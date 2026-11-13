"""Read-only date projections of the already audited rolling score cache.

New evaluation dates never move the original training folds or trigger fitting.
Execution-window boundaries and unavailable-date rules are applied by callers.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .consensus_search_research import verified_store_identity, preflight_predictions
from .event_engine import DAY_MS, utc_ms
from .high_win_research import SPLIT, END
from .model_search_research import KEEP, SYMBOLS
from .model_search_rules import MODEL_IDS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'model-search'
COLUMNS = list(KEEP) + ['p_'+name for name in MODEL_IDS] + [
    'model_fit_cutoff_ms', 'train_latest_settlement_ms']
MINUTE_MS = 60_000


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _key(value):
    return json.dumps(value, sort_keys=True)


def _validate_fold(meta):
    fold = meta['model_metadata']
    cutoff = fold['fit_cutoff_ms']
    models = fold.get('models', {})
    if (meta.get('symbol') not in SYMBOLS or fold.get('train_days') != 180
            or fold.get('refit_days') != 30 or not models
            or any(not model.get('training_converged') for model in models.values())):
        raise ValueError('Unsupported asset, rolling clock, or unconverged cached model')
    latest = max(model['latest_settlement_ms'] for model in models.values())
    if (latest >= cutoff or cutoff % MINUTE_MS
            or not cutoff <= fold['test_start_ms'] < fold['test_end_exclusive_ms'] <= cutoff+30*DAY_MS
            or fold['prediction_count'] <= 0):
        raise ValueError('Noncausal or invalid cached fold metadata')
    return fold, latest


def _validate_part(part, meta, latest):
    fold = meta['model_metadata']; cutoff = fold['fit_cutoff_ms']
    clock_names = ['signal_ms','entry_ms','settlement_ms',
                   'model_fit_cutoff_ms','train_latest_settlement_ms']
    clocks = part[clock_names].to_numpy(dtype=float)
    numeric = part[COLUMNS].to_numpy(dtype=float)
    if (len(part) != fold['prediction_count'] or not np.isfinite(numeric).all()
            or np.any(clocks != np.floor(clocks)) or np.any(clocks % MINUTE_MS)
            or part.signal_ms.duplicated().any()):
        raise ValueError('Nonfinite, duplicate, or incomplete cached prediction rows')
    if ((part.model_fit_cutoff_ms != cutoff).any()
            or (part.train_latest_settlement_ms != latest).any()
            or (part.signal_ms < fold['test_start_ms']).any()
            or (part.signal_ms >= fold['test_end_exclusive_ms']).any()
            or (part.entry_ms != part.signal_ms+MINUTE_MS).any()
            or (part.settlement_ms != part.signal_ms+31*MINUTE_MS).any()):
        raise ValueError('Noncausal or inconsistent cached prediction clocks')
    prices = part[['entry','exit']].to_numpy()
    scores = part[['p_'+name for name in MODEL_IDS]].to_numpy()
    if np.any(prices <= 0) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Invalid cached index price or model score')


def load_recent_predictions(start: date, end: date):
    """Load an arbitrary subwindow without changing audited model fit dates.

    The full original post-selection cache contract is verified first. Only
    overlapping slabs are materialized; all fold witnesses must still match
    the immutable original report. Both assets must be present at a scan time.
    """
    if (not isinstance(start,date) or not isinstance(end,date)
            or isinstance(start,datetime) or isinstance(end,datetime)
            or not SPLIT <= start < end <= END):
        raise ValueError('Require UTC calendar dates inside the original post-selection period')
    first, stop = utc_ms(start), utc_ms(end)
    identity = verified_store_identity()
    # Passing the new evaluation start here would request a different cache
    # filename and could cause an unnecessary, differently bounded fit.
    verified = preflight_predictions(SPLIT, END, identity)
    report_bytes = (ROOT/'research'/'model-search-results.json').read_bytes()
    report = json.loads(report_bytes)
    if report['provenance']['feature_store_identity'] != identity:
        raise ValueError('Original report and verified feature store disagree')
    witnesses = Counter(_key(fold) for fold in report['provenance']['later']['folds'])
    if len(verified) != sum(witnesses.values()):
        raise ValueError('Original later-fold population is incomplete')
    pieces = []; files = []; seen = set(); source_rows = 0; loaded_rows = 0
    for item in verified:
        name = item['file']
        if Path(name).name != name or not name.endswith('.npz') or name in seen:
            raise ValueError('Unexpected or duplicate verified cache filename')
        seen.add(name)
        path = CACHE/'predictions'/name
        meta_bytes = path.with_suffix('.json').read_bytes()
        if _hash(meta_bytes) != item['metadata_sha256']:
            raise ValueError('Prediction metadata changed after preflight')
        meta = json.loads(meta_bytes)
        witness = _key(meta['model_metadata'])
        if not witnesses[witness]:
            raise ValueError('Cached fold differs from the original model report witness')
        witnesses[witness] -= 1
        fold, latest = _validate_fold(meta)
        source_rows += fold['prediction_count']
        intersects = fold['test_start_ms'] < stop and fold['test_end_exclusive_ms'] > first
        entry = {**item, 'symbol':meta['symbol'], 'fit_cutoff_ms':fold['fit_cutoff_ms'],
                 'latest_settlement_ms':latest, 'source_rows':fold['prediction_count'],
                 'loaded':intersects, 'retained_before_joint_filter':0}
        if intersects:
            binary = path.read_bytes()
            if _hash(binary) != item['sha256'] or meta['sha256'] != item['sha256']:
                raise ValueError('Cached predictions changed after preflight')
            with np.load(io.BytesIO(binary), allow_pickle=False) as data:
                if not set(COLUMNS).issubset(data.files):
                    raise ValueError('Cached prediction columns are incomplete')
                part = pd.DataFrame({column:data[column] for column in COLUMNS})
            _validate_part(part, meta, latest)
            loaded_rows += len(part)
            part = part.loc[(part.signal_ms >= first)&(part.signal_ms < stop)].copy()
            part['symbol'] = meta['symbol']
            entry['retained_before_joint_filter'] = len(part)
            pieces.append(part)
        files.append(entry)
    if any(witnesses.values()) or not pieces:
        raise ValueError('Incomplete fold witnesses or empty requested prediction population')
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(['symbol','signal_ms']).any():
        raise ValueError('Overlapping source prediction slabs')
    before_joint = len(frame)
    counts = frame.groupby('signal_ms').symbol.transform('count')
    frame = frame.loc[counts == 2].sort_values(['signal_ms','symbol']).reset_index(drop=True)
    if frame.empty:
        raise ValueError('No scan times with both BTC and ETH available')
    return frame, {
        'feature_store_identity':identity, 'original_report_sha256':_hash(report_bytes),
        'original_start_utc':str(SPLIT), 'original_end_exclusive_utc':str(END),
        'start_utc':str(start), 'end_exclusive_utc':str(end),
        'source_prediction_rows':source_rows, 'loaded_slab_rows':loaded_rows,
        'rows_excluded_outside_requested_window':source_rows-before_joint,
        'loaded_rows_trimmed_at_window_edges':loaded_rows-before_joint,
        'joint_asset_rows_dropped':before_joint-len(frame), 'prediction_rows':len(frame),
        'verified_cache_files':files, 'training_or_download_performed':False,
        'preserved_original_fit_schedule':True,
        'execution_end_filter_applied':False,
    }
