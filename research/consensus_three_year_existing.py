"""Read-only sparse projection of the original discovery and later predictions.

The first 180 days of the original feature cache were training history. Existing
predictions start on 2024-05-07, not at the feature-cache start in November 2023.
No fitting, downloading, cache rewriting, strategy selection or publication.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .consensus_search_research import calendar_mask, preflight_predictions, verified_store_identity
from .consensus_search_rules import candidates, directions
from .event_engine import KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import START, SPLIT, END
from .recent_window_data import COLUMNS, _validate_fold, _validate_part

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'model-search'/'predictions'
RULE_ID = 'pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
RULE = next(rule for rule in candidates() if rule['id'] == RULE_ID)
SYMBOLS = ('BTCUSDT', 'ETHUSDT')
MINUTE = 60_000
OUTPUT_COLUMNS = ['symbol', 'signal_ms', 'entry_ms', 'settlement_ms', 'entry', 'exit',
                  'p_logistic_1m', 'p_gbm_deeper_5m', 'model_fit_cutoff_ms',
                  'train_latest_settlement_ms', 'direction', 'rule_id']


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _key(value):
    return json.dumps(value, sort_keys=True)


def _iso(stamp):
    return datetime.fromtimestamp(stamp/1000, timezone.utc).isoformat()


def _read_part(item, symbol, witnesses, seen):
    name = item['file']
    match = re.fullmatch(r'(BTCUSDT|ETHUSDT)-(\d+)-(\d+)-(\d+)\.npz', name)
    if not match or match[1] != symbol or name in seen:
        raise ValueError('Unexpected or duplicate original prediction filename')
    seen.add(name)
    path = CACHE/name
    metadata_bytes = path.with_suffix('.json').read_bytes()
    if _hash(metadata_bytes) != item['metadata_sha256']:
        raise ValueError('Original prediction metadata changed after verification')
    meta = json.loads(metadata_bytes)
    witness = _key(meta['model_metadata'])
    if witnesses[witness] <= 0:
        raise ValueError('Prediction fold differs from original report witness')
    witnesses[witness] -= 1
    fold, latest = _validate_fold(meta)
    cutoff, first, stop = map(int, match.groups()[1:])
    if meta['symbol'] != symbol or fold['fit_cutoff_ms'] != cutoff or not cutoff <= first < stop:
        raise ValueError('Prediction filename and metadata clocks disagree')
    binary = path.read_bytes()
    if _hash(binary) != item['sha256'] or meta['sha256'] != item['sha256']:
        raise ValueError('Original prediction contents changed after verification')
    with np.load(io.BytesIO(binary), allow_pickle=False) as arrays:
        if not set(COLUMNS).issubset(arrays.files):
            raise ValueError('Original prediction columns are incomplete')
        part = pd.DataFrame({name:arrays[name] for name in COLUMNS})
    _validate_part(part, meta, latest)
    if ((part.signal_ms < first).any() or (part.signal_ms >= stop).any()
            or int(part.signal_ms.min()) != fold['test_start_ms']
            or int(part.signal_ms.max())+MINUTE != fold['test_end_exclusive_ms']):
        raise ValueError('Original prediction population differs from fold bounds')
    part = part.sort_values('signal_ms').reset_index(drop=True)
    part['symbol'] = symbol
    return part, (cutoff, first, stop), {
        **item, 'symbol':symbol, 'fit_cutoff_ms':cutoff,
        'first_signal_ms':int(part.signal_ms.min()), 'last_signal_ms':int(part.signal_ms.max()),
        'source_rows':len(part), 'train_latest_settlement_ms':latest,
    }


def load_existing_opportunities(start=START, end=END):
    """Return fixed-rule opportunities and coverage without materializing all years.

    Both original phases are fully verified and bound to their original report
    witnesses. The shared cutoff at SPLIT remains two nonoverlapping prediction
    slabs. Cross-SPLIT execution labels are retained, not truncated or reset.
    End-of-evaluation settlement filtering belongs to the continuous simulator.
    """
    if type(start) is not date or type(end) is not date or not START <= start < end <= END:
        raise ValueError('Requested dates must lie inside existing 2024-05-07 onward predictions')
    first, stop = utc_ms(start), utc_ms(end)
    identity = verified_store_identity()
    original_bytes = (ROOT/'research'/'model-search-results.json').read_bytes()
    original = json.loads(original_bytes)
    if original['provenance']['feature_store_identity'] != identity:
        raise ValueError('Original model report has a different verified feature store')
    coverage = []
    for year in range(start.year, end.year+1):
        low, high = max(first, utc_ms(date(year,1,1))), min(stop, utc_ms(date(year+1,1,1)))
        if low >= high:
            continue
        coverage.append({'year':year, 'start_ms':low, 'end_exclusive_ms':high,
                         'requested_minutes':(high-low)//MINUTE,
                         'asset_rows_before_joint':0, 'joint_asset_rows_dropped':0,
                         'jointly_available_minutes':0, 'calendar_excluded_joint_minutes':0,
                         'candidate_opportunities_before_calendar':0,
                         'opportunities_excluded_by_calendar':0, 'raw_opportunities':0})
    pieces = []; files = []; seen = set(); phase_proofs = {}
    phase_edge = {}; source_rows = retained_rows = dropped_rows = 0
    previous_last = {symbol:None for symbol in SYMBOLS}
    for phase, low, high in (('discovery', START, SPLIT), ('later', SPLIT, END)):
        verified = preflight_predictions(low, high, identity)
        witnesses = Counter(_key(fold) for fold in original['provenance'][phase]['folds'])
        if len(verified) != sum(witnesses.values()) or len(verified)%2:
            raise ValueError('Original phase fold population is incomplete')
        phase_count = phase_drop = 0; phase_edge[phase] = {}
        for offset in range(0, len(verified), 2):
            parts = []; clocks = []
            for index, symbol in enumerate(SYMBOLS):
                part, clock, proof = _read_part(verified[offset+index], symbol, witnesses, seen)
                if previous_last[symbol] is not None and part.signal_ms.iloc[0] <= previous_last[symbol]:
                    raise ValueError('Overlapping or out-of-order original prediction slabs')
                previous_last[symbol] = int(part.signal_ms.iloc[-1])
                phase_edge[phase].setdefault(symbol, {'first_signal_ms':int(part.signal_ms.iloc[0])})
                phase_edge[phase][symbol]['last_signal_ms'] = int(part.signal_ms.iloc[-1])
                source_rows += len(part); phase_count += len(part)
                if not utc_ms(low) <= clock[1] < clock[2] <= utc_ms(high):
                    raise ValueError('Original prediction slab crosses its report phase')
                files.append(proof); parts.append(part); clocks.append(clock)
            if clocks[0] != clocks[1]:
                raise ValueError('BTC and ETH original slabs do not share a test interval')
            joint = np.intersect1d(parts[0].signal_ms, parts[1].signal_ms, assume_unique=True)
            phase_drop += sum(len(part) for part in parts)-2*len(joint)
            for part in parts:
                part = part.loc[(part.signal_ms >= first)&(part.signal_ms < stop)].copy()
                keep = part.signal_ms.isin(joint).to_numpy()
                retained_rows += len(part); dropped_rows += int((~keep).sum())
                for year in coverage:
                    belongs = ((part.signal_ms >= year['start_ms'])&(part.signal_ms < year['end_exclusive_ms'])).to_numpy()
                    year['asset_rows_before_joint'] += int(belongs.sum())
                    year['joint_asset_rows_dropped'] += int((belongs&~keep).sum())
                part = part.loc[keep].copy()
                signs = directions(part, RULE)
                allowed = calendar_mask(part)
                for year in coverage:
                    belongs = ((part.signal_ms >= year['start_ms'])&(part.signal_ms < year['end_exclusive_ms'])).to_numpy()
                    if part.symbol.eq('BTCUSDT').all():
                        year['jointly_available_minutes'] += int(belongs.sum())
                        year['calendar_excluded_joint_minutes'] += int((belongs&~allowed).sum())
                    year['candidate_opportunities_before_calendar'] += int((belongs&(signs != 0)).sum())
                    year['opportunities_excluded_by_calendar'] += int((belongs&(signs != 0)&~allowed).sum())
                    year['raw_opportunities'] += int((belongs&(signs != 0)&allowed).sum())
                selected = (signs != 0)&allowed
                part = part.loc[selected].copy()
                part['direction'] = signs[selected]
                part['rule_id'] = RULE_ID
                pieces.append(part[OUTPUT_COLUMNS])
        if any(witnesses.values()):
            raise ValueError('Original report fold witnesses were not fully consumed')
        expected = original['provenance'][phase]
        if (phase_drop != expected['joint_asset_rows_dropped']
                or phase_count-phase_drop != expected['prediction_rows']):
            raise ValueError('Original report minute populations changed')
        phase_proofs[phase] = {'source_rows':phase_count, 'joint_asset_rows_dropped':phase_drop,
                               'joint_prediction_rows':phase_count-phase_drop, 'verified_files':len(verified)}
        print(f'EXISTING {phase}: {phase_count:,} rows verified; no fitting', flush=True)
    frame = pd.concat(pieces, ignore_index=True).sort_values(['signal_ms','symbol']).reset_index(drop=True)
    if frame.duplicated(['symbol','signal_ms']).any():
        raise ValueError('Duplicate sparse opportunities after phase merge')
    for year in coverage:
        year['missing_joint_minutes'] = year['requested_minutes']-year['jointly_available_minutes']
    boundary = {symbol:{'last_discovery_signal_ms':phase_edge['discovery'][symbol]['last_signal_ms'],
                        'first_later_signal_ms':phase_edge['later'][symbol]['first_signal_ms'],
                        'missing_signal_minutes':max(0, (phase_edge['later'][symbol]['first_signal_ms']-
                             phase_edge['discovery'][symbol]['last_signal_ms'])//MINUTE-1)} for symbol in SYMBOLS}
    return frame, {'rule':RULE, 'feature_store_identity':identity,
                   'original_report_sha256':_hash(original_bytes),
                   'original_start_utc':str(START), 'original_end_exclusive_utc':str(END),
                   'start_utc':str(start), 'end_exclusive_utc':str(end),
                   'source_prediction_rows':source_rows, 'requested_asset_rows_before_joint':retained_rows,
                   'joint_asset_rows_dropped':dropped_rows, 'raw_opportunities':len(frame),
                   'phase_proofs':phase_proofs, 'phase_boundary':boundary,
                   'coverage_by_year':coverage, 'verified_cache_files':files,
                   'known_unavailable_dates_utc':list(KNOWN_UNAVAILABLE_DATES),
                   'cross_phase_execution_labels_retained':True,
                   'execution_end_filter_applied':False, 'preserved_original_fit_schedule':True,
                   'training_or_download_performed':False}
