"""Local-only weekly threshold adaptation on previously computed OOS scores.

This is another exploratory historical study, not an untouched holdout.
No fit, download, notification, model replacement or remote operation occurs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .event_engine import DAY_MS, KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import (START, SPLIT, CONFIRM, END, BASE_CAPITAL,
    detail, file_hash, paired_week_interval, write_json)
from .model_search_research import simulate_frame, choose_discovery
from .model_search_rules import candidates as model_candidates, directions as model_directions

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'model-search'
MINUTE_MS = 60000


def common_start():
    day = START + timedelta(days=180)
    return day + timedelta(days=(-day.weekday()) % 7)


def source_hash():
    names = ['research/adaptive_threshold_research.py', 'research/adaptive_threshold_rules.py',
             'research/model_search_research.py', 'research/model_search_rules.py',
             'research/high_win_research.py', 'research/event_engine.py',
             'research/model_search_models.py', 'research/model_search_data.py',
             'research/event_strategy_research.py', 'research/minute_scan_research.py',
             'research/derivative_features.py', 'research/derivatives_data.py',
             'research/qd_three_year_data.py', 'live/features.py']
    return hashlib.sha256(b''.join(n.encode()+bytes.fromhex(file_hash(ROOT/n)) for n in names)).hexdigest()


def calendar_mask(frame):
    """Exclude any signal-to-expiry interval touching a known unavailable day."""
    stamps = frame.signal_ms.to_numpy(dtype=np.int64)
    expiry = stamps + 31*MINUTE_MS
    mask = np.ones(len(frame), dtype=bool)
    for day in KNOWN_UNAVAILABLE_DATES:
        first = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
        mask &= ~((stamps < first+DAY_MS) & (expiry >= first))
    return mask


def load_cached_predictions():
    """Read a bounded projection of the exact verified 64-fold score cache.

    Fail closed on a changed contract, missing fold or corrupt file. There is
    deliberately no fallback to training or downloading observations.
    """
    from .model_search_data import source_identity
    store = json.loads((CACHE/'features'/'manifest.json').read_text(encoding='utf-8'))
    identity = hashlib.sha256(json.dumps(store, sort_keys=True).encode()).hexdigest()
    report = json.loads((ROOT/'research'/'model-search-results.json').read_text(encoding='utf-8'))
    if (not store.get('complete') or store.get('sources_sha256') != source_identity()
            or identity != report['provenance']['feature_store_identity']):
        raise ValueError('The feature manifest differs from the verified model study')
    contract = {'models_sha256': file_hash(ROOT/'research'/'model_search_models.py'),
                'data_code_sha256': file_hash(ROOT/'research'/'model_search_data.py'),
                'feature_store_identity': identity, 'pipeline_version': 1}
    columns = ['signal_ms','entry_ms','settlement_ms','entry','exit','ret_60','rv_60','rv_ratio',
               'model_fit_cutoff_ms','train_latest_settlement_ms',
               'p_logistic_halfhour','p_logistic_1m','p_gbm_deeper_5m']
    pieces, provenance = [], []
    paths = sorted((CACHE/'predictions').glob('*.json'))
    expected_metadata = report['provenance']['discovery']['folds'] + report['provenance']['later']['folds']
    remaining_metadata = Counter(json.dumps(meta,sort_keys=True) for meta in expected_metadata)
    if len(paths) != len(expected_metadata):
        raise ValueError('Missing or unexpected cached prediction folds')
    for path in paths:
        meta = json.loads(path.read_text(encoding='utf-8'))
        binary = path.with_suffix('.npz')
        if meta['contract'] != contract or file_hash(binary) != meta['sha256']:
            raise ValueError(f'Prediction provenance mismatch: {path.name}')
        model = meta['model_metadata']
        metadata_key = json.dumps(model,sort_keys=True)
        if not remaining_metadata[metadata_key]:
            raise ValueError('Fold metadata differs from the original verified model study')
        remaining_metadata[metadata_key] -= 1
        cutoff = model['fit_cutoff_ms']
        latest = max(m['latest_settlement_ms'] for m in model['models'].values())
        if latest >= cutoff or not all(m['training_converged'] for m in model['models'].values()):
            raise ValueError('Noncausal or unconverged source fit')
        with np.load(binary, allow_pickle=False) as data:
            part = pd.DataFrame({name:data[name] for name in columns})
        if (len(part) != model['prediction_count'] or
                (part.model_fit_cutoff_ms != cutoff).any() or
                (part.train_latest_settlement_ms != latest).any() or
                (part.signal_ms < max(cutoff, model['test_start_ms'])).any() or
                (part.signal_ms >= min(cutoff+30*DAY_MS, model['test_end_exclusive_ms'])).any() or
                (part.entry_ms != part.signal_ms+MINUTE_MS).any() or
                (part.settlement_ms != part.signal_ms+31*MINUTE_MS).any()):
            raise ValueError(f'Invalid prediction clocks: {path.name}')
        part['symbol'] = meta['symbol']
        pieces.append(part)
        provenance.append({'file':binary.name,'sha256':meta['sha256'],
                           'fit_cutoff_ms':cutoff,'latest_settlement_ms':latest,
                           'symbol':meta['symbol'],'rows':len(part)})
    frame = pd.concat(pieces, ignore_index=True)
    del pieces, part
    gc.collect()
    if frame.duplicated(['signal_ms','symbol']).any() or not frame.symbol.isin(['BTCUSDT','ETHUSDT']).all():
        raise ValueError('Duplicate score rows or unsupported asset')
    counts = frame.groupby('signal_ms').symbol.transform('count')
    dropped = int((counts != 2).sum())
    frame = frame.loc[(counts == 2)&(frame.signal_ms >= utc_ms(START))&(frame.signal_ms < utc_ms(END))]
    frame = frame.sort_values(['signal_ms','symbol']).reset_index(drop=True)
    expected_rows = report['provenance']['discovery']['prediction_rows'] + report['provenance']['later']['prediction_rows']
    if len(frame) != expected_rows:
        raise ValueError('Cached population differs from original verified model study')
    return frame, {'feature_store_identity':identity,'folds':provenance,
                   'joint_asset_rows_dropped':dropped,'prediction_rows':len(frame)}


def historical_checks(selected, summaries):
    v, c, post = (summaries[k] for k in ('validation','historical_confirmation','continuous_post_selection'))
    return {'selection_exists':selected is not None,
            'validation_min400':v['decisive_trades'] >= 400,
            'confirmation_min100':c['decisive_trades'] >= 100,
            'validation_58pct':(v['win_rate'] or 0) >= .58,
            'confirmation_58pct':(c['win_rate'] or 0) >= .58,
            'validation_positive':v['net_units'] > 0,
            'confirmation_positive':c['net_units'] > 0,
            'weekly_lower_above_break_even':(post['weekly_block_interval']['win_rate_lower_95'] or 0) > 1/1.85,
            'weekly_net_lower_positive':(post['weekly_block_interval']['net_lower_95'] or 0) > 0}


def run():
    from .adaptive_threshold_rules import candidates, apply_rule
    begin = time.perf_counter()
    out = ROOT/'research'
    rules = candidates()
    first = utc_ms(common_start())
    ranges = {'discovery':(first,utc_ms(SPLIT)),
              'validation':(utc_ms(SPLIT),utc_ms(CONFIRM)),
              'historical_confirmation':(utc_ms(CONFIRM),utc_ms(END)),
              'continuous_post_selection':(utc_ms(SPLIT),utc_ms(END))}
    plan = {'version':1,'rules':rules,'candidate_count':len(rules),
            'score_history_start_utc':str(START),'common_warmup_days_minimum':180,
            'discovery_start_utc':str(common_start()),'discovery_end_exclusive_utc':str(SPLIT),
            'validation_end_exclusive_utc':str(CONFIRM),'confirmation_end_exclusive_utc':str(END),
            'selection':'Discovery only; >=500 decisive, >=100 per asset, >=58% win and positive net; maximum Bonferroni Wilson lower then lexical ID.',
            'weekly_selection':'Monday 00:00 UTC; per asset; prior settled OOS labels only,00/30 signal grid; full lookback; >=120 decisive; observed win>=58%; Wilson one-sided alpha .05/8 lower>1/1.85; maximum lower, higher threshold tie.',
            'calibration_threshold_grid':[.56,.60,.64,.68],
            'known_unavailable_dates':list(KNOWN_UNAVAILABLE_DATES),
            'calendar_applies_to_calibration_signals_and_baseline':True,
            'capital':BASE_CAPITAL,'source_sha256':source_hash(),
            'historical_data_previously_researched':True,'pristine_holdout':False,
            'forward_validated':False,'post_selection_changes_allowed':False,
            'local_only':True,'publish_or_deploy':False,
            'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'adaptive-threshold-plan.json',plan)
    frame, provenance = load_cached_predictions()
    print(f'CACHE VERIFIED {len(frame)} rows, {len(provenance["folds"])} folds', flush=True)
    records, diagnostics_by_rule = [], {}
    # Generate/evaluate discovery before invoking adaptation on later labels.
    discovery = frame.loc[frame.signal_ms < utc_ms(SPLIT)].copy()
    for number, rule in enumerate(rules, 1):
        direction, diagnostics = apply_rule(discovery, rule, utc_ms(START), first, utc_ms(SPLIT))
        _, stats = simulate_frame(discovery,direction,first,utc_ms(SPLIT),rule=rule)
        records.append({'rule':rule,'discovery':stats})
        diagnostics_by_rule[rule['id']] = {'discovery':diagnostics}
        print(f'DISCOVERY {number}/{len(rules)} {rule["id"]}: {stats["signals"]} trades, win={stats["win_rate"]}', flush=True)
    selected, ranking = choose_discovery(records)
    selection = {'selected_rule':selected,'ranking':ranking,'selection_cutoff_ms':utc_ms(SPLIT),
                 'locked_at_utc':datetime.now(timezone.utc).isoformat(),
                 'plan_sha256':file_hash(out/'adaptive-threshold-plan.json')}
    write_json(out/'adaptive-threshold-selection.json',selection)
    print('LOCKED '+json.dumps(selected,ensure_ascii=False),flush=True)
    del discovery
    gc.collect()
    baseline_rule = next(r for r in model_candidates() if r['id']=='logistic_halfhour_0.56_exhaustion')
    base_direction = model_directions(frame,baseline_rule)
    base_direction[~calendar_mask(frame)] = 0
    base_ledgers = {}; baseline = {}
    for name,(lo,hi) in ranges.items():
        ledger,_ = simulate_frame(frame,base_direction,lo,hi,rule=baseline_rule)
        base_ledgers[name] = ledger
        baseline[name] = detail(ledger,lo,hi)
    matrix = []; chosen_ledgers = {}; selected_direction = None
    for number,row in enumerate(records,1):
        rule = row['rule']
        direction, diagnostics = apply_rule(frame,rule,utc_ms(START),first,utc_ms(END))
        diagnostics_by_rule[rule['id']]['all_weeks'] = diagnostics
        measured = dict(row)
        for name,(lo,hi) in ranges.items():
            ledger,stats = simulate_frame(frame,direction,lo,hi,rule=rule)
            if name=='discovery' and stats != row['discovery']:
                raise ValueError('Future rows changed discovery admissions')
            measured[name] = stats
            if selected and rule['id']==selected['id']:
                chosen_ledgers[name] = ledger
                selected_direction = direction
        matrix.append(measured)
        print(f'LATER {number}/{len(rules)} completed without reselection',flush=True)
    summaries = {}; paired = {}
    for name,(lo,hi) in ranges.items():
        ledger = chosen_ledgers.get(name,[])
        summaries[name] = detail(ledger,lo,hi)
        paired[name] = paired_week_interval(ledger,base_ledgers[name],lo,hi)
        if selected:
            pd.DataFrame(ledger).to_csv(out/f'adaptive-threshold-{name}-trades.csv',index=False)
    stresses = []
    if selected_direction is not None:
        for name,config in [('adverse_index_entry_2bps',{'adverse_bps':2.}),('payout_80pct',{'payout':.8})]:
            _,stats = simulate_frame(frame,selected_direction,utc_ms(SPLIT),utc_ms(END),stress=config,rule=selected)
            stresses.append({'name':name,'configuration':config,'test':stats})
    checks = historical_checks(selected,summaries)
    write_json(out/'adaptive-threshold-weekly-decisions.json',diagnostics_by_rule)
    if source_hash() != plan['source_sha256']:
        raise RuntimeError('Source changed during research')
    result = {'schema_version':1,'engine':'settled-OOS-weekly-threshold-adaptation-v1',
              'plan':plan,'selection':selection,'selected':summaries,'baseline':baseline,
              'all_candidates':matrix,'paired_comparison':paired,'stresses':stresses,'checks':checks,
              'historical_research_gates_passed':all(checks.values()),'live_approved':False,
              'provenance':{**provenance,'source_sha256':plan['source_sha256'],
                'weekly_decisions_sha256':file_hash(out/'adaptive-threshold-weekly-decisions.json'),
                'runtime_seconds':round(time.perf_counter()-begin,2),
                'generated_at_utc':datetime.now(timezone.utc).isoformat()},
              'limitations':[
                'This is a repeated search of previously researched history, not pristine holdout or forward validation.',
                'All candidates have the same 180-day initial warmup; discovery is consequently shorter than earlier studies.',
                'Halfhour calibration samples and admitted every-minute/shared-capital events are different populations.',
                'Wilson threshold bounds are selection heuristics, not a calibrated guarantee for accepted trades.',
                'Model refits and dependent market outcomes can invalidate calibration; final evidence uses weekly blocks.',
                'Historical availability calendar is incomplete; exact index opens are execution proxies.',
                'No production code, remote repository, server, alert, order or live model was changed.']}
    write_json(out/'adaptive-threshold-results.json',result)
    print(json.dumps({'selected':selected,'checks':checks,'runtime_seconds':result['provenance']['runtime_seconds']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    run()
