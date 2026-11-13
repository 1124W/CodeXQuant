"""Fixed cross-model consensus study using verified local predictions only.

The plan precedes data inspection; discovery chooses once. Later candidate
results are disclosed for audit and never replace the locked selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .consensus_search_rules import candidates, directions
from .event_engine import DAY_MS, KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import (START, SPLIT, CONFIRM, END, BASE_CAPITAL,
                                detail, paired_week_interval, write_json, file_hash)
from .minute_scan_research import cutoff_for
from .model_search_models import specs
from .model_search_research import (CACHE, SYMBOLS, choose_discovery,
                                    prediction_phase, simulate_frame)
from .model_search_rules import candidates as model_candidates, directions as model_directions

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = 'logistic_halfhour_0.56_exhaustion'
LEDGER_COLUMNS = ['symbol', 'signal_ms', 'entry_ms', 'settlement_ms', 'entry',
                  'exit', 'direction', 'outcome', 'net_units', 'stake',
                  'horizon_minutes', 'model_fit_cutoff_ms',
                  'train_latest_settlement_ms', 'rule_id']


def source_hash():
    names = ['research/consensus_search_research.py', 'research/consensus_search_rules.py',
             'research/model_search_research.py', 'research/model_search_models.py',
             'research/model_search_rules.py', 'research/model_search_data.py',
             'research/high_win_research.py', 'research/event_engine.py',
             'research/event_strategy_research.py', 'research/minute_scan_research.py',
             'research/derivative_features.py', 'research/derivatives_data.py',
             'research/qd_three_year_data.py', 'live/features.py']
    return hashlib.sha256(b''.join(name.encode() + bytes.fromhex(file_hash(ROOT/name))
                                  for name in names)).hexdigest()


def make_plan():
    rules = candidates()
    return {'version': 1, 'models': specs(), 'rules': rules, 'candidate_count': len(rules),
            'discovery_start_utc': str(START), 'discovery_end_exclusive_utc': str(SPLIT),
            'validation_end_exclusive_utc': str(CONFIRM), 'confirmation_end_exclusive_utc': str(END),
            'symbols': list(SYMBOLS), 'scan_minutes': 1, 'holding_minutes': 30, 'entry_delay_minutes': 1,
            'train_days': 180, 'refit_days': 30, 'baseline_rule_id': BASELINE_ID,
            'prediction_policy': 'reuse complete source/content-verified cache only; no training or downloads',
            'calendar_policy': 'primary matches earlier unmasked studies; selected-only calendar exclusion is a disclosed sensitivity, with no reselection',
            'known_unavailable_utc_dates': list(KNOWN_UNAVAILABLE_DATES),
            'selection': 'discovery only; >=500 decisive, >=100 each asset, >=58% point win, positive PnL; maximum Bonferroni-Wilson lower; lexical tie',
            'minimum_validation_trades': 400, 'minimum_confirmation_trades': 100,
            'target_win_rate': .58, 'capital': dict(BASE_CAPITAL),
            'pending_outcomes_reserved': True, 'joint_asset_feature_availability_required': True,
            'historical_data_previously_researched': True, 'pristine_holdout': False,
            'forward_validated': False, 'post_selection_changes_allowed': False,
            'local_only': True, 'publish_or_deploy': False, 'source_sha256': source_hash(),
            'frozen_at_utc': datetime.now(timezone.utc).isoformat()}


def verified_store_identity(store_dir=None):
    """Fail closed instead of invoking the feature-store rebuilding path."""
    from live.features import FEATURES
    from .model_search_data import START as STORE_START, END as STORE_END
    from .model_search_data import source_identity, _windows, _valid_file
    folder = Path(store_dir or CACHE/'features')
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    expected = {f'{symbol}:{month}': (symbol, month, utc_ms(first), utc_ms(last))
                for month, first, last in _windows(STORE_START, STORE_END) for symbol in SYMBOLS}
    contract = {'schema_version': 1, 'complete': True, 'features': FEATURES,
                'start_utc': str(STORE_START), 'end_exclusive_utc': str(STORE_END),
                'sources_sha256': source_identity(), 'minute_delay': 1, 'horizon_minutes': 30,
                'offline': True}
    if any(manifest.get(key) != value for key, value in contract.items()):
        raise ValueError('Feature manifest is missing, stale or outside the fixed study contract')
    if set(manifest.get('files', {})) != set(expected):
        raise ValueError('Feature manifest does not contain every expected asset/month')
    for key, (symbol, month, first, last) in expected.items():
        item = manifest['files'][key]
        if (item.get('symbol') != symbol or item.get('month') != month
                or item.get('start_ms') != first or item.get('end_ms') != last
                or not _valid_file(folder, item)):
            raise ValueError('Feature cache failed read-only verification: ' + key)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    print('FEATURE STORE VERIFIED WITHOUT REBUILD', flush=True)
    return identity


def preflight_predictions(start, end, identity, cache_dir=None):
    """Verify every required slab before prediction_phase can be invoked."""
    folder = Path(cache_dir or CACHE/'predictions')
    contract = {'models_sha256': file_hash(ROOT/'research/model_search_models.py'),
                'data_code_sha256': file_hash(ROOT/'research/model_search_data.py'),
                'feature_store_identity': identity, 'pipeline_version': 1}
    cutoff = utc_ms(cutoff_for(start)); first_ms, end_ms = utc_ms(start), utc_ms(end)
    verified = []
    while cutoff < end_ms:
        stop = min(cutoff + 30*DAY_MS, end_ms); first = max(first_ms, cutoff)
        for symbol in SYMBOLS:
            path = folder/f'{symbol}-{cutoff}-{first}-{stop}.npz'
            meta_path = path.with_suffix('.json')
            if not path.is_file() or not meta_path.is_file():
                raise ValueError('Missing prediction cache; retraining forbidden: ' + path.name)
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('contract') != contract or meta.get('sha256') != file_hash(path):
                raise ValueError('Stale/corrupt prediction cache; retraining forbidden: ' + path.name)
            fold = meta.get('model_metadata', {})
            if meta.get('symbol') != symbol or fold.get('fit_cutoff_ms') != cutoff:
                raise ValueError('Prediction cache asset/cutoff metadata mismatch: ' + path.name)
            verified.append({'file': path.name, 'sha256': meta['sha256'],
                             'metadata_sha256': file_hash(meta_path)})
        cutoff = stop
    return verified


def load_verified_predictions(start, end, identity):
    cache_files = preflight_predictions(start, end, identity)
    frame, metadata = prediction_phase(start, end, identity)
    metadata['verified_cache_files'] = cache_files
    return frame, metadata


def save_ledger(path, ledger):
    pd.DataFrame(ledger, columns=LEDGER_COLUMNS).to_csv(path, index=False)


def calendar_mask(frame):
    stamps = frame.signal_ms.to_numpy(dtype=np.int64)
    expiry = frame.settlement_ms.to_numpy(dtype=np.int64)
    allowed = np.ones(len(frame), dtype=bool)
    for day in KNOWN_UNAVAILABLE_DATES:
        first = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
        allowed &= ~((stamps < first+DAY_MS) & (expiry >= first))
    return allowed


def run(output_dir=None):
    began = time.perf_counter(); out = Path(output_dir or ROOT/'research')
    plan = make_plan()
    write_json(out/'consensus-search-plan.json', plan)
    identity = verified_store_identity()
    discovery, discovery_meta = load_verified_predictions(START, SPLIT, identity)
    rules = plan['rules']; records = []
    for number, rule in enumerate(rules, 1):
        _, stats = simulate_frame(discovery, directions(discovery, rule), utc_ms(START), utc_ms(SPLIT), rule=rule)
        records.append({'rule': rule, 'discovery': stats})
        print(f'DISCOVERY {number}/{len(rules)}', flush=True)
    selected, ranking = choose_discovery(records)
    selection = {'selected_rule': selected, 'ranking': ranking, 'selection_cutoff_ms': utc_ms(SPLIT),
                 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': file_hash(out/'consensus-search-plan.json')}
    write_json(out/'consensus-search-selection.json', selection)
    print('LOCKED CONSENSUS ' + json.dumps(selected, ensure_ascii=False), flush=True)
    baseline_rule = next(rule for rule in model_candidates() if rule['id'] == BASELINE_ID)
    selected_discovery, _ = (simulate_frame(discovery, directions(discovery, selected),
        utc_ms(START), utc_ms(SPLIT), rule=selected) if selected else ([], {}))
    baseline_discovery, _ = simulate_frame(discovery, model_directions(discovery, baseline_rule),
        utc_ms(START), utc_ms(SPLIT), rule=baseline_rule)
    del discovery; gc.collect()
    later, later_meta = load_verified_predictions(SPLIT, END, identity)
    ranges = {'validation': (utc_ms(SPLIT), utc_ms(CONFIRM)),
              'historical_confirmation': (utc_ms(CONFIRM), utc_ms(END)),
              'continuous_post_selection': (utc_ms(SPLIT), utc_ms(END))}
    selected_ledgers = {}; matrix = []
    for number, record in enumerate(records, 1):
        rule = record['rule']; direction = directions(later, rule); measured = dict(record)
        for name, (first, last) in ranges.items():
            ledger, stats = simulate_frame(later, direction, first, last, rule=rule)
            measured[name] = stats
            if selected and rule['id'] == selected['id']: selected_ledgers[name] = ledger
        matrix.append(measured)
        print(f'LATER {number}/{len(rules)} without reselection', flush=True)
    baseline_direction = model_directions(later, baseline_rule)
    summaries = {'discovery': detail(selected_discovery, utc_ms(START), utc_ms(SPLIT))}
    baseline = {'discovery': detail(baseline_discovery, utc_ms(START), utc_ms(SPLIT))}
    paired = {}; baseline_ledgers = {}
    for name, (first, last) in ranges.items():
        ledger = selected_ledgers.get(name, [])
        base, _ = simulate_frame(later, baseline_direction, first, last, rule=baseline_rule)
        baseline_ledgers[name] = base
        summaries[name] = detail(ledger, first, last); baseline[name] = detail(base, first, last)
        paired[name] = paired_week_interval(ledger, base, first, last)
        print(json.dumps({'period': name, 'selected': {key: summaries[name][key]
              for key in ('signals', 'win_rate', 'net_units', 'max_drawdown_units')}}, ensure_ascii=False), flush=True)
    stresses = []
    if selected:
        direction = directions(later, selected)
        for name, config in [('adverse_index_entry_2bps', {'adverse_bps': 2.}),
                             ('fee_1pct', {'fee': .01}), ('payout_80pct', {'payout': .8})]:
            _, stats = simulate_frame(later, direction, utc_ms(SPLIT), utc_ms(END), stress=config, rule=selected)
            stresses.append({'name': name, 'configuration': config, 'test': stats})
        allowed = calendar_mask(later)
        masked = np.where(allowed, direction, 0).astype(np.int8)
        calendar_results = {}
        for name, (first, last) in ranges.items():
            ledger, _ = simulate_frame(later, masked, first, last, rule=selected)
            calendar_results[name] = detail(ledger, first, last)
            save_ledger(out/f'consensus-search-calendar-{name}-trades.csv', ledger)
        stresses.append({'name': 'known_unavailable_calendar',
                         'configuration': {'unavailable_dates_utc': list(KNOWN_UNAVAILABLE_DATES),
                                           'excluded_raw_opportunities': int(((direction != 0) & ~allowed).sum()),
                                           'reselection': False},
                         'test': calendar_results['continuous_post_selection'], 'periods': calendar_results})
    v = summaries['validation']; c = summaries['historical_confirmation']; post = summaries['continuous_post_selection']
    folds = discovery_meta['folds'] + later_meta['folds']
    checks = {'selection_exists': selected is not None,
              'validation_min400': v['decisive_trades'] >= 400,
              'confirmation_min100': c['decisive_trades'] >= 100,
              'validation_58pct': (v['win_rate'] or 0) >= .58,
              'confirmation_58pct': (c['win_rate'] or 0) >= .58,
              'validation_positive': v['net_units'] > 0, 'confirmation_positive': c['net_units'] > 0,
              'weekly_lower_above_break_even': (post['weekly_block_interval']['win_rate_lower_95'] or 0) > 1/1.85,
              'weekly_net_lower_positive': (post['weekly_block_interval']['net_lower_95'] or 0) > 0,
              'all_models_converged': all(m['training_converged'] for f in folds for m in f['models'].values()),
              'training_labels_settled': all(m['latest_settlement_ms'] < f['fit_cutoff_ms']
                                           for f in folds for m in f['models'].values())}
    selected_ledgers['discovery'] = selected_discovery; baseline_ledgers['discovery'] = baseline_discovery
    ledger_hashes = {}
    for name in ['discovery', *ranges]:
        for kind, ledgers in [('selected', selected_ledgers), ('baseline', baseline_ledgers)]:
            path = out/f'consensus-search-{kind}-{name}-trades.csv'
            save_ledger(path, ledgers.get(name, [])); ledger_hashes[path.name] = file_hash(path)
    result = {'schema_version': 1, 'engine': 'local-fixed-consensus-v1', 'plan': plan,
              'selection': selection, 'selected': summaries, 'baseline': baseline,
              'all_candidates': matrix, 'paired_comparison': paired, 'stresses': stresses,
              'checks': checks, 'historical_research_gates_passed': all(checks.values()), 'live_approved': False,
              'provenance': {'feature_store_identity': identity, 'source_sha256': plan['source_sha256'],
                             'plan_sha256': selection['plan_sha256'], 'ledger_sha256': ledger_hashes,
                             'discovery': discovery_meta, 'later': later_meta,
                             'runtime_seconds': round(time.perf_counter()-began, 2),
                             'generated_at_utc': datetime.now(timezone.utc).isoformat()},
              'limitations': [
                  'Repeated study of already researched history is exploratory, not pristine or prospective confirmation.',
                  'Exactly one discovery selection is locked. All 16 later results are disclosed only for audit.',
                  'Consensus members are correlated; agreement and directional scores are not calibrated win probabilities.',
                  'Bonferroni-Wilson ranking covers these 16 candidates only, not every prior research round; it assumes independent Bernoulli observations.',
                  'Seven-day joint BTC/ETH blocks address short dependence only; 58% point estimates do not establish future 58% win rates.',
                  'Minute index opens proxy event strike/expiry quotes. Two-basis-point adverse entry stress reruns outcomes and capital admissions.',
                  'Actual modeled payout is 85% and fee zero. Fee 1% and payout 80% are hypothetical robustness stresses.',
                  'Primary study intentionally matches earlier unmasked comparisons; known unavailable January 2026 dates are excluded only in the disclosed selected-only calendar sensitivity. Primary results include potentially non-tradable dates.',
                  'No production setting, notification, repository remote, server or order was changed.']}
    if source_hash() != plan['source_sha256']: raise RuntimeError('Research source changed during run')
    write_json(out/'consensus-search-results.json', result)
    print('LOCAL CONSENSUS REPORT SAVED ' + str(out/'consensus-search-results.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args().output_dir)
