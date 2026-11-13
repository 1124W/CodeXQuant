"""Bounded local study: predict movements beyond a fixed +/-2bp neutral band.

The eight-rule plan precedes evaluation. One discovery selection is locked
before later folds are fitted or scored. Existing history has been reused in
many studies; no result here constitutes untouched or forward validation.
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

from .adaptive_threshold_rules import availability_mask
from .consensus_search_research import verified_store_identity, preflight_predictions, LEDGER_COLUMNS
from .event_engine import DAY_MS, MINUTE_MS, SYMBOLS, KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import (START, SPLIT, CONFIRM, END, BASE_CAPITAL,
                                detail, paired_week_interval, write_json, file_hash)
from .margin_search_models import fit_predict, specs, MARGIN_BPS
from .margin_search_rules import MODEL_IDS, candidates, directions
from .minute_scan_research import cutoff_for
from .model_search_data import load_window
from .model_search_research import choose_discovery, simulate_frame
from .model_search_rules import candidates as baseline_candidates, directions as baseline_directions

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'margin-search'
BASELINE_CACHE = ROOT/'research'/'.cache'/'model-search'/'predictions'
BASELINE_ID = 'logistic_halfhour_0.56_exhaustion'
KEEP = ['signal_ms', 'entry_ms', 'settlement_ms', 'entry', 'exit', 'ret_60', 'rv_60', 'rv_ratio']
PROBABILITY_COLUMNS = [f'p_{model}_{side}' for model in MODEL_IDS for side in ('up', 'down', 'neutral')]


def source_hash():
    names = ['research/margin_search_research.py', 'research/margin_search_rules.py',
             'research/margin_search_models.py', 'research/adaptive_threshold_rules.py',
             'research/consensus_search_research.py', 'research/model_search_research.py',
             'research/model_search_models.py', 'research/model_search_rules.py',
             'research/model_search_data.py', 'research/high_win_research.py',
             'research/event_engine.py', 'research/event_strategy_research.py',
             'research/minute_scan_research.py', 'research/derivative_features.py',
             'research/derivatives_data.py', 'research/qd_three_year_data.py', 'live/features.py']
    return hashlib.sha256(b''.join(name.encode()+bytes.fromhex(file_hash(ROOT/name)) for name in names)).hexdigest()


def make_plan():
    rules = candidates()
    return {'version': 1, 'candidate_count': len(rules), 'rules': rules, 'models': specs(),
            'discovery_start_utc': str(START), 'discovery_end_exclusive_utc': str(SPLIT),
            'validation_end_exclusive_utc': str(CONFIRM), 'confirmation_end_exclusive_utc': str(END),
            'symbols': list(SYMBOLS), 'scan_minutes': 1, 'entry_delay_minutes': 1,
            'holding_minutes': 30, 'train_days': 180, 'refit_days': 30,
            'target': {'margin_bps': MARGIN_BPS, 'classes': ['down', 'neutral', 'up'],
                       'neutral_includes_boundaries_and_refunds': True,
                       'neutral_training_samples_retained': True},
            'decision': 'Strictly larger up/down absolute three-class score; ties abstain; no renormalization after removing neutral.',
            'selection': 'Discovery only; >=500 decisive, >=100 each asset, >=58% win and positive net; highest Bonferroni-Wilson lower, lexical tie.',
            'later_disclosure': 'All eight candidates and their continuous 2bp adverse-entry stresses; no reselection.',
            'execution_robustness_gates': 'Selected 2bp adverse-entry net must be positive in validation, confirmation and continuous later history; every stressed settled bankroll must still fund one stake.',
            'feature_policy': 'Read-only source/content verification of existing 60-feature store; no rebuild or network.',
            'prediction_policy': 'Separate resumable atomic cache; each asset/fold trains only on labels settled strictly before cutoff.',
            'baseline_rule_id': BASELINE_ID, 'baseline_policy': 'Original verified 60-feature scores; same calendar and observation population.',
            'known_unavailable_dates': list(KNOWN_UNAVAILABLE_DATES),
            'calendar_applies_to_candidate_and_baseline': True,
            'capital': dict(BASE_CAPITAL), 'pending_outcomes_reserved': True,
            'joint_asset_feature_availability_required': True,
            'historical_data_previously_researched': True, 'pristine_holdout': False,
            'forward_validated': False, 'post_selection_changes_allowed': False,
            'local_only': True, 'publish_or_deploy': False, 'source_sha256': source_hash(),
            'frozen_at_utc': datetime.now(timezone.utc).isoformat()}


def _fold_windows(start, end):
    cutoff = utc_ms(cutoff_for(start))
    lower, upper = utc_ms(start), utc_ms(end)
    while cutoff < upper:
        stop = min(cutoff+30*DAY_MS, upper)
        yield cutoff, max(cutoff, lower), stop
        cutoff = stop


def _contract(identity):
    return {'models_sha256': file_hash(ROOT/'research'/'margin_search_models.py'),
            'pipeline_sha256': file_hash(ROOT/'research'/'margin_search_research.py'),
            'data_code_sha256': file_hash(ROOT/'research'/'model_search_data.py'),
            'feature_store_identity': identity, 'pipeline_version': 1, 'margin_bps': MARGIN_BPS}


def _save_npz(path, frame):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **{name: frame[name].to_numpy() for name in frame})
    temporary.replace(path)


def _validate_prediction(part, metadata, symbol, cutoff, first, stop, *, margin=True):
    """Validate content and fitted provenance for each saved slab, including reuse."""
    if (metadata.get('symbol') != symbol or metadata.get('first_signal_bound_ms') != first
            or metadata.get('end_exclusive_ms') != stop):
        raise ValueError('Prediction slab scope differs from its declared window')
    model = metadata['model_metadata']
    if (model.get('fit_cutoff_ms') != cutoff or model.get('prediction_count') != len(part)
            or not len(part) or model.get('train_days') != 180 or model.get('refit_days') != 30):
        raise ValueError('Invalid prediction fold metadata')
    fits = model['models']
    if margin and (set(fits) != set(MODEL_IDS) or model.get('specs') != specs()):
        raise ValueError('Unexpected margin model specifications')
    latest = max(item['latest_settlement_ms'] for item in fits.values())
    if latest >= cutoff:
        raise ValueError('Training contains labels not strictly settled before fitting')
    stamps = part.signal_ms.to_numpy()
    if (not np.isfinite(stamps).all() or np.any(stamps != np.floor(stamps))
            or np.any(stamps % MINUTE_MS) or len(np.unique(stamps)) != len(stamps)
            or np.any(stamps < max(first, cutoff)) or np.any(stamps >= min(stop, cutoff+30*DAY_MS))
            or model.get('test_start_ms') != int(stamps.min())
            or model.get('test_end_exclusive_ms') != int(stamps.max())+MINUTE_MS
            or (part.entry_ms != part.signal_ms+MINUTE_MS).any()
            or (part.settlement_ms != part.signal_ms+31*MINUTE_MS).any()
            or (part.model_fit_cutoff_ms != cutoff).any()
            or (part.train_latest_settlement_ms != latest).any()):
        raise ValueError('Invalid prediction event or provenance clocks')
    if margin:
        for name in MODEL_IDS:
            scores = part[[f'p_{name}_{side}' for side in ('up', 'down', 'neutral')]].to_numpy()
            if (not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                    or not np.allclose(scores.sum(axis=1), 1., rtol=1e-7, atol=1e-9)):
                raise ValueError('Invalid cached three-class probabilities')
    else:
        score = part.p_logistic_halfhour.to_numpy()
        if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
            raise ValueError('Invalid original baseline probabilities')


def _combine(parts):
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(['symbol', 'signal_ms']).any() or not frame.symbol.isin(SYMBOLS).all():
        raise ValueError('Duplicate prediction slabs or unsupported assets')
    counts = frame.groupby('signal_ms').symbol.transform('count')
    dropped = int((counts != 2).sum())
    frame = frame.loc[counts == 2].sort_values(['signal_ms', 'symbol']).reset_index(drop=True)
    return frame, dropped


def prediction_phase(start, end, identity, cache_dir=None):
    """Sequential fits with per-fold checkpoints; never alters the original cache."""
    folder = Path(cache_dir or CACHE)
    folder.mkdir(parents=True, exist_ok=True)
    contract = _contract(identity)
    parts, folds, files = [], [], []
    for cutoff, first, stop in _fold_windows(start, end):
        for symbol in SYMBOLS:
            path = folder/f'{symbol}-{cutoff}-{first}-{stop}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() or meta_path.exists():
                if not path.is_file() or not meta_path.is_file():
                    raise ValueError('Incomplete margin cache checkpoint: '+path.name)
                metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                if metadata.get('contract') != contract or metadata.get('sha256') != file_hash(path):
                    raise ValueError('Stale or corrupt margin cache checkpoint: '+path.name)
                with np.load(path, allow_pickle=False) as data:
                    predicted = pd.DataFrame({name: data[name] for name in KEEP+PROBABILITY_COLUMNS+
                                               ['model_fit_cutoff_ms', 'train_latest_settlement_ms']})
                print(f'REUSE MARGIN {symbol} {datetime.fromtimestamp(cutoff/1000, timezone.utc).date()}', flush=True)
            else:
                print(f'FIT MARGIN {symbol} {datetime.fromtimestamp(cutoff/1000, timezone.utc).date()}', flush=True)
                train = load_window(symbol, cutoff-180*DAY_MS, cutoff)
                test = load_window(symbol, first, stop)
                probabilities, model_meta = fit_predict(train, test, cutoff,
                    progress=lambda message: print('  '+message, flush=True))
                predicted = test[KEEP].reset_index(drop=True)
                for name in PROBABILITY_COLUMNS:
                    predicted[name] = probabilities[name].to_numpy()
                predicted['model_fit_cutoff_ms'] = cutoff
                predicted['train_latest_settlement_ms'] = max(item['latest_settlement_ms'] for item in model_meta['models'].values())
                metadata = {'contract': contract, 'symbol': symbol,
                            'first_signal_bound_ms': first, 'end_exclusive_ms': stop,
                            'model_metadata': model_meta}
                _validate_prediction(predicted, metadata, symbol, cutoff, first, stop)
                _save_npz(path, predicted)
                metadata['sha256'] = file_hash(path)
                write_json(meta_path, metadata)
                del train, test, probabilities
                gc.collect()
            _validate_prediction(predicted, metadata, symbol, cutoff, first, stop)
            predicted['symbol'] = symbol
            parts.append(predicted)
            folds.append(metadata['model_metadata'])
            files.append({'file': path.name, 'sha256': metadata['sha256'],
                          'metadata_sha256': file_hash(meta_path), 'symbol': symbol,
                          'fit_cutoff_ms': cutoff, 'first_signal_bound_ms': first,
                          'end_exclusive_ms': stop, 'rows': len(predicted)})
            write_json(folder/'progress.json', {'contract': contract, 'phase_start_utc': str(start),
                       'phase_end_exclusive_utc': str(end), 'completed_folds': files,
                       'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    frame, dropped = _combine(parts)
    del parts
    gc.collect()
    return frame, {'feature_store_identity': identity, 'folds': folds,
                   'verified_cache_files': files, 'joint_asset_rows_dropped': dropped,
                   'prediction_rows': len(frame)}


def load_baseline(start, end, identity):
    """Read only the original baseline projection after verifying every slab."""
    verified = preflight_predictions(start, end, identity, cache_dir=BASELINE_CACHE)
    parts, folds = [], []
    for cutoff, first, stop in _fold_windows(start, end):
        for symbol in SYMBOLS:
            path = BASELINE_CACHE/f'{symbol}-{cutoff}-{first}-{stop}.npz'
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path, allow_pickle=False) as data:
                part = pd.DataFrame({name: data[name] for name in KEEP+
                                     ['p_logistic_halfhour', 'model_fit_cutoff_ms', 'train_latest_settlement_ms']})
            scoped = {**metadata, 'first_signal_bound_ms': first, 'end_exclusive_ms': stop}
            _validate_prediction(part, scoped, symbol, cutoff, first, stop, margin=False)
            part['symbol'] = symbol
            parts.append(part)
            folds.append(metadata['model_metadata'])
    frame, dropped = _combine(parts)
    return frame, {'folds': folds, 'verified_cache_files': verified,
                   'joint_asset_rows_dropped': dropped, 'prediction_rows': len(frame)}


def assert_same_population(candidate, baseline):
    for name in ['symbol']+KEEP:
        if not np.array_equal(candidate[name].to_numpy(), baseline[name].to_numpy()):
            raise ValueError('Candidate and original baseline populations differ: '+name)


def _baseline_ledger(frame, rule, first, stop):
    signal = baseline_directions(frame, rule)
    signal[~availability_mask(frame.signal_ms.to_numpy())] = 0
    return simulate_frame(frame, signal, first, stop, rule=rule)[0]


def _save_ledger(path, ledger):
    pd.DataFrame(ledger, columns=LEDGER_COLUMNS).to_csv(path, index=False)
    return {'file': Path(path).name, 'sha256': file_hash(path), 'rows': len(ledger)}


def stress_detail(ledger, first, stop):
    result = detail(ledger, first, stop)
    capital = BASE_CAPITAL['initial_bankroll']
    balances = capital+np.cumsum([trade['net_units'] for trade in ledger])
    result['minimum_settled_bankroll'] = float(min(capital, balances.min())) if len(balances) else float(capital)
    result['bankroll_can_fund_next_stake'] = result['minimum_settled_bankroll'] >= BASE_CAPITAL['stake']
    return result


def historical_checks(selected, summaries, stressed_periods, folds):
    """Keep nominal statistical evidence separate from execution robustness."""
    v, c, post = (summaries[name] for name in ('validation', 'historical_confirmation', 'continuous_post_selection'))
    nominal = {'selection_exists': selected is not None,
               'validation_min400': v['decisive_trades'] >= 400,
               'confirmation_min100': c['decisive_trades'] >= 100,
               'validation_58pct': (v['win_rate'] or 0) >= .58,
               'confirmation_58pct': (c['win_rate'] or 0) >= .58,
               'validation_positive': v['net_units'] > 0, 'confirmation_positive': c['net_units'] > 0,
               'weekly_lower_above_break_even': (post['weekly_block_interval']['win_rate_lower_95'] or 0) > 1/1.85,
               'weekly_net_lower_positive': (post['weekly_block_interval']['net_lower_95'] or 0) > 0,
               'all_models_converged': bool(folds) and all(m['training_converged'] for f in folds for m in f['models'].values()),
               'training_labels_settled': bool(folds) and all(m['latest_settlement_ms'] < f['fit_cutoff_ms'] for f in folds for m in f['models'].values())}
    stress = {}
    for name in ('validation', 'historical_confirmation', 'continuous_post_selection'):
        stats = stressed_periods.get(name, {})
        stress['adverse_2bps_'+name+'_positive'] = stats.get('net_units', 0) > 0
        stress['adverse_2bps_'+name+'_no_ruin'] = bool(stats.get('bankroll_can_fund_next_stake', False))
    return nominal, stress


def run(phase='all', output_dir=None):
    if phase not in ('all', 'discovery'):
        raise ValueError('Unknown study phase')
    began = time.perf_counter()
    out = Path(output_dir or ROOT/'research')
    plan = make_plan()
    write_json(out/'margin-search-plan.json', plan)
    identity = verified_store_identity()
    discovery, discovery_meta = prediction_phase(START, SPLIT, identity)
    records = []
    for number, rule in enumerate(plan['rules'], 1):
        _, stats = simulate_frame(discovery, directions(discovery, rule), utc_ms(START), utc_ms(SPLIT), rule=rule)
        records.append({'rule': rule, 'discovery': stats})
        print(f'DISCOVERY {number}/8 {rule["id"]}: n={stats["decisive_trades"]}, win={stats["win_rate"]}', flush=True)
    selected, ranking = choose_discovery(records)
    selection = {'selected_rule': selected, 'ranking': ranking, 'selection_cutoff_ms': utc_ms(SPLIT),
                 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': file_hash(out/'margin-search-plan.json')}
    write_json(out/'margin-search-selection.json', selection)
    print('LOCKED MARGIN '+json.dumps(selected, ensure_ascii=False), flush=True)
    if phase == 'discovery':
        if source_hash() != plan['source_sha256']:
            raise RuntimeError('Source changed during margin discovery')
        return selection
    selected_discovery = (simulate_frame(discovery, directions(discovery, selected),
                          utc_ms(START), utc_ms(SPLIT), rule=selected)[0] if selected else [])
    ledger_files = []
    baseline_rule = next(rule for rule in baseline_candidates() if rule['id'] == BASELINE_ID)
    base, base_discovery_meta = load_baseline(START, SPLIT, identity)
    assert_same_population(discovery, base)
    baseline_discovery = _baseline_ledger(base, baseline_rule, utc_ms(START), utc_ms(SPLIT))
    ledger_files.append(_save_ledger(out/'margin-search-baseline-discovery-trades.csv', baseline_discovery))
    del base, discovery
    gc.collect()
    later, later_meta = prediction_phase(SPLIT, END, identity)
    ranges = {'validation': (utc_ms(SPLIT), utc_ms(CONFIRM)),
              'historical_confirmation': (utc_ms(CONFIRM), utc_ms(END)),
              'continuous_post_selection': (utc_ms(SPLIT), utc_ms(END))}
    matrix, selected_ledgers = [], {'discovery': selected_discovery}
    for number, record in enumerate(records, 1):
        rule = record['rule']; signal = directions(later, rule); measured = dict(record)
        for name, (first, stop) in ranges.items():
            ledger, stats = simulate_frame(later, signal, first, stop, rule=rule)
            measured[name] = stats
            if selected and rule['id'] == selected['id']:
                selected_ledgers[name] = ledger
        _, stressed = simulate_frame(later, signal, utc_ms(SPLIT), utc_ms(END),
                                     stress={'adverse_bps': MARGIN_BPS}, rule=rule)
        measured['continuous_adverse_entry_2bps'] = stressed
        matrix.append(measured)
        print(f'LATER {number}/8 disclosed without reselection', flush=True)
    base, base_later_meta = load_baseline(SPLIT, END, identity)
    assert_same_population(later, base)
    summaries = {'discovery': detail(selected_discovery, utc_ms(START), utc_ms(SPLIT))}
    baseline = {'discovery': detail(baseline_discovery, utc_ms(START), utc_ms(SPLIT))}
    paired = {'discovery': paired_week_interval(selected_discovery, baseline_discovery, utc_ms(START), utc_ms(SPLIT))}
    for name, (first, stop) in ranges.items():
        ledger = selected_ledgers.get(name, [])
        base_ledger = _baseline_ledger(base, baseline_rule, first, stop)
        summaries[name] = detail(ledger, first, stop)
        baseline[name] = detail(base_ledger, first, stop)
        paired[name] = paired_week_interval(ledger, base_ledger, first, stop)
        ledger_files.append(_save_ledger(out/f'margin-search-{name}-trades.csv', ledger))
        ledger_files.append(_save_ledger(out/f'margin-search-baseline-{name}-trades.csv', base_ledger))
    ledger_files.append(_save_ledger(out/'margin-search-discovery-trades.csv', selected_discovery))
    stresses = []
    periods = {}
    if selected:
        signal = directions(later, selected)
        for name, (first, stop) in ranges.items():
            ledger, _ = simulate_frame(later, signal, first, stop, stress={'adverse_bps': MARGIN_BPS}, rule=selected)
            periods[name] = stress_detail(ledger, first, stop)
            ledger_files.append(_save_ledger(out/f'margin-search-adverse2bps-{name}-trades.csv', ledger))
        stresses.append({'name': 'adverse_index_entry_2bps', 'configuration': {'adverse_bps': MARGIN_BPS},
                         'test': periods['continuous_post_selection'], 'periods': periods})
    folds = discovery_meta['folds']+later_meta['folds']
    nominal_checks, stress_checks = historical_checks(selected, summaries, periods, folds)
    checks = {**nominal_checks, **stress_checks}
    if source_hash() != plan['source_sha256']:
        raise RuntimeError('Source changed during margin research')
    result = {'schema_version': 1, 'engine': 'three-class-fixed-two-basis-point-margin-v1',
              'plan': plan, 'selection': selection, 'selected': summaries, 'baseline': baseline,
              'all_candidates': matrix, 'paired_comparison': paired, 'stresses': stresses, 'checks': checks,
              'nominal_history_gates_passed': all(nominal_checks.values()),
              'execution_robustness_gates_passed': all(stress_checks.values()),
              'historical_research_gates_passed': all(checks.values()), 'live_approved': False,
              'provenance': {'feature_store_identity': identity, 'source_sha256': plan['source_sha256'],
                             'plan_sha256': file_hash(out/'margin-search-plan.json'),
                             'selection_sha256': file_hash(out/'margin-search-selection.json'),
                             'ledger_files': ledger_files,
                             'discovery': discovery_meta, 'later': later_meta,
                             'baseline_discovery': base_discovery_meta, 'baseline_later': base_later_meta,
                             'runtime_seconds': round(time.perf_counter()-began, 2),
                             'generated_at_utc': datetime.now(timezone.utc).isoformat()},
              'limitations': [
                  'This is another bounded family after more than one hundred variations on reused history; not pristine validation.',
                  'The two-basis-point label margin is predeclared, not tuned using later returns; it cannot guarantee live execution robustness.',
                  'Scores measure three target classes and are not calibrated event win probabilities; neutral outcomes remain in training.',
                  'All eight later outcomes and 2bp stresses are disclosed for audit, never for replacing the discovery selection.',
                  'Five-minute training targets overlap; weekly blocks do not rule out longer dependence or regime shifts.',
                  'Minute index opens are proxies for live strike/expiry prices; known availability dates are an incomplete calendar.',
                  'No live model, production rule, alert, remote repository, server or order was changed.']}
    write_json(out/'margin-search-results.json', result)
    print(json.dumps({'selected': selected, 'checks': checks,
                      'runtime_seconds': result['provenance']['runtime_seconds']}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'discovery'), default='all')
    run(parser.parse_args().phase)
