"""Paired base60 versus base60+8 position-metrics study, local research only.

Eight augmented candidates are selected using discovery only. Their eight
matched controls use the same models, labels and exact training/inference
populations, and cannot enter the selection ranking. Historical archive time
plus ten minutes is an availability assumption, not point-in-time proof.
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

from .consensus_search_research import verified_store_identity
from .event_engine import DAY_MS, MINUTE_MS, SYMBOLS, KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import (START, SPLIT, CONFIRM, END, BASE_CAPITAL,
                                detail, paired_week_interval, write_json, file_hash)
from .margin_search_research import (_fold_windows, _save_npz, _combine,
                                    _save_ledger, stress_detail, historical_checks)
from .metrics_data import (START as METRICS_START, END as METRICS_END,
                           load_metrics, verify_metrics_store, store_identity)
from .metrics_features import METRIC_FEATURES, load_enriched_window
from .metrics_search_models import fit_predict, specs
from .metrics_search_rules import candidates, directions, matched_control
from .model_search_research import choose_discovery, simulate_frame

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'metrics-search'
KEEP = ['signal_ms', 'entry_ms', 'settlement_ms', 'entry', 'exit', 'ret_60', 'rv_60', 'rv_ratio',
        'metrics_record_ms', 'metrics_available_ms', 'metrics_age_ms']
MODEL_IDS = ('logistic_base', 'logistic_aug', 'gbm_base', 'gbm_aug')
PROBABILITY_COLUMNS = ['p_'+name for name in MODEL_IDS]


def source_hash():
    names = ['research/metrics_search_research.py', 'research/metrics_search_rules.py',
             'research/metrics_search_models.py', 'research/metrics_features.py', 'research/metrics_data.py',
             'research/margin_search_research.py', 'research/margin_search_models.py',
             'research/margin_search_rules.py', 'research/adaptive_threshold_rules.py',
             'research/consensus_search_research.py', 'research/model_search_research.py',
             'research/model_search_models.py', 'research/model_search_rules.py', 'research/model_search_data.py',
             'research/high_win_research.py', 'research/event_engine.py', 'research/event_strategy_research.py',
             'research/minute_scan_research.py', 'research/derivative_features.py',
             'research/derivatives_data.py', 'research/qd_three_year_data.py', 'live/features.py']
    return hashlib.sha256(b''.join(name.encode()+bytes.fromhex(file_hash(ROOT/name)) for name in names)).hexdigest()


def make_plan():
    augmented = candidates()
    controls = [matched_control(rule) for rule in augmented]
    return {'version': 1, 'rules': augmented, 'matched_controls': controls, 'candidate_count': 8,
            'selection_candidate_count': 8, 'control_count': 8, 'total_evaluated_rule_count': 16,
            'previously_researched_configurations': 113,
            'models': specs(), 'base_feature_count': 60, 'added_features': list(METRIC_FEATURES),
            'hypothesis': 'Position build/unwind and trader positioning provide information absent from price/volume/premium/funding features.',
            'discovery_start_utc': str(START), 'discovery_end_exclusive_utc': str(SPLIT),
            'validation_end_exclusive_utc': str(CONFIRM), 'confirmation_end_exclusive_utc': str(END),
            'symbols': list(SYMBOLS), 'entry_delay_minutes': 1, 'holding_minutes': 30,
            'scan_minutes': 1, 'train_days': 180, 'refit_days': 30,
            'metrics_availability_assumption': {'record_lag_minutes': 10,
                'available_age_minutes_inclusive_min': 0, 'available_age_minutes_exclusive_max': 5,
                'point_in_time_proven': False,
                'stock_change_endpoints': 'Exact timestamps; missing intermediate snapshots do not invent or change stock endpoints.',
                'raw_warmup_policy': 'Clamp to fixed 2023-11-09 coverage; incomplete first 4h10m drops from both paired fits.'},
            'paired_ablation': 'Per model, both feature arms share exact training timestamp/label hashes and identical inference rows; same estimator, parameters, seed and settled-label purge.',
            'selection': 'Eight augmented rules only; discovery >=500 decisive, >=100 each asset, >=58% win and positive net; maximum Bonferroni-Wilson lower over eight then lexical ID.',
            'baseline_policy': 'The selected candidate is compared only with its same-model/threshold/regime base60 control fitted in the same paired cache.',
            'incremental_evidence': 'Separate historical flag requires selected-minus-control weekly net lower95 >0; profitability alone is not evidence that metrics add information.',
            'execution_robustness_gates': 'Selected 2bp adverse-entry net positive in validation, confirmation and continuous later history; each stressed bankroll must still fund one stake.',
            'later_disclosure': 'All eight augmented candidates and eight matched controls, including continuous2bp stresses; no later reselection.',
            'feature_policy': 'Read-only verified existing base feature store and completed verified metrics archive; never download or rebuild in this runner.',
            'known_unavailable_dates': list(KNOWN_UNAVAILABLE_DATES),
            'calendar_applies_to_both_arms': True, 'joint_asset_feature_availability_required': True,
            'capital': dict(BASE_CAPITAL), 'historical_data_previously_researched': True,
            'pristine_holdout': False, 'forward_validated': False, 'post_selection_changes_allowed': False,
            'local_only': True, 'publish_or_deploy': False, 'source_sha256': source_hash(),
            'frozen_at_utc': datetime.now(timezone.utc).isoformat()}


def _contract(base_identity, metrics_identity):
    return {'models_sha256': file_hash(ROOT/'research'/'metrics_search_models.py'),
            'features_sha256': file_hash(ROOT/'research'/'metrics_features.py'),
            'metrics_data_sha256': file_hash(ROOT/'research'/'metrics_data.py'),
            'base_data_sha256': file_hash(ROOT/'research'/'model_search_data.py'),
            'base_models_sha256': file_hash(ROOT/'research'/'model_search_models.py'),
            'pipeline_sha256': file_hash(ROOT/'research'/'metrics_search_research.py'),
            'feature_store_identity': base_identity, 'metrics_store_identity': metrics_identity,
            'pipeline_version': 1, 'metrics_features': list(METRIC_FEATURES),
            'availability_lag_ms': 10*MINUTE_MS, 'max_available_age_exclusive_ms': 5*MINUTE_MS}


def _validate_prediction(part, metadata, symbol, cutoff, first, stop):
    if (metadata.get('symbol') != symbol or metadata.get('first_signal_bound_ms') != first
            or metadata.get('end_exclusive_ms') != stop):
        raise ValueError('Metrics prediction scope differs from its declared window')
    model = metadata['model_metadata']
    if (model.get('fit_cutoff_ms') != cutoff or model.get('prediction_count') != len(part)
            or not len(part) or model.get('train_days') != 180 or model.get('refit_days') != 30
            or model.get('specs') != specs() or set(model.get('models', {})) != set(MODEL_IDS)):
        raise ValueError('Invalid paired model fold metadata')
    fitted = model['models']
    latest = max(item['latest_settlement_ms'] for item in fitted.values())
    if latest >= cutoff:
        raise ValueError('Paired fitting includes labels not strictly settled before cutoff')
    pairs = model.get('pairs', {})
    if set(pairs) != {'logistic', 'gbm'}:
        raise ValueError('Missing paired training population proofs')
    for family in ('logistic', 'gbm'):
        base, aug = fitted[family+'_base'], fitted[family+'_aug']
        for key in ('training_count', 'latest_settlement_ms', 'earliest_signal_ms'):
            if base.get(key) != aug.get(key):
                raise ValueError('Base and augmented training populations differ')
        pair = pairs[family]
        if (not pair.get('identical_train_rows_and_labels') or not pair.get('identical_test_rows')
                or pair.get('base_feature_count') != 60 or pair.get('augmented_feature_count') != 68):
            raise ValueError('Missing exact feature-ablation population proof')
        for key in ('train_signal_sha256', 'train_label_sha256', 'test_signal_sha256'):
            value = base.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in '0123456789abcdef' for character in value)
                    or any(record.get(key) != value for record in
                           (aug, pair, pair.get('base', {}), pair.get('aug', {})))):
                raise ValueError('Paired training/prediction hashes differ')
        for key in ('training_count', 'prediction_count'):
            if any(record.get(key) != base.get(key) for record in
                   (aug, pair, pair.get('base', {}), pair.get('aug', {}))):
                raise ValueError('Paired training/prediction counts differ')
    stamps = part.signal_ms.to_numpy()
    if (not np.isfinite(stamps).all() or np.any(stamps != np.floor(stamps))
            or np.any(stamps % MINUTE_MS) or len(np.unique(stamps)) != len(stamps)
            or np.any(stamps < max(cutoff, first)) or np.any(stamps >= min(stop, cutoff+30*DAY_MS))
            or model.get('test_start_ms') != int(stamps.min())
            or model.get('test_end_exclusive_ms') != int(stamps.max())+MINUTE_MS
            or (part.entry_ms != part.signal_ms+MINUTE_MS).any()
            or (part.settlement_ms != part.signal_ms+31*MINUTE_MS).any()
            or (part.model_fit_cutoff_ms != cutoff).any()
            or (part.train_latest_settlement_ms != latest).any()):
        raise ValueError('Invalid paired prediction clocks')
    if (not np.isfinite(part[['metrics_record_ms', 'metrics_available_ms', 'metrics_age_ms']].to_numpy()).all()
            or (part.metrics_record_ms % (5*MINUTE_MS) != 0).any()
            or (part.metrics_available_ms != part.metrics_record_ms+10*MINUTE_MS).any()
            or (part.metrics_age_ms != part.signal_ms-part.metrics_available_ms).any()
            or (part.metrics_age_ms < 0).any() or (part.metrics_age_ms >= 5*MINUTE_MS).any()):
        raise ValueError('Metrics publication availability or freshness violated')
    actual_test_hash = hashlib.sha256(np.asarray(stamps, dtype='<i8').tobytes()).hexdigest()
    if any(record['test_signal_sha256'] != actual_test_hash for record in fitted.values()):
        raise ValueError('Cached prediction population hash does not match actual rows')
    metrics_identity = metadata['contract']['metrics_store_identity']
    for name in ('training_feature_quality', 'inference_feature_quality'):
        if metadata.get(name, {}).get('manifest_identity') != metrics_identity:
            raise ValueError('Paired features refer to a different metrics manifest')
    scores = part[PROBABILITY_COLUMNS].to_numpy()
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Invalid paired directional probabilities')


def prediction_phase(start, end, base_identity, metrics_identity, raw_metrics, raw_quality, cache_dir=None):
    folder = Path(cache_dir or CACHE)
    folder.mkdir(parents=True, exist_ok=True)
    contract = _contract(base_identity, metrics_identity)
    parts, folds, files = [], [], []
    for cutoff, first, stop in _fold_windows(start, end):
        for symbol in SYMBOLS:
            path = folder/f'{symbol}-{cutoff}-{first}-{stop}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() or meta_path.exists():
                if not path.is_file() or not meta_path.is_file():
                    raise ValueError('Incomplete paired metrics checkpoint: '+path.name)
                metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                if metadata.get('contract') != contract or metadata.get('sha256') != file_hash(path):
                    raise ValueError('Stale or corrupt paired metrics checkpoint: '+path.name)
                with np.load(path, allow_pickle=False) as data:
                    predicted = pd.DataFrame({name: data[name] for name in KEEP+PROBABILITY_COLUMNS+
                                               ['model_fit_cutoff_ms', 'train_latest_settlement_ms']})
                print(f'REUSE METRICS {symbol} {datetime.fromtimestamp(cutoff/1000, timezone.utc).date()}', flush=True)
            else:
                print(f'FIT METRICS PAIRED {symbol} {datetime.fromtimestamp(cutoff/1000, timezone.utc).date()}', flush=True)
                extra = {'metrics_frame': raw_metrics[symbol], 'metrics_quality': raw_quality[symbol]}
                train, train_quality = load_enriched_window(symbol, cutoff-180*DAY_MS, cutoff, **extra)
                test, test_quality = load_enriched_window(symbol, first, stop, **extra)
                probabilities, model_meta = fit_predict(train, test, cutoff,
                    progress=lambda message: print('  '+message, flush=True))
                predicted = test[KEEP].reset_index(drop=True)
                for name in PROBABILITY_COLUMNS:
                    predicted[name] = probabilities[name].to_numpy()
                predicted['model_fit_cutoff_ms'] = cutoff
                predicted['train_latest_settlement_ms'] = max(item['latest_settlement_ms'] for item in model_meta['models'].values())
                metadata = {'contract': contract, 'symbol': symbol, 'first_signal_bound_ms': first,
                            'end_exclusive_ms': stop, 'model_metadata': model_meta,
                            'training_feature_quality': train_quality, 'inference_feature_quality': test_quality}
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
                          'end_exclusive_ms': stop, 'rows': len(predicted),
                          'training_feature_quality': metadata['training_feature_quality'],
                          'inference_feature_quality': metadata['inference_feature_quality']})
            write_json(folder/'progress.json', {'contract': contract, 'phase_start_utc': str(start),
                       'phase_end_exclusive_utc': str(end), 'completed_folds': files,
                       'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    frame, dropped = _combine(parts)
    del parts
    gc.collect()
    return frame, {'feature_store_identity': base_identity, 'metrics_store_identity': metrics_identity,
                   'folds': folds, 'verified_cache_files': files,
                   'joint_asset_rows_dropped': dropped, 'prediction_rows': len(frame)}


def incremental_evidence(selected, paired):
    comparison = paired.get('continuous_post_selection', {})
    lower = comparison.get('delta_net_lower_95')
    return bool(selected is not None and lower is not None and lower > 0)


def run(phase='all', output_dir=None):
    if phase not in ('all', 'discovery'):
        raise ValueError('Unknown study phase')
    began = time.perf_counter()
    out = Path(output_dir or ROOT/'research')
    plan = make_plan()
    write_json(out/'metrics-search-plan.json', plan)
    base_identity = verified_store_identity()
    manifest = verify_metrics_store()
    metrics_identity = store_identity(manifest)
    raw_metrics, raw_quality = {}, {}
    for symbol in SYMBOLS:
        raw_metrics[symbol], raw_quality[symbol] = load_metrics(symbol, METRICS_START, METRICS_END)
        if raw_quality[symbol].get('manifest_identity') != metrics_identity:
            raise ValueError('Metrics manifest changed while preloading verified observations')
        print(f'METRICS VERIFIED {symbol}: {len(raw_metrics[symbol])} raw observations', flush=True)
    discovery, discovery_meta = prediction_phase(START, SPLIT, base_identity, metrics_identity, raw_metrics, raw_quality)
    records, control_records = [], []
    for number, rule in enumerate(plan['rules'], 1):
        control = matched_control(rule)
        for chosen, destination in ((rule, records), (control, control_records)):
            _, stats = simulate_frame(discovery, directions(discovery, chosen), utc_ms(START), utc_ms(SPLIT), rule=chosen)
            destination.append({'rule': chosen, 'discovery': stats})
        stats = records[-1]['discovery']
        print(f'DISCOVERY {number}/8 AUG {rule["id"]}: n={stats["decisive_trades"]}, win={stats["win_rate"]}', flush=True)
    # Controls are deliberately absent from the selection input and correction count.
    selected, ranking = choose_discovery(records)
    control = matched_control(selected) if selected else None
    selection = {'selected_rule': selected, 'matched_control': control, 'ranking': ranking,
                 'selection_candidate_count': len(records), 'controls_eligible_for_selection': False,
                 'selection_cutoff_ms': utc_ms(SPLIT), 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': file_hash(out/'metrics-search-plan.json')}
    write_json(out/'metrics-search-selection.json', selection)
    print('LOCKED METRICS '+json.dumps(selected, ensure_ascii=False), flush=True)
    if phase == 'discovery':
        if source_hash() != plan['source_sha256']:
            raise RuntimeError('Source changed during metrics discovery')
        return selection
    chosen_ledgers, control_ledgers = {}, {}
    for rule, destination in ((selected, chosen_ledgers), (control, control_ledgers)):
        destination['discovery'] = (simulate_frame(discovery, directions(discovery, rule),
            utc_ms(START), utc_ms(SPLIT), rule=rule)[0] if rule else [])
    del discovery
    gc.collect()
    later, later_meta = prediction_phase(SPLIT, END, base_identity, metrics_identity, raw_metrics, raw_quality)
    del raw_metrics
    gc.collect()
    ranges = {'validation': (utc_ms(SPLIT), utc_ms(CONFIRM)),
              'historical_confirmation': (utc_ms(CONFIRM), utc_ms(END)),
              'continuous_post_selection': (utc_ms(SPLIT), utc_ms(END))}
    augmented_matrix, control_matrix = [], []
    for number, (record, destination) in enumerate(
            [(r, augmented_matrix) for r in records]+[(r, control_matrix) for r in control_records], 1):
        rule = record['rule']; signal = directions(later, rule); measured = dict(record)
        for name, (first, stop) in ranges.items():
            ledger, stats = simulate_frame(later, signal, first, stop, rule=rule)
            measured[name] = stats
            if selected and rule['id'] == selected['id']:
                chosen_ledgers[name] = ledger
            if control and rule['id'] == control['id']:
                control_ledgers[name] = ledger
        _, stressed = simulate_frame(later, signal, utc_ms(SPLIT), utc_ms(END), stress={'adverse_bps': 2.}, rule=rule)
        measured['continuous_adverse_entry_2bps'] = stressed
        destination.append(measured)
        print(f'LATER {number}/16 augmented/control disclosed without reselection', flush=True)
    all_ranges = {'discovery': (utc_ms(START), utc_ms(SPLIT)), **ranges}
    summaries, baseline, paired, ledger_files = {}, {}, {}, []
    for name, (first, stop) in all_ranges.items():
        chosen = chosen_ledgers.get(name, []); base = control_ledgers.get(name, [])
        summaries[name], baseline[name] = detail(chosen, first, stop), detail(base, first, stop)
        paired[name] = paired_week_interval(chosen, base, first, stop)
        ledger_files.append(_save_ledger(out/f'metrics-search-{name}-trades.csv', chosen))
        ledger_files.append(_save_ledger(out/f'metrics-search-matched-control-{name}-trades.csv', base))
    stresses, stressed_periods = [], {}
    if selected:
        signal = directions(later, selected)
        for name, (first, stop) in ranges.items():
            ledger, _ = simulate_frame(later, signal, first, stop, stress={'adverse_bps': 2.}, rule=selected)
            stressed_periods[name] = stress_detail(ledger, first, stop)
            ledger_files.append(_save_ledger(out/f'metrics-search-adverse2bps-{name}-trades.csv', ledger))
        stresses.append({'name': 'adverse_index_entry_2bps', 'configuration': {'adverse_bps': 2.},
                         'test': stressed_periods['continuous_post_selection'], 'periods': stressed_periods})
    folds = discovery_meta['folds']+later_meta['folds']
    nominal, execution = historical_checks(selected, summaries, stressed_periods, folds)
    checks = {**nominal, **execution}
    if source_hash() != plan['source_sha256']:
        raise RuntimeError('Source changed during metrics research')
    if store_identity(verify_metrics_store()) != metrics_identity:
        raise RuntimeError('Metrics archive identity changed during research')
    result = {'schema_version': 1, 'engine': 'paired-base60-vs-position-metrics-v1',
              'plan': plan, 'selection': selection, 'selected': summaries, 'baseline': baseline,
              'all_candidates': augmented_matrix, 'matched_controls': control_matrix,
              'paired_comparison': paired, 'stresses': stresses, 'checks': checks,
              'nominal_history_gates_passed': all(nominal.values()),
              'execution_robustness_gates_passed': all(execution.values()),
              'historical_research_gates_passed': all(checks.values()),
              'incremental_historical_evidence_passed': incremental_evidence(selected, paired),
              'live_approved': False,
              'provenance': {'feature_store_identity': base_identity, 'metrics_store_identity': metrics_identity,
                             'source_sha256': plan['source_sha256'], 'plan_sha256': file_hash(out/'metrics-search-plan.json'),
                             'selection_sha256': file_hash(out/'metrics-search-selection.json'),
                             'ledger_files': ledger_files, 'raw_metrics_quality': raw_quality,
                             'discovery': discovery_meta, 'later': later_meta,
                             'runtime_seconds': round(time.perf_counter()-began, 2),
                             'generated_at_utc': datetime.now(timezone.utc).isoformat()},
              'limitations': [
                  'Prior rounds already explored 113 configurations on this history; new information does not make the historical holdout pristine.',
                  'Archive record time plus ten minutes is an explicit lag assumption; these files do not prove contemporaneous publication or absence of revisions.',
                  'Missing/stale/invalid metrics reduce the common population of both feature arms; no values or missing endpoints are filled.',
                  'Positive strategy profit alone does not show incremental metrics value; inspect the paired net interval and separate evidence flag.',
                  'All16 later candidate/control results are disclosed for audit; only eight augmented discovery candidates participate in selection.',
                  'Scores are not calibrated probabilities; overlapping targets and structural regime changes limit statistical confidence.',
                  'Minute index prices proxy actual contract strike/expiry; 2bp stress and known unavailable dates do not cover every execution risk.',
                  'No production code, live model, alert, order, GitHub remote or server was changed.']}
    write_json(out/'metrics-search-results.json', result)
    print(json.dumps({'selected': selected, 'checks': checks,
                      'incremental_historical_evidence_passed': result['incremental_historical_evidence_passed'],
                      'runtime_seconds': result['provenance']['runtime_seconds']}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'discovery'), default='all')
    run(parser.parse_args().phase)
