"""Paired cache provenance, candidate-only selection and incremental evidence."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import metrics_search_research as study
from research.event_engine import DAY_MS, MINUTE_MS, utc_ms


def observations(symbol, first):
    stamps = np.array([first, first+MINUTE_MS], dtype=np.int64)
    record = stamps//(5*MINUTE_MS)*(5*MINUTE_MS)-10*MINUTE_MS
    return pd.DataFrame({'symbol': symbol, 'signal_ms': stamps, 'entry_ms': stamps+MINUTE_MS,
        'settlement_ms': stamps+31*MINUTE_MS, 'entry': [100., 100.], 'exit': [101., 99.],
        'ret_60': [-.01, -.01], 'rv_60': [.001, .001], 'rv_ratio': [.8, .8],
        'metrics_record_ms': record, 'metrics_available_ms': record+10*MINUTE_MS,
        'metrics_age_ms': stamps-record-10*MINUTE_MS})


def fake_fit(train, test, cutoff, **unused):
    output = pd.DataFrame(index=test.index); models = {}; pairs = {}
    test_hash = hashlib.sha256(test.signal_ms.to_numpy(dtype='<i8').tobytes()).hexdigest()
    for family in ('logistic', 'gbm'):
        proof = {'training_count': 2001, 'prediction_count': len(test),
                 'train_signal_sha256': 'a'*64, 'train_label_sha256': 'b'*64,
                 'test_signal_sha256': test_hash}
        pair = {**proof, 'base_feature_count': 60, 'augmented_feature_count': 68,
                'identical_train_rows_and_labels': True, 'identical_test_rows': True}
        for arm in ('base', 'aug'):
            name = family+'_'+arm
            output['p_'+name] = .65
            models[name] = {**proof, 'latest_settlement_ms': cutoff-MINUTE_MS,
                'earliest_signal_ms': cutoff-180*DAY_MS, 'training_converged': True}
            pair[arm] = dict(proof)
        pairs[family] = pair
    metadata = {'fit_cutoff_ms': cutoff, 'prediction_count': len(test), 'train_days': 180, 'refit_days': 30,
                'test_start_ms': int(test.signal_ms.min()),
                'test_end_exclusive_ms': int(test.signal_ms.max())+MINUTE_MS,
                'models': models, 'pairs': pairs, 'specs': study.specs()}
    return output, metadata


def saved_fixture():
    first = utc_ms(study.START); frame = observations('BTCUSDT', first)
    scores, model = fake_fit(frame, frame, first)
    for name in scores:
        frame[name] = scores[name]
    frame['model_fit_cutoff_ms'] = first
    frame['train_latest_settlement_ms'] = first-MINUTE_MS
    metadata = {'symbol': 'BTCUSDT', 'first_signal_bound_ms': first, 'end_exclusive_ms': first+DAY_MS,
                'model_metadata': model, 'contract': {'metrics_store_identity': 'metrics-fixture'},
                'training_feature_quality': {'manifest_identity': 'metrics-fixture'},
                'inference_feature_quality': {'manifest_identity': 'metrics-fixture'}}
    return frame, metadata, first


class MetricsRunnerTests(unittest.TestCase):
    def test_cache_reuses_all_four_scores_and_does_not_refit_corrupt_or_stale(self):
        first = study.START; last = first+timedelta(days=1)
        raw = {symbol: pd.DataFrame() for symbol in study.SYMBOLS}
        quality = {symbol: {'manifest_identity': 'metrics-fixture'} for symbol in study.SYMBOLS}
        def enriched(symbol, a, b, **unused):
            return observations(symbol, a), {'manifest_identity': 'metrics-fixture', 'offered_base_rows': 2, 'retained_rows': 2}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(study, 'load_enriched_window', side_effect=enriched), \
                patch.object(study, 'fit_predict', side_effect=fake_fit) as fitter:
            folder = Path(directory)
            frame, meta = study.prediction_phase(first, last, 'base-fixture', 'metrics-fixture', raw, quality, folder)
            self.assertEqual(fitter.call_count, 2)
            again, again_meta = study.prediction_phase(first, last, 'base-fixture', 'metrics-fixture', raw, quality, folder)
            self.assertEqual(fitter.call_count, 2)
            pd.testing.assert_frame_equal(frame, again)
            self.assertEqual(meta, again_meta)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                study.prediction_phase(first, last, 'base-fixture', 'different', raw, quality, folder)
            path = next(folder.glob('*.npz'))
            with path.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                study.prediction_phase(first, last, 'base-fixture', 'metrics-fixture', raw, quality, folder)
            self.assertEqual(fitter.call_count, 2)

    def test_paired_row_label_and_test_hashes_must_match_all_arms_and_actual_rows(self):
        frame, metadata, first = saved_fixture()
        study._validate_prediction(frame, metadata, 'BTCUSDT', first, first, first+DAY_MS)
        for key in ('train_signal_sha256', 'train_label_sha256', 'test_signal_sha256'):
            changed = deepcopy(metadata)
            changed['model_metadata']['models']['logistic_aug'][key] = 'c'*64
            with self.assertRaisesRegex(ValueError, 'hashes differ'):
                study._validate_prediction(frame, changed, 'BTCUSDT', first, first, first+DAY_MS)
        changed = deepcopy(metadata)
        for family in ('logistic', 'gbm'):
            pair = changed['model_metadata']['pairs'][family]
            for record in (pair, pair['base'], pair['aug']):
                record['test_signal_sha256'] = 'c'*64
            for arm in ('base', 'aug'):
                changed['model_metadata']['models'][family+'_'+arm]['test_signal_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'actual rows'):
            study._validate_prediction(frame, changed, 'BTCUSDT', first, first, first+DAY_MS)

    def test_cache_rejects_unsettled_labels_stale_metrics_and_changed_manifest(self):
        frame, metadata, first = saved_fixture()
        bad = deepcopy(metadata)
        bad['model_metadata']['models']['gbm_aug']['latest_settlement_ms'] = first
        with self.assertRaisesRegex(ValueError, 'strictly settled'):
            study._validate_prediction(frame, bad, 'BTCUSDT', first, first, first+DAY_MS)
        stale = frame.copy()
        stale.loc[0, 'metrics_record_ms'] -= 5*MINUTE_MS
        stale.loc[0, 'metrics_available_ms'] -= 5*MINUTE_MS
        stale.loc[0, 'metrics_age_ms'] += 5*MINUTE_MS
        with self.assertRaisesRegex(ValueError, 'freshness'):
            study._validate_prediction(stale, metadata, 'BTCUSDT', first, first, first+DAY_MS)
        bad = deepcopy(metadata); bad['training_feature_quality']['manifest_identity'] = 'other'
        with self.assertRaisesRegex(ValueError, 'manifest'):
            study._validate_prediction(frame, bad, 'BTCUSDT', first, first, first+DAY_MS)

    def test_controls_never_enter_selection_and_lock_precedes_later_prediction(self):
        first = utc_ms(study.START)
        frame = pd.concat([observations(symbol, first) for symbol in study.SYMBOLS], ignore_index=True)
        scores, _ = fake_fit(frame, frame, first)
        for name in scores:
            frame[name] = scores[name]
        class LaterReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            folder = Path(directory)
            stack.enter_context(patch.object(study, 'source_hash', return_value='fixture-source'))
            stack.enter_context(patch.object(study, 'verified_store_identity', return_value='base-fixture'))
            stack.enter_context(patch.object(study, 'verify_metrics_store', return_value={'complete': True}))
            stack.enter_context(patch.object(study, 'store_identity', return_value='metrics-fixture'))
            stack.enter_context(patch.object(study, 'load_metrics', return_value=(pd.DataFrame(), {'manifest_identity': 'metrics-fixture'})))
            def choose(records):
                self.assertEqual(len(records), 8)
                self.assertTrue(all(record['rule']['feature_set'] == 'aug' for record in records))
                return records[0]['rule'], []
            stack.enter_context(patch.object(study, 'choose_discovery', side_effect=choose))
            def predict(start, end, *unused):
                self.assertTrue((folder/'metrics-search-plan.json').is_file())
                if start == study.SPLIT:
                    saved = json.loads((folder/'metrics-search-selection.json').read_text(encoding='utf-8'))
                    self.assertEqual(saved['selection_candidate_count'], 8)
                    self.assertFalse(saved['controls_eligible_for_selection'])
                    self.assertEqual(saved['matched_control'], study.matched_control(saved['selected_rule']))
                    raise LaterReached()
                self.assertFalse((folder/'metrics-search-selection.json').exists())
                return frame, {}
            predictor = stack.enter_context(patch.object(study, 'prediction_phase', side_effect=predict))
            with self.assertRaises(LaterReached):
                study.run(output_dir=folder)
            self.assertEqual(predictor.call_count, 2)

    def test_metrics_manifest_change_blocks_preloading_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(study, 'source_hash', return_value='fixture-source'))
            stack.enter_context(patch.object(study, 'verified_store_identity', return_value='base'))
            stack.enter_context(patch.object(study, 'verify_metrics_store', return_value={'complete': True}))
            stack.enter_context(patch.object(study, 'store_identity', return_value='metrics'))
            stack.enter_context(patch.object(study, 'load_metrics', return_value=(pd.DataFrame(), {'manifest_identity': 'changed'})))
            predictor = stack.enter_context(patch.object(study, 'prediction_phase'))
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                study.run(output_dir=directory)
            predictor.assert_not_called()

    def test_incremental_value_requires_positive_paired_net_lower_not_strategy_profit(self):
        chosen = {'id': 'fixed'}
        for lower in (None, -.1, 0.):
            self.assertFalse(study.incremental_evidence(chosen, {'continuous_post_selection': {'delta_net_lower_95': lower}}))
        self.assertFalse(study.incremental_evidence(None, {'continuous_post_selection': {'delta_net_lower_95': 100.}}))
        self.assertTrue(study.incremental_evidence(chosen, {'continuous_post_selection': {'delta_net_lower_95': .01}}))


if __name__ == '__main__':
    unittest.main()
