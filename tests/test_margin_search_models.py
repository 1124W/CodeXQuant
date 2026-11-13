"""Execution-margin targets, time purges and three-class output invariants."""
import unittest
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES
from research.margin_search_models import (DAY_MS, MARGIN_BPS, fit_predict,
                                           make_model, margin_labels, specs)
from research.model_search_models import specs as old_specs

MINUTE = 60000
CUTOFF = 21000*DAY_MS


def population():
    base = CUTOFF-20*DAY_MS+np.arange(3000, dtype=np.int64)*5*MINUTE
    stamps = np.sort(np.concatenate([base, base+MINUTE,
        [CUTOFF-181*DAY_MS, CUTOFF-35*MINUTE, CUTOFF-31*MINUTE,
         CUTOFF-30*MINUTE, CUTOFF, CUTOFF+MINUTE]]))
    identity = np.arange(len(stamps), dtype=float)
    train = pd.DataFrame({name: identity+position/100 for position, name in enumerate(FEATURES)})
    train['signal_ms'] = stamps; train['entry_ms'] = stamps+MINUTE
    train['settlement_ms'] = stamps+31*MINUTE
    train['entry'] = 100.; train['exit'] = 100.+(np.arange(len(stamps)) % 3-1)*.1
    train['symbol'] = 'BTCUSDT'
    train.index = np.arange(len(train))*2+1
    test = train.iloc[:5].copy(); test['signal_ms'] = CUTOFF+np.arange(5)*MINUTE
    test.index = [7, 2, 80, 3, 6]
    return train, test


class FakeModel:
    classes_ = np.array([-1, 0, 1])
    def __init__(self, name, records): self.name = name; self.records = records
    def fit(self, x, y):
        self.records[self.name] = {'x': x.copy(), 'y': y.copy()}
        self.rates = np.array([(y == label).mean() for label in self.classes_])
        return self
    def predict_proba(self, x):
        self.records[self.name]['test_x'] = x.copy()
        return np.tile(self.rates, (len(x), 1))


def factories(records, cls=FakeModel):
    return {name: (lambda model=name: cls(model, records)) for name in specs()}


class MarginSearchModelTests(unittest.TestCase):
    def test_fixed_two_models_retain_exact_prior_tree_parameters(self):
        self.assertEqual(len(specs()), 2)
        self.assertEqual(MARGIN_BPS, 2.)
        for key, previous in [('margin_shallow', 'gbm_shallow_5m'), ('margin_deep', 'gbm_deeper_5m')]:
            self.assertEqual(specs()[key]['parameters'], old_specs()[previous]['parameters'])
            self.assertEqual(specs()[key]['sample_minutes'], 5)
            model = make_model(key)
            self.assertFalse(model.early_stopping)
            self.assertEqual(model.min_samples_leaf, 300)
        changed = specs(); changed['margin_deep']['parameters']['max_depth'] = 99
        self.assertEqual(specs()['margin_deep']['parameters']['max_depth'], 3)
        with self.assertRaises(ValueError): make_model('logistic_halfhour')

    def test_deadzone_keeps_refunds_and_exact_boundaries_neutral(self):
        prices = np.array([100., 200., 300., 400., 500., 600., 700.])
        exits = prices*np.array([.9997, .9998, .9999, 1., 1.0001, 1.0002, 1.0003])
        np.testing.assert_array_equal(margin_labels(prices, exits), [-1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(margin_labels(prices, exits).dtype, np.int8)
        for opening, closing in [([0.], [1.]), ([1.], [np.nan]), ([1.], [-1.]), ([[1.]], [[1.]]), ([1.], [1., 2.])]:
            with self.assertRaises(ValueError): margin_labels(opening, closing)

    def test_only_180_day_settled_five_minute_rows_fit_and_neutrals_remain(self):
        train, test = population(); records = {}
        scores, meta = fit_predict(train, test, CUTOFF, factories=factories(records))
        self.assertEqual(list(scores.index), list(test.index))
        self.assertEqual(list(scores), [f'p_{model}_{label}' for model in specs() for label in ('up', 'down', 'neutral')])
        expected = train[(train.signal_ms >= CUTOFF-180*DAY_MS)
                         & (train.settlement_ms < CUTOFF) & (train.signal_ms % (5*MINUTE) == 0)]
        for model, record in records.items():
            used = train.iloc[record['x'][:, 0].astype(int)]
            self.assertEqual(len(used), len(expected))
            self.assertTrue((used.signal_ms % (5*MINUTE) == 0).all())
            self.assertTrue((used.signal_ms >= CUTOFF-180*DAY_MS).all())
            self.assertTrue((used.settlement_ms < CUTOFF).all())
            self.assertEqual(record['x'].shape[1], 60)
            self.assertGreater(int((record['y'] == 0).sum()), 0)
            self.assertLess(meta['models'][model]['latest_settlement_ms'], CUTOFF)
            self.assertEqual(sum(meta['models'][model]['class_counts'].values()), len(used))
        self.assertFalse(meta['raw_scores_are_calibrated_win_probabilities'])
        self.assertTrue(meta['local_research_only'])

    def test_future_and_too_old_training_prices_cannot_change_fit(self):
        train, test = population(); records = {}
        first, _ = fit_predict(train, test, CUTOFF, factories=factories(records))
        changed = train.copy()
        unused = (changed.settlement_ms >= CUTOFF) | (changed.signal_ms < CUTOFF-180*DAY_MS)
        changed.loc[unused, FEATURES] = np.nan
        changed.loc[unused, ['entry', 'exit']] = np.nan
        again = {}; second, _ = fit_predict(changed, test, CUTOFF, factories=factories(again))
        pd.testing.assert_frame_equal(first, second)
        for model in records:
            np.testing.assert_array_equal(records[model]['x'], again[model]['x'])
            np.testing.assert_array_equal(records[model]['y'], again[model]['y'])

    def test_prediction_outcomes_execution_clocks_and_binary_labels_are_not_read(self):
        train, test = population()
        first, _ = fit_predict(train, test, CUTOFF, factories=factories({}))
        changed = test.drop(columns=['entry', 'exit', 'entry_ms', 'settlement_ms'])
        changed['y'] = 99; changed['is_flat'] = True
        second, _ = fit_predict(train, changed, CUTOFF, factories=factories({}))
        pd.testing.assert_frame_equal(first, second)
        changed['entry'] = -np.inf; changed['exit'] = np.nan
        changed['entry_ms'] = -1; changed['settlement_ms'] = 0
        third, _ = fit_predict(train, changed, CUTOFF, factories=factories({}))
        pd.testing.assert_frame_equal(first, third)

    def test_permuted_class_order_is_mapped_to_correct_semantic_columns(self):
        class Permuted(FakeModel):
            classes_ = np.array([1, -1, 0])
            def predict_proba(self, x): return np.tile([.6, .3, .1], (len(x), 1))
        train, test = population(); scores, meta = fit_predict(train, test, CUTOFF, factories=factories({}, Permuted))
        for name in specs():
            np.testing.assert_allclose(scores[f'p_{name}_up'], .6)
            np.testing.assert_allclose(scores[f'p_{name}_down'], .3)
            np.testing.assert_allclose(scores[f'p_{name}_neutral'], .1)
            self.assertEqual(meta['models'][name]['class_order'], [1, -1, 0])

    def test_invalid_estimator_classes_and_probability_mass_are_rejected(self):
        train, test = population()
        for labels in ([0, 1], [-1, 0, 2], [-1, -1, 1], []):
            class InvalidClasses(FakeModel):
                classes_ = np.asarray(labels)
                def predict_proba(self, x): return np.tile([.3, .3, .4], (len(x), 1))
            with self.assertRaisesRegex(ValueError, 'exactly the down/neutral/up'):
                fit_predict(train, test, CUTOFF, factories=factories({}, InvalidClasses))
        for values in ([.2, .2, .2], [np.nan, .4, .6], [-.1, .5, .6], [1.1, 0., -.1]):
            class InvalidMass(FakeModel):
                def predict_proba(self, x): return np.tile(values, (len(x), 1))
            with self.assertRaisesRegex(ValueError, 'Invalid three-class'):
                fit_predict(train, test, CUTOFF, factories=factories({}, InvalidMass))

    def test_missing_target_class_and_invalid_historical_prices_fail_closed(self):
        train, test = population(); changed = train.copy(); changed['exit'] = 100.
        with self.assertRaisesRegex(ValueError, 'requires all'):
            fit_predict(changed, test, CUTOFF, factories=factories({}))
        selected = train.index[(train.signal_ms % (5*MINUTE) == 0) & (train.settlement_ms < CUTOFF)
                               & (train.signal_ms >= CUTOFF-180*DAY_MS)][0]
        changed = train.copy(); changed.loc[selected, 'entry'] = 0.
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            fit_predict(changed, test, CUTOFF, factories=factories({}))

    def test_wrong_clock_asset_missing_feature_and_nonfinite_input_fail_closed(self):
        train, test = population()
        changed = train.copy(); changed['entry_ms'] += MINUTE
        with self.assertRaisesRegex(ValueError, 'one-minute entry'):
            fit_predict(changed, test, CUTOFF, factories=factories({}))
        changed = test.copy(); changed['symbol'] = 'ETHUSDT'
        with self.assertRaisesRegex(ValueError, 'assets differ'):
            fit_predict(train, changed, CUTOFF, factories=factories({}))
        changed = test.copy(); changed.loc[changed.index[0], 'signal_ms'] = CUTOFF+30*DAY_MS
        with self.assertRaisesRegex(ValueError, 'outside'):
            fit_predict(train, changed, CUTOFF, factories=factories({}))
        changed = test.copy(); changed.loc[changed.index[0], FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite prediction'):
            fit_predict(train, changed, CUTOFF, factories=factories({}))
        with self.assertRaisesRegex(ValueError, 'Missing prediction features'):
            fit_predict(train, test.drop(columns=FEATURES[:1]), CUTOFF, factories=factories({}))

    def test_convergence_warnings_are_recorded_without_altering_parameters(self):
        from sklearn.exceptions import ConvergenceWarning
        class Unconverged(FakeModel):
            n_iter_ = 140
            def fit(self, x, y):
                warnings.warn('synthetic convergence failure', ConvergenceWarning)
                return super().fit(x, y)
        train, test = population(); _, meta = fit_predict(train, test, CUTOFF, factories=factories({}, Unconverged))
        self.assertTrue(all(not m['training_converged'] for m in meta['models'].values()))
        self.assertTrue(all(m['convergence_warning_count'] == 1 for m in meta['models'].values()))
        self.assertEqual(meta['specs'], specs())


if __name__ == '__main__': unittest.main()
