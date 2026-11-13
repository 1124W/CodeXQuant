"""Eight augmented decisions have exact controls and never read target prices."""
import math
import unittest

import numpy as np
import pandas as pd

from research.metrics_search_rules import candidates, directions, matched_control

MINUTE = 60_000
ORIGIN = int(pd.Timestamp('2025-01-06', tz='UTC').timestamp()*1000)


def row(minute=0, **changes):
    return {'symbol': 'BTCUSDT', 'signal_ms': ORIGIN+minute*MINUTE,
            'p_logistic_aug': .65, 'p_logistic_base': .35,
            'p_gbm_aug': .65, 'p_gbm_base': .35,
            'ret_60': -2*math.sqrt(60), 'rv_60': 1., 'rv_ratio': 1., **changes}


def selected(threshold=.60, regime='unrestricted'):
    return next(rule for rule in candidates() if rule['model'] == 'logistic'
                and rule['threshold'] == threshold and rule['regime'] == regime)


class MetricsSearchRuleTests(unittest.TestCase):
    def test_eight_augmented_candidates_and_exact_eight_unranked_controls(self):
        augmented = candidates(); all_rules = candidates(True)
        self.assertEqual(len(augmented), 8)
        self.assertEqual(len(all_rules), 16)
        self.assertEqual(len({r['id'] for r in all_rules}), 16)
        self.assertTrue(all(r['selection_candidate'] and r['feature_set'] == 'aug' for r in augmented))
        for rule in augmented:
            control = matched_control(rule)
            self.assertFalse(control['selection_candidate'])
            self.assertEqual(control['feature_set'], 'base')
            self.assertEqual({k: rule[k] for k in ('model', 'threshold', 'regime')},
                             {k: control[k] for k in ('model', 'threshold', 'regime')})
            with self.assertRaises(ValueError):
                matched_control(control)

    def test_control_uses_only_its_corresponding_base_score(self):
        frame = pd.DataFrame([row(), row(1, p_logistic_aug=.4, p_logistic_base=.7)])
        rule = selected()
        np.testing.assert_array_equal(directions(frame, rule), [1, -1])
        np.testing.assert_array_equal(directions(frame, matched_control(rule)), [-1, 1])

    def test_catalog_copies_and_conflicting_parameters_are_rejected(self):
        rule = selected()
        changed = candidates(); changed[0]['threshold'] = .50
        self.assertEqual(candidates()[0]['threshold'], .56)
        for mutation in ({'threshold': .50}, {'feature_set': 'base'}, {'model': 'other'},
                         {'selection_candidate': False}, {'new_filter': True}):
            with self.assertRaises(ValueError):
                directions(pd.DataFrame([row()]), {**rule, **mutation})

    def test_direction_exact_threshold_and_invalid_scores(self):
        values = [.6, .4, .59, .5, np.nan, np.inf, -.1, 1.1]
        frame = pd.DataFrame([row(i, p_logistic_aug=value) for i, value in enumerate(values)])
        np.testing.assert_array_equal(directions(frame, selected()), [1, -1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(directions(frame, selected()).dtype, np.int8)

    def test_exhaustion_is_the_original_reversal_volatility_condition(self):
        frame = pd.DataFrame([row(), row(1, ret_60=20.), row(2, ret_60=-.01),
                              row(3, rv_ratio=1.251), row(4, rv_60=-1.),
                              row(5, p_logistic_aug=.30, ret_60=20.)])
        np.testing.assert_array_equal(directions(frame, selected(regime='exhaustion')), [1, 0, 0, 0, 0, -1])

    def test_known_unavailable_calendar_is_identical_for_both_arms(self):
        day = int(pd.Timestamp('2026-01-17', tz='UTC').timestamp()*1000)
        frame = pd.DataFrame([row(0, signal_ms=day-32*MINUTE),
                              row(1, signal_ms=day-31*MINUTE), row(2, signal_ms=day)])
        rule = selected()
        np.testing.assert_array_equal(directions(frame, rule), [1, 0, 0])
        np.testing.assert_array_equal(directions(frame, matched_control(rule)), [-1, 0, 0])

    def test_prices_labels_metric_receipt_metadata_and_other_models_are_not_read(self):
        frame = pd.DataFrame([row(), row(1, symbol='ETHUSDT')])
        before = frame.copy(deep=True)
        expected = directions(frame, selected())
        pd.testing.assert_frame_equal(frame, before)
        for name in ('entry', 'exit', 'y', 'is_flat', 'entry_ms', 'settlement_ms', 'metrics_available_ms'):
            frame[name] = ['bad future value', object()]
        frame['p_gbm_aug'] = [-99., 99.]
        np.testing.assert_array_equal(directions(frame, selected()), expected)
        np.testing.assert_array_equal(directions(frame.iloc[::-1], selected()), expected[::-1])

    def test_scope_validation_and_unrestricted_inputs(self):
        for malformed in ([row(), row()], [row(symbol='SOLUSDT')], [row(signal_ms=ORIGIN+1)]):
            with self.assertRaises(ValueError):
                directions(pd.DataFrame(malformed), selected())
        frame = pd.DataFrame([row()]).drop(columns=['ret_60', 'rv_60', 'rv_ratio'])
        np.testing.assert_array_equal(directions(frame, selected()), [1])


if __name__ == '__main__':
    unittest.main()
