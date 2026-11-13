"""Synthetic tests for the fixed margin target decision rules."""
import math
import unittest

import numpy as np
import pandas as pd

from research.margin_search_rules import MODEL_IDS, candidates, directions

MINUTE = 60_000
ORIGIN = int(pd.Timestamp('2025-01-06', tz='UTC').timestamp()*1000)


def row(minute=0, up=.65, down=.25, neutral=.10, **changes):
    return {'symbol': 'BTCUSDT', 'signal_ms': ORIGIN+minute*MINUTE,
            **{f'p_{model}_{side}': value for model in MODEL_IDS
               for side, value in [('up', up), ('down', down), ('neutral', neutral)]},
            'ret_60': -2*math.sqrt(60), 'rv_60': 1., 'rv_ratio': 1., **changes}


def selected(threshold=.60, regime='unrestricted'):
    return next(rule for rule in candidates() if rule['model'] == 'margin_shallow'
                and rule['threshold'] == threshold and rule['regime'] == regime)


class MarginSearchRuleTests(unittest.TestCase):
    def test_catalog_is_fixed_copied_and_mutations_rejected(self):
        rules = candidates()
        self.assertEqual(len(rules), 8)
        self.assertEqual(len({r['id'] for r in rules}), 8)
        self.assertTrue(all(r['margin_bps'] == 2. for r in rules))
        self.assertTrue(all(not r['score_is_calibrated_win_probability'] for r in rules))
        rules[0]['threshold'] = .50
        self.assertEqual(candidates()[0]['threshold'], .56)
        for mutation in ({'threshold': .50}, {'margin_bps': 1.}, {'model': 'other'},
                         {'regime': 'trend'}, {'conditioning_on_non_neutral': True}):
            with self.assertRaises(ValueError):
                directions(pd.DataFrame([row()]), {**selected(), **mutation})
        with self.assertRaises(ValueError):
            directions(pd.DataFrame([row()]), 'unknown')

    def test_direction_uses_larger_score_and_exact_absolute_threshold(self):
        frame = pd.DataFrame([row(0, .60, .30, .10), row(1, .30, .60, .10),
                              row(2, .56, .34, .10), row(3, .40, .40, .20)])
        np.testing.assert_array_equal(directions(frame, selected()), [1, -1, 0, 0])
        np.testing.assert_array_equal(directions(frame, selected(.56)), [1, -1, 1, 0])
        self.assertEqual(directions(frame, selected()).dtype, np.int8)

    def test_neutral_is_not_removed_and_direction_scores_are_not_renormalized(self):
        frame = pd.DataFrame([row(0, .45, .05, .50), row(1, .05, .45, .50),
                              row(2, .60, .05, .35)])
        np.testing.assert_array_equal(directions(frame, selected()), [0, 0, 1])

    def test_invalid_probability_simplex_fails_closed(self):
        frame = pd.DataFrame([row(0, np.nan, .20, .10), row(1, np.inf, .20, .10),
                              row(2, .90, .20, .10), row(3, 1.01, -.01, 0.),
                              row(4, .80, .20, -.01)])
        self.assertFalse(directions(frame, selected()).any())

    def test_exhaustion_regime_preserves_reversal_and_volatility_requirements(self):
        frame = pd.DataFrame([row(0), row(1, ret_60=20.), row(2, ret_60=-.01),
                              row(3, rv_ratio=1.251), row(4, rv_60=-1.),
                              row(5, .20, .70, .10, ret_60=20.)])
        np.testing.assert_array_equal(directions(frame, selected(regime='exhaustion')), [1, 0, 0, 0, 0, -1])

    def test_known_unavailable_date_and_exact_expiry_boundary_are_masked(self):
        unavailable = int(pd.Timestamp('2026-01-17', tz='UTC').timestamp()*1000)
        frame = pd.DataFrame([row(0, signal_ms=unavailable-32*MINUTE),
                              row(1, signal_ms=unavailable-31*MINUTE),
                              row(2, signal_ms=unavailable)])
        np.testing.assert_array_equal(directions(frame, selected()), [1, 0, 0])

    def test_future_labels_prices_and_other_models_cannot_change_a_signal(self):
        frame = pd.DataFrame([row(), row(1, symbol='ETHUSDT')])
        before = frame.copy(deep=True)
        expected = directions(frame, selected())
        pd.testing.assert_frame_equal(frame, before)
        for column in ('entry', 'exit', 'entry_ms', 'settlement_ms', 'y', 'is_flat'):
            frame[column] = ['malformed future', object()]
        frame['p_margin_deep_up'] = [-999., 999.]
        np.testing.assert_array_equal(directions(frame, selected()), expected)
        np.testing.assert_array_equal(directions(frame.iloc[::-1], selected()), expected[::-1])

    def test_scope_checks_and_unrestricted_needs_no_regime_inputs(self):
        for frame in (pd.DataFrame([row(), row()]), pd.DataFrame([row(symbol='SOLUSDT')]),
                      pd.DataFrame([row(signal_ms=ORIGIN+1)])):
            with self.assertRaises(ValueError):
                directions(frame, selected())
        frame = pd.DataFrame([row()]).drop(columns=['ret_60', 'rv_60', 'rv_ratio'])
        np.testing.assert_array_equal(directions(frame, selected()), [1])


if __name__ == '__main__':
    unittest.main()
