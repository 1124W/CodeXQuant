"""Causal filter behavior under chronology, missing observations and asset joins."""
import unittest

import numpy as np
import pandas as pd

from research.high_win_filters import candidates, filter_mask

MINUTE = 60_000
ORIGIN = 1_800_000_000_000


def observation(minute=0, symbol="BTCUSDT", **changes):
    return {"symbol": symbol, "signal_ms": ORIGIN + minute * MINUTE,
            "p_up": .63, "ret_60": -30., "rv_60": 1., "rv_ratio": .9,
            "ret_240": 5., "rv_240": 1., "ret_3": 1., "imb_ma5": .2,
            "imb_ma60": .1, "clv": .3, "peer_imb_ma15": .1,
            "premium_close_bps": -1., **changes}


class HighWinFilterTests(unittest.TestCase):
    def test_catalog_is_fixed_and_returned_rules_cannot_mutate_it(self):
        rules = candidates()
        self.assertEqual(len(rules), 16)
        self.assertEqual(len({row["id"] for row in rules}), 16)
        rules[0]["id"] = "changed"
        self.assertEqual(candidates()[0]["id"], "baseline")
        with self.assertRaises(ValueError):
            filter_mask(pd.DataFrame([observation()]), {"id": "changed"})

    def test_all_filters_retain_existing_model_direction_and_exhaustion_gate(self):
        frame = pd.DataFrame([observation(0, p_up=.52), observation(1, ret_60=30),
                              observation(2, ret_60=-1), observation(3, rv_ratio=1.6)])
        for rule in candidates():
            with self.subTest(rule=rule["id"]):
                self.assertFalse(filter_mask(frame, rule).any())

    def test_more_extreme_model_score_does_not_get_labeled_calibrated_probability(self):
        frame = pd.DataFrame([observation(0, p_up=.59), observation(1, p_up=.61),
                              observation(2, p_up=.65)])
        np.testing.assert_array_equal(filter_mask(frame, "score_060"), [False, True, True])
        np.testing.assert_array_equal(filter_mask(frame, "score_064"), [False, False, True])

    def test_price_and_flow_confirmation_is_direction_symmetric(self):
        frame = pd.DataFrame([observation(0), observation(1, imb_ma5=-.2),
                              observation(2, p_up=.35, ret_60=30, ret_3=-1, imb_ma5=-.2)])
        np.testing.assert_array_equal(filter_mask(frame, "price_and_flow_turn"), [True, False, True])
        np.testing.assert_array_equal(filter_mask(frame, "contrarian_premium"), [True, True, False])

    def test_persistence_uses_prior_minute_same_asset_and_preserves_input_order(self):
        frame = pd.DataFrame([observation(2), observation(1, "ETHUSDT"),
                              observation(0), observation(1), observation(2, "ETHUSDT")])
        np.testing.assert_array_equal(filter_mask(frame, "persistent_2m"), [True, False, False, True, True])

    def test_missing_minute_failed_gate_or_changed_direction_breaks_persistence(self):
        frame = pd.DataFrame([observation(0), observation(2), observation(3, p_up=.5),
                              observation(4), observation(5, p_up=.35, ret_60=30)])
        self.assertFalse(filter_mask(frame, "persistent_2m").any())

    def test_execution_prices_and_future_outcomes_cannot_change_any_past_filter(self):
        past = pd.DataFrame([observation(0), observation(1), observation(2)])
        changed = past.copy()
        changed["entry"] = [1e9, 0, -1]
        changed["exit"] = [-1e9, 0, 1]
        changed["y"] = [False, True, False]
        changed["net_units"] = [1e12, -1e12, 99]
        changed = pd.concat([changed, pd.DataFrame([observation(99, p_up=.01, ret_60=1e6)])], ignore_index=True)
        for rule in candidates():
            with self.subTest(rule=rule["id"]):
                np.testing.assert_array_equal(filter_mask(past, rule), filter_mask(changed, rule)[:3])

    def test_invalid_indicator_fails_closed_and_duplicate_or_nonminute_time_fails(self):
        frame = pd.DataFrame([observation(0, ret_3=np.nan), observation(1, p_up=np.inf)])
        self.assertFalse(filter_mask(frame, "price_turn_3m").any())
        with self.assertRaises(ValueError):
            filter_mask(pd.DataFrame([observation(), observation()]), "baseline")
        with self.assertRaises(ValueError):
            filter_mask(pd.DataFrame([observation(signal_ms=ORIGIN + 1)]), "baseline")
        with self.assertRaises(ValueError):
            filter_mask(pd.DataFrame([observation(symbol="OTHER")]), "baseline")


if __name__ == "__main__":
    unittest.main()
