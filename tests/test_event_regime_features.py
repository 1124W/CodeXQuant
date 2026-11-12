"""Causality and contemporaneous peer-alignment checks for regime inputs."""

import copy
import unittest

import numpy as np
import pandas as pd

from research.event_regime_features import ADDED_FEATURES, enrich_samples
from research.workbuddy_strategy import RESEARCH_FEATURES


def samples(count=12):
    result = {}
    for multiplier, symbol in enumerate(("BTCUSDT", "ETHUSDT"), start=1):
        t = np.arange(count, dtype=float)
        frame = pd.DataFrame({name: .1 + .01 * t for name in RESEARCH_FEATURES})
        frame["ret_15"] = multiplier * (1 + t)
        frame["ret_30"] = multiplier * (2 + t)
        frame["ret_60"] = multiplier * (4 + t)
        frame["rv_60"] = multiplier * 2.
        frame["rv_ratio"] = 1.2
        frame["rv_ratio_long"] = .8
        frame["dist_hi"] = -3.
        frame["dist_lo"] = 1.
        frame["imb_ma5"] = .3
        frame["imb_ma15"] = .2
        frame["imb_ma60"] = .1
        frame["signal_ms"] = 1_800_000 * (1 + np.arange(count))
        frame["entry_ms"] = frame["signal_ms"] + 60_000
        frame["settlement_ms"] = frame["entry_ms"] + 1_800_000
        frame["entry"] = 100.
        frame["exit"] = 101.
        frame["y"] = 1
        frame["is_flat"] = False
        frame["symbol"] = symbol
        frame.attrs = {"features": RESEARCH_FEATURES.copy(), "profile": "research",
                       "quantdinger": {"custom_factor_count": 38}}
        result[symbol] = frame
    return result


class EventRegimeFeatureTests(unittest.TestCase):
    def test_twelve_features_have_expected_units_and_formula(self):
        source = samples()
        actual = enrich_samples(source)["BTCUSDT"]
        self.assertEqual(len(ADDED_FEATURES), 12)
        self.assertEqual(actual.attrs["features"], RESEARCH_FEATURES + ADDED_FEATURES)
        self.assertEqual(actual.attrs["profile"], "research")
        self.assertEqual(actual.attrs["quantdinger"]["custom_factor_count"], 38)
        row = actual.iloc[0]
        self.assertAlmostEqual(row.norm_ret_15, 1 / (2 * np.sqrt(15)))
        self.assertAlmostEqual(row.norm_ret_30, 2 / (2 * np.sqrt(30)))
        self.assertAlmostEqual(row.norm_ret_60, 4 / (2 * np.sqrt(60)))
        self.assertAlmostEqual(row.range_position_120, -.25)
        self.assertAlmostEqual(row.flow_momentum_15, .2 * row.norm_ret_15)
        self.assertAlmostEqual(row.flow_slope_5_60, .2)
        self.assertAlmostEqual(row.volatility_acceleration, .4)
        self.assertAlmostEqual(row.relative_norm_ret_60, (4 - 8) / (np.sqrt(20) * np.sqrt(60)))
        self.assertEqual(row.peer_ret_60, 8.)
        self.assertEqual(row.peer_imb_ma15, .2)
        self.assertEqual(row.peer_rv_ratio, 1.2)

    def test_future_execution_and_labels_are_never_read_or_filtered(self):
        source = samples()
        baseline = enrich_samples(source)
        changed = copy.deepcopy(source)
        for frame in changed.values():
            frame["entry"] = np.nan
            frame["exit"] = np.inf
            frame["y"] = np.nan
            frame["is_flat"] = True
            frame["entry_ms"] = -1
            frame["settlement_ms"] = -2
        actual = enrich_samples(changed)
        for symbol in source:
            columns = ["signal_ms"] + RESEARCH_FEATURES + ADDED_FEATURES
            pd.testing.assert_frame_equal(baseline[symbol][columns], actual[symbol][columns])
            self.assertEqual(len(actual[symbol]), len(source[symbol]))
            self.assertTrue(actual[symbol].is_flat.all())
            self.assertTrue(actual[symbol].y.isna().all())
            self.assertEqual(actual[symbol].attrs["regime_features"]["dropped_nonfinite_historical_rows"], 0)

    def test_prefix_does_not_change_when_later_feature_values_change(self):
        source = samples()
        baseline = enrich_samples(source)
        changed = copy.deepcopy(source)
        for frame in changed.values():
            frame.loc[6:, RESEARCH_FEATURES] *= 200
        actual = enrich_samples(changed)
        prefix = enrich_samples({symbol: frame.iloc[:6].copy() for symbol, frame in source.items()})
        for symbol in source:
            pd.testing.assert_frame_equal(baseline[symbol].iloc[:6], actual[symbol].iloc[:6])
            pd.testing.assert_frame_equal(baseline[symbol].iloc[:6].reset_index(drop=True), prefix[symbol])

    def test_exact_peer_intersection_never_fills_from_later_rows(self):
        source = samples()
        source["ETHUSDT"] = source["ETHUSDT"].drop(index=[0, 4])
        actual = enrich_samples(source)
        for symbol in actual:
            self.assertEqual(len(actual[symbol]), 10)
            self.assertNotIn(1_800_000, actual[symbol].signal_ms.to_list())
            self.assertNotIn(5 * 1_800_000, actual[symbol].signal_ms.to_list())
        btc = actual["BTCUSDT"].iloc[0]
        self.assertEqual(btc.signal_ms, 2 * 1_800_000)
        self.assertEqual(btc.peer_ret_60, 10.)
        self.assertEqual(actual["BTCUSDT"].attrs["regime_features"]["dropped_unaligned_rows"], 2)

    def test_inputs_and_nested_attrs_are_not_mutated(self):
        source = samples()
        before = copy.deepcopy(source)
        actual = enrich_samples(source)
        actual["BTCUSDT"].attrs["quantdinger"]["custom_factor_count"] = -1
        for symbol in source:
            pd.testing.assert_frame_equal(before[symbol], source[symbol])
            self.assertEqual(before[symbol].attrs, source[symbol].attrs)

    def test_only_nonfinite_historical_rows_are_dropped(self):
        source = samples()
        source["BTCUSDT"].loc[1, "rv_60"] = np.nan
        source["ETHUSDT"].loc[2, "ret_60"] = np.inf
        actual = enrich_samples(source)
        for symbol in actual:
            self.assertEqual(len(actual[symbol]), 10)
            self.assertTrue(np.isfinite(actual[symbol][actual[symbol].attrs["features"]]).all().all())
            self.assertEqual(actual[symbol].attrs["regime_features"]["dropped_nonfinite_historical_rows"], 2)

    def test_zero_volatility_and_flat_range_remain_finite(self):
        source = samples()
        for frame in source.values():
            frame[["rv_60", "dist_hi", "dist_lo"]] = 0.
        actual = enrich_samples(source)
        for frame in actual.values():
            self.assertEqual(len(frame), 12)
            self.assertTrue(np.isfinite(frame[ADDED_FEATURES]).all().all())
            self.assertTrue(frame.range_position_120.eq(0).all())

    def test_duplicate_or_fractional_timestamps_are_rejected(self):
        source = samples()
        source["BTCUSDT"].loc[1, "signal_ms"] = source["BTCUSDT"].loc[0, "signal_ms"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            enrich_samples(source)
        source = samples()
        source["BTCUSDT"]["signal_ms"] = source["BTCUSDT"].signal_ms.astype(float) + .5
        with self.assertRaisesRegex(ValueError, "half-hour"):
            enrich_samples(source)

    def test_target_feature_in_attribute_list_is_rejected(self):
        source = samples()
        source["BTCUSDT"].attrs["features"] += ["y"]
        with self.assertRaisesRegex(ValueError, "historical WorkBuddy"):
            enrich_samples(source)

    def test_missing_peer_symbol_or_no_common_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            enrich_samples({"BTCUSDT": samples()["BTCUSDT"]})
        source = samples()
        source["ETHUSDT"]["signal_ms"] += 100 * 1_800_000
        with self.assertRaisesRegex(ValueError, "no common"):
            enrich_samples(source)


if __name__ == "__main__":
    unittest.main()
