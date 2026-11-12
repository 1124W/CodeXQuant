"""Causality and publication-time checks for signed derivative inputs."""

import copy
import unittest

import numpy as np
import pandas as pd

from research.derivative_features import (
    FUNDING_FEATURES, FUNDING_PUBLICATION_LAG_MS, HOUR_MS,
    MAX_FUNDING_AGE_MS, MINUTE_MS, PREMIUM_FEATURES, add_derivative_features,
)


def premium_history(minutes=3000):
    t = np.arange(minutes)
    close = -.0003 + .00001 * np.sin(t / 7)
    return pd.DataFrame({"timestamp": t * MINUTE_MS, "open": close,
                         "high": close + .000001, "low": close - .000001,
                         "close": close})


def funding_history():
    return pd.DataFrame({"funding_time": [0, 2 * HOUR_MS, 4 * HOUR_MS + 1, 8 * HOUR_MS],
                         "rate": [-.0001, .0002, .0003, -.0004],
                         "interval_hours": [np.nan, 2., 2., 4.]})


def samples(signals=None):
    if signals is None:
        signals = np.arange(10, 48) * 30 * MINUTE_MS
    signals = np.asarray(signals, dtype=np.int64)
    frame = pd.DataFrame({"signal_ms": signals, "known": np.sin(signals / HOUR_MS),
                          "entry_ms": signals + MINUTE_MS,
                          "settlement_ms": signals + 31 * MINUTE_MS,
                          "entry": 100., "exit": 101., "y": 1, "is_flat": False})
    frame.attrs = {"features": ["known"], "profile": "research",
                   "reference": {"commit": "pinned"}}
    return frame


class DerivativeFeatureTests(unittest.TestCase):
    def test_signed_premium_six_features_formula_and_attrs(self):
        source = samples([5 * HOUR_MS])
        p, f = premium_history(), funding_history()
        actual = add_derivative_features(source, p, f)
        self.assertEqual(len(actual), 1)
        row = actual.iloc[0]
        observed = p.iloc[300-60:300].close.to_numpy() * 10_000
        self.assertLess(row.premium_close_bps, 0)
        self.assertAlmostEqual(row.premium_close_bps, observed[-1])
        self.assertAlmostEqual(row.premium_mean15_bps, observed[-15:].mean())
        self.assertAlmostEqual(row.premium_mean60_bps, observed.mean())
        self.assertAlmostEqual(row.premium_deviation60_bps, observed[-1] - observed.mean())
        self.assertAlmostEqual(row.premium_slope15_60_bps, observed[-15:].mean() - observed.mean())
        self.assertAlmostEqual(row.premium_std60_bps, observed.std(ddof=1))
        self.assertEqual(row.premium_observation_ms, 5 * HOUR_MS - MINUTE_MS)
        self.assertEqual(row.premium_close_available_ms, 5 * HOUR_MS)
        self.assertEqual(actual.attrs["features"], ["known"] + PREMIUM_FEATURES + FUNDING_FEATURES)
        self.assertEqual(actual.attrs["premium_features"], PREMIUM_FEATURES)
        self.assertEqual(actual.attrs["funding_features"], FUNDING_FEATURES)
        self.assertEqual(actual.attrs["reference"], source.attrs["reference"])
        self.assertEqual(source.attrs["features"], ["known"])
        self.assertNotIn("premium_close_bps", source)

    def test_funding_exact_millisecond_and_publication_lag(self):
        times = [4 * HOUR_MS, 4 * HOUR_MS + MINUTE_MS,
                 4 * HOUR_MS + 2 * MINUTE_MS, 8 * HOUR_MS, 8 * HOUR_MS + MINUTE_MS]
        actual = add_derivative_features(samples(times), premium_history(), funding_history())
        # Third observation at 04:00:00.001 is not available at 04:01:00.000.
        self.assertEqual(actual.signal_ms.tolist(), times[2:])
        np.testing.assert_allclose(actual.funding_rate_bps.to_numpy(), [3., 3., -4.])
        first = actual.iloc[0]
        self.assertEqual(first.funding_available_ms, 4 * HOUR_MS + MINUTE_MS + 1)
        self.assertAlmostEqual(first.funding_delta_bps, 1.)
        self.assertAlmostEqual(first.funding_mean3_bps, 4 / 3)
        self.assertAlmostEqual(first.funding_age_hours, (MINUTE_MS - 1) / HOUR_MS)
        self.assertEqual(actual.iloc[-1].funding_available_ms, 8 * HOUR_MS + MINUTE_MS)
        self.assertEqual(actual.iloc[-1].funding_age_hours, 0.)

    def test_funding_uses_actual_variable_intervals_not_interval_field(self):
        source = samples()
        f = funding_history()
        baseline = add_derivative_features(source, premium_history(), f)
        f["interval_hours"] = [99., np.nan, .001, 123.]
        actual = add_derivative_features(source, premium_history(), f)
        pd.testing.assert_frame_equal(baseline, actual)
        self.assertFalse(actual.attrs["derivative_features"]["funding_interval_used_as_feature"])

    def test_outcome_and_execution_columns_are_never_read_or_filtered(self):
        source = samples()
        baseline = add_derivative_features(source, premium_history(), funding_history())
        changed = copy.deepcopy(source)
        changed["entry"] = np.nan
        changed["exit"] = np.inf
        changed["y"] = np.nan
        changed["is_flat"] = True
        changed["entry_ms"] = -1
        changed["settlement_ms"] = -2
        actual = add_derivative_features(changed, premium_history(), funding_history())
        columns = ["signal_ms", "known"] + PREMIUM_FEATURES + FUNDING_FEATURES
        pd.testing.assert_frame_equal(baseline[columns], actual[columns])
        self.assertEqual(len(actual), len(baseline))
        self.assertTrue(actual.y.isna().all())
        self.assertTrue(actual.is_flat.all())
        without_targets = source.drop(columns=["entry", "exit", "y", "is_flat", "entry_ms", "settlement_ms"])
        no_targets = add_derivative_features(without_targets, premium_history(), funding_history())
        pd.testing.assert_frame_equal(baseline[columns], no_targets[columns])

    def test_future_premium_and_funding_mutation_cannot_change_past_factors(self):
        source, p, f = samples(), premium_history(), funding_history()
        cutoff = 8 * HOUR_MS
        baseline = add_derivative_features(source, p, f)
        changed_p, changed_f = p.copy(), f.copy()
        mask = changed_p.timestamp >= cutoff
        changed_p.loc[mask, ["open", "high", "low", "close"]] += 100.
        changed_f.loc[changed_f.funding_time >= cutoff, "rate"] = 10.
        actual = add_derivative_features(source, changed_p, changed_f)
        columns = ["signal_ms"] + PREMIUM_FEATURES + FUNDING_FEATURES
        pd.testing.assert_frame_equal(
            baseline.loc[baseline.signal_ms <= cutoff, columns].reset_index(drop=True),
            actual.loc[actual.signal_ms <= cutoff, columns].reset_index(drop=True))

    def test_prefix_invariance_without_future_source_rows(self):
        source, p, f = samples(), premium_history(), funding_history()
        cutoff = 8 * HOUR_MS
        baseline = add_derivative_features(source, p, f)
        prefix = add_derivative_features(source[source.signal_ms <= cutoff],
                                         p[p.timestamp < cutoff], f[f.funding_time < cutoff])
        columns = ["signal_ms"] + PREMIUM_FEATURES + FUNDING_FEATURES
        pd.testing.assert_frame_equal(
            baseline.loc[baseline.signal_ms <= cutoff, columns].reset_index(drop=True),
            prefix[columns].reset_index(drop=True))

    def test_premium_missing_internal_minute_is_rejected(self):
        p = premium_history().drop(index=299)
        with self.assertRaisesRegex(ValueError, "continuous"):
            add_derivative_features(samples(), p, funding_history())

    def test_declared_premium_gap_drops_full_affected_windows_without_filling(self):
        p = premium_history().drop(index=299)
        source = samples(np.array([300, 330, 359, 360, 390]) * MINUTE_MS)
        actual = add_derivative_features(source, p, funding_history(), allow_premium_gaps=True)
        self.assertEqual(actual.signal_ms.tolist(), [360 * MINUTE_MS, 390 * MINUTE_MS])
        self.assertTrue((actual.premium_gap_minutes60 == 0).all())
        metadata = actual.attrs["derivative_features"]
        self.assertTrue(metadata["allow_premium_gaps"])
        self.assertEqual(metadata["missing_premium_minutes"], 1)
        self.assertEqual(metadata["dropped_premium_gap_windows"], 3)
        self.assertEqual(metadata["dropped_premium_not_current"], 1)
        self.assertEqual(metadata["dropped_rows"], 3)
        self.assertIn("historical-only", metadata["feature_population_drop"])
        # Signal 06:00 sees the actual 05:00--05:59 observations, not a
        # previous-observation window compressed around the missing minute.
        expected = p.set_index("timestamp").loc[np.arange(300, 360) * MINUTE_MS, "close"] * 10_000
        self.assertAlmostEqual(actual.iloc[0].premium_mean60_bps, expected.mean())

    def test_declared_gap_handling_preserves_prefix_invariance(self):
        p = premium_history().drop(index=[299, 720])
        source = samples()
        baseline = add_derivative_features(source, p, funding_history(), allow_premium_gaps=True)
        cutoff = 10 * HOUR_MS
        prefix = add_derivative_features(source[source.signal_ms <= cutoff],
                                         p[p.timestamp < cutoff], funding_history(),
                                         allow_premium_gaps=True)
        columns = ["signal_ms"] + PREMIUM_FEATURES + FUNDING_FEATURES
        pd.testing.assert_frame_equal(
            baseline.loc[baseline.signal_ms <= cutoff, columns].reset_index(drop=True),
            prefix[columns].reset_index(drop=True))
        self.assertEqual(baseline.attrs["derivative_features"]["missing_premium_minutes"], 2)
        self.assertEqual(prefix.attrs["derivative_features"]["missing_premium_minutes"], 1)

    def test_allowing_gaps_still_rejects_duplicates_and_offgrid_times(self):
        p = premium_history()
        broken = p.copy()
        broken.loc[299, "timestamp"] += 1
        for invalid in (broken, pd.concat([p, p.iloc[[299]]])):
            with self.assertRaisesRegex(ValueError, "unique and clock aligned"):
                add_derivative_features(samples(), invalid, funding_history(), allow_premium_gaps=True)

    def test_premium_stale_end_and_uncompleted_future_bar_are_not_filled(self):
        # A bar beginning 05:00 is still open at the 05:00 signal.
        source = samples([5 * HOUR_MS, 5 * HOUR_MS + MINUTE_MS,
                          5 * HOUR_MS + 2 * MINUTE_MS])
        p = premium_history(301)
        actual = add_derivative_features(source, p, funding_history())
        self.assertEqual(actual.signal_ms.tolist(), source.signal_ms.tolist()[:2])
        self.assertEqual(actual.attrs["derivative_features"]["dropped_premium_not_current"], 1)
        self.assertEqual(actual.iloc[0].premium_observation_ms, 299 * MINUTE_MS)

    def test_first_59_premium_minutes_and_first_two_fundings_are_warmup(self):
        f = pd.DataFrame({"funding_time": [0, MINUTE_MS, 2 * MINUTE_MS],
                          "rate": [0., .0001, .0002]})
        source = samples(np.arange(1, 62) * MINUTE_MS)
        actual = add_derivative_features(source, premium_history(), f)
        self.assertEqual(actual.signal_ms.tolist(), [60 * MINUTE_MS, 61 * MINUTE_MS])
        metadata = actual.attrs["derivative_features"]
        self.assertEqual(metadata["dropped_premium_warmup"], 59)
        self.assertEqual(metadata["dropped_funding_warmup"], 2)

    def test_funding_staleness_after_24h_excludes_without_future_fill(self):
        f = pd.DataFrame({"funding_time": [0, 2 * HOUR_MS, 4 * HOUR_MS],
                          "rate": [0., .0001, .0002]})
        available = 4 * HOUR_MS + FUNDING_PUBLICATION_LAG_MS
        source = samples([available + MAX_FUNDING_AGE_MS,
                          available + MAX_FUNDING_AGE_MS + MINUTE_MS])
        actual = add_derivative_features(source, premium_history(), f)
        self.assertEqual(actual.signal_ms.tolist(), source.signal_ms.tolist()[:1])
        self.assertEqual(actual.iloc[0].funding_age_hours, 24.)
        self.assertEqual(actual.attrs["derivative_features"]["dropped_funding_unavailable_or_stale"], 1)

    def test_duplicate_times_bad_ohlc_and_nonfinite_rates_are_rejected(self):
        p, f = premium_history(), funding_history()
        cases = [pd.concat([p, p.iloc[[0]]]), p.assign(high=p.low - 1)]
        for broken in cases:
            with self.assertRaises(ValueError):
                add_derivative_features(samples(), broken, f)
        with self.assertRaisesRegex(ValueError, "unique"):
            add_derivative_features(samples(), p, pd.concat([f, f.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            add_derivative_features(samples(), p, f.assign(rate=np.inf))

    def test_target_cannot_be_declared_as_source_feature(self):
        source = samples()
        source.attrs["features"] = ["known", "y"]
        with self.assertRaisesRegex(ValueError, "historical"):
            add_derivative_features(source, premium_history(), funding_history())

    def test_input_metadata_copies_and_new_feature_collision_is_rejected(self):
        source = samples()
        actual = add_derivative_features(source, premium_history(), funding_history())
        actual.attrs["reference"]["commit"] = "changed"
        self.assertEqual(source.attrs["reference"]["commit"], "pinned")
        with self.assertRaisesRegex(ValueError, "already been added"):
            add_derivative_features(actual, premium_history(), funding_history())


if __name__ == "__main__":
    unittest.main()
