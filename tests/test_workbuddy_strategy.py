"""Causality, event-price and rolling-model boundary regression tests."""

import hashlib
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.event_engine import DAY_MS, MINUTE_MS, utc_ms
from research.qd_factors import DEFAULT_UPSTREAM_DIR
from research.workbuddy_strategy import (
    DEFAULT_REFERENCE, FEATURES, MODEL_PARAMETERS, PARAMETERS, RESEARCH_FEATURES,
    RESEARCH_MODEL_PARAMETERS, build_samples, load_reference, make_model,
    reference_provenance, walk_forward_signals,
)


SOURCE_READY = ((DEFAULT_REFERENCE / "datahub" / "features.py").exists()
                and (DEFAULT_UPSTREAM_DIR / ".git").exists())


def bars(count=3600):
    t = np.arange(count)
    opened = 100 + .003 * t + .5 * np.sin(t / 17)
    close = opened + .03 * np.sin(t / 7)
    volume = 20 + t % 19 + np.sin(t / 6)
    f = pd.DataFrame({"timestamp": utc_ms(date(2025, 1, 1)) + t * MINUTE_MS,
                      "open": opened, "high": np.maximum(opened, close) + .03,
                      "low": np.minimum(opened, close) - .03, "close": close,
                      "volume": volume, "quote_volume": volume * close,
                      "trades": 50 + t % 13, "taker_buy_base": volume * (.5 + .2 * np.sin(t / 11)),
                      "taker_buy_quote": volume * close * (.5 + .2 * np.sin(t / 11))})
    index = f[["timestamp", "open", "high", "low", "close"]].copy()
    return f, index


def model_samples(days=76):
    origin = utc_ms(date(2025, 2, 1))
    n = days * 48
    signals = origin + np.arange(n) * 30 * MINUTE_MS
    f = pd.DataFrame({name: np.arange(n, dtype=float) * .001 for name in FEATURES})
    f["ret_60"] = -30.
    f["signal_ms"] = signals
    f["entry_ms"] = signals + MINUTE_MS
    f["settlement_ms"] = signals + 31 * MINUTE_MS
    f["entry"] = 100.
    f["exit"] = np.where(np.arange(n) % 2, 101., 99.)
    f["y"] = (f["exit"] > f["entry"]).astype(int)
    f["is_flat"] = False
    return f, origin


class FakeModel:
    fitted = []

    def fit(self, x, y):
        self.fitted.append((x.copy(), y.copy()))
        return self

    def predict_proba(self, x):
        return np.tile([.2, .8], (len(x), 1))


@unittest.skipUnless(SOURCE_READY, "Download the pinned WorkBuddy and QuantDinger research sources first")
class WorkBuddyStrategyTests(unittest.TestCase):
    def test_pinned_reference_and_exact_three_condition_function(self):
        provenance = reference_provenance()
        self.assertEqual(provenance["commit"], "1f4e26e948ff566a5307d821e811aa7d6228928b")
        _, decide, _ = load_reference()
        self.assertEqual(decide(.8, -30, **PARAMETERS["BTCUSDT"])["action"], "long")
        self.assertEqual(decide(.2, 30, **PARAMETERS["ETHUSDT"])["action"], "short")
        self.assertEqual(decide(.6, -30, **PARAMETERS["BTCUSDT"])["action"], "skip")
        self.assertEqual(decide(.8, 30, **PARAMETERS["BTCUSDT"])["action"], "skip")
        self.assertEqual(decide(.8, -22, **PARAMETERS["ETHUSDT"])["action"], "skip")

    def test_modified_source_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "datahub" / "features.py"
            file.parent.mkdir()
            file.write_text("raise RuntimeError('untrusted')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned Git blob"):
                reference_provenance(folder)

    def test_closed_half_hour_samples_use_delayed_index_opens(self):
        spot, index = bars()
        samples = build_samples(spot, index, "BTCUSDT")
        self.assertGreater(len(samples), 50)
        self.assertTrue((samples.signal_ms % (30 * MINUTE_MS) == 0).all())
        self.assertTrue((samples.entry_ms == samples.signal_ms + MINUTE_MS).all())
        self.assertTrue((samples.settlement_ms == samples.entry_ms + 30 * MINUTE_MS).all())
        for row in samples.iloc[:5].itertuples():
            entry = index.loc[index.timestamp == row.entry_ms, "open"].iloc[0]
            exit_price = index.loc[index.timestamp == row.settlement_ms, "open"].iloc[0]
            self.assertEqual(row.entry, entry)
            self.assertEqual(row.exit, exit_price)
        self.assertEqual(samples.attrs["quantdinger"]["custom_factor_count"], 28)
        self.assertTrue(samples.attrs["quantdinger"]["ema_parity_passed"])

    def test_chunking_and_future_mutation_do_not_change_past_features(self):
        spot, index = bars()
        baseline = build_samples(spot, index, "BTCUSDT")
        with patch("research.workbuddy_strategy.FEATURE_CHUNK_SIZE", 600):
            chunked = build_samples(spot, index, "BTCUSDT")
        pd.testing.assert_frame_equal(baseline, chunked, rtol=1e-8, atol=1e-8)
        changed = spot.copy()
        boundary = int(spot.timestamp.iloc[2400])
        changed.loc[changed.timestamp >= boundary, ["open", "high", "low", "close"]] *= 2
        mutated = build_samples(changed, index, "BTCUSDT")
        columns = ["signal_ms"] + FEATURES
        a = baseline.loc[baseline.signal_ms <= boundary, columns].reset_index(drop=True)
        b = mutated.loc[mutated.signal_ms <= boundary, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, rtol=1e-8, atol=1e-8)

    def test_future_flat_events_are_retained_at_prediction_time(self):
        spot, index = bars()
        index[["open", "high", "low", "close"]] = 100.
        samples = build_samples(spot, index, "ETHUSDT")
        self.assertGreater(len(samples), 50)
        self.assertTrue(samples.is_flat.all())
        self.assertEqual(samples.attrs["flat_events_retained"], len(samples))

    def test_training_labels_are_purged_and_later_outcomes_cannot_change_predictions(self):
        samples, start = model_samples()
        FakeModel.fitted = []
        result = walk_forward_signals(samples, "BTCUSDT", start, start + 75 * DAY_MS,
                                      train_days=45, test_days=15, model_factory=FakeModel, progress=None)
        self.assertEqual(len(result["folds"]), 2)
        self.assertTrue(all(f["train_latest_settlement_ms"] < f["test_start_ms"] for f in result["folds"]))
        self.assertTrue(all(t["horizon_minutes"] == 30 for t in result["raw_trades"]))
        first_fit = tuple(v.copy() for v in FakeModel.fitted[0])
        changed = samples.copy()
        boundary = start + 45 * DAY_MS
        future = changed.signal_ms >= boundary
        changed.loc[future, "y"] = 1 - changed.loc[future, "y"]
        changed.loc[future, "exit"] = 100.5
        FakeModel.fitted = []
        again = walk_forward_signals(changed, "BTCUSDT", start, start + 75 * DAY_MS,
                                     train_days=45, test_days=15, model_factory=FakeModel, progress=None)
        np.testing.assert_array_equal(first_fit[0], FakeModel.fitted[0][0])
        np.testing.assert_array_equal(first_fit[1], FakeModel.fitted[0][1])
        pd.testing.assert_frame_equal(result["predictions"][["signal_ms", "p", "direction"]],
                                      again["predictions"][["signal_ms", "p", "direction"]])

    def test_wrong_symbol_or_horizon_execution_is_rejected(self):
        samples, start = model_samples()
        with self.assertRaisesRegex(ValueError, "Only BTCUSDT"):
            walk_forward_signals(samples, "SOLUSDT", start, start + 75 * DAY_MS)
        samples.loc[0, "settlement_ms"] += MINUTE_MS
        with self.assertRaisesRegex(ValueError, "30-minute"):
            walk_forward_signals(samples, "BTCUSDT", start, start + 75 * DAY_MS, train_days=45)

    def test_fixed_research_profile_uses_source_model_and_thirty_eight_features(self):
        self.assertEqual(len(RESEARCH_FEATURES), 38)
        reference = reference_provenance(profile="research")
        self.assertEqual(reference["default_train_days"], 180)
        self.assertEqual(reference["feature_count"], 38)
        self.assertEqual(reference["files"]["research/ceiling.py"]["git_blob_sha1"],
                         "6a4d60994e93d34c27ed9f5193fe605f4d06ed54")
        try:
            model = make_model(profile="research")
        except ImportError:
            self.skipTest("scikit-learn is required to inspect the real model factory")
        for key, value in RESEARCH_MODEL_PARAMETERS.items():
            self.assertEqual(model.get_params()[key], value)

    def test_research_profile_chunking_causality_and_flat_retention(self):
        spot, index = bars(3600)
        index[["open", "high", "low", "close"]] = 100.
        baseline = build_samples(spot, index, "BTCUSDT", profile="research")
        self.assertGreater(len(baseline), 50)
        self.assertTrue(baseline.is_flat.all())
        self.assertEqual(baseline.attrs["features"], RESEARCH_FEATURES)
        self.assertEqual(baseline.attrs["quantdinger"]["custom_factor_count"], 38)
        with patch("research.workbuddy_strategy.FEATURE_CHUNK_SIZE", 600):
            chunked = build_samples(spot, index, "BTCUSDT", profile="research")
        pd.testing.assert_frame_equal(baseline, chunked, rtol=1e-8, atol=1e-8)
        changed = spot.copy()
        boundary = int(spot.timestamp.iloc[2400])
        changed.loc[changed.timestamp >= boundary, ["open", "high", "low", "close"]] *= 2
        mutated = build_samples(changed, index, "BTCUSDT", profile="research")
        columns = ["signal_ms"] + RESEARCH_FEATURES
        pd.testing.assert_frame_equal(
            baseline.loc[baseline.signal_ms <= boundary, columns].reset_index(drop=True),
            mutated.loc[mutated.signal_ms <= boundary, columns].reset_index(drop=True),
            rtol=1e-8, atol=1e-8)

    def test_research_walk_forward_honors_dynamic_features(self):
        samples, start = model_samples()
        for column in RESEARCH_FEATURES[len(FEATURES):]:
            samples[column] = np.arange(len(samples)) * .002
        samples.attrs["features"] = RESEARCH_FEATURES.copy()
        samples.attrs["profile"] = "research"
        FakeModel.fitted = []
        result = walk_forward_signals(samples, "ETHUSDT", start, start + 75 * DAY_MS,
                                      train_days=45, test_days=15, model_factory=FakeModel,
                                      progress=None, profile="research")
        self.assertEqual(FakeModel.fitted[0][0].shape[1], 38)
        self.assertEqual(result["profile"], "research")
        self.assertEqual(result["model_parameters"], RESEARCH_MODEL_PARAMETERS)
        with self.assertRaisesRegex(ValueError, "features/profile"):
            walk_forward_signals(samples, "ETHUSDT", start, start + 75 * DAY_MS,
                                 train_days=45, model_factory=FakeModel, progress=None)


if __name__ == "__main__":
    unittest.main()
