"""Chronological research protocol checks using synthetic events and fake models."""

import copy
import unittest
from datetime import date

import numpy as np
import pandas as pd

from research.event_engine import DAY_MS, MINUTE_MS, utc_ms
from research.event_strategy_research import (
    CAPITAL, MODEL_CONFIGS, PLAN, build_raw, candidates, choose_on_discovery,
    evaluate, make_model, moving_week_interval, period_raw, rolling_predictions, rule_directions,
)


def event(signal, *, win=True, flat=False, rule="synthetic"):
    return {"symbol": "BTCUSDT", "rule_id": rule, "signal_ms": signal,
            "entry_ms": signal + MINUTE_MS,
            "settlement_ms": signal + 31 * MINUTE_MS,
            "entry": 100., "exit": 100. if flat else 101. if win else 99.,
            "direction": 1, "horizon_minutes": 30}


def prediction_samples(count=4):
    origin = utc_ms(date(2025, 1, 1))
    signs = np.where(np.arange(count) % 2, -1, 1)
    signals = origin + np.arange(count) * 60 * MINUTE_MS
    frame = pd.DataFrame({
        "signal_ms": signals, "entry_ms": signals + MINUTE_MS,
        "settlement_ms": signals + 31 * MINUTE_MS,
        "entry": 100., "exit": 100., "y": 0, "is_flat": True,
        "ret_60": -30. * signs, "rv_60": 1., "ret_15": -5. * signs,
        "ret_240": 60. * signs, "rv_240": 1., "rv_ratio": 1.,
        "model_fit_cutoff_ms": origin - DAY_MS,
        "train_latest_settlement_ms": origin - 2 * DAY_MS,
    })
    for name in MODEL_CONFIGS:
        frame[f"p_{name}"] = np.where(signs == 1, .8, .2)
    return frame


def model_samples(days=212):
    origin = utc_ms(date(2025, 1, 1))
    count = days * 48
    t = np.arange(count)
    signals = origin + t * 30 * MINUTE_MS
    frame = pd.DataFrame({
        "known_feature": t.astype(float), "other_feature": np.sin(t / 5),
        "signal_ms": signals, "entry_ms": signals + MINUTE_MS,
        "settlement_ms": signals + 31 * MINUTE_MS,
        "entry": 100., "exit": np.where(t % 3 == 0, 101., 99.),
        "y": (t % 3 == 0).astype(int), "is_flat": False, "symbol": "BTCUSDT",
    })
    # This outcome-looking column must not enter the factory's fitted arrays.
    frame["unused_target_copy"] = frame["y"] * 1_000_000
    frame.attrs["features"] = ["known_feature", "other_feature"]
    return frame, origin


class FakeModel:
    def __init__(self, name, records):
        self.name = name
        self.records = records

    def fit(self, x, y):
        self.records.append({"name": self.name, "x": x.copy(), "y": y.copy()})
        self.probability = .2 + .6 * float(np.mean(y))
        return self

    def predict_proba(self, x):
        return np.tile([1 - self.probability, self.probability], (len(x), 1))


def factories(records):
    return {name: (lambda current=name: FakeModel(current, records)) for name in MODEL_CONFIGS}


class EventStrategyResearchTests(unittest.TestCase):
    def test_candidate_universe_is_finite_unique_and_fixed(self):
        rules = candidates()
        self.assertEqual(len(rules), 33)
        self.assertEqual(len({rule["id"] for rule in rules}), 33)
        self.assertEqual(sum(rule["family"] == "model" for rule in rules), 27)
        self.assertEqual(sum(rule["family"] == "direct_exhaustion" for rule in rules), 3)
        self.assertEqual(sum(rule["family"] == "direct_trend_pullback" for rule in rules), 3)
        self.assertEqual(candidates(), rules)
        self.assertFalse(PLAN["post_selection_changes_permitted"])
        self.assertFalse(PLAN["pristine_holdout"])

    def test_settlement_exactly_on_end_boundary_is_excluded(self):
        start = utc_ms(date(2025, 1, 1))
        end = start + DAY_MS
        before = event(start + MINUTE_MS)
        exact = event(end - 31 * MINUTE_MS)
        earlier_signal = event(start - MINUTE_MS)
        later = event(end + MINUTE_MS)
        self.assertEqual(exact["settlement_ms"], end)
        self.assertEqual(period_raw([before, exact, earlier_signal, later], start, end), [before])
        ledger, stats = evaluate([before, exact, earlier_signal, later], start, end)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(stats["wins"], 1)

    def test_discovery_choice_cannot_change_when_all_later_outcomes_change(self):
        start = utc_ms(date(2024, 1, 1))
        end = start + 220 * DAY_MS
        rules = [{"id": "alpha"}, {"id": "beta"}]
        raw = {
            "alpha": [event(start + i * DAY_MS, win=i < 154, rule="alpha") for i in range(220)],
            "beta": [event(start + i * DAY_MS, win=i < 110, rule="beta") for i in range(220)],
        }
        for rule in rules:
            name = rule["id"]
            # Both the exact boundary and every subsequent event are inaccessible
            # for selection, even when their outcome strongly favors one rule.
            raw[name].append(event(end - 31 * MINUTE_MS, win=name == "beta", rule=name))
            raw[name].extend(event(end + i * DAY_MS, win=name == "beta", rule=name) for i in range(20))
        selected, ranking = choose_on_discovery(raw, rules, start, end)
        changed = copy.deepcopy(raw)
        for name, trades in changed.items():
            for trade in trades:
                if trade["settlement_ms"] >= end:
                    trade["exit"] = 1_000_000. if name == "alpha" else .0001
        again, later_ranking = choose_on_discovery(changed, rules, start, end)
        self.assertEqual(selected["id"], "alpha")
        self.assertEqual(selected, again)
        self.assertEqual(ranking, later_ranking)
        self.assertEqual(ranking[0]["discovery"]["decisive_trades"], 220)

    def test_direction_rules_use_no_label_or_execution_columns(self):
        frame = prediction_samples()
        forbidden = ["entry", "exit", "entry_ms", "settlement_ms", "y", "is_flat"]
        historical = frame.drop(columns=forbidden)
        for rule in candidates():
            expected = rule_directions(frame, rule)
            actual = rule_directions(historical, rule)
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(len(actual), len(frame))

    def test_raw_signals_retain_flat_events_and_ignore_changed_labels(self):
        frame = prediction_samples()
        rules = candidates()
        baseline = build_raw({"BTCUSDT": frame}, rules)
        changed = frame.copy()
        changed["is_flat"] = False
        changed["y"] = 1
        changed["exit"] = 200.
        actual = build_raw({"BTCUSDT": changed}, rules)
        for rule in rules:
            before, after = baseline[rule["id"]], actual[rule["id"]]
            self.assertEqual(len(before), len(frame))
            self.assertEqual([t["direction"] for t in before], [t["direction"] for t in after])
            self.assertEqual([t["signal_ms"] for t in before], [t["signal_ms"] for t in after])
            self.assertTrue(all(t["entry"] == t["exit"] for t in before))
        first = next(iter(baseline.values()))
        _, stats = evaluate(first, int(frame.signal_ms.min()), int(frame.settlement_ms.max()) + 1)
        self.assertEqual(stats["refunds"], len(frame))
        self.assertEqual(stats["net_units"], 0.)

    def test_wins_losses_and_refunds_use_85_percent_with_zero_extra_fee(self):
        start = utc_ms(date(2025, 1, 1))
        raw = [event(start, win=True), event(start + 60 * MINUTE_MS, win=False),
               event(start + 120 * MINUTE_MS, flat=True)]
        ledger, stats = evaluate(raw, start, start + DAY_MS)
        self.assertEqual(CAPITAL["payout"], .85)
        self.assertEqual(CAPITAL["fee"], 0.)
        self.assertEqual([trade["net_units"] for trade in ledger], [4.25, -5., 0.])
        self.assertEqual((stats["wins"], stats["losses"], stats["refunds"]), (1, 1, 1))
        self.assertEqual(stats["net_units"], -.75)
        self.assertAlmostEqual(stats["effective_break_even_win_rate"], 1 / 1.85)

    def test_rolling_fits_use_only_attrs_features_and_strictly_settled_history(self):
        samples, start = model_samples()
        records = []
        prediction, folds = rolling_predictions(samples, start, start + 212 * DAY_MS,
                                                factories=factories(records), progress=None)
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(records), 2 * len(MODEL_CONFIGS))
        for fold in folds:
            self.assertLess(fold["train_latest_settlement_ms"], fold["fit_cutoff_ms"])
        for index, record in enumerate(records):
            fold = folds[index // len(MODEL_CONFIGS)]
            positions = record["x"][:, 0].astype(int)
            self.assertEqual(record["x"].shape[1], len(samples.attrs["features"]))
            self.assertTrue((samples.settlement_ms.iloc[positions] < fold["fit_cutoff_ms"]).all())
            self.assertTrue((samples.signal_ms.iloc[positions] >= fold["fit_cutoff_ms"] - 180 * DAY_MS).all())
            np.testing.assert_array_equal(record["y"], samples.y.iloc[positions].to_numpy())
        first_cutoff = start + 180 * DAY_MS
        # This last half-hour signal settles one minute after the fit cutoff.
        purged_position = 180 * 48 - 1
        self.assertNotIn(purged_position, records[0]["x"][:, 0])
        self.assertEqual(int(prediction.signal_ms.min()), first_cutoff)
        self.assertTrue((prediction.settlement_ms < start + 212 * DAY_MS).all())

    def test_later_outcomes_cannot_change_earlier_fit_arrays_or_predictions(self):
        samples, start = model_samples()
        cutoff = start + 180 * DAY_MS
        first_records = []
        baseline, _ = rolling_predictions(samples, start, start + 212 * DAY_MS,
                                           factories=factories(first_records), progress=None)
        changed = samples.copy()
        future = changed.signal_ms >= cutoff
        changed.loc[future, "y"] = 1
        changed.loc[future, "exit"] = 1_000_000.
        later_records = []
        actual, _ = rolling_predictions(changed, start, start + 212 * DAY_MS,
                                         factories=factories(later_records), progress=None)
        for index in range(len(MODEL_CONFIGS)):
            np.testing.assert_array_equal(first_records[index]["x"], later_records[index]["x"])
            np.testing.assert_array_equal(first_records[index]["y"], later_records[index]["y"])
        columns = ["signal_ms"] + [f"p_{name}" for name in MODEL_CONFIGS]
        earlier = baseline.signal_ms < cutoff + 30 * DAY_MS
        pd.testing.assert_frame_equal(baseline.loc[earlier, columns], actual.loc[earlier, columns])
        # The second fit is allowed to learn newly settled labels, showing that
        # this test does not pass only because a fake model ignores its inputs.
        self.assertFalse(np.array_equal(first_records[3]["y"], later_records[3]["y"]))
        self.assertFalse(np.array_equal(baseline.loc[~earlier, "p_regularized_gbm"],
                                        actual.loc[~earlier, "p_regularized_gbm"]))

    def test_prediction_time_flats_are_retained_but_training_flats_are_excluded(self):
        samples, start = model_samples()
        cutoff = start + 180 * DAY_MS
        samples.loc[5, "is_flat"] = True
        samples.loc[samples.signal_ms >= cutoff, "is_flat"] = True
        samples.loc[samples.signal_ms >= cutoff, "exit"] = samples.loc[samples.signal_ms >= cutoff, "entry"]
        records = []
        prediction, _ = rolling_predictions(samples, start, start + 212 * DAY_MS,
                                             factories=factories(records), progress=None)
        self.assertTrue(prediction.is_flat.all())
        self.assertEqual(len(prediction), 32 * 48 - 1)
        self.assertNotIn(5., records[0]["x"][:, 0])

    def test_logistic_scaler_is_unfitted_and_inside_training_pipeline(self):
        model = make_model("regularized_logistic")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.steps[0][1], StandardScaler)
        self.assertFalse(hasattr(model.steps[0][1], "mean_"))
        model.fit(np.array([[1., 2.], [3., 6.], [5., 10.], [7., 14.]]), [0, 1, 0, 1])
        np.testing.assert_allclose(model.steps[0][1].mean_, [4., 8.])
        before = model.steps[0][1].mean_.copy()
        model.predict_proba(np.array([[1_000_000., 2_000_000.]]))
        np.testing.assert_array_equal(model.steps[0][1].mean_, before)

    def test_week_bootstrap_is_seeded_and_order_invariant(self):
        start = utc_ms(date(2025, 1, 1))
        ledger = []
        for day in range(23):
            win = day % 5 < 3
            ledger.append({"settlement_ms": start + day * DAY_MS + MINUTE_MS,
                           "outcome": "win" if win else "loss", "net_units": 4.25 if win else -5.})
        actual = moving_week_interval(ledger, start, start + 23 * DAY_MS, iterations=250)
        again = moving_week_interval(list(reversed(ledger)), start, start + 23 * DAY_MS, iterations=250)
        self.assertEqual(actual, again)
        self.assertEqual(actual["iterations"], 250)
        self.assertEqual(actual["block_days"], 7)
        self.assertLessEqual(actual["win_rate_lower_95"], actual["win_rate_upper_95"])
        self.assertLessEqual(actual["net_lower_95"], actual["net_upper_95"])

    def test_week_bootstrap_preserves_seven_day_patterns(self):
        start = utc_ms(date(2025, 1, 1))
        ledger = [{"settlement_ms": start + day * DAY_MS + MINUTE_MS,
                   "outcome": "win" if day % 7 < 4 else "loss",
                   "net_units": 4.25 if day % 7 < 4 else -5.}
                  for day in range(14)]
        actual = moving_week_interval(ledger, start, start + 14 * DAY_MS, iterations=250)
        # Every circular 7-day block has precisely four wins and three losses.
        # An independent-day sampler would introduce variance in these totals.
        self.assertAlmostEqual(actual["win_rate_lower_95"], 4 / 7)
        self.assertAlmostEqual(actual["win_rate_upper_95"], 4 / 7)
        self.assertEqual(actual["net_lower_95"], 4.)
        self.assertEqual(actual["net_upper_95"], 4.)

    def test_week_bootstrap_keeps_btc_eth_same_day_outcomes_joint(self):
        start = utc_ms(date(2025, 1, 1))
        ledger = []
        for day in range(21):
            for _ in range(1 + day % 3):
                timestamp = start + day * DAY_MS + MINUTE_MS
                ledger.extend([
                    {"symbol": "BTCUSDT", "settlement_ms": timestamp, "outcome": "win", "net_units": 4.25},
                    {"symbol": "ETHUSDT", "settlement_ms": timestamp, "outcome": "loss", "net_units": -5.},
                ])
        actual = moving_week_interval(ledger, start, start + 21 * DAY_MS, iterations=250)
        self.assertEqual(actual["win_rate_lower_95"], .5)
        self.assertEqual(actual["win_rate_upper_95"], .5)
        self.assertLess(actual["net_upper_95"], 0.)

    def test_week_bootstrap_all_wins_empty_and_refund_populations(self):
        start = utc_ms(date(2025, 1, 1))
        ledger = [{"settlement_ms": start + day * DAY_MS + MINUTE_MS,
                   "outcome": "win", "net_units": 4.25} for day in range(21)]
        actual = moving_week_interval(ledger, start, start + 21 * DAY_MS, iterations=250)
        self.assertEqual(actual["win_rate_lower_95"], 1.)
        self.assertEqual(actual["win_rate_upper_95"], 1.)
        self.assertEqual(actual["net_lower_95"], 21 * 4.25)
        self.assertEqual(actual["net_upper_95"], 21 * 4.25)
        for population in ([], [{"settlement_ms": start + MINUTE_MS, "outcome": "refund", "net_units": 0.}]):
            empty = moving_week_interval(population, start, start + 21 * DAY_MS, iterations=250)
            for key in ("win_rate_lower_95", "win_rate_upper_95", "net_lower_95", "net_upper_95"):
                self.assertIsNone(empty[key])

    def test_week_bootstrap_rejects_settlements_outside_interval(self):
        start = utc_ms(date(2025, 1, 1))
        end = start + 21 * DAY_MS
        for timestamp in (start - MINUTE_MS, end):
            ledger = [{"settlement_ms": timestamp, "outcome": "win", "net_units": 4.25}]
            with self.assertRaisesRegex(ValueError, "outside"):
                moving_week_interval(ledger, start, end, iterations=250)


if __name__ == "__main__":
    unittest.main()
