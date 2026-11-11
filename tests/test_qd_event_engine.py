import copy
import unittest

import pandas as pd

from research.event_engine import DAY_MS, MINUTE_MS, SYMBOLS, simulate_portfolio
from research.qd_event_engine import (
    GRID_SHA256, _config, build_candidates, evaluate_demo, generate_rules,
    select_rule,
)


def minute_rows(days=3, reverse_after=None):
    result = []
    for index in range(days * 1440):
        price = 100 + index / 10000
        if reverse_after is not None and index * MINUTE_MS >= reverse_after:
            price = 150 - index / 10000
        result.append((index * MINUTE_MS, price, price, price, price))
    return result


def feature_frames(days=3):
    frames = {}
    for symbol in SYMBOLS:
        for timeframe in (5, 15):
            count = days * 1440 // timeframe
            frames[(symbol, timeframe)] = pd.DataFrame({
                "timestamp": [(i + 1) * timeframe * MINUTE_MS for i in range(count)],
                "rsi2": [2. if i % 2 else 50. for i in range(count)],
                "rsi14": [50.] * count,
                "z20": [0.] * count,
                "trend": [0.] * count,
            })
    return frames


class GridTests(unittest.TestCase):
    def test_grid_is_fixed_unique_and_complete(self):
        rules = generate_rules()
        self.assertEqual(len(rules), 72)
        self.assertEqual(len({rule["id"] for rule in rules}), 72)
        self.assertEqual({r["horizon_minutes"] for r in rules}, {10, 30})
        self.assertEqual({r["timeframe_minutes"] for r in rules}, {5, 15})
        self.assertEqual(len(GRID_SHA256), 64)
        rules[0]["threshold"] = 99
        self.assertEqual(generate_rules()[0]["threshold"], 5.)

    def test_configuration_rejects_invalid_capital_costs_and_unknown_keys(self):
        for config in ({"stake": 0}, {"payout": float("nan")},
                       {"latency_minutes": 0}, {"max_open_positions": True},
                       {"initial_bankroll": 1}, {"fee": .9},
                       {"daily_loss_limit": 1}, {"test_tuning": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _config(config)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.rows = minute_rows()
        self.candles = {symbol: self.rows for symbol in SYMBOLS}
        self.features = feature_frames()
        self.rule = generate_rules()[0]

    def test_entry_and_expiry_use_delayed_exact_opens(self):
        candidates = build_candidates(self.candles, self.features, rules=[self.rule])
        self.assertTrue(candidates[self.rule["id"]])
        for trade in candidates[self.rule["id"]]:
            self.assertEqual(trade["entry_ms"], trade["signal_ms"] + MINUTE_MS)
            self.assertEqual(trade["settlement_ms"], trade["entry_ms"] + 10 * MINUTE_MS)
            self.assertEqual(trade["entry"], self.rows[trade["entry_ms"] // MINUTE_MS][1])
            self.assertEqual(trade["exit"], self.rows[trade["settlement_ms"] // MINUTE_MS][1])

    def test_future_data_mutation_cannot_change_preboundary_candidates(self):
        cutoff = 2 * DAY_MS
        original = build_candidates(self.candles, self.features, end_ms=cutoff)
        changed = minute_rows(reverse_after=cutoff)
        changed_features = {key: frame.copy() for key, frame in self.features.items()}
        for frame in changed_features.values():
            frame.loc[frame.timestamp >= cutoff, ["rsi2", "rsi14", "z20", "trend"]] = [99., 99., 3., 2.]
        altered = build_candidates({symbol: changed for symbol in SYMBOLS}, changed_features, end_ms=cutoff)
        self.assertEqual(original, altered)

    def test_intraminute_high_low_close_cannot_change_execution_open(self):
        original = build_candidates(self.candles, self.features, rules=[self.rule])
        changed = [(t, opened, opened + 5, opened - 5, opened + 1)
                   for t, opened, *_ in self.rows]
        altered = build_candidates({symbol: changed for symbol in SYMBOLS}, self.features, rules=[self.rule])
        self.assertEqual(original, altered)

    def test_cutoff_purges_expiry_equal_to_end_and_preserves_signal_edge(self):
        all_candidates = build_candidates(self.candles, self.features, rules=[self.rule])[self.rule["id"]]
        candidate = all_candidates[5]
        cutoff = candidate["settlement_ms"]
        purged = build_candidates(self.candles, self.features, rules=[self.rule], end_ms=cutoff)[self.rule["id"]]
        self.assertNotIn(candidate, purged)
        self.assertTrue(all(trade["settlement_ms"] < cutoff for trade in purged))
        window_start = DAY_MS
        expected = [trade for trade in all_candidates if trade["signal_ms"] >= window_start]
        windowed = build_candidates(self.candles, self.features, rules=[self.rule], start_ms=window_start)[self.rule["id"]]
        self.assertEqual(windowed, expected)

    def test_bad_feature_timestamps_and_missing_columns_fail(self):
        for invalid in ("timestamp", "rsi2"):
            frames = dict(self.features)
            frame = frames[("BTCUSDT", 5)].copy()
            if invalid == "timestamp":
                frame.loc[10, "timestamp"] += MINUTE_MS
            else:
                frame = frame.drop(columns=[invalid])
            frames[("BTCUSDT", 5)] = frame
            with self.assertRaises(ValueError):
                build_candidates(self.candles, frames, rules=[self.rule])


class SelectionTests(unittest.TestCase):
    def small_config(self):
        return {"min_train_trades": 1, "min_validation_trades": 1, "min_test_trades": 1}

    def test_holdout_outcomes_cannot_change_prior_selection(self):
        frames = feature_frames()
        candles = {symbol: minute_rows() for symbol in SYMBOLS}
        first = evaluate_demo(candles, 0, DAY_MS, 2 * DAY_MS, 3 * DAY_MS,
                              self.small_config(), frames)
        changed = {symbol: minute_rows(reverse_after=2 * DAY_MS) for symbol in SYMBOLS}
        second = evaluate_demo(changed, 0, DAY_MS, 2 * DAY_MS, 3 * DAY_MS,
                               self.small_config(), frames)
        self.assertEqual(first["selected_rule"], second["selected_rule"])
        self.assertEqual(first["selection_table"], second["selection_table"])
        self.assertEqual(first["train"], second["train"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertNotEqual(first["test"], second["test"])
        self.assertEqual(first["validation_context"]["test_rules_evaluated"], 1)

    def test_empty_prior_has_explicit_rejected_demo_not_success(self):
        frames = feature_frames()
        for frame in frames.values():
            frame.loc[:, "rsi2"] = 50.
        report = evaluate_demo({symbol: minute_rows() for symbol in SYMBOLS},
            0, DAY_MS, 2 * DAY_MS, 3 * DAY_MS, self.small_config(), frames)
        self.assertFalse(report["selection_eligible"])
        self.assertFalse(report["target_met"])
        self.assertFalse(report["demo_pass"])
        self.assertFalse(report["research_qualified"])
        self.assertEqual(report["selection_mode"], "rejected_prior_winner_demo_only")
        self.assertEqual(report["test"]["signals"], 0)
        self.assertIsNone(report["test"]["win_rate"])

    def test_refund_still_pays_friction_and_fixed_stake_is_five(self):
        rule = generate_rules()[0]
        raw = {"symbol": "BTCUSDT", "signal_ms": 0,
               "entry_ms": MINUTE_MS, "settlement_ms": 11 * MINUTE_MS,
               "direction": 1, "entry": 100., "exit": 100., "rule_id": rule["id"]}
        config = _config(None)
        trades = simulate_portfolio([raw], **{key: config[key] for key in (
            "payout", "fee", "initial_bankroll", "stake", "max_open_positions", "daily_loss_limit", "adverse_bps")})
        self.assertEqual(trades[0]["stake"], 5.)
        self.assertEqual(trades[0]["outcome"], "refund")
        self.assertAlmostEqual(trades[0]["net_units"], -.05)

    def test_invalid_minute_grid_fails_before_evaluation(self):
        rows = minute_rows()
        rows[100] = rows[99]
        with self.assertRaises(ValueError):
            evaluate_demo({symbol: rows for symbol in SYMBOLS}, 0, DAY_MS,
                          2 * DAY_MS, 3 * DAY_MS, self.small_config(), feature_frames())


if __name__ == "__main__":
    unittest.main()
