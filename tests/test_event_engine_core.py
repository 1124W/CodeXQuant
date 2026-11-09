import json
import math
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from research.event_engine import (DAY_MS, MINUTE_MS, STRATEGY_GRID, SYMBOLS, KNOWN_UNAVAILABLE_DATES,
                                  _window, _monthly, aggregate_bars, build_candidates,
                                  evaluate_events, event_stats, select_candidate,
                                  simulate_portfolio, utc_ms)


def trade(entry_ms, outcome="win", symbol="BTCUSDT", duration=30, **extras):
    return {"symbol": symbol, "signal_ms": entry_ms - MINUTE_MS,
            "entry_ms": entry_ms, "settlement_ms": entry_ms + duration * MINUTE_MS,
            "direction": 1, "entry": 100, "exit": {"win": 101, "loss": 99, "refund": 100}[outcome],
            "rule_id": STRATEGY_GRID[0]["id"], **extras}


class ExecutionTests(unittest.TestCase):
    def test_flat_refund_cost_and_decisive_break_even(self):
        entries = [trade(i * 60 * MINUTE_MS, outcome) for i, outcome in enumerate(("win", "loss", "refund"))]
        executed = simulate_portfolio(entries, daily_loss_limit=5)
        self.assertEqual([t["outcome"] for t in executed], ["win", "loss", "refund"])
        sample = event_stats(executed, 0, DAY_MS)
        self.assertEqual(sample["refunds"], 1)
        self.assertEqual(sample["decisive_trades"], 2)
        self.assertEqual(sample["win_rate"], .5)
        self.assertAlmostEqual(sample["net_units"], -.23)
        self.assertAlmostEqual(sample["effective_break_even_win_rate"], 1.015 / 1.8)

    def test_settle_first_and_symbol_priority_without_outcome_priority(self):
        first = trade(0, "loss", "BTCUSDT")
        same_time_winner = trade(0, "win", "ETHUSDT")
        next_trade = trade(30 * MINUTE_MS, "win")
        executed = simulate_portfolio([same_time_winner, next_trade, first])
        self.assertEqual([(t["entry_ms"], t["symbol"]) for t in executed], [(0, "BTCUSDT"), (30 * MINUTE_MS, "BTCUSDT")])
        self.assertEqual(executed[0]["outcome"], "loss")

    def test_budget_counts_losing_premium_not_net_profit(self):
        entries = [trade(i * 30 * MINUTE_MS, outcome) for i, outcome in enumerate(("win", "win", "loss", "loss", "win"))]
        executed = simulate_portfolio(entries, daily_loss_limit=2)
        self.assertEqual(len(executed), 4)
        # New UTC day starts a fresh budget, irrespective of the previous day.
        entries.append(trade(DAY_MS, "win"))
        self.assertEqual(len(simulate_portfolio(entries, daily_loss_limit=2)), 5)

    def test_cash_reservation_and_adverse_entry_change_outcomes(self):
        entries = [trade(0), trade(MINUTE_MS, symbol="ETHUSDT")]
        self.assertEqual(len(simulate_portfolio(entries, max_open_positions=2, initial_bankroll=1.01)), 1)
        small_move = [trade(0, exit=100.01)]
        self.assertEqual(simulate_portfolio(small_move)[0]["outcome"], "win")
        self.assertEqual(simulate_portfolio(small_move, adverse_bps=2)[0]["outcome"], "loss")

    def test_expiry_equal_to_cutoff_is_purged(self):
        cutoff = DAY_MS
        before = trade(cutoff - 31 * MINUTE_MS)
        boundary = trade(cutoff - 30 * MINUTE_MS)
        after = trade(cutoff + MINUTE_MS)
        self.assertEqual(_window([before, boundary, after], 0, cutoff), [before])
        self.assertEqual(_window([before, boundary, after], cutoff, 2 * cutoff), [after])


class SignalTests(unittest.TestCase):
    def make_minutes(self, days=3):
        rows, price = [], 100.
        for i in range(days * 1440):
            opened = price
            price += .02 * math.sin(i / 20) + .001
            rows.append((i * MINUTE_MS, opened, max(opened, price) + .01, min(opened, price) - .01, price))
        return rows

    def test_closed_aggregation_and_entry_expiry_exact_open(self):
        rows = self.make_minutes()
        bars = aggregate_bars(rows[:35])
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0][4], rows[29][4])
        candidates = build_candidates({s: rows for s in SYMBOLS}, latency_minutes=1, horizon_minutes=30)
        entries = [t for values in candidates.values() for t in values]
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry["entry_ms"] - entry["signal_ms"], MINUTE_MS)
            self.assertEqual(entry["settlement_ms"] - entry["entry_ms"], 30 * MINUTE_MS)
            self.assertEqual(entry["entry"], rows[entry["entry_ms"] // MINUTE_MS][1])
            self.assertEqual(entry["exit"], rows[entry["settlement_ms"] // MINUTE_MS][1])

    def test_future_suffix_cannot_change_past_signals_or_prices(self):
        rows = self.make_minutes()
        cutoff = 2 * DAY_MS
        original = build_candidates({s: rows for s in SYMBOLS})
        changed = [r if r[0] < cutoff else (r[0], *(v * 2 for v in r[1:])) for r in rows]
        altered = build_candidates({s: changed for s in SYMBOLS})
        for rule in original:
            self.assertEqual([t for t in original[rule] if t["settlement_ms"] < cutoff],
                             [t for t in altered[rule] if t["settlement_ms"] < cutoff])

    def test_known_unavailable_days_exclude_contracts_without_changing_other_signals(self):
        origin = utc_ms(date(2026, 1, 15))
        rows = [(r[0] + origin, *r[1:]) for r in self.make_minutes(days=12)]
        with patch("research.event_engine._UNAVAILABLE_DAY_NUMBERS", frozenset()):
            unrestricted = build_candidates({s: rows for s in SYMBOLS})
        restricted = build_candidates({s: rows for s in SYMBOLS})
        unavailable = {utc_ms(date.fromisoformat(day)) // DAY_MS for day in KNOWN_UNAVAILABLE_DATES}
        removed = 0
        for rule, entries in unrestricted.items():
            expected = [t for t in entries if not any(day in unavailable for day in
                        range(t["signal_ms"] // DAY_MS, t["settlement_ms"] // DAY_MS + 1))]
            self.assertEqual(restricted[rule], expected)
            removed += len(entries) - len(expected)
        self.assertGreater(removed, 0)
        self.assertTrue(any(restricted.values()))

    def test_day_bootstrap_is_deterministic_and_zero_days_not_dropped(self):
        entries = simulate_portfolio([trade(MINUTE_MS)])
        first = event_stats(entries, 0, 10 * DAY_MS)
        self.assertEqual(first, event_stats(entries, 0, 10 * DAY_MS))
        self.assertEqual(first["block_bootstrap_lower_95"], 0)
        self.assertEqual(first["block_bootstrap_net_lower_95"], 0)

    def test_monthly_frequency_uses_calendar_coverage_and_keeps_empty_months(self):
        start = utc_ms(date(2025, 1, 30))
        end = utc_ms(date(2025, 3, 2))
        entries = simulate_portfolio([trade(start + DAY_MS - 30 * MINUTE_MS)])
        months = _monthly(entries, start, end, {"payout": .8, "fee": .01})
        self.assertEqual([m["month"] for m in months], ["2025-01", "2025-02", "2025-03"])
        self.assertEqual(months[0]["per_day"], .5)
        self.assertEqual(months[1]["signals"], 0)
        self.assertEqual(months[2]["per_day"], 0)


class EvaluationTests(unittest.TestCase):
    def test_holdout_outcomes_cannot_change_rule_choices_or_development_results(self):
        start, end = date(2025, 1, 1), date(2025, 1, 11)
        origin = utc_ms(start)
        rows = [(origin + i * MINUTE_MS, 100., 100., 100., 100.) for i in range(10 * 1440)]
        chosen_id = STRATEGY_GRID[0]["id"]
        entries = [trade(origin + i * 60 * MINUTE_MS + MINUTE_MS,
                         "loss" if i % 4 == 3 else "win") for i in range(10 * 24)]
        candidates = {rule["id"]: entries if rule["id"] == chosen_id else [] for rule in STRATEGY_GRID}
        options = dict(train_days=3, validation_days=2, holdout_days=3, min_train_trades=10, min_test_trades=10)
        with patch("research.event_engine.build_candidates", return_value=candidates):
            first = evaluate_events({s: rows for s in SYMBOLS}, start, end, **options)
        altered = [{**t, "exit": 99} if t["signal_ms"] >= origin + 7 * DAY_MS else t for t in entries]
        changed = {rule["id"]: altered if rule["id"] == chosen_id else [] for rule in STRATEGY_GRID}
        with patch("research.event_engine.build_candidates", return_value=changed):
            second = evaluate_events({s: rows for s in SYMBOLS}, start, end, **options)
        self.assertEqual(first["walk_forward"], second["walk_forward"])
        self.assertEqual(first["final_holdout"]["chosen"], second["final_holdout"]["chosen"])
        self.assertNotEqual(first["final_holdout"]["test"], second["final_holdout"]["test"])
        self.assertTrue(first["trade_ledger"])
        self.assertFalse(first["live_approved"])
        self.assertEqual(first["walk_forward"]["aggregate"]["signals"],
                         sum(f["test"]["signals"] for f in first["walk_forward"]["folds"]))
        self.assertAlmostEqual(first["walk_forward"]["aggregate"]["net_units"],
            sum(t["net_units"] for t in first["trade_ledger"] if t["split"] == "walk_forward"))
        json.dumps(first, allow_nan=False)

    def test_no_train_edge_produces_explicit_no_trade_report(self):
        start, end = date(2025, 1, 1), date(2025, 1, 6)
        rows = [(utc_ms(start) + i * MINUTE_MS, 100., 100., 100., 100.) for i in range(5 * 1440)]
        report = evaluate_events({s: rows for s in SYMBOLS}, start, end,
                                 train_days=2, validation_days=1, holdout_days=1)
        self.assertEqual(report["schema_version"], 3)
        self.assertIsNone(report["final_holdout"]["chosen"])
        self.assertEqual(report["walk_forward"]["aggregate"]["signals"], 0)
        self.assertFalse(report["publishable"])
        self.assertFalse(report["live_approved"])
        self.assertEqual(report["trade_ledger"], [])
        json.dumps(report, allow_nan=False)

    def test_invalid_data_and_configuration_are_rejected(self):
        start, end = date(2025, 1, 1), date(2025, 1, 6)
        rows = [(utc_ms(start) + i * MINUTE_MS, 100., 100., 100., 100.) for i in range(5 * 1440)]
        for kwargs in ({"target_win_rate": float("nan")}, {"latency_minutes": 0}, {"stake": -1}, {"fee": .9}):
            with self.assertRaises(ValueError):
                evaluate_events({s: rows for s in SYMBOLS}, start, end, train_days=2,
                                validation_days=1, holdout_days=1, **kwargs)
        corrupt = rows[:]
        corrupt[1] = (corrupt[0][0], *corrupt[1][1:])
        with self.assertRaises(ValueError):
            evaluate_events({"BTCUSDT": corrupt, "ETHUSDT": rows}, start, end,
                            train_days=2, validation_days=1, holdout_days=1)


if __name__ == "__main__":
    unittest.main()
