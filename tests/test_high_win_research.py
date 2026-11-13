"""Independent causal selection, settlement and paired-inference regressions."""
import copy
import unittest

from research.high_win_research import (
    evaluate, paired_week_interval, select_on_discovery, simulate,
)
from research.minute_scan_research import simulate_alert_admission

MINUTE = 60_000
DAY = 86_400_000
ORIGIN = 20_000 * DAY


def trade(offset_minutes=0, *, symbol="BTCUSDT", direction=1, exit_price=101.):
    signal = ORIGIN + offset_minutes * MINUTE
    return {"symbol": symbol, "signal_ms": signal, "entry_ms": signal + MINUTE,
            "settlement_ms": signal + 31 * MINUTE, "entry": 100., "exit": exit_price,
            "direction": direction, "p_up": .6 if direction == 1 else .4,
            "horizon_minutes": 30}


def daily_population(count, *, alternate=True, wins_per_ten=7, start_day=0):
    return [trade((start_day + day) * 1440,
                  symbol="ETHUSDT" if alternate and day % 2 else "BTCUSDT",
                  exit_price=101. if day % 10 < wins_per_ten else 99.)
            for day in range(count)]


class HighWinResearchTests(unittest.TestCase):
    def test_validation_outcomes_cannot_change_discovery_selection_or_ranking(self):
        rules = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]
        initial = {"a": daily_population(10, wins_per_ten=8),
                   "b": daily_population(10, wins_per_ten=6)}
        start, end = ORIGIN, ORIGIN + 20 * DAY
        before = select_on_discovery(initial, rules, start, end, min_trades=4, min_per_symbol=2)
        changed = copy.deepcopy(initial)
        changed["a"] += daily_population(30, start_day=30, wins_per_ten=0)
        changed["b"] += daily_population(30, start_day=30, wins_per_ten=10)
        after = select_on_discovery(changed, rules, start, end, min_trades=4, min_per_symbol=2)
        self.assertEqual(before, after)
        self.assertEqual(before[0]["id"], "a")

    def test_default_minimum_requires_both_assets_even_if_single_asset_win_rate_is_higher(self):
        rules = [{"id": "single", "label": "one coin"}, {"id": "both", "label": "both coins"}]
        raw = {"single": daily_population(600, alternate=False, wins_per_ten=10),
               "both": daily_population(600, wins_per_ten=7)}
        selected, ranking = select_on_discovery(raw, rules, ORIGIN, ORIGIN + 601 * DAY)
        self.assertEqual(selected["id"], "both")
        by_id = {row["rule"]["id"]: row for row in ranking}
        self.assertFalse(by_id["single"]["eligible"])
        self.assertTrue(by_id["both"]["eligible"])
        too_small = {"single": daily_population(499), "both": daily_population(499)}
        self.assertIsNone(select_on_discovery(too_small, rules, ORIGIN, ORIGIN + 601 * DAY)[0])

    def test_evaluation_excludes_prestart_signals_and_settlements_at_or_beyond_end(self):
        end = ORIGIN + DAY
        raw = [trade(-20), trade(0), trade(1440 - 31), trade(1440 - 30)]
        ledger, stats = evaluate(raw, ORIGIN, end)
        self.assertEqual([row["signal_ms"] for row in ledger], [ORIGIN])
        self.assertEqual(stats["signals"], 1)
        self.assertEqual(stats["net_units"], 4.25)

    def test_zero_stress_simulation_matches_existing_minute_portfolio_exactly(self):
        raw = [trade(0, symbol="ETHUSDT"), trade(0), trade(1), trade(30, exit_price=99),
               trade(31), trade(60, direction=-1, exit_price=99), trade(61),
               trade(90, exit_price=100), trade(120), trade(180, exit_price=99)]
        expected, actual = simulate_alert_admission(raw), simulate(raw)
        fields = ("symbol", "signal_ms", "entry_ms", "settlement_ms", "direction", "entry", "exit",
                  "outcome", "net_units", "stake", "horizon_minutes")
        self.assertEqual([{k: row[k] for k in fields} for row in expected],
                         [{k: row[k] for k in fields} for row in actual])

    def test_adverse_execution_relabels_losses_before_later_risk_admission(self):
        raw = [trade(index * 35, exit_price=100.01) for index in range(8)]
        self.assertEqual(len(simulate(raw)), 8)
        stressed = simulate(raw, adverse_bps=2.)
        self.assertEqual(len(stressed), 5)
        self.assertTrue(all(row["outcome"] == "loss" for row in stressed))
        self.assertEqual(sum(row["net_units"] for row in stressed), -25.)
        short = simulate([trade(direction=-1, exit_price=99.99)], adverse_bps=2.)
        self.assertEqual(short[0]["outcome"], "loss")

    def test_fee_reduces_profit_and_is_reserved_when_testing_available_cash(self):
        raw = [trade(0), trade(35)]
        charged = simulate(raw, fee=.01)
        self.assertEqual(len(charged), 2)
        self.assertAlmostEqual(sum(row["net_units"] for row in charged), 8.40)
        # One event/day avoids the daily loss cap. Eighteen wins at4.20 and
        # 212 losses at5.05 leave exactly5.00; the next5.05 admission must fail.
        near_empty = (daily_population(18, wins_per_ten=10)
                      + daily_population(212, start_day=18, wins_per_ten=0)
                      + [trade(230 * 1440)])
        charged = simulate(near_empty, fee=.01)
        self.assertEqual(len(charged), 230)
        self.assertAlmostEqual(1000 + sum(row["net_units"] for row in charged), 5.)

    def test_paired_resampling_identical_ledgers_has_zero_difference(self):
        ledger = simulate(daily_population(35, wins_per_ten=6))
        interval = paired_week_interval(ledger, copy.deepcopy(ledger), ORIGIN, ORIGIN + 36 * DAY,
                                        iterations=100)
        self.assertEqual(interval["delta_win_rate_lower_95"], 0.)
        self.assertEqual(interval["delta_win_rate_upper_95"], 0.)
        for key, value in interval.items():
            if key.startswith("delta_") and (key.endswith("lower_95") or key.endswith("upper_95")):
                self.assertEqual(value, 0., key)


if __name__ == "__main__":
    unittest.main()
