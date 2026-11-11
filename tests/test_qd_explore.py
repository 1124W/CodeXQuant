import copy
from unittest.mock import patch
import unittest

from research.event_data import data_fingerprint
from research.event_engine import DAY_MS, SYMBOLS
from research.qd_event_engine import GRID_SHA256, _config, generate_rules
from research.qd_explore import run_exploration
from tests.test_qd_event_engine import feature_frames, minute_rows


def primary_for(candles):
    return {
        "config": _config(None), "grid_sha256": GRID_SHA256,
        "strategy_grid": generate_rules(), "candidate_count": 72,
        "start_utc": "1970-01-01", "test_start_utc": "1970-01-03",
        "end_exclusive_utc": "1970-01-04",
        "provenance": {"run_id": "primary-test", "data_sha256": data_fingerprint(candles)},
    }


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.candles = {symbol: minute_rows() for symbol in SYMBOLS}
        self.features = feature_frames()
        self.primary = primary_for(self.candles)

    def test_all_rules_disclosed_and_flags_never_claim_validation(self):
        before = copy.deepcopy(self.primary)
        report = run_exploration(self.primary, self.candles, self.features)
        self.assertEqual(self.primary, before)
        self.assertEqual(report["search_count"], 72)
        self.assertEqual(len(report["search_table"]), 72)
        self.assertEqual(report["evidence_type"], "in_sample_historical_search")
        self.assertTrue(report["selection_uses_test"])
        self.assertFalse(report["out_of_sample"])
        self.assertFalse(report["research_qualified"])
        self.assertFalse(report["live_approved"])
        self.assertFalse(report["demo_pass"])
        self.assertTrue(all(trade["split"] == "historical_exploration" for trade in report["trade_ledger"]))

    def test_grid_or_data_change_is_rejected(self):
        altered_grid = copy.deepcopy(self.primary)
        altered_grid["strategy_grid"][0]["threshold"] = 99
        with self.assertRaisesRegex(ValueError, "grid"):
            run_exploration(altered_grid, self.candles, self.features)
        altered_data = {symbol: minute_rows(reverse_after=2 * DAY_MS) for symbol in SYMBOLS}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            run_exploration(self.primary, altered_data, self.features)

    def test_no_trade_is_not_success_or_zero_percent(self):
        for frame in self.features.values():
            frame.loc[:, "rsi2"] = 50.
        report = run_exploration(self.primary, self.candles, self.features)
        self.assertEqual(report["test"]["signals"], 0)
        self.assertIsNone(report["test"]["win_rate"])
        self.assertFalse(report["historical_target_met"])
        self.assertEqual(report["historical_target_candidates"], 0)

    def test_sufficient_sample_target_does_not_convert_search_to_out_of_sample(self):
        rules = generate_rules()
        raw = {rule["id"]: [] for rule in rules}
        fake_stats = {
            "decisive_trades": 110, "signals": 110, "win_rate": .6,
            "win_rate_lower_95": .57, "net_units": 20.,
        }
        with patch("research.qd_explore.build_candidates", return_value=raw), \
             patch("research.qd_explore._sample", return_value=([], fake_stats)):
            report = run_exploration(self.primary, self.candles, self.features)
        self.assertTrue(report["historical_target_met"])
        self.assertEqual(report["historical_target_candidates"], 72)
        self.assertEqual(report["selected_rule"], rules[0])
        self.assertFalse(report["research_qualified"])
        self.assertFalse(report["out_of_sample"])


if __name__ == "__main__":
    unittest.main()
