"""Verify the published canonical report independently of model training."""

import copy
import csv
import io
import json
import math
import unittest
from pathlib import Path

from research.qd_demo_web import create_app, validate_report
from research.workbuddy_run import REPORT, TRADES, csv_text, years_before
from datetime import date


@unittest.skipUnless(REPORT.exists(), "Generate the WorkBuddy report first")
class PublishedWorkBuddyTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_all_trade_prices_payouts_and_summaries_reconcile(self):
        validate_report(self.report)
        self.assertEqual(self.report["symbols"], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(self.report["config"]["horizon_minutes"], 30)
        self.assertEqual(self.report["config"]["payout"], .85)
        self.assertEqual(self.report["config"]["fee"], 0)
        self.assertFalse(self.report["live_approved"])
        self.assertEqual(sum(s["signals"] for s in self.report["by_symbol"].values()), self.report["test"]["signals"])
        self.assertAlmostEqual(sum(s["net_units"] for s in self.report["monthly"]), self.report["test"]["net_units"])
        self.assertTrue(all(f["train_latest_settlement_ms"] < f["test_start_ms"] for f in self.report["folds"] if f["status"] == "evaluated"))

    def test_invalid_scope_or_forged_profit_is_rejected(self):
        for mutate in (lambda r: r["config"].update(horizon_minutes=10),
                       lambda r: r["config"].update(payout=.80),
                       lambda r: r["test"].update(net_units=r["test"]["net_units"] + 1)):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_csv_and_api_downloads_match_canonical_run(self):
        records = list(csv.DictReader(io.StringIO(csv_text(self.report))))
        self.assertEqual(len(records), len(self.report["trade_ledger"]))
        self.assertEqual({row["run_id"] for row in records}, {self.report["provenance"]["run_id"]})
        client = create_app(REPORT, TRADES).test_client()
        self.assertEqual(client.get("/api/event-contracts/report").json["data"]["provenance"]["run_id"], self.report["provenance"]["run_id"])
        for path in ("/demo/quantdinger.html", "/workbuddy.js", "/index.html"):
            with client.get(path) as response:
                self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/external/events-workBuddy/live/strategy.py").status_code, 404)

    def test_three_year_calendar_and_final_equity(self):
        self.assertEqual(years_before(date(2026, 11, 9)), date(2023, 11, 9))
        self.assertAlmostEqual(self.report["equity_curve"][-1]["balance_units"],
                               self.report["config"]["initial_bankroll"] + self.report["test"]["net_units"])


if __name__ == "__main__":
    unittest.main()
