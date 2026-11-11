"""The local demo must not present a corrupt or forged report as valid."""

import copy
import csv
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from research.event_engine import event_stats, utc_ms
from research.qd_demo import csv_payload, months_before
from research.qd_demo_web import create_app, validate_report
from research.qd_event_engine import DEFAULT_CONFIG, generate_rules


def report_fixture():
    start, end = utc_ms(date(2026, 8, 9)), utc_ms(date(2026, 11, 9))
    config = {**DEFAULT_CONFIG, "min_test_trades": 1}
    rule = generate_rules()[0]
    trades = []
    for i, (outcome, price, pnl) in enumerate((("win", 101., 3.95), ("loss", 99., -5.05), ("refund", 100., -.05))):
        signal = start + i * 60 * 60000
        entry = signal + 60000
        trades.append({"symbol": "BTCUSDT", "rule_id": rule["id"], "signal_ms": signal,
                       "entry_ms": entry, "settlement_ms": entry + 10 * 60000,
                       "direction": 1, "entry": 100., "exit": price, "outcome": outcome,
                       "stake": 5., "net_units": pnl, "horizon_minutes": 10,
                       "timeframe_minutes": 5, "split": "test"})
    return {"engine": "quantdinger-event-demo-v1", "config": config,
            "test": event_stats(trades, start, end, bootstrap=False), "trade_ledger": trades,
            "windows": {"test": {"start_utc": "2026-08-09", "end_exclusive_utc": "2026-11-09"}},
            "selected_rule": rule, "symbols": ["BTCUSDT", "ETHUSDT"],
            "selection_eligible": False, "demo_pass": False, "live_approved": False,
            "provenance": {"run_id": "fixture-run"}}


class DemoReportTests(unittest.TestCase):
    def test_calendar_months_not_fixed_ninety_days(self):
        self.assertEqual(months_before(date(2026, 11, 9)), date(2026, 8, 9))
        self.assertEqual(months_before(date(2026, 5, 31)), date(2026, 2, 28))

    def test_refund_counts_costs_and_csv_reconcile(self):
        report = report_fixture()
        self.assertIs(validate_report(report), report)
        rows = list(csv.DictReader(io.StringIO(csv_payload(report))))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["run_id"] for row in rows}, {"fixture-run"})
        self.assertAlmostEqual(sum(float(row["net_units"]) for row in rows), report["test"]["net_units"])

    def test_forged_pass_bad_counts_cost_or_settlement_are_rejected(self):
        for mutate in (
            lambda r: r.update(demo_pass=True),
            lambda r: r["test"].update(wins=9),
            lambda r: r["trade_ledger"][0].update(net_units=100),
            lambda r: r["trade_ledger"][0].update(exit=99),
            lambda r: r["trade_ledger"][0].update(entry_ms=r["trade_ledger"][0]["entry_ms"] + 60000),
        ):
            report = report_fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_api_exposes_only_validated_read_only_results(self):
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "report.json"
            report_path.write_text(json.dumps(report_fixture()), encoding="utf-8")
            app = create_app(report_path, Path(folder) / "missing.csv")
            client = app.test_client()
            response = client.get("/api/event-contracts/report")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["data"]["test"]["signals"], 3)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            # CSV is derived from canonical JSON even when the optional CSV file is absent.
            response = client.get("/api/event-contracts/trades.csv")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(response.text)))), 3)
            self.assertFalse(client.get("/api/event-contracts/health").json["data"]["order_execution_enabled"])
            self.assertEqual(client.post("/api/event-contracts/report").status_code, 405)
            self.assertEqual(client.get("/.git/config").status_code, 404)
            corrupt = report_fixture()
            corrupt["demo_pass"] = True
            report_path.write_text(json.dumps(corrupt), encoding="utf-8")
            self.assertEqual(client.get("/api/event-contracts/report").status_code, 503)


if __name__ == "__main__":
    unittest.main()
