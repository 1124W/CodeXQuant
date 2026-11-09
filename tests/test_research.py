import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from research.engine import (STEP_MS, SYMBOLS, Trade, consume_archive, evaluate,
                             midnight_ms, rsi_series, signal_rows, split_trades,
                             stats, validate_candles)


def candles(start, days, seed=5):
    rng, price, rows = random.Random(seed), 100, []
    for i in range(days * 48):
        opened = price
        price += rng.uniform(-1, 1)
        rows.append((midnight_ms(start) + i * STEP_MS, opened,
                     max(opened, price) + 1, min(opened, price) - 1, price))
    return rows


class SettlementTests(unittest.TestCase):
    def test_boundary_uses_settlement_not_candle_open(self):
        cutoff = 10 * STEP_MS
        trades = [Trade("BTCUSDT", i * STEP_MS, (i + 1) * STEP_MS, 1, 100, 101, True)
                  for i in (8, 9, 10)]
        train, test = split_trades(trades, cutoff)
        self.assertEqual(train, [trades[0]])
        self.assertEqual(test, [trades[2]])
        self.assertNotIn(trades[1], train + test)

    def test_signal_at_close_and_flat_is_loss(self):
        series = [(i * STEP_MS, 100 - i, 101 - i, 98 - i, 99 - i) for i in range(32)]
        series[-1] = (31 * STEP_MS, 69, 70, 68, 69)
        trades = signal_rows(series, 3, 1, 65, "BTCUSDT")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].signal_ms, 31 * STEP_MS)
        self.assertEqual(trades[0].settlement_ms, 32 * STEP_MS)
        self.assertEqual(trades[0].entry, trades[0].exit)
        self.assertFalse(trades[0].win)

    def test_rsi_flat_up_down(self):
        self.assertEqual(rsi_series([100] * 20)[-1], 50)
        self.assertEqual(rsi_series(list(range(20)))[-1], 100)
        self.assertEqual(rsi_series(list(range(20, 0, -1)))[-1], 0)

    def test_costs_and_simultaneous_settlement(self):
        trades = [Trade("BTCUSDT", 0, STEP_MS, 1, 100, 101, True),
                  Trade("ETHUSDT", 0, STEP_MS, 1, 100, 99, False)]
        result = stats(trades, 1, payout=.8, fee=.02)
        self.assertEqual(result["net_units"], -.24)
        self.assertEqual(result["max_drawdown_units"], .24)
        self.assertEqual(result, stats(list(reversed(trades)), 1, payout=.8, fee=.02))
        self.assertLess(result["win_rate_lower_95"], .5)
        self.assertGreater(result["win_rate_upper_95"], .5)

    def test_no_samples_is_not_zero_percent(self):
        result = stats([], 1)
        self.assertIsNone(result["win_rate"])
        self.assertIsNone(result["win_rate_lower_95"])
        json.dumps(result, allow_nan=False)
        trades = [Trade("BTCUSDT", i * STEP_MS, (i + 1) * STEP_MS, 1, 100, 101, i == 0)
                  for i in range(3)]
        # Reporting must not round the quantities used in statistical gates.
        self.assertEqual(stats(trades, 1)["win_rate"], 1 / 3)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2025, 1, 1)
        self.end = self.start + timedelta(days=1)
        self.rows = candles(self.start, 1)

    def test_valid_continuous_data(self):
        validate_candles("BTCUSDT", self.rows, self.start, self.end)

    def test_missing_duplicate_and_off_grid_rejected(self):
        for rows in (self.rows[:-1], self.rows[:10] + [self.rows[9]] + self.rows[11:],
                     [tuple([r[0] + 1, *r[1:]]) for r in self.rows]):
            with self.assertRaises(ValueError):
                validate_candles("BTCUSDT", rows, self.start, self.end)

    def test_invalid_ohlc_rejected(self):
        for value in (float("nan"), float("inf"), -1, 10000):
            rows = list(self.rows)
            rows[0] = (*rows[0][:4], value)
            with self.assertRaises(ValueError):
                validate_candles("BTCUSDT", rows, self.start, self.end)

    def test_archive_checksum_microseconds_and_cache_integrity(self):
        stream = io.BytesIO()
        timestamp = midnight_ms(self.start)
        with zipfile.ZipFile(stream, "w") as zipped:
            zipped.writestr("candles.csv", f"{timestamp * 1000},100,102,99,101\n")
        archive = stream.getvalue()
        checksum = hashlib.sha256(archive).hexdigest().encode() + b"  candles.zip"
        with tempfile.TemporaryDirectory() as directory:
            with patch("research.engine.fetch_bytes", side_effect=[archive, checksum]):
                rows, manifest = {}, []
                consume_archive("test.zip", rows, directory, manifest)
                self.assertEqual(rows[timestamp], (timestamp, 100, 102, 99, 101))
                self.assertEqual(manifest[0]["sha256"], checksum.decode().split()[0])
            with patch("research.engine.fetch_bytes", side_effect=AssertionError("cache missed")):
                consume_archive("test.zip", {}, directory)
            Path(directory, "test.zip").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                consume_archive("test.zip", {}, directory)


class EvaluationTests(unittest.TestCase):
    def test_holdout_cannot_change_selection_or_training(self):
        start = date(2025, 1, 1)
        end = start + timedelta(days=90)
        series = {s: candles(start, 90, i + 5) for i, s in enumerate(SYMBOLS)}
        first = evaluate(series, start, end, fee=.02)
        self.assertIsNotNone(first["chosen"])
        # Keep the first 60 days unchanged, radically change only the holdout.
        changed = {s: rows[:60 * 48] + [(r[0], r[1] * 2, r[2] * 2, r[3] * 2, r[4] * 2)
                                       for r in rows[60 * 48:]] for s, rows in series.items()}
        second = evaluate(changed, start, end, fee=.02)
        self.assertEqual(first["all_candidates"], second["all_candidates"])
        self.assertEqual(first["chosen"]["parameters"], second["chosen"]["parameters"])
        self.assertAlmostEqual(first["break_even_win_rate"], 1.02 / 1.8, places=6)
        self.assertFalse(first["live_approved"])
        self.assertTrue(all("test" not in candidate for candidate in first["all_candidates"]))
        ledger = first["trade_ledger"]
        for split in ("train", "test"):
            entries = [t for t in ledger if t["split"] == split]
            self.assertEqual(len(entries), first["chosen"][split]["signals"])
            self.assertAlmostEqual(sum(t["net_units"] for t in entries), first["chosen"][split]["net_units"])
        json.dumps(first, allow_nan=False)

    def test_short_period_has_no_candidate(self):
        start = date(2025, 1, 1)
        report = evaluate({s: candles(start, 2) for s in SYMBOLS}, start, start + timedelta(days=2), 1)
        self.assertIsNone(report["chosen"])
        self.assertFalse(report["publishable"])

    def test_cli_invalid_request_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "results.json")
            output.write_text("original", encoding="utf-8")
            result = subprocess.run([sys.executable, "research/run.py", "--days", "30", "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("existing report preserved", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
