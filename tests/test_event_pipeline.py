"""Independent checks for verified event-index data and repeatable offline runs."""

import hashlib
import io
import random
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from research.event_data import (MINUTE_MS, archive_paths, data_fingerprint,
                                 download_index_minutes, read_archive, read_rest_day,
                                 validate_minutes)
from research.engine import SYMBOLS, midnight_ms
from research.event_engine import build_candidates
from research.backtest import parser, save_report, validate_request


def minute_rows(start, days=1):
    base = midnight_ms(start)
    return [(base + i * MINUTE_MS, 100.0, 102.0, 99.0, 101.0)
            for i in range(days * 1440)]


def packed(rows, header="open_time,open,high,low,close"):
    stream = io.BytesIO()
    csv = header + "\n" + "\n".join(",".join(str(value) for value in row) for row in rows)
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.csv", csv)
    blob = stream.getvalue()
    checksum = (hashlib.sha256(blob).hexdigest() + "  index.zip\n").encode()
    return blob, checksum


class EventDataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2025, 1, 1)
        self.end = self.start + timedelta(days=1)

    def test_only_index_archive_paths_and_namespaced_verified_cache_are_used(self):
        paths = list(archive_paths("BTCUSDT", self.start, self.end))
        self.assertEqual(paths, ["daily/indexPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2025-01-01.zip"])
        blob, checksum = packed([(midnight_ms(self.start) * 1000, 100, 102, 99, 101)])
        with tempfile.TemporaryDirectory() as directory:
            with patch("research.event_data.fetch_bytes", side_effect=[blob, checksum]) as fetch:
                rows, manifest = read_archive(paths[0], directory)
            self.assertEqual(rows[0], (midnight_ms(self.start), 100, 102, 99, 101))
            self.assertIn("/data/futures/um/daily/indexPriceKlines/", fetch.call_args_list[0].args[0])
            self.assertEqual(manifest["sha256"], hashlib.sha256(blob).hexdigest())
            cached = Path(directory, "futures", "um", paths[0])
            self.assertTrue(cached.is_file())
            with patch("research.event_data.fetch_bytes", side_effect=AssertionError("unexpected network")):
                cached_rows, _ = read_archive(paths[0], directory, offline=True)
            self.assertEqual(cached_rows, rows)
            cached.write_bytes(b"tampered archive")
            with patch("research.event_data.fetch_bytes", side_effect=AssertionError("unexpected network")):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    read_archive(paths[0], directory, offline=True)

    def test_missing_minute_duplicate_off_grid_and_invalid_ohlc_are_rejected(self):
        original = minute_rows(self.start)
        validate_minutes("BTCUSDT", original, self.start, self.end)
        invalid = [original[:-1], original[:5] + [original[4]] + original[6:],
                   [(row[0] + 1, *row[1:]) for row in original]]
        for value in (float("nan"), float("inf"), -1, 103):
            rows = list(original)
            rows[5] = (*rows[5][:4], value)
            invalid.append(rows)
        for rows in invalid:
            with self.subTest(first=rows[0], count=len(rows)):
                with self.assertRaises(ValueError):
                    validate_minutes("BTCUSDT", rows, self.start, self.end)

    def test_empty_or_nonhex_checksum_and_submillisecond_time_are_rejected(self):
        path = next(archive_paths("BTCUSDT", self.start, self.end))
        blob, checksum = packed([(midnight_ms(self.start), 100, 102, 99, 101)])
        for invalid_checksum in (b"", b"z" * 64):
            with patch("research.event_data.fetch_bytes", side_effect=[blob, invalid_checksum]):
                with self.assertRaisesRegex(ValueError, "Invalid checksum"):
                    read_archive(path)
        blob, checksum = packed([(midnight_ms(self.start) * 1000 + 1, 100, 102, 99, 101)])
        with patch("research.event_data.fetch_bytes", side_effect=[blob, checksum]):
            with self.assertRaisesRegex(ValueError, "Non-millisecond"):
                read_archive(path)

    def test_download_rejects_duplicate_rows_instead_of_silently_deduplicating(self):
        rows = minute_rows(self.start)
        with patch("research.event_data.read_archive", return_value=(rows + rows[:1], {"path": "fake"})):
            with self.assertRaisesRegex(ValueError, "Duplicate index candle"):
                download_index_minutes(self.start, self.end, workers=1)

    def test_monthly_404_fallback_remains_reproducible_offline(self):
        start, end = date(2025, 1, 30), date(2025, 2, 1)
        archives = {}
        for symbol in SYMBOLS:
            for offset in range(2):
                day = start + timedelta(days=offset)
                path = f"daily/indexPriceKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                archives[path] = packed(minute_rows(day))

        def fetch(url):
            path = url.split("/data/futures/um/", 1)[1]
            if path.startswith("monthly/"):
                raise urllib.error.HTTPError(url, 404, "Monthly bundle not published", {}, None)
            is_checksum = path.endswith(".CHECKSUM")
            return archives[path.removesuffix(".CHECKSUM")][int(is_checksum)]

        with tempfile.TemporaryDirectory() as directory:
            with patch("research.event_data.fetch_bytes", side_effect=fetch):
                online, manifest = download_index_minutes(start, end, directory, workers=1)
            self.assertEqual(len(manifest), 4)
            with patch("research.event_data.fetch_bytes", side_effect=AssertionError("unexpected network")):
                offline, offline_manifest = download_index_minutes(start, end, directory, offline=True, workers=1)
            self.assertEqual(data_fingerprint(online), data_fingerprint(offline))
            self.assertEqual(manifest, offline_manifest)
            self.assertTrue(all(len(series) == 2880 for series in offline.values()))

    def test_fingerprint_is_market_order_independent_and_price_sensitive(self):
        rows = minute_rows(self.start)
        first = {"BTCUSDT": rows, "ETHUSDT": rows}
        second = {"ETHUSDT": rows, "BTCUSDT": rows}
        self.assertEqual(data_fingerprint(first), data_fingerprint(second))
        changed = list(rows)
        changed[-1] = (*changed[-1][:4], 100.5)
        self.assertNotEqual(data_fingerprint(first), data_fingerprint({**first, "BTCUSDT": changed}))

    def test_rest_repair_is_complete_persisted_and_tamper_evident_offline(self):
        rows = minute_rows(self.start)
        payload = json.dumps(rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            with patch("research.event_data.fetch_bytes", return_value=payload) as fetch:
                recovered, manifest = read_rest_day("BTCUSDT", self.start, directory)
            self.assertEqual(recovered, rows)
            url = fetch.call_args.args[0]
            self.assertTrue(url.startswith("https://fapi.binance.com/fapi/v1/indexPriceKlines?pair=BTCUSDT&interval=1m"))
            self.assertIn(f"startTime={midnight_ms(self.start)}", url)
            self.assertEqual(manifest["verification"], "local_sha256_of_official_rest_snapshot")
            self.assertEqual(manifest["sha256"], hashlib.sha256(payload).hexdigest())
            with patch("research.event_data.fetch_bytes", side_effect=AssertionError("unexpected network")):
                offline, offline_manifest = read_rest_day("BTCUSDT", self.start, directory, offline=True)
            self.assertEqual(recovered, offline)
            self.assertEqual(manifest, offline_manifest)
            cached = Path(directory, "futures", "um", manifest["path"])
            cached.write_bytes(payload + b" ")
            with patch("research.event_data.fetch_bytes", side_effect=AssertionError("unexpected network")):
                with self.assertRaisesRegex(ValueError, "REST snapshot checksum mismatch"):
                    read_rest_day("BTCUSDT", self.start, directory, offline=True)

    def test_rest_repair_rejects_incomplete_grid_invalid_ohlc_and_malformed_payload(self):
        original = minute_rows(self.start)
        wrong_ohlc = list(original)
        wrong_ohlc[7] = (*wrong_ohlc[7][:4], 1000.)
        duplicate = original[:5] + [original[4]] + original[6:]
        malformed = [original[:-1], wrong_ohlc, duplicate, {"code": -1}, [[1]], "not a list"]
        for raw in malformed:
            with self.subTest(kind=type(raw).__name__, size=len(raw)):
                with patch("research.event_data.fetch_bytes", return_value=json.dumps(raw).encode()):
                    with self.assertRaises(ValueError):
                        read_rest_day("BTCUSDT", self.start)

    def test_missing_minutes_are_repaired_with_observed_rest_rows_without_interpolation(self):
        full = minute_rows(self.start)
        missing = {5, 7, 90}
        observed = [row for index, row in enumerate(full) if index not in missing]
        repaired = list(full)
        for index in missing:
            repaired[index] = (full[index][0], 200., 202., 199., 201.)

        def archived(path, *_):
            return observed, {"path": path}

        def snapshot(symbol, day, *_):
            self.assertEqual(day, self.start)
            return repaired, {"path": f"rest/{symbol}/{day}", "verification": "local_sha256_of_official_rest_snapshot"}

        with patch("research.event_data.read_archive", side_effect=archived), \
             patch("research.event_data.read_rest_day", side_effect=snapshot) as rest:
            recovered, manifest = download_index_minutes(self.start, self.end, workers=1)
        self.assertEqual(rest.call_count, 2)
        self.assertEqual(len(manifest), 4)
        for symbol in SYMBOLS:
            self.assertEqual(recovered[symbol], repaired)
            self.assertEqual(recovered[symbol][5][1], 200.)
            self.assertEqual(recovered[symbol][6][1], 100.)

    def test_conflicting_archive_and_rest_rows_are_rejected(self):
        full = minute_rows(self.start)
        observed = full[:5] + full[6:]
        conflicting = list(full)
        conflicting[7] = (full[7][0], 200., 202., 199., 201.)
        with patch("research.event_data.read_archive", return_value=(observed, {"path": "archive"})), \
             patch("research.event_data.read_rest_day", return_value=(conflicting, {"path": "rest"})):
            with self.assertRaisesRegex(ValueError, "REST/archive index conflict"):
                download_index_minutes(self.start, self.end, workers=1)


class EventExecutionCausalityTests(unittest.TestCase):
    def test_changed_future_cannot_change_previously_settled_signals_or_fills(self):
        start, generator, price = date(2025, 1, 1), random.Random(314), 100.
        rows = []
        for index in range(6 * 1440):
            opened = price
            price = max(1, price + generator.uniform(-.6, .6))
            rows.append((midnight_ms(start) + index * MINUTE_MS, opened,
                         max(opened, price) + .1, min(opened, price) - .1, price))
        boundary = midnight_ms(start + timedelta(days=4))
        changed = [row if row[0] < boundary else (row[0], *(value * 2 for value in row[1:]))
                   for row in rows]
        original = build_candidates({symbol: rows for symbol in SYMBOLS})
        mutated = build_candidates({symbol: changed for symbol in SYMBOLS})
        count = 0
        for rule in original:
            prior = [trade for trade in original[rule] if trade["settlement_ms"] < boundary]
            count += len(prior)
            self.assertEqual(prior, [trade for trade in mutated[rule] if trade["settlement_ms"] < boundary])
        self.assertGreater(count, 20, "fixture must exercise actual strategy decisions")

    def test_later_intraminute_high_low_close_cannot_change_open_price_execution(self):
        rows = minute_rows(date(2025, 1, 1), 3)
        changed = list(rows)
        # First eligible signal is the close of completed bar index 60.
        # Fill one minute later, expire thirty minutes after that, using opens.
        entry_index, exit_index = 61 * 30 + 1, 61 * 30 + 31
        changed[entry_index] = (changed[entry_index][0], 100., 121., 99., 120.)
        changed[exit_index] = (changed[exit_index][0], 100., 102., 79., 80.)

        def one_signal(rule, bars, features, index):
            return 1 if index == 60 else 0

        with patch("research.event_engine._direction", side_effect=one_signal):
            original = build_candidates({symbol: rows for symbol in SYMBOLS})
            mutated = build_candidates({symbol: changed for symbol in SYMBOLS})
        self.assertEqual(original, mutated)
        self.assertTrue(original)
        for trades in original.values():
            self.assertEqual(len(trades), 2)
            for trade in trades:
                self.assertEqual(trade["entry"], 100.)
                self.assertEqual(trade["exit"], 100.)
                self.assertEqual(trade["entry_ms"] - trade["signal_ms"], MINUTE_MS)
                self.assertEqual(trade["settlement_ms"] - trade["entry_ms"], 30 * MINUTE_MS)


class EventCliArtifactTests(unittest.TestCase):
    def test_invalid_request_preserves_existing_report_and_trade_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ledger = Path(directory, "events.json"), Path(directory, "trades.csv")
            output.write_text("original report", encoding="utf-8")
            ledger.write_text("original ledger", encoding="utf-8")
            result = subprocess.run([sys.executable, "research/backtest.py", "--days", "1",
                                     "--output", str(output), "--trades-output", str(ledger)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("existing report preserved", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "original report")
            self.assertEqual(ledger.read_text(encoding="utf-8"), "original ledger")

    def test_saving_minimal_valid_run_produces_json_csv_and_history(self):
        report = {"provenance": {"run_id": "fixture0001"}, "validation_checks": {},
                  "walk_forward": {"aggregate": {}}, "final_holdout": {"test": {}},
                  "publishable": False, "trade_ledger": [
                      {"symbol": "BTCUSDT", "signal_ms": 1, "entry_ms": 60001,
                       "settlement_ms": 1860001, "direction": 1, "entry": 100, "exit": 101,
                       "outcome": "win", "net_units": .79, "split": "walk_forward", "fold_id": "fold_01"}]}
        with tempfile.TemporaryDirectory() as directory:
            output, ledger = Path(directory, "events.json"), Path(directory, "trades.csv")
            save_report(report, output, ledger)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertIn("BTCUSDT", ledger.read_text(encoding="utf-8"))
            history = Path(directory, ".runs", "fixture0001.json")
            self.assertEqual(json.loads(history.read_text(encoding="utf-8"))["provenance"]["run_id"], "fixture0001")

    def test_unfundable_cost_and_invalid_horizons_fail_before_download(self):
        cli = parser()
        for extra in (["--initial-bankroll", ".5"], ["--fee", "inf"], ["--horizon-minutes", "17"],
                      ["--target-win-rate", ".5"], ["--latency-minutes", "0"],
                      ["--adverse-bps", "9998"], ["--daily-loss-limit", ".5"]):
            with self.subTest(arguments=extra):
                with self.assertRaises(ValueError):
                    validate_request(cli.parse_args(extra))


if __name__ == "__main__":
    unittest.main()
