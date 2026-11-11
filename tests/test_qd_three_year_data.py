"""Regression checks for source integrity and minute-grid alignment."""

from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

import numpy as np
import pandas as pd

from research.qd_three_year_data import (
    DAY_MS, INDEX_COLUMNS, MINUTE_MS, SPOT_COLUMNS, _ms, archive_paths,
    _rest_day, load_kind, load_market, parse_archive, read_archive, validate_frame,
)


def zipped(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("observations.csv", "\n".join(",".join(map(str, row)) for row in rows) + "\n")
    return stream.getvalue()


def raw_row(timestamp):
    return [timestamp, 100, 102, 99, 101, 10, timestamp + 59999, 1000, 5, 4, 400, 0]


def day_frame(kind, day=date(2026, 1, 1)):
    timestamps = np.arange(_ms(day), _ms(day) + DAY_MS, MINUTE_MS, dtype=np.int64)
    frame = pd.DataFrame({"timestamp": timestamps, "open": 100., "high": 102., "low": 99., "close": 101.})
    if kind == "spot":
        for column, value in (("volume", 10.), ("quote_volume", 1000.), ("trades", 5.),
                              ("taker_buy_base", 4.), ("taker_buy_quote", 400.)):
            frame[column] = value
    return frame


class ArchiveTests(unittest.TestCase):
    def test_months_and_daily_tail_stay_in_correct_source_namespace(self):
        paths = list(archive_paths("spot", "BTCUSDT", date(2023, 11, 9), date(2026, 11, 9)))
        self.assertEqual(paths[0], "monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-11.zip")
        self.assertEqual(paths[-1], "daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-11-08.zip")
        self.assertEqual(len(paths), 44)
        self.assertTrue(all("indexPriceKlines" in p for p in archive_paths(
            "index", "ETHUSDT", date(2026, 10, 1), date(2026, 11, 1))))

    def test_microseconds_and_field_mapping_are_exact(self):
        timestamp = _ms(date(2026, 1, 1))
        payload = zipped([raw_row(timestamp * 1000), raw_row((timestamp + MINUTE_MS) * 1000)])
        spot = parse_archive(payload, "spot")
        self.assertEqual(list(spot.columns), SPOT_COLUMNS)
        self.assertEqual(spot.timestamp.tolist(), [timestamp, timestamp + MINUTE_MS])
        self.assertEqual(spot.trades.tolist(), [5., 5.])
        self.assertEqual(spot.taker_buy_base.tolist(), [4., 4.])
        self.assertEqual(list(parse_archive(payload, "index").columns), INDEX_COLUMNS)
        with self.assertRaisesRegex(ValueError, "Non-millisecond"):
            parse_archive(zipped([raw_row(timestamp * 1000 + 1)]), "spot")

    def test_exchange_checksum_and_offline_cache_tampering_fail_closed(self):
        payload = zipped([raw_row(_ms(date(2026, 1, 1)))])
        checksum = (hashlib.sha256(payload).hexdigest() + "  observations.zip\n").encode()
        relative = "daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01-01.zip"
        with tempfile.TemporaryDirectory() as directory:
            with patch("research.qd_three_year_data._fetch", side_effect=[payload, checksum]):
                frame, source = read_archive("spot", relative, directory)
            self.assertEqual(len(frame), 1)
            self.assertEqual(source["verification"], "exchange_archive_sha256")
            cached = Path(directory) / "spot" / "1m" / relative
            self.assertTrue(cached.is_file())
            cached.write_bytes(payload + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_archive("spot", relative, directory, offline=True)


class GridTests(unittest.TestCase):
    def test_official_website_fallback_is_persisted_and_offline_reproducible(self):
        day = date(2024, 8, 12)
        raw = [raw_row(_ms(day) + i * MINUTE_MS) for i in range(1440)]
        payload = json.dumps(raw).encode()
        restricted = urllib.error.HTTPError("https://fapi.binance.com/", 451, "restricted", {}, None)
        with tempfile.TemporaryDirectory() as folder:
            with patch("research.qd_three_year_data._fetch", side_effect=[restricted, payload]):
                observed, source = _rest_day("index", "BTCUSDT", day, Path(folder), False)
            self.assertEqual(len(observed), 1440)
            self.assertTrue(source["urls"][0].startswith("https://www.binance.com/fapi/v1/indexPriceKlines"))
            cached = Path(folder) / source["path"]
            self.assertEqual(cached.read_bytes(), payload)
            with patch("research.qd_three_year_data._fetch", side_effect=AssertionError("network")):
                offline, source_again = _rest_day("index", "BTCUSDT", day, Path(folder), True)
            pd.testing.assert_frame_equal(observed, offline)
            self.assertEqual(source_again["sha256"], source["sha256"])

    def test_invalid_ohlc_volume_duplicates_and_missing_minutes_are_rejected(self):
        frame = day_frame("spot")
        validate_frame(frame, "spot", start=date(2026, 1, 1), end=date(2026, 1, 2))
        for name, mutate in (
            ("ohlc", lambda x: x.__setitem__("low", 105.)),
            ("volume", lambda x: x.__setitem__("volume", -1.)),
            ("tradecount", lambda x: x.__setitem__("trades", 1.5)),
            ("taker", lambda x: x.__setitem__("taker_buy_base", 50.)),
        ):
            broken = frame.copy()
            mutate(broken)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_frame(broken, "spot")
        with self.assertRaises(ValueError):
            validate_frame(pd.concat([frame, frame.iloc[-1:]], ignore_index=True), "spot")
        with self.assertRaises(ValueError):
            validate_frame(frame.drop(index=2), "spot")

    def test_missing_minutes_repair_only_observed_rest_rows(self):
        day = date(2026, 1, 1)
        complete = day_frame("index")
        sparse = complete.drop(index=[10, 100]).reset_index(drop=True)
        def archive(kind, path, cache_dir, offline):
            if path == "month.zip":
                return sparse, {"kind": "index", "path": path}
            raise urllib.error.HTTPError(path, 404, "missing", {}, None)
        with tempfile.TemporaryDirectory() as folder, \
                patch("research.qd_three_year_data.archive_paths", return_value=iter(["month.zip"])), \
                patch("research.qd_three_year_data.read_archive", side_effect=archive), \
                patch("research.qd_three_year_data._rest_day", return_value=(complete, {"kind": "index", "path": "rest.json"})) as repair:
            recovered, sources = load_kind("index", "BTCUSDT", day, day + timedelta(days=1), folder)
        pd.testing.assert_frame_equal(recovered, complete)
        self.assertEqual(len(sources), 2)
        self.assertEqual(repair.call_args.args[2], day)

    def test_conflicting_rest_overlap_is_not_silently_overwritten(self):
        day = date(2026, 1, 1)
        complete = day_frame("index")
        sparse = complete.drop(index=10).reset_index(drop=True)
        conflict = complete.copy()
        conflict.loc[20, "close"] = 100.5
        def archive(kind, path, cache_dir, offline):
            if path == "month.zip":
                return sparse, {"kind": "index", "path": path}
            raise urllib.error.HTTPError(path, 404, "missing", {}, None)
        with tempfile.TemporaryDirectory() as folder, \
                patch("research.qd_three_year_data.archive_paths", return_value=iter(["month.zip"])), \
                patch("research.qd_three_year_data.read_archive", side_effect=archive), \
                patch("research.qd_three_year_data._rest_day", return_value=(conflict, {"kind": "index", "path": "rest.json"})):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_kind("index", "BTCUSDT", day, day + timedelta(days=1), folder)


if __name__ == "__main__":
    unittest.main()
