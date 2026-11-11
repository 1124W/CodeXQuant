"""Verified dual-source minutes: spot features and USD-M index settlement.

Archive hashes are exchange-issued CHECKSUM files. REST repairs instead retain
official observations and their original response pages with local hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT")
MINUTE_MS = 60_000
DAY_MS = 86_400_000
VISION = "https://data.binance.vision/data"
INDEX_COLUMNS = ["timestamp", "open", "high", "low", "close"]
SPOT_COLUMNS = INDEX_COLUMNS + ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _ms(day):
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _fetch(url):
    curl = shutil.which("curl.exe") or shutil.which("curl")
    for attempt in range(3):
        try:
            if curl:
                result = subprocess.run([curl, "--http1.1", "--silent", "--show-error", "--location",
                                         "--fail", "--connect-timeout", "8", "--max-time", "20", url],
                                        capture_output=True, timeout=23)
                if result.returncode == 0:
                    return result.stdout
                message = result.stderr.decode("utf-8", errors="replace")
                for code in (404, 403, 451):
                    if str(code) in message:
                        raise urllib.error.HTTPError(url, code, message, {}, None)
                raise OSError(f"curl download failed ({result.returncode}): {message.strip()}")
            with urllib.request.urlopen(url, timeout=20) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (OSError, urllib.error.URLError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)


def _atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def archive_paths(kind, symbol, start, end):
    if kind not in ("spot", "index") or symbol not in SYMBOLS or start >= end:
        raise ValueError("Require spot/index, BTCUSDT/ETHUSDT, and an increasing date range")
    month = start.replace(day=1)
    product = "klines" if kind == "spot" else "indexPriceKlines"
    while month < end:
        following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following <= end:
            yield f"monthly/{product}/{symbol}/1m/{symbol}-1m-{month:%Y-%m}.zip"
        else:
            day = max(start, month)
            while day < end:
                yield f"daily/{product}/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                day += timedelta(days=1)
        month = following


def _locations(kind, relative, cache_dir):
    if ".." in relative or "\\" in relative or ":" in relative or relative.startswith("/"):
        raise ValueError("Invalid archive path")
    if kind == "spot":
        return Path(cache_dir) / "spot" / "1m" / relative, f"{VISION}/spot/{relative}"
    return Path(cache_dir) / "futures" / "um" / relative, f"{VISION}/futures/um/{relative}"


def parse_archive(payload, kind):
    columns = SPOT_COLUMNS if kind == "spot" else INDEX_COLUMNS
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one archive CSV")
        with archive.open(names[0]) as stream:
            frame = pd.read_csv(stream, header=None, names=RAW_COLUMNS,
                                usecols=[RAW_COLUMNS.index(name) for name in columns],
                                dtype="str")
    if frame.empty:
        raise ValueError("Archive has no observations")
    if frame.iloc[0]["timestamp"] in ("open_time", "openTime"):
        frame = frame.iloc[1:].copy()
    raw = pd.to_numeric(frame["timestamp"], errors="raise").to_numpy(dtype=np.int64)
    micro = raw > 100_000_000_000_000
    if np.any(raw[micro] % 1000):
        raise ValueError("Non-millisecond archive timestamp")
    raw[micro] //= 1000
    frame["timestamp"] = raw
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame = frame[columns].reset_index(drop=True)
    validate_frame(frame, kind, continuous=False)
    return frame


def validate_frame(frame, kind, *, start=None, end=None, continuous=True):
    columns = SPOT_COLUMNS if kind == "spot" else INDEX_COLUMNS
    if list(frame.columns) != columns or frame.empty:
        raise ValueError("Wrong or empty minute data schema")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite minute observations")
    timestamps = frame["timestamp"].to_numpy()
    if np.any(timestamps != timestamps.astype(np.int64)) or np.any(timestamps % MINUTE_MS):
        raise ValueError("Off-grid minute timestamp")
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0) or continuous and np.any(diffs != MINUTE_MS):
        raise ValueError("Missing, duplicate or unordered minute observations")
    o, h, l, c = (frame[name].to_numpy() for name in ("open", "high", "low", "close"))
    if (np.any(values[:, 1:5] <= 0) or np.any(l > np.minimum(o, c))
            or np.any(h < np.maximum(o, c)) or np.any(h < l)):
        raise ValueError("Invalid minute OHLC")
    if kind == "spot":
        if np.any(values[:, 5:] < 0):
            raise ValueError("Negative spot volume or trade count")
        if (np.any(frame["trades"] != np.floor(frame["trades"]))
                or np.any(frame["taker_buy_base"] > frame["volume"] + 1e-7)
                or np.any(frame["taker_buy_quote"] > frame["quote_volume"] + 1e-5)):
            raise ValueError("Invalid spot trade count or taker volume")
    if start is not None and end is not None:
        expected = (end - start).days * 1440
        if (len(frame) != expected or timestamps[0] != _ms(start)
                or timestamps[-1] != _ms(end) - MINUTE_MS):
            raise ValueError(f"Incomplete {kind} minute history: {len(frame)} of {expected}")


def read_archive(kind, relative, cache_dir, offline=False):
    cached, url = _locations(kind, relative, cache_dir)
    checksum_path = cached.with_suffix(".zip.CHECKSUM")
    if cached.is_file() and checksum_path.is_file():
        payload, checksum = cached.read_bytes(), checksum_path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Verified archive cache missing: {cached}")
        payload, checksum = _fetch(url), _fetch(url + ".CHECKSUM")
    tokens = checksum.decode("ascii").split()
    expected = tokens[0].lower() if tokens else ""
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid exchange checksum")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Archive checksum mismatch: {cached}")
    frame = parse_archive(payload, kind)
    if not cached.is_file() or not checksum_path.is_file():
        _atomic_bytes(cached, payload)
        _atomic_bytes(checksum_path, checksum)
    return frame, {"kind": kind, "path": relative, "url": url,
                   "sha256": expected, "verification": "exchange_archive_sha256"}


def _rest_day(kind, symbol, day, cache_dir, offline):
    # Preserve compatibility with existing verified index REST snapshots.
    prefix = Path(cache_dir) / "futures" / "um" / "rest" / "indexPriceKlines" if kind == "index" else Path(cache_dir) / "spot" / "1m" / "rest" / "klines"
    cached = prefix / symbol / "1m" / f"{symbol}-1m-{day}.json"
    hash_path = cached.with_suffix(".json.sha256")
    source_path = cached.with_suffix(".json.sources.json")
    source_urls = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else []
    if cached.is_file() and hash_path.is_file():
        payload = cached.read_bytes()
        if hashlib.sha256(payload).hexdigest() != hash_path.read_text(encoding="ascii").strip():
            raise ValueError("REST snapshot checksum mismatch")
        observations = json.loads(payload)
    else:
        if offline:
            raise FileNotFoundError(f"Verified REST repair missing: {cached}")
        endpoint = "https://fapi.binance.com/fapi/v1/indexPriceKlines" if kind == "index" else "https://api.binance.com/api/v3/klines"
        pair_key = "pair" if kind == "index" else "symbol"
        maximum = 1500 if kind == "index" else 1000
        _log(f"{symbol} {kind}: requesting official REST day {day}")
        observations, response_pages, cursor = [], [], _ms(day)
        while cursor < _ms(day) + DAY_MS:
            url = (f"{endpoint}?{pair_key}={symbol}&interval=1m&startTime={cursor}"
                   f"&endTime={_ms(day)+DAY_MS-1}&limit={maximum}")
            try:
                raw_response = _fetch(url)
            except urllib.error.HTTPError as error:
                if kind != "index" or error.code != 451:
                    raise
                # Binance's public website exposes the same read-only index API.
                # Preserve the actual endpoint used in the local source manifest.
                url = url.replace("https://fapi.binance.com/", "https://www.binance.com/")
                raw_response = _fetch(url)
            chunk = json.loads(raw_response)
            if not isinstance(chunk, list) or not chunk:
                raise ValueError(f"Incomplete official {kind} REST repair: {day}")
            observations.extend(chunk)
            response_pages.append(raw_response)
            source_urls.append(url)
            next_cursor = int(chunk[-1][0]) + MINUTE_MS
            if next_cursor <= cursor:
                raise ValueError("REST pagination did not advance")
            cursor = next_cursor
        payload = response_pages[0] if len(response_pages) == 1 else json.dumps(observations, separators=(",", ":")).encode("utf-8")
        # Keep exact response bytes even when spot pagination needs two pages.
        for page_number, raw_response in enumerate(response_pages):
            page_path = cached.with_suffix(f".response-{page_number}.json")
            _atomic_bytes(page_path, raw_response)
            _atomic_bytes(page_path.with_suffix(".json.sha256"),
                          (hashlib.sha256(raw_response).hexdigest() + "\n").encode("ascii"))
        _atomic_bytes(source_path, json.dumps(source_urls, indent=2).encode("utf-8"))
    required_length = 11 if kind == "spot" else 5
    if not isinstance(observations, list) or any(not isinstance(row, list) or len(row) < required_length for row in observations):
        raise ValueError("Malformed official REST snapshot")
    columns = SPOT_COLUMNS if kind == "spot" else INDEX_COLUMNS
    positions = [RAW_COLUMNS.index(column) for column in columns]
    frame = pd.DataFrame([[row[i] for i in positions] for row in observations], columns=columns)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    validate_frame(frame, kind, start=day, end=day + timedelta(days=1))
    sha = hashlib.sha256(payload).hexdigest()
    if not cached.is_file() or not hash_path.is_file():
        _atomic_bytes(cached, payload)
        _atomic_bytes(hash_path, (sha + "\n").encode("ascii"))
    return frame, {"kind": kind, "path": str(cached.relative_to(cache_dir)),
                   "sha256": sha, "urls": source_urls,
                   "verification": "local_sha256_of_official_rest_snapshot",
                   "purpose": "repair_missing_archive_minutes"}


def load_kind(kind, symbol, start, end, cache_dir, offline=False, workers=8):
    tasks = list(archive_paths(kind, symbol, start, end))
    lower, upper = _ms(start), _ms(end)

    def consume(relative):
        try:
            frame, entry = read_archive(kind, relative, cache_dir, offline)
            return [frame], [entry]
        except (FileNotFoundError, urllib.error.HTTPError) as error:
            missing = isinstance(error, FileNotFoundError) or error.code == 404
            if not missing or not relative.startswith("monthly/"):
                if missing and relative.startswith("daily/"):
                    day = date.fromisoformat(Path(relative).stem[-10:])
                    frame, entry = _rest_day(kind, symbol, day, cache_dir, offline)
                    return [frame], [entry]
                raise
            month = date.fromisoformat(Path(relative).stem[-7:] + "-01")
            following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
            frames, entries, day = [], [], max(start, month)
            product = "klines" if kind == "spot" else "indexPriceKlines"
            while day < min(end, following):
                daily = f"daily/{product}/{symbol}/1m/{symbol}-1m-{day}.zip"
                try:
                    frame, entry = read_archive(kind, daily, cache_dir, offline)
                except (FileNotFoundError, urllib.error.HTTPError) as daily_error:
                    if not isinstance(daily_error, FileNotFoundError) and daily_error.code != 404:
                        raise
                    frame, entry = _rest_day(kind, symbol, day, cache_dir, offline)
                frames.append(frame)
                entries.append(entry)
                day += timedelta(days=1)
            return frames, entries

    frames, manifest = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(consume, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            chunks, entries = future.result()
            frames.extend(chunk.loc[(chunk.timestamp >= lower) & (chunk.timestamp < upper)] for chunk in chunks)
            manifest.extend(entries)
            if number % 6 == 0 or number == len(tasks):
                _log(f"{symbol} {kind}: verified {number}/{len(tasks)} archive groups")
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    validate_frame(combined, kind, continuous=False)
    # A published bundle can omit a whole UTC day. Recover only exact official observations.
    observed = combined["timestamp"].to_numpy(dtype=np.int64)
    missing = np.setdiff1d(np.arange(lower, upper, MINUTE_MS, dtype=np.int64), observed, assume_unique=True)
    if missing.size:
        days = sorted({datetime.fromtimestamp(int(value) / 1000, timezone.utc).date() for value in missing})
        _log(f"{symbol} {kind}: repairing incomplete UTC days {', '.join(map(str, days))}")
        repairs = []
        def recover_day(day):
            product = "klines" if kind == "spot" else "indexPriceKlines"
            relative = f"daily/{product}/{symbol}/1m/{symbol}-1m-{day}.zip"
            try:
                frame, entry = read_archive(kind, relative, cache_dir, offline)
                if len(frame) == 1440:
                    validate_frame(frame, kind, start=day, end=day + timedelta(days=1))
                    return frame, entry
                _log(f"{symbol} {kind}: daily archive {day} has {len(frame)}/1440 rows; using official REST")
                return _rest_day(kind, symbol, day, cache_dir, offline)
            except (FileNotFoundError, urllib.error.HTTPError) as error:
                if not isinstance(error, FileNotFoundError) and error.code != 404:
                    raise
                return _rest_day(kind, symbol, day, cache_dir, offline)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for recovered, entry in pool.map(recover_day, days):
                overlap = recovered[recovered.timestamp.isin(combined.timestamp)]
                if not overlap.empty:
                    archived = combined.set_index("timestamp").loc[overlap.timestamp]
                    if not np.array_equal(archived.to_numpy(), overlap.set_index("timestamp").to_numpy()):
                        raise ValueError(f"Conflicting {kind} archive and REST observations")
                repairs.append(recovered[recovered.timestamp.isin(missing)])
                manifest.append(entry)
        combined = pd.concat([combined, *repairs], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    validate_frame(combined, kind, start=start, end=end)
    return combined, sorted(manifest, key=lambda entry: (entry["kind"], entry["path"]))


def load_market(symbol, start, end, cache_dir, offline=False):
    """Return strictly aligned spot features and index settlement observations."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    cache_dir = Path(cache_dir)
    index, index_sources = load_kind("index", symbol, start, end, cache_dir, offline)
    spot, spot_sources = load_kind("spot", symbol, start, end, cache_dir, offline)
    if not np.array_equal(index.timestamp.to_numpy(), spot.timestamp.to_numpy()):
        raise ValueError("Spot features and index settlement grids differ")
    return spot, index, index_sources + spot_sources


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--start", type=date.fromisoformat, default=date(2023, 11, 6))
    cli.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 9))
    cli.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "research" / ".cache")
    cli.add_argument("--offline", action="store_true")
    args = cli.parse_args()
    if args.start >= args.end or args.end > datetime.now(timezone.utc).date():
        cli.error("Require ordered historical UTC dates")
    for symbol in SYMBOLS:
        spot, index, manifest = load_market(symbol, args.start, args.end, args.cache_dir, args.offline)
        _log(f"READY {symbol}: spot={len(spot):,}, index={len(index):,}, sources={len(manifest)}")
        del spot, index


if __name__ == "__main__":
    main()
