"""Verified Binance USD-M index archives for minute-resolution event research."""

import csv
import hashlib
import io
import json
import math
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

try:
    from .engine import SYMBOLS, fetch_bytes, midnight_ms
except ImportError:
    from engine import SYMBOLS, fetch_bytes, midnight_ms

INDEX_BASE = "https://data.binance.vision/data/futures/um"
MINUTE_MS = 60000


def archive_paths(symbol, start, end):
    if symbol not in SYMBOLS or start >= end:
        raise ValueError("Require BTCUSDT/ETHUSDT and a positive date range")
    month = start.replace(day=1)
    while month < end:
        following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following <= end:
            yield f"monthly/indexPriceKlines/{symbol}/1m/{symbol}-1m-{month:%Y-%m}.zip"
        else:
            day = max(start, month)
            while day < end:
                yield f"daily/indexPriceKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                day += timedelta(days=1)
        month = following


def read_archive(path, cache_dir=None, offline=False):
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path or ":" in path or not path.endswith(".zip"):
        raise ValueError("Invalid archive path")
    cached = Path(cache_dir) / "futures" / "um" / Path(path) if cache_dir else None
    checksum_path = cached.with_suffix(".zip.CHECKSUM") if cached else None
    if cached and cached.is_file() and checksum_path.is_file():
        archive, checksum = cached.read_bytes(), checksum_path.read_bytes()
    else:
        if offline:
            raise ValueError(f"Verified index cache missing: {path}; run once without --offline")
        archive = fetch_bytes(f"{INDEX_BASE}/{path}")
        checksum = fetch_bytes(f"{INDEX_BASE}/{path}.CHECKSUM")
    tokens = checksum.decode("ascii").split()
    expected = tokens[0].lower() if tokens else ""
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid checksum: {path}")
    if hashlib.sha256(archive).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}; corrupt cache will not be used")
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [name for name in zipped.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one CSV: {path}")
        with zipped.open(names[0]) as source:
            for line in csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig")):
                if not line or line[0] in ("open_time", "openTime"):
                    continue
                if len(line) < 5:
                    raise ValueError(f"Malformed index candle: {path}")
                raw = int(line[0])
                timestamp = raw // 1000 if raw > 10**14 else raw
                if raw > 10**14 and raw % 1000:
                    raise ValueError(f"Non-millisecond timestamp: {path}")
                rows.append((timestamp, *(float(value) for value in line[1:5])))
    # Cache only archives which can be parsed. Each worker owns a distinct path.
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        for target, payload in ((cached, archive), (checksum_path, checksum)):
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
    return rows, {"path": path, "url": f"{INDEX_BASE}/{path}", "sha256": expected}


def validate_minutes(symbol, series, start, end):
    expected = (end - start).days * 1440
    if len(series) != expected:
        raise ValueError(f"Incomplete {symbol} index history: {len(series)} of {expected} minute bars")
    for i, row in enumerate(series):
        if len(row) != 5:
            raise ValueError(f"Malformed {symbol} minute candle")
        timestamp, opened, high, low, close = row
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp != midnight_ms(start) + i * MINUTE_MS:
            raise ValueError(f"{symbol}: missing, duplicate or off-grid minute at {i}")
        if (not all(math.isfinite(v) and v > 0 for v in (opened, high, low, close))
                or not low <= min(opened, close) <= max(opened, close) <= high):
            raise ValueError(f"{symbol}: invalid index OHLC at {timestamp}")


def read_rest_day(symbol, day, cache_dir=None, offline=False):
    """Repair an absent archive day with an explicit, persisted official REST snapshot.

    Its SHA-256 is a local reproducibility fingerprint, not an exchange checksum.
    """
    start = midnight_ms(day)
    path = f"rest/indexPriceKlines/{symbol}/1m/{symbol}-1m-{day}.json"
    url = (f"https://fapi.binance.com/fapi/v1/indexPriceKlines?pair={symbol}&interval=1m"
           f"&startTime={start}&endTime={start + 86400000 - 1}&limit=1500")
    cached = Path(cache_dir) / "futures" / "um" / path if cache_dir else None
    fingerprint_path = cached.with_suffix(".json.sha256") if cached else None
    if cached and cached.is_file() and fingerprint_path.is_file():
        payload = cached.read_bytes()
        if hashlib.sha256(payload).hexdigest() != fingerprint_path.read_text(encoding="ascii").strip():
            raise ValueError(f"REST snapshot checksum mismatch: {path}")
    else:
        if offline:
            raise ValueError(f"Verified REST repair snapshot missing: {path}")
        payload = fetch_bytes(url)
    raw = json.loads(payload)
    if not isinstance(raw, list) or any(not isinstance(row, list) or len(row) < 5 for row in raw):
        raise ValueError(f"Malformed official REST index snapshot: {path}")
    rows = [(row[0], *(float(v) for v in row[1:5])) for row in raw]
    validate_minutes(symbol, rows, day, day + timedelta(days=1))
    fingerprint = hashlib.sha256(payload).hexdigest()
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        for target, content in ((cached, payload), (fingerprint_path, (fingerprint + "\n").encode("ascii"))):
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
    return rows, {"path": path, "url": url, "sha256": fingerprint,
                  "verification": "local_sha256_of_official_rest_snapshot",
                  "purpose": "repair_missing_archive_minutes"}


def download_index_minutes(start, end, cache_dir=None, offline=False, workers=4):
    tasks = [(symbol, path) for symbol in SYMBOLS for path in archive_paths(symbol, start, end)]
    grouped, manifest = {symbol: {} for symbol in SYMBOLS}, []
    lower, upper = midnight_ms(start), midnight_ms(end)

    def consume(task):
        symbol, path = task
        try:
            rows, entry = read_archive(path, cache_dir, offline)
            return symbol, rows, [entry]
        except (urllib.error.HTTPError, ValueError) as error:
            # A completed month may not yet have its monthly bundle. Daily
            # index bundles are the same source, not a fallback to spot prices.
            missing_month = (isinstance(error, urllib.error.HTTPError) and error.code == 404
                             or offline and isinstance(error, ValueError) and str(error).startswith("Verified index cache missing:"))
            if not missing_month or not path.startswith("monthly/"):
                raise
            month_text = PurePosixPath(path).stem[-7:]
            month = date.fromisoformat(month_text + "-01")
            following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
            rows, entries, day = [], [], max(start, month)
            while day < min(end, following):
                daily = f"daily/indexPriceKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                subset, entry = read_archive(daily, cache_dir, offline)
                rows.extend(subset)
                entries.append(entry)
                day += timedelta(days=1)
            return symbol, rows, entries

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for symbol, rows, entries in executor.map(consume, tasks):
            manifest.extend(entries)
            for row in rows:
                timestamp = row[0]
                if lower <= timestamp < upper:
                    if timestamp in grouped[symbol]:
                        raise ValueError(f"Duplicate index candle: {symbol} at {timestamp}")
                    grouped[symbol][timestamp] = row
    missing_days = []
    for symbol, rows in grouped.items():
        days = {date.fromordinal(start.toordinal() + (timestamp - lower) // 86400000)
                for timestamp in range(lower, upper, MINUTE_MS) if timestamp not in rows}
        missing_days.extend((symbol, day) for day in sorted(days))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        repairs = executor.map(lambda task: (task[0], *read_rest_day(task[0], task[1], cache_dir, offline)), missing_days)
        for symbol, rows, entry in repairs:
            manifest.append(entry)
            for row in rows:
                if row[0] in grouped[symbol] and grouped[symbol][row[0]] != row:
                    raise ValueError(f"REST/archive index conflict: {symbol} at {row[0]}")
                grouped[symbol][row[0]] = row
    candles = {symbol: [rows[t] for t in sorted(rows)] for symbol, rows in grouped.items()}
    for symbol, series in candles.items():
        validate_minutes(symbol, series, start, end)
    return candles, sorted(manifest, key=lambda entry: entry["path"])


def data_fingerprint(candles):
    digest = hashlib.sha256()
    for symbol in sorted(candles):
        digest.update((symbol + "\n").encode())
        for row in candles[symbol]:
            digest.update((",".join(str(v) for v in row) + "\n").encode())
    return digest.hexdigest()
