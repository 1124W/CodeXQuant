"""Evaluate fixed contrarian 30m event-contract candidates on Binance public data.

Run: python3 research/run.py --output research/results.json
The last 30 days are held out and are not used to select parameters.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/spot"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
STEP_MS = 30 * 60 * 1000
PAYOUT = 0.8
BREAK_EVEN = 1 / (1 + PAYOUT)


def download_rows(symbol, start, end):
    rows = {}
    first_month = start.replace(day=1)
    month = first_month
    while month < end:
        following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following <= end:
            name = f"{symbol}-30m-{month:%Y-%m}.zip"
            path = f"monthly/klines/{symbol}/30m/{name}"
            consume_archive(path, rows)
        else:
            day = max(month, start)
            while day < end:
                name = f"{symbol}-30m-{day:%Y-%m-%d}.zip"
                consume_archive(f"daily/klines/{symbol}/30m/{name}", rows)
                day += timedelta(days=1)
        month = following
    start_ms = int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.combine(end, datetime.min.time(), timezone.utc).timestamp() * 1000)
    return [rows[t] for t in sorted(rows) if start_ms <= t < end_ms]


def consume_archive(path, out):
    url = f"{BASE}/{path}"
    with urllib.request.urlopen(url, timeout=40) as response:
        archive = response.read()
    with urllib.request.urlopen(url + ".CHECKSUM", timeout=20) as response:
        expected = response.read().decode().split()[0]
    if hashlib.sha256(archive).hexdigest().lower() != expected.lower():
        raise ValueError(f"Checksum mismatch: {path}")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for line in csv.reader(io.TextIOWrapper(zipped.open(zipped.namelist()[0]))):
            if not line or not line[0].isdigit():
                continue
            raw_time = int(line[0])
            time = raw_time // 1000 if raw_time > 10**14 else raw_time
            out[time] = (time, float(line[1]), float(line[2]), float(line[3]), float(line[4]))


def rsi_series(closes, period=14):
    result = [None] * len(closes)
    gain = loss = 0
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        up, down = max(change, 0), max(-change, 0)
        if i <= period:
            gain += up
            loss += down
            if i == period:
                gain /= period
                loss /= period
        else:
            gain = (gain * (period - 1) + up) / period
            loss = (loss * (period - 1) + down) / period
        if i >= period:
            result[i] = 100 - 100 / (1 + gain / loss) if loss else 100
    return result


def signal_rows(candles, streak, z_limit, rsi_limit):
    closes = [c[4] for c in candles]
    rsi = rsi_series(closes)
    trades = []
    previous_direction = 0
    for i in range(30, len(candles) - 1):
        window = closes[i - 19:i + 1]
        mean = sum(window) / 20
        std = math.sqrt(sum((v - mean) ** 2 for v in window) / 20)
        z = (closes[i] - mean) / std if std else 0
        down_streak = all(closes[j] < closes[j - 1] for j in range(i - streak + 1, i + 1))
        up_streak = all(closes[j] > closes[j - 1] for j in range(i - streak + 1, i + 1))
        direction = 1 if down_streak and z <= -z_limit and rsi[i] <= 100 - rsi_limit else -1 if up_streak and z >= z_limit and rsi[i] >= rsi_limit else 0
        if direction and direction != previous_direction:
            win = (closes[i + 1] - closes[i]) * direction > 0
            trades.append((candles[i][0], win))
        previous_direction = direction
    return trades


def stats(trades, days):
    count = len(trades)
    wins = sum(win for _, win in trades)
    losses = count - wins
    balance = peak = drawdown = 0
    for _, win in sorted(trades):
        balance += PAYOUT if win else -1
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    rate = wins / count if count else 0
    if count:
        z = 1.96
        denom = 1 + z * z / count
        center = (rate + z * z / (2 * count)) / denom
        radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denom
        lower = center - radius
    else:
        lower = 0
    return {"signals": count, "per_day": round(count / days, 2), "wins": wins,
            "win_rate": round(rate, 4), "win_rate_lower_95": round(lower, 4),
            "net_units": round(balance, 2), "max_drawdown_units": round(drawdown, 2),
            "profit_factor": round(PAYOUT * wins / losses, 3) if losses else None}


def run(days):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    cutoff = end - timedelta(days=30)
    cutoff_ms = int(datetime.combine(cutoff, datetime.min.time(), timezone.utc).timestamp() * 1000)
    candles = {symbol: download_rows(symbol, start, end) for symbol in SYMBOLS}
    expected = days * 48
    for symbol, series in candles.items():
        if len(series) < expected * .98:
            raise ValueError(f"Incomplete {symbol} history: {len(series)} of {expected} expected bars")
    candidates = []
    for streak in (2, 3):
        for z_limit in (1.0, 1.5):
            for rsi_limit in (60, 65):
                trades = []
                for series in candles.values():
                    trades += signal_rows(series, streak, z_limit, rsi_limit)
                train = [t for t in trades if t[0] + STEP_MS < cutoff_ms]
                test = [t for t in trades if t[0] >= cutoff_ms]
                candidates.append({"parameters": {"streak": streak, "z_limit": z_limit, "rsi_limit": rsi_limit},
                                   "train": stats(train, days - 30), "test": stats(test, 30)})
    eligible = [c for c in candidates if c["train"]["signals"] >= 100]
    chosen = max(eligible, key=lambda c: c["train"]["net_units"] / c["train"]["signals"]) if eligible else None
    passed = bool(chosen and chosen["test"]["signals"] >= 50 and chosen["test"]["win_rate_lower_95"] > BREAK_EVEN and chosen["test"]["net_units"] > 0)
    return {"source": BASE, "symbols": SYMBOLS, "interval": "30m", "start_utc": str(start), "end_exclusive_utc": str(end),
            "cutoff_utc": str(cutoff), "payout": PAYOUT, "break_even_win_rate": round(BREAK_EVEN, 4),
            "bars": {s: len(v) for s, v in candles.items()}, "selection": "highest train net units per signal, at least 100 train signals",
            "settlement_boundary": "training outcomes close before the holdout begins",
            "publishable": passed, "chosen": chosen, "all_candidates": candidates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("research/results.json"))
    args = parser.parse_args()
    report = run(args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("start_utc", "end_exclusive_utc", "bars", "break_even_win_rate", "publishable", "chosen")}, ensure_ascii=False))
