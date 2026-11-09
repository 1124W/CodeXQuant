"""Reproducible close-to-close event-contract research; standard library only."""

import csv
import hashlib
import io
import math
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/spot"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
STEP_MS = 30 * 60 * 1000
PAYOUT = 0.8


def midnight_ms(day):
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)


def fetch_bytes(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if (error.code < 500 and error.code != 429) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def consume_archive(path, out, cache_dir=None, manifest=None):
    url = f"{BASE}/{path}"
    cached = Path(cache_dir) / path if cache_dir else None
    checksum_file = cached.with_suffix(".zip.CHECKSUM") if cached else None
    if cached and cached.exists() and checksum_file.exists():
        archive, checksum = cached.read_bytes(), checksum_file.read_bytes()
    else:
        archive, checksum = fetch_bytes(url), fetch_bytes(url + ".CHECKSUM")
    expected = checksum.decode().split()[0].lower()
    if len(expected) != 64 or hashlib.sha256(archive).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}; remove any corrupt cached archive")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(archive)
        checksum_file.write_bytes(checksum)
    if manifest is not None:
        manifest.append({"path": path, "sha256": expected})
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [name for name in zipped.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path}")
        with zipped.open(names[0]) as source:
            for line in csv.reader(io.TextIOWrapper(source)):
                if not line or line[0] == "open_time":
                    continue
                raw_time = int(line[0])
                timestamp = raw_time // 1000 if raw_time > 10**14 else raw_time
                row = (timestamp, *(float(value) for value in line[1:5]))
                if len(row) != 5:
                    raise ValueError(f"Malformed candle in {path}")
                if timestamp in out and out[timestamp] != row:
                    raise ValueError(f"Conflicting candle at {timestamp} in {path}")
                out[timestamp] = row


def download_rows(symbol, start, end, cache_dir=None, manifest=None):
    rows = {}
    month = start.replace(day=1)
    while month < end:
        following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following <= end:
            name = f"{symbol}-30m-{month:%Y-%m}.zip"
            consume_archive(f"monthly/klines/{symbol}/30m/{name}", rows, cache_dir, manifest)
        else:
            day = max(month, start)
            while day < end:
                name = f"{symbol}-30m-{day:%Y-%m-%d}.zip"
                consume_archive(f"daily/klines/{symbol}/30m/{name}", rows, cache_dir, manifest)
                day += timedelta(days=1)
        month = following
    return [rows[t] for t in sorted(rows) if midnight_ms(start) <= t < midnight_ms(end)]


def validate_candles(symbol, series, start, end):
    expected = (end - start).days * 48
    if len(series) != expected:
        raise ValueError(f"Incomplete {symbol} history: {len(series)} of {expected} expected bars")
    for index, (timestamp, opened, high, low, close) in enumerate(series):
        if timestamp != midnight_ms(start) + index * STEP_MS:
            raise ValueError(f"{symbol}: gap, duplicate or misaligned candle at index {index}")
        if (not all(math.isfinite(v) and v > 0 for v in (opened, high, low, close))
                or not low <= min(opened, close) <= max(opened, close) <= high):
            raise ValueError(f"{symbol}: invalid OHLC at {timestamp}")


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
            result[i] = 100 - 100 / (1 + gain / loss) if loss else (100 if gain else 50)
    return result


@dataclass(frozen=True)
class Trade:
    symbol: str
    signal_ms: int
    settlement_ms: int
    direction: int
    entry: float
    exit: float
    win: bool


def signal_rows(candles, streak, z_limit, rsi_limit, symbol=""):
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
        direction = (1 if down_streak and z <= -z_limit and rsi[i] <= 100 - rsi_limit
                     else -1 if up_streak and z >= z_limit and rsi[i] >= rsi_limit else 0)
        if direction and direction != previous_direction:
            # Close i is observable only at the END of candle i.
            trades.append(Trade(symbol, candles[i][0] + STEP_MS, candles[i + 1][0] + STEP_MS,
                                direction, closes[i], closes[i + 1],
                                (closes[i + 1] - closes[i]) * direction > 0))
        previous_direction = direction
    return trades


def split_trades(trades, cutoff_ms):
    return ([t for t in trades if t.settlement_ms < cutoff_ms],
            [t for t in trades if t.signal_ms >= cutoff_ms])


def stats(trades, days, payout=PAYOUT, fee=0):
    count = len(trades)
    wins = sum(t.win for t in trades)
    losses = count - wins
    balance = peak = drawdown = 0
    settlements = {}
    for trade in trades:
        settlements.setdefault(trade.settlement_ms, []).append((payout if trade.win else -1) - fee)
    # Simultaneous outcomes settle together; symbol order cannot change drawdown.
    for timestamp in sorted(settlements):
        balance += math.fsum(settlements[timestamp])
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    rate = wins / count if count else None
    lower = upper = None
    if count:
        z = 1.96
        denom = 1 + z * z / count
        center = (rate + z * z / (2 * count)) / denom
        radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denom
        lower, upper = max(0, center - radius), min(1, center + radius)
    return {"signals": count, "per_day": round(count / days, 2), "wins": wins, "losses": losses,
            # Keep full precision for validation gates; round only in the UI.
            "win_rate": rate, "win_rate_lower_95": lower, "win_rate_upper_95": upper,
            "net_units": round(balance, 6), "max_drawdown_units": round(drawdown, 6),
            "profit_factor": round((payout - fee) * wins / ((1 + fee) * losses), 3) if losses else None}


def evaluate(candles, start, end, holdout_days=30, payout=PAYOUT, fee=0):
    days = (end - start).days
    if holdout_days < 1 or days <= holdout_days:
        raise ValueError("History must contain positive training and holdout periods")
    if not math.isfinite(payout) or payout <= 0 or not math.isfinite(fee) or not 0 <= fee < payout:
        raise ValueError("Require finite payout > 0 and 0 <= fee < payout")
    if set(candles) != set(SYMBOLS):
        raise ValueError(f"Expected markets: {SYMBOLS}")
    for symbol, series in candles.items():
        validate_candles(symbol, series, start, end)
    cutoff = end - timedelta(days=holdout_days)
    cutoff_ms = midnight_ms(cutoff)
    candidates, ledgers = [], []
    for streak in (2, 3):
        for z_limit in (1.0, 1.5):
            for rsi_limit in (60, 65):
                trades = [t for symbol, series in candles.items()
                          for t in signal_rows(series, streak, z_limit, rsi_limit, symbol)]
                train, _ = split_trades(trades, cutoff_ms)
                candidates.append({"parameters": {"streak": streak, "z_limit": z_limit, "rsi_limit": rsi_limit},
                                   "train": stats(train, days - holdout_days, payout, fee)})
                ledgers.append(trades)
    eligible = [i for i, c in enumerate(candidates) if c["train"]["signals"] >= 100]
    chosen_index = max(eligible, key=lambda i: candidates[i]["train"]["net_units"] /
                       candidates[i]["train"]["signals"]) if eligible else None
    chosen, ledger = None, []
    if chosen_index is not None:
        chosen = dict(candidates[chosen_index])
        trades = ledgers[chosen_index]
        _, test = split_trades(trades, cutoff_ms)
        chosen["test"] = stats(test, holdout_days, payout, fee)
        chosen["test_by_symbol"] = {s: stats([t for t in test if t.symbol == s], holdout_days, payout, fee)
                                    for s in SYMBOLS}
        for trade in sorted(trades, key=lambda t: (t.settlement_ms, t.symbol)):
            split = ("train" if trade.settlement_ms < cutoff_ms else
                     "test" if trade.signal_ms >= cutoff_ms else "purged")
            ledger.append({**asdict(trade), "split": split,
                           "net_units": round((payout if trade.win else -1) - fee, 6)})
    break_even = (1 + fee) / (1 + payout)
    checks = {"selected_on_training_only": chosen is not None,
              "holdout_min_50_signals": bool(chosen and chosen["test"]["signals"] >= 50),
              "holdout_lower_95_above_break_even": bool(chosen and chosen["test"]["win_rate_lower_95"] is not None
                                                       and chosen["test"]["win_rate_lower_95"] > break_even),
              "holdout_positive_net": bool(chosen and chosen["test"]["net_units"] > 0)}
    return {"schema_version": 2, "source": BASE, "symbols": list(SYMBOLS), "interval": "30m",
            "start_utc": str(start), "end_exclusive_utc": str(end), "cutoff_utc": str(cutoff),
            "train_days": days - holdout_days, "holdout_days": holdout_days,
            "payout": payout, "fee_per_trade": fee, "break_even_win_rate": break_even,
            "bars": {s: len(v) for s, v in candles.items()},
            "data_quality": {"complete": True, "continuous": True, "valid_ohlc": True},
            "selection": "highest train net units per signal, at least 100 train signals; fixed grid order breaks ties",
            "settlement_boundary": "signal at candle close; next candle close settles; training settlement strictly before cutoff",
            "assumptions": ["One unit per trade; flat settlement counts as loss",
                            "Entry at observed close; no execution latency or spread model",
                            "Fee is a fixed cost per unit stake on every trade",
                            "Wilson interval assumes independent trials; BTC/ETH and serial dependence can overstate confidence",
                            "Repeated runs on overlapping holdouts are not new independent validation"],
            "validation_checks": checks, "publishable": all(checks.values()), "live_approved": False,
            "chosen": chosen, "all_candidates": candidates, "trade_ledger": ledger}


def run(days, end=None, holdout_days=30, payout=PAYOUT, fee=0, cache_dir=None):
    end = end or datetime.now(timezone.utc).date() - timedelta(days=1)
    if days <= holdout_days or holdout_days < 1:
        raise ValueError("--days must be greater than a positive --holdout-days")
    if end > datetime.now(timezone.utc).date():
        raise ValueError("--end must not be in the future")
    # Validate cost arguments before network access as well.
    if not math.isfinite(payout) or payout <= 0 or not math.isfinite(fee) or not 0 <= fee < payout:
        raise ValueError("Require finite payout > 0 and 0 <= fee < payout")
    start = end - timedelta(days=days)
    manifest = []
    candles = {s: download_rows(s, start, end, cache_dir, manifest) for s in SYMBOLS}
    report = evaluate(candles, start, end, holdout_days, payout, fee)
    report["archives"] = manifest
    return report
