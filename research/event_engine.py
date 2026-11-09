"""Chronological index-price event research. All execution prices are minute proxies.

The fixed rule grid is declared here before observing any evaluation window. The
engine never adjusts its rules to reach a requested win rate.
"""

import heapq
import math
import random
from collections import defaultdict
from datetime import datetime, timezone

MINUTE_MS = 60_000
DAY_MS = 86_400_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")
KNOWN_UNAVAILABLE_DATES = ("2026-01-17", "2026-01-18", "2026-01-19", "2026-01-24", "2026-01-25")
_UNAVAILABLE_DAY_NUMBERS = frozenset(int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) // 86_400
                                    for day in KNOWN_UNAVAILABLE_DATES)

# A small, predeclared research universe; this is not a test-result leaderboard.
STRATEGY_GRID = [
    {"id": f"revert_z{z}_rsi{rsi}_regime{regime}", "family": "mean_reversion",
     "z_limit": z, "rsi_limit": rsi, "max_trend_atr": regime}
    for z in (1.5, 2.0) for rsi in (60, 65) for regime in (.6, 1.0)
] + [
    {"id": f"pullback_{depth}_trend{strength}", "family": "trend_pullback",
     "pullback_atr": depth, "min_trend_atr": strength}
    for depth in (.2, .5) for strength in (.25, .5)
] + [
    {"id": f"breakout_{lookback}_body{body}", "family": "trend_breakout",
     "lookback": lookback, "min_body_atr": body}
    for lookback in (12, 24) for body in (.3, .6)
]


def utc_ms(day):
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)


def utc_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat()


def aggregate_bars(series, minutes=30):
    """Aggregate only complete clock-aligned bars, with an exclusive close time."""
    result = []
    size = minutes * MINUTE_MS
    for offset in range(0, len(series), minutes):
        chunk = series[offset:offset + minutes]
        if len(chunk) != minutes or chunk[0][0] % size:
            continue
        if any(row[0] != chunk[0][0] + i * MINUTE_MS for i, row in enumerate(chunk)):
            raise ValueError("Incomplete minute grid during aggregation")
        result.append((chunk[0][0], chunk[0][1], max(r[2] for r in chunk),
                       min(r[3] for r in chunk), chunk[-1][4]))
    return result


def _ema(values, period):
    current, factor, result = values[0], 2 / (period + 1), []
    for value in values:
        current = current * (1 - factor) + value * factor
        result.append(current)
    return result


def _rsi(values, period=14):
    result = [50.] * len(values)
    gain = loss = 0.
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        up, down = max(0, change), max(0, -change)
        if i <= period:
            gain += up / period
            loss += down / period
        else:
            gain = (gain * (period - 1) + up) / period
            loss = (loss * (period - 1) + down) / period
        if i >= period:
            result[i] = 100 - 100 / (1 + gain / loss) if loss else (100 if gain else 50)
    return result


def _features(bars):
    closes = [r[4] for r in bars]
    fast, slow, long = (_ema(closes, p) for p in (9, 21, 50))
    rsi = _rsi(closes)
    true_ranges, features = [], []
    for i, (_, opened, high, low, close) in enumerate(bars):
        previous = closes[max(0, i - 1)]
        true_ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
        atr = sum(true_ranges[max(0, i - 13):i + 1]) / min(i + 1, 14)
        window = closes[max(0, i - 19):i + 1]
        mean = sum(window) / len(window)
        deviation = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
        features.append({"fast": fast[i], "slow": slow[i], "long": long[i], "rsi": rsi[i],
                         "atr": atr, "z": (close - mean) / deviation if deviation else 0,
                         "trend": (fast[i] - slow[i]) / atr if atr else 0,
                         "body": (close - opened) / atr if atr else 0})
    return features


def _direction(rule, bars, features, i):
    f, close = features[i], bars[i][4]
    if not f["atr"]:
        return 0
    if rule["family"] == "mean_reversion":
        if abs(f["trend"]) > rule["max_trend_atr"]:
            return 0
        if f["z"] <= -rule["z_limit"] and f["rsi"] <= 100 - rule["rsi_limit"]:
            return 1
        if f["z"] >= rule["z_limit"] and f["rsi"] >= rule["rsi_limit"]:
            return -1
    elif rule["family"] == "trend_pullback":
        depth = rule["pullback_atr"] * f["atr"]
        if (f["trend"] >= rule["min_trend_atr"] and f["slow"] > f["long"]
                and close <= f["fast"] - depth and 35 <= f["rsi"] <= 55):
            return 1
        if (f["trend"] <= -rule["min_trend_atr"] and f["slow"] < f["long"]
                and close >= f["fast"] + depth and 45 <= f["rsi"] <= 65):
            return -1
    else:
        previous = bars[i - rule["lookback"]:i]
        if close > max(r[2] for r in previous) and f["body"] >= rule["min_body_atr"] and f["trend"] > 0:
            return 1
        if close < min(r[3] for r in previous) and f["body"] <= -rule["min_body_atr"] and f["trend"] < 0:
            return -1
    return 0


def build_candidates(candles, *, latency_minutes=1, horizon_minutes=30):
    """Future rows determine outcomes only, never a signal or its ranking."""
    candidates = {rule["id"]: [] for rule in STRATEGY_GRID}
    for symbol in SYMBOLS:
        series = candles[symbol]
        bars = aggregate_bars(series)
        if not bars:
            continue
        features = _features(bars)
        origin = series[0][0]
        for rule in STRATEGY_GRID:
            previous_direction = 0
            for i in range(60, len(bars)):
                direction = _direction(rule, bars, features, i)
                new_signal = direction and direction != previous_direction
                previous_direction = direction
                if not new_signal:
                    continue
                signal_ms = bars[i][0] + 30 * MINUTE_MS
                entry_ms = signal_ms + latency_minutes * MINUTE_MS
                settlement_ms = entry_ms + horizon_minutes * MINUTE_MS
                # The official event FAQ lists these UTC dates as unavailable.
                # Exclude any contract touching them, including a delayed entry
                # or an expiry exactly at an unavailable day's start.
                if any(day in _UNAVAILABLE_DAY_NUMBERS
                       for day in range(signal_ms // DAY_MS, settlement_ms // DAY_MS + 1)):
                    continue
                entry_index = (entry_ms - origin) // MINUTE_MS
                exit_index = (settlement_ms - origin) // MINUTE_MS
                if not 0 <= entry_index < exit_index < len(series):
                    continue
                candidates[rule["id"]].append({"symbol": symbol, "signal_ms": signal_ms,
                    "entry_ms": entry_ms, "settlement_ms": settlement_ms, "direction": direction,
                    "entry": series[entry_index][1], "exit": series[exit_index][1], "rule_id": rule["id"]})
    return candidates


def simulate_portfolio(trades, *, payout=.8, fee=.01, initial_bankroll=1000, stake=1,
                       max_open_positions=1, daily_loss_limit=5, adverse_bps=0):
    """Fixed stakes; reserve premium+cost. Losing premiums plus open risk cap a day.

    All positions due at an entry timestamp settle first. Equal-time signals use
    symbol name order, established without reading outcomes. Loss budget is our
    conservative research constraint, not an assertion of Binance account limits.
    """
    accepted, pending = [], []
    cash, daily_losses = float(initial_bankroll), defaultdict(float)
    sequence = 0

    def settle_until(timestamp):
        nonlocal cash
        while pending and pending[0][0] <= timestamp:
            _, _, trade = heapq.heappop(pending)
            outcome = trade["outcome"]
            cash += stake * (1 + payout) if outcome == "win" else stake if outcome == "refund" else 0
            if outcome == "loss":
                daily_losses[trade["settlement_ms"] // DAY_MS] += stake

    for raw in sorted(trades, key=lambda t: (t["entry_ms"], t["symbol"], t.get("rule_id", ""))):
        settle_until(raw["entry_ms"])
        day = raw["entry_ms"] // DAY_MS
        # Across-midnight positions are also reserved against the new day's cap.
        exposure = len(pending) * stake
        if (len(pending) >= max_open_positions or cash + 1e-12 < stake * (1 + fee)
                or daily_losses[day] + exposure + stake > daily_loss_limit + 1e-12):
            continue
        trade = dict(raw)
        trade["entry"] = raw["entry"] * (1 + raw["direction"] * adverse_bps / 10_000)
        change = (trade["exit"] - trade["entry"]) * trade["direction"]
        outcome = "win" if change > 0 else "loss" if change < 0 else "refund"
        trade["outcome"] = outcome
        trade["net_units"] = round(stake * ((payout if outcome == "win" else -1 if outcome == "loss" else 0) - fee), 8)
        trade["stake"] = stake
        cash -= stake * (1 + fee)
        accepted.append(trade)
        heapq.heappush(pending, (trade["settlement_ms"], sequence, trade))
        sequence += 1
    settle_until(math.inf)
    return sorted(accepted, key=lambda t: (t["settlement_ms"], t["symbol"]))


def _wilson(wins, count):
    if not count:
        return None, None
    rate, z = wins / count, 1.96
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return max(0, center - radius), min(1, center + radius)


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def _bootstrap(trades, start_ms, end_ms, iterations=1000):
    days = max(1, math.ceil((end_ms - start_ms) / DAY_MS))
    buckets = [[0, 0, 0.] for _ in range(days)]
    for trade in trades:
        index = (trade["settlement_ms"] - start_ms) // DAY_MS
        if not 0 <= index < days:
            raise ValueError("Trade settlement outside statistics window")
        buckets[index][0] += trade["outcome"] == "win"
        buckets[index][1] += trade["outcome"] != "refund"
        buckets[index][2] += trade["net_units"]
    if not any(b[1] for b in buckets):
        return None, None, None, None
    generator, rates, nets = random.Random(7219), [], []
    for _ in range(iterations):
        sampled = [buckets[generator.randrange(days)] for _ in range(days)]
        count = sum(b[1] for b in sampled)
        # Empty sampled blocks are evidence of sparse activity, not discarded.
        rates.append(sum(b[0] for b in sampled) / count if count else 0.)
        nets.append(math.fsum(b[2] for b in sampled))
    return _quantile(rates, .025), _quantile(rates, .975), _quantile(nets, .025), _quantile(nets, .975)


def event_stats(trades, start_ms, end_ms, *, payout=.8, fee=.01, bootstrap=True):
    wins = sum(t["outcome"] == "win" for t in trades)
    losses = sum(t["outcome"] == "loss" for t in trades)
    decisive, count = wins + losses, len(trades)
    lower, upper = _wilson(wins, decisive)
    balance = peak = drawdown = 0.
    settlements = defaultdict(list)
    for trade in trades:
        settlements[trade["settlement_ms"]].append(trade["net_units"])
    for timestamp in sorted(settlements):
        balance += math.fsum(settlements[timestamp])
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    gross_profit = math.fsum(max(0, t["net_units"]) for t in trades)
    gross_loss = -math.fsum(min(0, t["net_units"]) for t in trades)
    block = _bootstrap(trades, start_ms, end_ms) if bootstrap else (None,) * 4
    return {"signals": count, "wins": wins, "losses": losses, "refunds": count - decisive,
            "decisive_trades": decisive, "win_rate": wins / decisive if decisive else None,
            "per_day": count / max((end_ms - start_ms) / DAY_MS, 1 / 1440),
            "win_rate_lower_95": lower, "win_rate_upper_95": upper,
            "block_bootstrap_lower_95": block[0], "block_bootstrap_upper_95": block[1],
            "block_bootstrap_net_lower_95": block[2], "block_bootstrap_net_upper_95": block[3],
            "effective_break_even_win_rate": (1 + fee * count / decisive) / (1 + payout) if decisive else None,
            "net_units": round(balance, 8), "max_drawdown_units": round(drawdown, 8),
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "expectancy_per_trade": balance / count if count else None}


def _window(trades, start_ms, end_ms):
    # Signal must be observable inside this window; both entry and expiry fit.
    return [t for t in trades if t["signal_ms"] >= start_ms and t["settlement_ms"] < end_ms]


def _portfolio_config(config):
    return {key: config[key] for key in ("payout", "fee", "initial_bankroll", "stake",
            "max_open_positions", "daily_loss_limit", "adverse_bps")}


def select_candidate(candidate_trades, start_ms, end_ms, config):
    best = None
    for rule in STRATEGY_GRID:
        trades = simulate_portfolio(_window(candidate_trades[rule["id"]], start_ms, end_ms),
                                    **_portfolio_config(config))
        sample = event_stats(trades, start_ms, end_ms, payout=config["payout"], fee=config["fee"], bootstrap=False)
        if (sample["decisive_trades"] < config["min_train_trades"] or sample["net_units"] <= 0
                or sample["win_rate"] < config["target_win_rate"]):
            continue
        # Confidence-adjusted expectancy; ties preserve the predeclared grid order.
        score = sample["win_rate_lower_95"] * (1 + config["payout"]) - 1 - config["fee"]
        if best is None or score > best[0]:
            best = (score, rule["id"], sample)
    return (best[1], best[2]) if best else None


def _checks(sample, config):
    break_even = sample["effective_break_even_win_rate"]
    return {"min_samples": sample["decisive_trades"] >= config["min_test_trades"],
            "target_met": sample["win_rate"] is not None and sample["win_rate"] >= config["target_win_rate"],
            "positive_net": sample["net_units"] > 0,
            "confidence_above_break_even": break_even is not None and sample["win_rate_lower_95"] > break_even,
            "block_confidence_above_break_even": break_even is not None and sample["block_bootstrap_lower_95"] > break_even,
            "block_net_positive": sample["block_bootstrap_net_lower_95"] is not None and sample["block_bootstrap_net_lower_95"] > 0}


def _by_symbol(trades, start_ms, end_ms, config):
    return {symbol: event_stats([t for t in trades if t["symbol"] == symbol], start_ms, end_ms,
                                payout=config["payout"], fee=config["fee"])
            for symbol in SYMBOLS}


def _equity(trades, initial_bankroll, split):
    groups, curve = defaultdict(list), []
    balance = peak = initial_bankroll
    for trade in trades:
        groups[trade["settlement_ms"]].append(trade["net_units"])
    for timestamp in sorted(groups):
        balance += math.fsum(groups[timestamp])
        peak = max(peak, balance)
        curve.append({"timestamp_ms": timestamp, "balance_units": round(balance, 8),
                      "net_units": round(balance - initial_bankroll, 8),
                      "drawdown_units": round(peak - balance, 8), "split": split})
    return curve


def _monthly(trades, start_ms, end_ms, config):
    groups = defaultdict(list)
    for trade in trades:
        groups[utc_date(trade["settlement_ms"])[:7]].append(trade)
    cursor = datetime.fromtimestamp(start_ms / 1000, timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = []
    while int(cursor.timestamp() * 1000) < end_ms:
        following = cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
        month = cursor.strftime("%Y-%m")
        result.append({"month": month, **event_stats(groups[month], max(start_ms, int(cursor.timestamp() * 1000)),
            min(end_ms, int(following.timestamp() * 1000)), payout=config["payout"], fee=config["fee"], bootstrap=False)})
        cursor = following
    return result


def _validate(candles, start, end, config):
    if set(candles) != set(SYMBOLS):
        raise ValueError(f"Expected markets {SYMBOLS}")
    for key in ("train_days", "validation_days", "holdout_days", "latency_minutes", "horizon_minutes",
                "min_train_trades", "min_test_trades", "max_open_positions"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (end - start).days <= config["train_days"] + config["holdout_days"]:
        raise ValueError("Require at least one development test window after training and before final holdout")
    for key in ("payout", "fee", "target_win_rate", "adverse_bps", "initial_bankroll", "stake", "daily_loss_limit"):
        if not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if (config["payout"] <= 0 or not 0 <= config["fee"] < config["payout"]
            or not 0 < config["target_win_rate"] < 1 or not 0 <= config["adverse_bps"] < 9_998
            or config["stake"] <= 0 or config["initial_bankroll"] < config["stake"] * (1 + config["fee"])
            or config["daily_loss_limit"] < config["stake"]):
        raise ValueError("Invalid payout, fee, target or portfolio constraints")
    first, expected = utc_ms(start), (end - start).days * 1440
    for symbol, series in candles.items():
        if len(series) != expected:
            raise ValueError(f"Incomplete {symbol}: expected {expected} minute bars, got {len(series)}")
        for i, row in enumerate(series):
            if len(row) != 5:
                raise ValueError(f"Invalid {symbol} minute row")
            timestamp, opened, high, low, close = row
            if (not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp != first + i * MINUTE_MS
                    or not all(math.isfinite(v) and v > 0 for v in row[1:])
                    or not low <= min(opened, close) <= max(opened, close) <= high):
                raise ValueError(f"Invalid {symbol} timestamp/grid/OHLC at row {i}")


def evaluate_events(candles, start, end, *, train_days=60, validation_days=15, holdout_days=30,
                    payout=.8, fee=.01, target_win_rate=.58, latency_minutes=1, horizon_minutes=30,
                    adverse_bps=0, min_train_trades=100, min_test_trades=100, initial_bankroll=1000,
                    stake=1, max_open_positions=1, daily_loss_limit=5):
    """Rolling train selection, disjoint OOS folds, then a chronological final holdout.

    The final holdout is withheld from every development window and every selector.
    It is 'unseen by this run', not guaranteed never observed by prior researchers.
    """
    config = {key: value for key, value in locals().copy().items() if key not in ("candles", "start", "end")}
    _validate(candles, start, end, config)
    candidates = build_candidates(candles, latency_minutes=latency_minutes, horizon_minutes=horizon_minutes)
    first, last = utc_ms(start), utc_ms(end)
    holdout_start = last - holdout_days * DAY_MS
    cursor, folds, schedule = first + train_days * DAY_MS, [], []
    rules = {rule["id"]: rule for rule in STRATEGY_GRID}
    while cursor < holdout_start:
        test_end = min(cursor + validation_days * DAY_MS, holdout_start)
        training_start = cursor - train_days * DAY_MS
        chosen = select_candidate(candidates, training_start, cursor, config)
        fold_id = f"fold_{len(folds) + 1:02d}"
        if chosen:
            entries = _window(candidates[chosen[0]], cursor, test_end)
            schedule.extend({**t, "split": "walk_forward", "fold_id": fold_id} for t in entries)
        folds.append({"id": fold_id, "train_start": utc_date(training_start), "train_end": utc_date(cursor),
                      "test_start": utc_date(cursor), "test_end": utc_date(test_end),
                      "chosen": {"parameters": rules[chosen[0]], "train": chosen[1]} if chosen else None})
        cursor = test_end
    oos_start = first + train_days * DAY_MS
    oos = simulate_portfolio(schedule, **_portfolio_config(config))
    for fold in folds:
        entries = [t for t in oos if t["fold_id"] == fold["id"]]
        fold["test"] = event_stats(entries, utc_ms(datetime.fromisoformat(fold["test_start"]).date()),
            utc_ms(datetime.fromisoformat(fold["test_end"]).date()), payout=payout, fee=fee)
    final_train_start = holdout_start - train_days * DAY_MS
    chosen = select_candidate(candidates, final_train_start, holdout_start, config)
    final_schedule = ([{**t, "split": "final_holdout", "fold_id": "final_holdout"}
                       for t in _window(candidates[chosen[0]], holdout_start, last)] if chosen else [])
    final = simulate_portfolio(final_schedule, **_portfolio_config(config))
    aggregate = event_stats(oos, oos_start, holdout_start, payout=payout, fee=fee)
    final_stats = event_stats(final, holdout_start, last, payout=payout, fee=fee)
    active_folds = sum(f["test"]["signals"] > 0 for f in folds)
    positive_folds = sum(f["test"]["net_units"] > 0 for f in folds)
    checks = {f"walk_forward_{key}": value for key, value in _checks(aggregate, config).items()}
    checks.update({f"holdout_{key}": value for key, value in _checks(final_stats, config).items()})
    checks["walk_forward_stability"] = active_folds >= 3 and positive_folds / active_folds >= 2 / 3
    checks["selected_on_training_only"] = True
    stress = []
    for name, stress_payout, stress_fee, bps in [
            ("base", payout, fee, adverse_bps),
            ("lower_payout_075", min(payout, .75), fee, adverse_bps),
            ("lower_payout_070", min(payout, .70), fee, adverse_bps),
            ("higher_cost", payout, max(fee, .02), adverse_bps),
            ("adverse_strike_2bps", payout, fee, adverse_bps + 2)]:
        stress_config = {**_portfolio_config(config), "payout": stress_payout, "fee": stress_fee, "adverse_bps": bps}
        stress_oos = simulate_portfolio(schedule, **stress_config)
        stress_final = simulate_portfolio(final_schedule, **stress_config)
        stress.append({"name": name, "payout": stress_payout, "fee_per_trade": stress_fee, "adverse_bps": bps,
            "walk_forward": event_stats(stress_oos, oos_start, holdout_start, payout=stress_payout, fee=stress_fee, bootstrap=False),
            "final_holdout": event_stats(stress_final, holdout_start, last, payout=stress_payout, fee=stress_fee, bootstrap=False)})
    # Historical gates, not authority to trade or evidence of second-exact execution.
    return {"schema_version": 3, "engine": "event-walk-forward-v1", "symbols": list(SYMBOLS),
        "start_utc": str(start), "end_exclusive_utc": str(end), "cutoff_utc": utc_date(holdout_start),
        "target_win_rate": target_win_rate, "payout": payout, "fee_per_trade": fee,
        "break_even_win_rate": (1 + fee) / (1 + payout), "config": config,
        "bars": {symbol: len(series) for symbol, series in candles.items()},
        "known_unavailable_dates": list(KNOWN_UNAVAILABLE_DATES),
        "strategy_grid": STRATEGY_GRID, "candidate_count": len(STRATEGY_GRID),
        "selection": "Only training data; at least min_train_trades decisive outcomes, positive net, training win rate >= target; maximize Wilson-lower-bound expectancy; fixed grid tie order; otherwise no trade",
        "data_model": "Binance USD-M index 1m OHLC; fully closed 30m signals; entry and expiry use exact minute OPEN samples. Minute approximation cannot reproduce next-second event strike or settlement.",
        "assumptions": [
            "Fixed payout is an assumption; actual event payout varies by order and must be recorded separately.",
            "Flat entry/expiry price refunds principal; configured research cost still applies. Win rate excludes refunds.",
            "Fee is a conservative per-stake research friction, not a claim Binance charges an extra event fee.",
            "Positive latency is whole minutes; this is not the actual next-second execution rule.",
            "Only complete 30m bars are observable to the signal; entry/expiry minute opens are used only for execution.",
            "Each evaluation window purges positions settling at or beyond its end; no cross-window outcomes enter selection.",
            "Official FAQ lists 2026-01-17/18/19/24/25 as UTC unavailable dates; exclude every contract touching them from signal through expiry. This is a known-date calendar, not proof of uninterrupted availability on all other dates.",
            "Fixed stake, reserved premium plus cost; no martingale. Same-time signals use BTC before ETH.",
            "Daily loss budget counts losing premiums plus open premium risk against UTC days; research units, not an exchange minimum or actual account limit.",
            "Walk-forward portfolio carries capital continuously; final holdout is evaluated as a separate portfolio with initial bankroll.",
            "Day bootstrap resamples joint BTC/ETH calendar-day outcomes, including zero-trade days. Cross-day dependence and strategy-search multiplicity remain.",
            "All stress scenarios freeze the chosen rule schedule; losses can alter later risk-limit admission. They never reselect on stress or test results.",
            "Previously viewed or repeatedly reused dates are not fresh independent evidence. Passing historical gates still requires new forward observations."],
        "walk_forward": {"start_utc": utc_date(oos_start), "end_exclusive_utc": utc_date(holdout_start),
                         "folds": folds, "aggregate": aggregate, "by_symbol": _by_symbol(oos, oos_start, holdout_start, config),
                         "monthly": _monthly(oos, oos_start, holdout_start, config), "active_folds": active_folds, "positive_folds": positive_folds},
        "final_holdout": {"start_utc": utc_date(holdout_start), "end_exclusive_utc": str(end),
                          "chosen": {"parameters": rules[chosen[0]], "train": chosen[1]} if chosen else None,
                          "train": chosen[1] if chosen else None, "test": final_stats,
                          "by_symbol": _by_symbol(final, holdout_start, last, config)},
        "validation_checks": checks, "target_met": checks["walk_forward_target_met"] and checks["holdout_target_met"],
        "publishable": all(checks.values()), "live_approved": False,
        "equity_curve": _equity(oos, initial_bankroll, "walk_forward") + _equity(final, initial_bankroll, "final_holdout"),
        "trade_ledger": oos + final, "cost_scenarios": stress,
        "data_quality": {"complete": True, "continuous": True, "valid_ohlc": True},
        "confidence_method": {"wilson": "95% on decisive outcomes", "block_bootstrap": "1000 seeded calendar-day resamples; both symbols jointly, zero-trade days included", "seed": 7219}}
