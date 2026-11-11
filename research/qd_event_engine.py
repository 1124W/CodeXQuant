"""Fixed-grid event-contract experiments driven by QuantDinger factors.

The grid is declared before any evaluation. A rule is selected using only the
training and validation windows; only that rule is evaluated in the test window.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math

from .event_engine import (
    DAY_MS, MINUTE_MS, SYMBOLS, KNOWN_UNAVAILABLE_DATES,
    _by_symbol, _equity, _monthly, event_stats, simulate_portfolio, utc_date,
    utc_ms,
)


DEFAULT_CONFIG = {
    "payout": .8,
    "fee": .01,
    "target_win_rate": .58,
    "latency_minutes": 1,
    "initial_bankroll": 1000.,
    "stake": 5.,
    "max_open_positions": 1,
    "daily_loss_limit": 25.,
    "adverse_bps": 0.,
    "min_train_trades": 100,
    "min_validation_trades": 40,
    "min_test_trades": 100,
}


def generate_rules():
    """Return the complete predeclared 72-rule research grid."""
    rules = []
    for family, thresholds in (
        ("rsi2_reversal", (5., 10., 20.)),
        ("rsi14_reversal", (25., 30., 35.)),
        ("z20_reversal", (1.5, 2., 2.5)),
    ):
        for threshold in thresholds:
            for timeframe in (5, 15):
                for horizon in (10, 30):
                    for trend_limit in (None, .6):
                        gate = "all" if trend_limit is None else str(trend_limit)
                        rules.append({
                            "id": f"{family}_t{threshold:g}_bar{timeframe}_expiry{horizon}_trend{gate}",
                            "family": family,
                            "threshold": threshold,
                            "timeframe_minutes": timeframe,
                            "horizon_minutes": horizon,
                            "max_abs_trend": trend_limit,
                        })
    return rules


STRATEGY_GRID = generate_rules()
GRID_SHA256 = hashlib.sha256(
    json.dumps(STRATEGY_GRID, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_UNAVAILABLE_DAYS = frozenset(
    utc_ms(date.fromisoformat(day)) // DAY_MS for day in KNOWN_UNAVAILABLE_DATES
)


def _as_ms(value):
    if isinstance(value, bool):
        raise ValueError("Boolean timestamps are invalid")
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000) if value.tzinfo is None else int(value.timestamp() * 1000)
    if isinstance(value, date):
        return utc_ms(value)
    if isinstance(value, str):
        return utc_ms(date.fromisoformat(value))
    if isinstance(value, int):
        return value
    raise ValueError("Use a UTC millisecond integer or an ISO date")


def _config(overrides):
    supplied = dict(overrides or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown demo configuration: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    for key in ("latency_minutes", "max_open_positions", "min_train_trades",
                "min_validation_trades", "min_test_trades"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in set(config) - {"latency_minutes", "max_open_positions", "min_train_trades",
                              "min_validation_trades", "min_test_trades"}:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if (config["payout"] <= 0 or not 0 <= config["fee"] < config["payout"]
            or not .5 < config["target_win_rate"] < 1
            or config["stake"] <= 0
            or config["initial_bankroll"] < config["stake"] * (1 + config["fee"])
            or config["daily_loss_limit"] < config["stake"]
            or not 0 <= config["adverse_bps"] < 9998):
        raise ValueError("Invalid payout, costs, target or portfolio configuration")
    return config


def _validate_minutes(candles, first, end):
    if set(candles) != set(SYMBOLS):
        raise ValueError(f"Expected exactly {SYMBOLS}")
    for symbol in SYMBOLS:
        rows = candles[symbol]
        if not rows or rows[0][0] > first or rows[-1][0] < end - MINUTE_MS:
            raise ValueError(f"Minute history does not cover all windows: {symbol}")
        origin = rows[0][0]
        for index, row in enumerate(rows):
            if len(row) != 5:
                raise ValueError(f"Expected timestamp and OHLC: {symbol}")
            timestamp, opened, high, low, close = row
            if (not isinstance(timestamp, int) or isinstance(timestamp, bool)
                    or timestamp % MINUTE_MS or timestamp != origin + index * MINUTE_MS
                    or not all(math.isfinite(value) and value > 0 for value in row[1:])
                    or not low <= min(opened, close) <= max(opened, close) <= high):
                raise ValueError(f"Invalid minute timestamp, grid or OHLC: {symbol} row {index}")


def _feature_frame(features, symbol, timeframe):
    if (symbol, timeframe) in features:
        return features[(symbol, timeframe)]
    values = features.get(symbol, {})
    if timeframe in values:
        return values[timeframe]
    if str(timeframe) in values:
        return values[str(timeframe)]
    raise ValueError(f"Missing features for {symbol} at {timeframe} minutes")


def _direction(rule, row):
    field = {"rsi2_reversal": "rsi2", "rsi14_reversal": "rsi14", "z20_reversal": "z20"}[rule["family"]]
    value = getattr(row, field)
    if not math.isfinite(value):
        return 0
    limit = rule["max_abs_trend"]
    if limit is not None and (not math.isfinite(row.trend) or abs(row.trend) > limit):
        return 0
    threshold = rule["threshold"]
    if field == "z20":
        return 1 if value <= -threshold else -1 if value >= threshold else 0
    return 1 if value <= threshold else -1 if value >= 100 - threshold else 0


def build_candidates(candles, features_by_symbol_timeframe, *, rules=None,
                     latency_minutes=1, start_ms=None, end_ms=None):
    """Build trades from closed-bar features and exact delayed minute opens.

    ``end_ms`` is exclusive for settlement. The earlier frame prefix is read to
    keep edge-triggered signals identical when a new evaluation window starts.
    """
    if not isinstance(latency_minutes, int) or isinstance(latency_minutes, bool) or latency_minutes < 1:
        raise ValueError("Latency must be a positive whole number of minutes")
    selected_rules = generate_rules() if rules is None else list(rules)
    result = {rule["id"]: [] for rule in selected_rules}
    for symbol in SYMBOLS:
        series = candles[symbol]
        if not series:
            continue
        origin = series[0][0]
        frames = {}
        for rule in selected_rules:
            timeframe = rule["timeframe_minutes"]
            if timeframe not in frames:
                frame = _feature_frame(features_by_symbol_timeframe, symbol, timeframe)
                required = {"timestamp", "rsi2", "rsi14", "z20", "trend"}
                if not required.issubset(frame.columns):
                    raise ValueError(f"Missing QuantDinger feature columns: {required - set(frame.columns)}")
                frames[timeframe] = list(frame.itertuples(index=False))
            previous_direction = 0
            last_signal_ms = None
            for index, row in enumerate(frames[timeframe]):
                signal_ms = int(row.timestamp)
                if signal_ms != row.timestamp or signal_ms % (timeframe * MINUTE_MS):
                    raise ValueError("Features must use clock-aligned complete-bar close times")
                if last_signal_ms is not None and signal_ms <= last_signal_ms:
                    raise ValueError("Feature timestamps must be strictly increasing")
                last_signal_ms = signal_ms
                if end_ms is not None and signal_ms >= end_ms:
                    break
                if index < 60:
                    continue
                direction = _direction(rule, row)
                new_signal = direction and direction != previous_direction
                previous_direction = direction
                if not new_signal:
                    continue
                if start_ms is not None and signal_ms < start_ms:
                    continue
                entry_ms = signal_ms + latency_minutes * MINUTE_MS
                settlement_ms = entry_ms + rule["horizon_minutes"] * MINUTE_MS
                if end_ms is not None and settlement_ms >= end_ms:
                    continue
                if any(day in _UNAVAILABLE_DAYS
                       for day in range(signal_ms // DAY_MS, settlement_ms // DAY_MS + 1)):
                    continue
                entry_index = (entry_ms - origin) // MINUTE_MS
                exit_index = (settlement_ms - origin) // MINUTE_MS
                if not 0 <= entry_index < exit_index < len(series):
                    continue
                if series[entry_index][0] != entry_ms or series[exit_index][0] != settlement_ms:
                    raise ValueError("Entry or settlement is missing from the exact minute grid")
                result[rule["id"]].append({
                    "symbol": symbol, "signal_ms": signal_ms,
                    "entry_ms": entry_ms, "settlement_ms": settlement_ms,
                    "direction": direction, "entry": series[entry_index][1],
                    "exit": series[exit_index][1], "rule_id": rule["id"],
                    "timeframe_minutes": timeframe,
                    "horizon_minutes": rule["horizon_minutes"],
                })
    return result


def _portfolio_config(config):
    return {key: config[key] for key in ("payout", "fee", "initial_bankroll", "stake",
            "max_open_positions", "daily_loss_limit", "adverse_bps")}


def _window(trades, start_ms, end_ms):
    return [trade for trade in trades
            if trade["signal_ms"] >= start_ms and trade["settlement_ms"] < end_ms]


def _sample(raw, start_ms, end_ms, config, *, bootstrap=False):
    ledger = simulate_portfolio(_window(raw, start_ms, end_ms), **_portfolio_config(config))
    return ledger, event_stats(ledger, start_ms, end_ms, payout=config["payout"],
                               fee=config["fee"], bootstrap=bootstrap)


def select_rule(development_candidates, start_ms, validation_start_ms, test_start_ms, config):
    """Select using the two earlier windows and a deterministic grid-order tie."""
    table = []
    for rule in generate_rules():
        raw = development_candidates[rule["id"]]
        _, train = _sample(raw, start_ms, validation_start_ms, config)
        _, validation = _sample(raw, validation_start_ms, test_start_ms, config)
        reasons = []
        for split, stats, minimum in (("train", train, config["min_train_trades"]),
                                      ("validation", validation, config["min_validation_trades"])):
            if stats["decisive_trades"] < minimum:
                reasons.append(f"{split}_insufficient_samples")
            if stats["win_rate"] is None or stats["win_rate"] < config["target_win_rate"]:
                reasons.append(f"{split}_below_target")
            if stats["net_units"] <= 0:
                reasons.append(f"{split}_nonpositive_net")
        # The validation lower bound controls ranking. The training score breaks
        # a tie without allowing the final test to influence any decision.
        empty_score = -2 - config["fee"]
        train_score = ((1 + config["payout"]) * train["win_rate_lower_95"] - 1 - config["fee"]
                       if train["win_rate_lower_95"] is not None else empty_score)
        score = ((1 + config["payout"]) * validation["win_rate_lower_95"] - 1 - config["fee"]
                 if validation["win_rate_lower_95"] is not None else empty_score)
        table.append({"rule": rule, "train": train, "validation": validation,
                      "eligible": not reasons, "rejection_reasons": reasons,
                      "selection_score": score, "train_score": train_score})
    qualified = [item for item in table if item["eligible"]]
    sufficient = [item for item in table
                  if item["train"]["decisive_trades"] >= config["min_train_trades"]
                  and item["validation"]["decisive_trades"] >= config["min_validation_trades"]]
    pool = qualified or sufficient or table
    winner = max(pool, key=lambda item: (item["selection_score"], item["train_score"]))
    return winner, table


def evaluate_demo(candles, start, validation_start, test_start, end, config=None, features=None):
    """Evaluate one pre-test-selected rule on a single fixed final test window."""
    config = _config(config)
    first, validation_ms, test_ms, last = map(_as_ms, (start, validation_start, test_start, end))
    if not first < validation_ms < test_ms < last or any(value % MINUTE_MS for value in (first, validation_ms, test_ms, last)):
        raise ValueError("Require ordered, minute-aligned train, validation and test windows")
    _validate_minutes(candles, first, last)
    if features is None:
        from .qd_factors import build_features
        features = {(symbol, timeframe): build_features(candles[symbol], timeframe)
                    for symbol in SYMBOLS for timeframe in (5, 15)}
    development = build_candidates(candles, features, latency_minutes=config["latency_minutes"],
                                   start_ms=first, end_ms=test_ms)
    winner, table = select_rule(development, first, validation_ms, test_ms, config)
    chosen = winner["rule"]
    # Only the selected rule gets final-test executions. No test leaderboard is
    # generated, and no rule is replaced after viewing its final-test outcome.
    test_candidates = build_candidates(candles, features, rules=[chosen],
        latency_minutes=config["latency_minutes"], start_ms=test_ms, end_ms=last)[chosen["id"]]
    test_ledger, test_stats = _sample(test_candidates, test_ms, last, config, bootstrap=True)
    for trade in test_ledger:
        trade["split"] = "test"
        trade["fold_id"] = "fixed_three_month_test"
    _, train = _sample(development[chosen["id"]], first, validation_ms, config, bootstrap=True)
    _, validation = _sample(development[chosen["id"]], validation_ms, test_ms, config, bootstrap=True)
    break_even = test_stats["effective_break_even_win_rate"]
    checks = {
        "pretest_selection_eligible": winner["eligible"],
        "min_test_samples": test_stats["decisive_trades"] >= config["min_test_trades"],
        "target_win_rate": test_stats["win_rate"] is not None and test_stats["win_rate"] >= config["target_win_rate"],
        "positive_net": test_stats["net_units"] > 0,
        "wilson_above_break_even": break_even is not None and test_stats["win_rate_lower_95"] > break_even,
        "block_win_rate_above_break_even": break_even is not None and test_stats["block_bootstrap_lower_95"] is not None and test_stats["block_bootstrap_lower_95"] > break_even,
        "block_net_positive": test_stats["block_bootstrap_net_lower_95"] is not None and test_stats["block_bootstrap_net_lower_95"] > 0,
    }
    scenarios = []
    for name, payout, fee, adverse in (
        ("base", config["payout"], config["fee"], config["adverse_bps"]),
        ("payout_075", min(config["payout"], .75), config["fee"], config["adverse_bps"]),
        ("payout_070", min(config["payout"], .70), config["fee"], config["adverse_bps"]),
        ("friction_002", config["payout"], max(config["fee"], .02), config["adverse_bps"]),
        ("adverse_strike_2bps", config["payout"], config["fee"], config["adverse_bps"] + 2),
    ):
        scenario_config = {**config, "payout": payout, "fee": fee, "adverse_bps": adverse}
        _, sample = _sample(test_candidates, test_ms, last, scenario_config)
        scenarios.append({"name": name, "payout": payout, "fee_per_stake": fee,
                          "adverse_bps": adverse, "test": sample})
    target_met = all(checks[key] for key in ("min_test_samples", "target_win_rate", "positive_net"))
    demo_pass = target_met and checks["pretest_selection_eligible"]
    return {
        "schema_version": 1,
        "engine": "quantdinger-event-demo-v1",
        "symbols": list(SYMBOLS),
        "start_utc": utc_date(first),
        "validation_start_utc": utc_date(validation_ms),
        "test_start_utc": utc_date(test_ms),
        "end_exclusive_utc": utc_date(last),
        "windows": {
            "train": {"start_utc": utc_date(first), "end_exclusive_utc": utc_date(validation_ms)},
            "validation": {"start_utc": utc_date(validation_ms), "end_exclusive_utc": utc_date(test_ms)},
            "test": {"start_utc": utc_date(test_ms), "end_exclusive_utc": utc_date(last)},
        },
        "config": config,
        "target_win_rate": config["target_win_rate"],
        "break_even_win_rate": (1 + config["fee"]) / (1 + config["payout"]),
        "candidate_count": len(STRATEGY_GRID),
        "strategy_grid": generate_rules(),
        "grid_sha256": GRID_SHA256,
        "selected_rule": chosen,
        "selection_eligible": winner["eligible"],
        "selection_mode": "qualified_prior_winner" if winner["eligible"] else "rejected_prior_winner_demo_only",
        "selection_table": table,
        "selection": "Fixed 72-rule grid; train and validation must independently meet sample, 58% and positive-net gates; rank validation Wilson-lower expectancy with train tie-break. If no rule qualifies, show the best prior-only rule as a rejected demo. Final test is never used to select or replace a rule.",
        "train": train,
        "validation": validation,
        "test": test_stats,
        "by_symbol": _by_symbol(test_ledger, test_ms, last, config),
        "monthly": _monthly(test_ledger, test_ms, last, config),
        "equity_curve": _equity(test_ledger, config["initial_bankroll"], "test"),
        "trade_ledger": test_ledger,
        "cost_scenarios": scenarios,
        "validation_checks": checks,
        "target_met": target_met,
        "demo_pass": demo_pass,
        "research_qualified": all(checks.values()),
        "live_approved": False,
        "validation_context": {
            "selected_on_pretest_only": True,
            "test_rules_evaluated": 1,
            "historical_dates_previously_observed": True,
            "pristine_unobserved_holdout": False,
            "forward_sample_trades": 0,
        },
        "assumptions": [
            "Binance USD-M index minute OHLC; features use only completed 5/15-minute bars.",
            "Entry is the minute OPEN after configured latency; expiry is the minute OPEN exactly 10/30 minutes after entry. This is not second-exact exchange execution.",
            "Fixed payout is a modeled scenario, not recovered order-specific historical odds.",
            "Configured friction is a conservative research assumption per stake, not an official additional event fee.",
            "Unchanged entry and expiry index values refund principal while modeled friction remains; win rate excludes refunds.",
            "Constant 5 USDT default stake, one open event globally, no compounding or martingale; BTC precedes ETH on simultaneous signals.",
            "UTC daily risk cap counts losing premiums and currently open premium risk; settled winners do not reset the loss budget.",
            "Train and validation each start with the configured bankroll. Trades expiring on or after their split boundary are purged.",
            "Only the one prior-selected rule is evaluated on the final test. All 72 prior-window results are disclosed.",
            "Confidence bounds are unadjusted for strategy selection; historical dates overlap earlier research and require fresh forward verification.",
            "Stress scenarios retain the selected signals; altered execution costs can change risk admission.",
        ],
    }
