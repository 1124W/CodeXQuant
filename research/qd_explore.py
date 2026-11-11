"""Explicit in-sample historical search over the already frozen 72-rule grid.

This exploratory artifact is intentionally separate from the primary pre-test
selected run. Its selected winner is subject to selection bias and is never
classified as out-of-sample or approved for live trading.
"""

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from .event_data import data_fingerprint, download_index_minutes
from .event_engine import SYMBOLS, _by_symbol, _equity, _monthly, event_stats
from .qd_demo import DEFAULT_REPORT, DEFAULT_UPSTREAM, ROOT, csv_payload, log, write_atomic
from .qd_event_engine import (
    GRID_SHA256, _as_ms, _config, _sample, _validate_minutes,
    build_candidates, generate_rules,
)


DEFAULT_EXPLORATION = ROOT / "research" / "quantdinger-exploration.json"
DEFAULT_TRADES = ROOT / "research" / "quantdinger-exploration-trades.csv"


def _fingerprint():
    digest = hashlib.sha256()
    for name in ("qd_explore.py", "qd_event_engine.py", "qd_factors.py", "event_engine.py", "event_data.py", "engine.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run_exploration(primary, candles, features):
    """Explore exactly the primary grid and data, disclosing every test result."""
    config = _config(primary["config"])
    if primary.get("grid_sha256") != GRID_SHA256 or primary.get("strategy_grid") != generate_rules():
        raise ValueError("Exploration requires the unchanged primary 72-rule grid")
    if primary.get("candidate_count") != 72:
        raise ValueError("The primary experiment must contain exactly 72 candidates")
    observed_hash = data_fingerprint(candles)
    if observed_hash != primary["provenance"]["data_sha256"]:
        raise ValueError("Exploration data does not match the primary experiment fingerprint")
    first = _as_ms(primary["start_utc"])
    test_start, end = map(_as_ms, (primary["test_start_utc"], primary["end_exclusive_utc"]))
    _validate_minutes(candles, first, end)
    candidates = build_candidates(candles, features, latency_minutes=config["latency_minutes"],
                                  start_ms=test_start, end_ms=end)
    minimum = max(100, config["min_test_trades"])
    table = []
    for order, rule in enumerate(generate_rules()):
        _, stats = _sample(candidates[rule["id"]], test_start, end, config)
        passed = (stats["decisive_trades"] >= minimum
                  and stats["win_rate"] is not None
                  and stats["win_rate"] >= config["target_win_rate"]
                  and stats["net_units"] > 0)
        score = ((1 + config["payout"]) * stats["win_rate_lower_95"] - 1 - config["fee"]
                 if stats["win_rate_lower_95"] is not None else -2 - config["fee"])
        table.append({"grid_order": order, "rule": rule, "test": stats,
                      "historical_target_met": passed, "selection_score": score})
    passed = [row for row in table if row["historical_target_met"]]
    sufficient = [row for row in table if row["test"]["decisive_trades"] >= minimum]
    pool = passed or sufficient or table
    selected = max(pool, key=lambda row: (row["selection_score"], row["test"]["decisive_trades"], -row["grid_order"]))
    rule = selected["rule"]
    raw = candidates[rule["id"]]
    ledger, stats = _sample(raw, test_start, end, config, bootstrap=True)
    for trade in ledger:
        trade["split"] = "historical_exploration"
        trade["fold_id"] = "historical_exploration"
    stresses = []
    for name, payout, fee, adverse in (
        ("base", config["payout"], config["fee"], config["adverse_bps"]),
        ("payout_075", min(config["payout"], .75), config["fee"], config["adverse_bps"]),
        ("payout_070", min(config["payout"], .70), config["fee"], config["adverse_bps"]),
        ("friction_002", config["payout"], max(config["fee"], .02), config["adverse_bps"]),
        ("adverse_strike_2bps", config["payout"], config["fee"], config["adverse_bps"] + 2),
    ):
        _, sample = _sample(raw, test_start, end, {**config, "payout": payout, "fee": fee, "adverse_bps": adverse})
        stresses.append({"name": name, "payout": payout, "fee_per_stake": fee,
                         "adverse_bps": adverse, "test": sample})
    identity = {
        "primary_run_id": primary["provenance"]["run_id"],
        "data_sha256": observed_hash,
        "grid_sha256": GRID_SHA256,
        "source_code_sha256": _fingerprint(),
        "selected_rule_id": rule["id"],
        "test_start": primary["test_start_utc"],
        "end_exclusive": primary["end_exclusive_utc"],
        "configuration": config,
        "selection_uses_test": True,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {
        "schema_version": 1,
        "engine": "quantdinger-event-exploration-v1",
        "evidence_type": "in_sample_historical_search",
        "selection_uses_test": True,
        "out_of_sample": False,
        "research_qualified": False,
        "live_approved": False,
        "target_met": selected["historical_target_met"],
        "historical_target_met": selected["historical_target_met"],
        "exploration_target_met": selected["historical_target_met"],
        "selection_eligible": False,
        "demo_pass": False,
        "selection_mode": "test_period_historical_search",
        "selection": "Explore the same 72 frozen primary rules on the already viewed target period. Require at least 100 decisive trades, target win rate and positive modeled net. Rank Wilson-lower expectancy, sample count, then original grid order. The result is selected in-sample and is not an independent validation result.",
        "search_count": len(table),
        "candidate_count": len(table),
        "historical_target_candidates": len(passed),
        "min_decisive_trades": minimum,
        "target_win_rate": config["target_win_rate"],
        "break_even_win_rate": (1 + config["fee"]) / (1 + config["payout"]),
        "config": config,
        "symbols": list(SYMBOLS),
        "test_start_utc": primary["test_start_utc"],
        "end_exclusive_utc": primary["end_exclusive_utc"],
        "windows": {"test": {"start_utc": primary["test_start_utc"],
                              "end_exclusive_utc": primary["end_exclusive_utc"]}},
        "strategy_grid": generate_rules(),
        "grid_sha256": GRID_SHA256,
        "selected_rule": rule,
        "test": stats,
        "search_table": table,
        "selection_table": table,
        "by_symbol": _by_symbol(ledger, test_start, end, config),
        "monthly": _monthly(ledger, test_start, end, config),
        "equity_curve": _equity(ledger, config["initial_bankroll"], "historical_exploration"),
        "trade_ledger": ledger,
        "cost_scenarios": stresses,
        "upstream": primary.get("upstream"),
        "data_quality": primary.get("data_quality"),
        "provenance": {**identity, "run_id": run_id,
                       "generated_at_utc": datetime.now(timezone.utc).isoformat()},
        "validation_context": {
            "target_period_already_viewed": True,
            "selection_uses_test": True,
            "pristine_unobserved_holdout": False,
            "forward_sample_trades": 0,
            "search_count": len(table),
            "confidence_bounds_adjusted_for_search": False,
            "primary_report_overwritten": False,
        },
        "limitations": [
            "The best rule is chosen after evaluating all 72 rules on this same period; its performance is subject to selection bias and overfitting.",
            "Wilson and daily bootstrap bounds describe the selected historical sample and are not corrected for the 72-rule search.",
            "Positive historical performance does not establish forward profitability or trading approval.",
            "Fixed payout and friction are modeled assumptions, and execution uses minute-open index proxies.",
            "The original prior-selected primary run remains unchanged and must be reported alongside this exploration.",
        ],
    }


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary-report", type=Path, default=DEFAULT_REPORT)
    cli.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 9))
    cli.add_argument("--cache-dir", type=Path, default=ROOT / "research" / ".cache")
    cli.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    cli.add_argument("--offline", action="store_true")
    cli.add_argument("--output", type=Path, default=DEFAULT_EXPLORATION)
    cli.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    started = time.perf_counter()
    try:
        forbidden = {args.primary_report.resolve(), DEFAULT_REPORT.resolve(),
                     (ROOT / "research" / "quantdinger-demo-trades.csv").resolve()}
        if args.output.resolve() in forbidden or args.trades_output.resolve() in forbidden:
            raise ValueError("Exploration cannot overwrite primary report or primary ledger")
        if args.output.resolve() == args.trades_output.resolve():
            raise ValueError("Exploration report and CSV need different paths")
        primary = json.loads(args.primary_report.read_text(encoding="utf-8"))
        if str(args.end) != primary["end_exclusive_utc"]:
            raise ValueError("Exploration end date must match the existing primary run")
        data_start = date.fromisoformat(primary["data_quality"]["start_utc"])
        log(f"Historical exploration only; loading the identical primary data {data_start} to {args.end}")
        candles, _ = download_index_minutes(data_start, args.end, args.cache_dir, args.offline)
        from .qd_factors import build_features
        features = {}
        for symbol in SYMBOLS:
            for timeframe in (5, 15):
                log(f"Computing original QuantDinger features: {symbol} {timeframe}m")
                features[symbol, timeframe] = build_features(candles[symbol], timeframe, args.upstream_dir)
        log("Exploring all 72 original rules on the already viewed three-month period; no out-of-sample claim")
        report = run_exploration(primary, candles, features)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        write_atomic(args.trades_output, csv_payload(report))
        write_atomic(args.output, payload)
        log(f"Saved exploratory results in {time.perf_counter() - started:.1f}s")
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        cli.exit(1, f"Historical exploration failed: {error}\n")
    print(json.dumps({"run_id": report["provenance"]["run_id"],
                      "evidence_type": report["evidence_type"],
                      "selection_uses_test": True,
                      "search_count": report["search_count"],
                      "historical_target_candidates": report["historical_target_candidates"],
                      "selected_rule": report["selected_rule"], "test": report["test"],
                      "historical_target_met": report["historical_target_met"],
                      "research_qualified": False, "live_approved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
