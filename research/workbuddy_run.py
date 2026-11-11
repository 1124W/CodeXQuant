"""Three-year BTC/ETH event study using QuantDinger and the user's WorkBuddy rule."""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .event_engine import (SYMBOLS, _by_symbol, _equity, _monthly, event_stats,
                           simulate_portfolio, utc_ms)
from .qd_demo import months_before, write_atomic

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "workbuddy-results.json"
TRADES = ROOT / "research" / "workbuddy-trades.csv"
RULE = {"id": "workbuddy_contrarian_gbm_30m", "family": "workbuddy_contrarian_gbm",
        "horizon_minutes": 30, "timeframe_minutes": 1}


def log(message):
    print(message, file=sys.stderr, flush=True)


def years_before(day, years=3):
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def source_hash():
    digest = hashlib.sha256()
    for name in ("workbuddy_run.py", "workbuddy_strategy.py", "qd_three_year_data.py", "qd_factors.py", "event_engine.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def csv_text(report):
    output = io.StringIO(newline="")
    fields = ["run_id", "symbol", "rule_id", "signal_ms", "entry_ms", "settlement_ms", "direction",
              "entry", "exit", "p", "ret_60", "stake", "outcome", "net_units", "horizon_minutes", "fold_id"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({**trade, "run_id": report["provenance"]["run_id"]} for trade in report["trade_ledger"])
    return output.getvalue()


def portfolio_config(config):
    return {name: config[name] for name in ("payout", "fee", "initial_bankroll", "stake", "max_open_positions", "daily_loss_limit", "adverse_bps")}


def summarize(raw, start, end, config, *, bootstrap=True):
    ledger = simulate_portfolio(raw, **portfolio_config(config))
    for trade in ledger:
        trade["split"] = "walk_forward"
    stats = event_stats(ledger, start, end, payout=config["payout"], fee=config["fee"], bootstrap=bootstrap)
    return ledger, stats


def run(args):
    import gc
    import pandas as pd
    from .qd_factors import upstream_provenance
    from .qd_three_year_data import load_market
    from .workbuddy_strategy import build_samples, reference_provenance, walk_forward_signals

    started = time.perf_counter()
    end = args.end
    start = args.start or years_before(end)
    train_days = args.train_days if args.train_days is not None else (180 if args.profile == "research" else 240)
    if start >= end or end > datetime.now(timezone.utc).date():
        raise ValueError("Require a historical start before the exclusive UTC end")
    if train_days < 1 or args.test_days < 1:
        raise ValueError("Training and prediction windows must be positive")
    if args.output.resolve() == args.trades_output.resolve():
        raise ValueError("JSON and CSV output paths must differ")
    config = {"payout": .85, "fee": args.fee, "target_win_rate": .58,
              "initial_bankroll": args.initial_bankroll, "stake": args.stake,
              "max_open_positions": 1, "daily_loss_limit": args.daily_loss_limit,
              "adverse_bps": 0., "latency_minutes": 1, "horizon_minutes": 30,
              "train_days": train_days, "test_days": args.test_days, "min_test_trades": 100,
              "model_profile": args.profile}
    if (any(not math.isfinite(config[key]) for key in ("fee", "initial_bankroll", "stake", "daily_loss_limit"))
            or not 0 <= config["fee"] < .85 or config["stake"] < 5
            or config["initial_bankroll"] < config["stake"] * (1 + config["fee"])
            or config["daily_loss_limit"] < config["stake"]):
        raise ValueError("Invalid costs/capital; the fixed stake must be at least 5 USDT")
    reference = reference_provenance()
    upstream = upstream_provenance()
    plan = {"start": str(start), "end_exclusive": str(end), "config": config, "rule": RULE,
            "reference": reference, "quantdinger": upstream, "source_code_sha256": source_hash(),
            "strategy_search_count": 1, "thresholds_frozen_from_reference": True}
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    write_atomic(ROOT / "research" / ".qd-runs" / f"workbuddy-plan-{plan_sha[:20]}.json",
                 json.dumps({**plan, "recorded_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    data_start = start - timedelta(days=3)
    raw, folds, sources, quality, diagnostics = [], [], [], {}, {}
    digest = hashlib.sha256()
    predictions, model_metrics = {}, {}
    first_oos = None
    for symbol in SYMBOLS:
        log(f"[{symbol}] Loading verified spot features + index settlement: {data_start} to {end}")
        spot, index, manifest = load_market(symbol, data_start, end, args.cache_dir, args.offline)
        sources.extend(manifest)
        quality[symbol] = {"spot_minutes": len(spot), "index_minutes": len(index)}
        for name, frame in (("spot", spot), ("index", index)):
            digest.update(f"{symbol}:{name}\n".encode())
            digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
        log(f"[{symbol}] Building reference features in bounded chunks")
        samples = build_samples(spot, index, symbol, profile=args.profile)
        diagnostics[symbol] = dict(samples.attrs)
        del spot, index
        gc.collect()
        log(f"[{symbol}] {len(samples):,} closed-window samples; starting rolling training")
        result = walk_forward_signals(samples, symbol, start, end, train_days=train_days,
                                      test_days=args.test_days, progress=log, profile=args.profile)
        raw.extend(result["raw_trades"])
        folds.extend(result["folds"])
        predictions[symbol] = result["prediction_count"]
        model_metrics[symbol] = {"auc": result["auc"], "prediction_flat_count": result["prediction_flat_count"],
                                 "model_parameters": result["model_parameters"],
                                 "feature_count": len(samples.attrs.get("features", reference["features"]))}
        current_first = result["oos_start_utc"]
        first_oos = current_first if first_oos is None else min(first_oos, current_first)
        del samples, result
        gc.collect()
    first_ms, end_ms = utc_ms(date.fromisoformat(str(first_oos)[:10])), utc_ms(end)
    ledger, sample = summarize(raw, first_ms, end_ms, config)
    latest_start = max(months_before(end), date.fromisoformat(str(first_oos)[:10]))
    recent = [trade for trade in ledger if trade["signal_ms"] >= utc_ms(latest_start)]
    recent_stats = event_stats(recent, utc_ms(latest_start), end_ms, payout=.85, fee=config["fee"])
    recent_stats.update(start_utc=str(latest_start), end_exclusive_utc=str(end))
    monthly = _monthly(ledger, first_ms, end_ms, config)
    yearly = []
    for year in range(date.fromisoformat(str(first_oos)[:10]).year, end.year + 1):
        lower = max(first_ms, utc_ms(date(year, 1, 1)))
        upper = min(end_ms, utc_ms(date(year + 1, 1, 1)))
        if lower < upper:
            values = [trade for trade in ledger if lower <= trade["settlement_ms"] < upper]
            yearly.append({"year": str(year), **event_stats(values, lower, upper, payout=.85, fee=config["fee"], bootstrap=False)})
    scenarios = []
    for name, payout, friction, adverse in (("base", .85, config["fee"], 0),
        ("payout_080", .80, config["fee"], 0), ("payout_075", .75, config["fee"], 0),
        ("no_friction", .85, 0., 0), ("friction_001", .85, .01, 0), ("friction_002", .85, .02, 0),
        ("adverse_strike_2bps", .85, config["fee"], 2)):
        _, stats = summarize(raw, first_ms, end_ms, {**config, "payout": payout, "fee": friction, "adverse_bps": adverse}, bootstrap=False)
        scenarios.append({"name": name, "payout": payout, "fee_per_stake": friction, "adverse_bps": adverse, "test": stats})
    break_even = sample["effective_break_even_win_rate"]
    checks = {"only_btc_eth_30m": all(t["symbol"] in SYMBOLS and t["horizon_minutes"] == 30 for t in ledger),
              "min_test_samples": sample["decisive_trades"] >= 100,
              "target_win_rate": sample["win_rate"] is not None and sample["win_rate"] >= .58,
              "positive_net": sample["net_units"] > 0,
              "wilson_above_break_even": break_even is not None and sample["win_rate_lower_95"] > break_even,
              "block_net_positive": sample["block_bootstrap_net_lower_95"] is not None and sample["block_bootstrap_net_lower_95"] > 0,
              "training_labels_settled_before_fit": all(f.get("purge_passed", False) for f in folds)}
    identity = {"plan_sha256": plan_sha, "data_sha256": digest.hexdigest(), "source_code_sha256": source_hash(),
                "configuration": config, "reference_commit": reference["commit"], "upstream_commit": upstream["commit"]}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report = {"schema_version": 1, "engine": "quantdinger-workbuddy-v1", "symbols": list(SYMBOLS),
              "data_start_utc": str(start), "feature_warmup_start_utc": str(data_start),
              "test_start_utc": str(first_oos)[:10], "oos_start_utc": str(first_oos)[:10], "end_exclusive_utc": str(end),
              "windows": {"test": {"start_utc": str(first_oos)[:10], "end_exclusive_utc": str(end)}},
              "config": config, "selected_rule": RULE, "strategy": {"margin": .12,
                "min_abs_ret_bps": {"BTCUSDT": 20., "ETHUSDT": 28.},
                "description": "GBM probability differs from 0.5 by at least 0.12; recent 60-minute return exceeds the symbol threshold; predicted direction opposes that return.",
                "train_days": train_days, "test_days": args.test_days, "model_profile": args.profile},
              "test": sample, "latest_three_months": recent_stats, "by_symbol": _by_symbol(ledger, first_ms, end_ms, config),
              "monthly": monthly, "yearly": yearly, "equity_curve": _equity(ledger, config["initial_bankroll"], "walk_forward"),
              "trade_ledger": ledger, "folds": folds, "cost_scenarios": scenarios, "validation_checks": checks,
              "target_win_rate": .58, "break_even_win_rate": (1 + config["fee"]) / 1.85,
              "selection_eligible": True, "demo_pass": checks["min_test_samples"] and checks["target_win_rate"] and checks["positive_net"],
              "research_qualified": all(checks.values()), "live_approved": False,
              "strategy_search_count": 1, "reference": reference, "upstream": upstream,
              "data_quality": {"complete": True, "interval": "1m", "sources": ["Binance spot klines for features", "Binance USD-M indexPriceKlines for labels and settlement"],
                               "bars": quality, "hash_method": "pandas.hash_pandas_object uint64 bytes; pandas version recorded"},
              "model_predictions": predictions, "model_metrics": model_metrics, "factor_diagnostics": diagnostics, "sources": sources,
              "provenance": {**identity, "run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                             "runtime_seconds": round(time.perf_counter() - started, 2), "pandas_version": pd.__version__},
              "validation_context": {"thresholds_frozen_from_reference": True, "strategy_search_count": 1,
                                     "historical_reference_already_researched": True, "pristine_unobserved_holdout": False,
                                     "forward_sample_trades": 0, "initial_training_days_excluded_from_oos": train_days},
              "assumptions": ["Only BTCUSDT/ETHUSDT; fixed 30-minute expiry and 85% net payout on wins, as specified by the user.",
                  "Features use only completed spot minute bars; index minute OPEN prices determine training labels and event outcomes.",
                  "Signals occur on clock-aligned 30-minute boundaries; index entry is delayed one minute, expiry is 30 minutes after entry.",
                  f"Rolling models train on {train_days} earlier days and predict the next {args.test_days} days; training outcomes must settle strictly before each fit.",
                  "Equal entry/expiry index values refund principal and still incur the modeled friction; test refunds are never dropped in advance.",
                  "85% payout and no additional fee are user-confirmed current terms, held fixed across the historical scenario; 1%/2% research friction is shown separately as stress, not charged in the zero-fee baseline.",
                  "The initial training window belongs to the three-year data span and has no out-of-sample trades.",
                  "BTC and ETH share one open position and a daily risk budget; simultaneous signals give BTC deterministic priority.",
                  "The strategy and thresholds were already researched in the reference repository; rolling predictions are chronological but are not untouched forward evidence."]}
    comparison_path = ROOT / "research" / "workbuddy-live-results.json"
    if args.profile == "research" and comparison_path.is_file():
        previous = json.loads(comparison_path.read_text(encoding="utf-8"))
        if previous["provenance"]["data_sha256"] == identity["data_sha256"]:
            comparable_sample = previous["test"]
            if previous["config"]["fee"] != config["fee"] and config["fee"] == 0:
                comparable_sample = next(s["test"] for s in previous["cost_scenarios"] if s["name"] == "no_friction")
            report["profile_comparison"] = [
                {"profile": "live", "train_days": previous["config"]["train_days"], "feature_count": 28,
                 "oos_start_utc": previous["oos_start_utc"], "test": comparable_sample, "run_id": previous["provenance"]["run_id"],
                 "fee": config["fee"], "scenario": "no_friction" if previous["config"]["fee"] != config["fee"] else "base"},
                {"profile": "research", "train_days": train_days,
                 "feature_count": model_metrics[SYMBOLS[0]]["feature_count"],
                 "oos_start_utc": report["oos_start_utc"], "test": sample, "run_id": run_id, "fee": config["fee"]},
            ]
    from .qd_demo_web import validate_report
    validate_report(report)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_atomic(args.trades_output, csv_text(report))
    write_atomic(args.output, payload)
    write_atomic(ROOT / "research" / ".qd-runs" / f"workbuddy-{run_id}.json", payload)
    log(f"Finished in {time.perf_counter() - started:.1f}s: {sample['decisive_trades']} trades, win rate={sample['win_rate']}, net={sample['net_units']}")
    return report


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 9))
    cli.add_argument("--start", type=date.fromisoformat)
    cli.add_argument("--profile", choices=("research", "live"), default="research")
    cli.add_argument("--train-days", type=int)
    cli.add_argument("--test-days", type=int, default=30)
    cli.add_argument("--fee", type=float, default=0.)
    cli.add_argument("--initial-bankroll", type=float, default=1000.)
    cli.add_argument("--stake", type=float, default=5.)
    cli.add_argument("--daily-loss-limit", type=float, default=25.)
    cli.add_argument("--cache-dir", type=Path, default=ROOT / "research" / ".cache")
    cli.add_argument("--output", type=Path, default=REPORT)
    cli.add_argument("--trades-output", type=Path, default=TRADES)
    cli.add_argument("--offline", action="store_true")
    cli.add_argument("--serve", action="store_true")
    cli.add_argument("--port", type=int, default=4180)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.serve:
            from .qd_demo_web import create_app
            log(f"CodeXQuant three-year research: http://127.0.0.1:{args.port}")
            create_app(args.output, args.trades_output).run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        else:
            report = run(args)
            print(json.dumps({"run_id": report["provenance"]["run_id"], "test": report["test"],
                              "latest_three_months": report["latest_three_months"], "demo_pass": report["demo_pass"]}))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        cli.exit(1, f"WorkBuddy study failed: {error}\n")


if __name__ == "__main__":
    main()
