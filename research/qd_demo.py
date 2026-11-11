"""Run the QuantDinger event research extension and its read-only local demo."""

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .event_data import data_fingerprint, download_index_minutes
from .qd_event_engine import DEFAULT_CONFIG, GRID_SHA256, evaluate_demo, generate_rules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "research" / "quantdinger-demo.json"
DEFAULT_TRADES = ROOT / "research" / "quantdinger-demo-trades.csv"
DEFAULT_UPSTREAM = ROOT / "external" / "QuantDinger-repo"


def months_before(day, count=3):
    month_number = day.year * 12 + day.month - 1 - count
    year, month = divmod(month_number, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def log(message):
    print(message, file=sys.stderr, flush=True)


def fingerprint_sources():
    digest = hashlib.sha256()
    for name in ("qd_demo.py", "qd_factors.py", "qd_event_engine.py", "event_data.py", "event_engine.py", "engine.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def csv_payload(report):
    stream = io.StringIO(newline="")
    fields = ["run_id", "symbol", "rule_id", "signal_ms", "entry_ms", "settlement_ms", "direction",
              "entry", "exit", "outcome", "stake", "net_units", "horizon_minutes", "timeframe_minutes", "split"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({**trade, "run_id": report["provenance"]["run_id"]} for trade in report["trade_ledger"])
    return stream.getvalue()


def write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
    temporary.replace(path)


def run(args):
    from .qd_factors import build_features, upstream_provenance

    started = time.perf_counter()
    if args.end > datetime.now(timezone.utc).date():
        raise ValueError("The exclusive end date cannot be in the future")
    if args.train_days < 1 or args.validation_days < 1:
        raise ValueError("Train and validation windows must be positive")
    if args.output.resolve() == args.trades_output.resolve():
        raise ValueError("Report and trade CSV require different output paths")
    test_start = months_before(args.end)
    validation_start = test_start - timedelta(days=args.validation_days)
    start = validation_start - timedelta(days=args.train_days)
    data_start = start - timedelta(days=3)
    configuration = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    # Validate numeric configuration before any network or artifact mutation.
    from .qd_event_engine import _config
    configuration = _config(configuration)
    upstream = upstream_provenance(args.upstream_dir)
    plan = {"test_start": str(test_start), "end_exclusive": str(args.end), "train_start": str(start),
            "validation_start": str(validation_start), "configuration": configuration,
            "strategy_grid": generate_rules(), "grid_sha256": GRID_SHA256,
            "upstream": upstream, "source_sha256": fingerprint_sources(),
            "selection_uses_test": False}
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan_path = ROOT / "research" / ".qd-runs" / f"plan-{plan_sha[:20]}.json"
    if not plan_path.exists():
        write_atomic(plan_path, json.dumps({**plan, "recorded_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    log(f"Plan frozen: {plan_sha[:20]}; final test {test_start} to {args.end} (UTC, exclusive)")
    log(f"Loading official index data {data_start} to {args.end}; offline={args.offline}")
    candles, sources = download_index_minutes(data_start, args.end, args.cache_dir, args.offline, workers=4)
    log(f"Data verified: { {symbol: len(rows) for symbol, rows in candles.items()} }")
    features = {}
    for symbol, rows in candles.items():
        for timeframe in (5, 15):
            log(f"Computing QuantDinger factors: {symbol}, {timeframe}m closed bars")
            features[symbol, timeframe] = build_features(rows, timeframe, args.upstream_dir)
    log("Selecting on earlier train/validation windows; evaluating one frozen rule on the final test")
    report = evaluate_demo(candles, start, validation_start, test_start, args.end,
                           configuration, features=features)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    identity = {"plan_sha256": plan_sha, "data_sha256": data_fingerprint(candles),
                "source_code_sha256": fingerprint_sources(), "upstream_commit": upstream.get("commit"),
                "configuration": configuration, "test_start": str(test_start), "end_exclusive": str(args.end)}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report.update({
        "provenance": {**identity, "run_id": run_id, "git_revision": revision,
                       "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "runtime_seconds": round(time.perf_counter() - started, 3)},
        "upstream": upstream,
        "data_quality": {"source": "Binance USD-M indexPriceKlines", "interval": "1m",
                         "complete": True, "continuous": True, "valid_ohlc": True,
                         "archive_exchange_checksums_verified": True,
                         "rest_repairs_use_local_sha256": True,
                         "bars": {symbol: len(rows) for symbol, rows in candles.items()},
                         "start_utc": str(data_start), "end_exclusive_utc": str(args.end)},
        "sources": sources,
        "rules_source": "https://www.binance.com/en/support/faq/detail/172247f1c4b8401ab106fb63900f3825",
        "runtime_mode": "quantdinger_factor_research_extension",
        "full_platform_services_started": False,
    })
    from .qd_demo_web import validate_report
    validate_report(report)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_atomic(args.trades_output, csv_payload(report))
    write_atomic(args.output, payload)
    write_atomic(ROOT / "research" / ".qd-runs" / f"{run_id}.json", payload)
    log(f"Saved report and ledger in {time.perf_counter() - started:.1f}s")
    return report


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date() - timedelta(days=2))
    cli.add_argument("--train-days", type=int, default=90)
    cli.add_argument("--validation-days", type=int, default=30)
    cli.add_argument("--cache-dir", type=Path, default=ROOT / "research" / ".cache")
    cli.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    cli.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    cli.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    cli.add_argument("--offline", action="store_true")
    cli.add_argument("--serve", action="store_true", help="Serve the existing report on loopback; does not launch the full platform")
    cli.add_argument("--port", type=int, default=4180)
    for key, default in DEFAULT_CONFIG.items():
        cli.add_argument("--" + key.replace("_", "-"), type=int if isinstance(default, int) else float, default=default)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.serve:
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            from .qd_demo_web import create_app
            app = create_app(args.output, args.trades_output)
            log(f"QuantDinger event research demo: http://127.0.0.1:{args.port}")
            app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
            return
        report = run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        cli.exit(1, f"QuantDinger event demo failed: {error}\n")
    print(json.dumps({"run_id": report["provenance"]["run_id"], "selected_rule": report["selected_rule"],
                      "selection_eligible": report["selection_eligible"], "test": report["test"],
                      "demo_pass": report["demo_pass"], "research_qualified": report["research_qualified"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
