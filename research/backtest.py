"""Run auditable BTC/ETH index-based event-contract research (no trading orders)."""

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .event_data import INDEX_BASE, data_fingerprint, download_index_minutes
    from .event_engine import evaluate_events
else:
    from event_data import INDEX_BASE, data_fingerprint, download_index_minutes
    from event_engine import evaluate_events

ROOT = Path(__file__).resolve().parents[1]
RULES_URL = "https://www.binance.com/en/support/faq/detail/172247f1c4b8401ab106fb63900f3825"


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ("engine.py", "event_engine.py", "event_data.py", "backtest.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def validate_request(args):
    if args.days < args.train_days + args.validation_days + args.holdout_days:
        raise ValueError("--days must cover training, at least one validation window and final holdout")
    if min(args.train_days, args.validation_days, args.holdout_days) < 1:
        raise ValueError("All date windows must be positive")
    if args.end > datetime.now(timezone.utc).date():
        raise ValueError("--end must not be in the future")
    if not .5 < args.target_win_rate < 1:
        raise ValueError("--target-win-rate must be between .5 and 1")
    if not math.isfinite(args.payout) or args.payout <= 0 or not math.isfinite(args.fee) or not 0 <= args.fee < args.payout:
        raise ValueError("Require finite payout > 0 and 0 <= fee < payout")
    if args.latency_minutes < 1 or args.horizon_minutes not in (10, 30, 60, 1440):
        raise ValueError("Require latency >=1 minute and horizon of 10/30/60/1440 minutes")
    if args.min_train_trades < 1 or args.min_test_trades < 1 or args.max_open_positions < 1:
        raise ValueError("Trade counts and position limit must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.initial_bankroll, args.stake, args.daily_loss_limit)):
        raise ValueError("Bankroll, stake and daily loss limit must be finite and positive")
    if args.initial_bankroll < args.stake * (1 + args.fee):
        raise ValueError("Initial bankroll cannot fund a single trade")
    if not math.isfinite(args.adverse_bps) or not 0 <= args.adverse_bps < 9998:
        raise ValueError("Adverse entry basis points must be finite and in [0, 9998)")
    if args.daily_loss_limit < args.stake:
        raise ValueError("Daily loss limit must fund at least one stake")
    if args.output.resolve() == args.trades_output.resolve():
        raise ValueError("Report and trade CSV must have different paths")


def run_backtest(args):
    validate_request(args)
    start = args.end - timedelta(days=args.days)
    print(f"Loading verified index minutes: {start} to {args.end} (exclusive)", file=sys.stderr, flush=True)
    candles, archives = download_index_minutes(start, args.end, args.cache_dir, args.offline)
    print("Index data verified; evaluating frozen strategy grid and chronological windows", file=sys.stderr, flush=True)
    report = evaluate_events(candles, start, args.end, train_days=args.train_days,
                             validation_days=args.validation_days, holdout_days=args.holdout_days,
                             payout=args.payout, fee=args.fee, target_win_rate=args.target_win_rate,
                             latency_minutes=args.latency_minutes, horizon_minutes=args.horizon_minutes,
                             adverse_bps=args.adverse_bps, min_train_trades=args.min_train_trades,
                             min_test_trades=args.min_test_trades, initial_bankroll=args.initial_bankroll,
                             stake=args.stake, max_open_positions=args.max_open_positions,
                             daily_loss_limit=args.daily_loss_limit)
    configuration = {key: value for key, value in vars(args).items()
                     if key not in ("cache_dir", "output", "trades_output", "offline")}
    configuration["end"] = str(args.end)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    metadata = {"configuration": configuration, "data_sha256": data_fingerprint(candles),
                "source_code_sha256": source_fingerprint(), "git_revision": revision,
                "grid_sha256": hashlib.sha256(json.dumps(report["strategy_grid"], sort_keys=True).encode()).hexdigest()}
    metadata["run_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report.update({"source": INDEX_BASE, "archives": archives, "provenance": metadata,
                   "rules_source": RULES_URL,
                   "data_quality": {"complete": True, "continuous": True, "valid_ohlc": True,
                                    "archives_checksum_verified": True,
                                    "rest_repair_snapshots": sum(a.get("purpose") == "repair_missing_archive_minutes" for a in archives),
                                    "interval": "1m"},
                   "bars": {s: len(rows) for s, rows in candles.items()}})
    old_report = ROOT / "research" / "results.json"
    overlaps = False
    if old_report.is_file():
        try:
            old = json.loads(old_report.read_text(encoding="utf-8"))
            overlaps = start < date.fromisoformat(old["end_exclusive_utc"]) and args.end > date.fromisoformat(old["start_utc"])
        except (ValueError, KeyError, TypeError):
            pass
    report["validation_context"] = {
        "historical_research_only": True, "previous_report_date_overlap": overlaps,
        "final_holdout_not_used_by_current_selector": True,
        "pristine_unobserved_holdout": False,
        "forward_sample_trades": 0,
        "note": "Chronological holdout is excluded from current selection. Historical dates may have been seen in earlier research; this is not forward trading evidence."
    }
    report["live_approved"] = False
    return report


def trade_csv(report):
    output = io.StringIO(newline="")
    fields = ["run_id", "symbol", "signal_ms", "entry_ms", "settlement_ms", "direction", "entry", "exit",
              "outcome", "net_units", "split", "fold_id"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({"run_id": report["provenance"]["run_id"], **trade} for trade in report["trade_ledger"])
    return output.getvalue()


def save_report(report, output, trades_output):
    # Fully serialize/validate before replacing either artifact.
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    csv_payload = trade_csv(report)
    for path, content in ((trades_output, csv_payload), (output, payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        temporary.replace(path)
    history = output.parent / ".runs"
    history.mkdir(parents=True, exist_ok=True)
    record = {"provenance": report["provenance"], "validation_checks": report["validation_checks"],
              "walk_forward": report["walk_forward"]["aggregate"],
              "final_holdout": report["final_holdout"]["test"], "publishable": report["publishable"]}
    (history / f"{report['provenance']['run_id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--days", type=int, default=360)
    result.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date() - timedelta(days=2),
                        help="Exclusive UTC end date; default two days ago to allow archive publication")
    result.add_argument("--train-days", type=int, default=90)
    result.add_argument("--validation-days", type=int, default=30)
    result.add_argument("--holdout-days", type=int, default=60)
    result.add_argument("--payout", type=float, default=.8, help="Net profit per stake on a win, a fixed scenario not historical odds")
    result.add_argument("--fee", type=float, default=.01, help="Conservative cost per stake, including refunded trades")
    result.add_argument("--target-win-rate", type=float, default=.58)
    result.add_argument("--latency-minutes", type=int, default=1, help="Delayed minute-open execution proxy, not exchange next-second replay")
    result.add_argument("--horizon-minutes", type=int, default=30)
    result.add_argument("--adverse-bps", type=float, default=0)
    result.add_argument("--min-train-trades", type=int, default=100)
    result.add_argument("--min-test-trades", type=int, default=100)
    result.add_argument("--initial-bankroll", type=float, default=1000)
    result.add_argument("--stake", type=float, default=1, help="Fixed research stake in units; no compounding or martingale")
    result.add_argument("--max-open-positions", type=int, default=1)
    result.add_argument("--daily-loss-limit", type=float, default=5)
    result.add_argument("--cache-dir", type=Path, default=Path("research/.cache"))
    result.add_argument("--offline", action="store_true", help="Use verified cached index archives only")
    result.add_argument("--output", type=Path, default=Path("research/event-results.json"))
    result.add_argument("--trades-output", type=Path, default=Path("research/event-trades.csv"))
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        report = run_backtest(args)
        save_report(report, args.output, args.trades_output)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        cli.exit(1, f"Event research failed; existing report preserved: {error}\n")
    print(json.dumps({"run_id": report["provenance"]["run_id"], "publishable": report["publishable"],
                      "walk_forward": report["walk_forward"]["aggregate"],
                      "final_holdout": report["final_holdout"]["test"],
                      "validation_checks": report["validation_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
