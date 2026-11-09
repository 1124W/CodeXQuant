"""CLI for reproducible Binance 30m research. Run from the repository root."""

import argparse
import json
import zipfile
from datetime import date
from pathlib import Path

if __package__:
    from .engine import PAYOUT, run
else:
    from engine import PAYOUT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--end", type=date.fromisoformat,
                        help="Exclusive UTC end date (YYYY-MM-DD); defaults to yesterday for archive lag")
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--payout", type=float, default=PAYOUT)
    parser.add_argument("--fee", type=float, default=0, help="Fixed cost per unit stake per trade")
    parser.add_argument("--cache-dir", type=Path, default=Path("research/.cache"))
    parser.add_argument("--output", type=Path, default=Path("research/results.json"))
    args = parser.parse_args()
    try:
        report = run(args.days, args.end, args.holdout_days, args.payout, args.fee, args.cache_dir)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Research failed; existing report preserved: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({k: report[k] for k in ("start_utc", "end_exclusive_utc", "bars", "publishable", "chosen")}))


if __name__ == "__main__":
    main()
