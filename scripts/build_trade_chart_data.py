"""Build an audited, compact index-candle and execution dataset for 442 trades.

Uses only verified local official history. Does not refit or select a strategy.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from research.event_engine import utc_ms
from research.event_strategy_research import build_raw, evaluate
from research.qd_three_year_data import load_kind


START = date(2026, 8, 9)
END = date(2026, 11, 9)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
OUTPUT = Path("C:/Users/23977/.codex/visualizations/2026/11/08/01a1bbb3-59be-7fb2-88ac-902805232dec/trade-442-data.json")


def trade_key(trade):
    return (trade["symbol"], int(trade["signal_ms"]), int(trade["entry_ms"]),
            int(trade["settlement_ms"]), int(trade["direction"]))


def main():
    first, last = utc_ms(START), utc_ms(END)
    report_path = ROOT / "research" / "derivative-study-results.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    selected = report["selected_rule"]
    if selected["id"] != "both_regularized_logistic":
        raise ValueError("The requested 56.79% cohort belongs to a different rule")
    trades = [row for row in report["trade_ledger"]
              if row["signal_ms"] >= first and row["settlement_ms"] < last]
    trades.sort(key=lambda row: (row["entry_ms"], row["symbol"]))
    assert len(trades) == 442
    assert sum(row["outcome"] == "win" for row in trades) == 251
    assert sum(row["outcome"] == "loss" for row in trades) == 191
    assert math.isclose(math.fsum(row["net_units"] for row in trades), 111.75)

    # Independently regenerate this period's accepted ledger from ALL raw model
    # predictions, including portfolio capacity that earlier rejected trades used.
    predictions = {}
    prediction_hashes = {}
    needed = ["signal_ms", "entry_ms", "settlement_ms", "entry", "exit", "ret_15",
              "ret_60", "ret_240", "rv_60", "rv_240", "rv_ratio",
              "model_fit_cutoff_ms", "train_latest_settlement_ms", "p_" + selected["model"]]
    for symbol in SYMBOLS:
        path = ROOT / "research" / ".cache" / "derivative-study" / f"{symbol}-predictions.csv.gz"
        prediction_hashes[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, usecols=needed)
        predictions[symbol] = frame[(frame.signal_ms >= first) & (frame.settlement_ms < last)].copy()
    raw = build_raw(predictions, [selected])[selected["id"]]
    independently_accepted, stats = evaluate(raw, first, last)
    independently_accepted.sort(key=lambda row: (row["entry_ms"], row["symbol"]))
    assert [trade_key(row) for row in independently_accepted] == [trade_key(row) for row in trades]
    for a, b in zip(independently_accepted, trades):
        for column in ("entry", "exit", "net_units"):
            assert math.isclose(float(a[column]), float(b[column]), rel_tol=0, abs_tol=1e-8)
        assert a["outcome"] == b["outcome"]
    assert stats["signals"] == 442 and stats["wins"] == 251 and stats["losses"] == 191
    assert stats["net_units"] == 111.75

    candles, source_hashes, per_symbol = {}, {}, {}
    minute_verifications = 0
    for symbol in SYMBOLS:
        minutes, manifest = load_kind("index", symbol, START, END, ROOT / "research" / ".cache", True)
        minute_opens = minutes.set_index("timestamp")["open"]
        symbol_trades = [row for row in trades if row["symbol"] == symbol]
        for row in symbol_trades:
            assert row["settlement_ms"] - row["entry_ms"] == 30 * 60_000
            assert row["entry_ms"] - row["signal_ms"] == 60_000
            assert row["entry_ms"] in minute_opens.index and row["settlement_ms"] in minute_opens.index
            assert math.isclose(float(minute_opens.loc[row["entry_ms"]]), row["entry"], rel_tol=0, abs_tol=1e-8)
            assert math.isclose(float(minute_opens.loc[row["settlement_ms"]]), row["exit"], rel_tol=0, abs_tol=1e-8)
            actual_change = (row["exit"] - row["entry"]) * row["direction"]
            assert row["outcome"] == ("win" if actual_change > 0 else "loss" if actual_change < 0 else "refund")
            assert row["net_units"] == (4.25 if row["outcome"] == "win" else -5 if row["outcome"] == "loss" else 0)
            minute_verifications += 2
        grouped = minutes.groupby(minutes.timestamp // 1_800_000, sort=True).agg(
            timestamp=("timestamp", "first"), open=("open", "first"),
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            observations=("timestamp", "size"))
        assert (grouped.observations == 30).all()
        assert len(grouped) == 4416
        candles[symbol] = [[int(row.timestamp // 1000),
                            round(float(row.open), 2), round(float(row.high), 2),
                            round(float(row.low), 2), round(float(row.close), 2)]
                           for row in grouped.itertuples()]
        source_hashes[symbol] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        per_symbol[symbol] = {
            "trades": len(symbol_trades),
            "wins": sum(row["outcome"] == "win" for row in symbol_trades),
            "losses": sum(row["outcome"] == "loss" for row in symbol_trades),
            "net_units": round(math.fsum(row["net_units"] for row in symbol_trades), 8),
            "candles": len(candles[symbol]), "source_minutes": len(minutes),
        }

    columns = ("symbol", "entry_ms", "settlement_ms", "signal_ms", "entry", "exit",
               "direction", "outcome", "net_units")
    compact_trades = [{"id": number, **{name: row[name] for name in columns}}
                      for number, row in enumerate(trades, 1)]
    output = {
        "meta": {
            "start_utc": str(START), "end_exclusive_utc": str(END),
            "start_ms": first, "end_exclusive_ms": last,
            "source_run_id": report["provenance"]["run_id"],
            "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "source_prediction_sha256": prediction_hashes,
            "source_index_manifest_sha256": source_hashes,
            "rule_id": selected["id"], "trades": 442, "wins": 251, "losses": 191,
            "win_rate": 251 / 442, "net_units": 111.75, "max_drawdown_units": stats["max_drawdown_units"],
            "payout": .85, "fee": 0, "stake": 5, "candle_minutes": 30,
            "execution_price_source": "official index minute OPEN at exact execution timestamp",
            "candle_rounding_decimals": 2, "trade_price_precision": "original unrounded report values",
            "entry_exit_observations_verified": minute_verifications,
            "ledger_independently_regenerated": True,
            "per_symbol": per_symbol,
        },
        "candles": candles,
        "trades": compact_trades,
    }
    payload = json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    OUTPUT.write_text(payload, encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "bytes": len(payload.encode()), "meta": output["meta"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
