"""Finite, chronological BTC/ETH 30-minute strategy research, separate from baseline.

Run offline with Python 3.12. Raw model scores are not calibrated win probabilities.
The historical confirmation period was previously observed in older research.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .event_engine import (DAY_MS, MINUTE_MS, KNOWN_UNAVAILABLE_DATES,
                           _by_symbol, _equity, _monthly, event_stats,
                           simulate_portfolio, utc_ms)
from .qd_demo import write_atomic

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
MODEL_CONFIGS = {
    "regularized_gbm": {"max_iter": 120, "max_depth": 2, "learning_rate": .04,
                        "min_samples_leaf": 300, "l2_regularization": 10.,
                        "early_stopping": False, "random_state": 17},
    "workbuddy_gbm": {"max_iter": 200, "max_depth": 4, "learning_rate": .06,
                     "min_samples_leaf": 80, "l2_regularization": 1.,
                     "early_stopping": False, "random_state": 7},
    "regularized_logistic": {"C": .1, "max_iter": 400, "solver": "lbfgs",
                             "random_state": 17},
}
CAPITAL = {"payout": .85, "fee": 0., "initial_bankroll": 1000., "stake": 5.,
           "max_open_positions": 1, "daily_loss_limit": 25., "adverse_bps": 0.}
PLAN = {
    "version": 1, "data_start_utc": "2023-11-09", "end_exclusive_utc": "2026-11-09",
    "train_days": 180, "refit_days": 30,
    "discovery_start_utc": "2024-05-07", "discovery_end_exclusive_utc": "2025-05-07",
    "validation_end_exclusive_utc": "2026-08-09",
    "confirmation_end_exclusive_utc": "2026-11-09",
    "confidence_scores": [.56, .60, .64],
    "regimes": ["unrestricted", "exhaustion", "trend_pullback"],
    "direct_normalized_thresholds": [1., 1.5, 2.],
    "model_configs": MODEL_CONFIGS, "capital": CAPITAL,
    "selection": "maximum Bonferroni-Wilson lower bound on discovery only; minimum 200 decisive accepted trades; lexical tie break",
    "min_discovery_trades": 200, "min_validation_trades": 400,
    "min_confirmation_trades": 100, "target_win_rate": .58,
    "post_selection_changes_permitted": False,
    "historical_data_previously_observed": True, "pristine_holdout": False,
    "forward_trades": 0,
}


def log(message):
    print(message, flush=True)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_hash():
    names = ("event_strategy_research.py", "event_regime_features.py", "event_engine.py",
             "workbuddy_strategy.py", "qd_three_year_data.py", "qd_factors.py")
    return sha_bytes(b"".join(name.encode() + (ROOT / "research" / name).read_bytes() for name in names))


def candidates():
    rules = []
    for model in MODEL_CONFIGS:
        for threshold in PLAN["confidence_scores"]:
            for regime in PLAN["regimes"]:
                rules.append({"id": f"{model}_{threshold:.2f}_{regime}", "model": model,
                              "threshold": threshold, "regime": regime, "family": "model"})
    for family in ("direct_exhaustion", "direct_trend_pullback"):
        for threshold in PLAN["direct_normalized_thresholds"]:
            rules.append({"id": f"{family}_{threshold:.1f}", "family": family,
                          "threshold": threshold})
    return rules


def wilson_lower(wins, count, z):
    if count <= 0:
        return None
    p = wins / count
    return ((p + z*z/(2*count) - z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)))
            / (1 + z*z/count))


def period_raw(raw, start, end):
    return [t for t in raw if t["signal_ms"] >= start and t["settlement_ms"] < end]


def evaluate(raw, start, end, *, bootstrap=False, config=None):
    ledger = simulate_portfolio(period_raw(raw, start, end), **(config or CAPITAL))
    return ledger, event_stats(ledger, start, end, payout=(config or CAPITAL)["payout"],
                              fee=(config or CAPITAL)["fee"], bootstrap=bootstrap)


def choose_on_discovery(raw_by_rule, rules, start, end):
    """Never inspect prices/labels at or beyond the discovery settlement boundary."""
    z = NormalDist().inv_cdf(1 - .05 / (2 * len(rules)))
    ranking = []
    for rule in rules:
        _, stats = evaluate(raw_by_rule[rule["id"]], start, end)
        n = stats["decisive_trades"]
        eligible = n >= PLAN["min_discovery_trades"]
        lower = wilson_lower(stats["wins"], n, z) if eligible else None
        ranking.append({"rule": rule, "discovery": stats, "eligible": eligible,
                        "multiplicity_adjusted_lower_95": lower})
    valid = [row for row in ranking if row["eligible"]]
    valid.sort(key=lambda row: (-row["multiplicity_adjusted_lower_95"], row["rule"]["id"]))
    return (valid[0]["rule"] if valid else None), ranking


def make_model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "regularized_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(**MODEL_CONFIGS[name]))
    return HistGradientBoostingClassifier(**MODEL_CONFIGS[name])


def rolling_predictions(samples, first, end, *, factories=None, progress=log):
    """Training-only transforms and strictly settled labels; all prediction flats retained."""
    from threadpoolctl import threadpool_limits
    features = samples.attrs["features"]
    d = samples.sort_values("signal_ms").reset_index(drop=True)
    if d.signal_ms.duplicated().any() or not np.isfinite(d[features].to_numpy()).all():
        raise ValueError("Features must be finite and timestamps unique")
    if (not (d.entry_ms == d.signal_ms + MINUTE_MS).all()
            or not (d.settlement_ms == d.entry_ms + 30 * MINUTE_MS).all()):
        raise ValueError("Only delayed 30-minute events are supported")
    parts, folds = [], []
    cutoff = first + PLAN["train_days"] * DAY_MS
    while cutoff < end:
        stop = min(cutoff + PLAN["refit_days"] * DAY_MS, end)
        train = d[(d.signal_ms >= cutoff - PLAN["train_days"] * DAY_MS)
                  & (d.signal_ms < cutoff) & (d.settlement_ms < cutoff) & ~d.is_flat]
        test = d[(d.signal_ms >= cutoff) & (d.signal_ms < stop) & (d.settlement_ms < end)].copy()
        if len(train) < 2000 or train.y.nunique() < 2 or test.empty:
            raise ValueError("Insufficient training/test population")
        latest = int(train.settlement_ms.max())
        if latest >= cutoff:
            raise ValueError("Unsettled training label")
        for name in MODEL_CONFIGS:
            model = factories[name]() if factories else make_model(name)
            with threadpool_limits(limits=2):
                model.fit(train[features].to_numpy(), train.y.to_numpy(dtype=int))
                probabilities = np.asarray(model.predict_proba(test[features].to_numpy()))[:, 1]
            if (len(probabilities) != len(test) or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError("Invalid model scores")
            test[f"p_{name}"] = probabilities
        test["model_fit_cutoff_ms"] = cutoff
        test["train_latest_settlement_ms"] = latest
        parts.append(test)
        folds.append({"symbol": str(d.symbol.iloc[0]), "fit_cutoff_ms": cutoff,
                      "train_latest_settlement_ms": latest, "test_end_ms": stop,
                      "training_count": len(train), "prediction_count": len(test)})
        if progress:
            progress(f"{d.symbol.iloc[0]} {datetime.fromtimestamp(cutoff/1000, timezone.utc).date()}: {len(train)} train / {len(test)} predict / 3 models")
        cutoff = stop
    return pd.concat(parts, ignore_index=True), folds


def rule_directions(frame, rule):
    r60 = frame.ret_60.to_numpy()
    rv60 = np.maximum(frame.rv_60.to_numpy(), 1e-8)
    z60 = r60 / (rv60 * math.sqrt(60))
    z15 = frame.ret_15.to_numpy() / (rv60 * math.sqrt(15))
    z240 = frame.ret_240.to_numpy() / (np.maximum(frame.rv_240.to_numpy(), 1e-8) * math.sqrt(240))
    reversal = -np.sign(r60)
    trend = np.sign(frame.ret_240.to_numpy())
    calm = frame.rv_ratio.to_numpy() <= 1.25
    pullback = ((np.sign(frame.ret_15.to_numpy()) == -trend)
                & (np.abs(z15) >= .5) & (frame.rv_ratio.to_numpy() <= 1.5))
    if rule["family"] == "direct_exhaustion":
        return np.where((np.abs(z60) >= rule["threshold"]) & calm, reversal, 0).astype(int)
    if rule["family"] == "direct_trend_pullback":
        return np.where((np.abs(z240) >= rule["threshold"]) & pullback, trend, 0).astype(int)
    p = frame[f"p_{rule['model']}"].to_numpy()
    direction = np.where(p >= .5, 1, -1)
    mask = np.maximum(p, 1-p) >= rule["threshold"]
    if rule["regime"] == "exhaustion":
        mask &= (direction == reversal) & (np.abs(z60) >= 1) & calm
    elif rule["regime"] == "trend_pullback":
        mask &= (direction == trend) & (np.abs(z240) >= 1) & pullback
    return np.where(mask, direction, 0).astype(int)


def build_raw(predictions, rules):
    unavailable = {utc_ms(date.fromisoformat(day)) // DAY_MS for day in KNOWN_UNAVAILABLE_DATES}
    output = {rule["id"]: [] for rule in rules}
    for symbol, frame in predictions.items():
        records = frame.to_dict("records")
        for rule in rules:
            directions = rule_directions(frame, rule)
            for row, direction in zip(records, directions):
                if not direction or any(day in unavailable for day in range(int(row["signal_ms"])//DAY_MS, int(row["settlement_ms"])//DAY_MS+1)):
                    continue
                z240 = abs(row["ret_240"]) / (max(row["rv_240"], 1e-8)*math.sqrt(240))
                regime = "volatility_shock" if row["rv_ratio"] > 1.25 else "trend" if z240 >= 1 else "range"
                trade = {k: row[k] for k in ("signal_ms", "entry_ms", "settlement_ms", "entry", "exit",
                                             "model_fit_cutoff_ms", "train_latest_settlement_ms")}
                trade.update(symbol=symbol, direction=int(direction), rule_id=rule["id"],
                             horizon_minutes=30, timeframe_minutes=1, regime=regime)
                output[rule["id"]].append(trade)
    return output


def prepare_samples(cache_dir):
    from .event_regime_features import enrich_samples
    from .qd_three_year_data import load_market
    from .workbuddy_strategy import build_samples
    frames, manifests = {}, {}
    for symbol in SYMBOLS:
        log(f"{symbol}: verifying cached 3-year minute archives and building historical features")
        spot, index, manifests[symbol] = load_market(symbol, date(2023, 11, 6), date(2026, 11, 9), cache_dir, True)
        frames[symbol] = build_samples(spot, index, symbol, profile="research")
        del spot, index
    return enrich_samples(frames), manifests


def stats_for_ledger(ledger, start, end):
    return event_stats([t for t in ledger if start <= t["signal_ms"] and t["settlement_ms"] < end],
                       start, end, payout=.85, fee=0., bootstrap=False)


def moving_week_interval(ledger, start, end, iterations=2000):
    """Circular 7-day resampling; BTC/ETH share daily buckets and block draws."""
    days = max(1, math.ceil((end-start)/DAY_MS))
    buckets = np.zeros((days, 3), dtype=float)
    for trade in ledger:
        day = (trade["settlement_ms"]-start)//DAY_MS
        if not 0 <= day < days:
            raise ValueError("Settlement outside bootstrap interval")
        buckets[day] += (trade["outcome"] == "win", trade["outcome"] != "refund", trade["net_units"])
    if not buckets[:,1].sum():
        return {"block_days":7,"iterations":iterations,"win_rate_lower_95":None,
                "win_rate_upper_95":None,"net_lower_95":None,"net_upper_95":None}
    generator=np.random.default_rng(1729)
    starts=generator.integers(0,days,size=(iterations,math.ceil(days/7)))
    indices=((starts[:,:,None]+np.arange(7))%days).reshape(iterations,-1)[:,:days]
    totals=buckets[indices].sum(axis=1)
    rates=np.divide(totals[:,0],totals[:,1],out=np.zeros(iterations),where=totals[:,1]>0)
    return {"block_days":7,"iterations":iterations,"win_rate_lower_95":float(np.quantile(rates,.025)),
            "win_rate_upper_95":float(np.quantile(rates,.975)),
            "net_lower_95":float(np.quantile(totals[:,2],.025)),
            "net_upper_95":float(np.quantile(totals[:,2],.975))}


def run(output_dir):
    started = time.perf_counter()
    rules = candidates()
    frozen_plan = {**PLAN, "rules": rules, "source_sha256": source_hash()}
    plan_hash = sha_bytes(canonical(frozen_plan).encode())
    # Write a concrete plan before preparing data, training, or inspecting any results.
    write_atomic(output_dir / "event-strategy-plan.json", json.dumps(frozen_plan, indent=2)+"\n")
    cache = ROOT / "research" / ".cache" / "event-strategy"
    cache.mkdir(parents=True, exist_ok=True)
    samples, manifests = prepare_samples(ROOT / "research" / ".cache")
    dataset_hash = hashlib.sha256()
    predictions, folds = {}, []
    first, end = utc_ms(date(2023,11,9)), utc_ms(date(2026,11,9))
    for symbol, frame in samples.items():
        dataset_hash.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
        frame.to_csv(cache / f"{symbol}-samples.csv.gz", index=False, compression="gzip")
        predictions[symbol], current = rolling_predictions(frame, first, end)
        predictions[symbol].to_csv(cache / f"{symbol}-predictions.csv.gz", index=False, compression="gzip")
        folds.extend(current)
    log("33 candidates: selecting using discovery period only")
    raw = build_raw(predictions, rules)
    begin = utc_ms(date.fromisoformat(PLAN["discovery_start_utc"]))
    split = utc_ms(date.fromisoformat(PLAN["discovery_end_exclusive_utc"]))
    confirm = utc_ms(date.fromisoformat(PLAN["validation_end_exclusive_utc"]))
    selected, ranking = choose_on_discovery(raw, rules, begin, split)
    selection_record = {"plan_sha256": plan_hash, "selected_rule": selected,
                        "selection_cutoff_ms": split,
                        "ranking": ranking, "locked_at_utc": datetime.now(timezone.utc).isoformat()}
    write_atomic(output_dir / "event-strategy-selection.json", json.dumps(selection_record, indent=2, allow_nan=False)+"\n")
    # No later result is read for candidate selection. From here the rule cannot change.
    chosen = raw[selected["id"]] if selected else []
    ledger, total = evaluate(chosen, split, end, bootstrap=True)
    discovery_ledger, discovery = evaluate(chosen, begin, split, bootstrap=True)
    # Fresh capital at each independent evaluation segment is explicit.
    validation_ledger, validation = evaluate(chosen, split, confirm, bootstrap=True)
    confirmation_ledger, confirmation = evaluate(chosen, confirm, end, bootstrap=True)
    continuation_recent = stats_for_ledger(ledger, confirm, end)
    weekly_intervals = {
        "post_selection":moving_week_interval(ledger,split,end),
        "validation":moving_week_interval(validation_ledger,split,confirm),
        "historical_confirmation":moving_week_interval(confirmation_ledger,confirm,end),
    }
    yearly = []
    for year in (2025, 2026):
        lo, hi = max(split, utc_ms(date(year,1,1))), min(end, utc_ms(date(year+1,1,1)))
        yearly.append({"year": str(year), **stats_for_ledger(ledger,lo,hi)})
    monthly = _monthly(ledger, split, end, CAPITAL)
    by_symbol = _by_symbol(ledger, split, end, CAPITAL)
    regimes = {name: event_stats([t for t in ledger if t["regime"] == name], split, end,
                                 payout=.85, fee=0, bootstrap=False)
               for name in ("range", "trend", "volatility_shock")}
    stresses = []
    for name, override in (("fee_1pct", {"fee":.01}), ("fee_2pct", {"fee":.02}),
                           ("adverse_entry_2bps", {"adverse_bps":2.}), ("payout_80pct", {"payout":.8})):
        _, stats = evaluate(chosen,split,end,config={**CAPITAL, **override})
        stresses.append({"name":name,"configuration":{**CAPITAL,**override},"test":stats})
    best = next((r for r in ranking if selected and r["rule"]["id"] == selected["id"]), None)
    be = 1/1.85
    checks = {
        "discovery_multiplicity_adjusted_lower_above_break_even": bool(best and best["multiplicity_adjusted_lower_95"] > be),
        "validation_400_trades": validation["decisive_trades"] >= PLAN["min_validation_trades"],
        "validation_58pct": (validation["win_rate"] or 0) >= .58,
        "confirmation_100_trades": confirmation["decisive_trades"] >= PLAN["min_confirmation_trades"],
        "confirmation_58pct": (confirmation["win_rate"] or 0) >= .58,
        "validation_positive": validation["net_units"] > 0,
        "confirmation_positive": confirmation["net_units"] > 0,
        "confirmation_wilson_above_break_even": (confirmation["win_rate_lower_95"] or 0) > be,
        "post_selection_daily_bootstrap_net_lower_positive": (total["block_bootstrap_net_lower_95"] or 0) > 0,
        "post_selection_weekly_bootstrap_net_lower_positive": (weekly_intervals["post_selection"]["net_lower_95"] or 0) > 0,
        "each_post_selection_year_profitable": all(row["net_units"] > 0 for row in yearly),
        "each_symbol_100_trades_and_positive": all(by_symbol[s]["decisive_trades"] >= 100 and by_symbol[s]["net_units"] > 0 for s in SYMBOLS),
        "fee_1pct_stress_positive": stresses[0]["test"]["net_units"] > 0,
        "training_strictly_purged": all(f["train_latest_settlement_ms"] < f["fit_cutoff_ms"] for f in folds),
    }
    identity = {"plan_sha256":plan_hash,"data_sha256":dataset_hash.hexdigest(),"source_sha256":source_hash()}
    if identity["source_sha256"] != frozen_plan["source_sha256"]:
        raise RuntimeError("Research source changed during execution; refusing inconsistent provenance")
    report = {"schema_version":1,"engine":"event-strategy-research-v1","plan":frozen_plan,
              "provenance":{**identity,"run_id":sha_bytes(canonical(identity).encode())[:20],
                            "generated_at_utc":datetime.now(timezone.utc).isoformat(),
                            "runtime_seconds":round(time.perf_counter()-started,2)},
              "candidate_count":len(rules),"selected_rule":selected,"discovery_ranking":ranking,
              "discovery":discovery,"validation":validation,"historical_confirmation":confirmation,
              "post_selection":total,"confirmation_with_continuing_capital":continuation_recent,
              "yearly":yearly,"monthly":monthly,"by_symbol":by_symbol,"by_market_state":regimes,
              "cost_scenarios":stresses,"validation_checks":checks,
              "moving_week_bootstrap":weekly_intervals,
              "historical_stability_criteria_met":all(checks.values()),
              "forward_validated":False,"live_approved":False,
              "trade_ledger":ledger,"validation_ledger":validation_ledger,"confirmation_ledger":confirmation_ledger,
              "equity_curve":_equity(ledger,CAPITAL["initial_bankroll"],"post_selection"),
              "folds":folds,"sources":manifests,
              "limitations":["All history was previously accessible/observed; confirmation is not pristine forward evidence.",
                 "Only discovery chooses among 33 fixed candidates; later results cannot alter this selection.",
                 "Model scores are uncalibrated and do not state true trade win probabilities.",
                 "Bonferroni-Wilson is a candidate ranking score; serially dependent outcomes weaken its nominal coverage.",
                 "Minutes proxy next-second entry/expiry; actual contract quotes/payout history are unavailable.",
                 "Capital restarts at 1000 USDT for independent discovery/validation/confirmation; full post-selection ledger is continuous.",
                 "Joint BTC/ETH circular 7-day bootstrap complements independent-day intervals; neither corrects all prior research or long-range dependence.",
                 "85% payout and zero extra fee are fixed user scenarios, not historical quote observations."]}
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    write_atomic(output_dir / "event-strategy-results.json",payload)
    stream=io.StringIO(newline="")
    fields=["symbol","rule_id","signal_ms","entry_ms","settlement_ms","direction","entry","exit",
            "outcome","stake","net_units","horizon_minutes","regime","model_fit_cutoff_ms","train_latest_settlement_ms"]
    writer=csv.DictWriter(stream,fields,extrasaction="ignore")
    writer.writeheader();writer.writerows(ledger)
    write_atomic(output_dir / "event-strategy-trades.csv",stream.getvalue())
    log(json.dumps({"selected_rule":selected,"discovery":discovery,"validation":validation,
                    "confirmation":confirmation,"post_selection":total,"checks":checks},ensure_ascii=False))
    log(f"Finished in {time.perf_counter()-started:.1f}s; original WorkBuddy baseline untouched")
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir",type=Path,default=ROOT/"research")
    args=cli.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
