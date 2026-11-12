"""Eight frozen feature ablations with newly downloaded public Binance history.

Acquisition is separate: scripts/update_binance_research_data.py. This command
only consumes verified local archives and never places orders.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .derivative_features import PREMIUM_FEATURES, FUNDING_FEATURES, add_derivative_features
from .derivatives_data import load_premium, load_funding
from .event_engine import DAY_MS, MINUTE_MS, _by_symbol, _equity, _monthly, utc_ms
from .event_regime_features import enrich_samples
from .event_strategy_research import (CAPITAL, MODEL_CONFIGS, build_raw, canonical,
    choose_on_discovery, evaluate, make_model, moving_week_interval, stats_for_ledger)
from .qd_demo import write_atomic
from .qd_three_year_data import load_market
from .workbuddy_strategy import build_samples, _auc

ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=("BTCUSDT","ETHUSDT")
START=date(2023,11,9)
OLD_END=date(2026,11,9)
DISCOVERY=date(2024,5,7)
SPLIT=date(2025,5,7)
CONFIRM=date(2026,8,9)
MODELS=("regularized_gbm","regularized_logistic")
BLOCKS=("base50","premium","funding","both")


def log(value):
    print(value,flush=True)


def source_hash():
    files=("derivative_research.py","derivative_features.py","derivatives_data.py",
           "event_strategy_research.py","event_regime_features.py","event_engine.py",
           "workbuddy_strategy.py","qd_three_year_data.py","qd_factors.py")
    return hashlib.sha256(b"".join(name.encode()+(ROOT/"research"/name).read_bytes() for name in files)).hexdigest()


def rules():
    return [{"id":f"{block}_{model}","block":block,"base_model":model,
             "model":f"{block}_{model}","family":"model","regime":"exhaustion","threshold":.56}
            for block in BLOCKS for model in MODELS]


def feature_blocks(frame):
    base=[name for name in frame.attrs["features"] if name not in PREMIUM_FEATURES+FUNDING_FEATURES]
    if len(base)!=50:
        raise ValueError("Expected the frozen 50-feature baseline")
    return {"base50":base,"premium":base+PREMIUM_FEATURES,"funding":base+FUNDING_FEATURES,
            "both":base+PREMIUM_FEATURES+FUNDING_FEATURES}


def predict(frame,end,*,factories=None,progress=log):
    """Same cutoff schedule as prior study, including the newly extended last fold."""
    from threadpoolctl import threadpool_limits
    groups=feature_blocks(frame)
    d=frame.sort_values("signal_ms").reset_index(drop=True)
    if d.signal_ms.duplicated().any() or not np.isfinite(d[frame.attrs["features"]].to_numpy()).all():
        raise ValueError("Nonfinite or duplicate training population")
    if not (d.entry_ms==d.signal_ms+MINUTE_MS).all() or not (d.settlement_ms==d.entry_ms+30*MINUTE_MS).all():
        raise ValueError("Only one-minute-delayed 30-minute events are supported")
    cutoff=utc_ms(START)+180*DAY_MS
    keep=["symbol","signal_ms","entry_ms","settlement_ms","entry","exit","y","is_flat",
          "ret_15","ret_60","ret_240","rv_60","rv_240","rv_ratio",
          "premium_observation_ms","premium_close_available_ms","funding_observation_ms","funding_available_ms"]
    parts,folds=[],[]
    while cutoff<end:
        stop=min(cutoff+30*DAY_MS,end)
        train=d[(d.signal_ms>=cutoff-180*DAY_MS)&(d.signal_ms<cutoff)&(d.settlement_ms<cutoff)&~d.is_flat]
        test=d[(d.signal_ms>=cutoff)&(d.signal_ms<stop)&(d.settlement_ms<end)]
        if len(train)<2000 or train.y.nunique()<2 or test.empty:
            raise ValueError("Insufficient settled training population")
        latest=int(train.settlement_ms.max())
        if latest>=cutoff:
            raise ValueError("Training label not yet settled")
        predicted=test[keep].copy()
        for rule in rules():
            model=factories[rule["id"]]() if factories else make_model(rule["base_model"])
            with threadpool_limits(limits=2):
                model.fit(train[groups[rule["block"]]].to_numpy(),train.y.to_numpy(dtype=int))
                p=np.asarray(model.predict_proba(test[groups[rule["block"]]].to_numpy()))[:,1]
            if len(p)!=len(test) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                raise ValueError("Invalid raw model score")
            predicted[f"p_{rule['id']}"]=p
        predicted["model_fit_cutoff_ms"]=cutoff
        predicted["train_latest_settlement_ms"]=latest
        parts.append(predicted)
        folds.append({"symbol":str(d.symbol.iloc[0]),"fit_cutoff_ms":cutoff,
                      "train_latest_settlement_ms":latest,"test_end_ms":stop,
                      "n_train":len(train),"n_predict":len(test),"model_fits":8})
        if progress:
            progress(f"{d.symbol.iloc[0]} {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()}: train={len(train)}, predict={len(test)}, eight ablations")
        cutoff=stop
    return pd.concat(parts,ignore_index=True),folds


def score_metrics(predictions,rule,start,end):
    frames=[frame[(frame.signal_ms>=start)&(frame.settlement_ms<end)&~frame.is_flat] for frame in predictions.values()]
    frame=pd.concat(frames,ignore_index=True)
    if frame.empty:
        return {"n":0,"auc":None,"brier_raw_score":None}
    p=frame[f"p_{rule['id']}"].to_numpy()
    y=frame.y.to_numpy(dtype=int)
    return {"n":len(frame),"auc":_auc(y,p),"brier_raw_score":float(np.mean((p-y)**2))}


def make_plan(end):
    return {"version":1,"data_start_utc":str(START),"end_exclusive_utc":str(end),
            "discovery_start_utc":str(DISCOVERY),"discovery_end_exclusive_utc":str(SPLIT),
            "validation_end_exclusive_utc":str(CONFIRM),"historical_confirmation_end_exclusive_utc":str(OLD_END),
            "new_history_start_utc":str(OLD_END),"train_days":180,"refit_days":30,
            "capital":CAPITAL,"candidate_count":8,"rules":rules(),
            "model_parameters":{name:MODEL_CONFIGS[name] for name in MODELS},
            "premium_features":PREMIUM_FEATURES,"funding_features":FUNDING_FEATURES,
            "trade_gate":"score>=.56; oppose ret60; abs(ret60)/(rv60*sqrt60)>=1; rv_ratio<=1.25",
            "selection":"discovery only, min200 decisive trades; Bonferroni-Wilson lower-score; lexical tie break",
            "target_win_rate":.58,"min_validation_trades":400,"min_confirmation_trades":100,
            "min_new_history_trades":100,"min_new_history_days":30,
            "publication_lag_ms":{"premium":60000,"funding":60000},
            "funding_max_age_hours":24,"premium_gap_policy":"no fill; omit any 60-minute historical window crossing missing observations",
            "feature_ablation_population":"same complete observed-feature population for all eight candidates",
            "new_history_is_retrospective":True,"pristine_holdout":False,"forward_trades":0,
            "source_sha256":source_hash()}


def prepare(end):
    cache=ROOT/"research"/".cache"
    frames,sources,quality={},{},{}
    for symbol in SYMBOLS:
        log(f"{symbol}: offline verify updated spot/index minutes")
        spot,index,manifest=load_market(symbol,START-timedelta(days=3),end,cache,True)
        frames[symbol]=build_samples(spot,index,symbol,profile="research")
        quality[symbol]={"spot_minutes":len(spot),"index_minutes":len(index)}
        sources[symbol]={"spot_and_index":manifest}
        del spot,index
        gc.collect()
    frames=enrich_samples(frames)
    for symbol in SYMBOLS:
        log(f"{symbol}: offline verify premium and funding history, causally align")
        premium,premium_sources=load_premium(symbol,START-timedelta(days=3),end,cache,True,allow_gaps=True)
        funding,funding_sources=load_funding(symbol,START-timedelta(days=3),end,cache,True)
        sources[symbol].update(premium=premium_sources,funding=funding_sources)
        quality[symbol].update(premium_observed_minutes=len(premium),funding_events=len(funding),
            funding_latest_time_ms=int(funding.funding_time.iloc[-1]),
            funding_largest_observed_gap_hours=float(funding.funding_time.diff().max()/3_600_000))
        original=len(frames[symbol])
        frames[symbol]=add_derivative_features(frames[symbol],premium,funding,allow_premium_gaps=True)
        quality[symbol].update(half_hour_population_before=original,half_hour_population_after=len(frames[symbol]),
                               feature_metadata=frames[symbol].attrs["derivative_features"])
        del premium,funding
        gc.collect()
    return frames,sources,quality


def run(end):
    started=time.perf_counter()
    if end<=OLD_END or end>datetime.now(timezone.utc).date():
        raise ValueError("Require newly downloaded complete UTC days after 2026-11-09")
    out=ROOT/"research"
    plan=make_plan(end)
    plan_sha=hashlib.sha256(canonical(plan).encode()).hexdigest()
    write_atomic(out/"derivative-study-plan.json",json.dumps(plan,indent=2)+"\n")
    frames,sources,quality=prepare(end)
    digest=hashlib.sha256()
    predictions,folds={},[]
    cache=out/".cache"/"derivative-study"
    cache.mkdir(parents=True,exist_ok=True)
    feature_names={}
    for symbol,frame in frames.items():
        feature_names[symbol]=feature_blocks(frame)
        digest.update(symbol.encode())
        digest.update(pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype="<u8").tobytes())
        predictions[symbol],current=predict(frame,utc_ms(end))
        predictions[symbol].to_csv(cache/f"{symbol}-predictions.csv.gz",index=False,compression="gzip")
        folds.extend(current)
    begin,split,confirm,old_end,finish=map(utc_ms,(DISCOVERY,SPLIT,CONFIRM,OLD_END,end))
    raw=build_raw(predictions,rules())
    selected,ranking=choose_on_discovery(raw,rules(),begin,split)
    selection={"plan_sha256":plan_sha,"selected_rule":selected,"discovery_ranking":ranking,
               "locked_at_utc":datetime.now(timezone.utc).isoformat(),"selection_cutoff_ms":split}
    write_atomic(out/"derivative-study-selection.json",json.dumps(selection,indent=2,allow_nan=False)+"\n")
    chosen=raw[selected["id"]] if selected else []
    _,discovery=evaluate(chosen,begin,split,bootstrap=True)
    validation_ledger,validation=evaluate(chosen,split,confirm,bootstrap=True)
    confirmation_ledger,confirmation=evaluate(chosen,confirm,old_end,bootstrap=True)
    new_ledger,new_stats=evaluate(chosen,old_end,finish,bootstrap=True)
    ledger,total=evaluate(chosen,split,finish,bootstrap=True)
    ablations=[]
    for rule in rules():
        measurements={}
        for name,start,stop in (("validation",split,confirm),("historical_confirmation",confirm,old_end),
                                 ("new_history",old_end,finish)):
            _,measurements[name]=evaluate(raw[rule["id"]],start,stop)
            measurements[name]["prediction_metrics"]=score_metrics(predictions,rule,start,stop)
        ablations.append({"rule":rule,**measurements})
    stresses=[]
    for name,override in (("fee_1pct",{"fee":.01}),("adverse_entry_2bps",{"adverse_bps":2.}),
                          ("payout_80pct",{"payout":.8})):
        _,stats=evaluate(chosen,split,finish,config={**CAPITAL,**override})
        stresses.append({"name":name,"test":stats,"configuration":{**CAPITAL,**override}})
    best=next((r for r in ranking if selected and r["rule"]["id"]==selected["id"]),None)
    be=1/1.85
    week={"post_selection":moving_week_interval(ledger,split,finish),
          "validation":moving_week_interval(validation_ledger,split,confirm),
          "historical_confirmation":moving_week_interval(confirmation_ledger,confirm,old_end),
          "new_history":{"status":"insufficient_calendar_days"} if (end-OLD_END).days<28 else moving_week_interval(new_ledger,old_end,finish)}
    checks={"discovery_adjusted_score_above_break_even":bool(best and best["multiplicity_adjusted_lower_95"]>be),
            "validation_min400":validation["decisive_trades"]>=400,
            "validation_target58":(validation["win_rate"] or 0)>=.58,
            "confirmation_min100":confirmation["decisive_trades"]>=100,
            "confirmation_target58":(confirmation["win_rate"] or 0)>=.58,
            "validation_positive":validation["net_units"]>0,"confirmation_positive":confirmation["net_units"]>0,
            "post_selection_week_net_lower_positive":(week["post_selection"]["net_lower_95"] or 0)>0,
            "new_history_min100":new_stats["decisive_trades"]>=100,
            "new_history_min30_days":(end-OLD_END).days>=30,
            "new_history_target58":(new_stats["win_rate"] or 0)>=.58,
            "new_history_positive":new_stats["net_units"]>0,
            "training_labels_settled":all(f["train_latest_settlement_ms"]<f["fit_cutoff_ms"] for f in folds),
            "premium_available_before_signal":all((p.premium_close_available_ms<=p.signal_ms).all() for p in predictions.values()),
            "funding_available_before_signal":all((p.funding_available_ms<=p.signal_ms).all() for p in predictions.values())}
    actual_source=source_hash()
    if actual_source!=plan["source_sha256"]:
        raise RuntimeError("Source changed during execution; refusing inconsistent report")
    identity={"source_sha256":actual_source,"plan_sha256":plan_sha,"data_sha256":digest.hexdigest()}
    report={"schema_version":1,"engine":"derivative-feature-research-v1","plan":plan,
            "selected_rule":selected,"discovery_ranking":ranking,"discovery":discovery,
            "validation":validation,"historical_confirmation":confirmation,"new_history":new_stats,
            "post_selection":total,"new_history_continuing_capital":stats_for_ledger(ledger,old_end,finish),
            "feature_ablation":ablations,"feature_columns":feature_names,"data_quality":quality,"sources":sources,
            "monthly":_monthly(ledger,split,finish,CAPITAL),"by_symbol":_by_symbol(ledger,split,finish,CAPITAL),
            "moving_week_bootstrap":week,"cost_scenarios":stresses,"validation_checks":checks,
            "historical_criteria_met":all(checks.values()),"forward_validated":False,"live_approved":False,
            "trade_ledger":ledger,"new_history_trade_ledger":new_ledger,
            "equity_curve":_equity(ledger,CAPITAL["initial_bankroll"],"post_selection"),"folds":folds,
            "provenance":{**identity,"run_id":hashlib.sha256(canonical(identity).encode()).hexdigest()[:20],
                          "generated_at_utc":datetime.now(timezone.utc).isoformat(),
                          "runtime_seconds":round(time.perf_counter()-started,2)},
            "limitations":["All old history was previously researched; new days are retrospectively downloaded, not timestamped live signals.",
                "Eight feature ablations share one complete-observation population. Missing official premium minutes are not interpolated; affected trailing60-minute windows are skipped.",
                "Realized funding is delayed60 seconds, not a prediction of future funding. Funding is a feature, not an event-contract cash flow.",
                "Model scores are uncalibrated. Brier/AUC summarize directional scores, not certainty of profitable bets.",
                "Bonferroni-Wilson is a discovery selection heuristic; dependent outcomes and repeated prior studies weaken nominal confidence.",
                "Independent segments restart1000USDT capital; post-selection ledger is continuous from2025-05-07.",
                "Minute entry/expiry and constant85% net payout are proxies; actual second-level execution and historical event quotes are unavailable.",
                "No stable-strategy conclusion from fewer than100 new trades or fewer than30 new calendar days; no actual forward validation has occurred."]}
    write_atomic(out/"derivative-study-results.json",json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    stream=io.StringIO(newline="")
    fields=["symbol","rule_id","signal_ms","entry_ms","settlement_ms","direction","entry","exit","outcome","stake","net_units","horizon_minutes","regime","model_fit_cutoff_ms","train_latest_settlement_ms"]
    writer=csv.DictWriter(stream,fields,extrasaction="ignore");writer.writeheader();writer.writerows(ledger)
    write_atomic(out/"derivative-study-trades.csv",stream.getvalue())
    log(json.dumps({"selected_rule":selected,"validation":validation,"historical_confirmation":confirmation,
                    "new_history":new_stats,"post_selection":total,"checks":checks},ensure_ascii=False))
    log(f"Finished {report['provenance']['run_id']} in {time.perf_counter()-started:.1f}s")
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--end",type=date.fromisoformat,default=datetime.now(timezone.utc).date())
    args=cli.parse_args()
    run(args.end)


if __name__=="__main__":
    main()
