"""Train/export and load the fixed BTC/ETH 30-minute research alert model.

The bundle is produced locally by this project. Its SHA-256 manifest detects
corruption, not an untrusted publisher; never load someone else's joblib bundle.
No account credentials, broker API, or order submission is supported.
"""
from __future__ import annotations

import argparse
from datetime import date,datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (BASE_FEATURES,PAIR_FEATURES,FEATURES,MINUTE_MS,SYMBOLS,
                       base_frame,enrich_pair_frames,build_latest_features)

DAY_MS=86_400_000
RULE_ID="both_regularized_logistic"
MODEL_PARAMETERS={"C":.1,"max_iter":400,"solver":"lbfgs","random_state":17}


def _ms(day):
    return int(datetime.combine(day,datetime.min.time(),timezone.utc).timestamp()*1000)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decision(p_up,features,signal_ms):
    """Apply the predeclared .56/exhaustion gate; scores are uncalibrated."""
    p_up=float(p_up)
    if not math.isfinite(p_up) or not 0<=p_up<=1:
        raise ValueError("Invalid directional model score")
    if isinstance(signal_ms,bool) or int(signal_ms)!=signal_ms or signal_ms%MINUTE_MS:
        raise ValueError("Signal must be an integer minute boundary")
    ret=float(features["ret_60"]);sigma=float(features["rv_60"]);ratio=float(features["rv_ratio"])
    if not np.isfinite([ret,sigma,ratio]).all() or sigma<0:
        raise ValueError("Invalid gate inputs")
    side="LONG" if p_up>=.5 else "SHORT"
    direction=1 if side=="LONG" else -1
    score=max(p_up,1-p_up)
    normalized=ret/(max(sigma,1e-8)*math.sqrt(60))
    reason="signal"
    if score<.56:
        reason="model_score_below_0.56"
    elif direction!=-np.sign(ret):
        reason="model_does_not_oppose_60m_move"
    elif abs(normalized)<1:
        reason="60m_move_not_extreme"
    elif ratio>1.25:
        reason="short_volatility_expanding"
    if reason!="signal":
        side="NONE"
    return {"side":side,"direction":direction if side!="NONE" else 0,
            "p_up":p_up,"confidence_score":score,"score_is_calibrated_win_probability":False,
            "reason":reason,"signal_ms":int(signal_ms),"feature_close_ms":int(signal_ms)-MINUTE_MS,
            "entry_due_ms":int(signal_ms)+MINUTE_MS,"expiry_due_ms":int(signal_ms)+31*MINUTE_MS,
            "horizon_minutes":30,"rule_id":RULE_ID,"research_only":True,
            "order_execution_enabled":False}


class Predictor:
    def __init__(self,model_bundle_directory,max_model_age_days=35):
        import joblib
        import sklearn
        self.directory=Path(model_bundle_directory)
        self.manifest=json.loads((self.directory/"manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("schema_version")!=1 or self.manifest.get("features")!=FEATURES
                or self.manifest.get("rule_id")!=RULE_ID or self.manifest.get("order_execution_enabled") is not False):
            raise ValueError("Model manifest does not match the fixed 60-feature research contract")
        if self.manifest.get("sklearn_version")!=sklearn.__version__:
            raise ValueError("Install the exact scikit-learn version specified by the model manifest")
        self.models={}
        self.max_model_age_days=max_model_age_days
        for symbol in SYMBOLS:
            entry=self.manifest["models"][symbol]
            if entry["file"]!=f"{symbol}.joblib":
                raise ValueError("Unexpected model filename")
            path=self.directory/entry["file"]
            if _sha(path)!=entry["sha256"]:
                raise ValueError("Model checksum mismatch")
            model=joblib.load(path)
            if getattr(model,"n_features_in_",None)!=60:
                raise ValueError("Model expects a different feature vector")
            self.models[symbol]=model

    def predict(self,features_by_symbol,signal_ms):
        from threadpoolctl import threadpool_limits
        if set(features_by_symbol)!=set(SYMBOLS):
            raise ValueError("Both BTCUSDT and ETHUSDT feature rows are required")
        cutoff=int(self.manifest["training_cutoff_ms"])
        if signal_ms<cutoff:
            raise ValueError("A model cannot predict before its training cutoff")
        if self.max_model_age_days is not None and signal_ms-cutoff>self.max_model_age_days*DAY_MS:
            raise ValueError("Model exceeds its declared maximum age; retrain before alerts resume")
        result=[]
        for symbol in SYMBOLS:
            row=features_by_symbol[symbol]
            if set(row)!=set(FEATURES):
                raise ValueError("Feature names differ from the frozen model order")
            vector=np.asarray([[row[name] for name in FEATURES]],dtype=float)
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite prediction feature vector")
            with threadpool_limits(limits=2):
                p=float(self.models[symbol].predict_proba(vector)[0,1])
            output=decision(p,row,signal_ms)
            output.update(symbol=symbol,model_training_cutoff_ms=cutoff,
                          model_version=self.manifest["model_version"])
            result.append(output)
        return result

    def predict_market(self,spot_by_symbol,premium_by_symbol,funding_by_symbol,signal_ms):
        rows=build_latest_features(spot_by_symbol,premium_by_symbol,funding_by_symbol,signal_ms)
        return self.predict(rows,signal_ms)


def train_export(output_dir,end=date(2026,11,12),*,offline=True,cache_dir=None,progress=print):
    """Fit last180 days of settled labels using only public, verified archives.

    This path does not import/load the private feature repository. Downloading is
    opt-in through offline=False; serving only needs this bundle and live data.
    """
    import joblib
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from research.qd_three_year_data import load_market
    from research.derivatives_data import load_premium,load_funding
    from research.derivative_features import add_derivative_features
    end=date.fromisoformat(end) if isinstance(end,str) else end
    if end>datetime.now(timezone.utc).date():
        raise ValueError("Training must end on a completed UTC date")
    cutoff=_ms(end)
    start=end-timedelta(days=180)
    warmup=start-timedelta(days=3)
    root=Path(__file__).resolve().parents[1]
    cache=Path(cache_dir or root/"research"/".cache")
    samples={};manifests={}
    digest=hashlib.sha256()
    for symbol in SYMBOLS:
        progress(f"{symbol}: load verified public 183-day spot/index history")
        spot,index,sources=load_market(symbol,warmup,end,cache,offline)
        frame=base_frame(spot)
        frame["signal_ms"]=frame.timestamp+MINUTE_MS
        frame=frame[(frame.signal_ms%(30*MINUTE_MS)==0)&(frame.signal_ms>=_ms(start))].copy()
        frame=frame.dropna(subset=BASE_FEATURES).reset_index(drop=True)
        frame["entry_ms"]=frame.signal_ms+MINUTE_MS
        frame["settlement_ms"]=frame.signal_ms+31*MINUTE_MS
        frame=frame[frame.settlement_ms<cutoff].copy()
        opens=index.set_index("timestamp")["open"]
        frame["entry"]=opens.loc[frame.entry_ms].to_numpy()
        frame["exit"]=opens.loc[frame.settlement_ms].to_numpy()
        frame["y"]=(frame.exit>frame.entry).astype(int)
        frame["is_flat"]=frame.exit==frame.entry
        frame["symbol"]=symbol
        frame.attrs["features"]=BASE_FEATURES
        samples[symbol]=frame
        manifests[symbol]={"spot_index":sources}
        del spot,index
    samples=enrich_pair_frames(samples)
    for symbol in SYMBOLS:
        premium,p_sources=load_premium(symbol,warmup,end,cache,offline,allow_gaps=True)
        funding,f_sources=load_funding(symbol,warmup,end,cache,offline)
        samples[symbol]=add_derivative_features(samples[symbol],premium,funding,allow_premium_gaps=True)
        if samples[symbol].attrs["features"]!=FEATURES:
            raise ValueError("Training feature order differs from serving features")
        manifests[symbol].update(premium=p_sources,funding=f_sources)
    return retrain_from_rows(output_dir,samples,cutoff,progress,data_sources=manifests)


def retrain_from_rows(output_dir,rows_by_symbol,cutoff_ms,progress=print,*,data_sources=None):
    """Small-memory monthly refresh from seed + every settled live feature row.

    Minute observations may be supplied, but training deliberately retains the
    original half-hour sampling grid. Mixing a sparse historical seed with
    dense new minute rows would otherwise change the training distribution.
    All minute observations can remain in FeatureHistory for later research.

    The caller should write a new directory and switch the serving pointer only
    after this function and Predictor verification both succeed. Alert-only rows
    are not a representative training population; collect NONE decisions too.
    """
    import joblib
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    if set(rows_by_symbol)!=set(SYMBOLS):
        raise ValueError("Require exactly BTCUSDT/ETHUSDT training populations")
    if (isinstance(cutoff_ms,bool) or int(cutoff_ms)!=cutoff_ms
            or cutoff_ms%(30*MINUTE_MS)):
        raise ValueError("Training cutoff must be an integer half-hour boundary")
    cutoff=int(cutoff_ms)
    start_ms=cutoff-180*DAY_MS
    end=datetime.fromtimestamp(cutoff/1000,timezone.utc).date()
    start=datetime.fromtimestamp(start_ms/1000,timezone.utc).date()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    models={}
    for symbol in SYMBOLS:
        incoming=pd.DataFrame(rows_by_symbol[symbol])
        columns=FEATURES+["signal_ms","settlement_ms","y","is_flat"]
        if set(columns)-set(incoming.columns):
            raise ValueError("Missing declared training feature/label columns")
        d=incoming.loc[(incoming.signal_ms>=start_ms)&(incoming.signal_ms<cutoff)
                       &(incoming.settlement_ms<cutoff),columns].copy().sort_values("signal_ms").reset_index(drop=True)
        if (d.empty or d.signal_ms.duplicated().any() or np.any(d.signal_ms.to_numpy()%MINUTE_MS)
                or not (d.settlement_ms==d.signal_ms+31*MINUTE_MS).all()):
            raise ValueError("Training event times must be unique delayed 30-minute events")
        # Preserve the population used by the deployed coefficients and the
        # paired cadence comparison; dense minute observations are not weights.
        d=d[d.signal_ms%(30*MINUTE_MS)==0].copy().reset_index(drop=True)
        if (not d.y.isin([0,1]).all() or not d.is_flat.isin([False,True]).all()
                or not np.isfinite(d[FEATURES].to_numpy(dtype=float)).all()):
            raise ValueError("Invalid training features or settled directional labels")
        d["signal_ms"]=d.signal_ms.astype(np.int64)
        d["settlement_ms"]=d.settlement_ms.astype(np.int64)
        d["y"]=d.y.astype(np.int8)
        d["is_flat"]=d.is_flat.astype(bool)
        train=d[~d.is_flat].copy()
        if len(train)<2000 or train.y.nunique()!=2:
            raise ValueError("Insufficient decisive training labels")
        latest=int(train.settlement_ms.max())
        if latest>=cutoff:
            raise ValueError("Training label crosses fit cutoff")
        matrix=train[FEATURES].to_numpy()
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite training values")
        model=make_pipeline(StandardScaler(),LogisticRegression(**MODEL_PARAMETERS))
        with threadpool_limits(limits=2):
            model.fit(matrix,train.y.to_numpy(dtype=int))
        path=out/f"{symbol}.joblib"
        joblib.dump(model,path,compress=3)
        seed_path=out/f"{symbol}-training.csv.gz"
        d.to_csv(seed_path,index=False,compression={"method":"gzip","compresslevel":6,"mtime":0})
        digest.update(symbol.encode());digest.update(pd.util.hash_pandas_object(train,index=False).to_numpy(dtype="<u8").tobytes())
        models[symbol]={"file":path.name,"sha256":_sha(path),"training_count":len(train),
                        "latest_settlement_ms":latest,"iterations":int(model[-1].n_iter_[0]),
                        "training_samples_file":seed_path.name,"training_samples_sha256":_sha(seed_path),
                        "training_samples_rows":len(d)}
        progress(f"{symbol}: exported {len(train)} strictly settled labels")
    source=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name("features.py").read_bytes()).hexdigest()
    identity={"training_cutoff_ms":cutoff,"data_sha256":digest.hexdigest(),"source_sha256":source,
              "models":{symbol:models[symbol]["sha256"] for symbol in SYMBOLS}}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    manifest={"schema_version":1,"rule_id":RULE_ID,"features":FEATURES,
              "model_parameters":MODEL_PARAMETERS,"model_version":version,
              "training_start_utc":str(start),"training_cutoff_utc":str(end),"training_cutoff_ms":cutoff,
              "training_days":180,"default_max_model_age_days":35,"sklearn_version":sklearn.__version__,
              "training_sampling_minutes":30,"supported_inference_sampling_minutes":[1,30],
              "models":models,"source_sha256":source,"data_sha256":identity["data_sha256"],
              "public_data_manifests":data_sources or {},"generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "score_is_calibrated_win_probability":False,"research_only":True,
              "historical_stable_58pct_demonstrated":False,"order_execution_enabled":False}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return manifest


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir",type=Path,required=True)
    cli.add_argument("--end",type=date.fromisoformat,default=datetime.now(timezone.utc).date())
    cli.add_argument("--online",action="store_true",help="permit official historical downloads; default offline")
    args=cli.parse_args()
    result=train_export(args.output_dir,args.end,offline=not args.online)
    print(json.dumps({"model_version":result["model_version"],"output_dir":str(args.output_dir),"training_cutoff_utc":result["training_cutoff_utc"]}))


if __name__=="__main__":
    main()
