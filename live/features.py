"""Self-contained causal feature formulas for the frozen 60-input alert model.

No private repository, serialized research function, future label, or index
execution price is loaded to build a prediction. Spot/funding/premium timestamps
must use exchange milliseconds and inputs must include the last completed bar.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

MINUTE_MS=60_000
HOUR_MS=3_600_000
SYMBOLS=("BTCUSDT","ETHUSDT")
BASE_FEATURES=["ret_1","ret_2","ret_3","ret_5","ret_10","ret_15","ret_30","ret_60",
    "imb","imb_ma5","imb_ma15","imb_ma60","imb_std15","imb_std60",
    "vol_z15","vol_z60","cnt_z15","cnt_z60","ticket_z60","rv_15","rv_60",
    "rv_ratio","clv","amihud_z60","hour_sin","hour_cos","ret_5_smooth","ret_15_smooth",
    "ret_120","ret_240","ret_480","rv_240","rv_ratio_long","dist_hi","dist_lo","streak","dow_sin","dow_cos"]
PAIR_FEATURES=["norm_ret_15","norm_ret_30","norm_ret_60","trend_efficiency_60",
    "range_position_120","flow_momentum_15","flow_slope_5_60","volatility_acceleration",
    "relative_norm_ret_60","peer_ret_60","peer_imb_ma15","peer_rv_ratio"]
PREMIUM_FEATURES=["premium_close_bps","premium_mean15_bps","premium_mean60_bps",
    "premium_deviation60_bps","premium_slope15_60_bps","premium_std60_bps"]
FUNDING_FEATURES=["funding_rate_bps","funding_delta_bps","funding_mean3_bps","funding_age_hours"]
FEATURES=BASE_FEATURES+PAIR_FEATURES+PREMIUM_FEATURES+FUNDING_FEATURES
SPOT_COLUMNS=["timestamp","open","high","low","close","volume","quote_volume","trades","taker_buy_base"]


def _times(values,label):
    result=pd.to_numeric(values,errors="raise").to_numpy()
    if (not np.isfinite(result).all() or np.any(result!=result.astype(np.int64))):
        raise ValueError(f"Invalid {label} millisecond timestamps")
    return result.astype(np.int64)


def base_frame(spot):
    """38 historical indicators, deliberately without a future-target expression."""
    if set(SPOT_COLUMNS)-set(spot.columns):
        raise ValueError("Missing spot OHLC/order-flow fields")
    d=spot[SPOT_COLUMNS].copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"]=_times(d.timestamp,"spot")
    if (d.empty or np.any(d.timestamp.to_numpy()%MINUTE_MS)
            or np.any(np.diff(d.timestamp.to_numpy())!=MINUTE_MS)):
        raise ValueError("Spot minutes must be ordered, unique, continuous and clock aligned")
    for name in SPOT_COLUMNS[1:]:
        d[name]=pd.to_numeric(d[name],errors="raise").astype(float)
    raw=d[SPOT_COLUMNS[1:]].to_numpy()
    if (not np.isfinite(raw).all() or np.any(d[["open","high","low","close"]].to_numpy()<=0)
            or (d.high<np.maximum(d.open,d.close)).any() or (d.low>np.minimum(d.open,d.close)).any()
            or (d[["volume","quote_volume","trades","taker_buy_base"]]<0).any().any()):
        raise ValueError("Invalid spot OHLC/order-flow observations")
    log_close=np.log(d.close)
    out=pd.DataFrame({"timestamp":d.timestamp})
    for period in (1,2,3,5,10,15,30,60,120,240,480):
        out[f"ret_{period}"]=log_close.diff(period)*10_000
    out["imb"]=(2*d.taker_buy_base-d.volume)/d.volume.replace(0,np.nan)
    for period in (5,15,60):
        out[f"imb_ma{period}"]=out.imb.rolling(period).mean()
        out[f"imb_std{period}"]=out.imb.rolling(period).std(ddof=1)
    for period in (15,60):
        for column,prefix in (("volume","vol"),("trades","cnt")):
            rolling=d[column].rolling(period)
            out[f"{prefix}_z{period}"]=(d[column]-rolling.mean())/rolling.std(ddof=1).replace(0,np.nan)
    ticket=d.quote_volume/d.trades.replace(0,np.nan)
    out["ticket_z60"]=(ticket-ticket.rolling(60).mean())/ticket.rolling(60).std(ddof=1).replace(0,np.nan)
    for period in (15,60,240):
        out[f"rv_{period}"]=out.ret_1.rolling(period).std(ddof=1)
    out["rv_ratio"]=out.rv_15/out.rv_60.replace(0,np.nan)
    out["rv_ratio_long"]=out.rv_60/out.rv_240.replace(0,np.nan)
    out["clv"]=(d.close-d.low)/(d.high-d.low).replace(0,np.nan)-.5
    impact=(out.ret_1.abs()/(d.quote_volume/1_000_000)).replace([np.inf,-np.inf],np.nan)
    out["amihud_z60"]=(impact-impact.rolling(60).mean())/impact.rolling(60).std(ddof=1).replace(0,np.nan)
    stamp=pd.to_datetime(d.timestamp,unit="ms",utc=True)
    hour=stamp.dt.hour+stamp.dt.minute/60
    out["hour_sin"]=np.sin(2*np.pi*hour/24)
    out["hour_cos"]=np.cos(2*np.pi*hour/24)
    for period in (5,15):
        out[f"ret_{period}_smooth"]=out.ret_1.rolling(period).mean()
    out["dist_hi"]=(d.close/d.high.rolling(120).max()-1)*10_000
    out["dist_lo"]=(d.close/d.low.rolling(120).min()-1)*10_000
    rising=(out.ret_1>0).to_numpy()
    positions=np.arange(len(d))
    starts=np.maximum.accumulate(np.where(np.r_[True,rising[1:]!=rising[:-1]],positions,0))
    out["streak"]=(positions-starts+1)*np.where(rising,1,-1)
    out["dow_sin"]=np.sin(2*np.pi*stamp.dt.dayofweek/7)
    out["dow_cos"]=np.cos(2*np.pi*stamp.dt.dayofweek/7)
    return out[["timestamp"]+BASE_FEATURES]


def enrich_pair_frames(frames):
    """Append the frozen 12 cross-asset/regime features using equal signal times."""
    if set(frames)!=set(SYMBOLS):
        raise ValueError("Both BTCUSDT and ETHUSDT are required")
    indexed={symbol:frame.set_index("signal_ms").sort_index() for symbol,frame in frames.items()}
    if any(frame.index.duplicated().any() for frame in indexed.values()):
        raise ValueError("Duplicate signal timestamps")
    common=indexed[SYMBOLS[0]].index.intersection(indexed[SYMBOLS[1]].index)
    output={}
    for symbol in SYMBOLS:
        d=indexed[symbol].loc[common].copy()
        peer=indexed[SYMBOLS[1] if symbol==SYMBOLS[0] else SYMBOLS[0]].loc[common]
        sigma=d.rv_60.clip(lower=1e-6)
        for period in (15,30,60):
            d[f"norm_ret_{period}"]=d[f"ret_{period}"]/(sigma*math.sqrt(period))
        d["trend_efficiency_60"]=d.norm_ret_60.abs()
        width=d.dist_lo-d.dist_hi
        d["range_position_120"]=d.dist_lo.div(width.where(width.abs()>1e-6)).fillna(.5)-.5
        d["flow_momentum_15"]=d.imb_ma15*d.norm_ret_15
        d["flow_slope_5_60"]=d.imb_ma5-d.imb_ma60
        d["volatility_acceleration"]=d.rv_ratio-d.rv_ratio_long
        pair_sigma=np.sqrt(sigma**2+peer.rv_60.clip(lower=1e-6)**2)
        d["relative_norm_ret_60"]=(d.ret_60-peer.ret_60)/(pair_sigma*math.sqrt(60))
        d["peer_ret_60"]=peer.ret_60
        d["peer_imb_ma15"]=peer.imb_ma15
        d["peer_rv_ratio"]=peer.rv_ratio
        d=d.reset_index()
        d.attrs["features"]=BASE_FEATURES+PAIR_FEATURES
        output[symbol]=d
    return output


def _derivative_values(premium,funding,signal_ms):
    if set(["timestamp","open","high","low","close"])-set(premium):
        raise ValueError("Missing premium OHLC")
    p=premium.loc[(premium.timestamp>=signal_ms-60*MINUTE_MS)&(premium.timestamp<signal_ms)].copy().sort_values("timestamp")
    times=_times(p.timestamp,"premium")
    if not np.array_equal(times,np.arange(signal_ms-60*MINUTE_MS,signal_ms,MINUTE_MS)):
        raise ValueError("Incomplete trailing 60 premium minutes; no filling allowed")
    for name in ("open","high","low","close"):
        p[name]=pd.to_numeric(p[name],errors="raise").astype(float)
    if (not np.isfinite(p[["open","high","low","close"]].to_numpy()).all()
            or (p.high<np.maximum(p.open,p.close)).any() or (p.low>np.minimum(p.open,p.close)).any()):
        raise ValueError("Invalid signed premium OHLC")
    value=p.close.to_numpy()*10_000
    m15=float(np.mean(value[-15:]));m60=float(np.mean(value))
    answer=dict(zip(PREMIUM_FEATURES,[float(value[-1]),m15,m60,float(value[-1])-m60,m15-m60,float(np.std(value,ddof=1))]))
    if set(["funding_time","rate"])-set(funding):
        raise ValueError("Missing realized funding observations")
    f=funding.loc[funding.funding_time+MINUTE_MS<=signal_ms,["funding_time","rate"]].copy().sort_values("funding_time")
    if len(f)<3:
        raise ValueError("At least three published realized funding events are required")
    funding_times=_times(f.funding_time,"funding")
    if np.any(np.diff(funding_times)<=0):
        raise ValueError("Duplicate funding observations")
    rates=pd.to_numeric(f.rate,errors="raise").to_numpy(dtype=float)*10_000
    age=(signal_ms-int(funding_times[-1])-MINUTE_MS)/HOUR_MS
    if not np.isfinite(rates).all() or not 0<=age<=24:
        raise ValueError("Realized funding is invalid or more than 24 hours stale")
    answer.update(dict(zip(FUNDING_FEATURES,[float(rates[-1]),float(rates[-1]-rates[-2]),float(np.mean(rates[-3:])),float(age)])))
    return answer


def build_latest_features(spot_by_symbol,premium_by_symbol,funding_by_symbol,signal_ms):
    """Build one feature row per symbol after any completed minute.

    Indicator lookbacks are measured in one-minute bars, independently of how
    frequently the model is evaluated. No unfinished minute enters the vector.
    """
    if isinstance(signal_ms,bool) or int(signal_ms)!=signal_ms or signal_ms%MINUTE_MS:
        raise ValueError("Signal must be an integer minute millisecond boundary")
    signal_ms=int(signal_ms)
    if any(set(mapping)!=set(SYMBOLS) for mapping in (spot_by_symbol,premium_by_symbol,funding_by_symbol)):
        raise ValueError("Exact BTCUSDT/ETHUSDT inputs are required")
    compact={}
    for symbol in SYMBOLS:
        spot=spot_by_symbol[symbol]
        closed=spot.loc[spot.timestamp+MINUTE_MS<=signal_ms].copy()
        frame=base_frame(closed)
        if len(frame)<600 or int(frame.timestamp.iloc[-1])!=signal_ms-MINUTE_MS:
            raise ValueError(f"{symbol}: require >=600 completed spot minutes ending at signal-1m")
        if abs(int(frame.streak.iloc[-1]))>=len(frame)-1:
            raise ValueError("Current streak has no observed start in the supplied history")
        frame=frame.iloc[[-1]].copy().rename(columns={"timestamp":"last_bar_ms"})
        frame.insert(0,"signal_ms",signal_ms)
        compact[symbol]=frame
    enriched=enrich_pair_frames(compact)
    answer={}
    for symbol in SYMBOLS:
        row={name:float(enriched[symbol][name].iloc[0]) for name in BASE_FEATURES+PAIR_FEATURES}
        row.update(_derivative_values(premium_by_symbol[symbol],funding_by_symbol[symbol],signal_ms))
        if not np.isfinite([row[name] for name in FEATURES]).all():
            raise ValueError(f"{symbol}: nonfinite historical input; no prediction")
        answer[symbol]=row
    return answer
