"""Eight frozen open-interest/positioning features from verified public metrics.

Record time plus ten minutes is a conservative research availability assumption,
not evidence of original publication time or an unrevised point-in-time feed.
The latest available observation expires after five minutes. No missing sample
is interpolated, carried beyond that interval, or substituted for an exact lag.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd

from live.features import FEATURES
from .event_engine import MINUTE_MS,SYMBOLS

GRID_MS=5*MINUTE_MS
AVAILABILITY_DELAY_MS=10*MINUTE_MS
MAX_AVAILABLE_AGE_MS=5*MINUTE_MS
OI_LAGS=(30,60,240)
RAW_COLUMNS=('sum_open_interest','sum_open_interest_value',
    'count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio',
    'count_long_short_ratio','sum_taker_long_short_vol_ratio')
METRIC_FEATURES=['metrics_oi_log_change_30','metrics_oi_log_change_60','metrics_oi_log_change_240',
    'metrics_log_top_account_ratio','metrics_log_top_position_ratio',
    'metrics_log_global_account_ratio','metrics_log_taker_buy_sell_ratio',
    'metrics_oi_price_interaction_60']
AUGMENTED_FEATURES=FEATURES+METRIC_FEATURES
PROVENANCE_COLUMNS=['metrics_record_ms','metrics_available_ms','metrics_age_ms']


def specification():
    return deepcopy({'features':METRIC_FEATURES,'base_features':FEATURES,
        'record_grid_minutes':5,'availability_delay_minutes':10,'maximum_available_age_minutes_exclusive':5,
        'raw_record_age_minutes':'[10,15)','oi_log_change_lags_minutes':list(OI_LAGS),
        'oi_change':'log(current_sum_open_interest)-log(exact_prior_sum_open_interest)',
        'ratio_features':'natural logs of current top-account,top-position,global-account,taker-buy/sell ratios',
        'interaction':'metrics_oi_log_change_60 multiplied by current completed-minute norm_ret_60',
        'notional_interest':'quality check only; excluded to avoid mechanical price duplication',
        'missing_policy':'No interpolation; no stale carry; exact positive observed lag endpoints; intermediate stock gaps need not invalidate endpoints.',
        'duplicate_policy':'Identical observations collapse; every conflicting asset/timestamp group is dropped.',
        'availability_is_publication_time_proof':False,'local_only':True})


def _clean_metrics(metrics):
    required=['record_ms','symbol']+list(RAW_COLUMNS)
    if not isinstance(metrics,pd.DataFrame) or set(required)-set(metrics):
        raise ValueError('Missing official metrics schema')
    if not metrics.symbol.isin(SYMBOLS).all():raise ValueError('Unsupported metrics asset')
    values=metrics[required].copy()
    for name in ['record_ms']+list(RAW_COLUMNS):values[name]=pd.to_numeric(values[name],errors='coerce')
    identical=int(values.duplicated(required).sum())
    values=values.drop_duplicates(required)
    conflict=values.duplicated(['symbol','record_ms'],keep=False)
    conflicting=int(conflict.sum());values=values.loc[~conflict].copy()
    times=values.record_ms.to_numpy(dtype=float);numbers=values[list(RAW_COLUMNS)].to_numpy(dtype=float)
    valid=(np.isfinite(times)&(times>0)&(times==np.floor(times))&(times%GRID_MS==0)
           &np.isfinite(numbers).all(axis=1)&(numbers>0).all(axis=1))
    invalid=int((~valid).sum());values=values.loc[valid]
    values['record_ms']=values.record_ms.astype(np.int64)
    return values.sort_values(['symbol','record_ms']).reset_index(drop=True),{
        'invalid_metrics_rows_dropped':invalid,'identical_metrics_duplicates_collapsed':identical,
        'conflicting_metrics_rows_dropped':conflicting}


def augment_frame(frame,metrics):
    """Lazy causal merge, preserving retained base-row order/index and columns."""
    if not isinstance(frame,pd.DataFrame) or set(['signal_ms','symbol','norm_ret_60'])-set(frame):
        raise ValueError('Require symbol, completed-minute signal and norm_ret_60')
    if not frame.symbol.isin(SYMBOLS).all():raise ValueError('Only BTC/ETH are supported')
    times=pd.to_numeric(frame.signal_ms,errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(times).all() or np.any(times!=np.floor(times)) or np.any(times<=0)
            or np.any(times%MINUTE_MS) or frame.duplicated(['symbol','signal_ms']).any()):
        raise ValueError('Require unique asset/minute base observations')
    stamps=times.astype(np.int64);clean,cleaning=_clean_metrics(metrics)
    result=frame.copy();new=np.full((len(frame),len(METRIC_FEATURES)),np.nan)
    chosen_record=np.full(len(frame),-1,dtype=np.int64)
    chosen_available=np.full(len(frame),-1,dtype=np.int64)
    ages=np.full(len(frame),-1,dtype=np.int64)
    timely=np.zeros(len(frame),dtype=bool)
    symbols=frame.symbol.to_numpy()
    for symbol in SYMBOLS:
        positions=np.flatnonzero(symbols==symbol)
        observed=clean.loc[clean.symbol==symbol]
        if not len(positions) or observed.empty:continue
        recorded=observed.record_ms.to_numpy(dtype=np.int64)
        available=recorded+AVAILABILITY_DELAY_MS
        raw=observed[list(RAW_COLUMNS)].to_numpy(dtype=float)
        log_interest=np.log(raw[:,0])
        features=np.full((len(observed),7),np.nan)
        lookup=pd.Index(recorded)
        for column,lag in enumerate(OI_LAGS):
            prior=lookup.get_indexer(recorded-lag*MINUTE_MS)
            exists=prior>=0
            features[exists,column]=log_interest[exists]-log_interest[prior[exists]]
        features[:,3:]=np.log(raw[:,2:])
        loc=np.searchsorted(available,stamps[positions],side='right')-1
        exists=loc>=0;safe=np.maximum(loc,0)
        age=stamps[positions]-available[safe]
        valid_time=exists&(age>=0)&(age<MAX_AVAILABLE_AGE_MS)
        timely[positions]=valid_time
        chosen_record[positions]=np.where(exists,recorded[safe],-1)
        chosen_available[positions]=np.where(exists,available[safe],-1)
        ages[positions]=np.where(exists,age,-1)
        new[positions,:7]=features[safe]
    normalized=pd.to_numeric(frame.norm_ret_60,errors='coerce').to_numpy(dtype=float)
    with np.errstate(over='ignore',invalid='ignore'):
        new[:,7]=new[:,1]*normalized
    finite=np.isfinite(new).all(axis=1);keep=timely&finite
    for column,name in enumerate(METRIC_FEATURES):result[name]=new[:,column]
    for name,values in zip(PROVENANCE_COLUMNS,[chosen_record,chosen_available,ages]):result[name]=values
    result=result.iloc[np.flatnonzero(keep)].copy()
    result.attrs={**frame.attrs,'features':AUGMENTED_FEATURES.copy(),'metrics_specification':specification()}
    quality={**cleaning,'offered_base_rows':len(frame),'retained_rows':int(keep.sum()),
        'offered_metrics_rows':len(metrics),'clean_metrics_rows':len(clean),
        'missing_or_expired_current_metrics_rows':int((~timely).sum()),
        'timely_but_missing_exact_lag_or_nonfinite_feature_rows':int((timely&~finite).sum()),
        'availability_delay_minutes':10,'maximum_available_age_minutes_exclusive':5,
        'maximum_observed_available_age_ms':int(ages[keep].max()) if keep.any() else None,
        'maximum_observed_raw_record_age_ms':int((stamps[keep]-chosen_record[keep]).max()) if keep.any() else None,
        'availability_is_publication_time_proof':False}
    return result,quality


def load_enriched_window(symbol,start_ms,end_ms,*,feature_store_dir=None,metrics_cache_dir=None,
                         metrics_frame=None,metrics_quality=None):
    """Merge a bounded base-feature window; optional raw history avoids reparsing."""
    from .model_search_data import load_window
    frame=load_window(symbol,start_ms,end_ms,store_dir=feature_store_dir)
    if metrics_frame is None:
        from .metrics_data import load_metrics,START,END
        start=max(START,datetime.fromtimestamp(start_ms/1000,timezone.utc).date()-timedelta(days=1))
        final=min(END,datetime.fromtimestamp((end_ms-1)/1000,timezone.utc).date()+timedelta(days=1))
        metrics_frame,metrics_quality=load_metrics(symbol,start,final,cache_dir=metrics_cache_dir)
    metrics_quality=metrics_quality or {}
    enriched,quality=augment_frame(frame,metrics_frame)
    # Full daily archive lists are captured once by the study's provenance.
    quality['manifest_identity']=metrics_quality.get('manifest_identity')
    quality['symbol']=symbol;quality['window_start_ms']=start_ms;quality['window_end_exclusive_ms']=end_ms
    return enriched,quality
