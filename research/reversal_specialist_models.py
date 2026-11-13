"""One fixed pooled exhaustion-reversal model with side-oriented inputs.

The hypothesis is conditional: after an extreme completed-hour move, weakening
aggressor flow and relative dislocation may distinguish reversal from continued
pressure. This model learns reversal success inside that fixed population,
instead of learning ordinary up/down direction on all market minutes.
"""
from __future__ import annotations

from copy import deepcopy
import time
import warnings

import numpy as np
import pandas as pd

from .adaptive_threshold_rules import availability_mask
from .event_engine import DAY_MS, MINUTE_MS, SYMBOLS

SIGNED_FEATURES = ('ret_1','ret_3','ret_15','ret_60','ret_240',
                   'imb','imb_ma5','imb_ma60','peer_ret_60',
                   'premium_deviation60_bps','premium_slope15_60_bps','funding_rate_bps')
UNSIGNED_FEATURES = ('rv_60','rv_ratio','rv_ratio_long','vol_z60','cnt_z60',
                     'premium_std60_bps','funding_age_hours')
INPUT_FEATURES = SIGNED_FEATURES + UNSIGNED_FEATURES
TRANSFORMED_FEATURES = tuple('reversal_side_x_'+x for x in SIGNED_FEATURES) + UNSIGNED_FEATURES + ('is_eth',)
PARAMETERS = {'C':.1, 'max_iter':400, 'solver':'lbfgs', 'random_state':17}
MIN_TRAIN_ROWS = 500
MIN_TRAIN_PER_ASSET = 100
RULE_ID = 'pooled_exhaustion_reversal_logistic_score_0.60'


def specification():
    return deepcopy({'id':RULE_ID, 'family':'conditional_exhaustion_reversal',
        'model':'pooled_regularized_logistic', 'parameters':PARAMETERS,
        'threshold':.60, 'train_days':180, 'refit_days':30, 'train_sample_minutes':30,
        'minimum_training_rows':MIN_TRAIN_ROWS, 'minimum_training_rows_per_asset':MIN_TRAIN_PER_ASSET,
        'signed_features':list(SIGNED_FEATURES), 'unsigned_features':list(UNSIGNED_FEATURES),
        'asset_indicator':'is_eth', 'transformed_features':list(TRANSFORMED_FEATURES),
        'direction':'minus sign of completed60minute return',
        'population':'abs(ret60)/(max(rv60,1e-8)*sqrt60)>=1;0<=rv_ratio<=1.25; known available dates',
        'target':'reversal-side index change >0 over entry+30minutes; exact flats omitted from training',
        'score_is_calibrated_win_probability':False})


def _times(frame, name):
    if name not in frame:raise ValueError('Missing '+name)
    values=pd.to_numeric(frame[name],errors='raise').to_numpy()
    if (not np.isfinite(values).all() or np.any(values!=np.floor(values))
            or np.any(values<=0) or np.any(values%MINUTE_MS)):
        raise ValueError('Require positive integer minute clocks: '+name)
    return values.astype(np.int64)


def _scope(frame):
    if not isinstance(frame,pd.DataFrame) or frame.empty or 'symbol' not in frame:
        raise ValueError('Require nonempty symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():raise ValueError('Only BTC/ETH are supported')
    stamps=_times(frame,'signal_ms')
    if frame.duplicated(['symbol','signal_ms']).any():raise ValueError('Duplicate asset/minute observations')
    return stamps


def reversal_sides(frame):
    """Observable fixed exhaustion population; no execution/label columns read."""
    stamps=_scope(frame)
    ret,sigma,ratio=(pd.to_numeric(frame[k],errors='raise').to_numpy(dtype=float)
                     for k in ('ret_60','rv_60','rv_ratio'))
    side=-np.sign(ret)
    z=np.abs(ret)/(np.maximum(sigma,1e-8)*np.sqrt(60))
    valid=(np.isfinite(ret)&np.isfinite(sigma)&(sigma>=0)&np.isfinite(ratio)
           &(ratio>=0)&(ratio<=1.25)&(z>=1)&(side!=0)&availability_mask(stamps))
    return np.where(valid,side,0).astype(np.int8)


def transformed_inputs(frame):
    """Signed inputs face the proposed reversal; state inputs preserve magnitude."""
    _scope(frame)
    missing=set(INPUT_FEATURES)-set(frame.columns)
    if missing:raise ValueError('Missing specialist inputs: '+','.join(sorted(missing)))
    raw=frame[list(INPUT_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite specialist inputs')
    side=-np.sign(frame.ret_60.to_numpy(dtype=float))
    signed=raw[:,:len(SIGNED_FEATURES)]*side[:,None]
    unsigned=raw[:,len(SIGNED_FEATURES):]
    return np.column_stack((signed,unsigned,(frame.symbol.to_numpy()=='ETHUSDT').astype(float)))


def directions(frame):
    """One fixed .60 operating point, with no later-result threshold search."""
    sides=reversal_sides(frame)
    scores=pd.to_numeric(frame['p_reversal'],errors='raise').to_numpy(dtype=float)
    return np.where(np.isfinite(scores)&(scores>=.60)&(scores<=1),sides,0).astype(np.int8)


def fit_predict(train,test,cutoff_ms,*,factory=None,progress=None):
    """Pool assets, purge unsettled labels, condition training before fitting."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    if (isinstance(cutoff_ms,(bool,np.bool_)) or not np.isscalar(cutoff_ms)
            or not np.isfinite(cutoff_ms) or int(cutoff_ms)!=cutoff_ms
            or cutoff_ms<=0 or cutoff_ms%(30*MINUTE_MS)):
        raise ValueError('Fit cutoff must be a positive halfhour boundary')
    cutoff=int(cutoff_ms);train_stamps=_scope(train);test_stamps=_scope(test)
    entries=_times(train,'entry_ms');settlements=_times(train,'settlement_ms')
    if np.any(entries!=train_stamps+MINUTE_MS) or np.any(settlements!=train_stamps+31*MINUTE_MS):
        raise ValueError('Require one-minute entry delay and30minute holding')
    if np.any(test_stamps<cutoff) or np.any(test_stamps>=cutoff+30*DAY_MS):
        raise ValueError('Prediction rows outside the frozen30day fold')
    sides=reversal_sides(train)
    eligible=((train_stamps>=cutoff-180*DAY_MS)&(settlements<cutoff)
              &(train_stamps%(30*MINUTE_MS)==0)&(sides!=0))
    positions=np.flatnonzero(eligible)
    # Future/unsettled execution observations never enter a numeric conversion.
    opened=pd.to_numeric(train.entry.iloc[positions],errors='raise').to_numpy(dtype=float)
    closed=pd.to_numeric(train.exit.iloc[positions],errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(opened).all() or not np.isfinite(closed).all()
            or np.any(opened<=0) or np.any(closed<=0)):
        raise ValueError('Invalid settled historical index prices')
    change=(closed-opened)*sides[positions]
    decisive=change!=0;positions=positions[decisive];target=(change[decisive]>0).astype(np.int8)
    selected=train.iloc[positions]
    counts={symbol:int((selected.symbol==symbol).sum()) for symbol in SYMBOLS}
    if len(selected)<MIN_TRAIN_ROWS or min(counts.values())<MIN_TRAIN_PER_ASSET:
        raise ValueError('Insufficient conditioned training rows across both assets')
    if len(np.unique(target))!=2:raise ValueError('Training requires reversal wins and losses')
    matrix=transformed_inputs(selected);testing=transformed_inputs(test)
    model=factory() if factory else make_pipeline(StandardScaler(),LogisticRegression(**PARAMETERS))
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=2):
            model.fit(matrix,target);probability=np.asarray(model.predict_proba(testing),dtype=float)
    convergence=[item for item in recorded if issubclass(item.category,ConvergenceWarning)]
    for item in recorded:
        if not issubclass(item.category,ConvergenceWarning):warnings.warn(str(item.message),item.category,stacklevel=2)
    classes=np.asarray(getattr(model,'classes_',[]))
    if classes.shape!=(2,) or set(classes)!={0,1}:raise ValueError('Require binary reversal classes')
    if (probability.shape!=(len(test),2) or not np.isfinite(probability).all()
            or np.any(probability<0) or np.any(probability>1)
            or not np.allclose(probability.sum(axis=1),1.,rtol=1e-7,atol=1e-9)):
        raise ValueError('Invalid reversal scores')
    fitted=model[-1] if hasattr(model,'steps') else model
    iterations=getattr(fitted,'n_iter_',None)
    meta={'fit_cutoff_ms':cutoff,'test_start_ms':int(test_stamps.min()),
          'test_end_exclusive_ms':int(test_stamps.max())+MINUTE_MS,'prediction_count':len(test),
          'specification':specification(),'training_count':len(selected),'training_by_symbol':counts,
          'training_wins':int(target.sum()),'training_losses':int((target==0).sum()),
          'latest_settlement_ms':int(settlements[positions].max()),
          'earliest_signal_ms':int(train_stamps[positions].min()),
          'training_converged':not convergence,'convergence_warning_count':len(convergence),
          'iterations':np.atleast_1d(iterations).astype(int).tolist() if iterations is not None else None,
          'runtime_seconds':round(time.perf_counter()-started,4)}
    if progress:progress(f'pooled specialist: {len(selected)} rows; {meta["runtime_seconds"]:.2f}s; converged={not convergence}')
    output=pd.DataFrame({'p_reversal':probability[:,int(np.flatnonzero(classes==1)[0])]},index=test.index)
    return output,meta
