"""Sixty fixed causal rules for the local model/sampling comparison.

Model scores are uncalibrated directional scores. A score threshold is not a
claim that an admitted trade has that win probability. Every rule operates on
the current completed-minute inputs; execution prices and labels are ignored.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

MINUTE_MS=60_000
SYMBOLS=('BTCUSDT','ETHUSDT')
MODEL_IDS=('logistic_halfhour','logistic_5m','logistic_1m',
           'gbm_shallow_5m','gbm_deeper_5m','ensemble_5m')
THRESHOLDS=(.56,.60,.64)
REGIMES=('exhaustion','trend_pullback','unrestricted')


def candidates(model_ids=None):
    """Return the preregistered catalog, optionally for a named model subset.

    Six deterministic rules remain present with any subset; no supplied value
    can create a new model, threshold or regime combination.
    """
    models=MODEL_IDS if model_ids is None else tuple(model_ids)
    if len(set(models))!=len(models) or any(name not in MODEL_IDS for name in models):
        raise ValueError('Require unique predeclared model IDs')
    names={'exhaustion':'极端偏离反转','trend_pullback':'4小时趋势回调','unrestricted':'仅模型分数'}
    rules=[]
    for model in models:
        for threshold in THRESHOLDS:
            for regime in REGIMES:
                rules.append({'id':f'{model}_{threshold:.2f}_{regime}',
                              'family':'model','model':model,'threshold':threshold,'regime':regime,
                              'label':f'{model} / 分数≥{threshold:.2f} / {names[regime]}',
                              'score_is_calibrated_win_probability':False})
    for threshold in (1.5,2.,2.5):
        for confirmation in (False,True):
            suffix='price_turn' if confirmation else 'unconfirmed'
            rules.append({'id':f'direct_reversal_{threshold:.1f}_{suffix}',
                          'family':'direct','model':None,'threshold':threshold,
                          'regime':'direct_reversal','require_price_turn':confirmation,
                          'label':f'标准化60分钟偏离≥{threshold:g} / '+('3分钟价格确认' if confirmation else '直接反转'),
                          'score_is_calibrated_win_probability':False})
    return rules


_RULES={rule['id']:rule for rule in candidates()}


def _numeric(frame,name):
    if name not in frame:raise ValueError(f'Missing historical input: {name}')
    return pd.to_numeric(frame[name],errors='raise').to_numpy(dtype=float)


def _canonical(rule):
    key=rule if isinstance(rule,str) else rule.get('id') if isinstance(rule,dict) else None
    if key not in _RULES:raise ValueError('Unknown predeclared rule')
    canonical=_RULES[key]
    # A caller may supply only an ID, but conflicting parameters cannot silently
    # masquerade as a candidate from the locked plan.
    if isinstance(rule,dict):
        for name in ('family','model','threshold','regime','require_price_turn'):
            if name in rule and rule[name]!=canonical.get(name):
                raise ValueError('Rule parameters differ from the predeclared catalog')
    return canonical


def directions(frame,rule):
    """Return {-1,0,+1} int8 directions in original row order, without labels."""
    rule=_canonical(rule)
    if not isinstance(frame,pd.DataFrame) or 'symbol' not in frame:
        raise ValueError('Require symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():raise ValueError('Only BTC/ETH are supported')
    times=_numeric(frame,'signal_ms')
    if (not np.isfinite(times).all() or np.any(times!=np.floor(times))
            or np.any(times%MINUTE_MS) or frame.duplicated(['symbol','signal_ms']).any()):
        raise ValueError('Require unique asset observations at exact minute boundaries')
    if rule['family']=='model':
        p=_numeric(frame,'p_'+rule['model'])
        direction=np.where(p>=.5,1,-1)
        mask=np.isfinite(p)&(p>=0)&(p<=1)&(np.maximum(p,1-p)>=rule['threshold'])
        if rule['regime']=='unrestricted':
            return np.where(mask,direction,0).astype(np.int8)
    else:
        direction=None
        mask=np.ones(len(frame),dtype=bool)
    ratio=_numeric(frame,'rv_ratio')
    sigma60=_numeric(frame,'rv_60')
    mask &= np.isfinite(ratio)&(ratio>=0)&np.isfinite(sigma60)&(sigma60>=0)
    if rule['regime']=='trend_pullback':
        ret240=_numeric(frame,'ret_240');sigma240=_numeric(frame,'rv_240')
        ret15=_numeric(frame,'ret_15')
        trend=np.sign(ret240)
        z240=ret240/(np.maximum(sigma240,1e-8)*math.sqrt(240))
        z15=ret15/(np.maximum(sigma60,1e-8)*math.sqrt(15))
        mask &= (np.isfinite(ret240)&np.isfinite(sigma240)&(sigma240>=0)&np.isfinite(ret15)
                 &(direction==trend)&(np.abs(z240)>=1)&(np.sign(ret15)==-trend)
                 &(np.abs(z15)>=.5)&(ratio<=1.5))
    else:
        ret60=_numeric(frame,'ret_60')
        z60=ret60/(np.maximum(sigma60,1e-8)*math.sqrt(60))
        reversal=-np.sign(ret60)
        if rule['family']=='direct':
            direction=reversal
            mask &= np.isfinite(ret60)&(np.abs(z60)>=rule['threshold'])&(ratio<=1)
            if rule['require_price_turn']:
                ret3=_numeric(frame,'ret_3')
                mask &= np.isfinite(ret3)&(direction*ret3>0)
        else:
            mask &= np.isfinite(ret60)&(direction==reversal)&(np.abs(z60)>=1)&(ratio<=1.25)
    return np.where(mask,direction,0).astype(np.int8)
