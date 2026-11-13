"""Sixteen fixed cross-model consensus rules for local event research.

Agreement between correlated models is a selection condition, not independent
evidence or a calibrated win probability. No execution price or label is read.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd

MINUTE_MS=60_000
SYMBOLS=('BTCUSDT','ETHUSDT')
TRAINED_MODELS=('logistic_halfhour','logistic_5m','logistic_1m',
                'gbm_shallow_5m','gbm_deeper_5m')
PAIRS=(('logistic_halfhour','gbm_deeper_5m'),
       ('logistic_1m','gbm_deeper_5m'),
       ('gbm_shallow_5m','gbm_deeper_5m'))
THRESHOLDS=(.56,.60)
REGIMES=('exhaustion','unrestricted')


def candidates():
    """Return exactly the fixed12pair and4unanimity rules, as independent data."""
    rules=[]
    regimes={'exhaustion':'原有极端偏离反转门槛','unrestricted':'仅同方向共识'}
    for members in PAIRS:
        for threshold in THRESHOLDS:
            for regime in REGIMES:
                rules.append({'id':f"pair_{'_'.join(members)}_{threshold:.2f}_{regime}",
                              'family':'consensus','consensus_kind':'pair',
                              'members':list(members),'aggregation':'minimum',
                              'threshold':threshold,'regime':regime,
                              'label':f"{' + '.join(members)} / 各自分数≥{threshold:.2f} / {regimes[regime]}",
                              'score_is_calibrated_win_probability':False,
                              'members_not_independent':True})
    for threshold in THRESHOLDS:
        for regime in REGIMES:
            rules.append({'id':f'unanimous_five_{threshold:.2f}_{regime}',
                          'family':'consensus','consensus_kind':'unanimous_five',
                          'members':list(TRAINED_MODELS),'aggregation':'mean',
                          'threshold':threshold,'regime':regime,
                          'label':f'5个训练模型方向一致 / 平均方向分数≥{threshold:.2f} / {regimes[regime]}',
                          'score_is_calibrated_win_probability':False,
                          'members_not_independent':True})
    return deepcopy(rules)


_RULES={rule['id']:rule for rule in candidates()}


def _numeric(frame,name):
    if name not in frame:raise ValueError(f'Missing historical input: {name}')
    return pd.to_numeric(frame[name],errors='raise').to_numpy(dtype=float)


def _canonical(rule):
    key=rule if isinstance(rule,str) else rule.get('id') if isinstance(rule,dict) else None
    if key not in _RULES:raise ValueError('Unknown predeclared consensus rule')
    canonical=_RULES[key]
    if isinstance(rule,dict):
        for name in ('family','consensus_kind','members','aggregation','threshold','regime'):
            if name in rule and rule[name]!=canonical.get(name):
                raise ValueError('Consensus parameters differ from the predeclared catalog')
    return canonical


def directions(frame,rule):
    """Return int8{-1,0,+1} without consulting outcomes or future rows.

    Every member must have a valid score and a strictly non-neutral direction.
    Pair rules gate the minimum individual direction score; unanimity rules
    gate the mean across the five fitted models, excluding their ensemble.
    """
    rule=_canonical(rule)
    if not isinstance(frame,pd.DataFrame) or 'symbol' not in frame:
        raise ValueError('Require symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():raise ValueError('Only BTC/ETH are supported')
    times=_numeric(frame,'signal_ms')
    if (not np.isfinite(times).all() or np.any(times!=np.floor(times))
            or np.any(times%MINUTE_MS) or frame.duplicated(['symbol','signal_ms']).any()):
        raise ValueError('Require unique asset observations at exact minute boundaries')
    scores=np.column_stack([_numeric(frame,'p_'+name) for name in rule['members']])
    valid=np.isfinite(scores).all(axis=1)&((scores>=0)&(scores<=1)).all(axis=1)
    member_directions=np.where(scores>.5,1,np.where(scores<.5,-1,0))
    direction=member_directions[:,0]
    agreement=(direction!=0)&(member_directions==direction[:,None]).all(axis=1)
    directional=np.maximum(scores,1-scores)
    confidence=(directional.min(axis=1) if rule['aggregation']=='minimum'
                else directional.mean(axis=1))
    mask=valid&agreement&(confidence>=rule['threshold'])
    if rule['regime']=='exhaustion':
        ret60=_numeric(frame,'ret_60');sigma60=_numeric(frame,'rv_60')
        ratio=_numeric(frame,'rv_ratio')
        z60=ret60/(np.maximum(sigma60,1e-8)*math.sqrt(60))
        mask &= (np.isfinite(ret60)&np.isfinite(sigma60)&(sigma60>=0)
                 &np.isfinite(ratio)&(ratio>=0)&(ratio<=1.25)
                 &(direction==-np.sign(ret60))&(np.abs(z60)>=1))
    return np.where(mask,direction,0).astype(np.int8)
