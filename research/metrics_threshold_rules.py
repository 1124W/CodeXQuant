"""Fixed sixteen augmented confidence gates and sixteen matched controls.

This catalog extends a related eight-candidate study; it is not a new holdout.
"""
from copy import deepcopy
import math
import numpy as np
import pandas as pd

from .adaptive_threshold_rules import availability_mask
from .event_engine import MINUTE_MS, SYMBOLS

MODELS = ('logistic', 'gbm')
THRESHOLDS = (.56, .60, .64, .68)
REGIMES = ('exhaustion', 'unrestricted')


def candidates(include_controls=False):
    return deepcopy([{'id': f'metrics_threshold_{model}_{arm}_{threshold:.2f}_{regime}',
        'family': 'fixed_metrics_confidence_extension', 'model': model, 'feature_set': arm,
        'threshold': threshold, 'regime': regime, 'selection_candidate': arm == 'aug',
        'score_is_calibrated_win_probability': False, 'previous_study_threshold': threshold in (.56,.60)}
        for arm in (('aug','base') if include_controls else ('aug',))
        for model in MODELS for threshold in THRESHOLDS for regime in REGIMES])


_RULES = {rule['id']:rule for rule in candidates(True)}


def _canonical(rule):
    key=rule if isinstance(rule,str) else rule.get('id') if isinstance(rule,dict) else None
    if key not in _RULES: raise ValueError('Unknown frozen confidence rule')
    fixed=_RULES[key]
    if isinstance(rule,dict) and any(name not in fixed or value!=fixed[name] for name,value in rule.items()):
        raise ValueError('Confidence rule differs from frozen catalog')
    return fixed


def matched_control(rule):
    fixed=_canonical(rule)
    if fixed['feature_set']!='aug': raise ValueError('Require augmented rule for matched control')
    return deepcopy(_RULES[f'metrics_threshold_{fixed["model"]}_base_{fixed["threshold"]:.2f}_{fixed["regime"]}'])


def directions(frame, rule):
    fixed=_canonical(rule)
    if not isinstance(frame,pd.DataFrame) or 'symbol' not in frame or not frame.symbol.isin(SYMBOLS).all():
        raise ValueError('Require BTC/ETH observations')
    def numeric(name):
        if name not in frame: raise ValueError('Missing historical input: '+name)
        return pd.to_numeric(frame[name],errors='raise').to_numpy(dtype=float)
    times=numeric('signal_ms')
    if (not np.isfinite(times).all() or np.any(times!=np.floor(times))
            or np.any(times%MINUTE_MS) or frame.duplicated(['symbol','signal_ms']).any()):
        raise ValueError('Require unique exact-minute asset observations')
    p=numeric(f'p_{fixed["model"]}_{fixed["feature_set"]}')
    side=np.where(p>.5,1,np.where(p<.5,-1,0))
    mask=(np.isfinite(p)&(p>=0)&(p<=1)&(side!=0)&(np.maximum(p,1-p)>=fixed['threshold'])&availability_mask(times))
    if fixed['regime']=='exhaustion':
        ret,sigma,ratio=(numeric(name) for name in ('ret_60','rv_60','rv_ratio'))
        z=ret/(np.maximum(sigma,1e-8)*math.sqrt(60))
        mask &= (np.isfinite(ret)&np.isfinite(sigma)&(sigma>=0)&np.isfinite(ratio)
                 &(ratio>=0)&(ratio<=1.25)&(side==-np.sign(ret))&(np.abs(z)>=1))
    return np.where(mask,side,0).astype(np.int8)
