"""Eight fixed augmented rules and their eight exact matched base controls."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd

from .adaptive_threshold_rules import availability_mask
from .event_engine import MINUTE_MS, SYMBOLS

MODEL_IDS = ('logistic', 'gbm')
THRESHOLDS = (.56, .60)
REGIMES = ('exhaustion', 'unrestricted')


def candidates(include_controls=False):
    """Only augmented members are selection candidates; controls are not ranked."""
    arms = ('aug', 'base') if include_controls else ('aug',)
    return deepcopy([
        {'id': f'metrics_{model}_{arm}_{threshold:.2f}_{regime}',
         'family': 'metrics_feature_ablation', 'model': model, 'feature_set': arm,
         'threshold': threshold, 'regime': regime, 'selection_candidate': arm == 'aug',
         'score_is_calibrated_win_probability': False,
         'label': f'{model} / {"60+8持仓特征" if arm == "aug" else "60特征配对对照"} / 分数≥{threshold:.2f} / {regime}'}
        for arm in arms for model in MODEL_IDS for threshold in THRESHOLDS for regime in REGIMES
    ])


_RULES = {rule['id']: rule for rule in candidates(True)}


def _canonical(rule):
    key = rule if isinstance(rule, str) else rule.get('id') if isinstance(rule, dict) else None
    if key not in _RULES:
        raise ValueError('Unknown predeclared metrics rule')
    fixed = _RULES[key]
    if isinstance(rule, dict):
        for name, value in rule.items():
            if name != 'label' and (name not in fixed or value != fixed[name]):
                raise ValueError('Metrics rule parameters differ from the fixed catalog')
    return fixed


def matched_control(rule):
    rule = _canonical(rule)
    if rule['feature_set'] != 'aug':
        raise ValueError('Only an augmented candidate has a matched control')
    return deepcopy(_RULES[f'metrics_{rule["model"]}_base_{rule["threshold"]:.2f}_{rule["regime"]}'])


def _numeric(frame, name):
    if name not in frame:
        raise ValueError(f'Missing historical input: {name}')
    return pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)


def directions(frame, rule):
    """Causal completed-minute rule; all execution and label columns are ignored."""
    rule = _canonical(rule)
    if not isinstance(frame, pd.DataFrame) or 'symbol' not in frame:
        raise ValueError('Require symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():
        raise ValueError('Only BTC/ETH are supported')
    times = _numeric(frame, 'signal_ms')
    if (not np.isfinite(times).all() or np.any(times != np.floor(times))
            or np.any(times % MINUTE_MS) or frame.duplicated(['symbol', 'signal_ms']).any()):
        raise ValueError('Require unique asset observations at exact minute boundaries')
    p = _numeric(frame, f'p_{rule["model"]}_{rule["feature_set"]}')
    side = np.where(p > .5, 1, np.where(p < .5, -1, 0))
    mask = (np.isfinite(p) & (p >= 0) & (p <= 1) & (side != 0)
            & (np.maximum(p, 1-p) >= rule['threshold']) & availability_mask(times))
    if rule['regime'] == 'exhaustion':
        ret60, sigma60, ratio = (_numeric(frame, name) for name in ('ret_60', 'rv_60', 'rv_ratio'))
        z60 = ret60/(np.maximum(sigma60, 1e-8)*math.sqrt(60))
        mask &= (np.isfinite(ret60) & np.isfinite(sigma60) & (sigma60 >= 0)
                 & np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1.25)
                 & (side == -np.sign(ret60)) & (np.abs(z60) >= 1))
    return np.where(mask, side, 0).astype(np.int8)
