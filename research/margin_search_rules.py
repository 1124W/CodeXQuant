"""Eight fixed rules for three-class +/-2bp margin models, with no labels read."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd

from .adaptive_threshold_rules import availability_mask
from .event_engine import MINUTE_MS, SYMBOLS

MODEL_IDS = ('margin_shallow', 'margin_deep')
THRESHOLDS = (.56, .60)
REGIMES = ('exhaustion', 'unrestricted')


def candidates():
    return deepcopy([
        {'id': f'{model}_{threshold:.2f}_{regime}', 'family': 'three_class_margin',
         'model': model, 'threshold': threshold, 'regime': regime, 'margin_bps': 2.,
         'score_is_calibrated_win_probability': False,
         'label': f'{model} / 方向超过2bp分数≥{threshold:.2f} / {regime}'}
        for model in MODEL_IDS for threshold in THRESHOLDS for regime in REGIMES
    ])


_RULES = {rule['id']: rule for rule in candidates()}


def _canonical(rule):
    key = rule if isinstance(rule, str) else rule.get('id') if isinstance(rule, dict) else None
    if key not in _RULES:
        raise ValueError('Unknown predeclared margin rule')
    fixed = _RULES[key]
    if isinstance(rule, dict):
        for name, value in rule.items():
            if name != 'label' and (name not in fixed or value != fixed[name]):
                raise ValueError('Margin parameters differ from the predeclared catalog')
    return fixed


def _numeric(frame, name):
    if name not in frame:
        raise ValueError(f'Missing historical input: {name}')
    return pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)


def directions(frame, rule):
    """Pick the strictly larger up/down score; ties and invalid rows abstain.

    Probabilities are not renormalized after removing neutral. Thus .56 is
    an absolute three-class score threshold, not conditional direction odds.
    Execution/label columns are never consulted, including calendar masking.
    """
    rule = _canonical(rule)
    if not isinstance(frame, pd.DataFrame) or 'symbol' not in frame:
        raise ValueError('Require symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():
        raise ValueError('Only BTC/ETH are supported')
    times = _numeric(frame, 'signal_ms')
    if (not np.isfinite(times).all() or np.any(times != np.floor(times))
            or np.any(times % MINUTE_MS) or frame.duplicated(['symbol', 'signal_ms']).any()):
        raise ValueError('Require unique asset observations at exact minute boundaries')
    scores = np.column_stack([_numeric(frame, f'p_{rule["model"]}_{side}')
                              for side in ('up', 'down', 'neutral')])
    valid = (np.isfinite(scores).all(axis=1) & ((scores >= 0) & (scores <= 1)).all(axis=1)
             & np.isclose(scores.sum(axis=1), 1., rtol=1e-7, atol=1e-9))
    up, down = scores[:, 0], scores[:, 1]
    side = np.where(up > down, 1, np.where(down > up, -1, 0))
    mask = (valid & (side != 0) & (np.maximum(up, down) >= rule['threshold'])
            & availability_mask(times))
    if rule['regime'] == 'exhaustion':
        ret60, sigma60, ratio = (_numeric(frame, name) for name in ('ret_60', 'rv_60', 'rv_ratio'))
        z60 = ret60 / (np.maximum(sigma60, 1e-8) * math.sqrt(60))
        mask &= (np.isfinite(ret60) & np.isfinite(sigma60) & (sigma60 >= 0)
                 & np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1.25)
                 & (side == -np.sign(ret60)) & (np.abs(z60) >= 1))
    return np.where(mask, side, 0).astype(np.int8)
