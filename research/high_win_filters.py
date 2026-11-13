"""Sixteen predeclared, causal filters for local 30-minute event research.

Every rule contains the existing exhaustion gate. Model scores remain raw
direction scores, not calibrated win probabilities. Rules consume observations
available at ``signal_ms``; neither execution prices nor outcomes are inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MINUTE_MS = 60_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")

_RULES = (
    {"id": "baseline", "label": "原始分钟反转规则"},
    {"id": "score_058", "label": "方向分数至少0.58", "threshold": .58},
    {"id": "score_060", "label": "方向分数至少0.60", "threshold": .60},
    {"id": "score_062", "label": "方向分数至少0.62", "threshold": .62},
    {"id": "score_064", "label": "方向分数至少0.64", "threshold": .64},
    {"id": "volatility_cap_100", "label": "短期波动不高于60分钟波动"},
    {"id": "exhaustion_150", "label": "60分钟标准化偏离至少1.5"},
    {"id": "avoid_strong_opposing_trend", "label": "避开强烈反向4小时趋势"},
    {"id": "align_4h_trend", "label": "与4小时趋势同向"},
    {"id": "price_turn_3m", "label": "近3分钟价格确认反转"},
    {"id": "flow_turn", "label": "短期主动买卖流确认反转"},
    {"id": "candle_confirmation", "label": "最新分钟收盘位置确认方向"},
    {"id": "peer_flow_confirmation", "label": "另一币种主动买卖流同向"},
    {"id": "contrarian_premium", "label": "溢价方向支持反向交易"},
    {"id": "persistent_2m", "label": "连续两分钟同方向满足原始规则"},
    {"id": "price_and_flow_turn", "label": "3分钟价格与主动买卖流同时确认"},
)
_BY_ID = {rule["id"]: rule for rule in _RULES}


def candidates():
    """Return copies of the fixed rule catalog; no parameter cross-product."""
    return [dict(rule) for rule in _RULES]


def _numeric(frame, name):
    if name not in frame:
        raise ValueError(f"Missing historical input: {name}")
    return pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)


def filter_mask(frame, rule):
    """Return an acceptance mask in the input's original row order.

    ``persistent_2m`` needs both the current and immediately prior minute's
    completed observations for the same asset and model direction. Missing
    minutes break persistence; neither another asset nor a later row can fill
    that missing observation. Unavailable numeric indicators fail closed.
    """
    key = rule if isinstance(rule, str) else rule.get("id")
    if key not in _BY_ID:
        raise ValueError("Unknown predeclared filter")
    if not isinstance(frame, pd.DataFrame) or "symbol" not in frame:
        raise ValueError("Require symbol-tagged historical observations")
    if not frame.symbol.isin(SYMBOLS).all():
        raise ValueError("Only BTCUSDT and ETHUSDT are supported")
    times = _numeric(frame, "signal_ms")
    if (not np.isfinite(times).all() or np.any(times != np.floor(times))
            or np.any(times % MINUTE_MS)):
        raise ValueError("Signal times must be exact minute boundaries")
    if frame.duplicated(["symbol", "signal_ms"]).any():
        raise ValueError("Duplicate asset signal times")
    p = _numeric(frame, "p_up")
    ret60 = _numeric(frame, "ret_60")
    sigma60 = _numeric(frame, "rv_60")
    ratio = _numeric(frame, "rv_ratio")
    valid = np.isfinite(np.column_stack([p, ret60, sigma60, ratio])).all(axis=1)
    valid &= (p >= 0) & (p <= 1) & (sigma60 >= 0) & (ratio >= 0)
    direction = np.where(p >= .5, 1, -1)
    z60 = ret60 / (np.maximum(sigma60, 1e-8) * math.sqrt(60))
    threshold = _BY_ID[key].get("threshold", .56)
    base = (valid & (np.maximum(p, 1 - p) >= threshold)
            & (direction == -np.sign(ret60)) & (np.abs(z60) >= 1)
            & (ratio <= 1.25))
    if key in ("baseline", "score_058", "score_060", "score_062", "score_064"):
        return base
    if key == "volatility_cap_100":
        return base & (ratio <= 1)
    if key == "exhaustion_150":
        return base & (np.abs(z60) >= 1.5)
    if key in ("avoid_strong_opposing_trend", "align_4h_trend"):
        ret240, sigma240 = _numeric(frame, "ret_240"), _numeric(frame, "rv_240")
        z240 = ret240 / (np.maximum(sigma240, 1e-8) * math.sqrt(240))
        limit = -1. if key == "avoid_strong_opposing_trend" else 0.
        return base & np.isfinite(ret240) & np.isfinite(sigma240) & (sigma240 >= 0) & (direction * z240 >= limit)
    if key in ("price_turn_3m", "price_and_flow_turn"):
        ret3 = _numeric(frame, "ret_3")
        base &= np.isfinite(ret3) & (direction * ret3 > 0)
        if key == "price_turn_3m":
            return base
    if key in ("flow_turn", "price_and_flow_turn"):
        fast, slow = _numeric(frame, "imb_ma5"), _numeric(frame, "imb_ma60")
        return base & np.isfinite(fast) & np.isfinite(slow) & (direction * (fast - slow) > 0)
    if key == "candle_confirmation":
        clv = _numeric(frame, "clv")
        return base & np.isfinite(clv) & (direction * clv > 0)
    if key == "peer_flow_confirmation":
        flow = _numeric(frame, "peer_imb_ma15")
        return base & np.isfinite(flow) & (direction * flow >= 0)
    if key == "contrarian_premium":
        premium = _numeric(frame, "premium_close_bps")
        return base & np.isfinite(premium) & (direction * premium <= 0)
    if key == "persistent_2m":
        accepted = np.zeros(len(frame), dtype=bool)
        symbols = frame.symbol.to_numpy()
        for symbol in SYMBOLS:
            positions = np.flatnonzero(symbols == symbol)
            positions = positions[np.argsort(times[positions], kind="stable")]
            current, previous = positions[1:], positions[:-1]
            accepted[current] = (base[current] & base[previous]
                                 & (times[current] - times[previous] == MINUTE_MS)
                                 & (direction[current] == direction[previous]))
        return accepted
    raise AssertionError("Predeclared filter has no implementation")
