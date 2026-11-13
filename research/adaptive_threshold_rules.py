"""Twelve fixed weekly adaptive score gates, for exploratory local research.

Only settled, previously out-of-sample predictions on the UTC 00/30 minute
grid can choose a gate. A gate is frozen from Monday to Monday. Its Wilson
bound is a conservative selection statistic, not a calibrated live win
probability: serial dependence, model refits and the minute-level portfolio
selector can all invalidate a literal coverage interpretation.
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .event_engine import DAY_MS, KNOWN_UNAVAILABLE_DATES, MINUTE_MS, SYMBOLS

WEEK_MS = 7 * DAY_MS
MONDAY_OFFSET_MS = 4 * DAY_MS  # 1970-01-05, the first Monday after the epoch.
MODEL_IDS = ('logistic_halfhour', 'logistic_1m', 'gbm_deeper_5m')
LOOKBACK_DAYS = (90, 180)
REGIMES = ('exhaustion', 'unrestricted')
THRESHOLDS = (.56, .60, .64, .68)
MIN_DECISIVE = 120
TARGET_WIN_RATE = .58
BREAK_EVEN = 1 / 1.85
WILSON_Z = NormalDist().inv_cdf(1 - .05 / 8)
_UNAVAILABLE_STARTS = tuple(int(pd.Timestamp(day, tz='UTC').timestamp() * 1000)
                            for day in KNOWN_UNAVAILABLE_DATES)


def candidates():
    """Return copied, immutable-in-meaning members of the fixed 12-rule family."""
    return deepcopy([
        {'id': f'adaptive_{model}_{days}d_{regime}',
         'family': 'weekly_adaptive_threshold', 'model': model,
         'lookback_days': days, 'regime': regime,
         'thresholds': list(THRESHOLDS), 'min_decisive': MIN_DECISIVE,
         'target_win_rate': TARGET_WIN_RATE, 'break_even': BREAK_EVEN,
         'one_sided_alpha': .05, 'within_week_comparisons': 8,
         'sample_minutes': 30, 'update_weekday_utc': 'Monday',
         'score_is_calibrated_win_probability': False,
         'label': f'{model} / {days}天已结算样本 / 每周自适应门槛 / {regime}'}
        for model in MODEL_IDS for days in LOOKBACK_DAYS for regime in REGIMES
    ])


_RULES = {rule['id']: rule for rule in candidates()}


def _canonical(rule):
    key = rule if isinstance(rule, str) else rule.get('id') if isinstance(rule, dict) else None
    if key not in _RULES:
        raise ValueError('Unknown predeclared adaptive rule')
    canonical = _RULES[key]
    if isinstance(rule, dict):
        for name, value in rule.items():
            if name != 'label' and (name not in canonical or value != canonical[name]):
                raise ValueError('Adaptive parameters differ from the predeclared catalog')
    return canonical


def _numeric(frame, name):
    if name not in frame:
        raise ValueError(f'Missing historical input: {name}')
    return pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)


def _boundary(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value != int(value) or value % MINUTE_MS):
        raise ValueError(f'{name} must be an integer minute boundary')
    return int(value)


def monday_floor(timestamp_ms):
    """UTC Monday containing timestamp_ms; no machine-local timezone involved."""
    return ((int(timestamp_ms) - MONDAY_OFFSET_MS) // WEEK_MS) * WEEK_MS + MONDAY_OFFSET_MS


def availability_mask(signal_ms):
    """Known-date mask for fixed signal+1 minute entry, signal+31 minute expiry.

    An expiry exactly at an unavailable day's start is excluded, matching the
    event engine. This calendar does not prove availability on other dates.
    """
    stamps = np.asarray(signal_ms, dtype=np.int64)
    available = np.ones(stamps.shape, dtype=bool)
    for start in _UNAVAILABLE_STARTS:
        available &= ~((stamps < start + DAY_MS) & (stamps + 31 * MINUTE_MS >= start))
    return available


def _lower(wins, n):
    if not n:
        return None
    p = wins / n
    square = WILSON_Z ** 2
    return float((p + square / (2 * n)
                  - WILSON_Z * math.sqrt(p * (1 - p) / n + square / (4 * n * n)))
                 / (1 + square / n))


def _sorted_indices(mask, stamps):
    indices = np.flatnonzero(mask)
    return indices[np.argsort(stamps[indices], kind='stable')]


def _slice(indices, stamps, first, stop):
    values = stamps[indices]
    return indices[np.searchsorted(values, first):np.searchsorted(values, stop)]


def _historical_outcomes(frame, indices, stamps, sides, cutoff):
    """Parse prices only after signal/calendar/grid/time eligibility is known.

    The caller has already required expected settlement < cutoff. Actual
    settlement is independently checked before any outcome prices are read.
    Current and future prices, y and is_flat are never parsed or validated.
    """
    subset = frame.iloc[indices]
    settlements = _numeric(subset, 'settlement_ms')
    if not np.isfinite(settlements).all() or np.any(settlements != np.floor(settlements)):
        raise ValueError('Historical settlement times must be integer milliseconds')
    settled = settlements < cutoff
    indices, settlements = indices[settled], settlements[settled].astype(np.int64)
    subset = frame.iloc[indices]
    if np.any(settlements != stamps[indices] + 31 * MINUTE_MS):
        raise ValueError('Historical events must expire 31 minutes after the signal')
    if 'entry_ms' in subset:
        entries = _numeric(subset, 'entry_ms')
        if np.any(entries != stamps[indices] + MINUTE_MS):
            raise ValueError('Historical events must enter one minute after the signal')
    opened, closed = _numeric(subset, 'entry'), _numeric(subset, 'exit')
    if (not np.isfinite(opened).all() or not np.isfinite(closed).all()
            or np.any(opened <= 0) or np.any(closed <= 0)):
        raise ValueError('Invalid settled historical index prices')
    change = (closed - opened) * sides[indices]
    fits = None
    if 'model_fit_cutoff_ms' in subset:
        fits = _numeric(subset, 'model_fit_cutoff_ms')
        if (not np.isfinite(fits).all() or np.any(fits != np.floor(fits))
                or np.any(fits > stamps[indices])):
            raise ValueError('Calibration predictions must be out of sample')
        fits = fits.astype(np.int64)
        if 'train_latest_settlement_ms' in subset:
            latest = _numeric(subset, 'train_latest_settlement_ms')
            if not np.isfinite(latest).all() or np.any(latest >= fits):
                raise ValueError('Calibration models must train on strictly settled labels')
    return indices, settlements, change, fits


def _trial(threshold, scores, indices, stamps, settlements, change, fits):
    included = scores[indices] >= threshold
    values = change[included]
    wins, losses, flats = (int(np.count_nonzero(values > 0)),
                           int(np.count_nonzero(values < 0)),
                           int(np.count_nonzero(values == 0)))
    n = wins + losses
    lower = _lower(wins, n)
    reasons = []
    if n < MIN_DECISIVE:
        reasons.append('insufficient_decisive_samples')
    if not n or wins / n < TARGET_WIN_RATE:
        reasons.append('point_win_rate_below_target')
    if lower is None or lower <= BREAK_EVEN:
        reasons.append('corrected_lower_not_above_break_even')
    used_stamps, used_settlements = stamps[indices[included]], settlements[included]
    return {'threshold': threshold, 'n': n, 'wins': wins, 'losses': losses,
            'flat_count': flats, 'win_rate': wins / n if n else None,
            'lower': lower, 'eligible': not reasons, 'rejection_reasons': reasons,
            'latest_settlement_ms': int(used_settlements.max()) if len(used_settlements) else None,
            'calibration_signal_bounds_ms': ([int(used_stamps.min()), int(used_stamps.max())]
                                              if len(used_stamps) else None),
            'distinct_fit_cutoffs_ms': (sorted(int(value) for value in np.unique(fits[included]))
                                       if fits is not None else [])}


def apply_rule(frame, rule, history_start_ms, evaluation_start_ms, evaluation_end_ms):
    """Return original-order int8 directions and weekly per-asset diagnostics.

    ``frame`` includes previous OOS history as well as the evaluation period.
    The caller declares the OOS coverage start; a full lookback in calendar
    time is required, but missing observations are never filled. Diagnostics
    show selected-threshold statistics, or broadest-threshold statistics if
    abstaining, with all four threshold trials retained for audit. Nonzero
    directions occur only inside [evaluation_start_ms, evaluation_end_ms).
    """
    rule = _canonical(rule)
    history_start = _boundary(history_start_ms, 'History start')
    first = _boundary(evaluation_start_ms, 'Evaluation start')
    stop = _boundary(evaluation_end_ms, 'Evaluation end')
    if not history_start <= first < stop:
        raise ValueError('Require history_start <= evaluation_start < evaluation_end')
    if not isinstance(frame, pd.DataFrame) or 'symbol' not in frame:
        raise ValueError('Require symbol-tagged observations')
    if not frame.symbol.isin(SYMBOLS).all():
        raise ValueError('Only BTC/ETH are supported')
    raw_times = _numeric(frame, 'signal_ms')
    if (not np.isfinite(raw_times).all() or np.any(raw_times != np.floor(raw_times))
            or np.any(raw_times % MINUTE_MS)
            or frame.duplicated(['symbol', 'signal_ms']).any()):
        raise ValueError('Require unique asset observations at exact minute boundaries')
    stamps = raw_times.astype(np.int64)
    del raw_times
    p = _numeric(frame, 'p_' + rule['model'])
    side = np.where(p > .5, 1, np.where(p < .5, -1, 0)).astype(np.int8)
    score = np.maximum(p, 1 - p)
    base = (np.isfinite(p) & (p >= 0) & (p <= 1) & (side != 0)
            & (score >= THRESHOLDS[0]) & availability_mask(stamps))
    del p
    if rule['regime'] == 'exhaustion':
        ret60, sigma60, ratio = (_numeric(frame, name) for name in ('ret_60', 'rv_60', 'rv_ratio'))
        z60 = ret60 / (np.maximum(sigma60, 1e-8) * math.sqrt(60))
        base &= (np.isfinite(ret60) & np.isfinite(sigma60) & (sigma60 >= 0)
                 & np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1.25)
                 & (side == -np.sign(ret60)) & (np.abs(z60) >= 1))
        del ret60, sigma60, ratio, z60
    symbols = frame.symbol.to_numpy()
    all_calibration = base & (stamps % (30 * MINUTE_MS) == 0) & (stamps >= history_start) & (stamps < stop)
    all_signals = base & (stamps >= first) & (stamps < stop)
    calibration = {s: _sorted_indices(all_calibration & (symbols == s), stamps) for s in SYMBOLS}
    signals = {s: _sorted_indices(all_signals & (symbols == s), stamps) for s in SYMBOLS}
    del all_calibration, all_signals, base
    result = np.zeros(len(frame), dtype=np.int8)
    diagnostics = []
    lookback = rule['lookback_days'] * DAY_MS
    for week in range(monday_floor(first), stop, WEEK_MS):
        warm = week - lookback >= history_start
        for symbol in SYMBOLS:
            record = {'rule_id': rule['id'], 'symbol': symbol, 'week_start_ms': week,
                      'week_end_ms': week + WEEK_MS, 'lookback_start_ms': week - lookback,
                      'history_start_ms': history_start, 'full_calendar_history': warm,
                      'threshold': None, 'n': 0, 'wins': 0, 'losses': 0, 'flat_count': 0,
                      'win_rate': None, 'lower': None, 'latest_settlement_ms': None,
                      'calibration_signal_bounds_ms': None, 'distinct_fit_cutoffs_ms': [],
                      'threshold_diagnostics': [], 'raw_signal_count': 0,
                      'statistic_scope': 'no_calibration', 'no_trade_reason': 'warmup'}
            if warm:
                # The stop is exclusive: expected settlement == Monday is not admitted.
                indices = _slice(calibration[symbol], stamps, week - lookback, week - 31 * MINUTE_MS)
                if len(indices):
                    indices, settlements, change, fits = _historical_outcomes(frame, indices, stamps, side, week)
                else:
                    settlements = np.empty(0, dtype=np.int64)
                    change, fits = np.empty(0), None
                trials = [_trial(t, score, indices, stamps, settlements, change, fits) for t in THRESHOLDS]
                eligible = [trial for trial in trials if trial['eligible']]
                chosen = max(eligible, key=lambda trial: (trial['lower'], trial['threshold'])) if eligible else None
                summary = chosen or trials[0]
                for name in ('n', 'wins', 'losses', 'flat_count', 'win_rate', 'lower',
                             'latest_settlement_ms', 'calibration_signal_bounds_ms', 'distinct_fit_cutoffs_ms'):
                    record[name] = summary[name]
                record['threshold_diagnostics'] = trials
                record['statistic_scope'] = 'selected_threshold' if chosen else 'broadest_threshold'
                record['no_trade_reason'] = 'no_eligible_threshold'
                if chosen:
                    record['threshold'] = chosen['threshold']
                    current = _slice(signals[symbol], stamps, max(week, first), min(week + WEEK_MS, stop))
                    current = current[score[current] >= chosen['threshold']]
                    result[current] = side[current]
                    record['raw_signal_count'] = int(len(current))
                    record['no_trade_reason'] = None if len(current) else 'no_current_score_regime_match'
            diagnostics.append(record)
    return result, diagnostics
