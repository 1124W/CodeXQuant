"""Causal signed-premium and realized-funding inputs for event research.

Premium closes are observable only after the minute has completed. Funding
observations become usable one minute after their exact exchange timestamp;
that timestamp is not rounded to a presumed eight-hour schedule.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


MINUTE_MS = 60_000
HOUR_MS = 3_600_000
FUNDING_PUBLICATION_LAG_MS = MINUTE_MS
MAX_FUNDING_AGE_MS = 24 * HOUR_MS
PREMIUM_FEATURES = [
    "premium_close_bps", "premium_mean15_bps", "premium_mean60_bps",
    "premium_deviation60_bps", "premium_slope15_60_bps", "premium_std60_bps",
]
FUNDING_FEATURES = [
    "funding_rate_bps", "funding_delta_bps", "funding_mean3_bps", "funding_age_hours",
]
_FORBIDDEN_SOURCE_FEATURES = {
    "entry", "exit", "y", "is_flat", "entry_ms", "settlement_ms", "fwd_bp",
    "outcome", "net_units", "direction", "p",
}


def _integer_times(values, label):
    times = np.asarray(values)
    if (not np.issubdtype(times.dtype, np.number)
            or not np.isfinite(times).all()
            or np.any(times != times.astype(np.int64))):
        raise ValueError(f"{label} must be finite integer millisecond timestamps")
    return times.astype(np.int64)


def _premium_inputs(premium, *, allow_gaps=False):
    required = ["timestamp", "open", "high", "low", "close"]
    if not isinstance(premium, pd.DataFrame) or set(required) - set(premium.columns):
        raise ValueError("Premium history requires timestamp and signed OHLC")
    if premium.empty:
        raise ValueError("Premium history is empty")
    p = premium[required].copy().sort_values("timestamp").reset_index(drop=True)
    times = _integer_times(p.timestamp.to_numpy(), "Premium times")
    if np.any(times % MINUTE_MS) or np.any(np.diff(times) <= 0):
        raise ValueError("Premium minute grid must be continuous, unique and clock aligned")
    has_gaps = np.any(np.diff(times) != MINUTE_MS)
    if has_gaps and not allow_gaps:
        raise ValueError("Premium minute grid must be continuous unless gap handling is explicitly enabled")
    prices = p[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all()
            or np.any(prices[:, 1] < np.maximum(prices[:, 0], prices[:, 3]))
            or np.any(prices[:, 2] > np.minimum(prices[:, 0], prices[:, 3]))):
        raise ValueError("Invalid signed premium OHLC")
    missing_count = 0
    if has_gaps:
        full_times = np.arange(times[0], times[-1] + MINUTE_MS, MINUTE_MS, dtype=np.int64)
        missing_count = len(full_times) - len(times)
        p = p.set_index("timestamp").reindex(full_times).rename_axis("timestamp").reset_index()
        times = full_times
    observed = p["close"].notna().to_numpy()
    bp = p["close"].astype(float) * 10_000
    mean15 = bp.rolling(15, min_periods=15).mean()
    mean60 = bp.rolling(60, min_periods=60).mean()
    result = pd.DataFrame({
        "signal_ms": times + MINUTE_MS,
        "premium_observation_ms": np.where(observed, times, np.nan),
        "premium_close_available_ms": np.where(observed, times + MINUTE_MS, np.nan),
        "premium_gap_minutes60": pd.Series(~observed).rolling(60, min_periods=1).sum(),
        "premium_close_bps": bp,
        "premium_mean15_bps": mean15,
        "premium_mean60_bps": mean60,
        "premium_deviation60_bps": bp - mean60,
        "premium_slope15_60_bps": mean15 - mean60,
        "premium_std60_bps": bp.rolling(60, min_periods=60).std(ddof=1),
    })
    result.attrs["missing_premium_minutes"] = int(missing_count)
    return result


def _funding_inputs(funding):
    required = ["funding_time", "rate"]
    if not isinstance(funding, pd.DataFrame) or set(required) - set(funding.columns):
        raise ValueError("Funding history requires funding_time and rate")
    if funding.empty:
        raise ValueError("Funding history is empty")
    f = funding[required].copy().sort_values("funding_time").reset_index(drop=True)
    times = _integer_times(f.funding_time.to_numpy(), "Funding times")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Funding timestamps must be unique")
    rates = f["rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("Funding rates must be finite")
    bp = pd.Series(rates * 10_000)
    return pd.DataFrame({
        "funding_observation_ms": times,
        "funding_available_ms": times + FUNDING_PUBLICATION_LAG_MS,
        "funding_rate_bps": bp,
        "funding_delta_bps": bp.diff(),
        "funding_mean3_bps": bp.rolling(3, min_periods=3).mean(),
    })


def add_derivative_features(samples, premium, funding, *, allow_premium_gaps=False):
    """Append 6 premium and 4 funding factors without inspecting future outcomes.

    Input sample ``attrs['features']`` lists historical base inputs. Premium
    history requires a complete minute grid by default. Explicitly enabling
    ``allow_premium_gaps`` retains missing observations as NaNs and excludes
    every signal without a fully observed 60-minute historical premium window.
    No missing price is invented. Sample joins require the exact
    preceding completed premium minute, so stale bars are never carried forward.
    Realized funding alone is carried forward after the publication lag, up to
    24 hours. The first two funding observations and first 59 premium minutes
    are warmup. Execution/target columns are copied, never read or filtered.
    """
    if not isinstance(samples, pd.DataFrame) or "signal_ms" not in samples:
        raise ValueError("Samples require signal_ms")
    features = list(samples.attrs.get("features", []))
    if (not features or len(features) != len(set(features))
            or set(features) & _FORBIDDEN_SOURCE_FEATURES):
        raise ValueError("Source features must be unique historical inputs only")
    if set(features) - set(samples.columns):
        raise ValueError("Samples are missing declared historical features")
    new_columns = set(PREMIUM_FEATURES + FUNDING_FEATURES + [
        "premium_observation_ms", "premium_close_available_ms",
        "premium_gap_minutes60",
        "funding_observation_ms", "funding_available_ms",
    ])
    if new_columns & set(samples.columns):
        raise ValueError("Derivative factors have already been added or columns collide")
    times = _integer_times(samples.signal_ms.to_numpy(), "Signal times")
    if len(np.unique(times)) != len(times):
        raise ValueError("Sample signal times must be unique")
    p = _premium_inputs(premium, allow_gaps=allow_premium_gaps)
    f = _funding_inputs(funding)
    frame = samples.copy().sort_values("signal_ms").reset_index(drop=True)
    frame["signal_ms"] = _integer_times(frame.signal_ms.to_numpy(), "Signal times")
    # Equality, rather than an asof premium join, forbids stale or future bars.
    frame = frame.merge(p, on="signal_ms", how="left", sort=False, validate="one_to_one")
    frame = pd.merge_asof(frame.sort_values("signal_ms"), f,
                          left_on="signal_ms", right_on="funding_available_ms",
                          direction="backward", allow_exact_matches=True,
                          tolerance=MAX_FUNDING_AGE_MS)
    frame["funding_age_hours"] = (frame.signal_ms - frame.funding_available_ms) / HOUR_MS
    premium_missing = frame["premium_observation_ms"].isna()
    premium_gap_affected = frame["premium_gap_minutes60"].fillna(0).to_numpy() > 0
    premium_finite = np.isfinite(frame[PREMIUM_FEATURES].to_numpy(dtype=float)).all(axis=1)
    funding_missing = frame["funding_observation_ms"].isna()
    funding_finite = np.isfinite(frame[FUNDING_FEATURES].to_numpy(dtype=float)).all(axis=1)
    historical_finite = np.isfinite(frame[features].to_numpy(dtype=float)).all(axis=1)
    keep = premium_finite & funding_finite & historical_finite
    metadata = {
        "version": "1.0.0", "premium_features": PREMIUM_FEATURES.copy(),
        "funding_features": FUNDING_FEATURES.copy(),
        "premium_availability": "open timestamp + 60000 ms; exact preceding completed minute only",
        "allow_premium_gaps": bool(allow_premium_gaps),
        "missing_premium_minutes": p.attrs["missing_premium_minutes"],
        "premium_gap_handling": "calendar-minute NaN grid; require complete trailing 60 observations; no filling",
        "funding_publication_lag_ms": FUNDING_PUBLICATION_LAG_MS,
        "funding_max_age_ms": MAX_FUNDING_AGE_MS,
        "funding_age_origin": "publication availability, not rounded event timestamp",
        "funding_interval_used_as_feature": False,
        "dropped_premium_not_current": int(premium_missing.sum()),
        "dropped_premium_warmup": int((~premium_missing & ~premium_finite & ~premium_gap_affected).sum()),
        "dropped_premium_gap_windows": int(premium_gap_affected.sum()),
        "dropped_funding_unavailable_or_stale": int(funding_missing.sum()),
        "dropped_funding_warmup": int((~funding_missing & ~funding_finite).sum()),
        "dropped_nonfinite_base_inputs": int((~historical_finite).sum()),
        "dropped_rows": int((~keep).sum()),
        "feature_population_drop": "historical-only: missing/stale premium, incomplete 60m premium history, unavailable/stale realized funding, funding warmup, nonfinite base inputs",
    }
    frame = frame.loc[keep].reset_index(drop=True)
    # Provenance timestamps are exact after unknown observations are removed.
    for name in ("premium_observation_ms", "premium_close_available_ms",
                 "funding_observation_ms", "funding_available_ms"):
        frame[name] = frame[name].astype(np.int64)
    if (not (frame.premium_close_available_ms == frame.signal_ms).all()
            or not (frame.premium_observation_ms == frame.signal_ms - MINUTE_MS).all()
            or not (frame.funding_available_ms <= frame.signal_ms).all()
            or not (frame.funding_age_hours.between(0, 24)).all()):
        raise ValueError("Derivative publication boundary violation")
    frame.attrs = deepcopy(samples.attrs)
    frame.attrs["features"] = features + PREMIUM_FEATURES + FUNDING_FEATURES
    frame.attrs["premium_features"] = PREMIUM_FEATURES.copy()
    frame.attrs["funding_features"] = FUNDING_FEATURES.copy()
    frame.attrs["derivative_features"] = metadata
    return frame
