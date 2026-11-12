"""Causal regime and cross-asset inputs for BTC/ETH 30-minute research.

WorkBuddy ``ret_*`` and ``rv_*`` are both measured in basis points; ``rv_*``
is a standard deviation of one-minute returns, not a horizon return. Every
derived input below uses only the completed-bar features of the current signal.
The peer is joined on exactly the same ``signal_ms``. Missing peer observations
are never filled from an earlier or later signal, and settlement/label columns
are carried through without being read to construct or filter features.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .workbuddy_strategy import RESEARCH_FEATURES


SYMBOLS = ("BTCUSDT", "ETHUSDT")
EPSILON_BPS = 1e-6
ADDED_FEATURES = [
    "norm_ret_15", "norm_ret_30", "norm_ret_60", "trend_efficiency_60",
    "range_position_120", "flow_momentum_15", "flow_slope_5_60",
    "volatility_acceleration", "relative_norm_ret_60", "peer_ret_60",
    "peer_imb_ma15", "peer_rv_ratio",
]
FEATURE_UNITS = {
    name: "dimensionless" for name in ADDED_FEATURES
} | {"peer_ret_60": "basis_points"}
_REQUIRED = (
    "ret_15", "ret_30", "ret_60", "rv_60", "rv_ratio", "rv_ratio_long",
    "dist_hi", "dist_lo", "imb_ma5", "imb_ma15", "imb_ma60",
)


def _validate(frame: pd.DataFrame, symbol: str) -> list[str]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{symbol} samples must be a DataFrame")
    features = list(frame.attrs.get("features", RESEARCH_FEATURES))
    if not features or len(features) != len(set(features)):
        raise ValueError(f"{symbol} source features must be nonempty and unique")
    if set(features) - set(RESEARCH_FEATURES):
        raise ValueError("Source feature list must contain historical WorkBuddy inputs only")
    missing = {"signal_ms", *_REQUIRED, *features} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {symbol} historical inputs: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{symbol} samples are empty")
    times = frame["signal_ms"].to_numpy()
    if (not np.issubdtype(times.dtype, np.number)
            or not np.isfinite(times).all()
            or np.any(times != times.astype(np.int64))
            or np.any(times % 1_800_000)):
        raise ValueError("Signal timestamps must be finite integer half-hour boundaries")
    if frame["signal_ms"].duplicated().any():
        raise ValueError(f"Duplicate {symbol} signal timestamps")
    if "symbol" in frame and not frame["symbol"].eq(symbol).all():
        raise ValueError(f"Sample symbol does not match {symbol}")
    return features


def enrich_samples(samples_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Append 12 fixed features while keeping each event's execution and labels.

    Rows without an exact contemporaneous peer or with nonfinite historical
    inputs are omitted. No row is dropped because its future label, entry,
    expiry, or flat-event flag is missing/changed. Source frames are not mutated.
    A small fixed volatility floor prevents division by zero without fitting a
    scale to future data. ``trend_efficiency_60`` is an absolute standardized
    return proxy; it is not the bounded path-length efficiency ratio.
    """
    if set(samples_by_symbol) != set(SYMBOLS):
        raise ValueError("Require exactly BTCUSDT and ETHUSDT samples")
    source_features = {
        symbol: _validate(samples_by_symbol[symbol], symbol) for symbol in SYMBOLS
    }
    common = np.intersect1d(
        samples_by_symbol[SYMBOLS[0]]["signal_ms"].to_numpy(dtype=np.int64),
        samples_by_symbol[SYMBOLS[1]]["signal_ms"].to_numpy(dtype=np.int64),
    )
    if not len(common):
        raise ValueError("BTC/ETH samples have no common signal timestamps")
    aligned = {
        symbol: frame.set_index("signal_ms").loc[common].copy()
        for symbol, frame in samples_by_symbol.items()
    }
    result = {}
    for symbol in SYMBOLS:
        peer_symbol = SYMBOLS[1] if symbol == SYMBOLS[0] else SYMBOLS[0]
        frame = aligned[symbol].copy()
        peer = aligned[peer_symbol]
        sigma = frame["rv_60"].clip(lower=EPSILON_BPS)
        for horizon in (15, 30, 60):
            frame[f"norm_ret_{horizon}"] = frame[f"ret_{horizon}"] / (sigma * np.sqrt(horizon))
        frame["trend_efficiency_60"] = frame["norm_ret_60"].abs()
        width = frame["dist_lo"] - frame["dist_hi"]
        # A truly flat historical range has a neutral midpoint.
        frame["range_position_120"] = (
            frame["dist_lo"].div(width.where(width.abs() > EPSILON_BPS)).fillna(.5) - .5
        )
        frame["flow_momentum_15"] = frame["imb_ma15"] * frame["norm_ret_15"]
        frame["flow_slope_5_60"] = frame["imb_ma5"] - frame["imb_ma60"]
        frame["volatility_acceleration"] = frame["rv_ratio"] - frame["rv_ratio_long"]
        pair_sigma = np.sqrt(sigma ** 2 + peer["rv_60"].clip(lower=EPSILON_BPS) ** 2)
        frame["relative_norm_ret_60"] = (
            (frame["ret_60"] - peer["ret_60"]) / (pair_sigma * np.sqrt(60))
        )
        frame["peer_ret_60"] = peer["ret_60"]
        frame["peer_imb_ma15"] = peer["imb_ma15"]
        frame["peer_rv_ratio"] = peer["rv_ratio"]
        features = source_features[symbol] + ADDED_FEATURES
        # Inspect historical inputs too: a guarded denominator must not hide an
        # invalid volatility/range value that happened to become finite later.
        finite = np.isfinite(frame[list(dict.fromkeys([*features, *_REQUIRED]))].to_numpy(dtype=float)).all(axis=1)
        finite &= np.isfinite(peer[list(_REQUIRED)].to_numpy(dtype=float)).all(axis=1)
        finite &= (frame["rv_60"].to_numpy() >= 0) & (peer["rv_60"].to_numpy() >= 0)
        dropped = int((~finite).sum())
        frame = frame.loc[finite].reset_index()
        frame.attrs = deepcopy(samples_by_symbol[symbol].attrs)
        frame.attrs["features"] = features
        frame.attrs["regime_features"] = {
            "version": "1.0.0", "features": ADDED_FEATURES.copy(),
            "feature_units": FEATURE_UNITS.copy(), "peer_symbol": peer_symbol,
            "alignment": "exact same signal_ms; no filling",
            "dropped_unaligned_rows": len(samples_by_symbol[symbol]) - len(common),
            "dropped_nonfinite_historical_rows": dropped,
            "volatility_floor_bps": EPSILON_BPS,
        }
        result[symbol] = frame
    return result
