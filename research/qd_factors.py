"""Causal event-research features backed by the pinned QuantDinger registry.

The upstream registry supplies EMA and true-range series directly. RSI, ATR and
rolling z-score are evaluated in one pass for speed and checked against actual
upstream ``compute_factor`` calls on historical prefixes on every feature build.
This module does not import QuantDinger's Flask application or its live runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_DIR = ROOT / "external" / "QuantDinger-repo"
UPSTREAM_COMMIT = "d1cf2681bc84fba2551973b4bcab011ad483e8ad"
REGISTRY_BLOB = "4222d4709b61493767c56998928d3d92c2a04e4c"
REGISTRY_RELATIVE = Path("backend_api_python/app/services/factors/registry.py")
MINUTE_MS = 60_000
FEATURE_COLUMNS = ["timestamp", "open", "high", "low", "close", "rsi2", "rsi14",
                   "z20", "ema9", "ema21", "trend", "atr14", "body"]


def _root(upstream_dir=None):
    return Path(upstream_dir).resolve() if upstream_dir is not None else DEFAULT_UPSTREAM_DIR


def upstream_provenance(upstream_dir=None):
    directory = _root(upstream_dir)
    registry = directory / REGISTRY_RELATIVE
    if not registry.is_file():
        raise FileNotFoundError(
            f"QuantDinger registry missing at {registry}; run py -3.12 scripts/download_quantdinger.py"
        )
    raw = registry.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    blob = hashlib.sha1(f"blob {len(canonical)}\0".encode() + canonical).hexdigest()
    if blob != REGISTRY_BLOB:
        raise ValueError("QuantDinger registry differs from the pinned upstream Git blob")
    try:
        commit = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("QuantDinger source requires a verifiable local Git checkout") from error
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"QuantDinger commit is {commit}, expected {UPSTREAM_COMMIT}")
    return {
        "repository": "https://github.com/OpenByteInc/QuantDinger",
        "commit": commit,
        "checkout": str(directory),
        "registry_path": str(registry),
        "registry_relative_path": REGISTRY_RELATIVE.as_posix(),
        "git_blob_sha1": blob,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_lf_sha256": hashlib.sha256(canonical).hexdigest(),
        "license": "Apache-2.0",
        "runtime": "QuantDinger research module with event-contract extension",
        "full_platform_started": False,
    }


@lru_cache(maxsize=4)
def _load_verified_registry(directory, canonical_hash):
    # Register the real module before execution: dataclasses resolves its module.
    module_name = "_quantdinger_event_registry_" + canonical_hash[:12]
    spec = importlib.util.spec_from_file_location(module_name, Path(directory) / REGISTRY_RELATIVE)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the QuantDinger factor registry")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_registry(upstream_dir=None):
    metadata = upstream_provenance(upstream_dir)
    return _load_verified_registry(str(_root(upstream_dir)), metadata["canonical_lf_sha256"])


def _wilder_series(values, period):
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return output
    value = float(values[:period].mean())
    output[period - 1] = value
    for index in range(period, len(values)):
        value = (value * (period - 1) + float(values[index])) / period
        output[index] = value
    return output


def _rsi_series(close, period):
    changes = np.diff(np.asarray(close, dtype=float))
    gains = _wilder_series(np.maximum(changes, 0), period)
    losses = _wilder_series(np.maximum(-changes, 0), period)
    output = np.full(len(close), np.nan, dtype=float)
    valid = ~np.isnan(losses)
    calculated = np.full(len(changes), np.nan, dtype=float)
    positive_loss = valid & (losses > 0)
    calculated[positive_loss] = 100 - 100 / (1 + gains[positive_loss] / losses[positive_loss])
    # Preserve the upstream contract, including RSI=100 for an entirely flat run.
    calculated[valid & (losses <= 0)] = 100.0
    output[1:] = calculated
    return output


def _aggregate(rows, timeframe):
    if isinstance(timeframe, bool) or not isinstance(timeframe, int) or timeframe < 1:
        raise ValueError("timeframe must be a positive integer number of minutes")
    array = np.asarray(rows)
    if array.size == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError("Expected timestamp and index OHLC minute rows")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError("Minute rows must be finite")
    times = array[:, 0]
    if (np.any(times % MINUTE_MS) or np.any(np.diff(times) != MINUTE_MS)
            or np.any(times != times.astype(np.int64))):
        raise ValueError("Minute rows must be chronological, unique and continuous")
    opened, high, low, close = array[:, 1:].T
    if (np.any(array[:, 1:] <= 0) or np.any(low > np.minimum(opened, close))
            or np.any(high < np.maximum(opened, close)) or np.any(high < low)):
        raise ValueError("Invalid index OHLC")
    width = timeframe * MINUTE_MS
    source = pd.DataFrame(array, columns=["timestamp", "open", "high", "low", "close"])
    source["bucket"] = (times.astype(np.int64) // width) * width
    grouped = source.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), count=("timestamp", "size"),
        first=("timestamp", "first"), last=("timestamp", "last"),
    )
    complete = ((grouped["count"] == timeframe) & (grouped["first"] == grouped.index)
                & (grouped["last"] == grouped.index + width - MINUTE_MS))
    grouped = grouped.loc[complete, ["open", "high", "low", "close"]].copy()
    grouped.insert(0, "timestamp", (grouped.index + width).astype(np.int64))
    return grouped.reset_index(drop=True)


def _check_scalar_parity(registry, bars, features):
    lengths = sorted({n for n in (60, 100, 500, len(bars)) if 21 <= n <= len(bars)})
    checks = [
        ("rsi2", "rsi", {"period": 2}),
        ("rsi14", "rsi", {"period": 14}),
        ("z20", "mean_reversion_zscore", {"period": 20}),
        ("atr14", "atr", {"period": 14}),
        ("ema9", "ema", {"period": 9}),
        ("ema21", "ema", {"period": 21}),
    ]
    for length in lengths:
        history = bars.iloc[:length]
        for column, factor_id, params in checks:
            actual = registry.compute_factor(factor_id, history, params)
            optimized = float(features[column].iloc[length - 1])
            if not math.isclose(actual, optimized, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"QuantDinger parity failed: {column} at prefix {length}")
    return {"passed": True, "prefix_lengths": lengths, "factor_checks": len(checks) * len(lengths)}


def build_features(rows, timeframe, upstream_dir=None):
    """Return features available at each completed bar's exclusive UTC close.

    Index candles contain no executable volume: this adapter never fabricates a
    volume field. Appending future minutes cannot change historical features.
    """
    provenance = upstream_provenance(upstream_dir)
    registry = load_registry(upstream_dir)
    bars = _aggregate(rows, timeframe)
    result = bars.copy()
    if bars.empty:
        result = pd.DataFrame(columns=FEATURE_COLUMNS)
        result.attrs["upstream"] = provenance
        result.attrs["scalar_parity"] = {"passed": True, "prefix_lengths": [], "factor_checks": 0}
        return result
    close = bars["close"]
    for period in (9, 21):
        result[f"ema{period}"] = (registry._ema_values(close, period) if len(bars) >= period
                                    else np.full(len(bars), np.nan))
    result["rsi2"] = _rsi_series(close, 2)
    result["rsi14"] = _rsi_series(close, 14)
    mean = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=1)
    result["z20"] = (close - mean) / deviation.replace(0, np.nan)
    result.loc[deviation == 0, "z20"] = 0.0
    result["atr14"] = _wilder_series(registry._true_range(bars).to_numpy(), 14)
    denominator = result["atr14"].replace(0, np.nan)
    result["trend"] = (result["ema9"] - result["ema21"]) / denominator
    result["body"] = (bars["close"] - bars["open"]) / denominator
    parity = _check_scalar_parity(registry, bars, result)
    result = result[FEATURE_COLUMNS]
    result.attrs["upstream"] = provenance
    result.attrs["scalar_parity"] = parity
    result.attrs["factor_definitions"] = [registry.get_factor(key).metadata()
                                           for key in ("rsi", "mean_reversion_zscore", "ema", "atr")]
    return result
