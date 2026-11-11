"""Pinned WorkBuddy contrarian strategy in the QuantDinger factor runtime.

Spot OHLC and order flow generate features; Binance index opens define labels
and event settlements. Prediction-time flat events remain in the population.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import sys
from types import SimpleNamespace
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .event_engine import DAY_MS, MINUTE_MS, KNOWN_UNAVAILABLE_DATES, utc_ms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "external" / "events-workBuddy"
REFERENCE_COMMIT = "1f4e26e948ff566a5307d821e811aa7d6228928b"
REFERENCE_BLOBS = {
    "datahub/features.py": "84b6253bf72d86e4af563e81ed044978a19ff8f1",
    "live/strategy.py": "4ec75d2fcab8c206626bec9413cf42eb969ff48c",
    "live/train.py": "bf2aa91cbf1cc0cbd05c8058536fcbdeaf2a15d8",
}
RESEARCH_BLOB = "6a4d60994e93d34c27ed9f5193fe605f4d06ed54"
PARAMETERS = {"BTCUSDT": {"margin": .12, "min_abs_ret": 20.},
              "ETHUSDT": {"margin": .12, "min_abs_ret": 28.}}
MODEL_PARAMETERS = {"max_iter": 150, "max_depth": 3, "learning_rate": .04,
                    "min_samples_leaf": 200, "l2_regularization": 1.,
                    "random_state": 0, "early_stopping": False}
RESEARCH_MODEL_PARAMETERS = {"max_iter": 200, "max_depth": 4, "learning_rate": .06,
                             "min_samples_leaf": 80, "l2_regularization": 1.,
                             "random_state": 7, "early_stopping": False}
RULE_ID = "workbuddy_contrarian_gbm_30m"
FEATURE_CHUNK_SIZE = 80_000
FEATURE_OVERLAP = 120
FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_15", "ret_30", "ret_60",
    "imb", "imb_ma5", "imb_ma15", "imb_ma60", "imb_std15", "imb_std60",
    "vol_z15", "vol_z60", "cnt_z15", "cnt_z60", "ticket_z60",
    "rv_15", "rv_60", "rv_ratio", "clv", "amihud_z60",
    "hour_sin", "hour_cos", "ret_5_smooth", "ret_15_smooth",
]
RESEARCH_FEATURES = FEATURES + ["ret_120", "ret_240", "ret_480", "rv_240", "rv_ratio_long",
                               "dist_hi", "dist_lo", "streak", "dow_sin", "dow_cos"]
_UNAVAILABLE_DAYS = frozenset(utc_ms(date.fromisoformat(day)) // DAY_MS
                            for day in KNOWN_UNAVAILABLE_DATES)


def _profile(profile):
    if profile not in ("live", "research"):
        raise ValueError("Profile must be live or research")
    return (RESEARCH_FEATURES, RESEARCH_MODEL_PARAMETERS, 180, 600) if profile == "research" else (FEATURES, MODEL_PARAMETERS, 240, FEATURE_OVERLAP)


def reference_provenance(upstream_dir=None, profile="live"):
    directory = Path(upstream_dir or DEFAULT_REFERENCE).resolve()
    features, model_parameters, train_days, overlap = _profile(profile)
    files = {}
    expected_files = {**REFERENCE_BLOBS, **({"research/ceiling.py": RESEARCH_BLOB} if profile == "research" else {})}
    for relative, expected_blob in expected_files.items():
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned WorkBuddy source is missing: {path}")
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        blob = hashlib.sha1(f"blob {len(canonical)}\0".encode() + canonical).hexdigest()
        if blob != expected_blob:
            raise ValueError(f"WorkBuddy source does not match pinned Git blob: {relative}")
        files[relative] = {"git_blob_sha1": blob, "sha256": hashlib.sha256(canonical).hexdigest()}
    return {"repository": "https://github.com/1124W/events-workBuddy", "private": True,
            "commit": REFERENCE_COMMIT, "checkout": str(directory), "files": files,
            "strategy_function": "live/strategy.py:decide", "feature_count": len(features),
            "features": features.copy(), "parameters": PARAMETERS,
            "profile": profile, "default_train_days": train_days,
            "feature_function": "research/ceiling.py:build_ext" if profile == "research" else "datahub/features.py:build_features",
            "model_function": "research/ceiling.py:make_model" if profile == "research" else "live/train.py:make_model",
            "model_parameters": model_parameters.copy(),
            "adaptations": [
                "Train and settle on delayed Binance index minute opens instead of spot-close labels.",
                "Purge training labels whose settlement is not strictly before the fit cutoff.",
                "Retain flat future events at prediction time; exclude them from classifier fitting only.",
                "Disable automatic random validation/early stopping and limit native threads to two.",
                f"Build historical features in bounded chunks with a {overlap}-minute prefix overlap.",
                *(["Remove source build_ext future-label/drop-flat row filters; carry streak history exactly across chunks."] if profile == "research" else []),
            ]}


def _isolated_function(path, name, namespace=None, transform=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if node is None:
        raise ValueError(f"Pinned function is absent: {path.name}:{name}")
    if transform:
        node = transform(node)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@lru_cache(maxsize=4)
def _load_reference(directory, profile="live"):
    path = Path(directory) / "datahub" / "features.py"
    name = "_workbuddy_pinned_event_features"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.FEATURES != FEATURES:
        raise ValueError("Reference feature order differs from the pinned training contract")
    decide = _isolated_function(Path(directory) / "live" / "strategy.py", "decide")
    if profile == "research":
        ceiling = Path(directory) / "research" / "ceiling.py"
        namespace = {"np": np, "pd": pd, "BP": 1e4, "BASE_FEATURES": FEATURES,
                     "build_features": module.build_features}
        historical_ext = _isolated_function(ceiling, "build_ext", namespace, _historical_only_ext)
        source = SimpleNamespace(FEATURES=RESEARCH_FEATURES,
                                 build_features=lambda bars, horizon=30: historical_ext(bars, horizon, 0))
        return source, decide, _isolated_function(ceiling, "make_model")
    return module, decide, _isolated_function(Path(directory) / "live" / "train.py", "make_model")


def _historical_only_ext(node):
    """Keep the pinned feature expressions while removing outcome-based sampling."""
    body, changed = [], {"dropna": 0, "flat_assignment": 0, "flat_filter": 0, "flat_attrs": 0}
    for statement in node.body:
        text = ast.unparse(statement)
        if text.startswith("df = df.dropna(subset=feats + ['y', 'fwd_bp'])"):
            statement = ast.parse("df = df.dropna(subset=feats).reset_index(drop=True)").body[0]
            changed["dropna"] += 1
        elif text.startswith("flat = float((df['fwd_bp'] == 0).mean())"):
            changed["flat_assignment"] += 1
            continue
        elif text.startswith("df = df[df['fwd_bp'] != 0]"):
            changed["flat_filter"] += 1
            continue
        elif text.startswith("df.attrs['flat_share'] = flat"):
            changed["flat_attrs"] += 1
            continue
        body.append(statement)
    if changed != {"dropna": 1, "flat_assignment": 1, "flat_filter": 1, "flat_attrs": 2}:
        raise ValueError(f"The pinned historical feature filter structure changed: {changed}")
    node.body = body
    return node


def load_reference(upstream_dir=None, profile="live"):
    provenance = reference_provenance(upstream_dir, profile)
    return _load_reference(provenance["checkout"], profile)


def make_model(upstream_dir=None, profile="live"):
    _, _, factory = load_reference(upstream_dir, profile)
    model = factory("gbm")
    # A deterministic fixed iteration count avoids a random validation partition.
    model.set_params(early_stopping=False)
    return model


def _validate_minutes(frame, required, label):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {label} columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Empty {label} minute history")
    timestamps = frame["timestamp"].to_numpy()
    if (not np.isfinite(timestamps).all() or np.any(timestamps != timestamps.astype(np.int64))
            or np.any(timestamps % MINUTE_MS) or np.any(np.diff(timestamps) != MINUTE_MS)):
        raise ValueError(f"{label} minute grid must be continuous, unique and clock aligned")
    values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values <= 0)
            or np.any(values[:, 2] > np.minimum(values[:, 0], values[:, 3]))
            or np.any(values[:, 1] < np.maximum(values[:, 0], values[:, 3]))):
        raise ValueError(f"Invalid {label} OHLC")


def _quantdinger_contracts(samples, diagnostic_bars, features):
    from .qd_factors import load_registry, upstream_provenance
    registry = load_registry()
    for name in features:
        factor_id = "workbuddy_" + name
        if factor_id not in registry._FACTORS:
            registry._FACTORS[factor_id] = registry.FactorDefinition(
                factor_id=factor_id, version="1.0.0", name_i18n_key=factor_id,
                description_i18n_key=factor_id, category="workbuddy_event",
                factor_type="technical", required_fields=(name,),
                supported_contexts=("cta", "portfolio"), default_warmup_bars=120,
                compute=lambda frame, params, column=name: float(frame[column].iloc[-1]))
    checked = 0
    if not samples.empty:
        for position in sorted({0, len(samples) - 1}):
            frame = samples.iloc[[position]]
            for name in features:
                actual = registry.compute_factor("workbuddy_" + name, frame)
                if not math.isclose(actual, float(frame[name].iloc[0]), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"QuantDinger custom factor contract failed: {name}")
                checked += 1
    ema = registry._ema_values(diagnostic_bars["close"], 20)
    scalar_ema = registry.compute_factor("ema", diagnostic_bars, {"period": 20})
    if not math.isclose(float(ema.iloc[-1]), scalar_ema, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("QuantDinger EMA scalar/series parity failed")
    ranges = registry._true_range(diagnostic_bars)
    if len(ranges) != len(diagnostic_bars) or not np.isfinite(ranges).all():
        raise ValueError("QuantDinger true-range diagnostics failed")
    return {"upstream": upstream_provenance(), "custom_factor_count": len(features),
            "factor_contract_checks": checked, "ema_parity_passed": True,
            "true_range_checked": True, "factor_values_source": "pinned WorkBuddy build_features"}


def build_samples(spot, index, symbol, upstream_dir=None, profile="live"):
    """Build observable half-hour signals without filtering by future outcomes."""
    if symbol not in PARAMETERS:
        raise ValueError("Only BTCUSDT and ETHUSDT 30-minute events are supported")
    source, _, _ = load_reference(upstream_dir, profile)
    features, _, _, overlap = _profile(profile)
    required = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_base", "taker_buy_quote"]
    _validate_minutes(spot, required, "spot")
    _validate_minutes(index, ["timestamp", "open", "high", "low", "close"], "index")
    if len(spot) < overlap + 1:
        raise ValueError(f"At least {overlap + 1} historical spot minutes are required")
    batches = []
    chunk_size = FEATURE_CHUNK_SIZE
    streak = None
    if profile == "research":
        # Streak length has unbounded history; a fixed overlap alone is not exact.
        up = np.zeros(len(spot), dtype=bool)
        up[1:] = np.diff(np.log(spot["close"].to_numpy())) > 0
        positions = np.arange(len(spot))
        starts = np.maximum.accumulate(np.where(np.r_[True, up[1:] != up[:-1]], positions, 0))
        streak = ((positions - starts + 1) * np.where(up, 1, -1)).astype(np.int32)
    for offset in range(0, len(spot), chunk_size):
        stop = min(offset + chunk_size, len(spot))
        first = max(0, offset - overlap)
        bars = spot.iloc[first:stop][required].copy()
        bars["ts"] = pd.to_datetime(bars["timestamp"], unit="ms", utc=True)
        frame = source.build_features(bars, horizon=30)
        if profile == "research":
            # The source drops historical warmup NaNs and resets its row index.
            offsets = ((frame["timestamp"].to_numpy(dtype=np.int64) - int(spot["timestamp"].iloc[0])) // MINUTE_MS).astype(np.int64)
            frame["streak"] = streak[offsets]
        signal = frame["timestamp"].to_numpy(dtype=np.int64) + MINUTE_MS
        # Keep only rows owned by this chunk; the overlap is historical context.
        mask = (frame["timestamp"].to_numpy() >= int(spot["timestamp"].iloc[offset])) & (signal % (30 * MINUTE_MS) == 0)
        compact = frame.loc[mask, features].copy()
        compact.insert(0, "signal_ms", signal[mask])
        compact = compact.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
        batches.append(compact)
    samples = pd.concat(batches, ignore_index=True)
    samples["entry_ms"] = samples["signal_ms"] + MINUTE_MS
    samples["settlement_ms"] = samples["entry_ms"] + 30 * MINUTE_MS
    origin = int(index["timestamp"].iloc[0])
    entry_offsets = ((samples["entry_ms"].to_numpy() - origin) // MINUTE_MS).astype(np.int64)
    exit_offsets = ((samples["settlement_ms"].to_numpy() - origin) // MINUTE_MS).astype(np.int64)
    covered = (entry_offsets >= 0) & (exit_offsets < len(index))
    samples = samples.loc[covered].copy().reset_index(drop=True)
    entry_offsets, exit_offsets = entry_offsets[covered], exit_offsets[covered]
    timestamps = index["timestamp"].to_numpy(dtype=np.int64)
    if (not np.array_equal(timestamps[entry_offsets], samples["entry_ms"].to_numpy())
            or not np.array_equal(timestamps[exit_offsets], samples["settlement_ms"].to_numpy())):
        raise ValueError("Index entry and expiry must match exact minute observations")
    opens = index["open"].to_numpy(dtype=float)
    samples["entry"] = opens[entry_offsets]
    samples["exit"] = opens[exit_offsets]
    samples["is_flat"] = samples["exit"] == samples["entry"]
    samples["y"] = (samples["exit"] > samples["entry"]).astype(np.int8)
    samples["symbol"] = symbol
    samples.attrs["reference"] = reference_provenance(upstream_dir, profile)
    samples.attrs["features"] = features.copy()
    samples.attrs["profile"] = profile
    diagnostics = spot.iloc[:min(500, len(spot))][["open", "high", "low", "close", "volume"]]
    samples.attrs["quantdinger"] = _quantdinger_contracts(samples, diagnostics, features)
    samples.attrs["flat_events_retained"] = int(samples["is_flat"].sum())
    return samples


def _as_ms(value):
    if isinstance(value, bool):
        raise ValueError("Boolean timestamps are invalid")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, date):
        return utc_ms(value)
    if isinstance(value, str):
        return utc_ms(date.fromisoformat(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError("Use an ISO UTC date or millisecond timestamp")


def _auc(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
    if not positive or not negative:
        return None
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def walk_forward_signals(samples, symbol, start, end, train_days=None, test_days=30,
                         model_factory=None, progress=print, upstream_dir=None, profile="live"):
    """Fit on settled history only and generate disjoint chronological predictions."""
    if symbol not in PARAMETERS:
        raise ValueError("Only BTCUSDT and ETHUSDT are supported")
    expected_features, model_parameters, default_train, _ = _profile(profile)
    train_days = default_train if train_days is None else train_days
    features = list(samples.attrs.get("features", expected_features))
    if features != expected_features or samples.attrs.get("profile", profile) != profile:
        raise ValueError("Sample features/profile do not match the requested fixed model profile")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (train_days, test_days)):
        raise ValueError("Train and test days must be positive integers")
    first, last = _as_ms(start), _as_ms(end)
    if last <= first + train_days * DAY_MS:
        raise ValueError("History must include training and at least one test window")
    required = features + ["signal_ms", "entry_ms", "settlement_ms", "entry", "exit", "is_flat", "y"]
    if set(required) - set(samples.columns):
        raise ValueError("Missing sample features or event execution columns")
    d = samples.sort_values("signal_ms").reset_index(drop=True)
    if d["signal_ms"].duplicated().any() or not np.isfinite(d[features].to_numpy()).all():
        raise ValueError("Prediction samples must be unique with finite features")
    if (np.any(d["entry_ms"].to_numpy() != d["signal_ms"].to_numpy() + MINUTE_MS)
            or np.any(d["settlement_ms"].to_numpy() != d["entry_ms"].to_numpy() + 30 * MINUTE_MS)):
        raise ValueError("Only delayed 30-minute event execution is supported")
    _, decide, _ = load_reference(upstream_dir, profile)
    factory = model_factory or (lambda: make_model(upstream_dir, profile))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        if model_factory is None:
            raise
        from contextlib import nullcontext
        threadpool_limits = lambda **_: nullcontext()
    raw, folds, predictions = [], [], []
    cutoff = first + train_days * DAY_MS
    oos_start = cutoff
    while cutoff < last:
        test_end = min(last, cutoff + test_days * DAY_MS)
        train = d[(d["signal_ms"] >= cutoff - train_days * DAY_MS)
                  & (d["signal_ms"] < cutoff) & (d["settlement_ms"] < cutoff) & ~d["is_flat"]]
        test = d[(d["signal_ms"] >= cutoff) & (d["signal_ms"] < test_end)
                 & (d["settlement_ms"] < last)]
        fold = {"symbol": symbol, "test_start_ms": cutoff, "test_end_ms": test_end,
                "train_start_ms": cutoff - train_days * DAY_MS,
                "train_start_utc": datetime.fromtimestamp((cutoff - train_days * DAY_MS) / 1000, timezone.utc).date().isoformat(),
                "test_start_utc": datetime.fromtimestamp(cutoff / 1000, timezone.utc).date().isoformat(),
                "test_end_exclusive_utc": datetime.fromtimestamp(test_end / 1000, timezone.utc).date().isoformat(),
                "n_train": len(train), "n_test": len(test),
                "train_latest_settlement_ms": int(train["settlement_ms"].max()) if len(train) else None}
        if len(train) < 2000 or train["y"].nunique() < 2 or test.empty:
            fold.update(status="skipped", reason="insufficient_training_or_test_samples", auc=None,
                        purge_passed=False)
            folds.append(fold)
            cutoff = test_end
            continue
        if fold["train_latest_settlement_ms"] >= cutoff:
            raise ValueError("A training label crosses the model fit cutoff")
        model = factory()
        with threadpool_limits(limits=2):
            model.fit(train[features].to_numpy(), train["y"].to_numpy(dtype=int))
            probabilities = np.asarray(model.predict_proba(test[features].to_numpy()))[:, 1]
        if len(probabilities) != len(test) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("The model returned invalid prediction probabilities")
        decisive = ~test["is_flat"].to_numpy()
        fold.update(status="evaluated", purge_passed=True,
                    auc=_auc(test["y"].to_numpy()[decisive], probabilities[decisive]),
                    prediction_flat_count=int((~decisive).sum()))
        trade_count = 0
        for row, probability in zip(test.itertuples(index=False), probabilities):
            decision = decide(float(probability), float(row.ret_60), **PARAMETERS[symbol])
            direction = 1 if decision["action"] == "long" else -1 if decision["action"] == "short" else 0
            predictions.append({"signal_ms": int(row.signal_ms), "p": float(probability),
                                "y": int(row.y), "is_flat": bool(row.is_flat), "direction": direction})
            if not direction:
                continue
            if any(day in _UNAVAILABLE_DAYS for day in range(int(row.signal_ms) // DAY_MS, int(row.settlement_ms) // DAY_MS + 1)):
                continue
            raw.append({"symbol": symbol, "signal_ms": int(row.signal_ms),
                        "entry_ms": int(row.entry_ms), "settlement_ms": int(row.settlement_ms),
                        "direction": direction, "entry": float(row.entry), "exit": float(row.exit),
                        "rule_id": RULE_ID, "profile": profile, "timeframe_minutes": 1, "horizon_minutes": 30,
                        "p": float(probability), "ret_60": float(row.ret_60),
                        "fold_id": f"{symbol}_{cutoff}", "model_fit_cutoff_ms": cutoff,
                        "train_latest_settlement_ms": fold["train_latest_settlement_ms"]})
            trade_count += 1
        fold["signals"] = trade_count
        folds.append(fold)
        if progress:
            dt = datetime.fromtimestamp(cutoff / 1000, timezone.utc).date()
            progress(f"{symbol} fold {dt}: train={len(train)}, predict={len(test)}, signals={trade_count}, auc={fold['auc']}")
        cutoff = test_end
    decisive_predictions = [row for row in predictions if not row["is_flat"]]
    auc = _auc([row["y"] for row in decisive_predictions], [row["p"] for row in decisive_predictions])
    return {"raw_trades": raw, "folds": folds, "prediction_count": len(predictions),
            "prediction_flat_count": sum(row["is_flat"] for row in predictions), "auc": auc,
            "oos_start_utc": datetime.fromtimestamp(oos_start / 1000, timezone.utc).date().isoformat(),
            "model_parameters": model_parameters.copy(), "parameters": PARAMETERS[symbol].copy(),
            "profile": profile, "features": features,
            "reference": reference_provenance(upstream_dir, profile),
            "quantdinger": samples.attrs.get("quantdinger"),
            "fit_policy": f"rolling settled history; {default_train}-day default train and 30-day disjoint prediction windows",
            "predictions": pd.DataFrame(predictions)}
