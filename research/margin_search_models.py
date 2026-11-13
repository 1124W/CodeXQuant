"""Two fixed three-class models with a two-basis-point execution dead zone.

The target is down / neutral / up over the next 30-minute index interval.
Neutral examples remain in training, including unchanged-price refunds. Raw
three-class scores are not calibrated probabilities of an event-contract win.
"""
from __future__ import annotations

from copy import deepcopy
import time
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES, MINUTE_MS, SYMBOLS

DAY_MS = 86_400_000
TRAIN_DAYS = 180
REFIT_DAYS = 30
MIN_TRAIN_ROWS = 2000
MARGIN_BPS = 2.
CLASSES = (-1, 0, 1)
CLASS_NAMES = {-1: 'down', 0: 'neutral', 1: 'up'}
_SPECS = {
    'margin_shallow': {'family': 'gbm_three_class', 'sample_minutes': 5,
        'parameters': {'max_iter': 100, 'max_depth': 2, 'min_samples_leaf': 300,
                       'l2_regularization': 10., 'learning_rate': .04,
                       'early_stopping': False, 'random_state': 17}},
    'margin_deep': {'family': 'gbm_three_class', 'sample_minutes': 5,
        'parameters': {'max_iter': 140, 'max_depth': 3, 'min_samples_leaf': 300,
                       'l2_regularization': 20., 'learning_rate': .035,
                       'early_stopping': False, 'random_state': 17}},
}


def specs():
    return deepcopy(_SPECS)


def make_model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if name not in _SPECS:
        raise ValueError('Unknown fixed margin model')
    return HistGradientBoostingClassifier(**_SPECS[name]['parameters'])


def margin_labels(entry, exit_price):
    """Strictly outside +/-2bp is directional; both exact boundaries are neutral."""
    opened = np.asarray(entry, dtype=float); closed = np.asarray(exit_price, dtype=float)
    if opened.ndim != 1 or closed.shape != opened.shape:
        raise ValueError('Entry/exit observations must be matching one-dimensional arrays')
    if (not np.isfinite(opened).all() or not np.isfinite(closed).all()
            or np.any(opened <= 0) or np.any(closed <= 0)):
        raise ValueError('Require finite positive historical entry/exit prices')
    margin = MARGIN_BPS / 10000
    return np.where(closed > opened*(1+margin), 1,
                    np.where(closed < opened*(1-margin), -1, 0)).astype(np.int8)


def _timestamps(values, label):
    values = pd.to_numeric(values, errors='raise').to_numpy()
    if not np.isfinite(values).all() or np.any(values != values.astype(np.int64)):
        raise ValueError(label + ' must be finite integer milliseconds')
    result = values.astype(np.int64)
    if np.any(result <= 0) or np.any(result % MINUTE_MS):
        raise ValueError(label + ' must be positive minute boundaries')
    return result


def _scope(train, test):
    for label, frame in [('training', train), ('prediction', test)]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError('Empty ' + label + ' population')
        if set(FEATURES + ['signal_ms', 'symbol']) - set(frame.columns):
            raise ValueError('Missing ' + label + ' features, signal clock or asset')
        if frame.symbol.nunique() != 1 or not frame.symbol.isin(SYMBOLS).all():
            raise ValueError('Each fit requires one BTC/ETH asset')
    if train.symbol.iloc[0] != test.symbol.iloc[0]:
        raise ValueError('Training and prediction assets differ')
    if set(['entry_ms', 'settlement_ms', 'entry', 'exit']) - set(train.columns):
        raise ValueError('Missing historical index execution observations')


def fit_predict(train, test, cutoff_ms, *, factories=None, progress=None):
    """Predict using only 60 observable features and a strict settled-label purge.

    Training prices outside the past 180 days or not settled strictly before
    cutoff are never inspected. Prediction execution/label columns are never
    accessed. Factories are test doubles for the two fixed model identifiers.
    """
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    if (isinstance(cutoff_ms, (bool, np.bool_)) or not np.isscalar(cutoff_ms)
            or not np.isfinite(cutoff_ms) or int(cutoff_ms) != cutoff_ms
            or cutoff_ms <= 0 or cutoff_ms % (30*MINUTE_MS)):
        raise ValueError('Fit cutoff must be an integer half-hour boundary')
    cutoff = int(cutoff_ms); _scope(train, test)
    train_time = _timestamps(train.signal_ms, 'Training signals')
    entry_time = _timestamps(train.entry_ms, 'Training entries')
    settle_time = _timestamps(train.settlement_ms, 'Training settlements')
    test_time = _timestamps(test.signal_ms, 'Prediction signals')
    if len(np.unique(train_time)) != len(train_time) or len(np.unique(test_time)) != len(test_time):
        raise ValueError('Duplicate event timestamps')
    if np.any(entry_time != train_time+MINUTE_MS) or np.any(settle_time != entry_time+30*MINUTE_MS):
        raise ValueError('Labels require one-minute entry delay and 30-minute expiry')
    if np.any(test_time < cutoff) or np.any(test_time >= cutoff+REFIT_DAYS*DAY_MS):
        raise ValueError('Prediction rows outside the frozen 30-day fold')
    historical = ((train_time >= cutoff-TRAIN_DAYS*DAY_MS) & (train_time < cutoff)
                  & (settle_time < cutoff))
    historical_positions = np.flatnonzero(historical)
    selected = historical_positions[train_time[historical_positions] % (5*MINUTE_MS) == 0]
    if len(selected) < MIN_TRAIN_ROWS:
        raise ValueError('Insufficient settled five-minute training rows')
    labels = margin_labels(train.entry.iloc[selected], train.exit.iloc[selected])
    if not np.array_equal(np.unique(labels), np.asarray(CLASSES)):
        raise ValueError('Training requires all down/neutral/up classes')
    matrix = train.iloc[selected, train.columns.get_indexer(FEATURES)].to_numpy(dtype=float)
    test_matrix = test[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(matrix).all(): raise ValueError('Nonfinite historical features')
    if not np.isfinite(test_matrix).all(): raise ValueError('Nonfinite prediction features')
    if factories is not None and set(factories) != set(_SPECS):
        raise ValueError('Test factories must supply exactly two margin models')
    output = pd.DataFrame(index=test.index); models = {}; began = time.perf_counter()
    for name, spec in _SPECS.items():
        model = factories[name]() if factories is not None else make_model(name)
        current = time.perf_counter()
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter('always', ConvergenceWarning)
            with threadpool_limits(limits=2):
                model.fit(matrix, labels)
                probabilities = np.asarray(model.predict_proba(test_matrix), dtype=float)
        convergence = [item for item in fit_warnings if issubclass(item.category, ConvergenceWarning)]
        for item in fit_warnings:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn(str(item.message), item.category, stacklevel=2)
        classes = np.asarray(getattr(model, 'classes_', []))
        if (classes.shape != (3,) or not np.isin(classes, CLASSES).all()
                or len(np.unique(classes)) != 3):
            raise ValueError('Models must provide exactly the down/neutral/up classes')
        if (probabilities.shape != (len(test), 3) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=1e-7, atol=1e-9)):
            raise ValueError('Invalid three-class probabilities')
        for label in (1, -1, 0):
            output['p_'+name+'_'+CLASS_NAMES[label]] = probabilities[:, np.flatnonzero(classes == label)[0]]
        iterations = getattr(model, 'n_iter_', None)
        models[name] = {'training_count': len(selected), 'sample_minutes': spec['sample_minutes'],
                        'class_order': classes.astype(int).tolist(),
                        'class_counts': {CLASS_NAMES[label]: int((labels == label).sum()) for label in CLASSES},
                        'margin_bps': MARGIN_BPS,
                        'latest_settlement_ms': int(settle_time[selected].max()),
                        'earliest_signal_ms': int(train_time[selected].min()),
                        'training_converged': not convergence, 'convergence_warning_count': len(convergence),
                        'iterations': np.atleast_1d(iterations).astype(int).tolist() if iterations is not None else None,
                        'runtime_seconds': round(time.perf_counter()-current, 4)}
        if progress:
            progress(f'{name}: {len(selected)} settled rows; {models[name]["runtime_seconds"]:.2f}s; converged={not convergence}')
        del model, probabilities
    metadata = {'fit_cutoff_ms': cutoff, 'train_days': TRAIN_DAYS, 'refit_days': REFIT_DAYS,
                'test_start_ms': int(test_time.min()), 'test_end_exclusive_ms': int(test_time.max())+MINUTE_MS,
                'prediction_count': len(test), 'features': FEATURES.copy(), 'specs': specs(), 'models': models,
                'source_population_rows': len(train),
                'historical_settled_population_rows': len(historical_positions),
                'historical_five_minute_population_rows': len(selected),
                'target': {'margin_bps': MARGIN_BPS, 'class_names': {str(k): v for k, v in CLASS_NAMES.items()},
                           'neutral_includes_exact_boundaries_and_refunds': True},
                'raw_scores_are_calibrated_win_probabilities': False,
                'local_research_only': True, 'runtime_seconds': round(time.perf_counter()-began, 4)}
    return output, metadata
