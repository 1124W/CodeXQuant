"""Fixed six-model chronological comparison, for local research only.

No live model is loaded or replaced here. All models see the same historical
60 features. Denser samples change observation weights and therefore use a
proportionally smaller logistic C; none of the scores is a calibrated win rate.
"""
from __future__ import annotations

from copy import deepcopy
import time
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES, MINUTE_MS, SYMBOLS

DAY_MS=86_400_000
TRAIN_DAYS=180
REFIT_DAYS=30
MIN_TRAIN_ROWS=2000
_SPECS={
    'logistic_halfhour':{'family':'logistic','sample_minutes':30,
        'parameters':{'C':.1,'max_iter':400,'solver':'lbfgs','random_state':17}},
    'logistic_5m':{'family':'logistic','sample_minutes':5,
        'parameters':{'C':.1*5/30,'max_iter':400,'solver':'lbfgs','random_state':17}},
    'logistic_1m':{'family':'logistic','sample_minutes':1,
        'parameters':{'C':.1/30,'max_iter':400,'solver':'lbfgs','random_state':17}},
    'gbm_shallow_5m':{'family':'gbm','sample_minutes':5,
        'parameters':{'max_iter':100,'max_depth':2,'min_samples_leaf':300,
                      'l2_regularization':10.,'learning_rate':.04,
                      'early_stopping':False,'random_state':17}},
    'gbm_deeper_5m':{'family':'gbm','sample_minutes':5,
        'parameters':{'max_iter':140,'max_depth':3,'min_samples_leaf':300,
                      'l2_regularization':20.,'learning_rate':.035,
                      'early_stopping':False,'random_state':17}},
    'ensemble_5m':{'family':'mean','members':['logistic_5m','gbm_shallow_5m'],
                   'weights':[.5,.5]},
}


def specs():
    """A copied finite catalog, never a mutable parameter search grid."""
    return deepcopy(_SPECS)


def make_model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name not in _SPECS:raise ValueError('Unknown fixed model')
    spec=_SPECS[name]
    if spec['family']=='logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(**spec['parameters']))
    if spec['family']=='gbm':return HistGradientBoostingClassifier(**spec['parameters'])
    raise ValueError('The ensemble averages independently fitted component scores')


def _timestamps(values,label):
    values=pd.to_numeric(values,errors='raise').to_numpy()
    if not np.isfinite(values).all() or np.any(values!=values.astype(np.int64)):
        raise ValueError(f'{label} must be finite integer milliseconds')
    result=values.astype(np.int64)
    if np.any(result<=0) or np.any(result%MINUTE_MS):
        raise ValueError(f'{label} must be positive minute boundaries')
    return result


def _validate_scope(train,test):
    for name,frame in [('train',train),('test',test)]:
        if not isinstance(frame,pd.DataFrame) or frame.empty:raise ValueError(f'Empty {name} population')
        if set(FEATURES+['signal_ms'])-set(frame.columns):raise ValueError(f'Missing {name} features or event time')
        if 'symbol' in frame and (frame.symbol.nunique()!=1 or not frame.symbol.isin(SYMBOLS).all()):
            raise ValueError('Each fit must contain exactly one BTC/ETH asset')
    if 'symbol' in train and 'symbol' in test and train.symbol.iloc[0]!=test.symbol.iloc[0]:
        raise ValueError('Training and prediction assets differ')
    if set(['settlement_ms','y','is_flat'])-set(train.columns):raise ValueError('Missing settled training labels')


def fit_predict(train,test,cutoff_ms,*,factories=None,progress=None):
    """Fit five fixed models sequentially, then their fixed-weight ensemble.

    ``train`` may include future rows: they are purged before feature/label
    validation and fitting. ``test`` is one original30-day chronological fold;
    target/execution columns in it are never consulted. Factories are test
    doubles keyed by all five trained model IDs; they cannot change the grid.
    Returns a probability-only frame preserving the test index plus metadata.
    """
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    if (isinstance(cutoff_ms,(bool,np.bool_)) or not np.isscalar(cutoff_ms)
            or not np.isfinite(cutoff_ms) or int(cutoff_ms)!=cutoff_ms
            or cutoff_ms<=0 or cutoff_ms%(30*MINUTE_MS)):
        raise ValueError('Fit cutoff must be an integer half-hour boundary')
    cutoff=int(cutoff_ms)
    _validate_scope(train,test)
    train_time=_timestamps(train.signal_ms,'Training signals')
    settle_time=_timestamps(train.settlement_ms,'Training settlements')
    test_time=_timestamps(test.signal_ms,'Prediction signals')
    if len(np.unique(train_time))!=len(train_time) or len(np.unique(test_time))!=len(test_time):
        raise ValueError('Duplicate event timestamps')
    if np.any(settle_time!=train_time+31*MINUTE_MS):raise ValueError('Labels require entry+1minute and30minute expiry')
    if np.any(test_time<cutoff) or np.any(test_time>=cutoff+REFIT_DAYS*DAY_MS):
        raise ValueError('Prediction rows outside the frozen30-day fold')
    historical=(train_time>=cutoff-TRAIN_DAYS*DAY_MS)&(train_time<cutoff)&(settle_time<cutoff)
    # Purge outcomes before inspecting their values, including the event whose
    # settlement lies exactly on the fit boundary.
    history=train.loc[historical,['signal_ms','settlement_ms','y','is_flat']]
    if not history.y.isin([0,1]).all() or not history.is_flat.isin([False,True]).all():
        raise ValueError('Invalid historical settled labels')
    if ((history.is_flat.astype(bool))&(history.y!=0)).any():raise ValueError('Flat outcomes cannot be up labels')
    base_positions=np.flatnonzero(historical)
    base_positions=base_positions[~history.is_flat.to_numpy(dtype=bool)]
    test_matrix=test[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(test_matrix).all():raise ValueError('Nonfinite prediction features')
    trained_ids=[key for key,spec in _SPECS.items() if spec['family']!='mean']
    if factories is not None and set(factories)!=set(trained_ids):
        raise ValueError('Test factories must provide exactly the five trained models')
    output=pd.DataFrame(index=test.index);model_meta={};started=time.perf_counter()
    for name in trained_ids:
        spec=_SPECS[name]
        selected=base_positions[train_time[base_positions]%(spec['sample_minutes']*MINUTE_MS)==0]
        if len(selected)<MIN_TRAIN_ROWS:raise ValueError(f'{name}: insufficient settled training rows')
        labels=train.y.iloc[selected].to_numpy(dtype=np.int8)
        if np.unique(labels).size!=2:raise ValueError(f'{name}: training requires both directions')
        matrix=train.iloc[selected,train.columns.get_indexer(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite historical features')
        model=factories[name]() if factories is not None else make_model(name)
        current=time.perf_counter()
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter('always',ConvergenceWarning)
            with threadpool_limits(limits=2):
                model.fit(matrix,labels)
                values=np.asarray(model.predict_proba(test_matrix),dtype=float)
        convergence_warnings=[item for item in fit_warnings if issubclass(item.category,ConvergenceWarning)]
        for item in fit_warnings:
            if not issubclass(item.category,ConvergenceWarning):
                warnings.warn(str(item.message),item.category,stacklevel=2)
        if (values.shape!=(len(test),2) or not np.isfinite(values).all()
                or np.any(values<0) or np.any(values>1)
                or not np.allclose(values.sum(axis=1),1.,rtol=1e-7,atol=1e-9)):
            raise ValueError('Invalid directional probabilities')
        classes=getattr(model,'classes_',None)
        if classes is not None and not np.array_equal(classes,np.array([0,1])):
            raise ValueError('Models must provide ordered down/up classes')
        output['p_'+name]=values[:,1]
        estimator=model[-1] if hasattr(model,'named_steps') else model
        iterations=getattr(estimator,'n_iter_',None)
        model_meta[name]={'training_count':len(selected),'sample_minutes':spec['sample_minutes'],
                          'latest_settlement_ms':int(settle_time[selected].max()),
                          'earliest_signal_ms':int(train_time[selected].min()),
                          'training_converged':not convergence_warnings,
                          'convergence_warning_count':len(convergence_warnings),
                          'iterations':np.atleast_1d(iterations).astype(int).tolist() if iterations is not None else None,
                          'runtime_seconds':round(time.perf_counter()-current,4)}
        if progress:progress(f'{name}: {len(selected)} settled rows; {model_meta[name]["runtime_seconds"]:.2f}s; converged={not convergence_warnings}')
        del model,matrix,labels,values
    ensemble=_SPECS['ensemble_5m']
    output['p_ensemble_5m']=sum(weight*output['p_'+member] for member,weight in zip(ensemble['members'],ensemble['weights']))
    metadata={'fit_cutoff_ms':cutoff,'train_days':TRAIN_DAYS,'refit_days':REFIT_DAYS,
              'test_start_ms':int(test_time.min()),'test_end_exclusive_ms':int(test_time.max())+MINUTE_MS,
              'prediction_count':len(test),'features':FEATURES.copy(),'specs':specs(),'models':model_meta,
              'source_population_rows':len(train),'historical_decisive_population_rows':len(base_positions),
              'raw_scores_are_calibrated_win_probabilities':False,'local_research_only':True,
              'runtime_seconds':round(time.perf_counter()-started,4)}
    return output,metadata
