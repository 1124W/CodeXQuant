"""Paired 60-feature controls and 68-feature metrics models on identical rows.

The two architectures and parameters are fixed before results. Each augmented
fit and its control share the exact asset, signal timestamps, labels, stride,
purge and evaluation population; only the eight additional inputs differ.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES
from .event_engine import DAY_MS,MINUTE_MS,SYMBOLS
from .metrics_features import METRIC_FEATURES,AUGMENTED_FEATURES,PROVENANCE_COLUMNS,GRID_MS,AVAILABILITY_DELAY_MS,MAX_AVAILABLE_AGE_MS

TRAIN_DAYS=180
REFIT_DAYS=30
MIN_TRAIN_ROWS=2000
MODEL_IDS=('logistic_base','logistic_aug','gbm_base','gbm_aug')
_SPECS={
    'logistic':{'family':'logistic','sample_minutes':30,
        'parameters':{'C':.1,'max_iter':400,'solver':'lbfgs','random_state':17}},
    'gbm':{'family':'gbm','sample_minutes':5,
        'parameters':{'max_iter':140,'max_depth':3,'min_samples_leaf':300,
            'l2_regularization':20.,'learning_rate':.035,'early_stopping':False,'random_state':17}}}


def specs():return deepcopy(_SPECS)


def make_model(family):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    if family not in _SPECS:raise ValueError('Unknown fixed paired model family')
    params=_SPECS[family]['parameters']
    return (make_pipeline(StandardScaler(),LogisticRegression(**params)) if family=='logistic'
            else HistGradientBoostingClassifier(**params))


def _times(frame,name):
    if name not in frame:raise ValueError('Missing event clock: '+name)
    values=pd.to_numeric(frame[name],errors='raise').to_numpy()
    if (not np.isfinite(values).all() or np.any(values!=np.floor(values))
            or np.any(values<=0) or np.any(values%MINUTE_MS)):
        raise ValueError('Require positive integer minute timestamps: '+name)
    return values.astype(np.int64)


def _scope(train,test):
    for label,frame in (('training',train),('prediction',test)):
        if not isinstance(frame,pd.DataFrame) or frame.empty or set(AUGMENTED_FEATURES+PROVENANCE_COLUMNS+['signal_ms','symbol'])-set(frame):
            raise ValueError('Missing '+label+' paired features or scope')
        if frame.symbol.nunique()!=1 or not frame.symbol.isin(SYMBOLS).all():
            raise ValueError('Each paired fit requires one BTC/ETH asset')
    if train.symbol.iloc[0]!=test.symbol.iloc[0]:raise ValueError('Training/prediction assets differ')
    if set(['entry_ms','settlement_ms','entry','exit'])-set(train):
        raise ValueError('Missing historical entry/settlement observations')


def _sha(values,dtype):return hashlib.sha256(np.asarray(values,dtype=dtype).tobytes()).hexdigest()


def _availability(frame):
    values=frame[PROVENANCE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values!=np.floor(values)):
        raise ValueError('Invalid metrics availability provenance')
    recorded,available,age=values.T
    if (np.any(recorded<=0) or np.any(recorded%GRID_MS)
            or np.any(available!=recorded+AVAILABILITY_DELAY_MS)
            or np.any(age!=frame.signal_ms.to_numpy()-available)
            or np.any(age<0) or np.any(age>=MAX_AVAILABLE_AGE_MS)):
        raise ValueError('Metrics must be historically available and fresh')


def fit_predict(train,test,cutoff_ms,*,factories=None,progress=None):
    """Return four score columns and paired train-row/label hash evidence."""
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    if (isinstance(cutoff_ms,(bool,np.bool_)) or not np.isscalar(cutoff_ms)
            or not np.isfinite(cutoff_ms) or int(cutoff_ms)!=cutoff_ms
            or cutoff_ms<=0 or cutoff_ms%(30*MINUTE_MS)):
        raise ValueError('Fit cutoff must be a positive halfhour boundary')
    cutoff=int(cutoff_ms);_scope(train,test)
    stamps=_times(train,'signal_ms');entries=_times(train,'entry_ms');expiry=_times(train,'settlement_ms')
    test_stamps=_times(test,'signal_ms')
    if len(np.unique(stamps))!=len(stamps) or len(np.unique(test_stamps))!=len(test_stamps):
        raise ValueError('Duplicate asset/minute samples')
    if np.any(entries!=stamps+MINUTE_MS) or np.any(expiry!=entries+30*MINUTE_MS):
        raise ValueError('Require one-minute entry delay and30minute expiry')
    if np.any(test_stamps<cutoff) or np.any(test_stamps>=cutoff+REFIT_DAYS*DAY_MS):
        raise ValueError('Prediction rows outside frozen30day fold')
    if factories is not None and set(factories)!=set(MODEL_IDS):
        raise ValueError('Factories must provide the four fixed paired models')
    historical=(stamps>=cutoff-TRAIN_DAYS*DAY_MS)&(expiry<cutoff)
    _availability(test)
    testing_aug=test[AUGMENTED_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(testing_aug).all():raise ValueError('Nonfinite paired prediction features')
    output=pd.DataFrame(index=test.index);models={};pairs={};began=time.perf_counter()
    for family,spec in _SPECS.items():
        positions=np.flatnonzero(historical&(stamps%(spec['sample_minutes']*MINUTE_MS)==0))
        opened=pd.to_numeric(train.entry.iloc[positions],errors='raise').to_numpy(dtype=float)
        closed=pd.to_numeric(train.exit.iloc[positions],errors='raise').to_numpy(dtype=float)
        if (not np.isfinite(opened).all() or not np.isfinite(closed).all()
                or np.any(opened<=0) or np.any(closed<=0)):
            raise ValueError('Invalid selected historical index prices')
        decisive=opened!=closed;positions=positions[decisive];target=(closed[decisive]>opened[decisive]).astype(np.int8)
        if len(positions)<MIN_TRAIN_ROWS or len(np.unique(target))!=2:
            raise ValueError('Insufficient paired settled decisive training population')
        _availability(train.iloc[positions])
        training_aug=train.iloc[positions][AUGMENTED_FEATURES].to_numpy(dtype=float)
        if not np.isfinite(training_aug).all():raise ValueError('Nonfinite paired training features')
        signal_hash=_sha(stamps[positions],'<i8');label_hash=_sha(target,'i1')
        pair={'training_count':len(positions),'sample_minutes':spec['sample_minutes'],
              'train_signal_sha256':signal_hash,'train_label_sha256':label_hash,
              'test_signal_sha256':_sha(test_stamps,'<i8'),'prediction_count':len(test),
              'base_feature_count':len(FEATURES),'augmented_feature_count':len(AUGMENTED_FEATURES),
              'identical_train_rows_and_labels':True,'identical_test_rows':True}
        for arm in ('base','aug'):
            name=family+'_'+arm
            width=len(FEATURES) if arm=='base' else len(AUGMENTED_FEATURES)
            matrix=training_aug[:,:width];testing=testing_aug[:,:width]
            model=factories[name]() if factories is not None else make_model(family)
            current=time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always',ConvergenceWarning)
                with threadpool_limits(limits=2):
                    model.fit(matrix,target);probability=np.asarray(model.predict_proba(testing),dtype=float)
            convergence=[w for w in captured if issubclass(w.category,ConvergenceWarning)]
            for warning in captured:
                if not issubclass(warning.category,ConvergenceWarning):warnings.warn(str(warning.message),warning.category,stacklevel=2)
            classes=np.asarray(getattr(model,'classes_',[]))
            if classes.shape!=(2,) or set(classes)!={0,1}:raise ValueError('Require binary down/up model classes')
            if (probability.shape!=(len(test),2) or not np.isfinite(probability).all()
                    or np.any(probability<0) or np.any(probability>1)
                    or not np.allclose(probability.sum(axis=1),1.,rtol=1e-7,atol=1e-9)):
                raise ValueError('Invalid paired directional scores')
            output['p_'+name]=probability[:,int(np.flatnonzero(classes==1)[0])]
            estimator=model[-1] if hasattr(model,'steps') else model
            iterations=getattr(estimator,'n_iter_',None)
            record={'family':family,'feature_set':arm,'training_count':len(positions),
                    'sample_minutes':spec['sample_minutes'],'features':FEATURES.copy() if arm=='base' else AUGMENTED_FEATURES.copy(),
                    'train_signal_sha256':signal_hash,'train_label_sha256':label_hash,
                    'test_signal_sha256':pair['test_signal_sha256'],'prediction_count':len(test),
                    'latest_settlement_ms':int(expiry[positions].max()),'earliest_signal_ms':int(stamps[positions].min()),
                    'training_converged':not convergence,'convergence_warning_count':len(convergence),
                    'iterations':np.atleast_1d(iterations).astype(int).tolist() if iterations is not None else None,
                    'runtime_seconds':round(time.perf_counter()-current,4)}
            models[name]=record
            pair[arm]={'training_count':len(positions),'train_signal_sha256':signal_hash,'train_label_sha256':label_hash,
                       'test_signal_sha256':pair['test_signal_sha256'],'prediction_count':len(test)}
            if progress:progress(f'{name}: {len(positions)} pairedrows; {record["runtime_seconds"]:.2f}s; converged={not convergence}')
            del model,probability
        pairs[family]=pair
    metadata={'fit_cutoff_ms':cutoff,'train_days':TRAIN_DAYS,'refit_days':REFIT_DAYS,
        'test_start_ms':int(test_stamps.min()),'test_end_exclusive_ms':int(test_stamps.max())+MINUTE_MS,
        'prediction_count':len(test),'symbol':str(test.symbol.iloc[0]),'specs':specs(),'models':models,'pairs':pairs,
        'source_population_rows':len(train),'base_features':FEATURES.copy(),'added_features':METRIC_FEATURES.copy(),
        'raw_scores_are_calibrated_win_probabilities':False,'local_research_only':True,
        'runtime_seconds':round(time.perf_counter()-began,4)}
    return output,metadata
