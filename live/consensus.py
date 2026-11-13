"""Fixed two-model BTC/ETH research alerts; no broker or order operations.

Only locally generated joblib bundles are supported. Manifest hashes detect
corruption; they do not establish trust in an outside model publisher.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .features import FEATURES,MINUTE_MS,SYMBOLS,build_latest_features

DAY_MS=86_400_000
RULE_ID='pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
MODEL_IDS=('logistic_1m','gbm_deeper_5m')
THRESHOLD=.60
ROW_COLUMNS=FEATURES+['signal_ms','settlement_ms','y','is_flat']
MIN_DENSE_COVERAGE=.99
MIN_DAILY_COVERAGE=.90
MAX_MISSING_MINUTES=180
_SPECS={
    'logistic_1m':{'family':'logistic','sample_minutes':1,
        'parameters':{'C':.1/30,'max_iter':400,'solver':'lbfgs','random_state':17}},
    'gbm_deeper_5m':{'family':'gbm','sample_minutes':5,
        'parameters':{'max_iter':140,'max_depth':3,'min_samples_leaf':300,
            'l2_regularization':20.,'learning_rate':.035,'early_stopping':False,'random_state':17}},
}


def specs():return deepcopy(_SPECS)


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('features.py').read_bytes()).hexdigest()


def _minute(value):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
            or not math.isfinite(value) or value!=int(value) or value<=0 or value%MINUTE_MS):
        raise ValueError('Require a positive integer minute timestamp')
    return int(value)


def decision(scores,signal_ms):
    """The historical pair gate, without exhaustion or any new filter."""
    stamp=_minute(signal_ms)
    if not isinstance(scores,dict) or set(scores)!=set(MODEL_IDS):raise ValueError('Require both fixed model scores')
    clean={}
    for name,value in scores.items():
        if isinstance(value,(bool,np.bool_)):raise ValueError('Invalid directional model score')
        p=float(value)
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid directional model score')
        clean[name]=p
    directions={name:1 if p>.5 else -1 if p<.5 else 0 for name,p in clean.items()}
    confidence={name:max(p,1-p) for name,p in clean.items()}
    member_reasons={name:('neutral_model_direction' if directions[name]==0 else
        'model_score_below_0.60' if confidence[name]<THRESHOLD else 'qualified_direction') for name in MODEL_IDS}
    direction=directions[MODEL_IDS[0]];reason='signal'
    if 0 in directions.values():reason='neutral_model_direction'
    elif len(set(directions.values()))!=1:reason='model_directions_disagree'
    elif min(confidence.values())<THRESHOLD:reason='consensus_score_below_0.60'
    active=reason=='signal'
    return {'side':('LONG' if direction==1 else 'SHORT') if active else 'NONE',
        'direction':direction if active else 0,'p_up':sum(clean.values())/2,
        'confidence_score':min(confidence.values()),'model_scores':clean,
        'member_confidence_scores':confidence,'member_reasons':member_reasons,
        'score_is_calibrated_win_probability':False,'reason':reason,
        'signal_ms':stamp,'feature_close_ms':stamp-MINUTE_MS,
        'entry_due_ms':stamp+MINUTE_MS,'expiry_due_ms':stamp+31*MINUTE_MS,
        'horizon_minutes':30,'rule_id':RULE_ID,'research_only':True,'order_execution_enabled':False}


def make_model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name not in MODEL_IDS:raise ValueError('Unknown fixed member')
    spec=_SPECS[name]
    if spec['family']=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(**spec['parameters']))
    return HistGradientBoostingClassifier(**spec['parameters'])


def _check_model(model,name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if name=='logistic_1m':
        if (not isinstance(model,Pipeline) or len(model.steps)!=2
                or not isinstance(model[0],StandardScaler) or not isinstance(model[1],LogisticRegression)):
            raise ValueError('Wrong logistic model type')
        estimator=model[1]
        if not model[0].with_mean or not model[0].with_std:raise ValueError('Wrong feature scaling')
    else:
        if not isinstance(model,HistGradientBoostingClassifier):raise ValueError('Wrong GBM model type')
        estimator=model
    actual=estimator.get_params()
    if any(actual.get(key)!=value for key,value in _SPECS[name]['parameters'].items()):
        raise ValueError('Model parameters differ from fixed research specification')
    if getattr(model,'n_features_in_',None)!=len(FEATURES) or not np.array_equal(getattr(model,'classes_',None),[0,1]):
        raise ValueError('Wrong model features or class order')


class ConsensusPredictor:
    def __init__(self,model_bundle_directory,max_model_age_days=35):
        import joblib
        import sklearn
        self.directory=Path(model_bundle_directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        m=self.manifest
        if (m.get('schema_version')!=2 or m.get('features')!=FEATURES or m.get('rule_id')!=RULE_ID
                or m.get('model_specs')!=specs() or m.get('training_days')!=180
                or m.get('training_sampling_minutes')!=1 or m.get('member_sampling_minutes')!={'logistic_1m':1,'gbm_deeper_5m':5}
                or m.get('threshold')!=THRESHOLD or m.get('regime')!='unrestricted'
                or m.get('order_execution_enabled') is not False or set(m.get('models',{}))!=set(SYMBOLS)):
            raise ValueError('Manifest differs from the fixed consensus contract')
        if m.get('sklearn_version')!=sklearn.__version__:raise ValueError('Require the exact bundle scikit-learn version')
        cutoff=_minute(m.get('training_cutoff_ms'))
        if cutoff%(30*MINUTE_MS):raise ValueError('Invalid model training cutoff')
        if not re.fullmatch('[a-f0-9]{20}',m.get('model_version','')):raise ValueError('Invalid model identity')
        self.max_model_age_days=max_model_age_days;self.models={}
        for symbol in SYMBOLS:
            entry=m['models'][symbol]
            if set(entry.get('members',{}))!=set(MODEL_IDS):raise ValueError('Missing consensus member')
            if (entry.get('training_sampling_minutes')!=1 or entry.get('training_samples_file')!=f'{symbol}-training.csv.gz'
                    or entry.get('training_samples_rows',0)<int(.99*(180*1440-31))
                    or not re.fullmatch('[a-f0-9]{64}',entry.get('training_samples_sha256',''))):
                raise ValueError('Consensus requires a dense verified training seed')
            self.models[symbol]={}
            for name in MODEL_IDS:
                item=entry['members'][name]
                if item.get('file')!=f'{symbol}-{name}.joblib':raise ValueError('Unexpected model filename')
                if (item.get('training_converged') is not True or item.get('sample_minutes')!=_SPECS[name]['sample_minutes']
                        or not 0<item.get('latest_settlement_ms',cutoff)<cutoff or item.get('training_count',0)<2000):
                    raise ValueError('Unconverged or noncausal consensus model')
                path=self.directory/item['file']
                if path.is_symlink() or _sha(path)!=item.get('sha256'):raise ValueError('Model checksum mismatch')
                model=joblib.load(path);_check_model(model,name);self.models[symbol][name]=model

    def predict(self,features_by_symbol,signal_ms):
        from threadpoolctl import threadpool_limits
        stamp=_minute(signal_ms);cutoff=int(self.manifest['training_cutoff_ms'])
        if set(features_by_symbol)!=set(SYMBOLS):raise ValueError('Both BTC and ETH feature rows are required')
        if stamp<cutoff:raise ValueError('A model cannot predict before its training cutoff')
        if self.max_model_age_days is not None and stamp-cutoff>self.max_model_age_days*DAY_MS:
            raise ValueError('Model exceeds its declared maximum age; retrain before alerts resume')
        result=[]
        for symbol in SYMBOLS:
            row=features_by_symbol[symbol]
            if set(row)!=set(FEATURES) or any(isinstance(value,(bool,np.bool_)) for value in row.values()):
                raise ValueError('Feature names/values differ from the frozen model contract')
            vector=np.asarray([[row[name] for name in FEATURES]],dtype=float)
            if not np.isfinite(vector).all():raise ValueError('Nonfinite prediction feature vector')
            scores={}
            with threadpool_limits(limits=2):
                for name,model in self.models[symbol].items():
                    values=np.asarray(model.predict_proba(vector),dtype=float)
                    if values.shape!=(1,2) or not np.isfinite(values).all() or not np.isclose(values.sum(),1):
                        raise ValueError('Invalid model probability result')
                    scores[name]=float(values[0,1])
            output=decision(scores,stamp);output.update(symbol=symbol,model_training_cutoff_ms=cutoff,
                model_version=self.manifest['model_version']);result.append(output)
        return result

    def predict_market(self,spot_by_symbol,premium_by_symbol,funding_by_symbol,signal_ms):
        return self.predict(build_latest_features(spot_by_symbol,premium_by_symbol,funding_by_symbol,signal_ms),signal_ms)


def prepare_training_rows(incoming,cutoff_ms):
    """Require a dense 180-day population, then purge unsettled outcomes first."""
    cutoff=_minute(cutoff_ms);start=cutoff-180*DAY_MS
    if cutoff%(30*MINUTE_MS):raise ValueError('Training cutoff must be a half-hour boundary')
    frame=pd.DataFrame(incoming)
    if set(ROW_COLUMNS)-set(frame):raise ValueError('Missing training feature or label columns')
    clock=frame[['signal_ms','settlement_ms']].to_numpy(dtype=float)
    if (not np.isfinite(clock).all() or np.any(clock!=np.floor(clock)) or np.any(clock%MINUTE_MS)
            or not (frame.settlement_ms==frame.signal_ms+31*MINUTE_MS).all()):
        raise ValueError('Invalid training event clocks')
    d=frame.loc[(frame.signal_ms>=start)&(frame.signal_ms<cutoff)&(frame.settlement_ms<cutoff),ROW_COLUMNS]
    if not d.signal_ms.is_monotonic_increasing:d=d.sort_values('signal_ms')
    d.reset_index(drop=True,inplace=True)
    if (d.empty or d.signal_ms.duplicated().any() or not d.y.isin([0,1]).all()
            or not d.is_flat.isin([False,True]).all() or (d.is_flat.astype(bool)&(d.y!=0)).any()
            or any(not np.isfinite(d[name].to_numpy(dtype=float)).all() for name in FEATURES)):
        raise ValueError('Invalid settled training population')
    d=d.astype({'signal_ms':'int64','settlement_ms':'int64','y':'int8','is_flat':'bool'},copy=False)
    expected=np.arange(start,cutoff-31*MINUTE_MS,MINUTE_MS,dtype=np.int64)
    observed=d.signal_ms.to_numpy(dtype=np.int64)
    coverage=len(d)/len(expected)
    counts=pd.Series(observed//DAY_MS).value_counts()
    expected_counts=pd.Series(expected//DAY_MS).value_counts()
    daily=min(int(counts.get(day,0))/int(n) for day,n in expected_counts.items())
    gap=max(0,int(np.diff(np.concatenate(([start-MINUTE_MS],observed,[cutoff-31*MINUTE_MS]))).max()//MINUTE_MS)-1)
    if coverage<MIN_DENSE_COVERAGE or daily<MIN_DAILY_COVERAGE or gap>MAX_MISSING_MINUTES:
        raise ValueError('Consensus requires dense 180-day minute history; sparse/alert-only seeds are forbidden')
    return d,{'expected_minutes':len(expected),'observed_minutes':len(d),'coverage':coverage,
        'minimum_daily_coverage':daily,'maximum_missing_run_minutes':gap,'sampling_minutes':1}


def retrain_from_rows(output_dir,rows_by_symbol,cutoff_ms,progress=print,*,data_sources=None):
    """Train the fixed pair using representative dense past data; never tune it."""
    import joblib
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    if set(rows_by_symbol)!=set(SYMBOLS):raise ValueError('Require exactly BTC/ETH training populations')
    cutoff=_minute(cutoff_ms);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Write a new model directory; existing bundles are immutable')
    out.mkdir(parents=True,exist_ok=True)
    models={};digest=hashlib.sha256()
    for symbol in SYMBOLS:
        frame,coverage=prepare_training_rows(rows_by_symbol[symbol],cutoff)
        digest.update(symbol.encode());digest.update(pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype='<u8').tobytes())
        members={}
        for name,spec in _SPECS.items():
            selected=np.flatnonzero((~frame.is_flat)&(frame.signal_ms%(spec['sample_minutes']*MINUTE_MS)==0))
            labels=frame.y.iloc[selected].to_numpy(dtype=np.int8)
            if len(selected)<2000 or np.unique(labels).size!=2:raise ValueError('Insufficient decisive training labels')
            matrix=frame.iloc[selected,frame.columns.get_indexer(FEATURES)].to_numpy(dtype=float,copy=False)
            model=make_model(name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                with threadpool_limits(limits=2):model.fit(matrix,labels)
            if any(issubclass(item.category,ConvergenceWarning) for item in caught):
                raise ValueError('Fixed consensus member did not converge; no parameter adjustment permitted')
            _check_model(model,name)
            path=out/f'{symbol}-{name}.joblib';joblib.dump(model,path,compress=3)
            estimator=model[-1] if hasattr(model,'named_steps') else model
            members[name]={'file':path.name,'sha256':_sha(path),'sample_minutes':spec['sample_minutes'],
                'training_count':len(selected),'latest_settlement_ms':int(frame.settlement_ms.iloc[selected].max()),
                'earliest_signal_ms':int(frame.signal_ms.iloc[selected].min()),'training_converged':True,
                'iterations':np.atleast_1d(getattr(estimator,'n_iter_',[])).astype(int).tolist()}
            progress(f'{symbol} {name}: {len(selected):,} strictly settled rows; converged')
            del model,matrix,labels,selected
        seed=out/f'{symbol}-training.csv.gz'
        frame.to_csv(seed,index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
        models[symbol]={'members':members,'training_samples_file':seed.name,'training_samples_sha256':_sha(seed),
            'training_samples_rows':len(frame),'training_sampling_minutes':1,'dense_coverage':coverage,
            'training_count':members['logistic_1m']['training_count'],
            'latest_settlement_ms':max(item['latest_settlement_ms'] for item in members.values())}
        progress(f'{symbol}: dense seed exported ({seed.stat().st_size:,} bytes)')
    source=_source_hash();identity={'training_cutoff_ms':cutoff,'data_sha256':digest.hexdigest(),'source_sha256':source,
        'models':{symbol:{name:item['sha256'] for name,item in models[symbol]['members'].items()} for symbol in SYMBOLS}}
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    manifest={'schema_version':2,'rule_id':RULE_ID,'features':FEATURES,'model_specs':specs(),'threshold':THRESHOLD,
        'regime':'unrestricted','model_version':version,'training_days':180,'training_cutoff_ms':cutoff,
        'training_start_utc':datetime.fromtimestamp((cutoff-180*DAY_MS)/1000,timezone.utc).date().isoformat(),
        'training_cutoff_utc':datetime.fromtimestamp(cutoff/1000,timezone.utc).date().isoformat(),
        'training_sampling_minutes':1,'member_sampling_minutes':{'logistic_1m':1,'gbm_deeper_5m':5},
        'supported_inference_sampling_minutes':[1],'default_max_model_age_days':35,'sklearn_version':sklearn.__version__,
        'models':models,'source_sha256':source,'data_sha256':identity['data_sha256'],'public_data_manifests':data_sources or {},
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),'score_is_calibrated_win_probability':False,
        'research_only':True,'historical_stable_58pct_demonstrated':False,'order_execution_enabled':False}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    ConsensusPredictor(out)
    return manifest
