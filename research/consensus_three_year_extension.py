"""Isolated early extension of the fixed Logistic-1m + GBM-5m consensus.

Only the two existing model specifications are fitted. Existing stores, scores,
reports and services are read-only. All additional observations are official
Binance archives with CHECKSUMs, or explicitly identified official REST repairs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date,datetime,timedelta,timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES,MINUTE_MS,SYMBOLS
from . import model_search_data as feature_data
from . import model_search_models as original_models
from . import qd_three_year_data as market_data
from . import derivatives_data
from .event_engine import DAY_MS,utc_ms
from .high_win_research import file_hash,write_json
from .minute_scan_research import cutoff_for
from .model_search_research import KEEP

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'research'/'.cache'/'consensus-three-year-extension'
MODEL_IDS=('logistic_1m','gbm_deeper_5m')
START=date(2023,11,12)
END=date(2024,5,7)
FEATURE_START=cutoff_for(START)-timedelta(days=180)
# Preserve executions crossing the join with existing predictions on May 7.
FEATURE_END=END+timedelta(days=1)
COLUMNS=KEEP+['p_'+name for name in MODEL_IDS]+['model_fit_cutoff_ms','train_latest_settlement_ms']


def selected_specs():
    catalog=original_models.specs()
    return {name:catalog[name] for name in MODEL_IDS}


def fit_pair(train,test,cutoff_ms,*,factories=None,progress=None):
    """Exactly the original sample/purge/model procedure for two fixed members."""
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    if (isinstance(cutoff_ms,(bool,np.bool_)) or not np.isscalar(cutoff_ms)
            or not np.isfinite(cutoff_ms) or int(cutoff_ms)!=cutoff_ms
            or cutoff_ms<=0 or cutoff_ms%(30*MINUTE_MS)):
        raise ValueError('Fit cutoff must be an integer half-hour boundary')
    cutoff=int(cutoff_ms)
    original_models._validate_scope(train,test)
    train_time=original_models._timestamps(train.signal_ms,'Training signals')
    settle_time=original_models._timestamps(train.settlement_ms,'Training settlements')
    test_time=original_models._timestamps(test.signal_ms,'Prediction signals')
    if len(np.unique(train_time))!=len(train_time) or len(np.unique(test_time))!=len(test_time):
        raise ValueError('Duplicate event timestamps')
    if np.any(settle_time!=train_time+31*MINUTE_MS):raise ValueError('Incorrect training event clock')
    if np.any(test_time<cutoff) or np.any(test_time>=cutoff+30*DAY_MS):
        raise ValueError('Prediction rows outside the frozen 30-day fold')
    historical=(train_time>=cutoff-180*DAY_MS)&(train_time<cutoff)&(settle_time<cutoff)
    history=train.loc[historical,['signal_ms','settlement_ms','y','is_flat']]
    if not history.y.isin([0,1]).all() or not history.is_flat.isin([False,True]).all():
        raise ValueError('Invalid historical settled labels')
    if ((history.is_flat.astype(bool))&(history.y!=0)).any():raise ValueError('Invalid flat outcome')
    positions=np.flatnonzero(historical)
    positions=positions[~history.is_flat.to_numpy(dtype=bool)]
    test_matrix=test[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(test_matrix).all():raise ValueError('Nonfinite prediction features')
    if factories is not None and set(factories)!=set(MODEL_IDS):
        raise ValueError('Factories must provide exactly the two fixed members')
    output=pd.DataFrame(index=test.index);model_meta={};started=time.perf_counter()
    for name,spec in selected_specs().items():
        selected=positions[train_time[positions]%(spec['sample_minutes']*MINUTE_MS)==0]
        if len(selected)<original_models.MIN_TRAIN_ROWS:raise ValueError('Insufficient settled rows: '+name)
        labels=train.y.iloc[selected].to_numpy(dtype=np.int8)
        if np.unique(labels).size!=2:raise ValueError('Training requires both directions')
        matrix=train.iloc[selected,train.columns.get_indexer(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite historical features')
        model=factories[name]() if factories is not None else original_models.make_model(name)
        current=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            with threadpool_limits(limits=2):
                model.fit(matrix,labels)
                values=np.asarray(model.predict_proba(test_matrix),dtype=float)
        convergence=[item for item in caught if issubclass(item.category,ConvergenceWarning)]
        for item in caught:
            if not issubclass(item.category,ConvergenceWarning):warnings.warn(str(item.message),item.category,stacklevel=2)
        if (values.shape!=(len(test),2) or not np.isfinite(values).all()
                or np.any(values<0) or np.any(values>1)
                or not np.allclose(values.sum(axis=1),1.,rtol=1e-7,atol=1e-9)):
            raise ValueError('Invalid directional probabilities')
        classes=getattr(model,'classes_',None)
        if classes is not None and not np.array_equal(classes,np.array([0,1])):raise ValueError('Invalid class order')
        output['p_'+name]=values[:,1]
        estimator=model[-1] if hasattr(model,'named_steps') else model
        iterations=getattr(estimator,'n_iter_',None)
        model_meta[name]={'training_count':len(selected),'sample_minutes':spec['sample_minutes'],
            'latest_settlement_ms':int(settle_time[selected].max()),
            'earliest_signal_ms':int(train_time[selected].min()),
            'training_converged':not convergence,'convergence_warning_count':len(convergence),
            'iterations':np.atleast_1d(iterations).astype(int).tolist() if iterations is not None else None,
            'runtime_seconds':round(time.perf_counter()-current,4)}
        if progress:progress(f'{name}: {len(selected)} settled rows; {model_meta[name]["runtime_seconds"]:.2f}s; converged={not convergence}')
        del model,matrix,labels,values
    return output,{'fit_cutoff_ms':cutoff,'train_days':180,'refit_days':30,
        'test_start_ms':int(test_time.min()),'test_end_exclusive_ms':int(test_time.max())+MINUTE_MS,
        'prediction_count':len(test),'features':FEATURES.copy(),'specs':selected_specs(),'models':model_meta,
        'source_population_rows':len(train),'historical_decisive_population_rows':len(positions),
        'raw_scores_are_calibrated_win_probabilities':False,'local_research_only':True,
        'runtime_seconds':round(time.perf_counter()-started,4)}


def seed_archives(raw):
    """Copy only needed immutable archives into the new cache; old cache is read-only."""
    old=ROOT/'research'/'.cache';first=FEATURE_START-timedelta(days=3)
    for symbol in SYMBOLS:
        for kind in ('spot','index','premium','funding'):
            paths=(market_data.archive_paths(kind,symbol,first,FEATURE_END) if kind in ('spot','index')
                   else derivatives_data.archive_paths(kind,symbol,first,FEATURE_END))
            for relative in paths:
                rel=(Path('spot/1m')/relative if kind=='spot' else Path('futures/um')/relative
                     if kind=='index' else Path('derivatives')/relative)
                for candidate in (rel,rel.with_suffix('.zip.CHECKSUM')):
                    source=old/candidate;target=raw/candidate
                    if source.is_file() and not target.exists():
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def prepare_features(*,offline=False,cache=CACHE):
    raw=cache/'raw';raw.mkdir(parents=True,exist_ok=True);seed_archives(raw)
    first=FEATURE_START-timedelta(days=3)
    def fetch(task):
        symbol,kind=task
        if kind in ('spot','index'):
            frame,sources=market_data.load_kind(kind,symbol,first,FEATURE_END,raw,offline,workers=4)
        elif kind=='premium':frame,sources=derivatives_data.load_premium(symbol,first,FEATURE_END,raw,offline,allow_gaps=True)
        else:frame,sources=derivatives_data.load_funding(symbol,first,FEATURE_END,raw,offline)
        count=len(frame);del frame;gc.collect()
        print(f'RAW VERIFIED {symbol} {kind}: {count:,} observations',flush=True)
        return {'symbol':symbol,'kind':kind,'rows':count,'sources':sources}
    # Keep archive parsing bounded while concurrent requests hide network latency.
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources=list(pool.map(fetch,[(symbol,kind) for symbol in SYMBOLS for kind in ('spot','index','premium','funding')]))
    write_json(cache/'raw-sources.json',{'start_utc':str(first),'end_exclusive_utc':str(FEATURE_END),
        'sources':sources,'official_observations_only':True})
    # Labels are fetched from FEATURE_START, so its partial initial month is built
    # through the same monthly transformation as the original feature store.
    store=feature_data.ensure_feature_store(start=FEATURE_START,end=FEATURE_END,
        cache_dir=raw,store_dir=cache/'features')
    identity=hashlib.sha256(json.dumps(store,sort_keys=True).encode()).hexdigest()
    return store,identity


def expected_folds():
    cutoff=utc_ms(cutoff_for(START));last=utc_ms(END)
    while cutoff<last:
        stop=min(cutoff+30*DAY_MS,last)
        yield cutoff,max(utc_ms(START),cutoff),stop
        cutoff=stop


def contract(identity):
    return {'extension_source_sha256':file_hash(Path(__file__)),
        'original_models_sha256':file_hash(ROOT/'research/model_search_models.py'),
        'feature_store_identity':identity,'specs':selected_specs(),'start_utc':str(START),
        'end_exclusive_utc':str(END),'pipeline_version':1}


def run(*,offline=False,cache=CACHE):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();store,identity=prepare_features(offline=offline,cache=cache)
    expected=contract(identity);folder=cache/'predictions';folder.mkdir(exist_ok=True)
    manifest={'complete':False,'contract':expected,'files':[],
        'feature_start_utc':str(FEATURE_START),'feature_end_exclusive_utc':str(FEATURE_END),
        'source_metadata_sha256':file_hash(cache/'raw-sources.json'),
        'earlier_training_added':True,'original_caches_modified':False,'publish_or_deploy':False}
    write_json(cache/'manifest.json',manifest)
    for cutoff,first,stop in expected_folds():
        for symbol in SYMBOLS:
            name=f'{symbol}-{cutoff}-{first}-{stop}.npz';path=folder/name;meta_path=path.with_suffix('.json')
            reused=False
            if path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text(encoding='utf-8'))
                reused=meta.get('contract')==expected and meta.get('sha256')==file_hash(path)
            if reused:
                print('REUSE '+name,flush=True)
            else:
                print('FIT PAIR '+symbol+' '+datetime.fromtimestamp(cutoff/1000,timezone.utc).date().isoformat(),flush=True)
                train=feature_data.load_window(symbol,cutoff-180*DAY_MS,cutoff,store_dir=cache/'features')
                test=feature_data.load_window(symbol,first,stop,store_dir=cache/'features')
                scores,metadata=fit_pair(train,test,cutoff,progress=lambda msg:print('  '+msg,flush=True))
                if not all(m['training_converged'] for m in metadata['models'].values()):
                    raise ValueError('Fixed model did not converge; no parameter adjustment permitted')
                predicted=test[KEEP].reset_index(drop=True)
                for column in scores:predicted[column]=scores[column].to_numpy()
                predicted['model_fit_cutoff_ms']=cutoff
                predicted['train_latest_settlement_ms']=max(m['latest_settlement_ms'] for m in metadata['models'].values())
                temporary=path.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:np.savez(stream,**{key:predicted[key].to_numpy() for key in predicted})
                temporary.replace(path)
                meta={'contract':expected,'sha256':file_hash(path),'symbol':symbol,'model_metadata':metadata}
                write_json(meta_path,meta)
                del train,test,scores,predicted;gc.collect()
            manifest['files'].append({'file':name,'sha256':file_hash(path),'metadata_sha256':file_hash(meta_path),
                'symbol':symbol,'fit_cutoff_ms':cutoff,'requested_start_ms':first,'requested_end_exclusive_ms':stop,
                'model_metadata':meta['model_metadata']})
            write_json(cache/'manifest.json',manifest)
    manifest.update(complete=True,elapsed_seconds=round(time.perf_counter()-started,2),
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(cache/'manifest.json',manifest)
    print('EXTENSION COMPLETE '+json.dumps({'files':len(manifest['files']),'seconds':manifest['elapsed_seconds']}),flush=True)
    return manifest


def load_verified_extension(*,cache=CACHE):
    """Hash-check all two-model scores; return jointly available minute rows."""
    cache=Path(cache);manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    store=json.loads((cache/'features/manifest.json').read_text(encoding='utf-8'))
    identity=hashlib.sha256(json.dumps(store,sort_keys=True).encode()).hexdigest()
    if not manifest.get('complete') or manifest['contract']!=contract(identity):raise ValueError('Unverified early extension')
    if not store.get('complete') or store['sources_sha256']!=feature_data.source_identity():raise ValueError('Stale early features')
    for item in store['files'].values():
        if not feature_data._valid_file(cache/'features',item):raise ValueError('Early feature checksum mismatch')
    expected={(s,c,f,e) for c,f,e in expected_folds() for s in SYMBOLS};seen=set();parts=[]
    for item in manifest['files']:
        key=(item['symbol'],item['fit_cutoff_ms'],item['requested_start_ms'],item['requested_end_exclusive_ms'])
        if key not in expected or key in seen or Path(item['file']).name!=item['file']:raise ValueError('Wrong extension fold')
        seen.add(key);path=cache/'predictions'/item['file'];meta_path=path.with_suffix('.json')
        if file_hash(path)!=item['sha256'] or file_hash(meta_path)!=item['metadata_sha256']:raise ValueError('Early score checksum mismatch')
        meta=json.loads(meta_path.read_text(encoding='utf-8'));fold=meta['model_metadata']
        if meta['contract']!=manifest['contract'] or fold!=item['model_metadata']:raise ValueError('Early score provenance changed')
        if fold['specs']!=selected_specs() or set(fold['models'])!=set(MODEL_IDS):raise ValueError('Wrong early models')
        latest=max(m['latest_settlement_ms'] for m in fold['models'].values())
        if latest>=key[1] or not all(m['training_converged'] for m in fold['models'].values()):raise ValueError('Noncausal/unconverged early fold')
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(COLUMNS):raise ValueError('Incomplete early score columns')
            frame=pd.DataFrame({column:arrays[column] for column in COLUMNS})
        if len(frame)!=fold['prediction_count'] or not np.isfinite(frame.to_numpy(dtype=float)).all():raise ValueError('Invalid early score rows')
        if (frame.signal_ms.duplicated().any() or (frame.signal_ms<key[2]).any() or (frame.signal_ms>=key[3]).any()
                or (frame.entry_ms!=frame.signal_ms+MINUTE_MS).any() or (frame.settlement_ms!=frame.signal_ms+31*MINUTE_MS).any()
                or (frame.model_fit_cutoff_ms!=key[1]).any() or (frame.train_latest_settlement_ms!=latest).any()):
            raise ValueError('Invalid early chronology')
        scores=frame[['p_'+name for name in MODEL_IDS]].to_numpy()
        if np.any(scores<0) or np.any(scores>1) or np.any(frame[['entry','exit']].to_numpy()<=0):raise ValueError('Invalid early scores/prices')
        frame['symbol']=key[0];parts.append(frame)
    if seen!=expected:raise ValueError('Missing early folds')
    full=pd.concat(parts,ignore_index=True)
    if full.duplicated(['symbol','signal_ms']).any():raise ValueError('Overlapping early folds')
    count=full.groupby('signal_ms').symbol.transform('count');dropped=int((count!=2).sum())
    full=full.loc[count==2].sort_values(['signal_ms','symbol']).reset_index(drop=True)
    return full,{'manifest_sha256':file_hash(cache/'manifest.json'),'feature_store_identity':identity,
        'manifest':manifest,'joint_asset_rows_dropped':dropped,'prediction_rows':len(full)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();run(offline=args.offline)
