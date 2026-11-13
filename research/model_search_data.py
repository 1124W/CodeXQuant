"""Verified, bounded-memory monthly feature store for local model research.

All sixty inputs use completed-minute observations. Execution labels come from
exact official index minute opens and are stored separately from feature arrays.
No network request is allowed: every loader runs with offline=True.
"""
from __future__ import annotations

from datetime import date,datetime,timedelta,timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from live.features import BASE_FEATURES,FEATURES,SYMBOLS,MINUTE_MS,base_frame,enrich_pair_frames
from .derivative_features import add_derivative_features
from .derivatives_data import load_premium,load_funding
from .qd_three_year_data import load_kind

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'research'/'.cache'
STORE=CACHE/'model-search'/'features'
START=date(2023,11,9)
END=date(2026,11,12)
ROW_COLUMNS=FEATURES+['symbol','signal_ms','entry_ms','settlement_ms','entry','exit','y','is_flat']


def _ms(day):
    return int(datetime.combine(day,datetime.min.time(),timezone.utc).timestamp()*1000)


def _next_month(day):
    return (day.replace(day=28)+timedelta(days=4)).replace(day=1)


def _windows(start,end):
    cursor=start.replace(day=1)
    while cursor<end:
        following=_next_month(cursor)
        yield cursor.strftime('%Y-%m'),max(cursor,start),min(following,end)
        cursor=following


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def source_identity():
    names=('research/model_search_data.py','live/features.py','research/derivative_features.py',
           'research/derivatives_data.py','research/qd_three_year_data.py')
    return {name:_sha(ROOT/name) for name in names}


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def build_month(spot,premium,funding,index_opens,start_ms,end_ms,global_end_ms):
    """Pure chunk transformation, with labels unable to enter the input vector.

    index_opens[symbol] contains a continuous open array and its start_ms. This
    separate label buffer can extend beyond end_ms so month-end signals retain
    their correctly delayed thirty-minute outcome.
    """
    if not start_ms<end_ms<=global_end_ms:raise ValueError('Invalid month bounds')
    if any(set(mapping)!=set(SYMBOLS) for mapping in (spot,premium,funding,index_opens)):
        raise ValueError('Require both BTC/ETH input populations')
    base={}
    for symbol in SYMBOLS:
        frame=base_frame(spot[symbol])
        frame['signal_ms']=frame.timestamp+MINUTE_MS
        frame=frame[(frame.signal_ms>=start_ms)&(frame.signal_ms<end_ms)
                    &(frame.signal_ms+31*MINUTE_MS<global_end_ms)].copy()
        frame.attrs['features']=BASE_FEATURES.copy()
        base[symbol]=frame
    paired=enrich_pair_frames(base)
    output={}
    for symbol in SYMBOLS:
        frame=add_derivative_features(paired[symbol],premium[symbol],funding[symbol],allow_premium_gaps=True)
        if frame.attrs['features']!=FEATURES:raise ValueError('Unexpected feature order')
        frame['entry_ms']=frame.signal_ms+MINUTE_MS
        frame['settlement_ms']=frame.signal_ms+31*MINUTE_MS
        source=index_opens[symbol]
        values=np.asarray(source['opens'],dtype=float)
        for destination,times in (('entry',frame.entry_ms),('exit',frame.settlement_ms)):
            offset=times.to_numpy(dtype=np.int64)-int(source['start_ms'])
            locations=offset//MINUTE_MS
            if (np.any(offset%MINUTE_MS) or np.any(locations<0) or np.any(locations>=len(values))):
                raise ValueError('Exact index label outside verified continuous history')
            frame[destination]=values[locations]
        if not np.isfinite(frame[['entry','exit']].to_numpy()).all() or (frame[['entry','exit']]<=0).any().any():
            raise ValueError('Invalid official index execution observations')
        frame['symbol']=symbol
        frame['y']=(frame.exit>frame.entry).astype(np.int8)
        frame['is_flat']=frame.exit==frame.entry
        quality=frame.attrs['derivative_features']
        frame=frame[ROW_COLUMNS].copy()
        frame.attrs={'features':FEATURES.copy(),'quality':quality}
        output[symbol]=frame
    return output


def _save_npz(path,frame):
    temporary=path.with_suffix('.npz.tmp')
    # Uncompressed numeric NPZ avoids CPU-heavy compression of1.5GB of floats.
    # The monthly boundary bounds both IO and memory during later training.
    with temporary.open('wb') as stream:
        np.savez(stream,features=frame[FEATURES].to_numpy(dtype=np.float64),
                 times=frame[['signal_ms','entry_ms','settlement_ms']].to_numpy(dtype=np.int64),
                 prices=frame[['entry','exit']].to_numpy(dtype=np.float64),
                 labels=frame[['y','is_flat']].to_numpy(dtype=np.int8))
    temporary.replace(path)


def _valid_file(folder,entry):
    name=entry.get('file','')
    if Path(name).name!=name or not name.endswith('.npz'):return False
    path=folder/name
    return path.is_file() and _sha(path)==entry.get('sha256')


def _progress(message):print(message,flush=True)


def ensure_feature_store(*,start=START,end=END,cache_dir=None,store_dir=None,progress=_progress):
    """Build/reuse a source- and content-hashed numeric store, one month at a time."""
    if not isinstance(start,date) or not isinstance(end,date) or not start<end:
        raise ValueError('Require increasing UTC calendar dates')
    cache=Path(cache_dir or CACHE);folder=Path(store_dir or STORE);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'manifest.json';identity=source_identity()
    contract={'schema_version':1,'features':FEATURES,'start_utc':str(start),'end_exclusive_utc':str(end),
              'sources_sha256':identity,'minute_delay':1,'horizon_minutes':30,'offline':True,
              'source_policy':'verified official observations; no interpolation; complete60minute premium windows'}
    manifest={**contract,'complete':False,'files':{},'index_sources':{}}
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if all(old.get(key)==value for key,value in contract.items()):manifest=old
    windows=list(_windows(start,end))
    expected=[f'{symbol}:{month}' for month,_,_ in windows for symbol in SYMBOLS]
    if manifest.get('complete') and set(manifest['files'])==set(expected) and all(_valid_file(folder,manifest['files'][key]) for key in expected):
        if progress:progress('FEATURE STORE VERIFIED: '+str(folder))
        return manifest
    manifest['complete']=False;_write_json(path,manifest)
    opens={}
    for symbol in SYMBOLS:
        if progress:progress(f'{symbol}: verify complete index labels {start} through {end}')
        index,sources=load_kind('index',symbol,start,end,cache,True)
        # load_kind verified a unique complete calendar grid; offsets are exact.
        opens[symbol]={'start_ms':_ms(start),'opens':index.open.to_numpy(dtype=np.float64,copy=True)}
        manifest['index_sources'][symbol]=sources
        del index;gc.collect()
    _write_json(path,manifest)
    for month,lower,upper in windows:
        keys=[f'{symbol}:{month}' for symbol in SYMBOLS]
        if all(key in manifest['files'] and _valid_file(folder,manifest['files'][key]) for key in keys):
            if progress:progress(f'FEATURE MONTH REUSED {month}')
            continue
        warmup=lower-timedelta(days=3)
        spot,premium,funding,sources={},{},{},{}
        if progress:progress(f'FEATURE MONTH {month}: completed-minute inputs {warmup} through {upper}')
        for symbol in SYMBOLS:
            spot[symbol],spot_sources=load_kind('spot',symbol,warmup,upper,cache,True)
            premium[symbol],premium_sources=load_premium(symbol,warmup,upper,cache,True,allow_gaps=True)
            # Partial current-month funding cache is the complete Nov1..Nov12
            # request because warmup begins in the preceding archived month.
            funding[symbol],funding_sources=load_funding(symbol,warmup,upper,cache,True)
            sources[symbol]={'spot':spot_sources,'premium':premium_sources,'funding':funding_sources}
        frames=build_month(spot,premium,funding,opens,_ms(lower),_ms(upper),_ms(end))
        for symbol in SYMBOLS:
            frame=frames[symbol];name=f'{symbol}-{month}.npz';target=folder/name
            _save_npz(target,frame)
            manifest['files'][f'{symbol}:{month}']={'file':name,'sha256':_sha(target),'symbol':symbol,'month':month,
                'start_ms':_ms(lower),'end_ms':_ms(upper),'rows':len(frame),'bytes':target.stat().st_size,
                'first_signal_ms':int(frame.signal_ms.iloc[0]) if len(frame) else None,
                'last_signal_ms':int(frame.signal_ms.iloc[-1]) if len(frame) else None,
                'quality':frame.attrs['quality'],'observed_sources':sources[symbol]}
        _write_json(path,manifest)
        if progress:progress(f'FEATURE MONTH DONE {month}: '+', '.join(f'{symbol}={len(frames[symbol])}' for symbol in SYMBOLS))
        del spot,premium,funding,frames,frame,sources;gc.collect()
    if source_identity()!=identity:raise RuntimeError('Feature source changed during cache construction')
    manifest['complete']=True
    manifest['total_rows']={symbol:sum(v['rows'] for v in manifest['files'].values() if v['symbol']==symbol) for symbol in SYMBOLS}
    manifest['created_at_utc']=datetime.now(timezone.utc).isoformat()
    _write_json(path,manifest)
    return manifest


def load_window(symbol,start_ms,end_ms,*,store_dir=None):
    """Load only monthly blocks intersecting a signal-time interval.

    The caller must purge settlement_ms>=its fitting cutoff before fitting;
    retaining outcomes here does not make them historically available.
    """
    if symbol not in SYMBOLS:raise ValueError('Only BTCUSDT/ETHUSDT are supported')
    if any(isinstance(t,bool) or int(t)!=t or t%MINUTE_MS for t in (start_ms,end_ms)) or start_ms>=end_ms:
        raise ValueError('Require increasing exact minute boundaries')
    folder=Path(store_dir or STORE)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if (not manifest.get('complete') or manifest.get('schema_version')!=1 or manifest.get('features')!=FEATURES
            or manifest.get('sources_sha256')!=source_identity()):raise ValueError('Unverified or stale feature manifest')
    if start_ms<_ms(date.fromisoformat(manifest['start_utc'])) or end_ms>_ms(date.fromisoformat(manifest['end_exclusive_utc'])):
        raise ValueError('Requested window outside cached history')
    blocks=[]
    for entry in sorted(manifest['files'].values(),key=lambda x:x['start_ms']):
        if entry['symbol']!=symbol or entry['end_ms']<=start_ms or entry['start_ms']>=end_ms:continue
        if not _valid_file(folder,entry):raise ValueError('Feature cache checksum mismatch')
        with np.load(folder/entry['file'],allow_pickle=False) as arrays:
            features=arrays['features'];times=arrays['times'];prices=arrays['prices'];labels=arrays['labels']
            count=len(features)
            if (features.shape!=(count,60) or times.shape!=(count,3) or prices.shape!=(count,2)
                    or labels.shape!=(count,2) or count!=entry['rows']):raise ValueError('Feature cache shape mismatch')
            keep=(times[:,0]>=start_ms)&(times[:,0]<end_ms)
            frame=pd.DataFrame(features[keep],columns=FEATURES)
            for index,name in enumerate(('signal_ms','entry_ms','settlement_ms')):frame[name]=times[keep,index]
            frame['entry']=prices[keep,0];frame['exit']=prices[keep,1]
            frame['y']=labels[keep,0];frame['is_flat']=labels[keep,1].astype(bool)
            frame['symbol']=symbol;blocks.append(frame)
    combined=pd.concat(blocks,ignore_index=True) if blocks else pd.DataFrame(columns=ROW_COLUMNS)
    combined=combined[ROW_COLUMNS].sort_values('signal_ms').reset_index(drop=True)
    if (combined.signal_ms.duplicated().any() or not np.isfinite(combined[FEATURES].to_numpy(dtype=float)).all()
            or not (combined.entry_ms==combined.signal_ms+MINUTE_MS).all()
            or not (combined.settlement_ms==combined.signal_ms+31*MINUTE_MS).all()
            or not combined.y.isin([0,1]).all()
            or not (combined.y==(combined.exit>combined.entry).astype(int)).all()
            or not (combined.is_flat==(combined.exit==combined.entry)).all()):
        raise ValueError('Feature cache chronology/labels invalid')
    combined.attrs['features']=FEATURES.copy()
    return combined


if __name__=='__main__':
    result=ensure_feature_store()
    print(json.dumps({'complete':result['complete'],'total_rows':result['total_rows'],'path':str(STORE)},ensure_ascii=False),flush=True)
