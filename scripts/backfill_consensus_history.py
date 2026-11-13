"""Build a new, isolated minute-history bridge from official public observations.

This never opens the active history database, predicts, admits trades, or sends
messages. Time bounds are UTC milliseconds and [start, end) is signal time.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd

from live.features import (FEATURES,BASE_FEATURES,PAIR_FEATURES,SPOT_COLUMNS,SYMBOLS,
    MINUTE_MS,base_frame,enrich_pair_frames,_derivative_values,build_latest_features)
from live.history import FeatureHistory,LABEL_DELAY_MS

DAY_MS=86_400_000
WARMUP_MINUTES=1000
ENDPOINTS={'spot':('/api/v3/klines',True),'premium':('/fapi/v1/premiumIndexKlines',False),
    'index':('/fapi/v1/indexPriceKlines',False),'funding':('/fapi/v1/fundingRate',False)}


def minute(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<=0 or value%MINUTE_MS:
        raise ValueError('Require positive integer minute boundaries')
    return value


class OfficialClient:
    def __init__(self,source_dir,opener=urlopen):
        self.source_dir=Path(source_dir);self.source_dir.mkdir(parents=True,exist_ok=False)
        self.opener=opener;self.sources=[]

    def request(self,kind,symbol,params,page):
        path,spot=ENDPOINTS[kind]
        hosts=('https://api.binance.com','https://data-api.binance.vision') if spot else ('https://fapi.binance.com','https://www.binance.com')
        for host in hosts:
            url=host+path+'?'+urlencode(params)
            try:
                with self.opener(Request(url,headers={'User-Agent':'CodeXQuant-history-backfill/1.0'}),timeout=15) as response:
                    raw=response.read()
                data=json.loads(raw)
                if not isinstance(data,list):raise ValueError('Expected official observation list')
            except (OSError,ValueError):continue
            name=f'{kind}-{symbol}-{page:04d}.json';target=self.source_dir/name
            if target.exists():raise ValueError('Duplicate source response filename')
            target.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            target.with_suffix('.json.sha256').write_text(digest+'\n',encoding='ascii')
            self.sources.append({'kind':kind,'symbol':symbol,'file':name,'sha256':digest,
                'url':url,'verification':'local_sha256_of_exact_official_public_response','rows':len(data)})
            return data
        raise RuntimeError('Official public endpoint unavailable for '+kind+' '+symbol)


def download_range(client,kind,symbol,start,end):
    """Bounded pagination, preserving exact observations; never fill missing bars."""
    if kind not in ENDPOINTS or symbol not in SYMBOLS or start>=end:raise ValueError('Invalid observation request')
    cursor=start;observations=[];page=0
    while cursor<end:
        params={'pair' if kind=='index' else 'symbol':symbol,'startTime':cursor,'endTime':end-1,'limit':1000}
        if kind!='funding':params['interval']='1m'
        rows=client.request(kind,symbol,params,page);page+=1
        if not rows:break
        stamps=[]
        for row in rows:
            if kind=='funding':
                if not isinstance(row,dict) or row.get('symbol')!=symbol:raise ValueError('Wrong funding asset')
                stamp=int(row['fundingTime'])
            else:
                if not isinstance(row,list) or len(row)<(11 if kind=='spot' else 5):raise ValueError('Malformed official kline')
                stamp=int(row[0])
                if stamp%MINUTE_MS:raise ValueError('Off-grid official kline')
            if stamp<cursor or stamp>=end or stamps and stamp<=stamps[-1]:raise ValueError('Out-of-range or repeated official observation')
            stamps.append(stamp)
        following=stamps[-1]+(1 if kind=='funding' else MINUTE_MS)
        if following<=cursor:raise ValueError('Pagination did not advance')
        observations.extend(rows);cursor=following
        if kind=='funding' and len(rows)<1000:break
    if kind=='funding':
        frame=pd.DataFrame({'funding_time':[int(row['fundingTime']) for row in observations],
            'rate':[float(row['fundingRate']) for row in observations]})
        if frame.empty or len(frame)<3 or frame.funding_time.duplicated().any() or not np.isfinite(frame.rate).all():
            raise ValueError('Incomplete or invalid realized funding observations')
        return frame
    names=['timestamp','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
    if kind=='spot':
        frame=pd.DataFrame([{name:row[index] for index,name in enumerate(names[:11])} for row in observations])
        selected=SPOT_COLUMNS+['taker_buy_quote'];frame=frame[selected]
    else:
        frame=pd.DataFrame([row[:5] for row in observations],columns=names[:5])
    for name in frame:frame[name]=pd.to_numeric(frame[name],errors='raise')
    frame['timestamp']=frame.timestamp.astype('int64')
    expected=np.arange(start,end,MINUTE_MS,dtype=np.int64)
    if not np.array_equal(frame.timestamp.to_numpy(),expected) or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Incomplete, duplicate or nonfinite official minute observations')
    if kind=='index' and (frame.open<=0).any():raise ValueError('Invalid official index open')
    return frame


def fetch_inputs(client,start,end,now_ms):
    minute(start);minute(end)
    if not start<end or end-start>14*DAY_MS or end>now_ms//MINUTE_MS*MINUTE_MS:
        raise ValueError('Require at most 14 completed UTC days of signal history')
    latest_label=min(end-MINUTE_MS+31*MINUTE_MS,(now_ms-LABEL_DELAY_MS)//MINUTE_MS*MINUTE_MS)
    spans={'spot':(start-WARMUP_MINUTES*MINUTE_MS,end),'premium':(start-60*MINUTE_MS,end),
        'funding':(start-4*DAY_MS,end),'index':(start+MINUTE_MS,latest_label+MINUTE_MS)}
    tasks=[(kind,symbol) for kind in ENDPOINTS for symbol in SYMBOLS
        if kind!='index' or latest_label>=start+31*MINUTE_MS]
    result={kind:{} for kind in ENDPOINTS}
    def task(item):
        kind,symbol=item;frame=download_range(client,kind,symbol,*spans[kind])
        print(f'FETCHED {symbol} {kind}: {len(frame):,} observations',flush=True)
        return kind,symbol,frame
    with ThreadPoolExecutor(max_workers=4) as pool:
        for kind,symbol,frame in pool.map(task,tasks):result[kind][symbol]=frame
    for symbol in SYMBOLS:
        result['index'].setdefault(symbol,pd.DataFrame(columns=['timestamp','open','high','low','close']))
    return result


def build_rows(inputs,start,end,now_ms,*,samples=7):
    """Compute spot/pair indicators once, then exact per-minute derivative inputs."""
    minute(start);minute(end)
    stamps=np.arange(start,end,MINUTE_MS,dtype=np.int64)
    base={}
    for symbol in SYMBOLS:
        frame=base_frame(inputs['spot'][symbol]);frame['signal_ms']=frame.timestamp+MINUTE_MS
        frame=frame.loc[(frame.signal_ms>=start)&(frame.signal_ms<end)].copy()
        if not np.array_equal(frame.signal_ms.to_numpy(),stamps):raise ValueError('Missing requested feature minutes')
        if (frame.streak.abs()>=WARMUP_MINUTES-1).any():raise ValueError('Streak start exceeds live history window')
        base[symbol]=frame
    enriched=enrich_pair_frames(base);rows={};maximum_delta=0.;checked=[]
    for symbol in SYMBOLS:
        frame=enriched[symbol].copy()
        derivatives=[_derivative_values(inputs['premium'][symbol],inputs['funding'][symbol],int(stamp)) for stamp in stamps]
        for name in FEATURES[len(BASE_FEATURES+PAIR_FEATURES):]:frame[name]=[row[name] for row in derivatives]
        frame=frame[['signal_ms']+FEATURES]
        if not np.isfinite(frame[FEATURES].to_numpy()).all():raise ValueError('Nonfinite historical feature; no filling allowed')
        rows[symbol]=frame
    for index in np.unique(np.linspace(0,len(stamps)-1,min(samples,len(stamps)),dtype=int)):
        stamp=int(stamps[index])
        # Exactly the serving formula, using only each signal's trailing 1000
        # completed spot bars. This also catches rolling numerical drift.
        spot={symbol:inputs['spot'][symbol].loc[(inputs['spot'][symbol].timestamp<stamp)
            &(inputs['spot'][symbol].timestamp>=stamp-WARMUP_MINUTES*MINUTE_MS)] for symbol in SYMBOLS}
        reference=build_latest_features(spot,inputs['premium'],inputs['funding'],stamp)
        for symbol in SYMBOLS:
            actual=rows[symbol].iloc[index][FEATURES].to_numpy(dtype=float)
            expected=np.array([reference[symbol][name] for name in FEATURES])
            delta=float(np.max(np.abs(actual-expected)));maximum_delta=max(maximum_delta,delta)
            if not np.allclose(actual,expected,rtol=1e-10,atol=1e-10):
                raise ValueError('Batch features differ from final serving formula beyond 1e-10 tolerance')
        checked.append(stamp)
    encoded=[];counts={}
    for symbol in SYMBOLS:
        prices=inputs['index'][symbol].set_index('timestamp').open.to_dict()
        settled=0
        for row in rows[symbol].itertuples(index=False,name=None):
            stamp=int(row[0]);expiry=stamp+31*MINUTE_MS;features=list(map(float,row[1:]))
            y=flat=opened=closed=source=None
            if expiry+LABEL_DELAY_MS<=now_ms:
                if stamp+MINUTE_MS not in prices or expiry not in prices:raise ValueError('Exact matured index open is missing')
                opened=float(prices[stamp+MINUTE_MS]);closed=float(prices[expiry])
                if not math.isfinite(opened) or not math.isfinite(closed) or min(opened,closed)<=0:raise ValueError('Invalid matured index execution price')
                y=int(closed>opened);flat=int(closed==opened);source='official_index_observation';settled+=1
            encoded.append((symbol,stamp,expiry,json.dumps(features,separators=(',',':'),allow_nan=False),
                y,flat,opened,closed,source))
        counts[symbol]={'feature_rows':len(rows[symbol]),'settled_rows':settled,'pending_rows':len(rows[symbol])-settled}
    return encoded,{'counts':counts,'parity_sample_signal_ms':checked,'parity_rtol':1e-10,'parity_atol':1e-10,
        'maximum_sample_feature_absolute_delta':maximum_delta,'warmup_minutes':WARMUP_MINUTES,
        'label_delay_ms':LABEL_DELAY_MS,'all_requested_minutes_present':True}


def run(start,end,output,audit,*,now_ms=None,client=None):
    minute(start);minute(end);now=int(time.time()*1000) if now_ms is None else int(now_ms)
    out=Path(output).resolve();audit=Path(audit).resolve();partial=out.with_suffix(out.suffix+'.partial')
    if out.name in ('history.sqlite3','history-consensus.sqlite3'):
        raise ValueError('Use a new bridge-source filename; active history paths are forbidden')
    if any(path.exists() or path.is_symlink() for path in (out,audit,partial)):
        raise ValueError('Backfill outputs must be new; existing files are never overwritten')
    out.parent.mkdir(parents=True,exist_ok=True);audit.parent.mkdir(parents=True,exist_ok=True)
    sources=audit.with_suffix('.sources')
    client=client or OfficialClient(sources)
    inputs=fetch_inputs(client,start,end,now);records,verification=build_rows(inputs,start,end,now)
    history=FeatureHistory(partial)
    with closing(sqlite3.connect(partial)) as db:
        if db.execute('SELECT count(*) FROM feature_history').fetchone()[0]:raise ValueError('Bridge target is not empty')
        db.executemany('''INSERT INTO feature_history(symbol,signal_ms,settlement_ms,features_json,
            y,is_flat,entry_price,exit_price,label_source) VALUES(?,?,?,?,?,?,?,?,?)''',records)
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Bridge database integrity failure')
    partial.replace(out)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    result={'schema_version':1,'kind':'official_public_minute_history_backfill','start_ms':start,'end_exclusive_ms':end,
        'observation_now_ms':now,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'database_sha256':digest,'database_file':out.name,'database_rows':len(records),**verification,
        'sources':sorted(client.sources,key=lambda item:(item['kind'],item['symbol'],item['file'])),
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'features_source_sha256':hashlib.sha256(Path(sys.modules['live.features'].__file__).read_bytes()).hexdigest(),
        'interpolated_minutes':0,'predictions_run':False,'messages_sent':False,'active_history_modified':False}
    audit.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('database_file','database_rows','counts','maximum_sample_feature_absolute_delta','database_sha256')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-ms',type=int,required=True);parser.add_argument('--end-ms',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args();run(args.start_ms,args.end_ms,args.output,args.audit)
