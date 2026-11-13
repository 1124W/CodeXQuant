"""Paired causal comparison of half-hour and minute opportunity scanning.

Run offline: python -m research.minute_scan_research
Both schedules use identical frozen logistic coefficients and gates. This is
retrospective evidence, not a search for a profitable threshold or a promise.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from live.features import BASE_FEATURES, FEATURES, MINUTE_MS, SYMBOLS, base_frame, enrich_pair_frames
from live.predictor import MODEL_PARAMETERS
from .derivative_features import add_derivative_features
from .derivatives_data import load_funding, load_premium
from .event_engine import DAY_MS, event_stats, utc_ms
from .event_strategy_research import moving_week_interval
from .qd_three_year_data import load_market

ROOT=Path(__file__).resolve().parents[1]
ANCHOR=date(2024,5,7)


def cutoff_for(start):
    return ANCHOR+timedelta(days=((start-ANCHOR).days//30)*30)


def prepare(start,end,cache,progress=print):
    first=cutoff_for(start)-timedelta(days=183)
    samples,sources,quality={},{},{}
    for symbol in SYMBOLS:
        progress(f'{symbol}: verify cached spot/index {first} through {end}',flush=True)
        spot,index,source=load_market(symbol,first,end,cache,True)
        frame=base_frame(spot)
        frame['signal_ms']=frame.timestamp+MINUTE_MS
        # Dense test population, original sparse training population.
        frame=frame[(frame.signal_ms>=utc_ms(start))|(frame.signal_ms%(30*MINUTE_MS)==0)].copy()
        frame=frame.dropna(subset=BASE_FEATURES).reset_index(drop=True)
        frame['entry_ms']=frame.signal_ms+MINUTE_MS
        frame['settlement_ms']=frame.signal_ms+31*MINUTE_MS
        frame=frame[frame.settlement_ms<utc_ms(end)].copy()
        opens=index.set_index('timestamp')['open']
        frame['entry']=opens.loc[frame.entry_ms].to_numpy()
        frame['exit']=opens.loc[frame.settlement_ms].to_numpy()
        frame['y']=(frame.exit>frame.entry).astype(int)
        frame['is_flat']=frame.exit==frame.entry
        frame['symbol']=symbol
        frame.attrs['features']=BASE_FEATURES
        samples[symbol]=frame
        sources[symbol]={'spot_index':source}
        quality[symbol]={'spot_minutes':len(spot),'index_minutes':len(index)}
        del spot,index,frame;gc.collect()
    samples=enrich_pair_frames(samples)
    for symbol in SYMBOLS:
        progress(f'{symbol}: align observed premium/funding',flush=True)
        premium,p_source=load_premium(symbol,first,end,cache,True,allow_gaps=True)
        funding,f_source=load_funding(symbol,first,end,cache,True)
        samples[symbol]=add_derivative_features(samples[symbol],premium,funding,allow_premium_gaps=True)
        sources[symbol].update(premium=p_source,funding=f_source)
        quality[symbol]['features']=samples[symbol].attrs['derivative_features']
        del premium,funding;gc.collect()
    return samples,sources,quality


def predict(frame,start,end):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    cutoff=utc_ms(cutoff_for(start));finish=utc_ms(end)
    parts,folds=[],[]
    while cutoff<finish:
        stop=min(cutoff+30*DAY_MS,finish)
        train=frame[(frame.signal_ms>=cutoff-180*DAY_MS)&(frame.settlement_ms<cutoff)
                    &(frame.signal_ms%(30*MINUTE_MS)==0)&~frame.is_flat]
        test=frame[(frame.signal_ms>=max(cutoff,utc_ms(start)))&(frame.signal_ms<stop)]
        if len(train)<2000 or train.y.nunique()!=2 or test.empty:
            raise ValueError('Insufficient settled chronological population')
        model=make_pipeline(StandardScaler(),LogisticRegression(**MODEL_PARAMETERS))
        with threadpool_limits(limits=2):
            model.fit(train[FEATURES].to_numpy(),train.y.to_numpy())
            scores=model.predict_proba(test[FEATURES].to_numpy())[:,1]
        columns=['symbol','signal_ms','entry_ms','settlement_ms','entry','exit','ret_60','rv_60','rv_ratio']
        predicted=test[columns].copy()
        predicted['p_up']=scores
        predicted['model_fit_cutoff_ms']=cutoff
        predicted['train_latest_settlement_ms']=int(train.settlement_ms.max())
        parts.append(predicted)
        folds.append({'symbol':str(frame.symbol.iloc[0]),'fit_cutoff_ms':cutoff,
                      'train_latest_settlement_ms':int(train.settlement_ms.max()),
                      'train_count':len(train),'predict_count':len(test),'end_ms':stop})
        print(f'{frame.symbol.iloc[0]} {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()}: train {len(train)}, predict {len(test)}',flush=True)
        cutoff=stop
    return pd.concat(parts,ignore_index=True),folds


def opportunities(frame,cadence):
    if cadence not in (1,30):raise ValueError('Only the two predeclared schedules are supported')
    d=frame[frame.signal_ms%(cadence*MINUTE_MS)==0].copy()
    p=d.p_up.to_numpy();direction=np.where(p>=.5,1,-1)
    normalized=d.ret_60.to_numpy()/(np.maximum(d.rv_60.to_numpy(),1e-8)*np.sqrt(60))
    keep=(np.maximum(p,1-p)>=.56)&(direction==-np.sign(d.ret_60.to_numpy()))&(np.abs(normalized)>=1)&(d.rv_ratio.to_numpy()<=1.25)
    d['direction']=direction
    return d.loc[keep].to_dict('records')


def simulate_alert_admission(raw):
    """Mirror signal-time reservations, with outcomes known after expiry+5s.

    Decisions occur before settlement polling in the same minute. A just-due
    old position therefore remains reserved during this minute's admission.
    Price outcomes are consulted only when settlement becomes observable.
    """
    pending,accepted=[],[]
    daily_losses=defaultdict(float);settled_net=0.
    for source in sorted(raw,key=lambda r:(r['signal_ms'],r['symbol'])):
        signal=int(source['signal_ms']);entry=int(source['entry_ms'])
        remaining=[]
        for trade in pending:
            if trade['settlement_ms']<signal:
                settled_net+=trade['net_units']
                if trade['outcome']=='loss':daily_losses[trade['settlement_ms']//DAY_MS]+=5
            else:remaining.append(trade)
        pending=remaining
        if any(t['settlement_ms']>entry for t in pending):continue
        if daily_losses[entry//DAY_MS]+len(pending)*5+5>25:continue
        if 1000+settled_net-len(pending)*5<5:continue
        trade=dict(source)
        change=(trade['exit']-trade['entry'])*trade['direction']
        trade['outcome']='win' if change>0 else 'loss' if change<0 else 'refund'
        trade['net_units']=4.25 if change>0 else -5 if change<0 else 0.
        trade['stake']=5.;trade['horizon_minutes']=30
        pending.append(trade);accepted.append(trade)
    return sorted(accepted,key=lambda r:(r['settlement_ms'],r['symbol']))


def run(start=date(2026,8,9),end=date(2026,11,12),output=None):
    if start>=end or end>datetime.now(timezone.utc).date():raise ValueError('Require completed historical UTC days')
    started=time.perf_counter();out=Path(output or ROOT/'research');out.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),ROOT/'live'/'features.py',ROOT/'live'/'predictor.py',ROOT/'research'/'derivative_features.py']
    source_sha=hashlib.sha256(b''.join(p.read_bytes() for p in source_paths)).hexdigest()
    plan={'start_utc':str(start),'end_exclusive_utc':str(end),'cadences_minutes':[30,1],
          'train_days':180,'refit_days':30,'training_sampling_minutes':30,'training_anchor_utc':str(ANCHOR),
          'rule_id':'both_regularized_logistic','model_parameters':MODEL_PARAMETERS,'score_threshold':.56,
          'gate':'oppose ret60; abs(ret60)/(rv60*sqrt60)>=1; rv_ratio<=1.25',
          'entry_delay_minutes':1,'holding_minutes':30,'stake':5,'payout':.85,'fee':0.,'initial_capital':1000,
          'daily_gross_loss_plus_pending_risk_limit':25,'cross_asset_priority':'BTCUSDT before ETHUSDT',
          'admission':'signal-time reservation; strict expiry<signal for settled knowledge; same-minute settlement polling follows decision',
          'selection':'none; compare both fixed schedules; no threshold tuning on this window',
          'pristine_holdout':False,'forward_validated':False,'source_sha256':source_sha}
    plan_text=json.dumps(plan,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    (out/'minute-scan-plan.json').write_text(plan_text,encoding='utf-8')
    frames,sources,quality=prepare(start,end,ROOT/'research'/'.cache')
    predictions,folds=[],[];digest=hashlib.sha256()
    for symbol in SYMBOLS:
        digest.update(symbol.encode());digest.update(pd.util.hash_pandas_object(frames[symbol],index=False).to_numpy(dtype='<u8').tobytes())
        predicted,current=predict(frames[symbol],start,end);predictions.append(predicted);folds.extend(current)
    combined=pd.concat(predictions,ignore_index=True);measurements={}
    for cadence in (30,1):
        raw=opportunities(combined,cadence);ledger=simulate_alert_admission(raw)
        stats=event_stats(ledger,utc_ms(start),utc_ms(end),payout=.85,fee=0.,bootstrap=True)
        stats['raw_opportunities']=len(raw)
        stats['weekly_block_interval']=moving_week_interval(ledger,utc_ms(start),utc_ms(end))
        stats['by_symbol']={s:event_stats([t for t in ledger if t['symbol']==s],utc_ms(start),utc_ms(end),payout=.85,fee=0.,bootstrap=False) for s in SYMBOLS}
        stats['by_month']={}
        cursor=start.replace(day=1)
        while cursor<end:
            following=(cursor.replace(day=28)+timedelta(days=4)).replace(day=1)
            lower,upper=utc_ms(max(start,cursor)),utc_ms(min(end,following))
            bucket=[t for t in ledger if lower<=t['settlement_ms']<upper]
            stats['by_month'][cursor.strftime('%Y-%m')]=event_stats(bucket,lower,upper,payout=.85,fee=0.,bootstrap=False)
            cursor=following
        stats['accepted_off_halfhour']=sum(t['signal_ms']%(30*MINUTE_MS)!=0 for t in ledger)
        measurements[str(cadence)]=stats
        pd.DataFrame(ledger).to_csv(out/f'minute-scan-{cadence}m-trades.csv',index=False)
        print(json.dumps({'cadence_minutes':cadence,'trades':stats['signals'],'win_rate':stats['win_rate'],'net':stats['net_units']},ensure_ascii=False),flush=True)
    report={'schema_version':1,'engine':'minute-opportunity-paired-v1','plan':plan,'results':measurements,
            'folds':folds,'quality':quality,'sources':sources,
            'provenance':{'source_sha256':source_sha,'plan_sha256':hashlib.sha256(plan_text.encode()).hexdigest(),'data_sha256':digest.hexdigest(),
                          'generated_at_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':round(time.perf_counter()-started,2)},
            'checks':{'training_labels_settled':all(f['train_latest_settlement_ms']<f['fit_cutoff_ms'] for f in folds),
                      'positive_minute_net':measurements['1']['net_units']>0,
                      'minute_win_rate_higher':(measurements['1']['win_rate'] or 0)>(measurements['30']['win_rate'] or 0),
                      'minute_win_rate_58pct':(measurements['1']['win_rate'] or 0)>=.58},
            'limitations':['This historical interval was previously researched; it is not pristine or prospective evidence.',
                'More frequent inference changes the sampled population while training remains on the original half-hour grid.',
                'No search or selection uses the two evaluation results; both schedules are reported, including losing results.',
                'Scores are uncalibrated. Minute index opens are proxies for actual event strike/settlement quotes.',
                'Ideal data availability and prompt settlement polling are assumed; outages and late notification can reduce live admissions.',
                'Correlated minute signals are not independent trades; shared capital, pending-risk reservation and weekly block intervals are applied.']}
    if source_sha!=hashlib.sha256(b''.join(p.read_bytes() for p in source_paths)).hexdigest():raise RuntimeError('Source changed during research')
    (out/'minute-scan-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=date.fromisoformat,default=date(2026,8,9))
    parser.add_argument('--end',type=date.fromisoformat,default=date(2026,11,12))
    args=parser.parse_args();run(args.start,args.end)
