"""Local-only, preregistered higher-win-rate filter study; never changes live rules.

python -m research.high_win_research
Uses only verified cached public observations. All dates are UTC/exclusive end.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np
import pandas as pd

from .event_engine import DAY_MS, event_stats, utc_ms, _monthly
from .event_strategy_research import moving_week_interval, wilson_lower
from .high_win_filters import candidates, filter_mask
from .minute_scan_research import prepare, predict

ROOT=Path(__file__).resolve().parents[1]
START=date(2024,5,7)
SPLIT=date(2025,5,7)
CONFIRM=date(2026,8,9)
END=date(2026,11,12)
SYMBOLS=('BTCUSDT','ETHUSDT')
EXTRA=['ret_240','rv_240','ret_3','imb_ma5','imb_ma60','clv','peer_imb_ma15','premium_close_bps']
BASE_CAPITAL={'payout':.85,'fee':0.,'initial_bankroll':1000.,'stake':5.,
              'max_open_positions':1,'daily_loss_limit':25.,'adverse_bps':0.}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_end_for(stop):
    """Read existing full-month archives while retaining next-day labels.

    Loaded later observations are never prediction/training inputs for an
    earlier timestamp; predict() still restricts each fold to its own cutoff.
    """
    extended=stop+timedelta(days=1)
    following=(extended.replace(day=28)+timedelta(days=4)).replace(day=1)
    return min(following,END)


def source_hash():
    names=['research/high_win_research.py','research/high_win_filters.py',
           'research/minute_scan_research.py','research/derivative_features.py',
           'research/event_engine.py','research/event_strategy_research.py',
           'live/features.py','live/predictor.py']
    return hashlib.sha256(b''.join(name.encode()+bytes.fromhex(file_hash(ROOT/name)) for name in names)).hexdigest()


def simulate(raw,payout=.85,fee=0.,adverse_bps=0.):
    """Reserve at signal time; only outcomes strictly before a signal are known."""
    if not (0<payout<=1 and 0<=fee<1 and 0<=adverse_bps<10000):raise ValueError('Invalid stress scenario')
    pending,accepted=[],[]
    daily_losses=defaultdict(float);settled_net=0.
    for source in sorted(raw,key=lambda r:(r['signal_ms'],r['symbol'])):
        signal=int(source['signal_ms']);entry=int(source['entry_ms'])
        if source['symbol'] not in SYMBOLS or source['direction'] not in (-1,1):raise ValueError('Invalid event scope')
        if entry!=signal+60000 or source['settlement_ms']!=entry+1800000:raise ValueError('Invalid event clock')
        remaining=[]
        for trade in pending:
            if trade['settlement_ms']<signal:
                settled_net+=trade['net_units']
                if trade['outcome']=='loss':daily_losses[trade['settlement_ms']//DAY_MS]+=5
            else:remaining.append(trade)
        pending=remaining
        if any(t['settlement_ms']>entry for t in pending):continue
        if daily_losses[entry//DAY_MS]+len(pending)*5+5>25:continue
        if 1000+settled_net-len(pending)*5*(1+fee)<5*(1+fee):continue
        trade=dict(source)
        trade['entry']=source['entry']*(1+source['direction']*adverse_bps/10000)
        change=(trade['exit']-trade['entry'])*trade['direction']
        trade['outcome']='win' if change>0 else 'loss' if change<0 else 'refund'
        trade['net_units']=round(5*((payout if change>0 else -1 if change<0 else 0)-fee),8)
        trade['stake']=5.;trade['horizon_minutes']=30
        pending.append(trade);accepted.append(trade)
    return sorted(accepted,key=lambda r:(r['settlement_ms'],r['symbol']))


def evaluate(raw,start_ms,end_ms,stress=None):
    config={'payout':.85,'fee':0.,'adverse_bps':0.,**(stress or {})}
    allowed=[t for t in raw if t['signal_ms']>=start_ms and t['settlement_ms']<end_ms]
    ledger=simulate(allowed,**config)
    stats=event_stats(ledger,start_ms,end_ms,payout=config['payout'],fee=config['fee'],bootstrap=False)
    stats['raw_opportunities']=len(allowed)
    stats['by_symbol_counts']={symbol:sum(t['symbol']==symbol and t['outcome']!='refund' for t in ledger) for symbol in SYMBOLS}
    return ledger,stats


def select_on_discovery(raw_by_rule,rules,start_ms,end_ms,min_trades=500,min_per_symbol=100):
    """Ranking cannot inspect outcomes crossing the discovery endpoint."""
    z=NormalDist().inv_cdf(1-.05/(2*len(rules)))
    ranking=[]
    for rule in rules:
        _,stats=evaluate(raw_by_rule[rule['id']],start_ms,end_ms)
        eligible=stats['decisive_trades']>=min_trades and min(stats['by_symbol_counts'].values())>=min_per_symbol
        lower=wilson_lower(stats['wins'],stats['decisive_trades'],z) if eligible else None
        ranking.append({'rule':rule,'stats':stats,'eligible':eligible,'adjusted_lower':lower})
    valid=sorted((r for r in ranking if r['eligible']),key=lambda r:(-r['adjusted_lower'],r['rule']['id']))
    return (valid[0]['rule'] if valid else None),ranking


def paired_week_interval(selected,baseline,start,end,iterations=2000):
    """Use the same seven-day draws for both strategies, preserving covariance."""
    days=max(1,math.ceil((end-start)/DAY_MS));buckets=np.zeros((days,6))
    for offset,ledger in ((0,selected),(3,baseline)):
        for trade in ledger:
            index=(trade['settlement_ms']-start)//DAY_MS
            if not 0<=index<days:raise ValueError('Trade outside paired comparison')
            buckets[index,offset:offset+3]+=(trade['outcome']=='win',trade['outcome']!='refund',trade['net_units'])
    rng=np.random.default_rng(28913)
    starts=rng.integers(0,days,size=(iterations,math.ceil(days/7)))
    indices=((starts[:,:,None]+np.arange(7))%days).reshape(iterations,-1)[:,:days]
    total=buckets[indices].sum(axis=1)
    valid=(total[:,1]>0)&(total[:,4]>0)
    difference=(total[valid,0]/total[valid,1]-total[valid,3]/total[valid,4])
    net=total[:,2]-total[:,5]
    return {'block_days':7,'iterations':iterations,'nonempty_rate_draws':int(valid.sum()),
            'delta_win_rate_lower_95':float(np.quantile(difference,.025)) if len(difference) else None,
            'delta_win_rate_upper_95':float(np.quantile(difference,.975)) if len(difference) else None,
            'delta_net_lower_95':float(np.quantile(net,.025)),
            'delta_net_upper_95':float(np.quantile(net,.975))}


def prepare_opportunities(chunk_days=90):
    """Bound memory to a 90-day prediction slab plus sparse historical training.

    A one-day label extension prevents artificial loss of the last31minutes of
    each chunk. Only original monthly fit boundaries train models. No missing
    premium/funding minutes are filled; both assets must be ready to scan.
    """
    cache=ROOT/'research'/'.cache'/'high-win';cache.mkdir(parents=True,exist_ok=True)
    output=cache/'baseline-opportunities.csv.gz';metadata=cache/'prediction-manifest.json'
    prediction_sources=['research/minute_scan_research.py','live/features.py','live/predictor.py',
                        'research/derivative_features.py','research/qd_three_year_data.py','research/derivatives_data.py']
    prediction_key={name:file_hash(ROOT/name) for name in prediction_sources}
    prediction_key.update(start=str(START),end=str(END),chunk_days=chunk_days,pipeline_version=2)
    if output.exists() and metadata.exists():
        meta=json.loads(metadata.read_text(encoding='utf-8'))
        if meta.get('prediction_key')==prediction_key and meta.get('csv_sha256')==file_hash(output):
            print('Using hash-verified local chronological prediction cache',flush=True)
            return pd.read_csv(output,float_precision='round_trip'),meta
    slabs,folds,quality=[],[],[];cursor=START
    feature_digest=hashlib.sha256();begin=time.perf_counter()
    while cursor<END:
        stop=min(cursor+timedelta(days=chunk_days),END)
        print(f'CHUNK {cursor} to {stop}: verify archives and compute causal features',flush=True)
        frames,sources,checks=prepare(cursor,archive_end_for(stop),ROOT/'research'/'.cache')
        common=np.intersect1d(frames[SYMBOLS[0]].signal_ms.to_numpy(),frames[SYMBOLS[1]].signal_ms.to_numpy())
        for symbol in SYMBOLS:
            frame=frames[symbol]
            feature_digest.update(symbol.encode());feature_digest.update(pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype='<u8').tobytes())
            predicted,current=predict(frame,cursor,stop)
            valid=predicted.signal_ms.isin(common)
            removed=int((~valid).sum());predicted=predicted.loc[valid].copy()
            predicted=predicted.merge(frame[['signal_ms']+EXTRA],on='signal_ms',validate='one_to_one')
            p=predicted.p_up.to_numpy();direction=np.where(p>=.5,1,-1)
            z60=predicted.ret_60.to_numpy()/(np.maximum(predicted.rv_60.to_numpy(),1e-8)*np.sqrt(60))
            mask=(np.maximum(p,1-p)>=.56)&(direction==-np.sign(predicted.ret_60.to_numpy()))&(np.abs(z60)>=1)&(predicted.rv_ratio.to_numpy()<=1.25)
            predicted['direction']=direction
            slabs.append(predicted.loc[mask].copy())
            for fold in current:fold['chunk_start_utc']=str(cursor)
            folds.extend(current)
            quality.append({'symbol':symbol,'chunk_start':str(cursor),'chunk_end':str(stop),
                            'prediction_rows':len(predicted),'joint_unavailable_dropped':removed,
                            'baseline_opportunities':int(mask.sum()),'observations':checks[symbol],
                            'source_manifest_sha256':hashlib.sha256(json.dumps(sources[symbol],sort_keys=True).encode()).hexdigest()})
            del predicted
        del frames,frame,sources,checks,common;gc.collect()
        print(f'CHUNK DONE {stop}; elapsed {time.perf_counter()-begin:.1f}s',flush=True)
        cursor=stop
    combined=pd.concat(slabs,ignore_index=True).sort_values(['signal_ms','symbol']).reset_index(drop=True)
    if combined.duplicated(['symbol','signal_ms']).any():raise ValueError('Duplicated chunk boundary')
    combined.to_csv(output,index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    meta={'prediction_key':prediction_key,'csv_sha256':file_hash(output),'data_sha256':feature_digest.hexdigest(),
          'folds':folds,'quality':quality,'rows':len(combined),'created_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(metadata,meta)
    return combined,meta


def detail(ledger,start,end):
    stats=event_stats(ledger,start,end,payout=.85,fee=0.,bootstrap=False)
    stats['weekly_block_interval']=moving_week_interval(ledger,start,end)
    stats['by_symbol']={s:event_stats([t for t in ledger if t['symbol']==s],start,end,payout=.85,fee=0.,bootstrap=False) for s in SYMBOLS}
    stats['by_month']=_monthly(ledger,start,end,BASE_CAPITAL)
    stats['positive_months']=sum(row['net_units']>0 for row in stats['by_month'])
    stats['active_months']=sum(row['signals']>0 for row in stats['by_month'])
    return stats


def run(chunk_days=90):
    started=time.perf_counter();out=ROOT/'research';rules=candidates()
    plan={'version':1,'discovery_start_utc':str(START),'discovery_end_exclusive_utc':str(SPLIT),
          'validation_end_exclusive_utc':str(CONFIRM),'confirmation_end_exclusive_utc':str(END),
          'symbols':list(SYMBOLS),'scan_minutes':1,'holding_minutes':30,'entry_delay_minutes':1,
          'model':'unchanged60feature regularized logistic','train_days':180,'refit_days':30,'training_sample_minutes':30,
          'score_is_calibrated_probability':False,'rules':rules,'candidate_count':len(rules),
          'selection':'discovery only; minimum500 accepted decisive trades and100perasset; maximum Bonferroni-Wilson lower bound; lexical tie break',
          'minimum_validation_trades':400,'minimum_confirmation_trades':100,'target_win_rate':.58,
          'capital':BASE_CAPITAL,'simulation':'signal-time pending-risk reservation; settle only expiry<signal',
          'all_filter_signals_resimulated':True,'joint_asset_feature_availability_required':True,
          'post_selection_tuning_allowed':False,'historical_data_previously_researched':True,'pristine_holdout':False,
          'forward_validation':False,'local_only':True,'publish_or_deploy':False,'chunk_days':chunk_days,
          'source_sha256':source_hash(),'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'high-win-plan.json',plan)
    frame,provenance=prepare_opportunities(chunk_days)
    print('Predictions complete; evaluating16 fixed filters on discovery only',flush=True)
    # Keep one immutable input record per opportunity. Sixteen independent
    # dictionaries per row would undo the memory savings from chunked features.
    # simulate() copies an admitted row before adding any outcome/stress fields.
    base_records=frame.to_dict('records')
    raw={}
    for rule in rules:
        positions=np.flatnonzero(filter_mask(frame,rule))
        raw[rule['id']]=[base_records[index] for index in positions]
    first,split,confirm,end=map(utc_ms,(START,SPLIT,CONFIRM,END))
    selected,ranking=select_on_discovery(raw,rules,first,split)
    selection={'selected_rule':selected,'ranking':ranking,'selection_cutoff_ms':split,
               'locked_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':file_hash(out/'high-win-plan.json')}
    write_json(out/'high-win-selection.json',selection)
    print('LOCKED SELECTION '+json.dumps(selected,ensure_ascii=False),flush=True)
    chosen=raw[selected['id']] if selected else []
    windows={'discovery':(first,split),'validation':(split,confirm),'historical_confirmation':(confirm,end),'continuous_post_selection':(split,end)}
    selected_stats,baseline_stats,paired={}, {}, {}
    ledgers={}
    for name,(lo,hi) in windows.items():
        ledger,_=evaluate(chosen,lo,hi);base,_=evaluate(raw['baseline'],lo,hi)
        selected_stats[name]=detail(ledger,lo,hi);baseline_stats[name]=detail(base,lo,hi)
        paired[name]=paired_week_interval(ledger,base,lo,hi)
        if name=='continuous_post_selection':
            ledgers['selected']=ledger;ledgers['baseline']=base
        print(json.dumps({'period':name,'selected':{k:selected_stats[name][k] for k in ('signals','win_rate','net_units','max_drawdown_units')},
                          'baseline':{k:baseline_stats[name][k] for k in ('signals','win_rate','net_units','max_drawdown_units')}},ensure_ascii=False),flush=True)
    matrix=[]
    for row in ranking:
        rule=row['rule'];other={'rule':rule,'discovery':row['stats'],'discovery_eligible':row['eligible']}
        for name,(lo,hi) in windows.items():
            if name=='discovery':continue
            _,other[name]=evaluate(raw[rule['id']],lo,hi)
        matrix.append(other)
    stresses=[]
    for name,config in [('fee_1pct',{'fee':.01}),('adverse_index_entry_2bps',{'adverse_bps':2.}),('payout_80pct',{'payout':.8})]:
        _,stats=evaluate(chosen,split,end,config)
        stresses.append({'name':name,'configuration':config,'test':stats})
    validation=selected_stats['validation'];confirmation=selected_stats['historical_confirmation'];post=selected_stats['continuous_post_selection']
    be=1/1.85
    checks={'selection_exists':selected is not None,'selection_is_improvement_candidate':selected is not None and selected['id']!='baseline',
            'validation_min400':validation['decisive_trades']>=400,'confirmation_min100':confirmation['decisive_trades']>=100,
            'validation_58pct':(validation['win_rate'] or 0)>=.58,'confirmation_58pct':(confirmation['win_rate'] or 0)>=.58,
            'validation_profitable':validation['net_units']>0,'confirmation_profitable':confirmation['net_units']>0,
            'post_weekly_lower_above_break_even':(post['weekly_block_interval']['win_rate_lower_95'] or 0)>be,
            'post_weekly_net_lower_positive':(post['weekly_block_interval']['net_lower_95'] or 0)>0,
            'paired_post_win_rate_lower_positive':(paired['continuous_post_selection']['delta_win_rate_lower_95'] or 0)>0,
            'training_labels_strictly_settled':all(f['train_latest_settlement_ms']<f['fit_cutoff_ms'] for f in provenance['folds'])}
    for name,ledger in ledgers.items():
        pd.DataFrame(ledger).assign(rule_id=selected['id'] if name=='selected' and selected else 'baseline').to_csv(out/f'high-win-{name}-trades.csv',index=False)
    report={'schema_version':1,'engine':'local-high-win-filters-v1','plan':plan,'selection':selection,
            'selected':selected_stats,'baseline':baseline_stats,'paired_comparison':paired,'all_candidates':matrix,
            'stresses':stresses,'checks':checks,'research_gates_passed':all(checks.values()),'live_approved':False,
            'provenance':{'source_sha256':plan['source_sha256'],'data_sha256':provenance['data_sha256'],
                          'prediction_manifest_sha256':file_hash(ROOT/'research'/'.cache'/'high-win'/'prediction-manifest.json'),
                          'plan_sha256':selection['plan_sha256'],'runtime_seconds':round(time.perf_counter()-started,2),
                          'generated_at_utc':datetime.now(timezone.utc).isoformat(),'folds':provenance['folds'],'quality':provenance['quality']},
            'limitations':['All historical partitions have been examined in prior work; this is not pristine or prospective evidence.',
                'Bonferroni-Wilson discovery ranking does not remove serial dependence or all earlier strategy-search bias.',
                'One discovery-selected candidate is locked before later evaluation; no replacement is selected using later results.',
                'Weekly block intervals address short dependence only; structural change may persist beyond a week.',
                'Model directional scores remain uncalibrated. Monthly training uses the original half-hour population.',
                'Simulation uses minute index opens, constant85%payout, prompt observations and admission ordering; real contract quotes may differ.',
                'Every filter reruns shared-capital admissions from raw signals; fewer signals can increase measured win rate while reducing profit.',
                'Nothing in this run changes GitHub, server code, notifications, or live model settings.']}
    if source_hash()!=plan['source_sha256']:raise RuntimeError('Research sources changed during run')
    write_json(out/'high-win-results.json',report)
    print('LOCAL REPORT SAVED '+str(out/'high-win-results.json'),flush=True)
    return report


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--chunk-days',type=int,default=90)
    args=cli.parse_args()
    if not 7<=args.chunk_days<=90:cli.error('chunk-days must be7..90 to bound memory')
    run(args.chunk_days)
