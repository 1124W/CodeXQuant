"""Local-only model/sampling study with one discovery-locked selection.

No network, publication, live model replacement, bot messages, or orders.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import date,datetime,timedelta,timezone
import gc
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import pandas as pd

from .event_engine import DAY_MS,event_stats,utc_ms
from .event_strategy_research import wilson_lower
from .high_win_research import (START,SPLIT,CONFIRM,END,detail,paired_week_interval,
                                write_json,file_hash,BASE_CAPITAL)
from .minute_scan_research import cutoff_for
from .model_search_models import specs,fit_predict
from .model_search_rules import candidates,directions

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'research'/'.cache'/'model-search'
SYMBOLS=('BTCUSDT','ETHUSDT')
KEEP=['signal_ms','entry_ms','settlement_ms','entry','exit','ret_15','ret_60','ret_240',
      'rv_60','rv_240','rv_ratio','ret_3']


def source_hash():
    names=['research/model_search_research.py','research/model_search_models.py',
           'research/model_search_rules.py','research/model_search_data.py',
           'research/high_win_research.py','research/event_engine.py','live/features.py']
    return hashlib.sha256(b''.join(name.encode()+bytes.fromhex(file_hash(ROOT/name)) for name in names)).hexdigest()


def simulate_frame(frame,direction,start_ms,end_ms,stress=None,rule=None):
    """Iterate numeric candidate indices; only admitted events become dicts."""
    config={'payout':.85,'fee':0.,'adverse_bps':0.,**(stress or {})}
    payout,fee,bps=config['payout'],config['fee'],config['adverse_bps']
    if not 0<payout<=1 or not 0<=fee<1 or not 0<=bps<10000:raise ValueError('Invalid simulation scenario')
    direction=np.asarray(direction)
    if direction.shape!=(len(frame),) or not np.isin(direction,[-1,0,1]).all():raise ValueError('Invalid directions')
    stamps=frame.signal_ms.to_numpy(dtype=np.int64)
    entries=frame.entry_ms.to_numpy(dtype=np.int64);expiries=frame.settlement_ms.to_numpy(dtype=np.int64)
    symbols=frame.symbol.to_numpy();entry_prices=frame.entry.to_numpy();exit_prices=frame.exit.to_numpy()
    if (not frame.symbol.isin(SYMBOLS).all() or np.any(entries!=stamps+60000)
            or np.any(expiries!=entries+1800000)):raise ValueError('Invalid event scope/clock')
    if not np.isfinite(entry_prices).all() or not np.isfinite(exit_prices).all() or np.any(entry_prices<=0) or np.any(exit_prices<=0):raise ValueError('Invalid observed index prices')
    indices=np.flatnonzero((direction!=0)&(stamps>=start_ms)&(expiries<end_ms))
    indices=indices[np.lexsort(((symbols[indices]=='ETHUSDT').astype(np.int8),stamps[indices]))]
    pending=[];accepted=[];net=0.;loss=defaultdict(float)
    fit=frame.model_fit_cutoff_ms.to_numpy() if 'model_fit_cutoff_ms' in frame else None
    latest=frame.train_latest_settlement_ms.to_numpy() if 'train_latest_settlement_ms' in frame else None
    for index in indices:
        stamp=int(stamps[index]);entry_ms=int(entries[index])
        remain=[]
        for trade in pending:
            if trade['settlement_ms']<stamp:
                net+=trade['net_units']
                if trade['outcome']=='loss':loss[trade['settlement_ms']//DAY_MS]+=5
            else:remain.append(trade)
        pending=remain
        if any(t['settlement_ms']>entry_ms for t in pending):continue
        if loss[entry_ms//DAY_MS]+len(pending)*5+5>25:continue
        if 1000+net-len(pending)*5*(1+fee)<5*(1+fee):continue
        side=int(direction[index]);opened=float(entry_prices[index])*(1+side*bps/10000)
        closed=float(exit_prices[index]);change=(closed-opened)*side
        outcome='win' if change>0 else 'loss' if change<0 else 'refund'
        trade={'symbol':str(symbols[index]),'signal_ms':stamp,'entry_ms':entry_ms,
               'settlement_ms':int(expiries[index]),'entry':opened,'exit':closed,'direction':side,
               'outcome':outcome,'net_units':round(5*((payout if change>0 else -1 if change<0 else 0)-fee),8),
               'stake':5.,'horizon_minutes':30}
        if fit is not None:trade['model_fit_cutoff_ms']=int(fit[index])
        if latest is not None:trade['train_latest_settlement_ms']=int(latest[index])
        if rule:trade['rule_id']=rule['id']
        pending.append(trade);accepted.append(trade)
    accepted.sort(key=lambda t:(t['settlement_ms'],t['symbol']))
    stats=event_stats(accepted,start_ms,end_ms,payout=payout,fee=fee,bootstrap=False)
    stats['raw_opportunities']=len(indices)
    stats['by_symbol_counts']={s:sum(t['symbol']==s and t['outcome']!='refund' for t in accepted) for s in SYMBOLS}
    return accepted,stats


def choose_discovery(records,min_trades=500,min_per_symbol=100):
    """Only the discovery field is consulted; no later winner can replace it."""
    z=NormalDist().inv_cdf(1-.05/(2*len(records)))
    ranking=[]
    for record in records:
        s=record['discovery']
        eligible=(s['decisive_trades']>=min_trades and
                  min(s['by_symbol_counts'].get(asset,0) for asset in SYMBOLS)>=min_per_symbol and
                  (s['win_rate'] or 0)>=.58 and s['net_units']>0)
        lower=wilson_lower(s['wins'],s['decisive_trades'],z) if eligible else None
        ranking.append({'rule':record['rule'],'discovery':s,'eligible':eligible,'adjusted_lower':lower})
    valid=sorted((r for r in ranking if r['eligible']),key=lambda r:(-r['adjusted_lower'],r['rule']['id']))
    return (valid[0]['rule'] if valid else None),ranking


def prediction_phase(start,end,store_identity):
    from .model_search_data import load_window
    out=CACHE/'predictions';out.mkdir(parents=True,exist_ok=True)
    contract={'models_sha256':file_hash(ROOT/'research'/'model_search_models.py'),
              'data_code_sha256':file_hash(ROOT/'research'/'model_search_data.py'),
              'feature_store_identity':store_identity,'pipeline_version':1}
    cutoff=utc_ms(cutoff_for(start));start_ms,end_ms=utc_ms(start),utc_ms(end)
    parts,folds=[],[]
    while cutoff<end_ms:
        stop=min(cutoff+30*DAY_MS,end_ms);first=max(start_ms,cutoff)
        for symbol in SYMBOLS:
            path=out/f'{symbol}-{cutoff}-{first}-{stop}.npz';meta_path=path.with_suffix('.json')
            reuse=False
            if path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text(encoding='utf-8'))
                reuse=meta.get('contract')==contract and meta.get('sha256')==file_hash(path)
            if reuse:
                with np.load(path,allow_pickle=False) as data:predicted=pd.DataFrame({k:data[k] for k in data.files})
                print(f'REUSE {symbol} {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()}',flush=True)
            else:
                print(f'FIT {symbol} {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()} six model variants',flush=True)
                train=load_window(symbol,cutoff-180*DAY_MS,cutoff)
                test=load_window(symbol,first,stop)
                probabilities,metadata=fit_predict(train,test,cutoff,progress=lambda message:print('  '+message,flush=True))
                predicted=test[KEEP].reset_index(drop=True)
                for column in probabilities:predicted[column]=probabilities[column].to_numpy()
                predicted['model_fit_cutoff_ms']=cutoff
                predicted['train_latest_settlement_ms']=max(v['latest_settlement_ms'] for v in metadata['models'].values())
                np.savez(path,**{name:predicted[name].to_numpy() for name in predicted})
                meta={'contract':contract,'sha256':file_hash(path),'model_metadata':metadata,'symbol':symbol}
                write_json(meta_path,meta)
                del train,test,probabilities;gc.collect()
            predicted['symbol']=symbol
            parts.append(predicted);folds.append(meta['model_metadata'])
        cutoff=stop
    combined=pd.concat(parts,ignore_index=True);del parts;gc.collect()
    # The service evaluates neither coin unless both current feature rows exist.
    counts=combined.groupby('signal_ms').symbol.transform('count')
    dropped=int((counts!=2).sum())
    combined=combined.loc[counts==2].sort_values(['signal_ms','symbol']).reset_index(drop=True)
    if combined.duplicated(['signal_ms','symbol']).any():raise ValueError('Overlapping prediction slabs')
    return combined,{'folds':folds,'joint_asset_rows_dropped':dropped,'prediction_rows':len(combined)}


def run(phase='all'):
    from .model_search_data import ensure_feature_store
    begin=time.perf_counter();rules=candidates();out=ROOT/'research'
    plan={'version':1,'models':specs(),'candidate_count':len(rules),'rules':rules,
          'discovery_start_utc':str(START),'discovery_end_exclusive_utc':str(SPLIT),
          'validation_end_exclusive_utc':str(CONFIRM),'confirmation_end_exclusive_utc':str(END),
          'train_days':180,'refit_days':30,'data_sampling':'full completed minutes; model-specific train stride; inference every minute',
          'selection':'discovery only; >=500decisive, >=100eachasset, >=58%pointwin, positivePNL; maxBonferroniWilsonlower; lexicaltie',
          'capital':BASE_CAPITAL,'pending_outcomes_reserved':True,'historical_data_previously_researched':True,
          'pristine_holdout':False,'forward_validated':False,'post_selection_changes_allowed':False,
          'local_only':True,'publish_or_deploy':False,'source_sha256':source_hash(),
          'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'model-search-plan.json',plan)
    store=ensure_feature_store()
    identity=hashlib.sha256(json.dumps(store,sort_keys=True).encode()).hexdigest()
    discovery,discovery_meta=prediction_phase(START,SPLIT,identity)
    records=[]
    for number,rule in enumerate(rules,1):
        _,stats=simulate_frame(discovery,directions(discovery,rule),utc_ms(START),utc_ms(SPLIT),rule=rule)
        records.append({'rule':rule,'discovery':stats})
        if number%10==0:print(f'DISCOVERY {number}/{len(rules)} evaluated',flush=True)
    selected,ranking=choose_discovery(records)
    selection={'selected_rule':selected,'ranking':ranking,'selection_cutoff_ms':utc_ms(SPLIT),
               'locked_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':file_hash(out/'model-search-plan.json')}
    write_json(out/'model-search-selection.json',selection)
    print('LOCKED MODEL CANDIDATE '+json.dumps(selected,ensure_ascii=False),flush=True)
    if phase=='discovery':return selection
    selected_discovery,_=simulate_frame(discovery,directions(discovery,selected),utc_ms(START),utc_ms(SPLIT),rule=selected) if selected else ([],{})
    del discovery;gc.collect()
    later,later_meta=prediction_phase(SPLIT,END,identity)
    ranges={'validation':(utc_ms(SPLIT),utc_ms(CONFIRM)),
            'historical_confirmation':(utc_ms(CONFIRM),utc_ms(END)),
            'continuous_post_selection':(utc_ms(SPLIT),utc_ms(END))}
    matrix=[];selected_ledgers={};baseline_ledgers={}
    baseline_id='logistic_halfhour_0.56_exhaustion'
    for number,row in enumerate(records,1):
        rule=row['rule'];direction=directions(later,rule);measured=dict(row)
        for name,(first,last) in ranges.items():
            ledger,stats=simulate_frame(later,direction,first,last,rule=rule)
            measured[name]=stats
            if selected and rule['id']==selected['id']:selected_ledgers[name]=ledger
            if rule['id']==baseline_id:baseline_ledgers[name]=ledger
        matrix.append(measured)
        if number%10==0:print(f'LATER {number}/{len(rules)} evaluated without reselection',flush=True)
    summaries={};baseline={};paired={}
    for name,(first,last) in ranges.items():
        ledger=selected_ledgers.get(name,[]);base=baseline_ledgers[name]
        summaries[name]=detail(ledger,first,last);baseline[name]=detail(base,first,last)
        paired[name]=paired_week_interval(ledger,base,first,last)
        print(json.dumps({'period':name,'selected':{k:summaries[name][k] for k in ('signals','win_rate','net_units','max_drawdown_units')}},ensure_ascii=False),flush=True)
    summaries['discovery']=detail(selected_discovery,utc_ms(START),utc_ms(SPLIT))
    stresses=[]
    if selected:
        direction=directions(later,selected)
        for name,config in [('adverse_index_entry_2bps',{'adverse_bps':2.}),('fee_1pct',{'fee':.01}),('payout_80pct',{'payout':.8})]:
            _,stats=simulate_frame(later,direction,utc_ms(SPLIT),utc_ms(END),stress=config,rule=selected)
            stresses.append({'name':name,'configuration':config,'test':stats})
    v=summaries['validation'];c=summaries['historical_confirmation'];post=summaries['continuous_post_selection']
    checks={'selection_exists':selected is not None,'validation_min400':v['decisive_trades']>=400,
            'confirmation_min100':c['decisive_trades']>=100,'validation_58pct':(v['win_rate'] or 0)>=.58,
            'confirmation_58pct':(c['win_rate'] or 0)>=.58,'validation_positive':v['net_units']>0,
            'confirmation_positive':c['net_units']>0,
            'weekly_lower_above_break_even':(post['weekly_block_interval']['win_rate_lower_95'] or 0)>1/1.85,
            'weekly_net_lower_positive':(post['weekly_block_interval']['net_lower_95'] or 0)>0,
            'all_models_converged':all(m['training_converged'] for f in discovery_meta['folds']+later_meta['folds'] for m in f['models'].values()),
            'training_labels_settled':all(m['latest_settlement_ms']<f['fit_cutoff_ms'] for f in discovery_meta['folds']+later_meta['folds'] for m in f['models'].values())}
    for name,ledger in selected_ledgers.items():pd.DataFrame(ledger).to_csv(out/f'model-search-{name}-trades.csv',index=False)
    result={'schema_version':1,'engine':'local-model-and-sampling-v1','plan':plan,'selection':selection,
            'selected':summaries,'baseline':baseline,'all_candidates':matrix,'paired_comparison':paired,
            'stresses':stresses,'checks':checks,'historical_research_gates_passed':all(checks.values()),
            'live_approved':False,'provenance':{'feature_store_identity':identity,'source_sha256':plan['source_sha256'],
            'discovery':discovery_meta,'later':later_meta,'runtime_seconds':round(time.perf_counter()-begin,2),
            'generated_at_utc':datetime.now(timezone.utc).isoformat()},
            'limitations':['Repeated search of already researched history is exploratory, not pristine confirmation.',
                'No later winner replaces the discovery-selected candidate. All60laterresults are disclosed for audit only.',
                'Minute training labels overlap; raw model scores remain uncalibrated and sample counts are not independent observations.',
                'Seven-day blocks cannot rule out longer structural dependence. Point58% is not proof truefuturewin>=58%.',
                'Exact index minute opens are proxies for real eventcontract strike/expiry quotes; execution uncertainty is material.',
                'No production program, notification, repository remote or server was changed.']}
    if source_hash()!=plan['source_sha256']:raise RuntimeError('Source changed during model research')
    write_json(out/'model-search-results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('all','discovery'),default='all')
    run(parser.parse_args().phase)
