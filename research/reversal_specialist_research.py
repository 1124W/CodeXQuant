"""One preregistered conditional reversal hypothesis; local, bounded research.

No grid expansion, remote operations or live-model changes. Discovery must
qualify before later outcomes can support the fixed candidate. Even passing
historical gates would remain exploratory because these dates were reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .consensus_search_research import verified_store_identity, LEDGER_COLUMNS
from .event_engine import DAY_MS, MINUTE_MS, SYMBOLS, KNOWN_UNAVAILABLE_DATES, utc_ms
from .high_win_research import START,SPLIT,CONFIRM,END,BASE_CAPITAL,detail,paired_week_interval,write_json,file_hash
from .margin_search_research import _fold_windows, load_baseline, assert_same_population, stress_detail
from .model_search_data import load_window
from .model_search_research import choose_discovery,simulate_frame
from .model_search_rules import directions as baseline_directions, candidates as baseline_candidates
from .reversal_specialist_models import specification,INPUT_FEATURES,fit_predict,reversal_sides,directions
from .adaptive_threshold_rules import availability_mask

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'research'/'.cache'/'reversal-specialist'
KEEP=['signal_ms','entry_ms','settlement_ms','entry','exit','ret_60','rv_60','rv_ratio']
PROJECTION=list(dict.fromkeys(['symbol']+list(INPUT_FEATURES)+KEEP))
BASELINE_ID='logistic_halfhour_0.56_exhaustion'


def source_hash():
    names=['research/reversal_specialist_research.py','research/reversal_specialist_models.py',
           'research/margin_search_research.py','research/margin_search_models.py','research/margin_search_rules.py',
           'research/consensus_search_research.py','research/adaptive_threshold_rules.py',
           'research/model_search_research.py','research/model_search_rules.py','research/model_search_models.py',
           'research/model_search_data.py','research/high_win_research.py','research/event_engine.py',
           'research/event_strategy_research.py','research/minute_scan_research.py',
           'research/derivative_features.py','research/derivatives_data.py','research/qd_three_year_data.py','live/features.py']
    return hashlib.sha256(b''.join(n.encode()+bytes.fromhex(file_hash(ROOT/n)) for n in names)).hexdigest()


def make_plan():
    rule=specification()
    return {'version':1,'candidate_count':1,'rules':[rule],
        'hypothesis':'A pooled side-oriented classifier trained only on fixed exhaustion states can distinguish failed impulses from persistent price pressure.',
        'discovery_start_utc':str(START),'discovery_end_exclusive_utc':str(SPLIT),
        'validation_end_exclusive_utc':str(CONFIRM),'confirmation_end_exclusive_utc':str(END),
        'selection':'Exactly one fixed candidate; discovery>=500 decisive,>=100eachasset,>=58%win,positive net. No replacement after later outcomes.',
        'additional_economic_gates':'Positive net after2bp adverse entry in validation, historical_confirmation and continuous_post_selection; no capital ruin.',
        'capital':dict(BASE_CAPITAL),'known_unavailable_dates':list(KNOWN_UNAVAILABLE_DATES),
        'calendar_applies_to_training_signals_and_baseline':True,
        'training':'PooledBTC/ETH,180days,only strictly settled labels,00/30minute grid,condition on exhaustion before fitting;30day refits.',
        'baseline_rule_id':BASELINE_ID,'baseline_comparison':'Same observations, calendar and shared-capital admission.',
        'single_mechanism_no_search_grid':True,'post_selection_changes_allowed':False,
        'historical_data_previously_researched':True,'pristine_holdout':False,'forward_validated':False,
        'local_only':True,'publish_or_deploy':False,'source_sha256':source_hash(),
        'frozen_at_utc':datetime.now(timezone.utc).isoformat()}


def _contract(identity):
    return {'pipeline_sha256':file_hash(ROOT/'research/reversal_specialist_research.py'),
            'models_sha256':file_hash(ROOT/'research/reversal_specialist_models.py'),
            'data_code_sha256':file_hash(ROOT/'research/model_search_data.py'),
            'calendar_code_sha256':file_hash(ROOT/'research/adaptive_threshold_rules.py'),
            'known_unavailable_dates':list(KNOWN_UNAVAILABLE_DATES),
            'feature_store_identity':identity,'pipeline_version':1}


def _load_projected(first,stop,*,training_cutoff=None):
    """Drop unused columns before retaining the second asset's minute slab."""
    parts=[]
    for symbol in SYMBOLS:
        frame=load_window(symbol,first,stop)
        if training_cutoff is not None:
            mask=((frame.settlement_ms<training_cutoff)&(frame.signal_ms%(30*MINUTE_MS)==0)
                  &(reversal_sides(frame)!=0))
            frame=frame.loc[mask,PROJECTION].copy()
        else:frame=frame[PROJECTION].copy()
        parts.append(frame)
        del frame
        gc.collect()
    combined=pd.concat(parts,ignore_index=True)
    return combined.sort_values(['signal_ms','symbol']).reset_index(drop=True)


def _validate(part,metadata,cutoff,first,stop):
    model=metadata['model_metadata']
    if (metadata['first_signal_bound_ms']!=first or metadata['end_exclusive_ms']!=stop
            or model['fit_cutoff_ms']!=cutoff or model['specification']!=specification()
            or model['prediction_count']!=len(part) or not len(part)):
        raise ValueError('Specialist checkpoint metadata differs from fixed specification')
    times=part.signal_ms.to_numpy()
    if (not np.isfinite(times).all() or np.any(times!=np.floor(times)) or np.any(times%MINUTE_MS)
            or np.any(times<max(first,cutoff)) or np.any(times>=min(stop,cutoff+30*DAY_MS))
            or part.duplicated(['symbol','signal_ms']).any() or not part.symbol.isin(SYMBOLS).all()
            or model['test_start_ms']!=int(times.min()) or model['test_end_exclusive_ms']!=int(times.max())+MINUTE_MS
            or (part.entry_ms!=part.signal_ms+MINUTE_MS).any()
            or (part.settlement_ms!=part.signal_ms+31*MINUTE_MS).any()
            or (part.model_fit_cutoff_ms!=cutoff).any()
            or model['latest_settlement_ms']>=cutoff
            or (part.train_latest_settlement_ms!=model['latest_settlement_ms']).any()):
        raise ValueError('Invalid specialist prediction/provenance clocks')
    score=part.p_reversal.to_numpy()
    if not np.isfinite(score).all() or np.any(score<0) or np.any(score>1):
        raise ValueError('Invalid specialist reversal scores')
    if (model['training_count']<specification()['minimum_training_rows']
            or set(model['training_by_symbol'])!=set(SYMBOLS)
            or min(model['training_by_symbol'].values())<specification()['minimum_training_rows_per_asset']):
        raise ValueError('Insufficient cached conditioned training population')


def prediction_phase(start,end,identity,cache_dir=None):
    folder=Path(cache_dir or CACHE);folder.mkdir(parents=True,exist_ok=True)
    contract=_contract(identity);parts=[];folds=[];files=[]
    for cutoff,first,stop in _fold_windows(start,end):
        path=folder/f'pooled-{cutoff}-{first}-{stop}.npz';meta_path=path.with_suffix('.json')
        if path.exists() or meta_path.exists():
            if not path.is_file() or not meta_path.is_file():raise ValueError('Incomplete specialist checkpoint')
            metadata=json.loads(meta_path.read_text(encoding='utf-8'))
            if metadata.get('contract')!=contract or metadata.get('sha256')!=file_hash(path):
                raise ValueError('Stale or corrupt specialist checkpoint')
            with np.load(path,allow_pickle=False) as values:
                part=pd.DataFrame({name:values[name] for name in values.files})
            print(f'REUSE SPECIALIST {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()}',flush=True)
        else:
            print(f'FIT SPECIALIST {datetime.fromtimestamp(cutoff/1000,timezone.utc).date()}',flush=True)
            train=_load_projected(cutoff-180*DAY_MS,cutoff,training_cutoff=cutoff)
            test=_load_projected(first,stop)
            scores,model=fit_predict(train,test,cutoff,progress=lambda m:print('  '+m,flush=True))
            part=test[['symbol']+KEEP].copy()
            part['symbol']=part.symbol.astype('U8')
            part['p_reversal']=scores.p_reversal.to_numpy()
            part['model_fit_cutoff_ms']=cutoff;part['train_latest_settlement_ms']=model['latest_settlement_ms']
            metadata={'contract':contract,'model_metadata':model,'first_signal_bound_ms':first,'end_exclusive_ms':stop}
            _validate(part,metadata,cutoff,first,stop)
            # Store asset tags as Unicode arrays, so np.load never permits pickle.
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,**{name:part[name].to_numpy(dtype='U8') if name=='symbol' else part[name].to_numpy() for name in part})
            temporary.replace(path);metadata['sha256']=file_hash(path);write_json(meta_path,metadata)
            del train,test,scores
            gc.collect()
        _validate(part,metadata,cutoff,first,stop)
        parts.append(part);folds.append(metadata['model_metadata'])
        files.append({'file':path.name,'sha256':metadata['sha256'],'metadata_sha256':file_hash(meta_path),
                      'fit_cutoff_ms':cutoff,'first_signal_bound_ms':first,'end_exclusive_ms':stop,'rows':len(part)})
        write_json(folder/'progress.json',{'contract':contract,'phase_start_utc':str(start),'phase_end_exclusive_utc':str(end),
                   'completed_folds':files,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(['symbol','signal_ms']).any():raise ValueError('Overlapping specialist slabs')
    counts=frame.groupby('signal_ms').symbol.transform('count');dropped=int((counts!=2).sum())
    frame=frame.loc[counts==2].sort_values(['signal_ms','symbol']).reset_index(drop=True)
    return frame,{'feature_store_identity':identity,'folds':folds,'verified_cache_files':files,
                  'joint_asset_rows_dropped':dropped,'prediction_rows':len(frame)}


def historical_checks(selected,summaries,stressed,folds):
    v,c,post=(summaries[n] for n in ('validation','historical_confirmation','continuous_post_selection'))
    checks={'selection_exists':selected is not None,'validation_min400':v['decisive_trades']>=400,
        'confirmation_min100':c['decisive_trades']>=100,'validation_58pct':(v['win_rate'] or 0)>=.58,
        'confirmation_58pct':(c['win_rate'] or 0)>=.58,'validation_positive':v['net_units']>0,
        'confirmation_positive':c['net_units']>0,
        'weekly_lower_above_break_even':(post['weekly_block_interval']['win_rate_lower_95'] or 0)>1/1.85,
        'weekly_net_lower_positive':(post['weekly_block_interval']['net_lower_95'] or 0)>0,
        'all_models_converged':bool(folds) and all(f['training_converged'] for f in folds),
        'training_labels_settled':bool(folds) and all(f['latest_settlement_ms']<f['fit_cutoff_ms'] for f in folds)}
    for name in ('validation','historical_confirmation','continuous_post_selection'):
        checks['adverse_2bps_'+name+'_positive']=bool(stressed.get(name)) and stressed[name]['net_units']>0
        checks['adverse_2bps_'+name+'_no_ruin']=bool(stressed.get(name,{}).get('bankroll_can_fund_next_stake',False))
    return checks


def _save_ledger(path,ledger):
    pd.DataFrame(ledger,columns=LEDGER_COLUMNS).to_csv(path,index=False)
    return {'file':path.name,'sha256':file_hash(path),'rows':len(ledger)}


def run(phase='all',output_dir=None):
    if phase not in ('all','discovery'):raise ValueError('Unknown study phase')
    began=time.perf_counter();out=Path(output_dir or ROOT/'research');plan=make_plan();rule=plan['rules'][0]
    write_json(out/'reversal-specialist-plan.json',plan)
    identity=verified_store_identity()
    discovery,discovery_meta=prediction_phase(START,SPLIT,identity)
    discovery_ledger,discovery_stats=simulate_frame(discovery,directions(discovery),utc_ms(START),utc_ms(SPLIT),rule=rule)
    selected,ranking=choose_discovery([{'rule':rule,'discovery':discovery_stats}])
    selection={'selected_rule':selected,'ranking':ranking,'selection_cutoff_ms':utc_ms(SPLIT),
               'locked_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':file_hash(out/'reversal-specialist-plan.json')}
    write_json(out/'reversal-specialist-selection.json',selection)
    print('LOCKED SPECIALIST '+json.dumps({'selected':selected,'discovery':discovery_stats},ensure_ascii=False),flush=True)
    if phase=='discovery':
        if source_hash()!=plan['source_sha256']:raise RuntimeError('Source changed during specialist discovery')
        return selection
    base_rule=next(r for r in baseline_candidates() if r['id']==BASELINE_ID)
    base,base_discovery_meta=load_baseline(START,SPLIT,identity);assert_same_population(discovery,base)
    base_signal=baseline_directions(base,base_rule);base_signal[~availability_mask(base.signal_ms.to_numpy())]=0
    base_discovery,_=simulate_frame(base,base_signal,utc_ms(START),utc_ms(SPLIT),rule=base_rule)
    baseline={'discovery':detail(base_discovery,utc_ms(START),utc_ms(SPLIT))}
    candidate={'discovery':detail(discovery_ledger,utc_ms(START),utc_ms(SPLIT))}
    ledgers={'discovery':discovery_ledger};baseline_ledgers={'discovery':base_discovery}
    del discovery,base,base_signal
    gc.collect()
    later,later_meta=prediction_phase(SPLIT,END,identity)
    signal=directions(later)
    ranges={'validation':(utc_ms(SPLIT),utc_ms(CONFIRM)),
            'historical_confirmation':(utc_ms(CONFIRM),utc_ms(END)),
            'continuous_post_selection':(utc_ms(SPLIT),utc_ms(END))}
    stressed={};stress_ledgers={};record={'rule':rule,'discovery':discovery_stats}
    for name,(first,stop) in ranges.items():
        ledger,stats=simulate_frame(later,signal,first,stop,rule=rule)
        stressed_ledger,_=simulate_frame(later,signal,first,stop,stress={'adverse_bps':2.},rule=rule)
        record[name]=stats;candidate[name]=detail(ledger,first,stop);ledgers[name]=ledger
        stressed[name]=stress_detail(stressed_ledger,first,stop);stress_ledgers[name]=stressed_ledger
    record['continuous_adverse_entry_2bps']=stressed['continuous_post_selection']
    base,base_later_meta=load_baseline(SPLIT,END,identity);assert_same_population(later,base)
    base_signal=baseline_directions(base,base_rule);base_signal[~availability_mask(base.signal_ms.to_numpy())]=0
    paired={};files=[]
    all_ranges={'discovery':(utc_ms(START),utc_ms(SPLIT)),**ranges}
    for name,(first,stop) in all_ranges.items():
        if name!='discovery':
            baseline_ledgers[name]=simulate_frame(base,base_signal,first,stop,rule=base_rule)[0]
            baseline[name]=detail(baseline_ledgers[name],first,stop)
        paired[name]=paired_week_interval(ledgers[name],baseline_ledgers[name],first,stop)
        files.append(_save_ledger(out/f'reversal-specialist-candidate-{name}-trades.csv',ledgers[name]))
        files.append(_save_ledger(out/f'reversal-specialist-baseline-{name}-trades.csv',baseline_ledgers[name]))
        files.append(_save_ledger(out/f'reversal-specialist-{name}-trades.csv',ledgers[name] if selected else []))
        if name in stress_ledgers:files.append(_save_ledger(out/f'reversal-specialist-adverse2bps-{name}-trades.csv',stress_ledgers[name]))
    # Candidate outcomes are disclosed even if discovery failed; the selected
    # portfolio then remains empty rather than silently promoting that candidate.
    summaries=candidate if selected else {name:detail([],first,stop) for name,(first,stop) in all_ranges.items()}
    selected_stress=stressed if selected else {}
    folds=discovery_meta['folds']+later_meta['folds']
    checks=historical_checks(selected,summaries,selected_stress,folds)
    if source_hash()!=plan['source_sha256']:raise RuntimeError('Source changed during specialist research')
    result={'schema_version':1,'engine':'fixed-pooled-conditional-exhaustion-reversal-v1','plan':plan,
        'selection':selection,'selected':summaries,'candidate':candidate,'baseline':baseline,
        'all_candidates':[record],'paired_comparison':paired,
        'paired_comparison_scope':'fixed_candidate_disclosure_not_promotion',
        'stresses':[{'name':'adverse_index_entry_2bps','configuration':{'adverse_bps':2.},
                     'scope':'fixed_candidate_disclosure_not_promotion','test':stressed['continuous_post_selection'],'periods':stressed}],
        'checks':checks,'historical_research_gates_passed':all(checks.values()),'live_approved':False,
        'provenance':{'feature_store_identity':identity,'source_sha256':plan['source_sha256'],
            'plan_sha256':selection['plan_sha256'],'selection_sha256':file_hash(out/'reversal-specialist-selection.json'),
            'discovery':discovery_meta,'later':later_meta,'baseline_discovery':base_discovery_meta,
            'baseline_later':base_later_meta,'ledgers':files,'runtime_seconds':round(time.perf_counter()-began,2),
            'generated_at_utc':datetime.now(timezone.utc).isoformat()},
        'limitations':['One fixed mechanistic hypothesis after repeated reuse of this history; this is exploratory, not pristine or forward confirmation.',
            'Side-normalized inputs and conditional training can reduce irrelevant population mixing but cannot guarantee a durable edge.',
            'Historical aggregate flow and premium features omit order-book depth and liquidation-event observations.',
            'Raw reversal scores are not calibrated probabilities; admitted minute-level portfolio outcomes differ from halfhour training samples.',
            'Known contract-unavailable dates are incomplete; index-minute execution prices are proxies for actual strike and expiry.',
            'No live model, repository, server, alert, order or deployment was changed.']}
    write_json(out/'reversal-specialist-results.json',result)
    print(json.dumps({'selected':selected,'candidate_later':{n:{k:candidate[n][k] for k in ('signals','win_rate','net_units')} for n in ranges},'checks':checks,'runtime_seconds':result['provenance']['runtime_seconds']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('all','discovery'),default='all')
    run(parser.parse_args().phase)
