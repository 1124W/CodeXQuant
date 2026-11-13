"""Discovery-locked confidence extension using only completed numeric caches.

Never trains, downloads, rebuilds features, or changes the earlier study.
The protocol JSON was frozen before the evaluator and before later inspection.
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

from .high_win_research import (START,SPLIT,CONFIRM,END,detail,paired_week_interval,
                                file_hash,write_json)
from .event_engine import SYMBOLS,utc_ms
from .metrics_threshold_rules import candidates,matched_control,directions
from .metrics_data import verify_metrics_store,store_identity
from .metrics_search_research import (CACHE,KEEP,PROBABILITY_COLUMNS,_contract,
    _validate_prediction,source_hash as stage7_source_hash,incremental_evidence)
from .margin_search_research import (_fold_windows,_combine,_save_ledger,
                                     stress_detail,historical_checks)
from .model_search_research import choose_discovery,simulate_frame

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'research'


def implementation_hash():
    names=['research/metrics_threshold_research.py','research/metrics_threshold_rules.py']
    return hashlib.sha256(bytes.fromhex(stage7_source_hash())+b''.join(
        name.encode()+bytes.fromhex(file_hash(ROOT/name)) for name in names)).hexdigest()


def read_plan(out):
    path=Path(out)/'metrics-threshold-plan.json'
    plan=json.loads(path.read_text(encoding='utf-8'))
    if (plan.get('rules')!=candidates() or plan.get('matched_controls')!=[matched_control(r) for r in candidates()]
            or plan.get('selection_candidate_count')!=16 or plan.get('matched_control_count')!=16
            or plan.get('thresholds')!=[.56,.60,.64,.68]
            or plan.get('rules_source_sha256')!=file_hash(ROOT/'research/metrics_threshold_rules.py')
            or plan.get('stage7_source_sha256')!=stage7_source_hash()
            or plan.get('stage7_plan_sha256')!=file_hash(ROOT/'research/metrics-search-plan.json')):
        raise ValueError('Frozen confidence plan or its earlier-study dependencies changed')
    return plan


def read_identities():
    """Manifest metadata only; never load original feature arrays or raw CSVs."""
    path=ROOT/'research/.cache/model-search/features/manifest.json'
    base=json.loads(path.read_text(encoding='utf-8'))
    if not base.get('complete') or base.get('schema_version')!=1 or len(base.get('features',[]))!=60:
        raise ValueError('Incomplete original feature manifest')
    metrics=verify_metrics_store(verify_files=False)
    return {'feature_store_identity':hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),
            'metrics_store_identity':store_identity(metrics),
            'base_manifest_sha256':file_hash(path),
            'metrics_manifest_sha256':file_hash(ROOT/'research/.cache/metrics/manifest.json')}


def freeze_implementation(out,plan,identities):
    path=Path(out)/'metrics-threshold-implementation.json'
    expected={'plan_sha256':file_hash(Path(out)/'metrics-threshold-plan.json'),
              'implementation_sha256':implementation_hash(),'source_identities':identities,
              'cache_policy':'read-only validated numeric arrays; no fit path or fallback'}
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if any(saved.get(key)!=value for key,value in expected.items()):
            raise ValueError('Frozen implementation or source identities changed')
        return saved
    result={**expected,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(path,result)
    return result


def load_cached_phase(start,end,identities,cache_dir=None):
    """Preflight all slabs, then independently validate paired proofs and clocks."""
    folder=Path(cache_dir or CACHE)
    expected=_contract(identities['feature_store_identity'],identities['metrics_store_identity'])
    validated=[]
    for cutoff,first,stop in _fold_windows(start,end):
        for symbol in SYMBOLS:
            path=folder/f'{symbol}-{cutoff}-{first}-{stop}.npz';meta_path=path.with_suffix('.json')
            if not path.is_file() or not meta_path.is_file():
                raise ValueError('Incomplete read-only paired cache: '+path.name)
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('contract')!=expected or meta.get('sha256')!=file_hash(path):
                raise ValueError('Stale or corrupt read-only paired cache: '+path.name)
            validated.append((symbol,cutoff,first,stop,path,meta_path,meta))
    parts=[];folds=[];files=[]
    columns=KEEP+PROBABILITY_COLUMNS+['model_fit_cutoff_ms','train_latest_settlement_ms']
    for symbol,cutoff,first,stop,path,meta_path,meta in validated:
        with np.load(path,allow_pickle=False) as values:
            part=pd.DataFrame({name:values[name] for name in columns})
        _validate_prediction(part,meta,symbol,cutoff,first,stop)
        part['symbol']=symbol;parts.append(part);folds.append(meta['model_metadata'])
        files.append({'file':path.name,'sha256':meta['sha256'],'metadata_sha256':file_hash(meta_path),
                      'symbol':symbol,'fit_cutoff_ms':cutoff,'rows':len(part)})
    frame,dropped=_combine(parts)
    del parts;gc.collect()
    return frame,{'folds':folds,'verified_cache_files':files,'prediction_rows':len(frame),
                  'joint_asset_rows_dropped':dropped}


def evaluate_discovery(frame):
    records=[];controls=[]
    for rule in candidates():
        for chosen,destination in ((rule,records),(matched_control(rule),controls)):
            _,stats=simulate_frame(frame,directions(frame,chosen),utc_ms(START),utc_ms(SPLIT),rule=chosen)
            destination.append({'rule':chosen,'discovery':stats})
    selected,ranking=choose_discovery(records)
    return records,controls,selected,ranking


def _verify_selection(selection,out,implementation):
    if (selection.get('plan_sha256')!=implementation['plan_sha256']
            or selection.get('implementation_sha256')!=implementation['implementation_sha256']
            or selection.get('source_identities')!=implementation['source_identities']):
        raise ValueError('Discovery selection no longer matches frozen plan and inputs')
    records=selection['candidate_discovery']
    if [r['rule'] for r in records]!=candidates() or len(records)!=16:
        raise ValueError('Discovery ranking does not contain the exact sixteen augmented candidates')
    chosen,ranking=choose_discovery(records)
    if chosen!=selection['selected_rule'] or ranking!=selection['ranking']:
        raise ValueError('Locked selection is not the discovery-only result')
    expected_control=matched_control(chosen) if chosen else None
    if expected_control!=selection.get('matched_control'):
        raise ValueError('Locked control differs from selected model/threshold/regime')
    for item in selection['discovery_ledgers']:
        if file_hash(Path(out)/item['file'])!=item['sha256']:
            raise ValueError('Frozen discovery ledger changed')


def discovery_phase(out,plan,implementation):
    path=Path(out)/'metrics-threshold-selection.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        _verify_selection(saved,out,implementation)
        return saved
    identities=implementation['source_identities']
    frame,metadata=load_cached_phase(START,SPLIT,identities)
    records,controls,selected,ranking=evaluate_discovery(frame)
    control=matched_control(selected) if selected else None
    files=[]
    for label,rule in [('selected',selected),('control',control)]:
        ledger=simulate_frame(frame,directions(frame,rule),utc_ms(START),utc_ms(SPLIT),rule=rule)[0] if rule else []
        files.append(_save_ledger(Path(out)/f'metrics-threshold-{label}-discovery-trades.csv',ledger))
    selection={'selected_rule':selected,'matched_control':control,'ranking':ranking,
        'candidate_discovery':records,'control_discovery':controls,'selection_candidate_count':16,
        'controls_eligible_for_selection':False,'selection_cutoff_ms':utc_ms(SPLIT),
        'plan_sha256':implementation['plan_sha256'],'implementation_sha256':implementation['implementation_sha256'],
        'source_identities':identities,'discovery_metadata':metadata,'discovery_ledgers':files,
        'locked_at_utc':datetime.now(timezone.utc).isoformat()}
    if implementation_hash()!=implementation['implementation_sha256']:
        raise RuntimeError('Implementation changed during discovery')
    write_json(path,selection)
    print('LOCKED THRESHOLD '+json.dumps({'selected':selected,'selection_candidate_count':16},ensure_ascii=False),flush=True)
    return selection


def stage7_completion_barrier(plan,identities,result_path=None):
    path=Path(result_path or ROOT/'research/metrics-search-results.json')
    if not path.is_file(): raise ValueError('Earlier paired study has not completed its result')
    result=json.loads(path.read_text(encoding='utf-8'))
    provenance=result.get('provenance',{})
    if (provenance.get('source_sha256')!=plan['stage7_source_sha256']
            or provenance.get('plan_sha256')!=plan['stage7_plan_sha256']
            or any(provenance.get(k)!=identities[k] for k in ('feature_store_identity','metrics_store_identity'))
            or not provenance.get('later',{}).get('folds')):
        raise ValueError('Earlier paired completion refers to different sources or inputs')
    # Deliberately return provenance only, never prior candidate rankings.
    return {'result_sha256':file_hash(path),'source_sha256':provenance['source_sha256'],
            'plan_sha256':provenance['plan_sha256']}


def later_phase(out,plan,implementation,selection):
    _verify_selection(selection,out,implementation)
    identities=implementation['source_identities']
    barrier=stage7_completion_barrier(plan,identities)
    frame,metadata=load_cached_phase(SPLIT,END,identities)
    selected,control=selection['selected_rule'],selection['matched_control']
    ranges={'validation':(utc_ms(SPLIT),utc_ms(CONFIRM)),
            'historical_confirmation':(utc_ms(CONFIRM),utc_ms(END)),
            'continuous_post_selection':(utc_ms(SPLIT),utc_ms(END))}
    chosen_ledgers={};control_ledgers={};augmented=[];controls=[]
    for number,(record,destination) in enumerate(
            [(r,augmented) for r in selection['candidate_discovery']]+[(r,controls) for r in selection['control_discovery']],1):
        rule=record['rule'];signal=directions(frame,rule);measured=dict(record)
        for name,(first,stop) in ranges.items():
            ledger,stats=simulate_frame(frame,signal,first,stop,rule=rule);measured[name]=stats
            if selected and rule['id']==selected['id']:chosen_ledgers[name]=ledger
            if control and rule['id']==control['id']:control_ledgers[name]=ledger
        measured['continuous_adverse_entry_2bps']=simulate_frame(frame,signal,utc_ms(SPLIT),utc_ms(END),stress={'adverse_bps':2.},rule=rule)[1]
        destination.append(measured)
        if number%4==0:print(f'THRESHOLD LATER {number}/32 disclosed without reselection',flush=True)
    for label,destination in [('selected',chosen_ledgers),('control',control_ledgers)]:
        path=Path(out)/f'metrics-threshold-{label}-discovery-trades.csv'
        destination['discovery']=pd.read_csv(path).to_dict('records')
    all_ranges={'discovery':(utc_ms(START),utc_ms(SPLIT)),**ranges}
    summaries={};baseline={};paired={};files=[]
    for name,(first,stop) in all_ranges.items():
        chosen=chosen_ledgers.get(name,[]);base=control_ledgers.get(name,[])
        summaries[name]=detail(chosen,first,stop);baseline[name]=detail(base,first,stop)
        paired[name]=paired_week_interval(chosen,base,first,stop)
        for label,ledger in [('selected',chosen),('control',base)]:
            path=Path(out)/f'metrics-threshold-{label}-{name}-trades.csv'
            if name=='discovery':
                files.append({'file':path.name,'sha256':file_hash(path),'rows':len(ledger)})
            else:files.append(_save_ledger(path,ledger))
    stressed={};stresses=[]
    if selected:
        signal=directions(frame,selected)
        for name,(first,stop) in ranges.items():
            ledger=simulate_frame(frame,signal,first,stop,stress={'adverse_bps':2.},rule=selected)[0]
            stressed[name]=stress_detail(ledger,first,stop)
            files.append(_save_ledger(Path(out)/f'metrics-threshold-adverse2bps-{name}-trades.csv',ledger))
        stresses=[{'name':'adverse_index_entry_2bps','configuration':{'adverse_bps':2.},
                   'test':stressed['continuous_post_selection'],'periods':stressed}]
    folds=selection['discovery_metadata']['folds']+metadata['folds']
    nominal,execution=historical_checks(selected,summaries,stressed,folds)
    return {'selected':summaries,'baseline':baseline,'all_candidates':augmented,'matched_controls':controls,
        'paired_comparison':paired,'stresses':stresses,'checks':{**nominal,**execution},
        'nominal_history_gates_passed':all(nominal.values()),
        'execution_robustness_gates_passed':all(execution.values()),
        'historical_research_gates_passed':all(nominal.values()) and all(execution.values()),
        'incremental_historical_evidence_passed':incremental_evidence(selected,paired),
        'later_metadata':metadata,'earlier_study_completion':barrier,'ledger_files':files}


def run(phase='all',output_dir=None):
    if phase not in ('all','discovery','later'):raise ValueError('Unknown confidence study phase')
    began=time.perf_counter();out=Path(output_dir or DEFAULT_OUT)
    plan=read_plan(out);identities=read_identities();implementation=freeze_implementation(out,plan,identities)
    selection_path=out/'metrics-threshold-selection.json'
    if phase=='later' and not selection_path.is_file():raise ValueError('Discovery selection must already be locked')
    selection=discovery_phase(out,plan,implementation)
    if phase=='discovery':return selection
    measured=later_phase(out,plan,implementation,selection)
    if implementation_hash()!=implementation['implementation_sha256'] or read_identities()!=identities:
        raise RuntimeError('Frozen execution source or data identities changed')
    read_plan(out);_verify_selection(selection,out,implementation)
    result={'schema_version':1,'engine':plan['engine'],'plan':plan,'implementation':implementation,
        'selection':selection,**measured,'live_approved':False,
        'provenance':{'plan_sha256':implementation['plan_sha256'],
            'implementation_sha256':implementation['implementation_sha256'],
            'selection_sha256':file_hash(selection_path),'source_identities':identities,
            'runtime_seconds':round(time.perf_counter()-began,2),'generated_at_utc':datetime.now(timezone.utc).isoformat()},
        'limitations':['This is another related exploratory search over reused history, not a fresh holdout.',
            'The catalog reuses eight .56/.60 augmented configurations and adds eight .64/.68 configurations; all sixteen enter discovery correction.',
            'Only the discovery winner is selected; all sixteen augmented and sixteen exact controls are disclosed without later replacement.',
            'Bonferroni correction for this catalog does not remove serial dependence or accumulated bias from earlier research rounds.',
            'Record+10 minute availability remains an explicit assumption; original API timing and archive revisions are not proven.',
            'Raw scores are not calibrated win probabilities. A higher threshold can reduce trade count without improving true future performance.',
            'No models were fitted and no original feature arrays were loaded. Only verified completed paired numeric predictions were reused.',
            'Index-minute prices proxy contract quotes; 2bp stress and known-calendar exclusions do not capture every execution risk.',
            'No live model, alert, server, GitHub remote or account was changed.']}
    write_json(out/'metrics-threshold-results.json',result)
    print(json.dumps({'selected':selection['selected_rule'],'checks':result['checks'],
                      'runtime_seconds':result['provenance']['runtime_seconds']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','discovery','later'),default='all')
    run(parser.parse_args().phase)
