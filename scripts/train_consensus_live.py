"""Export the fixed live consensus pair from verified dense historical features."""
from __future__ import annotations

import argparse
from datetime import date,datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np

from live.consensus import ConsensusPredictor,MODEL_IDS,retrain_from_rows,specs
from live.features import FEATURES,MINUTE_MS,SYMBOLS
from research.consensus_search_research import verified_store_identity
from research.consensus_three_year_extension import fit_pair,selected_specs
from research.event_engine import DAY_MS,utc_ms
from research.high_win_research import file_hash,write_json
from research.model_search_data import END,load_window


def run(output_dir=ROOT/'research/.cache/consensus-live-model',end=END):
    if end!=END:raise ValueError('This export binds the verified 2026-11-12 cache boundary')
    if specs()!=selected_specs():raise ValueError('Serving and historical model specifications differ')
    identity=verified_store_identity();cutoff=utc_ms(end)
    rows={symbol:load_window(symbol,cutoff-180*DAY_MS,cutoff) for symbol in SYMBOLS}
    manifest=retrain_from_rows(output_dir,rows,cutoff,progress=lambda s:print(s,flush=True),
        data_sources={'feature_store_identity':identity,'source':'verified_model_search_dense_features',
            'feature_manifest_sha256':file_hash(ROOT/'research/.cache/model-search/features/manifest.json'),
            'training_script_sha256':file_hash(Path(__file__))})
    predictor=ConsensusPredictor(output_dir);checks={}
    from threadpoolctl import threadpool_limits
    for symbol in SYMBOLS:
        # These are identity probes, not backtest results or actual future prices.
        probes=rows[symbol].iloc[-32:].copy()
        probes['signal_ms']=cutoff+np.arange(len(probes))*MINUTE_MS
        probes['settlement_ms']=probes.signal_ms+31*MINUTE_MS
        expected,meta=fit_pair(rows[symbol],probes,cutoff)
        members={}
        for name in MODEL_IDS:
            with threadpool_limits(limits=2):actual=predictor.models[symbol][name].predict_proba(probes[FEATURES].to_numpy())[:,1]
            delta=float(np.max(np.abs(actual-expected['p_'+name].to_numpy())))
            saved=manifest['models'][symbol]['members'][name]
            if delta>1e-12 or any(saved[key]!=meta['models'][name][key] for key in
                    ('training_count','sample_minutes','latest_settlement_ms','earliest_signal_ms')):
                raise ValueError('Serving model differs from the fixed historical training procedure')
            members[name]={'prediction_probes':len(probes),'maximum_probability_delta':delta,
                'training_count':saved['training_count'],'training_sample_identity_verified':True}
        checks[symbol]=members
    result={'model_version':manifest['model_version'],'training_cutoff_utc':str(end),
        'manifest_sha256':file_hash(Path(output_dir)/'manifest.json'),'fixed_research_pair_parity':checks,
        'probe_policy':'historical feature vectors assigned hypothetical times solely for model identity checks; not predictions',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),'local_only':True,'messages_sent':False}
    write_json(Path(output_dir)/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'research/.cache/consensus-live-model')
    args=parser.parse_args();run(args.output_dir)
