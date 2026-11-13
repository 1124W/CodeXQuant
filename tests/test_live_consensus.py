"""Fixed pair decision parity, dense training population and bundle integrity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from live import consensus
from live.features import FEATURES,SYMBOLS
from research.consensus_search_rules import directions
from research.consensus_three_year_extension import selected_specs

MINUTE=60_000
DAY=86_400_000
CUTOFF=21_000*DAY


def bundle(folder):
    import joblib
    import sklearn
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(721);matrix=rng.normal(size=(80,60));labels=np.arange(80)%2
    models={}
    for symbol in SYMBOLS:
        members={}
        for name in consensus.MODEL_IDS:
            model=consensus.make_model(name)
            with threadpool_limits(limits=2):model.fit(matrix,labels)
            target=folder/f'{symbol}-{name}.joblib';joblib.dump(model,target)
            members[name]={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'training_converged':True,'sample_minutes':consensus.specs()[name]['sample_minutes'],
                'latest_settlement_ms':CUTOFF-MINUTE,'training_count':30_000}
        models[symbol]={'members':members,'training_sampling_minutes':1,'training_samples_rows':259169,
            'training_samples_file':f'{symbol}-training.csv.gz','training_samples_sha256':'a'*64}
    value={'schema_version':2,'rule_id':consensus.RULE_ID,'features':FEATURES,'model_specs':consensus.specs(),
        'training_days':180,'training_sampling_minutes':1,'member_sampling_minutes':{'logistic_1m':1,'gbm_deeper_5m':5},
        'threshold':.60,'regime':'unrestricted','order_execution_enabled':False,'models':models,
        'sklearn_version':sklearn.__version__,'training_cutoff_ms':CUTOFF,'model_version':'b'*20}
    (folder/'manifest.json').write_text(json.dumps(value),encoding='utf-8')
    return value


class ConsensusDecisionTests(unittest.TestCase):
    def test_specs_match_frozen_historical_pair(self):
        self.assertEqual(consensus.specs(),selected_specs())

    def test_pair_gate_matches_research_for_all_directions_and_threshold_edges(self):
        values=[0.,.399999,.4,.400001,.499999,.5,.500001,.599999,.6,.600001,1.]
        rows=[{'symbol':'BTCUSDT','signal_ms':CUTOFF+i*MINUTE,
            'p_logistic_1m':a,'p_gbm_deeper_5m':b} for i,(a,b) in enumerate((a,b) for a in values for b in values)]
        expected=directions(pd.DataFrame(rows),consensus.RULE_ID)
        actual=[consensus.decision({name:row['p_'+name] for name in consensus.MODEL_IDS},row['signal_ms'])['direction'] for row in rows]
        np.testing.assert_array_equal(actual,expected)

    def test_clock_and_exhaustion_free_decision(self):
        result=consensus.decision({'logistic_1m':.6,'gbm_deeper_5m':.8},CUTOFF)
        self.assertEqual(result['side'],'LONG');self.assertEqual(result['confidence_score'],.6)
        self.assertEqual(result['entry_due_ms'],CUTOFF+MINUTE)
        self.assertEqual(result['expiry_due_ms'],CUTOFF+31*MINUTE)
        self.assertFalse(result['order_execution_enabled'])
        self.assertEqual(result['model_scores'],{'logistic_1m':.6,'gbm_deeper_5m':.8})

    def test_explicit_no_signal_reasons_and_invalid_inputs(self):
        for a,b,reason in [(.7,.3,'model_directions_disagree'),(.6,.59,'consensus_score_below_0.60'),(.5,.8,'neutral_model_direction')]:
            result=consensus.decision({'logistic_1m':a,'gbm_deeper_5m':b},CUTOFF)
            self.assertEqual(result['reason'],reason);self.assertEqual(result['direction'],0)
        for bad in (True,float('nan'),float('inf'),1.1,-.1):
            with self.assertRaises(ValueError):consensus.decision({'logistic_1m':bad,'gbm_deeper_5m':.7},CUTOFF)
        for stamp in (True,CUTOFF+1,0,float('nan')):
            with self.assertRaises(ValueError):consensus.decision({'logistic_1m':.7,'gbm_deeper_5m':.7},stamp)


class DensePopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times=CUTOFF-180*DAY+np.arange(180*1440+2,dtype=np.int64)*MINUTE
        cls.frame=pd.DataFrame(np.zeros((len(times),60),dtype=np.float32),columns=FEATURES)
        cls.frame['signal_ms']=times;cls.frame['settlement_ms']=times+31*MINUTE
        cls.frame['y']=np.arange(len(times))%2;cls.frame['is_flat']=False

    def test_dense_seed_preserves_every_settled_minute_and_purges_future(self):
        first,meta=consensus.prepare_training_rows(self.frame,CUTOFF)
        self.assertEqual(len(first),180*1440-31);self.assertEqual(meta['coverage'],1.)
        self.assertTrue((np.diff(first.signal_ms)==MINUTE).all())
        changed=self.frame.copy();future=changed.settlement_ms>=CUTOFF
        changed.loc[future,FEATURES]=np.nan;changed.loc[future,'y']=999
        second,other=consensus.prepare_training_rows(changed,CUTOFF)
        pd.testing.assert_frame_equal(first,second);self.assertEqual(meta,other)

    def test_old_halfhour_seed_and_missing_day_rejected(self):
        with self.assertRaisesRegex(ValueError,'dense 180-day'):
            consensus.prepare_training_rows(self.frame.iloc[::30],CUTOFF)
        missing=self.frame.loc[self.frame.signal_ms//DAY!=(CUTOFF//DAY-20)]
        with self.assertRaisesRegex(ValueError,'dense 180-day'):
            consensus.prepare_training_rows(missing,CUTOFF)

    def test_small_documented_gap_allowed_without_filling(self):
        changed=self.frame.drop(index=range(10_000,10_061))
        data,meta=consensus.prepare_training_rows(changed,CUTOFF)
        self.assertEqual(len(data),180*1440-31-61)
        self.assertEqual(meta['maximum_missing_run_minutes'],61)

    def test_duplicate_or_off_grid_clock_rejected(self):
        duplicated=pd.concat([self.frame,self.frame.iloc[[100]]],ignore_index=True)
        with self.assertRaises(ValueError):consensus.prepare_training_rows(duplicated,CUTOFF)
        changed=self.frame.copy();changed.loc[100,'settlement_ms']+=MINUTE
        with self.assertRaises(ValueError):consensus.prepare_training_rows(changed,CUTOFF)


class ConsensusBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.manifest=bundle(self.folder)

    def save(self):
        (self.folder/'manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')

    def test_predict_compatible_schema_and_model_age_guards(self):
        model=consensus.ConsensusPredictor(self.folder)
        rows={symbol:{name:0. for name in FEATURES} for symbol in SYMBOLS}
        result=model.predict(rows,CUTOFF)
        self.assertEqual({item['symbol'] for item in result},set(SYMBOLS))
        self.assertTrue(all(set(item['model_scores'])==set(consensus.MODEL_IDS) for item in result))
        with self.assertRaisesRegex(ValueError,'before'):model.predict(rows,CUTOFF-MINUTE)
        with self.assertRaisesRegex(ValueError,'maximum age'):model.predict(rows,CUTOFF+36*DAY)
        rows['BTCUSDT'][FEATURES[0]]=float('nan')
        with self.assertRaises(ValueError):model.predict(rows,CUTOFF)

    def test_corruption_rejected_before_joblib_loading(self):
        entry=self.manifest['models']['BTCUSDT']['members']['logistic_1m']
        path=self.folder/entry['file'];path.write_bytes(path.read_bytes()+b'corrupt')
        with patch('joblib.load') as loader,self.assertRaisesRegex(ValueError,'checksum'):
            consensus.ConsensusPredictor(self.folder)
        loader.assert_not_called()

    def test_filename_contract_parameter_and_sparse_seed_rejected(self):
        original=deepcopy(self.manifest)
        mutations=[lambda m:m['models']['BTCUSDT']['members']['logistic_1m'].update(file='../escape.joblib'),
            lambda m:m.update(features=list(reversed(FEATURES))),
            lambda m:m['model_specs']['logistic_1m']['parameters'].update(C=.1),
            lambda m:m['models']['BTCUSDT'].update(training_samples_rows=8640)]
        for mutate in mutations:
            self.manifest=deepcopy(original);mutate(self.manifest);self.save()
            with self.assertRaises(ValueError):consensus.ConsensusPredictor(self.folder)

    def test_unconverged_or_noncausal_member_rejected(self):
        original=deepcopy(self.manifest)
        for change in ({'training_converged':False},{'latest_settlement_ms':CUTOFF}):
            self.manifest=deepcopy(original)
            self.manifest['models']['BTCUSDT']['members']['logistic_1m'].update(change);self.save()
            with self.assertRaises(ValueError):consensus.ConsensusPredictor(self.folder)


if __name__=='__main__':unittest.main()
