"""Consensus routing, independent paper cohort and dense seed compatibility."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd

from live.app import Worker,CONSENSUS_RULE,LEGACY_RULE,strategy_mode,model_pointer
from live.features import FEATURES,SYMBOLS
from live.history import FeatureHistory,ROW_COLUMNS
from live.notify import WeComNotifier,format_signal,format_research_report,format_strategy_update
from live.stakes import PaperSettings

ROOT=Path(__file__).resolve().parents[1]
ORIGIN=1794441600000
MINUTE=60000
DAY=86400000


def manifest(rule=CONSENSUS_RULE):
    return {'schema_version':2 if rule==CONSENSUS_RULE else 1,'model_version':'a'*20,
            'training_cutoff_ms':ORIGIN,'training_cutoff_utc':'2026-11-12',
            'rule_id':rule,'features':FEATURES.copy(),'models':{}}


def signal(stamp=ORIGIN):
    return {'symbol':'BTCUSDT','signal_ms':stamp,'entry_due_ms':stamp+MINUTE,
            'expiry_due_ms':stamp+31*MINUTE,'direction':1,'side':'LONG',
            'rule_id':CONSENSUS_RULE,'model_id':'a'*20,'confidence_score':.61,
            'model_scores':{'logistic_1m':.61,'gbm_deeper_5m':.64}}


class ConsensusServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def worker(self,mode='consensus'):
        worker=Worker.__new__(Worker);worker.db=self.folder/'worker.sqlite3'
        worker.state_dir=self.folder;worker.model_dir=self.folder/'model'
        worker.paper_settings=PaperSettings.for_stake(30);worker.strategy_mode=mode
        worker.predictor=SimpleNamespace(manifest=manifest(CONSENSUS_RULE if mode=='consensus' else LEGACY_RULE))
        worker.lock=threading.RLock();worker.state={'symbols':{}}
        worker._ensure_tables();return worker

    def test_routing_defaults_to_legacy_and_consensus_has_separate_persistent_pointer(self):
        self.assertEqual(strategy_mode({}),'legacy')
        self.assertEqual(strategy_mode({'CODEXQUANT_STRATEGY':'consensus'}),'consensus')
        for value in ('','Consensus','other',True):
            with self.subTest(value=value),self.assertRaises(ValueError):strategy_mode({'CODEXQUANT_STRATEGY':value})
        self.assertNotEqual(model_pointer(self.folder,'legacy'),model_pointer(self.folder,'consensus'))

    def test_consensus_worker_uses_dense_history_and_never_loads_legacy_predictor(self):
        predictor=SimpleNamespace(manifest=manifest())
        with patch.dict('os.environ',{'CODEXQUANT_STRATEGY':'consensus'}), \
             patch('live.consensus.ConsensusPredictor',return_value=predictor) as construct, \
             patch('live.app.Predictor',side_effect=AssertionError('legacy fallback')), \
             patch('live.history.FeatureHistory') as history,patch('live.app.WeComNotifier'):
            worker=Worker(self.folder/'bundle',self.folder/'state',market=Mock())
        construct.assert_called_once_with(self.folder/'bundle')
        history.assert_called_once_with(self.folder/'state'/'history-consensus.sqlite3')
        history.return_value.import_seed.assert_called_once_with(self.folder/'bundle',predictor.manifest)
        self.assertEqual(worker.strategy_description()['rule_id'],CONSENSUS_RULE)

    def test_legacy_admissions_are_backfilled_without_changing_original_rows(self):
        path=self.folder/'worker.sqlite3'
        original=('ETHUSDT',ORIGIN,ORIGIN+MINUTE,ORIGIN+31*MINUTE,-1,'loss',-5.)
        with closing(sqlite3.connect(path)) as db:
            db.execute('CREATE TABLE admissions(symbol TEXT,signal_ms INTEGER,entry_ms INTEGER,settlement_ms INTEGER,direction INTEGER,outcome TEXT,net REAL,PRIMARY KEY(symbol,signal_ms))')
            db.execute('INSERT INTO admissions VALUES(?,?,?,?,?,?,?)',original);db.commit()
        worker=self.worker()
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute('SELECT * FROM admissions').fetchone(),original)
            self.assertEqual(db.execute('SELECT rule_id,model_id FROM admission_strategy').fetchone(),(LEGACY_RULE,'legacy_untracked'))
        paper=worker.paper_performance()
        self.assertEqual(paper['net_usdt'],-5)
        self.assertEqual(paper['current_strategy']['total'],0)
        self.assertFalse(paper['all_history_is_current_strategy'])

    def test_old_pending_settles_as_old_strategy_and_new_cohort_is_separate(self):
        old=self.worker('legacy')
        row={**signal(),'rule_id':LEGACY_RULE};old.paper_settings=PaperSettings.for_stake(5)
        self.assertTrue(old.admit(row)[0])
        current=self.worker()
        current.settle_admission('BTCUSDT',ORIGIN,100,101)
        new=signal(ORIGIN+60*MINUTE);self.assertTrue(current.admit(new)[0])
        current.settle_admission('BTCUSDT',new['signal_ms'],100,99)
        paper=current.paper_performance()
        self.assertEqual(paper['total'],2)
        self.assertEqual(paper['net_usdt'],-25.75)
        self.assertEqual(paper['current_strategy']['total'],1)
        self.assertEqual(paper['current_strategy']['net_usdt'],-30)
        self.assertEqual(paper['current_strategy']['rule_id'],CONSENSUS_RULE)
        self.assertTrue(paper['mixed_strategies'])
        saved=current.recent_settlements()
        self.assertEqual(saved[0]['rule_id'],CONSENSUS_RULE)
        self.assertEqual(saved[1]['rule_id'],LEGACY_RULE)
        self.assertEqual(saved[1]['stake_usdt'],5)
        self.assertEqual(self.worker().paper_performance()['current_strategy']['net_usdt'],-30)

    def test_mismatched_strategy_cannot_create_partial_admission(self):
        worker=self.worker()
        with self.assertRaisesRegex(ValueError,'strategy does not match'):
            worker.admit({**signal(),'rule_id':LEGACY_RULE})
        self.assertEqual(worker.paper_performance()['total'],0)

    def test_monthly_refresh_uses_all_minute_history_and_consensus_trainer(self):
        worker=self.worker();rows={'BTCUSDT':pd.DataFrame(),'ETHUSDT':pd.DataFrame()}
        worker.history=SimpleNamespace(training_rows=Mock(return_value=rows));worker.notifier=Mock()
        old_pointer=model_pointer(self.folder,'legacy');old_pointer.write_text('preserve-old-model',encoding='utf-8')
        replacement=SimpleNamespace(manifest=manifest())
        now=ORIGIN+31*DAY+123456;day=now//DAY*DAY
        with patch('live.consensus.retrain_from_rows') as fit, \
             patch('live.consensus.ConsensusPredictor',return_value=replacement), \
             patch('live.predictor.retrain_from_rows',side_effect=AssertionError('legacy fallback')):
            worker._refresh_model(now)
        worker.history.training_rows.assert_called_once_with(day,sample_interval_minutes=1)
        self.assertEqual(fit.call_args.args[:3],(self.folder/'models-consensus'/f'refresh-{day}',rows,day))
        self.assertIs(worker.predictor,replacement)
        self.assertTrue(model_pointer(self.folder,'consensus').exists())
        self.assertEqual(old_pointer.read_text(encoding='utf-8'),'preserve-old-model')

    def test_failed_consensus_refresh_never_falls_back_or_overwrites_pointer(self):
        worker=self.worker();worker.history=SimpleNamespace(training_rows=Mock(return_value={}))
        worker.notifier=Mock();previous=worker.predictor
        with patch('live.consensus.retrain_from_rows',side_effect=ValueError('insufficient dense history')), \
             patch('live.predictor.retrain_from_rows',side_effect=AssertionError('legacy fallback')):
            worker._refresh_model(ORIGIN+31*DAY)
        self.assertIs(worker.predictor,previous)
        self.assertFalse(model_pointer(self.folder,'consensus').exists())
        self.assertIn('ValueError',worker.state['model_refresh_error'])
        worker.notifier.send_status.assert_not_called()

    def test_three_year_historical_comparator_matches_actual_active_rule(self):
        data=json.loads((ROOT/'research/consensus-public.json').read_text(encoding='utf-8'))
        worker=self.worker()
        historical=worker.historical_research()
        self.assertTrue(historical['is_live_strategy'])
        self.assertTrue(historical['matches_current_strategy'])
        self.assertEqual(historical['signals'],1892)
        self.assertEqual(historical['net_usdt'],2236.5)
        self.assertEqual(historical['scope'],'three_year')
        legacy=self.worker('legacy').historical_research()
        self.assertFalse(legacy['is_live_strategy'])
        self.assertEqual(worker.paper_performance()['net_usdt'],0)
        # A malformed preferred report cannot silently impersonate the old comparator.
        public=self.folder/'research';public.mkdir();data['strategy']['rule_id']='other'
        (public/'consensus-public.json').write_text(json.dumps(data),encoding='utf-8')
        worker.historical_research_loaded_at=float('-inf')
        with patch('live.app.ROOT',self.folder):self.assertIsNone(worker.historical_research())

    def test_signal_reports_both_directional_scores_and_rejects_disagreement(self):
        worker=self.worker();row=signal();worker.admit(row)
        text=format_signal(row)
        for part in ('双模型同向确认','Logistic 61.00%','GBM 64.00%','30.00 USDT','150.00 USDT','不是胜率'):
            self.assertIn(part,text)
        with self.assertRaisesRegex(ValueError,'Both models'):
            format_signal({**row,'model_scores':{'logistic_1m':.61,'gbm_deeper_5m':.39}})
        with self.assertRaisesRegex(ValueError,'weaker'):
            format_signal({**row,'confidence_score':.99})

    def test_hourly_report_distinguishes_legacy_account_current_cohort_and_history(self):
        worker=self.worker()
        report={'scan_stats':{'evaluated_slots':60,'candidates':0,'admitted':0,'late_slots':0,
                             'reasons':{'model_directions_disagree':60}},
                'paper_performance':worker.paper_performance(),'historical_research':worker.historical_research()}
        text=format_research_report(report,ORIGIN)
        for part in ('双模型方向不同','当前双模型策略的独立前向记录','暂无已决样本',
                     '当前策略三年历史对照','1892 笔','-373.50U','-447.00U','不能把旧策略业绩算给新策略'):
            self.assertIn(part,text)
        self.assertNotIn('非当前线上模型',text)
        self.assertLessEqual(len(text.encode('utf-8')),4096)

    def test_explicit_update_is_under_limit_and_deduplicates_across_restart(self):
        report=json.loads((ROOT/'research/consensus-public.json').read_text(encoding='utf-8'))
        text=format_strategy_update(report,'http://example.com/demo/consensus.html')
        self.assertLessEqual(len(text.encode('utf-8')),4096)
        for part in ('1,892笔','56.18%','-373.50U','-447.00U','不自动下单','不是新策略前向业绩'):
            self.assertIn(part,text)
        path=self.folder/'notifications.sqlite3'
        first=WeComNotifier('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test',path,opener=Mock(),clock=lambda:ORIGIN/1000)
        with patch.object(first,'_post',return_value={'sent':True,'errcode':0}) as post:
            self.assertTrue(first.send_strategy_update(report,'consensus-release-test')['sent'])
            post.assert_called_once()
        restarted=WeComNotifier('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test',path,opener=Mock(),clock=lambda:ORIGIN/1000)
        with patch.object(restarted,'_post') as post:
            self.assertTrue(restarted.send_strategy_update(report,'consensus-release-test')['duplicate'])
            post.assert_not_called()
        self.assertEqual(restarted.status()['sent_by_kind']['strategy_update'],1)
        with self.assertRaises(ValueError):format_strategy_update(report,'https://example.com/?key=private')


class DenseHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.history=FeatureHistory(self.folder/'history-consensus.sqlite3')

    def seed(self,sampling=1,count=1030,alter=None):
        bundle=self.folder/'bundle';bundle.mkdir(exist_ok=True)
        info={'schema_version':2 if sampling==1 else 1,'features':FEATURES.copy(),
              'training_sampling_minutes':sampling,'training_cutoff_ms':ORIGIN+31*DAY,'models':{}}
        for symbol in SYMBOLS:
            data=[]
            for i in range(count):
                stamp=ORIGIN+i*sampling*MINUTE
                row={name:float(i)+j/100 for j,name in enumerate(FEATURES)}
                row.update(signal_ms=stamp,settlement_ms=stamp+31*MINUTE,y=i%2,is_flat=False)
                if alter:alter(symbol,i,row)
                data.append(row)
            path=bundle/f'{symbol}-training.csv.gz'
            pd.DataFrame(data,columns=ROW_COLUMNS).to_csv(path,index=False,compression='gzip')
            info['models'][symbol]={'training_samples_file':path.name,'training_samples_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return bundle,info

    def test_dense_seed_streaming_preserves_non_halfhour_rows_and_exact_features(self):
        bundle,info=self.seed()
        imported=self.history.import_seed(bundle,info)
        self.assertEqual(imported['BTCUSDT']['inserted'],1030)
        train=self.history.training_rows(info['training_cutoff_ms'],sample_interval_minutes=1)
        sparse=self.history.training_rows(info['training_cutoff_ms'],sample_interval_minutes=30)
        self.assertEqual(len(train['BTCUSDT']),1030)
        self.assertEqual(len(sparse['BTCUSDT']),35)
        self.assertEqual(int(train['ETHUSDT'].signal_ms.iloc[1]),ORIGIN+MINUTE)
        self.assertEqual(train['ETHUSDT'][FEATURES[59]].iloc[1029],1029.59)
        self.assertEqual(str(train['BTCUSDT'].signal_ms.dtype),'int64')
        self.assertEqual(str(train['BTCUSDT'].y.dtype),'int8')
        self.assertEqual(str(train['BTCUSDT'].is_flat.dtype),'bool')
        self.assertTrue(self.history.import_seed(bundle,info)['BTCUSDT']['already_imported'])

    def test_chunk_boundary_duplicate_rejects_every_asset_before_import(self):
        def alter(symbol,index,row):
            if symbol=='ETHUSDT' and index==1024:
                row['signal_ms']-=MINUTE;row['settlement_ms']-=MINUTE
        bundle,info=self.seed(alter=alter)
        with self.assertRaisesRegex(ValueError,'timestamps must be unique'):
            self.history.import_seed(bundle,info)
        self.assertEqual(self.history.status()['total_rows'],0)

    def test_seed_sampling_contract_is_explicit_and_legacy_halfhour_still_supported(self):
        bundle,info=self.seed(count=4)
        for sampling in (True,5,'1',None):
            with self.subTest(sampling=sampling),self.assertRaises(ValueError):
                self.history.import_seed(bundle,{**info,'training_sampling_minutes':sampling})
        with self.assertRaisesRegex(ValueError,'dense one-minute'):
            self.history.import_seed(bundle,{**info,'training_sampling_minutes':30})
        legacy=self.folder/'legacy';legacy.mkdir()
        sparse=deepcopy(info);sparse['schema_version']=1;sparse.pop('training_sampling_minutes')
        with self.assertRaisesRegex(ValueError,'timestamps must be unique'):
            self.history.import_seed(bundle,sparse)
        bundle,sparse=self.seed(sampling=30,count=4);sparse.pop('training_sampling_minutes')
        self.assertEqual(self.history.import_seed(bundle,sparse)['BTCUSDT']['inserted'],4)

    def test_empty_stream_and_pending_or_future_labels_do_not_enter_training(self):
        empty=self.history.training_rows(ORIGIN)
        self.assertEqual(list(empty),list(SYMBOLS));self.assertTrue(empty['BTCUSDT'].empty)
        features={name:1. for name in FEATURES}
        self.history.record_features('BTCUSDT',ORIGIN,features)
        self.assertTrue(self.history.training_rows(ORIGIN+DAY)['BTCUSDT'].empty)
        self.history.update_label('BTCUSDT',ORIGIN,100,101)
        self.assertTrue(self.history.training_rows(ORIGIN+31*MINUTE)['BTCUSDT'].empty)
        self.assertEqual(len(self.history.training_rows(ORIGIN+32*MINUTE)['BTCUSDT']),1)


if __name__=='__main__':unittest.main()
