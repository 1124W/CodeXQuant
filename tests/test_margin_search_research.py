"""Margin checkpoints fail closed and discovery locks before later evaluation."""
from contextlib import ExitStack
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import margin_search_research as study
from research.event_engine import DAY_MS, MINUTE_MS, utc_ms


def observations(symbol, first):
    stamps=np.array([first,first+MINUTE_MS],dtype=np.int64)
    return pd.DataFrame({'symbol':symbol,'signal_ms':stamps,'entry_ms':stamps+MINUTE_MS,
        'settlement_ms':stamps+31*MINUTE_MS,'entry':[100.,100.],'exit':[101.,99.],
        'ret_60':[-.01,-.01],'rv_60':[.001,.001],'rv_ratio':[.8,.8]})


def fake_fit(train,test,cutoff,**unused):
    probability=pd.DataFrame(index=test.index)
    models={}
    for name in study.MODEL_IDS:
        probability['p_'+name+'_up']=.6
        probability['p_'+name+'_down']=.3
        probability['p_'+name+'_neutral']=.1
        models[name]={'latest_settlement_ms':cutoff-MINUTE_MS,'training_converged':True}
    meta={'fit_cutoff_ms':cutoff,'prediction_count':len(test),'train_days':180,'refit_days':30,
          'test_start_ms':int(test.signal_ms.min()),
          'test_end_exclusive_ms':int(test.signal_ms.max())+MINUTE_MS,
          'models':models,'specs':study.specs()}
    return probability,meta


class MarginRunnerTests(unittest.TestCase):
    def test_atomic_fold_cache_reuses_scores_and_rejects_corruption(self):
        first=study.START;last=first+timedelta(days=1)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch.object(study,'load_window',side_effect=lambda s,a,b:observations(s,a)),patch.object(study,'fit_predict',side_effect=fake_fit) as fitter:
                frame,meta=study.prediction_phase(first,last,'fixture-identity',folder)
                self.assertEqual(fitter.call_count,2)
                self.assertEqual(len(frame),4)
                again,again_meta=study.prediction_phase(first,last,'fixture-identity',folder)
                self.assertEqual(fitter.call_count,2)
                pd.testing.assert_frame_equal(frame,again)
                self.assertEqual(meta,again_meta)
                binary=next(folder.glob('*.npz'))
                with binary.open('ab') as stream:stream.write(b'corrupt')
                with self.assertRaisesRegex(ValueError,'corrupt'):
                    study.prediction_phase(first,last,'fixture-identity',folder)
                self.assertEqual(fitter.call_count,2)

    def test_incomplete_and_stale_checkpoint_never_silently_retrains(self):
        first=study.START;last=first+timedelta(days=1)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch.object(study,'load_window',side_effect=lambda s,a,b:observations(s,a)),patch.object(study,'fit_predict',side_effect=fake_fit) as fitter:
                study.prediction_phase(first,last,'fixture-identity',folder)
                with self.assertRaisesRegex(ValueError,'Stale'):
                    study.prediction_phase(first,last,'other-identity',folder)
                next(folder.glob('*.npz')).unlink()
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    study.prediction_phase(first,last,'fixture-identity',folder)
                self.assertEqual(fitter.call_count,2)

    def test_prediction_validation_rejects_unsettled_labels_and_invalid_scores(self):
        first=utc_ms(study.START);frame=observations('BTCUSDT',first)
        scores,model=fake_fit(frame,frame,first)
        for name in scores:frame[name]=scores[name]
        frame['model_fit_cutoff_ms']=first;frame['train_latest_settlement_ms']=first-MINUTE_MS
        metadata={'symbol':'BTCUSDT','first_signal_bound_ms':first,'end_exclusive_ms':first+DAY_MS,'model_metadata':model}
        study._validate_prediction(frame,metadata,'BTCUSDT',first,first,first+DAY_MS)
        frame.loc[0,'p_margin_shallow_up']=.9
        with self.assertRaisesRegex(ValueError,'probabilities'):
            study._validate_prediction(frame,metadata,'BTCUSDT',first,first,first+DAY_MS)
        model['models']['margin_shallow']['latest_settlement_ms']=first
        with self.assertRaisesRegex(ValueError,'settled'):
            study._validate_prediction(frame,metadata,'BTCUSDT',first,first,first+DAY_MS)

    def test_discovery_plan_and_selection_written_without_later_predictions(self):
        first=utc_ms(study.START);frame=pd.concat([observations(s,first) for s in study.SYMBOLS],ignore_index=True)
        scores,_=fake_fit(frame,frame,first)
        for name in scores:frame[name]=scores[name]
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            folder=Path(directory);events=[]
            plan=study.make_plan()
            stack.enter_context(patch.object(study,'make_plan',return_value=plan))
            stack.enter_context(patch.object(study,'verified_store_identity',return_value='fixture'))
            def predict(start,end,identity):
                self.assertEqual((start,end),(study.START,study.SPLIT))
                self.assertTrue((folder/'margin-search-plan.json').exists())
                self.assertFalse((folder/'margin-search-selection.json').exists())
                events.append('discovery')
                return frame,{}
            predictor=stack.enter_context(patch.object(study,'prediction_phase',side_effect=predict))
            stack.enter_context(patch.object(study,'load_baseline',side_effect=AssertionError('No baseline/later read in discovery phase')))
            result=study.run(phase='discovery',output_dir=folder)
            self.assertEqual(predictor.call_count,1)
            self.assertEqual(events,['discovery'])
            self.assertIsNone(result['selected_rule'])
            self.assertEqual(json.loads((folder/'margin-search-selection.json').read_text(encoding='utf-8')),result)

    def test_selection_file_exists_before_first_later_prediction(self):
        first=utc_ms(study.START);frame=pd.concat([observations(s,first) for s in study.SYMBOLS],ignore_index=True)
        scores,_=fake_fit(frame,frame,first)
        for name in scores:frame[name]=scores[name]
        baseline=frame.copy();baseline['p_logistic_halfhour']=.6
        class LaterReached(Exception):pass
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            folder=Path(directory);events=[]
            stack.enter_context(patch.object(study,'verified_store_identity',return_value='fixture'))
            def predict(start,end,identity):
                events.append((start,end))
                if start==study.SPLIT:
                    saved=json.loads((folder/'margin-search-selection.json').read_text(encoding='utf-8'))
                    self.assertIsNone(saved['selected_rule'])
                    self.assertEqual(saved['selection_cutoff_ms'],utc_ms(study.SPLIT))
                    raise LaterReached()
                return frame,{}
            stack.enter_context(patch.object(study,'prediction_phase',side_effect=predict))
            stack.enter_context(patch.object(study,'load_baseline',return_value=(baseline,{})))
            with self.assertRaises(LaterReached):study.run(output_dir=folder)
            self.assertEqual(events,[(study.START,study.SPLIT),(study.SPLIT,study.END)])

    def test_price_population_mismatch_is_rejected(self):
        frame=observations('BTCUSDT',utc_ms(study.START));changed=frame.copy()
        changed.loc[0,'entry']+=.01
        with self.assertRaisesRegex(ValueError,'populations differ: entry'):
            study.assert_same_population(frame,changed)

    def test_nominal_success_cannot_override_adverse_execution_loss_or_ruin(self):
        summaries={name:{'decisive_trades':500,'win_rate':.60,'net_units':10.,
                        'weekly_block_interval':{'win_rate_lower_95':.56,'net_lower_95':1.}}
                   for name in ('validation','historical_confirmation','continuous_post_selection')}
        stressed={name:{'net_units':10.,'bankroll_can_fund_next_stake':True} for name in summaries}
        folds=[{'fit_cutoff_ms':2,'models':{'m':{'latest_settlement_ms':1,'training_converged':True}}}]
        nominal,stress=study.historical_checks({'id':'fixed'},summaries,stressed,folds)
        self.assertTrue(all(nominal.values()) and all(stress.values()))
        stressed['validation']['net_units']=-1.
        stressed['historical_confirmation']['bankroll_can_fund_next_stake']=False
        nominal,stress=study.historical_checks({'id':'fixed'},summaries,stressed,folds)
        self.assertTrue(all(nominal.values()))
        self.assertFalse(stress['adverse_2bps_validation_positive'])
        self.assertFalse(stress['adverse_2bps_historical_confirmation_no_ruin'])

    def test_missing_stress_evidence_fails_closed(self):
        summaries={name:{'decisive_trades':500,'win_rate':.60,'net_units':10.,
                        'weekly_block_interval':{'win_rate_lower_95':.56,'net_lower_95':1.}}
                   for name in ('validation','historical_confirmation','continuous_post_selection')}
        _,stress=study.historical_checks({'id':'fixed'},summaries,{},[])
        self.assertEqual(len(stress),6)
        self.assertFalse(any(stress.values()))


if __name__=='__main__':unittest.main()
