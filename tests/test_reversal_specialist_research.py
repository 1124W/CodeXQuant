"""Fixed specialist checkpoint and discovery-selection isolation tests."""
from contextlib import ExitStack
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import reversal_specialist_research as study
from research.event_engine import DAY_MS,MINUTE_MS,utc_ms
from tests.test_reversal_specialist_models import rows


def projected(first):
    return pd.concat([rows([first,first+MINUTE_MS],s) for s in study.SYMBOLS],ignore_index=True)


def fake_fit(train,test,cutoff,**unused):
    return pd.DataFrame({'p_reversal':np.full(len(test),.7)},index=test.index),{
        'fit_cutoff_ms':cutoff,'test_start_ms':int(test.signal_ms.min()),
        'test_end_exclusive_ms':int(test.signal_ms.max())+MINUTE_MS,'prediction_count':len(test),
        'specification':study.specification(),'training_count':600,
        'training_by_symbol':{'BTCUSDT':300,'ETHUSDT':300},
        'latest_settlement_ms':cutoff-MINUTE_MS,'training_converged':True}


class SpecialistRunnerTests(unittest.TestCase):
    def test_unicode_numeric_cache_reuses_verified_predictions_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch.object(study,'_load_projected',side_effect=lambda first,stop,**kw:projected(first)),patch.object(study,'fit_predict',side_effect=fake_fit) as fit:
                frame,meta=study.prediction_phase(study.START,study.START+timedelta(days=1),'fixture',folder)
                again,again_meta=study.prediction_phase(study.START,study.START+timedelta(days=1),'fixture',folder)
                self.assertEqual(fit.call_count,1)
                pd.testing.assert_frame_equal(frame,again);self.assertEqual(meta,again_meta)
                binary=next(folder.glob('*.npz'))
                with np.load(binary,allow_pickle=False) as values:
                    self.assertEqual(values['symbol'].dtype.kind,'U')
                    for name in values.files:self.assertNotEqual(values[name].dtype.kind,'O')
                with binary.open('ab') as stream:stream.write(b'corruption')
                with self.assertRaisesRegex(ValueError,'corrupt'):
                    study.prediction_phase(study.START,study.START+timedelta(days=1),'fixture',folder)
                self.assertEqual(fit.call_count,1)

    def test_contract_binds_pipeline_and_incomplete_cache_never_retrains(self):
        contract=study._contract('fixture')
        self.assertEqual(contract['pipeline_sha256'],study.file_hash(study.ROOT/'research/reversal_specialist_research.py'))
        self.assertEqual(contract['known_unavailable_dates'],list(study.KNOWN_UNAVAILABLE_DATES))
        with patch.object(study,'KNOWN_UNAVAILABLE_DATES',('2026-01-01',)):
            self.assertNotEqual(study._contract('fixture'),contract)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch.object(study,'_load_projected',side_effect=lambda first,stop,**kw:projected(first)),patch.object(study,'fit_predict',side_effect=fake_fit) as fit:
                study.prediction_phase(study.START,study.START+timedelta(days=1),'fixture',folder)
                with self.assertRaisesRegex(ValueError,'Stale'):
                    study.prediction_phase(study.START,study.START+timedelta(days=1),'other',folder)
                next(folder.glob('*.npz')).unlink()
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    study.prediction_phase(study.START,study.START+timedelta(days=1),'fixture',folder)
                self.assertEqual(fit.call_count,1)

    def test_perfect_later_results_cannot_promote_discovery_failure(self):
        discovery=projected(utc_ms(study.START))
        later=pd.concat([projected(utc_ms(study.SPLIT)+DAY_MS),projected(utc_ms(study.CONFIRM)+DAY_MS)],ignore_index=True)
        for frame in (discovery,later):
            frame['ret_60']=-.01;frame['entry']=100.;frame['exit']=101.;frame['p_reversal']=.7
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            folder=Path(directory);calls=[]
            stack.enter_context(patch.object(study,'verified_store_identity',return_value='fixture'))
            def predict(start,end,identity):
                calls.append((start,end))
                self.assertTrue((folder/'reversal-specialist-plan.json').exists())
                if start==study.SPLIT:
                    locked=json.loads((folder/'reversal-specialist-selection.json').read_text(encoding='utf-8'))
                    self.assertIsNone(locked['selected_rule'])
                frame=discovery if start==study.START else later
                return frame,{'folds':[{'fit_cutoff_ms':1,'latest_settlement_ms':0,'training_converged':True}]}
            def baseline(start,end,identity):
                frame=(discovery if start==study.START else later).copy()
                frame['p_logistic_halfhour']=.6
                return frame,{}
            stack.enter_context(patch.object(study,'prediction_phase',side_effect=predict))
            stack.enter_context(patch.object(study,'load_baseline',side_effect=baseline))
            result=study.run(output_dir=folder)
            self.assertEqual(calls,[(study.START,study.SPLIT),(study.SPLIT,study.END)])
            self.assertIsNone(result['selection']['selected_rule'])
            self.assertEqual(len(result['all_candidates']),1)
            self.assertEqual(result['all_candidates'][0]['validation']['win_rate'],1.)
            self.assertEqual(result['selected']['validation']['signals'],0)
            self.assertFalse(result['historical_research_gates_passed'])
            self.assertEqual(result['paired_comparison_scope'],'fixed_candidate_disclosure_not_promotion')
            self.assertTrue(pd.read_csv(folder/'reversal-specialist-validation-trades.csv').empty)
            self.assertFalse(pd.read_csv(folder/'reversal-specialist-candidate-validation-trades.csv').empty)

    def test_stress_profitability_and_pathwise_capital_are_hard_gates(self):
        names=('validation','historical_confirmation','continuous_post_selection')
        nominal={name:{'decisive_trades':500,'win_rate':.60,'net_units':10.,
            'weekly_block_interval':{'win_rate_lower_95':.56,'net_lower_95':1.}} for name in names}
        stressed={name:{'net_units':10.,'bankroll_can_fund_next_stake':True} for name in names}
        folds=[{'fit_cutoff_ms':1,'latest_settlement_ms':0,'training_converged':True}]
        self.assertTrue(all(study.historical_checks({'id':'fixed'},nominal,stressed,folds).values()))
        stressed['validation']['bankroll_can_fund_next_stake']=False
        stressed['historical_confirmation']['net_units']=0.
        checks=study.historical_checks({'id':'fixed'},nominal,stressed,folds)
        self.assertFalse(checks['adverse_2bps_validation_no_ruin'])
        self.assertFalse(checks['adverse_2bps_historical_confirmation_positive'])
        checks=study.historical_checks({'id':'fixed'},nominal,{},folds)
        self.assertFalse(any(value for key,value in checks.items() if key.startswith('adverse_2bps_')))

    def test_stress_capital_gate_detects_intermediate_failure_despite_positive_finish(self):
        first=utc_ms(study.START)
        # A deliberately invalid recovery after depletion tests the diagnostic,
        # independently of the simulator which would stop admitting positions.
        ledger=[{'symbol':'BTCUSDT','signal_ms':first,'entry_ms':first+MINUTE_MS,
                 'settlement_ms':first+31*MINUTE_MS,'entry':100.,'exit':99.,'direction':1,
                 'outcome':'loss','net_units':-996.,'stake':5.,'horizon_minutes':30},
                {'symbol':'BTCUSDT','signal_ms':first+60*MINUTE_MS,'entry_ms':first+61*MINUTE_MS,
                 'settlement_ms':first+91*MINUTE_MS,'entry':100.,'exit':101.,'direction':1,
                 'outcome':'win','net_units':1000.,'stake':5.,'horizon_minutes':30}]
        result=study.stress_detail(ledger,first,first+DAY_MS)
        self.assertGreater(result['net_units'],0)
        self.assertEqual(result['minimum_settled_bankroll'],4.)
        self.assertFalse(result['bankroll_can_fund_next_stake'])


if __name__=='__main__':unittest.main()
