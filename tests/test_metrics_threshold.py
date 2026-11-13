"""Finite confidence catalog, cache-only execution and locked selection."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import metrics_threshold_research as study
from research.metrics_threshold_rules import candidates,directions,matched_control
from research.metrics_search_rules import candidates as old_candidates,directions as old_directions


class MetricsThresholdTests(unittest.TestCase):
    def frame(self):
        times=np.arange(6)*60000+1736121600000
        return pd.DataFrame({'symbol':'BTCUSDT','signal_ms':times,'ret_60':-20.,'rv_60':1.,'rv_ratio':.8,
            **{f'p_{family}_{arm}':[.56,.60,.64,.68,.5,np.nan] for family in ('logistic','gbm') for arm in ('aug','base')}})

    def test_fixed_sixteen_plus_sixteen_catalog_and_exact_controls(self):
        rules=candidates();self.assertEqual(len(rules),16);self.assertEqual(len(candidates(True)),32)
        self.assertEqual(sum(r['previous_study_threshold'] for r in rules),8)
        for rule in rules:
            control=matched_control(rule)
            self.assertEqual(control['threshold'],rule['threshold'])
            self.assertEqual(control['regime'],rule['regime'])
            self.assertEqual(control['model'],rule['model'])
            self.assertFalse(control['selection_candidate'])
        mutated=dict(rules[0],threshold=.62)
        with self.assertRaises(ValueError):directions(self.frame(),mutated)

    def test_original_eight_thresholds_have_identical_signals_to_prior_study(self):
        frame=self.frame()
        for old in old_candidates(True):
            new=next(r for r in candidates(True) if all(r[k]==old[k] for k in ('model','feature_set','threshold','regime')))
            np.testing.assert_array_equal(directions(frame,new),old_directions(frame,old))

    def test_new_thresholds_are_absolute_gates_and_ignore_future_execution(self):
        frame=self.frame();rule=next(r for r in candidates() if r['threshold']==.68 and r['regime']=='unrestricted')
        np.testing.assert_array_equal(directions(frame,rule),[0,0,0,1,0,0])
        frame['entry']=['unknown']*6;frame['exit']=[object()]*6;frame['settlement_ms']=0
        np.testing.assert_array_equal(directions(frame,rule),[0,0,0,1,0,0])

    def test_controls_never_enter_discovery_selection_or_correction_count(self):
        frame=self.frame()
        def simulation(frame,signal,first,stop,rule=None):
            wins=594 if rule['feature_set']=='base' else 354
            return [],{'decisive_trades':600,'wins':wins,'win_rate':wins/600,
                       'net_units':wins*4.25-(600-wins)*5,
                       'by_symbol_counts':{'BTCUSDT':300,'ETHUSDT':300}}
        with patch.object(study,'simulate_frame',side_effect=simulation):
            records,controls,chosen,ranking=study.evaluate_discovery(frame)
        self.assertEqual(len(records),16);self.assertEqual(len(controls),16);self.assertEqual(len(ranking),16)
        self.assertEqual(chosen['feature_set'],'aug')
        self.assertTrue(all(r['rule']['feature_set']=='aug' for r in ranking))

    def test_missing_cache_fails_before_any_array_read_without_fit_fallback(self):
        identities={'feature_store_identity':'base','metrics_store_identity':'metrics'}
        with tempfile.TemporaryDirectory() as folder,patch.object(study.np,'load',side_effect=AssertionError('No partial load')):
            with self.assertRaisesRegex(ValueError,'Incomplete read-only'):
                study.load_cached_phase(study.START,study.START+timedelta(days=1),identities,folder)
        self.assertFalse(hasattr(study,'fit_predict'))

    def test_incomplete_prior_study_blocks_later_read(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'has not completed'):
                study.stage7_completion_barrier({}, {}, Path(folder)/'missing.json')

    def test_implementation_is_written_before_discovery_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'metrics-threshold-plan.json').write_text('{}')
            identities={'feature_store_identity':'a','metrics_store_identity':'b'}
            saved=study.freeze_implementation(out,{},identities)
            self.assertTrue((out/'metrics-threshold-implementation.json').is_file())
            again=study.freeze_implementation(out,{},identities)
            self.assertEqual(saved,again)
            with self.assertRaisesRegex(ValueError,'changed'):
                study.freeze_implementation(out,{},dict(identities,metrics_store_identity='changed'))


if __name__=='__main__':unittest.main()
