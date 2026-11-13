"""Independent synthetic tests for the fixed cross-model consensus catalog."""
import math
import unittest

import numpy as np
import pandas as pd

from research.consensus_search_rules import TRAINED_MODELS,candidates,directions

MINUTE=60_000
ORIGIN=1_800_000_000_000


def row(minute=0,**changes):
    return {'symbol':'BTCUSDT','signal_ms':ORIGIN+minute*MINUTE,
            **{'p_'+model:.7 for model in TRAINED_MODELS},
            'ret_60':-2*math.sqrt(60),'rv_60':1.,'rv_ratio':1.,**changes}


def selected(kind='pair',threshold=.60,regime='unrestricted'):
    return next(rule for rule in candidates() if rule['consensus_kind']==kind
                and rule['threshold']==threshold and rule['regime']==regime)


class ConsensusSearchRuleTests(unittest.TestCase):
    def test_catalog_is_fixed_copied_and_does_not_double_count_ensemble(self):
        rules=candidates()
        self.assertEqual(len(rules),16)
        self.assertEqual(len({rule['id'] for rule in rules}),16)
        self.assertEqual(sum(rule['consensus_kind']=='pair' for rule in rules),12)
        self.assertEqual(sum(rule['consensus_kind']=='unanimous_five' for rule in rules),4)
        self.assertTrue(all('ensemble_5m' not in rule['members'] for rule in rules))
        self.assertTrue(all(rule['score_is_calibrated_win_probability'] is False for rule in rules))
        self.assertTrue(all(rule['members_not_independent'] is True for rule in rules))
        rules[0]['members'][0]='ensemble_5m'
        self.assertEqual(candidates()[0]['members'][0],'logistic_halfhour')
        with self.assertRaises(ValueError):directions(pd.DataFrame([row()]),rules[0])

    def test_pair_requires_each_score_to_pass_and_directions_to_agree(self):
        rule=selected()
        frame=pd.DataFrame([row(0,p_logistic_halfhour=.60,p_gbm_deeper_5m=.60),
                            row(1,p_logistic_halfhour=.90,p_gbm_deeper_5m=.59),
                            row(2,p_logistic_halfhour=.80,p_gbm_deeper_5m=.20),
                            row(3,p_logistic_halfhour=.40,p_gbm_deeper_5m=.30)])
        np.testing.assert_array_equal(directions(frame,rule),[1,0,0,-1])
        self.assertEqual(directions(frame,rule).dtype,np.int8)

    def test_unanimity_uses_mean_not_individual_minimum(self):
        rule=selected('unanimous_five')
        first=row(0,p_logistic_halfhour=.51)
        second=row(1,**{'p_'+name:.59 for name in TRAINED_MODELS})
        third=row(2,**{'p_'+name:.30 for name in TRAINED_MODELS},p_unused=.99)
        third['p_logistic_halfhour']=.49
        frame=pd.DataFrame([first,second,third])
        np.testing.assert_array_equal(directions(frame,rule),[1,0,-1])
        # Scores from the already averaged ensemble are neither needed nor read.
        frame['p_ensemble_5m']=[0.,1.,np.nan]
        np.testing.assert_array_equal(directions(frame,rule),[1,0,-1])

    def test_neutral_or_one_opposing_member_blocks_unanimity(self):
        rule=selected('unanimous_five')
        frame=pd.DataFrame([row(0,p_logistic_5m=.5),row(1,p_logistic_5m=.49),
                            row(2,p_logistic_5m=.51)])
        np.testing.assert_array_equal(directions(frame,rule),[0,0,1])

    def test_exhaustion_preserves_reversal_extremeness_and_volatility_gate(self):
        rule=selected(regime='exhaustion')
        frame=pd.DataFrame([row(0),row(1,ret_60=20.),row(2,ret_60=-1.),
                            row(3,rv_ratio=1.251),row(4,rv_60=-1.),
                            row(5,p_logistic_halfhour=.3,p_gbm_deeper_5m=.4,ret_60=20.)])
        np.testing.assert_array_equal(directions(frame,rule),[1,0,0,0,0,-1])

    def test_invalid_member_scores_fail_closed_without_other_members_hiding_them(self):
        rule=selected('unanimous_five')
        frame=pd.DataFrame([row(0,p_logistic_5m=np.nan),row(1,p_logistic_5m=np.inf),
                            row(2,p_logistic_5m=1.01),row(3,p_logistic_5m=-.01)])
        np.testing.assert_array_equal(directions(frame,rule),[0,0,0,0])

    def test_future_outcomes_labels_and_reordering_do_not_change_past_signals(self):
        frame=pd.DataFrame([row(0),row(1,symbol='ETHUSDT'),row(2)])
        changed=frame.copy()
        for column in ('entry','exit','entry_ms','settlement_ms','y','is_flat','net_units'):
            changed[column]=[1e12,-1e12,0.]
        changed=pd.concat([changed,pd.DataFrame([row(99,p_logistic_5m=.1)])],ignore_index=True)
        for rule in candidates():
            np.testing.assert_array_equal(directions(frame,rule),directions(changed,rule)[:3])
            np.testing.assert_array_equal(directions(frame.iloc[[2,0,1]],rule),directions(frame,rule)[[2,0,1]])

    def test_scope_and_catalog_mutation_are_rejected(self):
        rule=selected()
        for frame in (pd.DataFrame([row(),row()]),pd.DataFrame([row(symbol='SOLUSDT')]),
                      pd.DataFrame([row(signal_ms=ORIGIN+1)])):
            with self.assertRaises(ValueError):directions(frame,rule)
        with self.assertRaises(ValueError):directions(pd.DataFrame([row()]),'unknown')
        for mutation in ({'threshold':.50},{'aggregation':'mean'},{'regime':'trend_pullback'}):
            with self.assertRaises(ValueError):directions(pd.DataFrame([row()]),{**rule,**mutation})
        unrestricted=pd.DataFrame([row()]).drop(columns=['ret_60','rv_60','rv_ratio'])
        np.testing.assert_array_equal(directions(unrestricted,rule),[1])


if __name__=='__main__':unittest.main()
