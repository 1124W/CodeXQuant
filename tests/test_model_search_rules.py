"""Search-space immutability and causal signal gate regressions."""
import unittest
import math
import numpy as np
import pandas as pd

from research.model_search_rules import MODEL_IDS,candidates,directions

MINUTE=60_000
ORIGIN=1_800_000_000_000


def row(minute=0,**changes):
    return {'symbol':'BTCUSDT','signal_ms':ORIGIN+minute*MINUTE,
            **{'p_'+model:.7 for model in MODEL_IDS},
            'ret_60':-3*math.sqrt(60),'rv_60':1.,'rv_ratio':.9,
            'ret_240':2*math.sqrt(240),'rv_240':1.,'ret_15':-math.sqrt(15),
            'ret_3':1.,**changes}


def rule(family='model',**changes):
    return next(r for r in candidates() if r['family']==family
                and all(r.get(k)==v for k,v in changes.items()))


class ModelSearchRuleTests(unittest.TestCase):
    def test_fixed_sixty_rules_and_subsets_cannot_expand_search_space(self):
        rules=candidates()
        self.assertEqual(len(rules),60)
        self.assertEqual(len({r['id'] for r in rules}),60)
        self.assertEqual(sum(r['family']=='model' for r in rules),54)
        self.assertEqual(sum(r['family']=='direct' for r in rules),6)
        self.assertEqual(len(candidates(['logistic_1m'])),15)
        rules[0]['threshold']=.99
        self.assertEqual(candidates()[0]['threshold'],.56)
        for models in (['not_a_model'],['logistic_1m','logistic_1m']):
            with self.assertRaises(ValueError):candidates(models)
        with self.assertRaises(ValueError):directions(pd.DataFrame([row()]),rules[0])
        self.assertTrue(all(r['score_is_calibrated_win_probability'] is False for r in candidates()))

    def test_unrestricted_uses_only_directional_score_and_honors_both_sides(self):
        r=rule(model='logistic_1m',threshold=.64,regime='unrestricted')
        frame=pd.DataFrame([{'symbol':'BTCUSDT','signal_ms':ORIGIN+i*MINUTE,
                             'p_logistic_1m':p} for i,p in enumerate((.64,.35,.63,.5,np.nan,1.01))])
        np.testing.assert_array_equal(directions(frame,r),[1,-1,0,0,0,0])

    def test_exhaustion_requires_opposing_move_extremeness_and_calm_volatility(self):
        r=rule(model='logistic_1m',threshold=.56,regime='exhaustion')
        frame=pd.DataFrame([row(0),row(1,ret_60=30),row(2,ret_60=-1),
                            row(3,rv_ratio=1.251),row(4,p_logistic_1m=.3,ret_60=30)])
        np.testing.assert_array_equal(directions(frame,r),[1,0,0,0,-1])

    def test_trend_pullback_needs_model_alignment_and_opposite_short_move(self):
        r=rule(model='logistic_1m',threshold=.56,regime='trend_pullback')
        frame=pd.DataFrame([row(0),row(1,ret_15=5),row(2,ret_240=1),
                            row(3,rv_ratio=1.51),row(4,p_logistic_1m=.3),
                            row(5,p_logistic_1m=.3,ret_240=-40,ret_15=5)])
        np.testing.assert_array_equal(directions(frame,r),[1,0,0,0,0,-1])

    def test_direct_thresholds_calm_volatility_and_price_confirmation(self):
        r=rule('direct',threshold=2.,require_price_turn=False)
        confirmed=rule('direct',threshold=2.,require_price_turn=True)
        frame=pd.DataFrame([row(0),row(1,ret_60=-math.sqrt(60)),
                            row(2,rv_ratio=1.01),row(3,ret_3=-1),
                            row(4,ret_60=30,ret_3=-1)])
        np.testing.assert_array_equal(directions(frame,r),[1,0,0,1,-1])
        np.testing.assert_array_equal(directions(frame,confirmed),[1,0,0,0,-1])
        self.assertEqual(directions(frame,r).dtype,np.int8)

    def test_execution_labels_and_later_rows_do_not_change_earlier_signals(self):
        initial=pd.DataFrame([row(0),row(1),row(2,symbol='ETHUSDT')])
        changed=initial.copy()
        for name in ('entry','exit','y','is_flat','net_units','settlement_ms','entry_ms'):
            changed[name]=[1e12,-1e12,0]
        changed=pd.concat([changed,pd.DataFrame([row(99,ret_60=1e12)])],ignore_index=True)
        for r in candidates():
            np.testing.assert_array_equal(directions(initial,r),directions(changed,r)[:3])

    def test_row_order_is_preserved_and_assets_do_not_mix(self):
        frame=pd.DataFrame([row(0),row(1,p_logistic_1m=.3,ret_60=30),row(2,symbol='ETHUSDT')])
        r=rule(model='logistic_1m',threshold=.56,regime='exhaustion')
        reordered=frame.iloc[[2,0,1]]
        np.testing.assert_array_equal(directions(reordered,r),directions(frame,r)[[2,0,1]])

    def test_invalid_needed_inputs_fail_closed_and_scope_errors_raise(self):
        r=rule(model='logistic_1m',threshold=.56,regime='exhaustion')
        frame=pd.DataFrame([row(0,rv_ratio=np.nan),row(1,rv_60=-1),row(2,ret_60=np.inf)])
        np.testing.assert_array_equal(directions(frame,r),[0,0,0])
        for frame in (pd.DataFrame([row(),row()]),pd.DataFrame([row(symbol='OTHER')]),
                      pd.DataFrame([row(signal_ms=ORIGIN+1)])):
            with self.assertRaises(ValueError):directions(frame,r)
        with self.assertRaises(ValueError):directions(pd.DataFrame([row()]),'not_a_rule')


if __name__=='__main__':unittest.main()
