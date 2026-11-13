"""Fixed model population, label purge and raw-score ensemble invariants."""
import copy
import unittest
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES
from research.model_search_models import specs,fit_predict,make_model,DAY_MS

MINUTE=60000
CUTOFF=21000*DAY_MS


def population():
    base=CUTOFF-70*DAY_MS+np.arange(3000,dtype=np.int64)*30*MINUTE
    stamps=np.sort(np.concatenate([base,base+MINUTE,base+5*MINUTE,
                                   [CUTOFF-31*MINUTE,CUTOFF-30*MINUTE,CUTOFF,CUTOFF+MINUTE]]))
    identity=np.arange(len(stamps),dtype=float)
    train=pd.DataFrame({name:identity+position/100 for position,name in enumerate(FEATURES)})
    train['signal_ms']=stamps;train['settlement_ms']=stamps+31*MINUTE
    train['y']=np.arange(len(stamps))%2;train['is_flat']=False;train['symbol']='BTCUSDT'
    test=train.iloc[:5].copy();test['signal_ms']=CUTOFF+np.arange(5)*MINUTE
    test['settlement_ms']=test.signal_ms+31*MINUTE
    test.index=[7,2,80,3,6]
    return train,test


class FakeModel:
    classes_=np.array([0,1])
    def __init__(self,name,records):self.name=name;self.records=records
    def fit(self,x,y):
        self.records[self.name]={'x':x.copy(),'y':y.copy()}
        self.mean=float(y.mean())
        return self
    def predict_proba(self,x):
        p=.4 if self.name=='logistic_5m' else .7 if self.name=='gbm_shallow_5m' else self.mean
        return np.tile([1-p,p],(len(x),1))


def factories(records):
    return {name:(lambda model=name:FakeModel(model,records)) for name,spec in specs().items() if spec['family']!='mean'}


class ModelSearchTests(unittest.TestCase):
    def test_model_catalog_is_finite_and_regularization_is_scaled_by_sampling_density(self):
        rules=specs();self.assertEqual(len(rules),6)
        self.assertEqual(rules['logistic_halfhour']['parameters']['C'],.1)
        self.assertAlmostEqual(rules['logistic_5m']['parameters']['C'],.1*5/30)
        self.assertAlmostEqual(rules['logistic_1m']['parameters']['C'],.1/30)
        self.assertFalse(rules['gbm_shallow_5m']['parameters']['early_stopping'])
        rules['logistic_1m']['parameters']['C']=99
        self.assertNotEqual(specs()['logistic_1m']['parameters']['C'],99)

    def test_each_cadence_is_clock_aligned_and_labels_strictly_precede_cutoff(self):
        train,test=population();records={}
        scores,meta=fit_predict(train,test,CUTOFF,factories=factories(records))
        self.assertEqual(list(scores.index),list(test.index))
        self.assertEqual(list(scores),['p_'+name for name in specs()])
        for name,record in records.items():
            positions=record['x'][:,0].astype(int)
            used=train.iloc[positions]
            stride=specs()[name]['sample_minutes']*MINUTE
            self.assertTrue((used.signal_ms%stride==0).all())
            self.assertTrue((used.settlement_ms<CUTOFF).all())
            self.assertTrue((used.signal_ms>=CUTOFF-180*DAY_MS).all())
            self.assertEqual(record['x'].shape[1],60)
            self.assertEqual(meta['models'][name]['training_count'],len(used))
            self.assertLess(meta['models'][name]['latest_settlement_ms'],CUTOFF)
        self.assertEqual(meta['models']['logistic_halfhour']['training_count'],3000)
        self.assertEqual(meta['models']['logistic_5m']['training_count'],6000)
        self.assertEqual(meta['models']['logistic_1m']['training_count'],9000)

    def test_future_training_outcomes_and_all_test_targets_cannot_change_scores(self):
        train,test=population();records={}
        a,_=fit_predict(train,test,CUTOFF,factories=factories(records))
        changed=train.copy()
        future=changed.settlement_ms>=CUTOFF
        changed.loc[future,FEATURES]=np.nan
        changed.loc[future,'y']=99
        changed.loc[future,'is_flat']=True
        altered=test.copy();altered['y']=99;altered['entry']=1e12;altered['exit']=.001
        again={};b,_=fit_predict(changed,altered,CUTOFF,factories=factories(again))
        pd.testing.assert_frame_equal(a,b)
        for name in records:
            np.testing.assert_array_equal(records[name]['x'],again[name]['x'])
            np.testing.assert_array_equal(records[name]['y'],again[name]['y'])

    def test_ensemble_is_fixed_mean_and_does_not_claim_calibration(self):
        train,test=population();scores,meta=fit_predict(train,test,CUTOFF,factories=factories({}))
        np.testing.assert_allclose(scores.p_ensemble_5m,.55)
        self.assertFalse(meta['raw_scores_are_calibrated_win_probabilities'])
        self.assertTrue(meta['local_research_only'])
        self.assertTrue(all(m['training_converged'] for m in meta['models'].values()))

    def test_nonconvergence_is_reported_without_changing_fixed_parameters(self):
        from sklearn.exceptions import ConvergenceWarning
        class Unconverged(FakeModel):
            n_iter_=np.array([400])
            def fit(self,x,y):
                warnings.warn('synthetic iteration limit',ConvergenceWarning)
                return super().fit(x,y)
        train,test=population();records={};builders=factories(records)
        builders['logistic_1m']=lambda:Unconverged('logistic_1m',records)
        _,meta=fit_predict(train,test,CUTOFF,factories=builders)
        item=meta['models']['logistic_1m']
        self.assertFalse(item['training_converged'])
        self.assertEqual(item['convergence_warning_count'],1)
        self.assertEqual(item['iterations'],[400])
        self.assertEqual(meta['specs']['logistic_1m']['parameters']['max_iter'],400)
        self.assertTrue(meta['models']['logistic_5m']['training_converged'])

    def test_flat_training_events_removed_but_prediction_population_unchanged(self):
        train,test=population()
        positions=train.index[(train.signal_ms% (30*MINUTE)==0)&(train.settlement_ms<CUTOFF)][:8]
        train.loc[positions,'is_flat']=True;train.loc[positions,'y']=0
        test['is_flat']=True
        scores,meta=fit_predict(train,test,CUTOFF,factories=factories({}))
        self.assertEqual(meta['models']['logistic_halfhour']['training_count'],2992)
        self.assertEqual(len(scores),len(test))

    def test_wrong_asset_nonfinite_history_and_outside_fold_fail_closed(self):
        train,test=population()
        changed=test.copy();changed['symbol']='ETHUSDT'
        with self.assertRaisesRegex(ValueError,'assets differ'):fit_predict(train,changed,CUTOFF,factories=factories({}))
        changed=train.copy();changed.loc[0,FEATURES[0]]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite historical'):fit_predict(changed,test,CUTOFF,factories=factories({}))
        changed=test.copy();changed.loc[changed.index[0],'signal_ms']=CUTOFF-60000
        with self.assertRaisesRegex(ValueError,'outside'):fit_predict(train,changed,CUTOFF,factories=factories({}))
        changed=test.copy();changed.loc[changed.index[0],'signal_ms']=CUTOFF+30*DAY_MS
        with self.assertRaisesRegex(ValueError,'outside'):fit_predict(train,changed,CUTOFF,factories=factories({}))

    def test_real_factories_match_declared_shapes_and_disable_implicit_random_validation(self):
        logistic=make_model('logistic_5m')
        self.assertEqual(logistic[-1].C,.1*5/30)
        shallow=make_model('gbm_shallow_5m');deep=make_model('gbm_deeper_5m')
        self.assertEqual((shallow.max_depth,shallow.max_iter),(2,100))
        self.assertEqual((deep.max_depth,deep.max_iter),(3,140))
        self.assertFalse(shallow.early_stopping)
        self.assertFalse(deep.early_stopping)
        with self.assertRaises(ValueError):make_model('ensemble_5m')


if __name__=='__main__':unittest.main()
