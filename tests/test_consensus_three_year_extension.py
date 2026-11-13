"""The early extension cannot change fixed models, training clocks or samples."""
from datetime import date
import unittest
import warnings

import numpy as np
import pandas as pd

from live.features import FEATURES
from research import consensus_three_year_extension as extension
from research import model_search_models as original

MINUTE=60_000
DAY=86_400_000
CUTOFF=21_000*DAY


def population():
    base=CUTOFF-70*DAY+np.arange(3000,dtype=np.int64)*30*MINUTE
    stamps=np.sort(np.concatenate([base,base+MINUTE,base+5*MINUTE,
        [CUTOFF-31*MINUTE,CUTOFF-30*MINUTE,CUTOFF,CUTOFF+MINUTE]]))
    identity=np.arange(len(stamps),dtype=float)
    train=pd.DataFrame({name:identity+index/100 for index,name in enumerate(FEATURES)})
    train['signal_ms']=stamps;train['settlement_ms']=stamps+31*MINUTE
    train['y']=np.arange(len(stamps))%2;train['is_flat']=False;train['symbol']='BTCUSDT'
    test=train.iloc[:5].copy();test['signal_ms']=CUTOFF+np.arange(5)*MINUTE
    test['settlement_ms']=test.signal_ms+31*MINUTE
    test.index=[7,2,80,3,6]
    return train,test


class RecordingModel:
    classes_=np.array([0,1])
    def __init__(self,name,records):self.name=name;self.records=records
    def fit(self,matrix,labels):
        self.records[self.name]=(matrix.copy(),labels.copy());self.mean=float(labels.mean());return self
    def predict_proba(self,matrix):return np.tile([1-self.mean,self.mean],(len(matrix),1))


def builders(names,records,model=RecordingModel):
    return {name:(lambda key=name:model(key,records)) for name in names}


class EarlyExtensionTests(unittest.TestCase):
    def test_pair_is_exact_original_catalog_subset(self):
        expected={key:original.specs()[key] for key in extension.MODEL_IDS}
        self.assertEqual(extension.selected_specs(),expected)
        changed=extension.selected_specs();changed['logistic_1m']['parameters']['C']=999
        self.assertNotEqual(extension.selected_specs(),changed)

    def test_samples_scores_and_causal_metadata_match_original_procedure(self):
        train,test=population();pair_records={};all_records={}
        pair,metadata=extension.fit_pair(train,test,CUTOFF,
            factories=builders(extension.MODEL_IDS,pair_records))
        all_names=[name for name,spec in original.specs().items() if spec['family']!='mean']
        reference,reference_meta=original.fit_predict(train,test,CUTOFF,
            factories=builders(all_names,all_records))
        pd.testing.assert_frame_equal(pair,reference[['p_'+name for name in extension.MODEL_IDS]])
        for name in extension.MODEL_IDS:
            np.testing.assert_array_equal(pair_records[name][0],all_records[name][0])
            np.testing.assert_array_equal(pair_records[name][1],all_records[name][1])
            for field in ('training_count','sample_minutes','latest_settlement_ms','earliest_signal_ms','training_converged'):
                self.assertEqual(metadata['models'][name][field],reference_meta['models'][name][field])
            self.assertLess(metadata['models'][name]['latest_settlement_ms'],CUTOFF)
        self.assertEqual(set(pair_records),set(extension.MODEL_IDS))

    def test_future_outcomes_and_all_prediction_targets_cannot_change_fit(self):
        train,test=population();before={};after={}
        first,_=extension.fit_pair(train,test,CUTOFF,factories=builders(extension.MODEL_IDS,before))
        train.loc[train.settlement_ms>=CUTOFF,FEATURES]=np.nan
        train.loc[train.settlement_ms>=CUTOFF,'y']=999
        test['y']=-99;test['exit']=1e9;test['entry']=.001
        second,_=extension.fit_pair(train,test,CUTOFF,factories=builders(extension.MODEL_IDS,after))
        pd.testing.assert_frame_equal(first,second)
        for name in before:
            np.testing.assert_array_equal(before[name][0],after[name][0])
            np.testing.assert_array_equal(before[name][1],after[name][1])

    def test_original_thirty_day_clock_and_join_label_buffer(self):
        folds=list(extension.expected_folds());self.assertEqual(len(folds),6)
        self.assertEqual(extension.FEATURE_START,date(2023,5,13))
        self.assertEqual(extension.FEATURE_END,date(2024,5,8))
        self.assertEqual(folds[0][0],extension.utc_ms(date(2023,11,9)))
        self.assertEqual(folds[0][1],extension.utc_ms(date(2023,11,12)))
        self.assertEqual(folds[-1][2],extension.utc_ms(date(2024,5,7)))
        self.assertTrue(all(folds[i][2]==folds[i+1][1] for i in range(5)))

    def test_exact_cutoff_settlement_and_flat_labels_are_excluded(self):
        train,test=population();records={}
        row=train.index[0];train.loc[row,'is_flat']=True;train.loc[row,'y']=0
        _,meta=extension.fit_pair(train,test,CUTOFF,factories=builders(extension.MODEL_IDS,records))
        for name,(matrix,labels) in records.items():
            used=train.iloc[matrix[:,0].astype(int)]
            self.assertTrue((used.settlement_ms<CUTOFF).all())
            self.assertFalse(used.is_flat.any())
            self.assertTrue((used.signal_ms%(extension.selected_specs()[name]['sample_minutes']*MINUTE)==0).all())

    def test_nonconvergence_is_recorded_without_parameter_change(self):
        from sklearn.exceptions import ConvergenceWarning
        class Unconverged(RecordingModel):
            def fit(self,x,y):warnings.warn('synthetic',ConvergenceWarning);return super().fit(x,y)
        train,test=population()
        _,meta=extension.fit_pair(train,test,CUTOFF,factories=builders(extension.MODEL_IDS,{},Unconverged))
        self.assertTrue(all(not item['training_converged'] for item in meta['models'].values()))
        self.assertEqual(meta['specs'],extension.selected_specs())

    def test_invalid_future_scope_asset_and_extra_model_fail(self):
        train,test=population();bad=test.copy();bad['signal_ms']=CUTOFF+30*DAY
        with self.assertRaises(ValueError):extension.fit_pair(train,bad,CUTOFF)
        bad=test.copy();bad['symbol']='ETHUSDT'
        with self.assertRaises(ValueError):extension.fit_pair(train,bad,CUTOFF)
        with self.assertRaises(ValueError):extension.fit_pair(train,test,CUTOFF,factories={})


if __name__=='__main__':unittest.main()
