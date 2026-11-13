"""Paired feature ablations must share every training row, label and test row."""
from datetime import date
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import metrics_search_models as model
from research.event_engine import DAY_MS,MINUTE_MS,utc_ms

CUTOFF=utc_ms(date(2026,5,20))


def rows(times):
    stamps=np.asarray(times,dtype=np.int64);seq=np.arange(len(stamps))
    frame=pd.DataFrame({name:np.full(len(stamps),.01+column*.001)+np.sin(seq/17)*.0001
        for column,name in enumerate(model.AUGMENTED_FEATURES)})
    frame['symbol']='BTCUSDT';frame['signal_ms']=stamps
    frame['entry_ms']=stamps+MINUTE_MS;frame['settlement_ms']=stamps+31*MINUTE_MS
    frame['entry']=100.;frame['exit']=100.+np.where(seq%7<4,1.,-1.)
    frame['metrics_record_ms']=stamps//(5*MINUTE_MS)*(5*MINUTE_MS)-10*MINUTE_MS
    frame['metrics_available_ms']=frame.metrics_record_ms+10*MINUTE_MS
    frame['metrics_age_ms']=stamps-frame.metrics_available_ms
    return frame


def fixture():
    train=rows(CUTOFF-(np.arange(12012,0,-1)+7)*5*MINUTE_MS)
    test=rows([CUTOFF,CUTOFF+MINUTE_MS]);test.index=[71,3]
    return train,test


class Recorder:
    classes_=np.array([1,0]);n_iter_=np.array([2])
    def fit(self,matrix,target):self.matrix=matrix.copy();self.target=target.copy();return self
    def predict_proba(self,matrix):self.testing=matrix.copy();return np.tile([.6,.4],(len(matrix),1))


def recordings():
    records={name:Recorder() for name in model.MODEL_IDS}
    return records,{name:(lambda name=name:records[name]) for name in records}


class MetricsModelTests(unittest.TestCase):
    def test_paired_rows_labels_and_base_columns_are_identical_with_hash_proofs(self):
        train,test=fixture();records,factories=recordings()
        predicted,meta=model.fit_predict(train,test,CUTOFF,factories=factories)
        self.assertEqual(predicted.index.tolist(),test.index.tolist())
        self.assertEqual(predicted.columns.tolist(),['p_'+name for name in model.MODEL_IDS])
        np.testing.assert_array_equal(predicted.to_numpy(),np.full((2,4),.6))
        for family in ('logistic','gbm'):
            base,aug=records[family+'_base'],records[family+'_aug']
            np.testing.assert_array_equal(base.matrix,aug.matrix[:,:60])
            np.testing.assert_array_equal(base.target,aug.target)
            np.testing.assert_array_equal(base.testing,aug.testing[:,:60])
            pair=meta['pairs'][family]
            self.assertEqual(pair['base'],pair['aug'])
            self.assertEqual(pair['train_label_sha256'],hashlib.sha256(base.target.astype('i1').tobytes()).hexdigest())
            for key in ('train_signal_sha256','train_label_sha256','test_signal_sha256','training_count'):
                self.assertEqual(meta['models'][family+'_base'][key],meta['models'][family+'_aug'][key])
        self.assertGreater(meta['pairs']['gbm']['training_count'],meta['pairs']['logistic']['training_count'])

    def test_future_unsettled_old_offgrid_execution_labels_never_enter_fits(self):
        train,test=fixture();extra=rows([CUTOFF-30*MINUTE_MS,CUTOFF-181*DAY_MS,CUTOFF-42*MINUTE_MS])
        extra['entry']='unobserved';extra['exit']='unobserved'
        changed=pd.concat([train,extra],ignore_index=True)
        test['entry']='future';test['exit']='future';test['y']='future';test['settlement_ms']='future'
        _,factories=recordings();_,meta=model.fit_predict(changed,test,CUTOFF,factories=factories)
        _,factories=recordings();_,original=model.fit_predict(train,test,CUTOFF,factories=factories)
        self.assertEqual(meta['pairs'],original['pairs'])

    def test_flats_and_missing_metrics_do_not_create_different_control_training_rows(self):
        train,test=fixture();train.loc[0,'exit']=train.loc[0,'entry']
        records,factories=recordings();model.fit_predict(train,test,CUTOFF,factories=factories)
        self.assertEqual(len(records['gbm_base'].target),len(train)-1)
        self.assertEqual(len(records['gbm_base'].target),len(records['gbm_aug'].target))
        train.loc[1,model.METRIC_FEATURES[0]]=np.nan
        _,factories=recordings()
        with self.assertRaisesRegex(ValueError,'Nonfinite paired training'):
            model.fit_predict(train,test,CUTOFF,factories=factories)

    def test_unavailable_or_expired_metrics_provenance_is_rejected(self):
        for kind in ('future','expired'):
            train,test=fixture()
            if kind=='future':
                test['metrics_record_ms']+=5*MINUTE_MS
                test['metrics_available_ms']+=5*MINUTE_MS
                test['metrics_age_ms']-=5*MINUTE_MS
            else:
                test['metrics_record_ms']-=5*MINUTE_MS
                test['metrics_available_ms']-=5*MINUTE_MS
                test['metrics_age_ms']+=5*MINUTE_MS
            _,factories=recordings()
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'available and fresh'):
                model.fit_predict(train,test,CUTOFF,factories=factories)

    def test_changed_new_features_cannot_change_control_input_matrix(self):
        train,test=fixture();records,factories=recordings()
        model.fit_predict(train,test,CUTOFF,factories=factories)
        altered=train.copy();altered[model.METRIC_FEATURES]*=100
        other,other_factories=recordings();model.fit_predict(altered,test,CUTOFF,factories=other_factories)
        for family in ('logistic','gbm'):
            np.testing.assert_array_equal(records[family+'_base'].matrix,other[family+'_base'].matrix)
            self.assertFalse(np.array_equal(records[family+'_aug'].matrix,other[family+'_aug'].matrix))

    def test_bad_probability_mass_or_missing_factory_fails_closed(self):
        class Bad(Recorder):
            def predict_proba(self,matrix):return np.tile([.7,.7],(len(matrix),1))
        train,test=fixture();_,factories=recordings();factories['logistic_base']=Bad
        with self.assertRaisesRegex(ValueError,'Invalid paired directional'):
            model.fit_predict(train,test,CUTOFF,factories=factories)
        with self.assertRaisesRegex(ValueError,'four fixed'):
            model.fit_predict(train,test,CUTOFF,factories={})

    def test_fixed_architectures_and_specs_cannot_be_mutated_through_public_copy(self):
        spec=model.specs();spec['logistic']['parameters']['C']=999
        self.assertEqual(model.specs()['logistic']['parameters']['C'],.1)
        self.assertEqual(model.make_model('logistic')[-1].C,.1)
        self.assertEqual(model.make_model('gbm').max_depth,3)
        self.assertEqual(model.make_model('gbm').max_iter,140)


if __name__=='__main__':unittest.main()
