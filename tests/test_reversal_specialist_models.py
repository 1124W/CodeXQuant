"""The specialist learns settled reversal labels without ordinary-direction leakage."""
from datetime import date
import unittest
import warnings

import numpy as np
import pandas as pd

from research import reversal_specialist_models as model
from research.event_engine import DAY_MS,MINUTE_MS,utc_ms

CUTOFF=utc_ms(date(2026,5,20))


def rows(times,symbol):
    stamps=np.asarray(times,dtype=np.int64);n=len(stamps);seq=np.arange(n)
    frame=pd.DataFrame({name:np.full(n,.01) for name in model.INPUT_FEATURES})
    frame['ret_60']=np.where(seq%2==0,-.01,.01)
    frame['rv_60']=.001;frame['rv_ratio']=.8
    frame['symbol']=symbol;frame['signal_ms']=stamps
    frame['entry_ms']=stamps+MINUTE_MS;frame['settlement_ms']=stamps+31*MINUTE_MS
    frame['entry']=100.;side=-np.sign(frame.ret_60.to_numpy())
    frame['exit']=100.+side*np.where(seq%3==0,-1.,1.)
    return frame


def fixture():
    stamps=CUTOFF-(np.arange(300,0,-1)+1)*30*MINUTE_MS
    train=pd.concat([rows(stamps,s) for s in model.SYMBOLS],ignore_index=True)
    test=pd.concat([rows([CUTOFF,CUTOFF+MINUTE_MS],s) for s in model.SYMBOLS],ignore_index=True)
    test.index=[17,31,8,45]
    return train,test


class Recorder:
    classes_=np.array([1,0])
    n_iter_=np.array([3])
    def fit(self,matrix,target):
        self.matrix=matrix.copy();self.target=target.copy();return self
    def predict_proba(self,matrix):
        self.prediction_matrix=matrix.copy()
        return np.tile([.7,.3],(len(matrix),1))


class SpecialistModelTests(unittest.TestCase):
    def test_actual_scaler_logistic_pipeline_returns_valid_frozen_model_scores(self):
        train,test=fixture()
        predicted,metadata=model.fit_predict(train,test,CUTOFF)
        self.assertEqual(predicted.shape,(4,1))
        self.assertTrue(predicted.p_reversal.between(0,1).all())
        self.assertTrue(metadata['training_converged'])
        self.assertEqual(metadata['specification']['parameters']['C'],.1)

    def test_side_orientation_flips_directional_inputs_only_and_identifies_asset(self):
        _,test=fixture();original=test.copy()
        transformed=model.transformed_inputs(test)
        side=-np.sign(test.ret_60.to_numpy())
        np.testing.assert_array_equal(transformed[:,:len(model.SIGNED_FEATURES)],test[list(model.SIGNED_FEATURES)].to_numpy()*side[:,None])
        np.testing.assert_array_equal(transformed[:,len(model.SIGNED_FEATURES):-1],test[list(model.UNSIGNED_FEATURES)].to_numpy())
        np.testing.assert_array_equal(transformed[:,-1],[0.,0.,1.,1.])
        pd.testing.assert_frame_equal(test,original)

    def test_fit_pools_assets_and_predicts_reversal_success_with_explicit_class_mapping(self):
        train,test=fixture();recorder=Recorder()
        prediction,meta=model.fit_predict(train,test,CUTOFF,factory=lambda:recorder)
        self.assertEqual(prediction.index.tolist(),test.index.tolist())
        np.testing.assert_array_equal(prediction.p_reversal,np.full(4,.7))
        expected=((train.exit-train.entry)*(-np.sign(train.ret_60))>0).astype(int)
        np.testing.assert_array_equal(recorder.target,expected)
        self.assertEqual(meta['training_by_symbol'],{'BTCUSDT':300,'ETHUSDT':300})
        self.assertLess(meta['latest_settlement_ms'],CUTOFF)
        self.assertTrue(meta['training_converged'])

    def test_unsettled_old_offgrid_nongate_closed_day_labels_never_read(self):
        train,test=fixture();extra=rows([CUTOFF-30*MINUTE_MS,CUTOFF-181*DAY_MS,
            CUTOFF-45*MINUTE_MS,CUTOFF-302*30*MINUTE_MS,utc_ms(date(2026,1,18))],'BTCUSDT')
        extra.loc[3,'ret_60']=.000001
        extra['entry']='not-observed';extra['exit']='not-observed'
        changed=pd.concat([train,extra],ignore_index=True)
        test['entry']='future-price';test['exit']='future-price';test['y']='future-label'
        test['settlement_ms']='future-clock'
        recorder=Recorder();_,meta=model.fit_predict(changed,test,CUTOFF,factory=lambda:recorder)
        self.assertEqual(meta['training_count'],600)

    def test_refund_labels_are_omitted_and_nonfinite_used_features_fail(self):
        train,test=fixture();train.loc[0,'exit']=train.loc[0,'entry']
        _,meta=model.fit_predict(train,test,CUTOFF,factory=Recorder)
        self.assertEqual(meta['training_count'],599)
        train.loc[1,'imb']=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite specialist'):
            model.fit_predict(train,test,CUTOFF,factory=Recorder)

    def test_missing_asset_or_classes_cannot_train(self):
        train,test=fixture()
        with self.assertRaisesRegex(ValueError,'both assets'):
            model.fit_predict(train[train.symbol=='BTCUSDT'],test,CUTOFF,factory=Recorder)
        train['exit']=train.entry-np.sign(train.ret_60)
        with self.assertRaisesRegex(ValueError,'wins and losses'):
            model.fit_predict(train,test,CUTOFF,factory=Recorder)

    def test_fixed_direction_gate_ignores_current_outcome_and_masks_unavailable_calendar(self):
        frame=rows([utc_ms(date(2026,1,17))-31*MINUTE_MS,utc_ms(date(2026,1,20)),CUTOFF],'BTCUSDT')
        frame['p_reversal']=[.9,.60,.599999]
        frame['entry']='future';frame['exit']='future';frame['y']='future'
        np.testing.assert_array_equal(model.directions(frame),[0,-1,0])

    def test_specification_copies_and_probability_mass_validation(self):
        spec=model.specification();spec['parameters']['C']=99;spec['signed_features'].clear()
        self.assertEqual(model.specification()['parameters']['C'],.1)
        self.assertTrue(model.specification()['signed_features'])
        class Bad(Recorder):
            def predict_proba(self,matrix):return np.tile([.7,.7],(len(matrix),1))
        train,test=fixture()
        with self.assertRaisesRegex(ValueError,'Invalid reversal scores'):
            model.fit_predict(train,test,CUTOFF,factory=Bad)

    def test_convergence_warning_is_retained_for_acceptance_checks(self):
        from sklearn.exceptions import ConvergenceWarning
        class Unconverged(Recorder):
            def fit(self,matrix,target):
                warnings.warn('did not converge',ConvergenceWarning)
                return super().fit(matrix,target)
        train,test=fixture();_,meta=model.fit_predict(train,test,CUTOFF,factory=Unconverged)
        self.assertFalse(meta['training_converged'])
        self.assertEqual(meta['convergence_warning_count'],1)


if __name__=='__main__':unittest.main()
