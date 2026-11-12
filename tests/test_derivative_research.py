"""Chronology and fair feature-ablation checks without market downloads or ML fits."""
from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import derivative_research as study
from research.derivative_features import FUNDING_FEATURES, PREMIUM_FEATURES
from research.event_engine import DAY_MS, MINUTE_MS, utc_ms
from research.event_regime_features import ADDED_FEATURES
from research.workbuddy_strategy import RESEARCH_FEATURES


ORIGIN_DAY=date(2025,1,1)
ORIGIN=utc_ms(ORIGIN_DAY)
BASE=RESEARCH_FEATURES+ADDED_FEATURES
ALL_FEATURES=BASE+PREMIUM_FEATURES+FUNDING_FEATURES


def samples(days=215):
    count=days*48
    position=np.arange(count)
    times=ORIGIN+position*30*MINUTE_MS
    frame=pd.DataFrame({name:np.sin(position/37+i)+i/20 for i,name in enumerate(ALL_FEATURES)})
    # The first matrix column identifies which historical rows reached fit().
    frame["ret_1"]=position.astype(float)
    frame["ret_15"]=-10.
    frame["ret_60"]=-30.
    frame["ret_240"]=100.
    frame["rv_60"]=2.
    frame["rv_240"]=2.
    frame["rv_ratio"]=1.
    frame["signal_ms"]=times
    frame["entry_ms"]=times+MINUTE_MS
    frame["settlement_ms"]=times+31*MINUTE_MS
    frame["entry"]=100.
    frame["y"]=(position%3==0).astype(np.int8)
    frame["exit"]=np.where(frame.y,101.,99.)
    frame["is_flat"]=False
    frame["symbol"]="BTCUSDT"
    frame["premium_observation_ms"]=times-MINUTE_MS
    frame["premium_close_available_ms"]=times
    frame["funding_observation_ms"]=times-2*MINUTE_MS-1
    frame["funding_available_ms"]=times-MINUTE_MS-1
    frame["unused_future_target"]=frame.y.astype(np.int64)*1_000_000
    frame.attrs["features"]=ALL_FEATURES.copy()
    return frame


class FakeModel:
    def __init__(self,name,records):
        self.name=name
        self.records=records

    def fit(self,x,y):
        self.records.append({"name":self.name,"x":x.copy(),"y":y.copy()})
        self.mean_label=float(y.mean())
        return self

    def predict_proba(self,x):
        # Both fit outcomes and observable prediction inputs matter, so future
        # mutation tests cannot pass merely because the stub is constant.
        p=np.clip(.2+.4*self.mean_label+.00001*x[:,0],.01,.99)
        return np.column_stack((1-p,p))


def factories(records):
    return {rule["id"]:(lambda name=rule["id"]:FakeModel(name,records)) for rule in study.rules()}


def run_prediction(frame,end_day,records):
    with patch.object(study,"START",ORIGIN_DAY):
        return study.predict(frame,ORIGIN+end_day*DAY_MS,factories=factories(records),progress=None)


class DerivativeResearchTests(unittest.TestCase):
    def test_eight_fixed_ablations_do_not_search_new_score_or_regime_thresholds(self):
        rules=study.rules()
        self.assertEqual(len(rules),8)
        self.assertEqual(len({rule["id"] for rule in rules}),8)
        self.assertEqual({r["block"] for r in rules},{"base50","premium","funding","both"})
        self.assertEqual({r["base_model"] for r in rules},{"regularized_gbm","regularized_logistic"})
        self.assertTrue(all(r["family"]=="model" and r["regime"]=="exhaustion" and r["threshold"]==.56 for r in rules))
        self.assertEqual(study.rules(),rules)

    def test_feature_block_counts_order_and_isolation(self):
        groups=study.feature_blocks(samples(1))
        self.assertEqual({name:len(cols) for name,cols in groups.items()},
                         {"base50":50,"premium":56,"funding":54,"both":60})
        self.assertEqual(groups["base50"],BASE)
        self.assertEqual(groups["premium"],BASE+PREMIUM_FEATURES)
        self.assertEqual(groups["funding"],BASE+FUNDING_FEATURES)
        self.assertEqual(groups["both"],ALL_FEATURES)
        forbidden={"y","entry","exit","signal_ms","settlement_ms","unused_future_target"}
        self.assertTrue(all(not forbidden.intersection(cols) for cols in groups.values()))

    def test_changed_baseline_feature_count_is_rejected(self):
        frame=samples(1)
        frame.attrs["features"]=frame.attrs["features"][1:]
        with self.assertRaisesRegex(ValueError,"50-feature"):
            study.feature_blocks(frame)

    def test_plan_separates_retrospective_new_history_from_forward_evidence(self):
        plan=study.make_plan(date(2026,11,12))
        self.assertEqual(plan["candidate_count"],8)
        self.assertEqual(plan["new_history_start_utc"],"2026-11-09")
        self.assertEqual(plan["end_exclusive_utc"],"2026-11-12")
        self.assertEqual(plan["publication_lag_ms"],{"premium":60_000,"funding":60_000})
        self.assertEqual((plan["min_new_history_days"],plan["min_new_history_trades"]),(30,100))
        self.assertTrue(plan["new_history_is_retrospective"])
        self.assertFalse(plan["pristine_holdout"])
        self.assertEqual(plan["forward_trades"],0)

    def test_train_labels_are_strictly_settled_and_all_ablations_share_population(self):
        frame=samples(212)
        records=[]
        predicted,folds=run_prediction(frame,212,records)
        self.assertEqual(len(folds),2)
        self.assertEqual(len(records),16)
        widths={r["id"]:len(study.feature_blocks(frame)[r["block"]]) for r in study.rules()}
        for i,record in enumerate(records):
            fold=folds[i//8]
            positions=record["x"][:,0].astype(int)
            self.assertEqual(record["x"].shape[1],widths[record["name"]])
            self.assertTrue((frame.settlement_ms.iloc[positions]<fold["fit_cutoff_ms"]).all())
            self.assertTrue((frame.signal_ms.iloc[positions]>=fold["fit_cutoff_ms"]-180*DAY_MS).all())
            np.testing.assert_array_equal(record["y"],frame.y.iloc[positions].to_numpy())
            # Every feature ablation uses exactly the same chronological rows.
            np.testing.assert_array_equal(positions,records[(i//8)*8]["x"][:,0].astype(int))
            self.assertLess(fold["train_latest_settlement_ms"],fold["fit_cutoff_ms"])
        self.assertNotIn(180*48-1,records[0]["x"][:,0])
        self.assertTrue((predicted.settlement_ms<ORIGIN+212*DAY_MS).all())
        self.assertTrue((predicted.premium_close_available_ms<=predicted.signal_ms).all())
        self.assertTrue((predicted.funding_available_ms<=predicted.signal_ms).all())

    def test_future_label_and_feature_mutations_cannot_change_first_fold(self):
        frame=samples(212)
        baseline_records=[]
        baseline,_=run_prediction(frame,212,baseline_records)
        changed=frame.copy()
        changed.loc[changed.signal_ms>=ORIGIN+180*DAY_MS,"y"]=1
        changed.loc[changed.signal_ms>=ORIGIN+180*DAY_MS,"exit"]=1_000_000.
        changed.loc[changed.signal_ms>=ORIGIN+210*DAY_MS,"ret_1"]+=1000
        future_records=[]
        actual,_=run_prediction(changed,212,future_records)
        for i in range(8):
            np.testing.assert_array_equal(baseline_records[i]["x"],future_records[i]["x"])
            np.testing.assert_array_equal(baseline_records[i]["y"],future_records[i]["y"])
        columns=["signal_ms"]+[f"p_{rule['id']}" for rule in study.rules()]
        before=baseline.signal_ms<ORIGIN+210*DAY_MS
        pd.testing.assert_frame_equal(baseline.loc[before,columns],actual.loc[before,columns])
        self.assertFalse(np.array_equal(baseline_records[8]["y"],future_records[8]["y"]))
        self.assertFalse(np.array_equal(baseline.loc[~before,columns[-1]],actual.loc[~before,columns[-1]]))

    def test_three_new_days_extend_last_fold_without_an_extra_fit(self):
        frame=samples(215)
        old_records=[]
        old,old_folds=run_prediction(frame,211,old_records)
        new_records=[]
        extended,new_folds=run_prediction(frame,214,new_records)
        self.assertEqual(len(old_records),16)
        self.assertEqual(len(new_records),16)
        self.assertEqual([f["fit_cutoff_ms"] for f in old_folds],
                         [f["fit_cutoff_ms"] for f in new_folds])
        self.assertEqual(new_folds[-1]["fit_cutoff_ms"],ORIGIN+210*DAY_MS)
        for old_fit,new_fit in zip(old_records,new_records):
            np.testing.assert_array_equal(old_fit["x"],new_fit["x"])
            np.testing.assert_array_equal(old_fit["y"],new_fit["y"])
        # The extended run also includes an old signal whose expiry crossed the
        # former end boundary. Compare only the exact previously eligible grid.
        shared=extended[extended.signal_ms.isin(old.signal_ms)].reset_index(drop=True)
        pd.testing.assert_frame_equal(old.reset_index(drop=True),shared)
        added=extended[extended.signal_ms>=ORIGIN+211*DAY_MS]
        self.assertGreater(len(added),100)
        self.assertTrue((added.model_fit_cutoff_ms==ORIGIN+210*DAY_MS).all())

    def test_prediction_flats_are_retained_and_training_flats_excluded(self):
        frame=samples(212)
        frame.loc[10,"is_flat"]=True
        future=frame.signal_ms>=ORIGIN+180*DAY_MS
        frame.loc[future,"is_flat"]=True
        frame.loc[future,"exit"]=100.
        records=[]
        predicted,_=run_prediction(frame,212,records)
        self.assertTrue(predicted.is_flat.all())
        self.assertEqual(len(predicted),32*48-1)
        self.assertNotIn(10.,records[0]["x"][:,0])

    def test_invalid_population_or_execution_timing_is_rejected_before_fit(self):
        frame=samples(212)
        duplicate=pd.concat([frame,frame.iloc[-1:]],ignore_index=True)
        duplicate.attrs=frame.attrs.copy()
        bad_feature=frame.copy();bad_feature.loc[2,"premium_mean60_bps"]=np.nan
        bad_entry=frame.copy();bad_entry.loc[2,"entry_ms"]+=MINUTE_MS
        bad_expiry=frame.copy();bad_expiry.loc[2,"settlement_ms"]-=MINUTE_MS
        for bad in (duplicate,bad_feature,bad_entry,bad_expiry):
            records=[]
            with self.subTest(bad_rows=len(bad)),self.assertRaises(ValueError):
                run_prediction(bad,212,records)
            self.assertEqual(records,[])

    def test_unbounded_model_scores_are_rejected(self):
        frame=samples(212)
        class BadModel:
            def fit(self,x,y):
                return self
            def predict_proba(self,x):
                return np.tile([-.1,1.1],(len(x),1))
        bad_factories={r["id"]:BadModel for r in study.rules()}
        with patch.object(study,"START",ORIGIN_DAY),self.assertRaisesRegex(ValueError,"Invalid raw model score"):
            study.predict(frame,ORIGIN+212*DAY_MS,factories=bad_factories,progress=None)


if __name__=="__main__":
    unittest.main()
