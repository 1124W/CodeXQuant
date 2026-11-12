"""Feature and outcome history remains causal across restart and retraining."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from live.features import FEATURES,SYMBOLS
from live.history import FeatureHistory,ROW_COLUMNS

ORIGIN=1794441600000
MINUTE=60_000
DAY=86_400_000


def features(value=1.):
    return {name:value+number/100 for number,name in enumerate(FEATURES)}


class FeatureHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.directory=Path(self.temp.name)
        self.path=self.directory/"history.sqlite3"
        self.history=FeatureHistory(self.path)
    def tearDown(self):
        self.temp.cleanup()

    def seed(self,rows=4,mutate=None):
        bundle=self.directory/"bundle"
        bundle.mkdir(exist_ok=True)
        manifest={"features":FEATURES.copy(),"training_cutoff_ms":ORIGIN+DAY,"models":{}}
        for symbol in SYMBOLS:
            records=[]
            for i in range(rows):
                record={**features(1.+i),"signal_ms":ORIGIN+i*30*MINUTE,
                        "settlement_ms":ORIGIN+i*30*MINUTE+31*MINUTE,
                        "y":i%2,"is_flat":False}
                if mutate:
                    mutate(symbol,i,record)
                records.append(record)
            path=bundle/f"{symbol}-training.csv.gz"
            pd.DataFrame(records,columns=ROW_COLUMNS).to_csv(path,index=False,compression="gzip")
            manifest["models"][symbol]={"training_samples_file":path.name,
                                       "training_samples_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        return bundle,manifest

    def test_pending_labels_wait_full_30min_and_five_second_observation_delay(self):
        self.history.record_features("BTCUSDT",ORIGIN,features())
        self.assertEqual(self.history.pending_labels(ORIGIN+31*MINUTE+4999),[])
        pending=self.history.pending_labels(ORIGIN+31*MINUTE+5000)
        self.assertEqual(pending,[{"symbol":"BTCUSDT","signal_ms":ORIGIN,
                                  "entry_ms":ORIGIN+MINUTE,"settlement_ms":ORIGIN+31*MINUTE}])

    def test_future_unknown_outcomes_never_enter_training(self):
        self.history.record_features("BTCUSDT",ORIGIN,features())
        self.assertTrue(self.history.training_rows(ORIGIN+10*DAY)["BTCUSDT"].empty)
        self.history.update_label("BTCUSDT",ORIGIN,100.,101.)
        train=self.history.training_rows(ORIGIN+DAY)["BTCUSDT"]
        self.assertEqual(len(train),1)
        self.assertEqual(train.y.iloc[0],1)
        self.assertEqual(self.history.pending_labels(ORIGIN+DAY),[])

    def test_strict_settlement_cutoff_and_180day_lower_bound(self):
        self.history.record_features("BTCUSDT",ORIGIN,features())
        self.history.update_label("BTCUSDT",ORIGIN,100,99)
        self.assertTrue(self.history.training_rows(ORIGIN+31*MINUTE)["BTCUSDT"].empty)
        self.assertEqual(len(self.history.training_rows(ORIGIN+31*MINUTE+1)["BTCUSDT"]),1)
        self.assertEqual(len(self.history.training_rows(ORIGIN+180*DAY)["BTCUSDT"]),1)
        self.assertTrue(self.history.training_rows(ORIGIN+180*DAY+1)["BTCUSDT"].empty)

    def test_recording_same_event_is_idempotent_and_material_mutations_cannot_overwrite(self):
        self.assertTrue(self.history.record_features("BTCUSDT",ORIGIN,features()))
        self.assertFalse(FeatureHistory(self.path).record_features("BTCUSDT",ORIGIN,features()))
        with self.assertRaises(ValueError):
            self.history.record_features("BTCUSDT",ORIGIN,features(5.))
        self.history.update_label("BTCUSDT",ORIGIN,100,102)
        frame=self.history.training_rows(ORIGIN+DAY)["BTCUSDT"]
        self.assertEqual(frame[FEATURES[0]].iloc[0],features()[FEATURES[0]])
        self.assertEqual(self.history.status()["total_rows"],1)

    def test_two_assets_and_none_action_population_are_retained(self):
        # There is deliberately no action/alert gate in the recording API.
        for symbol in SYMBOLS:
            self.history.record_features(symbol,ORIGIN,features())
            self.history.update_label(symbol,ORIGIN,100,101)
        self.assertEqual(self.history.status()["total_rows"],2)
        self.assertEqual({s:len(f) for s,f in self.history.training_rows(ORIGIN+DAY).items()},
                         {"BTCUSDT":1,"ETHUSDT":1})

    def test_label_observations_preserved_and_repeated_actual_prices_cannot_change(self):
        self.history.record_features("ETHUSDT",ORIGIN,features())
        self.assertTrue(self.history.update_label("ETHUSDT",ORIGIN,100,101))
        self.assertFalse(self.history.update_label("ETHUSDT",ORIGIN,100,101))
        for entry,exit_price in ((101,100),(100,100),(101,102)):
            with self.assertRaises(ValueError):
                self.history.update_label("ETHUSDT",ORIGIN,entry,exit_price)
        self.assertEqual(self.history.training_rows(ORIGIN+DAY)["ETHUSDT"].y.iloc[0],1)

    def test_flats_are_known_labels_retained_for_caller_to_exclude_from_classifier(self):
        self.history.record_features("BTCUSDT",ORIGIN,features())
        self.history.update_label("BTCUSDT",ORIGIN,100,100)
        frame=self.history.training_rows(ORIGIN+DAY)["BTCUSDT"]
        self.assertEqual(len(frame),1)
        self.assertEqual(frame.y.iloc[0],0)
        self.assertTrue(frame.is_flat.iloc[0])
        self.assertEqual(frame.attrs["features"],FEATURES)

    def test_invalid_feature_contract_time_price_or_unknown_label_target_fails(self):
        cases=[("SOLUSDT",ORIGIN,features()),("BTCUSDT",ORIGIN+MINUTE,features()),
               ("BTCUSDT",ORIGIN,{**features(),"exit":101}),
               ("BTCUSDT",ORIGIN,{**features(),FEATURES[0]:float("nan")})]
        for args in cases:
            with self.subTest(args=args[:2]),self.assertRaises(ValueError):
                self.history.record_features(*args)
        with self.assertRaises(KeyError):
            self.history.update_label("BTCUSDT",ORIGIN,100,101)
        self.history.record_features("BTCUSDT",ORIGIN,features())
        for value in (-1,0,float("nan"),True):
            with self.assertRaises(ValueError):
                self.history.update_label("BTCUSDT",ORIGIN,value,100)

    def test_seed_import_is_hash_checked_idempotent_across_restart_and_model_loading_forbidden(self):
        bundle,manifest=self.seed()
        with patch("pickle.loads",side_effect=AssertionError("Model deserialization is forbidden")):
            first=self.history.import_seed(bundle,manifest)
        self.assertEqual(first["BTCUSDT"]["inserted"],4)
        repeated=FeatureHistory(self.path).import_seed(bundle,manifest)
        self.assertTrue(repeated["BTCUSDT"]["already_imported"])
        self.assertEqual(repeated["ETHUSDT"]["inserted"],0)
        self.assertEqual(self.history.status()["total_rows"],8)
        frame=self.history.training_rows(ORIGIN+DAY)["BTCUSDT"]
        self.assertEqual(list(frame.columns),ROW_COLUMNS)
        self.assertEqual(frame.signal_ms.dtype,np.dtype("int64"))
        self.assertEqual(frame.is_flat.dtype,np.dtype("bool"))
        self.assertEqual(frame[FEATURES[1]].iloc[0],features()[FEATURES[1]])

    def test_seed_tamper_path_traversal_and_contract_mismatch_rejected_before_inserts(self):
        bundle,manifest=self.seed()
        entry=manifest["models"]["BTCUSDT"]
        path=bundle/entry["training_samples_file"]
        path.write_bytes(path.read_bytes()+b"tampered")
        with self.assertRaises(ValueError):
            self.history.import_seed(bundle,manifest)
        bundle,manifest=self.seed()
        manifest["models"]["BTCUSDT"]["training_samples_file"]="../escape.csv.gz"
        with self.assertRaises(ValueError):
            self.history.import_seed(bundle,manifest)
        self.assertEqual(self.history.status()["total_rows"],0)
        bundle,manifest=self.seed()
        manifest["features"]=FEATURES[::-1]
        with self.assertRaises(ValueError):
            self.history.import_seed(bundle,manifest)

    def test_seed_must_be_settled_at_manifest_cutoff_and_cannot_overwrite_collected_data(self):
        bundle,manifest=self.seed()
        manifest["training_cutoff_ms"]=ORIGIN+31*MINUTE
        with self.assertRaises(ValueError):
            self.history.import_seed(bundle,manifest)
        self.assertEqual(self.history.status()["total_rows"],0)
        bundle,manifest=self.seed()
        self.history.record_features("ETHUSDT",ORIGIN,features(99.))
        with self.assertRaises(ValueError):
            self.history.import_seed(bundle,manifest)
        # Both seed coins share a transaction: no partial BTC import survives.
        self.assertEqual(self.history.status()["total_rows"],1)
        self.assertEqual(self.history.status()["symbols"]["BTCUSDT"]["rows"],0)

    def test_new_seed_overlap_can_label_pending_but_never_delete_newer_rows(self):
        self.history.record_features("BTCUSDT",ORIGIN,features())
        self.history.record_features("BTCUSDT",ORIGIN+2*DAY,features(44.))
        bundle,manifest=self.seed()
        self.history.import_seed(bundle,manifest)
        self.assertEqual(self.history.status()["symbols"]["BTCUSDT"]["rows"],5)
        self.assertEqual(self.history.status()["pending_labels"],1)
        pending=self.history.pending_labels(ORIGIN+3*DAY)
        self.assertEqual(pending[0]["signal_ms"],ORIGIN+2*DAY)
        self.assertEqual(len(self.history.training_rows(ORIGIN+DAY)["BTCUSDT"]),4)


if __name__=="__main__":
    unittest.main()
