"""Date projections retain original rolling fits and fail on corrupt caches."""
from contextlib import ExitStack
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import recent_window_data as data
from research.event_engine import utc_ms


class RecentWindowDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'research'/'.cache'/'model-search'
        (self.cache/'predictions').mkdir(parents=True)
        self.cutoff = utc_ms(date(2025,12,28))
        self.boundary = utc_ms(date(2026,1,2))

    def fixture(self, *, missing_eth=False, noncausal=False, invalid_score=False, duplicate=False):
        self.items=[]; folds=[]
        latest = self.cutoff if noncausal else self.cutoff-60000
        for symbol in data.SYMBOLS:
            stamps=np.array([self.boundary-60000,self.boundary,self.boundary+60000],dtype=np.int64)
            if missing_eth and symbol=='ETHUSDT':stamps=stamps[:2]
            if duplicate and symbol=='BTCUSDT':stamps[-1]=stamps[-2]
            arrays={name:np.ones(len(stamps)) for name in data.COLUMNS}
            arrays.update(signal_ms=stamps,entry_ms=stamps+60000,settlement_ms=stamps+31*60000,
                model_fit_cutoff_ms=np.full(len(stamps),self.cutoff),
                train_latest_settlement_ms=np.full(len(stamps),latest))
            for name in data.MODEL_IDS:arrays['p_'+name]=np.full(len(stamps),.6)
            if invalid_score:arrays['p_logistic_halfhour'][0]=1.1
            fold={'fit_cutoff_ms':self.cutoff,'train_days':180,'refit_days':30,
                'test_start_ms':self.boundary-60000,'test_end_exclusive_ms':self.boundary+120000,
                'prediction_count':len(stamps),
                'models':{'test':{'latest_settlement_ms':latest,'training_converged':True}}}
            path=self.cache/'predictions'/f'{symbol}-original-fold.npz'
            np.savez(path,**arrays)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            meta={'symbol':symbol,'model_metadata':fold,'sha256':digest}
            metadata=path.with_suffix('.json');metadata.write_text(json.dumps(meta),encoding='utf-8')
            self.items.append({'file':path.name,'sha256':digest,
                               'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()})
            folds.append(fold)
        report={'provenance':{'feature_store_identity':'verified-identity','later':{'folds':folds}}}
        (self.root/'research'/'model-search-results.json').write_text(json.dumps(report),encoding='utf-8')

    def load(self,start=date(2026,1,2),end=date(2026,1,3)):
        with ExitStack() as stack:
            stack.enter_context(patch.object(data,'ROOT',self.root))
            stack.enter_context(patch.object(data,'CACHE',self.cache))
            verified=stack.enter_context(patch.object(data,'verified_store_identity',return_value='verified-identity'))
            preflight=stack.enter_context(patch.object(data,'preflight_predictions',return_value=self.items))
            result=data.load_recent_predictions(start,end)
            verified.assert_called_once_with()
            preflight.assert_called_once_with(data.SPLIT,data.END,'verified-identity')
            return result

    def test_new_start_slices_original_fold_without_reanchoring_or_writes(self):
        self.fixture()
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        frame,meta=self.load()
        self.assertEqual(len(frame),4)
        self.assertEqual(set(frame.model_fit_cutoff_ms),{self.cutoff})
        self.assertEqual(int(frame.signal_ms.min()),self.boundary)
        self.assertEqual(meta['loaded_rows_trimmed_at_window_edges'],2)
        self.assertEqual(meta['joint_asset_rows_dropped'],0)
        self.assertFalse(meta['training_or_download_performed'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_missing_asset_minute_is_dropped_for_both_asset_scan(self):
        self.fixture(missing_eth=True)
        frame,meta=self.load()
        self.assertEqual(len(frame),2)
        self.assertEqual(frame.signal_ms.nunique(),1)
        self.assertEqual(meta['joint_asset_rows_dropped'],1)

    def test_changed_prediction_bytes_fail_closed(self):
        self.fixture()
        path=self.cache/'predictions'/self.items[0]['file']
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed after preflight'):self.load()

    def test_fold_witness_prevents_relabeling_verified_scores(self):
        self.fixture()
        path=(self.cache/'predictions'/self.items[0]['file']).with_suffix('.json')
        meta=json.loads(path.read_text(encoding='utf-8'))
        meta['model_metadata']['other_fit']='different'
        path.write_text(json.dumps(meta),encoding='utf-8')
        self.items[0]['metadata_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'report witness'):self.load()

    def test_training_outcome_at_fit_boundary_is_not_causal(self):
        self.fixture(noncausal=True)
        with self.assertRaisesRegex(ValueError,'Noncausal'):self.load()

    def test_duplicate_minutes_and_invalid_model_scores_are_rejected(self):
        for kwargs,match in (({'duplicate':True},'duplicate'),({'invalid_score':True},'model score')):
            with self.subTest(kwargs=kwargs):
                self.fixture(**kwargs)
                with self.assertRaisesRegex(ValueError,match):self.load()

    def test_dates_outside_audited_period_fail_before_loading(self):
        for start,end in ((date(2025,1,1),data.END),(date(2026,1,1),date(2026,11,13)),
                          (data.END,data.END),(datetime(2026,1,1),data.END)):
            with self.subTest(start=start,end=end),patch.object(data,'verified_store_identity') as verified:
                with self.assertRaisesRegex(ValueError,'UTC calendar dates'):
                    data.load_recent_predictions(start,end)
                verified.assert_not_called()


if __name__=='__main__':unittest.main()
