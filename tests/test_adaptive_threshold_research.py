"""Fail-closed cache provenance and research acceptance boundary checks."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import adaptive_threshold_research as study
from research.model_search_data import source_identity
from research.event_engine import DAY_MS, utc_ms


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


class CachedScoreFixture:
    """Two three-row assets; all fixture writes remain under a temp directory."""
    def __enter__(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cache = self.root/'research'/'.cache'/'model-search'
        (self.cache/'features').mkdir(parents=True)
        (self.cache/'predictions').mkdir()
        self.sources = source_identity()
        for name in set(self.sources)|{'research/model_search_models.py'}:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((study.ROOT/name).read_bytes())
        self.store = {'complete':True, 'sources_sha256':self.sources}
        write_json(self.cache/'features'/'manifest.json', self.store)
        identity = hashlib.sha256(json.dumps(self.store, sort_keys=True).encode()).hexdigest()
        self.contract = {
            'models_sha256':study.file_hash(self.root/'research/model_search_models.py'),
            'data_code_sha256':study.file_hash(self.root/'research/model_search_data.py'),
            'feature_store_identity':identity, 'pipeline_version':1}
        start = utc_ms(study.START)
        self.metadata = {
            'fit_cutoff_ms':start, 'test_start_ms':start,
            'test_end_exclusive_ms':start+30*DAY_MS, 'prediction_count':3,
            'models':{'logistic_halfhour':{
                'latest_settlement_ms':start-60000, 'training_converged':True}}}
        stamps = start+np.arange(1,4,dtype=np.int64)*60000
        self.arrays = {
            'signal_ms':stamps, 'entry_ms':stamps+60000,
            'settlement_ms':stamps+31*60000, 'entry':np.full(3,100.),
            'exit':np.array([101.,99.,102.]), 'ret_60':np.full(3,-.01),
            'rv_60':np.full(3,.001), 'rv_ratio':np.full(3,.8),
            'model_fit_cutoff_ms':np.full(3,start,dtype=np.int64),
            'train_latest_settlement_ms':np.full(3,start-60000,dtype=np.int64),
            'p_logistic_halfhour':np.full(3,.65), 'p_logistic_1m':np.full(3,.61),
            'p_gbm_deeper_5m':np.full(3,.62)}
        self.paths = {}
        for symbol in ('BTCUSDT','ETHUSDT'):
            binary = self.cache/'predictions'/f'{symbol}-{start}.npz'
            np.savez(binary, **self.arrays)
            meta = {'contract':self.contract, 'sha256':study.file_hash(binary),
                    'model_metadata':self.metadata, 'symbol':symbol}
            write_json(binary.with_suffix('.json'), meta)
            self.paths[symbol] = binary
        self.report = {'provenance':{
            'feature_store_identity':identity,
            'discovery':{'folds':[self.metadata,deepcopy(self.metadata)], 'prediction_rows':6},
            'later':{'folds':[], 'prediction_rows':0}}}
        self.report_path = self.root/'research'/'model-search-results.json'
        write_json(self.report_path, self.report)
        self.stack.enter_context(patch.object(study,'ROOT',self.root))
        self.stack.enter_context(patch.object(study,'CACHE',self.cache))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def alter_meta(self, symbol, mutation):
        path = self.paths[symbol].with_suffix('.json')
        meta = json.loads(path.read_text(encoding='utf-8'))
        mutation(meta)
        write_json(path,meta)


class AdaptiveCacheTests(unittest.TestCase):
    def test_verified_minimal_pair_loads_without_fitting(self):
        with CachedScoreFixture() as fixture:
            frame, provenance = study.load_cached_predictions()
            self.assertEqual(len(frame),6)
            self.assertEqual(frame.symbol.tolist(),['BTCUSDT','ETHUSDT']*3)
            self.assertEqual(provenance['joint_asset_rows_dropped'],0)
            self.assertEqual(len(provenance['folds']),2)
            np.testing.assert_array_equal(frame.signal_ms,np.repeat(fixture.arrays['signal_ms'],2))

    def test_missing_fold_is_rejected(self):
        with CachedScoreFixture() as fixture:
            fixture.paths['ETHUSDT'].with_suffix('.json').unlink()
            with self.assertRaisesRegex(ValueError,'fold'):
                study.load_cached_predictions()

    def test_corrupt_binary_is_rejected_before_parsing(self):
        with CachedScoreFixture() as fixture:
            with fixture.paths['BTCUSDT'].open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'provenance'):
                study.load_cached_predictions()

    def test_changed_contract_is_rejected(self):
        with CachedScoreFixture() as fixture:
            fixture.alter_meta('BTCUSDT',lambda m:m['contract'].update(pipeline_version=99))
            with self.assertRaisesRegex(ValueError,'provenance'):
                study.load_cached_predictions()

    def test_changed_manifest_is_rejected(self):
        with CachedScoreFixture() as fixture:
            fixture.store['unexpected']='changed'
            write_json(fixture.cache/'features'/'manifest.json',fixture.store)
            with self.assertRaisesRegex(ValueError,'manifest'):
                study.load_cached_predictions()

    def test_noncausal_or_unconverged_source_fit_is_rejected(self):
        for update in ({'training_converged':False},
                       {'latest_settlement_ms':utc_ms(study.START)}):
            with self.subTest(update=update), CachedScoreFixture() as fixture:
                fixture.alter_meta('BTCUSDT',lambda m:m['model_metadata']['models']['logistic_halfhour'].update(update))
                fixture.report['provenance']['discovery']['folds'][0]['models']['logistic_halfhour'].update(update)
                write_json(fixture.report_path,fixture.report)
                with self.assertRaisesRegex(ValueError,'Noncausal or unconverged'):
                    study.load_cached_predictions()

    def test_changed_fold_metadata_is_rejected_against_original_report(self):
        with CachedScoreFixture() as fixture:
            fixture.alter_meta('BTCUSDT',lambda m:m['model_metadata'].update(runtime_seconds=99.))
            with self.assertRaisesRegex(ValueError,'Fold metadata'):
                study.load_cached_predictions()

    def test_changed_feature_source_is_rejected_with_unchanged_cache_hashes(self):
        with CachedScoreFixture() as fixture:
            path=fixture.root/'live/features.py'
            with path.open('ab') as stream:
                stream.write(b'\n# altered fixture feature source\n')
            with patch('research.model_search_data.ROOT',fixture.root):
                with self.assertRaisesRegex(ValueError,'manifest'):
                    study.load_cached_predictions()

    def test_missing_pair_or_changed_original_population_is_rejected(self):
        with CachedScoreFixture() as fixture:
            fixture.report['provenance']['discovery']['prediction_rows']=8
            write_json(fixture.report_path,fixture.report)
            with self.assertRaisesRegex(ValueError,'population'):
                study.load_cached_predictions()

    def test_duplicate_asset_rows_are_rejected(self):
        with CachedScoreFixture() as fixture:
            fixture.alter_meta('ETHUSDT',lambda m:m.update(symbol='BTCUSDT'))
            with self.assertRaises(ValueError):
                study.load_cached_predictions()

    def test_altered_event_clock_rejected_even_if_checksum_updated(self):
        with CachedScoreFixture() as fixture:
            fixture.arrays['settlement_ms']=fixture.arrays['settlement_ms']+60000
            binary=fixture.paths['BTCUSDT']
            np.savez(binary,**fixture.arrays)
            fixture.alter_meta('BTCUSDT',lambda m:m.update(sha256=study.file_hash(binary)))
            with self.assertRaisesRegex(ValueError,'clocks'):
                study.load_cached_predictions()


class AdaptiveGateTests(unittest.TestCase):
    def test_calendar_includes_expiry_at_closed_day_start_and_full_closed_day(self):
        start=utc_ms(date(2026,1,17));minute=60000
        stamps=np.array([start-32*minute,start-31*minute,start-30*minute,
                         start,start+DAY_MS-1,start+3*DAY_MS],dtype=np.int64)
        actual=study.calendar_mask(pd.DataFrame({'signal_ms':stamps}))
        np.testing.assert_array_equal(actual,[True,False,False,False,False,True])

    def test_common_warmup_is_first_monday_after_at_least_180_days(self):
        start=study.common_start()
        self.assertEqual(start.weekday(),0)
        self.assertGreaterEqual((start-study.START).days,180)
        self.assertLess((start-study.START).days,187)
        self.assertLess(start,study.SPLIT)

    def test_point_wins_cannot_override_sample_and_interval_gates(self):
        summaries={name:{'decisive_trades':n,'win_rate':.60,'net_units':10.,
                         'weekly_block_interval':{'win_rate_lower_95':.55,'net_lower_95':1.}}
                   for name,n in [('validation',400),('historical_confirmation',100),
                                  ('continuous_post_selection',500)]}
        self.assertTrue(all(study.historical_checks({'id':'candidate'},summaries).values()))
        summaries['validation']['decisive_trades']=399
        summaries['historical_confirmation']['decisive_trades']=99
        summaries['continuous_post_selection']['weekly_block_interval']={
            'win_rate_lower_95':1/1.85,'net_lower_95':0.}
        checks=study.historical_checks(None,summaries)
        for name in ('selection_exists','validation_min400','confirmation_min100',
                     'weekly_lower_above_break_even','weekly_net_lower_positive'):
            self.assertFalse(checks[name])


if __name__=='__main__':
    unittest.main()
