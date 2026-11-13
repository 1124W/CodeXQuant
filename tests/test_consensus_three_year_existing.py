"""Original fold witnesses, shared-minute filtering and split-boundary labels."""
from contextlib import ExitStack
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import consensus_three_year_existing as loader
from research.event_engine import utc_ms


class ExistingConsensusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root/'predictions'; self.cache.mkdir()
        (self.root/'research').mkdir()
        self.split = utc_ms(loader.SPLIT)
        self.cutoff = self.split-5*86400000
        self.verified = {}; self.folds = {}; self.arrays = {}

    def fixture(self, *, missing_eth=False, corrupt_clock=False):
        report = {'provenance':{'feature_store_identity':'fixture'}}
        for phase, times in [('discovery', [self.split-2*60000,self.split-60000]),
                             ('later', [self.split,self.split+60000])]:
            self.verified[phase] = []; self.folds[phase] = []
            population = 0
            for symbol in loader.SYMBOLS:
                values = times[:-1] if missing_eth and symbol == 'ETHUSDT' and phase == 'later' else times
                count = len(values); population += count
                arrays = {column:np.ones(count) for column in loader.COLUMNS}
                arrays.update(signal_ms=np.array(values), entry_ms=np.array(values)+60000,
                              settlement_ms=np.array(values)+31*60000,
                              model_fit_cutoff_ms=np.full(count,self.cutoff),
                              train_latest_settlement_ms=np.full(count,self.cutoff-60000),
                              entry=np.full(count,100.), exit=np.full(count,101.))
                for column in loader.COLUMNS:
                    if column.startswith('p_'):arrays[column] = np.full(count,.61)
                if corrupt_clock and symbol == 'BTCUSDT' and phase == 'later':
                    arrays['entry_ms'][0] += 60000
                fold = {'fit_cutoff_ms':self.cutoff,'train_days':180,'refit_days':30,
                        'test_start_ms':values[0],'test_end_exclusive_ms':values[-1]+60000,
                        'prediction_count':count, 'models':{'fixed':{
                            'training_converged':True,'latest_settlement_ms':self.cutoff-60000}}}
                first,stop = (self.split-2*60000,self.split) if phase == 'discovery' else (self.split,self.split+2*60000)
                name = f'{symbol}-{self.cutoff}-{first}-{stop}.npz'
                path = self.cache/name; np.savez(path,**arrays)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                meta = {'symbol':symbol,'model_metadata':fold,'sha256':digest}
                metadata = json.dumps(meta).encode(); path.with_suffix('.json').write_bytes(metadata)
                self.verified[phase].append({'file':name,'sha256':digest,
                                            'metadata_sha256':hashlib.sha256(metadata).hexdigest()})
                self.folds[phase].append(fold); self.arrays[name] = arrays
            dropped = 1 if missing_eth and phase == 'later' else 0
            report['provenance'][phase] = {'folds':self.folds[phase],
                                          'joint_asset_rows_dropped':dropped,
                                          'prediction_rows':population-dropped}
        self.report = report
        self.save_report()

    def save_report(self):
        (self.root/'research/model-search-results.json').write_text(json.dumps(self.report),encoding='utf-8')

    def run_loader(self, start=None, end=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(loader,'ROOT',self.root))
            stack.enter_context(patch.object(loader,'CACHE',self.cache))
            stack.enter_context(patch.object(loader,'verified_store_identity',return_value='fixture'))
            stack.enter_context(patch.object(loader,'preflight_predictions',side_effect=lambda first,stop,identity:
                                              self.verified['discovery' if first == loader.START else 'later']))
            return loader.load_existing_opportunities(start or loader.START,end or loader.END)

    def test_preserves_cross_split_events_and_original_fit_schedule(self):
        self.fixture(); frame,proof = self.run_loader()
        self.assertEqual(len(frame),8)
        self.assertEqual(set(frame.model_fit_cutoff_ms),{self.cutoff})
        cross = frame[(frame.signal_ms < self.split)&(frame.settlement_ms > self.split)]
        self.assertEqual(len(cross),4)
        self.assertTrue(proof['cross_phase_execution_labels_retained'])
        self.assertEqual(proof['phase_boundary']['BTCUSDT']['missing_signal_minutes'],0)
        self.assertEqual(proof['phase_boundary']['ETHUSDT']['missing_signal_minutes'],0)
        self.assertFalse(proof['training_or_download_performed'])

    def test_joint_asset_filter_precedes_strategy_and_year_coverage(self):
        self.fixture(missing_eth=True); frame,proof = self.run_loader()
        self.assertEqual(len(frame),6)
        self.assertNotIn(self.split+60000,set(frame.signal_ms))
        self.assertEqual(proof['joint_asset_rows_dropped'],1)
        year = next(row for row in proof['coverage_by_year'] if row['year'] == 2025)
        self.assertEqual(year['jointly_available_minutes'],3)
        self.assertEqual(year['joint_asset_rows_dropped'],1)

    def test_source_witness_change_fails_closed(self):
        self.fixture()
        self.report['provenance']['later']['folds'][0] = {**self.folds['later'][0],'prediction_count':999}
        self.save_report()
        with self.assertRaisesRegex(ValueError,'witness'):self.run_loader()

    def test_binary_tampering_and_clock_change_fail_closed(self):
        self.fixture()
        path = self.cache/self.verified['later'][0]['file']
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'contents'):self.run_loader()
        self.fixture(corrupt_clock=True)
        with self.assertRaisesRegex(ValueError,'clocks'):self.run_loader()

    def test_requested_subwindow_does_not_change_original_phase_verification(self):
        self.fixture(); frame,proof = self.run_loader(start=loader.SPLIT)
        self.assertEqual(len(frame),4)
        self.assertEqual(proof['source_prediction_rows'],8)
        self.assertEqual(len(proof['verified_cache_files']),4)

    def test_known_calendar_excludes_cross_boundary_and_in_day_opportunities(self):
        self.split = utc_ms(date(2026,1,17))
        self.cutoff = self.split-5*86400000
        self.fixture()
        with patch.object(loader,'SPLIT',date(2026,1,17)):
            frame,proof = self.run_loader()
        self.assertTrue(frame.empty)
        year = next(row for row in proof['coverage_by_year'] if row['year'] == 2026)
        self.assertEqual(year['opportunities_excluded_by_calendar'],8)
        self.assertEqual(year['calendar_excluded_joint_minutes'],4)

    def test_three_year_start_cannot_silently_treat_training_history_as_predictions(self):
        with self.assertRaises(ValueError):loader.load_existing_opportunities(date(2023,11,12),loader.END)


if __name__ == '__main__':unittest.main()
