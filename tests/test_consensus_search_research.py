"""Fail-closed cache reuse and research ordering for consensus evaluation."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import consensus_search_research as study
from research.event_engine import utc_ms


class ConsensusResearchTests(unittest.TestCase):
    def test_plan_freezes_full_scope_and_disallows_forward_claims(self):
        plan = study.make_plan()
        self.assertEqual(plan['candidate_count'], 16)
        self.assertEqual(len(plan['rules']), 16)
        self.assertEqual(plan['baseline_rule_id'], 'logistic_halfhour_0.56_exhaustion')
        self.assertEqual(plan['capital']['payout'], .85)
        self.assertEqual(plan['capital']['fee'], 0.)
        self.assertTrue(plan['local_only'])
        for key in ('publish_or_deploy', 'pristine_holdout', 'forward_validated', 'post_selection_changes_allowed'):
            self.assertFalse(plan[key])

    def test_plan_exists_before_data_access_and_failure_never_evaluates(self):
        with tempfile.TemporaryDirectory() as folder:
            def refuse():
                plan = json.loads((Path(folder)/'consensus-search-plan.json').read_text(encoding='utf-8'))
                self.assertEqual(plan['candidate_count'], 16)
                raise ValueError('unavailable cache')
            with patch.object(study, 'verified_store_identity', side_effect=refuse), patch.object(study, 'prediction_phase') as loader:
                with self.assertRaisesRegex(ValueError, 'unavailable cache'):
                    study.run(folder)
                loader.assert_not_called()
            self.assertFalse((Path(folder)/'consensus-search-selection.json').exists())

    def test_missing_cache_fails_before_training_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(study, 'CACHE', Path(folder)), patch.object(study, 'prediction_phase') as loader:
                with self.assertRaisesRegex(ValueError, 'Missing prediction cache'):
                    study.load_verified_predictions(study.START, study.START+timedelta(days=1), 'identity')
                loader.assert_not_called()

    def test_corrupt_cache_fails_before_training_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)/'predictions'; cache.mkdir()
            start = study.START; end = start+timedelta(days=1); cutoff = utc_ms(study.cutoff_for(start))
            path = cache/f'BTCUSDT-{cutoff}-{utc_ms(start)}-{utc_ms(end)}.npz'
            path.write_bytes(b'corrupt')
            contract = {'models_sha256': study.file_hash(study.ROOT/'research/model_search_models.py'),
                        'data_code_sha256': study.file_hash(study.ROOT/'research/model_search_data.py'),
                        'feature_store_identity': 'identity', 'pipeline_version': 1}
            path.with_suffix('.json').write_text(json.dumps({'contract': contract, 'sha256': 'not-the-hash'}), encoding='utf-8')
            with patch.object(study, 'CACHE', Path(folder)), patch.object(study, 'prediction_phase') as loader:
                with self.assertRaisesRegex(ValueError, 'Stale/corrupt prediction cache'):
                    study.load_verified_predictions(start, end, 'identity')
                loader.assert_not_called()

    def test_empty_ledger_retains_machine_readable_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'ledger.csv'
            study.save_ledger(target, [])
            self.assertEqual(target.read_text().strip().split(','), study.LEDGER_COLUMNS)

    def test_calendar_excludes_touching_expiry_but_allows_next_day(self):
        start = int(study.datetime.fromisoformat('2026-01-24').replace(tzinfo=study.timezone.utc).timestamp()*1000)
        stamps = np.array([start-32*60000, start-31*60000, start, start+2*study.DAY_MS])
        frame = pd.DataFrame({'signal_ms': stamps, 'settlement_ms': stamps+31*60000})
        np.testing.assert_array_equal(study.calendar_mask(frame), [True, False, False, True])


if __name__ == '__main__': unittest.main()
