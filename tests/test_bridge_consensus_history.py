"""The one-time bridge preserves original observations and never rewinds state."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from deploy.bridge_consensus_history import bridge, connection, coverage, COLUMNS, MINUTE, DAY, OFFICIAL, SEED
from live.features import FEATURES, SYMBOLS
from live.history import FeatureHistory

START = 1794441600000


class BridgeConsensusHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'old.sqlite3'
        self.target = Path(self.temp.name) / 'new.sqlite3'
        for path in (self.source, self.target):
            FeatureHistory(path)
        for symbol in SYMBOLS:
            self.insert(self.target, self.row(symbol, START - MINUTE, seed=True))
        self.now = START + DAY

    def row(self, symbol, stamp, *, pending=False, seed=False, feature=1.):
        row = dict(zip(COLUMNS, (symbol, stamp, stamp + 31 * MINUTE,
                   json.dumps([feature] * len(FEATURES)), None, None, None, None, None)))
        if not pending:
            row.update(y=1, is_flat=0, entry_price=None if seed else 100.,
                       exit_price=None if seed else 101., label_source=SEED if seed else OFFICIAL)
        return row

    def insert(self, path, row):
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            db.execute('INSERT INTO feature_history (' + ','.join(COLUMNS) + ') VALUES (' +
                       ','.join('?' for _ in COLUMNS) + ')', tuple(row[name] for name in COLUMNS))

    def records(self, path):
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            return db.execute('SELECT ' + ','.join(COLUMNS) + ' FROM feature_history ORDER BY symbol,signal_ms').fetchall()

    def run_bridge(self, end=START + 2 * MINUTE):
        return bridge(self.source, self.target, START, end, now_ms=self.now)

    def test_original_rows_pending_and_labels_are_copied_exactly_source_is_unchanged(self):
        for symbol in SYMBOLS:
            self.insert(self.source, self.row(symbol, START))
            self.insert(self.source, self.row(symbol, START + MINUTE, pending=True))
        before = self.source.read_bytes()
        report = self.run_bridge()
        target = self.records(self.target)
        for row in self.records(self.source):
            self.assertIn(row, target)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(report['source_read_only'])
        self.assertEqual(report['symbols']['BTCUSDT']['inserted'], 2)
        self.assertEqual(report['symbols']['BTCUSDT']['observations']['maximum_missing_run_minutes'], 1)
        self.assertEqual(self.run_bridge()['symbols']['BTCUSDT']['duplicates'], 2)

    def test_source_connection_rejects_writes(self):
        db = connection(self.source, 'ro')
        self.addCleanup(db.close)
        with self.assertRaises(sqlite3.OperationalError):
            db.execute('DELETE FROM feature_history')

    def test_conflicting_last_asset_rolls_back_every_new_row(self):
        self.insert(self.source, self.row('BTCUSDT', START))
        self.insert(self.source, self.row('ETHUSDT', START, feature=2.))
        self.insert(self.target, self.row('ETHUSDT', START, feature=3.))
        before = self.records(self.target)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.run_bridge()
        self.assertEqual(self.records(self.target), before)

    def test_known_price_conflict_fails_even_if_direction_matches(self):
        original = self.row('BTCUSDT', START)
        changed = dict(original, exit_price=102.)
        self.insert(self.source, original)
        self.insert(self.target, changed)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.run_bridge()

    def test_insert_failure_after_first_asset_rolls_back_the_transaction(self):
        for symbol in SYMBOLS:
            self.insert(self.source, self.row(symbol, START))
        with closing(sqlite3.connect(self.target, isolation_level=None)) as db:
            db.execute("CREATE TRIGGER fail_second BEFORE INSERT ON feature_history "
                       "WHEN NEW.symbol='ETHUSDT' AND NEW.signal_ms=" + str(START) +
                       " BEGIN SELECT RAISE(ABORT,'injected insert failure'); END")
        before = self.records(self.target)
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_bridge()
        self.assertEqual(self.records(self.target), before)

    def test_existing_label_progress_is_preserved_and_pending_is_never_enriched(self):
        self.insert(self.source, self.row('BTCUSDT', START, pending=True))
        self.insert(self.target, self.row('BTCUSDT', START))
        self.insert(self.source, self.row('ETHUSDT', START))
        self.insert(self.target, self.row('ETHUSDT', START, pending=True))
        before = self.records(self.target)
        report = self.run_bridge()
        self.assertEqual(self.records(self.target), before)
        self.assertEqual(report['symbols']['ETHUSDT']['pending_target_kept'], 1)

    def test_old_sparse_seed_and_out_of_window_rows_are_not_copied(self):
        self.insert(self.source, self.row('BTCUSDT', START, seed=True))
        self.insert(self.source, self.row('BTCUSDT', START - MINUTE))
        self.insert(self.source, self.row('ETHUSDT', START + 3 * MINUTE))
        before = self.records(self.target)
        report = self.run_bridge()
        self.assertEqual(report['symbols']['BTCUSDT']['skipped_sparse_seed_rows'], 1)
        self.assertEqual(self.records(self.target), before)

    def test_noncausal_and_malformed_outcomes_leave_target_unchanged(self):
        cases = [dict(y=0), dict(is_flat=1), dict(exit_price=None),
                 dict(label_source='unknown'), dict(features_json='[NaN]'),
                 dict(settlement_ms=START + 30 * MINUTE)]
        before = self.records(self.target)
        for change in cases:
            with self.subTest(change=change):
                with closing(sqlite3.connect(self.source, isolation_level=None)) as db:
                    db.execute('DELETE FROM feature_history')
                self.insert(self.source, {**self.row('BTCUSDT', START), **change})
                with self.assertRaises(ValueError):
                    self.run_bridge()
                self.assertEqual(self.records(self.target), before)
        with closing(sqlite3.connect(self.source, isolation_level=None)) as db:
            db.execute('DELETE FROM feature_history')
        self.insert(self.source, self.row('BTCUSDT', START))
        with self.assertRaises(ValueError):
            bridge(self.source, self.target, START, START, now_ms=START + 31 * MINUTE)

    def test_contract_boundary_identity_and_future_window_are_rejected(self):
        for source, target, start, end in ((self.source, self.source, START, START),
                (self.source, self.target, START + MINUTE, START + MINUTE),
                (self.source, self.target, START, self.now + MINUTE),
                (self.source, self.target, START, START + 7 * DAY)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    bridge(source, target, start, end, now_ms=self.now)
        with closing(sqlite3.connect(self.source, isolation_level=None)) as db:
            db.execute("UPDATE feature_history_meta SET value='[]' WHERE name='features'")
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.run_bridge()

    def test_coverage_counts_empty_day_and_missing_boundary_runs(self):
        start = START - MINUTE
        end = START + DAY + MINUTE
        report = coverage([start, end], start, end)
        self.assertEqual(report['expected_minutes'], 1443)
        self.assertEqual(report['maximum_missing_run_minutes'], 1441)
        self.assertEqual(report['minimum_daily_coverage'], 0.)
        self.assertEqual(len(report['daily']), 3)

    def test_seed_day_prefix_is_included_but_unsettled_tail_is_excluded_from_label_coverage(self):
        # A deployed seed ends 31 minutes before cutoff. Here its final minute
        # precedes START; observations begin at START and can be unlabelled.
        self.insert(self.source, self.row('BTCUSDT', START, pending=True))
        report = bridge(self.source, self.target, START, START, now_ms=START + 10 * MINUTE)
        row = report['symbols']['BTCUSDT']
        self.assertEqual(row['observations']['expected_minutes'], 1)
        self.assertEqual(row['settled']['expected_minutes'], 0)


if __name__ == '__main__':
    unittest.main()
