"""Protect the published ledger from corrupt provenance and altered event math."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_high_win_public', ROOT/'scripts/build_high_win_public.py')
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)
AUDIT = json.loads((ROOT/'research/calendar-audit-results.json').read_text(encoding='utf-8'))
SOURCE = next(r for r in AUDIT['rows'] if r['rule_id'] == public.RULE and r['period'] == 'continuous_post_selection')
LEDGER = ROOT/SOURCE['ledger'].replace('\\','/')


class PublicLedgerTest(unittest.TestCase):
    def read(self, path=LEDGER, checksum=None):
        return public.read_verified_trades(path, checksum or public.sha(path),
            public.milliseconds('2025-05-07'), public.milliseconds('2026-11-12'), AUDIT['known_unavailable_dates_utc'])

    def changed_ledger(self, change):
        rows = list(csv.DictReader(io.StringIO(LEDGER.read_text(encoding='utf-8-sig'))))
        change(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ledger.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            return self.read(path)

    def test_published_cohort_exactly_matches_audited_rows(self):
        trades = self.read(checksum=SOURCE['ledger_sha256'])
        payload = json.loads((ROOT/'research/high-win-public.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['trades'], trades)
        measured = public.stats(trades)
        self.assertEqual((measured['signals'], measured['wins'], measured['losses'], measured['net_units'], measured['max_drawdown_units']),
                         (1262, 719, 543, 340.75, 107.25))
        self.assertEqual(trades[0]['entry_exact'], '1827.24186047')

    def test_source_hash_is_required(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.read(checksum='0'*64)

    def test_incorrect_outcome_rejected_even_with_new_hash(self):
        with self.assertRaisesRegex(ValueError, 'Outcome'):
            self.changed_ledger(lambda rows: rows[0].update(outcome='win'))

    def test_duplicate_trade_rejected_even_with_new_hash(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.changed_ledger(lambda rows: rows.insert(1, dict(rows[0])))

    def test_shortened_holding_period_rejected(self):
        with self.assertRaisesRegex(ValueError, 'clock'):
            self.changed_ledger(lambda rows: rows[0].update(settlement_ms=str(int(rows[0]['settlement_ms'])-60000)))

    def test_training_future_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Noncausal'):
            self.changed_ledger(lambda rows: rows[0].update(train_latest_settlement_ms=rows[0]['model_fit_cutoff_ms']))

    def test_public_projection_has_no_private_provenance_paths(self):
        raw = (ROOT/'research/high-win-public.json').read_text(encoding='utf-8')
        for marker in ['D:\\', 'C:\\', '.private', 'webhook', 'password', 'api_key', 'ledger_sha256', 'cache_path']:
            self.assertNotIn(marker, raw)


if __name__ == '__main__':
    unittest.main()
