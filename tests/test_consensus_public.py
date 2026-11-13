"""The public three-year projection reconciles the immutable research artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import unittest
from research.public_stake_engine import simulate_opportunities

ROOT=Path(__file__).resolve().parents[1]


class ConsensusPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'research/consensus-public.json').read_text(encoding='utf-8'))

    def test_default_replay_and_negative_evidence_are_preserved(self):
        d=self.data;p=d['period']
        trades,stats=simulate_opportunities(d['opportunities'],p['start_ms'],p['end_exclusive_ms'])
        self.assertEqual(trades,d['published']['three_years']['trades'])
        self.assertEqual(stats,d['stats'])
        self.assertEqual((stats['signals'],stats['net_usdt'],stats['max_drawdown_usdt']),(1892,2236.5,895.5))
        self.assertEqual(next(row for row in d['yearly'] if row['year']==2024)['net_usdt'],-373.5)
        self.assertEqual(next(row for row in d['stress'] if row['adverse_bps']==2)['stats']['net_usdt'],-447)

    def test_all_configured_amounts_replay_the_raw_population(self):
        d=self.data;p=d['period']
        for witness in d['provenance']['numeric_replay_checks']:
            _,stats=simulate_opportunities(d['opportunities'],p['start_ms'],p['end_exclusive_ms'],stake=witness['stake'])
            for key,expected in witness.items():
                if key!='stake':self.assertEqual(stats[key],expected)

    def test_publication_has_auditable_scores_and_no_private_metadata(self):
        d=self.data
        self.assertTrue(d['strategy']['posthoc']);self.assertFalse(d['strategy']['forward_validated'])
        for row in d['opportunities']:
            self.assertLess(row['train_latest_settlement_ms'],row['model_fit_cutoff_ms'])
            self.assertLessEqual(row['model_fit_cutoff_ms'],row['signal_ms'])
            for name in ('p_logistic_1m','p_gbm_deeper_5m'):
                self.assertGreaterEqual(max(row[name],1-row[name]),.60)
                self.assertEqual(1 if row[name]>.5 else -1,row['direction'])
        body=json.dumps(d)
        for fragment in ('D:\\','C:\\','webhook/send','password','models/','.private'):
            self.assertNotIn(fragment,body)
        self.assertLess((ROOT/'research/consensus-public.json').stat().st_size,16*1024*1024)

    def test_csv_and_report_provenance_match_the_published_ledger(self):
        d=self.data
        with (ROOT/'research/consensus-public-trades.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),1892)
        self.assertEqual(sum(round(float(row['net_units'])*100) for row in rows),223650)
        report=ROOT/'research/consensus-three-year-2026-11-13/results.json'
        self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(),d['provenance']['report_sha256'])
        for name,expected in d['provenance']['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main()
