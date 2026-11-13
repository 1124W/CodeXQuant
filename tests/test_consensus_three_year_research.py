"""Frozen single-rule replay, continuous year boundaries and published witnesses."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from research import consensus_three_year_research as study
from research.event_engine import utc_ms

MINUTE=60_000
DAY=86_400_000


def observation(stamp, *, symbol='BTCUSDT', closed=101., a=.61, b=.62):
    cutoff=stamp-DAY
    return {'symbol':symbol,'signal_ms':stamp,'entry_ms':stamp+MINUTE,
            'settlement_ms':stamp+31*MINUTE,'entry':100.,'exit':closed,
            'p_logistic_1m':a,'p_gbm_deeper_5m':b,
            'model_fit_cutoff_ms':cutoff,'train_latest_settlement_ms':cutoff-MINUTE}


def normalized(rows):
    return study.normalize_opportunities(pd.DataFrame(rows),utc_ms(study.START),utc_ms(study.END))


class ConsensusThreeYearReportTests(unittest.TestCase):
    def test_frozen_plan_uses_only_requested_rule_and_three_complete_years(self):
        plan=study.make_plan()
        self.assertEqual((plan['start_utc'],plan['end_exclusive_utc']),('2023-11-12','2026-11-12'))
        self.assertEqual((study.END-study.START).days,1096)
        self.assertEqual(plan['candidate_count'],1)
        self.assertFalse(plan['selection_performed'])
        self.assertEqual(plan['rule']['id'],'pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted')
        self.assertEqual(plan['rule']['threshold'],.60)
        self.assertEqual(set(plan['models']),{'logistic_1m','gbm_deeper_5m'})
        self.assertEqual(plan['models']['logistic_1m']['sample_minutes'],1)
        self.assertEqual(plan['models']['gbm_deeper_5m']['sample_minutes'],5)
        self.assertEqual((plan['stake_usdt'],plan['initial_capital_usdt'],plan['daily_risk_limit_usdt']),
                         (30,10000,150))
        self.assertTrue(plan['historical_selection_bias_remains'])
        self.assertFalse(plan['pristine_holdout'])
        self.assertFalse(plan['publish_or_deploy'])

    def test_existing_frozen_plan_cannot_be_silently_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=study.freeze_plan(folder)
            before=(folder/'plan.json').read_bytes()
            self.assertEqual(study.freeze_plan(folder),original)
            revised=deepcopy(study.make_plan());revised['rule']['threshold']=.56
            with patch.object(study,'make_plan',return_value=revised),self.assertRaisesRegex(ValueError,'frozen plan changed'):
                study.freeze_plan(folder)
            self.assertEqual((folder/'plan.json').read_bytes(),before)

    def test_agreement_requires_each_score_at_threshold_and_never_reads_outcome(self):
        first=utc_ms(date(2024,1,1))
        rows=[observation(first,a=.6,b=.62),observation(first+MINUTE,a=.4,b=.38),
              observation(first+2*MINUTE,a=.61,b=.39),
              observation(first+3*MINUTE,a=.59999,b=.99),
              observation(first+4*MINUTE,a=.5,b=.5)]
        result=normalized(rows)
        self.assertEqual(result.direction.tolist(),[1,-1])
        changed=deepcopy(rows)
        for row in changed:row['exit']=10_000.
        self.assertEqual(normalized(changed).direction.tolist(),[1,-1])

    def test_future_settlement_or_future_fit_witness_is_rejected(self):
        original=observation(utc_ms(date(2024,1,1)))
        for update in ({'train_latest_settlement_ms':original['model_fit_cutoff_ms']},
                       {'train_latest_settlement_ms':original['model_fit_cutoff_ms']+MINUTE},
                       {'model_fit_cutoff_ms':original['signal_ms']+MINUTE}):
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'Noncausal fit witness'):
                normalized([{**original,**update}])

    def test_invalid_scores_duplicate_rows_and_forged_directions_fail_closed(self):
        original=observation(utc_ms(date(2024,1,1)))
        for value in (float('nan'),float('inf'),1.01,-.01):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'Invalid model score'):
                normalized([{**original,'p_logistic_1m':value}])
        with self.assertRaisesRegex(ValueError,'Duplicate source'):
            normalized([original,original])
        with self.assertRaisesRegex(ValueError,'differs from the frozen rule'):
            normalized([{**original,'direction':-1}])

    def test_known_closed_day_and_crossing_into_it_are_excluded(self):
        closed=utc_ms(date(2026,1,17))
        rows=[observation(closed-10*MINUTE),observation(closed),
              observation(utc_ms(date(2026,1,20)))]
        self.assertEqual(normalized(rows).signal_ms.tolist(),[utc_ms(date(2026,1,20))])

    def test_merge_keeps_early_trade_expiring_after_source_join(self):
        boundary=utc_ms(study.JOIN)
        early=normalized([observation(boundary-10*MINUTE)])
        existing=normalized([observation(boundary),observation(boundary+30*MINUTE)])
        result=study.combine_opportunities(early,existing)
        self.assertEqual(result.signal_ms.tolist(),[boundary-10*MINUTE,boundary,boundary+30*MINUTE])
        ledger,stats=study.replay(result)
        self.assertEqual([row['signal_ms'] for row in ledger],[boundary-10*MINUTE,boundary+30*MINUTE])
        self.assertGreater(ledger[0]['settlement_ms'],boundary)
        self.assertEqual(stats['skipped_overlap'],1)

    def test_merge_rejects_overlapping_or_misassigned_source_population(self):
        boundary=utc_ms(study.JOIN)
        first=normalized([observation(boundary-MINUTE)])
        later=normalized([observation(boundary)])
        with self.assertRaisesRegex(ValueError,'ranges overlap'):
            study.combine_opportunities(later,later)
        with self.assertRaisesRegex(ValueError,'ranges overlap'):
            study.combine_opportunities(first,first)

    def test_continuous_year_partition_keeps_boundary_trade_and_carries_equity(self):
        boundary=utc_ms(date(2024,11,12))
        frame=normalized([observation(boundary-DAY),
                          observation(boundary-10*MINUTE,closed=99.),
                          observation(boundary+5*MINUTE),
                          observation(boundary+40*MINUTE)])
        ledger,total=study.replay(frame)
        first=study.partition(ledger,date(2023,11,12),date(2024,11,12))
        second=study.partition(ledger,date(2024,11,12),date(2025,11,12))
        self.assertEqual(total['signals'],3)
        self.assertEqual(total['skipped_overlap'],1)
        self.assertEqual(first['signals'],1)
        self.assertEqual(first['net_usdt'],25.5)
        self.assertEqual(first['ending_balance'],10025.5)
        self.assertEqual(second['opening_balance'],10025.5)
        self.assertEqual(second['signals'],2)
        self.assertEqual(second['net_usdt'],-4.5)
        self.assertEqual(second['ending_balance'],total['ending_balance'])
        self.assertEqual(first['net_usdt']+second['net_usdt'],total['net_usdt'])

    def test_exact_settlement_boundary_belongs_only_to_following_year(self):
        boundary=utc_ms(date(2025,1,1))
        frame=normalized([observation(boundary-31*MINUTE)])
        ledger,_=study.replay(frame)
        before=study.partition(ledger,date(2024,1,1),date(2025,1,1))
        after=study.partition(ledger,date(2025,1,1),date(2026,1,1))
        self.assertEqual(before['signals'],0)
        self.assertEqual(after['signals'],1)
        self.assertEqual(after['opening_balance'],10000)
        self.assertEqual(after['ending_balance'],10025.5)

    def test_months_and_calendar_years_partition_total_without_restarts(self):
        frame=normalized([observation(utc_ms(date(2023,12,31))+1430*MINUTE,closed=99.),
                          observation(utc_ms(date(2024,12,31))+1430*MINUTE),
                          observation(utc_ms(date(2026,11,11)))])
        ledger,total=study.replay(frame)
        for unit in ('month','year'):
            rows=study.calendar_partitions(ledger,unit)
            self.assertEqual(sum(row['signals'] for row in rows),3)
            self.assertEqual(sum(round(row['net_usdt']*100) for row in rows),round(total['net_usdt']*100))
            for before,after in zip(rows,rows[1:]):
                self.assertEqual(before['ending_balance'],after['opening_balance'])
        self.assertEqual(len(study.calendar_partitions(ledger,'year')),4)

    def test_partition_refund_resets_losing_streak_like_main_engine(self):
        start=utc_ms(date(2024,1,1))
        frame=normalized([observation(start,closed=99.),
                          observation(start+40*MINUTE,closed=100.),
                          observation(start+80*MINUTE,closed=99.)])
        ledger,total=study.replay(frame)
        period=study.partition(ledger,date(2024,1,1),date(2024,2,1))
        self.assertEqual(total['longest_losing_streak'],1)
        self.assertEqual(period['longest_losing_streak'],1)

    def test_published_2221_opportunities_reconcile_both_recent_ledgers(self):
        published=json.loads((study.ROOT/'research/recent-public.json').read_text(encoding='utf-8'))
        rows=[]
        for row in published['opportunities']:
            item=observation(row['signal_ms'],symbol=row['symbol'],closed=row['exit'],
                             a=.61 if row['direction']==1 else .39,
                             b=.62 if row['direction']==1 else .38)
            item['entry']=row['entry'];rows.append(item)
        frame=normalized(rows)
        self.assertEqual(len(frame),2221)
        result=study.reconcile_recent(frame)
        self.assertEqual({row['period']:row['signals'] for row in result},
                         {'year_to_date':443,'recent_six_months':338})
        self.assertEqual({row['period']:row['net_usdt'] for row in result},
                         {'year_to_date':1473.,'recent_six_months':1126.5})
        self.assertTrue(all(row['per_trade_match'] for row in result))
        wrong=frame.copy()
        first=published['published']['year_to_date']['trades'][0]
        selected=(wrong.symbol==first['symbol'])&(wrong.signal_ms==first['signal_ms'])
        wrong.loc[selected,'exit']=first['entry']-10 if first['direction']==1 else first['entry']+10
        with self.assertRaisesRegex(ValueError,'changed a published recent trade'):
            study.reconcile_recent(wrong)


if __name__=='__main__':unittest.main()
