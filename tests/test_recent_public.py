"""Public stake replay preserves event clocks, cash constraints and provenance."""
from copy import deepcopy
from datetime import date
from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd

from research.public_stake_engine import simulate_opportunities, validate_opportunities
from research.stake_sizing_engine import simulate_stakes
from scripts.build_recent_public import project_opportunities, REFERENCE_FIELDS

ROOT=Path(__file__).resolve().parents[1]
MINUTE=60_000
DAY=86_400_000
START=20_000*DAY


def row(minute,*,symbol='BTCUSDT',closed=101.,side=1):
    stamp=START+minute*MINUTE
    return {'symbol':symbol,'direction':side,'signal_ms':stamp,'entry_ms':stamp+MINUTE,
            'settlement_ms':stamp+31*MINUTE,'entry':100.,'exit':closed}


def opportunities(rows):
    frame=pd.DataFrame(rows)
    return project_opportunities(frame,frame.direction.to_numpy(dtype=np.int8))


class PublicReplayTests(unittest.TestCase):
    def test_reference_engine_parity_for_min_default_max_and_fractional_amounts(self):
        rows=[row(0,symbol='ETHUSDT'),row(0),row(30),row(31),row(61,closed=99.)]
        rows += [row(100+i*35,closed=99.) for i in range(8)]+[row(1440),row(1470,closed=100.)]
        frame=pd.DataFrame(rows);raw=opportunities(rows);before=deepcopy(raw)
        for stake in (5,5.01,30,249.99,250):
            for initial in (100,10000):
                with self.subTest(stake=stake,initial=initial):
                    actual,stats=simulate_opportunities(raw,START,START+2*DAY,stake=stake,initial_capital=initial)
                    expected,other=simulate_stakes(frame,frame.direction.to_numpy(),START,START+2*DAY,
                        {'id':'test','kind':'fixed','amount':stake},initial_bankroll=initial)
                    for key in REFERENCE_FIELDS:self.assertEqual(stats[key],other[key],key)
                    fields=('symbol','signal_ms','entry_ms','settlement_ms','outcome','stake','net_units','balance_after_settlement')
                    self.assertEqual([{k:t[k] for k in fields} for t in actual],
                                     [{k:t[k] for k in fields} for t in expected])
        self.assertEqual(before,raw)

    def test_stake_change_replays_admission_instead_of_scaling_old_ledger(self):
        raw=opportunities([row(i*35,closed=99.) for i in range(5)])
        small,_=simulate_opportunities(raw,START,START+DAY,stake=5,initial_capital=100)
        large,stats=simulate_opportunities(raw,START,START+DAY,stake=30,initial_capital=100)
        self.assertEqual(len(small),5)
        self.assertEqual(len(large),3)
        self.assertEqual(stats['net_usdt'],-90)
        self.assertEqual(stats['skipped_insufficient'],2)

    def test_future_or_exact_signal_time_settlement_cannot_release_capital(self):
        raw=opportunities([row(0),row(30),row(31),row(32)])
        trades,stats=simulate_opportunities(raw,START,START+DAY,stake=30,initial_capital=30)
        self.assertEqual([t['signal_ms'] for t in trades],[START,START+32*MINUTE])
        self.assertEqual(stats['skipped_insufficient'],2)
        self.assertEqual(trades[1]['equity_at_signal'],55.5)

    def test_calendar_window_is_signal_inclusive_expiry_exclusive(self):
        raw=opportunities([row(-1),row(0),row(1440-31)])
        ledger,_=simulate_opportunities(raw,START,START+DAY)
        self.assertEqual(len(ledger),1)
        self.assertEqual(ledger[0]['signal_ms'],START)

    def test_daily_risk_five_stakes_and_reset_on_next_utc_day(self):
        raw=opportunities([row(i*35,closed=99.) for i in range(7)]+[row(1440)])
        ledger,stats=simulate_opportunities(raw,START,START+2*DAY,stake=30)
        self.assertEqual(len(ledger),6)
        self.assertEqual(stats['skipped_daily_limit'],2)
        self.assertEqual({t['daily_budget_usdt'] for t in ledger},{150})
        blocked,_=simulate_opportunities(raw,START,START+2*DAY,stake=30,
                                         daily_risk_mode='fixed',daily_risk_limit=25)
        self.assertEqual(blocked,[])

    def test_exact_cent_validation_and_half_up_reward_rounding(self):
        raw=opportunities([row(0)])
        for value in (True,4.99,250.01,30.001,float('nan'),float('inf'),'bad'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                simulate_opportunities(raw,START,START+DAY,stake=value)
        ledger,_=simulate_opportunities(raw,START,START+DAY,stake=5.1)
        self.assertEqual(ledger[0]['net_units'],4.34)

    def test_corrupt_raw_prices_identity_and_order_fail_closed(self):
        raw=opportunities([row(0),row(40)])
        for mutation in (lambda x:x[0].update(outcome='loss'),
                         lambda x:x[0].update(entry_exact='NaN'),
                         lambda x:x[1].update(id=1),
                         lambda x:x[1].update(signal_ms=x[0]['signal_ms'])):
            broken=deepcopy(raw);mutation(broken)
            with self.assertRaises(ValueError):validate_opportunities(broken)

    def test_published_payload_replays_default_amount_without_any_cache(self):
        payload=json.loads((ROOT/'research/recent-public.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['assumptions']['default_stake'],30)
        self.assertEqual(payload['assumptions']['initial_capital'],10000)
        self.assertEqual(payload['assumptions']['daily_risk_multiplier'],5)
        self.assertTrue(payload['strategy']['posthoc'])
        self.assertFalse(payload['strategy']['is_live_strategy'])
        self.assertTrue(payload['research']['original_selected_failed'])
        for period in payload['periods']:
            trades,stats=simulate_opportunities(payload['opportunities'],period['start_ms'],period['end_exclusive_ms'])
            self.assertEqual(trades,payload['published'][period['id']]['trades'])
            self.assertEqual(stats,payload['published'][period['id']]['stats'])
        serialized=json.dumps(payload)
        for secret in ('D:\\','C:\\','webhook/send','WECOM_WEBHOOK','password','models/'):
            self.assertNotIn(secret,serialized)


if __name__=='__main__':unittest.main()
