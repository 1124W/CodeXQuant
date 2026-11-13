"""Money rounding, causal sizing, fixed daily risk and return/drawdown units."""
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from research.model_search_research import simulate_frame
from research.stake_sizing_engine import simulate_stakes

MINUTE=60_000
DAY=86_400_000
ORIGIN=20_000*DAY


def event(minutes=0,*,symbol='BTCUSDT',opened=100.,closed=101.,direction=1):
    stamp=ORIGIN+minutes*MINUTE
    return {'symbol':symbol,'signal_ms':stamp,'entry_ms':stamp+MINUTE,
            'settlement_ms':stamp+31*MINUTE,'entry':opened,'exit':closed,'direction':direction}


def run(rows,policy=None,initial=10000,bps=0):
    frame=pd.DataFrame(rows)
    return simulate_stakes(frame,frame.direction.to_numpy(),ORIGIN,ORIGIN+10*DAY,
        policy or {'id':'fixed300','kind':'fixed','amount':300},
        adverse_bps=bps,initial_bankroll=initial)


class StakeSizingTests(unittest.TestCase):
    def test_five_dollar_reference_parity_and_no_input_mutation(self):
        rows=[event(0,symbol='ETHUSDT'),event(0),event(1),event(30,closed=99),
              event(31,symbol='ETHUSDT'),event(60,direction=-1,closed=99),
              event(61),event(90,closed=100),event(120)]
        rows += [event(150+i*35,closed=99) for i in range(8)]
        frame=pd.DataFrame(rows);before=frame.copy(deep=True)
        old,old_stats=simulate_frame(frame,frame.direction.to_numpy(),ORIGIN,ORIGIN+10*DAY)
        new,stats=run(rows,{'id':'parity','kind':'fixed','amount':5},initial=1000)
        keys=['symbol','signal_ms','entry_ms','settlement_ms','entry','exit','direction','outcome','stake','net_units']
        self.assertEqual([{k:t[k] for k in keys} for t in new],[{k:t[k] for k in keys} for t in old])
        self.assertEqual(stats['net_usdt'],old_stats['net_units'])
        self.assertEqual(stats['max_drawdown_usdt'],old_stats['max_drawdown_units'])
        self.assertEqual(stats['signals'],old_stats['signals'])
        pd.testing.assert_frame_equal(frame,before)

    def test_fraction_uses_available_settled_balance_without_future_payout(self):
        # Next entry is exactly the previous expiry, but at the signal its
        # outcome is unavailable. Reserve 1000, size 10% of remaining 9000.
        rows=[event(0),event(30),event(61)]
        ledger,_=run(rows,{'id':'fraction10','kind':'fraction','fraction':.1})
        self.assertEqual([t['stake'] for t in ledger],[1000,900,995])
        self.assertEqual(ledger[1]['equity_at_signal'],10000)
        self.assertEqual(ledger[1]['available_at_signal'],9000)
        self.assertEqual(ledger[2]['equity_at_signal'],10850)
        # Altering an as-yet-unsettled payout cannot alter the stake placed now.
        rows[1]['exit']=99
        changed,_=run(rows,{'id':'fraction10','kind':'fraction','fraction':.1})
        self.assertEqual(changed[2]['stake'],ledger[2]['stake'])

    def test_fraction_daily_budget_is_frozen_and_next_day_reanchors(self):
        rows=[event(i*35,closed=99) for i in range(14)]+[event(1440)]
        ledger,stats=run(rows,{'id':'fraction10','kind':'fraction','fraction':.1})
        first_day=[t for t in ledger if t['entry_ms']//DAY==ORIGIN//DAY]
        self.assertEqual({t['daily_budget_usdt'] for t in first_day},{5000})
        self.assertGreater(stats['skipped_daily_limit'],0)
        following=ledger[-1]
        expected=int(Decimal(str(following['equity_at_signal']))*100/2)/100
        self.assertEqual(following['daily_budget_usdt'],expected)
        self.assertNotEqual(following['daily_budget_usdt'],5000)

    def test_price_stress_relabels_and_replays_daily_risk(self):
        rows=[event(i*35,closed=100.01) for i in range(8)]
        base,base_stats=run(rows)
        stressed,stats=run(rows,bps=2)
        self.assertEqual(base_stats['signals'],8)
        self.assertEqual(stats['signals'],5)
        self.assertEqual(stats['net_usdt'],-1500)
        self.assertEqual(stats['skipped_daily_limit'],3)
        self.assertTrue(all(t['outcome']=='loss' for t in stressed))

    def test_cent_profit_half_up_and_fraction_stake_rounds_down(self):
        _,stats=run([event()],{'id':'small','kind':'fixed','amount':.1},initial=1)
        self.assertEqual(stats['net_usdt'],.09)  # 0.085 -> 0.09
        ledger,_=run([event()],{'id':'floor','kind':'fraction','fraction':.333},initial=1)
        self.assertEqual(ledger[0]['stake'],.33)
        self.assertEqual(ledger[0]['net_units'],.28)

    def test_high_water_and_initial_capital_drawdowns_differ(self):
        rows=[event(i*35) for i in range(5)]+[event(175+i*35,closed=99) for i in range(4)]
        ledger,stats=run(rows,{'id':'fixed1000','kind':'fixed','amount':1000})
        self.assertEqual(stats['max_drawdown_usdt'],4000)
        self.assertAlmostEqual(stats['max_drawdown_initial_pct'],.4)
        self.assertAlmostEqual(stats['max_drawdown_pct'],4000/14250)
        self.assertEqual(stats['longest_losing_streak'],4)
        self.assertEqual(stats['max_drawdown_peak_ms'],ledger[4]['settlement_ms'])
        self.assertEqual(stats['max_drawdown_trough_ms'],ledger[-1]['settlement_ms'])

    def test_insufficient_balance_skips_instead_of_scaling_fixed_stake(self):
        ledger,stats=run([event(0,closed=99),event(35)],
                         {'id':'full','kind':'fixed','amount':1000},initial=1000)
        self.assertEqual(len(ledger),1)
        self.assertEqual(stats['ending_balance'],0)
        self.assertEqual(stats['skipped_insufficient'],1)
        self.assertEqual(stats['max_drawdown_pct'],1)

    def test_refund_returns_principal_and_resets_losing_streak(self):
        rows=[event(0,closed=99),event(35,closed=100),event(70,closed=99)]
        ledger,stats=run(rows)
        self.assertEqual(stats['refunds'],1)
        self.assertEqual(ledger[1]['net_units'],0)
        self.assertEqual(ledger[1]['balance_after_settlement'],9700)
        self.assertEqual(stats['longest_losing_streak'],1)

    def test_window_boundaries_and_invalid_clock_fail_closed(self):
        rows=[event(-1),event(0),event(10*1440-31)]
        ledger,_=run(rows)
        self.assertEqual(len(ledger),1)
        rows[1]['entry_ms']+=MINUTE
        with self.assertRaisesRegex(ValueError,'clocks'):run(rows)


if __name__=='__main__':unittest.main()
