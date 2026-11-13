"""Causal schedule comparison and admission reservations."""
import unittest

import pandas as pd

from research.minute_scan_research import opportunities,simulate_alert_admission

MINUTE=60000
ORIGIN=1800000*1000000


def trade(offset,*,symbol='BTCUSDT',expiry_price=101):
    signal=ORIGIN+offset*MINUTE
    return {'symbol':symbol,'signal_ms':signal,'entry_ms':signal+MINUTE,
            'settlement_ms':signal+31*MINUTE,'entry':100.,'exit':expiry_price,
            'direction':1,'p_up':.6,'ret_60':-30.,'rv_60':1.,'rv_ratio':1.}


class MinuteScanTests(unittest.TestCase):
    def test_more_scans_do_not_change_gate_or_holding_period(self):
        frame=pd.DataFrame([trade(0),trade(7),trade(30),{**trade(9),'p_up':.55}])
        self.assertEqual(len(opportunities(frame,30)),2)
        minute=opportunities(frame,1)
        self.assertEqual(len(minute),3)
        self.assertEqual([row['signal_ms'] for row in minute],[ORIGIN,ORIGIN+7*MINUTE,ORIGIN+30*MINUTE])
        self.assertTrue(all(row['settlement_ms']-row['entry_ms']==30*MINUTE for row in minute))

    def test_pending_position_blocks_dense_opportunities_and_btc_has_priority(self):
        ledger=simulate_alert_admission([trade(0,symbol='ETHUSDT'),trade(0),trade(1),trade(29),trade(30)])
        self.assertEqual([(row['symbol'],row['signal_ms']) for row in ledger],
                         [('BTCUSDT',ORIGIN),('BTCUSDT',ORIGIN+30*MINUTE)])

    def test_unsettled_outcome_is_not_used_for_signal_admission(self):
        losing=[trade(i,expiry_price=99) for i in (0,31,62,93)]
        first_win=simulate_alert_admission(losing+[trade(124,expiry_price=101),trade(154)])
        first_loss=simulate_alert_admission(losing+[trade(124,expiry_price=99),trade(154)])
        # At signal 154 the fifth result settles at 155 and cannot release the
        # already-reserved daily risk, even if history says it will win.
        self.assertEqual(len(first_win),5)
        self.assertEqual(len(first_loss),5)
        self.assertEqual([r['signal_ms'] for r in first_win],[r['signal_ms'] for r in first_loss])
        self.assertEqual(first_win[-1]['net_units'],4.25)
        self.assertEqual(first_loss[-1]['net_units'],-5.)


if __name__=='__main__':unittest.main()
