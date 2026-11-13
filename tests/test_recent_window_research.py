"""Recent-window boundaries, chronological selection, and portfolio restarts."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import recent_window_research as study
from research.event_engine import utc_ms

MINUTE = 60_000
DAY = 86_400_000


def event(signal, *, symbol='BTCUSDT', closed=101.):
    return {'symbol':symbol, 'signal_ms':signal, 'entry_ms':signal+MINUTE,
            'settlement_ms':signal+31*MINUTE, 'entry':100., 'exit':closed}


def discovery(wins, count=300, btc=150):
    return {'decisive_trades':count, 'wins':wins, 'win_rate':wins/count,
            'net_units':wins*4.25-(count-wins)*5,
            'by_symbol_counts':{'BTCUSDT':btc, 'ETHUSDT':count-btc}}


class RecentWindowResearchTests(unittest.TestCase):
    def test_default_dates_use_six_calendar_months_and_same_year_discovery(self):
        windows=study.periods(date(2026,11,12))
        self.assertEqual(windows['year_to_date'],(date(2026,1,1),date(2026,11,12)))
        self.assertEqual(windows['discovery'],(date(2026,1,1),date(2026,5,12)))
        self.assertEqual(windows['recent_six_months'],(date(2026,5,12),date(2026,11,12)))
        self.assertEqual(windows['recent_first_half'],(date(2026,5,12),date(2026,8,12)))
        self.assertEqual(windows['recent_second_half'],(date(2026,8,12),date(2026,11,12)))
        self.assertEqual((windows['recent_six_months'][1]-windows['recent_six_months'][0]).days,184)
        self.assertEqual(study.shift_months(date(2024,8,31),-6),date(2024,2,29))

    def test_unverified_or_insufficient_selection_dates_are_rejected(self):
        for value in (date(2026,11,13),date(2026,7,1),date(2025,11,12),
                      datetime(2026,11,12,tzinfo=timezone.utc),'2026-11-12'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                study.periods(value)

    def test_signal_start_inclusive_and_expiry_end_strictly_exclusive(self):
        first,end=date(2026,5,12),date(2026,5,13)
        low,high=utc_ms(first),utc_ms(end)
        frame=pd.DataFrame([event(low-MINUTE),event(low),event(high-31*MINUTE),event(high)])
        ledger,stats=study.simulate_window(frame,{'id':'fixture'},first,end,
                                           direction=np.ones(len(frame),dtype=np.int8))
        self.assertEqual([row['signal_ms'] for row in ledger],[low])
        self.assertEqual(stats['signals'],1)
        self.assertTrue(all(row['settlement_ms']<high for row in ledger))
        last_valid=pd.DataFrame([event(high-32*MINUTE)])
        allowed,_=study.simulate_window(last_valid,{'id':'fixture'},first,end,
                                         direction=np.ones(1,dtype=np.int8))
        self.assertEqual(allowed[0]['settlement_ms'],high-MINUTE)

    def test_window_restart_replays_signals_instead_of_slicing_old_admissions(self):
        first,end=date(2026,5,12),date(2026,5,13)
        start=utc_ms(first)
        # The earlier position blocks the new-window signal in a continuous run.
        frame=pd.DataFrame([event(start-10*MINUTE),event(start),event(start+30*MINUTE)])
        signs=np.ones(len(frame),dtype=np.int8)
        old,_=study.simulate_window(frame,{'id':'fixture'},date(2026,5,11),end,direction=signs)
        fresh,_=study.simulate_window(frame,{'id':'fixture'},first,end,direction=signs)
        sliced=[t for t in old if t['signal_ms']>=start]
        self.assertEqual([t['signal_ms'] for t in sliced],[start+30*MINUTE])
        self.assertEqual([t['signal_ms'] for t in fresh],[start,start+30*MINUTE])

    def test_settlement_partition_retains_cross_boundary_trade_exactly_once(self):
        first,middle,end=date(2026,5,12),date(2026,8,12),date(2026,11,12)
        boundary=utc_ms(middle)
        frame=pd.DataFrame([event(boundary-61*MINUTE),
                            event(boundary-31*MINUTE),event(boundary,closed=99.)])
        ledger,total=study.simulate_window(frame,{'id':'fixture'},first,end,
                                           direction=np.ones(len(frame),dtype=np.int8))
        before=study.settlement_partition(ledger,first,middle)
        after=study.settlement_partition(ledger,middle,end)
        self.assertEqual([t['settlement_ms'] for t in ledger],
                         [boundary-30*MINUTE,boundary,boundary+31*MINUTE])
        self.assertEqual((before['signals'],after['signals']),(1,2))
        self.assertEqual(before['signals']+after['signals'],total['signals'])
        self.assertAlmostEqual(before['net_units']+after['net_units'],total['net_units'])

    def test_discovery_choice_does_not_read_later_or_year_to_date_outcomes(self):
        rows=[{'rule':{'id':'a'},'discovery':discovery(183)},
              {'rule':{'id':'b'},'discovery':discovery(177)}]
        original=deepcopy(rows)
        chosen,_=study.choose_discovery(rows,min_trades=200,min_per_symbol=50)
        rows[0].update(recent_six_months={'win_rate':0.,'net_units':-1000},
                       year_to_date={'win_rate':0.,'net_units':-1000})
        rows[1].update(recent_six_months={'win_rate':1.,'net_units':1000},
                       year_to_date={'win_rate':1.,'net_units':1000})
        again,_=study.choose_discovery(rows,min_trades=200,min_per_symbol=50)
        self.assertEqual(chosen['id'],'a')
        self.assertEqual(chosen,again)
        self.assertEqual([r['discovery'] for r in rows],[r['discovery'] for r in original])

    def test_no_discovery_candidate_is_not_replaced_by_a_later_winner(self):
        # One ineligible discovery event, then 240 well-spaced profitable events.
        records=[event(utc_ms(date(2026,1,2)),closed=99.)]
        start=utc_ms(date(2026,5,12))
        records += [event(start+i*12*60*MINUTE,symbol='BTCUSDT' if i%2==0 else 'ETHUSDT')
                    for i in range(240)]
        frame=pd.DataFrame(records)
        frame['p_logistic_halfhour']=.9
        frame['p_gbm_deeper_5m']=.1
        rules=[r for r in study.candidate_rules()
               if r['id'] in ('logistic_halfhour_0.56_unrestricted','gbm_deeper_5m_0.56_unrestricted')]
        with tempfile.TemporaryDirectory() as directory, \
                patch('research.recent_window_data.load_recent_predictions',return_value=(frame,{})), \
                patch.object(study,'candidate_rules',return_value=rules), \
                patch.object(study,'COMPARATORS',()),redirect_stdout(io.StringIO()):
            report=study.run(output_dir=Path(directory)/'new-run')
            frozen=json.loads((Path(directory)/'new-run'/'selection.json').read_text(encoding='utf-8'))
        self.assertIsNone(frozen['selected_rule'])
        self.assertIsNone(report['selection']['selected_rule'])
        self.assertFalse(report['historical_gates_passed'])
        self.assertEqual(report['details'],{})
        self.assertEqual(report['descriptive_top_rule_ids'],['logistic_halfhour_0.56_unrestricted',
                                                          'gbm_deeper_5m_0.56_unrestricted'])
        winner=next(r for r in report['matrix'] if r['rule']['id']==report['descriptive_top_rule_ids'][0])
        self.assertEqual(winner['recent_six_months']['signals'],240)
        self.assertEqual(winner['recent_six_months']['win_rate'],1.)


if __name__=='__main__':
    unittest.main()
