"""Dense-array research admission parity and discovery-only selection checks."""
import copy
import unittest

import numpy as np
import pandas as pd

from research.high_win_research import evaluate
from research.model_search_research import simulate_frame,choose_discovery

MINUTE=60000
DAY=86400000
ORIGIN=20000*DAY


def event(minutes=0,*,symbol='BTCUSDT',entry=100.,exit_price=101.,direction=1):
    signal=ORIGIN+minutes*MINUTE
    return {'symbol':symbol,'signal_ms':signal,'entry_ms':signal+MINUTE,
            'settlement_ms':signal+31*MINUTE,'entry':entry,'exit':exit_price,
            'direction':direction}


def discovery(count=600,wins=360,btc=300):
    return {'signals':count,'decisive_trades':count,'wins':wins,'losses':count-wins,'refunds':0,
            'win_rate':wins/count,'net_units':wins*4.25-(count-wins)*5,
            'by_symbol_counts':{'BTCUSDT':btc,'ETHUSDT':count-btc}}


def rule(name):
    return {'id':name,'model':'synthetic','threshold':.58,'regime':'synthetic'}


class ModelSearchPortfolioTests(unittest.TestCase):
    def compare_simulation(self,raw,direction=None,stress=None):
        frame=pd.DataFrame(raw)
        frame.index=np.arange(len(frame))*7+3
        before=frame.copy(deep=True)
        signs=frame.direction.to_numpy() if direction is None else np.asarray(direction)
        first,last=ORIGIN,ORIGIN+3*DAY
        allowed=[{**row,'direction':int(sign)} for row,sign in zip(raw,signs) if sign]
        expected,expected_stats=evaluate(allowed,first,last,stress=stress)
        actual,stats=simulate_frame(frame,signs,first,last,stress=stress,rule=rule('candidate'))
        columns=['symbol','signal_ms','entry_ms','settlement_ms','direction','entry','exit','outcome','net_units']
        self.assertEqual([{k:r[k] for k in columns} for r in expected],
                         [{k:r[k] for k in columns} for r in actual])
        for name in ('signals','wins','losses','refunds','win_rate','net_units','max_drawdown_units'):
            self.assertEqual(stats[name],expected_stats[name],name)
        pd.testing.assert_frame_equal(frame,before)
        return actual,stats

    def test_dense_array_admissions_match_reference_with_interleaved_assets_and_sparse_indices(self):
        raw=[event(0,symbol='ETHUSDT'),event(0),event(1),event(30,exit_price=99),
             event(31,symbol='ETHUSDT'),event(60,direction=-1,exit_price=99),
             event(61),event(90,exit_price=100),event(120)]
        actual,_=self.compare_simulation(raw)
        self.assertEqual(actual[0]['symbol'],'BTCUSDT')
        self.assertEqual(len(actual),5)

    def test_zero_direction_excludes_candidate_without_filtering_existing_accepted_ledger(self):
        raw=[event(0),event(1,symbol='ETHUSDT'),event(30),event(31,symbol='ETHUSDT')]
        actual,_=self.compare_simulation(raw,[0,1,1,1])
        self.assertEqual(actual[0]['symbol'],'ETHUSDT')
        self.assertEqual(actual[0]['signal_ms'],ORIGIN+MINUTE)

    def test_stress_relabels_outcomes_and_recomputes_daily_risk_admissions(self):
        raw=[event(index*35,exit_price=100.01) for index in range(8)]
        actual,stats=self.compare_simulation(raw,stress={'fee':.01,'adverse_bps':2.})
        self.assertEqual(len(actual),5)
        self.assertEqual(stats['losses'],5)
        self.assertEqual(stats['net_units'],-25.25)
        short=[event(direction=-1,exit_price=99.99)]
        self.assertEqual(self.compare_simulation(short,stress={'adverse_bps':2.})[0][0]['outcome'],'loss')

    def test_events_crossing_evaluation_end_or_before_start_do_not_enter(self):
        raw=[event(-1),event(0),event(3*1440-31),event(3*1440)]
        actual,_=self.compare_simulation(raw)
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['signal_ms'],ORIGIN)

    def test_discovery_choice_ignores_validation_and_confirmation_fields(self):
        rows=[{'rule':rule('a'),'discovery':discovery(wins=372)},
              {'rule':rule('b'),'discovery':discovery(wins=354)}]
        initial=copy.deepcopy(rows)
        selected,_=choose_discovery(rows)
        self.assertEqual(selected['id'],'a')
        changed=copy.deepcopy(rows)
        changed[0]['validation']={'win_rate':0.,'net_units':-100000}
        changed[0]['historical_confirmation']={'win_rate':0.,'net_units':-100000}
        changed[1]['validation']={'win_rate':1.,'net_units':100000}
        changed[1]['historical_confirmation']={'win_rate':1.,'net_units':100000}
        again,_=choose_discovery(changed)
        self.assertEqual(selected,again)
        self.assertEqual(rows,initial)

    def test_eligibility_requires_58pct_500_total_and_100_per_asset(self):
        rows=[{'rule':rule('too_few'),'discovery':discovery(count=499,wins=400,btc=250)},
              {'rule':rule('one_asset'),'discovery':discovery(count=600,wins=500,btc=599)},
              {'rule':rule('below58'),'discovery':discovery(count=600,wins=347,btc=300)},
              {'rule':rule('eligible'),'discovery':discovery(count=600,wins=354,btc=300)}]
        chosen,_=choose_discovery(rows)
        self.assertEqual(chosen['id'],'eligible')
        self.assertIsNone(choose_discovery(rows[:-1])[0])

    def test_adjusted_score_ties_have_deterministic_lexical_selection(self):
        rows=[{'rule':rule('z'),'discovery':discovery()},
              {'rule':rule('a'),'discovery':discovery()}]
        self.assertEqual(choose_discovery(rows)[0]['id'],'a')


if __name__=='__main__':unittest.main()
