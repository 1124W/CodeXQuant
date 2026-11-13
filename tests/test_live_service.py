"""Public dashboard boundaries without constructing a worker or using the network."""
import tempfile
import unittest
import gc
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock,patch

from live.app import Worker,create_app


class PublicWorker:
    webhook = 'not-a-public-setting'
    private_model_path = '/private/models'

    def snapshot(self):
        return {'mode':'research_alerts','symbols':{'BTCUSDT':{'index_price':65000.25,'price_time_ms':1794450000000,'stale':False}},
                'model':{'id':'test-model','trained_at':'2026-11-12'},
                'last_signal':{'symbol':'BTCUSDT','direction':-1,'signal_ms':1794450000000},
                'notifications':{'enabled':True},'order_execution_enabled':False,'forward_validated':False}


class LiveServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(PublicWorker()).test_client()

    def test_status_returns_public_worker_snapshot_without_worker_configuration(self):
        response = self.client.get('/api/live/status')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json(),PublicWorker().snapshot())
        self.assertEqual(response.get_json()['symbols']['BTCUSDT']['index_price'],65000.25)
        self.assertNotIn(b'not-a-public-setting',response.data)
        self.assertNotIn(b'/private/models',response.data)
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        self.assertEqual(response.headers['X-Content-Type-Options'],'nosniff')

    def test_health_is_read_only_and_missing_worker_is_explicit(self):
        health = self.client.get('/health').get_json()
        self.assertTrue(health['ok'])
        self.assertTrue(health['model_loaded'])
        self.assertFalse(health['order_execution_enabled'])
        no_worker = create_app().test_client()
        self.assertFalse(no_worker.get('/health').get_json()['model_loaded'])
        response = no_worker.get('/api/live/status')
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.get_json()['symbols'],{})
        self.assertFalse(response.get_json()['notifications']['enabled'])

    def test_existing_home_and_trade_page_publish_usable_relative_entry(self):
        home = self.client.get('/')
        self.addCleanup(home.close)
        self.assertEqual(home.status_code,200)
        self.assertIn(b'demo/trades.html',home.data)
        page = self.client.get('/demo/trades.html')
        self.addCleanup(page.close)
        self.assertEqual(page.status_code,200)
        self.assertIn(b'../trade-view.js',page.data)
        self.assertIn(b'../research/trade-chart-data.json',page.data)
        for route in ('/styles.css','/trade-view.js','/demo/quantdinger.html'):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.addCleanup(response.close)
                self.assertEqual(response.status_code,200)

    def test_public_trade_data_preserves_requested_cohort(self):
        response = self.client.get('/research/trade-chart-data.json')
        self.addCleanup(response.close)
        self.assertEqual(response.status_code,200)
        payload = response.get_json()
        self.assertEqual(len(payload['trades']),442)
        self.assertEqual(sum(row['outcome']=='win' for row in payload['trades']),251)
        self.assertEqual(sum(row['net_units'] for row in payload['trades']),111.75)

    def test_private_files_are_blocked_even_when_they_exist_in_app_root(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('.env','.private/production.env','external/private.py','live-state/history.sqlite3',
                         'models/BTCUSDT.joblib','research/.cache/raw.json','research/unlisted.json','demo/private.html'):
                path = folder/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('private-content-marker',encoding='utf-8')
            with patch('live.app.ROOT',folder):
                for route in ('/.env','/.private/production.env','/external/private.py','/live-state/history.sqlite3',
                              '/models/BTCUSDT.joblib','/research/.cache/raw.json','/research/unlisted.json',
                              '/demo/private.html','/research/../.env','/demo/%2e%2e/.env'):
                    with self.subTest(route=route):
                        response = self.client.get(route)
                        self.assertEqual(response.status_code,404)
                        self.assertNotIn(b'private-content-marker',response.data)
                        self.assertEqual(response.headers['X-Content-Type-Options'],'nosniff')

    def test_public_endpoints_accept_no_post_put_delete_mutations(self):
        for route in ('/','/api/live/status','/health','/demo/trades.html','/research/trade-chart-data.json'):
            for method in ('POST','PUT','DELETE'):
                with self.subTest(route=route,method=method):
                    self.assertEqual(self.client.open(route,method=method,json={'action':'order'}).status_code,405)


class WorkerAdmissionTests(unittest.TestCase):
    """Paper decisions use persistent fixed-stake risk, independent of model IO."""
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # sqlite3's transaction context does not close a connection. The worker
        # presently releases short-lived connections through GC; collect these
        # before Windows attempts to unlink this test's private database file.
        self.addCleanup(gc.collect)
        self.worker=Worker.__new__(Worker)
        self.worker.db=Path(self.directory.name)/'worker.sqlite3'
        with closing(sqlite3.connect(self.worker.db)) as con:
            con.execute('CREATE TABLE processed (slot INTEGER PRIMARY KEY,state TEXT NOT NULL)')
            con.execute('CREATE TABLE admissions (symbol TEXT,signal_ms INTEGER,entry_ms INTEGER,settlement_ms INTEGER,direction INTEGER,outcome TEXT,net REAL,PRIMARY KEY(symbol,signal_ms))')
            con.commit()
        self.worker._ensure_tables()
        self.day=int(datetime(2026,1,2,tzinfo=timezone.utc).timestamp()*1000)
        self.slot=self.day+3*1800000  # half past the hour, away from price-report scheduling

    def signal(self,slot=None,symbol='BTCUSDT',direction=1):
        slot=self.slot if slot is None else slot
        return {'symbol':symbol,'signal_ms':slot,'entry_due_ms':slot+60000,
                'expiry_due_ms':slot+31*60000,'direction':direction,
                'side':'LONG' if direction==1 else 'SHORT'}

    def stored(self):
        with closing(sqlite3.connect(self.worker.db)) as con:
            return con.execute('SELECT symbol,signal_ms,outcome,net FROM admissions ORDER BY signal_ms,symbol').fetchall()

    def configure_tick(self,decisions):
        now=self.slot+10000
        self.worker.lock=threading.RLock()
        self.worker.state={'symbols':{symbol:{'index_price':65000 if symbol=='BTCUSDT' else 2000,
                                              'price_time_ms':now,'stale':False}
                                      for symbol in ('BTCUSDT','ETHUSDT')},
                           'last_signal':None,'last_decisions':[],'last_error':None}
        self.worker.last_quote=now
        self.worker.last_label_poll=now
        self.worker.last_refresh_attempt=now
        self.worker.history=SimpleNamespace(record_features=Mock(),pending_labels=Mock(return_value=[]))
        self.worker.market=SimpleNamespace(quotes=Mock(),features_input=Mock(return_value=({}, {}, {})))
        self.worker.predictor=SimpleNamespace(manifest={'model_version':'synthetic','training_cutoff_ms':self.slot-86400000},
                                             predict=Mock(return_value=decisions))
        self.worker.notifier=SimpleNamespace(send_signal=Mock(return_value={'sent':True}),retry_pending=Mock(),
                                            send_settlement=Mock(return_value={'state':'sent','sent':True}))
        # Tick tests exercise maintenance deterministically; its non-blocking
        # scheduling/guard is tested independently below.
        self.worker.start_maintenance=self.worker._maintenance_work
        return now

    def test_tick_btc_priority_admits_only_one_of_two_same_entry_signals_and_no_replay(self):
        # Predictor's fixed SYMBOLS order is BTC then ETH, independent of outcomes.
        now=self.configure_tick([self.signal(),self.signal(symbol='ETHUSDT',direction=-1)])
        rows={'BTCUSDT':{},'ETHUSDT':{}}
        with patch('live.features.build_latest_features',return_value=rows),patch('live.app.time.time',return_value=now/1000):
            self.worker.tick(now)
            self.worker.tick(now+5000)
        self.assertEqual(len(self.stored()),1)
        self.assertEqual(self.stored()[0][0],'BTCUSDT')
        self.worker.notifier.send_signal.assert_called_once()
        self.assertEqual(self.worker.notifier.send_signal.call_args.args[0]['symbol'],'BTCUSDT')
        self.assertEqual(self.worker.state['last_decisions'][1]['paper_risk_status'],'paper_position_open')
        self.assertEqual(self.worker.predictor.predict.call_count,1)

    def test_overlap_is_blocked_but_next_entry_exactly_at_expiry_is_allowed(self):
        self.assertEqual(self.worker.admit(self.signal()),(True,'paper_risk_admitted'))
        self.assertEqual(self.worker.admit(self.signal(symbol='ETHUSDT')),(False,'paper_position_open'))
        next_signal=self.signal(self.slot+1800000,symbol='ETHUSDT')
        self.assertEqual(next_signal['entry_due_ms'],self.signal()['expiry_due_ms'])
        self.assertTrue(self.worker.admit(next_signal)[0])
        self.assertEqual(len(self.stored()),2)

    def test_unknown_outcomes_remain_reserved_instead_of_assuming_future_wins(self):
        for index in range(5):
            self.assertTrue(self.worker.admit(self.signal(self.slot+index*1800000))[0])
        # All positions would have expired by the next entry, but no official
        # observations were obtained: reserve all five possible losses (25 USDT).
        result=self.worker.admit(self.signal(self.slot+5*1800000))
        self.assertEqual(result,(False,'paper_daily_risk_budget'))
        self.assertTrue(all(row[2] is None for row in self.stored()))
        self.worker.settle_admission('BTCUSDT',self.slot,100,100)
        self.assertTrue(self.worker.admit(self.signal(self.slot+5*1800000))[0])

    def test_five_realized_losses_stop_today_and_next_day_budget_resets(self):
        for index in range(5):
            signal=self.signal(self.slot+index*1800000)
            self.assertTrue(self.worker.admit(signal)[0])
            self.worker.settle_admission(signal['symbol'],signal['signal_ms'],100,99)
        self.assertEqual(sum(row[3] for row in self.stored()),-25)
        self.assertEqual(self.worker.admit(self.signal(self.slot+5*1800000)),(False,'paper_daily_risk_budget'))
        self.assertTrue(self.worker.admit(self.signal(self.slot+86400000))[0])

    def test_long_short_win_loss_and_refund_settle_fixed_85pct_payout(self):
        cases=[(1,101,'win',4.25),(-1,99,'win',4.25),(1,99,'loss',-5),(-1,101,'loss',-5),(1,100,'refund',0)]
        for index,(direction,expiry,outcome,net) in enumerate(cases):
            with self.subTest(direction=direction,expiry=expiry):
                signal=self.signal(self.slot+index*1800000,direction=direction)
                self.assertTrue(self.worker.admit(signal)[0])
                self.worker.settle_admission(signal['symbol'],signal['signal_ms'],100,expiry)
                self.assertEqual(self.stored()[-1][2:],(outcome,net))

    def test_duplicate_admission_is_blocked_after_settlement_and_restart(self):
        signal=self.signal()
        self.assertTrue(self.worker.admit(signal)[0])
        self.worker.settle_admission(signal['symbol'],signal['signal_ms'],100,101)
        restarted=Worker.__new__(Worker)
        restarted.db=self.worker.db
        self.assertEqual(restarted.admit(signal),(False,'duplicate_paper_signal'))
        self.assertEqual(len(self.stored()),1)

    def test_bankroll_exhaustion_blocks_new_stake_independently_of_daily_budget(self):
        earlier=self.day-300*86400000
        with closing(sqlite3.connect(self.worker.db)) as con:
            con.executemany('INSERT INTO admissions VALUES (?,?,?,?,?,?,?)',
                            [('BTCUSDT',earlier+index*86400000,earlier+index*86400000+60000,
                              earlier+index*86400000+31*60000,1,'loss',-5) for index in range(200)])
            con.commit()
        self.assertEqual(self.worker.admit(self.signal()),(False,'paper_capital_insufficient'))

    def test_late_prediction_records_features_without_alert_or_admission(self):
        now=self.configure_tick([self.signal(),self.signal(symbol='ETHUSDT')])
        with patch('live.features.build_latest_features',return_value={'BTCUSDT':{},'ETHUSDT':{}}),patch('live.app.time.time',return_value=(self.slot+56000)/1000):
            self.worker.tick(now)
        self.assertEqual(self.stored(),[])
        self.worker.notifier.send_signal.assert_not_called()
        self.assertEqual(self.worker.history.record_features.call_count,2)
        with closing(sqlite3.connect(self.worker.db)) as con:
            self.assertEqual(con.execute('SELECT state FROM processed WHERE slot=?',(self.slot,)).fetchone()[0],'late_skipped')

    def test_non_halfhour_minute_is_evaluated_and_none_reason_is_durable(self):
        self.slot+=7*60000
        none={**self.signal(),'side':'NONE','direction':0,'reason':'model_score_below_0.56'}
        now=self.configure_tick([none,{**none,'symbol':'ETHUSDT'}])
        with patch('live.features.build_latest_features',return_value={'BTCUSDT':{},'ETHUSDT':{}}),patch('live.app.time.time',return_value=now/1000):
            self.worker.tick(now)
        stats=self.worker.scan_stats(now)
        self.assertEqual(stats['evaluated_slots'],1)
        self.assertEqual(stats['reasons'],{'model_score_below_0.56':2})
        self.assertEqual(stats['admitted'],0)
        self.worker.notifier.send_signal.assert_not_called()

    def test_settlement_outbox_survives_restart_and_is_queued_exactly_once(self):
        self.worker.admit(self.signal())
        self.worker.settle_admission('BTCUSDT',self.slot,100,101)
        restarted=Worker.__new__(Worker);restarted.db=self.worker.db
        restarted.notifier=SimpleNamespace(send_settlement=Mock(return_value={'state':'pending'}))
        restarted._flush_settlements()
        restarted._flush_settlements()
        restarted.notifier.send_settlement.assert_called_once()
        result=restarted.notifier.send_settlement.call_args.args[0]
        self.assertEqual(result['entry_price'],100)
        self.assertEqual(result['exit_price'],101)
        self.assertEqual(result['net_usdt'],4.25)
        self.assertEqual(restarted.paper_performance()['net_usdt'],4.25)
        self.assertEqual(restarted.paper_performance()['win_rate'],1)

    def test_old_settled_admission_is_backfilled_before_training_labels(self):
        self.worker.admit(self.signal())
        with closing(sqlite3.connect(self.worker.db)) as con:
            con.execute("UPDATE admissions SET outcome='loss',net=-5");con.commit()
        self.worker.lock=threading.RLock();self.worker.state={}
        order=[]
        self.worker.market=SimpleNamespace(index_open=Mock(side_effect=lambda symbol,stamp: order.append(stamp) or (100 if stamp==self.slot+60000 else 99)))
        self.worker.history=SimpleNamespace(pending_labels=Mock(side_effect=lambda *a,**k: order.append('history') or []))
        self.worker._settle_due(self.slot+40*60000)
        self.assertEqual(order,[self.slot+60000,self.slot+31*60000,'history'])
        self.assertEqual(self.worker.recent_settlements()[0]['net_usdt'],-5)

    def test_conflicting_settlement_cannot_replace_original_prices_or_pnl(self):
        self.worker.admit(self.signal())
        self.worker.settle_admission('BTCUSDT',self.slot,100,101)
        for entry,expiry in ((100,99),(101,102)):
            with self.assertRaises(ValueError):self.worker.settle_admission('BTCUSDT',self.slot,entry,expiry)
        self.assertEqual(self.worker.paper_performance()['net_usdt'],4.25)

    def test_disabled_delivery_leaves_durable_settlement_waiting(self):
        self.worker.admit(self.signal())
        self.worker.settle_admission('BTCUSDT',self.slot,100,100)
        self.worker.notifier=SimpleNamespace(send_settlement=Mock(return_value={'state':'disabled'}))
        self.worker._flush_settlements()
        with closing(sqlite3.connect(self.worker.db)) as con:
            self.assertEqual(con.execute('SELECT queued FROM settlement_outbox').fetchone()[0],0)

    def test_slow_maintenance_is_nonblocking_and_only_one_worker_runs(self):
        entered,release=threading.Event(),threading.Event()
        calls=[]
        self.worker.lock=threading.RLock();self.worker.state={}
        def blocked(now):
            calls.append(now);entered.set();release.wait(3)
        self.worker._maintenance_work=blocked
        started=time.monotonic()
        self.worker.start_maintenance(self.slot)
        self.assertLess(time.monotonic()-started,.5)
        try:
            self.assertTrue(entered.wait(1))
            self.worker.start_maintenance(self.slot+60000)
            self.assertEqual(calls,[self.slot])
        finally:
            release.set();self.worker.maintenance_thread.join(3)
        self.assertIsNone(self.worker.state['maintenance_error'])

    def test_monthly_refresh_starts_even_on_minute_boundary(self):
        self.worker.predictor=SimpleNamespace(manifest={'training_cutoff_ms':self.slot-31*86400000})
        self.worker.last_refresh_attempt=0
        self.worker._refresh_model=Mock()
        self.worker.maybe_refresh(self.slot)
        self.worker.refresh_thread.join(3)
        self.worker._refresh_model.assert_called_once_with(self.slot)

    def test_same_minute_expiry_stays_reserved_until_its_scan_completes(self):
        self.slot+=10*60000
        self.worker.admit(self.signal())
        self.worker.lock=threading.RLock();self.worker.state={};self.worker.last_label_poll=0
        self.worker.notifier=SimpleNamespace(retry_pending=Mock(),send_settlement=Mock(return_value={'state':'sent'}))
        self.worker.history=SimpleNamespace(pending_labels=Mock(return_value=[]))
        self.worker.market=SimpleNamespace(index_open=Mock(side_effect=[100,101]))
        expiry=self.slot+31*60000
        self.worker._maintenance_work(expiry+10000)
        self.assertEqual(self.worker.paper_performance()['pending'],1)
        self.worker.market.index_open.assert_not_called()
        # Completing this minute's scan releases its results for the NEXT scan,
        # without waiting an extra minute and missing its admission opportunity.
        self.worker._mark(expiry,'evaluated')
        self.worker._maintenance_work(expiry+15000)
        self.assertEqual(self.worker.paper_performance()['pending'],0)
        self.assertEqual(self.worker.paper_performance()['wins'],1)
        self.worker.notifier.send_settlement.assert_called_once()

    def test_older_maintenance_cutoff_never_settles_a_future_minute(self):
        self.slot+=10*60000
        self.worker.admit(self.signal())
        self.worker.lock=threading.RLock();self.worker.state={};self.worker.last_label_poll=0
        self.worker.notifier=SimpleNamespace(retry_pending=Mock(),send_settlement=Mock(return_value={'state':'sent'}))
        self.worker.history=SimpleNamespace(pending_labels=Mock(return_value=[]))
        self.worker.market=SimpleNamespace(index_open=Mock())
        expiry=self.slot+31*60000
        self.worker._mark(expiry-60000,'evaluated')
        self.worker._maintenance_work(expiry-1000)
        self.assertEqual(self.worker.paper_performance()['pending'],1)
        self.worker.market.index_open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
