"""No-network stake migration, exact payouts and independent research notices."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock,patch

from live.app import Worker
from live.notify import WeComNotifier,format_signal,format_settlement,format_research_report
from live.stakes import PaperSettings,amount_cents,stake_cents,settlement_cents

ORIGIN=1794441600000
MINUTE=60000


def signal(stamp=ORIGIN,**values):
    return {'symbol':'BTCUSDT','signal_ms':stamp,'entry_due_ms':stamp+MINUTE,
            'expiry_due_ms':stamp+31*MINUTE,'direction':1,'side':'LONG',
            'confidence_score':.62,**values}


class PaperStakeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'worker.sqlite3'
        with closing(sqlite3.connect(self.path)) as con:
            con.execute('''CREATE TABLE admissions (symbol TEXT,signal_ms INTEGER,entry_ms INTEGER,
                settlement_ms INTEGER,direction INTEGER,outcome TEXT,net REAL,PRIMARY KEY(symbol,signal_ms))''')
            con.commit()

    def worker(self,stake=30):
        worker=Worker.__new__(Worker);worker.db=self.path
        worker.paper_settings=PaperSettings.for_stake(stake)
        worker._ensure_tables();return worker

    def legacy(self,stamp=ORIGIN,outcome=None,net=None):
        with closing(sqlite3.connect(self.path)) as con:
            con.execute('INSERT INTO admissions VALUES (?,?,?,?,?,?,?)',
                        ('BTCUSDT',stamp,stamp+MINUTE,stamp+31*MINUTE,1,outcome,net));con.commit()

    def test_new_default_is_30_and_invalid_settings_fail_closed(self):
        settings=PaperSettings.from_env({})
        self.assertEqual(settings.stake_cents,3000)
        self.assertEqual(settings.daily_limit_cents,15000)
        self.assertEqual(settings.initial_capital_cents,100000)
        for value in (True,None,'',float('nan'),float('inf'),'4.99','250.01','30.001','30.000'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                PaperSettings.from_env({'CODEXQUANT_PAPER_STAKE':value})
        self.assertEqual(stake_cents('5'),500)
        self.assertEqual(stake_cents('250.00'),25000)
        self.assertEqual(PaperSettings.for_stake('30.01').daily_limit_cents,15005)
        self.assertEqual(PaperSettings.from_env({'CODEXQUANT_PAPER_STAKE':'30','CODEXQUANT_PAPER_DAILY_LIMIT':'25'}).daily_limit_cents,2500)

    def test_exact_half_up_payout_and_loss_have_no_float_drift(self):
        self.assertEqual(settlement_cents(3000,'win'),2550)
        self.assertEqual(settlement_cents(510,'win'),434)
        self.assertEqual(settlement_cents(501,'win'),426)
        self.assertEqual(settlement_cents(3000,'loss'),-3000)
        self.assertEqual(settlement_cents(3000,'refund'),0)
        self.assertEqual(amount_cents('-0.75'),-75)
        with self.assertRaises(ValueError):settlement_cents(3000,'win',8000)

    def test_migration_keeps_pending_legacy_5_and_new_30_terms_after_restart(self):
        self.legacy()
        worker=self.worker(30)
        with closing(sqlite3.connect(self.path)) as con:
            self.assertEqual(len(con.execute('PRAGMA table_info(admissions)').fetchall()),7)
            self.assertEqual(con.execute('SELECT stake_cents,payout_bps FROM admission_terms').fetchone(),(500,8500))
        worker.settle_admission('BTCUSDT',ORIGIN,100,101)
        next_signal=signal(ORIGIN+60*MINUTE)
        self.assertTrue(worker.admit(next_signal)[0])
        self.assertEqual(next_signal['stake_usdt'],30)
        self.assertEqual(next_signal['win_net_usdt'],25.5)
        restarted=self.worker(250)
        restarted.settle_admission('BTCUSDT',next_signal['signal_ms'],100,101)
        history=restarted.paper_performance()
        self.assertEqual(history['net_usdt'],29.75)
        self.assertEqual(history['balance_usdt'],1029.75)
        self.assertEqual(history['stake_usdt'],250)
        self.assertTrue(history['mixed_stakes'])
        self.assertEqual(history['stake_history'],[{'stake_usdt':5.,'trades':1},{'stake_usdt':30.,'trades':1}])
        rows=restarted.recent_settlements()
        self.assertEqual(rows[0]['stake_usdt'],30)
        self.assertEqual(rows[0]['net_usdt'],25.5)
        self.assertEqual(rows[1]['net_usdt'],4.25)

    def test_old_outbox_payload_and_queued_status_are_not_rewritten(self):
        self.legacy(outcome='win',net=4.25)
        worker=self.worker()
        payload={'symbol':'BTCUSDT','signal_ms':ORIGIN,'entry_ms':ORIGIN+MINUTE,
                 'settlement_ms':ORIGIN+31*MINUTE,'direction':1,'entry_price':100.,
                 'exit_price':101.,'outcome':'win','net_usdt':4.25}
        original=json.dumps(payload,separators=(',',':'))
        with closing(sqlite3.connect(self.path)) as con:
            con.execute('INSERT INTO settlement_outbox VALUES (?,?,?,1)',('BTCUSDT',ORIGIN,original));con.commit()
        restarted=self.worker(250)
        restarted.settle_admission('BTCUSDT',ORIGIN,100,101)
        with closing(sqlite3.connect(self.path)) as con:
            self.assertEqual(con.execute('SELECT payload,queued FROM settlement_outbox').fetchone(),(original,1))
        self.assertIn('+4.25',format_settlement(payload))
        self.assertIn('5.00 USDT',format_settlement(payload))

    def test_daily_risk_sums_each_actual_amount_including_legacy_pending(self):
        self.legacy(ORIGIN,outcome='loss',net=-5)
        worker=self.worker(30)
        for index in range(1,5):
            row=signal(ORIGIN+index*60*MINUTE)
            self.assertTrue(worker.admit(row)[0])
            worker.settle_admission('BTCUSDT',row['signal_ms'],100,99)
        self.assertEqual(worker.paper_performance()['net_usdt'],-125)
        self.assertEqual(worker.admit(signal(ORIGIN+5*60*MINUTE)),(False,'paper_daily_risk_budget'))
        self.assertTrue(worker.admit(signal(ORIGIN+24*60*MINUTE))[0])

    def test_legacy_delayed_queue_keeps_original_content_without_conflict(self):
        self.legacy(outcome='win',net=4.25)
        worker=self.worker(250)
        payload={'symbol':'BTCUSDT','signal_ms':ORIGIN,'entry_ms':ORIGIN+MINUTE,
                 'settlement_ms':ORIGIN+31*MINUTE,'direction':1,'entry_price':100.,
                 'exit_price':101.,'outcome':'win','net_usdt':4.25}
        original=json.dumps(payload,separators=(',',':'))
        with closing(sqlite3.connect(self.path)) as con:
            con.execute('INSERT INTO settlement_outbox VALUES (?,?,?,0)',('BTCUSDT',ORIGIN,original));con.commit()
        now=ORIGIN+60*MINUTE
        notifier=WeComNotifier('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test-only',
                              Path(self.temp.name)/'notifications.sqlite3',opener=Mock(),clock=lambda:now/1000)
        # The old formatter used this literal text. A restart must deliver the
        # durable old queue, not replace it with a new-stake or reformatted copy.
        content=format_settlement({**payload,'delayed_report':True}).replace('本笔固定本金：','固定本金：')
        with notifier._db() as con:
            con.execute('''INSERT INTO wecom_events(event_key,kind,content,status,created_ms,updated_ms)
                VALUES(?,? ,?,'pending',?,?)''',(f'settlement:BTCUSDT:{ORIGIN}','settlement',content,now,now))
        worker.notifier=notifier
        worker.settle_admission('BTCUSDT',ORIGIN,100,101)
        with patch.object(notifier,'_post',return_value={'sent':True,'errcode':0}) as posting:
            worker._flush_settlements();worker._flush_settlements()
            posting.assert_called_once_with(content)
        with closing(sqlite3.connect(self.path)) as con:
            self.assertEqual(con.execute('SELECT payload,queued FROM settlement_outbox').fetchone(),(original,1))
        with notifier._db() as con:
            saved=con.execute('SELECT content,status,attempts FROM wecom_events').fetchone()
            self.assertEqual(tuple(saved),(content,'sent',1))

    def test_unsettled_30_remains_reserved_after_amount_changes(self):
        worker=self.worker(30);first=signal()
        self.assertTrue(worker.admit(first)[0])
        restarted=self.worker(5)
        # The nominal position has expired, but its 30U downside is still unknown.
        self.assertEqual(restarted.admit(signal(ORIGIN+60*MINUTE)),(False,'paper_daily_risk_budget'))
        restarted.settle_admission('BTCUSDT',ORIGIN,100,100)
        self.assertTrue(restarted.admit(signal(ORIGIN+60*MINUTE))[0])

    def test_publication_hold_blocks_new_admission_but_keeps_old_settlement(self):
        self.legacy();worker=self.worker(30)
        guard=Path(self.temp.name)/'publication.hold';guard.write_text('hold',encoding='ascii')
        worker.publication_hold_file=guard
        self.assertTrue(worker.publication_held())
        self.assertEqual(worker.admit(signal(ORIGIN+60*MINUTE)),(False,'research_publication_hold'))
        worker.settle_admission('BTCUSDT',ORIGIN,100,101)
        self.assertEqual(worker.paper_performance()['net_usdt'],4.25)
        guard.unlink()
        self.assertFalse(worker.publication_held())
        self.assertTrue(worker.admit(signal(ORIGIN+60*MINUTE))[0])

    def test_notifications_use_admission_amount_and_separate_historical_comparison(self):
        worker=self.worker();row=signal();worker.admit(row)
        content=format_signal(row)
        for text in ('30.00 USDT','+25.50 USDT','-30.00 USDT','150.00 USDT'):
            self.assertIn(text,content)
        worker.settle_admission('BTCUSDT',ORIGIN,100,101)
        result=worker.recent_settlements()[0]
        self.assertIn('+25.50',format_settlement(result))
        bad={**result,'net_usdt':4.25}
        with self.assertRaises(ValueError):format_settlement(bad)
        report={'scan_stats':{'evaluated_slots':60,'candidates':0,'admitted':0,'late_slots':0},
                'paper_performance':worker.paper_performance(),
                'historical_research':{'start_utc':'2026-05-12','end_exclusive_utc':'2026-11-12',
                    'signals':338,'win_rate':203/338,'net_usdt':1126.5,'stake_usdt':30,
                    'initial_capital_usdt':10000,'url':'http://example.com/demo/recent.html'}}
        content=format_research_report(report,ORIGIN)
        for text in ('1,000.00 USDT','10,000.00 USDT','+25.50','+1126.50','非当前线上模型','不并入上方'):
            self.assertIn(text,content)
        self.assertEqual(report['paper_performance']['net_usdt'],25.5)

    def test_missing_or_malformed_historical_artifact_does_not_fail_live_status(self):
        worker=self.worker();folder=Path(self.temp.name)/'empty'
        with patch('live.app.ROOT',folder):
            self.assertIsNone(worker.historical_research())
        folder.mkdir();(folder/'research').mkdir()
        (folder/'research'/'recent-public.json').write_text('{broken',encoding='utf-8')
        worker.historical_research_loaded_at=float('-inf')
        with patch('live.app.ROOT',folder):
            self.assertIsNone(worker.historical_research())

    def test_valid_historical_artifact_is_separate_cached_and_requires_provenance(self):
        worker=self.worker()
        data={'schema_version':1,
              'assumptions':{'default_stake':30,'initial_capital':10000,'net_payout':.85,'fee':0},
              'strategy':{'label':'Historical pair','rule_id':'historical_pair',
                          'is_live_strategy':False,'posthoc':True,'forward_validated':False},
              'published':{'recent_six_months':{'stats':{'signals':338,'win_rate':203/338,'net_usdt':1126.5}}},
              'periods':[{'id':'recent_six_months','start_utc':'2026-05-12','end_exclusive_utc':'2026-11-12'}]}
        folder=Path(self.temp.name)/'publication';(folder/'research').mkdir(parents=True)
        artifact=folder/'research'/'recent-public.json'
        artifact.write_text(json.dumps(data),encoding='utf-8')
        with patch('live.app.ROOT',folder),patch.dict('os.environ',{'CODEXQUANT_PUBLIC_BASE_URL':'https://example.com/project/'}):
            historical=worker.historical_research()
            self.assertEqual(historical['net_usdt'],1126.5)
            self.assertEqual(historical['url'],'https://example.com/project/demo/recent.html')
            self.assertFalse(historical['is_live_strategy'])
            self.assertTrue(historical['posthoc'])
            self.assertEqual(worker.paper_performance()['net_usdt'],0)
            data['strategy']['is_live_strategy']=True
            artifact.write_text(json.dumps(data),encoding='utf-8')
            self.assertEqual(worker.historical_research(),historical)
            worker.historical_research_loaded_at=float('-inf')
            self.assertIsNone(worker.historical_research())


if __name__=='__main__':unittest.main()
