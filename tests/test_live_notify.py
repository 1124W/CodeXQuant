"""No-network checks for durable, correctly labeled research notifications."""
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from urllib.error import URLError

from live.notify import (WeComNotifier, format_signal, format_price_report,
                         format_settlement,format_research_report,validate_webhook)


WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=fixture-not-a-real-key"
ORIGIN=1794441600000  # 2026-11-12 00:00 UTC / 08:00 CST


class Clock:
    def __init__(self):
        self.value=ORIGIN/1000
    def __call__(self):
        return self.value
    def advance(self,seconds):
        self.value+=seconds


class Response:
    def __init__(self,value,status=200):
        self.value,self.status=value,status
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def read(self,limit):
        return self.value if isinstance(self.value,bytes) else json.dumps(self.value).encode()


class Opener:
    def __init__(self,*responses):
        self.responses=list(responses) or [{"errcode":0}]
        self.calls=[]
    def open(self,req,timeout):
        self.calls.append((req,timeout))
        value=self.responses.pop(0) if len(self.responses)>1 else self.responses[0]
        if isinstance(value,Exception):
            raise value
        return value if isinstance(value,Response) else Response(value)


def signal(**overrides):
    return {"symbol":"BTCUSDT","side":"LONG","signal_ms":ORIGIN,
            "entry_due_ms":ORIGIN+60_000,"expiry_due_ms":ORIGIN+1_860_000,
            "p_up":.62,"confidence_score":.62,"index_price":64852.06123,
            "model_id":"both_regularized_logistic",**overrides}


def prices():
    return {"BTCUSDT":{"index_price":64852.06123,"price_time_ms":ORIGIN},
            "ETHUSDT":{"index_price":1922.1467,"price_time_ms":ORIGIN+1000}}


def settlement(**overrides):
    return {'symbol':'BTCUSDT','signal_ms':ORIGIN,'entry_ms':ORIGIN+60_000,
            'settlement_ms':ORIGIN+1_860_000,'direction':1,'entry_price':100.,
            'exit_price':101.,'outcome':'win','net_usdt':4.25,**overrides}


def report():
    return {'scan_stats':{'evaluated_slots':60,'late_slots':0,'candidates':2,'admitted':0,
                         'reasons':{'model_score_below_0.56':118},'risk_blocks':{'paper_position_open':2}},
            'paper_performance':{'settled':2,'pending':1,'wins':1,'losses':1,'refunds':0,
                                 'win_rate':.5,'net_usdt':-.75}}


class WeComFormattingTests(unittest.TestCase):
    def test_settlement_is_price_checked_and_explicit_paper_result(self):
        text=format_settlement(settlement())
        for token in ('BTC','做多','100.00','101.00','+4.25','前向模拟结算','85%'):
            self.assertIn(token,text)
        with self.assertRaises(ValueError):format_settlement(settlement(net_usdt=5))
        with self.assertRaises(ValueError):format_settlement(settlement(exit_price=99))
        with self.assertRaises(ValueError):format_settlement(settlement(direction=True))

    def test_hourly_report_distinguishes_no_signal_and_forward_performance(self):
        text=format_research_report(report(),ORIGIN)
        for token in ('60 轮','无新准入信号','已有模拟持仓 2次','50.00%','-0.75','不是历史回测胜率'):
            self.assertIn(token,text)
        no_results=report();no_results['paper_performance']['win_rate']=None
        self.assertIn('暂无已决样本',format_research_report(no_results,ORIGIN))

    def test_signal_explicit_asset_side_planned_30m_times_and_uncalibrated_score(self):
        text=format_signal(signal())
        for label in ("BTC", "看涨 / 做多", "2026-11-12 08:00:00", "08:01:00", "08:31:00",
                      "62.00%", "未经胜率校准", "64,852.06", "不自动下单"):
            self.assertIn(label,text)
        short=format_signal(signal(symbol="ETHUSDT",side="SHORT",p_up=.35,confidence_score=None))
        self.assertIn("ETH",short)
        self.assertIn("看跌 / 做空",short)
        self.assertIn("65.00%",short)

    def test_price_report_uses_real_quote_timestamps_and_two_decimals(self):
        text=format_price_report(prices(),ORIGIN)
        self.assertIn("BTC：64,852.06",text)
        self.assertIn("ETH：1,922.15",text)
        self.assertIn("08:00:01",text)
        self.assertIn("每小时整点",text)

    def test_untradeable_side_wrong_horizon_and_bad_numbers_are_rejected(self):
        for overrides in ({"symbol":"SOLUSDT"},{"side":"NONE"},{"entry_due_ms":ORIGIN},
                          {"expiry_due_ms":ORIGIN+600_000},{"confidence_score":1.2},
                          {"confidence_score":float("nan")},{"index_price":-1}):
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                format_signal(signal(**overrides))
        with self.assertRaises(ValueError):
            format_price_report(prices(),ORIGIN+60_000)
        with self.assertRaises(ValueError):
            format_price_report({"BTCUSDT":prices()["BTCUSDT"]},ORIGIN)

    def test_webhook_accepts_only_official_exact_https_endpoint_and_never_echoes_key(self):
        self.assertEqual(validate_webhook(WEBHOOK),WEBHOOK)
        self.assertIsNone(validate_webhook(""))
        for value in (WEBHOOK.replace("https:","http:"),WEBHOOK.replace("qyapi.weixin.qq.com","attacker.example"),
                      WEBHOOK+"&extra=1",WEBHOOK+"#x",WEBHOOK.replace("send?","other?"),
                      "https://user@qyapi.weixin.qq.com/cgi-bin/webhook/send?key=private-fixture"):
            with self.subTest(value=value),self.assertRaises(ValueError) as caught:
                validate_webhook(value)
            self.assertNotIn("fixture",str(caught.exception))


class WeComQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/"events.sqlite3"
        self.clock=Clock()
    def tearDown(self):
        self.temp.cleanup()
    def notifier(self,opener=None,url=WEBHOOK):
        return WeComNotifier(url,self.path,opener=opener or Opener(),clock=self.clock)

    def test_successful_http_ack_and_signal_dedup_survive_restart(self):
        opener=Opener()
        result=self.notifier(opener).send_signal(signal())
        self.assertTrue(result["sent"])
        self.assertEqual(result["errcode"],0)
        repeated=self.notifier(opener).send_signal(signal())
        self.assertTrue(repeated["duplicate"])
        self.assertFalse(repeated["sent"])
        self.assertEqual(len(opener.calls),1)
        outgoing,timeout=opener.calls[0]
        self.assertEqual(outgoing.get_method(),"POST")
        self.assertEqual(timeout,8)
        payload=json.loads(outgoing.data)
        self.assertEqual(payload["msgtype"],"markdown")
        self.assertIn("BTC",payload["markdown"]["content"])

    def test_settlement_retry_survives_signal_deadline_and_deduplicates_restart(self):
        opener=Opener(URLError('fixture'),{'errcode':0})
        self.clock.advance(4000)
        first=self.notifier(opener).send_settlement(settlement())
        self.assertEqual(first['state'],'pending')
        self.clock.advance(5)
        restarted=self.notifier(opener)
        self.assertTrue(restarted.retry_pending()[0]['sent'])
        self.assertTrue(restarted.send_settlement(settlement())['duplicate'])
        self.assertEqual(len(opener.calls),2)
        payload=json.loads(opener.calls[-1][0].data)
        self.assertIn('延迟补报',payload['markdown']['content'])
        self.assertEqual(restarted.status()['sent_by_kind'],{'settlement':1})

    def test_hourly_report_has_persistent_identity_and_short_expiry(self):
        opener=Opener()
        self.assertTrue(self.notifier(opener).send_research_report(report(),ORIGIN)['sent'])
        self.assertTrue(self.notifier(opener).send_research_report(report(),ORIGIN)['duplicate'])
        self.clock.advance(3720)
        self.assertEqual(self.notifier(opener).send_research_report(report(),ORIGIN+3_600_000)['state'],'expired')
        self.assertEqual(len(opener.calls),1)

    def test_same_time_other_asset_and_model_have_distinct_event_identity(self):
        opener=Opener()
        notifier=self.notifier(opener)
        for item in (signal(),signal(symbol="ETHUSDT"),signal(model_id="model-v2")):
            self.assertTrue(notifier.send_signal(item)["sent"])
        self.assertEqual(len(opener.calls),3)

    def test_whole_hour_price_dedup_across_new_quote_values(self):
        opener=Opener()
        notifier=self.notifier(opener)
        self.assertTrue(notifier.send_price_report(prices(),ORIGIN)["sent"])
        updated=prices();updated["BTCUSDT"]["index_price"]=70000
        self.assertTrue(notifier.send_price_report(updated,ORIGIN)["duplicate"])
        self.assertEqual(len(opener.calls),1)

    def test_api_error_is_not_success_and_retries_original_queued_payload_after_backoff(self):
        opener=Opener({"errcode":93000,"errmsg":WEBHOOK},{"errcode":0})
        notifier=self.notifier(opener)
        first=notifier.send_signal(signal())
        self.assertFalse(first["sent"])
        self.assertEqual(first["state"],"pending")
        self.assertEqual(first["errcode"],93000)
        self.assertNotIn("fixture",json.dumps(first))
        immediate=notifier.send_signal(signal(index_price=99999))
        self.assertEqual(immediate["state"],"backoff")
        self.assertEqual(len(opener.calls),1)
        self.clock.advance(5)
        self.assertTrue(notifier.retry_pending()[0]["sent"])
        self.assertEqual(opener.calls[0][0].data,opener.calls[1][0].data)
        self.assertEqual(notifier.status()["pending_count"],0)

    def test_timeout_is_explicitly_uncertain_without_secret_exception_leak(self):
        opener=Opener(URLError(WEBHOOK),{"errcode":0})
        notifier=self.notifier(opener)
        result=notifier.send_signal(signal())
        self.assertEqual(result["error"],"delivery_uncertain")
        self.assertTrue(result["delivery_uncertain"])
        self.assertNotIn("fixture",json.dumps(result)+json.dumps(notifier.status()))
        self.clock.advance(5)
        self.assertTrue(notifier.retry_pending()[0]["sent"])
        self.assertEqual(notifier.status()["delivery_uncertain_count"],1)

    def test_three_attempts_exhaust_bound_without_discarding_event(self):
        opener=Opener({"errcode":45009})
        notifier=self.notifier(opener)
        self.assertEqual(notifier.send_signal(signal())["state"],"pending")
        self.clock.advance(5)
        self.assertEqual(notifier.retry_pending()[0]["state"],"pending")
        self.clock.advance(20)
        self.assertEqual(notifier.retry_pending()[0]["state"],"exhausted")
        self.clock.advance(100)
        self.assertEqual(notifier.retry_pending(),[])
        self.assertEqual(notifier.send_signal(signal())["state"],"expired")
        self.assertEqual(len(opener.calls),3)
        with sqlite3.connect(self.path) as db:
            row=db.execute("SELECT status,content FROM wecom_events").fetchone()
        db.close()
        self.assertEqual(row[0],"expired")
        self.assertIn("BTC",row[1])

    def test_rate_limit_counts_attempts_in_shared_database_and_event_resumes(self):
        opener=Opener()
        notifier=self.notifier(opener)
        for number in range(20):
            self.assertTrue(notifier.send_status(f"fixture {number}",key=str(number))["sent"])
        limited=self.notifier(opener).send_status("twenty-first",key="21")
        self.assertEqual(limited["state"],"rate_limited")
        self.assertEqual(len(opener.calls),20)
        self.clock.advance(60.002)
        self.assertTrue(notifier.retry_pending()[0]["sent"])
        self.assertEqual(len(opener.calls),21)

    def test_concurrent_workers_lease_prevents_double_post(self):
        entered,release=threading.Event(),threading.Event()
        class Blocking(Opener):
            def open(inner,req,timeout):
                entered.set()
                release.wait(timeout=5)
                return super().open(req,timeout)
        opener=Blocking()
        first=self.notifier(opener)
        second=self.notifier(opener)
        results=[]
        worker=threading.Thread(target=lambda:results.append(first.send_signal(signal())))
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            duplicate=second.send_signal(signal())
            self.assertEqual(duplicate["state"],"in_flight")
            self.assertTrue(duplicate["duplicate"])
        finally:
            release.set();worker.join(timeout=5)
        self.assertTrue(results[0]["sent"])
        self.assertEqual(len(opener.calls),1)

    def test_expired_crash_lease_retries_and_marks_uncertain_delivery(self):
        opener=Opener()
        notifier=self.notifier(opener)
        notifier.send_signal(signal())
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE wecom_events SET status='sending',attempts=1,lease_until_ms=?,claim_token='crashed'",(ORIGIN-1,))
        db.close()
        self.assertTrue(notifier.retry_pending()[0]["sent"])
        self.assertEqual(notifier.status()["delivery_uncertain_count"],1)

    def test_crash_after_third_attempt_does_not_remain_pending_forever(self):
        notifier=self.notifier()
        notifier.send_signal(signal())
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE wecom_events SET status='sending',attempts=3,lease_until_ms=?,claim_token='crashed'",(ORIGIN-1,))
        db.close()
        self.assertEqual(notifier.retry_pending(),[])
        self.assertEqual(notifier.status()["pending_count"],0)
        self.assertEqual(notifier.status()["exhausted_count"],1)
        self.assertEqual(notifier.status()["delivery_uncertain_count"],1)

    def test_invalid_response_or_non200_never_mark_sent(self):
        for reply in (b"not JSON",{"errcode":False},{"errmsg":"missing code"},Response({"errcode":0},status=503)):
            with self.subTest(reply=reply):
                opener=Opener(reply)
                result=self.notifier(opener).send_status("fixture",key=str(id(reply)))
                self.assertFalse(result["sent"])
        self.assertEqual(self.notifier().status()["sent_count"],0)

    def test_stale_signal_is_never_transmitted_at_or_after_planned_entry(self):
        opener=Opener()
        notifier=self.notifier(opener)
        self.clock.advance(60)
        result=notifier.send_signal(signal())
        self.assertEqual(result["state"],"expired")
        self.assertEqual(opener.calls,[])
        self.assertEqual(notifier.status()["expired_count"],1)

    def test_pending_signal_cannot_replay_after_restart_and_deadline(self):
        opener=Opener(URLError("fixture network failure"))
        self.notifier(opener).send_signal(signal())
        self.clock.advance(61)
        restarted=self.notifier(opener)
        self.assertEqual(restarted.retry_pending(),[])
        self.assertEqual(len(opener.calls),1)
        self.assertEqual(restarted.status()["expired_count"],1)
        self.assertEqual(restarted.status()["pending_count"],0)

    def test_hourly_price_report_expires_after_two_minutes_without_network(self):
        opener=Opener()
        notifier=self.notifier(opener)
        self.clock.advance(120)
        self.assertEqual(notifier.send_price_report(prices(),ORIGIN)["state"],"expired")
        self.assertEqual(opener.calls,[])

    def test_pending_price_report_not_replayed_in_later_hours(self):
        opener=Opener(URLError("fixture unavailable"))
        notifier=self.notifier(opener)
        notifier.send_price_report(prices(),ORIGIN)
        self.clock.advance(3600)
        self.assertEqual(notifier.retry_pending(),[])
        self.assertEqual(len(opener.calls),1)
        self.assertEqual(notifier.status()["expired_count"],1)

    def test_disabled_notifier_never_opens_network_or_enqueues(self):
        opener=Opener()
        notifier=self.notifier(opener,url="")
        self.assertEqual(notifier.send_signal(signal())["state"],"disabled")
        self.assertEqual(notifier.retry_pending(),[])
        self.assertFalse(notifier.status()["enabled"])
        self.assertEqual(opener.calls,[])

    def test_status_redacts_webhook_and_size_bound_fails_before_network(self):
        opener=Opener()
        notifier=self.notifier(opener)
        notifier.send_status("配置 "+WEBHOOK,key="redaction")
        self.assertNotIn("fixture-not-a-real-key",opener.calls[0][0].data.decode())
        self.assertNotIn("fixture-not-a-real-key",json.dumps(notifier.status()))
        with self.assertRaises(ValueError):
            notifier.send_status("测"*1500)
        self.assertEqual(len(opener.calls),1)


if __name__=="__main__":
    unittest.main()
