"""Offline backfill tests: bulk pagination, causal vectors and delayed labels."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import backfill_consensus_history as backfill

MINUTE=60_000
ORIGIN=1_790_000_040_000
START=ORIGIN+1000*MINUTE
END=START+90*MINUTE


def inputs():
    count=1200;t=np.arange(count);result={key:{} for key in ('spot','premium','funding','index')}
    for position,symbol in enumerate(backfill.SYMBOLS):
        opened=(100+.003*t+.6*np.sin(t/17))*(1+position*.5)
        close=opened+.03*np.sin(t/7);volume=20+t%19+np.sin(t/6)
        result['spot'][symbol]=pd.DataFrame({'timestamp':ORIGIN+t*MINUTE,'open':opened,
            'high':np.maximum(opened,close)+.03,'low':np.minimum(opened,close)-.03,'close':close,
            'volume':volume,'quote_volume':volume*close,'trades':50+t%13,'taker_buy_base':volume*(.5+.2*np.sin(t/11))})
        value=.0001*np.sin(t/17+position)
        result['premium'][symbol]=pd.DataFrame({'timestamp':ORIGIN+t*MINUTE,'open':value-.000001,
            'high':value+.000002,'low':value-.000002,'close':value})
        funding_times=np.arange(ORIGIN-4*backfill.DAY_MS,END+backfill.DAY_MS,8*60*MINUTE,dtype=np.int64)
        result['funding'][symbol]=pd.DataFrame({'funding_time':funding_times,'rate':.0001+np.arange(len(funding_times))*.000001})
        result['index'][symbol]=pd.DataFrame({'timestamp':ORIGIN+t*MINUTE,'open':opened,'high':opened+.03,'low':opened-.03,'close':opened})
    return result


class BackfillTests(unittest.TestCase):
    def test_batch_matches_live_formula_and_labels_wait_until_five_seconds(self):
        records,audit=backfill.build_rows(inputs(),START,END,END+10*MINUTE+4999)
        self.assertEqual(len(records),180)
        self.assertEqual(audit['counts']['BTCUSDT'],{'feature_rows':90,'settled_rows':69,'pending_rows':21})
        self.assertTrue(audit['all_requested_minutes_present'])
        self.assertEqual(len(audit['parity_sample_signal_ms']),7)
        for symbol,signal,expiry,features,y,flat,entry,exit_price,source in records:
            self.assertEqual(len(json.loads(features)),60)
            if y is not None:
                self.assertLessEqual(expiry+5000,END+10*MINUTE+4999)
                self.assertEqual(source,'official_index_observation')
            else:self.assertIsNone(source)
        _,after=backfill.build_rows(inputs(),START,END,END+10*MINUTE+5000,samples=1)
        self.assertEqual(after['counts']['BTCUSDT']['settled_rows'],70)

    def test_future_index_changes_do_not_enter_features_or_pending_labels(self):
        current=END+10*MINUTE+4999;original=inputs()
        first,_=backfill.build_rows(original,START,END,current,samples=1)
        for frame in original['index'].values():frame.loc[frame.timestamp>current-5000,'open']=1e9
        second,_=backfill.build_rows(original,START,END,current,samples=1)
        self.assertEqual(first,second)

    def test_missing_premium_or_matured_index_observation_rejected(self):
        original=inputs();original['premium']['ETHUSDT']=original['premium']['ETHUSDT'].drop(index=1005)
        with self.assertRaises(ValueError):backfill.build_rows(original,START,END,END+60*MINUTE,samples=1)
        original=inputs();original['index']['BTCUSDT']=original['index']['BTCUSDT'].drop(index=1031)
        with self.assertRaisesRegex(ValueError,'index open is missing'):
            backfill.build_rows(original,START,END,END+60*MINUTE,samples=1)

    def test_batch_pagination_bounds_and_missing_minutes_fail(self):
        class Client:
            def __init__(self):self.calls=[]
            def request(self,kind,symbol,params,page):
                self.calls.append(params)
                stop=min(params['endTime']+1,params['startTime']+1000*MINUTE)
                return [[stamp,'100','101','99','100.5'] for stamp in range(params['startTime'],stop,MINUTE)]
        client=Client();frame=backfill.download_range(client,'index','BTCUSDT',START,START+1500*MINUTE)
        self.assertEqual(len(frame),1500);self.assertEqual(len(client.calls),2)
        self.assertEqual(client.calls[1]['startTime'],START+1000*MINUTE)
        self.assertEqual(client.calls[0]['pair'],'BTCUSDT')
        class Missing(Client):
            def request(self,*args):return super().request(*args)[1:]
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            backfill.download_range(Missing(),'index','BTCUSDT',START,START+30*MINUTE)

    def test_run_creates_only_new_bridge_with_readable_history_schema(self):
        class FakeClient:sources=[]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'bridge.sqlite3';audit=root/'audit.json'
            with patch.object(backfill,'fetch_inputs',return_value=inputs()):
                result=backfill.run(START,END,out,audit,now_ms=END+10*MINUTE+4999,client=FakeClient())
            with closing(sqlite3.connect(out)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM feature_history').fetchone()[0],180)
                self.assertEqual(db.execute('SELECT count(*) FROM feature_history WHERE y IS NULL').fetchone()[0],42)
            self.assertFalse(result['active_history_modified']);self.assertFalse(result['messages_sent'])
            with self.assertRaisesRegex(ValueError,'never overwritten'):
                backfill.run(START,END,out,audit,client=FakeClient())
            with self.assertRaisesRegex(ValueError,'active history'):
                backfill.run(START,END,root/'history-consensus.sqlite3',root/'another.json',client=FakeClient())

    def test_future_end_or_excessive_range_refused_before_requests(self):
        client=object()
        with self.assertRaises(ValueError):backfill.fetch_inputs(client,START,END,END-MINUTE)
        with self.assertRaises(ValueError):backfill.fetch_inputs(client,START,START+15*backfill.DAY_MS,START+20*backfill.DAY_MS)


if __name__=='__main__':unittest.main()
