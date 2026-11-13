"""Monthly feature cache parity, availability and corruption checks."""
from datetime import date,timedelta
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from live.features import FEATURES,SYMBOLS
from research.model_search_data import _ms,build_month,ensure_feature_store,load_window

MINUTE=60_000
START=date(2025,1,31)
END=date(2025,2,2)
ORIGIN=_ms(START)


def spot(symbol,start,end):
    times=np.arange(_ms(start),_ms(end),MINUTE,dtype=np.int64)
    t=(times-ORIGIN)/MINUTE
    level=100 if symbol=='BTCUSDT' else 20
    close=level+np.sin(t/11)*.4+np.sin(t/37)*.2+t*.0001
    opened=level+np.sin((t-1)/11)*.4+np.sin((t-1)/37)*.2+(t-1)*.0001
    volume=10+np.mod(t,11)+np.sin(t/5)
    return pd.DataFrame({'timestamp':times,'open':opened,'high':np.maximum(opened,close)+.05,
                         'low':np.minimum(opened,close)-.05,'close':close,'volume':volume,
                         'quote_volume':volume*close,'trades':20+np.mod(t,13),
                         'taker_buy_base':volume*(.5+.2*np.sin(t/7))})


def premium(symbol,start,end):
    times=np.arange(_ms(start),_ms(end),MINUTE,dtype=np.int64);t=(times-ORIGIN)/MINUTE
    close=np.sin(t/23)*.0001-.0002
    return pd.DataFrame({'timestamp':times,'open':close,'high':close+.00001,'low':close-.00001,'close':close})


def funding(symbol,start,end):
    times=np.arange(_ms(start),_ms(end),8*60*MINUTE,dtype=np.int64)
    return pd.DataFrame({'funding_time':times,'rate':np.sin(times/(8*60*MINUTE))*.0001})


def inputs(lower=START,upper=END):
    warmup=lower-timedelta(days=3)
    index={s:{'start_ms':ORIGIN,'opens':100+np.sin(np.arange(2880)/17)} for s in SYMBOLS}
    return ({s:spot(s,warmup,upper) for s in SYMBOLS},
            {s:premium(s,warmup,upper) for s in SYMBOLS},
            {s:funding(s,warmup,upper) for s in SYMBOLS},index)


class ModelSearchDataTests(unittest.TestCase):
    def test_month_chunks_preserve_features_and_cross_month_execution_labels(self):
        whole=build_month(*inputs(),ORIGIN,_ms(END),_ms(END))
        first=build_month(*inputs(START,date(2025,2,1)),ORIGIN,_ms(date(2025,2,1)),_ms(END))
        second=build_month(*inputs(date(2025,2,1),END),_ms(date(2025,2,1)),_ms(END),_ms(END))
        for symbol in SYMBOLS:
            self.assertEqual(len(first[symbol]),1440)
            self.assertEqual(len(second[symbol]),1440-31)
            combined=pd.concat([first[symbol],second[symbol]],ignore_index=True)
            np.testing.assert_allclose(whole[symbol][FEATURES],combined[FEATURES],rtol=1e-9,atol=1e-9)
            for name in ('signal_ms','entry_ms','settlement_ms','entry','exit','y','is_flat'):
                np.testing.assert_array_equal(whole[symbol][name],combined[name])
            self.assertGreater(first[symbol].settlement_ms.iloc[-1],_ms(date(2025,2,1)))
            self.assertLess(second[symbol].settlement_ms.max(),_ms(END))

    def test_future_market_or_execution_mutation_cannot_change_past_inputs(self):
        original=inputs();before=build_month(*original,ORIGIN,_ms(END),_ms(END))
        changed=inputs();cutoff=ORIGIN+200*MINUTE
        for symbol in SYMBOLS:
            mask=changed[0][symbol].timestamp>=cutoff
            changed[0][symbol].loc[mask,['open','high','low','close','quote_volume']]*=1.2
            mask=changed[1][symbol].timestamp>=cutoff
            changed[1][symbol].loc[mask,['open','high','low','close']]+=1
            changed[3][symbol]['opens'][200:]+=50
        after=build_month(*changed,ORIGIN,_ms(END),_ms(END))
        for symbol in SYMBOLS:
            a=before[symbol].loc[before[symbol].signal_ms<=cutoff]
            b=after[symbol].loc[after[symbol].signal_ms<=cutoff]
            np.testing.assert_allclose(a[FEATURES],b[FEATURES],rtol=0,atol=0)
            self.assertFalse(np.array_equal(a.exit,b.exit))

    def test_missing_premium_minute_removes_exact_sixty_signal_windows_without_fill(self):
        incoming=inputs();before=build_month(*incoming,ORIGIN,_ms(END),_ms(END))
        missing=ORIGIN+100*MINUTE
        incoming[1]['BTCUSDT']=incoming[1]['BTCUSDT'].loc[incoming[1]['BTCUSDT'].timestamp!=missing]
        after=build_month(*incoming,ORIGIN,_ms(END),_ms(END))
        excluded=np.setdiff1d(before['BTCUSDT'].signal_ms,after['BTCUSDT'].signal_ms)
        np.testing.assert_array_equal(excluded,np.arange(missing+MINUTE,missing+61*MINUTE,MINUTE))
        self.assertEqual(len(before['ETHUSDT']),len(after['ETHUSDT']))

    def test_numeric_cache_reuses_verified_files_and_rejects_corruption(self):
        calls=[]
        def kind(name,symbol,start,end,cache,offline):
            self.assertTrue(offline);calls.append((name,symbol,start,end))
            if name=='spot':frame=spot(symbol,start,end)
            else:
                times=np.arange(_ms(start),_ms(end),MINUTE,dtype=np.int64)
                frame=pd.DataFrame({'timestamp':times,'open':100+np.sin((times-ORIGIN)/MINUTE/17)})
            return frame,[{'kind':name,'sha256':'test-observation-sha'}]
        def p(symbol,start,end,cache,offline,allow_gaps):
            self.assertTrue(offline and allow_gaps)
            return premium(symbol,start,end),[]
        def f(symbol,start,end,cache,offline):
            self.assertTrue(offline)
            return funding(symbol,start,end),[]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch('research.model_search_data.load_kind',side_effect=kind),patch('research.model_search_data.load_premium',side_effect=p),patch('research.model_search_data.load_funding',side_effect=f):
                manifest=ensure_feature_store(start=START,end=END,store_dir=folder,progress=None)
                count=len(calls)
                again=ensure_feature_store(start=START,end=END,store_dir=folder,progress=None)
            self.assertTrue(manifest['complete'])
            self.assertEqual(len(calls),count)
            self.assertEqual(manifest,again)
            self.assertEqual(sum(c[0]=='index' for c in calls),2)
            boundary=_ms(date(2025,2,1))
            frame=load_window('BTCUSDT',boundary-10*MINUTE,boundary+10*MINUTE,store_dir=folder)
            self.assertEqual(len(frame),20)
            self.assertEqual(frame.attrs['features'],FEATURES)
            self.assertTrue((frame.entry_ms==frame.signal_ms+MINUTE).all())
            entry=manifest['files']['BTCUSDT:2025-01']
            with (folder/entry['file']).open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_window('BTCUSDT',boundary-10*MINUTE,boundary,store_dir=folder)


if __name__=='__main__':unittest.main()
