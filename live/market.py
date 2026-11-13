"""Read-only public Binance observations for the alert worker."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import json
import math
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request,urlopen

import pandas as pd

SYMBOLS=('BTCUSDT','ETHUSDT')


class MarketDataError(RuntimeError):
    pass


class BinanceMarket:
    def __init__(self,opener=None):
        self.opener=opener or urlopen
        self._spot_cache={}
        self._index_cache=OrderedDict()
        self._cache_lock=threading.RLock()

    def _get(self,path,params,spot=False):
        hosts=('https://api.binance.com','https://data-api.binance.vision') if spot else ('https://fapi.binance.com','https://www.binance.com')
        for host in hosts:
            url=host+path+'?'+urlencode(params)
            try:
                with self.opener(Request(url,headers={'User-Agent':'CodeXQuant/1.0'}),timeout=8) as response:
                    payload=json.loads(response.read())
                if isinstance(payload,dict) and payload.get('code',0)<0:
                    continue
                return payload
            except (OSError,HTTPError,ValueError):
                continue
        raise MarketDataError('Official public market endpoints unavailable')

    def quote(self,symbol,now_ms=None):
        if symbol not in SYMBOLS:raise ValueError('Unsupported symbol')
        value=self._get('/fapi/v1/premiumIndex',{'symbol':symbol})
        try:
            price=float(value['indexPrice']);stamp=int(value['time'])
        except (KeyError,TypeError,ValueError):raise MarketDataError('Malformed official index quote') from None
        now_ms=int(time.time()*1000) if now_ms is None else now_ms
        if not math.isfinite(price) or price<=0 or abs(now_ms-stamp)>30000:
            raise MarketDataError('Stale or invalid official index quote')
        return {'index_price':price,'price_time_ms':stamp,'stale':False,'source':'Binance USD-M indexPrice'}

    def quotes(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            return dict(zip(SYMBOLS,pool.map(self.quote,SYMBOLS)))

    def _spot(self,symbol,signal_ms):
        if symbol not in SYMBOLS:raise ValueError('Unsupported symbol')
        params={'symbol':symbol,'interval':'1m','endTime':signal_ms-1,'limit':1000}
        with self._cache_lock:
            cached=self._spot_cache.get(symbol)
        # Keep 1,000 completed minutes but download only the new tail (plus two
        # overlap bars). A long interruption/backwards replay starts afresh.
        incremental=(cached is not None and 0<=signal_ms-int(cached.timestamp.iloc[-1])-60000<=300*60000)
        if incremental:
            params['startTime']=int(cached.timestamp.iloc[-1])-60000
        rows=self._get('/api/v3/klines',params,True)
        if not isinstance(rows,list) or not rows:raise MarketDataError('Insufficient official spot history')
        names=['timestamp','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
        frame=pd.DataFrame(rows,columns=names)
        cols=['timestamp','open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote']
        for name in cols:frame[name]=pd.to_numeric(frame[name],errors='raise')
        frame['timestamp']=frame.timestamp.astype('int64')
        frame=frame[cols]
        if incremental:frame=pd.concat([cached,frame],ignore_index=True).drop_duplicates('timestamp',keep='last')
        frame=frame.loc[frame.timestamp<signal_ms].sort_values('timestamp').tail(1000).reset_index(drop=True)
        if (len(frame)<600 or int(frame.timestamp.iloc[-1])!=signal_ms-60000
                or not frame.timestamp.diff().iloc[1:].eq(60000).all()):
            raise MarketDataError('Incomplete official spot minutes')
        with self._cache_lock:self._spot_cache[symbol]=frame.copy()
        return frame

    def _premium(self,symbol,signal_ms):
        rows=self._get('/fapi/v1/premiumIndexKlines',{'symbol':symbol,'interval':'1m','endTime':signal_ms-1,'limit':120})
        if not isinstance(rows,list) or len(rows)<60:raise MarketDataError('Insufficient official premium history')
        frame=pd.DataFrame([row[:5] for row in rows],columns=['timestamp','open','high','low','close'])
        for name in frame:frame[name]=pd.to_numeric(frame[name],errors='raise')
        frame['timestamp']=frame.timestamp.astype('int64')
        return frame

    def _funding(self,symbol,signal_ms):
        rows=self._get('/fapi/v1/fundingRate',{'symbol':symbol,'startTime':signal_ms-4*86400000,'endTime':signal_ms-60000,'limit':100})
        if not isinstance(rows,list) or len(rows)<3:raise MarketDataError('Insufficient published funding observations')
        if any(row.get('symbol')!=symbol for row in rows):raise MarketDataError('Funding symbol mismatch')
        return pd.DataFrame({'funding_time':[int(row['fundingTime']) for row in rows],
                             'rate':[float(row['fundingRate']) for row in rows]})

    def features_input(self,signal_ms):
        tasks=[(kind,symbol) for kind in ('spot','premium','funding') for symbol in SYMBOLS]
        def fetch(task):
            kind,symbol=task
            return kind,symbol,getattr(self,'_'+kind)(symbol,signal_ms)
        result={'spot':{},'premium':{},'funding':{}}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for kind,symbol,frame in pool.map(fetch,tasks):result[kind][symbol]=frame
        return result['spot'],result['premium'],result['funding']

    def index_open(self,symbol,timestamp_ms):
        if symbol not in SYMBOLS:raise ValueError('Unsupported symbol')
        if isinstance(timestamp_ms,bool) or int(timestamp_ms)!=timestamp_ms or timestamp_ms%60000:
            raise ValueError('Index open requires an exact minute timestamp')
        key=(symbol,int(timestamp_ms))
        with self._cache_lock:
            if key in self._index_cache:
                self._index_cache.move_to_end(key)
                return self._index_cache[key]
        rows=self._get('/fapi/v1/indexPriceKlines',{'pair':symbol,'interval':'1m','startTime':timestamp_ms,'endTime':timestamp_ms+59999,'limit':1})
        if not isinstance(rows,list) or not rows or int(rows[0][0])!=timestamp_ms:
            raise MarketDataError('Exact index minute observation unavailable')
        value=float(rows[0][1])
        if not math.isfinite(value) or value<=0:raise MarketDataError('Invalid index open')
        # Only immutable historical minute opens are cached. Do not retain a
        # speculative current/future minute response from a misbehaving source.
        if timestamp_ms+60000<=int(time.time()*1000):
            with self._cache_lock:
                self._index_cache[key]=value
                while len(self._index_cache)>2048:self._index_cache.popitem(last=False)
        return value
