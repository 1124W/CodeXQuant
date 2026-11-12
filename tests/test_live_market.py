"""Official public observations, request-time isolation and exact index execution."""
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs,urlsplit

from live.market import BinanceMarket,MarketDataError,SYMBOLS


class Response:
    def __init__(self,payload):
        self.payload=json.dumps(payload).encode('utf-8')
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False
    def read(self):
        return self.payload


class LiveMarketTests(unittest.TestCase):
    def test_quote_uses_index_price_instead_of_mark_or_other_prices(self):
        requests=[]
        def opener(request,timeout):
            requests.append(request.full_url)
            self.assertEqual(timeout,8)
            return Response({'indexPrice':'65000.125','markPrice':'99999','lastPrice':'1','time':1794450000000})
        quote=BinanceMarket(opener).quote('BTCUSDT',1794450001000)
        self.assertEqual(quote['index_price'],65000.125)
        self.assertEqual(quote['price_time_ms'],1794450000000)
        self.assertFalse(quote['stale'])
        self.assertEqual(urlsplit(requests[0]).path,'/fapi/v1/premiumIndex')
        self.assertEqual(parse_qs(urlsplit(requests[0]).query)['symbol'],['BTCUSDT'])

    def test_quote_rejects_stale_future_nonfinite_nonpositive_and_malformed_observations(self):
        now=1794450000000
        for value in ({'indexPrice':'65000','time':now-30001},{'indexPrice':'65000','time':now+30001},
                      {'indexPrice':'NaN','time':now},{'indexPrice':'0','time':now},
                      {'indexPrice':'-1','time':now},{'markPrice':'65000','time':now}):
            with self.subTest(value=value):
                with self.assertRaises(MarketDataError):
                    BinanceMarket(lambda *a,**k:Response(value)).quote('ETHUSDT',now)
        with self.assertRaises(ValueError):
            BinanceMarket(lambda *a,**k:Response({})).quote('OTHER',now)

    def test_futures_endpoint_falls_back_only_to_public_official_website(self):
        calls=[]
        def opener(request,timeout):
            calls.append(request.full_url)
            if urlsplit(request.full_url).hostname=='fapi.binance.com':
                raise HTTPError(request.full_url,451,'unavailable',{},None)
            return Response({'indexPrice':'2000.25','time':1794450000000})
        quote=BinanceMarket(opener).quote('ETHUSDT',1794450000000)
        self.assertEqual(quote['index_price'],2000.25)
        self.assertEqual([urlsplit(url).hostname for url in calls],['fapi.binance.com','www.binance.com'])

    def test_all_official_failures_raise_without_inventing_a_quote(self):
        calls=[]
        def unavailable(request,timeout):
            calls.append(request.full_url)
            raise OSError('offline')
        with self.assertRaises(MarketDataError):
            BinanceMarket(unavailable).quote('BTCUSDT',1794450000000)
        self.assertEqual(len(calls),2)
        market=BinanceMarket(lambda *a,**k:Response({'code':-1000,'msg':'not available'}))
        with self.assertRaises(MarketDataError):
            market.quote('BTCUSDT',1794450000000)

    def test_spot_requests_only_minutes_closed_before_signal_and_normalizes_columns(self):
        signal=1794450000000
        rows=[[signal-(600-index)*60000,'100','101','99','100.5','10',signal-(600-index)*60000+59999,'1000',20,'6','600','0'] for index in range(600)]
        market=BinanceMarket()
        with patch.object(market,'_get',return_value=rows) as get:
            frame=market._spot('BTCUSDT',signal)
        args=get.call_args.args
        self.assertEqual(args[0],'/api/v3/klines')
        self.assertEqual(args[1]['endTime'],signal-1)
        self.assertEqual(args[1]['interval'],'1m')
        self.assertTrue(args[2])
        self.assertEqual(len(frame),600)
        self.assertEqual(frame.timestamp.iloc[-1],signal-60000)
        self.assertIn('taker_buy_base',frame)
        self.assertEqual(frame.taker_buy_base.iloc[-1],6)
        with patch.object(market,'_get',return_value=rows[:500]):
            with self.assertRaises(MarketDataError):
                market._spot('BTCUSDT',signal)

    def test_premium_signed_values_are_preserved_and_current_unclosed_minute_not_requested(self):
        signal=1794450000000
        rows=[[signal-(60-index)*60000,-.001,0,-.002,-.0005,0] for index in range(60)]
        market=BinanceMarket()
        with patch.object(market,'_get',return_value=rows) as get:
            frame=market._premium('ETHUSDT',signal)
        self.assertEqual(get.call_args.args[1]['endTime'],signal-1)
        self.assertEqual(frame.close.iloc[-1],-.0005)
        self.assertEqual(frame.timestamp.iloc[-1],signal-60000)

    def test_funding_request_respects_one_minute_publication_lag_and_original_milliseconds(self):
        signal=1794450000000
        events=[{'symbol':'BTCUSDT','fundingTime':signal-gap+1,'fundingRate':'.0001'} for gap in (3*8*3600000,2*8*3600000,8*3600000)]
        market=BinanceMarket()
        with patch.object(market,'_get',return_value=events) as get:
            frame=market._funding('BTCUSDT',signal)
        self.assertEqual(get.call_args.args[1]['endTime'],signal-60000)
        self.assertEqual(get.call_args.args[1]['startTime'],signal-4*86400000)
        self.assertEqual(frame.funding_time.iloc[0],events[0]['fundingTime'])
        self.assertEqual(frame.rate.iloc[-1],.0001)
        self.assertNotIn('interval_hours',frame)
        wrong=[{**row,'symbol':'ETHUSDT'} for row in events]
        with patch.object(market,'_get',return_value=wrong):
            with self.assertRaises(MarketDataError):
                market._funding('BTCUSDT',signal)

    def test_index_execution_requires_exact_requested_open_timestamp_and_valid_open(self):
        market=BinanceMarket()
        stamp=1794450000000
        with patch.object(market,'_get',return_value=[[stamp,'65000.125','65010','64990','65005']]) as get:
            self.assertEqual(market.index_open('BTCUSDT',stamp),65000.125)
        params=get.call_args.args[1]
        self.assertEqual(params,{'pair':'BTCUSDT','interval':'1m','startTime':stamp,'endTime':stamp+59999,'limit':1})
        for rows in ([],[[stamp-60000,'65000']],[[stamp,'NaN']],[[stamp,'0']]):
            with self.subTest(rows=rows):
                with patch.object(market,'_get',return_value=rows):
                    with self.assertRaises(MarketDataError):
                        market.index_open('BTCUSDT',stamp)


if __name__=='__main__':
    unittest.main()
