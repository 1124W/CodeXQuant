"""Causal feature parity, frozen alert gates and local-bundle integrity."""
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from live.features import (BASE_FEATURES,PAIR_FEATURES,FEATURES,MINUTE_MS,SYMBOLS,
                            base_frame,build_latest_features)
from live.predictor import Predictor,decision,RULE_ID,DAY_MS,retrain_from_rows


def market(count=6000):
    origin=int(pd.Timestamp('2026-11-01',tz='UTC').timestamp()*1000)
    t=np.arange(count)
    spot={};premium={};funding={}
    for position,symbol in enumerate(SYMBOLS):
        open_price=(100+.003*t+.6*np.sin(t/17))*(1+position*.5)
        close=open_price+.03*np.sin(t/7)
        volume=20+t%19+np.sin(t/6)
        spot[symbol]=pd.DataFrame({"timestamp":origin+t*MINUTE_MS,"open":open_price,
            "high":np.maximum(open_price,close)+.03,"low":np.minimum(open_price,close)-.03,
            "close":close,"volume":volume,"quote_volume":volume*close,
            "trades":50+t%13,"taker_buy_base":volume*(.5+.2*np.sin(t/11)),
            "taker_buy_quote":volume*close*(.5+.2*np.sin(t/11))})
        v=.0001*np.sin(t/17+position)
        premium[symbol]=pd.DataFrame({"timestamp":origin+t*MINUTE_MS,
            "open":v-.000001,"high":v+.000002,"low":v-.000002,"close":v})
        ft=np.arange(-24,145,8,dtype=np.int64)
        funding[symbol]=pd.DataFrame({"funding_time":origin+ft*3_600_000+1,
            "rate":.0001*np.sin(ft/24+position),"interval_hours":8.})
    return spot,premium,funding,origin+5700*MINUTE_MS


class LiveFeatureTests(unittest.TestCase):
    def test_exact_60_feature_names_and_no_target_expression(self):
        spot,premium,funding,signal=market()
        rows=build_latest_features(spot,premium,funding,signal)
        self.assertEqual(len(FEATURES),60)
        self.assertEqual(len(set(FEATURES)),60)
        for symbol in SYMBOLS:
            self.assertEqual(list(rows[symbol]),FEATURES)
            self.assertTrue(np.isfinite(list(rows[symbol].values())).all())
        frame=base_frame(spot['BTCUSDT'])
        self.assertEqual(list(frame),['timestamp']+BASE_FEATURES)
        self.assertFalse(set(frame)&{'y','is_flat','fwd_bp','entry','exit'})

    def test_future_spot_premium_funding_and_target_mutations_do_not_change_past(self):
        spot,premium,funding,signal=market()
        baseline=build_latest_features(spot,premium,funding,signal)
        changed=[copy.deepcopy(x) for x in (spot,premium,funding)]
        for symbol in SYMBOLS:
            future=changed[0][symbol].timestamp>=signal
            changed[0][symbol].loc[future,['open','high','low','close']]*=9
            changed[0][symbol]['exit']=0
            changed[0][symbol]['y']=1
            changed[1][symbol].loc[changed[1][symbol].timestamp>=signal,['open','high','low','close']]*=12
            changed[2][symbol].loc[changed[2][symbol].funding_time+MINUTE_MS>signal,'rate']=.9
        again=build_latest_features(*changed,signal)
        self.assertEqual(baseline,again)
        prefix={s:spot[s][spot[s].timestamp<signal].copy() for s in SYMBOLS}
        self.assertEqual(baseline,build_latest_features(prefix,premium,funding,signal))

    def test_missing_or_stale_closed_observations_fail_without_filling(self):
        spot,premium,funding,signal=market()
        missing=copy.deepcopy(spot)
        missing['BTCUSDT']=missing['BTCUSDT'][missing['BTCUSDT'].timestamp!=signal-MINUTE_MS]
        with self.assertRaises(ValueError):build_latest_features(missing,premium,funding,signal)
        bad=copy.deepcopy(premium)
        bad['ETHUSDT']=bad['ETHUSDT'][bad['ETHUSDT'].timestamp!=signal-15*MINUTE_MS]
        with self.assertRaisesRegex(ValueError,'60 premium'):build_latest_features(spot,bad,funding,signal)
        old={s:f[f.funding_time<signal-25*3_600_000].copy() for s,f in funding.items()}
        with self.assertRaises(ValueError):build_latest_features(spot,premium,old,signal)

    def test_funding_millisecond_publication_lag_is_respected(self):
        spot,premium,funding,signal=market()
        baseline=build_latest_features(spot,premium,funding,signal)
        changed=copy.deepcopy(funding)
        for symbol in SYMBOLS:
            extra=pd.DataFrame({'funding_time':[signal-MINUTE_MS+1],'rate':[.5],'interval_hours':[1]})
            changed[symbol]=pd.concat([changed[symbol],extra],ignore_index=True)
        self.assertEqual(baseline,build_latest_features(spot,premium,changed,signal))
        for symbol in SYMBOLS:
            changed[symbol].loc[changed[symbol].funding_time==signal-MINUTE_MS+1,'funding_time']=signal-MINUTE_MS
        after=build_latest_features(spot,premium,changed,signal)
        self.assertEqual(after['BTCUSDT']['funding_rate_bps'],5000)

    def test_bad_symbol_boundary_duplicate_and_short_history_rejected(self):
        spot,premium,funding,signal=market()
        with self.assertRaises(ValueError):build_latest_features(spot,premium,funding,signal+MINUTE_MS)
        with self.assertRaises(ValueError):build_latest_features({'BTCUSDT':spot['BTCUSDT']},premium,funding,signal)
        short={s:d[(d.timestamp<signal)&(d.timestamp>=signal-500*MINUTE_MS)] for s,d in spot.items()}
        with self.assertRaises(ValueError):build_latest_features(short,premium,funding,signal)
        dup=copy.deepcopy(spot)
        dup['BTCUSDT']=pd.concat([dup['BTCUSDT'],dup['BTCUSDT'].iloc[[100]]])
        with self.assertRaises(ValueError):build_latest_features(dup,premium,funding,signal)

    @unittest.skipUnless(Path('external/events-workBuddy/datahub/features.py').exists() and Path('external/QuantDinger-repo/.git').exists(), 'Pinned private research sources absent; serving does not require them')
    def test_full_60_feature_parity_against_historical_pipeline(self):
        from research.workbuddy_strategy import build_samples
        from research.event_regime_features import enrich_samples
        from research.derivative_features import add_derivative_features
        spot,premium,funding,signal=market()
        historical={}
        for symbol in SYMBOLS:
            index=spot[symbol][['timestamp','open','high','low','close']].copy()
            historical[symbol]=build_samples(spot[symbol],index,symbol,profile='research')
        historical=enrich_samples(historical)
        actual=build_latest_features(spot,premium,funding,signal)
        for symbol in SYMBOLS:
            expected=add_derivative_features(historical[symbol],premium[symbol],funding[symbol])
            row=expected.loc[expected.signal_ms==signal,FEATURES].iloc[0].to_numpy(dtype=float)
            np.testing.assert_allclose([actual[symbol][name] for name in FEATURES],row,rtol=1e-8,atol=1e-9)


class AlertGateTests(unittest.TestCase):
    def test_gate_has_same_direction_threshold_and_execution_times(self):
        signal=1_800_000*1_000_000
        gate={'ret_60':-30.,'rv_60':1.,'rv_ratio':1.}
        result=decision(.6,gate,signal)
        self.assertEqual(result['side'],'LONG')
        self.assertEqual(result['entry_due_ms'],signal+MINUTE_MS)
        self.assertEqual(result['expiry_due_ms'],signal+31*MINUTE_MS)
        self.assertFalse(result['order_execution_enabled'])
        self.assertFalse(result['score_is_calibrated_win_probability'])
        self.assertEqual(decision(.4,{**gate,'ret_60':30},signal)['side'],'SHORT')
        self.assertEqual(decision(.55,gate,signal)['side'],'NONE')
        self.assertEqual(decision(.6,{**gate,'ret_60':30},signal)['side'],'NONE')
        self.assertEqual(decision(.6,{**gate,'ret_60':-1},signal)['side'],'NONE')
        self.assertEqual(decision(.6,{**gate,'rv_ratio':1.3},signal)['side'],'NONE')
        with self.assertRaises(ValueError):decision(float('nan'),gate,signal)


def bundle(folder):
    import sklearn,joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    generator=np.random.default_rng(17)
    x=generator.normal(size=(40,60));y=np.arange(40)%2
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=400)).fit(x,y)
    entries={}
    for symbol in SYMBOLS:
        path=folder/f'{symbol}.joblib';joblib.dump(model,path)
        entries[symbol]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    cutoff=int(pd.Timestamp('2026-11-12',tz='UTC').timestamp()*1000)
    manifest={'schema_version':1,'rule_id':RULE_ID,'features':FEATURES,'order_execution_enabled':False,
              'sklearn_version':sklearn.__version__,'models':entries,'training_cutoff_ms':cutoff,'model_version':'test'}
    (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


class BundleTests(unittest.TestCase):
    def test_checksum_verified_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);bundle(folder)
            with (folder/'BTCUSDT.joblib').open('ab') as stream:stream.write(b'tampered')
            with patch('joblib.load') as loader:
                with self.assertRaisesRegex(ValueError,'checksum'):Predictor(folder)
                loader.assert_not_called()

    def test_model_feature_order_cutoff_staleness_and_finite_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);manifest=bundle(folder)
            predictor=Predictor(folder)
            rows={s:{name:0. for name in FEATURES} for s in SYMBOLS}
            for row in rows.values():row.update(ret_60=-30,rv_60=1,rv_ratio=1)
            outputs=predictor.predict(rows,manifest['training_cutoff_ms'])
            self.assertEqual({x['symbol'] for x in outputs},set(SYMBOLS))
            with self.assertRaisesRegex(ValueError,'before'):predictor.predict(rows,manifest['training_cutoff_ms']-30*MINUTE_MS)
            with self.assertRaisesRegex(ValueError,'maximum age'):predictor.predict(rows,manifest['training_cutoff_ms']+36*DAY_MS)
            bad=copy.deepcopy(rows);bad['BTCUSDT']['ret_1']=float('nan')
            with self.assertRaises(ValueError):predictor.predict(bad,manifest['training_cutoff_ms'])
            bad=copy.deepcopy(rows);bad['BTCUSDT']['future_target']=1
            with self.assertRaises(ValueError):predictor.predict(bad,manifest['training_cutoff_ms'])

    def test_manifest_rejects_path_traversal_and_wrong_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);manifest=bundle(folder)
            manifest['models']['BTCUSDT']['file']='../BTCUSDT.joblib'
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'filename'):Predictor(folder)
            manifest=bundle(folder);manifest['features']=list(reversed(FEATURES))
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'contract'):Predictor(folder)

    def test_small_memory_refresh_excludes_unsettled_labels_and_exports_seed(self):
        cutoff=int(pd.Timestamp('2026-11-12',tz='UTC').timestamp()*1000)
        n=50*48+4
        signals=cutoff-50*DAY_MS+np.arange(n)*30*MINUTE_MS
        generator=np.random.default_rng(712)
        rows={}
        for symbol in SYMBOLS:
            frame=pd.DataFrame(generator.normal(size=(n,60)),columns=FEATURES)
            frame['signal_ms']=signals
            frame['settlement_ms']=signals+31*MINUTE_MS
            frame['y']=np.arange(n)%2
            frame['is_flat']=False
            frame.loc[1000,'is_flat']=True
            rows[symbol]=frame
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            first=retrain_from_rows(folder/'first',rows,cutoff,progress=lambda _:None)
            changed=copy.deepcopy(rows)
            for frame in changed.values():
                future=frame.settlement_ms>=cutoff
                frame.loc[future,FEATURES]=float('nan')
                frame.loc[future,'y']=1-frame.loc[future,'y']
            again=retrain_from_rows(folder/'again',changed,cutoff,progress=lambda _:None)
            self.assertEqual(first['model_version'],again['model_version'])
            for symbol in SYMBOLS:
                entry=first['models'][symbol]
                seed_path=folder/'first'/entry['training_samples_file']
                self.assertEqual(hashlib.sha256(seed_path.read_bytes()).hexdigest(),entry['training_samples_sha256'])
                seed=pd.read_csv(seed_path,float_precision='round_trip')
                self.assertEqual(list(seed),FEATURES+['signal_ms','settlement_ms','y','is_flat'])
                self.assertTrue((seed.settlement_ms<cutoff).all())
                self.assertEqual(entry['training_samples_rows'],2399)
                self.assertEqual(entry['training_count'],2398)
                self.assertEqual(int(seed.is_flat.sum()),1)
                self.assertLess(entry['latest_settlement_ms'],cutoff)
            self.assertIsInstance(Predictor(folder/'first'),Predictor)

    def test_small_memory_refresh_rejects_invalid_event_grid_and_insufficient_history(self):
        cutoff=int(pd.Timestamp('2026-11-12',tz='UTC').timestamp()*1000)
        row={name:0. for name in FEATURES}
        row.update(signal_ms=cutoff-60*MINUTE_MS,settlement_ms=cutoff-29*MINUTE_MS,y=0,is_flat=False)
        rows={s:pd.DataFrame([row]) for s in SYMBOLS}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'Insufficient'):retrain_from_rows(temporary,rows,cutoff,progress=lambda _:None)
            rows['BTCUSDT'].loc[0,'settlement_ms']-=MINUTE_MS
            with self.assertRaisesRegex(ValueError,'event times'):retrain_from_rows(temporary,rows,cutoff,progress=lambda _:None)


if __name__=='__main__':unittest.main()
