"""Publication-delay assumptions, exact OI lags and missing-record boundaries."""
from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import metrics_features as feature
from research.event_engine import MINUTE_MS,utc_ms

ORIGIN=utc_ms(date(2025,1,1))


def observations(symbol='BTCUSDT'):
    offsets=np.arange(-300,66,5)
    interest=1000*np.exp(offsets/10000)
    return pd.DataFrame({'record_ms':ORIGIN+offsets*MINUTE_MS,'symbol':symbol,
        'sum_open_interest':interest,'sum_open_interest_value':interest*2000,
        'count_toptrader_long_short_ratio':1.2,'sum_toptrader_long_short_ratio':1.4,
        'count_long_short_ratio':.9,'sum_taker_long_short_vol_ratio':1.1})


def base(minutes,symbol='BTCUSDT'):
    return pd.DataFrame({'signal_ms':ORIGIN+np.asarray(minutes)*MINUTE_MS,
                         'symbol':symbol,'norm_ret_60':2.},index=np.arange(len(minutes))*7+3)


class MetricsFeatureTests(unittest.TestCase):
    def test_exact_log_stock_changes_ratios_and_current_price_interaction(self):
        frame,quality=feature.augment_frame(base([10,11,14]),observations())
        np.testing.assert_allclose(frame[feature.METRIC_FEATURES[:3]],[[.003,.006,.024]]*3,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(frame[feature.METRIC_FEATURES[3:7]],np.tile(np.log([1.2,1.4,.9,1.1]),(3,1)))
        np.testing.assert_allclose(frame.metrics_oi_price_interaction_60,.012)
        self.assertEqual(frame.index.tolist(),[3,10,17])
        np.testing.assert_array_equal(frame.metrics_record_ms,[ORIGIN]*3)
        self.assertEqual(quality['maximum_observed_available_age_ms'],4*MINUTE_MS)
        self.assertFalse(quality['availability_is_publication_time_proof'])

    def test_missing_next_record_expires_exactly_at_five_minutes_without_carry(self):
        incoming=observations();incoming=incoming[incoming.record_ms!=ORIGIN+5*MINUTE_MS]
        result,_=feature.augment_frame(base([10,14,15,16,19,20]),incoming)
        np.testing.assert_array_equal(result.signal_ms,[ORIGIN+m*MINUTE_MS for m in [10,14,20]])
        self.assertTrue((result.metrics_age_ms<5*MINUTE_MS).all())

    def test_availability_boundary_and_future_changes_do_not_change_past_features(self):
        incoming=observations();before,_=feature.augment_frame(base([9,10]),incoming)
        self.assertEqual(before.metrics_record_ms.tolist(),[ORIGIN-5*MINUTE_MS,ORIGIN])
        changed=incoming.copy();changed.loc[changed.record_ms>ORIGIN,list(feature.RAW_COLUMNS)]*=10
        after,_=feature.augment_frame(base([9,10]),changed)
        pd.testing.assert_frame_equal(before,after)

    def test_exact_lag_endpoint_required_but_missing_intermediate_stock_is_allowed(self):
        incoming=observations();gap=incoming[incoming.record_ms!=ORIGIN-55*MINUTE_MS]
        result,_=feature.augment_frame(base([10]),gap)
        self.assertEqual(len(result),1)
        endpoint=incoming[incoming.record_ms!=ORIGIN-60*MINUTE_MS]
        result,quality=feature.augment_frame(base([10]),endpoint)
        self.assertTrue(result.empty)
        self.assertEqual(quality['timely_but_missing_exact_lag_or_nonfinite_feature_rows'],1)

    def test_conflicting_duplicates_all_drop_including_invalid_competing_value(self):
        incoming=observations();row=incoming[incoming.record_ms==ORIGIN].copy()
        duplicate=pd.concat([incoming,row],ignore_index=True)
        result,q=feature.augment_frame(base([10]),duplicate)
        self.assertEqual(len(result),1);self.assertEqual(q['identical_metrics_duplicates_collapsed'],1)
        row['sum_open_interest']=-1
        conflicting=pd.concat([incoming,row],ignore_index=True)
        result,q=feature.augment_frame(base([10]),conflicting)
        self.assertTrue(result.empty)
        self.assertEqual(q['conflicting_metrics_rows_dropped'],2)

    def test_symbols_remain_separate_and_execution_labels_are_not_read(self):
        btc=observations();eth=observations('ETHUSDT');eth['count_toptrader_long_short_ratio']=2.5
        frames=pd.concat([base([10]),base([10],'ETHUSDT')],ignore_index=True)
        frames['entry']='future';frames['exit']='future';frames['y']='future'
        result,_=feature.augment_frame(frames,pd.concat([btc,eth],ignore_index=True))
        np.testing.assert_allclose(result.metrics_log_top_account_ratio,np.log([1.2,2.5]))
        self.assertEqual(result.entry.tolist(),['future','future'])

    def test_loader_clamps_raw_coverage_and_preserves_compact_provenance(self):
        from research.metrics_data import START,END
        first=utc_ms(START)
        raw=observations();frame=base([10])
        with patch('research.model_search_data.load_window',return_value=frame),patch('research.metrics_data.load_metrics',return_value=(raw,{'manifest_identity':'fixture','source_sha256':{'many':'hash'}})) as loader:
            result,q=feature.load_enriched_window('BTCUSDT',first,first+60*MINUTE_MS)
            self.assertEqual(loader.call_args.args[1],START)
            self.assertEqual(q['manifest_identity'],'fixture')
            self.assertNotIn('source_sha256',q)
            self.assertEqual(len(result),1)

    def test_nonpositive_or_nonfinite_observations_and_nonfinite_interaction_fail_closed(self):
        incoming=observations();incoming.loc[incoming.record_ms==ORIGIN,'sum_taker_long_short_vol_ratio']=0
        result,_=feature.augment_frame(base([10]),incoming)
        self.assertTrue(result.empty)
        frame=base([10]);frame['norm_ret_60']=np.nan
        result,_=feature.augment_frame(frame,observations())
        self.assertTrue(result.empty)


if __name__=='__main__':unittest.main()
