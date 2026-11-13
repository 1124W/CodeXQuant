"""Official checksum, raw-time integrity and fail-closed metrics loading."""
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import zipfile

import numpy as np
import pandas as pd
import requests

from research import metrics_data as data


def csv_rows(times=('2023-11-09 00:00:00','2023-11-09 00:05:00')):
    return pd.DataFrame({'create_time':list(times),'symbol':'BTCUSDT',
                         **{name:np.arange(len(times),dtype=float)+1 for name in data.METRIC_COLUMNS}})


def zipped(frame, symbol='BTCUSDT', day=data.START):
    name=f'{symbol}-metrics-{day}.zip'; stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix('.zip')+'.csv',frame.to_csv(index=False))
    body=stream.getvalue();checksum=hashlib.sha256(body).hexdigest()+'  '+name+'\n'
    return body,checksum


class MetricsDataTests(unittest.TestCase):
    def test_scope_is_exact_1099_days_and_only_btc_eth(self):
        pairs=data.expected_files()
        self.assertEqual(len(pairs),2198)
        self.assertEqual(len({day for _,day in pairs}),1099)
        self.assertEqual(max(day for _,day in pairs),data.END-timedelta(days=1))
        for symbol,day in [('SOLUSDT',data.START),('BTCUSDT',data.START-timedelta(days=1)),('BTCUSDT',data.END)]:
            with self.assertRaises(ValueError):data._names(symbol,day)

    def test_checksum_content_filename_and_members_are_enforced(self):
        body,checksum=zipped(csv_rows())
        self.assertEqual(data.verify_bytes(body,checksum,'BTCUSDT-metrics-2023-11-09.zip'),hashlib.sha256(body).hexdigest())
        for bad_body,bad_sum in [(body+b'x',checksum),(body,checksum.replace('BTCUSDT','ETHUSDT')),(body,checksum+checksum)]:
            with self.assertRaises(ValueError):data.verify_bytes(bad_body,bad_sum,'BTCUSDT-metrics-2023-11-09.zip')
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('../escape.csv','bad')
        payload=stream.getvalue();check=hashlib.sha256(payload).hexdigest()+'  BTCUSDT-metrics-2023-11-09.zip'
        with self.assertRaisesRegex(ValueError,'members'):data.parse_archive(payload,check,'BTCUSDT',data.START)

    def test_shuffled_observations_sort_without_inventing_rows(self):
        body,checksum=zipped(csv_rows().iloc[::-1])
        raw,meta=data.parse_archive(body,checksum,'BTCUSDT',data.START)
        clean,quality=data.clean_rows(raw)
        self.assertTrue(clean.record_ms.is_monotonic_increasing)
        self.assertEqual(len(clean),2)
        self.assertFalse(meta['quality']['original_row_order_monotonic'])
        self.assertEqual(quality['accepted_rows'],2)

    def test_clock_offsets_and_bad_numeric_values_are_not_rounded_or_filled(self):
        frame=csv_rows(['2023-11-09 00:00:00','2023-11-09 00:05:02','2023-11-09 00:10:00',
                        '2023-11-09 00:15:00','2023-11-09 00:20:00','2023-11-08 23:55:00'])
        frame.loc[2,'sum_open_interest']=0.;frame.loc[3,'count_long_short_ratio']=np.nan
        frame.loc[4,'sum_taker_long_short_vol_ratio']=-1.
        raw,meta=data.parse_archive(*zipped(frame),'BTCUSDT',data.START)
        clean,quality=data.clean_rows(raw)
        self.assertEqual(len(clean),1)
        self.assertEqual(meta['quality']['invalid_or_offgrid_or_outside_day_rows_dropped'],2)
        self.assertEqual(quality['nonpositive_or_nonfinite_rows_dropped'],3)

    def test_next_midnight_is_retained_until_cross_file_conflict_check(self):
        frame=csv_rows(['2023-11-09 23:55:00','2023-11-10 00:00:00'])
        raw,_=data.parse_archive(*zipped(frame),'BTCUSDT',data.START)
        following=csv_rows(['2023-11-10 00:00:00','2023-11-10 00:05:00'])
        other,_=data.parse_archive(*zipped(following,day=data.START+timedelta(days=1)),'BTCUSDT',data.START+timedelta(days=1))
        clean,quality=data.clean_rows(pd.concat([raw,other],ignore_index=True))
        self.assertEqual(quality['conflicting_duplicate_timestamps'],1)
        self.assertEqual(quality['conflicting_duplicate_rows_dropped'],2)
        self.assertEqual(len(clean),2)
        self.assertFalse((clean.record_ms==data._ms(data.START+timedelta(days=1))).any())

    def test_identical_duplicates_collapse_but_valid_invalid_conflict_drops_both(self):
        raw,_=data.parse_archive(*zipped(csv_rows()),'BTCUSDT',data.START)
        duplicate=pd.concat([raw,raw.iloc[:1]],ignore_index=True)
        clean,quality=data.clean_rows(duplicate)
        self.assertEqual(len(clean),2);self.assertEqual(quality['identical_duplicate_rows_removed'],1)
        duplicate.loc[2,'sum_open_interest']=np.nan
        clean,quality=data.clean_rows(duplicate)
        self.assertEqual(len(clean),1);self.assertEqual(quality['conflicting_duplicate_rows_dropped'],2)

    def test_changed_schema_and_wrong_asset_are_rejected(self):
        changed=csv_rows().rename(columns={'sum_open_interest':'new_oi'})
        with self.assertRaisesRegex(ValueError,'schema'):data.parse_archive(*zipped(changed),'BTCUSDT',data.START)
        changed=csv_rows();changed['symbol']='ETHUSDT'
        with self.assertRaisesRegex(ValueError,'different asset'):data.parse_archive(*zipped(changed),'BTCUSDT',data.START)

    def test_local_probe_reuse_verifies_then_publishes_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);probe=root/'probe';probe.mkdir();folder=root/'store';folder.mkdir()
            body,checksum=zipped(csv_rows());name='BTCUSDT-metrics-2023-11-09.zip'
            (probe/name).write_bytes(body);(probe/(name+'.CHECKSUM')).write_text(checksum)
            state=data.FetchState()
            with patch.object(state,'get',side_effect=AssertionError('network forbidden')):
                key,record,origin=data._obtain(('BTCUSDT',data.START),folder,state,probe)
                self.assertEqual(origin,'verified_probe');self.assertEqual(key,'BTCUSDT:2023-11-09')
                self.assertTrue((folder/record['file']).is_file())
                again=data._obtain(('BTCUSDT',data.START),folder,state,probe)
                self.assertEqual(again[2],'verified_cache')

    def test_offline_load_checks_manifest_and_drops_boundary_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);files={}
            for offset,times in [(0,['2023-11-09 23:55:00','2023-11-10 00:00:00']),
                                 (1,['2023-11-10 00:00:00','2023-11-10 00:05:00'])]:
                day=data.START+timedelta(days=offset);body,checksum=zipped(csv_rows(times),day=day)
                _,binary,check=data._names('BTCUSDT',day)
                data._atomic_bytes(folder/binary,body);data._atomic_bytes(folder/check,checksum.encode())
                files[f'BTCUSDT:{day}']={'file':binary,'checksum_file':check,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
            manifest={**data._contract(),'complete':True,'files':files}
            data._write_json(folder/'manifest.json',manifest)
            pairs=[('BTCUSDT',data.START),('BTCUSDT',data.START+timedelta(days=1))]
            with patch.object(data,'expected_files',return_value=pairs),patch.object(requests.Session,'get',side_effect=AssertionError('offline')):
                data.verify_metrics_store(folder)
                frame,quality=data.load_metrics('BTCUSDT',data.START+timedelta(days=1),data.START+timedelta(days=2),folder)
                self.assertEqual(len(frame),1)
                self.assertEqual(quality['conflicting_duplicate_timestamps'],1)
                self.assertEqual(len(quality['source_sha256']),2)
                manifest['complete']=False;data._write_json(folder/'manifest.json',manifest)
                with self.assertRaisesRegex(ValueError,'incomplete'):data.load_metrics('BTCUSDT',data.START,data.START+timedelta(days=1),folder)

    def test_http404_is_not_retried_and_budget_is_bounded(self):
        state=data.FetchState(max_requests=1)
        response=requests.Response();response.status_code=404
        error=requests.exceptions.HTTPError('missing',response=response)
        session=Mock();session.get.side_effect=error
        with patch.object(state,'_session',return_value=session):
            getter=session.get
            with self.assertRaises(requests.exceptions.HTTPError):state.get(data.BASE+'/BTCUSDT/test.zip',100)
            self.assertEqual(getter.call_count,1);self.assertTrue(state.stop.is_set())
        state=data.FetchState(max_requests=0)
        with self.assertRaisesRegex(RuntimeError,'budget'):state.get(data.BASE+'/BTCUSDT/test.zip',100)

    def test_repeated_tls_failures_trip_circuit_without_disabling_verification(self):
        state=data.FetchState();error=requests.exceptions.SSLError('TLS handshake timeout')
        session=Mock();session.get.side_effect=error
        with patch.object(state,'_session',return_value=session),patch('time.sleep'):
            getter=session.get
            with self.assertRaises(requests.exceptions.SSLError):state.get(data.BASE+'/BTCUSDT/test.zip',100)
            self.assertEqual(getter.call_count,3);self.assertTrue(state.stop.is_set())
            self.assertEqual(state.requests,3)
            self.assertTrue(all(call.kwargs['verify'] is True for call in getter.call_args_list))
            self.assertTrue(all(call.args[0].startswith(data.TRANSPORT_BASE) for call in getter.call_args_list))

    def test_transport_reuses_one_session_per_worker_with_hidden_retries_disabled(self):
        state=data.FetchState()
        first=state._session();self.assertIs(first,state._session())
        self.assertEqual(first.adapters['https://'].max_retries.total,0)
        state.close()


if __name__=='__main__':unittest.main()
