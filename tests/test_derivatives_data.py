"""Signed premium, realized funding, provenance, and official repair checks."""
from datetime import date,timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

import numpy as np
import pandas as pd

from research.derivatives_data import (
    FUNDING_COLUMNS,PREMIUM_COLUMNS,archive_paths,load_funding,load_premium,
    parse_funding,parse_premium,read_archive,validate_funding,validate_premium,
    _rest_cached,
)
from research.qd_three_year_data import DAY_MS,MINUTE_MS,_ms


DAY=date(2026,1,1)


def zipped(rows):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as archive:
        archive.writestr("data.csv","\n".join(",".join(map(str,row)) for row in rows)+"\n")
    return stream.getvalue()


def premium_row(t):
    return [t,-.0002,-.0001,-.0004,-.0003,0,t+59999,0,0,0,0,0]


def full_day():
    times=np.arange(_ms(DAY),_ms(DAY)+DAY_MS,MINUTE_MS,dtype=np.int64)
    return pd.DataFrame({"timestamp":times,"open":-.0002,"high":-.0001,"low":-.0004,"close":-.0003})


class DerivativeParserTests(unittest.TestCase):
    def test_signed_zero_premium_and_real_header_are_supported(self):
        rows=[["open_time","open","high","low","close","volume","close_time","quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"],
              premium_row(_ms(DAY)),[_ms(DAY)+MINUTE_MS,0,0,0,0,0,0,0,0,0,0,0]]
        frame=parse_premium(zipped(rows))
        self.assertEqual(list(frame.columns),PREMIUM_COLUMNS)
        self.assertEqual(frame.open.tolist(),[-.0002,0])
        self.assertEqual(frame.timestamp.dtype,np.dtype("int64"))
        validate_premium(frame)

    def test_microseconds_are_normalized_without_rounding(self):
        frame=parse_premium(zipped([premium_row(_ms(DAY)*1000)]))
        self.assertEqual(frame.timestamp.iloc[0],_ms(DAY))
        with self.assertRaisesRegex(ValueError,"Non-millisecond"):
            parse_premium(zipped([premium_row(_ms(DAY)*1000+1)]))

    def test_premium_invalid_range_nonfinite_grid_and_duplicates_rejected(self):
        frame=full_day()
        validate_premium(frame,start=DAY,end=DAY+timedelta(days=1))
        for column,value in (("high",-.0008),("low",.01),("close",np.inf),("timestamp",_ms(DAY)+1)):
            bad=frame.copy();bad.loc[2,column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):
                validate_premium(bad)
        with self.assertRaisesRegex(ValueError,"Missing"):
            validate_premium(frame.drop(index=3))
        with self.assertRaises(ValueError):
            validate_premium(pd.concat([frame,frame.iloc[-1:]],ignore_index=True))

    def test_actual_funding_schema_preserves_nonrounded_event_milliseconds(self):
        raw=zipped([["calc_time","funding_interval_hours","last_funding_rate"],
                    [_ms(DAY)+1,8,.00007981],[_ms(DAY)+8*3600000+5,8,-.00004568]])
        frame=parse_funding(raw)
        self.assertEqual(list(frame.columns),FUNDING_COLUMNS)
        self.assertEqual(frame.funding_time.tolist(),[_ms(DAY)+1,_ms(DAY)+8*3600000+5])
        self.assertEqual(frame.funding_time.dtype,np.dtype("int64"))
        self.assertEqual(frame.rate.iloc[-1],-.00004568)

    def test_funding_rejects_nonfinite_rate_duplicate_or_invalid_interval(self):
        frame=pd.DataFrame({"funding_time":[_ms(DAY)+1,_ms(DAY)+10000],"rate":[.0001,-.0001],"interval_hours":[8.,8.]})
        for col,value in (("rate",np.nan),("funding_time",_ms(DAY)+1),("interval_hours",0.)):
            bad=frame.copy();bad.loc[1,col]=value
            with self.subTest(column=col),self.assertRaises(ValueError):
                validate_funding(bad)
        # REST does not provide interval hours; missing values stay explicitly unknown.
        frame["interval_hours"]=np.nan
        validate_funding(frame)

    def test_official_monthly_and_daily_tail_paths_are_exact(self):
        paths=list(archive_paths("premium","BTCUSDT",date(2023,11,6),date(2026,11,12)))
        self.assertEqual(paths[0],"monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2023-11.zip")
        self.assertEqual(paths[-1],"daily/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2026-11-11.zip")
        funding=list(archive_paths("funding","ETHUSDT",date(2023,11,6),date(2026,11,12)))
        self.assertEqual(len(funding),37)
        self.assertEqual(funding[-1],"monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-11.zip")
        with self.assertRaises(ValueError):
            list(archive_paths("funding","SOLUSDT",DAY,DAY+timedelta(days=1)))


class DerivativeProvenanceTests(unittest.TestCase):
    def test_exchange_checksum_offline_reuse_and_tampering_fail_closed(self):
        payload=zipped([premium_row(_ms(DAY))])
        checksum=(hashlib.sha256(payload).hexdigest()+"  data.zip\n").encode()
        relative="daily/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2026-01-01.zip"
        with tempfile.TemporaryDirectory() as folder:
            with patch("research.derivatives_data._fetch",side_effect=[payload,checksum]):
                frame,source=read_archive("premium",relative,folder)
            self.assertEqual(source["verification"],"exchange_archive_sha256")
            with patch("research.derivatives_data._fetch",side_effect=AssertionError("network")):
                offline,_=read_archive("premium",relative,folder,True)
            pd.testing.assert_frame_equal(frame,offline)
            (Path(folder)/source["path"]).write_bytes(payload+b"tampered")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                read_archive("premium",relative,folder,True)

    def test_archive_path_traversal_is_rejected_before_any_network(self):
        for relative in ("../evil.zip","C:/evil.zip","/evil.zip","bad\\evil.zip"):
            with self.subTest(relative=relative),self.assertRaises(ValueError):
                read_archive("premium",relative,"unused")

    def test_funding_rest_website_fallback_original_pages_and_offline_reuse(self):
        rows=[{"symbol":"BTCUSDT","fundingTime":_ms(DAY)+1,"fundingRate":"0.0001"},
              {"symbol":"BTCUSDT","fundingTime":_ms(DAY)+8*3600000+5,"fundingRate":"-0.00002"}]
        body=json.dumps(rows).encode()
        blocked=urllib.error.HTTPError("https://fapi.binance.com/",451,"restricted",{},None)
        with tempfile.TemporaryDirectory() as folder:
            with patch("research.derivatives_data._fetch",side_effect=[blocked,body,b"[]"]):
                frame,entry=_rest_cached("funding","BTCUSDT",DAY,DAY+timedelta(days=1),folder)
            self.assertTrue(entry["urls"][0].startswith("https://www.binance.com/fapi/v1/fundingRate"))
            self.assertEqual(frame.funding_time.tolist(),[r["fundingTime"] for r in rows])
            self.assertTrue(frame.interval_hours.isna().all())
            cached=Path(folder)/entry["path"]
            self.assertEqual(cached.with_suffix(".response-0.json").read_bytes(),body)
            with patch("research.derivatives_data._fetch",side_effect=AssertionError("network")):
                offline,source=_rest_cached("funding","BTCUSDT",DAY,DAY+timedelta(days=1),folder,True)
            pd.testing.assert_frame_equal(frame,offline)
            self.assertEqual(source["sha256"],entry["sha256"])
            cached.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError,"REST checksum"):
                _rest_cached("funding","BTCUSDT",DAY,DAY+timedelta(days=1),folder,True)

    def test_premium_rest_paginated_day_and_explicit_nonexchange_integrity(self):
        rows=[premium_row(_ms(DAY)+i*MINUTE_MS) for i in range(1440)]
        pages=[json.dumps(rows[:1000]).encode(),json.dumps(rows[1000:]).encode()]
        with tempfile.TemporaryDirectory() as folder:
            with patch("research.derivatives_data._fetch",side_effect=pages):
                frame,entry=_rest_cached("premium","ETHUSDT",DAY,DAY+timedelta(days=1),folder)
            validate_premium(frame,start=DAY,end=DAY+timedelta(days=1))
            self.assertEqual(entry["verification"],"local_sha256_of_official_rest_snapshot")
            cached=Path(folder)/entry["path"]
            self.assertEqual(cached.with_suffix(".response-1.json").read_bytes(),pages[1])

    def test_wrong_rest_symbol_or_nonadvancing_pagination_rejected(self):
        rows=[{"symbol":"ETHUSDT","fundingTime":_ms(DAY),"fundingRate":"0.0001"}]
        with tempfile.TemporaryDirectory() as folder,patch("research.derivatives_data._fetch",return_value=json.dumps(rows).encode()):
            with self.assertRaisesRegex(ValueError,"symbol mismatch"):
                _rest_cached("funding","BTCUSDT",DAY,DAY+timedelta(days=1),folder)


class DerivativeLoadTests(unittest.TestCase):
    def test_missing_premium_minutes_repaired_using_only_official_observations(self):
        complete=full_day();sparse=complete.drop(index=[10,100]).reset_index(drop=True)
        missing=urllib.error.HTTPError("url",404,"missing",{},None)
        with patch("research.derivatives_data.archive_paths",return_value=iter(["month.zip"])),\
             patch("research.derivatives_data.read_archive",side_effect=[(sparse,{"path":"month.zip"}),missing]),\
             patch("research.derivatives_data._rest_cached",return_value=(complete,{"path":"rest.json"})):
            frame,sources=load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),"cache",True)
        pd.testing.assert_frame_equal(frame,complete)
        self.assertEqual(len(sources),2)

    def test_complete_daily_archive_repairs_monthly_hole_before_any_rest(self):
        complete=full_day();sparse=complete.drop(index=10).reset_index(drop=True)
        with patch("research.derivatives_data.archive_paths",return_value=iter(["month.zip"])),\
             patch("research.derivatives_data.read_archive",side_effect=[(sparse,{"path":"month.zip"}),(complete,{"path":"daily.zip"})]),\
             patch("research.derivatives_data._rest_cached",side_effect=AssertionError("REST unnecessary")):
            frame,sources=load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),"cache",True)
        pd.testing.assert_frame_equal(frame,complete)
        self.assertEqual(len(sources),2)

    def test_genuine_official_gap_is_explicit_and_never_filled(self):
        complete=full_day();sparse=complete.drop(index=[10,11,100]).reset_index(drop=True)
        with tempfile.TemporaryDirectory() as folder,\
             patch("research.derivatives_data.archive_paths",side_effect=lambda *args:iter(["month.zip"])),\
             patch("research.derivatives_data.read_archive",return_value=(sparse,{"path":"archive.zip"})),\
             patch("research.derivatives_data._rest_cached",return_value=(sparse,{"path":"rest.json"})):
            with self.assertRaisesRegex(ValueError,"Missing"):
                load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),folder,True)
            observed,sources=load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),folder,True,allow_gaps=True)
            pd.testing.assert_frame_equal(observed,sparse)
            quality=next(s for s in sources if s.get("kind")=="premium_quality")
            self.assertFalse(quality["complete"])
            self.assertEqual(quality["missing_minutes"],3)
            self.assertEqual(quality["interpolated_minutes"],0)
            self.assertEqual(quality["missing_ranges"][0]["start_ms"],_ms(DAY)+10*MINUTE_MS)
            self.assertEqual(quality["missing_ranges"][0]["end_exclusive_ms"],_ms(DAY)+12*MINUTE_MS)
            self.assertEqual(quality["missing_ranges"][1]["missing_minutes"],1)
            inventory=(Path(folder)/quality["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(inventory).hexdigest(),quality["sha256"])

    def test_sparse_rest_snapshot_reuses_offline_but_strict_mode_still_rejects(self):
        rows=[premium_row(_ms(DAY)+i*MINUTE_MS) for i in range(1440) if i!=10]
        pages=[json.dumps(rows[:1000]).encode(),json.dumps(rows[1000:]).encode()]
        with tempfile.TemporaryDirectory() as folder:
            with patch("research.derivatives_data._fetch",side_effect=pages):
                frame,entry=_rest_cached("premium","BTCUSDT",DAY,DAY+timedelta(days=1),folder,allow_gaps=True)
            self.assertEqual(len(frame),1439)
            self.assertEqual(entry["missing_minutes"],1)
            with patch("research.derivatives_data._fetch",side_effect=AssertionError("network")):
                offline,source=_rest_cached("premium","BTCUSDT",DAY,DAY+timedelta(days=1),folder,True,allow_gaps=True)
                pd.testing.assert_frame_equal(frame,offline)
                self.assertEqual(entry["sha256"],source["sha256"])
                with self.assertRaisesRegex(ValueError,"Missing"):
                    _rest_cached("premium","BTCUSDT",DAY,DAY+timedelta(days=1),folder,True)

    def test_conflicting_premium_repair_is_rejected(self):
        complete=full_day();sparse=complete.drop(index=10).reset_index(drop=True)
        conflict=complete.copy();conflict.loc[20,"close"]=-.0002
        with patch("research.derivatives_data.archive_paths",return_value=iter(["month.zip"])),\
             patch("research.derivatives_data.read_archive",return_value=(sparse,{"path":"month.zip"})),\
             patch("research.derivatives_data._rest_cached",return_value=(conflict,{"path":"rest.json"})):
            with self.assertRaisesRegex(ValueError,"Conflicting"):
                load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),"cache",True)

    def test_missing_daily_premium_uses_official_rest_not_interpolation(self):
        missing=urllib.error.HTTPError("url",404,"missing",{},None)
        complete=full_day()
        with patch("research.derivatives_data.read_archive",side_effect=missing),\
             patch("research.derivatives_data._rest_cached",return_value=(complete,{"path":"rest.json"})) as rest:
            frame,_=load_premium("BTCUSDT",DAY,DAY+timedelta(days=1),"cache",True)
        pd.testing.assert_frame_equal(frame,complete)
        self.assertEqual(rest.call_args.args[0],"premium")

    def test_funding_missing_archive_falls_back_without_inventing_interval_hours(self):
        frame=pd.DataFrame({"funding_time":[_ms(DAY)+1],"rate":[.0001],"interval_hours":[np.nan]})
        missing=urllib.error.HTTPError("url",404,"missing",{},None)
        with patch("research.derivatives_data.read_archive",side_effect=missing),\
             patch("research.derivatives_data._rest_cached",return_value=(frame,{"path":"rest.json"})):
            observed,_=load_funding("ETHUSDT",DAY,DAY+timedelta(days=1),"cache",True)
        pd.testing.assert_frame_equal(observed,frame)
        self.assertTrue(observed.interval_hours.isna().all())


if __name__=="__main__":
    unittest.main()
