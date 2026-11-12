"""Verified public Binance premium-index minutes and realized funding events.

Premium OHLC can be negative or zero. Funding is an event-time observation, not
information about the next funding payment. Consumers must apply their declared
publication lag before joining rates to strategy signals. No synthetic filling.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import zipfile

import numpy as np
import pandas as pd

from .qd_three_year_data import DAY_MS, MINUTE_MS, SYMBOLS, VISION, _atomic_bytes, _fetch, _ms

PREMIUM_COLUMNS = ["timestamp", "open", "high", "low", "close"]
FUNDING_COLUMNS = ["funding_time", "rate", "interval_hours"]


def _dates(symbol, start, end):
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    if symbol not in SYMBOLS or not isinstance(start, date) or not isinstance(end, date) or start >= end:
        raise ValueError("Require BTCUSDT/ETHUSDT and an increasing UTC date range")
    return start, end


def _next_month(day):
    return (day.replace(day=28) + timedelta(days=4)).replace(day=1)


def archive_paths(kind, symbol, start, end):
    start, end = _dates(symbol, start, end)
    if kind not in ("premium", "funding"):
        raise ValueError("Unknown derivative history kind")
    month = start.replace(day=1)
    while month < end:
        following = _next_month(month)
        if kind == "funding":
            yield f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month:%Y-%m}.zip"
        elif following <= end:
            yield f"monthly/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{month:%Y-%m}.zip"
        else:
            day = max(start, month)
            while day < end:
                yield f"daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                day += timedelta(days=1)
        month = following


def _zip_csv(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one derivative CSV")
        return archive.read(names[0])


def _timestamps(values):
    raw = pd.to_numeric(values, errors="raise").to_numpy()
    if not np.isfinite(raw).all() or np.any(raw != raw.astype(np.int64)):
        raise ValueError("Invalid derivative timestamp")
    raw = raw.astype(np.int64)
    micros = raw > 100_000_000_000_000
    if np.any(raw[micros] % 1000):
        raise ValueError("Non-millisecond derivative timestamp")
    raw[micros] //= 1000
    return raw


def validate_premium(frame, *, start=None, end=None, continuous=True):
    if list(frame.columns) != PREMIUM_COLUMNS or frame.empty:
        raise ValueError("Wrong or empty premium minute schema")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite premium observations")
    times = frame.timestamp.to_numpy()
    if np.any(times != times.astype(np.int64)) or np.any(times % MINUTE_MS):
        raise ValueError("Off-grid premium minute")
    diffs = np.diff(times)
    if np.any(diffs <= 0) or continuous and np.any(diffs != MINUTE_MS):
        raise ValueError("Missing, duplicate or unordered premium minutes")
    opened, high, low, close = (frame[name].to_numpy() for name in PREMIUM_COLUMNS[1:])
    if (np.any(low > np.minimum(opened, close)) or np.any(high < np.maximum(opened, close))
            or np.any(high < low)):
        raise ValueError("Invalid premium OHLC")
    if start is not None and end is not None:
        expected = (end-start).days*1440
        if len(frame) != expected or times[0] != _ms(start) or times[-1] != _ms(end)-MINUTE_MS:
            raise ValueError(f"Incomplete premium minute history: {len(frame)} of {expected}")


def validate_funding(frame):
    if list(frame.columns) != FUNDING_COLUMNS or frame.empty:
        raise ValueError("Wrong or empty funding event schema")
    times = frame.funding_time.to_numpy()
    if (not np.isfinite(times).all() or np.any(times != times.astype(np.int64))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid, duplicate or unordered funding event times")
    if not np.isfinite(frame.rate.to_numpy()).all():
        raise ValueError("Non-finite funding rate")
    intervals = frame.interval_hours.to_numpy()
    observed = ~np.isnan(intervals)
    if np.any(~np.isfinite(intervals[observed])) or np.any(intervals[observed] <= 0):
        raise ValueError("Invalid officially reported funding interval")


def parse_premium(payload):
    raw = pd.read_csv(io.BytesIO(_zip_csv(payload)), header=None, dtype=str)
    if raw.empty or raw.shape[1] < 5:
        raise ValueError("Malformed premium archive")
    if str(raw.iloc[0,0]).lstrip("\ufeff") in ("open_time", "openTime"):
        raw = raw.iloc[1:].copy()
    frame = raw.iloc[:,:5].copy()
    frame.columns = PREMIUM_COLUMNS
    frame["timestamp"] = _timestamps(frame.timestamp)
    for column in PREMIUM_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column],errors="raise").astype(float)
    frame=frame.reset_index(drop=True)
    validate_premium(frame,continuous=False)
    return frame


def parse_funding(payload):
    raw=pd.read_csv(io.BytesIO(_zip_csv(payload)),dtype=str)
    raw.columns=[str(name).lstrip("\ufeff") for name in raw.columns]
    required={"calc_time","funding_interval_hours","last_funding_rate"}
    if not required.issubset(raw.columns) or raw.empty:
        raise ValueError("Malformed official funding archive columns")
    frame=pd.DataFrame({"funding_time":_timestamps(raw.calc_time),
                        "rate":pd.to_numeric(raw.last_funding_rate,errors="raise").astype(float),
                        "interval_hours":pd.to_numeric(raw.funding_interval_hours,errors="raise").astype(float)})
    validate_funding(frame)
    return frame


def _relative_path(relative):
    if (not isinstance(relative,str) or ".." in relative or "\\" in relative or ":" in relative
            or relative.startswith("/") or not relative.endswith(".zip")):
        raise ValueError("Invalid derivative archive path")
    return relative


def read_archive(kind,relative,cache_dir,offline=False):
    if kind not in ("premium","funding"):
        raise ValueError("Unknown derivative archive kind")
    relative=_relative_path(relative)
    cached=Path(cache_dir)/"derivatives"/relative
    checksum_path=cached.with_suffix(".zip.CHECKSUM")
    url=f"{VISION}/futures/um/{relative}"
    if cached.is_file() and checksum_path.is_file():
        payload,checksum=cached.read_bytes(),checksum_path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Verified derivative archive missing: {cached}")
        payload,checksum=_fetch(url),_fetch(url+".CHECKSUM")
    tokens=checksum.decode("ascii").split()
    expected=tokens[0].lower() if tokens else ""
    if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Malformed exchange derivative CHECKSUM")
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("Derivative archive checksum mismatch")
    frame=parse_premium(payload) if kind=="premium" else parse_funding(payload)
    if not cached.is_file() or not checksum_path.is_file():
        _atomic_bytes(cached,payload);_atomic_bytes(checksum_path,checksum)
    return frame,{"kind":kind,"path":str(cached.relative_to(cache_dir)),"url":url,
                  "sha256":expected,"verification":"exchange_archive_sha256"}


def _gap_ranges(missing):
    missing=np.asarray(missing,dtype=np.int64)
    if not len(missing):
        return []
    parts=np.split(missing,np.flatnonzero(np.diff(missing)!=MINUTE_MS)+1)
    def utc(t):
        return datetime.fromtimestamp(int(t)/1000,timezone.utc).isoformat()
    return [{"start_ms":int(part[0]),"end_exclusive_ms":int(part[-1]+MINUTE_MS),
             "start_utc":utc(part[0]),"end_exclusive_utc":utc(part[-1]+MINUTE_MS),
             "missing_minutes":len(part)} for part in parts]


def _rest_cached(kind,symbol,start,end,cache_dir,offline=False,*,allow_gaps=False):
    """Exact original public REST pages + local hashes, never exchange-signed hashes."""
    cache=Path(cache_dir)/"derivatives"/"rest"/kind/symbol/f"{symbol}-{start}-{end}.json"
    hash_path=cache.with_suffix(".json.sha256")
    sources_path=cache.with_suffix(".json.sources.json")
    if cache.is_file() and hash_path.is_file() and sources_path.is_file():
        payload=cache.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=hash_path.read_text(encoding="ascii").strip():
            raise ValueError("Derivative REST checksum mismatch")
        observations=json.loads(payload)
        urls=json.loads(sources_path.read_text(encoding="utf-8"))
    else:
        if offline:
            raise FileNotFoundError(f"Verified derivative REST snapshot missing: {cache}")
        cursor,upper=_ms(start),_ms(end)
        observations,urls=[],[]
        page_number=0
        while cursor < upper:
            endpoint="fundingRate" if kind=="funding" else "premiumIndexKlines"
            suffix="&interval=1m" if kind=="premium" else ""
            url=(f"https://fapi.binance.com/fapi/v1/{endpoint}?symbol={symbol}{suffix}"
                 f"&startTime={cursor}&endTime={upper-1}&limit=1000")
            try:
                response=_fetch(url)
            except urllib.error.HTTPError as error:
                if error.code != 451:
                    raise
                url=url.replace("https://fapi.binance.com/","https://www.binance.com/")
                response=_fetch(url)
            chunk=json.loads(response)
            if not isinstance(chunk,list):
                raise ValueError("Malformed official derivative REST response")
            # Even the terminating empty page remains reproducible.
            page=cache.with_suffix(f".response-{page_number}.json")
            _atomic_bytes(page,response)
            _atomic_bytes(page.with_suffix(".json.sha256"),(hashlib.sha256(response).hexdigest()+"\n").encode("ascii"))
            urls.append(url);page_number+=1
            if not chunk:
                break
            if kind=="funding":
                if any(not isinstance(row,dict) or row.get("symbol")!=symbol for row in chunk):
                    raise ValueError("Funding REST symbol mismatch")
                next_cursor=int(chunk[-1]["fundingTime"])+1
            else:
                if any(not isinstance(row,list) or len(row)<5 for row in chunk):
                    raise ValueError("Malformed premium REST kline")
                next_cursor=int(chunk[-1][0])+MINUTE_MS
            if next_cursor<=cursor:
                raise ValueError("Derivative REST pagination did not advance")
            observations.extend(chunk)
            cursor=next_cursor
        payload=json.dumps(observations,separators=(",",":")).encode("utf-8")
        # Validate before caching the joined observation payload below.
    if kind=="premium":
        frame=pd.DataFrame([[row[i] for i in range(5)] for row in observations],columns=PREMIUM_COLUMNS)
        frame["timestamp"]=_timestamps(frame.timestamp)
        for column in PREMIUM_COLUMNS[1:]:
            frame[column]=pd.to_numeric(frame[column],errors="raise").astype(float)
        if allow_gaps:
            if not frame.empty:
                validate_premium(frame,continuous=False)
            if np.any(frame.timestamp<_ms(start)) or np.any(frame.timestamp>=_ms(end)):
                raise ValueError("Premium REST observation outside requested window")
        else:
            validate_premium(frame,start=start,end=end)
    else:
        frame=pd.DataFrame({"funding_time":_timestamps(pd.Series([row["fundingTime"] for row in observations])),
                            "rate":pd.to_numeric(pd.Series([row["fundingRate"] for row in observations]),errors="raise").astype(float),
                            # The funding REST endpoint does not report historical interval hours.
                            "interval_hours":np.nan})
        validate_funding(frame)
        if np.any(frame.funding_time<_ms(start)) or np.any(frame.funding_time>=_ms(end)):
            raise ValueError("Funding REST event outside requested window")
    if not cache.is_file() or not hash_path.is_file() or not sources_path.is_file():
        _atomic_bytes(cache,payload)
        _atomic_bytes(hash_path,(hashlib.sha256(payload).hexdigest()+"\n").encode("ascii"))
        _atomic_bytes(sources_path,json.dumps(urls,indent=2).encode("utf-8"))
    entry={"kind":kind,"path":str(cache.relative_to(cache_dir)),"urls":urls,
           "sha256":hashlib.sha256(payload).hexdigest(),
           "verification":"local_sha256_of_official_rest_snapshot"}
    if kind=="premium":
        missing=np.setdiff1d(np.arange(_ms(start),_ms(end),MINUTE_MS,dtype=np.int64),frame.timestamp.to_numpy(),assume_unique=True)
        entry.update(complete=not len(missing),missing_minutes=len(missing),missing_ranges=_gap_ranges(missing))
    return frame,entry


def load_premium(symbol,start,end,cache_dir,offline=False,*,allow_gaps=False):
    """Return actual premium minutes; strict by default, optional declared gaps.

    With ``allow_gaps=True`` no row is invented or forward-filled. A hashed
    ``premium_quality`` manifest explicitly lists every unfilled minute range.
    Consumers must invalidate any feature window touching those missing minutes.
    """
    start,end=_dates(symbol,start,end)
    def consume(relative):
        try:
            return [read_archive("premium",relative,cache_dir,offline)]
        except (FileNotFoundError,urllib.error.HTTPError) as error:
            if not isinstance(error,FileNotFoundError) and error.code!=404:
                raise
            if relative.startswith("daily/"):
                day=date.fromisoformat(Path(relative).stem[-10:])
                return [_rest_cached("premium",symbol,day,day+timedelta(days=1),cache_dir,offline,allow_gaps=allow_gaps)]
            month=date.fromisoformat(Path(relative).stem[-7:]+"-01")
            result=[]
            day=max(start,month)
            while day<min(end,_next_month(month)):
                daily=f"daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
                result.extend(consume(daily));day+=timedelta(days=1)
            return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups=list(pool.map(consume,archive_paths("premium",symbol,start,end)))
    chunks=[(frame,entry) for group in groups for frame,entry in group]
    frames=[frame.loc[(frame.timestamp>=_ms(start))&(frame.timestamp<_ms(end))] for frame,_ in chunks]
    combined=pd.concat(frames,ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    validate_premium(combined,continuous=False)
    expected=np.arange(_ms(start),_ms(end),MINUTE_MS,dtype=np.int64)
    missing=np.setdiff1d(expected,combined.timestamp.to_numpy(),assume_unique=True)
    sources=[entry for _,entry in chunks]
    if len(missing):
        days=sorted({datetime.fromtimestamp(int(t)/1000,timezone.utc).date() for t in missing})
        for day in days:
            # A complete official daily bundle may repair an incomplete monthly one.
            daily=f"daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip"
            repairs=[]
            try:
                repairs.append(read_archive("premium",daily,cache_dir,offline))
            except (FileNotFoundError,urllib.error.HTTPError) as error:
                if not isinstance(error,FileNotFoundError) and error.code!=404:
                    raise
            for repaired,entry in repairs:
                overlap=repaired[repaired.timestamp.isin(combined.timestamp)]
                archived=combined.set_index("timestamp").loc[overlap.timestamp]
                if not np.array_equal(archived.to_numpy(),overlap.set_index("timestamp").to_numpy()):
                    raise ValueError("Conflicting premium monthly/daily archive observations")
                combined=pd.concat([combined,repaired[repaired.timestamp.isin(missing)]],ignore_index=True)
                sources.append(entry)
            remaining=np.setdiff1d(np.arange(_ms(day),_ms(day)+DAY_MS,MINUTE_MS,dtype=np.int64),combined.timestamp.to_numpy(),assume_unique=False)
            if len(remaining):
                repaired,entry=_rest_cached("premium",symbol,day,day+timedelta(days=1),cache_dir,offline,allow_gaps=allow_gaps)
                overlap=repaired[repaired.timestamp.isin(combined.timestamp)]
                archived=combined.set_index("timestamp").loc[overlap.timestamp]
                if not np.array_equal(archived.to_numpy(),overlap.set_index("timestamp").to_numpy()):
                    raise ValueError("Conflicting premium archive and REST observations")
                combined=pd.concat([combined,repaired[repaired.timestamp.isin(remaining)]],ignore_index=True)
                sources.append(entry)
        combined=combined.sort_values("timestamp").reset_index(drop=True)
    if allow_gaps:
        validate_premium(combined,continuous=False)
        missing=np.setdiff1d(expected,combined.timestamp.to_numpy(),assume_unique=True)
        quality={"symbol":symbol,"start_utc":str(start),"end_exclusive_utc":str(end),
                 "complete":not len(missing),"expected_minutes":len(expected),"observed_minutes":len(combined),
                 "missing_minutes":len(missing),"missing_ranges":_gap_ranges(missing),
                 "interpolated_minutes":0,"policy":"actual observations only; invalidate rolling windows crossing gaps"}
        payload=json.dumps(quality,sort_keys=True,indent=2).encode("utf-8")
        path=Path(cache_dir)/"derivatives"/"quality"/f"{symbol}-premium-{start}-{end}.json"
        _atomic_bytes(path,payload)
        sources.append({"kind":"premium_quality","path":str(path.relative_to(cache_dir)),
                        "sha256":hashlib.sha256(payload).hexdigest(),
                        "verification":"local_sha256_of_explicit_gap_inventory",**quality})
    else:
        validate_premium(combined,start=start,end=end)
    return combined,sorted(sources,key=lambda x:x["path"])


def load_funding(symbol,start,end,cache_dir,offline=False):
    start,end=_dates(symbol,start,end)
    def consume(relative):
        try:
            return read_archive("funding",relative,cache_dir,offline)
        except (FileNotFoundError,urllib.error.HTTPError) as error:
            if not isinstance(error,FileNotFoundError) and error.code!=404:
                raise
            month=date.fromisoformat(Path(relative).stem[-7:]+"-01")
            return _rest_cached("funding",symbol,max(start,month),min(end,_next_month(month)),cache_dir,offline)
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks=list(pool.map(consume,archive_paths("funding",symbol,start,end)))
    frames=[frame.loc[(frame.funding_time>=_ms(start))&(frame.funding_time<_ms(end))] for frame,_ in chunks]
    combined=pd.concat(frames,ignore_index=True).sort_values("funding_time").reset_index(drop=True)
    validate_funding(combined)
    # Funding is sparse: do not synthesize an assumed eight-hour schedule.
    return combined,sorted([entry for _,entry in chunks],key=lambda x:x["path"])
