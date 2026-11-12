"""Download public Binance research observations into the project cache.

No account keys, trading endpoints, orders, or uploads are used.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.qd_demo import write_atomic
from research.qd_three_year_data import _fetch, load_market


def inspect_formats(cache):
    base = "https://data.binance.vision/data/futures/um/"
    result = []
    for relative in ("monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2026-10.zip",
                     "monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-10.zip"):
        payload = _fetch(base + relative)
        checksum = _fetch(base + relative + ".CHECKSUM")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != checksum.decode().split()[0]:
            raise ValueError("Exchange archive checksum mismatch")
        path = cache / "derivatives" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.with_suffix(".zip.CHECKSUM").write_bytes(checksum)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            with archive.open(archive.namelist()[0]) as stream:
                lines = [stream.readline().decode().strip() for _ in range(3)]
        row = {"url":base+relative,"sha256":digest,"bytes":len(payload),"first_lines":lines}
        result.append(row)
        print(json.dumps(row),flush=True)
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--component",choices=("klines","derivatives","premium","funding","inspect"),default="klines")
    cli.add_argument("--symbols",nargs="+",choices=("BTCUSDT","ETHUSDT"),default=["BTCUSDT","ETHUSDT"])
    cli.add_argument("--start",type=date.fromisoformat,default=date(2026,11,9))
    cli.add_argument("--end",type=date.fromisoformat,default=datetime.now(timezone.utc).date())
    args=cli.parse_args()
    if args.start>=args.end or args.end>datetime.now(timezone.utc).date():
        raise ValueError("Only complete historical UTC days may be requested")
    cache=ROOT/"research"/".cache"
    if args.component=="inspect":
        inspect_formats(cache)
        return
    results={}
    for symbol in args.symbols:
        if args.component=="klines":
            spot,index,sources=load_market(symbol,args.start,args.end,cache,False)
            results[symbol]={"spot_minutes":len(spot),"index_minutes":len(index),
                             "first_open_ms":int(index.timestamp.iloc[0]),
                             "last_open_ms":int(index.timestamp.iloc[-1]),"sources":sources}
        else:
            from research.derivatives_data import load_premium,load_funding
            results[symbol]={}
            if args.component in ("derivatives","premium"):
                # Actual exchange gaps are recorded; feature windows crossing them
                # are excluded downstream. No missing minute is interpolated.
                premium,premium_sources=load_premium(symbol,args.start,args.end,cache,False,allow_gaps=True)
                results[symbol].update(premium_minutes=len(premium),premium_sources=premium_sources)
            if args.component in ("derivatives","funding"):
                funding,funding_sources=load_funding(symbol,args.start,args.end,cache,False)
                results[symbol].update(funding_records=len(funding),funding_sources=funding_sources)
        print(json.dumps({"symbol":symbol,**{k:v for k,v in results[symbol].items() if not k.endswith('sources')} }),flush=True)
    summary={"start_utc":str(args.start),"end_exclusive_utc":str(args.end),
             "component":args.component,"downloaded_at_utc":datetime.now(timezone.utc).isoformat(),
             "only_public_market_data":True,"results":results}
    write_atomic(ROOT/"research"/f"market-update-{args.component}.json",json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
