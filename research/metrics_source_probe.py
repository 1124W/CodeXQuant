"""Small read-only feasibility probe of official Binance UM metrics archives.

Lists archive object metadata and reads ten predeclared daily samples only.
No strategy, labels, live account, deployed service, or model is accessed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research'/'.cache'/'metrics-feasibility'
START, END = date(2023, 11, 9), date(2026, 11, 12)
SYMBOLS = ('BTCUSDT', 'ETHUSDT')
DAYS = ('2023-11-09', '2024-06-01', '2026-06-24', '2026-06-25', '2026-11-11')
BUCKET = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
BASE = 'https://data.binance.vision/data/futures/um/daily/metrics'
XML = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def get(url, limit=2_000_000):
    with urllib.request.urlopen(url, timeout=15) as response:
        body = response.read(limit+1)
        if len(body) > limit: raise ValueError('Response exceeds predeclared small-probe bound')
        return body


def inventory(symbol):
    prefix = f'data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-'
    query = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000',
             'start-after': prefix+str(START-timedelta(days=1))+'.zip.CHECKSUM'}
    items = {}; pages = 0
    for _ in range(6):
        raw = get(BUCKET+'?'+urllib.parse.urlencode(query))
        pages += 1
        root = ET.fromstring(raw)
        for node in root.findall('s:Contents', XML):
            key = node.findtext('s:Key', namespaces=XML)
            suffix = key.removeprefix(prefix)
            match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})\.zip(\.CHECKSUM)?', suffix)
            if match and str(START) <= match[1] < str(END):
                record = items.setdefault(match[1], {})
                record['checksum' if match[2] else 'zip'] = {
                    'key': key, 'bytes': int(node.findtext('s:Size', namespaces=XML)),
                    'last_modified': node.findtext('s:LastModified', namespaces=XML)}
        if root.findtext('s:IsTruncated', namespaces=XML) != 'true': break
        query.pop('start-after', None)
        query['continuation-token'] = root.findtext('s:NextContinuationToken', namespaces=XML)
    else:
        raise ValueError('Inventory needs more than the fixed six-page probe bound')
    expected = [str(START+timedelta(days=k)) for k in range((END-START).days)]
    zip_sizes = [entry['zip']['bytes'] for entry in items.values() if 'zip' in entry]
    result = {'symbol': symbol, 'listing_pages': pages, 'expected_days': len(expected),
            'zip_count': len(zip_sizes), 'checksum_count': sum('checksum' in e for e in items.values()),
            'missing_zip_dates': [day for day in expected if 'zip' not in items.get(day, {})],
            'missing_checksum_dates': [day for day in expected if 'checksum' not in items.get(day, {})],
            'compressed_zip_bytes': sum(zip_sizes), 'min_zip_bytes': min(zip_sizes, default=0),
            'max_zip_bytes': max(zip_sizes, default=0), 'objects': items}
    (OUT/(symbol+'-inventory.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def sample(pair):
    symbol, day = pair; name = f'{symbol}-metrics-{day}.zip'; url = f'{BASE}/{symbol}/{name}'
    path = OUT/name; checksum_path = OUT/(name+'.CHECKSUM')
    body = path.read_bytes() if path.is_file() else get(url)
    checksum = checksum_path.read_text(encoding='utf-8') if checksum_path.is_file() else get(url+'.CHECKSUM',4096).decode()
    digest = hashlib.sha256(body).hexdigest()
    if checksum.split()[0] != digest: raise ValueError('Publisher checksum mismatch: '+name)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith('.csv'): raise ValueError('Unexpected archive members')
        csv = archive.read(names[0])
    frame = pd.read_csv(io.BytesIO(csv)); times = pd.to_datetime(frame.create_time, utc=True, errors='raise')
    numbers = frame.drop(columns=['create_time', 'symbol']).apply(pd.to_numeric, errors='raise')
    offsets = times.astype('int64').to_numpy()//1_000_000_000
    delta = np.diff(offsets)
    info = {'symbol': symbol, 'file_date': day, 'url': url, 'sha256': digest,
            'publisher_checksum_verified': True, 'compressed_bytes': len(body), 'csv_bytes': len(csv),
            'columns': list(frame), 'rows': len(frame), 'first_label_utc': str(times.iloc[0]),
            'last_label_utc': str(times.iloc[-1]), 'duplicate_labels': int(times.duplicated().sum()),
            'min_label_utc': str(times.min()), 'max_label_utc': str(times.max()),
            'input_order_monotonic': bool(times.is_monotonic_increasing),
            'sorted_non_300_second_steps': int((np.diff(np.sort(offsets)) != 300).sum()),
            'off_five_minute_grid': int((offsets%300 != 0).sum()),
            'non_300_second_steps': int((delta != 300).sum()),
            'missing_numeric_values': int(numbers.isna().sum().sum()),
            'nonfinite_numeric_values': int((~np.isfinite(numbers.to_numpy())).sum()),
            'zero_values_by_column': {name: int((numbers[name] == 0).sum()) for name in numbers},
            'valid_asset_tags': bool((frame.symbol == symbol).all())}
    path.write_bytes(body); checksum_path.write_text(checksum, encoding='utf-8')
    return info, frame


def main(cached_samples_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'start_utc': str(START), 'end_exclusive_utc': str(END),
            'sample_dates': list(DAYS), 'symbols': list(SYMBOLS),
            'full_archive_download': False, 'strategy_evaluation': False,
            'official_bucket_verified_from': 'https://data.binance.vision/',
            'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUT/'probe-plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    with ThreadPoolExecutor(max_workers=2) as pool:
        inventories = list(pool.map(inventory, SYMBOLS))
        requested = [(symbol, day) for symbol in SYMBOLS for day in DAYS]
        available = [pair for pair in requested if not cached_samples_only
                     or ((OUT/f'{pair[0]}-metrics-{pair[1]}.zip').is_file()
                         and (OUT/f'{pair[0]}-metrics-{pair[1]}.zip.CHECKSUM').is_file())]
        sampled = list(pool.map(sample, available))
    transitions = []
    for symbol in SYMBOLS:
        if not all(any(info['symbol']==symbol and info['file_date']==day for info, _ in sampled)
                   for day in ('2026-06-24', '2026-06-25')):
            transitions.append({'symbol': symbol, 'status': 'Not checked: a sample archive was not verified'})
            continue
        older = next(frame for info, frame in sampled if info['symbol']==symbol and info['file_date']=='2026-06-24')
        newer = next(frame for info, frame in sampled if info['symbol']==symbol and info['file_date']=='2026-06-25')
        overlap = older.merge(newer, on=['create_time', 'symbol'], suffixes=('_older', '_newer'))
        differing = []
        for _, row in overlap.iterrows():
            fields = [name for name in older if name not in ('create_time','symbol') and row[name+'_older'] != row[name+'_newer']]
            differing.append({'create_time': row.create_time, 'differing_fields': fields})
        transitions.append({'symbol': symbol, 'shared_timestamp_rows': len(overlap), 'conflicts': differing})
    result = {'plan': plan, 'inventories': inventories, 'samples': [info for info, _ in sampled],
              'unverified_samples': [pair for pair in requested if pair not in available],
              'adjacent_transition_archives': transitions,
              'limitations': ['Listing presence is not validation of every daily CSV.',
                  'The ten sample archives do not establish complete three-year timestamp or numeric quality.',
                  'Archive create_time and object LastModified do not establish earliest public availability or row measurement semantics.',
                  'No timestamp shifting, interpolation, feature production, training or outcome evaluation was performed.']}
    (ROOT/'research'/'metrics-source-feasibility.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'inventories': [{k:v for k,v in item.items() if k!='objects'} for item in inventories],
                      'samples': result['samples'], 'adjacent_transition_archives': transitions}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-samples-only', action='store_true')
    main(parser.parse_args().cached_samples_only)
