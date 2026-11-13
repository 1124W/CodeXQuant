"""Verified local Binance UM open-interest/positioning observations.

Only the explicit download command uses public HTTP. Research loaders never
fetch, interpolate, round timestamps, or infer historical publication times.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'research'/'.cache'/'metrics'
PROBE = ROOT/'research'/'.cache'/'metrics-feasibility'
START, END = date(2023, 11, 9), date(2026, 11, 12)
SYMBOLS = ('BTCUSDT', 'ETHUSDT')
METRIC_COLUMNS = ['sum_open_interest', 'sum_open_interest_value',
                  'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                  'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
CSV_COLUMNS = ['create_time', 'symbol']+METRIC_COLUMNS
FRAME_COLUMNS = ['record_ms', 'symbol']+METRIC_COLUMNS
BASE = 'https://data.binance.vision/data/futures/um/daily/metrics'
TRANSPORT_BASE = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/daily/metrics'
MAX_ARCHIVE_BYTES = 2_000_000
MAX_CSV_BYTES = 2_000_000
MAX_REQUESTS = 2*len(SYMBOLS)*(END-START).days+128


def _sha(data): return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(content); temporary.replace(path)


def store_identity(manifest):
    return _sha(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def source_hash(): return _sha(Path(__file__).read_bytes())


def _ms(day):
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp()*1000)


def expected_files():
    return [(symbol, START+timedelta(days=k)) for k in range((END-START).days) for symbol in SYMBOLS]


def _names(symbol, day):
    if symbol not in SYMBOLS or not isinstance(day, date) or not START <= day < END:
        raise ValueError('Archive request outside the fixed BTC/ETH date scope')
    name = f'{symbol}-metrics-{day}.zip'
    return name, f'{symbol}/{name}', f'{symbol}/{name}.CHECKSUM'


def verify_bytes(body, checksum, name):
    if not body or len(body) > MAX_ARCHIVE_BYTES:
        raise ValueError('Invalid archive byte count')
    lines = checksum.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('Require one publisher checksum record')
    fields = lines[0].split()
    if len(fields) != 2 or not re.fullmatch(r'[0-9a-fA-F]{64}', fields[0]) or fields[1].lstrip('*') != name:
        raise ValueError('Publisher checksum schema or filename mismatch')
    digest = _sha(body)
    if digest.lower() != fields[0].lower():
        raise ValueError('Publisher archive checksum mismatch')
    return digest


def clean_rows(frame):
    """Drop every conflicting duplicate before checking numeric eligibility."""
    if set(frame) != set(FRAME_COLUMNS): raise ValueError('Unexpected metrics frame schema')
    if not frame.symbol.isin(SYMBOLS).all(): raise ValueError('Unsupported metrics asset')
    values = frame.record_ms.to_numpy()
    if (not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values % 300000)):
        raise ValueError('Metrics frame clocks must already be exact five-minute observations')
    duplicates = frame.duplicated(['symbol', 'record_ms'], keep=False)
    conflicts = set()
    if duplicates.any():
        unique = frame.loc[duplicates].groupby(['symbol', 'record_ms'])[METRIC_COLUMNS].nunique(dropna=False)
        conflicts = set(unique.index[(unique > 1).any(axis=1)].tolist())
    conflict_mask = np.array([(symbol, int(stamp)) in conflicts for symbol, stamp in zip(frame.symbol, frame.record_ms)], dtype=bool)
    remaining = frame.loc[~conflict_mask]
    deduplicated = remaining.drop_duplicates(['symbol', 'record_ms'], keep='first')
    numeric = deduplicated[METRIC_COLUMNS].to_numpy(dtype=float)
    valid = np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
    clean = deduplicated.loc[valid, FRAME_COLUMNS].sort_values(['record_ms', 'symbol']).reset_index(drop=True)
    return clean, {'raw_rows': len(frame), 'conflicting_duplicate_timestamps': len(conflicts),
                   'conflicting_duplicate_rows_dropped': int(conflict_mask.sum()),
                   'identical_duplicate_rows_removed': len(remaining)-len(deduplicated),
                   'nonpositive_or_nonfinite_rows_dropped': int((~valid).sum()),
                   'accepted_rows': len(clean),
                   'conflict_keys': [{'symbol': symbol, 'record_ms': int(stamp)} for symbol, stamp in sorted(conflicts)]}


def parse_archive(body, checksum, symbol, day):
    """Parse originals; retain invalid numeric rows until cross-file deduplication.

    Both historical label conventions are admitted: the file day's midnight
    through the next midnight inclusive. No timing convention is guessed.
    """
    name, _, _ = _names(symbol, day); digest = verify_bytes(body, checksum, name)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        items = archive.infolist()
        if (len(items) != 1 or items[0].filename != name.removesuffix('.zip')+'.csv'
                or items[0].file_size > MAX_CSV_BYTES):
            raise ValueError('Unexpected metrics archive members or expanded size')
        frame = pd.read_csv(io.BytesIO(archive.read(items[0])), dtype={'create_time': str, 'symbol': str})
    if list(frame) != CSV_COLUMNS or frame.empty:
        raise ValueError('Missing or changed official metrics CSV schema')
    if not (frame.symbol == symbol).all(): raise ValueError('Archive contains a different asset')
    stamp = pd.to_datetime(frame.create_time, format='%Y-%m-%d %H:%M:%S', utc=True, errors='coerce')
    numbers = frame[METRIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
    times = stamp.astype('int64').to_numpy()//1_000_000
    valid_time = stamp.notna().to_numpy() & (times % 300000 == 0) & (times >= _ms(day)) & (times <= _ms(day+timedelta(days=1)))
    raw = numbers.loc[valid_time].copy()
    raw.insert(0, 'symbol', symbol); raw.insert(0, 'record_ms', times[valid_time].astype(np.int64))
    raw = raw[FRAME_COLUMNS].reset_index(drop=True)
    cleaned, quality = clean_rows(raw)
    quality.update({'archive_rows': len(frame), 'invalid_or_offgrid_or_outside_day_rows_dropped': int((~valid_time).sum()),
                    'original_row_order_monotonic': bool(stamp.is_monotonic_increasing),
                    'first_record_ms': int(cleaned.record_ms.min()) if len(cleaned) else None,
                    'last_record_ms': int(cleaned.record_ms.max()) if len(cleaned) else None,
                    'within_file_missing_grid_points': int(((cleaned.record_ms.diff().dropna()/300000)-1).clip(lower=0).sum())})
    return raw, {'sha256': digest, 'bytes': len(body), 'quality': quality}


def _contract():
    return {'schema_version': 1, 'source_code_sha256': source_hash(), 'symbols': list(SYMBOLS),
            'start_utc': str(START), 'end_exclusive_utc': str(END), 'columns': CSV_COLUMNS,
            'clean_policy': 'positive finite values; exact observed 5-minute timestamps; all conflicting duplicates dropped; identical duplicates collapsed; no interpolation',
            'publication_time_semantics': 'unverified; raw record times only; availability buffering belongs to frozen feature protocol',
            'historical_revision_caveat': 'Archive Last-Modified describes current object publication/revision, not earliest original API availability; historical revisions and release latency are unverified. A record+10min feature buffer is an explicit research assumption, not point-in-time proof.'}


def verify_metrics_store(cache_dir=None, verify_files=True):
    folder = Path(cache_dir or CACHE)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if any(manifest.get(key) != value for key, value in _contract().items()) or manifest.get('complete') is not True:
        raise ValueError('Metrics manifest is incomplete or source/schema/date contract changed')
    expected = {f'{symbol}:{day}' for symbol, day in expected_files()}
    if set(manifest.get('files', {})) != expected: raise ValueError('Metrics manifest archive coverage is incomplete')
    for symbol, day in expected_files():
        name, binary_name, checksum_name = _names(symbol, day)
        entry = manifest['files'][f'{symbol}:{day}']
        if entry.get('file') != binary_name or entry.get('checksum_file') != checksum_name:
            raise ValueError('Metrics manifest contains an unexpected archive path')
        if verify_files:
            body = (folder/binary_name).read_bytes(); checksum = (folder/checksum_name).read_text(encoding='utf-8')
            if verify_bytes(body, checksum, name) != entry.get('sha256') or len(body) != entry.get('bytes'):
                raise ValueError('Metrics manifest content hash mismatch')
    return manifest


def load_metrics(symbol, start, end, cache_dir=None):
    """Strictly offline source verification; no as-of filling or lag inference."""
    if symbol not in SYMBOLS or not isinstance(start, date) or not isinstance(end, date) or start >= end:
        raise ValueError('Require one BTC/ETH asset and increasing UTC dates')
    folder = Path(cache_dir or CACHE); manifest = verify_metrics_store(folder, verify_files=False)
    first, stop = max(start, START), min(end, END)
    # Include the preceding archive to detect any midnight conflict at the
    # requested boundary, then restrict observations to the requested window.
    source_first = max(first-timedelta(days=1), START)
    parts, sources, archive_quality = [], {}, []
    if first < stop:
        for k in range((stop-source_first).days):
            day = source_first+timedelta(days=k); name, binary_name, checksum_name = _names(symbol, day)
            body = (folder/binary_name).read_bytes(); checksum = (folder/checksum_name).read_text(encoding='utf-8')
            frame, meta = parse_archive(body, checksum, symbol, day)
            if meta['sha256'] != manifest['files'][f'{symbol}:{day}']['sha256']:
                raise ValueError('Requested metrics archive changed after manifest publication')
            parts.append(frame); sources[name] = meta['sha256']; archive_quality.append(meta['quality'])
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FRAME_COLUMNS)
    if raw.empty:
        clean = pd.DataFrame({name: pd.Series(dtype='object' if name=='symbol' else 'float64') for name in FRAME_COLUMNS})
        quality = {'raw_rows': 0, 'accepted_rows': 0, 'conflict_keys': []}
    else:
        clean, quality = clean_rows(raw)
        clean = clean.loc[(clean.record_ms >= _ms(first)) & (clean.record_ms < _ms(stop))].reset_index(drop=True)
    quality.update({'symbol': symbol, 'requested_start_utc': str(start), 'requested_end_exclusive_utc': str(end),
                    'effective_start_utc': str(first), 'effective_end_exclusive_utc': str(stop),
                    'clipped_start': first != start, 'clipped_end': stop != end,
                    'returned_rows': len(clean), 'archive_quality': archive_quality,
                    'manifest_identity': store_identity(manifest), 'source_sha256': sources})
    return clean, quality


class FetchState:
    def __init__(self, max_requests=MAX_REQUESTS):
        self.lock = threading.Lock(); self.stop = threading.Event(); self.requests = 0
        self.max_requests = max_requests; self.errors = []; self.tls_failures = deque(); self.headers = {}
        self.local = threading.local(); self.sessions = []

    def _session(self):
        if not hasattr(self.local, 'session'):
            session = requests.Session()
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0, pool_connections=1, pool_maxsize=1))
            self.local.session = session
            with self.lock: self.sessions.append(session)
        return self.local.session

    def close(self):
        for session in self.sessions: session.close()

    def get(self, url, limit):
        if not url.startswith(BASE+'/'): raise ValueError('Only the fixed official public metrics source may be fetched')
        transport_url = TRANSPORT_BASE+url.removeprefix(BASE)
        for attempt in range(3):
            if self.stop.is_set(): raise RuntimeError('Download stopped after a transport or validation failure')
            with self.lock:
                if self.requests >= self.max_requests: raise RuntimeError('Bounded HTTP request budget exhausted')
                self.requests += 1
            try:
                with self._session().get(transport_url, timeout=(8,15), stream=True, verify=True) as response:
                    response.raise_for_status()
                    parts=[]; length=0
                    for part in response.iter_content(chunk_size=65536):
                        length += len(part)
                        if length > limit: raise ValueError('Public archive response exceeded byte limit')
                        parts.append(part)
                    content = b''.join(parts)
                    if len(content) > limit: raise ValueError('Public archive response exceeded byte limit')
                    with self.lock:
                        self.headers[url] = {key: response.headers.get(key) for key in ('Last-Modified','ETag','Content-Length','Date')}
                    return content
            except requests.exceptions.RequestException as error:
                text = str(error)
                certificate = 'CERTIFICATE_VERIFY_FAILED' in text or 'certificate verify failed' in text.lower()
                fatal_http = (isinstance(error, requests.exceptions.HTTPError) and error.response is not None
                              and error.response.status_code not in (408,429,500,502,503,504))
                tls = any(word in text.lower() for word in ('ssl', 'tls', 'handshake'))
                with self.lock:
                    self.errors.append({'url': url, 'transport_url': transport_url, 'attempt': attempt+1,
                                        'error': type(error).__name__, 'detail': text})
                    if tls:
                        now = time.monotonic(); self.tls_failures.append(now)
                        while self.tls_failures and self.tls_failures[0] < now-60: self.tls_failures.popleft()
                        if len(self.tls_failures) >= 3: self.stop.set()
                if certificate or fatal_http or attempt == 2 or self.stop.is_set():
                    self.stop.set(); raise
                time.sleep(.5*(attempt+1))
        raise RuntimeError('Unreachable retry state')


def _obtain(pair, folder, state, probe_dir=PROBE, inventory_entry=None):
    symbol, day = pair; name, binary_name, checksum_name = _names(symbol, day)
    target, checksum_target = folder/binary_name, folder/checksum_name
    origin = 'downloaded'
    if target.is_file() and checksum_target.is_file():
        body, checksum = target.read_bytes(), checksum_target.read_text(encoding='utf-8'); origin = 'verified_cache'
    elif (Path(probe_dir)/name).is_file() and (Path(probe_dir)/(name+'.CHECKSUM')).is_file():
        body = (Path(probe_dir)/name).read_bytes()
        checksum = (Path(probe_dir)/(name+'.CHECKSUM')).read_text(encoding='utf-8'); origin = 'verified_probe'
    else:
        url = f'{BASE}/{symbol}/{name}'
        body = state.get(url, MAX_ARCHIVE_BYTES)
        checksum = state.get(url+'.CHECKSUM', 4096).decode('utf-8')
    _, meta = parse_archive(body, checksum, symbol, day)
    if origin != 'verified_cache':
        _atomic_bytes(target, body); _atomic_bytes(checksum_target, checksum.encode('utf-8'))
    source_url = f'{BASE}/{symbol}/{name}'
    return f'{symbol}:{day}', {'file': binary_name, 'checksum_file': checksum_name, **meta,
                              'source_url': source_url, 'http_archive_headers': state.headers.get(source_url),
                              'download_transport_url': TRANSPORT_BASE+source_url.removeprefix(BASE) if origin=='downloaded' else None,
                              'listing_archive_last_modified': (inventory_entry or {}).get('zip',{}).get('last_modified'),
                              'last_modified_is_original_api_availability': False}, origin


def download_all(cache_dir=None, workers=4, batch_new_archives=None):
    """Explicit bounded fetch; exact frozen 1,099 UTC days for both assets."""
    if workers not in (2,4,8) or (workers==2 and batch_new_archives!=50):
        raise ValueError('Use four/eight workers, or two workers for exactly a fifty-new-archive probe')
    if batch_new_archives not in (None,50): raise ValueError('Only the fixed fifty-archive transport probe is supported')
    folder = Path(cache_dir or CACHE); folder.mkdir(parents=True, exist_ok=True)
    began = time.monotonic(); contract = _contract()
    previous = json.loads((folder/'manifest.json').read_text(encoding='utf-8')) if (folder/'manifest.json').exists() else {}
    previous_requests = previous.get('transport',{}).get('cumulative_http_requests',previous.get('transport',{}).get('requests',0))
    prior_attempts = previous.get('prior_attempts',[])
    if previous:
        prior_attempts = prior_attempts+[{'source_code_sha256': previous.get('source_code_sha256'),
            'manifest_sha256': _sha((folder/'manifest.json').read_bytes()), 'verified_records':len(previous.get('files',{})),
            'transport':previous.get('transport'),'failure':previous.get('failure')}]
    tasks=[]; new_tasks=0
    for pair in expected_files():
        name,binary,checksum=_names(*pair)
        cached=((folder/binary).is_file() and (folder/checksum).is_file())
        probe=((PROBE/name).is_file() and (PROBE/(name+'.CHECKSUM')).is_file())
        if cached or probe or batch_new_archives is None or new_tasks<batch_new_archives:
            tasks.append(pair)
            if not cached and not probe:new_tasks+=1
    plan = {**contract, 'expected_archives': 2198, 'scheduled_archives':len(tasks), 'workers': workers,
            'transport_probe_new_archives':batch_new_archives, 'connect_timeout_seconds':8, 'read_timeout_seconds': 15,
            'max_transient_retries_per_request': 2, 'max_http_requests': MAX_REQUESTS,
            'tls_circuit_breaker': 'stop after three SSL/TLS failures within sixty seconds',
            'local_only': True, 'source': BASE, 'transport_source':TRANSPORT_BASE,
            'transport_policy':'per-worker persistent requests.Session; official same public bucket; verify=True; no hidden adapter retries',
            'full_download_explicitly_authorized': True,
            'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    _write_json(folder/'download-plan.json', plan)
    manifest = {**contract, 'complete': False, 'files': {}, 'prior_attempts':prior_attempts,
                'download_plan_sha256': _sha((folder/'download-plan.json').read_bytes())}
    _write_json(folder/'manifest.json', manifest)
    state = FetchState(max_requests=MAX_REQUESTS-previous_requests)
    counts = {'downloaded': 0, 'verified_cache': 0, 'verified_probe': 0}
    inventories = {}
    for symbol in SYMBOLS:
        path = PROBE/(symbol+'-inventory.json')
        if path.is_file(): inventories[symbol] = json.loads(path.read_text(encoding='utf-8'))['objects']
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(_obtain, pair, folder, state, PROBE,
               inventories.get(pair[0],{}).get(str(pair[1]))): pair for pair in tasks}
    last_report = began; error = None
    try:
        for future in as_completed(futures):
            key, record, origin = future.result()
            prior_record=previous.get('files',{}).get(key,{})
            if origin=='verified_cache' and prior_record.get('sha256')==record['sha256']:
                for name in ('http_archive_headers','download_transport_url'):
                    if not record.get(name):record[name]=prior_record.get(name)
            manifest['files'][key] = record; counts[origin] += 1
            count = len(manifest['files']); now = time.monotonic()
            if count % 25 == 0 or now-last_report >= 30:
                manifest['progress'] = {'counts': dict(counts), 'requests': state.requests, 'elapsed_seconds': round(now-began, 2)}
                _write_json(folder/'manifest.json', manifest)
            if count % 100 == 0 or now-last_report >= 30:
                print(f'METRICS {count}/{len(tasks)} scheduled verified; full scope=2198; requests={state.requests}; elapsed={now-began:.1f}s; {counts}', flush=True)
                last_report = now
    except Exception as exception:
        error = exception; state.stop.set()
        for future in futures: future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        state.close()
    manifest['transport'] = {'requests': state.requests, 'cumulative_http_requests':previous_requests+state.requests,
                             'errors': state.errors, 'counts': counts, 'workers':workers,
                             'source':TRANSPORT_BASE,'certificate_verification':True,
                             'elapsed_seconds': round(time.monotonic()-began, 2)}
    if source_hash() != contract['source_code_sha256']:
        error = RuntimeError('Downloader source changed during the frozen fetch')
    if error is not None:
        manifest['failure'] = {'type': type(error).__name__, 'message': str(error)}
        _write_json(folder/'manifest.json', manifest)
        raise error
    if len(manifest['files']) != 2198 and batch_new_archives is not None:
        manifest['transport_probe_completed']=True
        _write_json(folder/'manifest.json',manifest)
        print(json.dumps({'complete':False,'transport_probe_completed':True,'archives':len(manifest['files']),
                          'transport':manifest['transport']},ensure_ascii=False),flush=True)
        return manifest
    if len(manifest['files']) != 2198: raise RuntimeError('Archive count differs from fixed download plan')
    manifest['complete'] = True; manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    _write_json(folder/'manifest.json', manifest)
    verify_metrics_store(folder, verify_files=True)
    print(json.dumps({'complete': True, 'archives': len(manifest['files']), 'identity': store_identity(manifest),
                      'transport': manifest['transport']}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', choices=(2,4,8), type=int, default=4)
    parser.add_argument('--batch-new-archives',choices=(50,),type=int)
    args = parser.parse_args()
    if args.download: download_all(workers=args.workers,batch_new_archives=args.batch_new_archives)
    else:
        checked = verify_metrics_store()
        print(json.dumps({'archives': len(checked['files']), 'identity': store_identity(checked)}))
