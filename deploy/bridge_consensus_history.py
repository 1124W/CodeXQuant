"""Append recent original minute observations to an independently seeded history.

Run from the release root with ``python -m deploy.bridge_consensus_history``.
The source is read-only. Existing target rows, models, admissions and notifications
are never changed. Coverage is reported, not repaired or represented as profit.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import time

from live.features import FEATURES, SYMBOLS

MINUTE = 60_000
DAY = 86_400_000
MAX_BRIDGE_DAYS = 7
COLUMNS = ('symbol', 'signal_ms', 'settlement_ms', 'features_json', 'y',
           'is_flat', 'entry_price', 'exit_price', 'label_source')
OFFICIAL = 'official_index_observation'
SEED = 'verified_training_seed'


def minute(value, name):
    if type(value) is not int or value <= 0 or value % MINUTE:
        raise ValueError(name + ' must be a positive integer minute boundary')
    return value


def connection(path, mode):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=' + mode,
                         uri=True, timeout=15, isolation_level=None)
    db.row_factory = sqlite3.Row
    if mode == 'ro':
        db.execute('PRAGMA query_only=ON')
    return db


def validate_contract(db):
    row = db.execute("SELECT value FROM feature_history_meta WHERE name='features'").fetchone()
    if row is None or json.loads(row['value']) != FEATURES:
        raise ValueError('History feature contract differs from the frozen inputs')
    names = {row['name'] for row in db.execute('PRAGMA table_info(feature_history)')}
    if names != set(COLUMNS):
        raise ValueError('Unexpected feature history schema')


def validate_row(row, now_ms):
    if row['symbol'] not in SYMBOLS:
        raise ValueError('Unexpected history asset')
    signal = minute(row['signal_ms'], 'signal_ms')
    if row['settlement_ms'] != signal + 31 * MINUTE:
        raise ValueError('History settlement must equal signal +31 minutes')
    values = json.loads(row['features_json'])
    if (not isinstance(values, list) or len(values) != len(FEATURES)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ValueError('Invalid historical feature vector')
    if row['y'] is None:
        if any(row[name] is not None for name in ('is_flat', 'entry_price', 'exit_price', 'label_source')):
            raise ValueError('Pending observation contains a partial outcome')
        return
    if (row['label_source'] != OFFICIAL or row['y'] not in (0, 1)
            or row['is_flat'] not in (0, 1) or row['settlement_ms'] + 5000 > now_ms):
        raise ValueError('Only already observed official index labels can be bridged')
    entry, exit_price = row['entry_price'], row['exit_price']
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (entry, exit_price)):
        raise ValueError('Official labels require original finite index observations')
    if row['y'] != int(exit_price > entry) or row['is_flat'] != int(exit_price == entry):
        raise ValueError('Official label disagrees with original observed prices')


def compatible(existing, incoming):
    # A copied pending row can subsequently settle in either database. Do not
    # rewind it or enrich an existing target row; its own label worker owns it.
    for name in ('symbol', 'signal_ms', 'settlement_ms', 'features_json'):
        if existing[name] != incoming[name]:
            return False
    if existing['y'] is not None and incoming['y'] is not None:
        return all(existing[name] == incoming[name] for name in
                   ('y', 'is_flat', 'entry_price', 'exit_price', 'label_source'))
    return True


def coverage(times, start, end):
    if end < start:
        return {'expected_minutes': 0, 'observed_minutes': 0, 'coverage': None,
                'minimum_daily_coverage': None, 'maximum_missing_run_minutes': 0, 'daily': []}
    times = sorted(set(t for t in times if start <= t <= end))
    expected = (end - start) // MINUTE + 1
    counts = {}
    for stamp in times:
        counts[stamp // DAY] = counts.get(stamp // DAY, 0) + 1
    daily = []
    for day in range(start // DAY, end // DAY + 1):
        n = (min(end, (day + 1) * DAY - MINUTE) - max(start, day * DAY)) // MINUTE + 1
        count = counts.get(day, 0)
        daily.append({'date_utc': datetime.fromtimestamp(day * DAY / 1000, timezone.utc).date().isoformat(),
                      'expected_minutes': n, 'observed_minutes': count, 'coverage': count / n})
    boundary = [start - MINUTE, *times, end + MINUTE]
    missing = max((right - left) // MINUTE - 1 for left, right in zip(boundary, boundary[1:]))
    return {'expected_minutes': expected, 'observed_minutes': len(times), 'coverage': len(times) / expected,
            'minimum_daily_coverage': min(row['coverage'] for row in daily),
            'maximum_missing_run_minutes': missing, 'daily': daily}


def bridge(source, target, start_ms, end_ms, *, now_ms=None):
    start = minute(start_ms, 'start_ms')
    end = minute(end_ms, 'end_ms')
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if type(now) is not int or start > end or end > now or end - start >= MAX_BRIDGE_DAYS * DAY:
        raise ValueError('Bridge must be a past/current interval of at most seven days')
    source, target = Path(source), Path(target)
    if (not source.is_file() or not target.is_file() or source.is_symlink() or target.is_symlink()
            or source.samefile(target)):
        raise ValueError('Require different existing regular source and target databases')
    rows = []
    skipped = {symbol: 0 for symbol in SYMBOLS}
    with closing(connection(source, 'ro')) as old:
        old.execute('BEGIN')
        validate_contract(old)
        for record in old.execute('SELECT ' + ','.join(COLUMNS) + ' FROM feature_history '
                                  'WHERE signal_ms>=? AND signal_ms<=? ORDER BY signal_ms,symbol', (start, end)):
            row = dict(record)
            if row['symbol'] not in SYMBOLS:
                raise ValueError('Unexpected source history asset')
            if row['label_source'] == SEED:
                skipped[row['symbol']] += 1
                continue
            validate_row(row, now)
            rows.append(row)
    report = {'start_ms': start, 'end_ms_inclusive': end, 'checked_at_ms': now,
              'source_read_only': True, 'existing_target_rows_modified': 0, 'symbols': {}}
    with closing(connection(target, 'rw')) as new:
        new.execute('BEGIN IMMEDIATE')
        try:
            validate_contract(new)
            for symbol in SYMBOLS:
                last = new.execute('SELECT MAX(signal_ms) FROM feature_history WHERE symbol=? AND label_source=?',
                                   (symbol, SEED)).fetchone()[0]
                if last is None or start != last + MINUTE:
                    raise ValueError('Bridge must start immediately after the last dense seed row for each asset')
            additions = []
            counts = {symbol: {'inserted': 0, 'duplicates': 0, 'pending_target_kept': 0,
                               'skipped_sparse_seed_rows': skipped[symbol]} for symbol in SYMBOLS}
            for row in rows:
                old = new.execute('SELECT ' + ','.join(COLUMNS) + ' FROM feature_history '
                                  'WHERE symbol=? AND signal_ms=?', (row['symbol'], row['signal_ms'])).fetchone()
                if old is None:
                    additions.append(row)
                    counts[row['symbol']]['inserted'] += 1
                else:
                    validate_row(old, now)
                    if not compatible(old, row):
                        raise ValueError('Conflicting original observation; no rows were imported')
                    counts[row['symbol']]['duplicates'] += 1
                    if old['y'] is None and row['y'] is not None:
                        counts[row['symbol']]['pending_target_kept'] += 1
            # All source rows and all target conflicts have passed before the
            # first INSERT. Any later failure still rolls back the transaction.
            new.executemany('INSERT INTO feature_history (' + ','.join(COLUMNS) + ') VALUES (' +
                            ','.join('?' for _ in COLUMNS) + ')',
                            [tuple(row[name] for name in COLUMNS) for row in additions])
            # Include the seeded part of the first UTC day, so the 31-minute
            # seed tail is assessed against the actual full day's population.
            continuity_start = start // DAY * DAY
            settled_end = min(end, ((now - 31 * MINUTE - 1) // MINUTE) * MINUTE)
            for symbol in SYMBOLS:
                observed = new.execute('SELECT signal_ms,y FROM feature_history WHERE symbol=? '
                                       'AND signal_ms>=? AND signal_ms<=? ORDER BY signal_ms',
                                       (symbol, continuity_start, end)).fetchall()
                report['symbols'][symbol] = {**counts[symbol],
                    'continuity_start_ms': continuity_start,
                    'observations': coverage([row['signal_ms'] for row in observed], continuity_start, end),
                    'settled': coverage([row['signal_ms'] for row in observed if row['y'] is not None],
                                        continuity_start, settled_end)}
            new.commit()
        except BaseException:
            new.rollback()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--start-ms', required=True, type=int)
    parser.add_argument('--end-ms', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(bridge(args.source, args.target, args.start_ms, args.end_ms), indent=2))


if __name__ == '__main__':
    main()
