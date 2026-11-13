"""Publish an audited research overlay without replacing runtime secrets or state.

Linux root entry point. The overlay contains publication-manifest.json with
schema_version=1, files={relative_path: sha256}, and expected_runtime_hashes
covering every existing live/*.py. Optional expected_environment_sha256 and
expected_active_release bind the inspected deployment. No messages are sent.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sqlite3
import subprocess
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

MANIFEST = 'publication-manifest.json'
ROOT_ASSETS = {'index.html', 'styles.css', 'app.js', 'research.js', 'workbuddy.js',
               'trade-view.js', 'research-view.js', 'research-view.css', '.nojekyll', '.gitattributes',
               'README.md', 'requirements-live.txt', 'requirements-quantdinger-demo.txt'}
STATIC_ADDITIONS = {'demo': {'research.html'},
                    'asset': {'research-view.js', 'research-view.css'},
                    'report': {'high-win-public.json', 'high-win-public-trades.csv'}}
FORBIDDEN_PARTS = {'models', 'live-state', 'state', 'private', 'cache', '__pycache__',
                   '.git', '.private', '.cache', '.runs', '.qd-runs'}
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
LOCAL_HTTP = build_opener(ProxyHandler({}))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def safe_name(name, *, directory=False):
    """Permit only explicit publication source/asset classes, never runtime data."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Invalid archive path')
    clean = name.rstrip('/') if directory else name
    path = PurePosixPath(clean)
    if path.is_absolute() or str(path) != clean or any(p in ('', '.', '..') for p in clean.split('/')):
        raise ValueError('Noncanonical archive path')
    if any(p in FORBIDDEN_PARTS or (p.startswith('.') and clean not in {'.nojekyll', '.gitattributes'}) for p in path.parts):
        raise ValueError('Private/cache archive path rejected')
    if any(p.lower().startswith('.env') or p.lower() == 'runtime.env' for p in path.parts):
        raise ValueError('Runtime environment must not be included')
    if directory:
        if path.parts[0] not in {'demo', 'research', 'tests', 'deploy', 'live', 'scripts'}:
            raise ValueError('Unexpected publication directory')
        return clean
    if clean == MANIFEST or clean in ROOT_ASSETS or clean == 'live/app.py':
        return clean
    allowed = {'demo': {'.html'}, 'research': {'.py', '.json', '.csv', '.txt'},
               'tests': {'.py', '.cjs'}, 'deploy': {'.py', '.md'}, 'scripts': {'.py'}}
    if len(path.parts) != 2 or path.parts[0] not in allowed or path.suffix not in allowed[path.parts[0]]:
        raise ValueError('File is outside the publication allowlist')
    return clean


def inspect_bundle(bundle, expected_sha, commit):
    if not is_sha(expected_sha) or digest(bundle) != expected_sha:
        raise ValueError('Deployment archive SHA256 mismatch')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Require an exact Git commit SHA')
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 2000:
            raise ValueError('Too many archive members')
        names = set(); regular = {}; total = 0
        for member in members:
            if not member.isfile() and not member.isdir():
                raise ValueError('Links and special archive members are forbidden')
            name = safe_name(member.name, directory=member.isdir())
            if name in names:
                raise ValueError('Duplicate archive member')
            names.add(name)
            if member.isfile():
                if not 0 <= member.size <= MAX_MEMBER_BYTES:
                    raise ValueError('Archive member size exceeded')
                total += member.size
                regular[name] = member
        if total > MAX_TOTAL_BYTES or MANIFEST not in regular:
            raise ValueError('Archive size exceeded or publication manifest missing')
        if regular[MANIFEST].size > 1024*1024:
            raise ValueError('Oversized publication manifest')
        manifest = json.load(archive.extractfile(regular[MANIFEST]))
        if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
            raise ValueError('Unsupported publication manifest')
        if manifest.get('commit', commit) != commit:
            raise ValueError('Manifest Git commit differs from requested deployment')
        files = manifest['files']
        if set(files) != set(regular)-{MANIFEST}:
            raise ValueError('Manifest and archive file populations differ')
        required = {'live/app.py', 'demo/research.html', 'research-view.js', 'research-view.css',
                    'research/high-win-public.json', 'research/high-win-public-trades.csv'}
        if not required.issubset(files):
            raise ValueError('Publication overlay is missing required assets')
        for name, expected in files.items():
            safe_name(name)
            if not is_sha(expected):
                raise ValueError('Invalid per-file SHA256')
            actual = hashlib.sha256(archive.extractfile(regular[name]).read()).hexdigest()
            if actual != expected:
                raise ValueError('Publication file content hash mismatch: '+name)
        hashes = manifest.get('expected_runtime_hashes')
        if not isinstance(hashes, dict) or 'live/app.py' not in hashes:
            raise ValueError('Expected runtime hashes are required')
        for name, expected in hashes.items():
            path = PurePosixPath(name)
            if len(path.parts) != 2 or path.parts[0] != 'live' or path.suffix != '.py' or not is_sha(expected):
                raise ValueError('Invalid expected runtime hash entry')
    return manifest


def _normalize_static_tuples(text):
    tree = ast.parse(text)
    create = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_app'), None)
    if create is None:
        raise ValueError('Runtime create_app function missing')
    seen = {}
    for route in create.body:
        if not isinstance(route, ast.FunctionDef) or route.name not in STATIC_ADDITIONS:
            continue
        matches = [node for node in ast.walk(route) if isinstance(node, ast.Compare)
                   and isinstance(node.left, ast.Name) and node.left.id == 'name'
                   and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn)
                   and len(node.comparators) == 1 and isinstance(node.comparators[0], (ast.Tuple, ast.List))]
        if len(matches) != 1:
            raise ValueError('Runtime static whitelist shape changed')
        comparator = matches[0]
        values = comparator.comparators[0].elts
        if any(not isinstance(v, ast.Constant) or not isinstance(v.value, str) for v in values):
            raise ValueError('Runtime whitelist must contain literal filenames')
        names = [v.value for v in values]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate runtime static whitelist entry')
        seen[route.name] = set(names)
        comparator.comparators[0] = ast.Tuple(elts=[], ctx=ast.Load())
    if set(seen) != set(STATIC_ADDITIONS):
        raise ValueError('Expected three static whitelist routes')
    return ast.dump(tree, include_attributes=False), seen


def validate_app_change(before, after):
    old_tree, old_names = _normalize_static_tuples(before)
    new_tree, new_names = _normalize_static_tuples(after)
    if old_tree != new_tree:
        raise ValueError('Only static whitelist additions in live/app.py are allowed')
    for route in STATIC_ADDITIONS:
        if not old_names[route].issubset(new_names[route]) or new_names[route]-old_names[route] != STATIC_ADDITIONS[route]-old_names[route]:
            raise ValueError('Runtime static whitelist additions differ from approved research assets')


def runtime_hashes(release):
    directory = Path(release)/'live'
    return {p.relative_to(release).as_posix(): digest(p) for p in directory.glob('*.py') if p.is_file()}


def safe_destination(release, name):
    release = Path(release).resolve()
    target = release.joinpath(*PurePosixPath(safe_name(name)).parts)
    cursor = target
    while cursor != release:
        if cursor.is_symlink():
            raise ValueError('Overlay destination traverses an existing symlink')
        cursor = cursor.parent
    if not target.resolve().is_relative_to(release):
        raise ValueError('Overlay destination escapes the fresh release')
    return target


def apply_overlay(bundle, release, manifest):
    with tarfile.open(bundle, 'r:gz') as archive:
        for name, expected in manifest['files'].items():
            target = safe_destination(release, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(name)
            with target.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            os.chmod(target, 0o644)
            if digest(target) != expected:
                raise ValueError('Extracted publication hash mismatch: '+name)


def replace_working_directory(unit, before, after):
    section = None; found = 0; result = []
    for line in unit.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped
        if section == '[Service]' and stripped.startswith('WorkingDirectory='):
            if stripped.split('=', 1)[1] != str(before):
                raise ValueError('Service WorkingDirectory differs from inspected active release')
            ending = '\r\n' if line.endswith('\r\n') else '\n'
            line = 'WorkingDirectory='+str(after)+ending
            found += 1
        result.append(line)
    if found != 1:
        raise ValueError('Require exactly one service WorkingDirectory')
    return ''.join(result)


def sqlite_backup(source, destination, timeout=12):
    """Consistent individual audit snapshot; never restored automatically."""
    deadline = time.monotonic()+timeout
    def progress(status, remaining, total):
        if time.monotonic() >= deadline:
            raise TimeoutError('SQLite snapshot exceeded its bounded deadline')
    uri = Path(source).resolve().as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri, uri=True, timeout=3)) as incoming:
        with closing(sqlite3.connect(destination, timeout=3)) as outgoing:
            incoming.backup(outgoing, pages=256, progress=progress, sleep=.01)
            if outgoing.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('SQLite audit snapshot failed integrity check')
    os.chmod(destination, 0o600)


def immutable_state(state):
    result = {}
    pointer = Path(state)/'active-model.txt'
    if pointer.exists():
        result['active-model.txt'] = digest(pointer)
    models = Path(state)/'models'
    if models.exists():
        for item in models.rglob('*'):
            if item.is_file():
                result[item.relative_to(state).as_posix()] = digest(item)
    return result


def request(path, port=80, *, deadline=None):
    timeout = 2 if deadline is None else min(2, deadline-time.monotonic())
    if timeout <= 0:
        raise TimeoutError('Readiness deadline exhausted')
    try:
        with LOCAL_HTTP.open(f'http://127.0.0.1:{port}{path}', timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def json_response(path, port=80, *, deadline=None):
    status, body = request(path, port, deadline=deadline)
    if status != 200:
        raise RuntimeError('Readiness endpoint failed: '+path)
    return json.loads(body)


def check_live_identity(port=80, *, expected_model, deadline=None):
    health = json_response('/health', port, deadline=deadline)
    snapshot = json_response('/api/live/status', port, deadline=deadline)
    if (health.get('ok') is not True or health.get('model_loaded') is not True
            or health.get('order_execution_enabled') is not False
            or snapshot.get('order_execution_enabled') is not False
            or snapshot.get('model', {}).get('id') != expected_model):
        raise RuntimeError('Published worker/model/order execution invariant failed')
    return snapshot


def check_public(port=80, *, staged=False, expected_model=None, deadline=None):
    health = json_response('/health', port, deadline=deadline)
    if health.get('ok') is not True or health.get('order_execution_enabled') is not False:
        raise RuntimeError('Health/order execution invariant failed')
    for path in ('/demo/research.html', '/demo/trades.html', '/research-view.js', '/research-view.css'):
        status, body = request(path, port, deadline=deadline)
        if status != 200 or not body:
            raise RuntimeError('Public asset missing: '+path)
    json_response('/research/high-win-public.json', port, deadline=deadline)
    if len(json_response('/research/trade-chart-data.json', port, deadline=deadline).get('trades', [])) != 442:
        raise RuntimeError('Existing 442-trade visualization changed')
    status, body = request('/research/high-win-public-trades.csv', port, deadline=deadline)
    if status != 200 or len(list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))) != 1262:
        raise RuntimeError('Research comparator must contain exactly 1262 trades')
    for path in ('/.private/runtime.env', '/models/manifest.json', '/research/.cache/x',
                 '/live-state/worker.sqlite3', '/research/metrics-search-results.json'):
        if request(path, port, deadline=deadline)[0] != 404:
            raise RuntimeError('Private or nonpublic path was exposed: '+path)
    status, body = request('/api/live/status', port, deadline=deadline)
    snapshot = json.loads(body)
    if staged:
        if health.get('model_loaded') is not False or status != 503 or snapshot.get('notifications', {}).get('enabled') is not False:
            raise RuntimeError('Staging must run without a worker or notifications')
    elif (health.get('model_loaded') is not True or status != 200
          or snapshot.get('order_execution_enabled') is not False
          or snapshot.get('model', {}).get('id') != expected_model):
        raise RuntimeError('Published worker/model/order execution invariant failed')
    return snapshot


def wait_ready(callback, seconds):
    deadline = time.monotonic()+seconds
    last = None
    while time.monotonic() < deadline:
        try:
            return callback(deadline)
        except (OSError, URLError, RuntimeError, ValueError) as error:
            last = error
            time.sleep(max(0, min(.5, deadline-time.monotonic())))
    raise RuntimeError('Bounded readiness check failed') from last


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode:
        raise RuntimeError('Command failed: '+' '.join(args[:3]))
    return result.stdout.strip()


def atomic_file(path, content):
    path = Path(path); original = path.stat()
    temporary = path.with_name(path.name+f'.publication-{os.getpid()}.tmp')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True
            stream.write(content)
        os.chmod(temporary, original.st_mode & 0o777)
        if hasattr(os, 'chown'):
            os.chown(temporary, original.st_uid, original.st_gid)
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def atomic_link(path, target):
    path = Path(path)
    temporary = path.with_name(path.name+f'.publication-{os.getpid()}.tmp')
    created = False
    try:
        temporary.symlink_to(target, target_is_directory=True)
        created = True
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


class PublicationFailure(RuntimeError):
    def __init__(self, error, *, switched=False, rollback_error=None):
        self.failure_type = type(error).__name__
        self.rollback_failure_type = type(rollback_error).__name__ if rollback_error else None
        self.status = ('rollback_failed' if rollback_error else 'rolled_back') if switched else 'failed_before_switch'
        super().__init__('Publication '+self.status+'; failure: '+self.failure_type)


def switch_service(unit, updated, current, release, old_unit, old_link, *, model, verify_immutable):
    """Rollback the executable/unit only; SQLite remains live and is never restored."""
    switched = False
    try:
        print('SWITCH service WorkingDirectory only; restart existing worker', flush=True)
        atomic_file(unit, updated); switched = True
        command('systemctl', 'daemon-reload'); command('systemctl', 'restart', 'ecq')
        wait_ready(lambda deadline: check_public(expected_model=model, deadline=deadline), 35)
        verify_immutable()
        atomic_link(current, str(release))
    except BaseException as error:
        rollback_error = None
        if switched:
            print('ROLLBACK service unit and current link; retain evolving SQLite state', flush=True)
            try:
                atomic_file(unit, old_unit); atomic_link(current, old_link)
                command('systemctl', 'daemon-reload'); command('systemctl', 'restart', 'ecq')
                wait_ready(lambda deadline: check_live_identity(expected_model=model, deadline=deadline), 25)
            except BaseException as caught:
                rollback_error = caught
        raise PublicationFailure(error, switched=switched, rollback_error=rollback_error) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.name != 'posix' or os.geteuid() != 0:
        raise RuntimeError('This publication helper requires Linux root')
    manifest = inspect_bundle(args.bundle, args.sha256, args.commit)
    base = Path('/opt/codexquant'); releases = (base/'releases').resolve()
    current = base/'current'
    if not current.is_symlink():
        raise ValueError('The active current pointer must be a symlink')
    old_link = os.readlink(current); old = current.resolve()
    if not old.is_dir() or old.parent != releases:
        raise ValueError('Active release is outside /opt/codexquant/releases')
    if manifest.get('expected_active_release', str(old)) != str(old):
        raise ValueError('Active release changed since inspection')
    if command('systemctl', 'show', 'ecq', '-p', 'WorkingDirectory', '--value') != str(old):
        raise ValueError('Service and current symlink point to different releases')
    if command('systemctl', 'show', 'ecq', '-p', 'DropInPaths', '--value'):
        raise ValueError('Service drop-ins need explicit inspection before scoped publication')
    unit = Path(command('systemctl', 'show', 'ecq', '-p', 'FragmentPath', '--value'))
    if unit != Path('/etc/systemd/system/ecq.service') or unit.is_symlink():
        raise ValueError('Unexpected service unit path')
    if manifest.get('expected_service_unit_sha256', digest(unit)) != digest(unit):
        raise ValueError('Service unit changed since inspection')
    environment = Path('/etc/codexquant.env'); state = Path('/var/lib/codexquant')
    env_hash = digest(environment)
    if manifest.get('expected_environment_sha256', env_hash) != env_hash:
        raise ValueError('Runtime environment changed since inspection')
    if runtime_hashes(old) != manifest['expected_runtime_hashes']:
        raise ValueError('Inspected runtime source hashes no longer match')
    previous_status = json_response('/api/live/status')
    model = previous_status.get('model', {}).get('id')
    if not model or previous_status.get('order_execution_enabled') is not False:
        raise RuntimeError('Existing live model/order execution state is not ready')
    if manifest.get('expected_model_id', model) != model:
        raise RuntimeError('Active live model changed since inspection')
    required_databases = {'history.sqlite3', 'notifications.sqlite3', 'worker.sqlite3'}
    if not required_databases.issubset({p.name for p in state.glob('*.sqlite3') if p.is_file()}):
        raise RuntimeError('Required existing live databases are missing')
    original_state = immutable_state(state)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    release = releases/f'{stamp}-{args.commit[:12]}'
    backup = Path('/var/backups/codexquant')/f'publication-{stamp}-{args.commit[:12]}'
    if release.exists() or backup.exists():
        raise ValueError('Publication release/backup already exists')
    backup.mkdir(parents=True, mode=0o700); os.chmod(backup, 0o700)
    old_unit = unit.read_bytes(); shutil.copy2(unit, backup/'ecq.service')
    (backup/'current-link.txt').write_text(old_link+'\n', encoding='utf-8')
    print('SNAPSHOT service unit and independent SQLite audit backups', flush=True)
    database_names = []
    snapshot_deadline = time.monotonic()+45
    for database in sorted(state.glob('*.sqlite3')):
        if database.is_symlink():
            raise ValueError('Unexpected symlinked live database')
        remaining = snapshot_deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Database snapshot phase exceeded 45 seconds')
        sqlite_backup(database, backup/database.name, timeout=min(12, remaining))
        database_names.append(database.name)
    print('COPY current release; preserve live models and persistent state', flush=True)
    shutil.copytree(old, release, symlinks=True)
    apply_overlay(args.bundle, release, manifest)
    after_hashes = runtime_hashes(release)
    if set(after_hashes) != set(manifest['expected_runtime_hashes']):
        raise ValueError('Runtime file population changed')
    for name, expected in manifest['expected_runtime_hashes'].items():
        if name != 'live/app.py' and after_hashes[name] != expected:
            raise ValueError('A non-static runtime module changed')
    validate_app_change((old/'live/app.py').read_text(encoding='utf-8-sig'),
                        (release/'live/app.py').read_text(encoding='utf-8-sig'))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8088))
    stage_env = dict(os.environ, WECOM_WEBHOOK_URL='', CODEXQUANT_STATE_DIR=str(state),
                     CODEXQUANT_MODEL_DIR=str(state/'models'/'initial'), PYTHONUNBUFFERED='1')
    print('STAGE no-worker on 127.0.0.1:8088', flush=True)
    with (backup/'stage.log').open('wb') as log:
        process = subprocess.Popen([str(base/'venv/bin/python'), '-m', 'live.app', '--host', '127.0.0.1',
                                    '--port', '8088', '--no-worker'], cwd=release, env=stage_env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            def staged_ready(deadline):
                if process.poll() is not None:
                    raise RuntimeError('No-worker staging process exited')
                result = check_public(8088, staged=True, deadline=deadline)
                if process.poll() is not None:
                    raise RuntimeError('No-worker staging process exited during readiness')
                return result
            wait_ready(staged_ready, 25)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
    if digest(environment) != env_hash or immutable_state(state) != original_state:
        raise RuntimeError('Environment/model state changed during staging; publication aborted')
    if unit.read_bytes() != old_unit or os.readlink(current) != old_link:
        raise RuntimeError('Service definition changed during staging; publication aborted')
    updated = replace_working_directory(old_unit.decode('utf-8'), old, release).encode('utf-8')
    record = {'commit':args.commit, 'bundle_sha256':args.sha256, 'release':str(release),
              'previous_release':str(old), 'backup':str(backup), 'model_id':model,
              'environment_sha256':env_hash, 'database_audit_snapshots':database_names,
              'database_rollback_performed':False, 'notifications_sent_by_helper':False,
              'overlay_files':manifest['files']}
    def verify_immutable():
        if digest(environment) != env_hash or immutable_state(state) != original_state:
            raise RuntimeError('Environment or active model changed across publication')
    (backup/'publication.json').write_text(json.dumps(dict(record, status='ready_to_switch'), indent=2)+'\n', encoding='utf-8')
    try:
        switch_service(unit, updated, current, release, old_unit, old_link,
                       model=model, verify_immutable=verify_immutable)
    except PublicationFailure as error:
        record['status'] = error.status
        record['failure_type'] = error.failure_type
        record['rollback_failure_type'] = error.rollback_failure_type
        (backup/'publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        raise
    record['status'] = 'published'
    record['published_at_utc'] = datetime.now(timezone.utc).isoformat()
    (backup/'publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':'published','release':str(release),'backup':str(backup),
                      'commit':args.commit,'model_id':model,'environment_preserved':True,
                      'database_snapshots':len(database_names),'order_execution_enabled':False}), flush=True)


if __name__ == '__main__':
    main()
