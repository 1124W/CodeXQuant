"""Offline checks for scoped research deployment and state-preserving rollback."""
import hashlib
import importlib.util
import io
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('publish_research', ROOT/'deploy'/'publish_research.py')
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)
COMMIT = 'a'*40
MODEL = '9cd43159f5a66dca527e'


def sha(body):
    return hashlib.sha256(body).hexdigest()


def app_source(extra=False):
    additions = pub.STATIC_ADDITIONS if extra else {key: set() for key in pub.STATIC_ADDITIONS}
    text = 'def create_app(worker=None):\n'
    for route, names in additions.items():
        text += f'    def {route}(name):\n        if name not in {tuple(sorted(names | {"old.txt"}))!r}: return 404\n        return name\n'
    return text+'    return worker\n'


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, *, extra=None, alter_manifest=None, extra_member=None):
        files = {name: b'content' for name in ('live/app.py', 'demo/research.html',
                 'research-view.js', 'research-view.css', 'research/high-win-public.json',
                 'research/high-win-public-trades.csv')}
        files.update(extra or {})
        manifest = {'schema_version': 1, 'files': {key: sha(value) for key, value in files.items()},
                    'expected_runtime_hashes': {'live/app.py': sha(b'previous')}, 'commit': COMMIT}
        if alter_manifest:
            alter_manifest(manifest)
        archive = self.root/'publication.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            for name, body in {**files, pub.MANIFEST: json.dumps(manifest).encode()}.items():
                info = tarfile.TarInfo(name); info.size = len(body)
                output.addfile(info, io.BytesIO(body))
            if extra_member:
                info, body = extra_member
                output.addfile(info, io.BytesIO(body) if body else None)
        return archive, pub.digest(archive), manifest

    def test_valid_bundle_allows_public_scripts_and_gitattributes(self):
        archive, digest, expected = self.bundle(extra={'scripts/build_high_win_public.py': b'pass',
                                                      '.gitattributes': b'*.json text eol=lf\n'})
        self.assertEqual(pub.inspect_bundle(archive, digest, COMMIT), expected)
        release = self.root/'release'; release.mkdir()
        pub.apply_overlay(archive, release, expected)
        for name, value in expected['files'].items():
            self.assertEqual(pub.digest(release/name), value)

    def test_rejects_unsafe_paths_and_state(self):
        for name in ('../live/app.py', '/live/app.py', 'live//app.py', 'C:/x',
                     'research\\x.py', '.env', 'research/.env', 'models/model.pkl',
                     'research/.cache/x.json', 'live-state/worker.sqlite3',
                     'live/predictor.py', 'research/subdir/x.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pub.safe_name(name)

    def test_rejects_links_duplicates_unmanifested_files(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            info = tarfile.TarInfo('research/link.json'); info.type = kind; info.linkname = '/etc/passwd'
            archive, digest, _ = self.bundle(extra_member=(info, b''))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Links'):
                pub.inspect_bundle(archive, digest, COMMIT)
        for name, expected in (('research/unlisted.json', 'populations'), ('live/app.py', 'Duplicate')):
            info = tarfile.TarInfo(name); info.size = 1
            archive, digest, _ = self.bundle(extra_member=(info, b'x'))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                pub.inspect_bundle(archive, digest, COMMIT)

    def test_rejects_hash_and_commit_mismatches(self):
        archive, digest, _ = self.bundle(alter_manifest=lambda m: m['files'].update({'live/app.py': '0'*64}))
        with self.assertRaisesRegex(ValueError, 'content hash'):
            pub.inspect_bundle(archive, digest, COMMIT)
        archive, digest, _ = self.bundle()
        with self.assertRaisesRegex(ValueError, 'archive SHA256'):
            pub.inspect_bundle(archive, '0'*64, COMMIT)
        with self.assertRaisesRegex(ValueError, 'commit differs'):
            pub.inspect_bundle(archive, digest, 'b'*40)

    def test_rejects_existing_destination_symlink(self):
        release = self.root/'release'; release.mkdir()
        outside = self.root/'outside'; outside.mkdir()
        try:
            (release/'research').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Creating symlinks is not permitted on this host')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            pub.safe_destination(release, 'research/high-win-public.json')

    def test_app_change_accepts_only_required_static_additions(self):
        before = app_source(); after = app_source(True)
        pub.validate_app_change(before, after)
        pub.validate_app_change(after, after)
        for changed in (after.replace('return worker', 'return None'),
                        after.replace("'research.html'", "'runtime.env'"),
                        after.replace("'old.txt', ", '')):
            with self.assertRaises(ValueError):
                pub.validate_app_change(before, changed)

    def test_service_rewrite_preserves_other_settings(self):
        original = '[Unit]\nDescription=Test\n[Service]\nEnvironmentFile=/etc/codexquant.env\nWorkingDirectory=/old\nExecStart=/venv/bin/python -m live.app\n'
        actual = pub.replace_working_directory(original, '/old', '/new')
        self.assertEqual(actual, original.replace('WorkingDirectory=/old', 'WorkingDirectory=/new'))
        for bad in (original.replace('/old', '/drift'), original+'WorkingDirectory=/old\n',
                    original.replace('WorkingDirectory=/old\n', '')):
            with self.assertRaises(ValueError):
                pub.replace_working_directory(bad, '/old', '/new')

    def test_sqlite_snapshot_does_not_rewind_live_updates(self):
        database = self.root/'worker.sqlite3'; snapshot = self.root/'snapshot.sqlite3'
        with closing(sqlite3.connect(database, isolation_level=None)) as connection:
            connection.execute('CREATE TABLE events (value INTEGER)')
            connection.execute('INSERT INTO events VALUES (1)')
        pub.sqlite_backup(database, snapshot)
        with closing(sqlite3.connect(database, isolation_level=None)) as connection:
            connection.execute('INSERT INTO events VALUES (2)')
        with closing(sqlite3.connect(snapshot)) as connection:
            self.assertEqual(connection.execute('SELECT value FROM events').fetchall(), [(1,)])
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(connection.execute('SELECT value FROM events').fetchall(), [(1,), (2,)])

    def test_atomic_write_cleans_temporary_after_failure(self):
        unit = self.root/'ecq.service'; unit.write_bytes(b'old')
        with patch.object(pub.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                pub.atomic_file(unit, b'new')
        self.assertEqual(unit.read_bytes(), b'old')
        self.assertEqual(list(self.root.glob('*.tmp')), [])
        pub.atomic_file(unit, b'new')
        self.assertEqual(unit.read_bytes(), b'new')

    def test_failed_publication_restores_unit_and_link_preserving_live_database(self):
        unit = self.root/'ecq.service'; unit.write_bytes(b'old unit')
        database = self.root/'worker.sqlite3'
        with closing(sqlite3.connect(database, isolation_level=None)) as connection:
            connection.execute('CREATE TABLE events (value INTEGER)')
            connection.execute('INSERT INTO events VALUES (1)')
        def rejected_health(**kwargs):
            with closing(sqlite3.connect(database, isolation_level=None)) as connection:
                connection.execute('INSERT INTO events VALUES (2)')
            raise RuntimeError('new release failed health')
        def immediate(callback, seconds):
            return callback(time.monotonic()+seconds)
        with patch.object(pub, 'command') as command, patch.object(pub, 'atomic_link') as link, \
             patch.object(pub, 'wait_ready', side_effect=immediate), \
             patch.object(pub, 'check_public', side_effect=rejected_health), \
             patch.object(pub, 'check_live_identity', return_value={'model': {'id': MODEL}}) as rollback:
            with self.assertRaises(pub.PublicationFailure) as failure:
                pub.switch_service(unit, b'new unit', self.root/'current', '/new', b'old unit', '/old',
                                   model=MODEL, verify_immutable=lambda: None)
        self.assertEqual(failure.exception.status, 'rolled_back')
        self.assertEqual(unit.read_bytes(), b'old unit')
        link.assert_called_once_with(self.root/'current', '/old')
        rollback.assert_called_once()
        self.assertEqual(command.call_count, 4)
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(connection.execute('SELECT value FROM events').fetchall(), [(1,), (2,)])

    def responses(self, staged=False):
        json_bytes = lambda value: json.dumps(value).encode()
        response = {'/health': (200, json_bytes({'ok': True, 'model_loaded': not staged,
                                                'order_execution_enabled': False})),
                    '/research/high-win-public.json': (200, b'{}'),
                    '/research/trade-chart-data.json': (200, json_bytes({'trades': [{}]*442})),
                    '/research/high-win-public-trades.csv': (200, b'symbol\n'+b'BTC\n'*1262),
                    '/api/live/status': (503 if staged else 200, json_bytes({'model': {'id': MODEL},
                        'notifications': {'enabled': not staged}, 'order_execution_enabled': False}))}
        for path in ('/demo/research.html', '/demo/trades.html', '/research-view.js', '/research-view.css'):
            response[path] = (200, b'asset')
        return response

    def test_readiness_enforces_public_counts_private_paths_and_no_worker(self):
        for staged in (False, True):
            response = self.responses(staged)
            def local(path, *args, **kwargs):
                return response.get(path, (404, b''))
            with self.subTest(staged=staged), patch.object(pub, 'request', side_effect=local):
                pub.check_public(staged=staged, expected_model=MODEL, deadline=time.monotonic()+5)
                response['/research/high-win-public-trades.csv'] = (200, b'symbol\nBTC\n')
                with self.assertRaisesRegex(RuntimeError, '1262'):
                    pub.check_public(staged=staged, expected_model=MODEL)
            response = self.responses(staged); response['/.private/runtime.env'] = (200, b'private')
            with patch.object(pub, 'request', side_effect=local), self.assertRaisesRegex(RuntimeError, 'exposed'):
                pub.check_public(staged=staged, expected_model=MODEL)

    def test_identity_rejects_order_execution_or_model_drift(self):
        response = self.responses()
        response['/api/live/status'] = (200, json.dumps({'model': {'id': 'changed'},
                                                      'order_execution_enabled': False}).encode())
        with patch.object(pub, 'request', side_effect=lambda path, *a, **k: response[path]), \
             self.assertRaisesRegex(RuntimeError, 'invariant'):
            pub.check_live_identity(expected_model=MODEL)

    def test_expired_request_never_opens_network(self):
        with patch.object(pub.LOCAL_HTTP, 'open') as opened, self.assertRaises(TimeoutError):
            pub.request('/health', deadline=time.monotonic()-1)
        opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
