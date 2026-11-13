"""Scoped stake publication boundaries without networking or service changes."""
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from deploy import publish_stake as publication

COMMIT='a'*40


class StakePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def bundle(self, *, extra=None, invalid_hash=False, member_type=None):
        files={name:('fixture:'+name).encode() for name in publication.REQUIRED}
        if extra is not None:files[extra]=b'private-fixture'
        manifest={'kind':'stake_update_v1','commit':COMMIT,
                  'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()}}
        if invalid_hash:manifest['files']['live/app.py']='0'*64
        package=self.root/'overlay.tgz'
        with tarfile.open(package,'w:gz') as archive:
            for name,content in {**files,publication.MANIFEST:json.dumps(manifest).encode()}.items():
                entry=tarfile.TarInfo(name)
                if member_type is not None and name=='live/app.py':
                    entry.type=member_type;entry.linkname='outside';entry.size=0
                    archive.addfile(entry)
                else:
                    entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
        return package,hashlib.sha256(package.read_bytes()).hexdigest(),manifest

    def test_valid_scoped_overlay_and_identity(self):
        package,sha,expected=self.bundle()
        self.assertEqual(publication.inspect_bundle(package,sha,COMMIT),expected)
        for digest,commit in (('0'*64,COMMIT),(sha,'b'*40),(sha,'main')):
            with self.subTest(digest=digest,commit=commit),self.assertRaises(ValueError):
                publication.inspect_bundle(package,digest,commit)

    def test_archive_forbids_private_traversal_and_unrelated_runtime(self):
        for name in ('.private/runtime.env','../live/app.py','/live/app.py',
                     'live/predictor.py','research/.cache/input.npz','live\\app.py'):
            with self.subTest(name=name):
                package,sha,_=self.bundle(extra=name)
                with self.assertRaisesRegex(ValueError,'Unexpected'):
                    publication.inspect_bundle(package,sha,COMMIT)

    def test_bad_file_hash_and_archive_links_are_rejected(self):
        package,sha,_=self.bundle(invalid_hash=True)
        with self.assertRaisesRegex(ValueError,'checksum'):
            publication.inspect_bundle(package,sha,COMMIT)
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
            with self.subTest(kind=kind):
                package,sha,_=self.bundle(member_type=kind)
                with self.assertRaisesRegex(ValueError,'Links'):
                    publication.inspect_bundle(package,sha,COMMIT)

    def test_environment_preserves_robot_and_only_changes_stake_hold_daily_auto(self):
        old=(b'# deployment settings\r\nWECOM_WEBHOOK_URL=opaque-original-value\r\n'
             b'CODEXQUANT_PAPER_STAKE=5\r\nCODEXQUANT_PAPER_STAKE=10\r\n'
             b'CODEXQUANT_PAPER_DAILY_LIMIT=25\r\nCODEXQUANT_PAPER_DAILY_LIMIT=50\r\n'
             b'CODEXQUANT_MODEL_DIR=/var/lib/codexquant/models/initial\r\n'
             b'CODEXQUANT_STATE_DIR=/var/lib/codexquant\r\nOTHER=keep=all=parts\r\n')
        actual=publication.updated_environment(old).decode()
        self.assertIn('WECOM_WEBHOOK_URL=opaque-original-value\n',actual)
        self.assertIn('CODEXQUANT_MODEL_DIR=/var/lib/codexquant/models/initial\n',actual)
        self.assertIn('CODEXQUANT_STATE_DIR=/var/lib/codexquant\n',actual)
        self.assertIn('OTHER=keep=all=parts\n',actual)
        self.assertNotIn('CODEXQUANT_PAPER_DAILY_LIMIT=',actual)
        self.assertEqual(actual.count('CODEXQUANT_PAPER_STAKE='),1)
        self.assertIn('CODEXQUANT_PAPER_STAKE=30\n',actual)
        self.assertEqual(actual.count('CODEXQUANT_PUBLICATION_HOLD_FILE='),1)
        self.assertIn('CODEXQUANT_PUBLICATION_HOLD_FILE='+str(publication.HOLD)+'\n',actual)
        self.assertEqual(publication.updated_environment(actual.encode()),actual.encode())

    def database(self):
        path=self.root/'worker.sqlite3'
        with closing(sqlite3.connect(path)) as con:
            con.execute('CREATE TABLE admissions(symbol TEXT,signal_ms INTEGER,entry_ms INTEGER,'
                        'settlement_ms INTEGER,direction INTEGER,outcome TEXT,net REAL)')
            con.execute('INSERT INTO admissions VALUES (?,?,?,?,?,?,?)',
                        ('BTCUSDT',1000,61000,1861000,1,'win',4.25))
            con.execute('INSERT INTO admissions VALUES (?,?,?,?,?,?,?)',
                        ('ETHUSDT',2000,62000,1862000,-1,None,None))
            con.commit()
        return path

    def test_legacy_seven_column_rows_are_read_without_migration_or_rewrite(self):
        path=self.database();before=path.read_bytes()
        self.assertFalse(publication.has_variable_terms(path))
        self.assertEqual(publication.preserved_settled_rows(path),
                         [('BTCUSDT',1000,61000,1861000,1,'win',4.25)])
        self.assertEqual(path.read_bytes(),before)

    def test_any_nonlegacy_term_prevents_fixed_five_dollar_rollback(self):
        path=self.database()
        with closing(sqlite3.connect(path)) as con:
            con.execute('CREATE TABLE admission_terms(symbol TEXT,signal_ms INTEGER,stake_cents INTEGER)')
            con.execute('INSERT INTO admission_terms VALUES (?,?,?)',('BTCUSDT',1000,500));con.commit()
        self.assertFalse(publication.has_variable_terms(path))
        with closing(sqlite3.connect(path)) as con:
            con.execute('INSERT INTO admission_terms VALUES (?,?,?)',('ETHUSDT',2000,3000));con.commit()
        before=path.read_bytes()
        self.assertTrue(publication.has_variable_terms(path))
        self.assertEqual(path.read_bytes(),before)

    def test_overlay_refuses_symlink_in_destination_chain_before_write(self):
        package,_,manifest=self.bundle()
        release=self.root/'release';release.mkdir()
        blocked=release/'live'
        real_is_symlink=Path.is_symlink
        # Windows test environments may lack symlink creation permission. Exercise
        # the same detected chain through a path-specific filesystem observation.
        with patch.object(Path,'is_symlink',lambda path: path==blocked or real_is_symlink(path)):
            with self.assertRaisesRegex(ValueError,'Symlinked'):
                publication.apply_overlay(package,release,manifest)
        self.assertFalse((release/'live/app.py').exists())

    def recovery_fixture(self):
        environment=self.root/'runtime.env';environment.write_bytes(b'new-stake=30\n')
        unit=self.root/'ecq.service';unit.write_bytes(b'new-unit\n')
        backup=self.root/'backup';backup.mkdir()
        return dict(stopped=True,switched=True,record={'database_rollback_performed':False},
                    database=self.database(),environment=environment,unit=unit,
                    current=self.root/'current',old_env=b'old-stake=5\n',old_unit=b'old-unit\n',
                    old_link='old-release',expected_model='unchanged-model',backup=backup)

    def recover(self,fixture,*,state='ActiveState=inactive\nSubState=dead\nMainPID=0',
                stop_timeout=False,show_error=False,ready_error=None):
        def command(*args):
            if args[1]=='stop' and stop_timeout:
                raise subprocess.TimeoutExpired(args,10)
            if args[1]=='show':
                if show_error:raise RuntimeError('state probe failed')
                return state
            return ''
        hold=self.root/'publication-hold'
        with patch.object(publication,'HOLD',hold), \
             patch.object(publication.common,'command',side_effect=command) as commands, \
             patch.object(publication.common,'atomic_link') as links, \
             patch.object(publication.common,'wait_ready',side_effect=ready_error) as ready:
            publication.recover_failure(ValueError('readiness failed'),**fixture)
        audit=json.loads((fixture['backup']/'publication.json').read_text())
        self.assertEqual(audit,fixture['record'])
        self.assertEqual(audit['failure_type'],'ValueError')
        self.assertFalse(audit['database_rollback_performed'])
        return audit,commands,links,ready

    def test_stop_timeout_recovers_only_after_systemd_confirms_inactive(self):
        fixture=self.recovery_fixture();before=fixture['database'].read_bytes()
        audit,commands,links,ready=self.recover(fixture,stop_timeout=True)
        self.assertEqual(audit['status'],'rolled_back_without_database_rewind')
        self.assertEqual(audit['recovery_stop_failure_type'],'TimeoutExpired')
        self.assertEqual(fixture['environment'].read_bytes(),fixture['old_env'])
        self.assertEqual(fixture['unit'].read_bytes(),fixture['old_unit'])
        self.assertEqual(fixture['database'].read_bytes(),before)
        self.assertFalse((self.root/'publication-hold').exists())
        links.assert_called_once_with(fixture['current'],fixture['old_link'])
        ready.assert_called_once()
        self.assertEqual([call.args[1] for call in commands.call_args_list],
                         ['stop','show','daemon-reload','start'])

    def test_unconfirmed_stop_never_rewrites_unit_environment_or_link(self):
        fixture=self.recovery_fixture();before=fixture['database'].read_bytes()
        for state in ('ActiveState=active\nSubState=running\nMainPID=123',
                      'ActiveState=deactivating\nSubState=stop-sigterm\nMainPID=123',
                      'ActiveState=failed\nSubState=failed\nMainPID=123',
                      'ActiveState=inactive\nSubState=dead',
                      'ActiveState=inactive\nSubState=running\nMainPID=0'):
            with self.subTest(state=state):
                audit,commands,links,ready=self.recover(fixture,state=state,stop_timeout=True)
                self.assertEqual(audit['status'],'recovery_required_service_not_stopped')
                self.assertEqual(fixture['environment'].read_bytes(),b'new-stake=30\n')
                self.assertEqual(fixture['unit'].read_bytes(),b'new-unit\n')
                self.assertEqual(fixture['database'].read_bytes(),before)
                self.assertTrue((self.root/'publication-hold').is_file())
                links.assert_not_called();ready.assert_not_called()
                self.assertEqual([call.args[1] for call in commands.call_args_list],['stop','show'])

    def test_state_probe_failure_is_audited_with_hold_and_without_rollback(self):
        fixture=self.recovery_fixture()
        audit,commands,links,ready=self.recover(fixture,stop_timeout=True,show_error=True)
        self.assertEqual(audit['status'],'recovery_required_service_not_stopped')
        self.assertEqual(audit['recovery_state_failure_type'],'RuntimeError')
        self.assertEqual(fixture['environment'].read_bytes(),b'new-stake=30\n')
        self.assertEqual(fixture['unit'].read_bytes(),b'new-unit\n')
        self.assertTrue((self.root/'publication-hold').is_file())
        links.assert_not_called();ready.assert_not_called()

    def test_variable_terms_keep_amount_aware_unit_and_reload_before_restart(self):
        fixture=self.recovery_fixture()
        with closing(sqlite3.connect(fixture['database'])) as con:
            con.execute('CREATE TABLE admission_terms(stake_cents INTEGER)')
            con.execute('INSERT INTO admission_terms VALUES(3000)');con.commit()
        before=fixture['database'].read_bytes()
        audit,commands,links,ready=self.recover(
            fixture,state='ActiveState=failed\nSubState=failed\nMainPID=0')
        self.assertEqual(audit['status'],'held_for_amount_safe_recovery')
        self.assertEqual(fixture['environment'].read_bytes(),b'new-stake=30\n')
        self.assertEqual(fixture['unit'].read_bytes(),b'new-unit\n')
        self.assertEqual(fixture['database'].read_bytes(),before)
        self.assertTrue((self.root/'publication-hold').is_file())
        links.assert_not_called();ready.assert_called_once()
        self.assertEqual([call.args[1] for call in commands.call_args_list],
                         ['stop','show','daemon-reload','start'])

    def test_restore_exception_still_writes_failure_audit_and_retains_hold(self):
        fixture=self.recovery_fixture();before=fixture['database'].read_bytes()
        with patch.object(publication.common,'atomic_file',side_effect=OSError('write failed')):
            audit,commands,links,ready=self.recover(fixture)
        self.assertEqual(audit['status'],'recovery_failed_hold_retained')
        self.assertEqual(audit['recovery_failure_type'],'OSError')
        self.assertTrue((self.root/'publication-hold').is_file())
        self.assertEqual(fixture['database'].read_bytes(),before)
        links.assert_not_called();ready.assert_not_called()

    def test_restored_service_readiness_failure_is_audited_and_keeps_hold(self):
        fixture=self.recovery_fixture();before=fixture['database'].read_bytes()
        audit,_,_,_=self.recover(fixture,ready_error=RuntimeError('readiness failed'))
        self.assertEqual(audit['status'],'recovery_failed_hold_retained')
        self.assertEqual(audit['recovery_failure_type'],'RuntimeError')
        self.assertTrue((self.root/'publication-hold').is_file())
        self.assertEqual(fixture['database'].read_bytes(),before)

    def test_failure_before_stop_is_audited_without_service_or_config_changes(self):
        fixture=self.recovery_fixture();fixture['stopped']=False
        audit,commands,links,ready=self.recover(fixture)
        self.assertEqual(audit['status'],'failed_before_stop')
        self.assertEqual(fixture['environment'].read_bytes(),b'new-stake=30\n')
        self.assertEqual(fixture['unit'].read_bytes(),b'new-unit\n')
        self.assertFalse((self.root/'publication-hold').exists())
        commands.assert_not_called();links.assert_not_called();ready.assert_not_called()


if __name__=='__main__':unittest.main()
