"""Offline rollout boundaries: private models, exact state preservation and recovery."""
from contextlib import closing
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from deploy import publish_consensus as publication

COMMIT='a'*40
MODEL='b'*20


class ConsensusPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def bundle(self,*,extra=None,mutate_model=None,mutate_manifest=None,member_type=None,duplicate=False):
        bodies={name:('test-fixture:'+name).encode() for name in publication.REQUIRED}
        model={'schema_version':2,'rule_id':publication.RULE,'model_version':MODEL,
               'order_execution_enabled':False,'models':{}}
        sha=lambda body:hashlib.sha256(body).hexdigest()
        for symbol in ('BTCUSDT','ETHUSDT'):
            seed=f'{symbol}-training.csv.gz'
            model['models'][symbol]={'training_samples_file':seed,
                'training_samples_sha256':sha(bodies['models/'+seed]),'members':{}}
            for member in ('logistic_1m','gbm_deeper_5m'):
                name=f'{symbol}-{member}.joblib'
                model['models'][symbol]['members'][member]={'file':name,'sha256':sha(bodies['models/'+name])}
        if mutate_model:mutate_model(model)
        bodies['models/manifest.json']=json.dumps(model).encode()
        if extra:bodies[extra]=b'unapproved-fixture'
        manifest={'kind':'consensus_rollout_v1','commit':COMMIT,'new_model_id':MODEL,
                  'files':{name:sha(body) for name,body in bodies.items()}}
        if mutate_manifest:mutate_manifest(manifest)
        entries=list({**bodies,publication.MANIFEST:json.dumps(manifest).encode()}.items())
        if duplicate:entries.append(entries[0])
        archive=self.root/'bundle.tgz'
        with tarfile.open(archive,'w:gz') as output:
            for name,body in entries:
                entry=tarfile.TarInfo(name)
                if member_type and name=='models/manifest.json':
                    entry.type=member_type;entry.linkname='/private-target';output.addfile(entry)
                else:
                    entry.size=len(body);output.addfile(entry,io.BytesIO(body))
        return archive,sha(archive.read_bytes()),manifest

    def test_reviewed_overlay_contains_exactly_seven_isolated_model_files(self):
        bundle,sha,manifest=self.bundle()
        self.assertEqual(publication.inspect_bundle(bundle,sha,COMMIT),manifest)
        self.assertEqual(len(publication.MODEL_FILES),7)
        self.assertEqual(len(set(manifest['files'])&publication.PRIVATE_FILES),7)
        for expected,commit in [('0'*64,COMMIT),(sha,'main'),(sha,'c'*40)]:
            with self.assertRaises(ValueError):publication.inspect_bundle(bundle,expected,commit)

    def test_archive_rejects_traversal_unknown_private_files_duplicate_and_links(self):
        for path in ('../models/manifest.json','/models/manifest.json','models/../manifest.json',
                     '.private/runtime.env','models/extra.joblib','live/predictor.py','live\\app.py'):
            bundle,sha,_=self.bundle(extra=path)
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Unapproved'):
                publication.inspect_bundle(bundle,sha,COMMIT)
        bundle,sha,_=self.bundle(duplicate=True)
        with self.assertRaises(ValueError):publication.inspect_bundle(bundle,sha,COMMIT)
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
            bundle,sha,_=self.bundle(member_type=kind)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'member type'):
                publication.inspect_bundle(bundle,sha,COMMIT)

    def test_model_identity_rule_seed_and_member_hashes_fail_closed(self):
        mutations=[lambda m:m.update(rule_id='other'),lambda m:m.update(model_version='c'*20),
                   lambda m:m.update(order_execution_enabled=True),lambda m:m.update(schema_version=1),
                   lambda m:m['models'].pop('ETHUSDT'),
                   lambda m:m['models']['BTCUSDT'].update(training_samples_file='../seed.csv.gz'),
                   lambda m:m['models']['BTCUSDT'].update(training_samples_sha256='0'*64),
                   lambda m:m['models']['BTCUSDT']['members'].pop('logistic_1m'),
                   lambda m:m['models']['BTCUSDT']['members']['logistic_1m'].update(file='different.joblib'),
                   lambda m:m['models']['BTCUSDT']['members']['logistic_1m'].update(sha256='0'*64)]
        for mutate in mutations:
            bundle,sha,_=self.bundle(mutate_model=mutate)
            with self.assertRaises(ValueError):publication.inspect_bundle(bundle,sha,COMMIT)
        bundle,sha,_=self.bundle(mutate_manifest=lambda m:m['files'].update({'live/app.py':'0'*64}))
        with self.assertRaisesRegex(ValueError,'Checksum'):publication.inspect_bundle(bundle,sha,COMMIT)

    def test_environment_changes_only_strategy_model_stake_hold_public_url_and_daily_default(self):
        old=(b'# original\r\nWECOM_WEBHOOK_URL=opaque-existing-value\r\n'
             b'CODEXQUANT_STATE_DIR=/var/lib/codexquant\r\nCODEXQUANT_STRATEGY=legacy\r\n'
             b'CODEXQUANT_MODEL_DIR=/old/model\r\nCODEXQUANT_MODEL_DIR=/duplicate\r\n'
             b'CODEXQUANT_PAPER_STAKE=5\r\nCODEXQUANT_PAPER_DAILY_LIMIT=25\r\n'
             b'CODEXQUANT_PUBLIC_BASE_URL=http://old-host\r\nOTHER=a=b=c\r\n')
        actual=publication.updated_environment(old,'/new/isolated-model').decode()
        for fragment in ('# original\n','WECOM_WEBHOOK_URL=opaque-existing-value\n',
                         'CODEXQUANT_STATE_DIR=/var/lib/codexquant\n','OTHER=a=b=c\n'):
            self.assertIn(fragment,actual)
        self.assertIn('CODEXQUANT_STRATEGY=consensus\n',actual)
        self.assertIn('CODEXQUANT_MODEL_DIR=/new/isolated-model\n',actual)
        self.assertIn('CODEXQUANT_PAPER_STAKE=30\n',actual)
        self.assertIn('CODEXQUANT_PUBLIC_BASE_URL=https://nightsleep.eu.org\n',actual)
        self.assertNotIn('CODEXQUANT_PAPER_DAILY_LIMIT=',actual)
        self.assertEqual(actual.count('CODEXQUANT_MODEL_DIR='),1)
        self.assertEqual(publication.updated_environment(actual.encode(),'/new/isolated-model'),actual.encode())

    def public_fixture(self):
        bodies={name:('public:'+name).encode() for name in publication.ASSETS}
        manifest={'new_model_id':MODEL,'files':{name:hashlib.sha256(body).hexdigest() for name,body in bodies.items()}}
        snapshot={'model':{'id':MODEL,'rule_id':publication.RULE},'publication_hold':True,
                  'paper_performance':{'stake_usdt':30,'daily_loss_limit_usdt':150,
                                       'current_strategy':{'rule_id':publication.RULE}},
                  'historical_research':{'available':True,'matches_current_strategy':True}}
        def request(path,*args,**kwargs):
            name=path.lstrip('/')
            return (200,bodies[name]) if name in bodies else (404,b'')
        return manifest,snapshot,request

    def test_public_readiness_binds_new_files_and_model_rule_with_admissions_held(self):
        manifest,snapshot,request=self.public_fixture()
        with patch.object(publication.common,'check_public',return_value=snapshot) as checked, \
             patch.object(publication.common,'request',side_effect=request):
            self.assertEqual(publication.verify_public(manifest),snapshot)
        self.assertEqual(checked.call_args.kwargs['expected_model'],MODEL)
        for mutate in (lambda s:s.update(publication_hold=False),
                       lambda s:s['model'].update(rule_id='old-rule'),
                       lambda s:s['paper_performance'].update(stake_usdt=5),
                       lambda s:s['paper_performance']['current_strategy'].update(rule_id='old-rule'),
                       lambda s:s['historical_research'].update(matches_current_strategy=False)):
            value=deepcopy(snapshot);mutate(value)
            with patch.object(publication.common,'check_public',return_value=value), \
                 patch.object(publication.common,'request',side_effect=request),self.assertRaises(RuntimeError):
                publication.verify_public(manifest)

    def test_staging_requires_no_worker_and_refuses_private_routes_or_changed_assets(self):
        manifest,snapshot,request=self.public_fixture()
        with patch.object(publication.common,'check_public',return_value={}) as checked, \
             patch.object(publication.common,'request',side_effect=request):
            self.assertEqual(publication.verify_public(manifest,8088,staged=True),{})
        self.assertTrue(checked.call_args.kwargs['staged'])
        for path in ('/models/manifest.json','/live/consensus.py','/demo/consensus.html'):
            def broken(url,*args,**kwargs):
                return (200,b'wrong') if url==path else request(url,*args,**kwargs)
            with patch.object(publication.common,'check_public',return_value=snapshot), \
                 patch.object(publication.common,'request',side_effect=broken),self.assertRaises(RuntimeError):
                publication.verify_public(manifest)

    def test_member_write_refuses_symlink_before_touching_target(self):
        bundle,_,manifest=self.bundle();target=self.root/'target'/'consensus.py'
        with tarfile.open(bundle) as archive, patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(ValueError,'Symlinked'):
                publication.write_member(archive,'live/consensus.py',target,manifest['files']['live/consensus.py'])
        self.assertFalse(target.exists())

    def rollback_fixture(self):
        environment=self.root/'runtime.env';environment.write_bytes(b'new-consensus-env\n')
        unit=self.root/'ecq.service';unit.write_bytes(b'new-unit\n')
        backup=self.root/'backup';backup.mkdir()
        database=self.root/'worker.sqlite3'
        with closing(sqlite3.connect(database)) as con:
            con.execute('CREATE TABLE admissions(symbol TEXT,signal_ms INTEGER,stake INTEGER,net REAL)')
            con.execute('INSERT INTO admissions VALUES(?,?,?,?)',('BTCUSDT',1,30,25.5));con.commit()
        return database,dict(unit=unit,environment=environment,current=self.root/'current',
            old_unit=b'old-amount-aware-unit\n',old_env=b'old-stake30-env\n',old_link='old-release',
            old_model='old-model',backup=backup,record={'database_rollback_performed':False})

    def rollback(self,fixture,*,stopped=True,ready_error=None):
        hold=self.root/'publication-hold'
        def retain():hold.write_text('held',encoding='utf-8')
        with patch.object(publication,'HOLD',hold), \
             patch.object(publication.previous,'retain_publication_hold',side_effect=retain), \
             patch.object(publication.previous,'recovery_service_stopped',return_value=stopped), \
             patch.object(publication.common,'atomic_link') as link, \
             patch.object(publication.common,'command') as commands, \
             patch.object(publication.common,'wait_ready',side_effect=ready_error), \
             patch.object(publication.common,'sqlite_backup') as database_writer:
            publication.rollback(ValueError('new strategy readiness failed'),**fixture)
        database_writer.assert_not_called()
        audit=json.loads((fixture['backup']/'publication.json').read_text())
        self.assertEqual(audit,fixture['record']);self.assertFalse(audit['database_rollback_performed'])
        return audit,hold,link,commands

    def test_rollback_restores_amount_aware_code_without_rewinding_new_or_old_paper_rows(self):
        database,fixture=self.rollback_fixture();before=database.read_bytes()
        audit,hold,link,commands=self.rollback(fixture)
        self.assertEqual(audit['status'],'rolled_back_without_database_rewind')
        self.assertEqual(database.read_bytes(),before)
        self.assertEqual(fixture['environment'].read_bytes(),fixture['old_env'])
        self.assertEqual(fixture['unit'].read_bytes(),fixture['old_unit'])
        self.assertFalse(hold.exists());link.assert_called_once_with(fixture['current'],fixture['old_link'])
        self.assertEqual([call.args[1] for call in commands.call_args_list],['daemon-reload','start'])

    def test_unconfirmed_stop_preserves_new_config_and_hold_with_audit(self):
        database,fixture=self.rollback_fixture();before=database.read_bytes()
        audit,hold,link,commands=self.rollback(fixture,stopped=False)
        self.assertEqual(audit['status'],'recovery_required_service_not_stopped')
        self.assertEqual(database.read_bytes(),before)
        self.assertEqual(fixture['environment'].read_bytes(),b'new-consensus-env\n')
        self.assertTrue(hold.exists());link.assert_not_called();commands.assert_not_called()

    def test_restore_or_readiness_failure_still_keeps_hold_and_failure_audit(self):
        database,fixture=self.rollback_fixture();before=database.read_bytes()
        audit,hold,_,_=self.rollback(fixture,ready_error=RuntimeError('old worker unavailable'))
        self.assertEqual(audit['status'],'recovery_failed_hold_retained')
        self.assertEqual(audit['recovery_failure_type'],'RuntimeError')
        self.assertEqual(database.read_bytes(),before);self.assertTrue(hold.exists())

    def test_old_model_guard_ignores_only_the_independent_new_consensus_models(self):
        state=self.root/'state';(state/'models'/'old').mkdir(parents=True)
        old=state/'models/old/model.joblib';old.write_bytes(b'immutable-old')
        (state/'active-model.txt').write_text('models/old',encoding='utf-8')
        expected=publication.common.immutable_state(state)
        (state/'models-consensus/new').mkdir(parents=True)
        (state/'models-consensus/new/member.joblib').write_bytes(b'new-consensus')
        self.assertEqual(publication.common.immutable_state(state),expected)
        old.write_bytes(b'changed-old')
        self.assertNotEqual(publication.common.immutable_state(state),expected)


if __name__=='__main__':unittest.main()
