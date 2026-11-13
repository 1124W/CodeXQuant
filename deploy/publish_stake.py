"""Scoped stake/UI publication with a hold until migration checks pass.

Does not send a deployment notification, replace models, or rewind databases.
After new-sized admissions exist, an old fixed-5U worker is not a safe rollback.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import tarfile

try:
    from . import publish_research as common
except ImportError:
    import publish_research as common

MANIFEST='stake-publication-manifest.json'
ALLOWED={'index.html','trade-view.js','demo/trades.html','demo/research.html','demo/recent.html',
         'recent-view.js','recent-view.css','research/recent-public.json','research/recent-public-trades.csv',
         'live/app.py','live/notify.py','live/stakes.py','README.md','deploy/README.md',
         'deploy/publish_stake.py','scripts/package_live.py'}
REQUIRED={'demo/recent.html','recent-view.js','recent-view.css','research/recent-public.json',
          'research/recent-public-trades.csv','live/app.py','live/notify.py','live/stakes.py'}
HOLD=Path('/var/lib/codexquant/publication-hold')


def inspect_bundle(path,sha,commit):
    if common.digest(path)!=sha or not common.is_sha(sha) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Publication identity mismatch')
    with tarfile.open(path,'r:gz') as archive:
        entries=archive.getmembers()
        names=[entry.name for entry in entries]
        if len(names)!=len(set(names)) or set(names)-ALLOWED-{MANIFEST}:
            raise ValueError('Unexpected or duplicate overlay path')
        if any(not e.isfile() or not 0<=e.size<=16*1024*1024 for e in entries):
            raise ValueError('Links, directories, special or oversized files are not allowed')
        if sum(e.size for e in entries)>64*1024*1024 or MANIFEST not in names:
            raise ValueError('Invalid publication package')
        manifest=json.load(archive.extractfile(MANIFEST))
        if manifest.get('kind')!='stake_update_v1' or manifest.get('commit')!=commit:
            raise ValueError('Wrong manifest or commit')
        hashes=manifest.get('files',{})
        if set(hashes)!=set(names)-{MANIFEST} or not REQUIRED.issubset(hashes):
            raise ValueError('Incomplete publication manifest')
        for name,expected in hashes.items():
            if not common.is_sha(expected) or hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=expected:
                raise ValueError('File checksum mismatch: '+name)
    return manifest


def updated_environment(original):
    updates={'CODEXQUANT_PAPER_STAKE':'30','CODEXQUANT_PUBLICATION_HOLD_FILE':str(HOLD)}
    lines=[];seen=set()
    for line in original.decode('utf-8-sig').splitlines():
        key=line.split('=',1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if key=='CODEXQUANT_PAPER_DAILY_LIMIT':
            # The agreed budget follows the stake (five times), including future edits.
            continue
        if key in updates:
            if key not in seen:lines.append(key+'='+updates[key]);seen.add(key)
        else:lines.append(line)
    lines.extend(k+'='+v for k,v in updates.items() if k not in seen)
    return ('\n'.join(lines)+'\n').encode('utf-8')


def apply_overlay(bundle, release, manifest):
    release=Path(release).resolve()
    with tarfile.open(bundle,'r:gz') as archive:
        for name,expected in manifest['files'].items():
            if name not in ALLOWED:raise ValueError('Unapproved destination')
            target=release/name;cursor=target
            while cursor!=release:
                if cursor.is_symlink():raise ValueError('Symlinked overlay destination')
                cursor=cursor.parent
            if not target.resolve().is_relative_to(release):raise ValueError('Escaping destination')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('wb') as stream:shutil.copyfileobj(archive.extractfile(name),stream)
            os.chmod(target,0o644)
            if common.digest(target)!=expected:raise ValueError('Extracted checksum mismatch')


def preserved_settled_rows(database):
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as con:
        return con.execute('SELECT symbol,signal_ms,entry_ms,settlement_ms,direction,outcome,net '
                           'FROM admissions WHERE outcome IS NOT NULL ORDER BY signal_ms,symbol').fetchall()


def has_variable_terms(database):
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='admission_terms'").fetchone():return False
        return bool(con.execute('SELECT 1 FROM admission_terms WHERE stake_cents!=500 LIMIT 1').fetchone())


def check_public(manifest, port=80, *, staged=False, deadline=None):
    # Keep the legacy cohort/public-boundary checks as well as the new projection.
    snapshot=common.check_public(port,staged=staged,expected_model=manifest['expected_model_id'],deadline=deadline)
    for name in ('demo/recent.html','recent-view.js','recent-view.css','research/recent-public.json',
                 'research/recent-public-trades.csv'):
        status,body=common.request('/'+name,port,deadline=deadline)
        if status!=200 or hashlib.sha256(body).hexdigest()!=manifest['files'][name]:
            raise RuntimeError('Published recent asset mismatch: '+name)
    if not staged:
        paper=snapshot['paper_performance']
        if (paper.get('stake_usdt')!=30 or paper.get('daily_loss_limit_usdt')!=150
                or snapshot.get('publication_hold') is not True
                or not snapshot.get('historical_research',{}).get('available')):
            raise RuntimeError('Stake/hold/historical-summary readiness failed')
    return snapshot


def retain_publication_hold():
    """Leave an existing regular hold intact; never follow a replacement link."""
    if HOLD.is_symlink():raise ValueError('Unexpected symlinked publication hold')
    try:
        with HOLD.open('x',encoding='utf-8') as stream:
            stream.write('Recovery required; pause new paper admissions\n')
        os.chmod(HOLD,0o644)
    except FileExistsError:
        if HOLD.is_symlink() or not HOLD.is_file():
            raise ValueError('Unexpected publication hold type')


def recovery_service_stopped(record):
    """A timed-out stop is inconclusive until systemd confirms no main process."""
    try:common.command('systemctl','stop','ecq')
    except BaseException as error:
        record['recovery_stop_failure_type']=type(error).__name__
    try:
        state=dict(line.split('=',1) for line in common.command(
            'systemctl','show','ecq','--property=ActiveState',
            '--property=SubState','--property=MainPID').splitlines() if '=' in line)
        record['recovery_service_state']=state
        return (state.get('ActiveState') in {'inactive','failed'}
                and state.get('SubState') in {'dead','failed'}
                and state.get('MainPID')=='0')
    except BaseException as error:
        record['recovery_state_failure_type']=type(error).__name__
        return False


def recover_failure(error, *, stopped, switched, record, database, environment,
                    unit, current, old_env, old_unit, old_link, expected_model, backup):
    """Recover executables only, and always record an incomplete recovery.

    Do not restore a fixed-5U worker while it may still run or while the database
    contains variable-sized admissions. Database audit snapshots are never used
    as a rollback source.
    """
    record['failure_type']=type(error).__name__
    try:
        if not stopped:
            record['status']='failed_before_stop'
            return
        retain_publication_hold()
        if not recovery_service_stopped(record):
            record['status']='recovery_required_service_not_stopped'
            return
        if not has_variable_terms(database):
            common.atomic_file(environment,old_env);common.atomic_file(unit,old_unit)
            common.atomic_link(current,old_link)
            common.command('systemctl','daemon-reload');common.command('systemctl','start','ecq')
            common.wait_ready(lambda deadline:common.check_live_identity(
                expected_model=expected_model,deadline=deadline),25)
            HOLD.unlink(missing_ok=True)
            record['status']='rolled_back_without_database_rewind'
        else:
            # Keep the amount-aware unit/environment even if its reload failed.
            if switched:
                common.command('systemctl','daemon-reload')
                common.command('systemctl','start','ecq')
                common.wait_ready(lambda deadline:common.check_live_identity(
                    expected_model=expected_model,deadline=deadline),25)
            record['status']='held_for_amount_safe_recovery'
    except BaseException as recovery_error:
        record['status']='recovery_failed_hold_retained'
        record['recovery_failure_type']=type(recovery_error).__name__
        try:retain_publication_hold()
        except BaseException as hold_error:
            record['hold_failure_type']=type(hold_error).__name__
    finally:
        (backup/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if os.name!='posix' or os.geteuid()!=0:raise RuntimeError('Linux root required')
    manifest=inspect_bundle(args.bundle,args.sha256,args.commit)
    base=Path('/opt/codexquant');current=base/'current';old=current.resolve()
    unit=Path('/etc/systemd/system/ecq.service');environment=Path('/etc/codexquant.env')
    state=Path('/var/lib/codexquant');database=state/'worker.sqlite3'
    if not current.is_symlink() or old.parent!=(base/'releases').resolve() or str(old)!=manifest['expected_active_release']:
        raise ValueError('Unexpected active release')
    if (common.command('systemctl','show','ecq','-p','WorkingDirectory','--value')!=str(old)
            or common.command('systemctl','show','ecq','-p','FragmentPath','--value')!=str(unit)
            or common.command('systemctl','show','ecq','-p','DropInPaths','--value')):
        raise ValueError('Service definition changed or has unsupported drop-ins')
    if unit.is_symlink() or environment.is_symlink() or HOLD.exists() or HOLD.is_symlink():raise ValueError('Unexpected unit/env/hold state')
    if (common.digest(unit)!=manifest['expected_service_unit_sha256']
            or common.digest(environment)!=manifest['expected_environment_sha256']
            or common.runtime_hashes(old)!=manifest['expected_runtime_hashes']):
        raise ValueError('Inspected source or configuration changed')
    previous=common.check_live_identity(expected_model=manifest['expected_model_id'])
    original_models=common.immutable_state(state)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    release=base/'releases'/f'{stamp}-{args.commit[:12]}'
    backup=Path('/var/backups/codexquant')/f'stake-{stamp}-{args.commit[:12]}'
    backup.mkdir(mode=0o700,parents=True,exist_ok=False)
    old_unit=unit.read_bytes();old_env=environment.read_bytes();old_link=os.readlink(current)
    shutil.copy2(unit,backup/'ecq.service');shutil.copy2(environment,backup/'runtime.env')
    os.chmod(backup/'runtime.env',0o600)
    print('COPY and verify scoped overlay',flush=True)
    shutil.copytree(old,release,symlinks=True);apply_overlay(args.bundle,release,manifest)
    actual=common.runtime_hashes(release)
    if actual!=manifest['new_runtime_hashes']:raise ValueError('New runtime does not match reviewed source')
    for name,value in manifest['expected_runtime_hashes'].items():
        if name not in {'live/app.py','live/notify.py'} and actual.get(name)!=value:
            raise ValueError('Unrelated runtime module changed')
    with socket.socket() as probe:probe.bind(('127.0.0.1',8088))
    print('STAGE no-worker public-data checks',flush=True)
    with (backup/'stage.log').open('wb') as output:
        stage=subprocess.Popen([str(base/'venv/bin/python'),'-m','live.app','--host','127.0.0.1',
             '--port','8088','--no-worker'],cwd=release,env=dict(os.environ,WECOM_WEBHOOK_URL=''),
             stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            def ready(deadline):
                if stage.poll() is not None:raise RuntimeError('Stage exited')
                return check_public(manifest,8088,staged=True,deadline=deadline)
            common.wait_ready(ready,25)
        finally:
            stage.terminate()
            try:stage.wait(timeout=5)
            except subprocess.TimeoutExpired:stage.kill();stage.wait(timeout=5)
    if common.immutable_state(state)!=original_models:raise RuntimeError('Model changed while staging')
    if unit.read_bytes()!=old_unit or environment.read_bytes()!=old_env or os.readlink(current)!=old_link:
        raise RuntimeError('Configuration changed during staging')
    record={'commit':args.commit,'release':str(release),'previous_release':str(old),'backup':str(backup),
            'model_id':manifest['expected_model_id'],'notifications_sent_by_helper':False,
            'database_rollback_performed':False,'previous_paper':previous['paper_performance']}
    stopped=False;switched=False
    try:
        print('STOP briefly; snapshot state; hold new admissions during validation',flush=True)
        stopped=True;common.command('systemctl','stop','ecq')
        settled=preserved_settled_rows(database)
        for source in sorted(state.glob('*.sqlite3')):
            if source.is_symlink():raise ValueError('Unexpected symlinked database')
            common.sqlite_backup(source,backup/source.name)
        with HOLD.open('x',encoding='utf-8') as hold:hold.write('stake publication validation\n')
        os.chmod(HOLD,0o644)
        common.atomic_file(environment,updated_environment(old_env))
        updated=common.replace_working_directory(old_unit.decode('utf-8'),old,release).encode('utf-8')
        common.atomic_file(unit,updated);switched=True
        common.command('systemctl','daemon-reload');common.command('systemctl','start','ecq')
        snapshot=common.wait_ready(lambda deadline:check_public(manifest,deadline=deadline),35)
        after=preserved_settled_rows(database)
        if not all(row in after for row in settled):raise RuntimeError('Existing settled paper outcomes changed')
        if snapshot['paper_performance']['total']<previous['paper_performance']['total']:
            raise RuntimeError('Paper records were lost')
        if common.immutable_state(state)!=original_models:raise RuntimeError('Active model changed')
        if environment.read_bytes()!=updated_environment(old_env):raise RuntimeError('Environment changed unexpectedly')
        common.atomic_link(current,str(release))
        record.update(status='published',new_paper=snapshot['paper_performance'],
                      environment_sha256=common.digest(environment),new_stake_usdt=30,daily_limit_usdt=150)
        (backup/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        # Commit point: all checks succeeded before allowing any new-sized admission.
        HOLD.unlink()
    except BaseException as error:
        recover_failure(error,stopped=stopped,switched=switched,record=record,database=database,
                        environment=environment,unit=unit,current=current,old_env=old_env,
                        old_unit=old_unit,old_link=old_link,expected_model=manifest['expected_model_id'],
                        backup=backup)
        raise
    print(json.dumps({k:record[k] for k in ('status','commit','release','backup','model_id','new_stake_usdt','daily_limit_usdt')}),flush=True)


if __name__=='__main__':main()
