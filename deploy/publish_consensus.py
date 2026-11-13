"""Install a reviewed two-model rollout without rewinding paper state or notifying.

The seven private model/seed files are never served or committed to GitHub.
Pre-import the independent dense history while the old worker stays online.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime,timezone
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
    from . import publish_stake as previous
except ImportError:
    import publish_research as common
    import publish_stake as previous

RULE='pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
MANIFEST='consensus-publication-manifest.json'
ASSETS=('demo/consensus.html','consensus-view.js','consensus-view.css',
        'research/consensus-public.json','research/consensus-public-trades.csv')
ALLOWED=set(ASSETS)|{'index.html','demo/trades.html','trade-view.js','live/app.py','live/history.py',
    'live/notify.py','live/consensus.py','README.md','deploy/README.md','deploy/publish_consensus.py',
    'scripts/package_live.py','scripts/train_consensus_live.py','deploy/install.sh','.env.example'}
MODEL_FILES={'manifest.json','BTCUSDT-training.csv.gz','ETHUSDT-training.csv.gz'}|{
    f'{symbol}-{member}.joblib' for symbol in ('BTCUSDT','ETHUSDT')
    for member in ('logistic_1m','gbm_deeper_5m')}
PRIVATE_FILES={'models/'+name for name in MODEL_FILES}
REQUIRED=set(ASSETS)|{'live/app.py','live/history.py','live/notify.py','live/consensus.py'}|PRIVATE_FILES
HOLD=previous.HOLD


def inspect_bundle(path,sha,commit):
    if not common.is_sha(sha) or common.digest(path)!=sha or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Publication identity mismatch')
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if (len(names)!=len(set(names)) or set(names)-ALLOWED-PRIVATE_FILES-{MANIFEST}
                or MANIFEST not in names or len(names)>64):raise ValueError('Unapproved overlay member')
        if (any(not m.isfile() or not 0<=m.size<=180*1024*1024 for m in members)
                or sum(m.size for m in members)>512*1024*1024):raise ValueError('Invalid overlay size or member type')
        manifest=json.load(archive.extractfile(MANIFEST));files=manifest.get('files',{})
        if (manifest.get('kind')!='consensus_rollout_v1' or manifest.get('commit')!=commit
                or set(files)!=set(names)-{MANIFEST} or not REQUIRED.issubset(files)):
            raise ValueError('Incomplete rollout manifest')
        for name,digest in files.items():
            if not common.is_sha(digest):raise ValueError('Invalid file digest')
            hashed=hashlib.sha256()
            with archive.extractfile(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):hashed.update(chunk)
            if hashed.hexdigest()!=digest:raise ValueError('Checksum mismatch: '+name)
        model=json.load(archive.extractfile('models/manifest.json'))
        if (model.get('schema_version')!=2 or model.get('rule_id')!=RULE
                or model.get('model_version')!=manifest.get('new_model_id')
                or not re.fullmatch('[0-9a-f]{20}',model['model_version'])
                or model.get('order_execution_enabled') is not False):raise ValueError('Wrong model identity')
        described={'manifest.json'}
        if set(model.get('models',{}))!={'BTCUSDT','ETHUSDT'}:raise ValueError('Wrong model assets')
        for symbol,item in model['models'].items():
            seed=item['training_samples_file']
            if seed!=symbol+'-training.csv.gz' or files['models/'+seed]!=item['training_samples_sha256']:
                raise ValueError('Wrong seed identity')
            described.add(seed)
            if set(item['members'])!={'logistic_1m','gbm_deeper_5m'}:raise ValueError('Wrong consensus members')
            for member,descriptor in item['members'].items():
                name=descriptor['file']
                if name!=f'{symbol}-{member}.joblib' or files['models/'+name]!=descriptor['sha256']:
                    raise ValueError('Wrong member identity')
                described.add(name)
        if described!=MODEL_FILES:raise ValueError('Unexpected model population')
    return manifest


def updated_environment(original,model_directory):
    updates={'CODEXQUANT_STRATEGY':'consensus','CODEXQUANT_MODEL_DIR':str(model_directory),
        'CODEXQUANT_PAPER_STAKE':'30','CODEXQUANT_PUBLICATION_HOLD_FILE':str(HOLD),
        'CODEXQUANT_PUBLIC_BASE_URL':'https://nightsleep.eu.org'}
    output=[];seen=set()
    for line in original.decode('utf-8-sig').splitlines():
        key=line.split('=',1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if key=='CODEXQUANT_PAPER_DAILY_LIMIT':continue
        if key in updates:
            if key not in seen:output.append(key+'='+updates[key]);seen.add(key)
        else:output.append(line)
    output.extend(key+'='+value for key,value in updates.items() if key not in seen)
    return ('\n'.join(output)+'\n').encode()


def write_member(archive,name,target,digest,mode=0o644,owner=None):
    target=Path(target)
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError('Symlinked destination')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as stream:shutil.copyfileobj(archive.extractfile(name),stream,1024*1024)
    os.chmod(target,mode)
    if owner:os.chown(target,*owner)
    if common.digest(target)!=digest:raise ValueError('Extracted content changed')


def read_settled(database):
    return previous.preserved_settled_rows(database)


def verify_public(manifest,port=80,*,staged=False,deadline=None):
    snapshot=common.check_public(port,staged=staged,expected_model=manifest['new_model_id'],deadline=deadline)
    for name in ASSETS:
        status,body=common.request('/'+name,port,deadline=deadline)
        if status!=200 or hashlib.sha256(body).hexdigest()!=manifest['files'][name]:
            raise RuntimeError('New public asset mismatch: '+name)
    for path in ('/models/manifest.json','/live/consensus.py','/research/consensus-three-year-2026-11-13/results.json'):
        if common.request(path,port,deadline=deadline)[0]!=404:raise RuntimeError('Private route exposed')
    if not staged:
        paper=snapshot['paper_performance'];history=snapshot.get('historical_research',{})
        if (snapshot['model']['rule_id']!=RULE or snapshot.get('publication_hold') is not True
                or paper.get('stake_usdt')!=30 or paper.get('daily_loss_limit_usdt')!=150
                or paper.get('current_strategy',{}).get('rule_id')!=RULE
                or not history.get('available') or not history.get('matches_current_strategy')):
            raise RuntimeError('Consensus readiness invariant failed')
    return snapshot


def rollback(error,*,unit,environment,current,old_unit,old_env,old_link,old_model,backup,record):
    record.update(status='rollback_required',failure_type=type(error).__name__)
    try:
        previous.retain_publication_hold()
        if not previous.recovery_service_stopped(record):
            record['status']='recovery_required_service_not_stopped';return
        common.atomic_file(environment,old_env);common.atomic_file(unit,old_unit)
        common.atomic_link(current,old_link)
        common.command('systemctl','daemon-reload');common.command('systemctl','start','ecq')
        common.wait_ready(lambda deadline:common.check_live_identity(expected_model=old_model,deadline=deadline),35)
        HOLD.unlink(missing_ok=True);record['status']='rolled_back_without_database_rewind'
    except BaseException as recovery:
        record['status']='recovery_failed_hold_retained';record['recovery_failure_type']=type(recovery).__name__
    finally:(backup/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--bundle',type=Path,required=True)
    cli.add_argument('--sha256',required=True);cli.add_argument('--commit',required=True);args=cli.parse_args()
    if os.name!='posix' or os.geteuid()!=0:raise RuntimeError('Linux root required')
    import pwd
    manifest=inspect_bundle(args.bundle,args.sha256,args.commit)
    base=Path('/opt/codexquant');state=Path('/var/lib/codexquant');current=base/'current';old=current.resolve()
    unit=Path('/etc/systemd/system/ecq.service');environment=Path('/etc/codexquant.env')
    if (not current.is_symlink() or old.parent!=(base/'releases').resolve()
            or str(old)!=manifest['expected_active_release']):raise ValueError('Active release changed')
    if (unit.is_symlink() or environment.is_symlink() or HOLD.exists() or HOLD.is_symlink()
            or (state/'active-consensus-model.txt').exists() or (state/'active-consensus-model.txt').is_symlink()):
        raise ValueError('Unexpected configuration or model pointer state')
    if (common.digest(unit)!=manifest['expected_service_unit_sha256']
            or common.digest(environment)!=manifest['expected_environment_sha256']
            or common.runtime_hashes(old)!=manifest['expected_runtime_hashes']):raise ValueError('Inspected source changed')
    if (common.command('systemctl','show','ecq','-p','WorkingDirectory','--value')!=str(old)
            or common.command('systemctl','show','ecq','-p','FragmentPath','--value')!=str(unit)
            or common.command('systemctl','show','ecq','-p','DropInPaths','--value')):raise ValueError('Service definition changed')
    initial=common.check_live_identity(expected_model=manifest['expected_model_id'])
    original_models=common.immutable_state(state)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    release=base/'releases'/f'{stamp}-{args.commit[:12]}'
    backup=Path('/var/backups/codexquant')/f'consensus-{stamp}-{args.commit[:12]}'
    backup.mkdir(parents=True,mode=0o700,exist_ok=False)
    old_unit=unit.read_bytes();old_env=environment.read_bytes();old_link=os.readlink(current)
    shutil.copy2(unit,backup/'ecq.service');shutil.copy2(environment,backup/'runtime.env');os.chmod(backup/'runtime.env',0o600)
    user=pwd.getpwnam('codexquant');owner=(user.pw_uid,user.pw_gid)
    model_dir=state/'models-consensus'/('initial-'+manifest['new_model_id'])
    for target in (model_dir, state/'history-consensus.sqlite3',
                   state/'history-consensus.sqlite3-wal', state/'history-consensus.sqlite3-shm'):
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
            raise ValueError('Symlinked consensus state destination')
    if (state/'history-consensus.sqlite3').resolve()==(state/'history.sqlite3').resolve():
        raise ValueError('Consensus history must remain independent')
    if model_dir.exists():raise ValueError('New model destination already exists')
    model_dir.mkdir(parents=True,mode=0o700);os.chown(model_dir.parent,*owner);os.chown(model_dir,*owner)
    print('COPY checked runtime and isolated consensus models',flush=True)
    shutil.copytree(old,release,symlinks=True)
    with tarfile.open(args.bundle,'r:gz') as archive:
        for name,digest in manifest['files'].items():
            if name in PRIVATE_FILES:write_member(archive,name,model_dir/name.split('/',1)[1],digest,0o600,owner)
            else:write_member(archive,name,release/name,digest)
    if common.runtime_hashes(release)!=manifest['new_runtime_hashes']:raise ValueError('Reviewed runtime differs')
    # Dense population import is deliberately outside the short service outage.
    seed_code=('import json;from pathlib import Path;from live.consensus import ConsensusPredictor;'
        'from live.history import FeatureHistory;'
        f'p=ConsensusPredictor({str(model_dir)!r});'
        f'h=FeatureHistory({str(state/"history-consensus.sqlite3")!r});'
        f'r=h.import_seed({str(model_dir)!r},p.manifest);'
        'print(json.dumps({"model_id":p.manifest["model_version"],"seed":r,"history":h.status()}))')
    print('PRESEED independent minute history; old worker remains online',flush=True)
    with (backup/'seed.log').open('wb') as output:
        imported=subprocess.run(['runuser','-u','codexquant','--',str(base/'venv/bin/python'),'-c',seed_code],
            cwd=release,stdout=output,stderr=subprocess.STDOUT,timeout=300)
    if imported.returncode:raise RuntimeError('Consensus model/seed preflight failed; previous worker is unchanged')
    with socket.socket() as probe:probe.bind(('127.0.0.1',8088))
    print('STAGE public assets without worker or notifications',flush=True)
    with (backup/'stage.log').open('wb') as output:
        stage=subprocess.Popen([str(base/'venv/bin/python'),'-m','live.app','--host','127.0.0.1','--port','8088','--no-worker'],
            cwd=release,env=dict(os.environ,WECOM_WEBHOOK_URL=''),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:common.wait_ready(lambda deadline:verify_public(manifest,8088,staged=True,deadline=deadline),30)
        finally:
            stage.terminate()
            try:stage.wait(timeout=5)
            except subprocess.TimeoutExpired:stage.kill();stage.wait(timeout=5)
    if (unit.read_bytes()!=old_unit or environment.read_bytes()!=old_env or os.readlink(current)!=old_link
            or common.immutable_state(state)!=original_models):raise ValueError('Existing configuration/model changed during stage')
    record={'commit':args.commit,'release':str(release),'previous_release':str(old),'backup':str(backup),
        'old_model_id':manifest['expected_model_id'],'new_model_id':manifest['new_model_id'],
        'new_rule_id':RULE,'model_directory':str(model_dir),'old_history_preserved':True,
        'database_rollback_performed':False,'notifications_sent_by_helper':False,
        'previous_paper':initial['paper_performance']}
    try:
        print('SWITCH under admission hold; preserve unsettled trades and old outcomes',flush=True)
        common.command('systemctl','stop','ecq')
        settled=read_settled(state/'worker.sqlite3')
        for source in (state/'worker.sqlite3',state/'notifications.sqlite3',state/'history.sqlite3'):
            if source.is_symlink():raise ValueError('Symlinked state database')
            common.sqlite_backup(source,backup/source.name)
        with HOLD.open('x',encoding='utf-8') as stream:stream.write('Consensus rollout validation\n')
        os.chmod(HOLD,0o644)
        common.atomic_file(environment,updated_environment(old_env,model_dir))
        common.atomic_file(unit,common.replace_working_directory(old_unit.decode(),old,release).encode())
        common.command('systemctl','daemon-reload');common.command('systemctl','start','ecq')
        snapshot=common.wait_ready(lambda deadline:verify_public(manifest,deadline=deadline),60)
        after=read_settled(state/'worker.sqlite3')
        if not all(row in after for row in settled):raise RuntimeError('Prior settled outcomes changed')
        if (snapshot['paper_performance']['total']<initial['paper_performance']['total']
                or snapshot['paper_performance']['current_strategy']['total']!=0):
            raise RuntimeError('New cohort or historical account migration is wrong')
        if common.immutable_state(state)!=original_models:raise RuntimeError('Original model was modified')
        common.atomic_link(current,str(release))
        record.update(status='published',new_paper=snapshot['paper_performance'])
        (backup/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        HOLD.unlink()
    except BaseException as error:
        rollback(error,unit=unit,environment=environment,current=current,old_unit=old_unit,old_env=old_env,
            old_link=old_link,old_model=manifest['expected_model_id'],backup=backup,record=record)
        raise
    print(json.dumps({key:record[key] for key in ('status','commit','release','backup','new_model_id','new_rule_id')}),flush=True)


if __name__=='__main__':main()
