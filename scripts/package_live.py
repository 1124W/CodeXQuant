"""Build a deployment archive from a public allowlist and locally trained models."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def model_files(model_dir):
    """Only manifest-declared, hash-verified model/seed files enter the bundle."""
    directory=Path(model_dir)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    schema=manifest.get('schema_version')
    symbols=('BTCUSDT','ETHUSDT')
    if schema not in (1,2) or set(manifest.get('models',{}))!=set(symbols):
        raise ValueError('Unsupported model bundle schema/assets')
    selected=['manifest.json']
    for symbol in symbols:
        descriptor=manifest['models'][symbol]
        entries=[(descriptor,'training_samples_file','training_samples_sha256',f'{symbol}-training.csv.gz')]
        if schema==1:
            entries.append((descriptor,'file','sha256',f'{symbol}.joblib'))
        else:
            members=descriptor.get('members',{})
            if set(members)!={'logistic_1m','gbm_deeper_5m'}:raise ValueError('Missing fixed consensus member')
            entries.extend((members[name],'file','sha256',f'{symbol}-{name}.joblib') for name in ('logistic_1m','gbm_deeper_5m'))
        for entry,name_key,hash_key,expected_name in entries:
            if entry.get(name_key)!=expected_name:raise ValueError('Unexpected model/seed filename')
            path=directory/expected_name
            if path.is_symlink() or not path.is_file():raise ValueError('Model/seed must be a regular local file')
            value=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
            if value.hexdigest()!=entry.get(hash_key):raise ValueError('Model/seed checksum mismatch')
            selected.append(expected_name)
    if (directory/'manifest.json').is_symlink():raise ValueError('Model manifest must not be a symlink')
    return selected


def build(model_dir=None):
    out=ROOT/'deploy-bundles';out.mkdir(exist_ok=True)
    path=out/'codexquant-deploy.tgz'
    files=['index.html','styles.css','app.js','research.js','workbuddy.js','trade-view.js',
           'requirements-live.txt','.env.example','demo/trades.html','demo/quantdinger.html',
           'demo/research.html','research-view.js','research-view.css',
           'demo/recent.html','recent-view.js','recent-view.css',
           'research/recent-public.json','research/recent-public-trades.csv',
           'demo/consensus.html','consensus-view.js','consensus-view.css',
           'research/consensus-public.json','research/consensus-public-trades.csv',
           'research/high-win-public.json','research/high-win-public-trades.csv',
           'research/trade-chart-data.json','research/workbuddy-results.json','research/workbuddy-trades.csv',
           'research/event-results.json','research/event-trades.csv','research/results.json',
           'research/derivative-study-results.json','research/minute-scan-results.json',
           'deploy/install.sh','deploy/README.md']
    files += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'live').glob('*.py')]
    models=Path(model_dir or ROOT/'research'/'.cache'/'consensus-live-model-final')
    selected_models=model_files(models)
    with tarfile.open(path,'w:gz') as archive:
        for name in files:archive.add(ROOT/name,arcname=name,recursive=False)
        for name in selected_models:
            archive.add(models/name,arcname='models/'+name,recursive=False)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'codexquant-deploy.tgz.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(path),'bytes':path.stat().st_size,'sha256':digest}))
    return path


def desktop_package(archive):
    desktop=Path.home()/'Desktop'
    target=desktop/'CodeXQuant-deploy-20261112'
    target.mkdir(parents=True,exist_ok=True)
    for source,name in ((archive,archive.name),(archive.with_suffix('.tgz.sha256'),'codexquant-deploy.tgz.sha256'),
        (ROOT/'deploy'/'install.sh','install.sh'),(ROOT/'deploy'/'README.md','README.md'),
        (ROOT/'.private'/'runtime.env','runtime.env')):
        shutil.copy2(source,target/name)
    guide='此安装器仅用于没有ecq服务和历史状态的全新服务器。现有服务器必须使用deploy/publish_consensus.py的核验升级流程，不能运行install.sh覆盖。\n把tgz和install.sh上传新服务器；将私有环境配置放到/etc/codexquant.env，chmod600，显式设置CODEXQUANT_STRATEGY=consensus和CODEXQUANT_MODEL_DIR=/var/lib/codexquant/models-consensus/initial。\n全新安装命令：bash /上传目录/install.sh /上传目录/codexquant-deploy.tgz\n程序先验收8088端口与模型，保留已有环境文件；不自动发送部署通知。模型不下单。\n'
    (target/'WorkBuddy-deploy.txt').write_text(guide,encoding='utf-8')
    zipped=desktop/'CodeXQuant-deploy-20261112.zip'
    with zipfile.ZipFile(zipped,'w',zipfile.ZIP_DEFLATED) as output:
        for item in target.iterdir():output.write(item,arcname=target.name+'/'+item.name)
    print(json.dumps({'desktop_folder':str(target),'desktop_zip':str(zipped),'bytes':zipped.stat().st_size}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--desktop',action='store_true')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'research'/'.cache'/'consensus-live-model-final')
    args=parser.parse_args();archive=build(args.model_dir)
    if args.desktop:desktop_package(archive)
