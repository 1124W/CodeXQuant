"""Build a deployment archive from a public allowlist and locally trained models."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    out=ROOT/'deploy-bundles';out.mkdir(exist_ok=True)
    path=out/'codexquant-deploy.tgz'
    files=['index.html','styles.css','app.js','research.js','workbuddy.js','trade-view.js',
           'requirements-live.txt','.env.example','demo/trades.html','demo/quantdinger.html',
           'research/trade-chart-data.json','research/workbuddy-results.json','research/workbuddy-trades.csv',
           'research/event-results.json','research/event-trades.csv','research/results.json',
           'research/derivative-study-results.json','research/minute-scan-results.json',
           'deploy/install.sh','deploy/README.md','deploy/verify_ready.py']
    files += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'live').glob('*.py')]
    models=ROOT/'research'/'.cache'/'live-model'
    with tarfile.open(path,'w:gz') as archive:
        for name in files:archive.add(ROOT/name,arcname=name,recursive=False)
        for model in models.iterdir():
            if model.is_file() and model.suffix in ('.json','.joblib','.gz'):
                archive.add(model,arcname='models/'+model.name,recursive=False)
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
    guide='使用用户授权的SSH连接 38.76.182.201:22，用户名root。密码沿用用户提供的凭据，不写入仓库或日志。\n把本目录的tgz和install.sh上传服务器；runtime.env仅私下上传至/etc/codexquant.env，chmod600。\n执行 bash /上传目录/install.sh /上传目录/codexquant-deploy.tgz\n程序先验收8088端口，再备份并替换ecq服务80端口，失败自动回滚。\n访问 http://38.76.182.201/demo/trades.html 。\n机器人已配置为信号即时提醒、北京时间每小时整点播报。模型不下单。\n'
    (target/'WorkBuddy-deploy.txt').write_text(guide,encoding='utf-8')
    zipped=desktop/'CodeXQuant-deploy-20261112.zip'
    with zipfile.ZipFile(zipped,'w',zipfile.ZIP_DEFLATED) as output:
        for item in target.iterdir():output.write(item,arcname=target.name+'/'+item.name)
    print(json.dumps({'desktop_folder':str(target),'desktop_zip':str(zipped),'bytes':zipped.stat().st_size}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--desktop',action='store_true')
    args=parser.parse_args();archive=build()
    if args.desktop:desktop_package(archive)
