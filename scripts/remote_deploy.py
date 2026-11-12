"""SSH inspection/deployment helper. Password is requested on hidden stdin."""
from __future__ import annotations
import argparse
import getpass
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'/'.cache'/'deploy-deps'))

INSPECT="""set -eu
printf 'SYSTEM\\n'
uname -a
cat /etc/os-release
printf '\\nRUNTIME\\n'
command -v python3 || true
python3 --version || true
command -v docker || true
command -v nginx || true
printf '\\nCAPACITY\\n'
free -m || true
df -h / /opt
printf '\\nPORT80\\n'
ss -ltnp '( sport = :80 or sport = :443 or sport = :8088 )' || true
printf '\\nSERVICES\\n'
systemctl list-units --type=service --state=running --no-pager | head -35 || true
printf '\\nHTTP\\n'
curl -sS -I --max-time 5 http://127.0.0.1/ || true
"""

def connect(host,user,port):
    import paramiko
    known=ROOT/'.private'/'known_hosts'
    known.parent.mkdir(parents=True,exist_ok=True)
    client=paramiko.SSHClient()
    if known.exists():client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password=getpass.getpass('SSH password: ')
    client.connect(host,port=port,username=user,password=password,look_for_keys=False,
                   allow_agent=False,timeout=12,auth_timeout=12,banner_timeout=12)
    client.get_transport().set_keepalive(15)
    client.save_host_keys(str(known))
    return client

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--host',required=True)
    cli.add_argument('--port',type=int,default=22)
    cli.add_argument('--user',default='root')
    cli.add_argument('--mode',choices=('inspect','deploy','finalize','session'),default='inspect')
    cli.add_argument('--bundle',type=Path)
    args=cli.parse_args()
    client=connect(args.host,args.user,args.port)
    try:
        if args.mode=='session':
            print('SSH_SESSION_READY',flush=True)
            while True:
                line=sys.stdin.readline()
                if not line:break
                item=json.loads(line)
                if item.get('action')=='close':break
                if item.get('action')=='upload':
                    with client.open_sftp() as sftp:
                        sftp.put(item['local'],item['remote'])
                        if 'mode' in item:sftp.chmod(item['remote'],int(item['mode']))
                    print('UPLOAD_OK',flush=True);continue
                if item.get('action')!='exec':raise ValueError('Unknown session action')
                stdin,stdout,stderr=client.exec_command(item['command'],timeout=item.get('timeout',300))
                # Stream rather than buffering package installation/deploy progress.
                for line in iter(stdout.readline,''):print(line.rstrip(),flush=True)
                error=stderr.read().decode('utf-8',errors='replace')
                if error:print(error,file=sys.stderr,flush=True)
                print('REMOTE_EXIT',stdout.channel.recv_exit_status(),flush=True)
            return
        if args.mode=='inspect':command=INSPECT
        elif args.mode=='finalize':
            _,result,_=client.exec_command('readlink -f /opt/codexquant/current')
            release=result.read().decode().strip()
            if not release.startswith('/opt/codexquant/releases/') or '..' in release:
                raise RuntimeError('Unexpected active release path')
            with client.open_sftp() as sftp:
                for local,remote in ((ROOT/'live'/'app.py',release+'/live/app.py'),
                    (ROOT/'deploy'/'verify_ready.py',release+'/deploy/verify_ready.py')):
                    sftp.put(str(local),remote);sftp.chmod(remote,0o644)
            command=(f'cd {release} && systemctl restart ecq && sleep 12 && '
                'set -a && . /etc/codexquant.env && set +a && '
                'runuser -u codexquant --preserve-environment -- /opt/codexquant/venv/bin/python deploy/verify_ready.py')
        else:
            if not args.bundle or not args.bundle.is_file():raise ValueError('Validated bundle is required')
            _,ready,_=client.exec_command('mkdir -p /root/codexquant-upload; chmod 700 /root/codexquant-upload; if [ -f /etc/codexquant.env ]; then cp -a /etc/codexquant.env /root/codexquant-upload/previous-runtime.env; fi')
            if ready.channel.recv_exit_status()!=0:raise RuntimeError('Cannot prepare private upload directory')
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(120)
                for local,remote,mode in ((args.bundle,'/root/codexquant-upload/codexquant-deploy.tgz',0o600),
                    (ROOT/'deploy'/'install.sh','/root/codexquant-upload/install.sh',0o700),
                    (ROOT/'.private'/'runtime.env','/etc/codexquant.env',0o600)):
                    sftp.put(str(local),remote);sftp.chmod(remote,mode)
            command='bash /root/codexquant-upload/install.sh /root/codexquant-upload/codexquant-deploy.tgz'
        stdin,stdout,stderr=client.exec_command(command,timeout=60)
        for line in iter(stdout.readline,''):print(line.rstrip(),flush=True)
        err=stderr.read().decode('utf-8',errors='replace')
        if err:print(err,file=sys.stderr)
        print('remote_exit',stdout.channel.recv_exit_status())
    finally:client.close()

if __name__=='__main__':main()
