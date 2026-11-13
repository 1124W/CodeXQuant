#!/usr/bin/env bash
# Fresh installs only. Use publish_consensus.py to upgrade an existing service.
set -euo pipefail
bundle=${1:?Usage: install.sh /absolute/BUNDLE.tgz}
[[ $(id -u) == 0 ]] || { echo 'Linux root is required'; exit 2; }
[[ "$bundle" = /* && -f "$bundle" ]] || { echo 'Bundle must be an existing absolute path'; exit 2; }
base=/opt/codexquant
state=/var/lib/codexquant
unit=/etc/systemd/system/ecq.service
load_state=$(systemctl show ecq -p LoadState --value 2>/dev/null || true)
if [[ -e "$unit" || -L "$unit" || -e "$base/current" || -L "$base/current" || ( -n "$load_state" && "$load_state" != not-found ) ]]; then
  echo 'Existing ecq installation detected. Use deploy/publish_consensus.py; install.sh never upgrades or replaces history.'
  exit 2
fi
if [[ -L "$state" || ( -d "$state" && -n $(find "$state" -mindepth 1 -maxdepth 1 -print -quit) ) ]]; then
  echo 'Existing runtime state detected. Use the reviewed publication migration; nothing was replaced.'
  exit 2
fi
[[ -f /etc/codexquant.env && ! -L /etc/codexquant.env ]] || { echo 'Provide a regular /etc/codexquant.env first'; exit 2; }
python3 - <<'PY'
import socket
for port in (80,8088):
    with socket.socket() as probe:probe.bind(('0.0.0.0',port))
PY
version=$(date -u +%Y%m%dT%H%M%SZ)
release="$base/releases/$version"
[[ ! -e "$release" ]] || { echo 'Release directory already exists'; exit 2; }
mkdir -p "$release"
# Read environment values as data; never source a secret file as shell code.
# Existing environment content is preserved and model choice must be explicit.
strategy=$(python3 - "$bundle" "$release" <<'PY'
import hashlib,json,shlex,sys,tarfile
from pathlib import Path,PurePosixPath
bundle,release=Path(sys.argv[1]),Path(sys.argv[2])
env={}
for raw in Path('/etc/codexquant.env').read_text(encoding='utf-8-sig').splitlines():
    line=raw.strip()
    if not line or line.startswith('#'):continue
    if '=' not in line:raise ValueError('Malformed runtime environment line')
    key,value=line.split('=',1);key=key.strip()
    if key in env:raise ValueError('Duplicate runtime environment key')
    pieces=shlex.split(value,comments=False)
    if len(pieces)>1:raise ValueError('Environment values with spaces must be quoted')
    env[key]=pieces[0] if pieces else ''
with tarfile.open(bundle,'r:gz') as archive:
    members=archive.getmembers();names=[item.name for item in members]
    if len(set(names))!=len(names):raise ValueError('Duplicate bundle path')
    for item in members:
        path=PurePosixPath(item.name)
        if (not item.isfile() or path.is_absolute() or '..' in path.parts
                or str(path)!=item.name or '\\' in item.name or ':' in item.name):
            raise ValueError('Bundle contains a noncanonical or linked file')
    manifest=json.load(archive.extractfile('models/manifest.json'))
    schema=manifest.get('schema_version')
    if schema not in (1,2):raise ValueError('Unsupported model schema')
    strategy='consensus' if schema==2 else 'legacy'
    model_dir='/var/lib/codexquant/'+('models-consensus' if schema==2 else 'models')+'/initial'
    if (env.get('CODEXQUANT_STRATEGY')!=strategy or env.get('CODEXQUANT_MODEL_DIR')!=model_dir
            or env.get('CODEXQUANT_STATE_DIR')!='/var/lib/codexquant'):
        raise ValueError('Environment must explicitly match bundle strategy, model directory and state directory')
    if set(manifest.get('models',{}))!={'BTCUSDT','ETHUSDT'}:raise ValueError('Unexpected model assets')
    required={'models/manifest.json'}
    for symbol in ('BTCUSDT','ETHUSDT'):
        item=manifest['models'][symbol]
        records=[(item,'training_samples_file','training_samples_sha256',f'{symbol}-training.csv.gz')]
        if schema==1:records.append((item,'file','sha256',f'{symbol}.joblib'))
        else:
            if set(item.get('members',{}))!={'logistic_1m','gbm_deeper_5m'}:raise ValueError('Missing consensus members')
            records += [(item['members'][name],'file','sha256',f'{symbol}-{name}.joblib') for name in ('logistic_1m','gbm_deeper_5m')]
        for record,file_key,sha_key,expected in records:
            if record.get(file_key)!=expected:raise ValueError('Unexpected model filename')
            name='models/'+expected;required.add(name);digest=hashlib.sha256()
            with archive.extractfile(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=record.get(sha_key):raise ValueError('Model or training seed checksum mismatch')
    if {name for name in names if name.startswith('models/')}!=required:raise ValueError('Unexpected model-bundle files')
    archive.extractall(release,filter='data')
print(strategy)
PY
)
test -f "$release/live/app.py"
echo 'Preparing a fresh isolated runtime'
if [[ ! -x "$base/venv/bin/python" ]]; then python3 -m venv "$base/venv"; fi
"$base/venv/bin/python" -m pip install --disable-pip-version-check -r "$release/requirements-live.txt"
cd "$release"
"$base/venv/bin/python" - "$strategy" <<'PY'
import sys
if sys.argv[1]=='consensus':
    from live.consensus import ConsensusPredictor as Predictor
else:
    from live.predictor import Predictor
p=Predictor('models');print('MODEL_OK',p.manifest['model_version'])
PY
stage="codexquant-stage-$version"
cleanup_stage(){ systemctl stop "$stage" >/dev/null 2>&1 || true; }
trap cleanup_stage EXIT
systemd-run --unit="$stage" --property="WorkingDirectory=$release" \
  --setenv="CODEXQUANT_MODEL_DIR=$release/models" --setenv="CODEXQUANT_STRATEGY=$strategy" \
  "$base/venv/bin/python" -m live.app --host 127.0.0.1 --port 8088 --no-worker >/dev/null
for i in $(seq 1 20); do
  if curl -fsS --max-time 3 http://127.0.0.1:8088/health >/dev/null; then break; fi
  sleep 1
done
curl -fsS --max-time 5 http://127.0.0.1:8088/research/trade-chart-data.json | "$base/venv/bin/python" -c 'import json,sys; assert len(json.load(sys.stdin)["trades"])==442'
if [[ "$strategy" == consensus ]]; then curl -fsS --max-time 5 http://127.0.0.1:8088/demo/consensus.html >/dev/null; fi
cleanup_stage
id codexquant >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin codexquant
model_root="$state/models"
history="$state/history.sqlite3"
if [[ "$strategy" == consensus ]]; then model_root="$state/models-consensus"; history="$state/history-consensus.sqlite3"; fi
mkdir -p "$model_root"
cp -a "$release/models" "$model_root/initial"
# Import dense seeds before starting the worker's readiness deadline.
"$base/venv/bin/python" - "$model_root/initial" "$history" <<'PY'
import json,sys
from pathlib import Path
from live.history import FeatureHistory
directory=Path(sys.argv[1]);manifest=json.loads((directory/'manifest.json').read_text())
print('SEED_OK',FeatureHistory(sys.argv[2]).import_seed(directory,manifest))
PY
chown -R codexquant:codexquant "$state"
chmod 700 "$state"
cat > "$unit" <<EOF
[Unit]
Description=CodeXQuant BTC ETH research dashboard and WeCom alerts
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=codexquant
Group=codexquant
WorkingDirectory=$release
EnvironmentFile=/etc/codexquant.env
Environment=PYTHONUNBUFFERED=1
Environment=LOKY_MAX_CPU_COUNT=2
Environment=OMP_NUM_THREADS=2
Environment=OPENBLAS_NUM_THREADS=2
ExecStart=$base/venv/bin/python -m live.app --host 0.0.0.0 --port 80
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/codexquant
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
UMask=0077
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable ecq >/dev/null
systemctl start ecq
ok=0
for i in $(seq 1 45); do
  if curl -fsS --max-time 3 http://127.0.0.1/health >/dev/null; then ok=1; break; fi
  sleep 1
done
if [[ "$ok" == 1 ]]; then
  if ! curl -fsS --max-time 5 http://127.0.0.1/api/live/status | "$base/venv/bin/python" -c 'import json,sys; from pathlib import Path; s=json.load(sys.stdin); m=json.loads(Path("models/manifest.json").read_text()); assert s["model"]["id"]==m["model_version"] and s["model"]["rule_id"]==m["rule_id"] and s["order_execution_enabled"] is False'; then ok=0; fi
fi
if [[ "$ok" != 1 ]]; then
  systemctl stop ecq
  echo 'Fresh installation failed readiness; stopped new service and retained files/state for diagnosis. No existing installation was replaced.'
  exit 1
fi
ln -s "$release" "$base/current"
echo "INSTALLED release=$release strategy=$strategy"
curl -fsS --max-time 5 http://127.0.0.1/health
