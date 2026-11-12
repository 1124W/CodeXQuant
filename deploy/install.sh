#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?Usage: install.sh BUNDLE.tgz}
[[ "$bundle" = /* && -f "$bundle" ]] || { echo 'Bundle must be an existing absolute path'; exit 2; }
base=/opt/codexquant
version=$(date -u +%Y%m%dT%H%M%SZ)
release="$base/releases/$version"
backup="/var/backups/codexquant/$version"
mkdir -p "$release" "$backup"
chmod 700 "$backup"
tar -xzf "$bundle" -C "$release"
test -f "$release/live/app.py"
test -f "$release/models/manifest.json"
test -f /etc/codexquant.env
echo 'Preparing isolated runtime; original port 80 stays online'
if [[ ! -x "$base/venv/bin/python" ]]; then
  python3 -m venv "$base/venv"
fi
"$base/venv/bin/python" -m pip install --disable-pip-version-check -r "$release/requirements-live.txt"
cd "$release"
"$base/venv/bin/python" -c 'from live.predictor import Predictor; p=Predictor("models"); print("MODEL_OK",p.manifest["model_version"])'
old_cwd=$(systemctl show ecq -p WorkingDirectory --value || true)
old_unit=$(systemctl show ecq -p FragmentPath --value || true)
if [[ -n "$old_unit" && -f "$old_unit" ]]; then cp -a "$old_unit" "$backup/ecq.service"; fi
if [[ "$old_cwd" = /* && -d "$old_cwd" && "$old_cwd" != / && "$old_cwd" != /root && "$old_cwd" != /opt && "$old_cwd" != /home ]]; then
  tar -czf "$backup/old-program.tgz" -C "$old_cwd" .
fi
printf '%s\n' "$old_cwd" > "$backup/old-working-directory.txt"
echo 'Staging candidate on 127.0.0.1:8088'
systemd-run --unit="codexquant-stage-$version" --property="WorkingDirectory=$release" --setenv="CODEXQUANT_MODEL_DIR=$release/models" "$base/venv/bin/python" -m live.app --host 127.0.0.1 --port 8088 --no-worker >/dev/null
stage="codexquant-stage-$version"
cleanup_stage(){ systemctl stop "$stage" >/dev/null 2>&1 || true; }
trap cleanup_stage EXIT
for i in $(seq 1 20); do
  if curl -fsS --max-time 3 http://127.0.0.1:8088/health >/dev/null; then break; fi
  sleep 1
done
curl -fsS --max-time 5 http://127.0.0.1:8088/demo/trades.html | grep -q '442'
curl -fsS --max-time 5 http://127.0.0.1:8088/research/trade-chart-data.json | "$base/venv/bin/python" -c 'import json,sys; d=json.load(sys.stdin); assert len(d["trades"])==442; print("CHART_DATA_OK")'
cleanup_stage
id codexquant >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin codexquant
mkdir -p /var/lib/codexquant/models
if [[ ! -d /var/lib/codexquant/models/initial ]]; then cp -a "$release/models" /var/lib/codexquant/models/initial; fi
chown -R codexquant:codexquant /var/lib/codexquant
chmod 700 /var/lib/codexquant
cat > "$release/deploy/rollback.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
test -f '$backup/ecq.service'
cp '$backup/ecq.service' /etc/systemd/system/ecq.service
systemctl daemon-reload
systemctl restart ecq
echo 'Original ecq service restored'
EOF
chmod 700 "$release/deploy/rollback.sh"
echo 'Staging passed; switching the ecq service'
cat > /etc/systemd/system/ecq.service <<EOF
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
systemctl restart ecq
ok=0
for i in $(seq 1 45); do
  if curl -fsS --max-time 3 http://127.0.0.1/health >/dev/null; then ok=1; break; fi
  sleep 1
done
if [[ "$ok" != 1 ]]; then
  echo 'Candidate failed; rolling back original service'
  "$release/deploy/rollback.sh"
  exit 1
fi
ln -sfn "$release" "$base/current"
printf '%s\n' "$backup" > /var/lib/codexquant/deployment-backup.txt
echo "DEPLOYED release=$release backup=$backup"
curl -fsS --max-time 5 http://127.0.0.1/health
