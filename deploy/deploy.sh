#!/usr/bin/env bash
#
# Deploy / update PhotoCapibara on the VPS.
#
# Usage:
#   deploy/deploy.sh            # sync code + rebuild + restart webui
#   deploy/deploy.sh --data     # also rsync the local data/ (excludes work/) — first deploy / refresh
#   deploy/deploy.sh --worker   # also rebuild + (re)start the worker (needs a valid ANTHROPIC_API_KEY)
#
# One-time bootstrap (DB snapshot, .env, Keycloak secrets, override file) is NOT done here —
# it was done during the initial deploy. This script is for pushing code changes afterwards.
# See AGENTS.md for the full picture.
set -euo pipefail

# --- config (override via env) ---------------------------------------------
HOST="${DEPLOY_HOST:-anderson@10.200.1.1}"
DIR="${DEPLOY_DIR:-/home/anderson/photos_capibara}"
PROFILE="${DEPLOY_PROFILE:-auth}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ConnectTimeout=20)

SYNC_DATA=0
SYNC_WORKER=0
for arg in "$@"; do
  case "$arg" in
    --data)   SYNC_DATA=1 ;;
    --worker) SYNC_WORKER=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."   # repo root

echo "==> syncing code to $HOST:$DIR"
rsync -az --delete --timeout=90 -e "ssh ${SSH_OPTS[*]}" \
  --exclude 'venv/' --exclude 'data/' --exclude 'ComfyUI/' --exclude '.git/' \
  --exclude '.claude/' --exclude 'web/node_modules/' --exclude 'web/dist/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.env' \
  --exclude 'models/gender_net.caffemodel' \
  --exclude 'docker-compose.override.yml' \
  --exclude 'keycloak/realm-photoraw.json' \
  ./ "$HOST:$DIR/"

if [[ "$SYNC_DATA" == 1 ]]; then
  echo "==> syncing data/ (excluding rebuildable work/ and the live SQLite WAL)"
  rsync -az --timeout=120 -e "ssh ${SSH_OPTS[*]}" \
    --exclude 'work/' --exclude 'pipeline.db-wal' --exclude 'pipeline.db-shm' \
    data/ "$HOST:$DIR/data/"
fi

TARGETS="webui"
[[ "$SYNC_WORKER" == 1 ]] && TARGETS="webui worker"

echo "==> building images ($TARGETS) on the server"
# shellcheck disable=SC2029
ssh "${SSH_OPTS[@]}" "$HOST" "cd '$DIR' && docker compose --profile '$PROFILE' build $TARGETS"

echo "==> (re)starting $TARGETS"
ssh "${SSH_OPTS[@]}" "$HOST" "cd '$DIR' && docker compose --profile '$PROFILE' up -d $TARGETS"

echo "==> waiting for webui health"
ssh "${SSH_OPTS[@]}" "$HOST" "cd '$DIR' && \
  until [ \"\$(docker inspect -f '{{.State.Health.Status}}' photos_capibara-webui-1 2>/dev/null)\" = healthy ]; do sleep 3; done && \
  docker compose --profile '$PROFILE' ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'"

echo "==> done. https://fotos.capibaraeventos.com  (direct: http://10.200.1.1:8765)"
