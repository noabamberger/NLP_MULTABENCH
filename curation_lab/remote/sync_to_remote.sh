#!/usr/bin/env bash
# Push the working tree's code to the remote GPU box.
#
# Streams a tar over ssh rather than using rsync: Git Bash on Windows ships tar
# but not rsync. Code only -- no .venv, no .git, no results, no secrets. The
# remote .env is written separately (see setup_remote_env.sh) and deliberately
# carries GPU=0 and nothing else.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-nlpgpu2025s-1010.westus.cloudapp.azure.com}"
REMOTE_USER="${REMOTE_USER:-student}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/home/student/MulTaBench}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/multabench_remote}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[sync] $REPO_ROOT -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_WORKDIR"

tar czf - \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.emb_cache' \
  --exclude='.tar_cache' \
  --exclude='catboost_info' \
  multabench curation_lab tests benchmark.py requirements.txt requirements.in \
| ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" \
    "mkdir -p '$REMOTE_WORKDIR' && tar xzf - -C '$REMOTE_WORKDIR' && echo '[sync] extracted ok'"

echo "[sync] done"
