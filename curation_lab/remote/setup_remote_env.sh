#!/usr/bin/env bash
# One-time remote-side config for TAR-only GPU experiments.
#
# Deliberately writes a minimal .env: GPU=0 is the only value the TAR path needs
# (multabench/constants.py turns it into DEVICE="cuda:0"). W&B / HF / Kaggle
# secrets are NOT copied here -- curation_lab/runner/run.py never calls
# wandb_run(), E5-small-v2 is a public model, and the anchor dataset is already
# in the kagglehub cache on this box.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/student/MulTaBench}"
PY=/home/student/mtb311/bin/python

cd "$WORKDIR"

if [ ! -f .env ]; then
  cat > .env <<'ENV'
GPU=0
ENV
  echo "[env] wrote $WORKDIR/.env with GPU=0"
else
  grep -q '^GPU=' .env || echo 'GPU=0' >> .env
  echo "[env] $WORKDIR/.env already present; GPU key ensured"
fi

# The repo root must be importable (init.sh normally does this via a .pth file;
# it is bash-only and assumes a uv venv it created, so do the .pth directly).
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
echo "$WORKDIR" > "$SITE/multabench_repo.pth"
echo "[env] wrote $SITE/multabench_repo.pth -> $WORKDIR"

echo "=== import check ==="
cd "$WORKDIR"
"$PY" - <<'PY'
import multabench, curation_lab
from multabench.e5.e5_finetune import finetune_e5_with_lora
from multabench.constants import DEVICE
import torch
print("  multabench     ok")
print("  curation_lab   ok")
print("  DEVICE         ", DEVICE)
print("  cuda_available ", torch.cuda.is_available())
PY
