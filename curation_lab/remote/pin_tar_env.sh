#!/usr/bin/env bash
# Align the remote venv's TAR-code-path packages to this repo's requirements.txt pins.
#
# Why only a subset: the full requirements.txt drags in autogluon-multimodal, ray,
# tabicl/tabdpt and streamlit, none of which the `ft` (TAR) path touches. The
# packages below are exactly the ones that can change a TAR number or break the
# HF Trainer call in multabench/e5/e5_finetune.py.
#
# torch is listed so pip refuses to silently swap the CUDA build; requirements.txt
# pins `torch==2.7.1`, which PEP 440 treats as satisfied by `2.7.1+cu126`.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
PY=/home/student/mtb311/bin/python

uv pip install --python "$PY" \
  "torch==2.7.1" \
  "transformers==4.57.3" \
  "peft==0.18.0" \
  "accelerate==1.12.0" \
  "huggingface-hub==0.36.0" \
  "tokenizers==0.22.1" \
  "safetensors==0.7.0" \
  "datasets==4.0.0" \
  "numpy==2.3.5" \
  "pandas==2.3.3" \
  "scikit-learn==1.6.1" \
  "scipy==1.16.3" \
  "sentence-transformers==5.1.2" \
  "tabstar==1.1.15" \
  "python-dotenv==1.2.1" \
  "wandb==0.23.0"

echo "=== post-install verification ==="
"$PY" - <<'PY'
import importlib.metadata as md
import torch
for p in ["torch", "transformers", "peft", "accelerate", "huggingface-hub",
          "tokenizers", "datasets", "numpy", "pandas", "scikit-learn", "tabstar"]:
    print(f"  {p:20s} {md.version(p)}")
print("  cuda_available     ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device             ", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
    # sm_52 (Maxwell) has no bf16 and no tensor cores; prove fp32 matmul lands.
    a = torch.randn(256, 256, device="cuda")
    print("  fp32 matmul ok     ", bool(torch.isfinite(a @ a).all()))
PY
