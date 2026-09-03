# The Azure remote-GPU lane (superseded)

These four scripts drove TAR (`ft`) experiments on a rented Azure GPU box —
`nlpgpu2025s-1010.westus.cloudapp.azure.com`, a Tesla M60 — before the Kaggle notebook lane
in `curation_lab/kaggle/` replaced it.

They are kept because they are the only record of how the remote lane was set up, and because
two of their design notes still apply to any future GPU lane:

- `pin_tar_env.sh` pins only the subset of `requirements.txt` that the TAR path can actually
  touch, rather than the whole file — the rest (autogluon-multimodal, ray, tabicl/tabdpt,
  streamlit) is irrelevant to `ft` and drags in a long install.
- `setup_remote_env.sh` writes a deliberately minimal `.env`: `GPU=0` is the only value the TAR
  path needs, and W&B / HF / Kaggle secrets are **not** copied to the remote box.
- `sync_to_remote.sh` streams a tar over ssh instead of using rsync, because Git Bash on Windows
  ships `tar` but not `rsync`. Code only — no `.venv`, no `.git`, no results, no secrets.
- `tar_smoke.py` exercises `multabench.e5.e5_finetune.finetune_e5_with_lora` alone, so a failure
  is unambiguously a TAR failure rather than something downstream.

**Status: not in use.** The Azure allocation was time-boxed and did not renew. Every grid in
`results/curation/` that has an `ft` state was produced by the Kaggle lane
(`curation_lab/kaggle/`), which is the documented primary path — see the repo README.

Nothing here has been re-run or re-verified since the allocation lapsed. Treat it as a starting
point for a future GPU lane, not as working infrastructure.
