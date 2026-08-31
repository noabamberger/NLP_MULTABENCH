"""Push the TAR notebook to Kaggle, wait for it, and print the run log.

This is the iteration loop: edit build_notebook.py -> `python -m
curation_lab.kaggle.push` -> read the traceback -> repeat. Without it every
change means a manual upload and a hand-refreshed browser tab.

Auth note: `.env` carries a new-style Kaggle token (`KGAT...`) under the legacy
name KAGGLE_KEY. kaggle>=2 only reads it from KAGGLE_API_TOKEN, and the old
~/.kaggle/kaggle.json path 401s on this token, so we map it across here.

Usage:
    python -m curation_lab.kaggle.push               # build, push, wait, log
    python -m curation_lab.kaggle.push --no-build    # push what is on disk
    python -m curation_lab.kaggle.push --log-only    # just refetch the last log
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KAGGLE_EXE = os.path.join(REPO_ROOT, ".venv-tools", "Scripts", "kaggle.exe")
DEFAULT_DIR = os.path.join(REPO_ROOT, "kaggle_uploads", "tar-gpu")
KERNEL_ID = "talkraicer/multabench-tar-gpu"
POLL_SECONDS = 20
# A cold GPU notebook is queue + container boot + pip + HF model pull before it
# reaches any of our code; 45 min covers a full ft run too.
TIMEOUT_SECONDS = 45 * 60


def load_token() -> str:
    """Read KAGGLE_KEY out of .env without importing dotenv."""
    env_path = os.path.join(REPO_ROOT, ".env")
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("KAGGLE_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"KAGGLE_KEY not found in {env_path}")


def _env(**extra: str) -> dict:
    """Child-process env, forced to UTF-8 end to end.

    PYTHONIOENCODING covers stdout/stderr -- needed because the repo path contains
    Hebrew and the console codepage here is cp1255/cp1252. PYTHONUTF8 additionally
    changes the default for open(), which is what the kaggle CLI uses to write the
    downloaded run log; without it the log lands empty (0 bytes) whenever the run
    printed an emoji, e.g. the "LoRA on last N layers" line in e5_finetune.py."""
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **extra)


def kaggle(*args: str, check: bool = True) -> str:
    env = _env(KAGGLE_API_TOKEN=load_token())
    proc = subprocess.run([KAGGLE_EXE, *args], capture_output=True, text=True,
                          env=env, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def build(out_dir: str, machine_shape: str | None = None, cpu: bool = False,
          full: bool = False, full_epochs: int = 10) -> None:
    cmd = [sys.executable, "-m", "curation_lab.kaggle.build_notebook", "--out", out_dir]
    if machine_shape:
        cmd += ["--machine-shape", machine_shape]
    if cpu:
        cmd += ["--cpu"]
    if full:
        cmd += ["--full", "--full-epochs", str(full_epochs)]
    subprocess.check_call(cmd, cwd=REPO_ROOT, env=_env())


def wait_for_run(kernel_id: str = KERNEL_ID) -> str:
    """Poll until the kernel leaves the running/queued state. Returns final status."""
    started = time.time()
    last = ""
    while time.time() - started < TIMEOUT_SECONDS:
        status = kaggle("kernels", "status", kernel_id, check=False).strip()
        if status != last:
            print(f"[{int(time.time() - started):5d}s] {status}", flush=True)
            last = status
        low = status.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            return status
        time.sleep(POLL_SECONDS)
    return f"TIMEOUT after {TIMEOUT_SECONDS}s (last: {last})"


def fetch_log(dest: str, kernel_id: str = KERNEL_ID) -> None:
    os.makedirs(dest, exist_ok=True)
    print(kaggle("kernels", "output", kernel_id, "-p", dest, check=False))
    logs = glob.glob(os.path.join(dest, "*.log"))
    if not logs:
        print(f"!! no .log in {dest}; files: {os.listdir(dest)}")
        return
    print(f"\n===== {logs[0]} =====")
    _print_log(logs[0])


def _print_log(path: str) -> None:
    """Kaggle logs are a JSON array of {stream_name, time, data} records."""
    import json

    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    if not raw.strip():
        print("!! log file is empty -- the CLI failed to write it (encoding?), "
              "or the kernel produced no output at all.")
        return
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return
    for rec in records:
        data = rec.get("data", "")
        if rec.get("stream_name") == "stderr":
            data = "".join(f"! {ln}\n" for ln in data.splitlines())
        sys.stdout.write(data if data.endswith("\n") else data + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--log-only", action="store_true", help="Refetch the last run's log and exit.")
    p.add_argument("--machine-shape", default=None,
                   help="Accelerator override, e.g. to escape the sm_60 P100 default.")
    p.add_argument("--cpu", action="store_true",
                   help="Push the CPU validation variant (separate kernel, no GPU quota).")
    p.add_argument("--full", action="store_true",
                   help="Run the all-vs-ft measurement instead of the smoke test.")
    p.add_argument("--full-epochs", type=int, default=10)
    args = p.parse_args()

    kernel_id = KERNEL_ID.replace("-tar-gpu", "-tar-cpu") if args.cpu else KERNEL_ID
    work_dir = args.dir + ("-cpu" if args.cpu and args.dir == DEFAULT_DIR else "")
    out_dir = os.path.join(work_dir, "output")
    if args.log_only:
        fetch_log(out_dir, kernel_id)
        return

    if not args.no_build:
        build(work_dir, args.machine_shape, cpu=args.cpu,
              full=args.full, full_epochs=args.full_epochs)
    push_args = ["kernels", "push", "-p", work_dir]
    if args.machine_shape:
        push_args += ["--accelerator", args.machine_shape]
    print(kaggle(*push_args))
    if args.no_wait:
        return
    status = wait_for_run(kernel_id)
    print(f"\nfinal status: {status}")
    fetch_log(out_dir, kernel_id)


if __name__ == "__main__":
    main()
