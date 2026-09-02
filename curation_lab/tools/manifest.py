"""Content-addressed manifest of the repo's evidence files.

A reorganization moves and renames files. Recording paths alone would flag every
rename as a change; recording content hashes means only genuine loss shows up.
`check` therefore asks "does this content still exist anywhere?", not "is it still
at this path?".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build(roots: list[Path], repo_root: Path) -> dict[str, str]:
    """Map repo-relative POSIX path -> sha256 for every file under `roots`."""
    entries: dict[str, str] = {}
    for root in roots:
        if root.is_file():
            entries[root.relative_to(repo_root).as_posix()] = sha256_of(root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries[path.relative_to(repo_root).as_posix()] = sha256_of(path)
    return dict(sorted(entries.items()))


def write_manifest(entries: dict[str, str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(sorted(entries.items())), indent=2) + "\n",
        encoding="utf-8",
    )


def _hashes_present(repo_root: Path) -> set[str]:
    present: set[str] = set()
    skip = {".git", ".venv", "__pycache__", ".tar_cache", ".emb_cache"}
    for path in repo_root.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            present.add(sha256_of(path))
    return present


def check(manifest_path: Path, repo_root: Path, exceptions: set[str]) -> list[str]:
    """Return manifest paths whose content is no longer anywhere in the tree."""
    recorded: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    present = _hashes_present(repo_root)
    return [
        path
        for path, digest in sorted(recorded.items())
        if path not in exceptions and digest not in present
    ]
