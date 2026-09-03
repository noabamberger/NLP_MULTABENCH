"""Content-addressed manifest of the repo's evidence files.

A reorganization moves and renames files. Recording paths alone would flag every
rename as a change; recording content hashes means only genuine loss shows up.
`check` therefore asks "does this content still exist anywhere?", not "is it still
at this path?".

The limit of that question is worth stating, because it bounds what a clean result
proves: `check` confirms custody, not correctness. Content moved somewhere
nonsensical still counts as present, and a file that was never recorded is invisible
to it. It answers "was anything destroyed", not "is everything where it belongs".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 1 << 20

# Shared by build() and _hashes_present(): if only the presence scan skipped these,
# build() could record an entry the scan never sees, and check() would report a
# false loss.
_SKIP = {".git", ".venv", "__pycache__", ".tar_cache", ".emb_cache"}


def _skipped(path: Path) -> bool:
    return any(part in _SKIP for part in path.parts)


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
            if path.is_file() and not _skipped(path):
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
    for path in repo_root.rglob("*"):
        if path.is_file() and not _skipped(path):
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


def _main(argv: list[str] | None = None) -> int:
    """`python -m curation_lab.tools.manifest check <manifest.json> [exceptions...]`

    Exists so the custody claim can be re-run by a reader rather than only by
    whoever wrote the throwaway script that first produced it.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--except",
        dest="exceptions",
        nargs="*",
        default=[],
        help="repo-relative paths whose content is expected to have changed",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    lost = check(args.manifest, args.root.resolve(), set(args.exceptions))
    if lost:
        print(f"LOST ({len(lost)}):")
        for path in lost:
            print(f"  {path}")
        return 1
    print("LOST: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
