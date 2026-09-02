"""The manifest is what turns "nothing was lost" from a claim into a check."""
import json
from pathlib import Path

from curation_lab.tools.manifest import build, check, sha256_of, write_manifest


def _seed(root: Path) -> None:
    (root / "results").mkdir(parents=True)
    (root / "results" / "a.csv").write_text("alpha\n", encoding="utf-8")
    (root / "results" / "b.log").write_text("bravo\n", encoding="utf-8")


def test_sha256_is_stable_and_content_addressed(tmp_path: Path) -> None:
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("same\n", encoding="utf-8")
    two.write_text("same\n", encoding="utf-8")
    assert sha256_of(one) == sha256_of(two)


def test_build_uses_repo_relative_posix_paths(tmp_path: Path) -> None:
    _seed(tmp_path)
    entries = build([tmp_path / "results"], tmp_path)
    assert set(entries) == {"results/a.csv", "results/b.log"}


def test_a_renamed_file_is_not_reported_lost(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "curation").mkdir()
    (tmp_path / "results" / "a.csv").rename(tmp_path / "results" / "curation" / "grid.csv")

    assert check(manifest, tmp_path, exceptions=set()) == []


def test_a_deleted_file_is_reported_lost(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "b.log").unlink()

    assert check(manifest, tmp_path, exceptions=set()) == ["results/b.log"]


def test_an_intentionally_rewritten_file_is_excused_by_exceptions(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "b.log").write_text("rewritten\n", encoding="utf-8")

    assert check(manifest, tmp_path, exceptions={"results/b.log"}) == []


def test_manifest_round_trips_as_sorted_json(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    entries = build([tmp_path / "results"], tmp_path)
    write_manifest(entries, manifest)

    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert loaded == entries
    assert list(loaded) == sorted(loaded)
