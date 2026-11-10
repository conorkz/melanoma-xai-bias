"""Tests for melanoma.tracking."""

import json
from pathlib import Path

from melanoma.tracking import build_manifest, git_commit, git_is_dirty, package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_git_commit_returns_hash_in_repo():
    commit = git_commit(PROJECT_ROOT)
    assert commit is not None
    assert len(commit) == 40
    int(commit, 16)


def test_git_helpers_return_none_outside_repo(tmp_path):
    assert git_commit(tmp_path) is None
    assert git_is_dirty(tmp_path) is None


def test_git_is_dirty_returns_bool_in_repo():
    assert isinstance(git_is_dirty(PROJECT_ROOT), bool)


def test_package_versions_includes_installed_and_missing():
    versions = package_versions(["torch", "definitely-not-a-package-xyz"])
    assert versions["torch"] is not None
    assert versions["definitely-not-a-package-xyz"] is None


def test_build_manifest_contains_provenance(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: test\n", encoding="utf-8")
    manifest = build_manifest(
        config_path,
        {"run_name": "test"},
        seed=7,
        device="cpu",
        command=["train.py"],
        cwd=PROJECT_ROOT,
    )
    assert manifest["seed"] == 7
    assert manifest["device"] == "cpu"
    assert manifest["command"] == ["train.py"]
    assert manifest["config"] == {"run_name": "test"}
    assert manifest["git_commit"] == git_commit(PROJECT_ROOT)
    assert "packages" in manifest
    assert "timestamp" in manifest
    json.dumps(manifest)
