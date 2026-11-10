"""Experiment provenance: git state, environment versions, run manifests."""

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_git(arguments, cwd):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_commit(cwd=None):
    """Return the current git commit hash, or None outside a repository."""
    return _run_git(["rev-parse", "HEAD"], cwd or Path.cwd())


def git_is_dirty(cwd=None):
    """Return True when the working tree has changes, or None outside a repo."""
    status = _run_git(["status", "--porcelain"], cwd or Path.cwd())
    if status is None:
        return None
    return bool(status)


def package_versions(names=("torch", "timm", "numpy", "scikit-learn", "pandas")):
    """Return installed versions for the requested packages, None if absent."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(config_path, config, seed, device, command=None, cwd=None):
    """Build a JSON-serializable record of how a run was produced."""
    working_directory = Path(cwd) if cwd is not None else Path.cwd()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": dict(config),
        "seed": seed,
        "device": str(device),
        "command": list(command) if command is not None else list(sys.argv),
        "git_commit": git_commit(working_directory),
        "git_dirty": git_is_dirty(working_directory),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
