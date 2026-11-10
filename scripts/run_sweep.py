#!/usr/bin/env python
"""Expand a sweep config into run configs and optionally execute them."""

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.utils import ensure_dir


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a grid of training configurations")
    parser.add_argument("--config", required=True, help="Sweep YAML with base, seeds, and variants")
    parser.add_argument("--out-dir", default=None, help="Where generated configs are written")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Where training outputs are written")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Write configs but do not train")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with metrics.json")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed run")
    return parser.parse_args(argv)


def set_nested(config, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key such as training.loss."""
    parts = dotted_key.split(".")
    if not all(parts):
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    current = config
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot descend into non-mapping key '{part}' of {dotted_key!r}")
        current = next_value
    current[parts[-1]] = value


def expand_sweep(sweep):
    """Return one run description per variant and seed combination."""
    name = str(sweep.get("name", "sweep"))
    variants = sweep.get("variants") or []
    if not variants:
        raise ValueError("Sweep config must define a non-empty 'variants' list")
    seeds = sweep.get("seeds")
    if seeds is None:
        seeds = [sweep.get("seed", 42)]
    runs = []
    for variant in variants:
        if "name" not in variant:
            raise ValueError("Every sweep variant needs a 'name'")
        for seed in seeds:
            overrides = dict(variant.get("overrides", {}))
            overrides["seed"] = int(seed)
            runs.append(
                {
                    "name": f"{name}_{variant['name']}_seed{seed}",
                    "config_id": f"{name}_{variant['name']}",
                    "overrides": overrides,
                }
            )
    return runs


def main(argv=None):
    args = parse_args(argv)
    sweep_path = resolve(args.config)
    if not sweep_path.exists():
        print(f"Sweep config not found: {sweep_path}", file=sys.stderr)
        return 1
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(sweep, dict) or "base" not in sweep:
        print("Sweep config must be a mapping with a 'base' entry", file=sys.stderr)
        return 1

    base_path = resolve(sweep["base"])
    if not base_path.exists():
        print(f"Base config not found: {base_path}", file=sys.stderr)
        return 1
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    runs = expand_sweep(sweep)

    sweep_name = str(sweep.get("name", sweep_path.stem))
    out_dir = ensure_dir(resolve(args.out_dir or f"outputs/sweeps/{sweep_name}"))
    configs_dir = ensure_dir(out_dir / "configs")

    generated = []
    for run in runs:
        config = copy.deepcopy(base)
        for key, value in run["overrides"].items():
            set_nested(config, key, value)
        config["run_name"] = run["name"]
        config["config_id"] = run["config_id"]
        path = configs_dir / f"{run['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        generated.append((run, path))
        print(f"Wrote {path}")

    if args.dry_run:
        print(f"Dry run: generated {len(generated)} configs in {configs_dir}")
        return 0

    runs_dir = resolve(args.runs_dir)
    checkpoints_dir = resolve(args.checkpoints_dir)
    for run, path in generated:
        metrics_path = runs_dir / run["name"] / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"Skipping {run['name']}: {metrics_path} exists")
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            str(path),
            "--output-dir",
            str(runs_dir / run["name"]),
            "--checkpoint",
            str(checkpoints_dir / f"{run['name']}.pt"),
        ]
        print("Running:", " ".join(command))
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Run failed: {run['name']}", file=sys.stderr)
            if not args.keep_going:
                return 1

    print(f"Finished {len(generated)} configurations from {sweep_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
