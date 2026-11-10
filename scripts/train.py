#!/usr/bin/env python
"""Config-driven training and evaluation for the RQ1 benchmark.

The script trains one configuration, restores the best validation checkpoint,
selects an operating threshold on validation at a target specificity, evaluates
the untouched test split once, and writes per-sample predictions plus a run
manifest for reproducible aggregation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.config import load_config
from melanoma.data.dataset import HAM10000Dataset
from melanoma.data.metadata import load_metadata
from melanoma.data.splits import make_splits, split_summary
from melanoma.data.subsets import stratified_subset
from melanoma.data.transforms import build_transforms
from melanoma.models.factory import build_model
from melanoma.tracking import build_manifest
from melanoma.training.losses import FocalLoss
from melanoma.training.metrics import binary_metrics, bootstrap_ci, sensitivity_at_specificity
from melanoma.training.sampler import compute_class_weights, make_weighted_sampler
from melanoma.training.trainer import evaluate, fit
from melanoma.utils import ensure_dir, get_device, seed_everything
from melanoma.viz.plots import plot_roc_curve, plot_training_history


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate a melanoma classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--img-size", type=int, default=None, help="Override config image size")
    parser.add_argument("--subset-size", type=int, default=None, help="Debug on a subset")
    parser.add_argument("--split-file", default=None, help="Override data.split.file")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug: limit batches per epoch")
    parser.add_argument("--no-test", action="store_true", help="Skip test split evaluation")
    return parser.parse_args(argv)


def config_id(config):
    """Stable identifier used to group seeds of the same configuration."""
    explicit = config.get("config_id")
    if explicit:
        return str(explicit)
    backbone = str(config.model.get("backbone", "model"))
    loss = str(config.training.get("loss", "ce"))
    parts = [backbone, loss]
    if loss == "focal":
        parts.append(f"g{config.training.get('focal_gamma', 2.0)}")
    parts.append(str(config.training.get("sampler", "random")))
    parts.append(str(config.training.get("augmentation", "standard")))
    parts.append(f"{config.training.get('img_size', 224)}px")
    return "_".join(parts)


def resolve_splits(frame: pd.DataFrame, data_config) -> pd.DataFrame:
    """Load a cached split file or create and persist a lesion-aware split.

    A cached split is reused across seeds so every run sees the same test set.
    If the cached file does not cover the current frame (for example because a
    different subset was requested), raise instead of silently mixing splits.
    """
    split_config = data_config.get("split", {}) or {}
    split_file = split_config.get("file")
    path = resolve(split_file) if split_file else None

    if path is not None and path.exists():
        saved = pd.read_csv(path)
        missing_columns = {"image_id", "split"} - set(saved.columns)
        if missing_columns:
            raise ValueError(f"Split file {path} is missing columns: {sorted(missing_columns)}")
        missing_images = set(frame["image_id"]) - set(saved["image_id"])
        if missing_images:
            raise ValueError(
                f"Split file {path} does not cover {len(missing_images)} images; "
                "delete it or point data.split.file at a new path"
            )
        merged = frame.merge(
            saved[["image_id", "split"]], on="image_id", how="left", validate="one_to_one"
        )
        invalid = set(merged["split"].dropna().unique()) - {"train", "val", "test"}
        if invalid or merged["split"].isna().any():
            raise ValueError(
                f"Split file {path} contains invalid split labels: {sorted(map(str, invalid))}"
            )
        return merged

    splits = make_splits(
        frame,
        val_fraction=float(split_config.get("val_fraction", 0.15)),
        test_fraction=float(split_config.get("test_fraction", 0.15)),
        seed=int(split_config.get("seed", 42)),
    )
    if path is not None:
        ensure_dir(path.parent)
        splits[["image_id", "split"]].to_csv(path, index=False)
    return splits


def build_loss(config, labels, device):
    """Create the configured loss function."""
    name = str(config.training.get("loss", "ce"))
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        weights = compute_class_weights(labels).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(
            gamma=float(config.training.get("focal_gamma", 2.0)),
            alpha=config.training.get("focal_alpha"),
        )
    raise ValueError(f"Unknown loss {name!r}; expected ce, weighted_ce, or focal")


def make_loader(dataset, config, train: bool, seed: int, device):
    """Create a DataLoader, optionally with class-balanced sampling."""
    sampler_name = str(config.training.get("sampler", "random"))
    if sampler_name not in ("random", "class_balanced"):
        raise ValueError(f"Unknown sampler {sampler_name!r}; expected random or class_balanced")
    batch_size = int(config.training.batch_size)
    num_workers = int(config.training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    if train and sampler_name == "class_balanced":
        sampler = make_weighted_sampler(
            dataset.labels, generator=torch.Generator().manual_seed(seed)
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def metrics_at_threshold(labels, probabilities, threshold: float) -> dict:
    """Binary metrics at a fixed operating threshold, with the threshold recorded."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    result = binary_metrics(labels, predictions, probabilities)
    result["threshold"] = float(threshold)
    return result


def save_predictions(
    path,
    dataset,
    probabilities,
    labels,
    threshold: float,
    split: str,
    run_id: str,
    config_identifier: str,
    seed: int,
):
    """Write one row per evaluated image for later fairness and XAI joins."""
    if not (len(dataset.image_ids) == len(probabilities) == len(labels)):
        raise ValueError(
            "Prediction export alignment failed: "
            f"image_ids={len(dataset.image_ids)} probs={len(probabilities)} labels={len(labels)}"
        )
    frame = pd.DataFrame(
        {
            "image_id": dataset.image_ids,
            "split": split,
            "y_true": np.asarray(labels, dtype=int),
            "y_prob": np.asarray(probabilities, dtype=float),
            "y_pred": (np.asarray(probabilities) >= threshold).astype(int),
            "threshold": float(threshold),
            "run_id": run_id,
            "config_id": config_identifier,
            "seed": int(seed),
        }
    )
    frame.to_csv(path, index=False)
    return path


def summary_as_dict(summary: pd.DataFrame) -> dict:
    """Convert a split summary to JSON-safe builtin types."""
    return {
        str(split): {str(column): int(value) for column, value in row.items()}
        for split, row in summary.iterrows()
    }


def json_safe(value):
    """Recursively convert numpy values and non-finite floats to strict JSON.

    NaN and infinity become null so the metrics file parses with any strict
    JSON reader, not only Python's permissive default.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv=None):
    args = parse_args(argv)
    config = load_config(resolve(args.config))

    seed = int(args.seed) if args.seed is not None else int(config.seed)
    config["seed"] = seed
    seed_everything(seed)

    device = get_device(args.device)
    run_name = str(config.get("run_name") or Path(args.config).stem)
    output_dir = ensure_dir(resolve(args.output_dir or f"outputs/runs/{run_name}"))
    checkpoint = resolve(args.checkpoint or f"checkpoints/{run_name}.pt")

    metadata_csv = resolve(config.data.metadata_csv)
    images_dir = resolve(config.data.images_dir)
    if not metadata_csv.exists():
        print(f"Metadata not found at {metadata_csv}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"Images directory not found at {images_dir}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    task = str(config.data.get("task", "binary"))
    if task != "binary":
        raise ValueError("train.py currently supports the binary melanoma task only")
    num_classes = int(config.model.get("num_classes", 2))
    if num_classes != 2:
        raise ValueError("Binary training requires model.num_classes: 2")

    epochs = int(args.epochs) if args.epochs is not None else int(config.training.epochs)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    config.training["epochs"] = epochs

    img_size = int(args.img_size) if args.img_size is not None else int(config.training.img_size)
    config.training["img_size"] = img_size

    if args.split_file is not None:
        config["data"].setdefault("split", {})
        config["data"]["split"]["file"] = args.split_file

    max_batches = (
        int(args.max_batches)
        if args.max_batches is not None
        else config.training.get("max_batches")
    )
    if max_batches is not None:
        max_batches = int(max_batches)
        config.training["max_batches"] = max_batches

    frame = load_metadata(metadata_csv)
    if args.subset_size is not None:
        split_seed = int(config.data.get("split", {}).get("seed", 42))
        frame = stratified_subset(frame, args.subset_size, seed=split_seed)

    splits = resolve_splits(frame, config.data)
    summary = split_summary(splits)
    print(summary.to_string())
    print("")

    train_frame = splits[splits["split"] == "train"]
    val_frame = splits[splits["split"] == "val"]
    test_frame = splits[splits["split"] == "test"]
    for name, split_frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        if split_frame.empty:
            print(f"Split '{name}' is empty; adjust data.split fractions.", file=sys.stderr)
            return 1

    augmentation = str(config.training.get("augmentation", "standard"))
    if augmentation not in ("standard", "strong"):
        raise ValueError(f"Unknown augmentation {augmentation!r}; expected standard or strong")

    train_dataset = HAM10000Dataset(
        train_frame,
        images_dir,
        transform=build_transforms(img_size, train=True, strong=(augmentation == "strong")),
        task="binary",
    )
    val_dataset = HAM10000Dataset(
        val_frame,
        images_dir,
        transform=build_transforms(img_size, train=False),
        task="binary",
    )
    test_dataset = HAM10000Dataset(
        test_frame,
        images_dir,
        transform=build_transforms(img_size, train=False),
        task="binary",
    )

    train_loader = make_loader(train_dataset, config, train=True, seed=seed, device=device)
    val_loader = make_loader(val_dataset, config, train=False, seed=seed, device=device)
    test_loader = make_loader(test_dataset, config, train=False, seed=seed, device=device)

    backbone = str(config.model.backbone)
    model = build_model(
        backbone,
        num_classes=2,
        pretrained=bool(config.model.get("pretrained", True)),
        drop_rate=float(config.model.get("drop_rate", 0.0)),
        img_size=img_size,
    )
    model.to(device)

    criterion = build_loss(config, train_dataset.labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    patience = config.training.get("early_stopping_patience")
    patience = int(patience) if patience is not None else None
    grad_clip = config.training.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    print(
        f"Run: {run_name} | device: {device} | backbone: {backbone} | "
        f"loss: {config.training.get('loss', 'ce')} | sampler: {config.training.get('sampler', 'random')} | "
        f"augmentation: {augmentation} | img_size: {img_size} | epochs: {epochs}"
    )
    start = time.time()
    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=epochs,
        scheduler=scheduler,
        patience=patience,
        checkpoint_path=checkpoint,
        grad_clip=grad_clip,
        max_batches=max_batches,
        verbose=True,
    )
    elapsed = time.time() - start

    val_probabilities, val_labels = evaluate(model, val_loader, device)
    evaluation_config = config.get("evaluation", {}) or {}
    target_specificity = float(evaluation_config.get("target_specificity", 0.95))
    selected = sensitivity_at_specificity(val_labels, val_probabilities, target_specificity)
    if np.isfinite(selected["threshold"]):
        threshold = float(selected["threshold"])
        threshold_source = "val_target_specificity"
    else:
        threshold = 0.5
        threshold_source = "fallback_0.5"

    val_at_05 = metrics_at_threshold(val_labels, val_probabilities, 0.5)
    val_at_target = metrics_at_threshold(val_labels, val_probabilities, threshold)

    test_payload = None
    test_labels = None
    test_probabilities = None
    if not args.no_test:
        test_probabilities, test_labels = evaluate(model, test_loader, device)
        test_at_05 = metrics_at_threshold(test_labels, test_probabilities, 0.5)
        test_at_target = metrics_at_threshold(test_labels, test_probabilities, threshold)
        bootstrap_config = evaluation_config.get("bootstrap", {}) or {}
        n_boot = int(bootstrap_config.get("n_boot", 1000))
        confidence = float(bootstrap_config.get("confidence", 0.95))
        bootstrap_seed = int(bootstrap_config.get("seed", 0))
        bootstrap = {}
        if n_boot > 0:
            for metric in ("sensitivity", "specificity", "balanced_accuracy", "auc"):
                bootstrap[metric] = bootstrap_ci(
                    test_labels,
                    test_probabilities,
                    metric=metric,
                    threshold=threshold,
                    n_boot=n_boot,
                    confidence=confidence,
                    seed=bootstrap_seed,
                )
        test_payload = {"at_0.5": test_at_05, "at_target": test_at_target, "bootstrap": bootstrap}

    identifier = config_id(config)
    save_predictions(
        output_dir / "predictions_val.csv",
        val_dataset,
        val_probabilities,
        val_labels,
        threshold,
        "val",
        run_name,
        identifier,
        seed,
    )
    if test_probabilities is not None:
        save_predictions(
            output_dir / "predictions_test.csv",
            test_dataset,
            test_probabilities,
            test_labels,
            threshold,
            "test",
            run_name,
            identifier,
            seed,
        )

    payload = {
        "run_id": run_name,
        "config_id": identifier,
        "run_name": run_name,
        "seed": seed,
        "backbone": backbone,
        "img_size": img_size,
        "loss": str(config.training.get("loss", "ce")),
        "focal_gamma": (
            config.training.get("focal_gamma")
            if str(config.training.get("loss", "ce")) == "focal"
            else None
        ),
        "sampler": str(config.training.get("sampler", "random")),
        "augmentation": augmentation,
        "epochs": epochs,
        "best_epoch": history["best_epoch"],
        "stopped_early": history["stopped_early"],
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "target_specificity": target_specificity,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "split_summary": summary_as_dict(summary),
        "val": {"at_0.5": val_at_05, "at_target": val_at_target},
        "test": test_payload,
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)

    manifest = build_manifest(
        config_path=resolve(args.config),
        config=config,
        seed=seed,
        device=device,
        command=[Path(__file__).name, *(list(argv) if argv is not None else list(sys.argv[1:]))],
    )
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, allow_nan=False)

    plot_training_history(history, output_dir / "training_curves.png")
    if test_probabilities is not None and len(np.unique(test_labels)) == 2:
        plot_roc_curve(test_labels, test_probabilities, output_dir / "roc_test.png")

    print("")
    print(f"Done in {elapsed / 60:.1f} minutes | best epoch {history['best_epoch']}")
    print(f"Threshold {threshold:.4f} ({threshold_source})")
    if test_payload is None:
        print(
            f"Validation | sensitivity={val_at_target['sensitivity']:.3f} "
            f"specificity={val_at_target['specificity']:.3f} AUC={val_at_target['auc']:.3f}"
        )
    else:
        print(
            f"Test | sensitivity={test_payload['at_target']['sensitivity']:.3f} "
            f"specificity={test_payload['at_target']['specificity']:.3f} "
            f"AUC={test_payload['at_target']['auc']:.3f}"
        )
        if test_payload["bootstrap"]:
            interval = test_payload["bootstrap"]["sensitivity"]
            print(
                f"Sensitivity 95% CI: {interval['low']:.3f} to {interval['high']:.3f} "
                f"({interval['n_boot']} bootstrap samples)"
            )
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {output_dir / 'predictions_val.csv'}")
    if test_probabilities is not None:
        print(f"Predictions: {output_dir / 'predictions_test.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
