#!/usr/bin/env python
"""Smoke test: train a small baseline on a HAM10000 subset and report metrics."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.config import load_config
from melanoma.data.dataset import HAM10000Dataset
from melanoma.data.metadata import load_metadata
from melanoma.data.splits import make_splits
from melanoma.data.subsets import stratified_subset
from melanoma.data.transforms import build_transforms
from melanoma.models.factory import build_model
from melanoma.training.losses import FocalLoss
from melanoma.training.sampler import compute_class_weights, make_weighted_sampler
from melanoma.training.trainer import evaluate_metrics, fit
from melanoma.utils import ensure_dir, get_device, seed_everything
from melanoma.viz.plots import plot_training_history


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HAM10000 smoke test training run")
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--subset-size", type=int, default=None, help="Override subset size")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics and plots")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(resolve(args.config))
    seed_everything(config.seed)

    device = get_device(args.device)
    epochs = args.epochs if args.epochs is not None else config.training.epochs
    subset_size = args.subset_size if args.subset_size is not None else config.data.get("subset_size")
    output_dir = ensure_dir(resolve(args.output_dir or f"outputs/{config.run_name}"))
    checkpoint = resolve(args.checkpoint or f"checkpoints/{config.run_name}.pt")
    images_dir = resolve(config.data.images_dir)
    metadata_csv = resolve(config.data.metadata_csv)

    if not metadata_csv.exists():
        print(f"Metadata not found at {metadata_csv}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    print(f"Device: {device} | epochs: {epochs} | subset: {subset_size} | backbone: {config.model.backbone}")
    frame = load_metadata(metadata_csv)
    subset = stratified_subset(frame, subset_size, seed=config.seed)
    splits = make_splits(
        subset,
        val_fraction=config.data.split.val_fraction,
        test_fraction=config.data.split.test_fraction,
        seed=config.seed,
    )
    train_frame = splits[splits["split"] == "train"]
    val_frame = splits[splits["split"] == "val"]
    print(f"Split sizes: train={len(train_frame)} val={len(val_frame)}")

    img_size = config.training.img_size
    train_dataset = HAM10000Dataset(
        train_frame, images_dir, transform=build_transforms(img_size, train=True), task=config.data.task
    )
    val_dataset = HAM10000Dataset(
        val_frame, images_dir, transform=build_transforms(img_size, train=False), task=config.data.task
    )

    num_workers = config.training.num_workers
    if config.training.sampler == "class_balanced":
        sampler = make_weighted_sampler(train_dataset.labels, generator=torch.Generator().manual_seed(config.seed))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=num_workers
    )

    model = build_model(config.model.backbone, num_classes=config.model.num_classes, pretrained=config.model.pretrained)
    model.to(device)

    if config.training.loss == "focal":
        criterion = FocalLoss(gamma=config.training.focal_gamma, alpha=config.training.focal_alpha)
    elif config.training.loss == "weighted_ce":
        weights = compute_class_weights(train_dataset.labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

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
        patience=config.training.get("early_stopping_patience"),
        checkpoint_path=checkpoint,
        verbose=True,
    )
    elapsed = time.time() - start

    final_metrics = evaluate_metrics(model, val_loader, device)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": config.run_name,
                "backbone": config.model.backbone,
                "epochs": epochs,
                "device": str(device),
                "elapsed_seconds": round(elapsed, 1),
                "history": history,
                "final_val_metrics": final_metrics,
            },
            handle,
            indent=2,
        )
    plot_training_history(history, output_dir / "training_curves.png")

    print("")
    print(f"Done in {elapsed / 60:.1f} minutes")
    print(
        f"Best epoch {history['best_epoch']} | balanced_accuracy={final_metrics['balanced_accuracy']:.3f} "
        f"sensitivity={final_metrics['sensitivity']:.3f} specificity={final_metrics['specificity']:.3f} "
        f"auc={final_metrics['auc']:.3f}"
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
