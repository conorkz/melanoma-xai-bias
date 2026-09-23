"""Minimal training and evaluation loops with checkpointing and early stopping."""

from pathlib import Path

import numpy as np
import torch

from melanoma.training.metrics import binary_metrics
from melanoma.utils import ensure_dir


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None, max_batches=None):
    """Run one training epoch and return the mean loss.

    Raises ValueError for an empty loader or max_batches < 1.
    """
    if len(loader) == 0:
        raise ValueError("Training loader is empty")
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches must be >= 1, got {max_batches}")
    model.train()
    total_loss = 0.0
    seen = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        seen += inputs.size(0)
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Return (probabilities, labels) as numpy arrays over the loader."""
    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty")
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches must be >= 1, got {max_batches}")
    model.eval()
    probs, labels = [], []
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        logits = model(inputs)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate_metrics(model, loader, device, threshold=0.5, max_batches=None):
    """Evaluate the model and return the binary_metrics dictionary."""
    probs, labels = evaluate(model, loader, device, max_batches=max_batches)
    preds = (probs >= threshold).astype(int)
    return binary_metrics(labels, preds, probs)


def save_checkpoint(path, model, extra=None):
    """Save a checkpoint dict with model weights and optional metadata."""
    path = Path(path)
    ensure_dir(path.parent)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, device="cpu"):
    """Load model weights from a checkpoint and return the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint at {path} has no 'model_state_dict' entry")
    model.load_state_dict(payload["model_state_dict"])
    return {key: value for key, value in payload.items() if key != "model_state_dict"}


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    scheduler=None,
    patience=None,
    checkpoint_path=None,
    grad_clip=None,
    max_batches=None,
    verbose=False,
):
    """Train with validation, early stopping, and best-checkpoint saving.

    Returns a history dict with per-epoch losses and metrics, the best epoch,
    and whether training stopped early.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    history = {"train_loss": [], "val_metrics": [], "best_epoch": None, "stopped_early": False}
    best_score = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip, max_batches
        )
        metrics = evaluate_metrics(model, val_loader, device, max_batches=max_batches)
        history["train_loss"].append(loss)
        history["val_metrics"].append(metrics)

        raw_score = metrics["balanced_accuracy"]
        score = raw_score if not np.isnan(raw_score) else -np.inf
        improved = score > best_score or history["best_epoch"] is None
        if improved:
            best_score = score
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, extra={"epoch": epoch, "metrics": metrics})
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"epoch {epoch}/{epochs} train_loss={loss:.4f} "
                f"balanced_acc={raw_score:.4f} auc={metrics['auc']:.4f}"
            )

        if patience is not None and epochs_without_improvement >= patience:
            history["stopped_early"] = True
            break

    return history
