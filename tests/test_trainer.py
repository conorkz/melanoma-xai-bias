"""Tests for melanoma.training.trainer."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from melanoma.training.trainer import (
    evaluate,
    evaluate_metrics,
    fit,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


def make_loaders(n=16, batch_size=4):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(n, 3, 8, 8, generator=generator)
    targets = torch.tensor([0, 1] * (n // 2))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset, batch_size=batch_size)


def empty_loader():
    dataset = TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long))
    return DataLoader(dataset, batch_size=4)


def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


def test_train_one_epoch_returns_finite_loss():
    model = TinyNet()
    train_loader, _ = make_loaders()
    loss = train_one_epoch(model, train_loader, make_optimizer(model), nn.CrossEntropyLoss(), "cpu")
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_train_one_epoch_empty_loader_raises():
    model = TinyNet()
    with pytest.raises(ValueError, match="empty"):
        train_one_epoch(model, empty_loader(), make_optimizer(model), nn.CrossEntropyLoss(), "cpu")


def test_train_one_epoch_zero_max_batches_raises():
    model = TinyNet()
    train_loader, _ = make_loaders()
    with pytest.raises(ValueError):
        train_one_epoch(
            model, train_loader, make_optimizer(model), nn.CrossEntropyLoss(), "cpu", max_batches=0
        )


def test_evaluate_returns_probabilities_and_labels():
    model = TinyNet()
    _, loader = make_loaders(n=12, batch_size=4)
    probs, labels = evaluate(model, loader, "cpu")
    assert probs.shape == (12,)
    assert labels.shape == (12,)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert set(labels.tolist()) == {0, 1}


def test_evaluate_empty_loader_raises():
    model = TinyNet()
    with pytest.raises(ValueError, match="empty"):
        evaluate(model, empty_loader(), "cpu")


def test_evaluate_max_batches_limits_batches():
    model = TinyNet()
    _, loader = make_loaders(n=16, batch_size=4)
    probs, _ = evaluate(model, loader, "cpu", max_batches=1)
    assert probs.shape == (4,)


def test_evaluate_metrics_has_expected_keys():
    model = TinyNet()
    _, loader = make_loaders(n=16, batch_size=4)
    metrics = evaluate_metrics(model, loader, "cpu")
    assert {"sensitivity", "specificity", "balanced_accuracy", "auc", "confusion"} <= set(metrics)


def test_fit_runs_and_saves_checkpoint(tmp_path):
    model = TinyNet()
    train_loader, val_loader = make_loaders()
    checkpoint = tmp_path / "nested" / "model.pt"
    history = fit(
        model,
        train_loader,
        val_loader,
        make_optimizer(model),
        nn.CrossEntropyLoss(),
        "cpu",
        epochs=2,
        checkpoint_path=checkpoint,
    )
    assert len(history["train_loss"]) == 2
    assert len(history["val_metrics"]) == 2
    assert history["best_epoch"] in {1, 2}
    assert history["stopped_early"] is False
    assert checkpoint.exists()

    metadata = load_checkpoint(checkpoint, TinyNet(), device="cpu")
    assert metadata["epoch"] == history["best_epoch"]
    assert "metrics" in metadata


@pytest.mark.parametrize("bad_epochs", [0, -2])
def test_fit_invalid_epochs_raises(bad_epochs):
    model = TinyNet()
    train_loader, val_loader = make_loaders()
    with pytest.raises(ValueError):
        fit(
            model,
            train_loader,
            val_loader,
            make_optimizer(model),
            nn.CrossEntropyLoss(),
            "cpu",
            epochs=bad_epochs,
        )


def test_fit_invalid_patience_raises():
    model = TinyNet()
    train_loader, val_loader = make_loaders()
    with pytest.raises(ValueError):
        fit(
            model,
            train_loader,
            val_loader,
            make_optimizer(model),
            nn.CrossEntropyLoss(),
            "cpu",
            epochs=1,
            patience=0,
        )


def test_fit_with_scheduler_and_patience():
    model = TinyNet()
    train_loader, val_loader = make_loaders()
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        "cpu",
        epochs=6,
        scheduler=scheduler,
        patience=1,
    )
    assert "stopped_early" in history
    assert history["best_epoch"] is not None


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = TinyNet()
    path = save_checkpoint(tmp_path / "model.pt", model, extra={"epoch": 3})
    fresh = TinyNet()
    metadata = load_checkpoint(path, fresh, device="cpu")
    assert metadata["epoch"] == 3
    for original, loaded in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(original, loaded)


def test_fit_restores_best_checkpoint(tmp_path, monkeypatch):
    import melanoma.training.trainer as trainer_module

    class BiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x * self.bias

    model = BiasNet()
    train_loader, val_loader = make_loaders(n=8, batch_size=4)
    scores = iter([0.9, 0.5, 0.4])

    def fake_train(model, loader, optimizer, criterion, device, grad_clip=None, max_batches=None):
        with torch.no_grad():
            model.bias.add_(1.0)
        return 0.1

    def fake_eval(model, loader, device, threshold=0.5, max_batches=None):
        return {"balanced_accuracy": next(scores), "auc": 0.5}

    monkeypatch.setattr(trainer_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(trainer_module, "evaluate_metrics", fake_eval)

    checkpoint = tmp_path / "best.pt"
    history = fit(
        model,
        train_loader,
        val_loader,
        make_optimizer(model),
        nn.CrossEntropyLoss(),
        "cpu",
        epochs=3,
        checkpoint_path=checkpoint,
    )

    assert history["best_epoch"] == 1
    assert model.bias.item() == pytest.approx(1.0)


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", TinyNet(), device="cpu")


def test_load_checkpoint_invalid_content_raises(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"something": 1}, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path, TinyNet(), device="cpu")
