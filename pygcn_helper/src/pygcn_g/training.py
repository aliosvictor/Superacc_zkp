from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .data import CoraDataset, accuracy, load_cora_dataset
from .models import GCN


@dataclass
class TrainingConfig:
    data_dir: Path
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    hidden_size: int = 16
    dropout: float = 0.5
    precision: str = "f32"
    seed: int = 42
    patience: int = 20
    use_gpu: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.precision not in {"f32", "f64"}:
            raise ValueError("precision must be 'f32' or 'f64'")


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float


@dataclass
class TrainingResult:
    config: TrainingConfig
    history: List[EpochStats] = field(default_factory=list)
    best_epoch: int = -1
    best_val_accuracy: float = 0.0
    best_state_dict: dict | None = None
    test_accuracy_at_best: float = 0.0


def _prepare_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_precision(
    dataset: CoraDataset, model: GCN, precision: str, device: torch.device
) -> tuple[CoraDataset, GCN]:
    if precision == "f64":
        torch.set_default_dtype(torch.float64)
        model = model.double()
        features = dataset.features.double()
        adjacency = dataset.adjacency.double()
    else:
        torch.set_default_dtype(torch.float32)
        model = model.float()
        features = dataset.features.float()
        adjacency = dataset.adjacency.float()

    adjacency = adjacency.coalesce()

    features = features.to(device)
    adjacency = adjacency.to(device)
    labels = dataset.labels.to(device)
    idx_train = dataset.idx_train.to(device)
    idx_val = dataset.idx_val.to(device)
    idx_test = dataset.idx_test.to(device)

    updated_dataset = CoraDataset(
        adjacency=adjacency,
        features=features,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )

    return updated_dataset, model.to(device)


def train_model(config: TrainingConfig) -> TrainingResult:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dataset = load_cora_dataset(config.data_dir)
    model = GCN(
        nfeat=dataset.features.shape[1],
        nhid=config.hidden_size,
        nclass=int(dataset.labels.max().item() + 1),
        dropout=config.dropout,
    )

    device = _prepare_device(config.use_gpu)
    dataset, model = _apply_precision(dataset, model, config.precision, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    result = TrainingResult(config=config)
    best_state_dict = None
    best_val_acc = -float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(dataset.features, dataset.adjacency)
        loss_train = F.nll_loss(output[dataset.idx_train], dataset.labels[dataset.idx_train])
        acc_train = accuracy(output[dataset.idx_train], dataset.labels[dataset.idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(dataset.features, dataset.adjacency)
            loss_val = F.nll_loss(logits[dataset.idx_val], dataset.labels[dataset.idx_val])
            acc_val = accuracy(logits[dataset.idx_val], dataset.labels[dataset.idx_val])
            loss_test = F.nll_loss(logits[dataset.idx_test], dataset.labels[dataset.idx_test])
            acc_test = accuracy(logits[dataset.idx_test], dataset.labels[dataset.idx_test])

        result.history.append(
            EpochStats(
                epoch=epoch,
                train_loss=float(loss_train.item()),
                train_accuracy=acc_train,
                val_loss=float(loss_val.item()),
                val_accuracy=acc_val,
                test_loss=float(loss_test.item()),
                test_accuracy=acc_test,
            )
        )

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            patience_counter = 0
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            result.best_epoch = epoch
            result.best_val_accuracy = acc_val
            result.test_accuracy_at_best = acc_test
        else:
            patience_counter += 1

        if config.patience > 0 and patience_counter >= config.patience:
            break

    result.best_state_dict = best_state_dict
    if result.best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    return result
