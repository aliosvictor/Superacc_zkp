from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import scipy.sparse as sp
import torch

CORADATA_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora."
CORADATA_FILES = ["content", "cites"]


@dataclass
class CoraDataset:
    adjacency: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor


def download_cora_dataset(destination: Path) -> None:
    """
    Download the Cora citation dataset into the provided directory.

    Parameters
    ----------
    destination:
        Directory that will contain the raw `cora.content` and `cora.cites`
        files. The directory is created when missing.
    """

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for suffix in CORADATA_FILES:
        target = destination / f"cora.{suffix}"
        if target.exists():
            continue
        url = f"{CORADATA_URL}{suffix}"
        urlretrieve(url, target)


def _encode_onehot(labels: np.ndarray) -> np.ndarray:
    classes = sorted(set(labels))
    mapping = {label: np.identity(len(classes))[i, :] for i, label in enumerate(classes)}
    return np.array([mapping[label] for label in labels], dtype=np.int32)


def _normalize_sparse(matrix: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(matrix)


def _sparse_to_torch(sparse_matrix: sp.spmatrix) -> torch.Tensor:
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_cora_dataset(
    data_dir: Path, dataset_name: str = "cora"
) -> CoraDataset:
    """
    Load the Cora dataset from disk.

    Parameters
    ----------
    data_dir:
        Directory that contains the `cora.content` and `cora.cites` files.
    dataset_name:
        Reserved for future extensions. Only `cora` is supported today.
    """

    data_dir = data_dir.expanduser().resolve()
    if dataset_name != "cora":
        raise ValueError("Only the Cora dataset is supported")

    content_path = data_dir / f"{dataset_name}.content"
    cites_path = data_dir / f"{dataset_name}.cites"

    if not content_path.exists() or not cites_path.exists():
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}. "
            f"Use download_cora_dataset() or scripts/download_cora.py first."
        )

    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = _encode_onehot(idx_features_labels[:, -1])

    indices = np.array(idx_features_labels[:, 0], dtype=np.int32)
    index_map = {index: i for i, index in enumerate(indices)}

    edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)
    edges = np.array(
        [index_map.get(edge) for edge in edges_unordered.flatten()],
        dtype=np.int32,
    ).reshape(edges_unordered.shape)

    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(
        adjacency.T > adjacency
    )

    features = _normalize_sparse(features)
    adjacency = _normalize_sparse(adjacency + sp.eye(adjacency.shape[0]))

    idx_train = torch.arange(0, 140, dtype=torch.long)
    idx_val = torch.arange(200, 500, dtype=torch.long)
    idx_test = torch.arange(500, 1500, dtype=torch.long)

    features_tensor = torch.FloatTensor(np.array(features.todense()))
    labels_tensor = torch.LongTensor(np.where(labels)[1])
    adjacency_tensor = _sparse_to_torch(adjacency)

    return CoraDataset(
        adjacency=adjacency_tensor,
        features=features_tensor,
        labels=labels_tensor,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double().sum()
    return float(correct / len(labels))
