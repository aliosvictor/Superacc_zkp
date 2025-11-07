"""
Core package for training and exporting PyTorch Graph Convolutional Networks
compatible with the Superacc ZKP pipeline.
"""

from .data import CoraDataset, load_cora_dataset, download_cora_dataset
from .models import GCN
from .training import TrainingConfig, train_model
from .export import export_weights_to_json, state_dict_to_weights_dict

__all__ = [
    "CoraDataset",
    "load_cora_dataset",
    "download_cora_dataset",
    "GCN",
    "TrainingConfig",
    "train_model",
    "export_weights_to_json",
    "state_dict_to_weights_dict",
]
