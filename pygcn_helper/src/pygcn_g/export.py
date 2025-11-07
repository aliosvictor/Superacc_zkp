from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import torch


REQUIRED_KEYS: tuple[str, ...] = (
    "gc1.weight",
    "gc1.bias",
    "gc2.weight",
    "gc2.bias",
)


def _ensure_required_keys(state_dict: Dict[str, torch.Tensor]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in state_dict]
    if missing:
        raise KeyError(f"Missing keys in state dict: {', '.join(missing)}")


def state_dict_to_weights_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, list[float]]:
    _ensure_required_keys(state_dict)
    weights: Dict[str, list[float]] = {}
    for key in REQUIRED_KEYS:
        tensor = state_dict[key].detach().cpu().contiguous().view(-1)
        weights[key] = [float(value) for value in tensor.tolist()]
    return weights


def export_weights_to_json(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    indent: int = 2,
) -> Path:
    """
    Convert a trained model state dictionary into the JSON layout expected
    by the Rust inference pipeline.
    """

    weights_dict = state_dict_to_weights_dict(state_dict)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(weights_dict, handle, indent=indent)

    return output_path
