from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pygcn_g.export import export_weights_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a saved PyTorch checkpoint to the JSON format expected by Superacc ZKP."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the *.pt file produced by training.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON path (defaults to <checkpoint>_weights.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a state dict dictionary in the checkpoint file.")

    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_weights.json"
    )
    export_weights_to_json(state_dict, output_path)
    print(f"Exported weights to {output_path.resolve()}")


if __name__ == "__main__":
    main()
