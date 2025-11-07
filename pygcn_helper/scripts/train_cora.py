from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from pygcn_g.export import export_weights_to_json
from pygcn_g.training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Cora GCN model and export weights for Superacc ZKP."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cora"),
        help="Directory containing cora.content and cora.cites.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="L2 regularization factor.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden dimension of the GCN.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=("f32", "f64"),
        default="f32",
        help="Floating-point precision for training and export.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable CUDA training when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for checkpoints and exported weights.",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export JSON weights compatible with Superacc ZKP after training.",
    )
    return parser.parse_args()


def _format_run_name(precision: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_gcn_{precision}"


def main() -> None:
    args = parse_args()

    run_name = _format_run_name(args.precision)
    output_root = args.output_dir.expanduser()
    checkpoints_dir = output_root / "checkpoints"
    weights_dir = output_root / "weights"
    reports_dir = output_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.export_json:
        weights_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        precision=args.precision,
        seed=args.seed,
        patience=args.patience,
        use_gpu=args.use_gpu,
    )

    result = train_model(config)
    checkpoint_path = checkpoints_dir / f"{run_name}_best.pt"
    torch.save(result.best_state_dict, checkpoint_path)

    report = {
        "run_name": run_name,
        "precision": args.precision,
        "best_epoch": result.best_epoch,
        "best_val_accuracy": result.best_val_accuracy,
        "test_accuracy_at_best": result.test_accuracy_at_best,
        "config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "patience": args.patience,
            "seed": args.seed,
            "use_gpu": args.use_gpu,
        },
    }

    report_path = reports_dir / f"{run_name}_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Best checkpoint saved to {checkpoint_path}")
    print(
        f"Validation accuracy: {result.best_val_accuracy:.4f} | "
        f"Test accuracy: {result.test_accuracy_at_best:.4f}"
    )

    if args.export_json:
        weights_path = weights_dir / f"{run_name}_weights.json"
        export_weights_to_json(result.best_state_dict, weights_path)
        print(f"Weight export written to {weights_path}")


if __name__ == "__main__":
    main()
