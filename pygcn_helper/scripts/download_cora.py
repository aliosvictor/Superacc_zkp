from __future__ import annotations

import argparse
from pathlib import Path

from pygcn_g.data import download_cora_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Cora citation dataset required for training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cora"),
        help="Destination directory for the raw Cora files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_cora_dataset(args.output_dir)
    print(f"Downloaded Cora dataset into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
