# Source Inventory

## Core Crate Layout
- `src/lib.rs`: exports primary modules.
- `src/main.rs`: CLI example for inference pipeline.
- `src/data/`: dataset loader and PyTorch weight importer.
- `src/layers/`: graph convolution layer implementation.
- `src/math/`: dense/sparse matrix ops and activations.
- `src/models/`: GCN model orchestration.
- `src/types.rs`: shared data structures and aliases.
- `src/zkp/`: prover, verifier, and utility modules for proof generation.

## Experiments
- `experiments/gcn_full_feature.rs`: benchmark pipeline with operation tracking.
- `experiments/README.md`: instructions for regenerating artifacts locally.

## Data and Weights
- Create `data/cora/` locally and populate it with the `cora.content` and `cora.cites` files downloaded from the PyGCN repository or the documented fallback mirrors (the dataset itself is not versioned).
- `model_weights/` hosts JSON exports for 32-bit and 64-bit trained models.
- `pygcn_helper/` embeds the PyGCN-G training helper (a curated fork of https://github.com/tkipf/pygcn) so new checkpoints/exports can be produced without leaving the repository. Its scripts (`scripts/*.py`) expect `PYTHONPATH=src` and write temporaries only under `pygcn_helper/data` and `pygcn_helper/outputs`, both ignored by Git.

## Cargo Configuration
- Default feature set enables pure GCN inference (`gcn-only`).
- Optional features: `pytorch`, `ndarray`, and `zkp` (pulling local `libspartan`).
- Multiple binaries registered for inference, verification, and experiment tooling.

## Excluded Assets For New Release
- Documentation under `docs/zkp/` is large and Chinese-language; will be summarized instead.
- Historical experiment outputs from the original workspace are omitted; regenerate fresh artifacts with the provided experiment runner.
- Integration tests rely on bulky fixtures and will be rewritten when the new layout stabilizes.
- The previous standalone `pygcn/` repository is no longer tracked separately; all relevant code lives inside `pygcn_helper/` with redundant notebooks, caches, and checkpoints removed.
