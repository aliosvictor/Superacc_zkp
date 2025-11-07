# Superacc ZKP Architecture Overview

## Core Crate Layout
- `src/lib.rs` wires together the public API, exposing datasets, models, math utilities, and ZKP utilities behind feature flags.
- `src/data/` provides loaders for the Cora dataset and PyTorch weight import utilities.
- `src/types.rs` defines shared matrix, tensor, and configuration structures used throughout the crate.
- `src/layers/graph_conv.rs` implements the GCN layer with sparse adjacency and dense feature handling.
- `src/models/gcn.rs` hosts the high-level forward pass, accuracy, and loss routines mirroring PyGCN semantics.
- `src/math/` contains dense and sparse kernels plus activation functions, optionally instrumented for ZKP operation tracking.
- `src/zkp/` includes Spartan-friendly witness generation, verifiers, and helper gadgets (FL2SA, MULFP, SA2FL) guarded by the `zkp` feature flag.

## Experiment Binary
- `experiments/gcn_full_feature.rs` orchestrates the end-to-end ZKP benchmark on the full Cora graph.
  - Supports half/single/double precision witness generation.
  - Collects constraint counts, witness sizes, and delta metrics per layer.
  - Persists results into `./artifacts/gcn_full_feature/` for local analysis.

## Data & Weights
- Create `data/cora` locally and populate it with the `cora.content` and `cora.cites` files. Follow the README instructions to download them from the original PyGCN repository or the documented fallback sources.
- `model_weights/` stores sample JSON exports (f32/f64) taken from PyTorch training runs; the experiment CLI accepts alternate paths via flags.

## Build & Features
- `gcn-only` keeps the dependency footprint light for inference-only users.
- `zkp` pulls in Spartan (`libspartan`), field arithmetic helpers, and witness tooling. Update the `libspartan` dependency path in `Cargo.toml` to match the location of your local Spartan checkout when enabling this feature.
- `pytorch` enables optional `tch` bindings for interoperability experiments.
- `ndarray` unlocks alternative dense tensor implementations for research use.

## Exclusions in This Release
- Historical experiment outputs from the legacy workspace are omitted; regenerate fresh artifacts by running the experiment binary.
- Extensive Chinese-language ZKP documentation from earlier internal studies is not bundled here.

## Next Steps
- Integrate continuous integration to run `cargo fmt`, `cargo clippy`, and `cargo check --all-features` on pull requests.
- Rebuild the test matrix atop the streamlined layout once the repository is published and feature coverage stabilizes.
