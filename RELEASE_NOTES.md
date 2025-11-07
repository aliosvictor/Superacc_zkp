# Release Notes - Superacc ZKP

## Version 0.2.2 (2025-11-08)

### Summary
- Adds PDF renderings (`overview.pdf`, `GCN.pdf`) plus PNG previews embedded directly in `README.md` so GitHub visitors can see the architecture diagrams without cloning the repo.
- README intro and experiment sections now avoid referencing specific feature counts so the documentation feels general-purpose.

### Dependencies and Tooling
- Same as v0.2.1 (Rust 1.74+, CUDA PyTorch helper when needed, Spartan checkout for `zkp`).

### Verification Checklist
- `cargo fmt`
- `cargo check`
- `cargo check --features zkp`

## Version 0.2.1 (2025-11-08)

### Summary
- Adds comprehensive Rust doc comments across the GCN model, graph convolution layer, data loader, and ZKP helpers so the codebase is self-documenting without referencing the legacy `rust_gcn` tree.
- Refreshes the crate metadata (`Cargo.toml`) to version `0.2.1` for the documentation-focused release.
- Keeps the repository artifact-free while reinforcing the workflow described in the README and architecture notes.

### Dependencies and Tooling
- Same toolchain requirements as v0.2.0: Rust 1.74+, CUDA-enabled PyTorch 2.5.1 (cu121 wheel) when running the helper, and a local Spartan checkout for `zkp` builds.

### Verification Checklist
- `cargo fmt`
- `cargo check`
- `cargo check --features zkp`

## Version 0.2.0 (2025-11-07)

### Summary
- Vendors the PyGCN-G helper directly inside `pygcn_helper/`, eliminating the external dependency on a sibling `pygcn` checkout.
- Refreshes `README.md`, `INVENTORY.md`, `experiments/README.md`, and helper docs with end-to-end instructions for downloading data, training, and exporting weights.
- Adds new reference weights (`gcn_weights_f32_20251106.json`, `gcn_weights_f64_20251106.json`) generated via the embedded pipeline and updates defaults in `gcn_full_feature.rs`.
- Expands `.gitignore` to cover Python virtual environments, caches, and outputs so the repository remains artifact-free.

### Dependencies and Tooling
- Rust toolchain 1.74 or newer.
- CUDA-enabled PyTorch 2.5.1 (cu121 wheel) plus `numpy`/`scipy` when running the embedded helper.
- Spartan checkout still required when enabling the `zkp` feature (path configured in `Cargo.toml`).
- Dataset download workflow now lives entirely inside `pygcn_helper/`.

### Verification Checklist
- `cargo fmt`
- `cargo check --features zkp`
- `PYTHONPATH=src python -m scripts.download_cora --output-dir data/cora`
- `PYTHONPATH=src python -m scripts.train_cora --precision f32 --export-json`

### Known Follow-Ups
- Evaluate packaging the PyGCN-G helper as a separate PyPI module and wiring it through `pip install`.
- Consider publishing pre-built weight archives and CI workflows once public release is finalized.

## Version 0.1.0 (2025-11-06)

### Summary
- Publicly packages the streamlined Superacc ZKP workspace for its first tagged release.
- Retains only the full 1433-feature Cora workload and associated zero-knowledge proof pipeline.
- Provides English-only documentation and reproducible experiment instructions without bundling datasets or generated artifacts.
- Includes reference PyTorch weight exports (`gcn_weights_f32_20251106.json`, `gcn_weights_f64_20251106.json`) generated via the companion PyGCN-G helper for quick validation.

### Dependencies and Tooling
- Rust toolchain 1.74 or newer (install via rustup).
- Optional: local checkout of the Spartan proving system when building with the `zkp` feature. Update `Cargo.toml` to point to the checkout location, or replace the path dependency with a published crate if available.
- Dataset: users must download the Cora `cora.content` and `cora.cites` files following the steps in `README.md`.

### Verification Checklist
- Run `cargo fmt` and `cargo clippy --all-features` before submitting changes (note that enabling the `pytorch` feature pulls in the `tch` crate and requires libtorch, pkg-config, and OpenSSL on the host).
- Validate builds with `cargo check` for the default pipeline and `cargo check --features zkp` for the proving flow.
- Execute the `gcn_full_feature` experiment binary after touching inference or ZKP code paths to ensure witness generation still succeeds.

### Known Follow-Ups
- Publish the repository to GitHub and create an official release tag.
- Evaluate replacing the local Spartan path dependency with a crates.io release once one is available.
