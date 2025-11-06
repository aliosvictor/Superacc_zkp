# Superacc ZKP

Official reference implementation of the paper "Accurate and Zero-Knowledge Floating-Point Arithmetic for Graph Neural Network Inference." A high-performance graph convolution network (GCN) inference and zero-knowledge proof (ZKP) toolkit written in Rust. This project reorganizes the original `rust_gcn` workspace into a streamlined structure suitable for public release, retaining the full 1433-dimensional Cora workload and the Spartan-based ZKP pipeline.

## Highlights
- **GCN inference parity** with PyTorch/PyGCN using safe, zero-copy Rust data structures.
- **End-to-end ZKP workflow** for the complete Cora configuration leveraging Spartan primitives.
- **Deterministic math kernels** for dense and sparse operations with optional ZKP instrumentation.
- **Modular layout** that keeps data loaders, model logic, math utilities, and ZKP gadgets cleanly separated.

## Directory Structure
```
Superacc_zkp/
|-- Cargo.toml            # crate metadata and feature flags
|-- README.md             # this guide
|-- LICENSE               # MIT license
|-- .gitignore
|-- data/               # create this directory locally for the Cora dataset (see instructions below)
|-- model_weights/        # example JSON weight exports from PyTorch
|-- src/                  # library crate (data, layers, math, models, ZKP)
|-- experiments/          # gcn_full_feature.rs experiment binary
|-- docs/                 # high level architecture notes
|-- INVENTORY.md          # migration notes from the original project
`-- STRUCTURE_PLAN.md     # target layout rationale
```

## Prerequisites
- Rust toolchain 1.74+ (`rustup` recommended)
- The Cora dataset files (`cora.content` and `cora.cites`) stored locally under `data/cora` (not included in the repository)
- Optional: local checkout of `spartan` if the `zkp` feature is enabled; update `Cargo.toml` to reference the location of your Spartan checkout.

## Getting the Cora Dataset
1. Create the target directory:
   ```bash
   mkdir -p data/cora
   ```
2. Download the dataset files from the original PyGCN repository:
   ```bash
   curl -L -o data/cora/cora.content https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content
   curl -L -o data/cora/cora.cites https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites
   ```
   If the GitHub raw mirrors are unavailable, use one of the fallbacks below:
   - Clone the reference implementation and copy its dataset:
     ```bash
     git clone https://github.com/tkipf/pygcn.git
     cp pygcn/data/cora/cora.* data/cora/
     ```
   - Download the Planetoid archive maintained by LINQS:
     ```bash
     curl -L -o cora.tgz https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
     tar -xzf cora.tgz cora/cora.content cora/cora.cites
     mv cora/cora.* data/cora/
     rm -rf cora cora.tgz
     ```
3. Rerun `cargo run --bin gcn_inference` to verify the loader can find `data/cora`.

## Running the Inference Demo
```bash
# From the repository root
cargo run --bin gcn_inference
```
The demo loads the Cora dataset, instantiates the GCN with random weights, and prints accuracy/loss metrics as a smoke test for the dense/sparse kernels.

## Running the 1433-Dimension ZKP Experiment
```bash
RUSTFLAGS="-C target_cpu=native" \
cargo run --release --features zkp --bin gcn_full_feature \
  -- --weights-single model_weights/2025-09-01_22-16-26_gcn_best_weights_f32.json \
     --weights-double model_weights/2025-09-01_22-16-27_gcn_best_weights_f64.json
```
Notes:
- The experiment consumes the full 2708-node, 1433-feature Cora graph by default.
- Results (witness statistics, constraint counts, CSV summaries) are written under `./artifacts/gcn_full_feature/`. These outputs are not committed; regenerate them locally as needed.
- Use `--precision single`, `--precision double`, or `--precision half` to restrict the run to specific numeric precisions.
- Expect peak memory usage above 24 GB when running the full proof pipeline with all precisions; provision a high-memory machine for production runs. For sanity checks on smaller hosts, cap the workload (for example: `--precision single --verification-level fast --node-limit 128 --feature-limit 256`).

## Feature Flags
- `gcn-only` (default): pure inference without PyTorch bindings or ZKP machinery.
- `pytorch`: enables the optional `tch` bindings for native PyTorch tensor interop.
- `ndarray`: exposes experiments backed by `ndarray` types.
- `zkp`: activates Spartan, witness generators, and verifiers; required for `gcn_full_feature`.
- `full`: convenience alias enabling all of the above.

## Development Workflow
- Format code with `cargo fmt`.
- Lint with `cargo clippy --all-features`.
- Run the inference binary (`cargo run`) and ZKP experiment as described above.
- `cargo check --all-features` verifies both inference and ZKP builds without generating witnesses.

## Documentation
Additional architecture notes are available in `docs/architecture.md` describing the module layout, data flow, and ZKP integration strategy. Historical Chinese-language documents from `rust_gcn/docs/zkp` were deliberately omitted to keep the release concise; consult the original repository if you need those archives.

## License
Released under the [MIT License](LICENSE).
