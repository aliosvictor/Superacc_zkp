# Superacc ZKP

The Official reference implementation of the paper "Accurate and Zero-Knowledge Floating-Point Arithmetic for Graph Neural Network Inference." A high-performance graph convolution network (GCN) inference and zero-knowledge proof (ZKP) toolkit written in Rust. This project packages the full 1433-dimensional Cora workload together with the Spartan-based ZKP pipeline in a streamlined structure ready for public release.

## Highlights
- **GCN inference parity** with PyTorch/PyGCN using safe, zero-copy Rust data structures.
- **End-to-end ZKP workflow** for the complete Cora configuration leveraging Spartan primitives.
- **Deterministic math kernels** for dense and sparse operations with optional ZKP instrumentation.
- **Companion PyTorch pipeline** (`pygcn_helper/`) to retrain Cora GCN weights when needed.
- **Modular layout** that keeps data loaders, model logic, math utilities, and ZKP gadgets cleanly separated.
- **Figures** located under `Figures/` show the full system overview (`overview.pdf`) and the proof-aware GCN/ZKP flow (`GCN.pdf`).

## Figures
- `Figures/overview.pdf` captures the end-to-end stack spanning data ingestion, PyTorch retraining, Rust inference, and Spartan-based proof verification.
- `Figures/GCN.pdf` zooms into the GCN forward pass instrumented for ZKP, showing where dense and sparse math connect to Spartan gadgets.

## Directory Structure
```
Superacc_zkp/
|-- Cargo.toml            # crate metadata and feature flags
|-- README.md             # this guide
|-- LICENSE               # MIT license
|-- .gitignore
|-- data/                 # create this directory locally for the Cora dataset (see instructions below)
|-- model_weights/        # example JSON weight exports from PyTorch
|-- pygcn_helper/         # embedded PyTorch training toolkit (fork of pygcn)
|-- src/                  # library crate (data, layers, math, models, ZKP)
|-- experiments/          # gcn_full_feature.rs experiment binary
|-- docs/                 # high level architecture notes
`-- INVENTORY.md          # migration notes that track structural changes
```

## Prerequisites
- Rust toolchain 1.74+ (`rustup` recommended)
- System packages that expose `pkg-config` and OpenSSL development headers (required when enabling `zkp` or `pytorch`)
- The Cora dataset files (`cora.content` and `cora.cites`) stored locally under `data/cora` (not included in the repository)
- Optional: local checkout of `spartan` if the `zkp` feature is enabled; update the `libspartan` path in `Cargo.toml` to match your checkout.

## Preparing the Spartan Dependency
Spartan is required for any build that turns on the `zkp` feature. Clone the upstream repository alongside this project or adjust the path dependency to where you keep it:
```bash
git clone https://github.com/microsoft/SPARTAN.git ../spartan
```
If you prefer a different location, edit the `libspartan` entry in `Cargo.toml` to point to that directory.

Install the native toolchain pieces Spartan expects before building proofs:
- **Ubuntu / Debian**: `sudo apt-get install pkg-config libssl-dev`
- **macOS**: `brew install pkg-config openssl@3`
Make sure your shell can locate the OpenSSL libraries (for example, `export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@3/lib/pkgconfig"` on Apple Silicon).

## Getting the Cora Dataset
1. Create the target directory:
   ```bash
   mkdir -p data/cora
   ```
2. Use the helper script from the embedded PyTorch project to download the raw files:
   ```bash
   cd pygcn_helper
   PYTHONPATH=src python3 -m scripts.download_cora --output-dir ../data/cora
   cd ..
   ```
   This script fetches `cora.content` and `cora.cites` from the official Planetoid mirror on GitHub.
3. If the mirror is unavailable, fall back to the Planetoid archive maintained by LINQS:
   ```bash
   curl -L -o cora.tgz https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
   tar -xzf cora.tgz cora/cora.content cora/cora.cites
   mv cora/cora.* data/cora/
   rm -rf cora cora.tgz
   ```
4. Rerun `cargo run --bin gcn_inference` to verify the loader can find `data/cora`.

## Regenerating Model Weights
The directory `pygcn_helper/` contains a curated PyTorch training pipeline derived from the original [pygcn](https://github.com/tkipf/pygcn) project. Use it to regenerate float32 and float64 checkpoints and export JSON weights for this repository:

```bash
cd pygcn_helper
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=src python3 -m scripts.train_cora --precision f32 --export-json
PYTHONPATH=src python3 -m scripts.train_cora --precision f64 --export-json  # optional
```

The JSON files produced in `pygcn_helper/outputs/weights/` can be copied into `model_weights/` before running the Rust experiments. This repository already includes sample exports (`gcn_weights_f32_20251106.json` and `gcn_weights_f64_20251106.json`) generated with the settings above for quick smoke tests.

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
  -- --weights-single model_weights/gcn_weights_f32_20251106.json \
     --weights-double model_weights/gcn_weights_f64_20251106.json
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
Additional architecture notes are available in `docs/architecture.md` describing the module layout, data flow, and ZKP integration strategy. Historical Chinese-language documents from earlier internal drafts were deliberately omitted to keep the release concise.

## License
Released under the [MIT License](LICENSE).
