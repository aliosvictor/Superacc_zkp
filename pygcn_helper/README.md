# PyGCN-G: Weight Generation Toolkit

PyGCN-G is a streamlined fork of [Thomas Kipf's PyGCN](https://github.com/tkipf/pygcn) tailored for the Superacc ZKP project. It trains Graph Convolutional Networks on the full 1433-feature Cora dataset and exports weights in the JSON layout consumed by the Rust verification pipeline.

## Features
- Reproducible training script with float32 or float64 precision.
- Deterministic export of checkpoints to `gc1/gc2` JSON tensors.
- Download helper for the public Cora citation dataset.
- English-only source tree aligned with the Superacc ZKP repository policies.

## Environment Setup

```bash
cd pygcn_helper
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional: pip install -e .
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

## Download the Dataset

The project does not ship with datasets. Use the helper script to fetch the raw Cora files:

```bash
PYTHONPATH=src python -m scripts.download_cora --output-dir data/cora
```

The downloader retrieves `cora.content` and `cora.cites` from the upstream [PyGCN repository](https://github.com/tkipf/pygcn). If that mirror becomes unavailable, download the Planetoid archive from LINQS and extract the same two files manually.

## Train and Export Weights

Run the training script to produce a checkpoint and, optionally, the JSON weights required by Superacc ZKP.

```bash
# Float32 training with JSON export
PYTHONPATH=src python -m scripts.train_cora \
  --data-dir data/cora \
  --precision f32 \
  --export-json

# Float64 training (longer runtime)
PYTHONPATH=src python -m scripts.train_cora \
  --data-dir data/cora \
  --precision f64 \
  --export-json
```

Outputs are saved under `outputs/`:
- `outputs/checkpoints/<timestamp>_gcn_<precision>_best.pt`
- `outputs/reports/<timestamp>_gcn_<precision>_metrics.json`
- `outputs/weights/<timestamp>_gcn_<precision>_weights.json` (when `--export-json` is set)

To convert an existing checkpoint to JSON later:

```bash
PYTHONPATH=src python -m scripts.export_weights \
  outputs/checkpoints/<timestamp>_gcn_f32_best.pt \
  --output outputs/weights/<timestamp>_gcn_f32_weights.json
```

## Integrating with Superacc ZKP

Copy the generated JSON weights into `../model_weights/` (relative to this folder) and point the Rust experiment CLI to the new files. See `Superacc_zkp/README.md` for the full workflow.

## Repository Hygiene

- Do not commit `data/`, `outputs/`, or virtual environments.
- All documentation and comments must remain in English and ASCII.
- PyGCN-G inherits the original PyGCN MIT license; no change was made to the upstream training algorithm apart from refactoring and export utilities.
