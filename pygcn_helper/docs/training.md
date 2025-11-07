# Training Workflow

This document records the choices behind the PyGCN-G helper project and explains how to regenerate weights for the Superacc ZKP repository.

## Objectives
- Reproduce the two-layer GCN from the original PyGCN project.
- Support float32 and float64 training for downstream zero-knowledge pipelines.
- Emit checkpoints and JSON exports without storing intermediate artifacts in Git.

## Key Commands

```bash
# 1. Move into the embedded helper (relative to the repo root)
cd pygcn_helper

# 2. Install dependencies inside an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# 3. Download the dataset from the upstream PyGCN mirror
PYTHONPATH=src python -m scripts.download_cora --output-dir data/cora

# 4. Train and export weights (float32 example)
PYTHONPATH=src python -m scripts.train_cora --precision f32 --export-json
```

## Output Layout
- `outputs/checkpoints/` - PyTorch state dictionaries.
- `outputs/weights/` - Flattened JSON exports (only created when `--export-json` is passed).
- `outputs/reports/` - Metrics snapshot per run.

All of these directories are ignored by version control via `.gitignore`.

## Integration Notes
- Copy JSON exports from `pygcn_helper/outputs/weights/` into `../model_weights/` so the Rust experiments can pick them up without extra configuration.
- The JSON format contains four keys: `gc1.weight`, `gc1.bias`, `gc2.weight`, `gc2.bias`.
- Float64 exports can be large; ensure downstream tooling handles double precision before enabling the `f64` flag.
- Do not leave `data/`, `outputs/`, `.venv/`, or `__pycache__/` folders in source control—`Superacc_zkp/.gitignore` already excludes them, so keep the working tree clean before committing.
