# Target Layout Proposal

```
Superacc_zkp/
|-- Cargo.toml               # Renamed crate metadata (`superacc_zkp`)
|-- Cargo.lock               # Generated after dependency audit
|-- README.md                # English project overview and usage
|-- LICENSE                  # MIT license migrated from original repo
|-- src/
|   |-- lib.rs               # Module exports for consumers
|   |-- main.rs              # Minimal CLI entry for inference demo
|   |-- types.rs             # Core data structures (cleaned comments)
|   |-- data/                # Dataset loaders and weight importers
|   |-- layers/              # Graph convolution layers
|   |-- math/                # Dense/Sparse math utilities
|   |-- models/              # GCN model definition
|   `-- zkp/                 # Prover/Verifier modules (English comments)
|-- experiments/
|   |-- gcn_full_feature.rs  # 1433-dim performance + proof experiment
|   `-- README.md            # How to run experiments
|-- model_weights/           # JSON exports required for 1433-dim run
|-- data/                    # Create locally to hold the Cora dataset
|-- docs/
|   `-- architecture.md      # Concise English summary of system layout
`-- .gitignore               # Exclude build artifacts, datasets, and caches
```

## Naming and Packaging Rules
- All module names use `snake_case`; exported types retain the original PascalCase identifiers for compatibility.
- Every source file and documentation page must be written in English; non-ASCII symbols removed unless unavoidable (e.g., mathematical notation).
- Binaries retain descriptive names but are limited to those required for the 1433-dimensional experiment pipeline.

## Feature Flags
- `gcn-only` remains default for inference.
- `zkp` feature bundles Spartan bindings and proof tooling.
- Optional `pytorch`/`ndarray` features kept for compatibility but documentation will describe prerequisites in English.

## Exclusions
- No historical results or plots are stored inside the repo. Instead the README links to the experiment instructions and regeneration steps.
- Tests are deferred until the reorganized structure stabilizes.
- ZKP documentation is summarized within `docs/architecture.md` rather than migrating the entire Chinese corpus.
