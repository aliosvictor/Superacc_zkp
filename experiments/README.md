# GCN Full Feature Experiment

`gcn_full_feature.rs` executes the end-to-end Spartan-backed proof pipeline on the full 2708-node, 1433-feature Cora dataset.

## Usage
```bash
RUSTFLAGS="-C target_cpu=native" \
cargo run --release --features zkp --bin gcn_full_feature \
  -- --weights-single model_weights/gcn_weights_f32_20251106.json \
     --weights-double model_weights/gcn_weights_f64_20251106.json
```
> **Resource note:** Running the full pipeline for all precisions requires upwards of 24&ndash;32 GB of RAM. On machines with less memory, reduce the scope, for example:
> ```bash
> cargo run --release --features zkp --bin gcn_full_feature \
>   -- --precision single --verification-level fast \
>      --node-limit 128 --feature-limit 256
> ```

### Common Flags
- `--precision single|double|half` &mdash; restrict the run to a subset of numeric precisions.
- `--nodes <N>` &mdash; limit the experiment to the first `N` nodes (defaults to all nodes).
- `--node-offset <K>` &mdash; skip the first `K` nodes before slicing the dataset.
- `--output-prefix <PATH>` &mdash; change the artifact directory (defaults to `./artifacts/gcn_full_feature`).
- `--feature-limit <K>` &mdash; cap the number of feature columns (defaults to 1433).
- `--verification-level full|optimized|fast` &mdash; trade proof checks for faster runs on constrained machines.

### Outputs
The runner creates CSV/Markdown summaries plus witness statistics under the configured artifact directory. These files are intentionally excluded from version control; regenerate them locally whenever you need fresh measurements.
