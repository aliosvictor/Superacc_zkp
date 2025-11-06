# Release Notes - Superacc ZKP

## Version 0.1.0 (2025-11-06)

### Summary
- Publicly packages the reorganized Superacc ZKP workspace derived from the original rust_gcn codebase.
- Retains only the full 1433-feature Cora workload and associated zero-knowledge proof pipeline.
- Provides English-only documentation and reproducible experiment instructions without bundling datasets or generated artifacts.

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
