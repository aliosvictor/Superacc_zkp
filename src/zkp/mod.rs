//! Zero-knowledge proof helpers that wrap Spartan gadgets. The module is only
//! compiled when the `zkp` feature is enabled; otherwise we expose a small
//! placeholder that guides the user to rebuild with proofs enabled.

#[cfg(feature = "zkp")]
pub mod utils;

#[cfg(feature = "zkp")]
pub mod prover;

#[cfg(feature = "zkp")]
pub mod verifiers;

#[cfg(feature = "zkp")]
pub mod operation_tracker;

#[cfg(feature = "zkp")]
pub mod constraint_metrics;

#[cfg(feature = "zkp")]
pub use utils::*;

#[cfg(feature = "zkp")]
pub use prover::{
    DenseLayerWitness, Layer1Prover, Layer1Witness, Layer3Prover, Layer3Witness, MulFPWitness,
};

#[cfg(feature = "zkp")]
pub use verifiers::{
    common::*,
    coordinator::GCNZKPCoordinator,
    layer1::{Layer1VerificationReport, Layer1Verifier},
    layer3::{Layer3VerificationReport, Layer3Verifier},
};

#[cfg(not(feature = "zkp"))]
pub mod placeholder {
    //!
    //! ```bash
    //! cargo build --features zkp
    //! ```
    //!
    //! ```bash
    //! cargo build --features full
    //! ```

    use std::fmt;

    #[derive(Debug)]
    pub struct ZKPDisabledError;

    impl fmt::Display for ZKPDisabledError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "ZKP functionality is not enabled. Please compile with --features zkp to enable zero-knowledge proof functionality."
            )
        }
    }

    impl std::error::Error for ZKPDisabledError {}

    pub fn zkp_not_enabled() -> ZKPDisabledError {
        ZKPDisabledError
    }
}

#[cfg(not(feature = "zkp"))]
pub use placeholder::*;

/// Controls verbosity, RNG seeding, and precision used when building proofs.
#[derive(Debug, Clone)]
pub struct ZKPConfig {
    pub verbose: bool,
    pub seed: Option<u64>,
    pub precision: ZKPPrecision,
}

/// Enumerates the floating point widths supported by the proof circuits.
#[derive(Debug, Clone, Copy)]
pub enum ZKPPrecision {
    Half,
    Single,
    Double,
}

impl Default for ZKPConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            seed: None,
            precision: ZKPPrecision::Single,
        }
    }
}

impl Default for ZKPPrecision {
    fn default() -> Self {
        ZKPPrecision::Single
    }
}

/// Human-readable status string describing whether ZKP code is active in the
/// current build and what gadgets become available.
pub fn zkp_info() -> String {
    #[cfg(feature = "zkp")]
    {
        format!(
            "Rust GCN ZKP module v{}\n\
             - Spartan zkSNARK integration: enabled\n\
             - Supported gadgets: FL2SA, MULFP, SA2FL\n\
             - Floating point precision: f16, f32, f64\n\
             - Target scenarios: privacy-preserving graph learning, verifiable ML, federated learning",
            env!("CARGO_PKG_VERSION")
        )
    }

    #[cfg(not(feature = "zkp"))]
    {
        format!(
            "Rust GCN ZKP module v{}\n\
             - Spartan zkSNARK integration: disabled\n\
             - Enable with: cargo build --features zkp\n\
             - Full stack: cargo build --features full",
            env!("CARGO_PKG_VERSION")
        )
    }
}
