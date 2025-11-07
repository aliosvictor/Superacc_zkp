//! Crate entry point that wires together data loading, graph neural network layers,
//! math utilities, model definitions, typing helpers, and optional ZKP gadgets.
//! Consumers typically build a `models::gcn::GCN` instance, feed it data from
//! `data::loader`, and optionally enable `zkp` features for proof generation.

pub mod data;
pub mod layers;
pub mod math;
pub mod models;
pub mod types;
pub mod zkp;

pub use types::{
    CoraDataset, CsrMatrix, DefaultFloat, DefaultInt, DenseMatrix, FloatType, GCNConfig,
    GCNWeights, SparseMatrix,
};

pub use data::{loader, pytorch_loader};
pub use layers::graph_conv::GraphConvolution;
pub use models::gcn::GCN;

pub use zkp::{zkp_info, ZKPConfig, ZKPPrecision};

#[cfg(feature = "zkp")]
pub use zkp::utils;

#[cfg(not(feature = "zkp"))]
pub use zkp::zkp_not_enabled;
