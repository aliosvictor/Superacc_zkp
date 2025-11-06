pub use crate::zkp::utils::fl2sa::{Fl2saBatchWitness, Fl2saWitness};
pub use crate::zkp::utils::mulfp::{MulFPBatchWitness, MulFPInputData, MulFPWitness};

pub mod fl2sa_utils;
pub mod layer1;
pub mod layer2;
pub mod layer3;
pub mod layer4;
pub mod sparse_product;

pub use crate::zkp::utils::linear::{LinearCombinationWitness, LinearRelationWitness};
pub use layer1::{Layer1Prover, Layer1Witness};
pub use layer2::{Layer2Prover, Layer2Witness, ReluWitnessEntry};
pub use layer3::{Layer3Prover, Layer3Witness};
pub use layer4::{Layer4Prover, Layer4Witness, SoftmaxNodeWitness, SoftmaxWitnessEntry};
pub use sparse_product::{SparseMatMulWitness, SparseProductProver, SparseProductResult};

pub struct DenseLayerWitness {
    pub num_rows: usize,
    pub num_cols: usize,
    pub shared_dim: usize,
    pub mulfp_witness: Vec<MulFPWitness>,
    pub mulfp_batches: Vec<MulFPBatchWitness>,
}
