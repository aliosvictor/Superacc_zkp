use std::sync::Arc;

use crate::types::{DenseMatrix, FloatType, OperationPrecision};
use crate::zkp::operation_tracker::record_mul;
use crate::zkp::prover::{DenseLayerWitness, MulFPBatchWitness};
use crate::zkp::utils::mulfp::{produce_r1cs_mulfp_detached, MulFPInputData};
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};

pub struct Layer1Prover {
    config: Arc<GCNZKPConfig>,
}

impl Layer1Prover {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn generate_witness<T: FloatType>(
        &self,
        features: &DenseMatrix<T>,
        weights: &DenseMatrix<T>,
    ) -> Result<Layer1Witness, ZKPError> {
        let precision = self.config.precision_mode;
        let (num_nodes, nfeat) = features.shape;
        let (w_in, nhid) = weights.shape;

        if w_in != nfeat {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer1 weight dimension mismatch: {} vs {}",
                w_in, nfeat
            )));
        }

        let mut mulfp_witness = Vec::with_capacity(num_nodes * nfeat * nhid);

        for node_idx in 0..num_nodes {
            for feat_idx in 0..nfeat {
                let feature_value = features
                    .get(node_idx, feat_idx)
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                let feature_bits = FloatBitExtractor::decompose(feature_value, precision);

                for hidden_idx in 0..nhid {
                    let weight_value = weights
                        .get(feat_idx, hidden_idx)
                        .to_f64()
                        .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                    let weight_bits = FloatBitExtractor::decompose(weight_value, precision);

                    let mulfp_input = MulFPInputData {
                        b1: feature_bits.sign,
                        v1: feature_bits.mantissa,
                        p1: feature_bits.exponent,
                        b2: weight_bits.sign,
                        v2: weight_bits.mantissa,
                        p2: weight_bits.exponent,
                        witness_values: None,
                    };

                    let artifacts =
                        produce_r1cs_mulfp_detached(&self.config.mulfp_params, Some(&mulfp_input))
                            .map_err(ZKPError::MULFPVerificationError)?;

                    mulfp_witness.push(artifacts.into_witness(mulfp_input));
                }
            }
        }

        let batch_size = self.config.batch_size.max(1);
        let mut mulfp_batches = Vec::new();
        let mut current_indices = Vec::with_capacity(batch_size);
        for idx in 0..mulfp_witness.len() {
            current_indices.push(idx);
            let is_last = idx + 1 == mulfp_witness.len();
            if current_indices.len() == batch_size || is_last {
                let batch = MulFPBatchWitness::from_indices(
                    &current_indices,
                    &mulfp_witness,
                    &self.config.mulfp_params,
                )
                .map_err(|e| {
                    ZKPError::MULFPVerificationError(format!(
                        "Failed to assemble Layer1 MULFP batch witness: {}",
                        e
                    ))
                })?;
                mulfp_batches.push(batch);
                current_indices.clear();
            }
        }

        record_mul(OperationPrecision::from(precision), mulfp_witness.len());

        Ok(Layer1Witness {
            layer: DenseLayerWitness {
                num_rows: num_nodes,
                num_cols: nhid,
                shared_dim: nfeat,
                mulfp_witness,
                mulfp_batches,
            },
        })
    }
}

pub struct Layer1Witness {
    pub layer: DenseLayerWitness,
}
