use std::sync::Arc;

use crate::math::dense_ops;
use crate::types::{DenseMatrix, FloatType, SparseMatrix};
use crate::zkp::prover::{Layer1Prover, Layer2Prover, Layer3Prover, Layer4Prover};
use crate::zkp::utils::sparse::{hash_dense_matrix_rows, hash_sparse_matrix};
use crate::zkp::verifiers::common::{ConstraintAccumulator, GCNZKPConfig, ZKPError};

use super::{
    Layer1VerificationReport, Layer1Verifier, Layer2VerificationReport, Layer2Verifier,
    Layer3VerificationReport, Layer3Verifier, Layer4VerificationReport, Layer4Verifier,
};

pub struct GCNZKPCoordinator {
    config: Arc<GCNZKPConfig>,
}

impl GCNZKPCoordinator {
    pub fn new(config: GCNZKPConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub fn verify_layer1<T: FloatType>(
        &self,
        features: &DenseMatrix<T>,
        weights: &DenseMatrix<T>,
    ) -> Result<Layer1VerificationReport, ZKPError> {
        let prover = Layer1Prover::new(self.config.clone());
        let witness = prover.generate_witness(features, weights)?;
        let verifier = Layer1Verifier::new(self.config.clone());
        verifier.verify(&witness)
    }

    pub fn verify_layer4<T: FloatType>(
        &self,
        support: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        bias: Option<&[T]>,
        logits: &DenseMatrix<T>,
        softmax_outputs: &DenseMatrix<T>,
    ) -> Result<Layer4VerificationReport, ZKPError> {
        let prover = Layer4Prover::new(self.config.clone());
        let witness = prover.generate_witness(support, adj, bias)?;

        let expected_adj_hash = hash_sparse_matrix(adj).map_err(|e| {
            ZKPError::ConfigError(format!("Layer4 adjacency matrix hash failed: {}", e))
        })?;
        if expected_adj_hash != witness.sparse_product.adj_hash {
            return Err(ZKPError::ConstraintUnsatisfied(
                "Layer4 adjacency matrix hash mismatch".to_string(),
            ));
        }
        let expected_commitments = hash_dense_matrix_rows(support)
            .map_err(|e| ZKPError::ConfigError(format!("Layer4 support hash failed: {}", e)))?;
        if expected_commitments != witness.sparse_product.support_commitments {
            return Err(ZKPError::ConstraintUnsatisfied(
                "Layer4 support promise mismatch".to_string(),
            ));
        }

        if softmax_outputs.shape != (witness.num_nodes, witness.num_classes) {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer4 output dimension mismatch: witness {}x{} vs activations {}x{}",
                witness.num_nodes,
                witness.num_classes,
                softmax_outputs.shape.0,
                softmax_outputs.shape.1
            )));
        }
        if logits.shape != (witness.num_nodes, witness.num_classes) {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer4 logits dimension mismatch: witness {}x{} vs logits {}x{}",
                witness.num_nodes, witness.num_classes, logits.shape.0, logits.shape.1
            )));
        }

        let comparison_slack = self.config.tolerance.max(0.05);
        for (node_idx, node) in witness.nodes.iter().enumerate() {
            for (class_idx, entry) in node.entries.iter().enumerate() {
                let expected_softmax = softmax_outputs
                    .get(node_idx, class_idx)
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                if (entry.softmax_value - expected_softmax).abs() > comparison_slack {
                    return Err(ZKPError::NumericalError(
                        entry.softmax_value - expected_softmax,
                    ));
                }

                let expected_logit = logits
                    .get(node_idx, class_idx)
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                if (entry.logit_value - expected_logit).abs() > comparison_slack {
                    return Err(ZKPError::NumericalError(entry.logit_value - expected_logit));
                }
            }
        }

        let verifier = Layer4Verifier::new(self.config.clone());
        verifier.verify(&witness)
    }

    pub fn verify_layer3<T: FloatType>(
        &self,
        activations: &DenseMatrix<T>,
        weights: &DenseMatrix<T>,
    ) -> Result<Layer3VerificationReport, ZKPError> {
        let prover = Layer3Prover::new(self.config.clone());
        let witness = prover.generate_witness(activations, weights)?;
        let verifier = Layer3Verifier::new(self.config.clone());
        verifier.verify(&witness)
    }

    pub fn verify_layer2<T: FloatType>(
        &self,
        support: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        bias: Option<&[T]>,
        relu_outputs: &DenseMatrix<T>,
    ) -> Result<Layer2VerificationReport, ZKPError> {
        let prover = Layer2Prover::new(self.config.clone());
        let witness = prover.generate_witness(support, adj, bias)?;

        let expected_adj_hash = hash_sparse_matrix(adj).map_err(|e| {
            ZKPError::ConfigError(format!("Layer2 adjacency matrix hash failed: {}", e))
        })?;
        if expected_adj_hash != witness.sparse_product.adj_hash {
            return Err(ZKPError::ConstraintUnsatisfied(
                "Layer2 adjacency matrix hash mismatch".to_string(),
            ));
        }
        let expected_commitments = hash_dense_matrix_rows(support)
            .map_err(|e| ZKPError::ConfigError(format!("Layer2 support hash failed: {}", e)))?;
        if expected_commitments != witness.sparse_product.support_commitments {
            return Err(ZKPError::ConstraintUnsatisfied(
                "Layer2 support promise mismatch".to_string(),
            ));
        }

        if relu_outputs.shape != (witness.num_nodes, witness.hidden_dim) {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer2 output dimension mismatch: witness {}x{} vs activations {}x{}",
                witness.num_nodes, witness.hidden_dim, relu_outputs.shape.0, relu_outputs.shape.1
            )));
        }

        let comparison_slack = self.config.tolerance.max(0.05);
        for (idx, entry) in witness.entries.iter().enumerate() {
            let row = idx / witness.hidden_dim;
            let col = idx % witness.hidden_dim;
            let expected = relu_outputs
                .get(row, col)
                .to_f64()
                .ok_or_else(|| ZKPError::NumericalError(0.0))?;
            if (entry.relu_value - expected).abs() > comparison_slack {
                return Err(ZKPError::NumericalError(entry.relu_value - expected));
            }
        }

        let verifier = Layer2Verifier::new(self.config.clone());
        verifier.verify(&witness)
    }

    pub fn verify_dense_layers<T: FloatType>(
        &self,
        features: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        layer1_weights: &DenseMatrix<T>,
        layer1_bias: Option<&[T]>,
        activations: &DenseMatrix<T>,
        layer3_weights: &DenseMatrix<T>,
    ) -> Result<DenseLayersVerificationReport, ZKPError> {
        let support_layer1 = dense_ops::dense_mm(features, layer1_weights);
        let layer1 = self.verify_layer1(features, layer1_weights)?;
        let layer2 = self.verify_layer2(&support_layer1, adj, layer1_bias, activations)?;
        let layer3 = self.verify_layer3(activations, layer3_weights)?;

        let mut total_constraints = ConstraintAccumulator::new();
        total_constraints.merge(layer1.constraints.clone());
        total_constraints.merge(layer2.constraints.clone());
        total_constraints.merge(layer3.constraints.clone());

        Ok(DenseLayersVerificationReport {
            layer1,
            layer2,
            layer3,
            total_constraints,
        })
    }

    pub fn verify_all_layers<T: FloatType>(
        &self,
        features: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        layer1_weights: &DenseMatrix<T>,
        layer1_bias: Option<&[T]>,
        activations: &DenseMatrix<T>,
        layer3_weights: &DenseMatrix<T>,
        layer3_bias: Option<&[T]>,
        logits: &DenseMatrix<T>,
        softmax_outputs: &DenseMatrix<T>,
    ) -> Result<FullLayersVerificationReport, ZKPError> {
        let support_layer1 = dense_ops::dense_mm(features, layer1_weights);
        let support_layer2 = dense_ops::dense_mm(activations, layer3_weights);
        let layer1 = self.verify_layer1(features, layer1_weights)?;
        let layer2 = self.verify_layer2(&support_layer1, adj, layer1_bias, activations)?;
        let layer3 = self.verify_layer3(activations, layer3_weights)?;
        let layer4 =
            self.verify_layer4(&support_layer2, adj, layer3_bias, logits, softmax_outputs)?;

        let mut total_constraints = ConstraintAccumulator::new();
        total_constraints.merge(layer1.constraints.clone());
        total_constraints.merge(layer2.constraints.clone());
        total_constraints.merge(layer3.constraints.clone());
        total_constraints.merge(layer4.constraints.clone());

        Ok(FullLayersVerificationReport {
            layer1,
            layer2,
            layer3,
            layer4,
            total_constraints,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayersVerificationReport {
    pub layer1: Layer1VerificationReport,
    pub layer2: Layer2VerificationReport,
    pub layer3: Layer3VerificationReport,
    pub total_constraints: ConstraintAccumulator,
}

impl DenseLayersVerificationReport {
    pub fn mulfp_constraints(&self) -> usize {
        self.total_constraints.stats.mulfp_count
    }
}

#[derive(Debug, Clone)]
pub struct FullLayersVerificationReport {
    pub layer1: Layer1VerificationReport,
    pub layer2: Layer2VerificationReport,
    pub layer3: Layer3VerificationReport,
    pub layer4: Layer4VerificationReport,
    pub total_constraints: ConstraintAccumulator,
}

impl FullLayersVerificationReport {
    pub fn mulfp_constraints(&self) -> usize {
        self.total_constraints.stats.mulfp_count
    }
}
