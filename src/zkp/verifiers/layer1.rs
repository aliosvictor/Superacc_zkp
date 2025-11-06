use std::sync::Arc;

use crate::types::OperationPrecision;
use crate::zkp::prover::{DenseLayerWitness, Layer1Witness};
use crate::zkp::verifiers::common::{
    ConstraintAccumulator, ConstraintKind, GCNZKPConfig, VerificationLevel, ZKPError,
};

pub struct Layer1Verifier {
    config: Arc<GCNZKPConfig>,
}

impl Layer1Verifier {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn verify(&self, witness: &Layer1Witness) -> Result<Layer1VerificationReport, ZKPError> {
        let DenseLayerWitness {
            num_rows,
            num_cols,
            shared_dim,
            mulfp_witness,
            mulfp_batches,
        } = &witness.layer;

        let expected = num_rows
            .checked_mul(*shared_dim)
            .and_then(|v| v.checked_mul(*num_cols))
            .ok_or_else(|| ZKPError::ConfigError("witness dimension overflow".to_owned()))?;

        if mulfp_witness.len() != expected {
            return Err(ZKPError::DimensionMismatch(format!(
                "MULFP witness number mismatch: {} vs {}",
                mulfp_witness.len(),
                expected
            )));
        }

        let mut accumulator = ConstraintAccumulator::new();
        let precision: OperationPrecision = self.config.precision_mode.into();
        let should_check_sat = !matches!(self.config.verification_level, VerificationLevel::Fast);

        for (idx, entry) in mulfp_witness.iter().enumerate() {
            accumulator.add_mulfp_constraint();

            if should_check_sat {
                match entry.instance.is_sat(&entry.vars, &entry.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer1 witness item {} failed constraint verification",
                            idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::MULFPVerificationError(format!(
                            "Layer1 witness {} item verification failed: {:?}",
                            idx, err
                        )));
                    }
                }
            }

            accumulator.record_field_metrics(
                ConstraintKind::MulFpCore,
                precision,
                &entry.field_ops,
            );
        }

        for (batch_idx, batch) in mulfp_batches.iter().enumerate() {
            accumulator.add_mulfp_constraints(batch.num_constraints);

            if should_check_sat {
                match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError:: ConstraintUnsatisfied(format!(
                            "Layer1 witness batch Double-Shuffling group {} failed constraint verification",
                            batch_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError:: MULFPVerificationError(format!(
                            "Layer1 witness batch Double-Shuffling No. {} group verification failed: {:?}",
                            batch_idx, err
                        )));
                    }
                }
            }

            accumulator.record_field_metrics(
                ConstraintKind::MulFpBatch,
                precision,
                &batch.field_ops,
            );
        }

        Ok(Layer1VerificationReport {
            constraints: accumulator,
            num_mulfp_constraints: expected,
            num_nodes: *num_rows,
            hidden_dim: *num_cols,
            input_dim: *shared_dim,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Layer1VerificationReport {
    pub constraints: ConstraintAccumulator,
    pub num_mulfp_constraints: usize,
    pub num_nodes: usize,
    pub hidden_dim: usize,
    pub input_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkp::prover::{DenseLayerWitness, MulFPBatchWitness, MulFPWitness};
    use crate::zkp::utils::mulfp::{
        produce_r1cs_mulfp_detached, produce_r1cs_mulfp_with_params, MulFPInputData,
    };
    use crate::zkp::verifiers::common::{FloatBitExtractor, VerificationLevel, ZKPError};
    use libspartan::VarsAssignment;
    use std::sync::Arc;

    fn mulfp_input_from_f32_pair(lhs: f32, rhs: f32) -> MulFPInputData {
        MulFPInputData {
            b1: FloatBitExtractor::extract_sign_bit_f32(lhs) as u64,
            v1: FloatBitExtractor::extract_mantissa_f32(lhs),
            p1: FloatBitExtractor::extract_exponent_f32(lhs) as i64 - 127,
            b2: FloatBitExtractor::extract_sign_bit_f32(rhs) as u64,
            v2: FloatBitExtractor::extract_mantissa_f32(rhs),
            p2: FloatBitExtractor::extract_exponent_f32(rhs) as i64 - 127,
            witness_values: None,
        }
    }

    #[test]
    fn verifier_accepts_valid_layer1_witness() {
        let mut raw_config = GCNZKPConfig::single_precision();
        raw_config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(raw_config);
        let verifier = Layer1Verifier::new(config.clone());

        let lhs = 1.5f32;
        let rhs = -2.25f32;
        let input = mulfp_input_from_f32_pair(lhs, rhs);

        let detached = produce_r1cs_mulfp_detached(&config.mulfp_params, Some(&input))
            .expect("generate mulfp instance");

        let sat_result = detached.instance.is_sat(&detached.vars, &detached.inputs);
        assert_eq!(sat_result, Ok(true));

        let mulfp_entries = vec![detached.into_witness(input.clone())];

        let batch = MulFPBatchWitness::from_indices(&[0], &mulfp_entries, &config.mulfp_params)
            .expect("assemble batch witness");

        let witness = Layer1Witness {
            layer: DenseLayerWitness {
                num_rows: 1,
                num_cols: 1,
                shared_dim: 1,
                mulfp_batches: vec![batch],
                mulfp_witness: mulfp_entries,
            },
        };

        let report = verifier.verify(&witness).expect("layer1 verification");

        assert_eq!(report.num_mulfp_constraints, 1);
        assert_eq!(report.num_nodes, 1);
        assert_eq!(report.hidden_dim, 1);
        assert_eq!(report.input_dim, 1);
        let expected_constraint_count = witness.layer.mulfp_witness.len()
            + witness
                .layer
                .mulfp_batches
                .iter()
                .map(|batch| batch.num_constraints)
                .sum::<usize>();
        assert_eq!(
            report.constraints.stats.mulfp_count,
            expected_constraint_count
        );
    }

    #[test]
    fn verifier_rejects_tampered_layer1_witness() {
        let mut baseline_config = GCNZKPConfig::single_precision();
        baseline_config.verification_level = VerificationLevel::Optimized;
        let baseline_config = Arc::new(baseline_config);
        let baseline_verifier = Layer1Verifier::new(baseline_config.clone());

        let honest_lhs = 1.5f32;
        let honest_rhs = -2.25f32;
        let honest_input = mulfp_input_from_f32_pair(honest_lhs, honest_rhs);

        let honest_artifacts_detached =
            produce_r1cs_mulfp_detached(&baseline_config.mulfp_params, Some(&honest_input))
                .expect("generate mulfp instance");

        let honest_entries = vec![honest_artifacts_detached.into_witness(honest_input.clone())];
        let honest_batch =
            MulFPBatchWitness::from_indices(&[0], &honest_entries, &baseline_config.mulfp_params)
                .expect("assemble honest batch");

        let honest_witness = Layer1Witness {
            layer: DenseLayerWitness {
                num_rows: 1,
                num_cols: 1,
                shared_dim: 1,
                mulfp_batches: vec![honest_batch],
                mulfp_witness: honest_entries,
            },
        };

        baseline_verifier
            .verify(&honest_witness)
            .expect("honest witness must pass");

        let mut strict_config = GCNZKPConfig::single_precision();
        strict_config.verification_level = VerificationLevel::Full;
        let strict_config = Arc::new(strict_config);
        let strict_verifier = Layer1Verifier::new(strict_config.clone());

        let tamper_lhs = -0.75f32;
        let tamper_rhs = 3.5f32;
        let tamper_input = mulfp_input_from_f32_pair(tamper_lhs, tamper_rhs);

        let honest_again_detached =
            produce_r1cs_mulfp_detached(&baseline_config.mulfp_params, Some(&honest_input))
                .expect("generate second honest instance");

        let tampered_artifacts =
            produce_r1cs_mulfp_with_params(&baseline_config.mulfp_params, Some(&tamper_input))
                .expect("generate tampered witness");

        let mut tampered_vars_raw = tampered_artifacts.assignment.clone();
        if let Some(bit) = tampered_vars_raw.get_mut(12) {
            bit[0] ^= 1;
        }
        let tampered_vars_assignment =
            VarsAssignment::new(&tampered_vars_raw).expect("construct tampered VarsAssignment");

        let metrics = honest_again_detached.metrics;
        let tampered_entries = vec![MulFPWitness {
            input: honest_input.clone(),
            instance: honest_again_detached.instance,
            vars: tampered_vars_assignment,
            inputs: honest_again_detached.inputs,
            assignment: tampered_vars_raw,
            double_shuffle: honest_again_detached.double_shuffle,
            field_ops: metrics,
        }];
        let tampered_batch =
            MulFPBatchWitness::from_indices(&[0], &tampered_entries, &baseline_config.mulfp_params)
                .expect("assemble tampered batch");

        let tampered_witness = Layer1Witness {
            layer: DenseLayerWitness {
                num_rows: 1,
                num_cols: 1,
                shared_dim: 1,
                mulfp_batches: vec![tampered_batch],
                mulfp_witness: tampered_entries,
            },
        };

        match strict_verifier.verify(&tampered_witness) {
            Err(ZKPError::ConstraintUnsatisfied(_)) | Err(ZKPError::MULFPVerificationError(_)) => {}
            other => panic!("expected constraint failure, got {:?}", other),
        }
    }
}
