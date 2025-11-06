use std::sync::Arc;

use crate::types::OperationPrecision;
use crate::zkp::prover::{DenseLayerWitness, Fl2saBatchWitness, Fl2saWitness, Layer3Witness};
use crate::zkp::utils::sa2fl::Sa2flBatchWitness;
use crate::zkp::verifiers::common::{
    ConstraintAccumulator, ConstraintKind, GCNZKPConfig, VerificationLevel, ZKPError,
};

pub struct Layer3Verifier {
    config: Arc<GCNZKPConfig>,
}

impl Layer3Verifier {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn verify(&self, witness: &Layer3Witness) -> Result<Layer3VerificationReport, ZKPError> {
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
        let should_check_sat = !matches!(self.config.verification_level, VerificationLevel::Fast);
        let precision: OperationPrecision = self.config.precision_mode.into();

        for (idx, entry) in mulfp_witness.iter().enumerate() {
            accumulator.add_mulfp_constraint();

            if should_check_sat {
                match entry.instance.is_sat(&entry.vars, &entry.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 witness item {} failed constraint verification",
                            idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::MULFPVerificationError(format!(
                            "Layer3 witness item {} failed verification: {:?}",
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
                            "Layer3 witness batch Double-Shuffling group {} failed constraint verification",
                            batch_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError:: MULFPVerificationError(format!(
                            "Layer3 witness batch Double-Shuffling Group {} verification failed: {:?}",
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

        let expected_outputs = num_rows
            .checked_mul(*num_cols)
            .ok_or_else(|| ZKPError::ConfigError("Layer3 output quantity overflow".to_owned()))?;
        if witness.fl2sa_outputs.len() != expected_outputs {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer3 FL2SA witness number mismatch: {} vs {}",
                witness.fl2sa_outputs.len(),
                expected_outputs
            )));
        }
        if witness.sa2fl_outputs.len() != witness.fl2sa_outputs.len() {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer3 SA2FL witness number mismatch: {} vs {}",
                witness.sa2fl_outputs.len(),
                witness.fl2sa_outputs.len()
            )));
        }

        self.verify_fl2sa_entries(&witness.fl2sa_outputs, should_check_sat, &mut accumulator)?;
        self.verify_fl2sa_batches(
            &witness.fl2sa_outputs,
            &witness.fl2sa_output_batches,
            should_check_sat,
            &mut accumulator,
        )?;
        self.verify_sa2fl_batches(
            &witness.sa2fl_output_batches,
            should_check_sat,
            &mut accumulator,
        )?;

        Ok(Layer3VerificationReport {
            constraints: accumulator,
            num_mulfp_constraints: expected,
            num_nodes: *num_rows,
            output_dim: *num_cols,
            hidden_dim: *shared_dim,
        })
    }

    fn verify_fl2sa_entries(
        &self,
        entries: &[Fl2saWitness],
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        for (idx, entry) in entries.iter().enumerate() {
            accumulator.add_fl2sa_constraint();
            if should_check_sat {
                match entry.instance.is_sat(&entry.vars, &entry.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA witness item {} failed constraint verification",
                            idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::FL2SAConversionError(format!(
                            "Layer3 output FL2SA witness verification failed for item {}: {:?}",
                            idx, err
                        )));
                    }
                }
            }
            accumulator.record_field_metrics(
                ConstraintKind::Fl2SaCore,
                precision,
                &entry.field_ops,
            );
        }
        Ok(())
    }

    fn verify_fl2sa_batches(
        &self,
        source: &[Fl2saWitness],
        batches: &[Fl2saBatchWitness],
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        for (batch_idx, batch) in batches.iter().enumerate() {
            if batch.entry_indices.is_empty() {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "Layer3 output FL2SA batch witness group {} is empty",
                    batch_idx
                )));
            }
            if batch.double_shuffle.core.batch_size != batch.entry_indices.len() {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "Layer3 output FL2SA batch witness group {} batch size is inconsistent",
                    batch_idx
                )));
            }
            if batch.double_shuffle.core.p_l_values.len() != batch.entry_indices.len()
                || batch.double_shuffle.core.r_bit_values.len() != batch.entry_indices.len()
                || batch.double_shuffle.ah.ah_values.len() != batch.entry_indices.len()
            {
                return Err(ZKPError:: ConstraintUnsatisfied(format!(
                    "Layer3 output FL2SA batch witness group {} Double-Shuffling record length is inconsistent",
                    batch_idx
                )));
            }

            accumulator.add_fl2sa_constraints(batch.num_constraints);

            if should_check_sat {
                match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError:: ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch witness group {} failed constraint verification",
                            batch_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch witness group {} verification failed: {:?}",
                            batch_idx, err
                        )));
                    }
                }

                for (position, &entry_idx) in batch.entry_indices.iter().enumerate() {
                    let witness = source.get(entry_idx).ok_or_else(|| {
                        ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch witness index {} out of bounds",
                            entry_idx
                        ))
                    })?;

                    match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                        Ok(true) => {}
                        Ok(false) => {
                            return Err(ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA witness index {} constraint not met",
                                entry_idx
                            )));
                        }
                        Err(err) => {
                            return Err(ZKPError::FL2SAConversionError(format!(
                                "Layer3 output FL2SA witness index {} Verification failed: {:?}",
                                entry_idx, err
                            )));
                        }
                    }

                    let recorded_p_bits = batch
                        .double_shuffle
                        .core
                        .p_l_values
                        .get(position)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA batch {} missing p_l bit record",
                                batch_idx
                            ))
                        })?;
                    let expected_p_bits = witness
                        .double_shuffle
                        .core
                        .p_l_values
                        .get(0)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA witness {} missing p_l bit record",
                                entry_idx
                            ))
                        })?;
                    if recorded_p_bits != expected_p_bits {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch {} p_l bit is inconsistent",
                            batch_idx
                        )));
                    }

                    let recorded_r_bits = batch
                        .double_shuffle
                        .core
                        .r_bit_values
                        .get(position)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA batch {} missing r_bit record",
                                batch_idx
                            ))
                        })?;
                    let expected_r_bits = witness
                        .double_shuffle
                        .core
                        .r_bit_values
                        .get(0)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA witness {} missing r_bit record",
                                entry_idx
                            ))
                        })?;
                    if recorded_r_bits != expected_r_bits {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch {} r_bit is inconsistent",
                            batch_idx
                        )));
                    }

                    let recorded_ah_bits = batch
                        .double_shuffle
                        .ah
                        .ah_values
                        .get(position)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA batch {} missing ah bit record",
                                batch_idx
                            ))
                        })?;
                    let expected_ah_bits =
                        witness.double_shuffle.ah.ah_values.get(0).ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer3 output FL2SA witness {} missing ah bit record",
                                entry_idx
                            ))
                        })?;
                    if recorded_ah_bits != expected_ah_bits {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Layer3 output FL2SA batch {} ah bit is inconsistent",
                            batch_idx
                        )));
                    }
                }
            }

            accumulator.record_field_metrics(
                ConstraintKind::Fl2SaBatch,
                precision,
                &batch.field_ops,
            );
        }
        Ok(())
    }

    fn verify_sa2fl_batches(
        &self,
        batches: &[Sa2flBatchWitness],
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        for (batch_idx, batch) in batches.iter().enumerate() {
            accumulator.add_sa2fl_constraints(batch.num_constraints);
            if should_check_sat {
                match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError:: ConstraintUnsatisfied(format!(
                            "Layer3 output SA2FL batch witness group {} failed constraint verification",
                            batch_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::SA2FLConversionError(format!(
                            "Layer3 output SA2FL batch witness group {} failed to verify: {:?}",
                            batch_idx, err
                        )));
                    }
                }
            }
            accumulator.record_field_metrics(
                ConstraintKind::Sa2FlBatch,
                precision,
                &batch.field_ops,
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Layer3VerificationReport {
    pub constraints: ConstraintAccumulator,
    pub num_mulfp_constraints: usize,
    pub num_nodes: usize,
    pub output_dim: usize,
    pub hidden_dim: usize,
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DenseMatrix;
    use crate::zkp::prover::Layer3Prover;
    use crate::zkp::verifiers::common::{VerificationLevel, ZKPError};
    use libspartan::VarsAssignment;
    use std::sync::Arc;

    #[test]
    fn verifier_accepts_valid_layer3_witness() {
        let mut raw_config = GCNZKPConfig::single_precision();
        raw_config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(raw_config);
        let verifier = Layer3Verifier::new(config.clone());
        let prover = Layer3Prover::new(config.clone());

        let activations = DenseMatrix::new(vec![0.5f32], (1, 1));
        let weights = DenseMatrix::new(vec![0.3f32], (1, 1));
        let witness = prover
            .generate_witness(&activations, &weights)
            .expect("generate layer3 witness");

        let report = verifier.verify(&witness).expect("layer3 verification");

        assert_eq!(report.num_mulfp_constraints, 1);
        assert_eq!(report.num_nodes, 1);
        assert_eq!(report.output_dim, 1);
        assert_eq!(report.hidden_dim, 1);
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
        assert!(report.constraints.stats.fl2sa_count > 0);
    }

    #[test]
    fn verifier_rejects_tampered_layer3_witness() {
        let mut baseline_config = GCNZKPConfig::single_precision();
        baseline_config.verification_level = VerificationLevel::Optimized;
        let baseline_config = Arc::new(baseline_config);
        let baseline_verifier = Layer3Verifier::new(baseline_config.clone());
        let baseline_prover = Layer3Prover::new(baseline_config.clone());

        let activations = DenseMatrix::new(vec![0.2f32], (1, 1));
        let weights = DenseMatrix::new(vec![1.5f32], (1, 1));

        let mut witness = baseline_prover
            .generate_witness(&activations, &weights)
            .expect("generate honest witness");

        baseline_verifier
            .verify(&witness)
            .expect("honest witness must pass");

        let mut strict_config = GCNZKPConfig::single_precision();
        strict_config.verification_level = VerificationLevel::Full;
        let strict_config = Arc::new(strict_config);
        let strict_verifier = Layer3Verifier::new(strict_config.clone());

        if let Some(entry) = witness.layer.mulfp_witness.first_mut() {
            if let Some(bit) = entry.assignment.get_mut(5) {
                bit[0] ^= 1;
            }
            entry.vars =
                VarsAssignment::new(&entry.assignment).expect("rebuild tampered VarsAssignment");
        }

        match strict_verifier.verify(&witness) {
            Err(ZKPError::ConstraintUnsatisfied(_)) | Err(ZKPError::MULFPVerificationError(_)) => {}
            other => panic!("expected constraint failure, got {:?}", other),
        }
    }
}
