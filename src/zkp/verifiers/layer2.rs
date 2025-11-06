use std::sync::Arc;

use crate::types::OperationPrecision;
use crate::zkp::prover::{
    Fl2saWitness, Layer2Witness, LinearCombinationWitness, MulFPBatchWitness, MulFPWitness,
    SparseMatMulWitness,
};
use crate::zkp::utils::fl2sa::Fl2saBatchWitness;
use crate::zkp::verifiers::common::{
    ConstraintAccumulator, ConstraintKind, FloatBitExtractor, GCNZKPConfig, VerificationLevel,
    ZKPError,
};

pub struct Layer2Verifier {
    config: Arc<GCNZKPConfig>,
}

impl Layer2Verifier {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn verify(&self, witness: &Layer2Witness) -> Result<Layer2VerificationReport, ZKPError> {
        let mut accumulator = ConstraintAccumulator::new();
        let should_check_sat = !matches!(self.config.verification_level, VerificationLevel::Fast);

        let tolerance = self.config.tolerance;

        self.verify_sparse_product(
            &witness.sparse_product,
            should_check_sat,
            &mut accumulator,
            "Layer2",
        )?;

        for (batch_idx, batch) in witness.fl2sa_input_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_inputs,
                should_check_sat,
                &mut accumulator,
                &format!("input-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_poly_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_polys,
                should_check_sat,
                &mut accumulator,
                &format!("poly-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_relu_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_relus,
                should_check_sat,
                &mut accumulator,
                &format!("relu-batch-{}", batch_idx),
            )?;
        }

        for (idx, entry) in witness.entries.iter().enumerate() {
            let x2 = self.resolve_mulfp(witness, entry.x2_idx, idx, "x^2")?;
            self.verify_mulfp_witness(x2, should_check_sat, &mut accumulator, idx, "x^2")?;

            let x4 = self.resolve_mulfp(witness, entry.x4_idx, idx, "x^4")?;
            self.verify_mulfp_witness(x4, should_check_sat, &mut accumulator, idx, "x^4")?;

            let x6 = self.resolve_mulfp(witness, entry.x6_idx, idx, "x^6")?;
            self.verify_mulfp_witness(x6, should_check_sat, &mut accumulator, idx, "x^6")?;

            let term_a2 = self.resolve_mulfp(witness, entry.term_a2_idx, idx, "a2*x^2")?;
            self.verify_mulfp_witness(term_a2, should_check_sat, &mut accumulator, idx, "a2*x^2")?;

            let term_a4 = self.resolve_mulfp(witness, entry.term_a4_idx, idx, "a4*x^4")?;
            self.verify_mulfp_witness(term_a4, should_check_sat, &mut accumulator, idx, "a4*x^4")?;

            let term_a6 = self.resolve_mulfp(witness, entry.term_a6_idx, idx, "a6*x^6")?;
            self.verify_mulfp_witness(term_a6, should_check_sat, &mut accumulator, idx, "a6*x^6")?;

            self.verify_linear_witness(
                &entry.poly_relation_witness,
                should_check_sat,
                &mut accumulator,
                idx,
                "poly-sum",
            )?;
            self.verify_linear_witness(
                &entry.relu_relation_witness,
                should_check_sat,
                &mut accumulator,
                idx,
                "relu-output",
            )?;

            let relu_fl2sa = self.resolve_fl2sa(
                &witness.fl2sa_relus,
                entry.fl2sa_relu_idx,
                idx,
                "relu-output",
            )?;
            let expected_relu = FloatBitExtractor::quantize_to_precision(
                self.decode_fl2sa_value(relu_fl2sa),
                self.config.precision_mode,
            );
            let relu_delta = (entry.relu_value - expected_relu).abs();
            if relu_delta > tolerance {
                return Err(ZKPError::NumericalError(relu_delta));
            }

            let negative_threshold = tolerance.max(0.05);
            if (entry.relu_value as f64) < -negative_threshold {
                return Err(ZKPError::NumericalError(entry.relu_value as f64));
            }
        }

        for (batch_idx, batch) in witness.mulfp_batches.iter().enumerate() {
            for &entry_idx in &batch.entry_indices {
                if entry_idx >= witness.mulfp_witness.len() {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 MULFP batch witness {} group index {} out of bounds",
                        batch_idx, entry_idx
                    )));
                }
            }
            self.verify_mulfp_batch(batch, should_check_sat, &mut accumulator, batch_idx)?;
        }

        Ok(Layer2VerificationReport {
            constraints: accumulator,
            num_entries: witness.entries.len(),
            tolerance,
        })
    }

    fn verify_sparse_product(
        &self,
        sparse: &SparseMatMulWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        label: &str,
    ) -> Result<(), ZKPError> {
        if sparse.csr_row_ptr.len() != sparse.num_rows + 1 {
            return Err(ZKPError::ConstraintUnsatisfied(format!(
                "{} sparse witness row_ptr length mismatch",
                label
            )));
        }
        for window in sparse.csr_row_ptr.windows(2) {
            if window[1] < window[0] {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "{} CSR row_ptr non-monotonic: {:?}",
                    label, window
                )));
            }
        }
        if let Some(&last) = sparse.csr_row_ptr.last() {
            if last != sparse.col_indices.len() {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "{} End of CSR row_ptr={} does not match column index number {}",
                    label,
                    last,
                    sparse.col_indices.len()
                )));
            }
        }
        if sparse.col_indices.len() != sparse.values.len() {
            return Err(ZKPError::ConstraintUnsatisfied(format!(
                "{} Sparse witness column index is inconsistent with the number of values",
                label
            )));
        }
        for &col in &sparse.col_indices {
            if col >= sparse.num_cols {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "{} Sparse witness column index {} Out of dimensions {}",
                    label, col, sparse.num_cols
                )));
            }
        }

        for (idx, mulfp) in sparse.mulfp_witness.iter().enumerate() {
            self.verify_mulfp_witness(mulfp, should_check_sat, accumulator, idx, "sparse-product")?;
        }
        for (batch_idx, batch) in sparse.mulfp_batches.iter().enumerate() {
            self.verify_mulfp_batch(batch, should_check_sat, accumulator, batch_idx)?;
        }
        for (idx, relation) in sparse.row_relations.iter().enumerate() {
            self.verify_linear_witness(relation, should_check_sat, accumulator, idx, "sparse-row")?;
        }

        if let Some(row_diff) = &sparse.row_diff_witness {
            self.verify_row_diffs(row_diff, sparse, should_check_sat, accumulator)?;
        }

        Ok(())
    }

    fn verify_row_diffs(
        &self,
        witness: &crate::zkp::prover::sparse_product::RowDiffWitness,
        sparse: &SparseMatMulWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        if witness.diffs.len() != sparse.num_rows {
            return Err(ZKPError:: ConstraintUnsatisfied(
                "The number of sparse witness row differences is inconsistent with the number of rows".to_string(),
            ));
        }

        for (idx, window) in sparse.csr_row_ptr.windows(2).enumerate() {
            let diff = window[1].saturating_sub(window[0]);
            if witness.diffs[idx] != diff {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "Sparse witness row {} differential mismatch: witness={} actual={}",
                    idx, witness.diffs[idx], diff
                )));
            }
        }

        accumulator.add_linear_constraints(witness.num_constraints);

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(
                        "Sparse row differential witness failed SAT".to_string(),
                    ));
                }
                Err(err) => {
                    return Err(ZKPError::ConfigError(format!(
                        "Verification of sparse row differential witness failed: {:?}",
                        err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::Auxiliary, precision, &witness.field_ops);

        Ok(())
    }

    fn verify_mulfp_witness(
        &self,
        witness: &MulFPWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        idx: usize,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_mulfp_constraint();

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 witness item {} {} failed constraint verification",
                        idx, label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::MULFPVerificationError(format!(
                        "Layer2 witness item {} {} failed verification: {:?}",
                        idx, label, err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::MulFpCore, precision, &witness.field_ops);

        Ok(())
    }

    fn resolve_mulfp<'a>(
        &self,
        layer_witness: &'a Layer2Witness,
        idx: usize,
        entry_idx: usize,
        label: &str,
    ) -> Result<&'a MulFPWitness, ZKPError> {
        layer_witness.mulfp_witness.get(idx).ok_or_else(|| {
            ZKPError::ConstraintUnsatisfied(format!(
                "Layer2 witness item {} {} index {} out of bounds",
                entry_idx, label, idx
            ))
        })
    }

    fn resolve_fl2sa<'a>(
        &self,
        source: &'a [Fl2saWitness],
        idx: usize,
        entry_idx: usize,
        label: &str,
    ) -> Result<&'a Fl2saWitness, ZKPError> {
        source.get(idx).ok_or_else(|| {
            ZKPError::ConstraintUnsatisfied(format!(
                "Layer2 witness item {} {} FL2SA index {} out of bounds",
                entry_idx, label, idx
            ))
        })
    }

    fn decode_fl2sa_value(&self, witness: &Fl2saWitness) -> f64 {
        let precision = self.config.fl2sa_effective_precision();
        let mantissa =
            FloatBitExtractor::mantissa_from_blocks(&witness.v, self.config.fl2sa_w, precision);
        FloatBitExtractor::compose_from_components(witness.b, witness.p, mantissa, precision)
    }

    fn verify_mulfp_batch(
        &self,
        batch: &MulFPBatchWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        batch_idx: usize,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_mulfp_constraints(batch.num_constraints);

        if should_check_sat {
            match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 MULFP batch witness group {} failed constraint verification",
                        batch_idx
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::MULFPVerificationError(format!(
                        "Layer2 MULFP batch witness group {} failed to verify: {:?}",
                        batch_idx, err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::MulFpBatch, precision, &batch.field_ops);

        Ok(())
    }

    fn verify_fl2sa_batch(
        &self,
        batch: &Fl2saBatchWitness,
        source: &[Fl2saWitness],
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_fl2sa_constraints(batch.num_constraints);

        if batch.double_shuffle.core.batch_size != batch.entry_indices.len() {
            return Err(ZKPError::ConstraintUnsatisfied(format!(
                "Layer2 {} batch sizes are inconsistent: batch_size={}, indices={}",
                label,
                batch.double_shuffle.core.batch_size,
                batch.entry_indices.len()
            )));
        }

        if batch.double_shuffle.core.p_l_values.len() != batch.entry_indices.len()
            || batch.double_shuffle.core.r_bit_values.len() != batch.entry_indices.len()
            || batch.double_shuffle.ah.ah_values.len() != batch.entry_indices.len()
        {
            return Err(ZKPError::ConstraintUnsatisfied(format!(
                "Layer2 {} Double-Shuffling record and index length are inconsistent",
                label
            )));
        }

        if should_check_sat {
            match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 {} batch Double-Shuffling constraints are not met",
                        label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 {} Batch Double-Shuffling SAT failed: {:?}",
                        label, err
                    )));
                }
            }

            for (position, &entry_idx) in batch.entry_indices.iter().enumerate() {
                let witness = source.get(entry_idx).ok_or_else(|| {
                    ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 {} batch verification index {} out of range",
                        label, entry_idx
                    ))
                })?;

                match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError:: ConstraintUnsatisfied(format!(
                            "Layer2 {} Index {} in the batch witness does not satisfy the constraint",
                            label, entry_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::FL2SAConversionError(format!(
                            "Layer2 {} batch witness index {} verification failed: {:?}",
                            label, entry_idx, err
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
                            "Layer2 {} batch {} missing p_l bit record",
                            label, position
                        ))
                    })?;
                let expected_p_bits =
                    witness
                        .double_shuffle
                        .core
                        .p_l_values
                        .get(0)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer2 {} witness {} missing p_l bit record",
                                label, entry_idx
                            ))
                        })?;
                if recorded_p_bits != expected_p_bits {
                    return Err(ZKPError:: ConstraintUnsatisfied(format!(
                        "Layer2 {} batch {} p_l bit record is inconsistent with the original witness",
                        label, position
                    )));
                }

                let recorded_r_bits = batch
                    .double_shuffle
                    .core
                    .r_bit_values
                    .get(position)
                    .ok_or_else(|| {
                        ZKPError::ConstraintUnsatisfied(format!(
                            "Layer2 {} batch {} missing r_bit record",
                            label, position
                        ))
                    })?;
                let expected_r_bits =
                    witness
                        .double_shuffle
                        .core
                        .r_bit_values
                        .get(0)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer2 {} witness {} missing r_bit record",
                                label, entry_idx
                            ))
                        })?;
                if recorded_r_bits != expected_r_bits {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 {} batch {} r_bit record is inconsistent with the original witness",
                        label, position
                    )));
                }

                let recorded_ah_bits =
                    batch
                        .double_shuffle
                        .ah
                        .ah_values
                        .get(position)
                        .ok_or_else(|| {
                            ZKPError::ConstraintUnsatisfied(format!(
                                "Layer2 {} batch {} missing ah bit record",
                                label, position
                            ))
                        })?;
                let expected_ah_bits =
                    witness.double_shuffle.ah.ah_values.get(0).ok_or_else(|| {
                        ZKPError::ConstraintUnsatisfied(format!(
                            "Layer2 {} witness {} missing ah bit record",
                            label, entry_idx
                        ))
                    })?;
                if recorded_ah_bits != expected_ah_bits {
                    return Err(ZKPError:: ConstraintUnsatisfied(format!(
                        "Layer2 {} batch {} ah bit record is inconsistent with the original witness",
                        label, position
                    )));
                }
            }
        }

        for &entry_idx in &batch.entry_indices {
            if let Some(witness) = source.get(entry_idx) {
                accumulator.record_field_metrics(
                    ConstraintKind::Fl2SaCore,
                    precision,
                    &witness.field_ops,
                );
            }
        }

        accumulator.record_field_metrics(ConstraintKind::Fl2SaBatch, precision, &batch.field_ops);

        Ok(())
    }

    fn verify_linear_witness(
        &self,
        witness: &LinearCombinationWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        idx: usize,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_linear_constraints(witness.num_constraints);

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 witness item {} {} linear constraint is not satisfied",
                        idx, label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer2 witness item {} {} Linear constraint verification failed: {:?}",
                        idx, label, err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(
            ConstraintKind::LinearCombination,
            precision,
            &witness.field_ops,
        );

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Layer2VerificationReport {
    pub constraints: ConstraintAccumulator,
    pub num_entries: usize,
    pub tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DenseMatrix, SparseMatrix};
    use crate::zkp::prover::Layer2Prover;
    use crate::zkp::verifiers::common::VerificationLevel;

    fn sample_input_matrix() -> DenseMatrix<f32> {
        let data = vec![-1.75f32, -0.25f32, 0.0f32, 1.5f32];
        DenseMatrix::new(data, (2, 2))
    }

    fn identity_sparse(size: usize) -> SparseMatrix<f32> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..size {
            indices.push((i as i64, i as i64));
            values.push(1.0);
        }
        SparseMatrix::new(indices, values, (size, size))
    }

    #[test]
    fn verifier_accepts_valid_layer2_witness() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);
        let prover = Layer2Prover::new(config.clone());
        let inputs = sample_input_matrix();
        let support = inputs.clone();
        let adj = identity_sparse(inputs.shape.0);
        let witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer2 witness");

        assert!(
            !witness.fl2sa_input_batches.is_empty(),
            "batch witness for inputs should be present"
        );
        assert!(
            witness
                .fl2sa_input_batches
                .iter()
                .any(|batch| batch.double_shuffle.core.batch_size > 1),
            "at least one input batch should aggregate multiple witnesses"
        );
        assert!(
            witness
                .fl2sa_poly_batches
                .iter()
                .any(|batch| batch.double_shuffle.core.batch_size > 1),
            "polynomial batches should exercise aggregated double-shuffle"
        );
        assert!(
            witness
                .fl2sa_relu_batches
                .iter()
                .any(|batch| batch.double_shuffle.core.batch_size > 1),
            "relu batches should exercise aggregated double-shuffle"
        );
        assert!(
            !witness.mulfp_batches.is_empty(),
            "mulfp batch witness should be present"
        );
        assert!(
            witness
                .mulfp_batches
                .iter()
                .any(|batch| batch.entry_indices.len() > 1),
            "at least one mulfp batch should aggregate multiple witnesses"
        );

        let verifier = Layer2Verifier::new(config.clone());
        let report = verifier.verify(&witness).expect("layer2 verification");

        assert_eq!(report.num_entries, 4);
        let expected_sparse_mulfp: usize = witness.sparse_product.mulfp_witness.len()
            + witness
                .sparse_product
                .mulfp_batches
                .iter()
                .map(|batch| batch.num_constraints)
                .sum::<usize>();
        let expected_mulfp_constraints: usize = witness.mulfp_witness.len()
            + witness
                .mulfp_batches
                .iter()
                .map(|batch| batch.num_constraints)
                .sum::<usize>()
            + expected_sparse_mulfp;
        assert_eq!(
            report.constraints.stats.mulfp_count,
            expected_mulfp_constraints
        );
        let expected_fl2sa_constraints: usize = witness
            .fl2sa_input_batches
            .iter()
            .chain(witness.fl2sa_poly_batches.iter())
            .chain(witness.fl2sa_relu_batches.iter())
            .map(|batch| batch.num_constraints)
            .sum();
        assert_eq!(
            report.constraints.stats.fl2sa_count,
            expected_fl2sa_constraints
        );
        let sparse_linear: usize = witness
            .sparse_product
            .row_relations
            .iter()
            .map(|relation| relation.num_constraints)
            .sum();
        let expected_linear = 2 * config.fl2sa_alpha * report.num_entries + sparse_linear;
        assert_eq!(report.constraints.stats.auxiliary_count, expected_linear);
    }

    #[test]
    fn verifier_rejects_tampered_relu_output() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);
        let prover = Layer2Prover::new(config.clone());

        let inputs = DenseMatrix::new(vec![0.5f32], (1, 1));
        let support = inputs.clone();
        let adj = identity_sparse(1);
        let mut witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer2 witness");
        witness.entries[0].relu_value += 0.1;

        let verifier = Layer2Verifier::new(config.clone());
        match verifier.verify(&witness) {
            Err(ZKPError::NumericalError(_)) => {}
            other => panic!("expected numerical error, got {:?}", other),
        }
    }

    #[test]
    fn verifier_rejects_fl2sa_trace_tampering() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);
        let prover = Layer2Prover::new(config.clone());

        let inputs = sample_input_matrix();
        let support = inputs.clone();
        let adj = identity_sparse(inputs.shape.0);
        let mut witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer2 witness");
        let batch = witness
            .fl2sa_input_batches
            .get_mut(0)
            .expect("expected at least one batch");
        if let Some(bits) = batch.double_shuffle.core.p_l_values.get_mut(0) {
            if let Some(first) = bits.get_mut(0) {
                *first ^= 1;
            }
        }

        let verifier = Layer2Verifier::new(config.clone());
        match verifier.verify(&witness) {
            Err(ZKPError::ConstraintUnsatisfied(msg)) => {
                assert!(msg.contains("p_l bit"), "unexpected message: {}", msg);
            }
            other => panic!("expected constraint unsatisfied, got {:?}", other),
        }
    }
}
