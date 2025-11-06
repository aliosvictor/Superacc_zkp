use std::sync::Arc;

use crate::types::OperationPrecision;
use crate::zkp::prover::{
    Fl2saWitness, Layer4Witness, LinearRelationWitness, MulFPBatchWitness, MulFPWitness,
    SoftmaxNodeWitness, SoftmaxWitnessEntry, SparseMatMulWitness,
};
use crate::zkp::utils::fl2sa::Fl2saBatchWitness;
use crate::zkp::utils::sa2fl::Sa2flBatchWitness;
use crate::zkp::verifiers::common::{
    ConstraintAccumulator, ConstraintKind, FloatBitExtractor, GCNZKPConfig, VerificationLevel,
    ZKPError,
};

const STABILIZED_RANGE: f64 = 2.0;
const STABILIZED_MARGIN: f64 = 0.5;

pub struct Layer4Verifier {
    config: Arc<GCNZKPConfig>,
}

impl Layer4Verifier {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn verify(&self, witness: &Layer4Witness) -> Result<Layer4VerificationReport, ZKPError> {
        let mut accumulator = ConstraintAccumulator::new();
        let should_check_sat = !matches!(self.config.verification_level, VerificationLevel::Fast);
        let tolerance = self.config.tolerance;
        let probability_tolerance = tolerance.max(1e-3);
        let negative_slack = probability_tolerance.max(1e-4);

        self.verify_sparse_product(
            &witness.sparse_product,
            should_check_sat,
            &mut accumulator,
            "Layer4",
        )?;

        for (batch_idx, batch) in witness.fl2sa_max_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_max_list,
                should_check_sat,
                &mut accumulator,
                &format!("max-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_denominator_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_denominator_list,
                should_check_sat,
                &mut accumulator,
                &format!("denominator-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_logit_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_logit_list,
                should_check_sat,
                &mut accumulator,
                &format!("logit-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_stabilized_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_stabilized_list,
                should_check_sat,
                &mut accumulator,
                &format!("stabilized-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_numerator_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_numerator_list,
                should_check_sat,
                &mut accumulator,
                &format!("numerator-batch-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.fl2sa_softmax_batches.iter().enumerate() {
            self.verify_fl2sa_batch(
                batch,
                &witness.fl2sa_softmax_list,
                should_check_sat,
                &mut accumulator,
                &format!("softmax-batch-{}", batch_idx),
            )?;
        }

        for (batch_idx, batch) in witness.sa2fl_max_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-max-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.sa2fl_denominator_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-denominator-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.sa2fl_logit_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-logit-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.sa2fl_stabilized_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-stabilized-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.sa2fl_numerator_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-numerator-{}", batch_idx),
            )?;
        }
        for (batch_idx, batch) in witness.sa2fl_softmax_batches.iter().enumerate() {
            self.verify_sa2fl_batch(
                batch,
                should_check_sat,
                &mut accumulator,
                &format!("sa2fl-softmax-{}", batch_idx),
            )?;
        }

        for (node_idx, node) in witness.nodes.iter().enumerate() {
            self.verify_node(
                node,
                node_idx,
                witness,
                should_check_sat,
                negative_slack,
                probability_tolerance,
                &mut accumulator,
            )?;
        }

        for (batch_idx, batch) in witness.mulfp_batches.iter().enumerate() {
            for &entry_idx in &batch.entry_indices {
                if entry_idx >= witness.mulfp_witness.len() {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 MULFP batch witness {} group index {} out of bounds",
                        batch_idx, entry_idx
                    )));
                }
            }
            self.verify_mulfp_batch(batch, should_check_sat, &mut accumulator, batch_idx)?;
        }

        if let Some(selectors) = &witness.softmax_selectors {
            self.verify_softmax_selectors(selectors, witness, should_check_sat, &mut accumulator)?;
        }

        Ok(Layer4VerificationReport {
            constraints: accumulator,
            num_nodes: witness.num_nodes,
            num_classes: witness.num_classes,
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
            self.verify_sparse_mulfp_witness(mulfp, should_check_sat, accumulator, idx, label)?;
        }
        for (batch_idx, batch) in sparse.mulfp_batches.iter().enumerate() {
            self.verify_mulfp_batch(batch, should_check_sat, accumulator, batch_idx)?;
        }
        for (idx, relation) in sparse.row_relations.iter().enumerate() {
            self.verify_linear_witness(
                relation,
                should_check_sat,
                accumulator,
                idx,
                None,
                "sparse-row",
            )?;
        }

        if let Some(row_diff) = &sparse.row_diff_witness {
            self.verify_row_diffs(row_diff, sparse, should_check_sat, accumulator)?;
        }

        Ok(())
    }

    fn verify_node(
        &self,
        node: &SoftmaxNodeWitness,
        node_idx: usize,
        layer_witness: &Layer4Witness,
        should_check_sat: bool,
        negative_slack: f64,
        probability_tolerance: f64,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let tolerance = self.config.tolerance;
        let max_logit_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_max_list,
            node.fl2sa_max_idx,
            node_idx,
            None,
            "max_logit",
        )?;
        self.assert_fl2sa_matches(node.max_logit_value, max_logit_fl2sa, tolerance)?;

        let denominator_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_denominator_list,
            node.fl2sa_denominator_idx,
            node_idx,
            None,
            "denominator",
        )?;
        self.assert_fl2sa_matches(
            node.denominator_value,
            denominator_fl2sa,
            probability_tolerance,
        )?;

        if node.denominator_value <= 0.0 || !node.denominator_value.is_finite() {
            return Err(ZKPError::NumericalError(node.denominator_value as f64));
        }

        for (class_idx, entry) in node.entries.iter().enumerate() {
            self.verify_entry(
                entry,
                node_idx,
                class_idx,
                layer_witness,
                should_check_sat,
                negative_slack,
                probability_tolerance,
                accumulator,
            )?;
        }

        self.verify_linear_witness(
            &node.denominator_relation_witness,
            should_check_sat,
            accumulator,
            node_idx,
            None,
            "denominator-sum",
        )?;
        self.verify_linear_witness(
            &node.probability_relation_witness,
            should_check_sat,
            accumulator,
            node_idx,
            None,
            "softmax-normalization",
        )?;

        Ok(())
    }

    fn verify_entry(
        &self,
        entry: &SoftmaxWitnessEntry,
        node_idx: usize,
        class_idx: usize,
        layer_witness: &Layer4Witness,
        should_check_sat: bool,
        negative_slack: f64,
        probability_tolerance: f64,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let tolerance = self.config.tolerance;
        let logit_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_logit_list,
            entry.fl2sa_logit_idx,
            node_idx,
            Some(class_idx),
            "logit",
        )?;
        self.assert_fl2sa_matches(entry.logit_value, logit_fl2sa, tolerance)?;
        let stabilized_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_stabilized_list,
            entry.fl2sa_stabilized_idx,
            node_idx,
            Some(class_idx),
            "stabilized",
        )?;
        self.assert_fl2sa_matches(entry.stabilized_value, stabilized_fl2sa, tolerance)?;
        let numerator_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_numerator_list,
            entry.fl2sa_numerator_idx,
            node_idx,
            Some(class_idx),
            "numerator",
        )?;
        self.assert_fl2sa_matches(entry.numerator_value, numerator_fl2sa, tolerance)?;
        let softmax_fl2sa = self.resolve_fl2sa(
            &layer_witness.fl2sa_softmax_list,
            entry.fl2sa_softmax_idx,
            node_idx,
            Some(class_idx),
            "softmax",
        )?;
        self.assert_fl2sa_matches(entry.softmax_value, softmax_fl2sa, probability_tolerance)?;

        let t2 = self.resolve_mulfp(layer_witness, entry.t2_idx, node_idx, class_idx, "t^2")?;
        self.verify_mulfp_witness(
            t2,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "t^2",
        )?;

        let t3 = self.resolve_mulfp(layer_witness, entry.t3_idx, node_idx, class_idx, "t^3")?;
        self.verify_mulfp_witness(
            t3,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "t^3",
        )?;

        let t4 = self.resolve_mulfp(layer_witness, entry.t4_idx, node_idx, class_idx, "t^4")?;
        self.verify_mulfp_witness(
            t4,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "t^4",
        )?;

        let t5 = self.resolve_mulfp(layer_witness, entry.t5_idx, node_idx, class_idx, "t^5")?;
        self.verify_mulfp_witness(
            t5,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "t^5",
        )?;

        let t6 = self.resolve_mulfp(layer_witness, entry.t6_idx, node_idx, class_idx, "t^6")?;
        self.verify_mulfp_witness(
            t6,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "t^6",
        )?;

        let term_c1 = self.resolve_mulfp(
            layer_witness,
            entry.term_c1_idx,
            node_idx,
            class_idx,
            "c1*t",
        )?;
        self.verify_mulfp_witness(
            term_c1,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c1*t",
        )?;

        let term_c2 = self.resolve_mulfp(
            layer_witness,
            entry.term_c2_idx,
            node_idx,
            class_idx,
            "c2*t^2",
        )?;
        self.verify_mulfp_witness(
            term_c2,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c2*t^2",
        )?;

        let term_c3 = self.resolve_mulfp(
            layer_witness,
            entry.term_c3_idx,
            node_idx,
            class_idx,
            "c3*t^3",
        )?;
        self.verify_mulfp_witness(
            term_c3,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c3*t^3",
        )?;

        let term_c4 = self.resolve_mulfp(
            layer_witness,
            entry.term_c4_idx,
            node_idx,
            class_idx,
            "c4*t^4",
        )?;
        self.verify_mulfp_witness(
            term_c4,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c4*t^4",
        )?;

        let term_c5 = self.resolve_mulfp(
            layer_witness,
            entry.term_c5_idx,
            node_idx,
            class_idx,
            "c5*t^5",
        )?;
        self.verify_mulfp_witness(
            term_c5,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c5*t^5",
        )?;

        let term_c6 = self.resolve_mulfp(
            layer_witness,
            entry.term_c6_idx,
            node_idx,
            class_idx,
            "c6*t^6",
        )?;
        self.verify_mulfp_witness(
            term_c6,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "c6*t^6",
        )?;

        let softmax_times_denom = self.resolve_mulfp(
            layer_witness,
            entry.softmax_times_denom_idx,
            node_idx,
            class_idx,
            "softmax*denom",
        )?;
        self.verify_mulfp_witness(
            softmax_times_denom,
            should_check_sat,
            accumulator,
            node_idx,
            class_idx,
            "softmax*denom",
        )?;

        self.verify_linear_witness(
            &entry.numerator_relation_witness,
            should_check_sat,
            accumulator,
            node_idx,
            Some(class_idx),
            "numerator-poly",
        )?;

        if !entry.stabilized_value.is_finite()
            || entry.stabilized_value < -STABILIZED_RANGE - STABILIZED_MARGIN
            || entry.stabilized_value > STABILIZED_RANGE + STABILIZED_MARGIN
        {
            return Err(ZKPError::NumericalError(entry.stabilized_value as f64));
        }

        if entry.numerator_value <= 0.0 || !entry.numerator_value.is_finite() {
            return Err(ZKPError::NumericalError(entry.numerator_value as f64));
        }

        if !entry.softmax_value.is_finite() {
            return Err(ZKPError::NumericalError(entry.softmax_value as f64));
        }

        if (entry.softmax_value as f64) < -negative_slack {
            return Err(ZKPError::NumericalError(entry.softmax_value as f64));
        }

        Ok(())
    }

    fn resolve_fl2sa<'a>(
        &self,
        source: &'a [Fl2saWitness],
        idx: usize,
        node_idx: usize,
        class_idx: Option<usize>,
        label: &str,
    ) -> Result<&'a Fl2saWitness, ZKPError> {
        source.get(idx).ok_or_else(|| {
            let scope = match class_idx {
                Some(c) => format!("Node{} Category{}", node_idx, c),
                None => format!("Node{}", node_idx),
            };
            ZKPError::ConstraintUnsatisfied(format!(
                "{} FL2SA index of Layer4 witness {} {} is out of range (len={})",
                scope,
                label,
                idx,
                source.len()
            ))
        })
    }

    fn assert_fl2sa_matches(
        &self,
        claimed: f64,
        witness: &Fl2saWitness,
        tolerance: f64,
    ) -> Result<(), ZKPError> {
        let expected = FloatBitExtractor::quantize_to_precision(
            self.decode_fl2sa_value(witness),
            self.config.precision_mode,
        );
        let delta = (claimed - expected).abs();
        if delta > tolerance {
            return Err(ZKPError::NumericalError(delta));
        }
        Ok(())
    }

    fn decode_fl2sa_value(&self, witness: &Fl2saWitness) -> f64 {
        let precision = self.config.fl2sa_effective_precision();
        let mantissa =
            FloatBitExtractor::mantissa_from_blocks(&witness.v, self.config.fl2sa_w, precision);
        FloatBitExtractor::compose_from_components(witness.b, witness.p, mantissa, precision)
    }

    fn verify_linear_witness(
        &self,
        witness: &LinearRelationWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        node_idx: usize,
        class_idx: Option<usize>,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_linear_constraints(witness.num_constraints);

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    let scope = match class_idx {
                        Some(class_idx) => format!("Node{} Category{}", node_idx, class_idx),
                        None => format!("Node{}", node_idx),
                    };
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 witness {}'s {} linear constraint is not satisfied",
                        scope, label
                    )));
                }
                Err(err) => {
                    let scope = match class_idx {
                        Some(class_idx) => format!("Node{} Category{}", node_idx, class_idx),
                        None => format!("Node{}", node_idx),
                    };
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Linear constraint verification failed for {} of Layer4 witness {}: {:?}",
                        scope, label, err
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

    fn verify_mulfp_witness(
        &self,
        witness: &MulFPWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        node_idx: usize,
        class_idx: usize,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_mulfp_constraint();

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 witness node {} class {} of {} failed constraint verification",
                        node_idx, class_idx, label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::MULFPVerificationError(format!(
                        "Verification failed for {} of Layer4 witness node {} class {}: {:?}",
                        node_idx, class_idx, label, err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::MulFpCore, precision, &witness.field_ops);

        Ok(())
    }

    fn verify_sparse_mulfp_witness(
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
                        "Layer4 {} item {} of sparse multiplication failed constraint verification",
                        label, idx
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::MULFPVerificationError(format!(
                        "Layer4 {} sparse multiplication item {} failed verification: {:?}",
                        label, idx, err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::MulFpCore, precision, &witness.field_ops);

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

    fn verify_softmax_selectors(
        &self,
        witness: &crate::zkp::prover::layer4::SoftmaxSelectorWitness,
        layer_witness: &Layer4Witness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        if witness.selectors.len() != layer_witness.num_nodes {
            return Err(ZKPError::ConstraintUnsatisfied(
                "The number of Softmax selectors is inconsistent with the number of nodes"
                    .to_string(),
            ));
        }

        for (node_idx, selector) in witness.selectors.iter().enumerate() {
            if selector.len() != layer_witness.num_classes {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "Softmax selector row {} length is inconsistent",
                    node_idx
                )));
            }
            let ones = selector.iter().filter(|&&b| b == 1).count();
            if ones != 1 {
                return Err(ZKPError::ConstraintUnsatisfied(format!(
                    "Softmax selector row {} is not one-hot (ones={})",
                    node_idx, ones
                )));
            }
            let selected_idx = selector.iter().position(|&b| b == 1).ok_or_else(|| {
                ZKPError::ConstraintUnsatisfied(format!(
                    "Softmax selector line {} is missing 1",
                    node_idx
                ))
            })?;
            let expected_idx = layer_witness
                .nodes
                .get(node_idx)
                .ok_or_else(|| {
                    ZKPError::ConstraintUnsatisfied(format!(
                        "Softmax selector row {} exceeds the number of nodes",
                        node_idx
                    ))
                })?
                .max_class_idx;
            if selected_idx != expected_idx {
                return Err(ZKPError:: ConstraintUnsatisfied(format!(
                    "Position 1 of Softmax selector row {} is inconsistent with max witness: expected {} actual {}",
                    node_idx, expected_idx, selected_idx
                )));
            }
        }

        accumulator.add_linear_constraints(witness.num_constraints);

        if should_check_sat {
            match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(
                        "Softmax selector witness failed SAT".to_string(),
                    ));
                }
                Err(err) => {
                    return Err(ZKPError::ConfigError(format!(
                        "Verification of Softmax selector witness failed: {:?}",
                        err
                    )));
                }
            }
        }

        accumulator.record_field_metrics(ConstraintKind::Auxiliary, precision, &witness.field_ops);

        Ok(())
    }

    fn resolve_mulfp<'a>(
        &self,
        layer_witness: &'a Layer4Witness,
        idx: usize,
        node_idx: usize,
        class_idx: usize,
        label: &str,
    ) -> Result<&'a MulFPWitness, ZKPError> {
        layer_witness.mulfp_witness.get(idx).ok_or_else(|| {
            ZKPError::ConstraintUnsatisfied(format!(
                "Layer4 witness node {} category {}'s {} index {} is out of bounds",
                node_idx, class_idx, label, idx
            ))
        })
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
                        "Layer4 MULFP batch witness group {} failed constraint verification",
                        batch_idx
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::MULFPVerificationError(format!(
                        "Layer4 MULFP batch witness group {} failed to verify: {:?}",
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
                "Layer4 {} batch sizes are inconsistent: batch_size={}, indices={}",
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
                "Layer4 {} Double-Shuffling record and index length are inconsistent",
                label
            )));
        }

        if should_check_sat {
            match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 {} batch Double-Shuffling constraints are not met",
                        label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 {} Batch Double-Shuffling SAT failed: {:?}",
                        label, err
                    )));
                }
            }

            for (position, &entry_idx) in batch.entry_indices.iter().enumerate() {
                let witness = source.get(entry_idx).ok_or_else(|| {
                    ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 {} batch verification index {} out of range",
                        label, entry_idx
                    ))
                })?;

                match witness.instance.is_sat(&witness.vars, &witness.inputs) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(ZKPError::ConstraintUnsatisfied(format!(
                            "Index {} in Layer4 {} batch witness does not satisfy the constraint",
                            label, entry_idx
                        )));
                    }
                    Err(err) => {
                        return Err(ZKPError::FL2SAConversionError(format!(
                            "Layer4 {} batch witness index {} verification failed: {:?}",
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
                            "Layer4 {} batch {} missing p_l bit record",
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
                                "Layer4 {} witness {} missing p_l bit record",
                                label, entry_idx
                            ))
                        })?;
                if recorded_p_bits != expected_p_bits {
                    return Err(ZKPError:: ConstraintUnsatisfied(format!(
                        "Layer4 {} batch {} p_l bit record is inconsistent with the original witness",
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
                            "Layer4 {} batch {} missing r_bit record",
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
                                "Layer4 {} witness {} missing r_bit record",
                                label, entry_idx
                            ))
                        })?;
                if recorded_r_bits != expected_r_bits {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 {} batch {} r_bit record is inconsistent with the original witness",
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
                                "Layer4 {} batch {} missing ah bit record",
                                label, position
                            ))
                        })?;
                let expected_ah_bits =
                    witness.double_shuffle.ah.ah_values.get(0).ok_or_else(|| {
                        ZKPError::ConstraintUnsatisfied(format!(
                            "Layer4 {} witness {} missing ah bit record",
                            label, entry_idx
                        ))
                    })?;
                if recorded_ah_bits != expected_ah_bits {
                    return Err(ZKPError:: ConstraintUnsatisfied(format!(
                        "Layer4 {} batch {} ah bit record is inconsistent with the original witness",
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

    fn verify_sa2fl_batch(
        &self,
        batch: &Sa2flBatchWitness,
        should_check_sat: bool,
        accumulator: &mut ConstraintAccumulator,
        label: &str,
    ) -> Result<(), ZKPError> {
        let precision: OperationPrecision = self.config.precision_mode.into();
        accumulator.add_sa2fl_constraints(batch.num_constraints);
        if should_check_sat {
            match batch.instance.is_sat(&batch.vars, &batch.inputs) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(ZKPError::ConstraintUnsatisfied(format!(
                        "Layer4 SA2FL batch witness {} failed constraint verification",
                        label
                    )));
                }
                Err(err) => {
                    return Err(ZKPError::ConfigError(format!(
                        "Layer4 SA2FL batch witness {} verification failed: {:?}",
                        label, err
                    )));
                }
            }
        }
        accumulator.record_field_metrics(ConstraintKind::Sa2FlBatch, precision, &batch.field_ops);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Layer4VerificationReport {
    pub constraints: ConstraintAccumulator,
    pub num_nodes: usize,
    pub num_classes: usize,
    pub tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DenseMatrix, SparseMatrix};
    use crate::zkp::prover::Layer4Prover;
    use crate::zkp::verifiers::common::VerificationLevel;
    use std::sync::Arc;

    fn sample_logits() -> DenseMatrix<f32> {
        let data = vec![0.25f32, -0.75f32, 1.5f32, 0.1f32, -0.3f32, 0.85f32];
        DenseMatrix::new(data, (2, 3))
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
    fn verifier_accepts_valid_layer4_witness() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);

        let prover = Layer4Prover::new(config.clone());
        let logits = sample_logits();
        let support = logits.clone();
        let adj = identity_sparse(logits.shape.0);
        let witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer4 witness");

        assert!(
            !witness.fl2sa_max_batches.is_empty(),
            "max batches should be populated"
        );
        let has_batched_shuffles = [
            &witness.fl2sa_max_batches,
            &witness.fl2sa_denominator_batches,
            &witness.fl2sa_logit_batches,
            &witness.fl2sa_stabilized_batches,
            &witness.fl2sa_numerator_batches,
            &witness.fl2sa_softmax_batches,
        ]
        .iter()
        .flat_map(|collection| collection.iter())
        .any(|batch| batch.double_shuffle.core.batch_size > 1);
        assert!(
            has_batched_shuffles,
            "expected at least one FL2SA batch to exercise aggregated double-shuffle"
        );

        let verifier = Layer4Verifier::new(config.clone());
        let report = verifier.verify(&witness).expect("layer4 verification");

        assert_eq!(report.num_nodes, 2);
        assert_eq!(report.num_classes, 3);
        assert!(report.constraints.stats.mulfp_count > 0);
        assert!(report.constraints.stats.fl2sa_count > 0);
        assert!(report.constraints.stats.sa2fl_count > 0);
    }

    #[test]
    fn verifier_rejects_tampered_softmax_distribution() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);

        let prover = Layer4Prover::new(config.clone());
        let logits = sample_logits();
        let support = logits.clone();
        let adj = identity_sparse(logits.shape.0);
        let mut witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer4 witness");

        witness.nodes[0].entries[0].softmax_value += 0.1;

        let verifier = Layer4Verifier::new(config.clone());
        match verifier.verify(&witness) {
            Err(ZKPError::NumericalError(_)) => {}
            other => panic!("expected numerical error, got {:?}", other),
        }
    }

    #[test]
    fn verifier_rejects_fl2sa_batch_trace_tampering() {
        let mut config = GCNZKPConfig::single_precision();
        config.verification_level = VerificationLevel::Optimized;
        let config = Arc::new(config);

        let prover = Layer4Prover::new(config.clone());
        let logits = sample_logits();
        let support = logits.clone();
        let adj = identity_sparse(logits.shape.0);
        let mut witness = prover
            .generate_witness(&support, &adj, None)
            .expect("layer4 witness");

        let batch = witness
            .fl2sa_max_batches
            .get_mut(0)
            .expect("expected max batch");
        if let Some(bits) = batch.double_shuffle.core.p_l_values.get_mut(0) {
            if let Some(first) = bits.get_mut(0) {
                *first ^= 1;
            }
        }

        let verifier = Layer4Verifier::new(config.clone());
        match verifier.verify(&witness) {
            Err(ZKPError::ConstraintUnsatisfied(msg)) => {
                assert!(msg.contains("p_l bit"), "unexpected error message: {}", msg);
            }
            other => panic!("expected constraint unsatisfied, got {:?}", other),
        }
    }
}
