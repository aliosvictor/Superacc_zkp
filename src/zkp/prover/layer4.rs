use std::cmp::Ordering;
use std::sync::Arc;

use crate::types::{DenseMatrix, FloatType, OperationPrecision, SparseMatrix};
use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};
use crate::zkp::operation_tracker::{record_add, record_mul};
use crate::zkp::prover::fl2sa_utils::{
    derive_fl2sa_params, fl2sa_witness_from_value, mulfp_input_from_pair, superacc_from_value,
};
use crate::zkp::prover::sparse_product::{
    SparseMatMulWitness, SparseProductProver, SparseProductResult,
};
use crate::zkp::prover::{Fl2saWitness, LinearRelationWitness, MulFPBatchWitness, MulFPWitness};
use crate::zkp::utils::fl2sa::{
    sample_ah_double_shuffle_randomness, AhDoubleShuffleContext, AhDoubleShuffleWitness,
    Fl2saBatchWitness,
};
use crate::zkp::utils::linear::{build_superacc_linear_witness, SuperaccLinearTerm};
use crate::zkp::utils::mulfp::produce_r1cs_mulfp_detached;
use crate::zkp::utils::sa2fl::{SA2FLParams, Sa2flBatchWitness, Sa2flWitness};
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};
use rand::rngs::OsRng;

const SOFTMAX_C0: f64 = 0.999_961_9;
const SOFTMAX_C1: f64 = 1.001_148_2;
const SOFTMAX_C2: f64 = 0.500_340_46;
const SOFTMAX_C3: f64 = 0.164_113_76;
const SOFTMAX_C4: f64 = 0.041_203_48;
const SOFTMAX_C5: f64 = 0.009_705_75;
const SOFTMAX_C6: f64 = 0.001_585_57;

pub struct Layer4Prover {
    config: Arc<GCNZKPConfig>,
}

impl Layer4Prover {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn generate_witness<T: FloatType>(
        &self,
        support: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        bias: Option<&[T]>,
    ) -> Result<Layer4Witness, ZKPError> {
        let precision = self.config.precision_mode;
        let sparse_prover = SparseProductProver::new(self.config.clone());
        let SparseProductResult {
            witness: sparse_witness,
            outputs: logit_values,
            output_shape,
            output_fl2sa,
        } = sparse_prover.prove(adj, support, bias, "Layer4")?;
        let (num_nodes, num_classes) = output_shape;
        if num_classes == 0 {
            return Err(ZKPError::ConfigError(
                "The number of softmax layer categories must be greater than 0".to_owned(),
            ));
        }

        let fl2sa_params = derive_fl2sa_params(&self.config);
        let precision_tag: OperationPrecision = precision.into();
        let coeff_c0 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C0, precision);
        let coeff_c1 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C1, precision);
        let coeff_c2 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C2, precision);
        let coeff_c3 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C3, precision);
        let coeff_c4 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C4, precision);
        let coeff_c5 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C5, precision);
        let coeff_c6 = FloatBitExtractor::quantize_to_precision(SOFTMAX_C6, precision);
        let coeff_one = FloatBitExtractor::quantize_to_precision(1.0, precision);

        let c0_superacc =
            superacc_from_value(&self.config, &fl2sa_params, coeff_c0, Some(precision))?;
        let one_superacc =
            superacc_from_value(&self.config, &fl2sa_params, coeff_one, Some(precision))?;
        let linear_radix = 1i64 << self.config.fl2sa_w;

        let approx_capacity = num_nodes.saturating_mul(num_classes.max(1));
        let mut fl2sa_max_list = Vec::with_capacity(num_nodes);
        let mut fl2sa_denominator_list = Vec::with_capacity(num_nodes);
        let fl2sa_logit_list = output_fl2sa;
        let mut fl2sa_stabilized_list = Vec::with_capacity(approx_capacity);
        let mut fl2sa_numerator_list = Vec::with_capacity(approx_capacity);
        let mut fl2sa_softmax_list = Vec::with_capacity(approx_capacity);
        let mut mulfp_witness = Vec::with_capacity(approx_capacity * 8);
        let mut store_mulfp = |witness: MulFPWitness| {
            let idx = mulfp_witness.len();
            mulfp_witness.push(witness);
            idx
        };
        let mut selector_rows = Vec::with_capacity(num_nodes);
        let mut nodes = Vec::with_capacity(num_nodes);

        for node_idx in 0..num_nodes {
            let mut row_logits = Vec::with_capacity(num_classes);
            for class_idx in 0..num_classes {
                let value = logit_values[node_idx * num_classes + class_idx];
                let quantized = FloatBitExtractor::quantize_to_precision(value, precision);
                if !quantized.is_finite() {
                    return Err(ZKPError::NumericalError(quantized));
                }
                row_logits.push(quantized);
            }

            let (max_selector_idx, max_logit_value) = row_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .ok_or_else(|| {
                    ZKPError::ConfigError("Softmax row missing maximum value".to_owned())
                })?;

            if !max_logit_value.is_finite() {
                return Err(ZKPError::NumericalError(*max_logit_value));
            }

            let mut selector_bits = vec![0u8; num_classes];
            selector_bits[max_selector_idx] = 1;
            selector_rows.push(selector_bits);

            let max_logit = *max_logit_value;

            let fl2sa_max =
                fl2sa_witness_from_value(&self.config, &fl2sa_params, max_logit, "softmax-max")?;
            let fl2sa_max_idx = fl2sa_max_list.len();
            fl2sa_max_list.push(fl2sa_max);

            let mut partial_entries = Vec::with_capacity(num_classes);
            let mut numerator_sum = 0.0f64;
            let mut numerator_superaccs = Vec::with_capacity(num_classes);

            for class_idx in 0..num_classes {
                let logit = row_logits[class_idx];
                record_add(precision_tag, 1);
                let stabilized =
                    FloatBitExtractor::quantize_to_precision(logit - max_logit, precision);

                record_mul(precision_tag, 1);
                let t2 =
                    FloatBitExtractor::quantize_to_precision(stabilized * stabilized, precision);
                record_mul(precision_tag, 1);
                let t3 = FloatBitExtractor::quantize_to_precision(t2 * stabilized, precision);
                record_mul(precision_tag, 1);
                let t4 = FloatBitExtractor::quantize_to_precision(t2 * t2, precision);
                record_mul(precision_tag, 1);
                let t5 = FloatBitExtractor::quantize_to_precision(t4 * stabilized, precision);
                record_mul(precision_tag, 1);
                let t6 = FloatBitExtractor::quantize_to_precision(t3 * t3, precision);

                record_mul(precision_tag, 1);
                let term_c1 =
                    FloatBitExtractor::quantize_to_precision(coeff_c1 * stabilized, precision);
                record_mul(precision_tag, 1);
                let term_c2 = FloatBitExtractor::quantize_to_precision(coeff_c2 * t2, precision);
                record_mul(precision_tag, 1);
                let term_c3 = FloatBitExtractor::quantize_to_precision(coeff_c3 * t3, precision);
                record_mul(precision_tag, 1);
                let term_c4 = FloatBitExtractor::quantize_to_precision(coeff_c4 * t4, precision);
                record_mul(precision_tag, 1);
                let term_c5 = FloatBitExtractor::quantize_to_precision(coeff_c5 * t5, precision);
                record_mul(precision_tag, 1);
                let term_c6 = FloatBitExtractor::quantize_to_precision(coeff_c6 * t6, precision);
                record_add(precision_tag, 6);
                let numerator = FloatBitExtractor::quantize_to_precision(
                    coeff_c0 + term_c1 + term_c2 + term_c3 + term_c4 + term_c5 + term_c6,
                    precision,
                );

                let t2_idx = store_mulfp(self.mulfp_witness_from_pair(stabilized, stabilized)?);
                let t3_idx = store_mulfp(self.mulfp_witness_from_pair(t2, stabilized)?);
                let t4_idx = store_mulfp(self.mulfp_witness_from_pair(t2, t2)?);
                let t5_idx = store_mulfp(self.mulfp_witness_from_pair(t4, stabilized)?);
                let t6_idx = store_mulfp(self.mulfp_witness_from_pair(t3, t3)?);

                let term_c1_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c1, stabilized)?);
                let term_c2_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c2, t2)?);
                let term_c3_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c3, t3)?);
                let term_c4_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c4, t4)?);
                let term_c5_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c5, t5)?);
                let term_c6_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_c6, t6)?);

                let fl2sa_logit_idx = node_idx * num_classes + class_idx;

                let fl2sa_stabilized = fl2sa_witness_from_value(
                    &self.config,
                    &fl2sa_params,
                    stabilized,
                    "softmax-stabilized",
                )?;
                let fl2sa_stabilized_idx = fl2sa_stabilized_list.len();
                fl2sa_stabilized_list.push(fl2sa_stabilized);

                let fl2sa_numerator = fl2sa_witness_from_value(
                    &self.config,
                    &fl2sa_params,
                    numerator,
                    "softmax-numerator",
                )?;
                let numerator_superacc = fl2sa_numerator.superaccumulator.clone();
                let fl2sa_numerator_idx = fl2sa_numerator_list.len();
                fl2sa_numerator_list.push(fl2sa_numerator);

                let term_c1_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c1, Some(precision))?;
                let term_c2_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c2, Some(precision))?;
                let term_c3_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c3, Some(precision))?;
                let term_c4_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c4, Some(precision))?;
                let term_c5_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c5, Some(precision))?;
                let term_c6_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_c6, Some(precision))?;

                let numerator_relation = build_superacc_linear_witness(
                    vec![
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: c0_superacc.clone(),
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c1_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c2_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c3_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c4_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c5_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_c6_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: -1,
                            limbs: numerator_superacc.clone(),
                        },
                    ],
                    linear_radix,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Failed to build Softmax molecule constraints: {}",
                        e
                    ))
                })?;

                record_add(precision_tag, 1);
                numerator_sum =
                    FloatBitExtractor::quantize_to_precision(numerator_sum + numerator, precision);
                numerator_superaccs.push(numerator_superacc);

                partial_entries.push(PendingSoftmaxEntry {
                    logit_value: logit,
                    stabilized_value: stabilized,
                    numerator_value: numerator,
                    t2_idx,
                    t3_idx,
                    t4_idx,
                    t5_idx,
                    t6_idx,
                    term_witnesses: SoftmaxTermWitnesses {
                        term_c1_idx,
                        term_c2_idx,
                        term_c3_idx,
                        term_c4_idx,
                        term_c5_idx,
                        term_c6_idx,
                    },
                    fl2sa_logit_idx,
                    fl2sa_stabilized_idx,
                    fl2sa_numerator_idx,
                    numerator_relation_witness: numerator_relation,
                });
            }

            if !numerator_sum.is_finite() || numerator_sum <= 0.0 {
                return Err(ZKPError::NumericalError(numerator_sum as f64));
            }

            let denominator_value =
                FloatBitExtractor::quantize_to_precision(numerator_sum, precision);
            let fl2sa_denominator = fl2sa_witness_from_value(
                &self.config,
                &fl2sa_params,
                denominator_value,
                "softmax-denominator",
            )?;
            let denominator_superacc = fl2sa_denominator.superaccumulator.clone();
            let fl2sa_denominator_idx = fl2sa_denominator_list.len();
            fl2sa_denominator_list.push(fl2sa_denominator);

            let mut denominator_terms: Vec<SuperaccLinearTerm> = numerator_superaccs
                .into_iter()
                .map(|limbs| SuperaccLinearTerm {
                    coefficient: 1,
                    limbs,
                })
                .collect();
            denominator_terms.push(SuperaccLinearTerm {
                coefficient: -1,
                limbs: denominator_superacc,
            });
            let denominator_relation =
                build_superacc_linear_witness(denominator_terms, linear_radix).map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Failed to build Softmax denominator constraint: {}",
                        e
                    ))
                })?;

            let mut entries = Vec::with_capacity(num_classes);
            let mut softmax_superaccs = Vec::with_capacity(num_classes);

            for partial in partial_entries {
                let softmax_value = FloatBitExtractor::quantize_to_precision(
                    partial.numerator_value / denominator_value,
                    precision,
                );

                let fl2sa_softmax = fl2sa_witness_from_value(
                    &self.config,
                    &fl2sa_params,
                    softmax_value,
                    "softmax-output",
                )?;
                let softmax_superacc = fl2sa_softmax.superaccumulator.clone();
                let fl2sa_softmax_idx = fl2sa_softmax_list.len();
                fl2sa_softmax_list.push(fl2sa_softmax);
                let softmax_times_denom_idx =
                    store_mulfp(self.mulfp_witness_from_pair(softmax_value, denominator_value)?);

                softmax_superaccs.push(softmax_superacc);

                let SoftmaxTermWitnesses {
                    term_c1_idx,
                    term_c2_idx,
                    term_c3_idx,
                    term_c4_idx,
                    term_c5_idx,
                    term_c6_idx,
                } = partial.term_witnesses;

                entries.push(SoftmaxWitnessEntry {
                    logit_value: partial.logit_value,
                    stabilized_value: partial.stabilized_value,
                    numerator_value: partial.numerator_value,
                    softmax_value,
                    t2_idx: partial.t2_idx,
                    t3_idx: partial.t3_idx,
                    t4_idx: partial.t4_idx,
                    t5_idx: partial.t5_idx,
                    t6_idx: partial.t6_idx,
                    term_c1_idx,
                    term_c2_idx,
                    term_c3_idx,
                    term_c4_idx,
                    term_c5_idx,
                    term_c6_idx,
                    fl2sa_logit_idx: partial.fl2sa_logit_idx,
                    fl2sa_stabilized_idx: partial.fl2sa_stabilized_idx,
                    fl2sa_numerator_idx: partial.fl2sa_numerator_idx,
                    fl2sa_softmax_idx,
                    softmax_times_denom_idx,
                    numerator_relation_witness: partial.numerator_relation_witness,
                });
            }

            let mut probability_terms: Vec<SuperaccLinearTerm> = softmax_superaccs
                .into_iter()
                .map(|limbs| SuperaccLinearTerm {
                    coefficient: 1,
                    limbs,
                })
                .collect();
            probability_terms.push(SuperaccLinearTerm {
                coefficient: -1,
                limbs: one_superacc.clone(),
            });
            let probability_relation =
                build_superacc_linear_witness(probability_terms, linear_radix).map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Failed to build Softmax normalization constraint: {}",
                        e
                    ))
                })?;

            nodes.push(SoftmaxNodeWitness {
                max_logit_value: max_logit,
                denominator_value,
                max_class_idx: max_selector_idx,
                fl2sa_max_idx,
                fl2sa_denominator_idx,
                entries,
                denominator_relation_witness: denominator_relation,
                probability_relation_witness: probability_relation,
            });
        }

        let softmax_selector_witness =
            build_softmax_selector_witness(&selector_rows).map_err(|e| {
                ZKPError::ConfigError(format!("Failed to build Softmax selector witness: {}", e))
            })?;

        let batch_size = self.config.batch_size.max(1);
        let assemble_batches =
            |list: &[Fl2saWitness], label: &str| -> Result<Vec<Fl2saBatchWitness>, ZKPError> {
                if list.is_empty() {
                    return Ok(Vec::new());
                }
                let mut batches = Vec::new();
                let mut current: Vec<usize> = Vec::with_capacity(batch_size);
                for idx in 0..list.len() {
                    current.push(idx);
                    let is_last = idx + 1 == list.len();
                    if current.len() == batch_size || is_last {
                        let indices = current.clone();
                        let batch = Fl2saBatchWitness::from_indices(&indices, list, &fl2sa_params)
                            .map_err(|e| {
                                ZKPError::ConfigError(format!(
                                    "Assembling Layer4 {} FL2SA batch witness failed: {}",
                                    label, e
                                ))
                            })?;
                        batches.push(batch);
                        current.clear();
                    }
                }
                Ok(batches)
            };

        let assemble_sa2fl_batches = |list: &[Sa2flWitness],
                                      label: &str|
         -> Result<Vec<Sa2flBatchWitness>, ZKPError> {
            if list.is_empty() {
                return Ok(Vec::new());
            }
            let mut batches = Vec::new();
            let mut current: Vec<usize> = Vec::with_capacity(batch_size);
            for idx in 0..list.len() {
                current.push(idx);
                let is_last = idx + 1 == list.len();
                if current.len() == batch_size || is_last {
                    let indices = current.clone();
                    let batch = Sa2flBatchWitness::from_indices(&indices, list).map_err(|e| {
                        ZKPError::ConfigError(format!(
                            "Assembling Layer4 {} SA2FL batch witness failed: {}",
                            label, e
                        ))
                    })?;
                    batches.push(batch);
                    current.clear();
                }
            }
            Ok(batches)
        };

        let fl2sa_max_batches = assemble_batches(&fl2sa_max_list, "max")?;
        let fl2sa_denominator_batches = assemble_batches(&fl2sa_denominator_list, "denominator")?;
        let fl2sa_logit_batches = assemble_batches(&fl2sa_logit_list, "logit")?;
        let fl2sa_stabilized_batches = assemble_batches(&fl2sa_stabilized_list, "stabilized")?;
        let fl2sa_numerator_batches = assemble_batches(&fl2sa_numerator_list, "numerator")?;
        let fl2sa_softmax_batches = assemble_batches(&fl2sa_softmax_list, "softmax")?;

        let sa2fl_params = &self.config.sa2fl_params;
        let sa2fl_max_list = build_sa2fl_list(sa2fl_params, &fl2sa_max_list).map_err(|e| {
            ZKPError::ConfigError(format!("Building Layer4 max SA2FL witness failed: {}", e))
        })?;
        let sa2fl_denominator_list = build_sa2fl_list(sa2fl_params, &fl2sa_denominator_list)
            .map_err(|e| {
                ZKPError::ConfigError(format!(
                    "Building Layer4 denominator SA2FL witness failed: {}",
                    e
                ))
            })?;
        let sa2fl_logit_list = build_sa2fl_list(sa2fl_params, &fl2sa_logit_list).map_err(|e| {
            ZKPError::ConfigError(format!("Building Layer4 logit SA2FL witness failed: {}", e))
        })?;
        let sa2fl_stabilized_list = build_sa2fl_list(sa2fl_params, &fl2sa_stabilized_list)
            .map_err(|e| {
                ZKPError::ConfigError(format!(
                    "Building Layer4 stabilized SA2FL witness failed: {}",
                    e
                ))
            })?;
        let sa2fl_numerator_list =
            build_sa2fl_list(sa2fl_params, &fl2sa_numerator_list).map_err(|e| {
                ZKPError::ConfigError(format!(
                    "Failed to build Layer4 numerator SA2FL witness: {}",
                    e
                ))
            })?;
        let sa2fl_softmax_list =
            build_sa2fl_list(sa2fl_params, &fl2sa_softmax_list).map_err(|e| {
                ZKPError::ConfigError(format!(
                    "Building Layer4 softmax SA2FL witness failed: {}",
                    e
                ))
            })?;

        let sa2fl_max_batches = assemble_sa2fl_batches(&sa2fl_max_list, "max")?;
        let sa2fl_denominator_batches =
            assemble_sa2fl_batches(&sa2fl_denominator_list, "denominator")?;
        let sa2fl_logit_batches = assemble_sa2fl_batches(&sa2fl_logit_list, "logit")?;
        let sa2fl_stabilized_batches =
            assemble_sa2fl_batches(&sa2fl_stabilized_list, "stabilized")?;
        let sa2fl_numerator_batches = assemble_sa2fl_batches(&sa2fl_numerator_list, "numerator")?;
        let sa2fl_softmax_batches = assemble_sa2fl_batches(&sa2fl_softmax_list, "softmax")?;

        let mut mulfp_batches = Vec::new();
        if !mulfp_witness.is_empty() {
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
                            "Failed to assemble Layer4 MULFP batch witness: {}",
                            e
                        ))
                    })?;
                    mulfp_batches.push(batch);
                    current_indices.clear();
                }
            }
        }

        Ok(Layer4Witness {
            sparse_product: sparse_witness,
            num_nodes,
            num_classes,
            nodes,
            mulfp_witness,
            mulfp_batches,
            fl2sa_max_list,
            fl2sa_denominator_list,
            fl2sa_logit_list,
            fl2sa_stabilized_list,
            fl2sa_numerator_list,
            fl2sa_softmax_list,
            fl2sa_max_batches,
            fl2sa_denominator_batches,
            fl2sa_logit_batches,
            fl2sa_stabilized_batches,
            fl2sa_numerator_batches,
            fl2sa_softmax_batches,
            sa2fl_max_list,
            sa2fl_denominator_list,
            sa2fl_logit_list,
            sa2fl_stabilized_list,
            sa2fl_numerator_list,
            sa2fl_softmax_list,
            sa2fl_max_batches,
            sa2fl_denominator_batches,
            sa2fl_logit_batches,
            sa2fl_stabilized_batches,
            sa2fl_numerator_batches,
            sa2fl_softmax_batches,
            softmax_selectors: softmax_selector_witness,
        })
    }

    fn mulfp_witness_from_pair(&self, lhs: f64, rhs: f64) -> Result<MulFPWitness, ZKPError> {
        let input = mulfp_input_from_pair(lhs, rhs, self.config.precision_mode);
        let artifacts = produce_r1cs_mulfp_detached(&self.config.mulfp_params, Some(&input))
            .map_err(ZKPError::MULFPVerificationError)?;

        Ok(artifacts.into_witness(input))
    }
}

pub struct Layer4Witness {
    pub sparse_product: SparseMatMulWitness,
    pub num_nodes: usize,
    pub num_classes: usize,
    pub nodes: Vec<SoftmaxNodeWitness>,
    pub mulfp_witness: Vec<MulFPWitness>,
    pub mulfp_batches: Vec<MulFPBatchWitness>,
    pub fl2sa_max_list: Vec<Fl2saWitness>,
    pub fl2sa_denominator_list: Vec<Fl2saWitness>,
    pub fl2sa_logit_list: Vec<Fl2saWitness>,
    pub fl2sa_stabilized_list: Vec<Fl2saWitness>,
    pub fl2sa_numerator_list: Vec<Fl2saWitness>,
    pub fl2sa_softmax_list: Vec<Fl2saWitness>,
    pub fl2sa_max_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_denominator_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_logit_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_stabilized_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_numerator_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_softmax_batches: Vec<Fl2saBatchWitness>,
    pub sa2fl_max_list: Vec<Sa2flWitness>,
    pub sa2fl_denominator_list: Vec<Sa2flWitness>,
    pub sa2fl_logit_list: Vec<Sa2flWitness>,
    pub sa2fl_stabilized_list: Vec<Sa2flWitness>,
    pub sa2fl_numerator_list: Vec<Sa2flWitness>,
    pub sa2fl_softmax_list: Vec<Sa2flWitness>,
    pub sa2fl_max_batches: Vec<Sa2flBatchWitness>,
    pub sa2fl_denominator_batches: Vec<Sa2flBatchWitness>,
    pub sa2fl_logit_batches: Vec<Sa2flBatchWitness>,
    pub sa2fl_stabilized_batches: Vec<Sa2flBatchWitness>,
    pub sa2fl_numerator_batches: Vec<Sa2flBatchWitness>,
    pub sa2fl_softmax_batches: Vec<Sa2flBatchWitness>,
    pub softmax_selectors: Option<SoftmaxSelectorWitness>,
}

pub struct SoftmaxNodeWitness {
    pub max_logit_value: f64,
    pub denominator_value: f64,
    pub max_class_idx: usize,
    pub fl2sa_max_idx: usize,
    pub fl2sa_denominator_idx: usize,
    pub entries: Vec<SoftmaxWitnessEntry>,
    pub denominator_relation_witness: LinearRelationWitness,
    pub probability_relation_witness: LinearRelationWitness,
}

pub struct SoftmaxWitnessEntry {
    pub logit_value: f64,
    pub stabilized_value: f64,
    pub numerator_value: f64,
    pub softmax_value: f64,
    pub t2_idx: usize,
    pub t3_idx: usize,
    pub t4_idx: usize,
    pub t5_idx: usize,
    pub t6_idx: usize,
    pub term_c1_idx: usize,
    pub term_c2_idx: usize,
    pub term_c3_idx: usize,
    pub term_c4_idx: usize,
    pub term_c5_idx: usize,
    pub term_c6_idx: usize,
    pub fl2sa_logit_idx: usize,
    pub fl2sa_stabilized_idx: usize,
    pub fl2sa_numerator_idx: usize,
    pub fl2sa_softmax_idx: usize,
    pub softmax_times_denom_idx: usize,
    pub numerator_relation_witness: LinearRelationWitness,
}

pub struct SoftmaxSelectorWitness {
    pub selectors: Vec<Vec<u8>>,
    pub num_classes: usize,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub double_shuffle: AhDoubleShuffleWitness,
    pub field_ops: R1csShapeMetrics,
}

struct PendingSoftmaxEntry {
    logit_value: f64,
    stabilized_value: f64,
    numerator_value: f64,
    t2_idx: usize,
    t3_idx: usize,
    t4_idx: usize,
    t5_idx: usize,
    t6_idx: usize,
    term_witnesses: SoftmaxTermWitnesses,
    fl2sa_logit_idx: usize,
    fl2sa_stabilized_idx: usize,
    fl2sa_numerator_idx: usize,
    numerator_relation_witness: LinearRelationWitness,
}

struct SoftmaxTermWitnesses {
    term_c1_idx: usize,
    term_c2_idx: usize,
    term_c3_idx: usize,
    term_c4_idx: usize,
    term_c5_idx: usize,
    term_c6_idx: usize,
}

fn build_softmax_selector_witness(
    selectors: &[Vec<u8>],
) -> Result<Option<SoftmaxSelectorWitness>, String> {
    if selectors.is_empty() {
        return Ok(None);
    }
    let num_classes = selectors[0].len();
    if num_classes == 0 {
        return Err("Softmax selector length must be greater than 0".to_owned());
    }
    if selectors.iter().any(|row| row.len() != num_classes) {
        return Err("Softmax selector row length is inconsistent".to_owned());
    }

    let batch = selectors.len();
    let num_bits = batch * num_classes;
    let mut vars = vec![Scalar::ZERO.to_bytes(); num_bits + 2];
    let const_one_idx = num_bits;
    let zero_var_idx = const_one_idx + 1;
    vars[const_one_idx] = Scalar::ONE.to_bytes();

    let mut bit_indices = vec![vec![0usize; num_classes]; batch];
    for (row, bits) in selectors.iter().enumerate() {
        for (col, &bit) in bits.iter().enumerate() {
            let idx = row * num_classes + col;
            bit_indices[row][col] = idx;
            let scalar = if bit == 0 { Scalar::ZERO } else { Scalar::ONE };
            vars[idx] = scalar.to_bytes();
        }
    }

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut cursor = 0usize;
    let one = Scalar::ONE.to_bytes();
    for row in 0..batch {
        let cid = cursor;
        cursor += 1;
        for col in 0..num_classes {
            a_entries.push((cid, row * num_classes + col, one));
        }
        b_entries.push((cid, const_one_idx, one));
        c_entries.push((cid, const_one_idx, one));
    }

    let mut rng = OsRng;
    let randomness = sample_ah_double_shuffle_randomness(batch, num_classes, &mut rng);
    let start_idx = vars.len();
    let context = AhDoubleShuffleContext::new(
        batch,
        num_classes,
        start_idx,
        randomness,
        bit_indices.clone(),
        zero_var_idx,
    )?;
    if context.layout().next_var_idx() > vars.len() {
        vars.resize(context.layout().next_var_idx(), Scalar::ZERO.to_bytes());
    }
    cursor = context.append_constraints(
        cursor,
        &mut a_entries,
        &mut b_entries,
        &mut c_entries,
        const_one_idx,
    );
    let double_shuffle = context.populate_witness(&mut vars)?;

    let num_constraints = cursor;
    let num_vars = vars.len();
    let metrics = compute_r1cs_metrics(num_constraints, &a_entries, &b_entries, &c_entries);
    let instance = Instance::new(
        num_constraints,
        num_vars,
        0,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("{:?}", e))?;
    let vars_assignment = VarsAssignment::new(&vars)
        .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
    let inputs_assignment = InputsAssignment::new(&[])
        .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

    Ok(Some(SoftmaxSelectorWitness {
        selectors: selectors.to_vec(),
        num_classes,
        num_constraints,
        num_vars,
        num_inputs: 0,
        instance,
        vars: vars_assignment,
        inputs: inputs_assignment,
        double_shuffle,
        field_ops: metrics,
    }))
}

fn build_sa2fl_list(
    params: &SA2FLParams,
    entries: &[Fl2saWitness],
) -> Result<Vec<Sa2flWitness>, String> {
    entries
        .iter()
        .map(|entry| Sa2flWitness::from_fl2sa(params, entry))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_selector_witness_enforces_one_hot() {
        let selectors = vec![vec![1, 0, 0], vec![0, 1, 0]];
        let witness = build_softmax_selector_witness(&selectors)
            .expect("builder should succeed")
            .expect("witness should exist");
        assert_eq!(witness.num_classes, 3);
        assert_eq!(witness.selectors, selectors);
        assert!(witness
            .instance
            .is_sat(&witness.vars, &witness.inputs)
            .expect("sat check should pass"));
    }
}
