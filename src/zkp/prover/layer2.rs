use std::sync::Arc;

use crate::types::{DenseMatrix, FloatType, OperationPrecision, SparseMatrix};
use crate::zkp::operation_tracker::{record_add, record_mul};
use crate::zkp::prover::fl2sa_utils::{
    derive_fl2sa_params, fl2sa_witness_from_value, mulfp_input_from_pair, superacc_from_value,
};
use crate::zkp::prover::sparse_product::{
    SparseMatMulWitness, SparseProductProver, SparseProductResult,
};
use crate::zkp::prover::{LinearCombinationWitness, MulFPBatchWitness, MulFPWitness};
use crate::zkp::utils::fl2sa::{Fl2saBatchWitness, Fl2saWitness};
use crate::zkp::utils::linear::{build_superacc_linear_combination, SuperaccLinearTerm};
use crate::zkp::utils::mulfp::produce_r1cs_mulfp_detached;
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};

const RELU_A0: f64 = 0.170_982_69;
const RELU_A2: f64 = 1.152_992_4;
const RELU_A4: f64 = -0.351_953_12;
const RELU_A6: f64 = 0.045_708_388;

pub struct Layer2Prover {
    config: Arc<GCNZKPConfig>,
}

impl Layer2Prover {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn generate_witness<T: FloatType>(
        &self,
        support: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        bias: Option<&[T]>,
    ) -> Result<Layer2Witness, ZKPError> {
        let precision = self.config.precision_mode;
        let sparse_prover = SparseProductProver::new(self.config.clone());
        let sparse_result = sparse_prover.prove(adj, support, bias, "Layer2")?;
        let SparseProductResult {
            witness: sparse_witness,
            outputs: pre_activation_values,
            output_shape,
            output_fl2sa,
        } = sparse_result;

        let (num_nodes, hidden_dim) = output_shape;
        let fl2sa_params = derive_fl2sa_params(&self.config);
        let linear_radix = 1i64 << self.config.fl2sa_w;
        let precision_tag: OperationPrecision = precision.into();

        let coeff_a0 = FloatBitExtractor::quantize_to_precision(RELU_A0, precision);
        let coeff_a2 = FloatBitExtractor::quantize_to_precision(RELU_A2, precision);
        let coeff_a4 = FloatBitExtractor::quantize_to_precision(RELU_A4, precision);
        let coeff_a6 = FloatBitExtractor::quantize_to_precision(RELU_A6, precision);
        let half_coeff = FloatBitExtractor::quantize_to_precision(0.5f64, precision);

        let superacc_a0 =
            superacc_from_value(&self.config, &fl2sa_params, coeff_a0, Some(precision))?;

        let total_entries = num_nodes * hidden_dim;
        let mut entries = Vec::with_capacity(total_entries);
        let mut mulfp_witness = Vec::with_capacity(total_entries * 6);
        let mut store_mulfp = |witness: MulFPWitness| {
            let idx = mulfp_witness.len();
            mulfp_witness.push(witness);
            idx
        };
        let fl2sa_inputs_vec = output_fl2sa;
        let mut fl2sa_polys_vec = Vec::with_capacity(total_entries);
        let mut fl2sa_relus_vec = Vec::with_capacity(total_entries);

        for node_idx in 0..num_nodes {
            for feat_idx in 0..hidden_dim {
                let idx = node_idx * hidden_dim + feat_idx;
                let input_value =
                    FloatBitExtractor::quantize_to_precision(pre_activation_values[idx], precision);

                let input_square_raw = input_value * input_value;
                record_mul(precision_tag, 1);
                let input_square =
                    FloatBitExtractor::quantize_to_precision(input_square_raw, precision);

                let input_fourth_raw = input_square * input_square;
                record_mul(precision_tag, 1);
                let input_fourth =
                    FloatBitExtractor::quantize_to_precision(input_fourth_raw, precision);

                let input_sixth_raw = input_fourth * input_square;
                record_mul(precision_tag, 1);
                let input_sixth =
                    FloatBitExtractor::quantize_to_precision(input_sixth_raw, precision);

                let term_a2_raw = coeff_a2 * input_square;
                record_mul(precision_tag, 1);
                let term_a2 = FloatBitExtractor::quantize_to_precision(term_a2_raw, precision);

                let term_a4_raw = coeff_a4 * input_fourth;
                record_mul(precision_tag, 1);
                let term_a4 = FloatBitExtractor::quantize_to_precision(term_a4_raw, precision);

                let term_a6_raw = coeff_a6 * input_sixth;
                record_mul(precision_tag, 1);
                let term_a6 = FloatBitExtractor::quantize_to_precision(term_a6_raw, precision);

                let mut poly_acc = coeff_a0 + term_a2;
                record_add(precision_tag, 1);
                poly_acc += term_a4;
                record_add(precision_tag, 1);
                poly_acc += term_a6;
                record_add(precision_tag, 1);
                let poly = FloatBitExtractor::quantize_to_precision(poly_acc, precision);

                record_add(precision_tag, 1);
                let sum_input_poly =
                    FloatBitExtractor::quantize_to_precision(input_value + poly, precision);

                let relu_mul_raw = half_coeff * sum_input_poly;
                record_mul(precision_tag, 1);
                let relu_output = FloatBitExtractor::quantize_to_precision(relu_mul_raw, precision);

                if cfg!(debug_assertions) {
                    let residual = coeff_a0 + term_a2 + term_a4 + term_a6 - poly;
                    if residual.abs() > f64::EPSILON {
                        println!(
                            "[layer2-debug] residual={} input={} poly={} terms=({}, {}, {}, {})",
                            residual, input_value, poly, term_a2, term_a4, term_a6, coeff_a0
                        );
                    }
                }

                let x2_idx = store_mulfp(self.mulfp_witness_from_pair(input_value, input_value)?);
                let x4_idx = store_mulfp(self.mulfp_witness_from_pair(input_square, input_square)?);
                let x6_idx = store_mulfp(self.mulfp_witness_from_pair(input_fourth, input_square)?);

                let term_a2_idx =
                    store_mulfp(self.mulfp_witness_from_pair(coeff_a2, input_square)?);
                let term_a4_idx =
                    store_mulfp(self.mulfp_witness_from_pair(coeff_a4, input_fourth)?);
                let term_a6_idx = store_mulfp(self.mulfp_witness_from_pair(coeff_a6, input_sixth)?);

                let fl2sa_poly =
                    fl2sa_witness_from_value(&self.config, &fl2sa_params, poly, "relu-poly")?;
                let fl2sa_relu = fl2sa_witness_from_value(
                    &self.config,
                    &fl2sa_params,
                    relu_output,
                    "relu-output",
                )?;

                let term_a2_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_a2, Some(precision))?;
                let term_a4_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_a4, Some(precision))?;
                let term_a6_superacc =
                    superacc_from_value(&self.config, &fl2sa_params, term_a6, Some(precision))?;

                if cfg!(debug_assertions) {
                    let mut carry = 0i128;
                    let radix = 1i128 << self.config.fl2sa_w;
                    for limb_idx in 0..fl2sa_poly.superaccumulator.len() {
                        let mut sum = carry;
                        sum += superacc_a0[limb_idx] as i128;
                        sum += term_a2_superacc[limb_idx] as i128;
                        sum += term_a4_superacc[limb_idx] as i128;
                        sum += term_a6_superacc[limb_idx] as i128;
                        let result = sum.rem_euclid(radix);
                        carry = (sum - result) / radix;
                        let expected = fl2sa_poly.superaccumulator[limb_idx] as i128;
                        if result != expected {
                            println!(
                                "[layer2-debug] limb {} mismatch: sum={} result={} expected={} carry_next={}",
                                limb_idx, sum, result, expected, carry
                            );
                        }
                    }
                    if carry != 0 {
                        println!("[layer2-debug] final carry {}", carry);
                    }
                }

                let poly_relation = build_superacc_linear_combination(
                    vec![
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: superacc_a0.clone(),
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_a2_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_a4_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: term_a6_superacc,
                        },
                        SuperaccLinearTerm {
                            coefficient: -1,
                            limbs: fl2sa_poly.superaccumulator.clone(),
                        },
                    ],
                    linear_radix,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Failed to build ReLU polynomial constraints: {}",
                        e
                    ))
                })?;

                let relu_relation = build_superacc_linear_combination(
                    vec![
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: fl2sa_inputs_vec[idx].superaccumulator.clone(),
                        },
                        SuperaccLinearTerm {
                            coefficient: 1,
                            limbs: fl2sa_poly.superaccumulator.clone(),
                        },
                        SuperaccLinearTerm {
                            coefficient: -2,
                            limbs: fl2sa_relu.superaccumulator.clone(),
                        },
                    ],
                    linear_radix,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!("Failed to build ReLU output constraints: {}", e))
                })?;

                let fl2sa_input_idx = idx;
                let fl2sa_poly_idx = fl2sa_polys_vec.len();
                fl2sa_polys_vec.push(fl2sa_poly);
                let fl2sa_relu_idx = fl2sa_relus_vec.len();
                fl2sa_relus_vec.push(fl2sa_relu);

                entries.push(ReluWitnessEntry {
                    input_value,
                    relu_value: relu_output,
                    poly_value: poly,
                    square_value: input_square,
                    fourth_value: input_fourth,
                    sixth_value: input_sixth,
                    x2_idx,
                    x4_idx,
                    x6_idx,
                    term_a2_idx,
                    term_a4_idx,
                    term_a6_idx,
                    fl2sa_input_idx,
                    fl2sa_poly_idx,
                    fl2sa_relu_idx,
                    poly_relation_witness: poly_relation,
                    relu_relation_witness: relu_relation,
                });
            }
        }
        let batch_size = self.config.batch_size.max(1);
        let mut fl2sa_input_batches = Vec::new();
        let mut fl2sa_poly_batches = Vec::new();
        let mut fl2sa_relu_batches = Vec::new();

        let mut current_chunk = Vec::with_capacity(batch_size);
        for idx in 0..fl2sa_inputs_vec.len() {
            current_chunk.push(idx);
            let is_last = idx + 1 == fl2sa_inputs_vec.len();
            if current_chunk.len() == batch_size || is_last {
                let batch_indices = current_chunk.clone();
                let input_batch = Fl2saBatchWitness::from_indices(
                    &batch_indices,
                    &fl2sa_inputs_vec,
                    &fl2sa_params,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Assembling Layer2 input FL2SA batch witness failed: {}",
                        e
                    ))
                })?;
                let poly_batch = Fl2saBatchWitness::from_indices(
                    &batch_indices,
                    &fl2sa_polys_vec,
                    &fl2sa_params,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Assembling Layer2 polynomial FL2SA batch witness failed: {}",
                        e
                    ))
                })?;
                let relu_batch = Fl2saBatchWitness::from_indices(
                    &batch_indices,
                    &fl2sa_relus_vec,
                    &fl2sa_params,
                )
                .map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Assembling Layer2 ReLU FL2SA batch witness failed: {}",
                        e
                    ))
                })?;

                fl2sa_input_batches.push(input_batch);
                fl2sa_poly_batches.push(poly_batch);
                fl2sa_relu_batches.push(relu_batch);

                current_chunk.clear();
            }
        }

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
                            "Failed to assemble Layer2 MULFP batch witness: {}",
                            e
                        ))
                    })?;
                    mulfp_batches.push(batch);
                    current_indices.clear();
                }
            }
        }

        Ok(Layer2Witness {
            sparse_product: sparse_witness,
            num_nodes,
            hidden_dim,
            entries,
            mulfp_witness,
            mulfp_batches,
            fl2sa_inputs: fl2sa_inputs_vec,
            fl2sa_polys: fl2sa_polys_vec,
            fl2sa_relus: fl2sa_relus_vec,
            fl2sa_input_batches,
            fl2sa_poly_batches,
            fl2sa_relu_batches,
        })
    }

    fn mulfp_witness_from_pair(&self, lhs: f64, rhs: f64) -> Result<MulFPWitness, ZKPError> {
        let input = mulfp_input_from_pair(lhs, rhs, self.config.precision_mode);
        let artifacts = produce_r1cs_mulfp_detached(&self.config.mulfp_params, Some(&input))
            .map_err(ZKPError::MULFPVerificationError)?;

        Ok(artifacts.into_witness(input))
    }
}

pub struct ReluWitnessEntry {
    pub input_value: f64,
    pub relu_value: f64,
    pub poly_value: f64,
    pub square_value: f64,
    pub fourth_value: f64,
    pub sixth_value: f64,
    pub x2_idx: usize,
    pub x4_idx: usize,
    pub x6_idx: usize,
    pub term_a2_idx: usize,
    pub term_a4_idx: usize,
    pub term_a6_idx: usize,
    pub fl2sa_input_idx: usize,
    pub fl2sa_poly_idx: usize,
    pub fl2sa_relu_idx: usize,
    pub poly_relation_witness: LinearCombinationWitness,
    pub relu_relation_witness: LinearCombinationWitness,
}

pub struct Layer2Witness {
    pub sparse_product: SparseMatMulWitness,
    pub num_nodes: usize,
    pub hidden_dim: usize,
    pub entries: Vec<ReluWitnessEntry>,
    pub mulfp_witness: Vec<MulFPWitness>,
    pub mulfp_batches: Vec<MulFPBatchWitness>,
    pub fl2sa_inputs: Vec<Fl2saWitness>,
    pub fl2sa_polys: Vec<Fl2saWitness>,
    pub fl2sa_relus: Vec<Fl2saWitness>,
    pub fl2sa_input_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_poly_batches: Vec<Fl2saBatchWitness>,
    pub fl2sa_relu_batches: Vec<Fl2saBatchWitness>,
}
