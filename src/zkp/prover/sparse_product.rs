use std::mem;
use std::sync::Arc;

use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};
use rand::rngs::OsRng;

use crate::types::{DenseMatrix, FloatType, OperationPrecision, SparseMatrix};
use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};
use crate::zkp::operation_tracker::{record_add, record_mul};
use crate::zkp::prover::fl2sa_utils::{
    derive_fl2sa_params, fl2sa_witness_from_value, mulfp_input_from_pair, superacc_from_value,
};
use crate::zkp::prover::{LinearRelationWitness, MulFPBatchWitness, MulFPWitness};
use crate::zkp::utils::fl2sa::{
    sample_ah_double_shuffle_randomness, AhDoubleShuffleContext, AhDoubleShuffleWitness,
    Fl2saWitness,
};
use crate::zkp::utils::linear::{build_superacc_linear_combination, SuperaccLinearTerm};
use crate::zkp::utils::mulfp::produce_r1cs_mulfp_detached;
use crate::zkp::utils::sparse::{hash_dense_matrix_rows, hash_sparse_matrix};
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};

const ROW_DIFF_MIN_BITS: usize = 1;

pub struct RowDiffWitness {
    pub diffs: Vec<usize>,
    pub bit_len: usize,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub double_shuffle: AhDoubleShuffleWitness,
    pub field_ops: R1csShapeMetrics,
}

pub struct SparseMatMulWitness {
    pub num_rows: usize,
    pub num_cols: usize,
    pub csr_row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub support_commitments: Vec<[u8; 32]>,
    pub adj_hash: [u8; 32],
    pub mulfp_witness: Vec<MulFPWitness>,
    pub mulfp_batches: Vec<MulFPBatchWitness>,
    pub row_relations: Vec<LinearRelationWitness>,
    pub row_diff_witness: Option<RowDiffWitness>,
}

pub struct SparseProductResult {
    pub witness: SparseMatMulWitness,
    pub outputs: Vec<f64>,
    pub output_shape: (usize, usize),
    pub output_fl2sa: Vec<Fl2saWitness>,
}

pub struct SparseProductProver {
    config: Arc<GCNZKPConfig>,
}

impl SparseProductProver {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn prove<T: FloatType>(
        &self,
        adj: &SparseMatrix<T>,
        support: &DenseMatrix<T>,
        bias: Option<&[T]>,
        label: &str,
    ) -> Result<SparseProductResult, ZKPError> {
        let (num_rows, adj_cols) = adj.shape;
        let (support_rows, num_cols) = support.shape;

        if support_rows != adj_cols {
            return Err(ZKPError::DimensionMismatch(format!(
                "{} support dimension mismatch: adj_cols={} vs support_rows={}",
                label, adj_cols, support_rows
            )));
        }

        if bias.map(|b| b.len() != num_cols).unwrap_or(false) {
            return Err(ZKPError::DimensionMismatch(format!(
                "{} bias dimension mismatch: bias_len={} vs cols={}",
                label,
                bias.map(|b| b.len()).unwrap_or(0),
                num_cols
            )));
        }

        let csr = adj.to_csr().map_err(|e| {
            ZKPError::ConfigError(format!("{} Failed to convert CSR: {}", label, e))
        })?;

        let support_commitments = hash_dense_matrix_rows(support)
            .map_err(|e| ZKPError::ConfigError(format!("{} support hash failed: {}", label, e)))?;
        let adj_hash = hash_sparse_matrix(adj)
            .map_err(|e| ZKPError::ConfigError(format!("{} adj hash failed: {}", label, e)))?;

        let precision = self.config.precision_mode;
        let precision_tag: OperationPrecision = precision.into();
        let fl2sa_params = derive_fl2sa_params(&self.config);
        let linear_radix = 1i64 << self.config.fl2sa_w;

        let mut outputs = vec![0.0f64; num_rows * num_cols];
        let mut output_terms: Vec<Vec<SuperaccLinearTerm>> =
            Vec::with_capacity(num_rows * num_cols);
        output_terms.resize_with(num_rows * num_cols, Vec::new);
        let mut mulfp_witness = Vec::new();

        let bias_values: Vec<f64> = bias
            .map(|b| {
                b.iter()
                    .map(|value| {
                        value
                            .to_f64()
                            .ok_or_else(|| ZKPError::NumericalError(0.0))
                            .map(|raw| FloatBitExtractor::quantize_to_precision(raw, precision))
                    })
                    .collect::<Result<Vec<f64>, ZKPError>>()
            })
            .transpose()?
            .unwrap_or_else(|| vec![0.0; num_cols]);

        for row in 0..num_rows {
            let start = csr.row_ptr[row];
            let end = csr.row_ptr[row + 1];

            for edge_idx in start..end {
                let col = csr.col_indices[edge_idx];
                let adj_value = csr.values[edge_idx]
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                let adj_quantized = FloatBitExtractor::quantize_to_precision(adj_value, precision);

                for feature in 0..num_cols {
                    let support_value = support
                        .get(col, feature)
                        .to_f64()
                        .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                    let support_quantized =
                        FloatBitExtractor::quantize_to_precision(support_value, precision);

                    record_mul(precision_tag, 1);
                    let product_value = FloatBitExtractor::quantize_to_precision(
                        adj_quantized * support_quantized,
                        precision,
                    );

                    let input = mulfp_input_from_pair(adj_quantized, support_quantized, precision);
                    let artifacts =
                        produce_r1cs_mulfp_detached(&self.config.mulfp_params, Some(&input))
                            .map_err(ZKPError::MULFPVerificationError)?;
                    mulfp_witness.push(artifacts.into_witness(input));

                    let idx = row * num_cols + feature;
                    record_add(precision_tag, 1);
                    outputs[idx] = FloatBitExtractor::quantize_to_precision(
                        outputs[idx] + product_value,
                        precision,
                    );
                    let product_superacc = superacc_from_value(
                        &self.config,
                        &fl2sa_params,
                        product_value,
                        Some(precision),
                    )?;
                    output_terms[idx].push(SuperaccLinearTerm {
                        coefficient: 1,
                        limbs: product_superacc,
                    });
                }
            }

            for feature in 0..num_cols {
                let bias_val = bias_values[feature];
                if bias_val != 0.0 {
                    let idx = row * num_cols + feature;
                    record_add(precision_tag, 1);
                    outputs[idx] = FloatBitExtractor::quantize_to_precision(
                        outputs[idx] + bias_val,
                        precision,
                    );
                    let bias_superacc = superacc_from_value(
                        &self.config,
                        &fl2sa_params,
                        bias_val,
                        Some(precision),
                    )?;
                    output_terms[idx].push(SuperaccLinearTerm {
                        coefficient: 1,
                        limbs: bias_superacc,
                    });
                }
            }
        }

        let mut row_relations = Vec::with_capacity(outputs.len());
        let mut fl2sa_outputs = Vec::with_capacity(outputs.len());
        for (idx, value) in outputs.iter().enumerate() {
            let fl2sa =
                fl2sa_witness_from_value(&self.config, &fl2sa_params, *value, "sparse-output")?;
            let mut terms = mem::take(&mut output_terms[idx]);
            terms.push(SuperaccLinearTerm {
                coefficient: -1,
                limbs: fl2sa.superaccumulator.clone(),
            });
            let relation = build_superacc_linear_combination(terms, linear_radix).map_err(|e| {
                ZKPError::ConfigError(format!("Failed to build sparse row constraint: {}", e))
            })?;
            row_relations.push(relation);
            fl2sa_outputs.push(fl2sa);
        }

        let mut mulfp_batches = Vec::new();
        let batch_size = self.config.batch_size.max(1);
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
                        "{} Sparse batch witness assembly failed: {}",
                        label, e
                    ))
                })?;
                mulfp_batches.push(batch);
                current_indices.clear();
            }
        }

        let mut stored_values = Vec::with_capacity(csr.values.len());
        for value in &csr.values {
            stored_values.push(
                value
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?,
            );
        }

        Ok(SparseProductResult {
            witness: SparseMatMulWitness {
                num_rows,
                num_cols,
                csr_row_ptr: csr.row_ptr.clone(),
                col_indices: csr.col_indices.clone(),
                values: stored_values,
                support_commitments,
                adj_hash,
                mulfp_witness,
                mulfp_batches,
                row_relations,
                row_diff_witness: build_row_diff_witness(&csr.row_ptr).map_err(|e| {
                    ZKPError::ConfigError(format!(
                        "Failed to build CSR differential constraints: {}",
                        e
                    ))
                })?,
            },
            outputs,
            output_shape: (num_rows, num_cols),
            output_fl2sa: fl2sa_outputs,
        })
    }
}

fn build_row_diff_witness(row_ptr: &[usize]) -> Result<Option<RowDiffWitness>, String> {
    if row_ptr.len() < 2 {
        return Ok(None);
    }

    let diffs: Vec<usize> = row_ptr
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect();
    let max_diff = diffs.iter().copied().max().unwrap_or(0);
    let mut bit_len =
        (usize::BITS as usize - max_diff.leading_zeros() as usize).max(ROW_DIFF_MIN_BITS);
    if bit_len > 63 {
        bit_len = 63;
    }

    let batch = diffs.len();
    let num_bits = batch * bit_len;
    let mut vars = vec![Scalar::ZERO.to_bytes(); num_bits + 2];
    let const_one_idx = num_bits;
    let zero_var_idx = const_one_idx + 1;
    vars[const_one_idx] = Scalar::ONE.to_bytes();

    let mut bit_indices = vec![vec![0usize; bit_len]; batch];
    for (row, diff) in diffs.iter().enumerate() {
        for bit in 0..bit_len {
            let idx = row * bit_len + bit;
            bit_indices[row][bit] = idx;
            let bit_val = if (diff >> bit) & 1 == 1 {
                Scalar::ONE
            } else {
                Scalar::ZERO
            };
            vars[idx] = bit_val.to_bytes();
        }
    }

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let one = Scalar::ONE.to_bytes();

    let mut cursor = 0usize;
    for (row, diff) in diffs.iter().enumerate() {
        let constraint_idx = cursor;
        cursor += 1;
        for bit in 0..bit_len {
            let coeff = 1u128 << bit;
            let scalar = Scalar::from(coeff as u64).to_bytes();
            a_entries.push((constraint_idx, row * bit_len + bit, scalar));
        }
        b_entries.push((constraint_idx, const_one_idx, one));
        let diff_scalar = Scalar::from(*diff as u64).to_bytes();
        c_entries.push((constraint_idx, const_one_idx, diff_scalar));
    }

    let mut rng = OsRng;
    let randomness = sample_ah_double_shuffle_randomness(batch, bit_len, &mut rng);
    let start_idx = vars.len();
    let context = AhDoubleShuffleContext::new(
        batch,
        bit_len,
        start_idx,
        randomness.clone(),
        bit_indices.clone(),
        zero_var_idx,
    )?;
    let layout = context.layout();
    let target_len = layout.next_var_idx().max(vars.len());
    if target_len > vars.len() {
        vars.resize(target_len, Scalar::ZERO.to_bytes());
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
    let inputs_assignment =
        InputsAssignment::new(&[]).map_err(|e| format!("Failed to create input: {:?}", e))?;

    Ok(Some(RowDiffWitness {
        diffs,
        bit_len,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_diff_witness_matches_row_ptr() {
        let row_ptr = vec![0, 2, 5, 7];
        let witness = build_row_diff_witness(&row_ptr)
            .expect("row diff witness should build")
            .expect("witness should exist");
        assert_eq!(witness.diffs, vec![2, 3, 2]);
        assert!(witness
            .instance
            .is_sat(&witness.vars, &witness.inputs)
            .expect("sat check should succeed"));
    }
}
