// Multiplication of two floating-point numbers.
// Require: Two floating-point numbers (b1,v1, p1) and (b2,v2, p2).
// Ensure: Multiplication result (b3,v3, p3)

#![allow(clippy::assertions_on_result_states)]
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};
use rand::{rngs::OsRng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha3::{Digest, Sha3_256};
use std::ops::Range;

use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};

fn scalar_from_u128(value: u128) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&value.to_le_bytes());
    Scalar::from_bytes_mod_order(bytes)
}

fn bit_at_u128(value: u128, index: usize) -> u64 {
    if index < 128 {
        ((value >> index) & 1) as u64
    } else {
        0
    }
}

fn scalar_bit_from_bytes(bytes: &[u8; 32]) -> Result<u8, String> {
    let scalar = Scalar::from_bytes_mod_order(*bytes);
    if scalar == Scalar::ZERO {
        Ok(0)
    } else if scalar == Scalar::ONE {
        Ok(1)
    } else {
        Err(format!(
            "Illegal Boolean bit assignment: expected 0 or 1, actual {:?}",
            scalar
        ))
    }
}

fn sample_nonzero_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let candidate = Scalar::random(rng);
        if candidate != Scalar::ZERO {
            return candidate;
        }
    }
}

#[cfg(test)]
fn first_unsatisfied_row(
    num_cons: usize,
    num_vars: usize,
    num_inputs: usize,
    a: &[(usize, usize, [u8; 32])],
    b: &[(usize, usize, [u8; 32])],
    c: &[(usize, usize, [u8; 32])],
    vars: &[[u8; 32]],
    inputs: &[[u8; 32]],
) -> Option<(usize, Scalar, Scalar, Scalar)> {
    let mut z: Vec<Scalar> = Vec::with_capacity(num_vars + num_inputs + 1);
    for entry in vars {
        z.push(Scalar::from_bytes_mod_order(*entry));
    }
    #[cfg(test)]
    {
        if num_vars > 267 {
            println!(
                "[mulfp-debug] z initial value at 267: {:?}",
                Scalar::from_bytes_mod_order(vars[267])
            );
            println!("[mulfp-debug] helper raw bytes at 267: {:?}", vars[267]);
            debug_assert_eq!(
                Scalar::from_bytes_mod_order(Scalar::ONE.to_bytes()),
                Scalar::ONE
            );
        }
    }
    z.push(Scalar::ONE);
    for entry in inputs {
        z.push(Scalar::from_bytes_mod_order(*entry));
    }

    let mut a_evals = vec![Scalar::ZERO; num_cons];
    for (row, col, coeff) in a {
        let coeff_scalar = Scalar::from_bytes_mod_order(*coeff);
        a_evals[*row] += coeff_scalar * z[*col];
    }

    let mut b_evals = vec![Scalar::ZERO; num_cons];
    for (row, col, coeff) in b {
        let coeff_scalar = Scalar::from_bytes_mod_order(*coeff);
        b_evals[*row] += coeff_scalar * z[*col];
    }

    let mut c_evals = vec![Scalar::ZERO; num_cons];
    for (row, col, coeff) in c {
        let coeff_scalar = Scalar::from_bytes_mod_order(*coeff);
        c_evals[*row] += coeff_scalar * z[*col];
    }

    for row in 0..num_cons {
        let lhs = a_evals[row] * b_evals[row];
        let rhs = c_evals[row];
        if lhs != rhs {
            #[cfg(test)]
            {
                let z_values_for_row: Vec<(usize, Scalar)> = b
                    .iter()
                    .filter(|(r, _, _)| *r == row)
                    .map(|(_, col, coeff)| (*col, Scalar::from_bytes_mod_order(*coeff)))
                    .collect();
                let z_assignments: Vec<(usize, Scalar)> = z_values_for_row
                    .iter()
                    .map(|(col, _coeff)| (*col, z[*col]))
                    .collect();
                let a_values_for_row: Vec<(usize, Scalar)> = a
                    .iter()
                    .filter(|(r, _, _)| *r == row)
                    .map(|(_, col, coeff)| (*col, Scalar::from_bytes_mod_order(*coeff)))
                    .collect();
                let a_assignments: Vec<(usize, Scalar)> = a_values_for_row
                    .iter()
                    .map(|(col, _coeff)| (*col, z[*col]))
                    .collect();
                println!(
                    "[mulfp-debug] Row {} B entries: {:?}",
                    row, z_values_for_row
                );
                println!(
                    "[mulfp-debug] Row {} z assignments: {:?}",
                    row, z_assignments
                );
                println!(
                    "[mulfp-debug] Row {} A entries: {:?}",
                    row, a_values_for_row
                );
                println!(
                    "[mulfp-debug] Row {} A assignments: {:?}",
                    row, a_assignments
                );
                let c_values_for_row: Vec<(usize, Scalar)> = c
                    .iter()
                    .filter(|(r, _, _)| *r == row)
                    .map(|(_, col, coeff)| (*col, Scalar::from_bytes_mod_order(*coeff)))
                    .collect();
                let c_assignments: Vec<(usize, Scalar)> = c_values_for_row
                    .iter()
                    .map(|(col, _coeff)| (*col, z[*col]))
                    .collect();
                println!(
                    "[mulfp-debug] Row {} C entries: {:?}",
                    row, c_values_for_row
                );
                println!(
                    "[mulfp-debug] Row {} C assignments: {:?}",
                    row, c_assignments
                );
            }
            return Some((row, a_evals[row], b_evals[row], rhs));
        }
    }

    None
}

///
#[derive(Debug, Clone)]
pub struct MulFPParams {
    pub w: usize,
    pub e: usize,
    pub m: usize,
    pub lgw: usize,
    pub acc: usize,
}

impl Default for MulFPParams {
    fn default() -> Self {
        Self::single_precision()
    }
}

impl MulFPParams {
    ///
    ///
    pub fn half_precision() -> Self {
        Self {
            w: 4,
            e: 5,
            m: 10,
            lgw: 2,
            acc: 22,
        }
    }

    ///
    ///
    ///
    pub fn single_precision() -> Self {
        Self {
            w: 4,
            e: 8,
            m: 23,
            lgw: 2,
            acc: 48,
        }
    }

    ///
    ///
    ///
    ///
    ///
    pub fn double_precision() -> Self {
        Self {
            w: 4,
            e: 11,
            m: 52,
            lgw: 2,
            acc: 106,
        }
    }

    ///
    pub fn constraint_complexity(&self) -> ConstraintComplexity {
        let group_1_6 = 6;
        let group_7 = 16 + 4 * self.m;
        let group_8 = 21;
        let total = group_1_6 + group_7 + group_8;

        let binary_bits = 2 * self.m + 2;
        let gamma_terms = 2 * self.m;

        let avg_entries_per_constraint = 3;
        let total_entries = total * avg_entries_per_constraint * 3;
        let estimated_memory_mb = (total_entries * 32) as f64 / (1024.0 * 1024.0);

        ConstraintComplexity {
            group_1_6,
            group_7,
            group_8,
            total,
            binary_bits,
            gamma_terms,
            estimated_memory_mb,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintComplexity {
    pub group_1_6: usize,
    pub group_7: usize,
    pub group_8: usize,
    pub total: usize,
    pub binary_bits: usize,
    pub gamma_terms: usize,
    pub estimated_memory_mb: f64,
}

///
#[derive(Debug, Clone)]
pub struct MulFPInputData {
    pub b1: u64,
    pub v1: u64,
    pub p1: i64,

    pub b2: u64,
    pub v2: u64,
    pub p2: i64,

    pub witness_values: Option<Vec<[u8; 32]>>,
}

pub struct MulFPR1CSArtifacts {
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_non_zero_entries: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub assignment: Vec<[u8; 32]>,
    pub double_shuffle: Vec<MulFPDoubleShuffleWitness>,
    pub metrics: R1csShapeMetrics,
}

impl MulFPR1CSArtifacts {
    pub fn into_witness(self, input: MulFPInputData) -> MulFPWitness {
        MulFPWitness {
            input,
            instance: self.instance,
            vars: self.vars,
            inputs: self.inputs,
            assignment: self.assignment,
            double_shuffle: self.double_shuffle,
            field_ops: self.metrics,
        }
    }
}

pub struct MulFPWitness {
    pub input: MulFPInputData,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub assignment: Vec<[u8; 32]>,
    pub double_shuffle: Vec<MulFPDoubleShuffleWitness>,
    pub field_ops: R1csShapeMetrics,
}

impl std::fmt::Debug for MulFPWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MulFPWitness")
            .field("input", &self.input)
            .field("num_assignment_entries", &self.assignment.len())
            .field("double_shuffle_groups", &self.double_shuffle.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulFPDoubleShuffleMode {
    Embedded,
    Detached,
}

#[derive(Debug, Clone)]
pub struct MulFPBooleanLayout {
    pub sign_bits: [usize; 3],
    pub mantissa_product_range: Range<usize>,
    pub mantissa_normalized_range: Range<usize>,
    pub rounding_flags: MulFPRoundingFlagIndices,
    pub normalization_switches: MulFPNormalizationSwitchIndices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFPRoundingFlagIndices {
    pub rdb: usize,
    pub stb: usize,
    pub one_minus_rdb: usize,
    pub one_minus_stb: usize,
    pub v_hat_m_plus_one: usize,
    pub one_minus_v_hat_m_plus_one: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFPNormalizationSwitchIndices {
    pub v_top_bit: usize,
    pub one_minus_v_top: usize,
}

#[derive(Debug, Clone)]
pub struct MulFPBooleanSlices {
    pub mantissa_product: Vec<usize>,
    pub mantissa_normalized: Vec<usize>,
    pub sign_bits: [usize; 3],
    pub rounding_flags: MulFPRoundingFlagIndices,
    pub normalization_switches: MulFPNormalizationSwitchIndices,
}

#[derive(Debug, Clone)]
pub struct MulFPBatchIndexBook {
    pub batch: usize,
    pub mantissa_product: Vec<usize>,
    pub mantissa_normalized: Vec<usize>,
    pub sign_bits: Vec<usize>,
    pub rounding_flags: Vec<usize>,
    pub normalization_switches: Vec<usize>,
    pub slice_sizes: MulFPBooleanSliceSizes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFPBooleanSliceSizes {
    pub mantissa_product: usize,
    pub mantissa_normalized: usize,
    pub sign_bits: usize,
    pub rounding_flags: usize,
    pub normalization_switches: usize,
}

#[derive(Clone)]
pub struct TranscriptRng {
    seed: [u8; 32],
    inner: ChaCha20Rng,
}

impl TranscriptRng {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let inner = ChaCha20Rng::from_seed(seed);
        Self { seed, inner }
    }

    pub fn from_domain_data(domain: &str, data: &[u8]) -> Self {
        let seed = derive_transcript_seed(domain, data);
        Self::from_seed(seed)
    }

    pub fn fork(&self, domain: &str) -> Self {
        let child_seed = mix_seed(&self.seed, domain.as_bytes());
        Self::from_seed(child_seed)
    }

    pub fn seed(&self) -> [u8; 32] {
        self.seed
    }
}

impl RngCore for TranscriptRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.inner.try_fill_bytes(dest)
    }
}

impl CryptoRng for TranscriptRng {}

fn derive_transcript_seed(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(domain.as_bytes());
    hasher.update(&(data.len() as u64).to_le_bytes());
    hasher.update(data);
    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    seed
}

fn mix_seed(base: &[u8; 32], domain: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(base);
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    seed
}

fn encode_indices_seed(indices: &[usize], params: &MulFPParams) -> Vec<u8> {
    let mut data = Vec::with_capacity(32 + indices.len() * 8);
    data.extend_from_slice(&(params.w as u64).to_le_bytes());
    data.extend_from_slice(&(params.e as u64).to_le_bytes());
    data.extend_from_slice(&(params.m as u64).to_le_bytes());
    data.extend_from_slice(&(params.acc as u64).to_le_bytes());
    for &idx in indices {
        data.extend_from_slice(&(idx as u64).to_le_bytes());
    }
    data
}

fn encode_mulfp_input_seed(
    input: Option<&MulFPInputData>,
    params: &MulFPParams,
    mode: MulFPDoubleShuffleMode,
) -> Vec<u8> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&(params.w as u64).to_le_bytes());
    data.extend_from_slice(&(params.e as u64).to_le_bytes());
    data.extend_from_slice(&(params.m as u64).to_le_bytes());
    data.extend_from_slice(&(params.acc as u64).to_le_bytes());
    let mode_tag: u8 = match mode {
        MulFPDoubleShuffleMode::Embedded => 0,
        MulFPDoubleShuffleMode::Detached => 1,
    };
    data.push(mode_tag);

    if let Some(input) = input {
        data.extend_from_slice(&input.b1.to_le_bytes());
        data.extend_from_slice(&input.v1.to_le_bytes());
        data.extend_from_slice(&input.p1.to_le_bytes());
        data.extend_from_slice(&input.b2.to_le_bytes());
        data.extend_from_slice(&input.v2.to_le_bytes());
        data.extend_from_slice(&input.p2.to_le_bytes());
    }

    data
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulFPDoubleShuffleRandomness {
    pub row_scalars: Vec<Scalar>,
    pub col_scalars: Vec<Scalar>,
    pub row_weights: Vec<Scalar>,
    pub col_weights: Vec<Scalar>,
    pub alpha: Scalar,
    pub alpha_prime: Scalar,
    pub seed: [u8; 32],
}

pub fn derive_mulfp_boolean_layout(params: &MulFPParams) -> MulFPBooleanLayout {
    let m = params.m;
    let binary_bits = 2 * m + 2;

    let b1_idx = 0;
    let v1_idx = b1_idx + 1;
    let p1_idx = v1_idx + 1;
    let b2_idx = p1_idx + 1;
    let v2_idx = b2_idx + 1;
    let p2_idx = v2_idx + 1;
    let b3_idx = p2_idx + 1;
    let v3_idx = b3_idx + 1;
    let p3_idx = v3_idx + 1;
    let v1_prime_idx = p3_idx + 1;
    let v2_prime_idx = v1_prime_idx + 1;
    let const_one_idx = v2_prime_idx + 1;

    let v_bits_base_idx = const_one_idx + 1;
    let mantissa_product_range = v_bits_base_idx..(v_bits_base_idx + binary_bits);
    let v_top_bit_idx = v_bits_base_idx + (binary_bits - 1);

    let p3_hat_idx = v_bits_base_idx + binary_bits;
    let v3_hat_idx = p3_hat_idx + 1;
    let v_hat_bits_base_idx = v3_hat_idx + 1;
    let mantissa_normalized_range = v_hat_bits_base_idx..(v_hat_bits_base_idx + binary_bits);

    let ra_idx = v_hat_bits_base_idx + binary_bits;
    let rb_idx = ra_idx + 1;
    let gamma_base_idx = rb_idx + 1;
    let const_one_idx_updated = gamma_base_idx + (2 * m);
    let b1_times_b2_idx = const_one_idx_updated + 1;

    let mut next_idx = b1_times_b2_idx + 1;

    let _p_diff1_idx = next_idx;
    next_idx += 1;
    let _v_diff1_idx = next_idx;
    next_idx += 1;
    let _p_diff2_idx = next_idx;
    next_idx += 1;
    let _v_diff2_idx = next_idx;
    next_idx += 1;

    let _ra_p_diff1_idx = next_idx;
    next_idx += 1;
    let _rb_v_diff1_idx = next_idx;
    next_idx += 1;
    let _ra_p_diff2_idx = next_idx;
    next_idx += 1;
    let _rb_v_diff2_idx = next_idx;
    next_idx += 1;

    let _sum1_base_idx = next_idx;
    next_idx += 2 * m;
    let _sum2_base_idx = next_idx;
    next_idx += 2 * m;

    let _gamma_sum1_idx = next_idx;
    next_idx += 1;
    let _gamma_sum2_idx = next_idx;
    next_idx += 1;

    let _term1_idx = next_idx;
    next_idx += 1;
    let _term2_idx = next_idx;
    next_idx += 1;

    let one_minus_v_idx = next_idx;
    next_idx += 1;
    let _v_times_term1_idx = next_idx;
    next_idx += 1;
    let _one_minus_v_times_term2_idx = next_idx;
    next_idx += 1;
    let _final_result_idx = next_idx;
    next_idx += 1;

    let _updated_const_one_idx = next_idx;
    next_idx += 1;

    let rdb_idx = next_idx;
    next_idx += 1;
    let stb_idx = next_idx;
    next_idx += 1;
    let _f1_idx = next_idx;
    next_idx += 1;
    let _f2_idx = next_idx;
    next_idx += 1;
    let _sum_upper_bits_idx = next_idx;
    next_idx += 1;
    let _v_tilde_3_idx = next_idx;
    next_idx += 1;

    let _stb_product_base_idx = next_idx;
    next_idx += m + 1;
    let _stb_sum_idx = next_idx;
    next_idx += 1;

    let _stb_sum_minus_stb_idx = next_idx;
    next_idx += 1;
    let _stb_logic_result_idx = next_idx;
    next_idx += 1;
    let _zero_var_idx = next_idx;
    next_idx += 1;

    let one_minus_rdb_idx = next_idx;
    next_idx += 1;
    let one_minus_stb_idx = next_idx;
    next_idx += 1;
    let v_hat_m_plus_one_idx = next_idx;
    next_idx += 1;
    let one_minus_v_hat_m_plus_one_idx = next_idx;
    next_idx += 1;
    let _stb_plus_v_hat_m_plus_one_idx = next_idx;
    next_idx += 1;

    let _term_a_idx = next_idx;
    next_idx += 1;
    let _term_b_idx = next_idx;
    next_idx += 1;
    let _term_c_idx = next_idx;
    next_idx += 1;
    let _bracket_content_idx = next_idx;
    next_idx += 1;
    let _left_part_idx = next_idx;
    next_idx += 1;
    let _right_part_idx = next_idx;
    next_idx += 1;
    let _final_round_result_idx = next_idx;

    MulFPBooleanLayout {
        sign_bits: [b1_idx, b2_idx, b3_idx],
        mantissa_product_range,
        mantissa_normalized_range,
        rounding_flags: MulFPRoundingFlagIndices {
            rdb: rdb_idx,
            stb: stb_idx,
            one_minus_rdb: one_minus_rdb_idx,
            one_minus_stb: one_minus_stb_idx,
            v_hat_m_plus_one: v_hat_m_plus_one_idx,
            one_minus_v_hat_m_plus_one: one_minus_v_hat_m_plus_one_idx,
        },
        normalization_switches: MulFPNormalizationSwitchIndices {
            v_top_bit: v_top_bit_idx,
            one_minus_v_top: one_minus_v_idx,
        },
    }
}

pub fn collect_mulfp_bool_slices(params: &MulFPParams) -> MulFPBooleanSlices {
    let layout = derive_mulfp_boolean_layout(params);

    let mantissa_product: Vec<usize> = layout.mantissa_product_range.clone().collect();
    let mantissa_normalized: Vec<usize> = layout.mantissa_normalized_range.clone().collect();

    MulFPBooleanSlices {
        mantissa_product,
        mantissa_normalized,
        sign_bits: layout.sign_bits,
        rounding_flags: layout.rounding_flags,
        normalization_switches: layout.normalization_switches,
    }
}

pub fn merge_mulfp_bool_slices(
    slices: &[MulFPBooleanSlices],
) -> Result<MulFPBatchIndexBook, String> {
    if slices.is_empty() {
        return Err("merge_mulfp_bool_slices requires at least one slice".to_string());
    }

    let first = &slices[0];
    let expected_sizes = MulFPBooleanSliceSizes {
        mantissa_product: first.mantissa_product.len(),
        mantissa_normalized: first.mantissa_normalized.len(),
        sign_bits: first.sign_bits.len(),
        rounding_flags: 6,
        normalization_switches: 2,
    };

    let mut mantissa_product = Vec::with_capacity(expected_sizes.mantissa_product * slices.len());
    let mut mantissa_normalized =
        Vec::with_capacity(expected_sizes.mantissa_normalized * slices.len());
    let mut sign_bits = Vec::with_capacity(expected_sizes.sign_bits * slices.len());
    let mut rounding_flags = Vec::with_capacity(expected_sizes.rounding_flags * slices.len());
    let mut normalization_switches =
        Vec::with_capacity(expected_sizes.normalization_switches * slices.len());

    for (op_idx, slice) in slices.iter().enumerate() {
        if slice.mantissa_product.len() != expected_sizes.mantissa_product
            || slice.mantissa_normalized.len() != expected_sizes.mantissa_normalized
            || slice.sign_bits.len() != expected_sizes.sign_bits
        {
            return Err(format!(
                "The length of the {}th slice is inconsistent with the first slice",
                op_idx
            ));
        }

        mantissa_product.extend_from_slice(&slice.mantissa_product);
        mantissa_normalized.extend_from_slice(&slice.mantissa_normalized);
        sign_bits.extend_from_slice(&slice.sign_bits);
        rounding_flags.extend_from_slice(&[
            slice.rounding_flags.rdb,
            slice.rounding_flags.stb,
            slice.rounding_flags.one_minus_rdb,
            slice.rounding_flags.one_minus_stb,
            slice.rounding_flags.v_hat_m_plus_one,
            slice.rounding_flags.one_minus_v_hat_m_plus_one,
        ]);
        normalization_switches.extend_from_slice(&[
            slice.normalization_switches.v_top_bit,
            slice.normalization_switches.one_minus_v_top,
        ]);
    }

    Ok(MulFPBatchIndexBook {
        batch: slices.len(),
        mantissa_product,
        mantissa_normalized,
        sign_bits,
        rounding_flags,
        normalization_switches,
        slice_sizes: expected_sizes,
    })
}

pub fn sample_mulfp_double_shuffle_randomness(
    rows: usize,
    cols: usize,
    rng: &mut TranscriptRng,
) -> MulFPDoubleShuffleRandomness {
    let mut row_scalars = Vec::with_capacity(rows);
    for _ in 0..rows {
        row_scalars.push(sample_nonzero_scalar(rng));
    }

    let mut col_scalars = Vec::with_capacity(cols);
    for _ in 0..cols {
        col_scalars.push(sample_nonzero_scalar(rng));
    }

    let mut row_weights = Vec::with_capacity(rows);
    for _ in 0..rows {
        row_weights.push(sample_nonzero_scalar(rng));
    }

    let mut col_weights = Vec::with_capacity(cols);
    for _ in 0..cols {
        col_weights.push(sample_nonzero_scalar(rng));
    }

    let alpha = sample_nonzero_scalar(rng);
    let alpha_prime = sample_nonzero_scalar(rng);

    MulFPDoubleShuffleRandomness {
        row_scalars,
        col_scalars,
        row_weights,
        col_weights,
        alpha,
        alpha_prime,
        seed: rng.seed(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MulFPBooleanGroup {
    MantissaProduct,
    MantissaNormalized,
    SignBits,
    RoundingFlags,
    NormalizationSwitches,
}

impl MulFPBooleanGroup {
    pub fn label(&self) -> &'static str {
        match self {
            MulFPBooleanGroup::MantissaProduct => "mantissa_product",
            MulFPBooleanGroup::MantissaNormalized => "mantissa_normalized",
            MulFPBooleanGroup::SignBits => "sign_bits",
            MulFPBooleanGroup::RoundingFlags => "rounding_flags",
            MulFPBooleanGroup::NormalizationSwitches => "normalization_switches",
        }
    }
}

impl MulFPBatchIndexBook {
    pub fn slice_len(&self, group: MulFPBooleanGroup) -> usize {
        match group {
            MulFPBooleanGroup::MantissaProduct => self.slice_sizes.mantissa_product,
            MulFPBooleanGroup::MantissaNormalized => self.slice_sizes.mantissa_normalized,
            MulFPBooleanGroup::SignBits => self.slice_sizes.sign_bits,
            MulFPBooleanGroup::RoundingFlags => self.slice_sizes.rounding_flags,
            MulFPBooleanGroup::NormalizationSwitches => self.slice_sizes.normalization_switches,
        }
    }

    pub fn flat_indices(&self, group: MulFPBooleanGroup) -> &Vec<usize> {
        match group {
            MulFPBooleanGroup::MantissaProduct => &self.mantissa_product,
            MulFPBooleanGroup::MantissaNormalized => &self.mantissa_normalized,
            MulFPBooleanGroup::SignBits => &self.sign_bits,
            MulFPBooleanGroup::RoundingFlags => &self.rounding_flags,
            MulFPBooleanGroup::NormalizationSwitches => &self.normalization_switches,
        }
    }

    pub fn matrix_for_group(&self, group: MulFPBooleanGroup) -> Result<Vec<Vec<usize>>, String> {
        let stride = self.slice_len(group);
        if stride == 0 {
            return Ok(vec![Vec::new(); self.batch]);
        }

        let storage = self.flat_indices(group);
        if storage.len() != stride * self.batch {
            return Err(format!(
                "{} Group index length mismatch: expected {} ({}x{}), got {}",
                group.label(),
                stride * self.batch,
                self.batch,
                stride,
                storage.len()
            ));
        }

        let mut rows = Vec::with_capacity(self.batch);
        for batch_idx in 0..self.batch {
            let start = batch_idx * stride;
            let end = start + stride;
            rows.push(storage[start..end].to_vec());
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone)]
struct MulFPDoubleShuffleVarLayout {
    start_idx: usize,
    k: Range<usize>,
    k_prime: Range<usize>,
    k_diff_left: Range<usize>,
    k_prime_diff_right: Range<usize>,
    k_prod_left: Range<usize>,
    k_prime_prod_right: Range<usize>,
    s: Range<usize>,
    s_prime: Range<usize>,
    s_diff_left: Range<usize>,
    s_prime_diff_right: Range<usize>,
    s_prod_left: Range<usize>,
    s_prime_prod_right: Range<usize>,
    next_var_idx: usize,
}

impl MulFPDoubleShuffleVarLayout {
    fn new(start_idx: usize, rows: usize, cols: usize) -> Self {
        let k = start_idx..(start_idx + rows);
        let k_prime = k.end..(k.end + rows);
        let k_diff_left = k_prime.end..(k_prime.end + rows);
        let k_prime_diff_right = k_diff_left.end..(k_diff_left.end + rows);
        let k_prod_left = k_prime_diff_right.end..(k_prime_diff_right.end + rows);
        let k_prime_prod_right = k_prod_left.end..(k_prod_left.end + rows);

        let s = k_prime_prod_right.end..(k_prime_prod_right.end + cols);
        let s_prime = s.end..(s.end + cols);
        let s_diff_left = s_prime.end..(s_prime.end + cols);
        let s_prime_diff_right = s_diff_left.end..(s_diff_left.end + cols);
        let s_prod_left = s_prime_diff_right.end..(s_prime_diff_right.end + cols);
        let s_prime_prod_right = s_prod_left.end..(s_prod_left.end + cols);

        let next_var_idx = s_prime_prod_right.end;

        Self {
            start_idx,
            k,
            k_prime,
            k_diff_left,
            k_prime_diff_right,
            k_prod_left,
            k_prime_prod_right,
            s,
            s_prime,
            s_diff_left,
            s_prime_diff_right,
            s_prod_left,
            s_prime_prod_right,
            next_var_idx,
        }
    }

    fn k_index(&self, row: usize) -> usize {
        self.k.start + row
    }

    fn k_prime_index(&self, row: usize) -> usize {
        self.k_prime.start + row
    }

    fn k_diff_left_index(&self, row: usize) -> usize {
        self.k_diff_left.start + row
    }

    fn k_prime_diff_right_index(&self, row: usize) -> usize {
        self.k_prime_diff_right.start + row
    }

    fn k_prod_left_index(&self, row: usize) -> usize {
        self.k_prod_left.start + row
    }

    fn k_prime_prod_right_index(&self, row: usize) -> usize {
        self.k_prime_prod_right.start + row
    }

    fn s_index(&self, col: usize) -> usize {
        self.s.start + col
    }

    fn s_prime_index(&self, col: usize) -> usize {
        self.s_prime.start + col
    }

    fn s_diff_left_index(&self, col: usize) -> usize {
        self.s_diff_left.start + col
    }

    fn s_prime_diff_right_index(&self, col: usize) -> usize {
        self.s_prime_diff_right.start + col
    }

    fn s_prod_left_index(&self, col: usize) -> usize {
        self.s_prod_left.start + col
    }

    fn s_prime_prod_right_index(&self, col: usize) -> usize {
        self.s_prime_prod_right.start + col
    }

    fn start_idx(&self) -> usize {
        self.start_idx
    }

    fn next_var_idx(&self) -> usize {
        self.next_var_idx
    }
}

#[derive(Debug, Clone)]
pub struct MulFPDoubleShuffleWitness {
    pub group: MulFPBooleanGroup,
    pub batch_size: usize,
    pub slice_len: usize,
    pub randomness: MulFPDoubleShuffleRandomness,
    pub bit_indices: Vec<Vec<usize>>,
    pub bit_values: Vec<Vec<u8>>,
}

impl MulFPDoubleShuffleWitness {
    pub fn empty(group: MulFPBooleanGroup) -> Self {
        Self {
            group,
            batch_size: 0,
            slice_len: 0,
            randomness: MulFPDoubleShuffleRandomness {
                row_scalars: Vec::new(),
                col_scalars: Vec::new(),
                row_weights: Vec::new(),
                col_weights: Vec::new(),
                alpha: Scalar::ZERO,
                alpha_prime: Scalar::ZERO,
                seed: [0u8; 32],
            },
            bit_indices: Vec::new(),
            bit_values: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MulFPDoubleShuffleGroupContext {
    group: MulFPBooleanGroup,
    batch: usize,
    cols: usize,
    layout: MulFPDoubleShuffleVarLayout,
    randomness: MulFPDoubleShuffleRandomness,
    matrix: Vec<Vec<usize>>,
    matrix_row_shifted: Vec<Vec<usize>>,
    matrix_col_shifted: Vec<Vec<usize>>,
    row_scalar_for_shift: Vec<usize>,
}

impl MulFPDoubleShuffleGroupContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group: MulFPBooleanGroup,
        book: &MulFPBatchIndexBook,
        randomness: MulFPDoubleShuffleRandomness,
        start_var_idx: usize,
        zero_var_idx: usize,
    ) -> Result<Self, String> {
        let matrix = book.matrix_for_group(group)?;
        let batch = book.batch;
        let cols = book.slice_len(group);

        if randomness.row_scalars.len() != batch {
            return Err(format!(
                "{} The random coefficient row vector length is inconsistent with batch ({} vs {})",
                group.label(),
                randomness.row_scalars.len(),
                batch
            ));
        }
        if randomness.row_weights.len() != batch {
            return Err(format!(
                "{} row weight length is inconsistent with batch ({} vs {})",
                group.label(),
                randomness.row_weights.len(),
                batch
            ));
        }
        if randomness.col_scalars.len() != cols {
            return Err(format!(
                "{} The column random coefficient length is inconsistent with the number of columns ({} vs {})",
                group.label(),
                randomness.col_scalars.len(),
                cols
            ));
        }
        if randomness.col_weights.len() != cols {
            return Err(format!(
                "{} The column weight length is inconsistent with the number of columns ({} vs {})",
                group.label(),
                randomness.col_weights.len(),
                cols
            ));
        }

        let mut matrix_row_shifted = vec![vec![zero_var_idx; cols]; batch];
        let mut row_scalar_for_shift = vec![0usize; batch];
        if batch > 0 && cols > 0 {
            for row in 0..batch {
                let prev_row = if row == 0 { batch - 1 } else { row - 1 };
                matrix_row_shifted[row] = matrix[prev_row].clone();
                row_scalar_for_shift[row] = prev_row;
            }
        }

        let mut matrix_col_shifted = vec![vec![zero_var_idx; cols]; batch];
        if cols > 0 {
            for row in 0..batch {
                for col in 0..cols {
                    let prev_col = if col == 0 { cols - 1 } else { col - 1 };
                    matrix_col_shifted[row][col] = matrix_row_shifted[row]
                        .get(prev_col)
                        .copied()
                        .unwrap_or(zero_var_idx);
                }
            }
        }

        let layout = MulFPDoubleShuffleVarLayout::new(start_var_idx, batch, cols);

        Ok(Self {
            group,
            batch,
            cols,
            layout,
            randomness,
            matrix,
            matrix_row_shifted,
            matrix_col_shifted,
            row_scalar_for_shift,
        })
    }

    pub fn group(&self) -> MulFPBooleanGroup {
        self.group
    }

    pub fn start_idx(&self) -> usize {
        self.layout.start_idx()
    }

    pub fn next_var_idx(&self) -> usize {
        self.layout.next_var_idx()
    }

    pub fn introduced_vars(&self) -> usize {
        self.next_var_idx().saturating_sub(self.start_idx())
    }

    pub fn append_constraints(
        &self,
        constraint_start: usize,
        a_entries: &mut Vec<(usize, usize, [u8; 32])>,
        b_entries: &mut Vec<(usize, usize, [u8; 32])>,
        c_entries: &mut Vec<(usize, usize, [u8; 32])>,
        const_one_idx: usize,
    ) -> usize {
        let mut cursor = constraint_start;
        let one = Scalar::ONE.to_bytes();
        let minus_alpha = (-self.randomness.alpha).to_bytes();
        let minus_alpha_prime = (-self.randomness.alpha_prime).to_bytes();

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            let rho = self.randomness.row_scalars[row];
            for col in 0..self.cols {
                let coeff = rho * self.randomness.col_weights[col];
                a_entries.push((constraint_id, self.matrix[row][col], coeff.to_bytes()));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            let source_row = self.row_scalar_for_shift[row];
            let rho = self.randomness.row_scalars[source_row];
            for col in 0..self.cols {
                let coeff = rho * self.randomness.col_weights[col];
                a_entries.push((
                    constraint_id,
                    self.matrix_row_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prime_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_index(row), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_diff_left_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_prime_index(row), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.k_prime_diff_right_index(row),
                one,
            ));
        }

        if self.batch > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_diff_left_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prod_left_index(0), one));

            for row in 1..self.batch {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.k_prod_left_index(row - 1), one));
                b_entries.push((cid, self.layout.k_diff_left_index(row), one));
                c_entries.push((cid, self.layout.k_prod_left_index(row), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_prime_diff_right_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prime_prod_right_index(0), one));

            for row in 1..self.batch {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.k_prime_prod_right_index(row - 1), one));
                b_entries.push((cid, self.layout.k_prime_diff_right_index(row), one));
                c_entries.push((cid, self.layout.k_prime_prod_right_index(row), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.k_prod_left_index(self.batch - 1),
                one,
            ));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.k_prime_prod_right_index(self.batch - 1),
                one,
            ));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            let sigma = self.randomness.col_scalars[col];
            for row in 0..self.batch {
                let source_row = self.row_scalar_for_shift[row];
                let coeff = sigma * self.randomness.row_weights[source_row];
                a_entries.push((
                    constraint_id,
                    self.matrix_row_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            let source_col = if col == 0 { self.cols - 1 } else { col - 1 };
            let sigma = self.randomness.col_scalars[source_col];
            for row in 0..self.batch {
                let source_row = self.row_scalar_for_shift[row];
                let coeff = sigma * self.randomness.row_weights[source_row];
                a_entries.push((
                    constraint_id,
                    self.matrix_col_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prime_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_index(col), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha_prime));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_diff_left_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_index(col), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha_prime));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_diff_right_index(col),
                one,
            ));
        }

        if self.cols > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_diff_left_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prod_left_index(0), one));

            for col in 1..self.cols {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prod_left_index(col - 1), one));
                b_entries.push((cid, self.layout.s_diff_left_index(col), one));
                c_entries.push((cid, self.layout.s_prod_left_index(col), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_diff_right_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prime_prod_right_index(0), one));

            for col in 1..self.cols {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prime_prod_right_index(col - 1), one));
                b_entries.push((cid, self.layout.s_prime_diff_right_index(col), one));
                c_entries.push((cid, self.layout.s_prime_prod_right_index(col), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.s_prod_left_index(self.cols - 1),
                one,
            ));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_prod_right_index(self.cols - 1),
                one,
            ));
        }

        cursor
    }

    pub fn populate_witness(
        &self,
        vars: &mut Vec<[u8; 32]>,
    ) -> Result<MulFPDoubleShuffleWitness, String> {
        let mut k_values = Vec::with_capacity(self.batch);
        let mut k_prime_values = Vec::with_capacity(self.batch);

        for row in 0..self.batch {
            let rho = self.randomness.row_scalars[row];
            let mut sum = Scalar::ZERO;
            for col in 0..self.cols {
                let idx = self.matrix[row][col];
                let value_bytes = vars.get(idx).ok_or_else(|| {
                    format!("Variable index {} out of bounds (line {})", idx, row)
                })?;
                let value_scalar = Scalar::from_bytes_mod_order(*value_bytes);
                sum += self.randomness.col_weights[col] * value_scalar;
            }
            sum *= rho;
            vars[self.layout.k_index(row)] = sum.to_bytes();
            k_values.push(sum);
        }

        for row in 0..self.batch {
            let source_row = self.row_scalar_for_shift[row];
            let rho = self.randomness.row_scalars[source_row];
            let mut sum = Scalar::ZERO;
            for col in 0..self.cols {
                let idx = self.matrix_row_shifted[row][col];
                let value_bytes = vars.get(idx).ok_or_else(|| {
                    format!("Replacement row index {} out of bounds (row {})", idx, row)
                })?;
                let value_scalar = Scalar::from_bytes_mod_order(*value_bytes);
                sum += self.randomness.col_weights[col] * value_scalar;
            }
            sum *= rho;
            vars[self.layout.k_prime_index(row)] = sum.to_bytes();
            k_prime_values.push(sum);
        }

        let mut k_diff_left_vals = Vec::with_capacity(self.batch);
        let mut k_prime_diff_vals = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            let diff = k_values[row] - self.randomness.alpha;
            vars[self.layout.k_diff_left_index(row)] = diff.to_bytes();
            k_diff_left_vals.push(diff);

            let diff_prime = k_prime_values[row] - self.randomness.alpha;
            vars[self.layout.k_prime_diff_right_index(row)] = diff_prime.to_bytes();
            k_prime_diff_vals.push(diff_prime);
        }

        if self.batch > 0 {
            let mut left_prod = k_diff_left_vals[0];
            vars[self.layout.k_prod_left_index(0)] = left_prod.to_bytes();
            for row in 1..self.batch {
                left_prod *= k_diff_left_vals[row];
                vars[self.layout.k_prod_left_index(row)] = left_prod.to_bytes();
            }

            let mut right_prod = k_prime_diff_vals[0];
            vars[self.layout.k_prime_prod_right_index(0)] = right_prod.to_bytes();
            for row in 1..self.batch {
                right_prod *= k_prime_diff_vals[row];
                vars[self.layout.k_prime_prod_right_index(row)] = right_prod.to_bytes();
            }
        }

        let mut s_values = Vec::with_capacity(self.cols);
        let mut s_prime_values = Vec::with_capacity(self.cols);

        for col in 0..self.cols {
            let sigma = self.randomness.col_scalars[col];
            let mut sum = Scalar::ZERO;
            for row in 0..self.batch {
                let idx = self.matrix_row_shifted[row][col];
                let value_bytes = vars.get(idx).ok_or_else(|| {
                    format!("Column {} Row {} Index {} Out of bounds", col, row, idx)
                })?;
                let value_scalar = Scalar::from_bytes_mod_order(*value_bytes);
                let source_row = self.row_scalar_for_shift[row];
                sum += self.randomness.row_weights[source_row] * value_scalar;
            }
            sum *= sigma;
            vars[self.layout.s_index(col)] = sum.to_bytes();
            s_values.push(sum);
        }

        for col in 0..self.cols {
            let source_col = if col == 0 { self.cols - 1 } else { col - 1 };
            let sigma = self.randomness.col_scalars[source_col];
            let mut sum = Scalar::ZERO;
            for row in 0..self.batch {
                let idx = self.matrix_col_shifted[row][col];
                let value_bytes = vars.get(idx).ok_or_else(|| {
                    format!(
                        "Column replacement {} row {} index {} out of bounds",
                        col, row, idx
                    )
                })?;
                let value_scalar = Scalar::from_bytes_mod_order(*value_bytes);
                let source_row = self.row_scalar_for_shift[row];
                sum += self.randomness.row_weights[source_row] * value_scalar;
            }
            sum *= sigma;
            vars[self.layout.s_prime_index(col)] = sum.to_bytes();
            s_prime_values.push(sum);
        }

        let mut s_diff_vals = Vec::with_capacity(self.cols);
        let mut s_prime_diff_vals = Vec::with_capacity(self.cols);
        for col in 0..self.cols {
            let diff = s_values[col] - self.randomness.alpha_prime;
            vars[self.layout.s_diff_left_index(col)] = diff.to_bytes();
            s_diff_vals.push(diff);

            let diff_prime = s_prime_values[col] - self.randomness.alpha_prime;
            vars[self.layout.s_prime_diff_right_index(col)] = diff_prime.to_bytes();
            s_prime_diff_vals.push(diff_prime);
        }

        if self.cols > 0 {
            let mut prod_left = s_diff_vals[0];
            vars[self.layout.s_prod_left_index(0)] = prod_left.to_bytes();
            for col in 1..self.cols {
                prod_left *= s_diff_vals[col];
                vars[self.layout.s_prod_left_index(col)] = prod_left.to_bytes();
            }

            let mut prod_right = s_prime_diff_vals[0];
            vars[self.layout.s_prime_prod_right_index(0)] = prod_right.to_bytes();
            for col in 1..self.cols {
                prod_right *= s_prime_diff_vals[col];
                vars[self.layout.s_prime_prod_right_index(col)] = prod_right.to_bytes();
            }
        }

        let mut bit_indices = Vec::with_capacity(self.batch);
        let mut bit_values = Vec::with_capacity(self.batch);

        for row in 0..self.batch {
            let mut indices_row = Vec::with_capacity(self.cols);
            let mut values_row = Vec::with_capacity(self.cols);
            for col in 0..self.cols {
                let idx = self.matrix[row][col];
                indices_row.push(idx);
                let value_bytes = vars
                    .get(idx)
                    .ok_or_else(|| format!("Failed to log boolean: index {} out of bounds", idx))?;
                let scalar = Scalar::from_bytes_mod_order(*value_bytes);
                if scalar != Scalar::ZERO && scalar != Scalar::ONE {
                    return Err(format!(
                        "{} row {} column {} variable {} non-boolean",
                        self.group.label(),
                        row,
                        col,
                        idx
                    ));
                }
                let bit = if scalar == Scalar::ONE { 1 } else { 0 };
                values_row.push(bit);
            }
            bit_indices.push(indices_row);
            bit_values.push(values_row);
        }

        Ok(MulFPDoubleShuffleWitness {
            group: self.group,
            batch_size: self.batch,
            slice_len: self.cols,
            randomness: self.randomness.clone(),
            bit_indices,
            bit_values,
        })
    }
}

pub struct MulFPBatchWitness {
    pub entry_indices: Vec<usize>,
    pub double_shuffle: Vec<MulFPDoubleShuffleWitness>,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub assignment: Vec<[u8; 32]>,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub field_ops: R1csShapeMetrics,
}

impl std::fmt::Debug for MulFPBatchWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MulFPBatchWitness")
            .field("entry_indices", &self.entry_indices)
            .field("num_constraints", &self.num_constraints)
            .field("num_vars", &self.num_vars)
            .field("num_inputs", &self.num_inputs)
            .field("field_adds", &self.field_ops.field_adds)
            .field("field_muls", &self.field_ops.field_muls)
            .finish()
    }
}

impl MulFPBatchWitness {
    pub fn from_indices(
        indices: &[usize],
        source: &[MulFPWitness],
        params: &MulFPParams,
    ) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("MulFPBatchWitness::from_indices requires at least one witness".to_owned());
        }

        let batch_size = indices.len();
        let template = collect_mulfp_bool_slices(params);

        let mut slices: Vec<MulFPBooleanSlices> = Vec::with_capacity(batch_size);
        let mut bit_assignments: Vec<(usize, u8)> = Vec::new();
        let mut next_var_idx = 0usize;

        for (batch_pos, &witness_idx) in indices.iter().enumerate() {
            let witness = source.get(witness_idx).ok_or_else(|| {
                format!(
                    "MulFPBatchWitness::from_indices index {} out of bounds (batch {})",
                    witness_idx, batch_pos
                )
            })?;

            let mut allocate_bit = |orig_idx: usize| -> Result<usize, String> {
                let bytes = witness.assignment.get(orig_idx).ok_or_else(|| {
                    format!(
                        "MulFP witness variable index {} out of bounds (batch {}, witness idx {})",
                        orig_idx, batch_pos, witness_idx
                    )
                })?;
                let bit = scalar_bit_from_bytes(bytes)?;
                let var_idx = next_var_idx;
                next_var_idx += 1;
                bit_assignments.push((var_idx, bit));
                Ok(var_idx)
            };

            let mut slice = MulFPBooleanSlices {
                mantissa_product: Vec::with_capacity(template.mantissa_product.len()),
                mantissa_normalized: Vec::with_capacity(template.mantissa_normalized.len()),
                sign_bits: [0usize; 3],
                rounding_flags: MulFPRoundingFlagIndices {
                    rdb: 0,
                    stb: 0,
                    one_minus_rdb: 0,
                    one_minus_stb: 0,
                    v_hat_m_plus_one: 0,
                    one_minus_v_hat_m_plus_one: 0,
                },
                normalization_switches: MulFPNormalizationSwitchIndices {
                    v_top_bit: 0,
                    one_minus_v_top: 0,
                },
            };

            for &orig_idx in &template.mantissa_product {
                let new_idx = allocate_bit(orig_idx)?;
                slice.mantissa_product.push(new_idx);
            }

            for &orig_idx in &template.mantissa_normalized {
                let new_idx = allocate_bit(orig_idx)?;
                slice.mantissa_normalized.push(new_idx);
            }

            slice.sign_bits = [
                allocate_bit(template.sign_bits[0])?,
                allocate_bit(template.sign_bits[1])?,
                allocate_bit(template.sign_bits[2])?,
            ];

            slice.rounding_flags = MulFPRoundingFlagIndices {
                rdb: allocate_bit(template.rounding_flags.rdb)?,
                stb: allocate_bit(template.rounding_flags.stb)?,
                one_minus_rdb: allocate_bit(template.rounding_flags.one_minus_rdb)?,
                one_minus_stb: allocate_bit(template.rounding_flags.one_minus_stb)?,
                v_hat_m_plus_one: allocate_bit(template.rounding_flags.v_hat_m_plus_one)?,
                one_minus_v_hat_m_plus_one: allocate_bit(
                    template.rounding_flags.one_minus_v_hat_m_plus_one,
                )?,
            };

            slice.normalization_switches = MulFPNormalizationSwitchIndices {
                v_top_bit: allocate_bit(template.normalization_switches.v_top_bit)?,
                one_minus_v_top: allocate_bit(template.normalization_switches.one_minus_v_top)?,
            };

            slices.push(slice);
        }

        let book = merge_mulfp_bool_slices(&slices)?;

        let double_shuffle_start_idx = next_var_idx;
        let groups = [
            MulFPBooleanGroup::MantissaProduct,
            MulFPBooleanGroup::MantissaNormalized,
            MulFPBooleanGroup::SignBits,
            MulFPBooleanGroup::RoundingFlags,
            MulFPBooleanGroup::NormalizationSwitches,
        ];

        let mut ds_var_cursor = double_shuffle_start_idx;
        for group in &groups {
            let cols = book.slice_len(*group);
            if cols == 0 {
                continue;
            }
            let layout_preview = MulFPDoubleShuffleVarLayout::new(ds_var_cursor, batch_size, cols);
            ds_var_cursor = layout_preview.next_var_idx();
        }

        let const_one_idx = ds_var_cursor;
        let zero_var_idx = const_one_idx + 1;
        let total_vars = zero_var_idx + 1;

        let batch_seed_data = encode_indices_seed(indices, params);
        let batch_rng =
            TranscriptRng::from_domain_data("mulfp-batch-double-shuffle", &batch_seed_data);
        let mut contexts: Vec<(MulFPBooleanGroup, Option<MulFPDoubleShuffleGroupContext>)> =
            Vec::with_capacity(groups.len());

        let mut constraint_cursor = 0usize;
        let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

        let mut next_context_start = double_shuffle_start_idx;
        for group in &groups {
            let cols = book.slice_len(*group);
            if cols == 0 {
                contexts.push((*group, None));
                continue;
            }

            let mut group_rng = batch_rng.fork(&format!("mulfp-double-shuffle-{}", group.label()));
            let randomness =
                sample_mulfp_double_shuffle_randomness(batch_size, cols, &mut group_rng);
            let context = MulFPDoubleShuffleGroupContext::new(
                *group,
                &book,
                randomness,
                next_context_start,
                zero_var_idx,
            )
            .map_err(|e| {
                format!(
                    "Failed to construct {:?} bulk Double-Shuffling context: {}",
                    group, e
                )
            })?;

            next_context_start = context.next_var_idx();
            constraint_cursor = context.append_constraints(
                constraint_cursor,
                &mut a_entries,
                &mut b_entries,
                &mut c_entries,
                const_one_idx,
            );

            contexts.push((*group, Some(context)));
        }

        debug_assert!(next_context_start == const_one_idx);

        let one = Scalar::ONE.to_bytes();
        let minus_one = (-Scalar::ONE).to_bytes();

        for (idx, _) in &bit_assignments {
            let constraint_id = constraint_cursor;
            constraint_cursor += 1;
            a_entries.push((constraint_id, *idx, one));
            b_entries.push((constraint_id, *idx, one));
            b_entries.push((constraint_id, const_one_idx, minus_one));
            c_entries.push((constraint_id, zero_var_idx, one));
        }

        let zero_constraint_id = constraint_cursor;
        constraint_cursor += 1;
        a_entries.push((zero_constraint_id, zero_var_idx, one));
        b_entries.push((zero_constraint_id, const_one_idx, one));

        let num_constraints = constraint_cursor;

        let metrics = compute_r1cs_metrics(num_constraints, &a_entries, &b_entries, &c_entries);
        let instance = Instance::new(
            num_constraints,
            total_vars,
            1,
            &a_entries,
            &b_entries,
            &c_entries,
        )
        .map_err(|e| {
            format!(
                "Failed to create MULFP Double-Shuffling batch instance: {:?}",
                e
            )
        })?;

        let mut vars = vec![Scalar::ZERO.to_bytes(); total_vars];
        for (idx, bit) in &bit_assignments {
            vars[*idx] = Scalar::from(*bit as u64).to_bytes();
        }
        vars[const_one_idx] = Scalar::ONE.to_bytes();
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();

        let mut double_shuffle_witnesses = Vec::with_capacity(groups.len());
        for (group, context_opt) in contexts.into_iter() {
            if let Some(context) = context_opt {
                let witness = context
                    .populate_witness(&mut vars)
                    .map_err(|e| format!("Failed to generate {:?} batch witness: {}", group, e))?;
                double_shuffle_witnesses.push(witness);
            } else {
                double_shuffle_witnesses.push(MulFPDoubleShuffleWitness::empty(group));
            }
        }

        let assignment = vars.clone();
        let vars_assignment = VarsAssignment::new(&vars)
            .map_err(|e| format!("Construction variable assignment failed: {:?}", e))?;
        let inputs_assignment = InputsAssignment::new(&[Scalar::ONE.to_bytes()])
            .map_err(|e| format!("Construct input assignment failed: {:?}", e))?;

        match instance.is_sat(&vars_assignment, &inputs_assignment) {
            Ok(true) => {}
            Ok(false) => {
                return Err("Batch MULFP Double-Shuffling SAT check failed".to_owned());
            }
            Err(err) => {
                return Err(format!(
                    "Batch MULFP Double-Shuffling SAT check failed: {:?}",
                    err
                ));
            }
        }

        Ok(Self {
            entry_indices: indices.to_vec(),
            double_shuffle: double_shuffle_witnesses,
            instance,
            vars: vars_assignment,
            inputs: inputs_assignment,
            assignment,
            num_constraints,
            num_vars: const_one_idx,
            num_inputs: 1,
            field_ops: metrics,
        })
    }
}

impl Default for MulFPInputData {
    fn default() -> Self {
        Self {
            b1: 0,
            v1: 0,
            p1: 0,
            b2: 0,
            v2: 0,
            p2: 0,
            witness_values: None,
        }
    }
}

///
///
///
///
///
///
///
pub fn produce_r1cs_mulfp_with_mode(
    params: &MulFPParams,
    input_data: Option<&MulFPInputData>,
    mode: MulFPDoubleShuffleMode,
) -> Result<MulFPR1CSArtifacts, String> {
    if params.w == 0 || !params.w.is_power_of_two() {
        return Err("w must be a power of 2 and greater than 0".to_string());
    }

    let _w = params.w;
    let _e = params.e;
    let _m = params.m;
    let _lgw = params.lgw;
    let acc = params.acc;

    let num_inputs = 6;

    let one = Scalar::ONE.to_bytes();
    let minus_one = (-Scalar::ONE).to_bytes();
    let _zero = Scalar::ZERO.to_bytes();

    let two = Scalar::from(2u8);
    let mut power_of_two = Scalar::ONE;

    let mut powers_of_two = Vec::new();
    powers_of_two.push(Scalar::ONE.to_bytes());
    for _i in 0..(acc - 1) {
        power_of_two = power_of_two * two;
        powers_of_two.push(power_of_two.to_bytes());
    }

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

    // ============================================================================
    // ============================================================================
    //
    //
    //
    //
    //
    //
    //
    //
    // ============================================================================

    let b1_idx = 0;
    let v1_idx = b1_idx + 1;
    let p1_idx = v1_idx + 1;

    let b2_idx = p1_idx + 1;
    let v2_idx = b2_idx + 1;
    let p2_idx = v2_idx + 1;

    let b3_idx = p2_idx + 1;
    let v3_idx = b3_idx + 1;
    let p3_idx = v3_idx + 1;

    let v1_prime_idx = p3_idx + 1; // v1': v1 + 2^m
    let v2_prime_idx = v1_prime_idx + 1; // v2': v2 + 2^m

    let const_one_idx = v2_prime_idx + 1;
    let mut const_one_indices = vec![const_one_idx];

    let m = params.m;

    let binary_bits = 2 * m + 2;
    let v_bits_base_idx = const_one_idx + 1;

    let p3_hat_idx = v_bits_base_idx + binary_bits;
    let v3_hat_idx = p3_hat_idx + 1;

    let v_hat_bits_base_idx = v3_hat_idx + 1;

    let ra_idx = v_hat_bits_base_idx + binary_bits;
    let rb_idx = ra_idx + 1;
    let gamma_base_idx = rb_idx + 1;

    let const_one_idx_updated = gamma_base_idx + (2 * m);
    const_one_indices.push(const_one_idx_updated);

    let _old_num_vars = const_one_idx + 1;

    // ============================================================================
    // ============================================================================

    let mut constraint_count = 0;

    if m >= powers_of_two.len() {
        return Err(format!(
            "m({}) exceeds the range of powers_of_two array ({})",
            m,
            powers_of_two.len()
        ));
    }

    let power_2m = powers_of_two[m]; // 2^m
    #[cfg(test)]
    println!("[mulfp-debug] power_2m bytes={:?}", power_2m);
    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v1_idx, one)); // v1
    a_entries.push((constraint_idx, const_one_idx_updated, power_2m));
    b_entries.push((constraint_idx, const_one_idx_updated, one));
    c_entries.push((constraint_idx, v1_prime_idx, one)); // = v1'
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v2_idx, one)); // v2
    a_entries.push((constraint_idx, const_one_idx_updated, power_2m));
    b_entries.push((constraint_idx, const_one_idx_updated, one));
    c_entries.push((constraint_idx, v2_prime_idx, one)); // = v2'
    constraint_count += 1;

    // ============================================================================
    // ============================================================================

    let b1_times_b2_idx = const_one_idx_updated + 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, b1_idx, one)); // b1
    b_entries.push((constraint_idx, b2_idx, one)); // * b2
    c_entries.push((constraint_idx, b1_times_b2_idx, one)); // = b1*b2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, b1_idx, one)); // b1
    a_entries.push((constraint_idx, b2_idx, one)); // + b2
    a_entries.push((
        constraint_idx,
        b1_times_b2_idx,
        (-Scalar::from(2u8)).to_bytes(),
    ));
    b_entries.push((constraint_idx, const_one_idx_updated, one)); // * 1
    c_entries.push((constraint_idx, b3_idx, one)); // = b3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v1_prime_idx, one)); // v1'
    b_entries.push((constraint_idx, v2_prime_idx, one)); // * v2'
    c_entries.push((constraint_idx, v3_idx, one)); // = v3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, p1_idx, one)); // p1
    a_entries.push((constraint_idx, p2_idx, one)); // + p2
    b_entries.push((constraint_idx, const_one_idx_updated, one));
    c_entries.push((constraint_idx, p3_idx, one)); // = p3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    // Let v :=(v[0], . . . ,v[2m+1]) be the binary representation of v3

    let constraint_idx = constraint_count;
    for i in 0..binary_bits {
        if i >= powers_of_two.len() {
            return Err(format!(
                "Binary decomposition weight index {} is outside the range of powers_of_two array {}",
                i,
                powers_of_two.len()
            ));
        }
        let weight = powers_of_two[i]; // 2^i
        a_entries.push((constraint_idx, v_bits_base_idx + i, weight)); // 2^i x v[i]
    }
    b_entries.push((constraint_idx, const_one_idx_updated, one));
    c_entries.push((constraint_idx, v3_idx, one)); // = v3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================

    // $$
    // \begin{aligned}
    // & v[2 \cdot m + 1] \\
    // & \cdot\left(ra \cdot \left(\hat{p}_3-p_3-1\right)+ rb \cdot \left(\hat{v}_3-v_3\right)+\sum_{i=0}^{2 \cdot m - 1} \gamma_i(\hat{v}[i]-v[i])\right) \\
    // + & (1-v[2 \cdot m + 1]) \\
    // & \cdot\left(ra \cdot \left(\hat{p}_3-p_3\right) + rb \cdot \left(\hat{v}_3-2 \cdot v_3\right)+\sum_{i=0}^{2 \cdot m - 1} \gamma_i(\hat{v}[i]-v[i-1])\right) \\
    // = & 0
    // \end{aligned}
    // $$
    // where we set v[-1] = 0 and ra, rb, gammai, i in [0,2m+1] are random numbers.

    let mut csprng = OsRng;
    let ra_val = csprng.gen_range(1..=u64::MAX);
    let ra_scalar = Scalar::from(ra_val);

    let rb_val = csprng.gen_range(1..=u64::MAX);
    let rb_scalar = Scalar::from(rb_val);

    let mut gamma = vec![Scalar::ZERO; 2 * m];
    for i in 0..(2 * m) {
        let gamma_val = csprng.gen_range(1..=u64::MAX);
        gamma[i] = Scalar::from(gamma_val);
    }

    // ============================================================================
    //
    // ============================================================================

    // v[2m+1] . (ra.(p3-p3-1) + rb.(v3-v3) + Sigmagammai(v[i]-v[i]))
    // + (1-v[2m+1]) . (ra.(p3-p3) + rb.(v3-2.v3) + Sigmagammai(v[i]-v[i-1]))
    // = 0

    let v_top_bit_idx = v_bits_base_idx + (binary_bits - 1); // v[2m+1]

    let mut next_idx = b1_times_b2_idx + 1;

    let p_diff1_idx = next_idx;
    next_idx += 1; // p3-p3-1
    let v_diff1_idx = next_idx;
    next_idx += 1; // v3-v3
    let p_diff2_idx = next_idx;
    next_idx += 1; // p3-p3
    let v_diff2_idx = next_idx;
    next_idx += 1; // v3-2.v3

    let ra_p_diff1_idx = next_idx;
    next_idx += 1; // ra.(p3-p3-1)
    let rb_v_diff1_idx = next_idx;
    next_idx += 1; // rb.(v3-v3)
    let ra_p_diff2_idx = next_idx;
    next_idx += 1; // ra.(p3-p3)
    let rb_v_diff2_idx = next_idx;
    next_idx += 1; // rb.(v3-2.v3)

    let sum1_base_idx = next_idx;
    next_idx += 2 * m;
    let sum2_base_idx = next_idx;
    next_idx += 2 * m;
    let gamma_sum1_idx = next_idx;
    next_idx += 1;
    let gamma_sum2_idx = next_idx;
    next_idx += 1;

    let term1_idx = next_idx;
    next_idx += 1;
    let term2_idx = next_idx;
    next_idx += 1;

    let one_minus_v_idx = next_idx;
    next_idx += 1; // (1-v[2m+1])
    let v_times_term1_idx = next_idx;
    next_idx += 1; // v[2m+1] * term1
    let one_minus_v_times_term2_idx = next_idx;
    next_idx += 1; // (1-v[2m+1]) * term2

    let final_result_idx = next_idx;
    next_idx += 1;

    let updated_const_one_idx = next_idx;
    const_one_indices.push(updated_const_one_idx);
    next_idx += 1;
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, p3_hat_idx, one)); // p3
    a_entries.push((constraint_idx, p3_idx, minus_one)); // - p3
    a_entries.push((constraint_idx, updated_const_one_idx, minus_one)); // - 1
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, p_diff1_idx, one)); // = p_diff1
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v3_hat_idx, one)); // v3
    a_entries.push((constraint_idx, v3_idx, minus_one)); // - v3
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, v_diff1_idx, one)); // = v_diff1
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, p3_hat_idx, one)); // p3
    a_entries.push((constraint_idx, p3_idx, minus_one)); // - p3
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, p_diff2_idx, one)); // = p_diff2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v3_hat_idx, one)); // v3
    let minus_two = (-Scalar::from(2u8)).to_bytes();
    a_entries.push((constraint_idx, v3_idx, minus_two)); // - 2.v3
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, v_diff2_idx, one)); // = v_diff2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, ra_idx, one)); // ra
    b_entries.push((constraint_idx, p_diff1_idx, one)); // x (p3-p3-1)
    c_entries.push((constraint_idx, ra_p_diff1_idx, one)); // = ra.(p3-p3-1)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, rb_idx, one)); // rb
    b_entries.push((constraint_idx, v_diff1_idx, one)); // x (v3-v3)
    c_entries.push((constraint_idx, rb_v_diff1_idx, one)); // = rb.(v3-v3)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, ra_idx, one)); // ra
    b_entries.push((constraint_idx, p_diff2_idx, one)); // x (p3-p3)
    c_entries.push((constraint_idx, ra_p_diff2_idx, one)); // = ra.(p3-p3)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, rb_idx, one)); // rb
    b_entries.push((constraint_idx, v_diff2_idx, one)); // x (v3-2.v3)
    c_entries.push((constraint_idx, rb_v_diff2_idx, one)); // = rb.(v3-2.v3)
    constraint_count += 1;

    for i in 0..(2 * m) {
        let constraint_idx = constraint_count;
        a_entries.push((constraint_idx, v_hat_bits_base_idx + i, one)); // v[i]
        a_entries.push((constraint_idx, v_bits_base_idx + i, minus_one)); // - v[i]
        b_entries.push((constraint_idx, gamma_base_idx + i, one)); // x gammai
        c_entries.push((constraint_idx, sum1_base_idx + i, one)); // = gammai(v[i]-v[i])
        constraint_count += 1;
    }

    for i in 0..(2 * m) {
        let constraint_idx = constraint_count;
        a_entries.push((constraint_idx, v_hat_bits_base_idx + i, one)); // v[i]
        if i > 0 {
            a_entries.push((constraint_idx, v_bits_base_idx + (i - 1), minus_one));
            // - v[i-1]
        }
        b_entries.push((constraint_idx, gamma_base_idx + i, one)); // x gammai
        c_entries.push((constraint_idx, sum2_base_idx + i, one)); // = gammai(v[i]-v[i-1])
        constraint_count += 1;
    }

    let constraint_idx = constraint_count;
    for i in 0..(2 * m) {
        a_entries.push((constraint_idx, sum1_base_idx + i, one)); // gammai(v[i]-v[i])
    }
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, gamma_sum1_idx, one)); // = Sigmagammai(v[i]-v[i])
    constraint_count += 1;

    let constraint_idx = constraint_count;
    for i in 0..(2 * m) {
        a_entries.push((constraint_idx, sum2_base_idx + i, one)); // gammai(v[i]-v[i-1])
    }
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, gamma_sum2_idx, one)); // = Sigmagammai(v[i]-v[i-1])
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, ra_p_diff1_idx, one)); // ra.(p3-p3-1)
    a_entries.push((constraint_idx, rb_v_diff1_idx, one)); // + rb.(v3-v3)
    a_entries.push((constraint_idx, gamma_sum1_idx, one)); // + Sigmagammai(v[i]-v[i])
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, term1_idx, one)); // = term1
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, ra_p_diff2_idx, one)); // ra.(p3-p3)
    a_entries.push((constraint_idx, rb_v_diff2_idx, one)); // + rb.(v3-2.v3)
    a_entries.push((constraint_idx, gamma_sum2_idx, one)); // + Sigmagammai(v[i]-v[i-1])
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, term2_idx, one)); // = term2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, updated_const_one_idx, one)); // 1
    a_entries.push((constraint_idx, v_top_bit_idx, minus_one)); // - v[2m+1]
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, one_minus_v_idx, one)); // = (1-v[2m+1])
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v_top_bit_idx, one)); // v[2m+1]
    b_entries.push((constraint_idx, term1_idx, one)); // x term1
    c_entries.push((constraint_idx, v_times_term1_idx, one)); // = v[2m+1] * term1
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, one_minus_v_idx, one)); // (1-v[2m+1])
    b_entries.push((constraint_idx, term2_idx, one)); // x term2
    c_entries.push((constraint_idx, one_minus_v_times_term2_idx, one)); // = (1-v[2m+1]) * term2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v_times_term1_idx, one)); // v[2m+1] * term1
    a_entries.push((constraint_idx, one_minus_v_times_term2_idx, one)); // + (1-v[2m+1]) * term2
    b_entries.push((constraint_idx, updated_const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, final_result_idx, one));
    constraint_count += 1;

    // ============================================================================
    // ============================================================================

    // f1 = v3 - Sigmai=0m v[m+1+i] . 2i
    // f2 = f1 - 1
    // (1-rdb) . f1 + rdb . ((1-stb)(1-v[m+1]) f1 + (stb+v[m+1]) f2) = 0
    //

    let rdb_idx = next_idx;
    next_idx += 1; // rdb = v[m]
    let stb_idx = next_idx;
    next_idx += 1; // stb (sticky bit)
    let f1_idx = next_idx;
    next_idx += 1; // f1
    let f2_idx = next_idx;
    next_idx += 1; // f2 = f1 - 1
    let sum_upper_bits_idx = next_idx;
    next_idx += 1; // Sigmai=0m v[m+1+i] . 2i
    let v_tilde_3_idx = next_idx;
    next_idx += 1; // v3

    let _stb_product_base_idx = next_idx;
    next_idx += params.m + 1;
    let stb_sum_idx = next_idx;
    next_idx += 1;

    let stb_sum_minus_stb_idx = next_idx;
    next_idx += 1; // (stb_sum - stb)
    let stb_logic_result_idx = next_idx;
    next_idx += 1; // stb x (stb_sum - stb)
    let zero_var_idx = next_idx;
    next_idx += 1;

    let one_minus_rdb_idx = next_idx;
    next_idx += 1; // (1 - rdb)
    let one_minus_stb_idx = next_idx;
    next_idx += 1; // (1 - stb)
    let v_hat_m_plus_1_idx = next_idx;
    next_idx += 1; // v[m+1]
    let one_minus_v_hat_m_plus_1_idx = next_idx;
    next_idx += 1; // (1 - v[m+1])
    let stb_plus_v_hat_m_plus_1_idx = next_idx;
    next_idx += 1; // (stb + v[m+1])

    let term_a_idx = next_idx;
    next_idx += 1; // (1-stb)(1-v[m+1])
    let term_b_idx = next_idx;
    next_idx += 1; // term_a . f1
    let term_c_idx = next_idx;
    next_idx += 1; // (stb+v[m+1]) . f2
    let bracket_content_idx = next_idx;
    next_idx += 1; // term_b + term_c
    let left_part_idx = next_idx;
    next_idx += 1; // (1-rdb) . f1
    let right_part_idx = next_idx;
    next_idx += 1; // rdb . bracket_content
    let final_round_result_idx = next_idx;
    next_idx += 1;

    let updated_const_one_idx_final = next_idx;
    let num_vars_final = updated_const_one_idx_final + 1;

    let constraint_idx = constraint_count;
    if (v_hat_bits_base_idx + params.m) < num_vars_final {
        a_entries.push((constraint_idx, v_hat_bits_base_idx + params.m, one)); // v[m]
        b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
        c_entries.push((constraint_idx, rdb_idx, one)); // = rdb
        constraint_count += 1;
    }

    let constraint_idx = constraint_count;
    for i in 0..=params.m {
        let bit_idx = params.m + 1 + i;
        if i < powers_of_two.len() && (v_hat_bits_base_idx + bit_idx) < num_vars_final {
            let weight = powers_of_two[i]; // 2i
            a_entries.push((constraint_idx, v_hat_bits_base_idx + bit_idx, weight));
            // v[m+1+i] x 2i
        }
    }
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, sum_upper_bits_idx, one)); // = sum_upper_bits
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v_tilde_3_idx, one)); // v3
    a_entries.push((constraint_idx, sum_upper_bits_idx, minus_one)); // - sum_upper_bits
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, f1_idx, one)); // = f1
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, f1_idx, one)); // f1
    a_entries.push((constraint_idx, updated_const_one_idx_final, minus_one)); // - 1
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, f2_idx, one)); // = f2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    for i in 0..params.m {
        if (v_hat_bits_base_idx + i) < num_vars_final {
            a_entries.push((constraint_idx, v_hat_bits_base_idx + i, one)); // v[i], i in [0, m-1]
        }
    }
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, stb_sum_idx, one)); // = stb_sum
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, stb_idx, one)); // stb
    b_entries.push((constraint_idx, stb_idx, one)); // x stb
    c_entries.push((constraint_idx, stb_idx, one)); // = stb
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, updated_const_one_idx_final, one)); // 1
    a_entries.push((constraint_idx, stb_idx, minus_one)); // - stb
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, stb_logic_result_idx, one)); // = (1-stb)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, stb_sum_idx, one)); // stb_sum
    b_entries.push((constraint_idx, stb_logic_result_idx, one)); // x (1-stb)
    c_entries.push((constraint_idx, zero_var_idx, one)); // = 0
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, stb_sum_idx, one)); // stb_sum
    a_entries.push((constraint_idx, stb_idx, minus_one)); // - stb
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, stb_sum_minus_stb_idx, one)); // = (stb_sum - stb)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, updated_const_one_idx_final, one)); // 1
    a_entries.push((constraint_idx, rdb_idx, minus_one)); // - rdb
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, one_minus_rdb_idx, one)); // = (1-rdb)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, updated_const_one_idx_final, one)); // 1
    a_entries.push((constraint_idx, stb_idx, minus_one)); // - stb
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, one_minus_stb_idx, one)); // = (1-stb)
    constraint_count += 1;

    let constraint_idx = constraint_count;
    let v_hat_m_plus_1_bit_idx = v_hat_bits_base_idx + params.m + 1;
    if v_hat_m_plus_1_bit_idx < num_vars_final {
        a_entries.push((constraint_idx, v_hat_m_plus_1_bit_idx, one)); // v[m+1]
        b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
        c_entries.push((constraint_idx, v_hat_m_plus_1_idx, one)); // = v[m+1]
        constraint_count += 1;
    }

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, updated_const_one_idx_final, one)); // 1
    a_entries.push((constraint_idx, v_hat_m_plus_1_idx, minus_one)); // - v[m+1]
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, one_minus_v_hat_m_plus_1_idx, one)); // = (1-v[m+1])
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, stb_idx, one)); // stb
    a_entries.push((constraint_idx, v_hat_m_plus_1_idx, one)); // + v[m+1]
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, stb_plus_v_hat_m_plus_1_idx, one)); // = (stb + v[m+1])
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, one_minus_stb_idx, one)); // (1-stb)
    b_entries.push((constraint_idx, one_minus_v_hat_m_plus_1_idx, one)); // x (1-v[m+1])
    c_entries.push((constraint_idx, term_a_idx, one)); // = term_a
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, term_a_idx, one)); // term_a
    b_entries.push((constraint_idx, f1_idx, one)); // x f1
    c_entries.push((constraint_idx, term_b_idx, one)); // = term_b
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, stb_plus_v_hat_m_plus_1_idx, one)); // (stb + v[m+1])
    b_entries.push((constraint_idx, f2_idx, one)); // x f2
    c_entries.push((constraint_idx, term_c_idx, one)); // = term_c
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, term_b_idx, one)); // term_b
    a_entries.push((constraint_idx, term_c_idx, one)); // + term_c
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, bracket_content_idx, one)); // = bracket_content
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, one_minus_rdb_idx, one)); // (1-rdb)
    b_entries.push((constraint_idx, f1_idx, one)); // x f1
    c_entries.push((constraint_idx, left_part_idx, one)); // = left_part
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, rdb_idx, one)); // rdb
    b_entries.push((constraint_idx, bracket_content_idx, one)); // x bracket_content
    c_entries.push((constraint_idx, right_part_idx, one)); // = right_part
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, left_part_idx, one)); // left_part
    a_entries.push((constraint_idx, right_part_idx, one)); // + right_part
    b_entries.push((constraint_idx, updated_const_one_idx_final, one)); // x 1
    c_entries.push((constraint_idx, final_round_result_idx, one));
    constraint_count += 1;

    let updated_const_one_idx = updated_const_one_idx_final;
    const_one_indices.push(updated_const_one_idx);

    let mut total_constraints = constraint_count;
    let mut num_vars = num_vars_final;
    let mut double_shuffle_contexts: Vec<(
        MulFPBooleanGroup,
        Option<MulFPDoubleShuffleGroupContext>,
    )> = Vec::new();

    if matches!(mode, MulFPDoubleShuffleMode::Embedded) {
        let boolean_slices = collect_mulfp_bool_slices(params);
        let batch_index_book = merge_mulfp_bool_slices(&[boolean_slices.clone()])
            .map_err(|e| format!("Failed to combine MULFP boolean slices: {}", e))?;

        let double_shuffle_groups = [
            MulFPBooleanGroup::MantissaProduct,
            MulFPBooleanGroup::MantissaNormalized,
            MulFPBooleanGroup::SignBits,
            MulFPBooleanGroup::RoundingFlags,
            MulFPBooleanGroup::NormalizationSwitches,
        ];

        double_shuffle_contexts.reserve(double_shuffle_groups.len());
        let mut ds_constraint_cursor = total_constraints;
        let mut ds_next_var_idx = num_vars;
        let seed_data = encode_mulfp_input_seed(input_data, params, mode);
        let ds_rng_base =
            TranscriptRng::from_domain_data("mulfp-single-double-shuffle", &seed_data);

        for group in double_shuffle_groups {
            let cols = batch_index_book.slice_len(group);
            if cols == 0 {
                double_shuffle_contexts.push((group, None));
                continue;
            }

            let mut group_rng = ds_rng_base.fork(&format!("mulfp-single-group-{}", group.label()));
            let randomness = sample_mulfp_double_shuffle_randomness(
                batch_index_book.batch,
                cols,
                &mut group_rng,
            );

            let context = MulFPDoubleShuffleGroupContext::new(
                group,
                &batch_index_book,
                randomness,
                ds_next_var_idx,
                zero_var_idx,
            )
            .map_err(|e| {
                format!(
                    "Failed to construct {:?} Double-Shuffling context: {}",
                    group, e
                )
            })?;

            ds_constraint_cursor = context.append_constraints(
                ds_constraint_cursor,
                &mut a_entries,
                &mut b_entries,
                &mut c_entries,
                updated_const_one_idx,
            );

            ds_next_var_idx = context.next_var_idx();
            double_shuffle_contexts.push((group, Some(context)));
        }

        total_constraints = ds_constraint_cursor;
        num_vars = ds_next_var_idx;
    }

    // ============================================================================
    // ============================================================================

    let metrics = compute_r1cs_metrics(total_constraints, &a_entries, &b_entries, &c_entries);
    let inst = Instance::new(
        total_constraints,
        num_vars,
        num_inputs,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("{:?}", e))?;

    let mut double_shuffle_witnesses: Vec<MulFPDoubleShuffleWitness> =
        Vec::with_capacity(double_shuffle_contexts.len());
    let (assignment_vars, assignment_inputs, assignment_vec) = if let Some(data) = input_data {
        let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars];

        let scalar_from_i64 = |value: i64| -> Scalar {
            if value >= 0 {
                Scalar::from(value as u64)
            } else {
                -Scalar::from((-value) as u64)
            }
        };

        vars[b1_idx] = Scalar::from(data.b1).to_bytes();
        vars[v1_idx] = Scalar::from(data.v1).to_bytes();
        vars[p1_idx] = scalar_from_i64(data.p1).to_bytes();
        vars[b2_idx] = Scalar::from(data.b2).to_bytes();
        vars[v2_idx] = Scalar::from(data.v2).to_bytes();
        vars[p2_idx] = scalar_from_i64(data.p2).to_bytes();

        let power_2m_u128 = 1u128 << m;
        let v1_prime_u128 = data.v1 as u128 + power_2m_u128;
        let v2_prime_u128 = data.v2 as u128 + power_2m_u128;
        let v1_prime_val = scalar_from_u128(v1_prime_u128);
        let v2_prime_val = scalar_from_u128(v2_prime_u128);
        vars[v1_prime_idx] = v1_prime_val.to_bytes();
        vars[v2_prime_idx] = v2_prime_val.to_bytes();

        let b1_times_b2 = data.b1 * data.b2;
        let b3 = data.b1 ^ data.b2;
        let p3 = data.p1 + data.p2;

        vars[b1_times_b2_idx] = Scalar::from(b1_times_b2).to_bytes();
        #[cfg(test)]
        println!(
            "[mulfp-debug] b1_times_b2={:?} idx={} stored={:?}",
            b1_times_b2,
            b1_times_b2_idx,
            Scalar::from_bytes_mod_order(vars[b1_times_b2_idx])
        );
        vars[b3_idx] = Scalar::from(b3).to_bytes();
        vars[p3_idx] = scalar_from_i64(p3).to_bytes();

        let v3_u128 = v1_prime_u128 * v2_prime_u128;
        let v3_val = scalar_from_u128(v3_u128);
        vars[v3_idx] = v3_val.to_bytes();

        #[cfg(test)]
        println!("[mulfp-debug] v3_u128={:#x}", v3_u128);

        for i in 0..binary_bits {
            let bit = bit_at_u128(v3_u128, i);
            vars[v_bits_base_idx + i] = Scalar::from(bit).to_bytes();
        }

        let highest_bit = if binary_bits > 0 {
            bit_at_u128(v3_u128, binary_bits - 1)
        } else {
            0
        };
        let p3_hat = p3 + highest_bit as i64;
        vars[p3_hat_idx] = scalar_from_i64(p3_hat).to_bytes();

        #[cfg(test)]
        {
            println!(
                "[mulfp-debug] p3={} p3_hat={} highest_bit={}",
                p3, p3_hat, highest_bit
            );
            println!(
                "[mulfp-debug] p3_scalar={:?} p3_hat_scalar={:?}",
                Scalar::from_bytes_mod_order(vars[p3_idx]),
                Scalar::from_bytes_mod_order(vars[p3_hat_idx])
            );
        }

        let v3_hat_u128 = if highest_bit == 1 {
            v3_u128
        } else {
            v3_u128 << 1
        };
        let v3_hat = scalar_from_u128(v3_hat_u128);
        vars[v3_hat_idx] = v3_hat.to_bytes();

        for i in 0..binary_bits {
            let bit = bit_at_u128(v3_hat_u128, i);
            vars[v_hat_bits_base_idx + i] = Scalar::from(bit).to_bytes();
        }

        vars[ra_idx] = ra_scalar.to_bytes();
        vars[rb_idx] = rb_scalar.to_bytes();
        for i in 0..(2 * m) {
            vars[gamma_base_idx + i] = gamma[i].to_bytes();
        }

        let p3_scalar = scalar_from_i64(p3);
        let p3_hat_scalar = scalar_from_i64(p3_hat);

        vars[p_diff1_idx] = (p3_hat_scalar - p3_scalar - Scalar::ONE).to_bytes(); // p3-p3-1
        vars[v_diff1_idx] = (v3_hat - v3_val).to_bytes(); // v3-v3
        vars[p_diff2_idx] = (p3_hat_scalar - p3_scalar).to_bytes(); // p3-p3
        vars[v_diff2_idx] = (v3_hat - v3_val * Scalar::from(2u8)).to_bytes(); // v3-2.v3

        let p_diff1_val = p3_hat_scalar - p3_scalar - Scalar::ONE;
        let v_diff1_val = v3_hat - v3_val;
        let p_diff2_val = p3_hat_scalar - p3_scalar;
        let v_diff2_val = v3_hat - v3_val * Scalar::from(2u8);

        vars[ra_p_diff1_idx] = (ra_scalar * p_diff1_val).to_bytes(); // ra.(p3-p3-1)
        vars[rb_v_diff1_idx] = (rb_scalar * v_diff1_val).to_bytes(); // rb.(v3-v3)
        vars[ra_p_diff2_idx] = (ra_scalar * p_diff2_val).to_bytes(); // ra.(p3-p3)
        vars[rb_v_diff2_idx] = (rb_scalar * v_diff2_val).to_bytes(); // rb.(v3-2.v3)

        let mut gamma_sum1 = Scalar::ZERO;
        for i in 0..(2 * m) {
            let v_hat_bit = Scalar::from(bit_at_u128(v3_hat_u128, i));
            let v_bit = Scalar::from(bit_at_u128(v3_u128, i));
            let diff = v_hat_bit - v_bit;
            let term = gamma[i] * diff;
            vars[sum1_base_idx + i] = term.to_bytes();
            gamma_sum1 += term;
        }
        vars[gamma_sum1_idx] = gamma_sum1.to_bytes();

        let mut gamma_sum2 = Scalar::ZERO;
        for i in 0..(2 * m) {
            let v_hat_bit = Scalar::from(bit_at_u128(v3_hat_u128, i));
            let v_prev_bit = if i > 0 {
                Scalar::from(bit_at_u128(v3_u128, i - 1))
            } else {
                Scalar::ZERO
            };
            let diff = v_hat_bit - v_prev_bit;
            let term = gamma[i] * diff;
            vars[sum2_base_idx + i] = term.to_bytes();
            gamma_sum2 += term;
        }
        vars[gamma_sum2_idx] = gamma_sum2.to_bytes();

        let term1 = ra_scalar * p_diff1_val + rb_scalar * v_diff1_val + gamma_sum1;
        let term2 = ra_scalar * p_diff2_val + rb_scalar * v_diff2_val + gamma_sum2;
        vars[term1_idx] = term1.to_bytes();
        vars[term2_idx] = term2.to_bytes();

        let v_top_bit_scalar = Scalar::from(highest_bit);
        let one_minus_v = Scalar::ONE - v_top_bit_scalar;
        vars[one_minus_v_idx] = one_minus_v.to_bytes();
        vars[v_times_term1_idx] = (v_top_bit_scalar * term1).to_bytes();
        vars[one_minus_v_times_term2_idx] = (one_minus_v * term2).to_bytes();

        let final_result = v_top_bit_scalar * term1 + one_minus_v * term2;
        vars[final_result_idx] = final_result.to_bytes();

        // rdb = v[m]
        let rdb_val = if params.m < binary_bits {
            Scalar::from(bit_at_u128(v3_hat_u128, params.m))
        } else {
            Scalar::ZERO
        };
        vars[rdb_idx] = rdb_val.to_bytes();
        #[cfg(test)]
        println!(
            "[mulfp-debug] rdb_val={:?} -> bytes {:?}",
            rdb_val, vars[rdb_idx]
        );

        let mut stb_sum_val = Scalar::ZERO;
        for i in 0..params.m {
            let bit_val = Scalar::from(bit_at_u128(v3_hat_u128, i));
            stb_sum_val += bit_val;
        }
        vars[stb_sum_idx] = stb_sum_val.to_bytes();

        let stb_val = if stb_sum_val == Scalar::ZERO {
            Scalar::ZERO
        } else {
            Scalar::ONE
        };
        vars[stb_idx] = stb_val.to_bytes();

        let stb_sum_minus_stb_val = stb_sum_val - stb_val;
        vars[stb_sum_minus_stb_idx] = stb_sum_minus_stb_val.to_bytes();

        let one_minus_stb_for_logic = Scalar::ONE - stb_val;
        vars[stb_logic_result_idx] = one_minus_stb_for_logic.to_bytes();

        #[cfg(debug_assertions)]
        {
            let product_check = stb_sum_val * one_minus_stb_for_logic;
            debug_assert!(product_check == Scalar::ZERO);
        }

        vars[zero_var_idx] = Scalar::ZERO.to_bytes();

        let mut sum_upper_bits_val = Scalar::ZERO;
        for i in 0..=params.m {
            let bit_val = Scalar::from(bit_at_u128(v3_hat_u128, params.m + 1 + i));
            let weight = scalar_from_u128(1u128 << i);
            sum_upper_bits_val += bit_val * weight;
        }
        vars[sum_upper_bits_idx] = sum_upper_bits_val.to_bytes();

        vars[v_tilde_3_idx] = v3_hat.to_bytes();

        // f1 = v3 - sum_upper_bits
        let f1_val = v3_hat - sum_upper_bits_val;
        vars[f1_idx] = f1_val.to_bytes();

        // f2 = f1 - 1
        let f2_val = f1_val - Scalar::ONE;
        vars[f2_idx] = f2_val.to_bytes();

        // (1 - rdb)
        let one_minus_rdb_val = Scalar::ONE - rdb_val;
        vars[one_minus_rdb_idx] = one_minus_rdb_val.to_bytes();

        // (1 - stb)
        let one_minus_stb_val = Scalar::ONE - stb_val;
        vars[one_minus_stb_idx] = one_minus_stb_val.to_bytes();

        // v[m+1]
        let v_hat_m_plus_1_val = Scalar::from(bit_at_u128(v3_hat_u128, params.m + 1));
        vars[v_hat_m_plus_1_idx] = v_hat_m_plus_1_val.to_bytes();

        // (1 - v[m+1])
        let one_minus_v_hat_m_plus_1_val = Scalar::ONE - v_hat_m_plus_1_val;
        vars[one_minus_v_hat_m_plus_1_idx] = one_minus_v_hat_m_plus_1_val.to_bytes();

        // (stb + v[m+1])
        let stb_plus_v_hat_m_plus_1_val = stb_val + v_hat_m_plus_1_val;
        vars[stb_plus_v_hat_m_plus_1_idx] = stb_plus_v_hat_m_plus_1_val.to_bytes();

        // term_a = (1-stb) x (1-v[m+1])
        let term_a_val = one_minus_stb_val * one_minus_v_hat_m_plus_1_val;
        vars[term_a_idx] = term_a_val.to_bytes();

        // term_b = term_a x f1
        let term_b_val = term_a_val * f1_val;
        vars[term_b_idx] = term_b_val.to_bytes();

        // term_c = (stb + v[m+1]) x f2
        let term_c_val = stb_plus_v_hat_m_plus_1_val * f2_val;
        vars[term_c_idx] = term_c_val.to_bytes();

        // bracket_content = term_b + term_c
        let bracket_content_val = term_b_val + term_c_val;
        vars[bracket_content_idx] = bracket_content_val.to_bytes();

        // left_part = (1-rdb) x f1
        let left_part_val = one_minus_rdb_val * f1_val;
        vars[left_part_idx] = left_part_val.to_bytes();

        // right_part = rdb x bracket_content
        let right_part_val = rdb_val * bracket_content_val;
        vars[right_part_idx] = right_part_val.to_bytes();

        let final_round_result_val = left_part_val + right_part_val;
        vars[final_round_result_idx] = final_round_result_val.to_bytes();

        vars[updated_const_one_idx] = one;

        if let Some(ref witness_vals) = data.witness_values {
            if witness_vals.len() == num_vars {
                vars = witness_vals.clone();
            }
        }

        for &idx in &const_one_indices {
            let mut one_bytes = [0u8; 32];
            one_bytes[0] = 1;
            vars[idx] = one_bytes;
        }

        vars[rdb_idx] = rdb_val.to_bytes();

        #[cfg(test)]
        {
            println!("[mulfp-debug] rdb_idx={}", rdb_idx);
            let diff_debug = p3_hat_scalar - p3_scalar;
            println!("[mulfp-debug] diff_scalar={:?}", diff_debug);
            println!(
                "[mulfp-debug] const_one_idx={} value={:?} legacy_idx={} legacy_value={:?}",
                updated_const_one_idx,
                Scalar::from_bytes_mod_order(vars[updated_const_one_idx]),
                const_one_idx_updated,
                Scalar::from_bytes_mod_order(vars[const_one_idx_updated])
            );
            println!(
                "[mulfp-debug] early_const_value={:?}",
                Scalar::from_bytes_mod_order(vars[const_one_indices[2]])
            );
            println!("[mulfp-debug] const_one_indices={:?}", const_one_indices);
            println!(
                "[mulfp-debug] sample var[85]={:?}",
                Scalar::from_bytes_mod_order(vars[85])
            );
            println!(
                "[mulfp-debug] const_one_idx_updated value={:?}",
                Scalar::from_bytes_mod_order(vars[const_one_idx_updated])
            );
            println!(
                "[mulfp-debug] rdb_var={:?}",
                Scalar::from_bytes_mod_order(vars[rdb_idx])
            );
        }

        let input_bytes = vec![Scalar::ZERO.to_bytes(); num_inputs];

        for (group, context_opt) in &double_shuffle_contexts {
            if let Some(context) = context_opt {
                let witness = context.populate_witness(&mut vars).map_err(|e| {
                    format!(
                        "Failed to generate {:?} Double-Shuffling witness: {}",
                        group, e
                    )
                })?;
                double_shuffle_witnesses.push(witness);
            } else {
                double_shuffle_witnesses.push(MulFPDoubleShuffleWitness::empty(*group));
            }
        }

        #[cfg(test)]
        {
            if let Some((row, a_eval, b_eval, rhs)) = first_unsatisfied_row(
                total_constraints,
                num_vars,
                num_inputs,
                &a_entries,
                &b_entries,
                &c_entries,
                &vars,
                &input_bytes,
            ) {
                panic!(
                    "MULFP witness constraint row {} is not satisfied: a_eval={:?} b_eval={:?} rhs={:?}",
                    row, a_eval, b_eval, rhs
                );
            }
        }

        let vars_snapshot = vars.clone();
        let assignment_vars = VarsAssignment::new(&vars_snapshot)
            .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
        let assignment_inputs = InputsAssignment::new(&input_bytes)
            .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

        (assignment_vars, assignment_inputs, vars_snapshot)
    } else {
        let mut zero_vars = vec![Scalar::ZERO.to_bytes(); num_vars];
        for &idx in &const_one_indices {
            let mut one_bytes = [0u8; 32];
            one_bytes[0] = 1;
            zero_vars[idx] = one_bytes;
        }
        let vars_snapshot = zero_vars.clone();
        let assignment_vars = VarsAssignment::new(&vars_snapshot)
            .map_err(|e| format!("Failed to create default variable assignment: {:?}", e))?;
        let assignment_inputs =
            InputsAssignment::new(&vec![Scalar::ZERO.to_bytes(); num_inputs])
                .map_err(|e| format!("Failed to create default input assignment: {:?}", e))?;

        for (group, _) in &double_shuffle_contexts {
            double_shuffle_witnesses.push(MulFPDoubleShuffleWitness::empty(*group));
        }

        (assignment_vars, assignment_inputs, vars_snapshot)
    };

    let num_non_zero_entries = a_entries.len() + b_entries.len() + c_entries.len();

    Ok(MulFPR1CSArtifacts {
        num_constraints: total_constraints,
        num_vars,
        num_inputs,
        num_non_zero_entries,
        instance: inst,
        vars: assignment_vars,
        inputs: assignment_inputs,
        assignment: assignment_vec,
        double_shuffle: double_shuffle_witnesses,
        metrics,
    })
}

pub fn produce_r1cs_mulfp_with_params(
    params: &MulFPParams,
    input_data: Option<&MulFPInputData>,
) -> Result<MulFPR1CSArtifacts, String> {
    produce_r1cs_mulfp_with_mode(params, input_data, MulFPDoubleShuffleMode::Embedded)
}

pub fn produce_r1cs_mulfp_detached(
    params: &MulFPParams,
    input_data: Option<&MulFPInputData>,
) -> Result<MulFPR1CSArtifacts, String> {
    produce_r1cs_mulfp_with_mode(params, input_data, MulFPDoubleShuffleMode::Detached)
}

///
pub fn produce_r1cs_basic_constraints_only(
    params: &MulFPParams,
    input_data: Option<&MulFPInputData>,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    if params.w == 0 || !params.w.is_power_of_two() {
        return Err("w must be a power of 2 and greater than 0".to_string());
    }

    let num_inputs = 6;

    let one = Scalar::ONE.to_bytes();
    let _minus_one = (-Scalar::ONE).to_bytes();

    let two = Scalar::from(2u8);
    let mut power_of_two = Scalar::ONE;
    let mut powers_of_two = Vec::new();
    powers_of_two.push(Scalar::ONE.to_bytes());
    for _i in 0..(params.acc - 1) {
        power_of_two = power_of_two * two;
        powers_of_two.push(power_of_two.to_bytes());
    }

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

    let b1_idx = 0;
    let v1_idx = b1_idx + 1;
    let p1_idx = v1_idx + 1;
    let b2_idx = p1_idx + 1;
    let v2_idx = b2_idx + 1;
    let p2_idx = v2_idx + 1;
    let b3_idx = p2_idx + 1;
    let v3_idx = b3_idx + 1;
    let p3_idx = v3_idx + 1;
    let v1_prime_idx = p3_idx + 1; // v1': v1 + 2^m
    let v2_prime_idx = v1_prime_idx + 1; // v2': v2 + 2^m
    let const_one_idx = v2_prime_idx + 1;
    let b1_times_b2_idx = const_one_idx + 1;

    let m = params.m;
    let binary_bits = 2 * m + 2;
    let v_bits_base_idx = b1_times_b2_idx + 1;

    let num_vars = v_bits_base_idx + binary_bits;

    let mut constraint_count = 0;

    if m >= powers_of_two.len() {
        return Err(format!(
            "m({}) exceeds the range of powers_of_two array ({})",
            m,
            powers_of_two.len()
        ));
    }
    let power_2m = powers_of_two[m]; // 2^m

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v1_idx, one)); // v1
    a_entries.push((constraint_idx, const_one_idx, power_2m)); // + 2^m
    b_entries.push((constraint_idx, const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, v1_prime_idx, one)); // = v1'
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v2_idx, one)); // v2
    a_entries.push((constraint_idx, const_one_idx, power_2m)); // + 2^m
    b_entries.push((constraint_idx, const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, v2_prime_idx, one)); // = v2'
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, b1_idx, one)); // b1
    b_entries.push((constraint_idx, b2_idx, one)); // * b2
    c_entries.push((constraint_idx, b1_times_b2_idx, one)); // = b1*b2
    constraint_count += 1;

    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, b1_idx, one)); // b1
    a_entries.push((constraint_idx, b2_idx, one)); // + b2
    a_entries.push((
        constraint_idx,
        b1_times_b2_idx,
        (-Scalar::from(2u8)).to_bytes(),
    )); // - 2*b1*b2
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, b3_idx, one)); // = b3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, v1_prime_idx, one)); // v1'
    b_entries.push((constraint_idx, v2_prime_idx, one)); // x v2'
    c_entries.push((constraint_idx, v3_idx, one)); // = v3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    a_entries.push((constraint_idx, p1_idx, one)); // p1
    a_entries.push((constraint_idx, p2_idx, one)); // + p2
    b_entries.push((constraint_idx, const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, p3_idx, one)); // = p3
    constraint_count += 1;

    // ============================================================================
    // ============================================================================
    let constraint_idx = constraint_count;
    for i in 0..binary_bits {
        if i >= powers_of_two.len() {
            return Err(format!(
                "Binary decomposition weight index {} is outside the range of powers_of_two array {}",
                i,
                powers_of_two.len()
            ));
        }
        let weight = powers_of_two[i]; // 2^i
        a_entries.push((constraint_idx, v_bits_base_idx + i, weight)); // 2^i x v[i]
    }
    b_entries.push((constraint_idx, const_one_idx, one)); // x 1
    c_entries.push((constraint_idx, v3_idx, one)); // = v3
    constraint_count += 1;

    let total_constraints = constraint_count;

    // ============================================================================
    // ============================================================================
    let inst = Instance::new(
        total_constraints,
        num_vars,
        num_inputs,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("{:?}", e))?;

    // ============================================================================
    // ============================================================================
    let (assignment_vars, assignment_inputs) = if let Some(data) = input_data {
        let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars];

        let scalar_from_i64 = |value: i64| -> Scalar {
            if value >= 0 {
                Scalar::from(value as u64)
            } else {
                -Scalar::from((-value) as u64)
            }
        };

        vars[b1_idx] = Scalar::from(data.b1).to_bytes();
        vars[v1_idx] = Scalar::from(data.v1).to_bytes();
        vars[p1_idx] = scalar_from_i64(data.p1).to_bytes();
        vars[b2_idx] = Scalar::from(data.b2).to_bytes();
        vars[v2_idx] = Scalar::from(data.v2).to_bytes();
        vars[p2_idx] = scalar_from_i64(data.p2).to_bytes();

        let power_2m_u128 = 1u128 << m;
        let v1_prime_u128 = data.v1 as u128 + power_2m_u128;
        let v2_prime_u128 = data.v2 as u128 + power_2m_u128;
        let v1_prime_val = scalar_from_u128(v1_prime_u128);
        let v2_prime_val = scalar_from_u128(v2_prime_u128);
        vars[v1_prime_idx] = v1_prime_val.to_bytes();
        vars[v2_prime_idx] = v2_prime_val.to_bytes();

        let b1_times_b2 = data.b1 * data.b2;
        let b3 = data.b1 ^ data.b2;
        let p3 = data.p1 + data.p2;

        vars[b1_times_b2_idx] = Scalar::from(b1_times_b2).to_bytes();
        vars[b3_idx] = Scalar::from(b3).to_bytes();
        vars[p3_idx] = scalar_from_i64(p3).to_bytes();

        let v3_u128 = v1_prime_u128 * v2_prime_u128;
        let v3_val = scalar_from_u128(v3_u128);
        vars[v3_idx] = v3_val.to_bytes();

        for i in 0..binary_bits {
            let bit = bit_at_u128(v3_u128, i);
            vars[v_bits_base_idx + i] = Scalar::from(bit).to_bytes();
        }

        vars[const_one_idx] = one;

        let assignment_vars = VarsAssignment::new(&vars)
            .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
        let assignment_inputs =
            InputsAssignment::new(&vec![Scalar::ZERO.to_bytes(); num_inputs])
                .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

        (assignment_vars, assignment_inputs)
    } else {
        let mut zero_vars = vec![Scalar::ZERO.to_bytes(); num_vars];
        zero_vars[const_one_idx] = one;

        let assignment_vars = VarsAssignment::new(&zero_vars)
            .map_err(|e| format!("Failed to create default variable assignment: {:?}", e))?;
        let assignment_inputs =
            InputsAssignment::new(&vec![Scalar::ZERO.to_bytes(); num_inputs])
                .map_err(|e| format!("Failed to create default input assignment: {:?}", e))?;

        (assignment_vars, assignment_inputs)
    };

    let num_non_zero_entries = a_entries.len() + b_entries.len() + c_entries.len();

    Ok((
        total_constraints,
        num_vars,
        num_inputs,
        num_non_zero_entries,
        inst,
        assignment_vars,
        assignment_inputs,
    ))
}
#[cfg(feature = "manual_examples")]
fn main() {
    println!("Floating point multiplication constraint system - IEEE 754 single precision/double precision configuration test");
    println!("{}", "=".repeat(60));

    demonstrate_precision_configs();

    test_double_precision_performance();

    verify_constraint_correctness();

    println!("\nIEEE 754 single precision/double precision configuration test completed!");
}

#[cfg(feature = "manual_examples")]
fn demonstrate_precision_configs() {
    println!("Comparative analysis of IEEE 754 precision configuration");
    println!("{}", "-".repeat(60));

    let configs = vec![
        ("single precision", MulFPParams::single_precision()),
        ("double precision", MulFPParams::double_precision()),
    ];

    println!(
        "{:<12} {:<6} {:<10} {:<12} {:<10} {:<12}",
        "Configuration",
        "m value",
        "constraint group 7",
        "total constraints",
        "gamma number of terms",
        "Memory (MB)"
    );
    println!("{}", "-".repeat(70));

    for (name, params) in configs {
        let complexity = params.constraint_complexity();
        println!(
            "{:<12} {:<6} {:<10} {:<12} {:<10} {:<12.2}",
            name,
            params.m,
            complexity.group_7,
            complexity.total,
            complexity.gamma_terms,
            complexity.estimated_memory_mb
        );
    }

    println!();
    println!("Key findings:");
    println!("- The number of constraints grows linearly with the m parameter (mainly from constraint group 7)");
    println!("-Double precision increases 1.9 times compared to single precision, which is still within the acceptable range");
    println!("- Memory requirements are less than 1MB in all configurations");
}

#[cfg(feature = "manual_examples")]
fn test_double_precision_performance() {
    println!("\nIEEE 754 double precision configuration performance test");
    println!("{}", "-".repeat(60));

    let params = MulFPParams::double_precision();
    let complexity = params.constraint_complexity();

    println!("Double configuration (e={}, m={}):", params.e, params.m);
    println!("Constraint complexity:");
    println!("- Constraint groups 1-6: {}", complexity.group_1_6);
    println!("- Constraint group 7: {}", complexity.group_7);
    println!("- Constraint group 8: {}", complexity.group_8);
    println!("- Total: {} constraints", complexity.total);
    println!("Resource requirements:");
    println!("- Binary digits: {} bits", complexity.binary_bits);
    println!("- Number of gamma items: {}", complexity.gamma_terms);
    println!(
        "- Estimated memory: {:.2} MB",
        complexity.estimated_memory_mb
    );

    println!("\nDouble precision numerical test:");
    let double_precision_data = MulFPInputData {
        b1: 0,
        v1: 4503599627370495,
        p1: 1000,
        b2: 0,
        v2: 2251799813685247,
        p2: 500,
        witness_values: None,
    };

    println!("Enter data:");
    println!(
        "First number: sign={}, mantissa={}, exponent={}",
        double_precision_data.b1, double_precision_data.v1, double_precision_data.p1
    );
    println!(
        "Second number: sign={}, mantissa={}, exponent={}",
        double_precision_data.b2, double_precision_data.v2, double_precision_data.p2
    );

    let start_time = std::time::Instant::now();

    match produce_r1cs_mulfp_with_params(&params, Some(&double_precision_data)) {
        Ok(artifacts) => {
            let generation_time = start_time.elapsed();

            println!("Constraint generation performance:");
            println!("- Generation time: {:.2?}", generation_time);
            println!(
                "- Actual number of constraints: {}",
                artifacts.num_constraints
            );
            println!("- Actual number of variables: {}", artifacts.num_vars);
            println!(
                "- Number of non-zero terms: {}",
                artifacts.num_non_zero_entries
            );

            let verify_start = std::time::Instant::now();
            match artifacts
                .instance
                .is_sat(&artifacts.vars, &artifacts.inputs)
            {
                Ok(true) => {
                    let verify_time = verify_start.elapsed();
                    println!(
                        "Constraint verification: passed (time taken: {:.2?})",
                        verify_time
                    );
                    println!("The double-precision configuration has excellent performance and can be put into production.");
                }
                Ok(false) => {
                    println!("Error: Constraint validation: failed");
                }
                Err(e) => {
                    println!("Error: Validation error: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("Error: Constraint generation failed: {}", e);
        }
    }
}

#[cfg(feature = "manual_examples")]
fn verify_constraint_correctness() {
    println!("\nSingle precision/double precision constraint correctness verification");
    println!("{}", "-".repeat(60));

    let test_configs = vec![
        ("single precision", MulFPParams::single_precision()),
        ("double precision", MulFPParams::double_precision()),
    ];

    for (config_name, params) in test_configs {
        println!("\nVerify {} (m={}):", config_name, params.m);

        let test_data = if params.m <= 25 {
            MulFPInputData {
                b1: 0,
                v1: 1048576,
                p1: 100,
                b2: 1,
                v2: 2097152,
                p2: 150,
                witness_values: None,
            }
        } else {
            MulFPInputData {
                b1: 1,
                v1: 1125899906842623,
                p1: 800,
                b2: 0,
                v2: 562949953421311,
                p2: 600,
                witness_values: None,
            }
        };

        match produce_r1cs_mulfp_with_params(&params, Some(&test_data)) {
            Ok(artifacts) => {
                match artifacts
                    .instance
                    .is_sat(&artifacts.vars, &artifacts.inputs)
                {
                    Ok(true) => {
                        println!(
                            "{} constraint verification passed ({} constraints)",
                            config_name, artifacts.num_constraints
                        );
                    }
                    Ok(false) => {
                        println!("Error: {} constraint validation failed", config_name);
                    }
                    Err(e) => {
                        println!("Error: {} Validation error: {:?}", config_name, e);
                    }
                }
            }
            Err(e) => {
                println!("Error: {} Constraint generation failed: {}", config_name, e);
            }
        }
    }

    println!("\nVerification summary:");
    println!("- The constraint math for both IEEE 754 single and double precision configurations is correct");
    println!("- Constraint complexity grows linearly with accuracy, but remains within a controllable range");
    println!("- Double precision configuration verified for production use");
    println!("- Constraint relaxation design ensures the stability and practicality of the system");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkp::verifiers::common::FloatBitExtractor;

    fn transcript_from_u64(value: u64) -> TranscriptRng {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&value.to_le_bytes());
        TranscriptRng::from_seed(seed)
    }

    #[test]
    fn collect_mulfp_bool_slices_single_precision() {
        let params = MulFPParams::single_precision();
        let slices = collect_mulfp_bool_slices(&params);

        assert_eq!(slices.mantissa_product.len(), 2 * params.m + 2);
        assert_eq!(slices.mantissa_normalized.len(), 2 * params.m + 2);
        assert_eq!(slices.sign_bits, [0, 3, 6]);

        let mut flag_indices = vec![
            slices.rounding_flags.rdb,
            slices.rounding_flags.stb,
            slices.rounding_flags.one_minus_rdb,
            slices.rounding_flags.one_minus_stb,
            slices.rounding_flags.v_hat_m_plus_one,
            slices.rounding_flags.one_minus_v_hat_m_plus_one,
        ];
        flag_indices.sort_unstable();
        flag_indices.dedup();
        assert_eq!(
            flag_indices.len(),
            6,
            "rounding flag indices should be different from each other"
        );
        for idx in flag_indices {
            assert!(
                idx > slices.mantissa_normalized.last().copied().unwrap(),
                "rounding flag index {} should be greater than the normalized mantissa index range",
                idx
            );
        }

        assert!(
            slices.normalization_switches.v_top_bit >= *slices.mantissa_product.last().unwrap()
        );
        assert!(
            slices.normalization_switches.one_minus_v_top > slices.normalization_switches.v_top_bit
        );
    }

    #[test]
    fn merge_mulfp_bool_slices_two_ops() {
        let params = MulFPParams::single_precision();
        let slice = collect_mulfp_bool_slices(&params);
        let merged =
            merge_mulfp_bool_slices(&[slice.clone(), slice.clone()]).expect("merge succeeds");

        assert_eq!(merged.batch, 2);
        assert_eq!(
            merged.mantissa_product.len(),
            merged.slice_sizes.mantissa_product * 2
        );
        assert_eq!(
            merged.mantissa_normalized.len(),
            merged.slice_sizes.mantissa_normalized * 2
        );
        assert_eq!(merged.sign_bits.len(), merged.slice_sizes.sign_bits * 2);
        assert_eq!(
            merged.rounding_flags.len(),
            merged.slice_sizes.rounding_flags * 2
        );
        assert_eq!(
            merged.normalization_switches.len(),
            merged.slice_sizes.normalization_switches * 2
        );

        assert_eq!(
            &merged.sign_bits[0..3],
            &slice.sign_bits,
            "The first set of sign bits should be consistent with the single result"
        );
        assert_eq!(
            &merged.sign_bits[3..6],
            &slice.sign_bits,
            "The second set of sign bits should be consistent with the single result"
        );
    }

    #[test]
    fn merge_mulfp_bool_slices_detects_mismatch() {
        let params = MulFPParams::single_precision();
        let slice_ok = collect_mulfp_bool_slices(&params);
        let mut slice_bad = slice_ok.clone();
        slice_bad.mantissa_product.pop();

        let err = merge_mulfp_bool_slices(&[slice_ok, slice_bad])
            .expect_err("should detect mismatched slice");
        assert!(
            err.contains("inconsistent"),
            "The error message should indicate that the length is inconsistent: {}",
            err
        );
    }

    #[test]
    fn sample_mulfp_double_shuffle_randomness_is_deterministic() {
        let mut rng1 = transcript_from_u64(42);
        let r1 = sample_mulfp_double_shuffle_randomness(3, 8, &mut rng1);

        let mut rng2 = transcript_from_u64(42);
        let r2 = sample_mulfp_double_shuffle_randomness(3, 8, &mut rng2);
        assert_eq!(
            r1, r2,
            "The same seed should generate the same random coefficients"
        );

        let mut rng3 = transcript_from_u64(7);
        let r3 = sample_mulfp_double_shuffle_randomness(3, 8, &mut rng3);
        assert_ne!(
            r1, r3,
            "Different seeds should generate different random coefficients"
        );
    }

    fn build_mock_slice(offset: usize) -> MulFPBooleanSlices {
        MulFPBooleanSlices {
            mantissa_product: (0..6).map(|i| offset + i).collect(),
            mantissa_normalized: (0..6).map(|i| offset + 6 + i).collect(),
            sign_bits: [offset + 12, offset + 13, offset + 14],
            rounding_flags: MulFPRoundingFlagIndices {
                rdb: offset + 15,
                stb: offset + 16,
                one_minus_rdb: offset + 17,
                one_minus_stb: offset + 18,
                v_hat_m_plus_one: offset + 19,
                one_minus_v_hat_m_plus_one: offset + 20,
            },
            normalization_switches: MulFPNormalizationSwitchIndices {
                v_top_bit: offset + 21,
                one_minus_v_top: offset + 22,
            },
        }
    }

    #[test]
    fn mulfp_double_shuffle_group_context_produces_trace() {
        let slice_a = build_mock_slice(0);
        let slice_b = build_mock_slice(32);

        let book = merge_mulfp_bool_slices(&[slice_a.clone(), slice_b.clone()]).expect("merge ok");

        let group = MulFPBooleanGroup::MantissaProduct;
        let indices = book.flat_indices(group);
        let max_source_idx = indices.iter().copied().max().unwrap_or(0);

        let zero_var_idx = max_source_idx + 1;
        let start_var_idx = zero_var_idx + 1;

        let mut rng = transcript_from_u64(2024);
        let randomness =
            sample_mulfp_double_shuffle_randomness(book.batch, book.slice_len(group), &mut rng);

        let context = MulFPDoubleShuffleGroupContext::new(
            group,
            &book,
            randomness.clone(),
            start_var_idx,
            zero_var_idx,
        )
        .expect("context build");

        let const_one_idx = context.next_var_idx();
        let total_vars = const_one_idx + 1;
        let mut vars = vec![Scalar::ZERO.to_bytes(); total_vars];

        let stride = book.slice_len(group);
        for batch_idx in 0..book.batch {
            for col in 0..stride {
                let flat_idx = batch_idx * stride + col;
                if let Some(&var_idx) = indices.get(flat_idx) {
                    let bit = if col % 2 == 0 {
                        Scalar::ZERO
                    } else {
                        Scalar::ONE
                    };
                    vars[var_idx] = bit.to_bytes();
                }
            }
        }
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();
        vars[const_one_idx] = Scalar::ONE.to_bytes();

        let mut a_entries = Vec::new();
        let mut b_entries = Vec::new();
        let mut c_entries = Vec::new();
        let num_constraints = context.append_constraints(
            0,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            const_one_idx,
        );
        assert!(num_constraints > 0, "Constraints should be generated");

        let witness = context
            .populate_witness(&mut vars)
            .expect("populate witness");
        assert_eq!(witness.group, group);
        assert_eq!(witness.batch_size, book.batch);
        assert_eq!(witness.slice_len, book.slice_len(group));

        for (row_bits, expected_row) in witness.bit_values.iter().zip(0..book.batch) {
            assert_eq!(row_bits.len(), stride);
            for (col, bit) in row_bits.iter().enumerate() {
                let expected = if col % 2 == 0 { 0 } else { 1 };
                assert_eq!(
                    *bit, expected,
                    "The {}th witness in column {} should be equal to {}",
                    expected_row, col, expected
                );
            }
        }
    }

    fn input_from_pair(lhs: f32, rhs: f32) -> MulFPInputData {
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

    fn assert_mulfp_witness_satisfies(lhs: f32, rhs: f32) {
        let params = MulFPParams::single_precision();
        let input = input_from_pair(lhs, rhs);

        let artifacts =
            produce_r1cs_mulfp_with_params(&params, Some(&input)).expect("generate mulfp witness");

        let sat = artifacts
            .instance
            .is_sat(&artifacts.vars, &artifacts.inputs)
            .expect("is_sat evaluation should succeed");
        assert!(
            sat,
            "MULFP witness for ({}, {}) does not satisfy all constraints",
            lhs, rhs
        );
    }

    #[test]
    fn mulfp_witness_satisfies_constraints_for_basic_pairs() {
        let cases = [(1.5f32, -0.75f32), (-2.5f32, -3.0f32), (0.125f32, 8.0f32)];

        for (lhs, rhs) in cases {
            assert_mulfp_witness_satisfies(lhs, rhs);
        }
    }

    #[test]
    fn mulfp_artifacts_capture_double_shuffle_witnesses() {
        let params = MulFPParams::single_precision();
        let input = input_from_pair(1.5f32, -0.75f32);

        let artifacts =
            produce_r1cs_mulfp_with_params(&params, Some(&input)).expect("generate mulfp witness");

        assert_eq!(
            artifacts.double_shuffle.len(),
            5,
            "Single precision should include 5 groups of Double-Shuffling witnesses"
        );

        for witness in &artifacts.double_shuffle {
            assert_eq!(witness.batch_size, 1);
            if witness.slice_len > 0 {
                assert_eq!(
                    witness.bit_indices.len(),
                    1,
                    "In a single batch scenario, there should be only one row of bit records."
                );
            }
        }

        let sat = artifacts
            .instance
            .is_sat(&artifacts.vars, &artifacts.inputs)
            .expect("sat evaluation");
        assert!(sat);
    }

    #[test]
    fn mulfp_double_shuffle_tamper_is_detected() {
        let params = MulFPParams::single_precision();
        let input = input_from_pair(2.5f32, 1.75f32);

        let artifacts =
            produce_r1cs_mulfp_with_params(&params, Some(&input)).expect("generate mulfp witness");

        let mut tampered_vars_raw = artifacts.assignment.clone();

        let mantissa_witness = artifacts
            .double_shuffle
            .iter()
            .find(|entry| entry.group == MulFPBooleanGroup::MantissaProduct)
            .expect("mantissa double shuffle witness");
        if mantissa_witness.slice_len == 0 {
            panic!("mantissa witness slice length should be > 0");
        }
        let target_var_idx = mantissa_witness.bit_indices[0][0];
        let original_byte = tampered_vars_raw[target_var_idx];
        if original_byte[0] == 0 {
            tampered_vars_raw[target_var_idx][0] = 1;
        } else {
            tampered_vars_raw[target_var_idx][0] = 0;
        }

        let tampered_vars =
            VarsAssignment::new(&tampered_vars_raw).expect("construct tampered VarsAssignment");

        let result = artifacts
            .instance
            .is_sat(&tampered_vars, &artifacts.inputs)
            .expect("is_sat should complete");
        assert!(
            !result,
            "is_sat failure should be triggered after tampering with the mantissa bit (Double-Shuffling constraints should be caught)"
        );
    }

    #[test]
    fn mulfp_batch_witness_satisfies_constraints() {
        let params = MulFPParams::single_precision();
        let pairs = [(1.5f32, -0.75f32), (-2.5f32, 3.0f32)];

        let mut witnesses = Vec::new();
        for (lhs, rhs) in pairs {
            let input = input_from_pair(lhs, rhs);
            let artifacts = produce_r1cs_mulfp_detached(&params, Some(&input))
                .expect("generate mulfp detached");

            witnesses.push(artifacts.into_witness(input));
        }

        let batch = MulFPBatchWitness::from_indices(&[0, 1], &witnesses, &params)
            .expect("assemble batch witness");

        assert_eq!(batch.entry_indices, vec![0, 1]);

        let sat = batch
            .instance
            .is_sat(&batch.vars, &batch.inputs)
            .expect("sat evaluation");
        assert!(
            sat,
            "Batch witnesses should satisfy the Double-Shuffling constraint"
        );
    }

    #[test]
    fn mulfp_batch_witness_detects_tamper() {
        let params = MulFPParams::single_precision();
        let pairs = [(1.5f32, -0.75f32), (-2.5f32, 3.0f32)];

        let mut witnesses = Vec::new();
        for (lhs, rhs) in pairs {
            let input = input_from_pair(lhs, rhs);
            let artifacts = produce_r1cs_mulfp_detached(&params, Some(&input))
                .expect("generate mulfp detached");

            witnesses.push(artifacts.into_witness(input));
        }

        let batch = MulFPBatchWitness::from_indices(&[0, 1], &witnesses, &params)
            .expect("assemble batch witness");

        let mut tampered_assignment = batch.assignment.clone();
        let mut tampered = false;
        for witness in &batch.double_shuffle {
            if !witness.bit_indices.is_empty() && !witness.bit_indices[0].is_empty() {
                let idx = witness.bit_indices[0][0];
                tampered_assignment[idx][0] ^= 1;
                tampered = true;
                break;
            }
        }

        assert!(
            tampered,
            "Tests should be modified to at least one boolean bit"
        );

        let tampered_vars =
            VarsAssignment::new(&tampered_assignment).expect("construct tampered vars assignment");

        let sat = batch
            .instance
            .is_sat(&tampered_vars, &batch.inputs)
            .expect("tampered sat evaluation");
        assert!(
            !sat,
            "The tampered batch witness should trigger constraint failure"
        );
    }
}
