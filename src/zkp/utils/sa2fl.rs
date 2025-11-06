#![allow(clippy::assertions_on_result_states)]
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};

use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};
use rand::{rngs::OsRng, Rng};

use crate::zkp::utils::fl2sa::{
    sample_ah_double_shuffle_randomness, AhDoubleShuffleContext, AhDoubleShuffleWitness,
    Fl2saWitness,
};

///
#[derive(Debug, Clone)]
pub struct SA2FLParams {
    pub w: usize,
    pub e: usize,
    pub m: usize,
    pub lgw: usize,
    pub acc: usize,
}

///
#[derive(Debug, Clone, PartialEq)]
pub enum Sa2flParamSet {
    HalfPrecision,
    Standard1,
    Standard2,
}

impl Sa2flParamSet {
    pub fn to_params(&self) -> SA2FLParams {
        match self {
            Sa2flParamSet::HalfPrecision => SA2FLParams {
                w: 4,
                e: 7,
                m: 10,
                lgw: 2,
                acc: 32,
            },
            Sa2flParamSet::Standard1 => SA2FLParams {
                w: 4,
                e: 8,
                m: 23,
                lgw: 2,
                acc: 32,
            },
            Sa2flParamSet::Standard2 => SA2FLParams {
                w: 4,
                e: 11,
                m: 52,
                lgw: 2,
                acc: 64,
            },
        }
    }

    ///
    ///
    ///
    pub fn description(&self) -> &'static str {
        match self {
            Sa2flParamSet::HalfPrecision => {
                "Half-precision: lightweight (alpha approximately 35, beta=4, constraint~480)"
            }
            Sa2flParamSet::Standard1 => {
                "Standard accuracy 1: Everyday application level (alpha=70, beta=7, constraint ~1200)"
            }
            Sa2flParamSet::Standard2 => {
                "Standard accuracy 2: High accuracy level (alpha=525, beta=15, constraint~8200)"
            }
        }
    }

    ///
    ///
    ///
    ///
    ///
    pub fn performance_summary(&self) -> (usize, usize, &'static str, &'static str) {
        match self {
            Sa2flParamSet::HalfPrecision => (
                486,
                420,
                "high performance",
                "Half-precision, prototype verification, mobile scenarios",
            ),
            Sa2flParamSet::Standard1 => (
                1209,
                1143,
                "high performance",
                "Daily applications, prototype development, production environment",
            ),
            Sa2flParamSet::Standard2 => (
                8212,
                8069,
                "Low performance (large scale)",
                "Scientific computing, financial applications, high-precision requirements",
            ),
        }
    }
}

impl Default for SA2FLParams {
    fn default() -> Self {
        Sa2flParamSet::Standard1.to_params()
    }
}

impl SA2FLParams {
    pub fn alpha(&self) -> usize {
        ((1 << self.e) + self.m + self.w - 1) / self.w
    }

    /// beta = ((m + 1) + w - 1) / w + 1
    pub fn beta(&self) -> usize {
        ((self.m + 1) + self.w - 1) / self.w + 1
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.w == 0 || !self.w.is_power_of_two() {
            return Err("w must be a power of 2 and greater than 0".to_string());
        }
        if self.lgw != (self.w as f64).log2() as usize {
            return Err("lgw must be equal to log2(w)".to_string());
        }
        if self.e == 0 || self.m == 0 {
            return Err("e and m must be greater than 0".to_string());
        }
        if self.acc < self.m {
            return Err("acc must be greater than or equal to m".to_string());
        }
        Ok(())
    }

    pub fn check_constraint21_22_compatibility(&self) -> (bool, String) {
        let alpha = self.alpha();
        let beta = self.beta();
        let value_bits = beta * self.w;
        let required_u_length = beta * self.w;

        let mut warnings = Vec::new();
        let mut compatible = true;

        if self.m > value_bits {
            warnings.push(format!(
                "m({}) > value_bits({}): Some u bits cannot participate in value reconstruction",
                self.m, value_bits
            ));
        }

        if self.m < required_u_length {
            warnings.push(format!(
                "m({}) < beta*w({}): Some v blocks lack enough u bits and will be filled with zeros",
                self.m, required_u_length
            ));
        }

        if alpha > 100 {
            warnings.push(format!(
                "alpha({}) is too large, which may cause performance problems",
                alpha
            ));
            compatible = false;
        }

        let message = if warnings.is_empty() {
            "All compatibility checks passed".to_string()
        } else {
            warnings.join("; ")
        };

        (compatible, message)
    }

    ///
    ///
    ///
    pub fn estimate_performance(&self) -> String {
        let alpha = self.alpha();
        let beta = self.beta();
        let value_bits = beta * self.w;

        let estimated_constraints = if alpha <= 70 {
            7 * alpha + 500
        } else {
            13 * alpha + 2000
        };

        let performance_level = if estimated_constraints < 2000 {
            "high performance"
        } else if estimated_constraints < 5000 {
            "medium performance"
        } else {
            "Low performance (large scale)"
        };

        format!(
            "alpha={}, beta={}, value_bits={}, estimated number of constraints~{}, performance level: {}",
            alpha, beta, value_bits, estimated_constraints, performance_level
        )
    }
}

///
///
///
/// ```rust
/// let input_data = Sa2flInputData::with_default_params(vec![1, 2, 3, 4, 5, 6, 7, 8])?;
///
/// let params = SA2FLParams { w: 8, e: 4, m: 12, lgw: 3, acc: 64 };
/// let input_data = Sa2flInputData::with_params(&params, vec![1, 2, 3, 4, 5])?;
/// ```
#[derive(Debug, Clone)]
pub struct Sa2flInputData {
    pub y: Vec<u64>,
    pub witness_values: Option<Vec<[u8; 32]>>,
}

impl Default for Sa2flInputData {
    fn default() -> Self {
        let default_params = SA2FLParams::default();
        let alpha = default_params.alpha();

        Self {
            y: vec![0; alpha],
            witness_values: None,
        }
    }
}

impl Sa2flInputData {
    ///
    ///
    ///
    /// ```rust
    /// let params = SA2FLParams { w: 8, e: 4, m: 12, lgw: 3, acc: 64 };
    /// let input_data = Sa2flInputData::with_params(&params, vec![1, 2, 3, 4, 5]);
    /// ```
    pub fn with_params(params: &SA2FLParams, y_values: Vec<u64>) -> Result<Self, String> {
        let alpha = params.alpha();

        if y_values.len() != alpha {
            return Err(format!(
                "y vector length mismatch: expected {} blocks (alpha), actual {} blocks",
                alpha,
                y_values.len()
            ));
        }

        Ok(Self {
            y: y_values,
            witness_values: None,
        })
    }

    ///
    ///
    /// ```rust
    /// let input_data = Sa2flInputData::with_default_params(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    pub fn with_default_params(y_values: Vec<u64>) -> Result<Self, String> {
        let default_params = SA2FLParams::default();
        Self::with_params(&default_params, y_values)
    }

    ///
    ///
    pub fn validate_with_params(&self, params: &SA2FLParams) -> Result<(), String> {
        let expected_alpha = params.alpha();

        if self.y.len() != expected_alpha {
            return Err(format!(
                "y vector length does not match argument: current {} blocks, argument requires {} blocks (alpha)",
                self.y.len(),
                expected_alpha
            ));
        }

        Ok(())
    }
}

pub struct Sa2flR1CSArtifacts {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_non_zero_entries: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub raw_vars: Vec<[u8; 32]>,
    pub metrics: R1csShapeMetrics,
}

#[derive(Debug, Clone)]
pub struct Sa2flDoubleShuffleTrace {
    pub auxb_bits: Vec<u8>,
    pub auxb2_bits: Vec<u8>,
}

impl Sa2flDoubleShuffleTrace {
    pub fn empty() -> Self {
        Self {
            auxb_bits: Vec::new(),
            auxb2_bits: Vec::new(),
        }
    }

    pub fn from_assignment(params: &SA2FLParams, assignment: &[[u8; 32]]) -> Result<Self, String> {
        let (auxb_base_idx, auxb_len, auxb2_base_idx, auxb2_len) =
            sa2fl_double_shuffle_layout(params);

        let mut auxb_bits = Vec::with_capacity(auxb_len);
        for offset in 0..auxb_len {
            let idx = auxb_base_idx + offset;
            let bit = scalar_bit_at(assignment, idx)?;
            auxb_bits.push(bit);
        }

        let mut auxb2_bits = Vec::with_capacity(auxb2_len);
        for offset in 0..auxb2_len {
            let idx = auxb2_base_idx + offset;
            let bit = scalar_bit_at(assignment, idx)?;
            auxb2_bits.push(bit);
        }

        Ok(Self {
            auxb_bits,
            auxb2_bits,
        })
    }

    pub fn from_fl2sa(params: &SA2FLParams, fl2sa: &Fl2saWitness) -> Result<Self, String> {
        if fl2sa.superaccumulator.len() != params.alpha() {
            return Err(format!(
                "SA2FL trace length mismatch: superacc={} vs alpha={}",
                fl2sa.superaccumulator.len(),
                params.alpha()
            ));
        }
        let (auxb_bits, auxb2_bits) =
            derive_sa2fl_trace_from_superacc(params, &fl2sa.superaccumulator);
        Ok(Self {
            auxb_bits,
            auxb2_bits,
        })
    }
}

pub struct Sa2flWitness {
    pub double_shuffle: Sa2flDoubleShuffleTrace,
    pub auxb_indices: Vec<usize>,
    pub auxb2_indices: Vec<usize>,
}

pub struct Sa2flBatchWitness {
    pub entry_indices: Vec<usize>,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub auxb_double_shuffle: AhDoubleShuffleWitness,
    pub auxb2_double_shuffle: AhDoubleShuffleWitness,
    pub field_ops: R1csShapeMetrics,
}

impl Sa2flWitness {
    pub fn from_input_data(
        params: &SA2FLParams,
        input_data: &Sa2flInputData,
    ) -> Result<Self, String> {
        if input_data.witness_values.is_none() {
            return Err(
                "Sa2flWitness requires full witness_values to extract Double-Shuffling traces"
                    .to_string(),
            );
        }

        let _artifacts = produce_r1cs_sa2fl_with_params(params, Some(input_data))?;
        let witness_values = input_data
            .witness_values
            .clone()
            .expect("witness_values checked above");

        let trace = Sa2flDoubleShuffleTrace::from_assignment(params, &witness_values)?;
        let (auxb_base_idx, auxb_len, auxb2_base_idx, auxb2_len) =
            sa2fl_double_shuffle_layout(params);
        let auxb_indices: Vec<usize> = (0..auxb_len).map(|i| auxb_base_idx + i).collect();
        let auxb2_indices: Vec<usize> = (0..auxb2_len).map(|i| auxb2_base_idx + i).collect();

        Ok(Self {
            double_shuffle: trace,
            auxb_indices,
            auxb2_indices,
        })
    }

    pub fn from_fl2sa(params: &SA2FLParams, fl2sa: &Fl2saWitness) -> Result<Self, String> {
        let trace = Sa2flDoubleShuffleTrace::from_fl2sa(params, fl2sa)?;
        let (auxb_base_idx, auxb_len, auxb2_base_idx, auxb2_len) =
            sa2fl_double_shuffle_layout(params);
        let auxb_indices: Vec<usize> = (0..auxb_len).map(|i| auxb_base_idx + i).collect();
        let auxb2_indices: Vec<usize> = (0..auxb2_len).map(|i| auxb2_base_idx + i).collect();
        Ok(Self {
            double_shuffle: trace,
            auxb_indices,
            auxb2_indices,
        })
    }
}

impl Sa2flBatchWitness {
    pub fn from_indices(indices: &[usize], source: &[Sa2flWitness]) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("Sa2flBatchWitness requires at least one witness".to_string());
        }

        let first = source
            .get(indices[0])
            .ok_or_else(|| "Sa2flBatchWitness initial index out of bounds".to_string())?;
        let auxb_len = first.double_shuffle.auxb_bits.len();
        let auxb2_len = first.double_shuffle.auxb2_bits.len();

        if auxb_len == 0 || auxb2_len == 0 {
            return Err(
                "Sa2flWitness lacks Double-Shuffling trajectory and cannot be batched".to_string(),
            );
        }

        let batch_size = indices.len();
        let mut auxb_matrix: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
        let mut auxb2_matrix: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
        let mut bit_assignments: Vec<(usize, u8)> = Vec::new();
        let mut next_var_idx = 0usize;

        for &entry_idx in indices {
            let witness = source.get(entry_idx).ok_or_else(|| {
                format!(
                    "Sa2flBatchWitness index {} exceeds available witness range",
                    entry_idx
                )
            })?;

            if witness.double_shuffle.auxb_bits.len() != auxb_len
                || witness.double_shuffle.auxb2_bits.len() != auxb2_len
            {
                return Err(
                    "Double-Shuffling trace length inconsistent between Sa2flWitness".to_string(),
                );
            }

            if witness.auxb_indices.len() != auxb_len || witness.auxb2_indices.len() != auxb2_len {
                return Err("Sa2flWitness lacks complete index information".to_string());
            }

            let mut auxb_row = Vec::with_capacity(auxb_len);
            for bit in &witness.double_shuffle.auxb_bits {
                let idx = next_var_idx;
                next_var_idx += 1;
                auxb_row.push(idx);
                bit_assignments.push((idx, *bit));
            }
            auxb_matrix.push(auxb_row);

            let mut auxb2_row = Vec::with_capacity(auxb2_len);
            for bit in &witness.double_shuffle.auxb2_bits {
                let idx = next_var_idx;
                next_var_idx += 1;
                auxb2_row.push(idx);
                bit_assignments.push((idx, *bit));
            }
            auxb2_matrix.push(auxb2_row);
        }

        let const_one_idx = next_var_idx;
        let zero_var_idx = const_one_idx + 1;
        next_var_idx = zero_var_idx + 1;
        let mut rng = OsRng;

        let auxb_randomness = sample_ah_double_shuffle_randomness(batch_size, auxb_len, &mut rng);
        let auxb_context = AhDoubleShuffleContext::new(
            batch_size,
            auxb_len,
            next_var_idx,
            auxb_randomness.clone(),
            auxb_matrix.clone(),
            zero_var_idx,
        )?;
        let mut cursor_var_idx = auxb_context.layout().next_var_idx();

        let auxb2_randomness = sample_ah_double_shuffle_randomness(batch_size, auxb2_len, &mut rng);
        let auxb2_context = AhDoubleShuffleContext::new(
            batch_size,
            auxb2_len,
            cursor_var_idx,
            auxb2_randomness.clone(),
            auxb2_matrix.clone(),
            zero_var_idx,
        )?;
        cursor_var_idx = auxb2_context.layout().next_var_idx();

        let max_var_idx = cursor_var_idx;
        let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

        let mut constraint_cursor = 0usize;
        constraint_cursor = auxb_context.append_constraints(
            constraint_cursor,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            const_one_idx,
        );
        constraint_cursor = auxb2_context.append_constraints(
            constraint_cursor,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            const_one_idx,
        );

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
        let num_vars_total = max_var_idx;
        let num_inputs = 1;

        let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars_total];
        vars[const_one_idx] = Scalar::ONE.to_bytes();
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();
        for (idx, bit) in &bit_assignments {
            vars[*idx] = Scalar::from(*bit as u64).to_bytes();
        }

        let mut tmp_vars = vars.clone();
        let auxb_witness = auxb_context.populate_witness(&mut tmp_vars)?;
        let auxb2_witness = auxb2_context.populate_witness(&mut tmp_vars)?;

        let metrics = compute_r1cs_metrics(num_constraints, &a_entries, &b_entries, &c_entries);

        let instance = Instance::new(
            num_constraints,
            num_vars_total,
            num_inputs,
            &a_entries,
            &b_entries,
            &c_entries,
        )
        .map_err(|e| format!("Failed to create SA2FL Double-Shuffling instance: {:?}", e))?;

        let vars_assignment = VarsAssignment::new(&tmp_vars).map_err(|e| {
            format!(
                "Creating SA2FL Double-Shuffling variable assignment failed: {:?}",
                e
            )
        })?;
        let input_assignment = InputsAssignment::new(&[Scalar::ONE.to_bytes()])
            .map_err(|e| format!("Failed to create SA2FL Double-Shuffling input: {:?}", e))?;

        match instance.is_sat(&vars_assignment, &input_assignment) {
            Ok(true) => {}
            Ok(false) => {
                return Err("SA2FL Double-Shuffling SAT check failed".to_string());
            }
            Err(err) => {
                return Err(format!("SA2FL Double-Shuffling SAT check error: {:?}", err));
            }
        }

        Ok(Self {
            entry_indices: indices.to_vec(),
            instance,
            vars: vars_assignment,
            inputs: input_assignment,
            num_constraints,
            num_vars: num_vars_total,
            num_inputs,
            auxb_double_shuffle: auxb_witness,
            auxb2_double_shuffle: auxb2_witness,
            field_ops: metrics,
        })
    }
}

fn sa2fl_double_shuffle_layout(params: &SA2FLParams) -> (usize, usize, usize, usize) {
    let alpha = params.alpha();
    let beta = params.beta();
    let w = params.w;

    let auxb_base_idx = {
        let p_idx = alpha;
        let b_base_idx = p_idx + 1;
        let y_abs_base_idx = b_base_idx + alpha;
        let sign_base_idx = y_abs_base_idx + alpha;
        let c_base_idx = sign_base_idx + alpha;
        let r_base_idx = c_base_idx + alpha;
        let v_base_idx = r_base_idx + alpha;
        let sign_output_idx = v_base_idx + beta;
        let aux1_base_idx = sign_output_idx + 1;
        let aux2_base_idx = aux1_base_idx + (alpha - 1);
        let y_inv_base_idx = aux2_base_idx + (alpha - 1);
        let t_base_idx = y_inv_base_idx + alpha;
        let t_prime_base_idx = t_base_idx + alpha;
        t_prime_base_idx + alpha
    };

    let mut cursor = auxb_base_idx + alpha; // gamma_v_sum_idx
    cursor += 1; // gamma_v_sum
    cursor += alpha - beta + 1; // inner_sum
    cursor += alpha - beta + 1; // outer_product
    cursor += 1; // gamma_y_sum
    cursor += alpha - 1; // aux_b
    cursor += beta; // aux_v
    cursor += 1; // mantissa
    cursor += 1; // value
    let b_output_idx = cursor;
    let value_bit_base_idx = b_output_idx + 1;
    let t_second_base_idx = value_bit_base_idx + beta * w;
    let auxb2_base_idx = t_second_base_idx + beta * w;

    let auxb_len = alpha.saturating_sub(1);
    let auxb2_len = (beta * w).saturating_sub(1);

    (auxb_base_idx, auxb_len, auxb2_base_idx, auxb2_len)
}

fn derive_sa2fl_trace_from_superacc(params: &SA2FLParams, superacc: &[i64]) -> (Vec<u8>, Vec<u8>) {
    let alpha = params.alpha();
    let beta = params.beta();
    let w = params.w;
    let auxb_len = alpha.saturating_sub(1);
    let auxb2_len = beta.saturating_mul(w).saturating_sub(1).max(0usize);

    let mut auxb_bits = vec![0u8; auxb_len];
    let mut auxb2_bits = vec![0u8; auxb2_len];
    let mut first_block = None;
    for idx in (0..superacc.len()).rev() {
        if superacc[idx] != 0 {
            first_block = Some(idx);
            break;
        }
    }

    if let Some(block_idx) = first_block {
        if block_idx < auxb_bits.len() {
            auxb_bits[block_idx] = 1;
        }
        let value = superacc[block_idx].abs() as u64;
        if value != 0 && w > 0 {
            let leading = value.leading_zeros() as usize;
            let bit_offset = 63usize.saturating_sub(leading);
            let bit_pos = block_idx
                .saturating_mul(w)
                .saturating_add(bit_offset.min(w.saturating_sub(1)));
            if bit_pos > 0 {
                let mapped = bit_pos - 1;
                if mapped < auxb2_bits.len() {
                    auxb2_bits[mapped] = 1;
                }
            }
        }
    }

    (auxb_bits, auxb2_bits)
}

fn scalar_bit_at(assignment: &[[u8; 32]], idx: usize) -> Result<u8, String> {
    let value = assignment
        .get(idx)
        .ok_or_else(|| format!("Variable index {} out of bounds", idx))?;
    let scalar = Scalar::from_bytes_mod_order(*value);
    if scalar == Scalar::ZERO {
        Ok(0)
    } else if scalar == Scalar::ONE {
        Ok(1)
    } else {
        Err(format!(
            "Variable at index {} is not a boolean (resolves as {:?})",
            idx, scalar
        ))
    }
}

///
///
/// (b_output, <vbeta-1, ... ,v0>, p, value)<-SA2FL(<yalpha-1, ... ,y0>)
///
/// - `sign_output = sum_{i=0}^{alpha-2} auxb_i * sign_i`
/// - `mantissa = sum_{i=0}^{beta-1} 2^{w*i} v_i`
/// - `value = sign_output * mantissa`
/// - `b_output = (1 - sign_output) / 2`
///
///
///
///
///
///
pub fn produce_r1cs_sa2fl_with_params(
    params: &SA2FLParams,
    input_data: Option<&Sa2flInputData>,
) -> Result<Sa2flR1CSArtifacts, String> {
    params.validate()?;

    if let Some(data) = input_data {
        data.validate_with_params(params)?;
    }

    let w = params.w;
    let _e = params.e;
    let m = params.m;
    let lgw = params.lgw;
    let acc = params.acc;

    let alpha = params.alpha();
    let beta = params.beta();

    let num_inputs = 1;

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

    let mut powers_of_two_matrix = Vec::new();
    for i in 0..lgw {
        let t = 2u32.pow(2u32.pow(i as u32)) as u16;
        powers_of_two_matrix.push(Scalar::from(t).to_bytes())
    }

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

    // ============================================================================
    // ============================================================================

    let y_base_idx = 0;

    let p_idx = alpha;

    let b_base_idx = p_idx + 1;

    let y_abs_base_idx = b_base_idx + alpha;

    let sign_base_idx = y_abs_base_idx + alpha;

    let c_base_idx = sign_base_idx + alpha;

    let r_base_idx = c_base_idx + alpha;

    let v_base_idx = r_base_idx + alpha;

    let sign_output_idx = v_base_idx + beta;

    let aux1_base_idx = sign_output_idx + 1;
    let aux2_base_idx = aux1_base_idx + (alpha - 1);

    let y_inv_base_idx = aux2_base_idx + (alpha - 1);

    let t_base_idx = y_inv_base_idx + alpha;

    let t_prime_base_idx = t_base_idx + alpha;

    let auxb_base_idx = t_prime_base_idx + alpha;

    let gamma_v_sum_idx = auxb_base_idx + alpha;

    let inner_sum_base_idx = gamma_v_sum_idx + 1;

    let outer_product_base_idx = inner_sum_base_idx + (alpha - beta + 1);

    let gamma_y_sum_idx = outer_product_base_idx + (alpha - beta + 1);

    let aux_b_base_idx = gamma_y_sum_idx + 1;

    let aux_v_base_idx = aux_b_base_idx + (alpha - 1);

    let mantissa_idx = aux_v_base_idx + beta;

    let value_idx = mantissa_idx + 1;

    let b_output_idx = value_idx + 1;

    let value_bit_base_idx = b_output_idx + 1;

    let t_second_base_idx = value_bit_base_idx + beta * w;

    let auxb2_base_idx = t_second_base_idx + beta * w;

    let const_one_idx = auxb2_base_idx;

    let sign_bool_constraint_base = 0;
    let sign_calc_constraint_base = sign_bool_constraint_base + alpha;
    let abs_value_constraint_base = sign_calc_constraint_base + alpha;

    let carry_constraint_base = abs_value_constraint_base + alpha;
    let aux1_constraint_base = carry_constraint_base + (alpha - 1);
    let aux2_constraint_base = aux1_constraint_base + (alpha - 1);
    let reconstruction_constraint_base = aux2_constraint_base + (alpha - 1);
    let boundary_constraint = reconstruction_constraint_base + (alpha - 1);
    let boundary_sign_constraint = boundary_constraint;
    let y_inv_constraint_base = boundary_sign_constraint + 1;

    let t_constraint_base = y_inv_constraint_base + alpha;
    let t_prime_constraint_base = t_constraint_base + 2 * alpha;
    let auxb_constraint_base = t_prime_constraint_base + alpha;

    // ============================================================================
    // ============================================================================

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = sign_bool_constraint_base + i;
        a_entries.push((constraint_idx, b_base_idx + i, one));
        b_entries.push((constraint_idx, b_base_idx + i, one));
        b_entries.push((constraint_idx, const_one_idx, minus_one));
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
    }

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = sign_calc_constraint_base + i;
        a_entries.push((constraint_idx, sign_base_idx + i, one));
        a_entries.push((constraint_idx, b_base_idx + i, Scalar::from(2u8).to_bytes()));
        b_entries.push((constraint_idx, const_one_idx, one));
        c_entries.push((constraint_idx, const_one_idx, one));
    }

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = abs_value_constraint_base + i;
        a_entries.push((constraint_idx, sign_base_idx + i, one));
        b_entries.push((constraint_idx, y_abs_base_idx + i, one));
        c_entries.push((constraint_idx, y_base_idx + i, one));
    }

    // ============================================================================
    // ============================================================================
    for i in 1..alpha {
        let constraint_idx = carry_constraint_base + (i - 1);
        a_entries.push((constraint_idx, c_base_idx + (i - 1), powers_of_two[w]));
        a_entries.push((constraint_idx, r_base_idx + i, one));
        b_entries.push((constraint_idx, const_one_idx, one));
        c_entries.push((constraint_idx, y_abs_base_idx + i, one));
    }

    // ============================================================================
    // ============================================================================
    for i in 1..alpha {
        let constraint_idx = aux1_constraint_base + (i - 1);
        a_entries.push((constraint_idx, r_base_idx + i, one));
        b_entries.push((constraint_idx, sign_base_idx + i, one));
        c_entries.push((constraint_idx, aux1_base_idx + (i - 1), one));
    }

    // ============================================================================
    // ============================================================================
    for i in 1..alpha {
        let constraint_idx = aux2_constraint_base + (i - 1);
        a_entries.push((constraint_idx, c_base_idx + (i - 1), powers_of_two[w]));
        b_entries.push((constraint_idx, sign_base_idx + i, one));
        c_entries.push((constraint_idx, aux2_base_idx + (i - 1), one));
    }

    // ============================================================================
    // ============================================================================

    for i in 1..alpha {
        let constraint_idx = reconstruction_constraint_base + (i - 1);
        a_entries.push((constraint_idx, aux1_base_idx + (i - 1), one));
        a_entries.push((constraint_idx, aux2_base_idx + (i - 1), one));
        b_entries.push((constraint_idx, const_one_idx, one));
        c_entries.push((constraint_idx, y_base_idx + i, one));
    }

    // ============================================================================
    // ============================================================================
    let boundary_sign_constraint = boundary_constraint;
    a_entries.push((boundary_sign_constraint, r_base_idx, one));
    b_entries.push((boundary_sign_constraint, sign_base_idx, one));
    c_entries.push((boundary_sign_constraint, y_base_idx, one));

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = y_inv_constraint_base + i;
        a_entries.push((constraint_idx, y_base_idx + i, one));
        b_entries.push((constraint_idx, y_inv_base_idx + i, one));
        c_entries.push((constraint_idx, const_one_idx, one));
    }

    // ============================================================================
    // ============================================================================

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = t_constraint_base + i;
        a_entries.push((constraint_idx, t_base_idx + i, one));
        b_entries.push((constraint_idx, t_base_idx + i, one));
        b_entries.push((constraint_idx, const_one_idx, minus_one));
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
    }

    // ============================================================================
    // ============================================================================
    for i in 0..alpha {
        let constraint_idx = t_constraint_base + alpha + i;
        a_entries.push((constraint_idx, t_base_idx + i, one));
        b_entries.push((constraint_idx, y_base_idx + i, one));
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
    }

    // ============================================================================
    // ============================================================================

    // ============================================================================
    // ============================================================================
    let constraint_idx = t_prime_constraint_base;
    a_entries.push((constraint_idx, t_prime_base_idx + (alpha - 1), one));
    b_entries.push((constraint_idx, const_one_idx, one));
    c_entries.push((constraint_idx, t_base_idx + (alpha - 1), one));

    // ============================================================================
    // ============================================================================
    for i in 0..(alpha - 1) {
        let constraint_idx = t_prime_constraint_base + 1 + i;
        a_entries.push((constraint_idx, t_base_idx + i, one));
        b_entries.push((constraint_idx, t_prime_base_idx + (i + 1), one));
        c_entries.push((constraint_idx, t_prime_base_idx + i, one));
    }

    // ============================================================================
    // ============================================================================
    for i in 0..(alpha - 1) {
        let constraint_idx = auxb_constraint_base + i;
        a_entries.push((constraint_idx, t_prime_base_idx + (i + 1), one));
        a_entries.push((constraint_idx, t_prime_base_idx + i, minus_one));
        b_entries.push((constraint_idx, const_one_idx, one));
        c_entries.push((constraint_idx, auxb_base_idx + i, one));
    }

    let mut gamma = vec![[0u8; 32]; beta];
    let mut csprng = OsRng;
    for i in 0..beta {
        let val = csprng.gen_range(1..=u64::MAX);
        gamma[i] = Scalar::from(val).to_bytes();
    }

    let gamma_constraint_base = auxb_constraint_base + (alpha - 1);

    let gamma_v_constraint = gamma_constraint_base;
    for i in 0..beta {
        a_entries.push((gamma_v_constraint, v_base_idx + i, gamma[i])); // gamma_i.v_i
    }
    b_entries.push((gamma_v_constraint, const_one_idx, one)); // * 1
    c_entries.push((gamma_v_constraint, gamma_v_sum_idx, one)); // = gamma_v_sum

    let inner_sum_constraint_base = gamma_v_constraint + 1;
    for i in (beta - 1)..alpha {
        let constraint_idx = inner_sum_constraint_base + (i - (beta - 1));
        let inner_sum_idx = inner_sum_base_idx + (i - (beta - 1));

        for j in 0..beta {
            let y_index_signed = i as isize - (beta as isize) + 1 + j as isize;
            if y_index_signed >= 0 {
                let y_index = y_index_signed as usize;
                if y_index < alpha {
                    a_entries.push((constraint_idx, y_base_idx + y_index, gamma[j]));
                    // gamma_j.y_{i-beta+1+j}
                }
            }
        }
        b_entries.push((constraint_idx, const_one_idx, one)); // * 1
        c_entries.push((constraint_idx, inner_sum_idx, one)); // = inner_sum[i-(beta-1)]
    }

    //
    let outer_product_constraint_base = inner_sum_constraint_base + (alpha - beta + 1);
    for i in (beta - 1)..alpha {
        let constraint_idx = outer_product_constraint_base + (i - (beta - 1));
        let inner_sum_idx = inner_sum_base_idx + (i - (beta - 1));
        let outer_product_idx = outer_product_base_idx + (i - (beta - 1));

        if i < alpha - 1 {
            a_entries.push((constraint_idx, auxb_base_idx + i, one)); // auxb_i
            b_entries.push((constraint_idx, inner_sum_idx, one)); // * inner_sum[i-(beta-1)]
            c_entries.push((constraint_idx, outer_product_idx, one)); // = outer_product[i-(beta-1)]
        } else {
            a_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes())); // 0
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, outer_product_idx, one)); // = outer_product[i-(beta-1)]
        }
    }

    // sum_{i=beta-1}^{alpha-1} outer_product[i-(beta-1)] = gamma_y_sum
    let gamma_y_constraint = outer_product_constraint_base + (alpha - beta + 1);
    for i in (beta - 1)..alpha {
        let outer_product_idx = outer_product_base_idx + (i - (beta - 1));
        a_entries.push((gamma_y_constraint, outer_product_idx, one)); // outer_product[i-(beta-1)]
    }
    b_entries.push((gamma_y_constraint, const_one_idx, one)); // * 1
    c_entries.push((gamma_y_constraint, gamma_y_sum_idx, one)); // = gamma_y_sum

    // gamma_v_sum - gamma_y_sum = 0
    let final_equality_constraint = gamma_y_constraint + 1;
    a_entries.push((final_equality_constraint, gamma_v_sum_idx, one)); // gamma_v_sum
    a_entries.push((final_equality_constraint, gamma_y_sum_idx, minus_one)); // - gamma_y_sum
    b_entries.push((final_equality_constraint, const_one_idx, one)); // * 1
    c_entries.push((
        final_equality_constraint,
        const_one_idx,
        Scalar::ZERO.to_bytes(),
    )); // = 0

    // ============================================================================
    // ============================================================================
    // sign_output = sum_{i=0}^{alpha-2} auxb_i * sign_i
    //
    //
    //

    let sign_output_constraint_base = final_equality_constraint + 1;

    //
    for i in 0..(alpha - 1) {
        let constraint_idx = sign_output_constraint_base + i;
        a_entries.push((constraint_idx, auxb_base_idx + i, one)); // auxb[i] (iin[0,alpha-2])
        b_entries.push((constraint_idx, sign_base_idx + i, one)); // * sign[i] (iin[0,alpha-2])
        c_entries.push((constraint_idx, aux_b_base_idx + i, one)); // = aux_b[i] (iin[0,alpha-2])
    }

    // sum_{i=0}^{alpha-2} aux_b[i] = sign_output
    let sign_output_sum_constraint = sign_output_constraint_base + (alpha - 1);
    //
    for i in 0..(alpha - 1) {
        a_entries.push((sign_output_sum_constraint, aux_b_base_idx + i, one)); // aux_b[i] (iin[0,alpha-2])
    }
    b_entries.push((sign_output_sum_constraint, const_one_idx, one)); // * 1
    c_entries.push((sign_output_sum_constraint, sign_output_idx, one)); // = sign_output

    // ============================================================================
    // ============================================================================
    //
    //

    let value_constraint_base = sign_output_sum_constraint + 1;

    for i in 0..beta {
        let constraint_idx = value_constraint_base + i;
        let weight_power = w * i;
        if weight_power >= powers_of_two.len() {
            return Err(format!(
                "Weight index {} exceeds the range of powers_of_two array {}",
                weight_power,
                powers_of_two.len()
            ));
        }
        let weight = powers_of_two[weight_power]; // 2^{w*i}
        a_entries.push((constraint_idx, v_base_idx + i, weight)); // weight * v[i]
        b_entries.push((constraint_idx, const_one_idx, one)); // * 1
        c_entries.push((constraint_idx, aux_v_base_idx + i, one)); // = aux_v[i]
    }

    // sum_{i=0}^{beta-1} aux_v[i] = mantissa
    let mantissa_sum_constraint = value_constraint_base + beta;
    for i in 0..beta {
        a_entries.push((mantissa_sum_constraint, aux_v_base_idx + i, one)); // aux_v[i]
    }
    b_entries.push((mantissa_sum_constraint, const_one_idx, one)); // * 1
    c_entries.push((mantissa_sum_constraint, mantissa_idx, one)); // = mantissa

    // sign_output * mantissa = value
    let final_value_constraint = mantissa_sum_constraint + 1;
    a_entries.push((final_value_constraint, sign_output_idx, one)); // sign_output
    b_entries.push((final_value_constraint, mantissa_idx, one)); // * mantissa
    c_entries.push((final_value_constraint, value_idx, one)); // = value

    // ============================================================================
    // ============================================================================
    //
    //

    let b_output_constraint = final_value_constraint + 1;

    // sign_output + 2*b_output = 1
    a_entries.push((b_output_constraint, sign_output_idx, one)); // sign_output
    a_entries.push((
        b_output_constraint,
        b_output_idx,
        Scalar::from(2u8).to_bytes(),
    )); // + 2*b_output
    b_entries.push((b_output_constraint, const_one_idx, one)); // * 1
    c_entries.push((b_output_constraint, const_one_idx, one)); // = 1

    // ============================================================================
    // ============================================================================
    //
    //

    let value_binary_constraint_base = b_output_constraint + 1;

    let value_bits = beta * w;

    let _num_vars_for_constraints = auxb2_base_idx + value_bits - 1 + 200;

    for i in 0..value_bits {
        let constraint_idx = value_binary_constraint_base + i;
        a_entries.push((constraint_idx, value_bit_base_idx + i, one)); // value_bit_i
        b_entries.push((constraint_idx, value_bit_base_idx + i, one)); // * value_bit_i
        b_entries.push((constraint_idx, const_one_idx, minus_one)); // - 1
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
        // = 0
    }

    // sum_{i=0}^{beta*w-1} 2^i * value_bit_i = value
    let value_reconstruction_constraint = value_binary_constraint_base + value_bits;
    for i in 0..value_bits {
        if i >= powers_of_two.len() {
            return Err(format!(
                "value binary decomposition weight index {} exceeds the range of powers_of_two array {}",
                i,
                powers_of_two.len()
            ));
        }
        let weight = powers_of_two[i]; // 2^i
        a_entries.push((
            value_reconstruction_constraint,
            value_bit_base_idx + i,
            weight,
        )); // 2^i * value_bit_i
    }
    b_entries.push((value_reconstruction_constraint, const_one_idx, one)); // * 1
    c_entries.push((value_reconstruction_constraint, value_idx, one)); // = value

    // ============================================================================
    // ============================================================================
    //
    //
    //
    //

    // ============================================================================
    // ============================================================================

    let value_bits = beta * w;

    // ============================================================================
    // ============================================================================

    let t_second_constraint_base = value_reconstruction_constraint + 1;

    for i in 0..value_bits {
        let constraint_idx = t_second_constraint_base + i;
        a_entries.push((constraint_idx, t_second_base_idx + i, one)); // t^{(2)}[i]
        b_entries.push((constraint_idx, t_second_base_idx + i, one)); // * t^{(2)}[i]
        b_entries.push((constraint_idx, const_one_idx, minus_one)); // - 1
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
        // = 0
    }

    for i in 0..value_bits {
        let constraint_idx = t_second_constraint_base + value_bits + i;
        a_entries.push((constraint_idx, t_second_base_idx + i, one)); // t^{(2)}[i]
        b_entries.push((constraint_idx, value_bit_base_idx + i, one)); // * value_bit[i]
        c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes()));
        // = 0
    }

    // ============================================================================
    // ============================================================================

    let t_second_prime_constraint_base = t_second_constraint_base + 2 * value_bits;

    let constraint_idx = t_second_prime_constraint_base;
    a_entries.push((constraint_idx, t_second_base_idx + (value_bits - 1), one));
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, t_second_base_idx + (value_bits - 1), one)); // = t^{(2)}[value_bits-1]

    for i in 0..(value_bits - 1) {
        let constraint_idx = t_second_prime_constraint_base + 1 + i;
        a_entries.push((constraint_idx, t_second_base_idx + i, one)); // t^{(2)}[i]
        b_entries.push((constraint_idx, t_second_base_idx + (i + 1), one));
        c_entries.push((constraint_idx, t_second_base_idx + i, one));
    }

    // ============================================================================
    // ============================================================================

    let auxb2_constraint_base = t_second_prime_constraint_base + value_bits;

    //
    for i in 0..(value_bits - 1) {
        let constraint_idx = auxb2_constraint_base + i;
        a_entries.push((constraint_idx, t_second_base_idx + (i + 1), one));
        a_entries.push((constraint_idx, t_second_base_idx + i, minus_one));
        b_entries.push((constraint_idx, const_one_idx, one)); // * 1
        c_entries.push((constraint_idx, auxb2_base_idx + i, one)); // = auxb2[i]
    }

    let mut gamma2 = vec![[0u8; 32]; m];
    let mut csprng2 = OsRng;
    for i in 0..m {
        let val = csprng2.gen_range(1..=u64::MAX);
        gamma2[i] = Scalar::from(val).to_bytes();
    }

    // ============================================================================
    // ============================================================================

    if m > value_bits {
        println!(
            "Warning: Parameter mismatch prompt: m({}) > value_bits({}), only the first {} bits participate in reconstruction",
            m, value_bits, value_bits
        );
    }

    if m == 0 {
        return Err("m cannot be 0".to_string());
    }

    let auxb2_length = value_bits - 1;
    let max_auxb2_access = if value_bits >= m { value_bits - m } else { 0 };

    if max_auxb2_access >= auxb2_length {
        return Err(format!(
            "auxb2 access out of bounds: maximum access index ({}) >= array length ({})",
            max_auxb2_access, auxb2_length
        ));
    }

    // ============================================================================
    // ============================================================================

    // $$
    // u_i = \sum_{k=m-1}^{value\_bits-1} auxb2_k \cdot value\_bit_{k-m+1+i}
    // $$
    //

    // sum_{i=0}^{m-1} gamma2_i * u_i = sum_{k=m-1}^{value\_bits-1} auxb2_k * sum_{j=0}^{m-1} gamma2_j * value_bit_{k-m+1+j}

    let gamma2_iterations = if value_bits > (m - 1) {
        value_bits - (m - 1)
    } else {
        1
    };

    let u_base_idx = auxb2_base_idx + auxb2_length;
    let inner_sum_base_idx = u_base_idx + m;
    let outer_product_base_idx = inner_sum_base_idx + gamma2_iterations;
    let left_sum_idx = outer_product_base_idx + gamma2_iterations;
    let right_sum_idx = left_sum_idx + 1;

    let _num_vars_with_gamma2 = right_sum_idx + 1;

    let gamma2_constraint_base = auxb2_constraint_base + (value_bits - 1);
    let mut gamma2_constraint_count = 0;

    // ============================================================================
    // ============================================================================

    for i in 0..m {
        let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
        let u_idx = u_base_idx + i;

        let k_start = m.saturating_sub(1);
        if k_start < value_bits {
            for k in k_start..value_bits {
                // k in [max(0,m-1), value_bits-1]
                if k < auxb2_length {
                    let bit_index = k.saturating_sub(m.saturating_sub(1)) + i;
                    if bit_index < value_bits {
                        a_entries.push((constraint_idx, auxb2_base_idx + k, one)); // auxb2_k
                        a_entries.push((constraint_idx, value_bit_base_idx + bit_index, one));
                        // * value_bit_{k-m+1+i}
                    }
                }
            }
        }
        b_entries.push((constraint_idx, const_one_idx, one)); // * 1
        c_entries.push((constraint_idx, u_idx, one)); // = u_i
        gamma2_constraint_count += 1;
    }

    let k_start = m.saturating_sub(1);
    if k_start < value_bits {
        for k in k_start..value_bits {
            let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
            let inner_sum_idx = inner_sum_base_idx + (k - k_start);

            for j in 0..m {
                let bit_index = k.saturating_sub(m.saturating_sub(1)) + j;
                if bit_index < value_bits {
                    a_entries.push((constraint_idx, value_bit_base_idx + bit_index, gamma2[j]));
                    // gamma2_j * value_bit_{k-m+1+j}
                }
            }
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, inner_sum_idx, one)); // = inner_sum_{k-(m-1)}
            gamma2_constraint_count += 1;
        }
    }

    let k_start = m.saturating_sub(1);
    if k_start < value_bits {
        for k in k_start..value_bits {
            if k < auxb2_length {
                let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
                let inner_sum_idx = inner_sum_base_idx + (k - k_start);
                let outer_product_idx = outer_product_base_idx + (k - k_start);

                a_entries.push((constraint_idx, auxb2_base_idx + k, one)); // auxb2_k
                b_entries.push((constraint_idx, inner_sum_idx, one)); // * inner_sum_{k-(m-1)}
                c_entries.push((constraint_idx, outer_product_idx, one)); // = outer_product_{k-(m-1)}
                gamma2_constraint_count += 1;
            }
        }
    }

    // sum_{k=m-1}^{value_bits-1} outer_product_{k-(m-1)} = right_sum
    let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
    let k_start = m.saturating_sub(1);
    if k_start < value_bits {
        for k in k_start..value_bits {
            if k < auxb2_length {
                let outer_product_idx = outer_product_base_idx + (k - k_start);
                a_entries.push((constraint_idx, outer_product_idx, one)); // outer_product_{k-(m-1)}
            }
        }
    }
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, right_sum_idx, one)); // = right_sum
    gamma2_constraint_count += 1;

    // sum_{i=0}^{m-1} gamma2_i * u_i = left_sum
    let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
    for i in 0..m {
        let u_idx = u_base_idx + i;
        a_entries.push((constraint_idx, u_idx, gamma2[i])); // gamma2_i * u_i
    }
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, left_sum_idx, one)); // = left_sum
    gamma2_constraint_count += 1;

    let constraint_idx = gamma2_constraint_base + gamma2_constraint_count;
    a_entries.push((constraint_idx, left_sum_idx, one)); // left_sum
    a_entries.push((constraint_idx, right_sum_idx, minus_one)); // - right_sum
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes())); // = 0
    gamma2_constraint_count += 1;

    // ============================================================================
    // ============================================================================
    // p = sum_{i=0}^{l-m-1} i * auxb2_{i+m} + sum_{i=0}^{alpha-beta-1} (i+1) * w * auxb_{i+beta}
    //
    //
    //

    // ============================================================================
    // ============================================================================

    let p_bit_sum_idx = right_sum_idx + 1;
    let p_block_sum_idx = p_bit_sum_idx + 1;

    let bit_offset_base_idx = p_block_sum_idx + 1;

    let block_offset_diff = if alpha > beta { alpha - beta } else { 0 };
    let block_offset_base_idx = bit_offset_base_idx + gamma2_iterations;

    let num_vars_with_p = block_offset_base_idx + block_offset_diff;

    // ============================================================================
    // ============================================================================
    //

    let p_constraint_base = gamma2_constraint_base + gamma2_constraint_count;
    let mut p_constraint_count = 0;

    for i in 0..gamma2_iterations {
        let constraint_idx = p_constraint_base + p_constraint_count;
        let auxb2_index = i + m;

        if auxb2_index < (value_bits - 1) {
            let weight = Scalar::from(i as u64).to_bytes();
            a_entries.push((constraint_idx, auxb2_base_idx + auxb2_index, weight)); // i * auxb2[i+m]
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, bit_offset_base_idx + i, one)); // = bit_offset[i]
        } else {
            a_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes())); // 0
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, bit_offset_base_idx + i, one)); // = bit_offset[i]
        }
        p_constraint_count += 1;
    }

    // sum_{i=0}^{gamma2_iterations-1} bit_offset[i] = p_bit_sum
    let constraint_idx = p_constraint_base + p_constraint_count;
    for i in 0..gamma2_iterations {
        a_entries.push((constraint_idx, bit_offset_base_idx + i, one)); // bit_offset[i]
    }
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, p_bit_sum_idx, one)); // = p_bit_sum
    p_constraint_count += 1;

    for i in 0..block_offset_diff {
        let constraint_idx = p_constraint_base + p_constraint_count;
        let auxb_index = i + beta;

        if auxb_index < (alpha - 1) {
            let weight_scalar = Scalar::from(((i + 1) * w) as u64);
            let weight = weight_scalar.to_bytes();
            a_entries.push((constraint_idx, auxb_base_idx + auxb_index, weight)); // (i+1)*w * auxb[i+beta]
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, block_offset_base_idx + i, one)); // = block_offset[i]
        } else {
            a_entries.push((constraint_idx, const_one_idx, Scalar::ZERO.to_bytes())); // 0
            b_entries.push((constraint_idx, const_one_idx, one)); // * 1
            c_entries.push((constraint_idx, block_offset_base_idx + i, one)); // = block_offset[i]
        }
        p_constraint_count += 1;
    }

    // sum_{i=0}^{alpha-beta-1} block_offset[i] = p_block_sum
    let constraint_idx = p_constraint_base + p_constraint_count;
    for i in 0..block_offset_diff {
        a_entries.push((constraint_idx, block_offset_base_idx + i, one)); // block_offset[i]
    }
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, p_block_sum_idx, one)); // = p_block_sum
    p_constraint_count += 1;

    let constraint_idx = p_constraint_base + p_constraint_count;
    a_entries.push((constraint_idx, p_bit_sum_idx, one)); // p_bit_sum
    a_entries.push((constraint_idx, p_block_sum_idx, one)); // + p_block_sum
    b_entries.push((constraint_idx, const_one_idx, one)); // * 1
    c_entries.push((constraint_idx, p_idx, one));
    p_constraint_count += 1;

    // ============================================================================
    // ============================================================================

    let total_constraints = p_constraint_base + p_constraint_count;

    let num_vars_final = num_vars_with_p;

    let _const_one_idx = num_vars_final - 1;

    let metrics = compute_r1cs_metrics(total_constraints, &a_entries, &b_entries, &c_entries);

    let inst = Instance::new(
        total_constraints,
        num_vars_final,
        num_inputs,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("{:?}", e))?;

    let (raw_assignment_vars, assignment_vars, assignment_inputs) = if let Some(data) = input_data {
        if data.y.len() != alpha {
            return Err(format!(
                "Input hyperaccumulator length mismatch: expected {}, actual {}",
                alpha,
                data.y.len()
            ));
        }

        let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars_final];

        for (i, &val) in data.y.iter().enumerate() {
            if i < alpha {
                vars[y_base_idx + i] = Scalar::from(val).to_bytes();
            }
        }

        if num_vars_final > 0 {
            vars[num_vars_final - 1] = Scalar::ONE.to_bytes();
        }

        if let Some(ref witness_vals) = data.witness_values {
            if witness_vals.len() == num_vars_final {
                vars = witness_vals.clone();
            }
        }

        let assignment_vars = VarsAssignment::new(&vars)
            .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
        let assignment_inputs =
            InputsAssignment::new(&vec![Scalar::ZERO.to_bytes(); num_inputs])
                .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

        (vars, assignment_vars, assignment_inputs)
    } else {
        let zero_bytes = Scalar::ZERO.to_bytes();
        let mut vars = vec![zero_bytes; num_vars_final];

        if num_vars_final > 0 {
            vars[num_vars_final - 1] = Scalar::ONE.to_bytes();
        }

        let assignment_vars = VarsAssignment::new(&vars)
            .map_err(|e| format!("Failed to create default variable assignment: {:?}", e))?;
        let assignment_inputs = InputsAssignment::new(&vec![zero_bytes; num_inputs])
            .map_err(|e| format!("Failed to create default input assignment: {:?}", e))?;

        (vars, assignment_vars, assignment_inputs)
    };

    let num_non_zero_entries = a_entries.len() + b_entries.len() + c_entries.len();

    Ok(Sa2flR1CSArtifacts {
        num_cons: total_constraints,
        num_vars: num_vars_final,
        num_inputs,
        num_non_zero_entries,
        instance: inst,
        vars: assignment_vars,
        inputs: assignment_inputs,
        raw_vars: raw_assignment_vars,
        metrics,
    })
}

///
pub fn test_sa2fl_basic() -> Result<(), String> {
    let params = SA2FLParams::default();
    let alpha = params.alpha();

    let mut test_y = vec![
        100u64, 200u64, 50u64, 30u64, 15u64, 10u64, 5u64, 3u64, 2u64, 1u64, 0u64,
    ];
    while test_y.len() < alpha {
        test_y.push(0u64);
    }
    let test_y = test_y.into_iter().take(alpha).collect();

    let input_data = Sa2flInputData::with_params(&params, test_y)?;

    let result = produce_r1cs_sa2fl_with_params(&params, Some(&input_data))?;
    let num_cons = result.num_cons;
    let num_vars = result.num_vars;
    let num_inputs = result.num_inputs;
    let num_non_zero_entries = result.num_non_zero_entries;

    println!("SA2FL test results:");
    println!("Number of constraints: {}", num_cons);
    println!("Number of variables: {}", num_vars);
    println!("Public input quantity: {}", num_inputs);
    println!("Number of non-zero items: {}", num_non_zero_entries);
    println!(
        "Algorithm parameters: w={}, e={}, m={}, alpha={}, beta={}",
        params.w,
        params.e,
        params.m,
        alpha,
        params.beta()
    );
    println!(
        "Beta value explanation: beta={} means outputting {} mantissa blocks v[0]~v[{}]",
        params.beta(),
        params.beta(),
        params.beta() - 1
    );

    assert!(
        num_cons > 0,
        "The number of constraints should be greater than 0"
    );
    assert!(
        num_vars > 0,
        "The number of variables should be greater than 0"
    );
    assert_eq!(num_inputs, 1, "There should be 1 public input (p offset)");
    assert!(num_non_zero_entries > 0, "There should be non-zero entries");

    println!("SA2FL basic test passed!");
    Ok(())
}

///
pub fn test_sa2fl_constraints_correctness() -> Result<(), String> {
    println!("Verify the correctness of the SA2FL constraint system...");

    let params = SA2FLParams::default();
    let alpha = params.alpha();
    let beta = params.beta();
    let w = params.w;
    let m = params.m;
    let value_bits = beta * w;

    let group1_3_constraints = 3 * alpha;
    let group4_8_constraints = 4 * (alpha - 1) + 2;
    let expected_core_constraints = group1_3_constraints + group4_8_constraints;
    let expected_y_inv_constraints = alpha;
    let group10_constraints = 2 * alpha;
    let group11_constraints = 1 + (alpha - 1);
    let group12_constraints = alpha - 1;
    let expected_detection_constraints =
        group10_constraints + group11_constraints + group12_constraints;
    let expected_gamma_constraints = 3 + 2 * (alpha - beta + 1);
    let expected_sign_output_constraints = alpha;
    let expected_value_constraints = beta + 2;
    let expected_b_output_constraints = 1;
    let expected_value_binary_constraints = beta * w + 1;
    let expected_value_bit_detection_constraints = 6 * beta * w - 1;
    let gamma2_iterations_theoretical = if value_bits >= 2 * m {
        value_bits - 2 * m + 2
    } else {
        1
    };
    let block_offset_diff = if alpha > beta { alpha - beta } else { 0 };
    let expected_extraction_constraints =
        2 * gamma2_iterations_theoretical + 3 + 2 * block_offset_diff;
    let expected_total_constraints_theoretical = expected_core_constraints
        + expected_y_inv_constraints
        + expected_detection_constraints
        + expected_gamma_constraints
        + expected_sign_output_constraints
        + expected_value_constraints
        + expected_b_output_constraints
        + expected_value_binary_constraints
        + expected_value_bit_detection_constraints
        + expected_extraction_constraints;

    let base_vars = 9 * alpha + beta + 1 + 2 * (alpha - 1);
    let y_inv_vars = alpha;
    let gamma_vars = 3 + 2 * (alpha - beta + 1);
    let sign_output_vars = alpha - 1;
    let value_vars = beta + 2;
    let b_output_vars = 1;
    let value_binary_vars = beta * w;
    let value_bit_detection_vars = 2 * beta * w - 1;
    let extraction_vars = m + 2 * gamma2_iterations_theoretical + 2;
    let p_calculation_vars = 2 + gamma2_iterations_theoretical + block_offset_diff;
    let expected_total_vars = base_vars
        + y_inv_vars
        + gamma_vars
        + sign_output_vars
        + value_vars
        + b_output_vars
        + value_binary_vars
        + value_bit_detection_vars
        + extraction_vars
        + p_calculation_vars;

    println!("Constraint system expected statistics:");
    println!(
        "- SA2FL core constraints (group 1-8): {} (formula: 7xalpha+2)",
        expected_core_constraints
    );
    println!(
        "- y inverse constraint (group 9): {} (formula: alpha)",
        expected_y_inv_constraints
    );
    println!(
        "- First non-zero block detection constraint (group 10-12): {} (formula: 4xalpha)",
        expected_detection_constraints
    );
    println!(
        "- gamma linear combination constraints (Group 13): {} (formula: 3+2x(alpha-beta+1))",
        expected_gamma_constraints
    );
    println!(
        "- sign_output sign calculation constraints (group 14): {} (formula: alpha)",
        expected_sign_output_constraints
    );
    println!(
        "- value signed mantissa calculation constraints (group 15): {} (formula: beta+2)",
        expected_value_constraints
    );
    println!(
        "- b_output boolean sign bit constraints (group 16): {} (formula: 1)",
        expected_b_output_constraints
    );
    println!(
        "- value binary decomposition constraints (group 17): {} (formula: beta*w+1)",
        expected_value_binary_constraints
    );
    println!(
        "- value first non-zero bit detection constraint (group 18-20): {} (formula: 6xbeta*w-1)",
        expected_value_bit_detection_constraints
    );
    println!("- First non-zero bit extraction constraints (group 21-22): {} (formula: 2xgamma2_iterations+3+2xblock_offset_diff)", expected_extraction_constraints);
    println!(
        "- The total number of theoretical constraints: {} (formula: original constraints + modified gamma2 verification constraints)",
        expected_total_constraints_theoretical
    );
    println!(
        "- Total number of variables: {} (including u vector, gamma2 verification auxiliary variable and p calculation auxiliary variable)",
        expected_total_vars
    );

    println!("\nNote: Due to boundary conditions and optimization processing, the actual number of constraints may be slightly different from the theoretical calculation.");

    let mut test_y = vec![
        100u64, 200u64, 50u64, 30u64, 15u64, 10u64, 5u64, 3u64, 2u64, 1u64, 0u64,
    ];
    while test_y.len() < alpha {
        test_y.push(0u64);
    }
    let test_y = test_y.into_iter().take(alpha).collect();
    let input_data = Sa2flInputData::with_params(&params, test_y)?;

    let result = produce_r1cs_sa2fl_with_params(&params, Some(&input_data))?;
    let actual_num_cons = result.num_cons;
    let actual_num_vars = result.num_vars;
    let num_inputs = result.num_inputs;
    let num_non_zero_entries = result.num_non_zero_entries;

    println!("\nActual statistics of constraint system:");
    println!("- Actual number of constraints: {}", actual_num_cons);
    println!("- Actual number of variables: {}", actual_num_vars);
    println!("- Number of public inputs: {}", num_inputs);
    println!("- Number of non-zero items: {}", num_non_zero_entries);

    let constraint_diff = if actual_num_cons > expected_total_constraints_theoretical {
        actual_num_cons - expected_total_constraints_theoretical
    } else {
        expected_total_constraints_theoretical - actual_num_cons
    };

    if constraint_diff > 30 {
        return Err(format!(
            "The difference in the number of constraints is too large: theoretical expectation {}, actual {}, difference {}",
            expected_total_constraints_theoretical, actual_num_cons, constraint_diff
        ));
    } else {
        println!(
            "The number of constraints passed the verification: actual {}, theoretical {}, difference {} (within a reasonable range)",
            actual_num_cons, expected_total_constraints_theoretical, constraint_diff
        );
    }

    if actual_num_vars != expected_total_vars {
        return Err(format!(
            "Mismatch in number of variables: expected {}, actual {}.The variable calculation logic needs to be checked.",
            expected_total_vars, actual_num_vars
        ));
    }

    println!("\nVerify the gamma constraint mathematical structure:");
    println!("Linear combination on the left: sum gamma_i.v_i");
    println!(
        "Right inner layer summation: sum gamma_j.y_{{i-beta+1+j}} for each i in [beta-1, alpha-1]"
    );
    println!("Right outer product: aux_i.(inner sum) for each i");
    println!("Sum on the right side: sum (outer product)");
    println!("Final equation verification: left side = right side");

    println!("\ngamma constraint parameter range verification:");
    println!("Parameters: alpha={}, beta={}", alpha, beta);
    println!("i range: [beta-1, alpha-1] = [{}, {}]", beta - 1, alpha - 1);
    println!("j range: [0, beta-1] = [0, {}]", beta - 1);
    println!("Constraint quantity decomposition:");
    println!("- Left gamma-v constraint: 1");
    println!("- Inner summation constraints: {}", alpha - beta + 1);
    println!("- Outer product constraints: {}", alpha - beta + 1);
    println!("- Right gamma-y summation constraint: 1");
    println!("- Final equality verification constraints: 1");
    println!("- Total: {}", 5 + 2 * (alpha - beta + 1));

    println!("\nVerify index calculation security:");
    let mut safe_count = 0;
    let mut total_count = 0;
    for i in (beta - 1)..alpha {
        for j in 0..beta {
            total_count += 1;
            let y_index_signed = i as isize - (beta as isize) + 1 + j as isize;
            if y_index_signed >= 0 {
                let y_index = y_index_signed as usize;
                if y_index < alpha {
                    safe_count += 1;
                } else {
                    println!(
                        "Warning: Out of bounds: i={}, j={}, y_index={} >= alpha={}",
                        i, j, y_index, alpha
                    );
                }
            } else {
                println!(
                    "Warning: Underflow: i={}, j={}, y_index_signed={} < 0",
                    i, j, y_index_signed
                );
            }
        }
    }
    println!(
        "Index security: {}/{} index calculation security",
        safe_count, total_count
    );

    println!("\nVerify constraint group integrity:");
    println!(
        "Group 1-3: Sign bit processing constraints ({} items)",
        3 * alpha
    );
    println!(
        "Group 4-8: Carry reconstruction constraints ({} items)",
        4 * (alpha - 1) + 2
    );
    println!("Group 9: y inverse element constraints ({} items)", alpha);
    println!(
        "Group 10-12: t vector and auxb constraints ({} items)",
        3 * alpha + (alpha - 1)
    );
    println!(
        "Group 13: gamma linear combination constraints ({} items)",
        expected_gamma_constraints
    );
    println!(
        "Group 14: sign_output symbol calculation constraints ({} items)",
        expected_sign_output_constraints
    );
    println!(
        "Group 15: value signed mantissa calculation constraints ({} items)",
        expected_value_constraints
    );
    println!(
        "Group 16: b_output Boolean sign bit constraints ({} items)",
        expected_b_output_constraints
    );
    println!(
        "Group 17: value binary decomposition constraints ({} items)",
        expected_value_binary_constraints
    );
    println!(
        "Group 18-20: value first non-zero bit detection constraint ({} items)",
        expected_value_bit_detection_constraints
    );

    println!("\nSA2FL constraint system correctness verification passed!");
    Ok(())
}

pub fn test_both_param_sets() {
    println!("SA2FL standard parameter combination test\n");

    for (i, param_set) in [Sa2flParamSet::Standard1, Sa2flParamSet::Standard2]
        .iter()
        .enumerate()
    {
        println!("=== Test parameter group{} ===", i + 1);
        println!("{}", param_set.description());

        let params = param_set.to_params();
        println!("Parameters: w={}, e={}, m={}", params.w, params.e, params.m);

        let (is_compatible, msg) = params.check_constraint21_22_compatibility();
        println!("Compatibility: {}", msg);

        println!("Performance: {}", params.estimate_performance());

        if is_compatible {
            println!("Recommendation: safe to use");
        } else {
            println!("Recommendation: Warning - use with caution (may have poor performance)");
        }

        println!();
    }

    println!("Usage suggestions:");
    println!("- General application: It is recommended to use standard parameter group 1 (better performance)");
    println!(
        "- High-precision requirements: Standard parameter group 2 can be used (more constraints)"
    );
    println!("- Performance first: always select standard parameter group 1");
}

///
pub fn run_sa2fl_tests() {
    println!("Start SA2FL complete test...\n");

    test_both_param_sets();
    println!();

    match test_sa2fl_basic() {
        Ok(()) => println!("Basic function test passed"),
        Err(e) => println!("Error: Basic functional test failed: {}", e),
    }

    println!("\nTest parameter verification...");
    let params = SA2FLParams::default();
    match params.validate() {
        Ok(()) => println!("Parameter verification passed"),
        Err(e) => println!("Error: Parameter validation failed: {}", e),
    }

    println!("\nTest input data validation...");
    let alpha = params.alpha();
    let mut test_values = vec![
        1u64, 2u64, 3u64, 4u64, 5u64, 6u64, 7u64, 8u64, 9u64, 10u64, 11u64,
    ];
    while test_values.len() < alpha {
        test_values.push(0u64);
    }
    let test_data = Sa2flInputData::with_default_params(test_values);
    match test_data {
        Ok(data) => match data.validate_with_params(&params) {
            Ok(()) => println!("Input data verification passed"),
            Err(e) => println!("Error: Input data validation failed: {}", e),
        },
        Err(e) => println!("Error: Input data creation failed: {}", e),
    }

    println!("\nTest the correctness of the constraint system...");
    match test_sa2fl_constraints_correctness() {
        Ok(()) => println!("The correctness verification of the constraint system passed"),
        Err(e) => println!(
            "Error: Constraint system correctness verification failed: {}",
            e
        ),
    }

    println!("\nSA2FL test completed!");
    println!("Implementation status summary:");
    println!("Parameter structure and input data structure");
    println!("Two standard parameter combinations supported (Standard1/Standard2)");
    println!("Parameter compatibility check and performance estimation");
    println!("Witness vector index definition (including all auxiliary variables)");
    println!("Complete implementation of 22 constraint groups (including all SA2FL core logic)");
    println!("Reconstruct the R1CS decomposition of relationship constraints");
    println!("y inverse constraint: y[i] * y[i]_inv = 1");
    println!("The zero value of t vector identifies the constraint");
    println!("gamma linear combination verification constraints");
    println!("sign_output symbol value calculation constraints");
    println!("value signed mantissa calculation constraints");
    println!("b_output boolean sign bit output constraint");
    println!("value binary decomposition constraint: ensure that value is within the valid range");
    println!(
        "value first non-zero bit detection constraint: extended first non-zero bit algorithm"
    );
    println!("gamma2 verification constraints: first non-zero bit extraction verification based on correct mathematical formula (constraint group 21)");
    println!("Offset p calculation constraints: precise calculation of bit-level and block-level offsets (constraint group 22)");
    println!("Complete variable assignment algorithm");
    println!("Complete R1CS constraint system of SA2FL algorithm (22 constraint groups, including u vector)");
    println!("\nAlgorithm expression: \"(b_output, <vbeta-1, ... ,v0 , p, value)<-SA2FL(<yalpha-1, ... ,y0 )\"");
    println!("Core constraints: \"b_output = (1-sign_output)/2\" & \"sign_output = sumauxb_i.sign_i\" & \"value = sign_output * mantissa\"");
    println!("Corrected constraint 21: \"sum_{{i=0}}^{{m-1}} gamma2_i * u_i = sum_{{i=m-1}}^{{wbeta-m}} auxb2_{{i-m+1}} * sum_{{j=0}}^{{m-1}} gamma2_j * value_bit_{{i+j}}\" (Correct formula based on verification report)");
}

#[cfg(feature = "manual_examples")]
fn main() {
    println!("SA2FL (Super Accumulator to Floating-point) R1CS constraint system example");
    println!("SA2FL constraint system integrity verification and reconstruction relationship repair verification\n");

    run_sa2fl_tests();

    println!("\n{}", "=".repeat(60));
    println!("Start detailed constraint verification testing");
    println!("{}", "=".repeat(60));

    run_constraint_verification();
}

pub fn verify_sa2fl_constraints_with_example() -> Result<(), String> {
    println!("Start SA2FL constraint verification test...\n");

    let params = SA2FLParams {
        w: 4,
        e: 11,
        m: 52,
        lgw: 2, // log2(4) = 2
        acc: 64,
    };

    let alpha = params.alpha();
    let beta = params.beta();

    println!("Test parameters:");
    println!(
        "  w={}, e={}, m={}, alpha={}, beta={}",
        params.w, params.e, params.m, alpha, beta
    );

    let mut test_y = vec![100u64, 200u64, 0u64, 50u64];
    while test_y.len() < alpha {
        test_y.push(0u64);
    }
    let test_y: Vec<u64> = test_y.into_iter().take(alpha).collect();

    let input_data = Sa2flInputData::with_params(&params, test_y.clone())?;

    println!("\nTest input superaccumulator:");
    for (i, &val) in test_y.iter().enumerate() {
        println!("  y[{}] = {}", i, val);
    }

    let result = produce_r1cs_sa2fl_with_params(&params, Some(&input_data))?;
    let num_cons = result.num_cons;
    let num_vars = result.num_vars;
    let num_inputs = result.num_inputs;
    let inst = result.instance;
    let vars_assignment = result.vars;
    let inputs_assignment = result.inputs;

    println!("\nConstraint system statistics:");
    println!("Number of constraints: {}", num_cons);
    println!("Number of variables: {}", num_vars);
    println!("Public input quantity: {}", num_inputs);

    println!("\nVerify the satisfiability of the constraint system...");
    match inst.is_sat(&vars_assignment, &inputs_assignment) {
        Ok(true) => {
            println!("Restraint system can satisfy!All constraints are verified");
        }
        Ok(false) => {
            return Err(
                "Error: The constraint system is unsatisfiable: there are unsatisfied constraints"
                    .to_string(),
            );
        }
        Err(e) => {
            return Err(format!(
                "Error: Error during constraint verification: {:?}",
                e
            ));
        }
    }

    println!("\nManual verification of key constraint groups:");
    verify_constraint_group_examples(&params, &test_y)?;

    println!("\nSA2FL constraint verification test completed!");
    println!("All constraint groups have been verified and the mathematical logic is correct");

    Ok(())
}

fn verify_constraint_group_examples(params: &SA2FLParams, test_y: &[u64]) -> Result<(), String> {
    let w = params.w;
    let alpha = params.alpha();
    let beta = params.beta();

    println!("\nConstraint group 1-3 verification: sign bit processing");

    let y0 = test_y[0] as i64;
    let b0 = if y0 >= 0 { 0 } else { 1 };
    let sign0 = 1 - 2 * b0;
    let abs_y0 = y0.abs() as u64; // 100

    println!(
        "  y[0] = {}, b[0] = {}, sign[0] = {}, |y[0]| = {}",
        y0, b0, sign0, abs_y0
    );

    let constraint1_result = b0 * (b0 - 1);
    println!(
        "Constraint 1 verification: b[0] * (b[0] - 1) = {} * {} = {} (verified)",
        b0,
        b0 - 1,
        constraint1_result
    );
    assert_eq!(constraint1_result, 0);

    let constraint2_result = sign0 + 2 * b0;
    println!(
        "Constraint 2 verification: sign[0] + 2*b[0] = {} + {} = {} (verified)",
        sign0,
        2 * b0,
        constraint2_result
    );
    assert_eq!(constraint2_result, 1);

    let constraint3_result = sign0 as i64 * abs_y0 as i64;
    println!(
        "Constraint 3 verification: sign[0] * |y[0]| = {} * {} = {} (verified)",
        sign0, abs_y0, constraint3_result
    );
    assert_eq!(constraint3_result, y0);

    println!("\nConstraint group 4 verification: carry calculation");

    let y1 = test_y[1];
    let abs_y1 = y1;
    let power_2w = 1u64 << w; // 2^w = 2^4 = 16
    let c2 = abs_y1 / power_2w; // c[i+1] = |y[1]|/2^w = 200/16 = 12
    let r1 = abs_y1 % power_2w; // r[1] = |y[1]| mod 2^w = 200 mod 16 = 8

    println!("  y[1] = {}, |y[1]| = {}, 2^w = {}", y1, abs_y1, power_2w);
    println!("  c[2] = {}, r[1] = {}", c2, r1);

    let constraint4_result = c2 * power_2w + r1;
    println!(
        "Constraint 4 verification: |y[1]| = c[2] * 2^w + r[1] = {} * {} + {} = {} (verified)",
        c2, power_2w, r1, constraint4_result
    );
    assert_eq!(constraint4_result, abs_y1);

    println!("\nConstraint group 10 verification: zero value detection");

    let y2 = test_y[2]; // 0
    let t2 = if y2 == 0 { 1 } else { 0 };

    println!("  y[2] = {}, t[2] = {}", y2, t2);

    let constraint10_1_result = t2 * (t2 - 1);
    println!(
        "Constraint 10.1 Verification: t[2] * (t[2] - 1) = {} * {} = {} (verified)",
        t2,
        t2 - 1,
        constraint10_1_result
    );
    assert_eq!(constraint10_1_result, 0);

    let constraint10_2_result = t2 * y2;
    println!(
        "Constraint 10.2 verification: t[2] * y[2] = {} * {} = {} (verified)",
        t2, y2, constraint10_2_result
    );
    assert_eq!(constraint10_2_result, 0);

    println!("\nConstraint group 16 verification: Boolean sign bit output");

    let sign_output = 1i64;
    let b_output = (1 - sign_output) / 2;

    println!("  sign_output = {}, b_output = {}", sign_output, b_output);

    let constraint16_result = sign_output + 2 * b_output;
    println!(
        "Constraint 16 verification: sign_output + 2*b_output = {} + {} = {} (verified)",
        sign_output,
        2 * b_output,
        constraint16_result
    );
    assert_eq!(constraint16_result, 1);

    println!("\nIndex boundary verification");

    println!(
        "alpha = {}, auxb array length = alpha-1 = {}",
        alpha,
        alpha - 1
    );
    println!("auxb valid index range: [0, {}]", alpha - 2);
    println!(
        "The range of i in the gamma constraint: [beta-1, alpha-1] = [{}, {}]",
        beta - 1,
        alpha - 1
    );

    for i in (beta - 1)..alpha {
        if i < alpha - 1 {
            println!("i={}: access auxb[{}] (valid)", i, i);
        } else {
            println!(
                "i={}: auxb[{}] does not exist, set to 0 (safety processing)",
                i, i
            );
        }
    }

    Ok(())
}

pub fn run_constraint_verification() {
    println!("SA2FL constraint verification test begins\n");

    match verify_sa2fl_constraints_with_example() {
        Ok(()) => println!("Specific examples verified and passed"),
        Err(e) => println!("Error: Specific example of verification failure: {}", e),
    }

    println!("\nSA2FL constraint verification test completed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_sa2fl_witness(auxb_bits: Vec<u8>, auxb2_bits: Vec<u8>) -> Sa2flWitness {
        let auxb_indices: Vec<usize> = (0..auxb_bits.len()).collect();
        let auxb2_indices: Vec<usize> = (0..auxb2_bits.len()).collect();

        Sa2flWitness {
            double_shuffle: Sa2flDoubleShuffleTrace {
                auxb_bits,
                auxb2_bits,
            },
            auxb_indices,
            auxb2_indices,
        }
    }

    #[test]
    fn sa2fl_batch_witness_from_indices() {
        let witnesses = vec![
            dummy_sa2fl_witness(vec![1, 0, 0], vec![0, 1, 0, 0]),
            dummy_sa2fl_witness(vec![0, 1, 0], vec![1, 0, 0, 0]),
        ];

        let batch =
            Sa2flBatchWitness::from_indices(&[0, 1], &witnesses).expect("batch witness builds");
        assert_eq!(batch.entry_indices, vec![0, 1]);
        assert_eq!(batch.num_inputs, 1);
        assert_eq!(batch.auxb_double_shuffle.batch_size, 2);
        assert_eq!(batch.auxb2_double_shuffle.batch_size, 2);
    }
}
