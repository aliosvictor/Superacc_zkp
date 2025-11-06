use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};

use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};

pub struct SuperaccLinearTerm {
    pub coefficient: i64,
    pub limbs: Vec<i64>,
}

pub struct LinearRelationWitness {
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_non_zero_entries: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub result_superacc: Vec<i64>,
    pub carry_values: Vec<i64>,
    pub field_ops: R1csShapeMetrics,
}

pub type LinearCombinationWitness = LinearRelationWitness;

fn scalar_from_i64(value: i64) -> Scalar {
    if value >= 0 {
        Scalar::from(value as u64)
    } else {
        -Scalar::from((-value) as u64)
    }
}

fn i128_to_scalar_bytes(value: i128) -> Result<[u8; 32], String> {
    if value > i64::MAX as i128 || value < i64::MIN as i128 {
        return Err(format!("Value outside i64 range: {}", value));
    }
    Ok(scalar_from_i64(value as i64).to_bytes())
}

///
pub fn build_superacc_linear_witness(
    terms: Vec<SuperaccLinearTerm>,
    radix: i64,
) -> Result<LinearRelationWitness, String> {
    if terms.is_empty() {
        return Err("The linear combination term cannot be empty".to_string());
    }

    if radix <= 1 {
        return Err("radix must be greater than 1".to_string());
    }

    let limb_len = terms[0].limbs.len();
    if limb_len == 0 {
        return Err(
            "The superaccumulator length of the linear combination term must be greater than 0"
                .to_string(),
        );
    }

    for term in &terms {
        if term.limbs.len() != limb_len {
            return Err("The length of the superaccumulator must be the same for all linear combination terms".to_string());
        }
    }

    let num_terms = terms.len();
    let term_var_count = num_terms * limb_len;
    let result_base_idx = term_var_count;
    let carry_base_idx = result_base_idx + limb_len;
    let const_one_idx = carry_base_idx + limb_len;
    let num_vars = const_one_idx + 1;
    let num_inputs = 0usize;

    let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars];

    for (term_idx, term) in terms.iter().enumerate() {
        for (limb_idx, &value) in term.limbs.iter().enumerate() {
            let var_idx = term_idx * limb_len + limb_idx;
            vars[var_idx] = scalar_from_i64(value).to_bytes();
        }
    }

    let mut result_superacc = Vec::with_capacity(limb_len);
    let mut carry_values = Vec::with_capacity(limb_len);
    let mut carry_in = 0i128;
    let radix_i128 = radix as i128;

    for limb_idx in 0..limb_len {
        let mut acc = carry_in;
        for term in &terms {
            acc += (term.coefficient as i128) * (term.limbs[limb_idx] as i128);
        }

        let remainder = acc.rem_euclid(radix_i128);
        let carry_out = acc.div_euclid(radix_i128);

        if remainder > i64::MAX as i128 || remainder < i64::MIN as i128 {
            return Err(format!(
                "The remainder of the {}th limb of the linear combination exceeds the i64 range: {}",
                limb_idx, remainder
            ));
        }
        if carry_out > i64::MAX as i128 || carry_out < i64::MIN as i128 {
            return Err(format!(
                "Linear combination of limb {} carries out of i64 range: {}",
                limb_idx, carry_out
            ));
        }

        vars[result_base_idx + limb_idx] = i128_to_scalar_bytes(remainder)?;
        vars[carry_base_idx + limb_idx] = i128_to_scalar_bytes(carry_out)?;

        result_superacc.push(remainder as i64);
        carry_values.push(carry_out as i64);

        carry_in = carry_out;
    }

    vars[const_one_idx] = Scalar::ONE.to_bytes();

    let num_constraints = limb_len;
    let mut a_entries = Vec::with_capacity(num_constraints * (num_terms + 3));
    let mut b_entries = Vec::with_capacity(num_constraints);
    let mut c_entries = Vec::with_capacity(num_constraints);

    let mut non_zero_entries = 0usize;

    for row in 0..limb_len {
        b_entries.push((row, const_one_idx, Scalar::ONE.to_bytes()));
        non_zero_entries += 1;

        for (term_idx, term) in terms.iter().enumerate() {
            let coeff = term.coefficient;
            if coeff == 0 {
                continue;
            }
            let var_idx = term_idx * limb_len + row;
            a_entries.push((row, var_idx, scalar_from_i64(coeff).to_bytes()));
            non_zero_entries += 1;
        }

        if row > 0 {
            a_entries.push((row, carry_base_idx + row - 1, Scalar::ONE.to_bytes()));
            non_zero_entries += 1;
        }

        a_entries.push((row, result_base_idx + row, scalar_from_i64(-1).to_bytes()));
        non_zero_entries += 1;

        c_entries.push((row, carry_base_idx + row, scalar_from_i64(radix).to_bytes()));
        non_zero_entries += 1;
    }

    let metrics = compute_r1cs_metrics(num_constraints, &a_entries, &b_entries, &c_entries);

    let instance = Instance::new(
        num_constraints,
        num_vars,
        num_inputs,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("{:?}", e))?;

    let vars_assignment = VarsAssignment::new(&vars)
        .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
    let inputs_assignment = InputsAssignment::new(&[])
        .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

    if let Ok(false) = instance.is_sat(&vars_assignment, &inputs_assignment) {
        return Err("The constructed linear combination witness does not satisfy SAT".to_string());
    }

    Ok(LinearRelationWitness {
        num_constraints,
        num_vars,
        num_inputs,
        num_non_zero_entries: non_zero_entries,
        instance,
        vars: vars_assignment,
        inputs: inputs_assignment,
        result_superacc,
        carry_values,
        field_ops: metrics,
    })
}

pub fn build_superacc_linear_combination(
    terms: Vec<SuperaccLinearTerm>,
    radix: i64,
) -> Result<LinearRelationWitness, String> {
    build_superacc_linear_witness(terms, radix)
}
