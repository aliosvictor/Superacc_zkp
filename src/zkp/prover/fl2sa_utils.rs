use crate::zkp::utils::fl2sa::{
    produce_r1cs_fl2sa_with_result, superacc_from_components_i64, FL2SAInputData, FL2SAParams,
    Fl2saWitness, FloatPrecision,
};
use crate::zkp::utils::mulfp::MulFPInputData;
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};

pub fn derive_fl2sa_params(config: &GCNZKPConfig) -> FL2SAParams {
    let sa = &config.sa2fl_params;
    FL2SAParams {
        w: config.fl2sa_w as usize,
        e: sa.e,
        m: sa.m,
        lgw: sa.lgw,
        acc: sa.acc.max(32),
    }
}

pub fn fl2sa_witness_from_value(
    config: &GCNZKPConfig,
    params: &FL2SAParams,
    value: f64,
    label: &str,
) -> Result<Fl2saWitness, ZKPError> {
    let precision = config.precision_mode;
    let (effective_value, effective_precision) =
        promote_fl2sa_value(value, precision, config.fl2sa_m);
    let quantized = FloatBitExtractor::quantize_to_precision(effective_value, effective_precision);
    let bits = FloatBitExtractor::decompose(quantized, effective_precision);
    let mut mantissa_blocks =
        FloatBitExtractor::float_to_mantissa_blocks(quantized, config.fl2sa_w, effective_precision);
    normalize_mantissa_blocks(params, &mut mantissa_blocks);

    let input_data = FL2SAInputData::with_params(
        params,
        bits.exponent_bits,
        bits.sign as u32,
        Some(mantissa_blocks.clone()),
    )
    .map_err(ZKPError::FL2SAConversionError)?;

    let (artifacts, _) = produce_r1cs_fl2sa_with_result::<f64>(params, &input_data, precision)
        .map_err(|e| {
            ZKPError::FL2SAConversionError(format!("{} value {}: {}", label, quantized, e))
        })?;

    let superaccumulator = superacc_from_components_i64(
        params,
        bits.exponent_bits,
        bits.sign as u32,
        &mantissa_blocks,
    )
    .map_err(ZKPError::FL2SAConversionError)?;

    Ok(artifacts.into_witness(
        bits.exponent_bits,
        bits.sign as u32,
        mantissa_blocks,
        superaccumulator,
    ))
}

pub fn superacc_from_value(
    config: &GCNZKPConfig,
    params: &FL2SAParams,
    value: f64,
    precision_override: Option<FloatPrecision>,
) -> Result<Vec<i64>, ZKPError> {
    let precision = precision_override.unwrap_or(config.precision_mode);
    let (effective_value, effective_precision) =
        promote_fl2sa_value(value, precision, config.fl2sa_m);
    let quantized = FloatBitExtractor::quantize_to_precision(effective_value, effective_precision);
    let mut mantissa_blocks =
        FloatBitExtractor::float_to_mantissa_blocks(quantized, config.fl2sa_w, effective_precision);
    let bits = FloatBitExtractor::decompose(quantized, effective_precision);
    normalize_mantissa_blocks(params, &mut mantissa_blocks);

    superacc_from_components_i64(
        params,
        bits.exponent_bits,
        bits.sign as u32,
        &mantissa_blocks,
    )
    .map_err(ZKPError::FL2SAConversionError)
}

pub fn mulfp_input_from_pair(lhs: f64, rhs: f64, precision: FloatPrecision) -> MulFPInputData {
    let lhs_bits = FloatBitExtractor::decompose(lhs, precision);
    let rhs_bits = FloatBitExtractor::decompose(rhs, precision);

    MulFPInputData {
        b1: lhs_bits.sign,
        v1: lhs_bits.mantissa,
        p1: lhs_bits.exponent,
        b2: rhs_bits.sign,
        v2: rhs_bits.mantissa,
        p2: rhs_bits.exponent,
        witness_values: None,
    }
}

pub fn promote_fl2sa_value(
    value: f64,
    source_precision: FloatPrecision,
    fl2sa_m: u32,
) -> (f64, FloatPrecision) {
    if matches!(source_precision, FloatPrecision::Half) && fl2sa_m >= 23 {
        ((value as f32) as f64, FloatPrecision::Single)
    } else {
        (value, source_precision)
    }
}

pub fn normalize_mantissa_blocks(params: &FL2SAParams, blocks: &mut Vec<u64>) {
    let expected = params.beta().saturating_sub(1);
    if blocks.len() < expected {
        blocks.resize(expected, 0);
    } else if blocks.len() > expected {
        blocks.truncate(expected);
    }
}
