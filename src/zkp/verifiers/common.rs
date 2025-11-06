use crate::types::OperationPrecision;
use crate::zkp::constraint_metrics::R1csShapeMetrics;
use crate::zkp::utils::fl2sa::FloatPrecision;
use crate::zkp::utils::mulfp::MulFPParams;
use crate::zkp::utils::sa2fl::{SA2FLParams, Sa2flParamSet};

use half::f16;
use std::collections::BTreeMap;
use std::time::Duration;

// ============================================================================
// ============================================================================

#[derive(Debug, Clone)]
pub enum ZKPError {
    FL2SAConversionError(String),
    MULFPVerificationError(String),
    SA2FLConversionError(String),
    ConstraintUnsatisfied(String),
    NumericalError(f64),
    DimensionMismatch(String),
    ConfigError(String),
}

impl std::fmt::Display for ZKPError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ZKPError::FL2SAConversionError(msg) => write!(f, "FL2SA conversion error: {}", msg),
            ZKPError::MULFPVerificationError(msg) => write!(f, "MULFP validation error: {}", msg),
            ZKPError::SA2FLConversionError(msg) => write!(f, "SA2FL conversion error: {}", msg),
            ZKPError::ConstraintUnsatisfied(msg) => write!(f, "Constraint not satisfied: {}", msg),
            ZKPError::NumericalError(error) => {
                write!(f, "Numerical error exceeds tolerance: {}", error)
            }
            ZKPError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            ZKPError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for ZKPError {}

// ============================================================================
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLevel {
    Full,
    Optimized,
    Fast,
}

// ============================================================================
// ============================================================================

#[derive(Debug, Clone)]
pub struct GCNZKPConfig {
    pub precision_mode: FloatPrecision,

    pub verification_level: VerificationLevel,

    pub tolerance: f64,

    pub batch_size: usize,

    pub fl2sa_w: u32,
    pub fl2sa_alpha: usize,
    pub fl2sa_m: u32,

    pub mulfp_params: MulFPParams,

    pub sa2fl_params: SA2FLParams,
}

impl GCNZKPConfig {
    pub fn half_precision() -> Self {
        let sa2fl_params = Sa2flParamSet::Standard1.to_params();
        Self {
            precision_mode: FloatPrecision::Half,
            verification_level: VerificationLevel::Full,
            tolerance: 5e-2,
            batch_size: 48,

            fl2sa_w: 4,
            fl2sa_alpha: sa2fl_params.alpha(),
            fl2sa_m: 23,

            mulfp_params: MulFPParams::half_precision(),

            sa2fl_params,
        }
    }

    pub fn single_precision() -> Self {
        Self {
            precision_mode: FloatPrecision::Single,
            verification_level: VerificationLevel::Full,
            tolerance: 1e-4,
            batch_size: 32,

            fl2sa_w: 4,
            fl2sa_alpha: 70,
            fl2sa_m: 23,

            mulfp_params: MulFPParams::single_precision(),

            sa2fl_params: Sa2flParamSet::Standard1.to_params(),
        }
    }

    pub fn double_precision() -> Self {
        Self {
            precision_mode: FloatPrecision::Double,
            verification_level: VerificationLevel::Full,
            tolerance: 1e-8,
            batch_size: 16,

            fl2sa_w: 4,
            fl2sa_alpha: 525,
            fl2sa_m: 52,

            mulfp_params: MulFPParams::double_precision(),

            sa2fl_params: Sa2flParamSet::Standard2.to_params(),
        }
    }

    pub fn alpha(&self) -> usize {
        self.fl2sa_alpha
    }

    pub fn block_width(&self) -> u32 {
        self.fl2sa_w
    }

    pub fn fl2sa_effective_precision(&self) -> FloatPrecision {
        if matches!(self.precision_mode, FloatPrecision::Half) && self.fl2sa_m >= 23 {
            FloatPrecision::Single
        } else {
            self.precision_mode
        }
    }
}

impl Default for GCNZKPConfig {
    fn default() -> Self {
        Self::single_precision()
    }
}

// ============================================================================
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct ConstraintStats {
    pub fl2sa_count: usize,
    pub mulfp_count: usize,
    pub sa2fl_count: usize,
    pub auxiliary_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintKind {
    MulFpCore,
    MulFpBatch,
    Fl2SaCore,
    Fl2SaBatch,
    Sa2FlCore,
    Sa2FlBatch,
    LinearCombination,
    Auxiliary,
}

impl ConstraintKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConstraintKind::MulFpCore => "mulfp_core",
            ConstraintKind::MulFpBatch => "mulfp_batch",
            ConstraintKind::Fl2SaCore => "fl2sa_core",
            ConstraintKind::Fl2SaBatch => "fl2sa_batch",
            ConstraintKind::Sa2FlCore => "sa2fl_core",
            ConstraintKind::Sa2FlBatch => "sa2fl_batch",
            ConstraintKind::LinearCombination => "linear_combination",
            ConstraintKind::Auxiliary => "auxiliary",
        }
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct AddCounter {
    pub fp16: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub fp64: usize,
}

impl AddCounter {
    pub fn record(&mut self, precision: OperationPrecision, count: usize) {
        match precision {
            OperationPrecision::Fp16 => self.fp16 = self.fp16.saturating_add(count),
            OperationPrecision::Bf16 => self.bf16 = self.bf16.saturating_add(count),
            OperationPrecision::Fp32 => self.fp32 = self.fp32.saturating_add(count),
            OperationPrecision::Fp64 => self.fp64 = self.fp64.saturating_add(count),
        }
    }

    pub fn merge(&mut self, other: &AddCounter) {
        self.fp16 = self.fp16.saturating_add(other.fp16);
        self.bf16 = self.bf16.saturating_add(other.bf16);
        self.fp32 = self.fp32.saturating_add(other.fp32);
        self.fp64 = self.fp64.saturating_add(other.fp64);
    }

    pub fn total(&self) -> usize {
        self.fp16
            .saturating_add(self.bf16)
            .saturating_add(self.fp32)
            .saturating_add(self.fp64)
    }
}

#[derive(Debug, Clone, Default)]
pub struct MulCounter {
    pub fp16: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub fp64: usize,
}

impl MulCounter {
    pub fn record(&mut self, precision: OperationPrecision, count: usize) {
        match precision {
            OperationPrecision::Fp16 => self.fp16 = self.fp16.saturating_add(count),
            OperationPrecision::Bf16 => self.bf16 = self.bf16.saturating_add(count),
            OperationPrecision::Fp32 => self.fp32 = self.fp32.saturating_add(count),
            OperationPrecision::Fp64 => self.fp64 = self.fp64.saturating_add(count),
        }
    }

    pub fn merge(&mut self, other: &MulCounter) {
        self.fp16 = self.fp16.saturating_add(other.fp16);
        self.bf16 = self.bf16.saturating_add(other.bf16);
        self.fp32 = self.fp32.saturating_add(other.fp32);
        self.fp64 = self.fp64.saturating_add(other.fp64);
    }

    pub fn total(&self) -> usize {
        self.fp16
            .saturating_add(self.bf16)
            .saturating_add(self.fp32)
            .saturating_add(self.fp64)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FieldOpKindCounter {
    pub adds: AddCounter,
    pub muls: MulCounter,
}

impl FieldOpKindCounter {
    pub fn record(&mut self, precision: OperationPrecision, field_adds: usize, field_muls: usize) {
        if field_adds > 0 {
            self.adds.record(precision, field_adds);
        }
        if field_muls > 0 {
            self.muls.record(precision, field_muls);
        }
    }

    pub fn merge(&mut self, other: &FieldOpKindCounter) {
        self.adds.merge(&other.adds);
        self.muls.merge(&other.muls);
    }

    pub fn total_adds(&self) -> usize {
        self.adds.total()
    }

    pub fn total_muls(&self) -> usize {
        self.muls.total()
    }
}

///
#[derive(Debug, Clone)]
pub struct ConstraintAccumulator {
    pub total_constraints: usize,

    pub stats: ConstraintStats,

    pub add_counter: AddCounter,

    pub mul_counter: MulCounter,

    pub field_add_counter: AddCounter,

    pub field_mul_counter: MulCounter,

    pub field_ops_by_kind: BTreeMap<ConstraintKind, FieldOpKindCounter>,
}

impl ConstraintAccumulator {
    pub fn new() -> Self {
        Self {
            total_constraints: 0,
            stats: ConstraintStats::default(),
            add_counter: AddCounter::default(),
            mul_counter: MulCounter::default(),
            field_add_counter: AddCounter::default(),
            field_mul_counter: MulCounter::default(),
            field_ops_by_kind: BTreeMap::new(),
        }
    }

    pub fn add_fl2sa_constraint(&mut self) {
        self.add_fl2sa_constraints(1);
    }

    pub fn add_fl2sa_constraints(&mut self, count: usize) {
        self.stats.fl2sa_count += count;
        self.total_constraints += count;
    }

    pub fn add_mulfp_constraint(&mut self) {
        self.add_mulfp_constraints(1);
    }

    pub fn add_mulfp_constraints(&mut self, count: usize) {
        self.stats.mulfp_count += count;
        self.total_constraints += count;
    }

    pub fn add_sa2fl_constraint(&mut self) {
        self.add_sa2fl_constraints(1);
    }

    pub fn add_sa2fl_constraints(&mut self, count: usize) {
        self.stats.sa2fl_count += count;
        self.total_constraints += count;
    }

    pub fn add_auxiliary_constraint(&mut self) {
        self.stats.auxiliary_count += 1;
        self.total_constraints += 1;
    }

    pub fn add_linear_constraints(&mut self, count: usize) {
        self.stats.auxiliary_count += count;
        self.total_constraints += count;
    }

    pub fn merge(&mut self, other: ConstraintAccumulator) {
        self.stats.fl2sa_count += other.stats.fl2sa_count;
        self.stats.mulfp_count += other.stats.mulfp_count;
        self.stats.sa2fl_count += other.stats.sa2fl_count;
        self.stats.auxiliary_count += other.stats.auxiliary_count;
        self.total_constraints += other.total_constraints;
        self.add_counter.merge(&other.add_counter);
        self.mul_counter.merge(&other.mul_counter);
        self.field_add_counter.merge(&other.field_add_counter);
        self.field_mul_counter.merge(&other.field_mul_counter);

        for (kind, counter) in other.field_ops_by_kind.iter() {
            self.field_ops_by_kind
                .entry(*kind)
                .or_insert_with(FieldOpKindCounter::default)
                .merge(counter);
        }
    }

    pub fn record_add_ops(&mut self, precision: OperationPrecision, count: usize) {
        if count > 0 {
            self.add_counter.record(precision, count);
        }
    }

    pub fn record_mul_ops(&mut self, precision: OperationPrecision, count: usize) {
        if count > 0 {
            self.mul_counter.record(precision, count);
        }
    }

    pub fn record_field_ops(
        &mut self,
        kind: ConstraintKind,
        precision: OperationPrecision,
        field_adds: usize,
        field_muls: usize,
    ) {
        if field_adds > 0 {
            self.field_add_counter.record(precision, field_adds);
        }
        if field_muls > 0 {
            self.field_mul_counter.record(precision, field_muls);
        }
        if field_adds > 0 || field_muls > 0 {
            self.field_ops_by_kind
                .entry(kind)
                .or_insert_with(FieldOpKindCounter::default)
                .record(precision, field_adds, field_muls);
        }
    }

    pub fn total_add_ops(&self) -> usize {
        self.add_counter.total()
    }

    pub fn total_mul_ops(&self) -> usize {
        self.mul_counter.total()
    }

    pub fn total_field_add_ops(&self) -> usize {
        self.field_add_counter.total()
    }

    pub fn total_field_mul_ops(&self) -> usize {
        self.field_mul_counter.total()
    }

    pub fn record_field_metrics(
        &mut self,
        kind: ConstraintKind,
        precision: OperationPrecision,
        metrics: &R1csShapeMetrics,
    ) {
        if metrics.is_empty() {
            return;
        }
        self.record_field_ops(kind, precision, metrics.field_adds, metrics.field_muls);
    }

    pub fn field_ops_for_kind(&self, kind: ConstraintKind) -> Option<&FieldOpKindCounter> {
        self.field_ops_by_kind.get(&kind)
    }

    pub fn print_statistics(&self) {
        println!("Constraint system statistics:");
        println!("FL2SA constraints: {}", self.stats.fl2sa_count);
        println!("MULFP constraint: {}", self.stats.mulfp_count);
        println!("SA2FL constraints: {}", self.stats.sa2fl_count);
        println!("Auxiliary constraints: {}", self.stats.auxiliary_count);
        println!("Total number of constraints: {}", self.total_constraints);
        println!("Floating point addition: {}", self.total_add_ops());
        println!("Floating point multiplication: {}", self.total_mul_ops());
        println!("Field addition: {}", self.total_field_add_ops());
        println!("Field multiplication: {}", self.total_field_mul_ops());
    }
}

impl Default for ConstraintAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl From<FloatPrecision> for OperationPrecision {
    fn from(value: FloatPrecision) -> Self {
        match value {
            FloatPrecision::Half => OperationPrecision::Fp16,
            FloatPrecision::Single => OperationPrecision::Fp32,
            FloatPrecision::Double => OperationPrecision::Fp64,
        }
    }
}

// ============================================================================
// ============================================================================

pub struct FloatBitExtractor;

pub struct FloatDecomposition {
    pub quantized: f64,
    pub sign: u64,
    pub mantissa: u64,
    pub exponent_bits: u64,
    pub exponent: i64,
}

impl FloatBitExtractor {
    pub fn quantize_to_precision(value: f64, precision: FloatPrecision) -> f64 {
        match precision {
            FloatPrecision::Half => f16::from_f64(value).to_f64(),
            FloatPrecision::Single => (value as f32) as f64,
            FloatPrecision::Double => value,
        }
    }

    pub fn decompose(value: f64, precision: FloatPrecision) -> FloatDecomposition {
        match precision {
            FloatPrecision::Half => {
                let quantized = f16::from_f64(value);
                let bits = quantized.to_bits();
                let sign = ((bits >> 15) & 1) as u64;
                let exponent_bits = ((bits >> 10) & 0x1F) as u64;
                let mantissa = (bits & 0x03FF) as u64;
                let exponent = exponent_bits as i64 - 15;
                FloatDecomposition {
                    quantized: quantized.to_f64(),
                    sign,
                    mantissa,
                    exponent_bits,
                    exponent,
                }
            }
            FloatPrecision::Single => {
                let quantized = value as f32;
                let bits = quantized.to_bits();
                let sign = ((bits >> 31) & 1) as u64;
                let exponent_bits = ((bits >> 23) & 0xFF) as u64;
                let mantissa = (bits & ((1u32 << 23) - 1)) as u64;
                let exponent = exponent_bits as i64 - 127;
                FloatDecomposition {
                    quantized: quantized as f64,
                    sign,
                    mantissa,
                    exponent_bits,
                    exponent,
                }
            }
            FloatPrecision::Double => {
                let quantized = value;
                let bits = quantized.to_bits();
                let sign = ((bits >> 63) & 1) as u64;
                let exponent_bits = ((bits >> 52) & 0x7FF) as u64;
                let mantissa = bits & ((1u64 << 52) - 1);
                let exponent = exponent_bits as i64 - 1023;
                FloatDecomposition {
                    quantized,
                    sign,
                    mantissa,
                    exponent_bits,
                    exponent,
                }
            }
        }
    }

    pub fn float_to_mantissa_blocks(
        value: f64,
        block_width: u32,
        precision: FloatPrecision,
    ) -> Vec<u64> {
        let total_bits = match precision {
            FloatPrecision::Half => 10,
            FloatPrecision::Single => 23,
            FloatPrecision::Double => 52,
        };

        let mantissa = match precision {
            FloatPrecision::Half => {
                let quantized = f16::from_f64(value);
                Self::extract_mantissa_f16(quantized)
            }
            FloatPrecision::Single => {
                let quantized = value as f32;
                Self::extract_mantissa_f32(quantized)
            }
            FloatPrecision::Double => {
                let quantized = value;
                Self::extract_mantissa_f64(quantized)
            }
        };

        Self::split_mantissa(mantissa, total_bits, block_width)
    }

    fn split_mantissa(mantissa: u64, total_bits: u32, block_width: u32) -> Vec<u64> {
        let num_blocks = (total_bits + block_width - 1) / block_width;
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mask = (1u64 << block_width) - 1;

        for i in 0..num_blocks {
            let shift_amount = (i * block_width) as u64;
            let block = (mantissa >> shift_amount) & mask;
            blocks.push(block);
        }

        blocks
    }

    pub fn mantissa_from_blocks(
        blocks: &[u64],
        block_width: u32,
        precision: FloatPrecision,
    ) -> u64 {
        let total_bits = match precision {
            FloatPrecision::Half => 10,
            FloatPrecision::Single => 23,
            FloatPrecision::Double => 52,
        };
        let mut mantissa = 0u128;
        let block_mask = if block_width >= 64 {
            u128::MAX
        } else {
            (1u128 << block_width) - 1
        };
        for (idx, &block) in blocks.iter().enumerate() {
            let shift = (idx as u32).saturating_mul(block_width);
            mantissa |= ((block as u128) & block_mask) << shift;
        }
        if total_bits >= 64 {
            mantissa as u64
        } else {
            (mantissa & ((1u128 << total_bits) - 1)) as u64
        }
    }

    pub fn compose_from_components(
        sign: u32,
        exponent_bits: u64,
        mantissa: u64,
        precision: FloatPrecision,
    ) -> f64 {
        match precision {
            FloatPrecision::Half => {
                let bits: u16 = (((sign & 1) as u16) << 15)
                    | (((exponent_bits as u16) & 0x1F) << 10)
                    | ((mantissa as u16) & 0x03FF);
                f16::from_bits(bits).to_f64()
            }
            FloatPrecision::Single => {
                let bits: u32 = (((sign & 1) as u32) << 31)
                    | (((exponent_bits as u32) & 0xFF) << 23)
                    | ((mantissa as u32) & 0x007F_FFFF);
                f32::from_bits(bits) as f64
            }
            FloatPrecision::Double => {
                let bits: u64 = (((sign as u64) & 1) << 63)
                    | ((exponent_bits & 0x7FF) << 52)
                    | (mantissa & 0x000F_FFFF_FFFF_FFFF);
                f64::from_bits(bits)
            }
        }
    }

    pub fn extract_sign_bit_f16(value: f16) -> u32 {
        let bits = value.to_bits();
        ((bits >> 15) & 1) as u32
    }

    pub fn extract_mantissa_f16(value: f16) -> u64 {
        (value.to_bits() as u64) & 0x03FF
    }

    pub fn extract_exponent_f16(value: f16) -> u64 {
        ((value.to_bits() >> 10) & 0x1F) as u64
    }

    pub fn extract_sign_bit_f32(value: f32) -> u32 {
        let bits = value.to_bits();
        (bits >> 31) & 1
    }

    pub fn extract_mantissa_f32(value: f32) -> u64 {
        let bits = value.to_bits();
        (bits & ((1u32 << 23) - 1)) as u64
    }

    pub fn extract_exponent_f32(value: f32) -> u64 {
        let bits = value.to_bits();
        ((bits >> 23) & 0xFF) as u64
    }

    pub fn extract_sign_bit_f64(value: f64) -> u32 {
        let bits = value.to_bits();
        ((bits >> 63) & 1) as u32
    }

    pub fn extract_mantissa_f64(value: f64) -> u64 {
        let bits = value.to_bits();
        bits & ((1u64 << 52) - 1)
    }

    pub fn extract_exponent_f64(value: f64) -> u64 {
        let bits = value.to_bits();
        (bits >> 52) & 0x7FF
    }

    pub fn float_to_mantissa_blocks_f32(value: f32, block_width: u32) -> Vec<u64> {
        let mantissa = Self::extract_mantissa_f32(value);
        let num_blocks = (23 + block_width - 1) / block_width;

        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mask = (1u64 << block_width) - 1;

        for i in 0..num_blocks {
            let shift_amount = (i * block_width) as u64;
            let block = (mantissa >> shift_amount) & mask;
            blocks.push(block);
        }

        blocks
    }

    pub fn float_to_mantissa_blocks_f64(value: f64, block_width: u32) -> Vec<u64> {
        let mantissa = Self::extract_mantissa_f64(value);
        let num_blocks = (52 + block_width - 1) / block_width;

        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mask = (1u64 << block_width) - 1;

        for i in 0..num_blocks {
            let shift_amount = (i * block_width) as u64;
            let block = (mantissa >> shift_amount) & mask;
            blocks.push(block);
        }

        blocks
    }

    pub fn float_to_offset_f32(value: f32) -> u64 {
        Self::extract_exponent_f32(value)
    }

    pub fn float_to_offset_f16(value: f16) -> u64 {
        Self::extract_exponent_f16(value)
    }

    pub fn float_to_offset_f64(value: f64) -> u64 {
        Self::extract_exponent_f64(value)
    }
}

// ============================================================================
// ============================================================================

pub struct SuperAccumulatorUtils;

impl SuperAccumulatorUtils {
    pub fn zero_superaccumulator(alpha: usize) -> Vec<f64> {
        vec![0.0; alpha]
    }

    ///
    pub fn superaccumulator_to_float(sa: &[f64]) -> f64 {
        sa.iter().sum()
    }

    ///
    pub fn float_to_superaccumulator(value: f64, alpha: usize) -> Vec<f64> {
        let mut sa = vec![0.0; alpha];
        if !sa.is_empty() {
            sa[0] = value;
        }
        sa
    }

    pub fn verify_superaccumulator_sum(sa_list: &[Vec<f64>]) -> Result<Vec<f64>, ZKPError> {
        if sa_list.is_empty() {
            return Err(ZKPError::ConfigError(
                "Superaccumulator list is empty".to_string(),
            ));
        }

        let alpha = sa_list[0].len();
        let mut result = vec![0.0; alpha];

        for sa in sa_list {
            if sa.len() != alpha {
                return Err(ZKPError::DimensionMismatch(format!(
                    "Inconsistent superaccumulator lengths: {} vs {}",
                    sa.len(),
                    alpha
                )));
            }

            for i in 0..alpha {
                result[i] += sa[i];
            }
        }

        Ok(result)
    }
}

// ============================================================================
// ============================================================================

#[derive(Debug, Clone)]
pub struct VerificationStatistics {
    pub layer1_constraints: usize,
    pub layer2_constraints: usize,
    pub total_constraints: usize,
    pub verification_time: Duration,
    pub fl2sa_conversions: usize,
    pub mulfp_verifications: usize,
    pub sa2fl_conversions: usize,
}

impl VerificationStatistics {
    pub fn new() -> Self {
        Self {
            layer1_constraints: 0,
            layer2_constraints: 0,
            total_constraints: 0,
            verification_time: Duration::from_secs(0),
            fl2sa_conversions: 0,
            mulfp_verifications: 0,
            sa2fl_conversions: 0,
        }
    }

    pub fn print(&self) {
        println!("Verification statistics:");
        println!("Layer1 constraints: {}", self.layer1_constraints);
        println!("Layer2 constraints: {}", self.layer2_constraints);
        println!("Total number of constraints: {}", self.total_constraints);
        println!("FL2SA conversion: {}", self.fl2sa_conversions);
        println!("MULFP verification: {}", self.mulfp_verifications);
        println!("SA2FL conversion: {}", self.sa2fl_conversions);
        println!("Verification time: {:?}", self.verification_time);
    }
}

impl Default for VerificationStatistics {
    fn default() -> Self {
        Self::new()
    }
}
