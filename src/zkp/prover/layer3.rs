use std::sync::Arc;

use crate::types::{DenseMatrix, FloatType, OperationPrecision};
use crate::zkp::operation_tracker::{record_add, record_mul};
use crate::zkp::prover::fl2sa_utils::{derive_fl2sa_params, fl2sa_witness_from_value};
use crate::zkp::prover::{DenseLayerWitness, Fl2saBatchWitness, Fl2saWitness, MulFPBatchWitness};
use crate::zkp::utils::mulfp::{produce_r1cs_mulfp_detached, MulFPInputData};
use crate::zkp::utils::sa2fl::{Sa2flBatchWitness, Sa2flWitness};
use crate::zkp::verifiers::common::{FloatBitExtractor, GCNZKPConfig, ZKPError};

pub struct Layer3Prover {
    config: Arc<GCNZKPConfig>,
}

impl Layer3Prover {
    pub fn new(config: Arc<GCNZKPConfig>) -> Self {
        Self { config }
    }

    pub fn generate_witness<T: FloatType>(
        &self,
        activations: &DenseMatrix<T>,
        weights: &DenseMatrix<T>,
    ) -> Result<Layer3Witness, ZKPError> {
        let precision = self.config.precision_mode;
        let (num_nodes, nhid) = activations.shape;
        let (w_in, nclass) = weights.shape;
        let precision_tag: OperationPrecision = precision.into();

        if w_in != nhid {
            return Err(ZKPError::DimensionMismatch(format!(
                "Layer3 weight dimension mismatch: {} vs {}",
                w_in, nhid
            )));
        }

        let total_mulfp = num_nodes
            .checked_mul(nhid)
            .and_then(|v| v.checked_mul(nclass))
            .ok_or_else(|| ZKPError::ConfigError("Layer3 witness number overflow".to_owned()))?;
        let mut mulfp_witness = Vec::with_capacity(total_mulfp);
        let output_len = num_nodes
            .checked_mul(nclass)
            .ok_or_else(|| ZKPError::ConfigError("Layer3 output quantity overflow".to_owned()))?;
        let mut outputs = vec![0.0f64; output_len];

        for node_idx in 0..num_nodes {
            for feat_idx in 0..nhid {
                let act_value = activations
                    .get(node_idx, feat_idx)
                    .to_f64()
                    .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                let act_bits = FloatBitExtractor::decompose(act_value, precision);

                for class_idx in 0..nclass {
                    let weight_value = weights
                        .get(feat_idx, class_idx)
                        .to_f64()
                        .ok_or_else(|| ZKPError::NumericalError(0.0))?;
                    let weight_bits = FloatBitExtractor::decompose(weight_value, precision);

                    let mulfp_input = MulFPInputData {
                        b1: act_bits.sign,
                        v1: act_bits.mantissa,
                        p1: act_bits.exponent,
                        b2: weight_bits.sign,
                        v2: weight_bits.mantissa,
                        p2: weight_bits.exponent,
                        witness_values: None,
                    };

                    let artifacts =
                        produce_r1cs_mulfp_detached(&self.config.mulfp_params, Some(&mulfp_input))
                            .map_err(ZKPError::MULFPVerificationError)?;

                    record_mul(precision_tag, 1);

                    mulfp_witness.push(artifacts.into_witness(mulfp_input));

                    let product_value = FloatBitExtractor::quantize_to_precision(
                        act_value * weight_value,
                        precision,
                    );
                    let idx = node_idx * nclass + class_idx;
                    record_add(precision_tag, 1);
                    outputs[idx] = FloatBitExtractor::quantize_to_precision(
                        outputs[idx] + product_value,
                        precision,
                    );
                }
            }
        }

        let batch_size = self.config.batch_size.max(1);
        let mut mulfp_batches = Vec::new();
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
                        "Failed to assemble Layer3 MULFP batch witness: {}",
                        e
                    ))
                })?;
                mulfp_batches.push(batch);
                current_indices.clear();
            }
        }

        let fl2sa_params = derive_fl2sa_params(&self.config);
        let fl2sa_outputs = outputs
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                fl2sa_witness_from_value(
                    &self.config,
                    &fl2sa_params,
                    *value,
                    &format!("layer3-output-{idx}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fl2sa_output_batches =
            assemble_fl2sa_batches(&fl2sa_outputs, &fl2sa_params, batch_size, "Layer3 output")?;

        let sa2fl_params = &self.config.sa2fl_params;
        let sa2fl_outputs = build_sa2fl_list(sa2fl_params, &fl2sa_outputs).map_err(|e| {
            ZKPError::SA2FLConversionError(format!("Building Layer3 SA2FL failed: {e}"))
        })?;
        let sa2fl_output_batches =
            assemble_sa2fl_batches(&sa2fl_outputs, batch_size, "Layer3 output")?;

        Ok(Layer3Witness {
            layer: DenseLayerWitness {
                num_rows: num_nodes,
                num_cols: nclass,
                shared_dim: nhid,
                mulfp_witness,
                mulfp_batches,
            },
            fl2sa_outputs,
            fl2sa_output_batches,
            sa2fl_outputs,
            sa2fl_output_batches,
        })
    }
}

pub struct Layer3Witness {
    pub layer: DenseLayerWitness,
    pub fl2sa_outputs: Vec<Fl2saWitness>,
    pub fl2sa_output_batches: Vec<Fl2saBatchWitness>,
    pub sa2fl_outputs: Vec<Sa2flWitness>,
    pub sa2fl_output_batches: Vec<Sa2flBatchWitness>,
}

fn assemble_fl2sa_batches(
    list: &[Fl2saWitness],
    params: &crate::zkp::utils::fl2sa::FL2SAParams,
    batch_size: usize,
    label: &str,
) -> Result<Vec<Fl2saBatchWitness>, ZKPError> {
    if list.is_empty() {
        return Ok(Vec::new());
    }
    let mut batches = Vec::new();
    let mut current = Vec::with_capacity(batch_size);
    for idx in 0..list.len() {
        current.push(idx);
        let is_last = idx + 1 == list.len();
        if current.len() == batch_size || is_last {
            let indices = current.clone();
            let batch = Fl2saBatchWitness::from_indices(&indices, list, params).map_err(|e| {
                ZKPError::FL2SAConversionError(format!(
                    "{label} Batch witness assembly failed: {e}"
                ))
            })?;
            batches.push(batch);
            current.clear();
        }
    }
    Ok(batches)
}

fn assemble_sa2fl_batches(
    list: &[Sa2flWitness],
    batch_size: usize,
    label: &str,
) -> Result<Vec<Sa2flBatchWitness>, ZKPError> {
    if list.is_empty() {
        return Ok(Vec::new());
    }
    let mut batches = Vec::new();
    let mut current = Vec::with_capacity(batch_size);
    for idx in 0..list.len() {
        current.push(idx);
        let is_last = idx + 1 == list.len();
        if current.len() == batch_size || is_last {
            let indices = current.clone();
            let batch = Sa2flBatchWitness::from_indices(&indices, list).map_err(|e| {
                ZKPError::SA2FLConversionError(format!("{label} SA2FL batch witness failed: {e}"))
            })?;
            batches.push(batch);
            current.clear();
        }
    }
    Ok(batches)
}

fn build_sa2fl_list(
    params: &crate::zkp::utils::sa2fl::SA2FLParams,
    entries: &[Fl2saWitness],
) -> Result<Vec<Sa2flWitness>, String> {
    entries
        .iter()
        .map(|entry| Sa2flWitness::from_fl2sa(params, entry))
        .collect()
}
