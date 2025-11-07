#[cfg(not(feature = "zkp"))]
compile_error!("gcn_full_feature binary requires compiling with --features zkp");

use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::{stdout, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use half::f16;
use superacc_zkp::data::loader::load_cora_data;
use superacc_zkp::data::pytorch_loader;
use superacc_zkp::math::{
    activations::{relu, softmax},
    dense_ops,
};
use superacc_zkp::models::gcn::GCN;
use superacc_zkp::types::{DenseMatrix, FloatType, GCNConfig, SparseMatrix};
use superacc_zkp::zkp::operation_tracker::{reset_operation_counters, take_operation_snapshot};
use superacc_zkp::zkp::prover::{
    Layer1Prover, Layer1Witness, Layer2Prover, Layer2Witness, Layer3Prover, Layer3Witness,
    Layer4Prover, Layer4Witness, SparseMatMulWitness,
};
use superacc_zkp::zkp::utils::fl2sa::{Fl2saBatchWitness, Fl2saWitness};
use superacc_zkp::zkp::utils::linear::LinearRelationWitness;
use superacc_zkp::zkp::utils::mulfp::{MulFPBatchWitness, MulFPWitness};
use superacc_zkp::zkp::utils::sa2fl::{Sa2flBatchWitness, Sa2flWitness};
use superacc_zkp::zkp::verifiers::common::{
    ConstraintAccumulator, ConstraintKind, GCNZKPConfig, VerificationLevel,
};
use superacc_zkp::zkp::verifiers::{
    Layer1Verifier, Layer2Verifier, Layer3Verifier, Layer4Verifier,
};

fn log_stage(message: &str) {
    println!("{}", message);
    let _ = stdout().flush();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrecisionCase {
    Single,
    Double,
    Half,
}

impl PrecisionCase {
    fn label(&self) -> &'static str {
        match self {
            Self::Single => "zkGCN-Single",
            Self::Double => "zkGCN-Double",
            Self::Half => "zkGCN-HALF",
        }
    }
}

impl std::str::FromStr for PrecisionCase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "single" | "f32" | "zk" | "sing" => Ok(Self::Single),
            "double" | "f64" | "dbl" => Ok(Self::Double),
            "half" | "f16" | "fp16" => Ok(Self::Half),
            other => Err(format!("Unknown precision option: {}", other)),
        }
    }
}

#[derive(Debug)]
struct AppConfig {
    node_limit: usize,
    node_offset: usize,
    precision_cases: Vec<PrecisionCase>,
    output_prefix: PathBuf,
    weights_single: PathBuf,
    weights_double: PathBuf,
    weights_half: PathBuf,
    verification_level: VerificationLevel,
    feature_limit: Option<usize>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            node_limit: usize::MAX,
            node_offset: 0,
            precision_cases: vec![
                PrecisionCase::Single,
                PrecisionCase::Double,
                PrecisionCase::Half,
            ],
            output_prefix: PathBuf::from("./artifacts/gcn_full_feature"),
            weights_single: PathBuf::from("model_weights/gcn_weights_f32_20251106.json"),
            weights_double: PathBuf::from("model_weights/gcn_weights_f64_20251106.json"),
            weights_half: PathBuf::from("model_weights/gcn_weights_f32_20251106.json"),
            verification_level: VerificationLevel::Full,
            feature_limit: None,
        }
    }
}

#[derive(Clone)]
struct LayerMeasurement {
    name: &'static str,
    prover: Duration,
    verifier: Duration,
    witness_bytes: usize,
    constraints: ConstraintAccumulator,
    max_output_delta: Option<f64>,
}

#[derive(Clone)]
struct CaseResult {
    precision: PrecisionCase,
    node_count: usize,
    prover_time: Duration,
    verifier_time: Duration,
    witness_bytes: usize,
    total_constraints: ConstraintAccumulator,
    accuracy: f64,
    layer_stats: Vec<LayerMeasurement>,
    notes: Option<String>,
}

fn parse_args() -> Result<AppConfig, Box<dyn Error>> {
    let mut cfg = AppConfig::default();
    let mut args = std::env::args().skip(1);
    let mut requested_cases: Vec<PrecisionCase> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--nodes" | "--node-limit" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--nodes requires integer argument".to_owned())?;
                let parsed: usize = value.parse()?;
                if parsed == 0 {
                    return Err("The number of nodes must be greater than 0".into());
                }
                cfg.node_limit = parsed;
            }
            "--node-offset" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--node-offset requires integer argument".to_owned())?;
                let parsed: usize = value.parse()?;
                cfg.node_offset = parsed;
            }
            "--precision" | "--precisions" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--precision requires comma separated list".to_owned())?;
                requested_cases = value
                    .split(',')
                    .map(|token| token.trim().parse())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--output-prefix" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-prefix requires parameters".to_owned())?;
                cfg.output_prefix = PathBuf::from(value);
            }
            "--weights-single" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--weights-single requires path argument".to_owned())?;
                cfg.weights_single = PathBuf::from(value);
            }
            "--weights-double" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--weights-double requires path argument".to_owned())?;
                cfg.weights_double = PathBuf::from(value);
            }
            "--weights-half" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--weights-half requires path argument".to_owned())?;
                cfg.weights_half = PathBuf::from(value);
            }
            "--verification-level" => {
                let value = args.next().ok_or_else(|| {
                    "--verification-level requires parameters (full|optimized|fast)".to_owned()
                })?;
                cfg.verification_level = match value.to_ascii_lowercase().as_str() {
                    "full" => VerificationLevel::Full,
                    "optimized" | "opt" => VerificationLevel::Optimized,
                    "fast" => VerificationLevel::Fast,
                    other => {
                        return Err(format!("Unknown verification-level: {}", other).into());
                    }
                };
            }
            "--feature-limit" | "--features" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--feature-limit requires integer argument".to_owned())?;
                let parsed: usize = value.parse()?;
                if parsed == 0 {
                    return Err("feature-limit must be greater than 0".into());
                }
                cfg.feature_limit = Some(parsed);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("Unknown parameter: {}", unknown).into());
            }
        }
    }

    if !requested_cases.is_empty() {
        cfg.precision_cases = requested_cases;
    }

    Ok(cfg)
}

fn print_usage() {
    println!("GCN performance experiment (full 1433-feature configuration)");
    println!("Usage: cargo run --release --features zkp --bin gcn_full_feature [options]");
    println!();
    println!("Options:");
    println!(
        "  --nodes <N>             Limit the run to the first N nodes (default: use all nodes)"
    );
    println!("  --node-offset <K>       Skip the first K nodes (default: 0)");
    println!("  --precision a,b,c       Precision list from {{single,double,half}} (default: all)");
    println!("  --output-prefix <PATH>  Directory prefix for generated artifacts (default: ./artifacts/gcn_full_feature)");
    println!("  --weights-single <PATH> Path to f32 weights JSON file");
    println!("  --weights-double <PATH> Path to f64 weights JSON file");
    println!(
        "  --weights-half <PATH>   Path to f16 weights JSON file (default reuses f32 weights)"
    );
    println!("  --verification-level <L> Full|Optimized|Fast (default: Full)");
    println!("  --feature-limit <K>     Limit to the first K feature columns (default: 1433)");
    println!("  --help                  Show this help message");
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;

    println!("=== GCN Performance Experiment ===");
    println!("Node limit: {}", config.node_limit);
    println!("Node offset: {}", config.node_offset);
    println!("Output prefix: {}", config.output_prefix.display());
    if let Some(limit) = config.feature_limit {
        println!("Feature upper limit: {}", limit);
    } else {
        println!("Feature cap: Use all features");
    }
    println!("Precision list: {:?}", config.precision_cases);
    println!("Verification level: {:?}", config.verification_level);
    println!();

    let mut results: Vec<CaseResult> = Vec::new();

    for precision in &config.precision_cases {
        match precision {
            PrecisionCase::Single => {
                println!("-> Run single precision (f32) experiment");
                match run_case::<f32>(&config, PrecisionCase::Single) {
                    Ok(result) => {
                        results.push(result);
                    }
                    Err(err) => {
                        eprintln!("Single precision experiment failed: {}", err);
                        return Err(err);
                    }
                }
            }
            PrecisionCase::Double => {
                println!("-> Run double precision (f64) experiment");
                match run_case::<f64>(&config, PrecisionCase::Double) {
                    Ok(result) => {
                        results.push(result);
                    }
                    Err(err) => {
                        eprintln!("Double precision experiment failed: {}", err);
                        return Err(err);
                    }
                }
            }
            PrecisionCase::Half => {
                println!("-> Run half-precision (f16) experiment");
                match run_case::<f16>(&config, PrecisionCase::Half) {
                    Ok(result) => {
                        results.push(result);
                    }
                    Err(err) => {
                        eprintln!("Half precision experiment failed: {}", err);
                        return Err(err);
                    }
                }
            }
        }
        println!();
    }

    if results.is_empty() {
        println!("No experiments were performed.");
        return Ok(());
    }

    persist_results(&config.output_prefix, &results)?;
    print_summary_table(&results);

    Ok(())
}

fn run_case<T: FloatType + std::fmt::Display>(
    config: &AppConfig,
    precision: PrecisionCase,
) -> Result<CaseResult, Box<dyn Error>> {
    let dataset = load_cora_data::<T>("data/cora")?;
    let total_nodes = dataset.features.shape.0;
    if config.node_offset >= total_nodes {
        return Err(format!(
            "node-offset={} exceeds the total number of data set nodes {}",
            config.node_offset, total_nodes
        )
        .into());
    }
    let available_nodes = total_nodes - config.node_offset;
    let node_cap = config.node_limit.min(available_nodes);
    if node_cap == 0 {
        return Err(format!(
            "node-offset={} causes the number of available nodes to be 0",
            config.node_offset
        )
        .into());
    }
    let original_feature_dim = dataset.features.shape.1;
    let feature_cap = config
        .feature_limit
        .unwrap_or(1433)
        .min(original_feature_dim);

    let features_for_proof = if feature_cap < original_feature_dim {
        trim_feature_columns(&dataset.features, feature_cap)
    } else {
        dataset.features.clone()
    };

    let (features_full, adj) = truncate_dataset(
        &features_for_proof,
        &dataset.adj,
        node_cap,
        config.node_offset,
    );
    let features = features_full;

    let loader_config = GCNConfig::default();
    let mut accuracy_config = loader_config.clone();
    accuracy_config.nfeat = original_feature_dim;
    let mut proof_config = loader_config.clone();

    let weights_path = match precision {
        PrecisionCase::Single => &config.weights_single,
        PrecisionCase::Double => &config.weights_double,
        PrecisionCase::Half => &config.weights_half,
    };
    let weights_path_str = weights_path.to_str().ok_or_else(|| {
        format!(
            "Weight file path contains non-UTF-8 characters: {}",
            weights_path.display()
        )
    })?;
    let accuracy_weights =
        pytorch_loader::load_weights_from_json::<T>(weights_path_str, &loader_config)?;
    let accuracy_model = GCN::from_weights(accuracy_weights, accuracy_config.clone());

    let mut proof_weights =
        pytorch_loader::load_weights_from_json::<T>(weights_path_str, &loader_config)?;
    if feature_cap < original_feature_dim {
        proof_weights.gc1_weight = trim_gc1_weight(&proof_weights.gc1_weight, feature_cap);
    }
    proof_config.nfeat = feature_cap;
    let model = GCN::from_weights(proof_weights, proof_config.clone());

    let support_layer1 = dense_ops::dense_mm(&features, &model.gc1.weight);
    let pre_activations = model.gc1.forward(&features, &adj);
    let activations = relu(&pre_activations);
    let support_layer2 = dense_ops::dense_mm(&activations, &model.gc2.weight);
    let logits = model.gc2.forward(&activations, &adj);
    let softmax_outputs = softmax(&logits, 1);

    let accuracy = accuracy_model.accuracy(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_test,
    );
    let accuracy = if accuracy.is_finite() {
        (accuracy * 10_000.0).round() / 10_000.0
    } else {
        accuracy
    };
    let mut zkp_config = match precision {
        PrecisionCase::Single => GCNZKPConfig::single_precision(),
        PrecisionCase::Double => GCNZKPConfig::double_precision(),
        PrecisionCase::Half => GCNZKPConfig::half_precision(),
    };
    zkp_config.verification_level = config.verification_level;
    zkp_config.batch_size = zkp_config.batch_size.min(node_cap.max(1));
    let zkp_config = Arc::new(zkp_config);

    let mut layer_stats = Vec::new();
    let mut total_prover = Duration::ZERO;
    let mut total_verifier = Duration::ZERO;
    let mut total_witness_bytes = 0usize;
    let mut total_constraints = ConstraintAccumulator::new();

    {
        log_stage("[progress] Counting Layer1 constraints and operators...");
        let stats = measure_layer1(zkp_config.clone(), &features, &model.gc1.weight)?;
        total_prover += stats.prover;
        total_verifier += stats.verifier;
        total_witness_bytes += stats.witness_bytes;
        total_constraints.merge(stats.constraints.clone());
        layer_stats.push(stats);
        log_stage("[progress] Layer1 completed.");
    }

    {
        log_stage("[progress] Counting Layer2 constraints and operators...");
        let stats = measure_layer2(
            zkp_config.clone(),
            &support_layer1,
            &adj,
            model.gc1.bias.as_deref(),
            &activations,
        )?;
        total_prover += stats.prover;
        total_verifier += stats.verifier;
        total_witness_bytes += stats.witness_bytes;
        total_constraints.merge(stats.constraints.clone());
        layer_stats.push(stats);
        log_stage("[progress] Layer2 completed.");
    }

    {
        log_stage("[progress] Counting Layer3 constraints and operators...");
        let stats = measure_layer3(zkp_config.clone(), &activations, &model.gc2.weight)?;
        total_prover += stats.prover;
        total_verifier += stats.verifier;
        total_witness_bytes += stats.witness_bytes;
        total_constraints.merge(stats.constraints.clone());
        layer_stats.push(stats);
        log_stage("[progress] Layer3 completed.");
    }

    {
        log_stage("[progress] Counting Layer4 constraints and operators...");
        let stats = measure_layer4(
            zkp_config,
            &support_layer2,
            &adj,
            model.gc2.bias.as_deref(),
            &logits,
            &softmax_outputs,
        )?;
        total_prover += stats.prover;
        total_verifier += stats.verifier;
        total_witness_bytes += stats.witness_bytes;
        total_constraints.merge(stats.constraints.clone());
        layer_stats.push(stats);
        log_stage("[progress] Layer4 completed.");
    }

    Ok(CaseResult {
        precision,
        node_count: node_cap,
        prover_time: total_prover,
        verifier_time: total_verifier,
        witness_bytes: total_witness_bytes,
        total_constraints,
        accuracy,
        layer_stats,
        notes: Some(format!(
            "nodes={} features={} offset={}",
            node_cap, feature_cap, config.node_offset
        )),
    })
}

fn truncate_dataset<T: FloatType>(
    features: &DenseMatrix<T>,
    adj: &SparseMatrix<T>,
    node_limit: usize,
    node_offset: usize,
) -> (DenseMatrix<T>, SparseMatrix<T>) {
    let total_rows = features.shape.0;
    let start = node_offset.min(total_rows);
    let end = (start + node_limit).min(total_rows);
    let rows = end.saturating_sub(start);
    let cols = features.shape.1;

    let mut feature_data = Vec::with_capacity(rows * cols);
    for row in start..end {
        let offset = row * cols;
        feature_data.extend_from_slice(&features.data[offset..offset + cols]);
    }
    let features = DenseMatrix::new(feature_data, (rows, cols));

    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (idx, &(row, col)) in adj.indices.iter().enumerate() {
        let row_usize = row as usize;
        let col_usize = col as usize;
        if row_usize >= start && row_usize < end && col_usize >= start && col_usize < end {
            indices.push(((row_usize - start) as i64, (col_usize - start) as i64));
            values.push(adj.values[idx]);
        }
    }
    let adj = SparseMatrix {
        indices,
        values,
        shape: (rows, rows),
    };

    (features, adj)
}

fn measure_layer1<T: FloatType>(
    config: Arc<GCNZKPConfig>,
    features: &DenseMatrix<T>,
    weights: &DenseMatrix<T>,
) -> Result<LayerMeasurement, Box<dyn Error>> {
    let prover = Layer1Prover::new(config.clone());
    reset_operation_counters();
    let start = Instant::now();
    let witness = prover
        .generate_witness(features, weights)
        .map_err(|e| format!("Layer1 failed to generate witness: {:?}", e))?;
    let prover_time = start.elapsed();

    let witness_bytes = layer1_witness_bytes(&witness);

    let verifier = Layer1Verifier::new(config);
    let start = Instant::now();
    let report = verifier
        .verify(&witness)
        .map_err(|e| format!("Layer1 verification failed: {:?}", e))?;
    let verifier_time = start.elapsed();
    let mut constraints = report.constraints;
    let snapshot = take_operation_snapshot();
    constraints.add_counter.merge(&snapshot.add);
    constraints.mul_counter.merge(&snapshot.mul);

    Ok(LayerMeasurement {
        name: "Layer1",
        prover: prover_time,
        verifier: verifier_time,
        witness_bytes,
        constraints,
        max_output_delta: None,
    })
}

fn measure_layer2<T: FloatType>(
    config: Arc<GCNZKPConfig>,
    support: &DenseMatrix<T>,
    adj: &SparseMatrix<T>,
    bias: Option<&[T]>,
    relu_outputs: &DenseMatrix<T>,
) -> Result<LayerMeasurement, Box<dyn Error>> {
    let prover = Layer2Prover::new(config.clone());
    reset_operation_counters();
    let start = Instant::now();
    let witness = prover
        .generate_witness(support, adj, bias)
        .map_err(|e| format!("Layer2 failed to generate witness: {:?}", e))?;
    let prover_time = start.elapsed();

    let witness_bytes = layer2_witness_bytes(&witness);

    let tolerance = config.tolerance.max(0.05);
    let max_delta = validate_layer2_outputs(&witness, relu_outputs).map_err(|e| format!("{e}"))?;
    if max_delta > tolerance {
        println!(
            "Warning: Layer2 ReLU has the largest deviation {:.6} (> tolerance {:.6})",
            max_delta, tolerance
        );
    }

    let verifier = Layer2Verifier::new(config);
    let start = Instant::now();
    let report = verifier
        .verify(&witness)
        .map_err(|e| format!("Layer2 verification failed: {:?}", e))?;
    let verifier_time = start.elapsed();
    let mut constraints = report.constraints;
    let snapshot = take_operation_snapshot();
    constraints.add_counter.merge(&snapshot.add);
    constraints.mul_counter.merge(&snapshot.mul);

    Ok(LayerMeasurement {
        name: "Layer2",
        prover: prover_time,
        verifier: verifier_time,
        witness_bytes,
        constraints,
        max_output_delta: Some(max_delta),
    })
}

fn measure_layer3<T: FloatType>(
    config: Arc<GCNZKPConfig>,
    activations: &DenseMatrix<T>,
    weights: &DenseMatrix<T>,
) -> Result<LayerMeasurement, Box<dyn Error>> {
    let prover = Layer3Prover::new(config.clone());
    reset_operation_counters();
    let start = Instant::now();
    let witness = prover
        .generate_witness(activations, weights)
        .map_err(|e| format!("Layer3 failed to generate witness: {:?}", e))?;
    let prover_time = start.elapsed();

    let witness_bytes = layer3_witness_bytes(&witness);

    let verifier = Layer3Verifier::new(config);
    let start = Instant::now();
    let report = verifier
        .verify(&witness)
        .map_err(|e| format!("Layer3 verification failed: {:?}", e))?;
    let verifier_time = start.elapsed();
    let mut constraints = report.constraints;
    let snapshot = take_operation_snapshot();
    constraints.add_counter.merge(&snapshot.add);
    constraints.mul_counter.merge(&snapshot.mul);

    Ok(LayerMeasurement {
        name: "Layer3",
        prover: prover_time,
        verifier: verifier_time,
        witness_bytes,
        constraints,
        max_output_delta: None,
    })
}

fn measure_layer4<T: FloatType>(
    config: Arc<GCNZKPConfig>,
    support: &DenseMatrix<T>,
    adj: &SparseMatrix<T>,
    bias: Option<&[T]>,
    _logits: &DenseMatrix<T>,
    softmax_outputs: &DenseMatrix<T>,
) -> Result<LayerMeasurement, Box<dyn Error>> {
    let prover = Layer4Prover::new(config.clone());
    reset_operation_counters();
    let start = Instant::now();
    let witness = prover
        .generate_witness(support, adj, bias)
        .map_err(|e| format!("Layer4 failed to generate witness: {:?}", e))?;
    let prover_time = start.elapsed();

    let witness_bytes = layer4_witness_bytes(&witness);

    let tolerance = config.tolerance.max(0.05);
    let max_delta =
        validate_layer4_outputs(&witness, softmax_outputs).map_err(|e| format!("{e}"))?;
    if max_delta > tolerance {
        println!(
            "Warning: Layer4 Softmax maximum deviation {:.6} (> tolerance {:.6})",
            max_delta, tolerance
        );
    }

    let verifier = Layer4Verifier::new(config);
    let start = Instant::now();
    let report = verifier
        .verify(&witness)
        .map_err(|e| format!("Layer4 verification failed: {:?}", e))?;
    let verifier_time = start.elapsed();
    let mut constraints = report.constraints;
    let snapshot = take_operation_snapshot();
    constraints.add_counter.merge(&snapshot.add);
    constraints.mul_counter.merge(&snapshot.mul);

    Ok(LayerMeasurement {
        name: "Layer4",
        prover: prover_time,
        verifier: verifier_time,
        witness_bytes,
        constraints,
        max_output_delta: Some(max_delta),
    })
}

fn layer1_witness_bytes(witness: &Layer1Witness) -> usize {
    let layer = &witness.layer;
    let mulfp_entries = layer
        .mulfp_witness
        .iter()
        .map(bytes_for_mulfp)
        .sum::<usize>();
    let batch_entries = layer
        .mulfp_batches
        .iter()
        .map(bytes_for_mulfp_batch)
        .sum::<usize>();
    mulfp_entries + batch_entries
}

fn layer2_witness_bytes(witness: &Layer2Witness) -> usize {
    let sparse_bytes = bytes_for_sparse_product(&witness.sparse_product);
    let mulfp_entries = witness
        .mulfp_witness
        .iter()
        .map(bytes_for_mulfp)
        .sum::<usize>();
    let mulfp_batches = witness
        .mulfp_batches
        .iter()
        .map(bytes_for_mulfp_batch)
        .sum::<usize>();

    let fl2sa_inputs = witness
        .fl2sa_inputs
        .iter()
        .map(bytes_for_fl2sa)
        .sum::<usize>();
    let fl2sa_polys = witness
        .fl2sa_polys
        .iter()
        .map(bytes_for_fl2sa)
        .sum::<usize>();
    let fl2sa_relus = witness
        .fl2sa_relus
        .iter()
        .map(bytes_for_fl2sa)
        .sum::<usize>();

    let fl2sa_input_batches = witness
        .fl2sa_input_batches
        .iter()
        .map(bytes_for_fl2sa_batch)
        .sum::<usize>();
    let fl2sa_poly_batches = witness
        .fl2sa_poly_batches
        .iter()
        .map(bytes_for_fl2sa_batch)
        .sum::<usize>();
    let fl2sa_relu_batches = witness
        .fl2sa_relu_batches
        .iter()
        .map(bytes_for_fl2sa_batch)
        .sum::<usize>();

    let linear_relations = witness
        .entries
        .iter()
        .flat_map(|entry| {
            [
                bytes_for_linear(&entry.poly_relation_witness),
                bytes_for_linear(&entry.relu_relation_witness),
            ]
        })
        .sum::<usize>();

    mulfp_entries
        + mulfp_batches
        + fl2sa_inputs
        + fl2sa_polys
        + fl2sa_relus
        + fl2sa_input_batches
        + fl2sa_poly_batches
        + fl2sa_relu_batches
        + linear_relations
        + sparse_bytes
}

fn layer3_witness_bytes(witness: &Layer3Witness) -> usize {
    let layer = &witness.layer;
    let mulfp_entries = layer
        .mulfp_witness
        .iter()
        .map(bytes_for_mulfp)
        .sum::<usize>();
    let mulfp_batches = layer
        .mulfp_batches
        .iter()
        .map(bytes_for_mulfp_batch)
        .sum::<usize>();
    let fl2sa_entries = witness
        .fl2sa_outputs
        .iter()
        .map(bytes_for_fl2sa)
        .sum::<usize>();
    let fl2sa_batches = witness
        .fl2sa_output_batches
        .iter()
        .map(bytes_for_fl2sa_batch)
        .sum::<usize>();
    let sa2fl_entries = witness
        .sa2fl_outputs
        .iter()
        .map(bytes_for_sa2fl)
        .sum::<usize>();
    let sa2fl_batches = witness
        .sa2fl_output_batches
        .iter()
        .map(bytes_for_sa2fl_batch)
        .sum::<usize>();
    mulfp_entries + mulfp_batches + fl2sa_entries + fl2sa_batches + sa2fl_entries + sa2fl_batches
}

fn layer4_witness_bytes(witness: &Layer4Witness) -> usize {
    let sparse_bytes = bytes_for_sparse_product(&witness.sparse_product);
    let mulfp_entries = witness
        .mulfp_witness
        .iter()
        .map(bytes_for_mulfp)
        .sum::<usize>();
    let mulfp_batches = witness
        .mulfp_batches
        .iter()
        .map(bytes_for_mulfp_batch)
        .sum::<usize>();

    let mut fl2sa_term = 0usize;
    for collection in [
        &witness.fl2sa_max_list,
        &witness.fl2sa_denominator_list,
        &witness.fl2sa_logit_list,
        &witness.fl2sa_stabilized_list,
        &witness.fl2sa_numerator_list,
        &witness.fl2sa_softmax_list,
    ] {
        fl2sa_term += collection.iter().map(bytes_for_fl2sa).sum::<usize>();
    }

    let mut fl2sa_batches = 0usize;
    for collection in [
        &witness.fl2sa_max_batches,
        &witness.fl2sa_denominator_batches,
        &witness.fl2sa_logit_batches,
        &witness.fl2sa_stabilized_batches,
        &witness.fl2sa_numerator_batches,
        &witness.fl2sa_softmax_batches,
    ] {
        fl2sa_batches += collection.iter().map(bytes_for_fl2sa_batch).sum::<usize>();
    }

    let mut sa2fl_term = 0usize;
    for collection in [
        &witness.sa2fl_max_list,
        &witness.sa2fl_denominator_list,
        &witness.sa2fl_logit_list,
        &witness.sa2fl_stabilized_list,
        &witness.sa2fl_numerator_list,
        &witness.sa2fl_softmax_list,
    ] {
        sa2fl_term += collection.iter().map(bytes_for_sa2fl).sum::<usize>();
    }

    let mut sa2fl_batches = 0usize;
    for collection in [
        &witness.sa2fl_max_batches,
        &witness.sa2fl_denominator_batches,
        &witness.sa2fl_logit_batches,
        &witness.sa2fl_stabilized_batches,
        &witness.sa2fl_numerator_batches,
        &witness.sa2fl_softmax_batches,
    ] {
        sa2fl_batches += collection.iter().map(bytes_for_sa2fl_batch).sum::<usize>();
    }

    let mut linear_relations = 0usize;
    for node in &witness.nodes {
        linear_relations += bytes_for_linear(&node.denominator_relation_witness);
        linear_relations += bytes_for_linear(&node.probability_relation_witness);
        linear_relations += node
            .entries
            .iter()
            .map(|entry| bytes_for_linear(&entry.numerator_relation_witness))
            .sum::<usize>();
    }

    mulfp_entries
        + mulfp_batches
        + fl2sa_term
        + fl2sa_batches
        + sa2fl_term
        + sa2fl_batches
        + linear_relations
        + sparse_bytes
}

fn bytes_for_sparse_product(witness: &SparseMatMulWitness) -> usize {
    let scalars = witness.csr_row_ptr.len() * size_of::<usize>()
        + witness.col_indices.len() * size_of::<usize>()
        + witness.values.len() * size_of::<f64>();
    let commitments = witness.support_commitments.len() * 32 + 32;
    let mulfp_entries = witness
        .mulfp_witness
        .iter()
        .map(bytes_for_mulfp)
        .sum::<usize>();
    let mulfp_batches = witness
        .mulfp_batches
        .iter()
        .map(bytes_for_mulfp_batch)
        .sum::<usize>();
    let relations = witness
        .row_relations
        .iter()
        .map(bytes_for_linear)
        .sum::<usize>();
    scalars + commitments + mulfp_entries + mulfp_batches + relations
}

fn bytes_for_mulfp(witness: &MulFPWitness) -> usize {
    witness.assignment.len() * 32
}

fn bytes_for_mulfp_batch(witness: &MulFPBatchWitness) -> usize {
    witness.assignment.len() * 32
}

fn bytes_for_fl2sa(witness: &Fl2saWitness) -> usize {
    witness.num_vars * 32
}

fn bytes_for_fl2sa_batch(witness: &Fl2saBatchWitness) -> usize {
    witness.num_vars * 32
}

fn bytes_for_sa2fl(witness: &Sa2flWitness) -> usize {
    witness.double_shuffle.auxb_bits.len()
        + witness.double_shuffle.auxb2_bits.len()
        + witness.auxb_indices.len() * size_of::<usize>()
        + witness.auxb2_indices.len() * size_of::<usize>()
}

fn bytes_for_sa2fl_batch(witness: &Sa2flBatchWitness) -> usize {
    witness.num_vars * 32
}

fn bytes_for_linear(witness: &LinearRelationWitness) -> usize {
    witness.num_vars * 32
}

fn trim_feature_columns<T: FloatType>(matrix: &DenseMatrix<T>, new_cols: usize) -> DenseMatrix<T> {
    let (rows, cols) = matrix.shape;
    assert!(
        new_cols <= cols,
        "feature_limit {} exceeds feature dimension {}",
        new_cols,
        cols
    );
    if new_cols == cols {
        return matrix.clone();
    }
    let mut data = Vec::with_capacity(rows * new_cols);
    for row in 0..rows {
        let start = row * cols;
        let end = start + new_cols;
        data.extend_from_slice(&matrix.data[start..end]);
    }
    DenseMatrix::new(data, (rows, new_cols))
}

fn trim_gc1_weight<T: FloatType>(matrix: &DenseMatrix<T>, new_rows: usize) -> DenseMatrix<T> {
    let (rows, cols) = matrix.shape;
    assert!(
        new_rows <= rows,
        "feature_limit {} The number of rows exceeding the weight {}",
        new_rows,
        rows
    );
    if new_rows == rows {
        return matrix.clone();
    }
    let mut data = Vec::with_capacity(new_rows * cols);
    for row in 0..new_rows {
        let start = row * cols;
        let end = start + cols;
        data.extend_from_slice(&matrix.data[start..end]);
    }
    DenseMatrix::new(data, (new_rows, cols))
}

fn validate_layer2_outputs<T: FloatType>(
    witness: &Layer2Witness,
    relu_outputs: &DenseMatrix<T>,
) -> Result<f64, String> {
    let expected_rows = witness.num_nodes;
    let expected_cols = witness.hidden_dim;
    if relu_outputs.shape.0 != expected_rows || relu_outputs.shape.1 != expected_cols {
        return Err(format!(
            "ReLU output dimension {}x{} is inconsistent with witness {}x{}",
            relu_outputs.shape.0, relu_outputs.shape.1, expected_rows, expected_cols
        ));
    }

    let mut max_delta = 0f64;
    for (idx, entry) in witness.entries.iter().enumerate() {
        let row = idx / expected_cols;
        let col = idx % expected_cols;
        let expected = relu_outputs
            .get(row, col)
            .to_f64()
            .ok_or_else(|| format!("Unable to convert ReLU output ({},{}) to f64", row, col))?;
        let delta = (entry.relu_value - expected).abs();
        if delta > max_delta {
            max_delta = delta;
        }
    }

    Ok(max_delta)
}

fn validate_layer4_outputs<T: FloatType>(
    witness: &Layer4Witness,
    softmax_outputs: &DenseMatrix<T>,
) -> Result<f64, String> {
    if softmax_outputs.shape.0 != witness.num_nodes
        || softmax_outputs.shape.1 != witness.num_classes
    {
        return Err(format!(
            "Softmax output dimension {}x{} is inconsistent with witness {}x{}",
            softmax_outputs.shape.0,
            softmax_outputs.shape.1,
            witness.num_nodes,
            witness.num_classes
        ));
    }

    let mut max_delta = 0f64;
    for (node_idx, node) in witness.nodes.iter().enumerate() {
        if node.entries.len() != witness.num_classes {
            return Err(format!(
                "Number of categories for Softmax node {} {} does not match expected {}",
                node_idx,
                node.entries.len(),
                witness.num_classes
            ));
        }
        for (class_idx, entry) in node.entries.iter().enumerate() {
            let expected = softmax_outputs
                .get(node_idx, class_idx)
                .to_f64()
                .ok_or_else(|| {
                    format!(
                        "Unable to convert Softmax output ({},{}) to f64",
                        node_idx, class_idx
                    )
                })?;
            let delta = (entry.softmax_value - expected).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
    }

    Ok(max_delta)
}

fn persist_results(output_prefix: &PathBuf, results: &[CaseResult]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = output_prefix.parent() {
        if !parent.exists() {
            create_dir_all(parent)?;
        }
    }

    let csv_path = output_prefix.with_extension("csv");
    let md_path = output_prefix.with_extension("md");

    {
        let mut file = File::create(&csv_path)?;
        writeln!(
            file,
            "precision,node_limit,prover_time_ms,verifier_time_ms,proof_size_bytes,accuracy,total_constraints,mulfp_constraints,fl2sa_constraints,sa2fl_constraints,add_ops_total,mul_ops_total,add_ops_fp16,add_ops_bf16,add_ops_fp32,add_ops_fp64,mul_ops_fp16,mul_ops_bf16,mul_ops_fp32,mul_ops_fp64,field_add_ops_total,field_mul_ops_total,field_add_ops_fp16,field_add_ops_bf16,field_add_ops_fp32,field_add_ops_fp64,field_mul_ops_fp16,field_mul_ops_bf16,field_mul_ops_fp32,field_mul_ops_fp64,field_add_ops_mulfp_core,field_mul_ops_mulfp_core,field_add_ops_mulfp_batch,field_mul_ops_mulfp_batch,field_add_ops_fl2sa_core,field_mul_ops_fl2sa_core,field_add_ops_fl2sa_batch,field_mul_ops_fl2sa_batch,field_add_ops_sa2fl_core,field_mul_ops_sa2fl_core,field_add_ops_sa2fl_batch,field_mul_ops_sa2fl_batch,field_add_ops_linear_combination,field_mul_ops_linear_combination,field_add_ops_auxiliary,field_mul_ops_auxiliary,max_output_delta,notes"
        )?;
        for result in results {
            let stats = &result.total_constraints.stats;
            let add_counter = &result.total_constraints.add_counter;
            let mul_counter = &result.total_constraints.mul_counter;
            let field_add_counter = &result.total_constraints.field_add_counter;
            let field_mul_counter = &result.total_constraints.field_mul_counter;
            let field_total_adds = field_add_counter.total();
            let field_total_muls = field_mul_counter.total();
            let kind_totals = |kind: ConstraintKind| {
                result
                    .total_constraints
                    .field_ops_for_kind(kind)
                    .map(|counter| (counter.total_adds(), counter.total_muls()))
                    .unwrap_or((0usize, 0usize))
            };
            let (field_add_mulfp_core, field_mul_mulfp_core) =
                kind_totals(ConstraintKind::MulFpCore);
            let (field_add_mulfp_batch, field_mul_mulfp_batch) =
                kind_totals(ConstraintKind::MulFpBatch);
            let (field_add_fl2sa_core, field_mul_fl2sa_core) =
                kind_totals(ConstraintKind::Fl2SaCore);
            let (field_add_fl2sa_batch, field_mul_fl2sa_batch) =
                kind_totals(ConstraintKind::Fl2SaBatch);
            let (field_add_sa2fl_core, field_mul_sa2fl_core) =
                kind_totals(ConstraintKind::Sa2FlCore);
            let (field_add_sa2fl_batch, field_mul_sa2fl_batch) =
                kind_totals(ConstraintKind::Sa2FlBatch);
            let (field_add_linear, field_mul_linear) =
                kind_totals(ConstraintKind::LinearCombination);
            let (field_add_aux, field_mul_aux) = kind_totals(ConstraintKind::Auxiliary);

            let notes = result.notes.as_ref().map(|s| s.as_str()).unwrap_or("");
            let max_delta = result
                .layer_stats
                .iter()
                .filter_map(|layer| layer.max_output_delta)
                .fold(0f64, f64::max);

            let mut row: Vec<String> = Vec::with_capacity(48);
            row.push(result.precision.label().to_string());
            row.push(result.node_count.to_string());
            row.push(format!("{:.3}", duration_ms(result.prover_time)));
            row.push(format!("{:.3}", duration_ms(result.verifier_time)));
            row.push(result.witness_bytes.to_string());
            row.push(format!("{:.4}", result.accuracy));
            row.push(result.total_constraints.total_constraints.to_string());
            row.push(stats.mulfp_count.to_string());
            row.push(stats.fl2sa_count.to_string());
            row.push(stats.sa2fl_count.to_string());
            row.push(result.total_constraints.total_add_ops().to_string());
            row.push(result.total_constraints.total_mul_ops().to_string());
            row.push(add_counter.fp16.to_string());
            row.push(add_counter.bf16.to_string());
            row.push(add_counter.fp32.to_string());
            row.push(add_counter.fp64.to_string());
            row.push(mul_counter.fp16.to_string());
            row.push(mul_counter.bf16.to_string());
            row.push(mul_counter.fp32.to_string());
            row.push(mul_counter.fp64.to_string());
            row.push(field_total_adds.to_string());
            row.push(field_total_muls.to_string());
            row.push(field_add_counter.fp16.to_string());
            row.push(field_add_counter.bf16.to_string());
            row.push(field_add_counter.fp32.to_string());
            row.push(field_add_counter.fp64.to_string());
            row.push(field_mul_counter.fp16.to_string());
            row.push(field_mul_counter.bf16.to_string());
            row.push(field_mul_counter.fp32.to_string());
            row.push(field_mul_counter.fp64.to_string());
            row.push(field_add_mulfp_core.to_string());
            row.push(field_mul_mulfp_core.to_string());
            row.push(field_add_mulfp_batch.to_string());
            row.push(field_mul_mulfp_batch.to_string());
            row.push(field_add_fl2sa_core.to_string());
            row.push(field_mul_fl2sa_core.to_string());
            row.push(field_add_fl2sa_batch.to_string());
            row.push(field_mul_fl2sa_batch.to_string());
            row.push(field_add_sa2fl_core.to_string());
            row.push(field_mul_sa2fl_core.to_string());
            row.push(field_add_sa2fl_batch.to_string());
            row.push(field_mul_sa2fl_batch.to_string());
            row.push(field_add_linear.to_string());
            row.push(field_mul_linear.to_string());
            row.push(field_add_aux.to_string());
            row.push(field_mul_aux.to_string());
            row.push(format!("{:.6}", max_delta));
            row.push(notes.to_string());

            writeln!(file, "{}", row.join(","))?;
        }
    }

    {
        let mut file = File::create(&md_path)?;
        writeln!(file, "# GCN performance experiment results")?;
        writeln!(file)?;
        writeln!(
            file,
            "| Accuracy | Number of nodes | Proof time (ms) | Verification time (ms) | Proof size (KB) | Accuracy | #Constraints | Number of additions | Number of multiplications | Field addition | Field multiplication | Max Delta | Remarks |"
        )?;
        writeln!(
            file,
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )?;
        for result in results {
            let proof_kb = result.witness_bytes as f64 / 1024.0;
            let notes = result.notes.clone().unwrap_or_else(|| "-".to_owned());
            let max_delta = result
                .layer_stats
                .iter()
                .filter_map(|layer| layer.max_output_delta)
                .fold(0f64, f64::max);
            let field_add_total = result.total_constraints.total_field_add_ops();
            let field_mul_total = result.total_constraints.total_field_mul_ops();
            writeln!(
                file,
                "| {} | {} | {:.3} | {:.3} | {:.1} | {:.4} | {} | {} | {} | {} | {} | {:.6} | {} |",
                result.precision.label(),
                result.node_count,
                duration_ms(result.prover_time),
                duration_ms(result.verifier_time),
                proof_kb,
                result.accuracy,
                result.total_constraints.total_constraints,
                result.total_constraints.total_add_ops(),
                result.total_constraints.total_mul_ops(),
                field_add_total,
                field_mul_total,
                max_delta,
                notes
            )?;
        }

        writeln!(file)?;
        writeln!(file, "## Hierarchical statistics")?;
        for result in results {
            writeln!(file)?;
            writeln!(file, "### {}", result.precision.label())?;
            if let Some(notes) = &result.notes {
                writeln!(file, "> {}", notes)?;
                continue;
            }
            writeln!(
                file,
                "| Layer | Prover(ms) | Verifier(ms) | Proof size (KB) | #Constraints | Number of additions | Number of multiplications | Max Delta |"
            )?;
            writeln!(file, "| --- | --- | --- | --- | --- | --- | --- | --- |")?;
            for layer in &result.layer_stats {
                writeln!(
                    file,
                    "| {} | {:.3} | {:.3} | {:.1} | {} | {} | {} | {:.6} |",
                    layer.name,
                    duration_ms(layer.prover),
                    duration_ms(layer.verifier),
                    layer.witness_bytes as f64 / 1024.0,
                    layer.constraints.total_constraints,
                    layer.constraints.total_add_ops(),
                    layer.constraints.total_mul_ops(),
                    layer.max_output_delta.unwrap_or(0.0)
                )?;
            }
        }
    }

    println!("Write result: {}", csv_path.display());
    println!("Write result: {}", md_path.display());

    Ok(())
}

fn print_summary_table(results: &[CaseResult]) {
    println!("=== Summary ===");
    println!(
        "{:<15} {:>8} {:>12} {:>12} {:>14} {:>10} {:>10}",
        "Precision", "Nodes", "Prover(ms)", "Verifier(ms)", "Proof(KB)", "Accuracy", "Max Delta"
    );
    for result in results {
        let proof_kb = result.witness_bytes as f64 / 1024.0;
        let acc = if result.accuracy.is_nan() {
            "N/A".to_owned()
        } else {
            format!("{:.4}", result.accuracy)
        };
        let max_delta = result
            .layer_stats
            .iter()
            .filter_map(|layer| layer.max_output_delta)
            .fold(0f64, f64::max);
        println!(
            "{:<15} {:>8} {:>12.3} {:>12.3} {:>14.1} {:>10} {:>10.6}",
            result.precision.label(),
            result.node_count,
            duration_ms(result.prover_time),
            duration_ms(result.verifier_time),
            proof_kb,
            acc,
            max_delta
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
