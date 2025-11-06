use crate::types::{DenseMatrix, FloatType, GCNConfig, GCNWeights};
use std::collections::HashMap;

#[derive(Debug)]
pub enum PyTorchLoaderError {
    FileNotFound(String),
    ParseError(String),
    DimensionMismatch(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for PyTorchLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(msg) => write!(f, "File not found: {}", msg),
            Self::ParseError(msg) => write!(f, "Parsing error: {}", msg),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            Self::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
        }
    }
}

impl std::error::Error for PyTorchLoaderError {}

///
///   - 'gc1.weight': torch.float32 [1433, 16]  
///   - 'gc1.bias': torch.float32 [16]
///   - 'gc2.weight': torch.float32 [16, 7]
///   - 'gc2.bias': torch.float32 [7]
///
pub fn load_pytorch_weights<T: FloatType>(
    model_path: &str,
) -> Result<GCNWeights<T>, PyTorchLoaderError> {
    println!("Try loading PyTorch model parameters: {}", model_path);

    //
    //
    //    ```rust
    //    use tch::{Tensor, Device};
    //    let vs = tch::nn::VarStore::new(Device::Cpu);
    //    vs.load(model_path)?;
    //    ```
    //
    //

    Err(PyTorchLoaderError::UnsupportedFormat(
        "PyTorch weight loading has not been implemented yet, please use weight dictionary format or implement tch binding".to_string(),
    ))
}

///
///
pub fn load_weights_from_dict<T: FloatType>(
    weights_dict: HashMap<String, Vec<f32>>,
    config: &GCNConfig,
) -> Result<GCNWeights<T>, PyTorchLoaderError> {
    println!("Load GCN parameters from weight dictionary...");

    let gc1_weight_f32 = weights_dict
        .get("gc1.weight")
        .ok_or_else(|| PyTorchLoaderError::ParseError("gc1.weight missing".to_string()))?;

    let expected_gc1_size = config.nfeat * config.nhid;
    if gc1_weight_f32.len() != expected_gc1_size {
        return Err(PyTorchLoaderError::DimensionMismatch(format!(
            "gc1.weight dimension error: {} vs {} ({}*{})",
            gc1_weight_f32.len(),
            expected_gc1_size,
            config.nfeat,
            config.nhid
        )));
    }

    let gc1_weight_data: Vec<T> = gc1_weight_f32
        .iter()
        .map(|&x| T::from_f64_exact(x as f64).unwrap())
        .collect();

    let gc1_bias_f32 = weights_dict
        .get("gc1.bias")
        .ok_or_else(|| PyTorchLoaderError::ParseError("gc1.bias missing".to_string()))?;

    if gc1_bias_f32.len() != config.nhid {
        return Err(PyTorchLoaderError::DimensionMismatch(format!(
            "gc1.bias dimension error: {} vs {}",
            gc1_bias_f32.len(),
            config.nhid
        )));
    }

    let gc1_bias: Vec<T> = gc1_bias_f32
        .iter()
        .map(|&x| T::from_f64_exact(x as f64).unwrap())
        .collect();

    let gc2_weight_f32 = weights_dict
        .get("gc2.weight")
        .ok_or_else(|| PyTorchLoaderError::ParseError("gc2.weight missing".to_string()))?;

    let expected_gc2_size = config.nhid * config.nclass;
    if gc2_weight_f32.len() != expected_gc2_size {
        return Err(PyTorchLoaderError::DimensionMismatch(format!(
            "gc2.weight dimension error: {} vs {} ({}*{})",
            gc2_weight_f32.len(),
            expected_gc2_size,
            config.nhid,
            config.nclass
        )));
    }

    let gc2_weight_data: Vec<T> = gc2_weight_f32
        .iter()
        .map(|&x| T::from_f64_exact(x as f64).unwrap())
        .collect();

    let gc2_bias_f32 = weights_dict
        .get("gc2.bias")
        .ok_or_else(|| PyTorchLoaderError::ParseError("gc2.bias missing".to_string()))?;

    if gc2_bias_f32.len() != config.nclass {
        return Err(PyTorchLoaderError::DimensionMismatch(format!(
            "gc2.bias dimension error: {} vs {}",
            gc2_bias_f32.len(),
            config.nclass
        )));
    }

    let gc2_bias: Vec<T> = gc2_bias_f32
        .iter()
        .map(|&x| T::from_f64_exact(x as f64).unwrap())
        .collect();

    let gc1_weight = DenseMatrix::new(gc1_weight_data, (config.nfeat, config.nhid));
    let gc2_weight = DenseMatrix::new(gc2_weight_data, (config.nhid, config.nclass));

    println!("Weight loading completed:");
    println!("  gc1.weight: {:?}", gc1_weight.shape);
    println!("  gc1.bias: {}", gc1_bias.len());
    println!("  gc2.weight: {:?}", gc2_weight.shape);
    println!("  gc2.bias: {}", gc2_bias.len());

    Ok(GCNWeights {
        gc1_weight,
        gc1_bias,
        gc2_weight,
        gc2_bias,
    })
}

///
/// ```json
/// {
///   "gc1.weight": [0.1, 0.2, ..., 0.5],
///   "gc1.bias": [0.1, 0.2, ..., 0.16],  
///   "gc2.weight": [0.3, 0.4, ..., 0.7],
///   "gc2.bias": [0.1, 0.2, ..., 0.7]
/// }
/// ```
///
/// ```python
/// import torch
/// import json
///
/// model = torch.load('model.pth')
/// weights_dict = {k: v.cpu().numpy().flatten().tolist()
///                for k, v in model.state_dict().items()}
///
/// with open('weights.json', 'w') as f:
///     json.dump(weights_dict, f)
/// ```
pub fn load_weights_from_json<T: FloatType>(
    json_path: &str,
    config: &GCNConfig,
) -> Result<GCNWeights<T>, PyTorchLoaderError> {
    use std::fs;

    println!("Load weights from JSON file: {}", json_path);

    let json_str = fs::read_to_string(json_path)
        .map_err(|e| PyTorchLoaderError::FileNotFound(format!("{}: {}", json_path, e)))?;

    let weights_dict: HashMap<String, Vec<f32>> = serde_json::from_str(&json_str)
        .map_err(|e| PyTorchLoaderError::ParseError(format!("JSON parsing failed: {}", e)))?;

    load_weights_from_dict(weights_dict, config)
}

///
pub fn create_random_weights<T: FloatType>(config: &GCNConfig, seed: u64) -> GCNWeights<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn simple_rand(seed: u64, i: usize) -> f64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    println!("Create random initialization weights (seed={})", seed);

    let mut gc1_weight_data = Vec::with_capacity(config.nfeat * config.nhid);
    for i in 0..(config.nfeat * config.nhid) {
        let val = simple_rand(seed, i) * 0.1;
        gc1_weight_data.push(T::from_f64_exact(val).unwrap());
    }

    let mut gc1_bias = Vec::with_capacity(config.nhid);
    for i in 0..config.nhid {
        let val = simple_rand(seed, 1000 + i) * 0.01;
        gc1_bias.push(T::from_f64_exact(val).unwrap());
    }

    let mut gc2_weight_data = Vec::with_capacity(config.nhid * config.nclass);
    for i in 0..(config.nhid * config.nclass) {
        let val = simple_rand(seed, 2000 + i) * 0.1;
        gc2_weight_data.push(T::from_f64_exact(val).unwrap());
    }

    let mut gc2_bias = Vec::with_capacity(config.nclass);
    for i in 0..config.nclass {
        let val = simple_rand(seed, 3000 + i) * 0.01;
        gc2_bias.push(T::from_f64_exact(val).unwrap());
    }

    let gc1_weight = DenseMatrix::new(gc1_weight_data, (config.nfeat, config.nhid));
    let gc2_weight = DenseMatrix::new(gc2_weight_data, (config.nhid, config.nclass));

    println!("Random weight creation is completed:");
    println!(
        "Total number of parameters: {}",
        gc1_weight.data.len() + gc1_bias.len() + gc2_weight.data.len() + gc2_bias.len()
    );

    GCNWeights {
        gc1_weight,
        gc1_bias,
        gc2_weight,
        gc2_bias,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_load_weights_from_dict() {
        let config = GCNConfig {
            nfeat: 3,
            nhid: 2,
            nclass: 2,
            dropout: 0.5,
        };

        let mut weights_dict = HashMap::new();
        weights_dict.insert("gc1.weight".to_string(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]); // 3*2=6
        weights_dict.insert("gc1.bias".to_string(), vec![0.1, 0.2]); // 2
        weights_dict.insert("gc2.weight".to_string(), vec![0.3, 0.4, 0.5, 0.6]); // 2*2=4
        weights_dict.insert("gc2.bias".to_string(), vec![0.1, 0.2]); // 2

        let weights = load_weights_from_dict::<DefaultFloat>(weights_dict, &config).unwrap();

        assert_eq!(weights.gc1_weight.shape, (3, 2));
        assert_eq!(weights.gc1_bias.len(), 2);
        assert_eq!(weights.gc2_weight.shape, (2, 2));
        assert_eq!(weights.gc2_bias.len(), 2);

        assert_eq!(weights.gc1_weight.get(0, 0), 0.1);
        assert_eq!(weights.gc1_weight.get(2, 1), 0.6);
        assert_eq!(weights.gc1_bias[1], 0.2);
    }

    #[test]
    fn test_create_random_weights() {
        let config = GCNConfig::default();
        let weights = create_random_weights::<DefaultFloat>(&config, 42);

        assert_eq!(weights.gc1_weight.shape, (1433, 16));
        assert_eq!(weights.gc1_bias.len(), 16);
        assert_eq!(weights.gc2_weight.shape, (16, 7));
        assert_eq!(weights.gc2_bias.len(), 7);

        assert!(weights.gc1_weight.data.iter().any(|&x| x != 0.0));
        assert!(weights.gc2_weight.data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = GCNConfig {
            nfeat: 3,
            nhid: 2,
            nclass: 2,
            dropout: 0.5,
        };

        let mut weights_dict = HashMap::new();
        weights_dict.insert("gc1.weight".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
        weights_dict.insert("gc1.bias".to_string(), vec![0.1, 0.2]);
        weights_dict.insert("gc2.weight".to_string(), vec![0.3, 0.4, 0.5, 0.6]);
        weights_dict.insert("gc2.bias".to_string(), vec![0.1, 0.2]);

        let result = load_weights_from_dict::<DefaultFloat>(weights_dict, &config);
        assert!(result.is_err());

        match result.unwrap_err() {
            PyTorchLoaderError::DimensionMismatch(_) => (),
            _ => panic!("It should be a dimension mismatch error"),
        }
    }
}
