use crate::types::{CoraDataset, DenseMatrix, FloatType, SparseMatrix};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

///
///
///
pub fn load_cora_data<T: FloatType>(
    data_path: &str,
) -> Result<CoraDataset<T>, Box<dyn std::error::Error>> {
    println!("Loading Cora dataset from: {}", data_path);

    if !Path::new(data_path).exists() {
        return Err(format!("Data path does not exist: {}", data_path).into());
    }

    let content_file = format!("{}/cora.content", data_path);
    let (features, labels, node_map) = load_features_and_labels::<T>(&content_file)?;

    println!(
        "Loading completed - number of nodes: {}, feature dimension: {}, number of categories: {}",
        features.shape.0,
        features.shape.1,
        labels.iter().max().unwrap_or(&0) + 1
    );

    let cites_file = format!("{}/cora.cites", data_path);
    let adj = load_adjacency_matrix::<T>(&cites_file, &node_map, features.shape.0)?;

    println!(
        "Adjacency matrix construction completed - non-zero elements: {}",
        adj.nnz()
    );

    let idx_train: Vec<i64> = (0..140).collect();
    let idx_val: Vec<i64> = (200..500).collect();
    let idx_test: Vec<i64> = (500..1500).collect();

    println!(
        "Dataset partitioning - training: {}, validation: {}, testing: {}",
        idx_train.len(),
        idx_val.len(),
        idx_test.len()
    );

    Ok(CoraDataset {
        features,
        adj,
        labels,
        idx_train,
        idx_val,
        idx_test,
    })
}

///
/// <node_id> <feature1> <feature2> ... <feature1433> <label>
///
fn load_features_and_labels<T: FloatType>(
    filepath: &str,
) -> Result<(DenseMatrix<T>, Vec<i64>, HashMap<String, usize>), Box<dyn std::error::Error>> {
    println!("Reading features and tags: {}", filepath);

    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();
    let mut node_map = HashMap::new();

    let mut num_features = 0;
    let mut node_idx = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.trim().split('\t').collect();

        if parts.len() < 3 {
            return Err(format!(
                "The data format of row {} is wrong and the number of fields is insufficient: {}",
                line_num + 1,
                parts.len()
            )
            .into());
        }

        let node_id = parts[0].to_string();
        node_map.insert(node_id, node_idx);
        node_idx += 1;

        let feature_parts = &parts[1..parts.len() - 1];
        if num_features == 0 {
            num_features = feature_parts.len();
            println!("Feature dimension detected: {}", num_features);
        } else if feature_parts.len() != num_features {
            return Err(format!(
                "The feature dimensions of row {} are inconsistent: {} vs {}",
                line_num + 1,
                feature_parts.len(),
                num_features
            )
            .into());
        }

        for &feat_str in feature_parts {
            let feat_val: f64 = feat_str.parse().map_err(|_| {
                format!(
                    "Characteristic value parsing failed at line {}: '{}'",
                    line_num + 1,
                    feat_str
                )
            })?;

            features_data.push(
                T::from_f64_exact(feat_val)
                    .ok_or_else(|| format!("Floating point conversion failed: {}", feat_val))?,
            );
        }

        let label = parts[parts.len() - 1];
        let label_idx = encode_cora_label(label)?;
        labels_data.push(label_idx);
    }

    let num_nodes = labels_data.len();
    println!(
        "Data reading completed - number of nodes: {}, feature dimension: {}",
        num_nodes, num_features
    );

    let mut features = DenseMatrix::new(features_data, (num_nodes, num_features));

    println!("Feature normalization in progress...");
    normalize_features(&mut features);

    Ok((features, labels_data, node_map))
}

///
///
fn load_adjacency_matrix<T: FloatType>(
    filepath: &str,
    node_map: &HashMap<String, usize>,
    num_nodes: usize,
) -> Result<SparseMatrix<T>, Box<dyn std::error::Error>> {
    println!("Reading adjacency matrix: {}", filepath);

    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let mut edges = Vec::new();
    let mut edge_count = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.trim().split('\t').collect();

        if parts.len() != 2 {
            return Err(format!("{} row edge data format error: {}", line_num + 1, line).into());
        }

        let src_idx = node_map
            .get(parts[0])
            .ok_or_else(|| format!("Source node not found: {}", parts[0]))?;
        let dst_idx = node_map
            .get(parts[1])
            .ok_or_else(|| format!("Target node not found: {}", parts[1]))?;

        edges.push((*src_idx, *dst_idx));
        edge_count += 1;
    }

    println!("Read edge number: {}", edge_count);

    println!("Constructing symmetric adjacency matrix...");
    let adj = build_symmetric_adjacency::<T>(edges, num_nodes);

    println!("Adding self-loops and normalizing...");
    let adj_normalized = add_self_loops_and_normalize(adj);

    adj_normalized
        .validate_indices()
        .map_err(|e| format!("Adjacency matrix validation failed: {}", e))?;

    Ok(adj_normalized)
}

///
/// adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
///
fn build_symmetric_adjacency<T: FloatType>(
    edges: Vec<(usize, usize)>,
    num_nodes: usize,
) -> SparseMatrix<T> {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    let mut edge_set = std::collections::HashSet::new();

    for (src, dst) in edges {
        if edge_set.insert((src, dst)) {
            indices.push((src as i64, dst as i64));
            values.push(T::one());
        }

        if src != dst && edge_set.insert((dst, src)) {
            indices.push((dst as i64, src as i64));
            values.push(T::one());
        }
    }

    println!("Number of edges after symmetrization: {}", indices.len());
    SparseMatrix::new(indices, values, (num_nodes, num_nodes))
}

///
fn add_self_loops_and_normalize<T: FloatType>(mut adj: SparseMatrix<T>) -> SparseMatrix<T> {
    let num_nodes = adj.shape.0;

    println!("Add self-loop ({} nodes)", num_nodes);
    for i in 0..num_nodes {
        adj.indices.push((i as i64, i as i64));
        adj.values.push(T::one());
    }

    println!("Perform row normalization...");
    normalize_adjacency_matrix(&mut adj);

    adj
}

///
fn normalize_features<T: FloatType>(features: &mut DenseMatrix<T>) {
    let (rows, _cols) = features.shape;

    for row in 0..rows {
        let row_data = features.get_row_mut(row);

        let row_sum: T = row_data.iter().fold(T::zero(), |acc, &x| acc + x);

        if row_sum > T::zero() {
            for val in row_data.iter_mut() {
                *val = *val / row_sum;
            }
        }
    }

    println!("Feature normalization completed");
}

///
/// rowsum = np.array(mx.sum(1))
/// r_inv = np.power(rowsum, -1).flatten()
/// r_inv[np.isinf(r_inv)] = 0.
/// r_mat_inv = sp.diags(r_inv)
/// mx = r_mat_inv.dot(mx)
fn normalize_adjacency_matrix<T: FloatType>(adj: &mut SparseMatrix<T>) {
    let num_nodes = adj.shape.0;
    let mut row_sums = vec![T::zero(); num_nodes];

    for (&(i, _), &val) in adj.indices.iter().zip(adj.values.iter()) {
        row_sums[i as usize] = row_sums[i as usize] + val;
    }

    for (val, &(i, _)) in adj.values.iter_mut().zip(adj.indices.iter()) {
        let row_sum = row_sums[i as usize];
        if row_sum > T::zero() {
            *val = *val / row_sum;
        }
    }

    println!("Adjacency matrix normalization completed");
}

///
fn encode_cora_label(label: &str) -> Result<i64, Box<dyn std::error::Error>> {
    let label_idx = match label {
        "Case_Based" => 0,
        "Genetic_Algorithms" => 1,
        "Neural_Networks" => 2,
        "Probabilistic_Methods" => 3,
        "Reinforcement_Learning" => 4,
        "Rule_Learning" => 5,
        "Theory" => 6,
        _ => return Err(format!("Unknown Cora tag: {}", label).into()),
    };

    Ok(label_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_data() -> Result<TempDir, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        let content_path = temp_dir.path().join("cora.content");
        let mut content_file = File::create(&content_path)?;
        writeln!(content_file, "1\t1.0\t0.0\tCase_Based")?;
        writeln!(content_file, "2\t0.0\t1.0\tNeural_Networks")?;
        writeln!(content_file, "3\t0.5\t0.5\tTheory")?;

        let cites_path = temp_dir.path().join("cora.cites");
        let mut cites_file = File::create(&cites_path)?;
        writeln!(cites_file, "1\t2")?;
        writeln!(cites_file, "2\t3")?;

        Ok(temp_dir)
    }

    #[test]
    fn test_label_encoding() {
        assert_eq!(encode_cora_label("Case_Based").unwrap(), 0);
        assert_eq!(encode_cora_label("Neural_Networks").unwrap(), 2);
        assert_eq!(encode_cora_label("Theory").unwrap(), 6);

        assert!(encode_cora_label("Unknown_Label").is_err());
    }

    #[test]
    fn test_load_features_and_labels() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = create_test_data()?;
        let content_path = temp_dir.path().join("cora.content");

        let (features, labels, node_map) =
            load_features_and_labels::<DefaultFloat>(content_path.to_str().unwrap())?;

        assert_eq!(features.shape, (3, 2));
        assert_eq!(labels.len(), 3);
        assert_eq!(node_map.len(), 3);

        assert_eq!(labels[0], 0); // Case_Based
        assert_eq!(labels[1], 2); // Neural_Networks
        assert_eq!(labels[2], 6); // Theory

        assert_eq!(node_map["1"], 0);
        assert_eq!(node_map["2"], 1);
        assert_eq!(node_map["3"], 2);

        Ok(())
    }

    #[test]
    fn test_adjacency_matrix_loading() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = create_test_data()?;
        let cites_path = temp_dir.path().join("cora.cites");

        let mut node_map = HashMap::new();
        node_map.insert("1".to_string(), 0);
        node_map.insert("2".to_string(), 1);
        node_map.insert("3".to_string(), 2);

        let adj =
            load_adjacency_matrix::<DefaultFloat>(cites_path.to_str().unwrap(), &node_map, 3)?;

        assert_eq!(adj.shape, (3, 3));

        let mut has_self_loops = vec![false; 3];
        for (&(i, j), _) in adj.indices.iter().zip(adj.values.iter()) {
            if i == j {
                has_self_loops[i as usize] = true;
            }
        }
        assert!(
            has_self_loops.iter().all(|&x| x),
            "All nodes should have self-loops"
        );

        assert!(adj.validate_indices().is_ok());

        Ok(())
    }
}
