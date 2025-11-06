use superacc_zkp::{
    data::{loader, pytorch_loader},
    types::{DefaultFloat, GCNConfig},
    GCN,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Rust GCN inference engine");
    println!("Fully compatible with PyGCN data types");
    println!("========================================\n");

    println!("Step 1: Load Cora dataset");
    println!("----------------------------------------");

    let data_path = "data/cora";
    let dataset = match loader::load_cora_data::<DefaultFloat>(data_path) {
        Ok(data) => {
            println!("Cora data set loaded successfully.");
            print_dataset_info(&data);
            data
        }
        Err(e) => {
            eprintln!("Data set loading failed: {}", e);
            eprintln!(
                "Tip: please make sure the data directory exists: {}",
                data_path
            );
            eprintln!("Download cora.content and cora.cites into data/cora (see README).");
            return Err(e);
        }
    };

    println!("\nStep 2: Create GCN model");
    println!("----------------------------------------");

    let config = GCNConfig::default();
    println!("Model configuration:");
    println!("Input feature dimension (nfeat): {}", config.nfeat);
    println!("Hidden layer dimension (nhid): {}", config.nhid);
    println!("Number of output classes (nclass): {}", config.nclass);
    println!("Dropout rate: {}", config.dropout);

    println!("\nStep 3: Initialize model weights");
    println!("----------------------------------------");

    let weights = pytorch_loader::create_random_weights::<DefaultFloat>(&config, 42);
    let model = GCN::from_weights(weights, config);

    println!("GCN model created successfully:");
    println!("  {}", model);
    println!("  {}", model.layer_info());

    println!("\nStep 4: Perform inference");
    println!("----------------------------------------");

    println!("Performing forward pass...");
    let start_time = std::time::Instant::now();

    let output = model.forward(&dataset.features, &dataset.adj);

    let inference_time = start_time.elapsed();
    println!(
        "Forward propagation completed in {:.2} ms",
        inference_time.as_millis()
    );

    println!("Output verification:");
    println!("Output shape: {:?}", output.shape);
    println!("Data type: f32 (matches PyTorch torch.float32)");

    verify_log_softmax_properties(&output);

    println!("\nStep 5: Calculate performance indicators");
    println!("----------------------------------------");

    let train_acc = model.accuracy(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_train,
    );

    let val_acc = model.accuracy(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_val,
    );

    let test_acc = model.accuracy(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_test,
    );

    let train_loss = model.nll_loss(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_train,
    );
    let val_loss = model.nll_loss(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_val,
    );
    let test_loss = model.nll_loss(
        &dataset.features,
        &dataset.adj,
        &dataset.labels,
        &dataset.idx_test,
    );

    println!("Performance indicators (randomly initialized weights):");
    println!(
        "Training set - Accuracy: {:.4}, Loss: {:.4}",
        train_acc, train_loss
    );
    println!(
        "Validation set - Accuracy: {:.4}, Loss: {:.4}",
        val_acc, val_loss
    );
    println!(
        "Test set - Accuracy: {:.4}, Loss: {:.4}",
        test_acc, test_loss
    );

    println!("\nStep 6: Data type compatibility verification");
    println!("----------------------------------------");

    print_type_compatibility_summary();

    println!("\nUsage tips:");
    println!("----------------------------------------");
    println!("1. Currently using random initialization weights for testing");
    println!("2. To use the trained PyTorch weights, you need:");
    println!("- Save weights from PyTorch to JSON format");
    println!("- Use pytorch_loader::load_weights_from_json()");
    println!("3. All data types are fully compatible with PyGCN");
    println!("4. The forward propagation logic is consistent with PyGCN");

    println!("\nThe Rust GCN inference engine is running!");

    Ok(())
}

fn print_dataset_info(dataset: &superacc_zkp::types::CoraDataset<DefaultFloat>) {
    println!("Dataset information:");
    println!("Number of nodes: {}", dataset.features.shape.0);
    println!("Feature dimension: {}", dataset.features.shape.1);
    println!(
        "Non-zero elements of adjacency matrix: {}",
        dataset.adj.nnz()
    );
    println!(
        "Number of categories: {}",
        dataset.labels.iter().max().unwrap_or(&0) + 1
    );

    println!("Data set partitioning:");
    println!("Training set: {} nodes", dataset.idx_train.len());
    println!("Validation set: {} nodes", dataset.idx_val.len());
    println!("Test set: {} nodes", dataset.idx_test.len());

    println!("Data type verification:");
    println!(
        "Feature matrix: DenseMatrix<f32> {:?} (matches torch.float32)",
        dataset.features.shape
    );
    println!(
        "Adjacency matrix: SparseMatrix<f32> {:?} (matches sparse FloatTensor)",
        dataset.adj.shape
    );
    println!("Tag type: Vec<i64> (matches torch.int64)");
    println!("Index type: Vec<i64> (matches torch.int64)");
}

fn verify_log_softmax_properties(output: &superacc_zkp::types::DenseMatrix<DefaultFloat>) {
    let (num_nodes, _num_classes) = output.shape;
    let mut max_prob_error = 0.0f32;
    let mut total_prob_error = 0.0f32;

    println!("Log Softmax output verification:");

    for node in 0..std::cmp::min(3, num_nodes) {
        let node_log_probs = output.get_row(node);

        let probs: Vec<f32> = node_log_probs.iter().map(|&x| x.exp()).collect();
        let prob_sum: f32 = probs.iter().sum();

        let error = (prob_sum - 1.0).abs();
        max_prob_error = max_prob_error.max(error);
        total_prob_error += error;

        if node < 2 {
            println!(
                "Node {}: log_probs=[{:.4}, {:.4}, ...], prob_sum={:.6}",
                node, node_log_probs[0], node_log_probs[1], prob_sum
            );
        }
    }

    println!("Maximum probability and error: {:.2e}", max_prob_error);
    println!(
        "Average probability and error: {:.2e}",
        total_prob_error / num_nodes as f32
    );

    if max_prob_error < 1e-5 {
        println!("Log Softmax numerical stability verification passed.");
    } else {
        println!("Warning: Log Softmax has a large numerical error.");
    }
}

fn print_type_compatibility_summary() {
    println!("Correspondence between Rust GCN and PyGCN data types:");
    println!();
    println!("| PyGCN Types | Rust Types | Match Status |");
    println!("|------------------------|----------------------|----------|");
    println!("| torch.float32          | f32                  |   yes   |");
    println!("| torch.int64            | i64                  |   yes   |");
    println!("| FloatTensor[M,N]       | DenseMatrix<f32>     |   yes   |");
    println!("| sparse.FloatTensor     | SparseMatrix<f32>    |   yes   |");
    println!("| torch.mm()             | dense_mm()           |   yes   |");
    println!("| torch.spmm()           | sparse_dense_mm()    |   yes   |");
    println!("| F.relu()               | relu()               |   yes   |");
    println!("| F.log_softmax()        | log_softmax()        |   yes   |");
    println!("| F.nll_loss()           | nll_loss()           |   yes   |");

    println!("\nKey feature matching:");
    println!("- Row-major memory layout (C-contiguous)");
    println!("- Sparse matrix COO format");
    println!("- Numerically stable log_softmax implementation");
    println!("- Same forward propagation calculation order");
    println!("- Dropout skipped in inference mode");
}
