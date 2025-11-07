use crate::layers::graph_conv::GraphConvolution;
use crate::math::activations::{log_softmax, relu};
use crate::types::{DenseMatrix, FloatType, GCNConfig, GCNWeights, SparseMatrix};

/// Two-layer Kipf-Welling style Graph Convolutional Network that mirrors the
/// reference PyTorch implementation: `relu(gc1(x, adj)) -> gc2(...) -> log_softmax`.
/// Shape validation is performed before any heavy linear algebra so that
/// mismatched exports fail fast with actionable errors.
#[derive(Debug)]
pub struct GCN<T: FloatType> {
    pub gc1: GraphConvolution<T>,
    pub gc2: GraphConvolution<T>,
    pub config: GCNConfig,
}

impl<T: FloatType> GCN<T> {
    /// Creates a zero-initialized model. Useful for tests or placeholder
    /// construction when weights are assigned later with [`set_weight`](crate::layers::graph_conv::GraphConvolution::set_weight).
    pub fn new(config: GCNConfig) -> Self {
        let gc1 = GraphConvolution::new(config.nfeat, config.nhid, true);
        let gc2 = GraphConvolution::new(config.nhid, config.nclass, true);

        Self { gc1, gc2, config }
    }

    /// Builds a model from serialized tensors produced by the PyTorch helper.
    /// Shape assertions prevent accidentally loading weights that target a
    /// different feature or hidden dimension.
    ///
    /// * `gc1`: `[nfeat, nhid]` weight plus `[nhid]` bias
    /// * `gc2`: `[nhid, nclass]` weight plus `[nclass]` bias
    pub fn from_weights(weights: GCNWeights<T>, config: GCNConfig) -> Self {
        assert_eq!(
            weights.gc1_weight.shape,
            (config.nfeat, config.nhid),
            "gc1 weight shape mismatch: {:?} vs ({}, {})",
            weights.gc1_weight.shape,
            config.nfeat,
            config.nhid
        );
        assert_eq!(
            weights.gc1_bias.len(),
            config.nhid,
            "gc1 offset length mismatch: {} vs {}",
            weights.gc1_bias.len(),
            config.nhid
        );
        assert_eq!(
            weights.gc2_weight.shape,
            (config.nhid, config.nclass),
            "gc2 weight shape mismatch: {:?} vs ({}, {})",
            weights.gc2_weight.shape,
            config.nhid,
            config.nclass
        );
        assert_eq!(
            weights.gc2_bias.len(),
            config.nclass,
            "gc2 offset length mismatch: {} vs {}",
            weights.gc2_bias.len(),
            config.nclass
        );

        let gc1 = GraphConvolution::from_weights(
            weights.gc1_weight.data,
            Some(weights.gc1_bias),
            config.nfeat,
            config.nhid,
        );

        let gc2 = GraphConvolution::from_weights(
            weights.gc2_weight.data,
            Some(weights.gc2_bias),
            config.nhid,
            config.nclass,
        );

        Self { gc1, gc2, config }
    }

    /// Executes the full inference pipeline,
    /// equivalent to the PyTorch snippet:
    ///
    /// ```python
    /// x = F.relu(self.gc1(x, adj))
    /// x = self.gc2(x, adj)
    /// return F.log_softmax(x, dim=1)
    /// ```
    pub fn forward(&self, x: &DenseMatrix<T>, adj: &SparseMatrix<T>) -> DenseMatrix<T> {
        let (num_nodes, input_features) = x.shape;
        assert_eq!(
            input_features, self.config.nfeat,
            "Input feature dimension mismatch: {} vs {}",
            input_features, self.config.nfeat
        );

        let (adj_rows, adj_cols) = adj.shape;
        assert_eq!(
            (adj_rows, adj_cols),
            (num_nodes, num_nodes),
            "Adjacency matrix dimension mismatch: ({}, {}) vs ({}, {})",
            adj_rows,
            adj_cols,
            num_nodes,
            num_nodes
        );

        // x1 = relu(gc1(x, adj))
        let x1 = self.gc1.forward(x, adj);
        let x1_relu = relu(&x1);

        let x1_after_dropout = x1_relu;

        // x2 = gc2(x1_after_dropout, adj)
        let x2 = self.gc2.forward(&x1_after_dropout, adj);

        // output = log_softmax(x2, dim=1)
        log_softmax(&x2, 1)
    }

    /// Applies argmax to the logits from [`GCN::forward`] and returns one label
    /// per node, mirroring `output.max(1)[1]` from the PyTorch training loop.
    pub fn predict(&self, x: &DenseMatrix<T>, adj: &SparseMatrix<T>) -> Vec<usize> {
        let output = self.forward(x, adj);
        let (num_nodes, _) = output.shape;

        let mut predictions = Vec::with_capacity(num_nodes);

        for node in 0..num_nodes {
            let node_logits = output.get_row(node);
            let max_class = node_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions.push(max_class);
        }

        predictions
    }

    /// Computes accuracy restricted to the indices in `mask`. Each entry of the
    /// mask selects a node that participates in the metric, matching the helper
    /// PyTorch script's `accuracy` function.
    pub fn accuracy(
        &self,
        x: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        labels: &[i64],
        mask: &[i64],
    ) -> f64 {
        let predictions = self.predict(x, adj);

        let mut correct = 0;
        let mut total = 0;

        for &node_idx in mask {
            let node_idx = node_idx as usize;

            if node_idx < predictions.len() && node_idx < labels.len() {
                let pred_class = predictions[node_idx];
                let true_class = labels[node_idx] as usize;

                if pred_class == true_class {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Negative log-likelihood loss averaged over the masked nodes. The values
    /// align with PyTorch's `NLLLoss` applied to log-softmax outputs.
    pub fn nll_loss(
        &self,
        x: &DenseMatrix<T>,
        adj: &SparseMatrix<T>,
        labels: &[i64],
        mask: &[i64],
    ) -> T {
        let output = self.forward(x, adj);
        let mut total_loss = T::zero();
        let mut count = 0;

        for &node_idx in mask {
            let node_idx = node_idx as usize;

            if node_idx < labels.len() && node_idx < output.shape.0 {
                let true_class = labels[node_idx] as usize;
                if true_class < output.shape.1 {
                    let log_prob = output.get(node_idx, true_class);
                    total_loss = total_loss - log_prob;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_loss / T::from_f64_exact(count as f64).unwrap()
        } else {
            T::zero()
        }
    }

    /// Returns the total number of learnt scalars in both graph convolution
    /// layers, useful for comparing against the PyTorch baseline.
    pub fn num_parameters(&self) -> usize {
        self.gc1.num_parameters() + self.gc2.num_parameters()
    }

    /// Provides read-only access to the configuration used when constructing the model.
    pub fn config(&self) -> &GCNConfig {
        &self.config
    }

    /// Human readable summary similar to PyTorch's `__repr__`, handy for logging.
    pub fn layer_info(&self) -> String {
        format!(
            "GCN(\n  {}\n  {}\n  dropout: {:.2}\n)",
            self.gc1, self.gc2, self.config.dropout
        )
    }
}

impl<T: FloatType> std::fmt::Display for GCN<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GCN(nfeat={}, nhid={}, nclass={}, params={})",
            self.config.nfeat,
            self.config.nhid,
            self.config.nclass,
            self.num_parameters()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_gcn_creation() {
        let config = GCNConfig {
            nfeat: 10,
            nhid: 5,
            nclass: 3,
            dropout: 0.5,
        };

        let model = GCN::<DefaultFloat>::new(config.clone());

        assert_eq!(model.config.nfeat, 10);
        assert_eq!(model.config.nhid, 5);
        assert_eq!(model.config.nclass, 3);

        assert_eq!(model.num_parameters(), 73);
    }

    #[test]
    fn test_gcn_forward() {
        let config = GCNConfig {
            nfeat: 2,
            nhid: 3,
            nclass: 2,
            dropout: 0.5,
        };

        let model = GCN::<DefaultFloat>::new(config);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = DenseMatrix::new(input_data, (3, 2));

        let adj_indices = vec![(0, 0), (1, 1), (2, 2)];
        let adj_values = vec![1.0, 1.0, 1.0];
        let adj = SparseMatrix::new(adj_indices, adj_values, (3, 3));

        let output = model.forward(&input, &adj);

        assert_eq!(output.shape, (3, 2));

        for node in 0..3 {
            let node_probs = output.get_row(node);
            let prob_sum: DefaultFloat = node_probs.iter().map(|&x| x.exp()).sum();
            assert!(
                (prob_sum - 1.0).abs() < 1e-5,
                "The probability sum of node {} is not equal to 1: {}",
                node,
                prob_sum
            );
        }
    }

    #[test]
    fn test_prediction_and_accuracy() {
        let config = GCNConfig {
            nfeat: 2,
            nhid: 2,
            nclass: 2,
            dropout: 0.0,
        };

        let model = GCN::<DefaultFloat>::new(config);

        let input_data = vec![1.0, 0.0, 0.0, 1.0];
        let input = DenseMatrix::new(input_data, (2, 2));

        let adj_indices = vec![(0, 0), (1, 1)];
        let adj_values = vec![1.0, 1.0];
        let adj = SparseMatrix::new(adj_indices, adj_values, (2, 2));

        let predictions = model.predict(&input, &adj);
        assert_eq!(predictions.len(), 2);

        let labels = vec![0i64, 1i64];
        let mask = vec![0i64, 1i64];

        let acc = model.accuracy(&input, &adj, &labels, &mask);
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "Accuracy should be between 0-1: {}",
            acc
        );
    }

    #[test]
    fn test_nll_loss() {
        let config = GCNConfig {
            nfeat: 2,
            nhid: 2,
            nclass: 2,
            dropout: 0.0,
        };

        let model = GCN::<DefaultFloat>::new(config);

        let input_data = vec![1.0, 0.0, 0.0, 1.0];
        let input = DenseMatrix::new(input_data, (2, 2));

        let adj_indices = vec![(0, 0), (1, 1)];
        let adj_values = vec![1.0, 1.0];
        let adj = SparseMatrix::new(adj_indices, adj_values, (2, 2));

        let labels = vec![0i64, 1i64];
        let mask = vec![0i64, 1i64];

        let loss = model.nll_loss(&input, &adj, &labels, &mask);
        assert!(loss >= 0.0, "NLL loss should be non-negative: {}", loss);
    }
}
