use crate::math::{dense_ops, sparse_ops};
use crate::types::{DenseMatrix, FloatType, SparseMatrix};

///
/// - weight: torch.Parameter([in_features, out_features], dtype=torch.float32)
///
#[derive(Debug)]
pub struct GraphConvolution<T: FloatType> {
    pub weight: DenseMatrix<T>,
    pub bias: Option<Vec<T>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl<T: FloatType> GraphConvolution<T> {
    ///
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = DenseMatrix::zeros(in_features, out_features);
        let bias = if bias {
            Some(vec![T::zero(); out_features])
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    ///
    ///
    pub fn from_weights(
        weight_data: Vec<T>,
        bias_data: Option<Vec<T>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(
            weight_data.len(),
            in_features * out_features,
            "Weight data length mismatch: {} vs {}*{}",
            weight_data.len(),
            in_features,
            out_features
        );

        let weight = DenseMatrix::new(weight_data, (in_features, out_features));

        if let Some(ref bias_vec) = bias_data {
            assert_eq!(
                bias_vec.len(),
                out_features,
                "Offset length mismatch: {} vs {}",
                bias_vec.len(),
                out_features
            );
        }

        Self {
            weight,
            bias: bias_data,
            in_features,
            out_features,
        }
    }

    ///
    ///
    /// 1. support = torch.mm(input, self.weight)  -> dense_mm(input, weight)
    /// 2. output = torch.spmm(adj, support)       -> sparse_dense_mm(adj, support)  
    /// 3. return output + self.bias               -> add_bias(output, bias)
    pub fn forward(&self, input: &DenseMatrix<T>, adj: &SparseMatrix<T>) -> DenseMatrix<T> {
        let (num_nodes, input_features) = input.shape;
        assert_eq!(
            input_features, self.in_features,
            "Input feature dimension mismatch: {} vs {}",
            input_features, self.in_features
        );

        let (adj_rows, adj_cols) = adj.shape;
        assert_eq!(
            adj_rows, num_nodes,
            "Adjacency matrix row number does not match node number: {} vs {}",
            adj_rows, num_nodes
        );
        assert_eq!(
            adj_cols, num_nodes,
            "The adjacency matrix should be a square matrix: {}x{}",
            adj_rows, adj_cols
        );

        // input: [num_nodes, in_features] @ weight: [in_features, out_features]
        // -> support: [num_nodes, out_features]
        let support = dense_ops::dense_mm(input, &self.weight);

        // adj: [num_nodes, num_nodes] @ support: [num_nodes, out_features]
        // -> output: [num_nodes, out_features]
        let mut output = sparse_ops::sparse_dense_mm(adj, &support);

        if let Some(ref bias) = self.bias {
            dense_ops::add_bias(&mut output, bias);
        }

        output
    }

    ///
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = self.bias.as_ref().map_or(0, |b| b.len());
        weight_params + bias_params
    }

    pub fn weight_shape(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }

    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    ///
    pub fn set_weight(&mut self, weight_data: Vec<T>) {
        assert_eq!(weight_data.len(), self.in_features * self.out_features);
        self.weight = DenseMatrix::new(weight_data, (self.in_features, self.out_features));
    }

    pub fn set_bias(&mut self, bias_data: Vec<T>) {
        assert_eq!(bias_data.len(), self.out_features);
        self.bias = Some(bias_data);
    }

    pub fn remove_bias(&mut self) {
        self.bias = None;
    }
}

impl<T: FloatType> std::fmt::Display for GraphConvolution<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphConvolution(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            self.has_bias()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_graph_convolution_creation() {
        let gc = GraphConvolution::<DefaultFloat>::new(10, 5, true);

        assert_eq!(gc.in_features, 10);
        assert_eq!(gc.out_features, 5);
        assert!(gc.has_bias());
        assert_eq!(gc.num_parameters(), 10 * 5 + 5);

        let gc_no_bias = GraphConvolution::<DefaultFloat>::new(10, 5, false);
        assert!(!gc_no_bias.has_bias());
        assert_eq!(gc_no_bias.num_parameters(), 10 * 5);
    }

    #[test]
    fn test_graph_convolution_forward() {
        let mut gc = GraphConvolution::<DefaultFloat>::new(2, 2, true);

        let weight_data = vec![1.0, 0.0, 0.0, 1.0];
        gc.set_weight(weight_data);

        let bias_data = vec![0.0, 0.0];
        gc.set_bias(bias_data);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]
        let input = DenseMatrix::new(input_data, (2, 2));

        let adj_indices = vec![(0, 0), (1, 1)];
        let adj_values = vec![1.0, 1.0];
        let adj = SparseMatrix::new(adj_indices, adj_values, (2, 2));

        let output = gc.forward(&input, &adj);

        assert_eq!(output.get(0, 0), 1.0);
        assert_eq!(output.get(0, 1), 2.0);
        assert_eq!(output.get(1, 0), 3.0);
        assert_eq!(output.get(1, 1), 4.0);
    }

    #[test]
    fn test_from_weights() {
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2, 0.3];

        let gc = GraphConvolution::from_weights(weight_data, Some(bias_data), 2, 3);

        assert_eq!(gc.in_features, 2);
        assert_eq!(gc.out_features, 3);
        assert!(gc.has_bias());

        assert_eq!(gc.weight.get(0, 0), 0.1);
        assert_eq!(gc.weight.get(0, 1), 0.2);
        assert_eq!(gc.weight.get(0, 2), 0.3);
        assert_eq!(gc.weight.get(1, 0), 0.4);
        assert_eq!(gc.weight.get(1, 1), 0.5);
        assert_eq!(gc.weight.get(1, 2), 0.6);

        let bias = gc.bias.as_ref().unwrap();
        assert_eq!(bias[0], 0.1);
        assert_eq!(bias[1], 0.2);
        assert_eq!(bias[2], 0.3);
    }

    #[test]
    #[should_panic(expected = "Input feature dimensions do not match")]
    fn test_dimension_mismatch() {
        let gc = GraphConvolution::<DefaultFloat>::new(3, 2, false);

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = DenseMatrix::new(input_data, (2, 2));

        let adj_indices = vec![(0, 0), (1, 1)];
        let adj_values = vec![1.0, 1.0];
        let adj = SparseMatrix::new(adj_indices, adj_values, (2, 2));

        let _output = gc.forward(&input, &adj);
    }
}
