#[cfg(feature = "zkp")]
use half::f16;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::{cmp::Ordering, fmt::Debug};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationPrecision {
    Fp16,
    Bf16,
    Fp32,
    Fp64,
}

impl OperationPrecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationPrecision::Fp16 => "fp16",
            OperationPrecision::Bf16 => "bf16",
            OperationPrecision::Fp32 => "fp32",
            OperationPrecision::Fp64 => "fp64",
        }
    }
}

///
pub trait FloatType: Float + FromPrimitive + Zero + One + Copy + Debug + Send + Sync {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn neg_infinity() -> Self;
    fn from_f64_exact(value: f64) -> Option<Self>;
    fn operation_precision() -> OperationPrecision;
}

impl FloatType for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn from_f64_exact(value: f64) -> Option<Self> {
        Some(value as f32)
    }
    fn operation_precision() -> OperationPrecision {
        OperationPrecision::Fp32
    }
}

impl FloatType for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn from_f64_exact(value: f64) -> Option<Self> {
        Some(value)
    }
    fn operation_precision() -> OperationPrecision {
        OperationPrecision::Fp64
    }
}

#[cfg(feature = "zkp")]
impl FloatType for f16 {
    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        f16::from_f32(self.to_f32().ln())
    }

    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }

    fn from_f64_exact(value: f64) -> Option<Self> {
        Some(f16::from_f64(value))
    }

    fn operation_precision() -> OperationPrecision {
        OperationPrecision::Fp16
    }
}

///
///
#[derive(Debug, Clone)]
pub struct DenseMatrix<T: FloatType> {
    pub data: Vec<T>,
    pub shape: (usize, usize),
}

impl<T: FloatType> DenseMatrix<T> {
    ///
    pub fn new(data: Vec<T>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Data length does not match matrix shape: {} vs {}*{}",
            data.len(),
            shape.0,
            shape.1
        );
        Self { data, shape }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::zero(); rows * cols],
            shape: (rows, cols),
        }
    }

    ///
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(
            row < self.shape.0 && col < self.shape.1,
            "Index out of range: ({},{}) vs ({},{})",
            row,
            col,
            self.shape.0,
            self.shape.1
        );
        self.data[row * self.shape.1 + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.shape.0 && col < self.shape.1);
        self.data[row * self.shape.1 + col] = value;
    }

    ///
    pub fn get_row(&self, row: usize) -> &[T] {
        let start = row * self.shape.1;
        let end = start + self.shape.1;
        &self.data[start..end]
    }

    pub fn get_row_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.shape.1;
        let end = start + self.shape.1;
        &mut self.data[start..end]
    }
}

///
/// - PyTorch: torch.sparse.FloatTensor -> indices: int64, values: float32
/// - Rust: SparseMatrix<f32> -> indices: (i64,i64), values: Vec<f32>
///
#[derive(Debug, Clone)]
pub struct SparseMatrix<T: FloatType> {
    pub indices: Vec<(i64, i64)>,
    pub values: Vec<T>,
    pub shape: (usize, usize),
}

///
#[derive(Debug, Clone)]
pub struct CsrMatrix<T: FloatType> {
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
    pub shape: (usize, usize),
}

impl<T: FloatType> CsrMatrix<T> {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn rows(&self) -> usize {
        self.shape.0
    }

    pub fn cols(&self) -> usize {
        self.shape.1
    }
}

impl<T: FloatType> SparseMatrix<T> {
    ///
    pub fn new(indices: Vec<(i64, i64)>, values: Vec<T>, shape: (usize, usize)) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "The number of indexes and values must be the same: {} vs {}",
            indices.len(),
            values.len()
        );
        Self {
            indices,
            values,
            shape,
        }
    }

    ///
    /// - row_indices: torch.int64 tensor
    /// - col_indices: torch.int64 tensor  
    /// - values: torch.float32 tensor
    pub fn from_pytorch_coo(
        row_indices: Vec<i64>,
        col_indices: Vec<i64>,
        values: Vec<T>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(row_indices.len(), col_indices.len());
        assert_eq!(row_indices.len(), values.len());

        let indices: Vec<(i64, i64)> = row_indices
            .into_iter()
            .zip(col_indices.into_iter())
            .collect();
        Self::new(indices, values, shape)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn validate_indices(&self) -> Result<(), String> {
        let (rows, cols) = self.shape;

        for &(i, j) in &self.indices {
            if i < 0 || j < 0 || i as usize >= rows || j as usize >= cols {
                return Err(format!(
                    "Index out of range: ({}, {}) vs shape ({}, {})",
                    i, j, rows, cols
                ));
            }
        }
        Ok(())
    }

    pub fn to_csr(&self) -> Result<CsrMatrix<T>, String> {
        self.validate_indices()?;

        let (rows, cols) = self.shape;
        let mut entries: Vec<(usize, usize, T)> = self
            .indices
            .iter()
            .zip(self.values.iter())
            .map(|(&(row, col), &value)| (row as usize, col as usize, value))
            .collect();

        entries.sort_by(
            |(row_a, col_a, _), (row_b, col_b, _)| match row_a.cmp(row_b) {
                Ordering::Equal => col_a.cmp(col_b),
                other => other,
            },
        );

        let mut row_ptr = vec![0usize; rows + 1];
        for (row, _, _) in &entries {
            row_ptr[row + 1] += 1;
        }
        for row in 1..=rows {
            row_ptr[row] += row_ptr[row - 1];
        }

        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for (_, col, value) in entries {
            col_indices.push(col);
            values.push(value);
        }

        Ok(CsrMatrix {
            row_ptr,
            col_indices,
            values,
            shape: (rows, cols),
        })
    }
}

///
/// - features: PyTorch torch.float32 [2708, 1433] -> Rust DenseMatrix<f32>
/// - adj: PyTorch sparse FloatTensor [2708, 2708] -> Rust SparseMatrix<f32>  
/// - labels: PyTorch torch.int64 [2708] -> Rust Vec<i64>
#[derive(Debug)]
pub struct CoraDataset<T: FloatType> {
    pub features: DenseMatrix<T>,
    pub adj: SparseMatrix<T>,
    pub labels: Vec<i64>,
    pub idx_train: Vec<i64>,
    pub idx_val: Vec<i64>,
    pub idx_test: Vec<i64>,
}

///
/// - gc1.weight: [1433, 16], gc1.bias: [16]
/// - gc2.weight: [16, 7], gc2.bias: [7]
#[derive(Debug)]
pub struct GCNWeights<T: FloatType> {
    pub gc1_weight: DenseMatrix<T>,
    pub gc1_bias: Vec<T>,
    pub gc2_weight: DenseMatrix<T>,
    pub gc2_bias: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct GCNConfig {
    pub nfeat: usize,
    pub nhid: usize,
    pub nclass: usize,
    pub dropout: f64,
}

impl Default for GCNConfig {
    fn default() -> Self {
        Self {
            nfeat: 1433,
            nhid: 16,
            nclass: 7,
            dropout: 0.5,
        }
    }
}

///
pub type DefaultFloat = f32;
pub type DefaultInt = i64;

#[cfg(test)]
mod tests {
    use super::{DefaultFloat, SparseMatrix};

    #[test]
    fn coo_to_csr_produces_sorted_rows() {
        let indices = vec![(1, 2), (0, 1), (1, 0)];
        let values = vec![2.0, 3.0, 4.0];
        let sparse = SparseMatrix::<DefaultFloat>::new(indices, values, (3, 3));

        let csr = sparse.to_csr().expect("Conversion failed");

        assert_eq!(csr.row_ptr, vec![0, 1, 3, 3]);
        assert_eq!(csr.col_indices, vec![1, 0, 2]);
        assert_eq!(csr.values, vec![3.0, 4.0, 2.0]);
    }

    #[test]
    fn coo_to_csr_validates_indices() {
        let indices = vec![(2, 3)];
        let values = vec![1.0];
        let sparse = SparseMatrix::<DefaultFloat>::new(indices, values, (2, 2));
        let err = sparse.to_csr().expect_err("Index error should be returned");
        assert!(err.contains("Index out of range"));
    }
}
