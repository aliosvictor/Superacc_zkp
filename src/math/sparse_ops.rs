use crate::types::{DenseMatrix, FloatType, SparseMatrix};

#[inline]
fn tracked_mul<T: FloatType>(lhs: T, rhs: T) -> T {
    #[cfg(feature = "zkp")]
    {
        crate::zkp::operation_tracker::mul_op(lhs, rhs)
    }
    #[cfg(not(feature = "zkp"))]
    {
        lhs * rhs
    }
}

#[inline]
fn tracked_add<T: FloatType>(lhs: T, rhs: T) -> T {
    #[cfg(feature = "zkp")]
    {
        crate::zkp::operation_tracker::add_op(lhs, rhs)
    }
    #[cfg(not(feature = "zkp"))]
    {
        lhs + rhs
    }
}

///
/// - PyTorch: torch.spmm(adj, support)
/// - adj.indices: torch.int64, adj.values: torch.float32
///
/// - sparse: SparseMatrix<f32> (indices: Vec<(i64,i64)>, values: Vec<f32>)
///
pub fn sparse_dense_mm<T: FloatType>(
    sparse: &SparseMatrix<T>,
    dense: &DenseMatrix<T>,
) -> DenseMatrix<T> {
    let (m, k) = sparse.shape;
    let (k2, n) = dense.shape;
    assert_eq!(
        k, k2,
        "Matrix dimensions do not match and cannot be multiplied: sparse[{},{}] @ dense[{},{}]",
        m, k, k2, n
    );

    sparse
        .validate_indices()
        .expect("Invalid sparse matrix index");

    let mut result = DenseMatrix::zeros(m, n);

    for (&(i, j), &val) in sparse.indices.iter().zip(sparse.values.iter()) {
        let i = i as usize;
        let j = j as usize;

        debug_assert!(
            i < m && j < k,
            "Sparse matrix index out of range: ({},{}) vs ({},{})",
            i,
            j,
            m,
            k
        );

        for col in 0..n {
            let current = result.get(i, col);
            let product = tracked_mul(val, dense.get(j, col));
            let update = tracked_add(current, product);
            result.set(i, col, update);
        }
    }

    result
}

///
pub fn validate_sparse_matrix<T: FloatType>(sparse: &SparseMatrix<T>) -> Result<(), String> {
    let (rows, cols) = sparse.shape;

    for &(i, j) in &sparse.indices {
        if i < 0 || j < 0 {
            return Err(format!("Negative index found: ({}, {})", i, j));
        }

        let i_usize = i as usize;
        let j_usize = j as usize;

        if i_usize >= rows || j_usize >= cols {
            return Err(format!(
                "Index out of range: ({}, {}) vs matrix shape ({}, {})",
                i, j, rows, cols
            ));
        }
    }

    if sparse.indices.len() != sparse.values.len() {
        return Err(format!(
            "The number of indexes ({}) does not match the number of values ({})",
            sparse.indices.len(),
            sparse.values.len()
        ));
    }

    Ok(())
}

///
#[allow(dead_code)]
pub fn sparse_to_dense<T: FloatType>(sparse: &SparseMatrix<T>) -> DenseMatrix<T> {
    let (rows, cols) = sparse.shape;
    let mut dense = DenseMatrix::zeros(rows, cols);

    for (&(i, j), &val) in sparse.indices.iter().zip(sparse.values.iter()) {
        dense.set(i as usize, j as usize, val);
    }

    dense
}

///
#[allow(dead_code)]
pub fn make_symmetric<T: FloatType>(sparse: &SparseMatrix<T>) -> SparseMatrix<T> {
    let mut new_indices = Vec::new();
    let mut new_values = Vec::new();

    for (&(i, j), &val) in sparse.indices.iter().zip(sparse.values.iter()) {
        new_indices.push((i, j));
        new_values.push(val);

        if i != j {
            new_indices.push((j, i));
            new_values.push(val);
        }
    }

    SparseMatrix::new(new_indices, new_values, sparse.shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_sparse_dense_multiply() {
        let indices = vec![(0, 0), (1, 1)];
        let values = vec![1.0, 2.0];
        let sparse = SparseMatrix::<DefaultFloat>::new(indices, values, (2, 2));

        let dense_data = vec![3.0, 4.0, 5.0, 6.0];
        let dense = DenseMatrix::<DefaultFloat>::new(dense_data, (2, 2));

        let result = sparse_dense_mm(&sparse, &dense);

        assert_eq!(result.get(0, 0), 3.0); // 1*3 + 0*5 = 3
        assert_eq!(result.get(0, 1), 4.0); // 1*4 + 0*6 = 4
        assert_eq!(result.get(1, 0), 10.0); // 0*3 + 2*5 = 10
        assert_eq!(result.get(1, 1), 12.0); // 0*4 + 2*6 = 12
    }

    #[test]
    fn test_sparse_matrix_validation() {
        let indices = vec![(0, 0), (1, 1)];
        let values = vec![1.0, 2.0];
        let sparse = SparseMatrix::<DefaultFloat>::new(indices, values, (2, 2));

        assert!(validate_sparse_matrix(&sparse).is_ok());

        let bad_indices = vec![(0, 0), (2, 1)];
        let bad_values = vec![1.0, 2.0];
        let bad_sparse = SparseMatrix::<DefaultFloat>::new(bad_indices, bad_values, (2, 2));

        assert!(validate_sparse_matrix(&bad_sparse).is_err());
    }
}
