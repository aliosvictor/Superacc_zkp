use crate::types::{DenseMatrix, FloatType};

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
///
pub fn dense_mm<T: FloatType>(a: &DenseMatrix<T>, b: &DenseMatrix<T>) -> DenseMatrix<T> {
    let (m, k) = a.shape;
    let (k2, n) = b.shape;
    assert_eq!(
        k, k2,
        "Matrix dimensions do not match and cannot be multiplied: A[{},{}] @ B[{},{}]",
        m, k, k2, n
    );

    let mut result = DenseMatrix::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                let product = tracked_mul(a.get(i, l), b.get(l, j));
                sum = tracked_add(sum, product);
            }
            result.set(i, j, sum);
        }
    }

    result
}

///
///
pub fn add_bias<T: FloatType>(matrix: &mut DenseMatrix<T>, bias: &[T]) {
    let (rows, cols) = matrix.shape;
    assert_eq!(
        bias.len(),
        cols,
        "The bias length does not match the number of matrix columns: bias[{}] vs matrix[{},{}]",
        bias.len(),
        rows,
        cols
    );

    for row in 0..rows {
        let row_data = matrix.get_row_mut(row);
        for (j, &bias_val) in bias.iter().enumerate() {
            row_data[j] = tracked_add(row_data[j], bias_val);
        }
    }
}

///
pub fn elementwise_apply<T: FloatType, F>(matrix: &DenseMatrix<T>, f: F) -> DenseMatrix<T>
where
    F: Fn(T) -> T,
{
    let data = matrix.data.iter().map(|&x| f(x)).collect();
    DenseMatrix::new(data, matrix.shape)
}

///
#[allow(dead_code)]
pub fn transpose<T: FloatType>(matrix: &DenseMatrix<T>) -> DenseMatrix<T> {
    let (rows, cols) = matrix.shape;
    let mut data = vec![T::zero(); rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            data[j * rows + i] = matrix.get(i, j);
        }
    }

    DenseMatrix::new(data, (cols, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_dense_matrix_multiply() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let a = DenseMatrix::<DefaultFloat>::new(a_data, (2, 2));

        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let b = DenseMatrix::<DefaultFloat>::new(b_data, (2, 2));

        let c = dense_mm(&a, &b);

        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_add_bias() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DenseMatrix::<DefaultFloat>::new(data, (2, 2));
        let bias = vec![0.5, 1.0];

        add_bias(&mut matrix, &bias);

        assert_eq!(matrix.get(0, 0), 1.5); // 1.0 + 0.5
        assert_eq!(matrix.get(0, 1), 3.0); // 2.0 + 1.0
        assert_eq!(matrix.get(1, 0), 3.5); // 3.0 + 0.5
        assert_eq!(matrix.get(1, 1), 5.0); // 4.0 + 1.0
    }
}
