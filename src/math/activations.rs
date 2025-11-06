use crate::math::dense_ops::elementwise_apply;
use crate::types::{DenseMatrix, FloatType};

///
/// - Rust: relu(DenseMatrix<f32>) -> DenseMatrix<f32>
///
pub fn relu<T: FloatType>(input: &DenseMatrix<T>) -> DenseMatrix<T> {
    elementwise_apply(input, |x| FloatType::max(x, T::zero()))
}

///
/// - Rust: log_softmax(DenseMatrix<f32>, 1) -> DenseMatrix<f32>
///
///
pub fn log_softmax<T: FloatType>(input: &DenseMatrix<T>, dim: usize) -> DenseMatrix<T> {
    let (rows, cols) = input.shape;
    let mut result = input.clone();

    match dim {
        1 => {
            for row in 0..rows {
                let row_data = result.get_row_mut(row);

                let max_val = row_data
                    .iter()
                    .fold(<T as FloatType>::neg_infinity(), |acc, &x| {
                        FloatType::max(acc, x)
                    });

                for x in row_data.iter_mut() {
                    *x = *x - max_val;
                }

                let sum_exp: T = row_data
                    .iter()
                    .map(|&x| FloatType::exp(x))
                    .fold(T::zero(), |acc, x| acc + x);
                let log_sum_exp = FloatType::ln(sum_exp);

                for x in row_data.iter_mut() {
                    *x = *x - log_sum_exp;
                }
            }
        }
        0 => {
            for col in 0..cols {
                let mut col_data: Vec<T> = (0..rows).map(|row| result.get(row, col)).collect();

                let max_val = col_data
                    .iter()
                    .fold(<T as FloatType>::neg_infinity(), |acc, &x| {
                        FloatType::max(acc, x)
                    });

                for x in col_data.iter_mut() {
                    *x = *x - max_val;
                }

                let sum_exp: T = col_data
                    .iter()
                    .map(|&x| FloatType::exp(x))
                    .fold(T::zero(), |acc, x| acc + x);
                let log_sum_exp = FloatType::ln(sum_exp);

                for (row, &val) in col_data.iter().enumerate() {
                    result.set(row, col, val - log_sum_exp);
                }
            }
        }
        _ => panic!(
            "Unsupported log_softmax dimension: {}, only 0 or 1 is supported",
            dim
        ),
    }

    result
}

///
#[allow(dead_code)]
pub fn softmax<T: FloatType>(input: &DenseMatrix<T>, dim: usize) -> DenseMatrix<T> {
    let log_probs = log_softmax(input, dim);
    elementwise_apply(&log_probs, |x| FloatType::exp(x))
}

///
#[allow(dead_code)]
pub fn dropout_inference<T: FloatType>(input: &DenseMatrix<T>, _rate: f64) -> DenseMatrix<T> {
    input.clone()
}

///
#[allow(dead_code)]
pub fn validate_log_softmax<T: FloatType + std::fmt::Display>(
    output: &DenseMatrix<T>,
    dim: usize,
    tolerance: T,
) -> bool {
    let (rows, cols) = output.shape;

    match dim {
        1 => {
            for row in 0..rows {
                let row_data = output.get_row(row);
                let prob_sum: T = row_data
                    .iter()
                    .map(|&x| FloatType::exp(x))
                    .fold(T::zero(), |acc, x| acc + x);

                let diff = (prob_sum - T::one()).abs();
                if diff > tolerance {
                    eprintln!(
                        "The probabilities at row {} do not sum to 1: {} (difference: {})",
                        row, prob_sum, diff
                    );
                    return false;
                }
            }
        }
        0 => {
            for col in 0..cols {
                let prob_sum: T = (0..rows)
                    .map(|row| FloatType::exp(output.get(row, col)))
                    .fold(T::zero(), |acc, x| acc + x);

                let diff = (prob_sum - T::one()).abs();
                if diff > tolerance {
                    eprintln!("The sum of the probabilities in column {} is not equal to 1: {} (difference: {})", col, prob_sum, diff);
                    return false;
                }
            }
        }
        _ => return false,
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DefaultFloat;

    #[test]
    fn test_relu() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let input = DenseMatrix::<DefaultFloat>::new(data, (2, 2));

        let output = relu(&input);

        assert_eq!(output.get(0, 0), 0.0); // max(-1, 0) = 0
        assert_eq!(output.get(0, 1), 0.0); // max(0, 0) = 0
        assert_eq!(output.get(1, 0), 1.0); // max(1, 0) = 1
        assert_eq!(output.get(1, 1), 2.0); // max(2, 0) = 2
    }

    #[test]
    fn test_log_softmax() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = DenseMatrix::<DefaultFloat>::new(data, (2, 3));

        let output = log_softmax(&input, 1);

        let tolerance = 1e-6;
        assert!(validate_log_softmax(&output, 1, tolerance));

        assert!(output.get(0, 2) > output.get(0, 1)); // 3 > 2
        assert!(output.get(0, 1) > output.get(0, 0)); // 2 > 1
        assert!(output.get(1, 2) > output.get(1, 1)); // 6 > 5
        assert!(output.get(1, 1) > output.get(1, 0)); // 5 > 4
    }

    #[test]
    fn test_softmax_probabilities() {
        let data = vec![0.0, 1.0, 2.0];
        let input = DenseMatrix::<DefaultFloat>::new(data, (1, 3));

        let probs = softmax(&input, 1);

        let sum: DefaultFloat = probs.get_row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        for &prob in probs.get_row(0) {
            assert!(prob > 0.0);
        }

        assert!(probs.get(0, 2) > probs.get(0, 1));
        assert!(probs.get(0, 1) > probs.get(0, 0));
    }
}
