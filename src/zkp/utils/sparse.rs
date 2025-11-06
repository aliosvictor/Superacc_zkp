#![allow(dead_code)]

use crate::types::{DenseMatrix, FloatType, SparseMatrix};
use sha3::{Digest, Sha3_256};

const SPARSE_HASH_DOMAIN: &[u8] = b"SPARSE_ADJ_HASH_V1";
const DENSE_ROW_HASH_DOMAIN: &[u8] = b"DENSE_ROW_HASH_V1";

fn encode_float<T: FloatType>(value: T) -> Result<[u8; 8], String> {
    value
        .to_f64()
        .map(|float| float.to_le_bytes())
        .ok_or_else(|| "Unable to convert float to f64 for hashing".to_string())
}

pub fn hash_sparse_matrix<T: FloatType>(matrix: &SparseMatrix<T>) -> Result<[u8; 32], String> {
    let csr = matrix.to_csr()?;
    let mut hasher = Sha3_256::new();
    hasher.update(SPARSE_HASH_DOMAIN);
    hasher.update(&(csr.shape.0 as u64).to_le_bytes());
    hasher.update(&(csr.shape.1 as u64).to_le_bytes());
    hasher.update(&(csr.nnz() as u64).to_le_bytes());

    for ptr in &csr.row_ptr {
        hasher.update(&(*ptr as u64).to_le_bytes());
    }
    for col in &csr.col_indices {
        hasher.update(&(*col as u64).to_le_bytes());
    }
    for value in &csr.values {
        hasher.update(&encode_float(*value)?);
    }

    let digest = hasher.finalize();
    Ok(digest.into())
}

pub fn hash_dense_matrix_rows<T: FloatType>(
    matrix: &DenseMatrix<T>,
) -> Result<Vec<[u8; 32]>, String> {
    let (rows, cols) = matrix.shape;
    let mut commitments = Vec::with_capacity(rows);

    for row in 0..rows {
        let mut hasher = Sha3_256::new();
        hasher.update(DENSE_ROW_HASH_DOMAIN);
        hasher.update(&(row as u64).to_le_bytes());
        hasher.update(&(cols as u64).to_le_bytes());

        let start = row * cols;
        let end = start + cols;
        for value in &matrix.data[start..end] {
            hasher.update(&encode_float(*value)?);
        }

        commitments.push(hasher.finalize().into());
    }

    Ok(commitments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DefaultFloat, DenseMatrix, SparseMatrix};

    fn make_sparse() -> SparseMatrix<DefaultFloat> {
        SparseMatrix::new(vec![(0, 1), (1, 0)], vec![1.5, -2.0], (2, 2))
    }

    #[test]
    fn sparse_hash_ignores_entry_order() {
        let hash1 = hash_sparse_matrix(&make_sparse()).expect("hash");
        let permuted = SparseMatrix::new(vec![(1, 0), (0, 1)], vec![-2.0, 1.5], (2, 2));
        let hash2 = hash_sparse_matrix(&permuted).expect("hash");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn dense_row_hashes_match_shape() {
        let matrix = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let hashes = hash_dense_matrix_rows(&matrix).expect("hash rows");
        assert_eq!(hashes.len(), 2);
        assert_ne!(
            hashes[0], hashes[1],
            "Different rows should produce different hashes"
        );
    }
}
