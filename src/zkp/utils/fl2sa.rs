#![allow(clippy::assertions_on_result_states)]
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, VarsAssignment};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, RngCore, SeedableRng};
use std::fmt;

use crate::zkp::constraint_metrics::{compute_r1cs_metrics, R1csShapeMetrics};

use half::f16;
use num_traits::Float;

#[derive(Debug, Clone, Copy)]
pub enum FloatPrecision {
    Half,   // f16
    Single, // f32
    Double, // f64
}

#[derive(Debug, Clone)]
pub struct DoubleShuffleRandomness {
    pub g: Vec<[u8; 32]>,
    pub g_prime: Vec<[u8; 32]>,
    pub a: [u8; 32],
    pub a_prime: [u8; 32],
    pub row_scalars: Vec<[u8; 32]>,
    pub col_scalars: Vec<[u8; 32]>,
    pub seed: Option<[u8; 32]>,
}

impl DoubleShuffleRandomness {
    pub fn new(beta: usize, w: usize, rng: &mut impl RngCore) -> Self {
        Self::with_batch(beta, w, 1, rng)
    }

    pub fn with_batch(beta: usize, w: usize, batch_size: usize, rng: &mut impl RngCore) -> Self {
        let mut g = Vec::with_capacity(w);
        for _ in 0..w {
            let val = rng.gen_range(1..=u64::MAX);
            g.push(Scalar::from(val).to_bytes());
        }

        let row_count = beta * batch_size;

        let mut g_prime = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let val = rng.gen_range(1..=u64::MAX);
            g_prime.push(Scalar::from(val).to_bytes());
        }

        let a = Scalar::from(rng.gen_range(1..=u64::MAX)).to_bytes();
        let a_prime = Scalar::from(rng.gen_range(1..=u64::MAX)).to_bytes();

        let mut row_scalars = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let val = rng.gen_range(1..=u64::MAX);
            row_scalars.push(Scalar::from(val).to_bytes());
        }

        let mut col_scalars = Vec::with_capacity(w);
        for _ in 0..w {
            let val = rng.gen_range(1..=u64::MAX);
            col_scalars.push(Scalar::from(val).to_bytes());
        }

        Self {
            g,
            g_prime,
            a,
            a_prime,
            row_scalars,
            col_scalars,
            seed: None,
        }
    }

    pub fn from_seed(beta: usize, w: usize, seed: [u8; 32]) -> Self {
        let mut rng = StdRng::from_seed(seed);
        let mut randomness = Self::with_batch(beta, w, 1, &mut rng);
        randomness.seed = Some(seed);
        randomness
    }

    pub fn with_seed_and_batch(beta: usize, w: usize, batch_size: usize, seed: [u8; 32]) -> Self {
        let mut rng = StdRng::from_seed(seed);
        let mut randomness = Self::with_batch(beta, w, batch_size, &mut rng);
        randomness.seed = Some(seed);
        randomness
    }
}

#[derive(Debug, Clone)]
pub struct DoubleShuffleWitness {
    pub beta: usize,
    pub w: usize,
    pub batch_size: usize,
    pub randomness: DoubleShuffleRandomness,
    pub p_l_indices: Vec<Vec<usize>>,        // [batch][p_l_bits]
    pub r_bit_indices: Vec<Vec<Vec<usize>>>, // [batch][beta-1][w]
    pub p_l_values: Vec<Vec<u8>>,            // [batch][p_l_bits]
    pub r_bit_values: Vec<Vec<Vec<u8>>>,     // [batch][beta-1][w]
}

impl DoubleShuffleWitness {
    pub fn single(beta: usize, w: usize, randomness: DoubleShuffleRandomness) -> Self {
        Self {
            beta,
            w,
            batch_size: 1,
            randomness,
            p_l_indices: Vec::new(),
            r_bit_indices: Vec::new(),
            p_l_values: Vec::new(),
            r_bit_values: Vec::new(),
        }
    }

    pub fn with_batch(
        batch_size: usize,
        beta: usize,
        w: usize,
        randomness: DoubleShuffleRandomness,
    ) -> Self {
        Self {
            beta,
            w,
            batch_size,
            randomness,
            p_l_indices: Vec::new(),
            r_bit_indices: Vec::new(),
            p_l_values: Vec::new(),
            r_bit_values: Vec::new(),
        }
    }

    pub fn with_indices(
        mut self,
        p_l_indices: Vec<Vec<usize>>,
        r_bit_indices: Vec<Vec<Vec<usize>>>,
    ) -> Self {
        self.p_l_indices = p_l_indices;
        self.r_bit_indices = r_bit_indices;
        self
    }

    pub fn with_trace(mut self, p_l_values: Vec<Vec<u8>>, r_bit_values: Vec<Vec<Vec<u8>>>) -> Self {
        self.p_l_values = p_l_values;
        self.r_bit_values = r_bit_values;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Fl2saDoubleShuffleBundle {
    pub core: DoubleShuffleWitness,
    pub ah: AhDoubleShuffleWitness,
}

// =============================================================================
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AhDoubleShuffleRandomness {
    pub row_scalars: Vec<Scalar>,
    pub col_scalars: Vec<Scalar>,
    pub row_weights: Vec<Scalar>,
    pub col_weights: Vec<Scalar>,
    pub alpha: Scalar,
    pub alpha_prime: Scalar,
    pub seed: [u8; 32],
}

fn sample_nonzero_scalar(rng: &mut impl RngCore) -> Scalar {
    loop {
        let candidate = Scalar::from(rng.gen_range(1..=u64::MAX));
        if candidate != Scalar::ZERO {
            return candidate;
        }
    }
}

pub fn sample_ah_double_shuffle_randomness(
    batch_size: usize,
    cols: usize,
    rng: &mut impl RngCore,
) -> AhDoubleShuffleRandomness {
    let mut row_scalars = Vec::with_capacity(batch_size);
    let mut row_weights = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        row_scalars.push(sample_nonzero_scalar(rng));
        row_weights.push(sample_nonzero_scalar(rng));
    }

    let mut col_scalars = Vec::with_capacity(cols);
    let mut col_weights = Vec::with_capacity(cols);
    for _ in 0..cols {
        col_scalars.push(sample_nonzero_scalar(rng));
        col_weights.push(sample_nonzero_scalar(rng));
    }

    let alpha = sample_nonzero_scalar(rng);
    let alpha_prime = sample_nonzero_scalar(rng);
    let mut seed_bytes = [0u8; 32];
    rng.fill_bytes(&mut seed_bytes);

    AhDoubleShuffleRandomness {
        row_scalars,
        col_scalars,
        row_weights,
        col_weights,
        alpha,
        alpha_prime,
        seed: seed_bytes,
    }
}

#[derive(Debug, Clone)]
pub struct AhDoubleShuffleWitness {
    pub batch_size: usize,
    pub cols: usize,
    pub randomness: AhDoubleShuffleRandomness,
    pub ah_indices: Vec<Vec<usize>>, // [batch][cols]
    pub ah_values: Vec<Vec<u8>>,     // [batch][cols]
}

impl AhDoubleShuffleWitness {
    pub fn empty(batch_size: usize, cols: usize, randomness: AhDoubleShuffleRandomness) -> Self {
        Self {
            batch_size,
            cols,
            randomness,
            ah_indices: vec![Vec::new(); batch_size],
            ah_values: vec![Vec::new(); batch_size],
        }
    }

    pub fn with_indices(mut self, indices: Vec<Vec<usize>>) -> Self {
        self.ah_indices = indices;
        self
    }

    pub fn with_values(mut self, values: Vec<Vec<u8>>) -> Self {
        self.ah_values = values;
        self
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AhDoubleShuffleVarLayout {
    start_idx: usize,
    k: std::ops::Range<usize>,
    k_prime: std::ops::Range<usize>,
    k_diff_left: std::ops::Range<usize>,
    k_prime_diff_right: std::ops::Range<usize>,
    k_prod_left: std::ops::Range<usize>,
    k_prime_prod_right: std::ops::Range<usize>,
    s: std::ops::Range<usize>,
    s_prime: std::ops::Range<usize>,
    s_diff_left: std::ops::Range<usize>,
    s_prime_diff_right: std::ops::Range<usize>,
    s_prod_left: std::ops::Range<usize>,
    s_prime_prod_right: std::ops::Range<usize>,
    next_idx: usize,
}

#[allow(dead_code)]
impl AhDoubleShuffleVarLayout {
    fn new(start_idx: usize, rows: usize, cols: usize) -> Self {
        let k = start_idx..(start_idx + rows);
        let k_prime = k.end..(k.end + rows);
        let k_diff_left = k_prime.end..(k_prime.end + rows);
        let k_prime_diff_right = k_diff_left.end..(k_diff_left.end + rows);
        let k_prod_left = k_prime_diff_right.end..(k_prime_diff_right.end + rows);
        let k_prime_prod_right = k_prod_left.end..(k_prod_left.end + rows);

        let s = k_prime_prod_right.end..(k_prime_prod_right.end + cols);
        let s_prime = s.end..(s.end + cols);
        let s_diff_left = s_prime.end..(s_prime.end + cols);
        let s_prime_diff_right = s_diff_left.end..(s_diff_left.end + cols);
        let s_prod_left = s_prime_diff_right.end..(s_prime_diff_right.end + cols);
        let s_prime_prod_right = s_prod_left.end..(s_prod_left.end + cols);

        let next_idx = s_prime_prod_right.end;

        Self {
            start_idx,
            k,
            k_prime,
            k_diff_left,
            k_prime_diff_right,
            k_prod_left,
            k_prime_prod_right,
            s,
            s_prime,
            s_diff_left,
            s_prime_diff_right,
            s_prod_left,
            s_prime_prod_right,
            next_idx,
        }
    }

    fn start_idx(&self) -> usize {
        self.start_idx
    }

    pub fn next_var_idx(&self) -> usize {
        self.next_idx
    }

    fn rows(&self) -> usize {
        self.k.len()
    }

    fn cols(&self) -> usize {
        self.s.len()
    }

    fn k_index(&self, row: usize) -> usize {
        self.k.start + row
    }

    fn k_prime_index(&self, row: usize) -> usize {
        self.k_prime.start + row
    }

    fn k_diff_left_index(&self, row: usize) -> usize {
        self.k_diff_left.start + row
    }

    fn k_prime_diff_right_index(&self, row: usize) -> usize {
        self.k_prime_diff_right.start + row
    }

    fn k_prod_left_index(&self, row: usize) -> usize {
        self.k_prod_left.start + row
    }

    fn k_prime_prod_right_index(&self, row: usize) -> usize {
        self.k_prime_prod_right.start + row
    }

    fn s_index(&self, col: usize) -> usize {
        self.s.start + col
    }

    fn s_prime_index(&self, col: usize) -> usize {
        self.s_prime.start + col
    }

    fn s_diff_left_index(&self, col: usize) -> usize {
        self.s_diff_left.start + col
    }

    fn s_prime_diff_right_index(&self, col: usize) -> usize {
        self.s_prime_diff_right.start + col
    }

    fn s_prod_left_index(&self, col: usize) -> usize {
        self.s_prod_left.start + col
    }

    fn s_prime_prod_right_index(&self, col: usize) -> usize {
        self.s_prime_prod_right.start + col
    }
}

#[derive(Debug, Clone)]
pub struct AhDoubleShuffleContext {
    batch: usize,
    cols: usize,
    layout: AhDoubleShuffleVarLayout,
    randomness: AhDoubleShuffleRandomness,
    matrix: Vec<Vec<usize>>,
    matrix_row_shifted: Vec<Vec<usize>>,
    matrix_col_shifted: Vec<Vec<usize>>,
    row_scalar_for_shift: Vec<usize>,
}

impl AhDoubleShuffleContext {
    pub(crate) fn new(
        batch: usize,
        cols: usize,
        start_idx: usize,
        randomness: AhDoubleShuffleRandomness,
        indices: Vec<Vec<usize>>,
        zero_var_idx: usize,
    ) -> Result<Self, String> {
        if indices.len() != batch {
            return Err(format!(
                "ah Double-Shuffling indices row count mismatch: expected {}, actual {}",
                batch,
                indices.len()
            ));
        }
        for (row, cols_vec) in indices.iter().enumerate() {
            if cols_vec.len() != cols {
                return Err(format!(
                    "ah Double-Shuffling No. {} row and column number mismatch: expected {}, actual {}",
                    row,
                    cols,
                    cols_vec.len()
                ));
            }
        }

        let layout = AhDoubleShuffleVarLayout::new(start_idx, batch, cols);

        let mut matrix = vec![vec![zero_var_idx; cols]; batch];
        for (row, row_indices) in indices.iter().enumerate() {
            for (col, idx) in row_indices.iter().enumerate() {
                matrix[row][col] = *idx;
            }
        }

        let mut matrix_row_shifted = vec![vec![zero_var_idx; cols]; batch];
        let mut row_scalar_for_shift = vec![0usize; batch];
        if batch > 0 {
            for row in 0..batch {
                let prev_row = if row == 0 { batch - 1 } else { row - 1 };
                matrix_row_shifted[row] = matrix[prev_row].clone();
                row_scalar_for_shift[row] = prev_row;
            }
        }

        let mut matrix_col_shifted = vec![vec![zero_var_idx; cols]; batch];
        if cols > 0 {
            for row in 0..batch {
                for col in 0..cols {
                    let prev_col = if col == 0 { cols - 1 } else { col - 1 };
                    matrix_col_shifted[row][col] = matrix_row_shifted[row]
                        .get(prev_col)
                        .copied()
                        .unwrap_or(zero_var_idx);
                }
            }
        }

        Ok(Self {
            batch,
            cols,
            layout,
            randomness,
            matrix,
            matrix_row_shifted,
            matrix_col_shifted,
            row_scalar_for_shift,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn layout(&self) -> &AhDoubleShuffleVarLayout {
        &self.layout
    }

    pub(crate) fn append_constraints(
        &self,
        constraint_start: usize,
        a_entries: &mut Vec<(usize, usize, [u8; 32])>,
        b_entries: &mut Vec<(usize, usize, [u8; 32])>,
        c_entries: &mut Vec<(usize, usize, [u8; 32])>,
        const_one_idx: usize,
    ) -> usize {
        let mut cursor = constraint_start;
        let one = Scalar::ONE.to_bytes();
        let minus_alpha = (-self.randomness.alpha).to_bytes();
        let minus_alpha_prime = (-self.randomness.alpha_prime).to_bytes();

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            let rho = self.randomness.row_scalars[row];
            for col in 0..self.cols {
                let coeff = rho * self.randomness.col_weights[col];
                a_entries.push((constraint_id, self.matrix[row][col], coeff.to_bytes()));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            let source_row = self.row_scalar_for_shift[row];
            let rho = self.randomness.row_scalars[source_row];
            for col in 0..self.cols {
                let coeff = rho * self.randomness.col_weights[col];
                a_entries.push((
                    constraint_id,
                    self.matrix_row_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prime_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_index(row), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_diff_left_index(row), one));
        }

        for row in 0..self.batch {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_prime_index(row), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.k_prime_diff_right_index(row),
                one,
            ));
        }

        if self.batch > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_diff_left_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prod_left_index(0), one));

            for row in 1..self.batch {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.k_prod_left_index(row - 1), one));
                b_entries.push((cid, self.layout.k_diff_left_index(row), one));
                c_entries.push((cid, self.layout.k_prod_left_index(row), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.k_prod_left_index(self.batch - 1),
                one,
            ));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.k_prime_prod_right_index(self.batch - 1),
                one,
            ));
        }

        if self.batch > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.k_prime_diff_right_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.k_prime_prod_right_index(0), one));

            for row in 1..self.batch {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.k_prime_prod_right_index(row - 1), one));
                b_entries.push((cid, self.layout.k_prime_diff_right_index(row), one));
                c_entries.push((cid, self.layout.k_prime_prod_right_index(row), one));
            }
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            let sigma = self.randomness.col_scalars[col];
            for row in 0..self.batch {
                let coeff = sigma * self.randomness.row_weights[row];
                a_entries.push((
                    constraint_id,
                    self.matrix_row_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            let prev_col = if col == 0 { self.cols - 1 } else { col - 1 };
            let sigma = self.randomness.col_scalars[prev_col];
            for row in 0..self.batch {
                let coeff = sigma * self.randomness.row_weights[row];
                a_entries.push((
                    constraint_id,
                    self.matrix_col_shifted[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prime_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_index(col), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha_prime));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_diff_left_index(col), one));
        }

        for col in 0..self.cols {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_index(col), one));
            a_entries.push((constraint_id, const_one_idx, minus_alpha_prime));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_diff_right_index(col),
                one,
            ));
        }

        if self.cols > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_diff_left_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prod_left_index(0), one));

            for col in 1..self.cols {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prod_left_index(col - 1), one));
                b_entries.push((cid, self.layout.s_diff_left_index(col), one));
                c_entries.push((cid, self.layout.s_prod_left_index(col), one));
            }

            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.s_prod_left_index(self.cols - 1),
                one,
            ));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_prod_right_index(self.cols - 1),
                one,
            ));
        }

        if self.cols > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_diff_right_index(0), one));
            b_entries.push((constraint_id, const_one_idx, one));
            c_entries.push((constraint_id, self.layout.s_prime_prod_right_index(0), one));

            for col in 1..self.cols {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prime_prod_right_index(col - 1), one));
                b_entries.push((cid, self.layout.s_prime_diff_right_index(col), one));
                c_entries.push((cid, self.layout.s_prime_prod_right_index(col), one));
            }
        }

        cursor
    }

    pub(crate) fn populate_witness(
        &self,
        vars: &mut [[u8; 32]],
    ) -> Result<AhDoubleShuffleWitness, String> {
        let mut k_values = vec![Scalar::ZERO; self.batch];
        let mut k_prime_values = vec![Scalar::ZERO; self.batch];

        for row in 0..self.batch {
            let mut acc = Scalar::ZERO;
            let rho = self.randomness.row_scalars[row];
            for col in 0..self.cols {
                let idx = self.matrix[row][col];
                let val = Scalar::from_bytes_mod_order(vars[idx]);
                acc += self.randomness.col_weights[col] * val;
            }
            acc *= rho;
            vars[self.layout.k_index(row)] = acc.to_bytes();
            k_values[row] = acc;
        }

        for row in 0..self.batch {
            let source_row = self.row_scalar_for_shift[row];
            let mut acc = Scalar::ZERO;
            let rho = self.randomness.row_scalars[source_row];
            for col in 0..self.cols {
                let idx = self.matrix_row_shifted[row][col];
                let val = Scalar::from_bytes_mod_order(vars[idx]);
                acc += self.randomness.col_weights[col] * val;
            }
            acc *= rho;
            vars[self.layout.k_prime_index(row)] = acc.to_bytes();
            k_prime_values[row] = acc;
        }

        for row in 0..self.batch {
            let diff = k_values[row] - self.randomness.alpha;
            vars[self.layout.k_diff_left_index(row)] = diff.to_bytes();
            let diff_prime = k_prime_values[row] - self.randomness.alpha;
            vars[self.layout.k_prime_diff_right_index(row)] = diff_prime.to_bytes();
        }

        if self.batch > 0 {
            vars[self.layout.k_prod_left_index(0)] = vars[self.layout.k_diff_left_index(0)];
            vars[self.layout.k_prime_prod_right_index(0)] =
                vars[self.layout.k_prime_diff_right_index(0)];

            for row in 1..self.batch {
                let prev =
                    Scalar::from_bytes_mod_order(vars[self.layout.k_prod_left_index(row - 1)]);
                let current =
                    Scalar::from_bytes_mod_order(vars[self.layout.k_diff_left_index(row)]);
                vars[self.layout.k_prod_left_index(row)] = (prev * current).to_bytes();

                let prev_prime = Scalar::from_bytes_mod_order(
                    vars[self.layout.k_prime_prod_right_index(row - 1)],
                );
                let current_prime =
                    Scalar::from_bytes_mod_order(vars[self.layout.k_prime_diff_right_index(row)]);
                vars[self.layout.k_prime_prod_right_index(row)] =
                    (prev_prime * current_prime).to_bytes();
            }
        }

        for col in 0..self.cols {
            let sigma = self.randomness.col_scalars[col];
            let mut acc = Scalar::ZERO;
            for row in 0..self.batch {
                let idx = self.matrix_row_shifted[row][col];
                let val = Scalar::from_bytes_mod_order(vars[idx]);
                acc += self.randomness.row_weights[row] * val;
            }
            acc *= sigma;
            vars[self.layout.s_index(col)] = acc.to_bytes();
        }

        for col in 0..self.cols {
            let prev_col = if col == 0 { self.cols - 1 } else { col - 1 };
            let sigma = self.randomness.col_scalars[prev_col];
            let mut acc = Scalar::ZERO;
            for row in 0..self.batch {
                let idx = self.matrix_col_shifted[row][col];
                let val = Scalar::from_bytes_mod_order(vars[idx]);
                acc += self.randomness.row_weights[row] * val;
            }
            acc *= sigma;
            vars[self.layout.s_prime_index(col)] = acc.to_bytes();
        }

        for col in 0..self.cols {
            let diff = Scalar::from_bytes_mod_order(vars[self.layout.s_index(col)])
                - self.randomness.alpha_prime;
            vars[self.layout.s_diff_left_index(col)] = diff.to_bytes();
            let diff_prime = Scalar::from_bytes_mod_order(vars[self.layout.s_prime_index(col)])
                - self.randomness.alpha_prime;
            vars[self.layout.s_prime_diff_right_index(col)] = diff_prime.to_bytes();
        }

        if self.cols > 0 {
            vars[self.layout.s_prod_left_index(0)] = vars[self.layout.s_diff_left_index(0)];
            vars[self.layout.s_prime_prod_right_index(0)] =
                vars[self.layout.s_prime_diff_right_index(0)];

            for col in 1..self.cols {
                let prev =
                    Scalar::from_bytes_mod_order(vars[self.layout.s_prod_left_index(col - 1)]);
                let current =
                    Scalar::from_bytes_mod_order(vars[self.layout.s_diff_left_index(col)]);
                vars[self.layout.s_prod_left_index(col)] = (prev * current).to_bytes();

                let prev_prime = Scalar::from_bytes_mod_order(
                    vars[self.layout.s_prime_prod_right_index(col - 1)],
                );
                let current_prime =
                    Scalar::from_bytes_mod_order(vars[self.layout.s_prime_diff_right_index(col)]);
                vars[self.layout.s_prime_prod_right_index(col)] =
                    (prev_prime * current_prime).to_bytes();
            }
        }

        let mut values = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            let mut row_values = Vec::with_capacity(self.cols);
            for col in 0..self.cols {
                let idx = self.matrix[row][col];
                let val = Scalar::from_bytes_mod_order(vars[idx]);
                if val == Scalar::ZERO {
                    row_values.push(0u8);
                } else if val == Scalar::ONE {
                    row_values.push(1u8);
                } else {
                    return Err(format!("ah Double-Shuffling index {} non-boolean", idx));
                }
            }
            values.push(row_values);
        }

        Ok(
            AhDoubleShuffleWitness::empty(self.batch, self.cols, self.randomness.clone())
                .with_indices(self.matrix.clone())
                .with_values(values),
        )
    }
}

#[derive(Debug, Clone)]
pub struct DoubleShuffleBatchIndices {
    pub p_l_bits: Vec<Vec<Option<usize>>>,    // [batch][w]
    pub r_bits: Vec<Vec<Vec<Option<usize>>>>, // [batch][beta-1][w]
}

impl DoubleShuffleBatchIndices {
    pub fn new(batch_size: usize, beta: usize, w: usize) -> Self {
        let p_l_bits = vec![vec![None; w]; batch_size];
        let r_bits = vec![vec![vec![None; w]; beta - 1]; batch_size];

        Self { p_l_bits, r_bits }
    }

    pub fn single(
        beta: usize,
        w: usize,
        lgw: usize,
        p_l_bits_base_idx: usize,
        r_bit_base_idx: usize,
    ) -> Self {
        let mut indices = Self::new(1, beta, w);

        for j in 0..w {
            if j < lgw {
                indices.p_l_bits[0][j] = Some(p_l_bits_base_idx + j);
            }
        }

        for row in 0..(beta - 1) {
            for j in 0..w {
                indices.r_bits[0][row][j] = Some(r_bit_base_idx + row * w + j);
            }
        }

        indices
    }
}

#[derive(Debug, Clone)]
struct DoubleShuffleVarLayout {
    _beta: usize,
    _w: usize,
    _batch_size: usize,
    row_count: usize,
    k: std::ops::Range<usize>,
    k_prime: std::ops::Range<usize>,
    k_diff_left: std::ops::Range<usize>,
    k_prime_diff_right: std::ops::Range<usize>,
    k_prod_left: std::ops::Range<usize>,
    k_prime_prod_right: std::ops::Range<usize>,
    s: std::ops::Range<usize>,
    s_prime: std::ops::Range<usize>,
    s_diff_left: std::ops::Range<usize>,
    s_prime_diff_right: std::ops::Range<usize>,
    s_prod_left: std::ops::Range<usize>,
    s_prime_prod_right: std::ops::Range<usize>,
    _next_var_idx: usize,
}

impl DoubleShuffleVarLayout {
    fn new(start_idx: usize, beta: usize, w: usize, batch_size: usize) -> Self {
        let row_count = beta * batch_size;

        let k = start_idx..(start_idx + row_count);
        let k_prime = k.end..(k.end + row_count);
        let k_diff_left = k_prime.end..(k_prime.end + row_count);
        let k_prime_diff_right = k_diff_left.end..(k_diff_left.end + row_count);
        let k_prod_left = k_prime_diff_right.end..(k_prime_diff_right.end + row_count);
        let k_prime_prod_right = k_prod_left.end..(k_prod_left.end + row_count);

        let s_start = k_prime_prod_right.end;
        let s = s_start..(s_start + w);
        let s_prime = s.end..(s.end + w);
        let s_diff_left = s_prime.end..(s_prime.end + w);
        let s_prime_diff_right = s_diff_left.end..(s_diff_left.end + w);
        let s_prod_left = s_prime_diff_right.end..(s_prime_diff_right.end + w);
        let s_prime_prod_right = s_prod_left.end..(s_prod_left.end + w);

        let next_var_idx = s_prime_prod_right.end;

        Self {
            _beta: beta,
            _w: w,
            _batch_size: batch_size,
            row_count,
            k,
            k_prime,
            k_diff_left,
            k_prime_diff_right,
            k_prod_left,
            k_prime_prod_right,
            s,
            s_prime,
            s_diff_left,
            s_prime_diff_right,
            s_prod_left,
            s_prime_prod_right,
            _next_var_idx: next_var_idx,
        }
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn next_var_idx(&self) -> usize {
        self._next_var_idx
    }

    fn k_index(&self, row: usize) -> usize {
        self.k.start + row
    }

    fn k_prime_index(&self, row: usize) -> usize {
        self.k_prime.start + row
    }

    fn k_diff_left_index(&self, row: usize) -> usize {
        self.k_diff_left.start + row
    }

    fn k_prime_diff_right_index(&self, row: usize) -> usize {
        self.k_prime_diff_right.start + row
    }

    fn k_prod_left_index(&self, row: usize) -> usize {
        self.k_prod_left.start + row
    }

    fn k_prime_prod_right_index(&self, row: usize) -> usize {
        self.k_prime_prod_right.start + row
    }

    fn s_index(&self, col: usize) -> usize {
        self.s.start + col
    }

    fn s_prime_index(&self, col: usize) -> usize {
        self.s_prime.start + col
    }

    fn s_diff_left_index(&self, col: usize) -> usize {
        self.s_diff_left.start + col
    }

    fn s_prime_diff_right_index(&self, col: usize) -> usize {
        self.s_prime_diff_right.start + col
    }

    fn s_prod_left_index(&self, col: usize) -> usize {
        self.s_prod_left.start + col
    }

    fn s_prime_prod_right_index(&self, col: usize) -> usize {
        self.s_prime_prod_right.start + col
    }
}

#[derive(Debug, Clone)]
struct DoubleShuffleBatchContext {
    beta: usize,
    w: usize,
    _lgw: usize,
    batch_size: usize,
    layout: DoubleShuffleVarLayout,
    randomness: DoubleShuffleRandomness,
    indices: DoubleShuffleBatchIndices,
    var_matrix: Vec<Vec<usize>>,
    var_matrix_shuffled_row: Vec<Vec<usize>>,
    var_matrix_shuffled_col: Vec<Vec<usize>>,
}

impl DoubleShuffleBatchContext {
    fn new(
        beta: usize,
        w: usize,
        lgw: usize,
        batch_size: usize,
        start_var_idx: usize,
        randomness: DoubleShuffleRandomness,
        indices: DoubleShuffleBatchIndices,
        zero_var_idx: usize,
    ) -> Self {
        let layout = DoubleShuffleVarLayout::new(start_var_idx, beta, w, batch_size);
        let row_count = layout.row_count();

        let mut var_matrix = vec![vec![zero_var_idx; w]; row_count];

        for batch in 0..batch_size {
            let row_offset = batch * beta;

            for col in 0..w {
                if let Some(idx) = indices.p_l_bits[batch][col] {
                    var_matrix[row_offset][col] = idx;
                }
            }

            for local_row in 0..(beta - 1) {
                let target_row = row_offset + local_row + 1;
                for col in 0..w {
                    if let Some(idx) = indices.r_bits[batch][local_row][col] {
                        var_matrix[target_row][col] = idx;
                    }
                }
            }
        }

        let mut var_matrix_shuffled_row = vec![vec![zero_var_idx; w]; row_count];
        for batch in 0..batch_size {
            let row_offset = batch * beta;
            for local_row in 0..beta {
                let target_row = row_offset + local_row;
                let source_row = if local_row == 0 {
                    row_offset + beta - 1
                } else {
                    target_row - 1
                };
                var_matrix_shuffled_row[target_row] = var_matrix[source_row].clone();
            }
        }

        let mut var_matrix_shuffled_col = vec![vec![zero_var_idx; w]; row_count];
        for row in 0..row_count {
            for col in 0..w {
                let source_col = if col == 0 { w - 1 } else { col - 1 };
                var_matrix_shuffled_col[row][col] = var_matrix_shuffled_row[row][source_col];
            }
        }

        Self {
            beta,
            w,
            _lgw: lgw,
            batch_size,
            layout,
            randomness,
            indices,
            var_matrix,
            var_matrix_shuffled_row,
            var_matrix_shuffled_col,
        }
    }

    fn extract_indices(&self) -> (Vec<Vec<usize>>, Vec<Vec<Vec<usize>>>) {
        let mut p_l_indices = Vec::with_capacity(self.batch_size);
        let mut r_bit_indices = Vec::with_capacity(self.batch_size);

        for batch in 0..self.batch_size {
            let mut p_indices = Vec::new();
            for col in 0..self.w {
                if let Some(idx) = self.indices.p_l_bits[batch][col] {
                    p_indices.push(idx);
                }
            }
            p_l_indices.push(p_indices);

            let mut r_rows = Vec::with_capacity(self.beta.saturating_sub(1));
            for local_row in 0..self.beta.saturating_sub(1) {
                let mut row_indices = Vec::new();
                for col in 0..self.w {
                    if let Some(idx) = self.indices.r_bits[batch][local_row][col] {
                        row_indices.push(idx);
                    }
                }
                r_rows.push(row_indices);
            }
            r_bit_indices.push(r_rows);
        }

        (p_l_indices, r_bit_indices)
    }

    fn append_constraints(
        &self,
        constraint_start: usize,
        a_entries: &mut Vec<(usize, usize, [u8; 32])>,
        b_entries: &mut Vec<(usize, usize, [u8; 32])>,
        c_entries: &mut Vec<(usize, usize, [u8; 32])>,
        num_vars: usize,
    ) -> usize {
        let one = Scalar::ONE.to_bytes();

        let g_scalars: Vec<Scalar> = self
            .randomness
            .g
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let g_prime_scalars: Vec<Scalar> = self
            .randomness
            .g_prime
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let row_scalars: Vec<Scalar> = self
            .randomness
            .row_scalars
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let col_scalars: Vec<Scalar> = self
            .randomness
            .col_scalars
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();

        let a_scalar = Scalar::from_bytes_mod_order(self.randomness.a);
        let minus_a = (-a_scalar).to_bytes();
        let a_prime_scalar = Scalar::from_bytes_mod_order(self.randomness.a_prime);
        let minus_a_prime = (-a_prime_scalar).to_bytes();

        let mut cursor = constraint_start;

        for row in 0..self.layout.row_count() {
            let batch_idx = row / self.beta;
            let rho = row_scalars[batch_idx];
            let constraint_id = cursor;
            cursor += 1;

            for col in 0..self.w {
                let coeff = rho * g_scalars[col];
                a_entries.push((constraint_id, self.var_matrix[row][col], coeff.to_bytes()));
            }
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.k_index(row), one));
        }

        for row in 0..self.layout.row_count() {
            let batch_idx = row / self.beta;
            let rho = row_scalars[batch_idx];
            let constraint_id = cursor;
            cursor += 1;

            for col in 0..self.w {
                let coeff = rho * g_scalars[col];
                a_entries.push((
                    constraint_id,
                    self.var_matrix_shuffled_row[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.k_prime_index(row), one));
        }

        for row in 0..self.layout.row_count() {
            let constraint_id = cursor;
            cursor += 1;

            a_entries.push((constraint_id, self.layout.k_index(row), one));
            a_entries.push((constraint_id, num_vars, minus_a));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.k_diff_left_index(row), one));
        }

        for row in 0..self.layout.row_count() {
            let constraint_id = cursor;
            cursor += 1;

            a_entries.push((constraint_id, self.layout.k_prime_index(row), one));
            a_entries.push((constraint_id, num_vars, minus_a));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((
                constraint_id,
                self.layout.k_prime_diff_right_index(row),
                one,
            ));
        }

        let left_prod_start = cursor;
        if self.layout.row_count() > 0 {
            a_entries.push((left_prod_start, self.layout.k_diff_left_index(0), one));
            b_entries.push((left_prod_start, num_vars, one));
            c_entries.push((left_prod_start, self.layout.k_prod_left_index(0), one));
            cursor += 1;

            for row in 1..self.layout.row_count() {
                let constraint_id = cursor;
                cursor += 1;
                a_entries.push((constraint_id, self.layout.k_prod_left_index(row - 1), one));
                b_entries.push((constraint_id, self.layout.k_diff_left_index(row), one));
                c_entries.push((constraint_id, self.layout.k_prod_left_index(row), one));
            }
        }

        let right_prod_start = cursor;
        if self.layout.row_count() > 0 {
            a_entries.push((
                right_prod_start,
                self.layout.k_prime_diff_right_index(0),
                one,
            ));
            b_entries.push((right_prod_start, num_vars, one));
            c_entries.push((
                right_prod_start,
                self.layout.k_prime_prod_right_index(0),
                one,
            ));
            cursor += 1;

            for row in 1..self.layout.row_count() {
                let constraint_id = cursor;
                cursor += 1;
                a_entries.push((
                    constraint_id,
                    self.layout.k_prime_prod_right_index(row - 1),
                    one,
                ));
                b_entries.push((
                    constraint_id,
                    self.layout.k_prime_diff_right_index(row),
                    one,
                ));
                c_entries.push((
                    constraint_id,
                    self.layout.k_prime_prod_right_index(row),
                    one,
                ));
            }
        }

        if self.layout.row_count() > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.k_prod_left_index(self.layout.row_count() - 1),
                one,
            ));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((
                constraint_id,
                self.layout
                    .k_prime_prod_right_index(self.layout.row_count() - 1),
                one,
            ));
        }

        for col in 0..self.w {
            let constraint_id = cursor;
            cursor += 1;
            for row in 0..self.layout.row_count() {
                let sigma = col_scalars[col];
                let coeff = sigma * g_prime_scalars[row];
                a_entries.push((
                    constraint_id,
                    self.var_matrix_shuffled_row[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.s_index(col), one));
        }

        for col in 0..self.w {
            let constraint_id = cursor;
            cursor += 1;
            for row in 0..self.layout.row_count() {
                let source_col = if col == 0 { self.w - 1 } else { col - 1 };
                let sigma = col_scalars[source_col];
                let coeff = sigma * g_prime_scalars[row];
                a_entries.push((
                    constraint_id,
                    self.var_matrix_shuffled_col[row][col],
                    coeff.to_bytes(),
                ));
            }
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.s_prime_index(col), one));
        }

        for col in 0..self.w {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_index(col), one));
            a_entries.push((constraint_id, num_vars, minus_a_prime));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.s_diff_left_index(col), one));
        }

        for col in 0..self.w {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_index(col), one));
            a_entries.push((constraint_id, num_vars, minus_a_prime));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_diff_right_index(col),
                one,
            ));
        }

        if self.w > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_diff_left_index(0), one));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.s_prod_left_index(0), one));

            for col in 1..self.w {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prod_left_index(col - 1), one));
                b_entries.push((cid, self.layout.s_diff_left_index(col), one));
                c_entries.push((cid, self.layout.s_prod_left_index(col), one));
            }
        }

        if self.w > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((constraint_id, self.layout.s_prime_diff_right_index(0), one));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((constraint_id, self.layout.s_prime_prod_right_index(0), one));

            for col in 1..self.w {
                let cid = cursor;
                cursor += 1;
                a_entries.push((cid, self.layout.s_prime_prod_right_index(col - 1), one));
                b_entries.push((cid, self.layout.s_prime_diff_right_index(col), one));
                c_entries.push((cid, self.layout.s_prime_prod_right_index(col), one));
            }
        }

        if self.w > 0 {
            let constraint_id = cursor;
            cursor += 1;
            a_entries.push((
                constraint_id,
                self.layout.s_prod_left_index(self.w - 1),
                one,
            ));
            b_entries.push((constraint_id, num_vars, one));
            c_entries.push((
                constraint_id,
                self.layout.s_prime_prod_right_index(self.w - 1),
                one,
            ));
        }

        cursor
    }

    fn populate_witness(
        &self,
        vars: &mut Vec<[u8; 32]>,
        max_var_idx: usize,
    ) -> Result<DoubleShuffleWitness, String> {
        let g_scalars: Vec<Scalar> = self
            .randomness
            .g
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let g_prime_scalars: Vec<Scalar> = self
            .randomness
            .g_prime
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let row_scalars: Vec<Scalar> = self
            .randomness
            .row_scalars
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();
        let col_scalars: Vec<Scalar> = self
            .randomness
            .col_scalars
            .iter()
            .map(|bytes| Scalar::from_bytes_mod_order(*bytes))
            .collect();

        let a_scalar = Scalar::from_bytes_mod_order(self.randomness.a);
        let a_prime_scalar = Scalar::from_bytes_mod_order(self.randomness.a_prime);

        let mut k_values = Vec::with_capacity(self.layout.row_count());
        for row in 0..self.layout.row_count() {
            let batch_idx = row / self.beta;
            let rho = row_scalars[batch_idx];
            let mut sum = Scalar::ZERO;
            for col in 0..self.w {
                let var_idx = self.var_matrix[row][col];
                if var_idx < max_var_idx {
                    let value_scalar = Scalar::from_bytes_mod_order(vars[var_idx]);
                    sum += g_scalars[col] * value_scalar;
                }
            }
            sum *= rho;
            vars[self.layout.k_index(row)] = sum.to_bytes();
            k_values.push(sum);
        }

        let mut k_prime_values = Vec::with_capacity(self.layout.row_count());
        for row in 0..self.layout.row_count() {
            let batch_idx = row / self.beta;
            let rho = row_scalars[batch_idx];
            let mut sum = Scalar::ZERO;
            for col in 0..self.w {
                let var_idx = self.var_matrix_shuffled_row[row][col];
                if var_idx < max_var_idx {
                    let value_scalar = Scalar::from_bytes_mod_order(vars[var_idx]);
                    sum += g_scalars[col] * value_scalar;
                }
            }
            sum *= rho;
            vars[self.layout.k_prime_index(row)] = sum.to_bytes();
            k_prime_values.push(sum);
        }

        let mut k_diff_left_vals = Vec::with_capacity(self.layout.row_count());
        for row in 0..self.layout.row_count() {
            let diff = k_values[row] - a_scalar;
            vars[self.layout.k_diff_left_index(row)] = diff.to_bytes();
            k_diff_left_vals.push(diff);
        }

        let mut k_prime_diff_right_vals = Vec::with_capacity(self.layout.row_count());
        for row in 0..self.layout.row_count() {
            let diff = k_prime_values[row] - a_scalar;
            vars[self.layout.k_prime_diff_right_index(row)] = diff.to_bytes();
            k_prime_diff_right_vals.push(diff);
        }

        if self.layout.row_count() > 0 {
            let mut prod_left = k_diff_left_vals[0];
            vars[self.layout.k_prod_left_index(0)] = prod_left.to_bytes();
            for row in 1..self.layout.row_count() {
                prod_left *= k_diff_left_vals[row];
                vars[self.layout.k_prod_left_index(row)] = prod_left.to_bytes();
            }

            let mut prod_right = k_prime_diff_right_vals[0];
            vars[self.layout.k_prime_prod_right_index(0)] = prod_right.to_bytes();
            for row in 1..self.layout.row_count() {
                prod_right *= k_prime_diff_right_vals[row];
                vars[self.layout.k_prime_prod_right_index(row)] = prod_right.to_bytes();
            }
        }

        let mut s_values = Vec::with_capacity(self.w);
        for col in 0..self.w {
            let mut sum = Scalar::ZERO;
            for row in 0..self.layout.row_count() {
                let sigma = col_scalars[col];
                let var_idx = self.var_matrix_shuffled_row[row][col];
                if var_idx < max_var_idx {
                    let value_scalar = Scalar::from_bytes_mod_order(vars[var_idx]);
                    sum += sigma * g_prime_scalars[row] * value_scalar;
                }
            }
            vars[self.layout.s_index(col)] = sum.to_bytes();
            s_values.push(sum);
        }

        let mut s_prime_values = Vec::with_capacity(self.w);
        for col in 0..self.w {
            let mut sum = Scalar::ZERO;
            for row in 0..self.layout.row_count() {
                let source_col = if col == 0 { self.w - 1 } else { col - 1 };
                let sigma = col_scalars[source_col];
                let var_idx = self.var_matrix_shuffled_col[row][col];
                if var_idx < max_var_idx {
                    let value_scalar = Scalar::from_bytes_mod_order(vars[var_idx]);
                    sum += sigma * g_prime_scalars[row] * value_scalar;
                }
            }
            vars[self.layout.s_prime_index(col)] = sum.to_bytes();
            s_prime_values.push(sum);
        }

        let mut s_diff_left_vals = Vec::with_capacity(self.w);
        for col in 0..self.w {
            let diff = s_values[col] - a_prime_scalar;
            vars[self.layout.s_diff_left_index(col)] = diff.to_bytes();
            s_diff_left_vals.push(diff);
        }

        let mut s_prime_diff_right_vals = Vec::with_capacity(self.w);
        for col in 0..self.w {
            let diff = s_prime_values[col] - a_prime_scalar;
            vars[self.layout.s_prime_diff_right_index(col)] = diff.to_bytes();
            s_prime_diff_right_vals.push(diff);
        }

        if self.w > 0 {
            let mut s_left_prod = s_diff_left_vals[0];
            vars[self.layout.s_prod_left_index(0)] = s_left_prod.to_bytes();
            for col in 1..self.w {
                s_left_prod *= s_diff_left_vals[col];
                vars[self.layout.s_prod_left_index(col)] = s_left_prod.to_bytes();
            }

            let mut s_right_prod = s_prime_diff_right_vals[0];
            vars[self.layout.s_prime_prod_right_index(0)] = s_right_prod.to_bytes();
            for col in 1..self.w {
                s_right_prod *= s_prime_diff_right_vals[col];
                vars[self.layout.s_prime_prod_right_index(col)] = s_right_prod.to_bytes();
            }
        }

        let mut p_l_values = Vec::with_capacity(self.batch_size);
        let mut r_bit_values = Vec::with_capacity(self.batch_size);
        let mut p_l_indices = Vec::with_capacity(self.batch_size);
        let mut r_bit_indices = Vec::with_capacity(self.batch_size);

        for batch in 0..self.batch_size {
            let mut p_bits = Vec::new();
            let mut p_indices_vec = Vec::new();
            for col in 0..self.w {
                if let Some(idx) = self.indices.p_l_bits[batch][col] {
                    let scalar = Scalar::from_bytes_mod_order(vars[idx]);
                    let bit = if scalar == Scalar::ZERO {
                        0
                    } else if scalar == Scalar::ONE {
                        1
                    } else {
                        return Err(format!("Double-Shuffling p_l index {} non-boolean", idx));
                    };
                    p_bits.push(bit);
                    p_indices_vec.push(idx);
                }
            }
            p_l_values.push(p_bits);
            p_l_indices.push(p_indices_vec);

            let mut r_rows_bits = Vec::with_capacity(self.beta.saturating_sub(1));
            let mut r_rows_indices = Vec::with_capacity(self.beta.saturating_sub(1));
            for local_row in 0..self.beta.saturating_sub(1) {
                let mut row_bits = Vec::with_capacity(self.w);
                let mut row_indices = Vec::with_capacity(self.w);
                for col in 0..self.w {
                    if let Some(idx) = self.indices.r_bits[batch][local_row][col] {
                        let scalar = Scalar::from_bytes_mod_order(vars[idx]);
                        let bit = if scalar == Scalar::ZERO {
                            0
                        } else if scalar == Scalar::ONE {
                            1
                        } else {
                            return Err(format!(
                                "Double-Shuffling r_bit index {} non-boolean",
                                idx
                            ));
                        };
                        row_bits.push(bit);
                        row_indices.push(idx);
                    }
                }
                r_rows_bits.push(row_bits);
                r_rows_indices.push(row_indices);
            }
            r_bit_values.push(r_rows_bits);
            r_bit_indices.push(r_rows_indices);
        }

        Ok(DoubleShuffleWitness::with_batch(
            self.batch_size,
            self.beta,
            self.w,
            self.randomness.clone(),
        )
        .with_indices(p_l_indices, r_bit_indices)
        .with_trace(p_l_values, r_bit_values))
    }
}

pub struct Fl2saR1CSArtifacts {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_non_zero_entries: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub double_shuffle: Fl2saDoubleShuffleBundle,
    pub metrics: R1csShapeMetrics,
}

impl Fl2saR1CSArtifacts {
    pub fn into_witness(
        self,
        p: u64,
        b: u32,
        v: Vec<u64>,
        superaccumulator: Vec<i64>,
    ) -> Fl2saWitness {
        Fl2saWitness {
            p,
            b,
            v,
            num_constraints: self.num_cons,
            num_vars: self.num_vars,
            instance: self.instance,
            vars: self.vars,
            inputs: self.inputs,
            superaccumulator,
            double_shuffle: self.double_shuffle,
            field_ops: self.metrics,
        }
    }
}

pub struct Fl2saWitness {
    pub p: u64,
    pub b: u32,
    pub v: Vec<u64>,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub superaccumulator: Vec<i64>,
    pub double_shuffle: Fl2saDoubleShuffleBundle,
    pub field_ops: R1csShapeMetrics,
}

impl fmt::Debug for Fl2saWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fl2saWitness")
            .field("p", &self.p)
            .field("b", &self.b)
            .field("v_len", &self.v.len())
            .field("num_constraints", &self.num_constraints)
            .field("num_vars", &self.num_vars)
            .field("superaccumulator_len", &self.superaccumulator.len())
            .field("double_shuffle", &self.double_shuffle)
            .field("instance", &"Instance { .. }")
            .field("vars", &"VarsAssignment { .. }")
            .field("inputs", &"InputsAssignment { .. }")
            .field("field_adds", &self.field_ops.field_adds)
            .field("field_muls", &self.field_ops.field_muls)
            .finish()
    }
}

pub struct Fl2saBatchWitness {
    pub entry_indices: Vec<usize>,
    pub double_shuffle: Fl2saDoubleShuffleBundle,
    pub instance: Instance,
    pub vars: VarsAssignment,
    pub inputs: InputsAssignment,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub field_ops: R1csShapeMetrics,
}

impl fmt::Debug for Fl2saBatchWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fl2saBatchWitness")
            .field("entry_indices", &self.entry_indices)
            .field("double_shuffle", &self.double_shuffle)
            .field("num_constraints", &self.num_constraints)
            .field("num_vars", &self.num_vars)
            .field("num_inputs", &self.num_inputs)
            .field("field_adds", &self.field_ops.field_adds)
            .field("field_muls", &self.field_ops.field_muls)
            .finish()
    }
}

impl Fl2saBatchWitness {
    pub fn from_indices(
        indices: &[usize],
        source: &[Fl2saWitness],
        params: &FL2SAParams,
    ) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("Fl2saBatchWitness::assemble requires at least one witness".to_owned());
        }

        let beta = params.beta();
        let w = params.w as usize;
        let lgw = params.lgw;
        let alpha = params.alpha();

        if let Some(first) = indices.get(0).and_then(|idx| source.get(*idx)) {
            let core = &first.double_shuffle.core;
            if core.beta != beta || core.w != w {
                return Err("The Fl2saBatchWitness::from_indices parameter is inconsistent with the witness configuration".to_owned());
            }
            if first.double_shuffle.ah.cols != alpha {
                return Err("Fl2saBatchWitness::from_indices ah The vector length is inconsistent with the parameters".to_owned());
            }
        }

        let batch_size = indices.len();
        let mut ds_indices = DoubleShuffleBatchIndices::new(batch_size, beta, w);
        let mut ah_indices: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
        let mut bit_indices: Vec<usize> = Vec::new();
        let mut bit_assignments: Vec<(usize, u8)> = Vec::new();

        let mut next_var_idx = 0usize;

        for (batch_pos, &idx) in indices.iter().enumerate() {
            let witness = source.get(idx).ok_or_else(|| {
                format!(
                    "Fl2saBatchWitness::from_indices index {} is out of range",
                    idx
                )
            })?;

            if witness.double_shuffle.core.batch_size != 1 {
                return Err(
                    "The FL2SA witness that has been batched is not supported for batching again."
                        .to_owned(),
                );
            }
            if witness.double_shuffle.ah.batch_size != 1 {
                return Err("It is not supported to batch ah Double-Shuffling again.".to_owned());
            }

            let core = &witness.double_shuffle.core;

            let p_bits_raw = core.p_l_values.get(0).cloned().unwrap_or_default();
            let mut p_bits_truncated = vec![0u8; lgw];
            for (col, value) in p_bits_raw.into_iter().enumerate().take(lgw) {
                p_bits_truncated[col] = value;
            }
            for col in 0..lgw {
                ds_indices.p_l_bits[batch_pos][col] = Some(next_var_idx);
                bit_indices.push(next_var_idx);
                bit_assignments.push((next_var_idx, p_bits_truncated[col]));
                next_var_idx += 1;
            }

            for row in 0..beta.saturating_sub(1) {
                let row_bits_raw = core
                    .r_bit_values
                    .get(0)
                    .and_then(|rows| rows.get(row))
                    .cloned()
                    .unwrap_or_default();
                let mut row_bits_full = vec![0u8; w];
                for (col, value) in row_bits_raw.into_iter().enumerate() {
                    if col < w {
                        row_bits_full[col] = value;
                    }
                }
                for col in 0..w {
                    ds_indices.r_bits[batch_pos][row][col] = Some(next_var_idx);
                    bit_indices.push(next_var_idx);
                    bit_assignments.push((next_var_idx, row_bits_full[col]));
                    next_var_idx += 1;
                }
            }

            let mut ah_indices_row = Vec::with_capacity(alpha);
            let ah_values_row = witness
                .double_shuffle
                .ah
                .ah_values
                .get(0)
                .cloned()
                .unwrap_or_else(|| vec![0u8; alpha]);
            if ah_values_row.len() != alpha {
                return Err(format!(
                    "ah Double-Shuffling length mismatch: expected {}, actual {}",
                    alpha,
                    ah_values_row.len()
                ));
            }
            for value in ah_values_row {
                let idx_var = next_var_idx;
                next_var_idx += 1;
                ah_indices_row.push(idx_var);
                bit_indices.push(idx_var);
                bit_assignments.push((idx_var, value));
            }
            ah_indices.push(ah_indices_row);
        }

        let double_shuffle_start_idx = next_var_idx;
        let layout_preview =
            DoubleShuffleVarLayout::new(double_shuffle_start_idx, beta, w, batch_size);
        let double_shuffle_var_end = layout_preview.next_var_idx();
        let ah_double_shuffle_start_idx = double_shuffle_var_end;
        let ah_layout_preview =
            AhDoubleShuffleVarLayout::new(ah_double_shuffle_start_idx, batch_size, alpha);
        let num_vars = ah_layout_preview.next_var_idx();
        let const_one_idx = num_vars;
        let zero_var_idx = const_one_idx + 1;
        let max_var_idx = zero_var_idx + 1;

        let mut rng = OsRng;
        let core_randomness = DoubleShuffleRandomness::with_batch(beta, w, batch_size, &mut rng);
        let ah_randomness = sample_ah_double_shuffle_randomness(batch_size, alpha, &mut rng);

        let context = DoubleShuffleBatchContext::new(
            beta,
            w,
            lgw,
            batch_size,
            double_shuffle_start_idx,
            core_randomness,
            ds_indices,
            zero_var_idx,
        );
        let ah_context = AhDoubleShuffleContext::new(
            batch_size,
            alpha,
            ah_double_shuffle_start_idx,
            ah_randomness,
            ah_indices.clone(),
            zero_var_idx,
        )?;

        let one = Scalar::ONE.to_bytes();
        let minus_one = (-Scalar::ONE).to_bytes();

        let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

        let mut constraint_cursor = 0usize;
        constraint_cursor = context.append_constraints(
            constraint_cursor,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            num_vars,
        );
        constraint_cursor = ah_context.append_constraints(
            constraint_cursor,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            const_one_idx,
        );

        for bit_idx in &bit_indices {
            let constraint_id = constraint_cursor;
            constraint_cursor += 1;
            a_entries.push((constraint_id, *bit_idx, one));
            b_entries.push((constraint_id, *bit_idx, one));
            b_entries.push((constraint_id, const_one_idx, minus_one));
            c_entries.push((constraint_id, zero_var_idx, one));
        }

        let zero_constraint_id = constraint_cursor;
        constraint_cursor += 1;
        a_entries.push((zero_constraint_id, zero_var_idx, one));
        b_entries.push((zero_constraint_id, const_one_idx, one));

        let num_constraints = constraint_cursor;
        let metrics = compute_r1cs_metrics(num_constraints, &a_entries, &b_entries, &c_entries);
        let instance = Instance::new(
            num_constraints,
            max_var_idx,
            1,
            &a_entries,
            &b_entries,
            &c_entries,
        )
        .map_err(|e| format!("Failed to create Double-Shuffling batch instance: {:?}", e))?;

        let mut vars = vec![Scalar::ZERO.to_bytes(); max_var_idx];
        vars[const_one_idx] = Scalar::ONE.to_bytes();
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();
        for (idx, value) in &bit_assignments {
            vars[*idx] = Scalar::from(*value as u64).to_bytes();
        }

        let core_double_shuffle_witness = context.populate_witness(&mut vars, max_var_idx)?;
        let ah_double_shuffle_witness = ah_context.populate_witness(&mut vars)?;

        let vars_assignment = VarsAssignment::new(&vars)
            .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
        let input_assignment = InputsAssignment::new(&[Scalar::ONE.to_bytes()])
            .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

        match instance.is_sat(&vars_assignment, &input_assignment) {
            Ok(true) => {}
            Ok(false) => {
                return Err("Batch Double-Shuffling SAT check returns false".to_owned());
            }
            Err(err) => {
                return Err(format!(
                    "Batch Double-Shuffling SAT check failed: {:?}",
                    err
                ));
            }
        }

        Ok(Self {
            entry_indices: indices.to_vec(),
            double_shuffle: Fl2saDoubleShuffleBundle {
                core: core_double_shuffle_witness,
                ah: ah_double_shuffle_witness,
            },
            instance,
            vars: vars_assignment,
            inputs: input_assignment,
            num_constraints,
            num_vars,
            num_inputs: 1,
            field_ops: metrics,
        })
    }
}

//
//
//
//
// +---------------------------------------------------------------------+
// +---------------------------------------------------------------------+
//                                    down
// +---------------------------------------------------------------------+
// +---------------------------------------------------------------------+
//                                    down
// +---------------------------------------------------------------------+
// |                                                                     |
// |               down       down       down                                   |
// |               down       down       down                                   |
// |               down       down       down                                   |
// |               down       down       down                                   |
// |              down       down       down       down                            |
// |              down       down       down       down                            |
// |                                                                     |
// |                                                                     |
// +---------------------------------------------------------------------+

// ============================================================================
// ============================================================================

///
///
pub fn fl2sa_complete_algorithm<F: Float>(
    b: u32,
    v: &[u64],
    p: u64,
    w: u32,
    alpha: usize,
    m: u32,
    precision: FloatPrecision,
) -> Vec<F> {
    let mut y = vec![F::zero(); alpha];

    fl2sa_generic(&mut y, b, v, p, w, m, precision);

    let mut result = Vec::new();
    for i in (0..alpha).rev() {
        result.push(y[i]);
    }

    result
}

///
///
pub fn fl2sa_batch_algorithm<F: Float>(
    inputs: &[(u32, Vec<u64>, u64)],
    w: u32,
    alpha: usize,
    m: u32,
    precision: FloatPrecision,
) -> Vec<F> {
    let mut y = vec![F::zero(); alpha];

    for (b, v, p) in inputs {
        fl2sa_generic(&mut y, *b, v, *p, w, m, precision);
    }

    let mut result = Vec::new();
    for i in (0..alpha).rev() {
        result.push(y[i]);
    }

    result
}

///
///
pub fn fl2sa_example_f64(b: u32, v: &[u64], p: u64) -> Vec<f64> {
    let w = 4;
    let alpha = 11;
    let m = 10;

    fl2sa_complete_algorithm(b, v, p, w, alpha, m, FloatPrecision::Double)
}
pub fn fl2sa_generic<F: Float>(
    y: &mut [F],
    b: u32,
    v: &[u64],
    p: u64,
    w: u32,
    m: u32,
    precision: FloatPrecision,
) {
    let alpha = y.len();
    let beta = v.len() + 1;

    let (_mantissa_bits, _exponent_bits) = match precision {
        FloatPrecision::Half => (10, 5),
        FloatPrecision::Single => (23, 8),
        FloatPrecision::Double => (52, 11),
    };

    let p_h = p.checked_div(w.into()).unwrap_or(0);
    let p_l = p.checked_rem(w.into()).unwrap_or(0);

    let z = if p == 0 { 1 } else { 0 };
    //

    //
    //
    //
    let mut v = v.to_vec();
    if beta >= 2 {
        let power_exp = if m as u32 >= (w as u32) * (beta as u32 - 2) {
            (m as u32) - (w as u32) * (beta as u32 - 2)
        } else {
            0
        };
        assert!(power_exp < m, "power_exp exceeds Po2 array bounds");
        let power_2_exp = 1u64 << power_exp;

        v[beta - 2] = v[beta - 2].saturating_add(power_2_exp * (1 - z) as u64);
    }

    let mut u = vec![0u64; beta - 1];
    for i in 0..(beta - 1) {
        u[i] = v[i].saturating_mul(1u64 << p_l); // u[i] = v[i] x 2^{p_l}
    }

    let mut d = vec![0u64; beta - 1];
    for i in 0..(beta - 1) {
        d[i] = u[i] / (1u64 << w);
    }

    let mut v_prime = vec![0i64; beta];
    for i in (1..(beta - 1)).rev() {
        let shifted = (d[i] as u64) << w;
        let remainder = u[i].saturating_sub(shifted) as i64;
        let carry = d[i - 1] as i64;
        v_prime[i] = remainder.saturating_add(carry);
    }
    v_prime[0] = (u[0].saturating_sub((d[0] as u64) << w)) as i64;
    v_prime[beta - 1] = d[beta - 2] as i64;

    let sign = 1 - 2 * (b as i64);
    for i in 0..beta {
        v_prime[i] = v_prime[i].saturating_mul(sign);
    }

    for i in 0..beta {
        let idx = (p_h as usize) + i;
        if idx < alpha {
            let value = match precision {
                FloatPrecision::Half => {
                    let f64_val = v_prime[i] as f64;
                    if f64_val > 65504.0 {
                        F::from(65504.0f32).unwrap()
                    } else if f64_val < -65504.0 {
                        F::from(-65504.0f32).unwrap()
                    } else {
                        let f16_val = f16::from_f64(f64_val);
                        F::from(f16_val.to_f32()).unwrap()
                    }
                }
                FloatPrecision::Single => {
                    let f64_val = v_prime[i] as f64;
                    if f64_val > 3.4028235e38 {
                        F::from(3.4028235e38f32).unwrap()
                    } else if f64_val < -3.4028235e38 {
                        F::from(-3.4028235e38f32).unwrap()
                    } else {
                        F::from(f64_val as f32).unwrap()
                    }
                }
                FloatPrecision::Double => {
                    if v_prime[i] > 9007199254740991 {
                        F::from(9007199254740991.0).unwrap()
                    } else if v_prime[i] < -9007199254740991 {
                        F::from(-9007199254740991.0).unwrap()
                    } else {
                        F::from(v_prime[i] as f64).unwrap()
                    }
                }
            };
            y[idx] = y[idx] + value;
        }
    }
}

///
///
///
///
/// ```rust
/// let default = FL2SAParams::default(); // w=4, e=5, m=10 -> alpha=11, beta=4
///
/// let high_precision = FL2SAParams { w: 8, e: 6, m: 16, lgw: 3, acc: 64 }; // alpha=6, beta=3
///
/// let lightweight = FL2SAParams { w: 2, e: 4, m: 8, lgw: 1, acc: 32 }; // alpha=13, beta=6
/// ```
#[derive(Debug, Clone)]
pub struct FL2SAParams {
    pub w: usize,
    pub e: usize,
    pub m: usize,
    pub lgw: usize,
    pub acc: usize,
}

impl Default for FL2SAParams {
    fn default() -> Self {
        Self {
            w: 4,
            e: 5,
            m: 10,
            lgw: 2,
            acc: 64,
        }
    }
}

impl FL2SAParams {
    pub fn alpha(&self) -> usize {
        ((1 << self.e) + self.m + self.w - 1) / self.w
    }

    pub fn beta(&self) -> usize {
        ((self.m + 1) + self.w - 1) / self.w + 1
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.w == 0 || !self.w.is_power_of_two() {
            return Err("w must be a power of 2 and greater than 0".to_string());
        }
        if self.lgw != (self.w as f64).log2() as usize {
            return Err("lgw must be equal to log2(w)".to_string());
        }
        if self.e == 0 || self.m == 0 {
            return Err("e and m must be greater than 0".to_string());
        }
        if self.acc < self.m {
            return Err("acc must be greater than or equal to m".to_string());
        }
        Ok(())
    }

    ///
    ///
    ///
    /// ```rust
    /// let params = FL2SAParams::single_precision(4);
    /// println!("Alpha: {}, Beta: {}", params.alpha(), params.beta());
    /// ```
    ///
    /// - alpha approximately 70
    /// - beta approximately 7
    pub fn single_precision(w: usize) -> Self {
        let lgw = (w as f64).log2() as usize;
        Self {
            w,
            e: 8,
            m: 23,
            lgw,
            acc: 64,
        }
    }

    ///
    ///
    ///
    /// ```rust
    /// let params = FL2SAParams::double_precision(8);
    /// println!("Alpha: {}, Beta: {}", params.alpha(), params.beta());
    /// ```
    ///
    /// - alpha approximately 525
    /// - beta approximately 15
    ///
    pub fn double_precision(w: usize) -> Self {
        let lgw = (w as f64).log2() as usize;
        Self {
            w,
            e: 11,
            m: 52,
            lgw,
            acc: 64,
        }
    }

    ///
    pub fn precision_info(&self) -> (&'static str, &'static str) {
        match (self.m, self.e) {
            (23, 8) => ("single precision", "f32 (IEEE 754)"),
            (52, 11) => ("double precision", "f64 (IEEE 754)"),
            _ => ("Customize", "Non-standard precision configuration"),
        }
    }
}

///
///
///
/// ```rust
/// let input_data = FL2SAInputData::with_default_params(3, 0, vec![0x5, 0x7, 0x2])?;
///
/// let params = FL2SAParams { w: 8, e: 4, m: 12, lgw: 3, acc: 64 };
/// let input_data = FL2SAInputData::with_params(&params, 2, 0, Some(vec![0x3F, 0x2A]))?;
///
/// let input_data = FL2SAInputData::with_params(&params, 5, 1, None)?;
/// ```
#[derive(Debug, Clone)]
pub struct FL2SAInputData {
    pub p: u64,
    pub b: u32,
    pub v: Vec<u64>,
    pub witness_values: Option<Vec<[u8; 32]>>,
    pub double_shuffle_randomness: Option<DoubleShuffleRandomness>,
    pub double_shuffle_seed: Option<[u8; 32]>,
    pub ah_double_shuffle_randomness: Option<AhDoubleShuffleRandomness>,
    pub ah_double_shuffle_seed: Option<[u8; 32]>,
}

impl Default for FL2SAInputData {
    fn default() -> Self {
        let default_params = FL2SAParams::default();
        let beta_minus_one = default_params.beta() - 1;

        Self {
            p: 0,
            b: 0,
            v: vec![0; beta_minus_one],
            witness_values: None,
            double_shuffle_randomness: None,
            double_shuffle_seed: None,
            ah_double_shuffle_randomness: None,
            ah_double_shuffle_seed: None,
        }
    }
}

impl FL2SAInputData {
    ///
    ///
    ///
    /// ```rust
    /// let params = FL2SAParams { w: 8, e: 4, m: 12, lgw: 3, acc: 64 };
    /// let input_data = FL2SAInputData::with_params(&params, 5, 0, Some(vec![0x1F, 0x2A]));
    /// ```
    pub fn with_params(
        params: &FL2SAParams,
        p: u64,
        b: u32,
        v_values: Option<Vec<u64>>,
    ) -> Result<Self, String> {
        let beta_minus_one = params.beta() - 1;

        let v = if let Some(values) = v_values {
            if values.len() != beta_minus_one {
                return Err(format!(
                    "v vector length mismatch: expected {} blocks (beta-1), actual {} blocks",
                    beta_minus_one,
                    values.len()
                ));
            }
            values
        } else {
            vec![0; beta_minus_one]
        };

        Ok(Self {
            p,
            b,
            v,
            witness_values: None,
            double_shuffle_randomness: None,
            double_shuffle_seed: None,
            ah_double_shuffle_randomness: None,
            ah_double_shuffle_seed: None,
        })
    }

    ///
    ///
    /// ```rust
    /// let input_data = FL2SAInputData::with_default_params(3, 0, vec![0x5, 0x7, 0x2]);
    /// ```
    pub fn with_default_params(p: u64, b: u32, v_values: Vec<u64>) -> Result<Self, String> {
        let default_params = FL2SAParams::default();
        Self::with_params(&default_params, p, b, Some(v_values))
    }

    ///
    ///
    pub fn validate_with_params(&self, params: &FL2SAParams) -> Result<(), String> {
        let expected_beta_minus_one = params.beta() - 1;

        if self.v.len() != expected_beta_minus_one {
            return Err(format!(
                "v vector length does not match parameters: current {} blocks, parameter requires {} blocks (beta-1)",
                self.v.len(),
                expected_beta_minus_one
            ));
        }

        if self.b > 1 {
            return Err("Sign bit b must be 0 or 1".to_string());
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
pub fn produce_r1cs_fl2sa_with_params(
    params: &FL2SAParams,
    input_data: Option<&FL2SAInputData>,
) -> Result<Fl2saR1CSArtifacts, String> {
    params.validate()?;

    let w = params.w;
    let _e = params.e;
    let m = params.m;
    let lgw = params.lgw;
    let acc = params.acc;

    let alpha = params.alpha();
    let beta = params.beta();

    let num_inputs = 1;

    let one = Scalar::ONE.to_bytes();
    let minus_one = (-Scalar::ONE).to_bytes();
    let _zero = Scalar::ZERO.to_bytes();

    let two = Scalar::from(2u8);
    let _four = Scalar::from(4u8);
    let mut power_of_two = Scalar::ONE;

    let mut Po2 = Vec::new();
    Po2.push(Scalar::ONE.to_bytes());
    for _i in 0..(acc - 1) {
        power_of_two = power_of_two * two;
        Po2.push(power_of_two.to_bytes());
    }

    let mut PPo2 = Vec::new();
    for i in 0..lgw {
        let t = 2u32.pow(2u32.pow(i as u32)) as u16;
        PPo2.push(Scalar::from(t).to_bytes())
    }

    // ============================================================================
    // ============================================================================

    let p_idx = 0;
    let p_h_idx = 1;
    let p_l_idx = 2;
    let p_l_bits_base_idx = 3;

    let z_var_idx = 3 + lgw;
    let p_inv_idx = z_var_idx + 1;
    let v_beta2_old_idx = p_inv_idx + 1; // z[7]: v[beta-2]_old
    let v_beta2_new_idx = v_beta2_old_idx + 1; // z[8]: v[beta-2]_new

    let v_base_idx = v_beta2_new_idx + 1;

    let u_base_idx = v_base_idx + (beta - 2);

    let pow_2_pl_intermediate_base = u_base_idx + (beta - 1);
    let pow_2_pl_intermediate_count = if lgw > 1 { 2 * (lgw - 1) } else { 0 };
    let pow_2_pl_idx = pow_2_pl_intermediate_base + pow_2_pl_intermediate_count; // z[16]: 2^p_l

    let d_base_idx = pow_2_pl_idx + 1;

    let r_base_idx = d_base_idx + (beta - 1);

    let v_prime_base_idx = r_base_idx + (beta - 1);

    let b_idx = v_prime_base_idx + beta;
    let sign_idx = b_idx + 1;
    let v_prime_signed_base_idx = sign_idx + 1;

    let r_bit_base_idx = v_prime_signed_base_idx + beta;

    let k_base_idx = r_bit_base_idx + (beta - 1) * w;
    let k_prime_base_idx = k_base_idx + beta;
    let k_diff_left_base_idx = k_prime_base_idx + beta;
    let k_prime_diff_right_base_idx = k_diff_left_base_idx + beta;
    let k_prod_left_base_idx = k_prime_diff_right_base_idx + beta;
    let k_prime_prod_right_base_idx = k_prod_left_base_idx + beta;
    let s_base_idx = k_prime_prod_right_base_idx + beta;
    let s_prime_base_idx = s_base_idx + w;
    let s_diff_left_base_idx = s_prime_base_idx + w;
    let s_prime_diff_right_base_idx = s_diff_left_base_idx + w;
    let s_prod_left_base_idx = s_prime_diff_right_base_idx + w;
    let s_prime_prod_right_base_idx = s_prod_left_base_idx + w;

    let ah_base_idx = s_prime_prod_right_base_idx + w;

    let aux_base_idx = ah_base_idx + alpha;
    let max_aux_pairs = alpha * beta;

    let delta_y_base_idx = aux_base_idx + max_aux_pairs;

    let gamma_delta_sum_idx = delta_y_base_idx + alpha; // z[159]: Sigmagamma_i.y'_i
    let gamma_ah_sum_base_idx = gamma_delta_sum_idx + 1;
    let gamma_v_product_base_idx = gamma_ah_sum_base_idx + beta; // z[164~167]: v'_k x Sigmagamma_i.a[...]
    let gamma_rhs_sum_idx = gamma_v_product_base_idx + beta; // z[168]: Sigma_k v'_k.Sigmagamma_i.a[...]

    let num_vars = gamma_rhs_sum_idx + 1;
    let zero_var_idx = num_vars + 1;

    // ============================================================================
    // ============================================================================

    let mut a_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c_entries: Vec<(usize, usize, [u8; 32])> = Vec::new();

    let constraint_0_p_l_decomp = 0;
    let constraint_1_p_combination = 1;
    let constraint_2_z_boolean = 2;
    let constraint_3_inverse_relation = 3;
    let constraint_4_consistency = 4;
    let constraint_5_v_update = 5;

    // ============================================================================
    // ============================================================================
    //
    //
    //
    // ============================================================================

    // ============================================================================
    // ============================================================================

    // p_l = sum(i=0 to lgw-1) 2^i x p_l[i]
    for i in 0..lgw {
        a_entries.push((constraint_0_p_l_decomp, p_l_bits_base_idx + i, Po2[i]));
    }
    b_entries.push((constraint_0_p_l_decomp, num_vars, one));
    c_entries.push((constraint_0_p_l_decomp, p_l_idx, one));

    // p_l + p_h x w = p
    a_entries.push((constraint_1_p_combination, p_l_idx, one));
    a_entries.push((
        constraint_1_p_combination,
        p_h_idx,
        Scalar::from(w as u64).to_bytes(),
    ));
    b_entries.push((constraint_1_p_combination, num_vars, one));
    c_entries.push((constraint_1_p_combination, p_idx, one));

    a_entries.push((constraint_2_z_boolean, z_var_idx, one));
    b_entries.push((constraint_2_z_boolean, z_var_idx, one));
    c_entries.push((constraint_2_z_boolean, z_var_idx, one));

    a_entries.push((constraint_3_inverse_relation, p_idx, one));
    b_entries.push((constraint_3_inverse_relation, p_inv_idx, one));
    c_entries.push((constraint_3_inverse_relation, num_vars, one));
    c_entries.push((constraint_3_inverse_relation, z_var_idx, minus_one));

    a_entries.push((constraint_4_consistency, z_var_idx, one));
    b_entries.push((constraint_4_consistency, p_idx, one));

    // v[beta-2]_old + 2^{m-w*(beta-2)} * (1-z) = v[beta-2]_new
    let power_exp = if m as u32 >= (w as u32) * (beta as u32 - 2) {
        (m as u32) - (w as u32) * (beta as u32 - 2)
    } else {
        0
    };
    assert!(power_exp < acc as u32, "power_exp exceeds Po2 array bounds");
    let power_constant = Po2[power_exp as usize];
    let minus_power_constant = (-Scalar::from_bytes_mod_order(power_constant)).to_bytes();

    a_entries.push((constraint_5_v_update, v_beta2_old_idx, one));
    a_entries.push((constraint_5_v_update, num_vars, power_constant));
    a_entries.push((constraint_5_v_update, z_var_idx, minus_power_constant));
    b_entries.push((constraint_5_v_update, num_vars, one));
    c_entries.push((constraint_5_v_update, v_beta2_new_idx, one));

    // power[0] = 1 + p_l[0]
    // term[i] = 1 + p_l[i] x (2^{2^i} - 1)  for i > 0
    // power[i] = power[i-1] x term[i]        for i > 0
    let pow_2_pl_constraint_base = 6;

    if lgw == 1 {
        a_entries.push((6, num_vars, one)); // 1
        a_entries.push((6, 3, one)); // + p_l[0]
        b_entries.push((6, num_vars, one)); // x 1
        c_entries.push((6, pow_2_pl_idx, one)); // = 2^p_l
    } else {
        for i in 0..lgw {
            if i == 0 {
                let power0_idx = pow_2_pl_intermediate_base;
                a_entries.push((6, num_vars, one)); // 1
                a_entries.push((6, 3, one)); // + p_l[0]
                b_entries.push((6, num_vars, one)); // x 1
                c_entries.push((6, power0_idx, one)); // = power[0]
            } else {
                let constraint_base = pow_2_pl_constraint_base + 2 * i - 1;
                let term_idx = pow_2_pl_intermediate_base + i;
                let prev_power_idx = if i == 1 {
                    pow_2_pl_intermediate_base // power[0]
                } else {
                    pow_2_pl_intermediate_base + lgw - 1 + i - 1 // power[i-1]
                };
                let power_idx = if i == lgw - 1 {
                    pow_2_pl_idx
                } else {
                    pow_2_pl_intermediate_base + lgw - 1 + i // power[i]
                };

                let multiplier = PPo2[i];
                let multiplier_minus_one =
                    (Scalar::from_bytes_mod_order(multiplier) - Scalar::ONE).to_bytes();
                a_entries.push((constraint_base, num_vars, one)); // 1
                a_entries.push((constraint_base, 3 + i, multiplier_minus_one)); // + p_l[i] x (2^{2^i} - 1)
                b_entries.push((constraint_base, num_vars, one)); // x 1
                c_entries.push((constraint_base, term_idx, one)); // = term[i]

                a_entries.push((constraint_base + 1, prev_power_idx, one)); // power[i-1]
                b_entries.push((constraint_base + 1, term_idx, one)); // x term[i]
                c_entries.push((constraint_base + 1, power_idx, one)); // = power[i]
            }
        }
    }

    let pow_2_pl_constraint_count = if lgw == 1 { 1 } else { 2 * lgw - 1 };
    let left_shift_constraint_base = pow_2_pl_constraint_base + pow_2_pl_constraint_count;

    for i in 0..(beta - 2) {
        a_entries.push((left_shift_constraint_base + i, v_base_idx + i, one)); // v[i]
        b_entries.push((left_shift_constraint_base + i, pow_2_pl_idx, one)); // 2^{p_l}
        c_entries.push((left_shift_constraint_base + i, u_base_idx + i, one)); // u[i]
    }
    a_entries.push((left_shift_constraint_base + beta - 2, v_beta2_new_idx, one)); // v[beta-2]
    b_entries.push((left_shift_constraint_base + beta - 2, pow_2_pl_idx, one)); // 2^{p_l}
    c_entries.push((
        left_shift_constraint_base + beta - 2,
        u_base_idx + beta - 2,
        one,
    )); // u[beta-2]

    // u[i] = d[i] x 2^w + r[i]
    let truncation_constraint_base = left_shift_constraint_base + (beta - 1);

    for i in 0..(beta - 1) {
        let constraint_id = truncation_constraint_base + i;
        // (2^w x d[i] + r[i]) x 1 = u[i]
        a_entries.push((constraint_id, d_base_idx + i, Po2[w])); // A: d[i] x 2^w
        a_entries.push((constraint_id, r_base_idx + i, one)); // A: + r[i]
        b_entries.push((constraint_id, num_vars, one)); // B: x 1
        c_entries.push((constraint_id, u_base_idx + i, one)); // C: = u[i]
    }

    let range_check_constraint_base = truncation_constraint_base + (beta - 1);

    for i in 0..(beta - 1) {
        let constraint_id = range_check_constraint_base + i;

        // r[i] = sum(j=0 to w-1) 2^j x r[i][j]
        for j in 0..w {
            a_entries.push((constraint_id, r_bit_base_idx + i * w + j, Po2[j]));
            // r[i][j] x 2^j
        }
        b_entries.push((constraint_id, num_vars, one)); // x 1
        c_entries.push((constraint_id, r_base_idx + i, one)); // = r[i]
    }

    // Double-Shuffling
    let batch_size = 1;
    let ds_indices =
        DoubleShuffleBatchIndices::single(beta, w, lgw, p_l_bits_base_idx, r_bit_base_idx);

    let (provided_randomness, provided_seed) = input_data
        .map(|data| {
            (
                data.double_shuffle_randomness.clone(),
                data.double_shuffle_seed,
            )
        })
        .unwrap_or((None, None));
    let mut fallback_rng = OsRng;
    let randomness_single = if let Some(randomness) = provided_randomness {
        if randomness.g.len() != w {
            return Err(format!(
                "Double-Shuffling g vector length mismatch: expected {}, actual {}",
                w,
                randomness.g.len()
            ));
        }
        let expected_rows = beta * batch_size;
        if randomness.g_prime.len() != expected_rows {
            return Err(format!(
                "Double-Shuffling g' vector length mismatch: expected {}, actual {}",
                expected_rows,
                randomness.g_prime.len()
            ));
        }
        if randomness.row_scalars.len() != batch_size {
            return Err(format!(
                "Double-Shuffling row_scalars length mismatch: expected {}, actual {}",
                batch_size,
                randomness.row_scalars.len()
            ));
        }
        if randomness.col_scalars.len() != w {
            return Err(format!(
                "Double-Shuffling col_scalars length mismatch: expected {}, actual {}",
                w,
                randomness.col_scalars.len()
            ));
        }
        randomness
    } else if let Some(seed) = provided_seed {
        DoubleShuffleRandomness::with_seed_and_batch(beta, w, batch_size, seed)
    } else {
        DoubleShuffleRandomness::with_batch(beta, w, batch_size, &mut fallback_rng)
    };

    let double_shuffle_start_idx = r_bit_base_idx + (beta - 1) * w;
    let double_shuffle_context = DoubleShuffleBatchContext::new(
        beta,
        w,
        lgw,
        batch_size,
        double_shuffle_start_idx,
        randomness_single,
        ds_indices,
        zero_var_idx,
    );

    let double_shuffle_constraint_base = range_check_constraint_base + (beta - 1);
    let v_prime_constraint_base = double_shuffle_context.append_constraints(
        double_shuffle_constraint_base,
        &mut a_entries,
        &mut b_entries,
        &mut c_entries,
        num_vars,
    );

    let sign_constraint_base = v_prime_constraint_base + beta;
    let ah_constraint_base = sign_constraint_base + 2 + beta;
    // ==================================================================================
    // ==================================================================================

    let constraint_id = v_prime_constraint_base;
    a_entries.push((constraint_id, u_base_idx, one)); // u[0]
    a_entries.push((
        constraint_id,
        d_base_idx,
        (-Scalar::from_bytes_mod_order(Po2[w])).to_bytes(),
    )); // -d[0] x 2^w
    b_entries.push((constraint_id, num_vars, one)); // x 1
    c_entries.push((constraint_id, v_prime_base_idx, one)); // = v_prime[0]

    for i in 1..(beta - 1) {
        let constraint_id = v_prime_constraint_base + i;

        a_entries.push((constraint_id, u_base_idx + i, one)); // u[i]
        a_entries.push((constraint_id, d_base_idx + i - 1, one));
        a_entries.push((
            constraint_id,
            d_base_idx + i,
            (-Scalar::from_bytes_mod_order(Po2[w])).to_bytes(),
        )); // - d[i] x 2^w

        b_entries.push((constraint_id, num_vars, one));

        c_entries.push((constraint_id, v_prime_base_idx + i, one));
    }

    let constraint_id = v_prime_constraint_base + beta - 1;
    a_entries.push((constraint_id, d_base_idx + beta - 2, one)); // d[beta-2]
    b_entries.push((constraint_id, num_vars, one)); // x 1
    c_entries.push((constraint_id, v_prime_base_idx + beta - 1, one)); // = v_prime[beta-1]

    // ==================================================================================
    // v_prime_signed[i] = v_prime[i] x (1 - 2b) for i in [0, beta-1]
    // ==================================================================================

    let sign_constraint_base = v_prime_constraint_base + beta;

    let constraint_id = sign_constraint_base;
    a_entries.push((constraint_id, b_idx, one)); // b
    b_entries.push((constraint_id, b_idx, one)); // x b
    c_entries.push((constraint_id, b_idx, one)); // = b

    let constraint_id = sign_constraint_base + 1;
    a_entries.push((constraint_id, num_vars, one));
    a_entries.push((constraint_id, b_idx, (-Scalar::from(2u8)).to_bytes())); // -2b
    b_entries.push((constraint_id, num_vars, one)); // x 1
    c_entries.push((constraint_id, sign_idx, one)); // = sign

    for i in 0..beta {
        let constraint_id = sign_constraint_base + 2 + i;
        a_entries.push((constraint_id, v_prime_base_idx + i, one)); // v_prime[i]
        b_entries.push((constraint_id, sign_idx, one)); // x sign
        c_entries.push((constraint_id, v_prime_signed_base_idx + i, one)); // = v_prime_signed[i]
    }

    // ==================================================================================
    // ==================================================================================
    //
    //
    //
    //
    //
    //
    // \begin{cases}
    //  y_i=y_i+\sum_{j=0}^i a[i-j] \cdot v_j^{\prime} & \text{if } i<\beta \\
    //  y_i=y_i+\sum_{j=0}^{\beta-1} a[i-j] \cdot v_j^{\prime} & \text{if } i \leq \alpha-\beta+1 \\
    //  y_i=y_i+\sum_{j=0}^{\alpha-1-i} a[i-\beta+1+j] \cdot v_{\beta-1-j} & \text{otherwise}
    // \end{cases}
    //
    //
    //

    // ==================================================================================
    // ==================================================================================
    //
    //

    // ==================================================================================
    // ==================================================================================

    for i in 0..alpha {
        let constraint_id = ah_constraint_base + i;
        a_entries.push((constraint_id, ah_base_idx + i, one)); // ah[i]
        b_entries.push((constraint_id, ah_base_idx + i, one)); // x ah[i]
        c_entries.push((constraint_id, ah_base_idx + i, one)); // = ah[i]
    }

    let sum_constraint_id = ah_constraint_base + alpha;
    for i in 0..alpha {
        a_entries.push((sum_constraint_id, ah_base_idx + i, one)); // ah[i]
    }
    b_entries.push((sum_constraint_id, num_vars, one)); // x 1
    c_entries.push((sum_constraint_id, num_vars, one)); // = 1

    let position_constraint_id = ah_constraint_base + alpha + 1;
    for i in 0..alpha {
        let weight = Scalar::from(i as u64).to_bytes();
        a_entries.push((position_constraint_id, ah_base_idx + i, weight)); // i x ah[i]
    }
    a_entries.push((position_constraint_id, p_h_idx, minus_one)); // - p_h
    b_entries.push((position_constraint_id, num_vars, one)); // x 1
    c_entries.push((position_constraint_id, num_vars, one)); // = 1

    let ah_constraint_count = alpha + 2;

    // ==================================================================================
    // ==================================================================================
    //
    // \begin{cases}
    //  y'_i=\sum_{j=0}^i ah[i-j] \cdot v'_{j} & \text{if } i<\beta \\
    //  y'_i=\sum_{j=0}^{\beta-1} ah[i-j] \cdot v'_{j} & \text{if } i \leq \alpha-\beta+1 \\
    //  y'_i=\sum_{j=0}^{\alpha-1-i} ah[i-\beta+1+j] \cdot v'_{\beta-1-j} & \text{otherwise}
    // \end{cases}
    //
    //

    let _product_constraint_base = ah_constraint_base + ah_constraint_count;
    let product_constraint_base = _product_constraint_base;
    let mut aux_count = 0;

    for i in 0..alpha {
        for j in 0..beta {
            let need_product = (i < beta && j <= i && (i - j) < alpha)
                || (i >= beta && i <= alpha - beta + 1 && j < beta && (i - j) < alpha)
                || (i > alpha - beta + 1
                    && j <= alpha - 1 - i
                    && (i - beta + 1 + j) < alpha
                    && (beta - 1 - j) < beta);

            if need_product {
                let constraint_id = product_constraint_base + aux_count;
                let aux_idx = aux_base_idx + aux_count;

                a_entries.push((constraint_id, ah_base_idx + i, one)); // ah[i]
                b_entries.push((constraint_id, v_prime_signed_base_idx + j, one)); // x v_prime_signed[j]
                c_entries.push((constraint_id, aux_idx, one)); // = aux[i][j]

                aux_count += 1;
            }
        }
    }

    // ========================================================================
    // ========================================================================
    //
    // sum_{i=0}^{alpha-1} gamma_i . y'_i = sum_{k=0}^{beta-1} v'_k . ( ... )
    //

    let mut rng_gamma = OsRng;
    let mut gamma_scalars: Vec<Scalar> = Vec::with_capacity(alpha);
    for _ in 0..alpha {
        let val = rng_gamma.gen_range(1..=u64::MAX);
        gamma_scalars.push(Scalar::from(val));
    }

    let mut gamma_ah_coeffs = vec![vec![Scalar::ZERO; alpha]; beta];
    let alpha_minus_beta_plus_1 = alpha.saturating_sub(beta).saturating_add(1);
    let alpha_minus_beta_plus_2 = alpha.saturating_sub(beta).saturating_add(2);
    for k in 0..beta {
        let coeffs = &mut gamma_ah_coeffs[k];

        let case1_end = beta.min(alpha);
        for i in k..case1_end {
            if i < alpha {
                let ah_idx = i - k;
                if ah_idx < alpha {
                    coeffs[ah_idx] += gamma_scalars[i];
                }
            }
        }

        if beta < alpha {
            let case2_start = beta;
            let case2_end = alpha_minus_beta_plus_1.min(alpha.saturating_sub(1));
            if case2_start <= case2_end {
                for i in case2_start..=case2_end {
                    if i < alpha {
                        let ah_idx = i.saturating_sub(k);
                        if ah_idx < alpha {
                            coeffs[ah_idx] += gamma_scalars[i];
                        }
                    }
                }
            }
        }

        if alpha > beta && k > 0 {
            let mut case3_start = alpha_minus_beta_plus_2;
            if case3_start >= alpha {
                case3_start = alpha.saturating_sub(1);
            }
            let case3_end = alpha
                .saturating_sub(beta)
                .saturating_add(k)
                .min(alpha.saturating_sub(1));
            if case3_start <= case3_end {
                for i in case3_start..=case3_end {
                    if i < alpha {
                        let ah_idx = i.saturating_sub(k);
                        if ah_idx < alpha {
                            coeffs[ah_idx] += gamma_scalars[i];
                        }
                    }
                }
            }
        }
    }

    let gamma_delta_constraint_id = product_constraint_base + aux_count;

    for i in 0..alpha {
        a_entries.push((
            gamma_delta_constraint_id,
            delta_y_base_idx + i,
            gamma_scalars[i].to_bytes(),
        ));
    }
    b_entries.push((gamma_delta_constraint_id, num_vars, one));
    c_entries.push((gamma_delta_constraint_id, gamma_delta_sum_idx, one));

    let gamma_ah_constraint_base = gamma_delta_constraint_id + 1;
    for k in 0..beta {
        let constraint_id = gamma_ah_constraint_base + k;
        for (ah_idx, coeff) in gamma_ah_coeffs[k].iter().enumerate() {
            if *coeff != Scalar::ZERO {
                a_entries.push((constraint_id, ah_base_idx + ah_idx, coeff.to_bytes()));
            }
        }
        b_entries.push((constraint_id, num_vars, one));
        c_entries.push((constraint_id, gamma_ah_sum_base_idx + k, one));
    }

    let gamma_product_constraint_base = gamma_ah_constraint_base + beta;
    for k in 0..beta {
        let constraint_id = gamma_product_constraint_base + k;
        a_entries.push((constraint_id, v_prime_signed_base_idx + k, one));
        b_entries.push((constraint_id, gamma_ah_sum_base_idx + k, one));
        c_entries.push((constraint_id, gamma_v_product_base_idx + k, one));
    }

    let gamma_rhs_sum_constraint_id = gamma_product_constraint_base + beta;
    for k in 0..beta {
        a_entries.push((
            gamma_rhs_sum_constraint_id,
            gamma_v_product_base_idx + k,
            one,
        ));
    }
    b_entries.push((gamma_rhs_sum_constraint_id, num_vars, one));
    c_entries.push((gamma_rhs_sum_constraint_id, gamma_rhs_sum_idx, one));

    let gamma_final_constraint_id = gamma_rhs_sum_constraint_id + 1;
    a_entries.push((gamma_final_constraint_id, gamma_delta_sum_idx, one));
    a_entries.push((gamma_final_constraint_id, gamma_rhs_sum_idx, minus_one));
    b_entries.push((gamma_final_constraint_id, num_vars, one));

    let product_constraint_count = aux_count;
    let rlc_constraint_count = 2 * beta + 3;
    let rlc_constraint_base = gamma_delta_constraint_id;
    let rlc_constraint_count_total = rlc_constraint_count;
    let _total_weighted_constraints = product_constraint_count + rlc_constraint_count_total;

    // ==================================================================================
    // ==================================================================================

    let mut actual_num_cons = rlc_constraint_base + rlc_constraint_count_total;

    let zero_constraint_id = actual_num_cons;
    a_entries.push((zero_constraint_id, zero_var_idx, one));
    b_entries.push((zero_constraint_id, num_vars, one));
    actual_num_cons += 1;

    let max_var_idx = zero_var_idx + 1;

    let total_entries = a_entries.len() + b_entries.len() + c_entries.len();
    let actual_num_non_zero_entries = total_entries.next_power_of_two();

    let metrics = compute_r1cs_metrics(actual_num_cons, &a_entries, &b_entries, &c_entries);
    let inst = Instance::new(
        actual_num_cons,
        max_var_idx,
        num_inputs,
        &a_entries,
        &b_entries,
        &c_entries,
    )
    .map_err(|e| format!("Failed to create R1CS instance: {:?}", e))?;

    fn scalar_from_i64(value: i64) -> Scalar {
        if value >= 0 {
            Scalar::from(value as u64)
        } else {
            -Scalar::from((-value) as u64)
        }
    }

    let (p_l_index_sets, r_bit_index_sets) = double_shuffle_context.extract_indices();

    let (
        assignment_vars,
        assignment_inputs,
        core_double_shuffle_witness,
        ah_double_shuffle_witness,
    ) = if let Some(data) = input_data {
        if data.v.len() != beta - 1 {
            return Err(format!(
                "v vector length mismatch: expected {} blocks, actual {} blocks",
                beta - 1,
                data.v.len()
            ));
        }

        let mut vars = vec![Scalar::ZERO.to_bytes(); max_var_idx];
        let const_one_idx = num_vars;
        vars[const_one_idx] = Scalar::ONE.to_bytes();
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();

        let w_u64 = w as u64;
        let lgw_usize = lgw;
        let p_scalar = Scalar::from(data.p);
        vars[p_idx] = p_scalar.to_bytes();

        let p_h = data.p / w_u64;
        let p_l = data.p % w_u64;
        vars[p_h_idx] = Scalar::from(p_h).to_bytes();
        vars[p_l_idx] = Scalar::from(p_l).to_bytes();

        for i in 0..lgw_usize {
            let bit = ((p_l >> i) & 1) as u64;
            vars[p_l_bits_base_idx + i] = Scalar::from(bit).to_bytes();
        }

        let z_val = if data.p == 0 { 1u64 } else { 0u64 };
        vars[z_var_idx] = Scalar::from(z_val).to_bytes();
        let p_inv_scalar = if data.p == 0 {
            Scalar::ZERO
        } else {
            p_scalar.invert()
        };
        vars[p_inv_idx] = p_inv_scalar.to_bytes();

        for i in 0..beta.saturating_sub(2) {
            vars[v_base_idx + i] = Scalar::from(data.v[i]).to_bytes();
        }

        let power_exp = if params.m >= w * (beta - 2) {
            params.m - w * (beta - 2)
        } else {
            0
        };
        if power_exp >= 64 {
            return Err(format!(
                "power_exp={} is outside the supported range (requires < 64)",
                power_exp
            ));
        }

        if p_l >= 64 {
            return Err(format!(
                "p_l={} is outside the supported range (requires < 64)",
                p_l
            ));
        }

        let mut adjusted_v = data.v.clone();
        let v_beta2_old = adjusted_v[beta - 2];
        vars[v_beta2_old_idx] = Scalar::from(v_beta2_old).to_bytes();
        let hidden_bit = if z_val == 0 {
            1u64 << (power_exp as u32)
        } else {
            0
        };
        let v_beta2_new = v_beta2_old + hidden_bit;
        adjusted_v[beta - 2] = v_beta2_new;
        vars[v_beta2_new_idx] = Scalar::from(v_beta2_new).to_bytes();

        let two_pow_pl_u64 = 1u64 << (p_l as u32);

        if lgw_usize > 1 {
            let bit0 = (p_l & 1) as u64;
            let mut power_scalar = Scalar::from(1 + bit0);
            vars[pow_2_pl_intermediate_base] = power_scalar.to_bytes();

            for i in 1..lgw_usize {
                let bit = ((p_l >> i) & 1) as u64;
                let pow_term = 1u64 << (1u32 << i);
                let term_val = if bit == 1 { pow_term } else { 1u64 };
                let term_scalar = Scalar::from(term_val);
                vars[pow_2_pl_intermediate_base + i] = term_scalar.to_bytes();

                let new_power = power_scalar * term_scalar;
                if i == lgw_usize - 1 {
                    vars[pow_2_pl_idx] = new_power.to_bytes();
                } else {
                    let power_idx = pow_2_pl_intermediate_base + lgw_usize - 1 + i;
                    vars[power_idx] = new_power.to_bytes();
                }
                power_scalar = new_power;
            }
        } else {
            vars[pow_2_pl_idx] = Scalar::from(1 + ((p_l & 1) as u64)).to_bytes();
        }

        vars[pow_2_pl_idx] = Scalar::from(two_pow_pl_u64).to_bytes();

        if w >= 64 {
            return Err(format!(
                "w={} is outside the supported range (requires < 64)",
                w
            ));
        }
        let w_u32 = w as u32;
        let mut u_values = Vec::with_capacity(beta - 1);
        let mut d_values = Vec::with_capacity(beta - 1);
        let mut r_values = Vec::with_capacity(beta - 1);
        let shift_amount = p_l as u32;
        let trunc_divisor = 1u64 << w_u32;

        for (idx, &block) in adjusted_v.iter().enumerate() {
            let u_val = (block as u128) << shift_amount;
            if u_val > u64::MAX as u128 {
                return Err("u value exceeds u64 range".to_string());
            }
            let u_u64 = u_val as u64;
            u_values.push(u_u64);
            vars[u_base_idx + idx] = Scalar::from(u_u64).to_bytes();

            let d_u64 = u_u64 / trunc_divisor;
            let r_u64 = u_u64 % trunc_divisor;
            d_values.push(d_u64);
            r_values.push(r_u64);
            vars[d_base_idx + idx] = Scalar::from(d_u64).to_bytes();
            vars[r_base_idx + idx] = Scalar::from(r_u64).to_bytes();

            for bit in 0..w {
                let bit_val = ((r_u64 >> bit) & 1) as u64;
                vars[r_bit_base_idx + idx * w + bit] = Scalar::from(bit_val).to_bytes();
            }
        }

        vars[b_idx] = Scalar::from(data.b).to_bytes();
        let sign_value = 1 - 2 * (data.b as i64);
        vars[sign_idx] = scalar_from_i64(sign_value).to_bytes();

        let mut v_prime = vec![0i64; beta];
        let mut v_prime_signed = vec![0i64; beta];

        v_prime[0] = r_values[0] as i64;
        for i in 1..(beta - 1) {
            v_prime[i] = r_values[i] as i64 + d_values[i - 1] as i64;
        }
        v_prime[beta - 1] = d_values[beta - 2] as i64;

        for i in 0..beta {
            vars[v_prime_base_idx + i] = scalar_from_i64(v_prime[i]).to_bytes();
            v_prime_signed[i] = v_prime[i] * sign_value;
            vars[v_prime_signed_base_idx + i] = scalar_from_i64(v_prime_signed[i]).to_bytes();
        }

        let p_h_usize = p_h as usize;
        if p_h_usize + 1 >= alpha {
            return Err(format!(
                "p_h={} exceeds superaccumulator range alpha={}",
                p_h_usize, alpha
            ));
        }
        let mut ah_vec = vec![0i64; alpha];
        ah_vec[p_h_usize + 1] = 1;
        for (idx, &value) in ah_vec.iter().enumerate() {
            vars[ah_base_idx + idx] = scalar_from_i64(value).to_bytes();
        }

        let mut aux_index = 0usize;
        for i in 0..alpha {
            for j in 0..beta {
                let need_product = (i < beta && j <= i && (i - j) < alpha)
                    || (i >= beta && i <= alpha - beta + 1 && j < beta && (i - j) < alpha)
                    || (i > alpha - beta + 1
                        && j <= alpha - 1 - i
                        && (i - beta + 1 + j) < alpha
                        && (beta - 1 - j) < beta);

                if need_product {
                    let product_value = ah_vec[i] * v_prime_signed[j];
                    vars[aux_base_idx + aux_index] = scalar_from_i64(product_value).to_bytes();
                    aux_index += 1;
                }
            }
        }

        if aux_index != aux_count {
            return Err(format!(
                "Inconsistent number of aux variables: expected {}, actual {}",
                aux_count, aux_index
            ));
        }

        let mut delta_y_scalars = vec![Scalar::ZERO; alpha];
        for i in 0..alpha {
            let delta_val = if i < beta {
                let mut sum = 0i64;
                for j in 0..=i {
                    let ah_idx = i - j;
                    sum += ah_vec[ah_idx] * v_prime_signed[j];
                }
                sum
            } else if i <= alpha - beta + 1 {
                let mut sum = 0i64;
                for j in 0..beta {
                    let ah_idx = i - j;
                    sum += ah_vec[ah_idx] * v_prime_signed[j];
                }
                sum
            } else {
                let mut sum = 0i64;
                for j in 0..=(alpha - 1 - i) {
                    let ah_idx = i - beta + 1 + j;
                    let v_idx = beta - 1 - j;
                    sum += ah_vec[ah_idx] * v_prime_signed[v_idx];
                }
                sum
            };
            let delta_scalar = scalar_from_i64(delta_val);
            delta_y_scalars[i] = delta_scalar;
            vars[delta_y_base_idx + i] = delta_scalar.to_bytes();
        }

        let mut gamma_delta_sum_scalar = Scalar::ZERO;
        for i in 0..alpha {
            gamma_delta_sum_scalar += gamma_scalars[i] * delta_y_scalars[i];
        }
        vars[gamma_delta_sum_idx] = gamma_delta_sum_scalar.to_bytes();

        let mut gamma_ah_sum_scalars = vec![Scalar::ZERO; beta];
        for k in 0..beta {
            let mut sum = Scalar::ZERO;
            for (ah_idx, coeff) in gamma_ah_coeffs[k].iter().enumerate() {
                if *coeff != Scalar::ZERO {
                    let ah_scalar = scalar_from_i64(ah_vec[ah_idx]);
                    sum += *coeff * ah_scalar;
                }
            }
            gamma_ah_sum_scalars[k] = sum;
            vars[gamma_ah_sum_base_idx + k] = sum.to_bytes();
        }

        let mut gamma_v_product_scalars = vec![Scalar::ZERO; beta];
        for k in 0..beta {
            let v_scalar = scalar_from_i64(v_prime_signed[k]);
            let product = v_scalar * gamma_ah_sum_scalars[k];
            gamma_v_product_scalars[k] = product;
            vars[gamma_v_product_base_idx + k] = product.to_bytes();
        }

        let mut gamma_rhs_sum_scalar = Scalar::ZERO;
        for value in &gamma_v_product_scalars {
            gamma_rhs_sum_scalar += *value;
        }
        vars[gamma_rhs_sum_idx] = gamma_rhs_sum_scalar.to_bytes();

        let core_double_shuffle_witness =
            double_shuffle_context.populate_witness(&mut vars, max_var_idx)?;
        let ah_indices_row: Vec<usize> = (0..alpha).map(|i| ah_base_idx + i).collect();
        let ah_bits: Vec<u8> = ah_vec
            .iter()
            .map(|&value| if value == 0 { 0 } else { 1 })
            .collect();
        let mut ah_rng = OsRng;
        let ah_randomness = sample_ah_double_shuffle_randomness(1, alpha, &mut ah_rng);
        let ah_double_shuffle_witness = AhDoubleShuffleWitness::empty(1, alpha, ah_randomness)
            .with_indices(vec![ah_indices_row])
            .with_values(vec![ah_bits]);
        let assignment_vars = VarsAssignment::new(&vars)
            .map_err(|e| format!("Failed to create variable assignment: {:?}", e))?;
        let input_bytes = vec![Scalar::from(data.p).to_bytes(); num_inputs];
        let assignment_inputs = InputsAssignment::new(&input_bytes)
            .map_err(|e| format!("Failed to create input assignment: {:?}", e))?;

        match inst.is_sat(&assignment_vars, &assignment_inputs) {
            Ok(true) => {}
            Ok(false) => {
                return Err(
                    "FL2SA witness does not satisfy R1CS constraints after generation".to_string(),
                );
            }
            Err(err) => {
                return Err(format!("FL2SA witness SAT check failed: {:?}", err));
            }
        }

        (
            assignment_vars,
            assignment_inputs,
            core_double_shuffle_witness,
            ah_double_shuffle_witness,
        )
    } else {
        let zero_bytes = Scalar::ZERO.to_bytes();
        let assignment_vars = VarsAssignment::new(&vec![zero_bytes; max_var_idx])
            .map_err(|e| format!("Failed to create default variable assignment: {:?}", e))?;
        let assignment_inputs = InputsAssignment::new(&vec![zero_bytes; num_inputs])
            .map_err(|e| format!("Failed to create default input assignment: {:?}", e))?;

        let p_l_traces: Vec<Vec<u8>> = p_l_index_sets
            .iter()
            .map(|indices| vec![0; indices.len()])
            .collect();
        let r_bit_traces: Vec<Vec<Vec<u8>>> = r_bit_index_sets
            .iter()
            .map(|rows| rows.iter().map(|cols| vec![0; cols.len()]).collect())
            .collect();
        let zero_witness = DoubleShuffleWitness::with_batch(
            batch_size,
            beta,
            w,
            double_shuffle_context.randomness.clone(),
        )
        .with_indices(p_l_index_sets.clone(), r_bit_index_sets.clone())
        .with_trace(p_l_traces, r_bit_traces);

        let zero_ah_values = vec![vec![0u8; alpha]];
        let mut ah_rng = OsRng;
        let zero_ah_randomness =
            sample_ah_double_shuffle_randomness(batch_size, alpha, &mut ah_rng);
        let zero_ah_indices: Vec<Vec<usize>> = vec![(0..alpha).map(|i| ah_base_idx + i).collect()];
        let zero_ah_witness = AhDoubleShuffleWitness::empty(batch_size, alpha, zero_ah_randomness)
            .with_indices(zero_ah_indices)
            .with_values(zero_ah_values);

        (
            assignment_vars,
            assignment_inputs,
            zero_witness,
            zero_ah_witness,
        )
    };

    let double_shuffle = Fl2saDoubleShuffleBundle {
        core: core_double_shuffle_witness,
        ah: ah_double_shuffle_witness,
    };

    Ok(Fl2saR1CSArtifacts {
        num_cons: actual_num_cons,
        num_vars: max_var_idx,
        num_inputs,
        num_non_zero_entries: actual_num_non_zero_entries,
        instance: inst,
        vars: assignment_vars,
        inputs: assignment_inputs,
        double_shuffle,
        metrics,
    })
}

#[allow(non_snake_case)]
pub fn produce_r1cs_fl2sa() -> (
    usize,
    usize,
    usize,
    usize,
    Instance,
    VarsAssignment,
    InputsAssignment,
) {
    let params = FL2SAParams::default();
    let artifacts = produce_r1cs_fl2sa_with_params(&params, None)
        .expect("Failed to generate R1CS with default parameters");
    artifacts_into_tuple(artifacts)
}

fn artifacts_into_tuple(
    artifacts: Fl2saR1CSArtifacts,
) -> (
    usize,
    usize,
    usize,
    usize,
    Instance,
    VarsAssignment,
    InputsAssignment,
) {
    (
        artifacts.num_cons,
        artifacts.num_vars,
        artifacts.num_inputs,
        artifacts.num_non_zero_entries,
        artifacts.instance,
        artifacts.vars,
        artifacts.inputs,
    )
}

pub fn produce_r1cs_fl2sa_custom(
    w: usize,
    e: usize,
    m: usize,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let lgw = (w as f64).log2() as usize;
    let params = FL2SAParams {
        w,
        e,
        m,
        lgw,
        acc: 64,
    };
    produce_r1cs_fl2sa_with_params(&params, None).map(artifacts_into_tuple)
}

pub fn produce_r1cs_fl2sa_with_input(
    p: u64,
    b: u32,
    v: Vec<u64>,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let params = FL2SAParams::default();
    let input_data = FL2SAInputData {
        p,
        b,
        v,
        witness_values: None,
        double_shuffle_randomness: None,
        double_shuffle_seed: None,
        ah_double_shuffle_randomness: None,
        ah_double_shuffle_seed: None,
    };
    produce_r1cs_fl2sa_with_params(&params, Some(&input_data)).map(artifacts_into_tuple)
}

///
///
///
///
/// ```rust
/// let (cons, vars, inputs, non_zero, inst, vars_assign, inputs_assign) =
///     produce_r1cs_fl2sa_single_precision(4)?;
/// ```
///
/// - Alpha: approximately70
/// - Beta: approximately7
pub fn produce_r1cs_fl2sa_single_precision(
    w: usize,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let params = FL2SAParams::single_precision(w);
    produce_r1cs_fl2sa_with_params(&params, None).map(artifacts_into_tuple)
}

///
///
///
///
/// ```rust
/// let (cons, vars, inputs, non_zero, inst, vars_assign, inputs_assign) =
///     produce_r1cs_fl2sa_double_precision(8)?;
/// ```
///
/// - Alpha: approximately525
/// - Beta: approximately15
///
pub fn produce_r1cs_fl2sa_double_precision(
    w: usize,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let params = FL2SAParams::double_precision(w);
    produce_r1cs_fl2sa_with_params(&params, None).map(artifacts_into_tuple)
}

///
///
///
/// ```rust
/// let result = produce_r1cs_fl2sa_single_precision_with_input(
///     4,                      // w=4
///     5,                      // p=5
/// )?;
/// ```
pub fn produce_r1cs_fl2sa_single_precision_with_input(
    w: usize,
    p: u64,
    b: u32,
    v: Vec<u64>,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let params = FL2SAParams::single_precision(w);
    let input_data = FL2SAInputData::with_params(&params, p, b, Some(v))?;
    produce_r1cs_fl2sa_with_params(&params, Some(&input_data)).map(artifacts_into_tuple)
}

///
///
///
/// ```rust
/// let result = produce_r1cs_fl2sa_double_precision_with_input(
///     3,                      // p=3
/// )?;
/// ```
pub fn produce_r1cs_fl2sa_double_precision_with_input(
    w: usize,
    p: u64,
    b: u32,
    v: Vec<u64>,
) -> Result<
    (
        usize,
        usize,
        usize,
        usize,
        Instance,
        VarsAssignment,
        InputsAssignment,
    ),
    String,
> {
    let params = FL2SAParams::double_precision(w);
    let input_data = FL2SAInputData::with_params(&params, p, b, Some(v))?;
    produce_r1cs_fl2sa_with_params(&params, Some(&input_data)).map(artifacts_into_tuple)
}

///
#[allow(non_snake_case)]
pub fn produce_r1cs_fl2sa_with_result<F: Float + Default + std::fmt::Debug>(
    params: &FL2SAParams,
    input_data: &FL2SAInputData,
    precision: FloatPrecision,
) -> Result<(Fl2saR1CSArtifacts, Vec<F>), String> {
    if input_data.v.len() != params.beta() - 1 {
        return Err(format!(
            "Mantissa block number mismatch: expected {}, actual {}",
            params.beta() - 1,
            input_data.v.len()
        ));
    }

    let artifacts = produce_r1cs_fl2sa_with_params(params, Some(input_data))?;

    let alpha = params.alpha();
    let mut y = vec![F::default(); alpha];

    fl2sa_generic(
        &mut y,
        input_data.b,
        &input_data.v,
        input_data.p,
        params.w as u32,
        params.m as u32,
        precision,
    );

    let mut result = Vec::new();
    for i in (0..alpha).rev() {
        result.push(y[i]);
    }

    Ok((artifacts, result))
}

///
///
///
///
///
/// ```rust
/// let (cons, vars, inputs, non_zero, inst, vars_assign, inputs_assign, superaccumulator) =
///     produce_r1cs_fl2sa_with_result_f64(3, 0, vec![0x5, 0x7, 0x2])?;
///
/// for (i, val) in superaccumulator.iter().enumerate() {
///     if *val != 0.0 {
///         println!("  y[{}] = {}", 10-i, val);
///     }
/// }
/// ```
///
///
///
pub fn produce_r1cs_fl2sa_with_result_f64(
    p: u64,
    b: u32,
    v: Vec<u64>,
) -> Result<(Fl2saR1CSArtifacts, Vec<f64>), String> {
    let params = FL2SAParams::default();
    let input_data = FL2SAInputData {
        p,
        b,
        v,
        witness_values: None,
        double_shuffle_randomness: None,
        double_shuffle_seed: None,
        ah_double_shuffle_randomness: None,
        ah_double_shuffle_seed: None,
    };
    produce_r1cs_fl2sa_with_result(&params, &input_data, FloatPrecision::Double)
}

pub fn produce_r1cs_fl2sa_with_result_f32(
    p: u64,
    b: u32,
    v: Vec<u64>,
) -> Result<(Fl2saR1CSArtifacts, Vec<f32>), String> {
    let params = FL2SAParams::default();
    let input_data = FL2SAInputData {
        p,
        b,
        v,
        witness_values: None,
        double_shuffle_randomness: None,
        double_shuffle_seed: None,
        ah_double_shuffle_randomness: None,
        ah_double_shuffle_seed: None,
    };
    produce_r1cs_fl2sa_with_result(&params, &input_data, FloatPrecision::Single)
}

///
pub fn compute_fl2sa_result<F: Float + Default>(
    params: &FL2SAParams,
    p: u64,
    b: u32,
    v: Vec<u64>,
    precision: FloatPrecision,
) -> Result<Vec<F>, String> {
    if v.len() != params.beta() - 1 {
        return Err(format!(
            "Mantissa block number mismatch: expected {}, actual {}",
            params.beta() - 1,
            v.len()
        ));
    }

    let alpha = params.alpha();
    let mut y = vec![F::default(); alpha];

    fl2sa_generic(
        &mut y,
        b,
        &v,
        p,
        params.w as u32,
        params.m as u32,
        precision,
    );

    let mut result = Vec::new();
    for i in (0..alpha).rev() {
        result.push(y[i]);
    }

    Ok(result)
}

pub fn superacc_from_components_i64(
    params: &FL2SAParams,
    p: u64,
    b: u32,
    mantissa_blocks: &[u64],
) -> Result<Vec<i64>, String> {
    let beta_minus_one = params.beta() - 1;
    if mantissa_blocks.len() != beta_minus_one {
        return Err(format!(
            "Mantissa block number mismatch: expected {}, actual {}",
            beta_minus_one,
            mantissa_blocks.len()
        ));
    }

    let alpha = params.alpha();
    let beta = params.beta();
    let w = params.w as u32;
    let m = params.m as u32;

    let p_h = p / w as u64;
    let p_l = p % w as u64;
    let z = if p == 0 { 1 } else { 0 };

    let mut v = mantissa_blocks.to_vec();
    if beta >= 2 {
        let power_exp = m.saturating_sub(w.saturating_mul((beta as u32).saturating_sub(2)));
        let power_2_exp = 1u64 << power_exp;
        if beta >= 2 && beta - 2 < v.len() {
            v[beta - 2] = v[beta - 2].saturating_add(power_2_exp.saturating_mul((1 - z) as u64));
        }
    }

    let mut u = vec![0u128; beta - 1];
    for i in 0..(beta - 1) {
        u[i] = (v[i] as u128) << p_l;
    }

    let mut d = vec![0u128; beta - 1];
    let shift = 1u128 << w;
    for i in 0..(beta - 1) {
        d[i] = u[i] / shift;
    }

    let mut v_prime = vec![0i128; beta];
    for i in (1..(beta - 1)).rev() {
        let shifted = (d[i] as u128) << w;
        let remainder = u[i].saturating_sub(shifted) as i128;
        let carry = d[i - 1] as i128;
        v_prime[i] = remainder.saturating_add(carry);
    }
    v_prime[0] = (u[0].saturating_sub((d[0] as u128) << w)) as i128;
    v_prime[beta - 1] = d[beta - 2] as i128;

    let sign = 1 - 2 * (b as i128);
    for value in &mut v_prime {
        *value = value.saturating_mul(sign);
    }

    let mut superacc = vec![0i128; alpha];
    for i in 0..beta {
        let idx = p_h as usize + i;
        if idx < alpha {
            superacc[idx] = superacc[idx].saturating_add(v_prime[i]);
        }
    }

    let mut result = Vec::with_capacity(alpha);
    for value in superacc {
        if value > i64::MAX as i128 || value < i64::MIN as i128 {
            return Err("Superaccumulator value is outside i64 range".to_string());
        }
        result.push(value as i64);
    }

    Ok(result)
}

// ============================================================================
// ============================================================================

#[cfg(test)]
mod sync_verification {
    use super::*;

    #[test]
    fn verify_code_doc_consistency() {
        let (num_cons, num_vars, num_inputs, num_non_zero_entries, _, _, _) = produce_r1cs_fl2sa();

        assert_eq!(
            num_cons, 136,
            "The number of constraints is inconsistent with the documentation!Current: {}, Documentation: 136",
            num_cons
        );

        assert_eq!(
            num_vars, 171,
            "The number of variables is inconsistent with the documentation!Current: {}, Documentation: 171",
            num_vars
        );

        assert_eq!(
            num_inputs, 1,
            "The input quantity is inconsistent with the document!Current: {}, Documentation: 1",
            num_inputs
        );

        let w = 4;
        let e = 5;
        let m = 10;
        let _lgw = 2;
        let _acc = 64;

        let expected_alpha = ((1 << e) + m + w - 1) / w; // 11
        let expected_beta = ((m + 1) + w - 1) / w + 1; // 4

        assert_eq!(
            expected_alpha, 11,
            "Alpha calculation is inconsistent with documentation!Calculation: {}, Documentation: 11",
            expected_alpha
        );
        assert_eq!(
            expected_beta, 4,
            "Beta calculations are inconsistent with documentation!Calculation: {}, Documents: 4",
            expected_beta
        );

        println!("Code document consistency verification passed!");
        println!("Number of constraints: {} check", num_cons);
        println!("Number of variables: {} check", num_vars);
        println!("Enter quantity: {} check", num_inputs);
        println!("Number of non-zero items: {} check", num_non_zero_entries);
        println!("   Alpha: {} check", expected_alpha);
        println!("   Beta: {} check", expected_beta);
    }

    #[test]
    fn print_current_parameters() {
        let w = 4;
        let e = 5;
        let m = 10;
        let lgw = 2;
        let acc = 64;
        let alpha = ((1 << e) + m + w - 1) / w;
        let beta = ((m + 1) + w - 1) / w + 1;

        println!("\nCurrent system parameters (for checking documentation):");
        println!("w (block width): {}", w);
        println!("e (exponential parameter): {}", e);
        println!("m (maximum number of digits): {}", m);
        println!("   lgw (log2(w)): {}", lgw);
        println!("acc (accuracy constant): {}", acc);
        println!("alpha (accumulator length): {}", alpha);
        println!("beta (number of blocks): {}", beta);

        let (num_cons, num_vars, num_inputs, num_non_zero_entries, _, _, _) = produce_r1cs_fl2sa();
        println!("\nCurrent performance data (for checking documentation):");
        println!("Number of constraints: {}", num_cons);
        println!("Number of variables: {}", num_vars);
        println!("Enter quantity: {}", num_inputs);
        println!("Number of non-zero items: {}", num_non_zero_entries);
    }

    #[test]
    fn double_shuffle_batch_context_handles_two_entries() {
        use curve25519_dalek::scalar::Scalar;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let beta = 4;
        let w = 4;
        let lgw = 2;
        let batch_size = 2;

        let start_idx = 200usize;
        let row_count = beta * batch_size;
        let double_shuffle_var_end = start_idx + 6 * row_count + 6 * w;
        let max_var_idx = double_shuffle_var_end + 16;
        let num_vars = max_var_idx - 2;
        let zero_var_idx = max_var_idx - 1;

        let mut vars = vec![Scalar::ZERO.to_bytes(); max_var_idx];
        vars[zero_var_idx] = Scalar::ZERO.to_bytes();

        let mut indices = DoubleShuffleBatchIndices::new(batch_size, beta, w);
        let mut expected_p_bits = vec![vec![0u8; lgw]; batch_size];
        let mut expected_r_bits = vec![vec![vec![0u8; w]; beta.saturating_sub(1)]; batch_size];

        for batch in 0..batch_size {
            let p_l_base = batch * 20;
            for j in 0..lgw {
                let idx = p_l_base + j;
                let bit = ((batch + j) % 2) as u64;
                indices.p_l_bits[batch][j] = Some(idx);
                vars[idx] = Scalar::from(bit).to_bytes();
                expected_p_bits[batch][j] = bit as u8;
            }

            for row in 0..(beta - 1) {
                let r_base = batch * 50 + row * w;
                for col in 0..w {
                    let idx = r_base + col;
                    let bit = ((batch + row + col) % 2) as u64;
                    indices.r_bits[batch][row][col] = Some(idx);
                    vars[idx] = Scalar::from(bit).to_bytes();
                    expected_r_bits[batch][row][col] = bit as u8;
                }
            }
        }

        let mut rng = StdRng::seed_from_u64(42);
        let randomness = DoubleShuffleRandomness::with_batch(beta, w, batch_size, &mut rng);
        let context = DoubleShuffleBatchContext::new(
            beta,
            w,
            lgw,
            batch_size,
            start_idx,
            randomness,
            indices,
            zero_var_idx,
        );

        let mut a_entries = Vec::new();
        let mut b_entries = Vec::new();
        let mut c_entries = Vec::new();
        let next_constraint = context.append_constraints(
            300,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            num_vars,
        );
        assert!(next_constraint > 300);
        assert!(!a_entries.is_empty());

        let mut vars_for_witness = vars.clone();
        vars_for_witness.resize(max_var_idx, Scalar::ZERO.to_bytes());

        let max_len = vars_for_witness.len();
        let witness = context
            .populate_witness(&mut vars_for_witness, max_len)
            .expect("batch witness generation should succeed");

        assert_eq!(witness.batch_size, batch_size);
        assert_eq!(witness.p_l_values, expected_p_bits);
        assert_eq!(witness.r_bit_values, expected_r_bits);
    }
}

// ============================================================================
// ============================================================================

#[cfg(feature = "manual_examples")]
fn main() {
    println!("FL2SA R1CS restraint system demonstration");
    println!("============================");

    let (num_cons, num_vars, num_inputs, num_non_zero_entries, _inst, _vars, _inputs) =
        produce_r1cs_fl2sa();

    println!("FL2SA R1CS constraint system generation is completed!");
    println!("System parameters:");
    println!("Number of constraints: {}", num_cons);
    println!("Number of variables: {}", num_vars);
    println!("Enter quantity: {}", num_inputs);
    println!("Number of non-zero items: {}", num_non_zero_entries);

    println!("\nPrivacy protection features:");
    println!("Check: One-Hot encoding ah vector hides p_h position information");
    println!("Check: Zero-knowledge proof protects floating-point number processing");
    println!("Check: R1CS constraint verification algorithm correctness");

    println!("\nAlgorithm performance:");
    println!(
        "check {} constraints implement the complete FL2SA algorithm",
        num_cons
    );
    println!("Check: Supports multi-precision floating point numbers (f16/f32/f64)");
    println!("Check: Linear complexity O(beta + w + alpha)");

    println!("\nRun the test to view detailed verification:");
    println!("   cargo test --example fl2sa -- --nocapture");

    // ========================================================================
    // ========================================================================

    println!("\nParameterized API demonstration");
    println!("==================");

    println!("Custom parameter demonstration (w=8, e=4, m=12):");
    match produce_r1cs_fl2sa_custom(8, 4, 12) {
        Ok((cons, vars, inputs, non_zero, _, _, _)) => {
            println!(
                "Successfully generated - Constraints: {}, Variables: {}, Input: {}, Non-zero items: {}",
                cons, vars, inputs, non_zero
            );
        }
        Err(e) => println!("Error: {}", e),
    }

    // ========================================================================
    // ========================================================================

    println!("\nIEEE 754 precision variant demonstration");
    println!("========================");

    println!("Single precision (f32: m=23, e=8) demonstration:");
    match produce_r1cs_fl2sa_single_precision(4) {
        Ok((cons, vars, inputs, non_zero, _, _, _)) => {
            let params = FL2SAParams::single_precision(4);
            let (precision_type, ieee_info) = params.precision_info();
            println!(
                "{} system generated successfully - {}",
                precision_type, ieee_info
            );
            println!(
                "System scale: Constraints: {}, Variables: {}, Input: {}, Non-zero items: {}",
                cons, vars, inputs, non_zero
            );
            println!(
                "Parameters: alpha={}, beta={}, w={}",
                params.alpha(),
                params.beta(),
                params.w
            );
        }
        Err(e) => println!("Error: Single precision system generation failed: {}", e),
    }

    println!("\nDouble precision (f64: m=52, e=11) demonstration:");
    match produce_r1cs_fl2sa_double_precision(4) {
        Ok((cons, vars, inputs, non_zero, _, _, _)) => {
            let params = FL2SAParams::double_precision(4);
            let (precision_type, ieee_info) = params.precision_info();
            println!(
                "{} system generated successfully - {}",
                precision_type, ieee_info
            );
            println!(
                "System scale: Constraints: {}, Variables: {}, Input: {}, Non-zero items: {}",
                cons, vars, inputs, non_zero
            );
            println!(
                "Parameters: alpha={}, beta={}, w={}",
                params.alpha(),
                params.beta(),
                params.w
            );

            let single_params = FL2SAParams::single_precision(4);
            let alpha_ratio = params.alpha() as f64 / single_params.alpha() as f64;
            let beta_ratio = params.beta() as f64 / single_params.beta() as f64;
            println!(
                "vs single precision: alpha growth{:.1}x, beta growth{:.1}x",
                alpha_ratio, beta_ratio
            );
        }
        Err(e) => println!("Error: Double precision system generation failed: {}", e),
    }

    println!("\nPrecision variant comparison table:");
    println!("Accuracy Type w alpha beta Constrained Estimation Variable Estimation");
    println!("   ------------------------------------------------");

    let precisions = [
        ("Single precision (f32)", FL2SAParams::single_precision(4)),
        ("Double precision (f64)", FL2SAParams::double_precision(4)),
    ];

    for (name, params) in &precisions {
        if let Ok(artifacts) = produce_r1cs_fl2sa_with_params(params, None) {
            println!(
                "   {:11}  {:2}   {:6} {:5}     {:8}    {:8}",
                name,
                params.w,
                params.alpha(),
                params.beta(),
                artifacts.num_cons,
                artifacts.num_vars
            );
        }
    }

    println!("\nAccuracy test with input data:");

    println!("Single precision input data test:");
    match produce_r1cs_fl2sa_single_precision_with_input(
        4,
        5,
        0,
        vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC],
    ) {
        Ok((cons, vars, _, _, _, _, _)) => {
            println!(
                "Single precision input test successful: Constraint: {}, Variable: {}",
                cons, vars
            );
        }
        Err(e) => println!("Error: Single precision input test failed: {}", e),
    }

    println!("Double precision input data test:");
    let double_params = FL2SAParams::double_precision(8);
    let beta_minus_one = double_params.beta() - 1;
    let test_v = vec![0x123; beta_minus_one];

    match produce_r1cs_fl2sa_double_precision_with_input(8, 3, 1, test_v) {
        Ok((cons, vars, _, _, _, _, _)) => {
            println!(
                "Double precision input test successful: constraint: {}, variable: {} (w=8)",
                cons, vars
            );
        }
        Err(e) => println!("Error: Double precision input test failed: {}", e),
    }

    println!("\nDetailed input data demonstration:");
    let input_p = 5;
    let input_b = 0;
    let input_v = vec![0x7, 0x4, 0x2];

    match produce_r1cs_fl2sa_with_input(input_p, input_b, input_v) {
        Ok((cons, vars, inputs, non_zero, _, _, _)) => {
            println!(
                "Generated successfully - enter p={}, b={}",
                input_p, input_b
            );
            println!(
                "System size - Constraints: {}, Variables: {}, Input: {}, Non-zero items: {}",
                cons, vars, inputs, non_zero
            );
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\nParameter verification demonstration:");
    let invalid_params = FL2SAParams {
        w: 3,
        e: 5,
        m: 10,
        lgw: 2,
        acc: 64,
    };
    match produce_r1cs_fl2sa_with_params(&invalid_params, None) {
        Ok(_) => println!("Error: unexpected success (expected this case to fail)"),
        Err(e) => println!("Correctly reject invalid parameters: {}", e),
    }

    // ========================================================================
    // ========================================================================

    println!("\nR1CS + Algorithm Result Demonstration");
    println!("=========================");

    println!(
        "Full FL2SA demo - getting R1CS constraints and superaccumulator results simultaneously:"
    );
    let test_p = 3;
    let test_b = 0;
    let test_v = vec![0x5, 0x7, 0x2];

    match produce_r1cs_fl2sa_with_result_f64(test_p, test_b, test_v.clone()) {
        Ok((artifacts, superaccumulator)) => {
            println!("Generated successfully!");
            println!(
                "R1CS system: Constraint: {}, Variable: {}, Input: {}, Non-zero item: {}",
                artifacts.num_cons,
                artifacts.num_vars,
                artifacts.num_inputs,
                artifacts.num_non_zero_entries
            );
            println!(
                "Input parameters: p={}, b={}, v={:?}",
                test_p, test_b, test_v
            );
            println!("Super accumulator result <y[alpha-1], . . . , y0 :");
            for (i, val) in superaccumulator.iter().enumerate() {
                if *val != 0.0 {
                    println!("      y[{}] = {}", 10 - i, val);
                }
            }
            if superaccumulator.iter().all(|&x| x == 0.0) {
                println!("(all positions are 0)");
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\nLightweight superaccumulator calculation demonstration:");
    let params = FL2SAParams::default();
    match compute_fl2sa_result::<f64>(
        &params,
        test_p,
        test_b,
        test_v.clone(),
        FloatPrecision::Double,
    ) {
        Ok(result) => {
            println!("Calculation successful!");
            println!("Super accumulator results <y[10], y[9], . . . , y[1], y[0] :");
            for (i, val) in result.iter().enumerate() {
                if *val != 0.0 {
                    println!("      y[{}] = {}", 10 - i, val);
                }
            }
            if result.iter().all(|&x| x == 0.0) {
                println!("(all positions are 0)");
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\nCustom parameter hyperaccumulator demonstration (w=8, e=4, m=12):");
    let custom_params = FL2SAParams {
        w: 8,
        e: 4,
        m: 12,
        lgw: 3,
        acc: 64,
    };
    let custom_beta = custom_params.beta();

    let custom_input_data =
        match FL2SAInputData::with_params(&custom_params, 2, 0, Some(vec![0x3F, 0x2A])) {
            Ok(data) => data,
            Err(e) => {
                println!("Error: Input data creation failed: {}", e);
                return;
            }
        };

    match compute_fl2sa_result::<f64>(
        &custom_params,
        custom_input_data.p,
        custom_input_data.b,
        custom_input_data.v.clone(),
        FloatPrecision::Double,
    ) {
        Ok(result) => {
            println!("Custom parameters calculated successfully!");
            println!(
                "System parameters: alpha={}, beta={}",
                custom_params.alpha(),
                custom_beta
            );
            println!(
                "Input: p={}, b={}, v={:?}",
                custom_input_data.p, custom_input_data.b, custom_input_data.v
            );
            println!("Superaccumulator result:");
            for (i, val) in result.iter().enumerate() {
                if *val != 0.0 {
                    println!("      y[{}] = {}", custom_params.alpha() - 1 - i, val);
                }
            }
            if result.iter().all(|&x| x == 0.0) {
                println!("(all positions are 0)");
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // ========================================================================
    // ========================================================================

    println!("\nFL2SA algorithm demonstration");
    println!("================");

    let b = 0;
    let v = vec![0x5, 0x3, 0x1];
    let p = 2;

    println!("Input parameters:");
    println!("Sign bit b: {}", b);
    println!("Mantissa block v: {:?}", v);
    println!("Offset p: {}", p);

    let result = fl2sa_example_f64(b, &v, p);
    println!("Superaccumulator output <y[10], y[9], ..., y[1], y[0] :");
    for (i, val) in result.iter().enumerate() {
        if *val != 0.0 {
            println!("   y[{}] = {}", 10 - i, val);
        }
    }

    println!("\nBatch FL2SA algorithm demonstration");
    println!("====================");

    let inputs = vec![
        (0, vec![0x5, 0x3, 0x1], 2),
        (1, vec![0x2, 0x7, 0x4], 1),
        (0, vec![0xA, 0xB, 0xC], 3),
    ];

    println!("Batch input parameters:");
    for (i, (b, v, p)) in inputs.iter().enumerate() {
        println!("Number {}: b={}, v={:?}, p={}", i + 1, b, v, p);
    }

    let batch_result = fl2sa_batch_algorithm::<f64>(&inputs, 4, 11, 10, FloatPrecision::Double);
    println!("Super accumulator after batch accumulation:");
    for (i, val) in batch_result.iter().enumerate() {
        if *val != 0.0 {
            println!("   y[{}] = {}", 10 - i, val);
        }
    }

    println!("\nFL2SA algorithm execution completed!");
}
