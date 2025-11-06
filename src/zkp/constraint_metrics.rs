use std::cmp::max;

///
#[derive(Debug, Clone, Copy, Default)]
pub struct R1csShapeMetrics {
    pub field_adds: usize,
    pub field_muls: usize,
    pub nnz_a: usize,
    pub nnz_b: usize,
    pub nnz_c: usize,
}

impl R1csShapeMetrics {
    pub fn accumulate(&mut self, other: &R1csShapeMetrics) {
        self.field_adds = self.field_adds.saturating_add(other.field_adds);
        self.field_muls = self.field_muls.saturating_add(other.field_muls);
        self.nnz_a = self.nnz_a.saturating_add(other.nnz_a);
        self.nnz_b = self.nnz_b.saturating_add(other.nnz_b);
        self.nnz_c = self.nnz_c.saturating_add(other.nnz_c);
    }

    pub fn total_nnz(&self) -> usize {
        self.nnz_a
            .saturating_add(self.nnz_b)
            .saturating_add(self.nnz_c)
    }

    pub fn is_empty(&self) -> bool {
        self.field_adds == 0 && self.field_muls == 0
    }
}

///
pub fn compute_r1cs_metrics(
    num_constraints: usize,
    entries_a: &[(usize, usize, [u8; 32])],
    entries_b: &[(usize, usize, [u8; 32])],
    entries_c: &[(usize, usize, [u8; 32])],
) -> R1csShapeMetrics {
    if num_constraints == 0 {
        return R1csShapeMetrics::default();
    }

    let mut a_counts = vec![0usize; num_constraints];
    let mut b_counts = vec![0usize; num_constraints];
    let mut c_counts = vec![0usize; num_constraints];

    for (row, _, _) in entries_a {
        if *row < num_constraints {
            a_counts[*row] = a_counts[*row].saturating_add(1);
        }
    }
    for (row, _, _) in entries_b {
        if *row < num_constraints {
            b_counts[*row] = b_counts[*row].saturating_add(1);
        }
    }
    for (row, _, _) in entries_c {
        if *row < num_constraints {
            c_counts[*row] = c_counts[*row].saturating_add(1);
        }
    }

    let mut field_adds = 0usize;
    let mut field_muls = 0usize;

    for row in 0..num_constraints {
        let a = a_counts[row];
        let b = b_counts[row];
        let c = c_counts[row];

        if a > 0 {
            field_muls = field_muls.saturating_add(a);
            field_adds = field_adds.saturating_add(a.saturating_sub(1));
        }
        if b > 0 {
            field_muls = field_muls.saturating_add(b);
            field_adds = field_adds.saturating_add(b.saturating_sub(1));
        }
        if c > 0 {
            field_muls = field_muls.saturating_add(c);
            field_adds = field_adds.saturating_add(c.saturating_sub(1));
        }

        if a > 0 && b > 0 {
            field_muls = field_muls.saturating_add(1);
        }

        if max(a, b) > 0 || c > 0 {
            field_adds = field_adds.saturating_add(1);
        }
    }

    R1csShapeMetrics {
        field_adds,
        field_muls,
        nnz_a: entries_a.len(),
        nnz_b: entries_b.len(),
        nnz_c: entries_c.len(),
    }
}
