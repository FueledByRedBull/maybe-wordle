use serde::{Deserialize, Serialize};

use crate::scoring::PATTERN_SPACE;

pub const SMALL_STATE_TABLE_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallStateTable {
    pub version: u32,
    pub max_size: usize,
    pub min_expected: Vec<f64>,
}

impl SmallStateTable {
    pub fn build(max_size: usize) -> Self {
        let mut min_expected = vec![0.0; max_size + 1];
        if max_size >= 1 {
            min_expected[1] = 1.0;
        }
        for size in 2..=max_size {
            let mut best = f64::INFINITY;
            let mut partition = Vec::with_capacity(size);
            enumerate_partitions(size, size, &mut partition, &min_expected, &mut best);
            min_expected[size] = best;
        }
        Self {
            version: SMALL_STATE_TABLE_VERSION,
            max_size,
            min_expected,
        }
    }

    pub fn lower_bound(&self, size: usize) -> f64 {
        if size == 0 {
            return 0.0;
        }
        self.min_expected
            .get(size)
            .copied()
            .unwrap_or_else(|| 1.0 + depth_like_floor(size))
    }
}

fn enumerate_partitions(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    min_expected: &[f64],
    best: &mut f64,
) {
    if current.len() >= PATTERN_SPACE {
        return;
    }
    if remaining == 0 {
        let total = current.iter().sum::<usize>() as f64;
        if total == 0.0 {
            return;
        }
        let mut candidate = 1.0;
        for part in current.iter().copied() {
            candidate += (part as f64 / total) * min_expected[part];
        }
        *best = best.min(candidate);
        return;
    }

    let upper = remaining.min(max_part);
    for part in (1..=upper).rev() {
        if current.is_empty() && part == remaining {
            continue;
        }
        current.push(part);
        enumerate_partitions(remaining - part, part, current, min_expected, best);
        current.pop();
    }
}

fn depth_like_floor(size: usize) -> f64 {
    if size <= 1 {
        return 1.0;
    }
    let mut depth = 1.0;
    let mut capacity = 1usize;
    while capacity < size {
        capacity = capacity.saturating_mul(PATTERN_SPACE);
        depth += 1.0;
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::SmallStateTable;

    #[test]
    fn lower_bound_is_positive_over_small_sizes() {
        let table = SmallStateTable::build(8);
        for size in 1..=8 {
            assert!(table.min_expected[size] >= 1.0);
        }
    }

    #[test]
    fn singleton_floor_is_exact() {
        let table = SmallStateTable::build(4);
        assert_eq!(table.lower_bound(1), 1.0);
        assert!(table.lower_bound(4) > 1.0);
    }
}
