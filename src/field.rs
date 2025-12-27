use crate::grid::Grid2;
use rayon::prelude::*;
use std::sync::OnceLock;

const PAR_THRESHOLD_DEFAULT: usize = 262_144;
const PAR_MIN_WORK_PER_THREAD: usize = 4096;

fn parallel_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("SIM_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(PAR_THRESHOLD_DEFAULT)
    })
}

fn should_parallel(len: usize) -> bool {
    if len < parallel_threshold() {
        return false;
    }
    let threads = rayon::current_num_threads().max(1);
    len / threads >= PAR_MIN_WORK_PER_THREAD
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field2 {
    grid: Grid2,
    data: Vec<f32>,
}

impl Field2 {
    pub fn new(grid: Grid2, fill: f32) -> Self {
        let data = vec![fill; grid.size()];
        Self { grid, data }
    }

    pub fn from_fn(grid: Grid2, f: impl Fn(usize, usize) -> f32 + Sync) -> Self {
        let width = grid.width();
        let mut data = vec![0.0; grid.size()];
        if should_parallel(data.len()) {
            data.par_iter_mut().enumerate().for_each(|(i, value)| {
                let x = i % width;
                let y = i / width;
                *value = f(x, y);
            });
        } else {
            for (i, value) in data.iter_mut().enumerate() {
                let x = i % width;
                let y = i / width;
                *value = f(x, y);
            }
        }
        Self { grid, data }
    }

    pub fn grid(&self) -> Grid2 {
        self.grid
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.grid.idx(x, y)]
    }

    pub fn sample_clamped(&self, x: i32, y: i32) -> f32 {
        let (cx, cy) = self.grid.clamp_coord(x, y);
        self.get(cx, cy)
    }

    pub fn sample_linear(&self, pos: (f32, f32)) -> f32 {
        let dx = self.grid.dx();
        let gx = pos.0 / dx - 0.5;
        let gy = pos.1 / dx - 0.5;
        let x0 = gx.floor() as i32;
        let y0 = gy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let sx = gx - x0 as f32;
        let sy = gy - y0 as f32;
        let v00 = self.sample_clamped(x0, y0);
        let v10 = self.sample_clamped(x1, y0);
        let v01 = self.sample_clamped(x0, y1);
        let v11 = self.sample_clamped(x1, y1);
        let vx0 = v00 + (v10 - v00) * sx;
        let vx1 = v01 + (v11 - v01) * sx;
        vx0 + (vx1 - vx0) * sy
    }

    pub fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Self {
        let mut data = vec![0.0; self.data.len()];
        if should_parallel(data.len()) {
            data.par_iter_mut().enumerate().for_each(|(i, value)| {
                *value = f(self.data[i]);
            });
        } else {
            for (value, src) in data.iter_mut().zip(self.data.iter()) {
                *value = f(*src);
            }
        }
        Self {
            grid: self.grid,
            data,
        }
    }

    pub fn map_with_index(&self, f: impl Fn(usize, usize, f32) -> f32 + Sync) -> Self {
        let width = self.grid.width();
        let mut data = vec![0.0; self.data.len()];
        if should_parallel(data.len()) {
            data.par_iter_mut().enumerate().for_each(|(i, value)| {
                let x = i % width;
                let y = i / width;
                *value = f(x, y, self.data[i]);
            });
        } else {
            for (i, value) in data.iter_mut().enumerate() {
                let x = i % width;
                let y = i / width;
                *value = f(x, y, self.data[i]);
            }
        }
        Self {
            grid: self.grid,
            data,
        }
    }

    pub fn fill_with_index(&mut self, f: impl Fn(usize, usize) -> f32 + Sync) {
        let width = self.grid.width();
        if should_parallel(self.data.len()) {
            self.data.par_iter_mut().enumerate().for_each(|(i, value)| {
                let x = i % width;
                let y = i / width;
                *value = f(x, y);
            });
        } else {
            for (i, value) in self.data.iter_mut().enumerate() {
                let x = i % width;
                let y = i / width;
                *value = f(x, y);
            }
        }
    }

    pub fn fill_with_index_and_dot(
        &mut self,
        dot_with: &Self,
        f: impl Fn(usize, usize) -> f32 + Sync,
    ) -> f32 {
        self.assert_same_grid(dot_with);
        let width = self.grid.width();
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .enumerate()
                .map(|(i, value)| {
                    let x = i % width;
                    let y = i / width;
                    let next = f(x, y);
                    *value = next;
                    next * dot_with.data[i]
                })
                .sum()
        } else {
            let mut dot = 0.0;
            for (i, value) in self.data.iter_mut().enumerate() {
                let x = i % width;
                let y = i / width;
                let next = f(x, y);
                *value = next;
                dot += next * dot_with.data[i];
            }
            dot
        }
    }

    pub fn update_with_index(&mut self, f: impl Fn(usize, usize, f32) -> f32 + Sync) {
        let width = self.grid.width();
        if should_parallel(self.data.len()) {
            self.data.par_iter_mut().enumerate().for_each(|(i, value)| {
                let x = i % width;
                let y = i / width;
                *value = f(x, y, *value);
            });
        } else {
            for (i, value) in self.data.iter_mut().enumerate() {
                let x = i % width;
                let y = i / width;
                *value = f(x, y, *value);
            }
        }
    }

    pub fn clone_from(&mut self, other: &Self) {
        self.assert_same_grid(other);
        self.data.clone_from(&other.data);
    }

    pub fn scale_in_place(&mut self, scale: f32) {
        if should_parallel(self.data.len()) {
            self.data.par_iter_mut().for_each(|value| {
                *value *= scale;
            });
        } else {
            for value in &mut self.data {
                *value *= scale;
            }
        }
    }

    pub fn add_scaled_in_place(&mut self, other: &Self, scale: f32) {
        self.assert_same_grid(other);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .zip(other.data.par_iter())
                .for_each(|(value, other_value)| {
                    *value += *other_value * scale;
                });
        } else {
            for (value, other_value) in self.data.iter_mut().zip(other.data.iter()) {
                *value += other_value * scale;
            }
        }
    }

    pub fn add_scaled_in_place_and_sum_sq(&mut self, other: &Self, scale: f32) -> f32 {
        self.assert_same_grid(other);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .zip(other.data.par_iter())
                .map(|(value, other_value)| {
                    *value += *other_value * scale;
                    *value * *value
                })
                .sum()
        } else {
            let mut sum = 0.0;
            for (value, other_value) in self.data.iter_mut().zip(other.data.iter()) {
                *value += other_value * scale;
                sum += *value * *value;
            }
            sum
        }
    }

    pub fn scale_and_add_in_place(&mut self, scale: f32, other: &Self) {
        self.assert_same_grid(other);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .zip(other.data.par_iter())
                .for_each(|(value, other_value)| {
                    *value = *value * scale + *other_value;
                });
        } else {
            for (value, other_value) in self.data.iter_mut().zip(other.data.iter()) {
                *value = *value * scale + other_value;
            }
        }
    }

    pub fn mul_pointwise_into(&mut self, left: &Self, right: &Self) {
        left.assert_same_grid(right);
        self.assert_same_grid(left);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .zip(left.data.par_iter())
                .zip(right.data.par_iter())
                .for_each(|((out, left_value), right_value)| {
                    *out = *left_value * *right_value;
                });
        } else {
            for ((out, left_value), right_value) in self
                .data
                .iter_mut()
                .zip(left.data.iter())
                .zip(right.data.iter())
            {
                *out = left_value * right_value;
            }
        }
    }

    pub fn mul_pointwise_into_and_dot_left(&mut self, left: &Self, right: &Self) -> f32 {
        left.assert_same_grid(right);
        self.assert_same_grid(left);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter_mut()
                .zip(left.data.par_iter())
                .zip(right.data.par_iter())
                .map(|((out, left_value), right_value)| {
                    *out = *left_value * *right_value;
                    *left_value * *out
                })
                .sum()
        } else {
            let mut dot = 0.0;
            for ((out, left_value), right_value) in self
                .data
                .iter_mut()
                .zip(left.data.iter())
                .zip(right.data.iter())
            {
                *out = left_value * right_value;
                dot += left_value * *out;
            }
            dot
        }
    }

    pub fn zip_with(&self, other: &Self, f: impl Fn(f32, f32) -> f32 + Sync) -> Self {
        self.assert_same_grid(other);
        let mut data = vec![0.0; self.data.len()];
        if should_parallel(data.len()) {
            data.par_iter_mut()
                .zip(self.data.par_iter())
                .zip(other.data.par_iter())
                .for_each(|((out, left), right)| {
                    *out = f(*left, *right);
                });
        } else {
            for ((out, left), right) in data
                .iter_mut()
                .zip(self.data.iter())
                .zip(other.data.iter())
            {
                *out = f(*left, *right);
            }
        }
        Self {
            grid: self.grid,
            data,
        }
    }

    pub fn laplacian(&self) -> Self {
        let dx2 = self.grid.dx() * self.grid.dx();
        self.map_with_index(|x, y, center| {
            let left = self.sample_clamped(x as i32 - 1, y as i32);
            let right = self.sample_clamped(x as i32 + 1, y as i32);
            let down = self.sample_clamped(x as i32, y as i32 - 1);
            let up = self.sample_clamped(x as i32, y as i32 + 1);
            (left + right + up + down - 4.0 * center) / dx2
        })
    }

    pub fn add_scaled(&self, other: &Self, scale: f32) -> Self {
        self.zip_with(other, |a, b| a + b * scale)
    }

    pub fn sum_with(&self, f: impl Fn(f32) -> f32 + Sync) -> f32 {
        if should_parallel(self.data.len()) {
            self.data.par_iter().map(|value| f(*value)).sum()
        } else {
            self.data.iter().map(|value| f(*value)).sum()
        }
    }

    pub fn sum(&self) -> f32 {
        if should_parallel(self.data.len()) {
            self.data.par_iter().sum()
        } else {
            self.data.iter().sum()
        }
    }

    pub fn abs_sum(&self) -> f32 {
        if should_parallel(self.data.len()) {
            self.data.par_iter().map(|value| value.abs()).sum()
        } else {
            self.data.iter().map(|value| value.abs()).sum()
        }
    }

    pub fn min_max(&self) -> (f32, f32) {
        let mut iter = self.data.iter().filter(|value| value.is_finite());
        let Some(first) = iter.next() else {
            return (0.0, 0.0);
        };
        let mut min_value = *first;
        let mut max_value = *first;
        for value in iter {
            if *value < min_value {
                min_value = *value;
            }
            if *value > max_value {
                max_value = *value;
            }
        }
        (min_value, max_value)
    }

    pub fn stats(&self) -> (f32, f32, f32, usize) {
        let mut sum = 0.0;
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        let mut non_finite = 0;
        for value in &self.data {
            if value.is_finite() {
                sum += value;
                if *value < min_value {
                    min_value = *value;
                }
                if *value > max_value {
                    max_value = *value;
                }
            } else {
                non_finite += 1;
            }
        }
        if min_value == f32::INFINITY {
            min_value = 0.0;
        }
        if max_value == f32::NEG_INFINITY {
            max_value = 0.0;
        }
        (sum, min_value, max_value, non_finite)
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.assert_same_grid(other);
        if should_parallel(self.data.len()) {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(a, b)| a * b)
                .sum()
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .sum()
        }
    }

    fn assert_same_grid(&self, other: &Self) {
        assert_eq!(self.grid, other.grid, "field grid mismatch");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} to be within {tol} of {b}"
        );
    }

    #[test]
    fn laplacian_constant_is_zero() {
        let grid = Grid2::new(4, 4, 1.0);
        let field = Field2::new(grid, 2.0);
        let lap = field.laplacian();
        for value in lap.data.iter() {
            assert_close(*value, 0.0, 1e-6);
        }
    }

    #[test]
    fn from_fn_maps_coords() {
        let grid = Grid2::new(3, 2, 1.0);
        let field = Field2::from_fn(grid, |x, y| (x + y * 10) as f32);
        assert_close(field.get(2, 1), 12.0, 1e-6);
    }

    #[test]
    fn sample_linear_matches_cell_center() {
        let grid = Grid2::new(2, 2, 1.0);
        let field = Field2::from_fn(grid, |x, y| (x + y * 2) as f32);
        let pos = grid.cell_center(1, 0);
        assert_close(field.sample_linear(pos), 1.0, 1e-6);
    }

    #[test]
    fn add_scaled_in_place_matches_add_scaled() {
        let grid = Grid2::new(3, 2, 1.0);
        let a = Field2::from_fn(grid, |x, y| (x + y * 3) as f32);
        let b = Field2::from_fn(grid, |x, y| (x * 2 + y) as f32);
        let mut in_place = a.clone();
        in_place.add_scaled_in_place(&b, 0.25);
        let expected = a.add_scaled(&b, 0.25);
        assert_eq!(in_place, expected);
    }

    #[test]
    fn mul_pointwise_into_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let left = Field2::from_fn(grid, |x, _y| (x + 1) as f32);
        let right = Field2::from_fn(grid, |_x, y| (y + 2) as f32);
        let mut out = Field2::new(grid, 0.0);
        out.mul_pointwise_into(&left, &right);
        assert_close(out.get(0, 0), 2.0, 1e-6);
        assert_close(out.get(1, 1), 6.0, 1e-6);
    }

    #[test]
    fn min_max_reports_bounds() {
        let grid = Grid2::new(2, 2, 1.0);
        let field = Field2::from_fn(grid, |x, y| (x + y * 2) as f32 - 1.0);
        let (min_value, max_value) = field.min_max();
        assert_close(min_value, -1.0, 1e-6);
        assert_close(max_value, 2.0, 1e-6);
    }

    #[test]
    fn sum_with_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let field = Field2::from_fn(grid, |x, y| (x + y) as f32);
        let sum = field.sum_with(|value| value * 2.0);
        assert_close(sum, 8.0, 1e-6);
    }

    #[test]
    fn stats_reports_min_max_sum_and_non_finite() {
        let grid = Grid2::new(2, 2, 1.0);
        let field = Field2::from_fn(grid, |x, y| (x + y) as f32);
        let (sum, min_value, max_value, non_finite) = field.stats();
        assert_close(sum, 4.0, 1e-6);
        assert_close(min_value, 0.0, 1e-6);
        assert_close(max_value, 2.0, 1e-6);
        assert_eq!(non_finite, 0);
    }

    #[test]
    fn fill_with_index_and_dot_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let dot_with = Field2::from_fn(grid, |x, y| (x + 1 + y * 2) as f32);
        let mut out = Field2::new(grid, 0.0);
        let dot = out.fill_with_index_and_dot(&dot_with, |x, y| (x + y) as f32);
        let manual = out
            .data
            .iter()
            .zip(dot_with.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_close(dot, manual, 1e-6);
    }

    #[test]
    fn add_scaled_in_place_and_sum_sq_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let base = Field2::from_fn(grid, |x, y| (x + y * 2) as f32);
        let delta = Field2::from_fn(grid, |x, y| (x * 2 + y) as f32);
        let mut out = base.clone();
        let sum_sq = out.add_scaled_in_place_and_sum_sq(&delta, 0.5);
        let manual: f32 = out.data.iter().map(|value| value * value).sum();
        assert_close(sum_sq, manual, 1e-6);
    }

    #[test]
    fn scale_and_add_in_place_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let base = Field2::from_fn(grid, |x, y| (x + y) as f32);
        let add = Field2::from_fn(grid, |x, y| (x + 2 * y) as f32);
        let mut out = base.clone();
        out.scale_and_add_in_place(0.25, &add);
        let expected = base
            .data
            .iter()
            .zip(add.data.iter())
            .map(|(a, b)| *a * 0.25 + *b)
            .collect::<Vec<_>>();
        assert_eq!(out.data, expected);
    }

    #[test]
    fn mul_pointwise_into_and_dot_left_matches_manual() {
        let grid = Grid2::new(2, 2, 1.0);
        let left = Field2::from_fn(grid, |x, y| (x + y * 2) as f32);
        let right = Field2::from_fn(grid, |x, y| (x * 3 + y + 1) as f32);
        let mut out = Field2::new(grid, 0.0);
        let dot = out.mul_pointwise_into_and_dot_left(&left, &right);
        let manual: f32 = left
            .data
            .iter()
            .zip(out.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_close(dot, manual, 1e-6);
    }

}
