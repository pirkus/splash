use crate::grid::Grid2;

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

    pub fn from_fn(grid: Grid2, f: impl Fn(usize, usize) -> f32) -> Self {
        let width = grid.width();
        let data = (0..grid.size())
            .map(|i| {
                let x = i % width;
                let y = i / width;
                f(x, y)
            })
            .collect();
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

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let data = self.data.iter().map(|value| f(*value)).collect();
        Self {
            grid: self.grid,
            data,
        }
    }

    pub fn map_with_index(&self, f: impl Fn(usize, usize, f32) -> f32) -> Self {
        let width = self.grid.width();
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let x = i % width;
                let y = i / width;
                f(x, y, *value)
            })
            .collect();
        Self {
            grid: self.grid,
            data,
        }
    }

    pub fn fill_with_index(&mut self, f: impl Fn(usize, usize) -> f32) {
        let width = self.grid.width();
        for (i, value) in self.data.iter_mut().enumerate() {
            let x = i % width;
            let y = i / width;
            *value = f(x, y);
        }
    }

    pub fn scale_in_place(&mut self, scale: f32) {
        for value in &mut self.data {
            *value *= scale;
        }
    }

    pub fn add_scaled_in_place(&mut self, other: &Self, scale: f32) {
        self.assert_same_grid(other);
        for (value, other_value) in self.data.iter_mut().zip(other.data.iter()) {
            *value += other_value * scale;
        }
    }

    pub fn mul_pointwise_into(&mut self, left: &Self, right: &Self) {
        left.assert_same_grid(right);
        self.assert_same_grid(left);
        for ((out, left_value), right_value) in self
            .data
            .iter_mut()
            .zip(left.data.iter())
            .zip(right.data.iter())
        {
            *out = left_value * right_value;
        }
    }

    pub fn zip_with(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        self.assert_same_grid(other);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| f(*a, *b))
            .collect();
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

    pub fn sum_with(&self, f: impl Fn(f32) -> f32) -> f32 {
        self.data.iter().map(|value| f(*value)).sum()
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn abs_sum(&self) -> f32 {
        self.data.iter().map(|value| value.abs()).sum()
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

    pub fn dot(&self, other: &Self) -> f32 {
        self.assert_same_grid(other);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
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
}
