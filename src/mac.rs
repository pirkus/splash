use crate::{Grid2, Vec2};
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacGrid2 {
    width: usize,
    height: usize,
    dx: f32,
}

impl MacGrid2 {
    pub fn new(width: usize, height: usize, dx: f32) -> Self {
        assert!(width > 0, "width must be > 0");
        assert!(height > 0, "height must be > 0");
        assert!(dx > 0.0, "dx must be > 0");
        Self { width, height, dx }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }

    pub fn cell_grid(&self) -> Grid2 {
        Grid2::new(self.width, self.height, self.dx)
    }

    pub fn u_grid(&self) -> StaggeredGrid2 {
        StaggeredGrid2::new(
            self.width + 1,
            self.height,
            self.dx,
            (0.0, 0.5 * self.dx),
        )
    }

    pub fn v_grid(&self) -> StaggeredGrid2 {
        StaggeredGrid2::new(
            self.width,
            self.height + 1,
            self.dx,
            (0.5 * self.dx, 0.0),
        )
    }

    pub fn cell_center(&self, x: usize, y: usize) -> (f32, f32) {
        (
            (x as f32 + 0.5) * self.dx,
            (y as f32 + 0.5) * self.dx,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaggeredGrid2 {
    width: usize,
    height: usize,
    dx: f32,
    origin: (f32, f32),
}

impl StaggeredGrid2 {
    pub fn new(width: usize, height: usize, dx: f32, origin: (f32, f32)) -> Self {
        assert!(width > 0, "width must be > 0");
        assert!(height > 0, "height must be > 0");
        assert!(dx > 0.0, "dx must be > 0");
        Self {
            width,
            height,
            dx,
            origin,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }

    pub fn origin(&self) -> (f32, f32) {
        self.origin
    }

    pub fn size(&self) -> usize {
        self.width * self.height
    }

    pub fn idx(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y * self.width + x
    }

    pub fn clamp_coord(&self, x: i32, y: i32) -> (usize, usize) {
        let max_x = (self.width - 1) as i32;
        let max_y = (self.height - 1) as i32;
        let cx = x.clamp(0, max_x) as usize;
        let cy = y.clamp(0, max_y) as usize;
        (cx, cy)
    }

    pub fn index_position(&self, x: usize, y: usize) -> (f32, f32) {
        (
            self.origin.0 + x as f32 * self.dx,
            self.origin.1 + y as f32 * self.dx,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StaggeredField2 {
    grid: StaggeredGrid2,
    data: Vec<f32>,
}

impl StaggeredField2 {
    pub fn new(grid: StaggeredGrid2, fill: f32) -> Self {
        let data = vec![fill; grid.size()];
        Self { grid, data }
    }

    pub fn from_data(grid: StaggeredGrid2, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), grid.size(), "staggered field data mismatch");
        Self { grid, data }
    }

    pub fn from_fn(grid: StaggeredGrid2, f: impl Fn(usize, usize) -> f32 + Sync) -> Self {
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

    pub fn grid(&self) -> StaggeredGrid2 {
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
        let origin = self.grid.origin();
        let gx = (pos.0 - origin.0) / dx;
        let gy = (pos.1 - origin.1) / dx;
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
        assert_eq!(self.grid, other.grid, "staggered grid mismatch");
        self.data.clone_from(&other.data);
    }

    pub fn zip_with(&self, other: &Self, f: impl Fn(f32, f32) -> f32 + Sync) -> Self {
        assert_eq!(self.grid, other.grid, "staggered grid mismatch");
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

    pub fn add_scaled(&self, other: &Self, scale: f32) -> Self {
        self.zip_with(other, |a, b| a + b * scale)
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

    pub fn max_abs(&self) -> f32 {
        if should_parallel(self.data.len()) {
            self.data
                .par_iter()
                .map(|value| value.abs())
                .reduce(|| 0.0_f32, f32::max)
        } else {
            self.data
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max)
        }
    }

    pub fn sum_squares(&self) -> f32 {
        if should_parallel(self.data.len()) {
            self.data
                .par_iter()
                .map(|value| value * value)
                .sum()
        } else {
            self.data.iter().map(|value| value * value).sum()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellType {
    Fluid,
    Solid,
    Air,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellFlags {
    grid: Grid2,
    data: Vec<CellType>,
}

impl CellFlags {
    pub fn new(grid: Grid2, fill: CellType) -> Self {
        let data = vec![fill; grid.size()];
        Self { grid, data }
    }

    pub fn from_fn(grid: Grid2, f: impl Fn(usize, usize) -> CellType) -> Self {
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

    pub fn get(&self, x: usize, y: usize) -> CellType {
        self.data[self.grid.idx(x, y)]
    }

    pub fn clone_from(&mut self, other: &Self) {
        assert_eq!(self.grid, other.grid, "cell flag grid mismatch");
        self.data.clone_from(&other.data);
    }

    pub fn fill_with_index(&mut self, f: impl Fn(usize, usize) -> CellType) {
        let width = self.grid.width();
        for (i, value) in self.data.iter_mut().enumerate() {
            let x = i % width;
            let y = i / width;
            *value = f(x, y);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacVelocity2 {
    grid: MacGrid2,
    u: StaggeredField2,
    v: StaggeredField2,
}

impl MacVelocity2 {
    pub fn new(grid: MacGrid2, fill: Vec2) -> Self {
        let u = StaggeredField2::new(grid.u_grid(), fill.x);
        let v = StaggeredField2::new(grid.v_grid(), fill.y);
        Self { grid, u, v }
    }

    pub fn from_components(grid: MacGrid2, u: StaggeredField2, v: StaggeredField2) -> Self {
        assert_eq!(u.grid(), grid.u_grid(), "u grid mismatch");
        assert_eq!(v.grid(), grid.v_grid(), "v grid mismatch");
        Self { grid, u, v }
    }

    pub fn grid(&self) -> MacGrid2 {
        self.grid
    }

    pub fn u(&self) -> &StaggeredField2 {
        &self.u
    }

    pub fn v(&self) -> &StaggeredField2 {
        &self.v
    }

    pub fn u_mut(&mut self) -> &mut StaggeredField2 {
        &mut self.u
    }

    pub fn v_mut(&mut self) -> &mut StaggeredField2 {
        &mut self.v
    }

    pub fn clone_from(&mut self, other: &Self) {
        self.u.clone_from(&other.u);
        self.v.clone_from(&other.v);
    }

    pub fn sample_linear(&self, pos: (f32, f32)) -> Vec2 {
        Vec2::new(self.u.sample_linear(pos), self.v.sample_linear(pos))
    }

    pub fn add_scaled(&self, other: &Self, scale: f32) -> Self {
        Self {
            grid: self.grid,
            u: self.u.add_scaled(&other.u, scale),
            v: self.v.add_scaled(&other.v, scale),
        }
    }

    pub fn max_abs(&self) -> f32 {
        self.u.max_abs().max(self.v.max_abs())
    }

    pub fn energy(&self) -> f32 {
        self.u.sum_squares() + self.v.sum_squares()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    NoSlip,
    Inflow(Vec2),
    Outflow,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryConfig {
    pub left: BoundaryCondition,
    pub right: BoundaryCondition,
    pub bottom: BoundaryCondition,
    pub top: BoundaryCondition,
}

impl BoundaryConfig {
    pub fn no_slip() -> Self {
        Self {
            left: BoundaryCondition::NoSlip,
            right: BoundaryCondition::NoSlip,
            bottom: BoundaryCondition::NoSlip,
            top: BoundaryCondition::NoSlip,
        }
    }
}

pub fn apply_domain_boundaries(velocity: &MacVelocity2, config: BoundaryConfig) -> MacVelocity2 {
    let grid = velocity.grid();
    let w = grid.width();
    let h = grid.height();
    let u_field = velocity.u();
    let v_field = velocity.v();
    let u = u_field.map_with_index(|x, y, value| {
        if x == 0 {
            return match config.left {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => u_field.get(1, y),
            };
        }
        if x == w {
            return match config.right {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => u_field.get(w - 1, y),
            };
        }
        if y == 0 {
            return match config.bottom {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => {
                    if h > 1 {
                        u_field.get(x, 1)
                    } else {
                        value
                    }
                }
            };
        }
        if y + 1 == h {
            return match config.top {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => {
                    if h > 1 {
                        u_field.get(x, h - 2)
                    } else {
                        value
                    }
                }
            };
        }
        value
    });
    let v = v_field.map_with_index(|x, y, value| {
        if y == 0 {
            return match config.bottom {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => v_field.get(x, 1),
            };
        }
        if y == h {
            return match config.top {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => v_field.get(x, h - 1),
            };
        }
        if x == 0 {
            return match config.left {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => {
                    if w > 1 {
                        v_field.get(1, y)
                    } else {
                        value
                    }
                }
            };
        }
        if x + 1 == w {
            return match config.right {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => {
                    if w > 1 {
                        v_field.get(w - 2, y)
                    } else {
                        value
                    }
                }
            };
        }
        value
    });
    MacVelocity2::from_components(grid, u, v)
}

pub fn apply_domain_boundaries_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    config: BoundaryConfig,
) {
    debug_assert_eq!(out.grid(), velocity.grid(), "velocity grid mismatch");
    let grid = velocity.grid();
    let w = grid.width();
    let h = grid.height();
    let u_field = velocity.u();
    let v_field = velocity.v();
    out.u_mut().fill_with_index(|x, y| {
        if x == 0 {
            return match config.left {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => u_field.get(1, y),
            };
        }
        if x == w {
            return match config.right {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => u_field.get(w - 1, y),
            };
        }
        if y == 0 {
            return match config.bottom {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => {
                    if h > 1 {
                        u_field.get(x, 1)
                    } else {
                        u_field.get(x, y)
                    }
                }
            };
        }
        if y + 1 == h {
            return match config.top {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.x,
                BoundaryCondition::Outflow => {
                    if h > 1 {
                        u_field.get(x, h - 2)
                    } else {
                        u_field.get(x, y)
                    }
                }
            };
        }
        u_field.get(x, y)
    });
    out.v_mut().fill_with_index(|x, y| {
        if y == 0 {
            return match config.bottom {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => v_field.get(x, 1),
            };
        }
        if y == h {
            return match config.top {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => v_field.get(x, h - 1),
            };
        }
        if x == 0 {
            return match config.left {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => {
                    if w > 1 {
                        v_field.get(1, y)
                    } else {
                        v_field.get(x, y)
                    }
                }
            };
        }
        if x + 1 == w {
            return match config.right {
                BoundaryCondition::NoSlip => 0.0,
                BoundaryCondition::Inflow(v) => v.y,
                BoundaryCondition::Outflow => {
                    if w > 1 {
                        v_field.get(w - 2, y)
                    } else {
                        v_field.get(x, y)
                    }
                }
            };
        }
        v_field.get(x, y)
    });
}

pub fn apply_solid_boundaries(velocity: &MacVelocity2, flags: &CellFlags) -> MacVelocity2 {
    assert_eq!(flags.grid(), velocity.grid().cell_grid(), "cell grid mismatch");
    let grid = velocity.grid();
    let w = grid.width();
    let h = grid.height();
    let u = velocity.u().map_with_index(|x, y, value| {
        let left_solid = if x == 0 {
            true
        } else {
            flags.get(x - 1, y) == CellType::Solid
        };
        let right_solid = if x == w {
            true
        } else {
            flags.get(x, y) == CellType::Solid
        };
        if left_solid || right_solid {
            0.0
        } else {
            value
        }
    });
    let v = velocity.v().map_with_index(|x, y, value| {
        let bottom_solid = if y == 0 {
            true
        } else {
            flags.get(x, y - 1) == CellType::Solid
        };
        let top_solid = if y == h {
            true
        } else {
            flags.get(x, y) == CellType::Solid
        };
        if bottom_solid || top_solid {
            0.0
        } else {
            value
        }
    });
    MacVelocity2::from_components(grid, u, v)
}

pub fn apply_solid_boundaries_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
) {
    debug_assert_eq!(flags.grid(), velocity.grid().cell_grid(), "cell grid mismatch");
    debug_assert_eq!(out.grid(), velocity.grid(), "velocity grid mismatch");
    let grid = velocity.grid();
    let w = grid.width();
    let h = grid.height();
    let u_field = velocity.u();
    let v_field = velocity.v();
    out.u_mut().fill_with_index(|x, y| {
        let left_solid = if x == 0 {
            true
        } else {
            flags.get(x - 1, y) == CellType::Solid
        };
        let right_solid = if x == w {
            true
        } else {
            flags.get(x, y) == CellType::Solid
        };
        if left_solid || right_solid {
            0.0
        } else {
            u_field.get(x, y)
        }
    });
    out.v_mut().fill_with_index(|x, y| {
        let bottom_solid = if y == 0 {
            true
        } else {
            flags.get(x, y - 1) == CellType::Solid
        };
        let top_solid = if y == h {
            true
        } else {
            flags.get(x, y) == CellType::Solid
        };
        if bottom_solid || top_solid {
            0.0
        } else {
            v_field.get(x, y)
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_grid_sizes() {
        let grid = MacGrid2::new(4, 3, 1.0);
        let u_grid = grid.u_grid();
        let v_grid = grid.v_grid();
        assert_eq!(u_grid.width(), 5);
        assert_eq!(u_grid.height(), 3);
        assert_eq!(v_grid.width(), 4);
        assert_eq!(v_grid.height(), 4);
    }

    #[test]
    fn staggered_field_samples_origin() {
        let grid = StaggeredGrid2::new(2, 2, 1.0, (0.0, 0.0));
        let field = StaggeredField2::from_fn(grid, |x, y| (x + y * 10) as f32);
        assert_eq!(field.sample_linear((0.0, 0.0)), 0.0);
        assert_eq!(field.sample_linear((1.0, 0.0)), 1.0);
    }

    #[test]
    fn apply_no_slip_zeroes_domain_normals() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::new(1.0, 2.0));
        let bounded = apply_domain_boundaries(&velocity, BoundaryConfig::no_slip());
        assert_eq!(bounded.u().get(0, 0), 0.0);
        assert_eq!(bounded.u().get(4, 3), 0.0);
        assert_eq!(bounded.u().get(2, 0), 0.0);
        assert_eq!(bounded.u().get(2, 3), 0.0);
        assert_eq!(bounded.v().get(1, 0), 0.0);
        assert_eq!(bounded.v().get(1, 4), 0.0);
        assert_eq!(bounded.v().get(0, 2), 0.0);
        assert_eq!(bounded.v().get(3, 2), 0.0);
        assert_eq!(bounded.u().get(2, 1), 1.0);
        assert_eq!(bounded.v().get(1, 2), 2.0);
    }

    #[test]
    fn apply_outflow_copies_neighbor() {
        let grid = MacGrid2::new(3, 3, 1.0);
        let u = StaggeredField2::from_fn(grid.u_grid(), |x, _| x as f32);
        let v = StaggeredField2::new(grid.v_grid(), 0.0);
        let velocity = MacVelocity2::from_components(grid, u, v);
        let config = BoundaryConfig {
            left: BoundaryCondition::Outflow,
            right: BoundaryCondition::Outflow,
            bottom: BoundaryCondition::NoSlip,
            top: BoundaryCondition::NoSlip,
        };
        let bounded = apply_domain_boundaries(&velocity, config);
        assert_eq!(bounded.u().get(0, 1), bounded.u().get(1, 1));
        assert_eq!(bounded.u().get(3, 1), bounded.u().get(2, 1));
    }

    #[test]
    fn apply_solid_boundaries_zeroes_adjacent_faces() {
        let grid = MacGrid2::new(3, 3, 1.0);
        let flags = CellFlags::from_fn(grid.cell_grid(), |x, y| {
            if x == 1 && y == 1 {
                CellType::Solid
            } else {
                CellType::Fluid
            }
        });
        let velocity = MacVelocity2::new(grid, Vec2::new(1.0, 1.0));
        let bounded = apply_solid_boundaries(&velocity, &flags);
        assert_eq!(bounded.u().get(1, 1), 0.0);
        assert_eq!(bounded.u().get(2, 1), 0.0);
        assert_eq!(bounded.v().get(1, 1), 0.0);
        assert_eq!(bounded.v().get(1, 2), 0.0);
    }

    #[test]
    fn apply_inflow_sets_tangential_components() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let config = BoundaryConfig {
            left: BoundaryCondition::NoSlip,
            right: BoundaryCondition::NoSlip,
            bottom: BoundaryCondition::NoSlip,
            top: BoundaryCondition::Inflow(Vec2::new(1.0, 0.0)),
        };
        let bounded = apply_domain_boundaries(&velocity, config);
        assert_eq!(bounded.u().get(1, 3), 1.0);
        assert_eq!(bounded.v().get(1, 4), 0.0);
    }
}
