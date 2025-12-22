use crate::{Field2, Grid2, Vec2};

#[derive(Clone, Debug, PartialEq)]
pub struct VecField2 {
    u: Field2,
    v: Field2,
}

impl VecField2 {
    pub fn from_fn(grid: Grid2, f: impl Fn(usize, usize) -> Vec2) -> Self {
        let u = Field2::from_fn(grid, |x, y| f(x, y).x);
        let v = Field2::from_fn(grid, |x, y| f(x, y).y);
        Self { u, v }
    }

    pub fn get(&self, x: usize, y: usize) -> Vec2 {
        Vec2::new(self.u.get(x, y), self.v.get(x, y))
    }

    pub fn u(&self) -> &Field2 {
        &self.u
    }

    pub fn v(&self) -> &Field2 {
        &self.v
    }

    pub fn sample_linear(&self, pos: (f32, f32)) -> Vec2 {
        Vec2::new(self.u.sample_linear(pos), self.v.sample_linear(pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_fn_samples_components() {
        let grid = Grid2::new(3, 2, 1.0);
        let field = VecField2::from_fn(grid, |x, y| Vec2::new(x as f32, y as f32));
        let v = field.get(2, 1);
        assert_eq!(v, Vec2::new(2.0, 1.0));
    }
}
