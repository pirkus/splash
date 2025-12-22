#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid2 {
    width: usize,
    height: usize,
    dx: f32,
}

impl Grid2 {
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

    pub fn cell_center(&self, x: usize, y: usize) -> (f32, f32) {
        (
            (x as f32 + 0.5) * self.dx,
            (y as f32 + 0.5) * self.dx,
        )
    }
}
