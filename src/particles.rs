use crate::{Field2, MacGrid2, MacVelocity2, StaggeredField2, StaggeredGrid2, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticlePhase {
    Liquid,
    Air,
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleBounds {
    pub min: Vec2,
    pub max: Vec2,
}

impl ParticleBounds {
    pub fn from_grid(grid: MacGrid2, margin: f32) -> Self {
        let margin = margin.max(0.0);
        let width = grid.width() as f32 * grid.dx();
        let height = grid.height() as f32 * grid.dx();
        let max_x = (width - margin).max(margin);
        let max_y = (height - margin).max(margin);
        Self {
            min: Vec2::new(margin, margin),
            max: Vec2::new(max_x, max_y),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParticleSystem {
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    phases: Vec<ParticlePhase>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            phases: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.positions
    }

    pub fn velocities(&self) -> &[Vec2] {
        &self.velocities
    }

    pub fn phases(&self) -> &[ParticlePhase] {
        &self.phases
    }

    pub fn push(&mut self, position: Vec2, velocity: Vec2, phase: ParticlePhase) {
        self.positions.push(position);
        self.velocities.push(velocity);
        self.phases.push(phase);
    }

    pub fn append(&mut self, other: ParticleSystem) {
        self.positions.extend(other.positions);
        self.velocities.extend(other.velocities);
        self.phases.extend(other.phases);
    }

    pub fn add_velocity(&mut self, delta: Vec2) {
        for vel in &mut self.velocities {
            *vel = *vel + delta;
        }
    }

    pub fn seed_pool(
        grid: MacGrid2,
        height_fraction: f32,
        spacing: f32,
        jitter: f32,
        phase: ParticlePhase,
    ) -> Self {
        if spacing <= 0.0 {
            return Self::new();
        }
        let hf = height_fraction.clamp(0.0, 1.0);
        let jitter = jitter.clamp(0.0, 1.0);
        let width = grid.width() as f32 * grid.dx();
        let height = grid.height() as f32 * grid.dx();
        let max_y = height * hf;
        if max_y <= spacing {
            return Self::new();
        }
        let start = spacing * 0.5;
        let end_x = width - spacing * 0.5;
        let end_y = max_y - spacing * 0.5;
        if end_x <= start || end_y <= start {
            return Self::new();
        }
        let nx = ((end_x - start) / spacing).floor() as usize + 1;
        let ny = ((end_y - start) / spacing).floor() as usize + 1;
        let mut system = Self::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let base_x = start + ix as f32 * spacing;
                let base_y = start + iy as f32 * spacing;
                let jx = (rand_unit(ix as u32, iy as u32, 0) - 0.5) * jitter * spacing;
                let jy = (rand_unit(ix as u32, iy as u32, 1) - 0.5) * jitter * spacing;
                let position = Vec2::new(base_x + jx, base_y + jy);
                system.push(position, Vec2::zero(), phase);
            }
        }
        system
    }

    pub fn seed_disk(
        center: Vec2,
        radius: f32,
        spacing: f32,
        jitter: f32,
        phase: ParticlePhase,
    ) -> Self {
        if spacing <= 0.0 || radius <= 0.0 {
            return Self::new();
        }
        let jitter = jitter.clamp(0.0, 1.0);
        let min_x = center.x - radius;
        let max_x = center.x + radius;
        let min_y = center.y - radius;
        let max_y = center.y + radius;
        let start_x = min_x + spacing * 0.5;
        let start_y = min_y + spacing * 0.5;
        if max_x <= start_x || max_y <= start_y {
            return Self::new();
        }
        let nx = ((max_x - start_x) / spacing).floor() as usize + 1;
        let ny = ((max_y - start_y) / spacing).floor() as usize + 1;
        let mut system = Self::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let base_x = start_x + ix as f32 * spacing;
                let base_y = start_y + iy as f32 * spacing;
                let jx = (rand_unit(ix as u32, iy as u32, 7) - 0.5) * jitter * spacing;
                let jy = (rand_unit(ix as u32, iy as u32, 13) - 0.5) * jitter * spacing;
                let position = Vec2::new(base_x + jx, base_y + jy);
                let dx = position.x - center.x;
                let dy = position.y - center.y;
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                system.push(position, Vec2::zero(), phase);
            }
        }
        system
    }

    pub fn to_grid(&self, grid: MacGrid2) -> MacVelocity2 {
        if self.is_empty() {
            return MacVelocity2::new(grid, Vec2::zero());
        }
        let u_grid = grid.u_grid();
        let v_grid = grid.v_grid();
        let mut u_accum = vec![0.0; u_grid.size()];
        let mut v_accum = vec![0.0; v_grid.size()];
        let mut u_weights = vec![0.0; u_grid.size()];
        let mut v_weights = vec![0.0; v_grid.size()];
        for i in 0..self.positions.len() {
            let pos = self.positions[i];
            let vel = self.velocities[i];
            splat_component(&mut u_accum, &mut u_weights, u_grid, pos, vel.x);
            splat_component(&mut v_accum, &mut v_weights, v_grid, pos, vel.y);
        }
        normalize_staggered(&mut u_accum, &u_weights);
        normalize_staggered(&mut v_accum, &v_weights);
        let u_field = StaggeredField2::from_data(u_grid, u_accum);
        let v_field = StaggeredField2::from_data(v_grid, v_accum);
        MacVelocity2::from_components(grid, u_field, v_field)
    }

    pub fn to_density(&self, grid: MacGrid2, scale: f32) -> Field2 {
        self.to_density_with_rest(grid, scale, self.rest_density(grid))
    }

    pub fn to_density_with_rest(
        &self,
        grid: MacGrid2,
        scale: f32,
        rest_density: f32,
    ) -> Field2 {
        let cell_grid = grid.cell_grid();
        let accum = self.density_accum(grid);
        let mut field = Field2::new(cell_grid, 0.0);
        let rest_density = rest_density.max(1e-6);
        let inv = scale / rest_density;
        field.update_with_index(|x, y, _| {
            let idx = cell_grid.idx(x, y);
            (accum[idx] * inv).clamp(0.0, 1.0)
        });
        field
    }

    pub fn update_velocities_from_grid(
        &mut self,
        new_grid: &MacVelocity2,
        prev_grid: Option<&MacVelocity2>,
        flip_ratio: f32,
    ) {
        if self.is_empty() {
            return;
        }
        let flip_ratio = flip_ratio.clamp(0.0, 1.0);
        for i in 0..self.positions.len() {
            let pos = self.positions[i];
            let pic = new_grid.sample_linear((pos.x, pos.y));
            let next = if let Some(prev) = prev_grid {
                let prev_sample = prev.sample_linear((pos.x, pos.y));
                let delta = pic - prev_sample;
                let flip = self.velocities[i] + delta;
                pic.scale(1.0 - flip_ratio) + flip.scale(flip_ratio)
            } else {
                pic
            };
            self.velocities[i] = next;
        }
    }

    pub fn advect_rk2_grid(
        &mut self,
        velocity: &MacVelocity2,
        dt: f32,
        bounds: ParticleBounds,
        bounce: f32,
    ) {
        if dt == 0.0 {
            return;
        }
        let bounce = bounce.clamp(0.0, 1.0);
        for i in 0..self.positions.len() {
            let pos = self.positions[i];
            let v1 = velocity.sample_linear((pos.x, pos.y));
            let mid = pos + v1.scale(0.5 * dt);
            let v2 = velocity.sample_linear((mid.x, mid.y));
            let mut next = pos + v2.scale(dt);
            let mut vel = self.velocities[i];
            if bounce > 0.0 {
                vel = v2;
            }
            apply_bounds(&mut next, &mut vel, bounds, bounce);
            self.positions[i] = next;
            self.velocities[i] = vel;
        }
    }

    pub fn advect_rk3_grid(
        &mut self,
        velocity: &MacVelocity2,
        dt: f32,
        bounds: ParticleBounds,
        bounce: f32,
    ) {
        if dt == 0.0 {
            return;
        }
        let bounce = bounce.clamp(0.0, 1.0);
        for i in 0..self.positions.len() {
            let pos = self.positions[i];
            let v1 = velocity.sample_linear((pos.x, pos.y));
            let p2 = pos + v1.scale(0.5 * dt);
            let v2 = velocity.sample_linear((p2.x, p2.y));
            let p3 = pos + v2.scale(0.75 * dt);
            let v3 = velocity.sample_linear((p3.x, p3.y));
            let mut next = pos + (v1.scale(2.0) + v2.scale(3.0) + v3.scale(4.0)).scale(dt / 9.0);
            let mut vel = self.velocities[i];
            if bounce > 0.0 {
                vel = v3;
            }
            apply_bounds(&mut next, &mut vel, bounds, bounce);
            self.positions[i] = next;
            self.velocities[i] = vel;
        }
    }

    pub fn rest_density(&self, grid: MacGrid2) -> f32 {
        let accum = self.density_accum(grid);
        accum
            .iter()
            .cloned()
            .fold(0.0_f32, f32::max)
            .max(1e-6)
    }

    fn density_accum(&self, grid: MacGrid2) -> Vec<f32> {
        let cell_grid = grid.cell_grid();
        let mut accum = vec![0.0; cell_grid.size()];
        for pos in &self.positions {
            splat_cell(&mut accum, cell_grid, *pos, 1.0);
        }
        accum
    }
}

fn splat_component(
    accum: &mut [f32],
    weights: &mut [f32],
    grid: StaggeredGrid2,
    pos: Vec2,
    value: f32,
) {
    let dx = grid.dx();
    let origin = grid.origin();
    let gx = (pos.x - origin.0) / dx;
    let gy = (pos.y - origin.1) / dx;
    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    for dy in 0..=1 {
        let wy = if dy == 0 { 1.0 - fy } else { fy };
        for dx_i in 0..=1 {
            let wx = if dx_i == 0 { 1.0 - fx } else { fx };
            let ix = x0 + dx_i;
            let iy = y0 + dy;
            if ix < 0 || iy < 0 {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            if ix >= grid.width() || iy >= grid.height() {
                continue;
            }
            let w = wx * wy;
            if w == 0.0 {
                continue;
            }
            let idx = grid.idx(ix, iy);
            accum[idx] += value * w;
            weights[idx] += w;
        }
    }
}

fn normalize_staggered(data: &mut [f32], weights: &[f32]) {
    for (value, weight) in data.iter_mut().zip(weights.iter()) {
        if *weight > 0.0 {
            *value /= *weight;
        }
    }
}

fn splat_cell(accum: &mut [f32], grid: crate::Grid2, pos: Vec2, value: f32) {
    let dx = grid.dx();
    let gx = pos.x / dx - 0.5;
    let gy = pos.y / dx - 0.5;
    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    for dy in 0..=1 {
        let wy = if dy == 0 { 1.0 - fy } else { fy };
        for dx_i in 0..=1 {
            let wx = if dx_i == 0 { 1.0 - fx } else { fx };
            let ix = x0 + dx_i;
            let iy = y0 + dy;
            if ix < 0 || iy < 0 {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            if ix >= grid.width() || iy >= grid.height() {
                continue;
            }
            let w = wx * wy;
            if w == 0.0 {
                continue;
            }
            let idx = grid.idx(ix, iy);
            accum[idx] += value * w;
        }
    }
}

fn apply_bounds(pos: &mut Vec2, vel: &mut Vec2, bounds: ParticleBounds, bounce: f32) {
    if pos.x < bounds.min.x {
        pos.x = bounds.min.x;
        if vel.x < 0.0 {
            vel.x = -vel.x * bounce;
        }
    } else if pos.x > bounds.max.x {
        pos.x = bounds.max.x;
        if vel.x > 0.0 {
            vel.x = -vel.x * bounce;
        }
    }
    if pos.y < bounds.min.y {
        pos.y = bounds.min.y;
        if vel.y < 0.0 {
            vel.y = -vel.y * bounce;
        }
    } else if pos.y > bounds.max.y {
        pos.y = bounds.max.y;
        if vel.y > 0.0 {
            vel.y = -vel.y * bounce;
        }
    }
}

fn rand_unit(ix: u32, iy: u32, salt: u32) -> f32 {
    let seed = ix.wrapping_mul(1664525)
        ^ iy.wrapping_mul(1013904223)
        ^ salt.wrapping_mul(2654435761);
    let hashed = mix_u32(seed);
    (hashed as f32) / (u32::MAX as f32)
}

fn mix_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846ca68b);
    value ^= value >> 16;
    value
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
    fn seed_pool_populates_particles() {
        let grid = MacGrid2::new(16, 16, 1.0);
        let system = ParticleSystem::seed_pool(grid, 0.5, 0.5, 0.2, ParticlePhase::Liquid);
        assert!(!system.is_empty());
        assert_eq!(system.positions.len(), system.velocities.len());
        assert_eq!(system.positions.len(), system.phases.len());
    }

    #[test]
    fn seed_pool_stays_within_domain() {
        let grid = MacGrid2::new(10, 10, 1.0);
        let system = ParticleSystem::seed_pool(grid, 0.6, 0.5, 0.4, ParticlePhase::Liquid);
        let width = grid.width() as f32 * grid.dx();
        let height = grid.height() as f32 * grid.dx() * 0.6;
        for pos in system.positions() {
            assert!(pos.x >= 0.0 && pos.x <= width);
            assert!(pos.y >= 0.0 && pos.y <= height);
        }
    }

    #[test]
    fn seed_pool_is_deterministic() {
        let grid = MacGrid2::new(12, 12, 1.0);
        let a = ParticleSystem::seed_pool(grid, 0.7, 0.5, 0.3, ParticlePhase::Liquid);
        let b = ParticleSystem::seed_pool(grid, 0.7, 0.5, 0.3, ParticlePhase::Liquid);
        assert_eq!(a.positions(), b.positions());
        assert_eq!(a.velocities(), b.velocities());
        assert_eq!(a.phases(), b.phases());
    }

    #[test]
    fn p2g_g2p_pic_round_trip() {
        let grid = MacGrid2::new(16, 16, 1.0);
        let mut system = ParticleSystem::new();
        let pos = Vec2::new(5.25, 6.75);
        let vel = Vec2::new(1.5, -0.6);
        system.push(pos, vel, ParticlePhase::Liquid);
        let grid_vel = system.to_grid(grid);
        let sampled = grid_vel.sample_linear((pos.x, pos.y));
        assert_close(sampled.x, vel.x, 1e-4);
        assert_close(sampled.y, vel.y, 1e-4);
        system.update_velocities_from_grid(&grid_vel, None, 0.0);
        assert_close(system.velocities()[0].x, vel.x, 1e-4);
        assert_close(system.velocities()[0].y, vel.y, 1e-4);
    }

    #[test]
    fn flip_blends_with_pic() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let mut system = ParticleSystem::new();
        let pos = Vec2::new(2.5, 2.5);
        system.push(pos, Vec2::new(0.2, 0.0), ParticlePhase::Liquid);
        let prev_grid = MacVelocity2::new(grid, Vec2::zero());
        let new_grid = MacVelocity2::new(grid, Vec2::new(1.0, 0.0));
        system.update_velocities_from_grid(&new_grid, Some(&prev_grid), 1.0);
        assert_close(system.velocities()[0].x, 1.2, 1e-4);
        system.velocities[0] = Vec2::new(0.2, 0.0);
        system.update_velocities_from_grid(&new_grid, Some(&prev_grid), 0.0);
        assert_close(system.velocities()[0].x, 1.0, 1e-4);
    }

    #[test]
    fn advect_rk2_clamps_to_bounds() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(7.5, 7.5), Vec2::new(1.0, 1.0), ParticlePhase::Liquid);
        let velocity = MacVelocity2::new(grid, Vec2::new(2.0, 2.0));
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        system.advect_rk2_grid(&velocity, 1.0, bounds, 0.0);
        let pos = system.positions()[0];
        assert!(pos.x <= bounds.max.x);
        assert!(pos.y <= bounds.max.y);
    }

    #[test]
    fn advect_rk3_moves_particle() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(2.0, 2.0), Vec2::new(0.0, 0.0), ParticlePhase::Liquid);
        let velocity = MacVelocity2::new(grid, Vec2::new(1.0, 0.0));
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        system.advect_rk3_grid(&velocity, 1.0, bounds, 0.0);
        let pos = system.positions()[0];
        assert!(pos.x > 2.5);
    }
}
