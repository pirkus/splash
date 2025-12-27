use crate::{Field2, MacGrid2, MacVelocity2, StaggeredField2, StaggeredGrid2, Vec2};
use rayon::prelude::*;
use std::sync::OnceLock;

const PAR_THRESHOLD_DEFAULT: usize = 32_768;
const PAR_MIN_WORK_PER_THREAD: usize = 2048;

fn particle_parallel_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("SIM_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(PAR_THRESHOLD_DEFAULT)
    })
}

fn particle_should_parallel(len: usize) -> bool {
    if len < particle_parallel_threshold() {
        return false;
    }
    let threads = rayon::current_num_threads().max(1);
    len / threads >= PAR_MIN_WORK_PER_THREAD
}

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
    affines: Vec<[Vec2; 2]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GpuParticle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
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
            affines: Vec::with_capacity(capacity),
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

    pub(crate) fn pack_gpu(&self) -> Vec<GpuParticle> {
        self.positions
            .iter()
            .zip(self.velocities.iter())
            .map(|(pos, vel)| GpuParticle {
                pos: [pos.x, pos.y],
                vel: [vel.x, vel.y],
            })
            .collect()
    }

    pub(crate) fn unpack_gpu(&mut self, data: &[GpuParticle]) -> bool {
        if data.len() != self.positions.len() {
            return false;
        }
        for (i, particle) in data.iter().enumerate() {
            self.positions[i] = Vec2::new(particle.pos[0], particle.pos[1]);
            self.velocities[i] = Vec2::new(particle.vel[0], particle.vel[1]);
        }
        true
    }

    pub fn max_speed(&self) -> f32 {
        let mut max_sq = 0.0_f32;
        for vel in &self.velocities {
            if !vel.x.is_finite() || !vel.y.is_finite() {
                return f32::INFINITY;
            }
            let sq = vel.x * vel.x + vel.y * vel.y;
            if sq > max_sq {
                max_sq = sq;
            }
        }
        max_sq.sqrt()
    }

    pub fn clamp_speeds(&mut self, max_speed: f32) -> usize {
        if max_speed <= 0.0 || !max_speed.is_finite() {
            return 0;
        }
        let max_sq = max_speed * max_speed;
        let mut clamped = 0usize;
        for vel in &mut self.velocities {
            if !vel.x.is_finite() || !vel.y.is_finite() {
                *vel = Vec2::zero();
                clamped += 1;
                continue;
            }
            let sq = vel.x * vel.x + vel.y * vel.y;
            if sq > max_sq {
                let scale = max_sq.sqrt() / sq.sqrt();
                *vel = vel.scale(scale);
                clamped += 1;
            }
        }
        clamped
    }

    pub fn xsph_viscosity(&mut self, bounds: ParticleBounds, radius: f32, c: f32) {
        if self.is_empty() || radius <= 0.0 || c <= 0.0 {
            return;
        }
        let c = c.clamp(0.0, 1.0);
        let (bins, w, h, cell, width, height) = build_bins(&self.positions, bounds, radius);
        if bins.is_empty() {
            return;
        }
        let h2 = radius * radius;
        let n = self.positions.len();
        let do_parallel = particle_should_parallel(n);
        let mut updates = vec![Vec2::zero(); n];
        if do_parallel {
            let positions = &self.positions;
            let velocities = &self.velocities;
            updates
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, update)| {
                    let pos = positions[i];
                    let vel = velocities[i];
                    let (bx, by) = bin_coord(pos, bounds, cell, width, height);
                    let mut sum = Vec2::zero();
                    let mut sum_w = 0.0;
                    for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                        for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                            let idx = ny as usize * w + nx as usize;
                            for &j in &bins[idx] {
                                if j == i {
                                    continue;
                                }
                                let dp = pos - positions[j];
                                let r2 = dp.x * dp.x + dp.y * dp.y;
                                if r2 >= h2 {
                                    continue;
                                }
                                let w_ij = pbf_kernel_w(r2, h2);
                                let dv = velocities[j] - vel;
                                sum = sum + dv.scale(w_ij);
                                sum_w += w_ij;
                            }
                        }
                    }
                    if sum_w > 0.0 {
                        *update = sum.scale(1.0 / sum_w);
                    }
                });
            self.velocities
                .par_iter_mut()
                .zip(updates.par_iter())
                .for_each(|(vel, update)| {
                    *vel = *vel + update.scale(c);
                });
        } else {
            for i in 0..n {
                let pos = self.positions[i];
                let vel = self.velocities[i];
                let (bx, by) = bin_coord(pos, bounds, cell, width, height);
                let mut sum = Vec2::zero();
                let mut sum_w = 0.0;
                for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                    for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                        let idx = ny as usize * w + nx as usize;
                        for &j in &bins[idx] {
                            if j == i {
                                continue;
                            }
                            let dp = pos - self.positions[j];
                            let r2 = dp.x * dp.x + dp.y * dp.y;
                            if r2 >= h2 {
                                continue;
                            }
                            let w_ij = pbf_kernel_w(r2, h2);
                            let dv = self.velocities[j] - vel;
                            sum = sum + dv.scale(w_ij);
                            sum_w += w_ij;
                        }
                    }
                }
                if sum_w > 0.0 {
                    updates[i] = sum.scale(1.0 / sum_w);
                }
            }
            for i in 0..n {
                self.velocities[i] = self.velocities[i] + updates[i].scale(c);
            }
        }
    }

    pub fn apply_air_drag(&mut self, occ: &Field2, threshold: f32, drag: f32, dt: f32) {
        if self.is_empty() || drag <= 0.0 || dt <= 0.0 {
            return;
        }
        let threshold = threshold.max(0.0);
        let scale = (1.0 - drag * dt).clamp(0.0, 1.0);
        if scale >= 1.0 {
            return;
        }
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    let occ_val = occ.sample_linear((pos.x, pos.y));
                    if occ_val < threshold {
                        *vel = vel.scale(scale);
                    }
                });
        } else {
            for i in 0..self.positions.len() {
                let pos = self.positions[i];
                let occ_val = occ.sample_linear((pos.x, pos.y));
                if occ_val < threshold {
                    self.velocities[i] = self.velocities[i].scale(scale);
                }
            }
        }
    }

    pub fn phases(&self) -> &[ParticlePhase] {
        &self.phases
    }

    pub fn push(&mut self, position: Vec2, velocity: Vec2, phase: ParticlePhase) {
        self.positions.push(position);
        self.velocities.push(velocity);
        self.phases.push(phase);
        self.affines.push([Vec2::zero(), Vec2::zero()]);
    }

    pub fn append(&mut self, other: ParticleSystem) {
        self.positions.extend(other.positions);
        self.velocities.extend(other.velocities);
        self.phases.extend(other.phases);
        self.affines.extend(other.affines);
    }

    pub fn swap_remove(&mut self, index: usize) -> bool {
        if index >= self.positions.len() {
            return false;
        }
        self.positions.swap_remove(index);
        self.velocities.swap_remove(index);
        self.phases.swap_remove(index);
        self.affines.swap_remove(index);
        true
    }

    pub fn set_velocity_field(&mut self, f: impl Fn(Vec2) -> Vec2 + Sync) {
        if self.is_empty() {
            return;
        }
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    *vel = f(*pos);
                });
        } else {
            for i in 0..self.positions.len() {
                let pos = self.positions[i];
                self.velocities[i] = f(pos);
            }
        }
    }

    pub fn resample_volume(
        &mut self,
        grid: MacGrid2,
        bounds: ParticleBounds,
        occ: &Field2,
        velocity: &MacVelocity2,
        target_count: usize,
        spawn_min: f32,
        low: f32,
        high: f32,
        max_change: usize,
        jitter: f32,
        salt: u32,
    ) -> (usize, usize) {
        let current = self.positions.len();
        if target_count == 0
            || max_change == 0
            || current == target_count
            || !low.is_finite()
            || !high.is_finite()
        {
            return (0, 0);
        }
        let low = low.max(spawn_min.max(0.0));
        let high = high.max(low);
        let cell_grid = grid.cell_grid();
        let mut removed = 0usize;
        let mut added = 0usize;
        if current > target_count {
            let remove_count = (current - target_count).min(max_change);
            let mut candidates: Vec<(f32, usize)> = Vec::new();
            for (i, pos) in self.positions.iter().enumerate() {
                let cx = (pos.x / cell_grid.dx()).floor() as i32;
                let cy = (pos.y / cell_grid.dx()).floor() as i32;
                let (gx, gy) = cell_grid.clamp_coord(cx, cy);
                let occ_val = occ.get(gx, gy);
                if occ_val > high {
                    candidates.push((occ_val, i));
                }
            }
            if !candidates.is_empty() {
                candidates.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut indices: Vec<usize> = candidates
                    .into_iter()
                    .take(remove_count.min(current))
                    .map(|(_, idx)| idx)
                    .collect();
                indices.sort_unstable();
                indices.dedup();
                for idx in indices.into_iter().rev() {
                    if self.swap_remove(idx) {
                        removed += 1;
                    }
                }
            }
        } else if target_count > current {
            let add_count = (target_count - current).min(max_change);
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for y in 0..cell_grid.height() {
                for x in 0..cell_grid.width() {
                    let occ_val = occ.get(x, y);
                    if occ_val >= spawn_min && occ_val < low {
                        candidates.push((x, y));
                    }
                }
            }
            if !candidates.is_empty() {
                let jitter = jitter.max(0.0);
                for i in 0..add_count {
                    let seed = mix_u32(salt.wrapping_add(i as u32));
                    let cell_idx = (seed as usize) % candidates.len();
                    let (x, y) = candidates[cell_idx];
                    let (cx, cy) = cell_grid.cell_center(x, y);
                    let jx = (rand_unit(x as u32, y as u32, salt ^ seed) - 0.5) * jitter;
                    let jy = (rand_unit(y as u32, x as u32, salt ^ seed.wrapping_add(17)) - 0.5)
                        * jitter;
                    let mut pos = Vec2::new(cx + jx, cy + jy);
                    let mut vel = velocity.sample_linear((pos.x, pos.y));
                    apply_bounds(&mut pos, &mut vel, bounds, 0.0);
                    self.push(pos, vel, ParticlePhase::Liquid);
                    added += 1;
                }
            }
        }
        (added, removed)
    }

    pub fn apply_circle_obstacle(&mut self, center: Vec2, radius: f32, bounce: f32) {
        if self.is_empty() || radius <= 0.0 {
            return;
        }
        let radius = radius.max(1e-6);
        let radius_sq = radius * radius;
        let bounce = bounce.clamp(0.0, 1.0);
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter_mut()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    let dx = pos.x - center.x;
                    let dy = pos.y - center.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq >= radius_sq {
                        return;
                    }
                    let (nx, ny) = if dist_sq <= 1e-12 {
                        (1.0, 0.0)
                    } else {
                        let dist = dist_sq.sqrt();
                        (dx / dist, dy / dist)
                    };
                    pos.x = center.x + nx * radius;
                    pos.y = center.y + ny * radius;
                    let vn = vel.x * nx + vel.y * ny;
                    if vn < 0.0 {
                        let impulse = (1.0 + bounce) * vn;
                        vel.x -= impulse * nx;
                        vel.y -= impulse * ny;
                    }
                });
        } else {
            for i in 0..self.positions.len() {
                let dx = self.positions[i].x - center.x;
                let dy = self.positions[i].y - center.y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq >= radius_sq {
                    continue;
                }
                let (nx, ny) = if dist_sq <= 1e-12 {
                    (1.0, 0.0)
                } else {
                    let dist = dist_sq.sqrt();
                    (dx / dist, dy / dist)
                };
                self.positions[i].x = center.x + nx * radius;
                self.positions[i].y = center.y + ny * radius;
                let vn = self.velocities[i].x * nx + self.velocities[i].y * ny;
                if vn < 0.0 {
                    let impulse = (1.0 + bounce) * vn;
                    self.velocities[i].x -= impulse * nx;
                    self.velocities[i].y -= impulse * ny;
                }
            }
        }
    }

    pub fn add_velocity(&mut self, delta: Vec2) {
        if particle_should_parallel(self.velocities.len()) {
            self.velocities.par_iter_mut().for_each(|vel| {
                *vel = *vel + delta;
            });
        } else {
            for vel in &mut self.velocities {
                *vel = *vel + delta;
            }
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

    pub fn seed_rect(
        min: Vec2,
        max: Vec2,
        spacing: f32,
        jitter: f32,
        phase: ParticlePhase,
    ) -> Self {
        if spacing <= 0.0 {
            return Self::new();
        }
        let jitter = jitter.clamp(0.0, 1.0);
        let min_x = min.x.min(max.x);
        let max_x = min.x.max(max.x);
        let min_y = min.y.min(max.y);
        let max_y = min.y.max(max.y);
        if max_x - min_x <= spacing || max_y - min_y <= spacing {
            return Self::new();
        }
        let start_x = min_x + spacing * 0.5;
        let start_y = min_y + spacing * 0.5;
        let end_x = max_x - spacing * 0.5;
        let end_y = max_y - spacing * 0.5;
        if end_x <= start_x || end_y <= start_y {
            return Self::new();
        }
        let nx = ((end_x - start_x) / spacing).floor() as usize + 1;
        let ny = ((end_y - start_y) / spacing).floor() as usize + 1;
        let mut system = Self::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let base_x = start_x + ix as f32 * spacing;
                let base_y = start_y + iy as f32 * spacing;
                let jx = (rand_unit(ix as u32, iy as u32, 11) - 0.5) * jitter * spacing;
                let jy = (rand_unit(ix as u32, iy as u32, 12) - 0.5) * jitter * spacing;
                let position = Vec2::new(base_x + jx, base_y + jy);
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

    pub fn to_grid_apic(&self, grid: MacGrid2) -> MacVelocity2 {
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
            let affine = self.affines.get(i).copied().unwrap_or([Vec2::zero(), Vec2::zero()]);
            for_each_staggered_neighbor(u_grid, pos, |ix, iy, w, node| {
                let d = node - pos;
                let u_val = vel.x + affine[0].x * d.x + affine[0].y * d.y;
                let idx = u_grid.idx(ix, iy);
                u_accum[idx] += u_val * w;
                u_weights[idx] += w;
            });
            for_each_staggered_neighbor(v_grid, pos, |ix, iy, w, node| {
                let d = node - pos;
                let v_val = vel.y + affine[1].x * d.x + affine[1].y * d.y;
                let idx = v_grid.idx(ix, iy);
                v_accum[idx] += v_val * w;
                v_weights[idx] += w;
            });
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

    pub fn to_level_set(
        &self,
        grid: MacGrid2,
        bounds: ParticleBounds,
        particle_radius: f32,
        max_distance: f32,
    ) -> Field2 {
        let cell_grid = grid.cell_grid();
        let particle_radius = particle_radius.max(1e-6);
        let mut max_distance = max_distance.max(particle_radius);
        max_distance = max_distance.max(cell_grid.dx());
        if self.is_empty() {
            return Field2::new(cell_grid, max_distance);
        }
        let cell_size = max_distance;
        let width = ((bounds.max.x - bounds.min.x) / cell_size).floor() as i32 + 1;
        let height = ((bounds.max.y - bounds.min.y) / cell_size).floor() as i32 + 1;
        if width <= 0 || height <= 0 {
            return Field2::new(cell_grid, max_distance);
        }
        let w = width as usize;
        let h = height as usize;
        let mut bins = vec![Vec::new(); w * h];
        for (i, pos) in self.positions.iter().enumerate() {
            let mut bx = ((pos.x - bounds.min.x) / cell_size).floor() as i32;
            let mut by = ((pos.y - bounds.min.y) / cell_size).floor() as i32;
            if bx < 0 || by < 0 || bx >= width || by >= height {
                bx = bx.clamp(0, width - 1);
                by = by.clamp(0, height - 1);
            }
            bins[by as usize * w + bx as usize].push(i);
        }
        let search_range = (max_distance / cell_size).ceil() as i32;
        let max_distance_sq = max_distance * max_distance;
        Field2::from_fn(cell_grid, |x, y| {
            let (px, py) = cell_grid.cell_center(x, y);
            let bx = ((px - bounds.min.x) / cell_size).floor() as i32;
            let by = ((py - bounds.min.y) / cell_size).floor() as i32;
            let mut min_d2 = max_distance_sq;
            let min_y = (by - search_range).max(0);
            let max_y = (by + search_range).min(height - 1);
            let min_x = (bx - search_range).max(0);
            let max_x = (bx + search_range).min(width - 1);
            for ny in min_y..=max_y {
                for nx in min_x..=max_x {
                    let idx = ny as usize * w + nx as usize;
                    for &pi in &bins[idx] {
                        let pos = self.positions[pi];
                        let dx = px - pos.x;
                        let dy = py - pos.y;
                        let d2 = dx * dx + dy * dy;
                        if d2 < min_d2 {
                            min_d2 = d2;
                        }
                    }
                }
            }
            if min_d2 >= max_distance_sq {
                max_distance
            } else {
                min_d2.sqrt() - particle_radius
            }
        })
    }

    pub fn to_sph_density_with_rest(
        &self,
        grid: MacGrid2,
        radius: f32,
        scale: f32,
        rest_density: f32,
    ) -> Field2 {
        let cell_grid = grid.cell_grid();
        let accum = self.sph_density_accum(grid, radius);
        let mut field = Field2::new(cell_grid, 0.0);
        let rest_density = rest_density.max(1e-6);
        let scale = scale.max(0.0);
        let inv = scale / rest_density;
        field.update_with_index(|x, y, _| {
            let idx = cell_grid.idx(x, y);
            (accum[idx] * inv).clamp(0.0, 1.0)
        });
        field
    }

    pub fn rest_sph_density(&self, grid: MacGrid2, radius: f32) -> f32 {
        let accum = self.sph_density_accum(grid, radius);
        accum
            .iter()
            .cloned()
            .fold(0.0_f32, f32::max)
            .max(1e-6)
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
        if particle_should_parallel(self.positions.len()) {
            let prev_grid = prev_grid;
            self.positions
                .par_iter()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    let pic = new_grid.sample_linear((pos.x, pos.y));
                    let next = if let Some(prev) = prev_grid {
                        let prev_sample = prev.sample_linear((pos.x, pos.y));
                        let delta = pic - prev_sample;
                        let flip = *vel + delta;
                        pic.scale(1.0 - flip_ratio) + flip.scale(flip_ratio)
                    } else {
                        pic
                    };
                    *vel = next;
                });
        } else {
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
    }

    pub fn update_velocities_from_grid_apic(
        &mut self,
        new_grid: &MacVelocity2,
        prev_grid: Option<&MacVelocity2>,
        flip_ratio: f32,
    ) {
        self.update_velocities_from_grid(new_grid, prev_grid, flip_ratio);
        self.update_affines_from_grid(new_grid);
    }

    pub fn update_affines_from_grid(&mut self, grid: &MacVelocity2) {
        if self.is_empty() {
            return;
        }
        if self.affines.len() != self.positions.len() {
            self.affines.resize(self.positions.len(), [Vec2::zero(), Vec2::zero()]);
        }
        let u_field = grid.u();
        let v_field = grid.v();
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter()
                .zip(self.affines.par_iter_mut())
                .for_each(|(pos, affine)| {
                    let row0 = apic_affine_row(u_field, *pos);
                    let row1 = apic_affine_row(v_field, *pos);
                    *affine = [row0, row1];
                });
        } else {
            for i in 0..self.positions.len() {
                let pos = self.positions[i];
                let row0 = apic_affine_row(u_field, pos);
                let row1 = apic_affine_row(v_field, pos);
                self.affines[i] = [row0, row1];
            }
        }
    }

    pub fn clamp_affines(&mut self, max_norm: f32) {
        if self.is_empty() || max_norm <= 0.0 || !max_norm.is_finite() {
            return;
        }
        let max_sq = max_norm * max_norm;
        if particle_should_parallel(self.affines.len()) {
            self.affines.par_iter_mut().for_each(|rows| {
                for row in rows.iter_mut() {
                    if !row.x.is_finite() || !row.y.is_finite() {
                        *row = Vec2::zero();
                        continue;
                    }
                    let sq = row.x * row.x + row.y * row.y;
                    if sq > max_sq {
                        let scale = max_norm / sq.sqrt();
                        *row = row.scale(scale);
                    }
                }
            });
        } else {
            for rows in &mut self.affines {
                for row in rows.iter_mut() {
                    if !row.x.is_finite() || !row.y.is_finite() {
                        *row = Vec2::zero();
                        continue;
                    }
                    let sq = row.x * row.x + row.y * row.y;
                    if sq > max_sq {
                        let scale = max_norm / sq.sqrt();
                        *row = row.scale(scale);
                    }
                }
            }
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
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter_mut()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    let cur = *pos;
                    let v1 = velocity.sample_linear((cur.x, cur.y));
                    let mid = cur + v1.scale(0.5 * dt);
                    let v2 = velocity.sample_linear((mid.x, mid.y));
                    let mut next = cur + v2.scale(dt);
                    let mut next_vel = *vel;
                    if bounce > 0.0 {
                        next_vel = v2;
                    }
                    apply_bounds(&mut next, &mut next_vel, bounds, bounce);
                    *pos = next;
                    *vel = next_vel;
                });
        } else {
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
        if particle_should_parallel(self.positions.len()) {
            self.positions
                .par_iter_mut()
                .zip(self.velocities.par_iter_mut())
                .for_each(|(pos, vel)| {
                    let cur = *pos;
                    let v1 = velocity.sample_linear((cur.x, cur.y));
                    let p2 = cur + v1.scale(0.5 * dt);
                    let v2 = velocity.sample_linear((p2.x, p2.y));
                    let p3 = cur + v2.scale(0.75 * dt);
                    let v3 = velocity.sample_linear((p3.x, p3.y));
                    let mut next =
                        cur + (v1.scale(2.0) + v2.scale(3.0) + v3.scale(4.0)).scale(dt / 9.0);
                    let mut next_vel = *vel;
                    if bounce > 0.0 {
                        next_vel = v3;
                    }
                    apply_bounds(&mut next, &mut next_vel, bounds, bounce);
                    *pos = next;
                    *vel = next_vel;
                });
        } else {
            for i in 0..self.positions.len() {
                let pos = self.positions[i];
                let v1 = velocity.sample_linear((pos.x, pos.y));
                let p2 = pos + v1.scale(0.5 * dt);
                let v2 = velocity.sample_linear((p2.x, p2.y));
                let p3 = pos + v2.scale(0.75 * dt);
                let v3 = velocity.sample_linear((p3.x, p3.y));
                let mut next =
                    pos + (v1.scale(2.0) + v2.scale(3.0) + v3.scale(4.0)).scale(dt / 9.0);
                let mut vel = self.velocities[i];
                if bounce > 0.0 {
                    vel = v3;
                }
                apply_bounds(&mut next, &mut vel, bounds, bounce);
                self.positions[i] = next;
                self.velocities[i] = vel;
            }
        }
    }

    pub fn separate(&mut self, radius: f32, bounds: ParticleBounds, iterations: usize) {
        if iterations == 0 || radius <= 0.0 || self.positions.len() < 2 {
            return;
        }
        let radius = radius.max(1e-6);
        let cell = radius;
        let width = ((bounds.max.x - bounds.min.x) / cell).floor() as i32 + 1;
        let height = ((bounds.max.y - bounds.min.y) / cell).floor() as i32 + 1;
        if width <= 0 || height <= 0 {
            return;
        }
        let w = width as usize;
        let h = height as usize;
        let radius_sq = radius * radius;
        for _ in 0..iterations {
            let mut bins = vec![Vec::new(); w * h];
            for (i, pos) in self.positions.iter().enumerate() {
                let bx = ((pos.x - bounds.min.x) / cell).floor() as i32;
                let by = ((pos.y - bounds.min.y) / cell).floor() as i32;
                let bx = bx.clamp(0, width - 1) as usize;
                let by = by.clamp(0, height - 1) as usize;
                bins[by * w + bx].push(i);
            }
            let mut delta = vec![Vec2::zero(); self.positions.len()];
            for by in 0..h {
                for bx in 0..w {
                    let bin_idx = by * w + bx;
                    for &i in &bins[bin_idx] {
                        let pi = self.positions[i];
                        let min_y = by.saturating_sub(1);
                        let max_y = (by + 1).min(h - 1);
                        let min_x = bx.saturating_sub(1);
                        let max_x = (bx + 1).min(w - 1);
                        for ny in min_y..=max_y {
                            for nx in min_x..=max_x {
                                let n_idx = ny * w + nx;
                                for &j in &bins[n_idx] {
                                    if j <= i {
                                        continue;
                                    }
                                    let pj = self.positions[j];
                                    let dx = pi.x - pj.x;
                                    let dy = pi.y - pj.y;
                                    let dist_sq = dx * dx + dy * dy;
                                    if dist_sq >= radius_sq || dist_sq <= 1e-12 {
                                        continue;
                                    }
                                    let dist = dist_sq.sqrt();
                                    let overlap = (radius - dist) / dist;
                                    let correction = Vec2::new(dx, dy).scale(0.5 * overlap);
                                    delta[i] = delta[i] + correction;
                                    delta[j] = delta[j] - correction;
                                }
                            }
                        }
                    }
                }
            }
            for i in 0..self.positions.len() {
                let mut pos = self.positions[i] + delta[i];
                let mut vel = self.velocities[i];
                apply_bounds(&mut pos, &mut vel, bounds, 0.0);
                self.positions[i] = pos;
                self.velocities[i] = vel;
            }
        }
    }

    pub fn pbf_rest_density(&self, bounds: ParticleBounds, radius: f32) -> f32 {
        if self.is_empty() || radius <= 0.0 {
            return 1.0;
        }
        let (bins, w, h, cell, width, height) = build_bins(&self.positions, bounds, radius);
        if bins.is_empty() {
            return 1.0;
        }
        let h2 = radius * radius;
        let mut total = 0.0;
        for pos in &self.positions {
            let (bx, by) = bin_coord(*pos, bounds, cell, width, height);
            let mut density = 0.0;
            for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                    let idx = ny as usize * w + nx as usize;
                    for &j in &bins[idx] {
                        let dp = *pos - self.positions[j];
                        let r2 = dp.x * dp.x + dp.y * dp.y;
                        if r2 >= h2 {
                            continue;
                        }
                        density += pbf_kernel_w(r2, h2);
                    }
                }
            }
            total += density;
        }
        let avg = total / self.positions.len() as f32;
        avg.max(1e-6)
    }

    pub fn pbf_project(
        &mut self,
        bounds: ParticleBounds,
        radius: f32,
        rest_density: f32,
        iterations: usize,
        dt: f32,
        bounce: f32,
        scorr_k: f32,
        scorr_n: f32,
        scorr_q: f32,
        prev_positions: &[Vec2],
    ) {
        if iterations == 0
            || radius <= 0.0
            || dt <= 0.0
            || self.positions.len() < 2
            || prev_positions.len() != self.positions.len()
        {
            return;
        }
        let rest_density = rest_density.max(1e-6);
        let radius = radius.max(1e-6);
        let h2 = radius * radius;
        let inv_rest = 1.0 / rest_density;
        let scorr_k = scorr_k.max(0.0);
        let scorr_n = scorr_n.max(0.0);
        let scorr_q = scorr_q.clamp(0.0, 1.0);
        let scorr_q2 = (scorr_q * radius) * (scorr_q * radius);
        let eps = 1e-6;
        let prev_positions = prev_positions;
        let n = self.positions.len();
        let do_parallel = particle_should_parallel(n);
        let mut densities = vec![0.0; n];
        let mut lambdas = vec![0.0; n];
        let mut deltas = vec![Vec2::zero(); n];
        for _ in 0..iterations {
            let (bins, w, h, cell, width, height) = build_bins(&self.positions, bounds, radius);
            if bins.is_empty() {
                break;
            }
            if do_parallel {
                let positions = &self.positions;
                densities
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, density)| {
                        let pos = positions[i];
                        let (bx, by) = bin_coord(pos, bounds, cell, width, height);
                        let mut accum = 0.0;
                        for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                            for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                                let idx = ny as usize * w + nx as usize;
                                for &j in &bins[idx] {
                                    let dp = pos - positions[j];
                                    let r2 = dp.x * dp.x + dp.y * dp.y;
                                    if r2 >= h2 {
                                        continue;
                                    }
                                    accum += pbf_kernel_w(r2, h2);
                                }
                            }
                        }
                        *density = accum;
                    });
                lambdas
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, lambda)| {
                        let pos = positions[i];
                        let (bx, by) = bin_coord(pos, bounds, cell, width, height);
                        let mut grad_i = Vec2::zero();
                        let mut sum_grad_sq = 0.0;
                        for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                            for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                                let idx = ny as usize * w + nx as usize;
                                for &j in &bins[idx] {
                                    if j == i {
                                        continue;
                                    }
                                    let dp = pos - positions[j];
                                    let r2 = dp.x * dp.x + dp.y * dp.y;
                                    if r2 >= h2 {
                                        continue;
                                    }
                                    let grad = pbf_kernel_grad(dp, r2, h2);
                                    sum_grad_sq += grad.x * grad.x + grad.y * grad.y;
                                    grad_i = grad_i + grad;
                                }
                            }
                        }
                        sum_grad_sq += grad_i.x * grad_i.x + grad_i.y * grad_i.y;
                        let c = densities[i] * inv_rest - 1.0;
                        *lambda = if sum_grad_sq > 0.0 {
                            -c / (sum_grad_sq + eps)
                        } else {
                            0.0
                        };
                    });
                deltas
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, delta_out)| {
                        let pos = positions[i];
                        let (bx, by) = bin_coord(pos, bounds, cell, width, height);
                        let mut delta = Vec2::zero();
                        for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                            for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                                let idx = ny as usize * w + nx as usize;
                                for &j in &bins[idx] {
                                    if j == i {
                                        continue;
                                    }
                                    let dp = pos - positions[j];
                                    let r2 = dp.x * dp.x + dp.y * dp.y;
                                    if r2 >= h2 {
                                        continue;
                                    }
                                    let grad = pbf_kernel_grad(dp, r2, h2);
                                    let scorr = if scorr_k > 0.0 {
                                        pbf_scorr(r2, h2, scorr_q2, scorr_k, scorr_n)
                                    } else {
                                        0.0
                                    };
                                    let scale = lambdas[i] + lambdas[j] + scorr;
                                    delta = delta + grad.scale(scale);
                                }
                            }
                        }
                        *delta_out = delta.scale(inv_rest);
                    });
                self.positions
                    .par_iter_mut()
                    .zip(self.velocities.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (pos, vel))| {
                        let mut next = *pos + deltas[i];
                        let mut next_vel = *vel;
                        apply_bounds(&mut next, &mut next_vel, bounds, 0.0);
                        *pos = next;
                        *vel = next_vel;
                    });
            } else {
                for (i, pos) in self.positions.iter().enumerate() {
                    let (bx, by) = bin_coord(*pos, bounds, cell, width, height);
                    let mut density = 0.0;
                    for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                        for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                            let idx = ny as usize * w + nx as usize;
                            for &j in &bins[idx] {
                                let dp = *pos - self.positions[j];
                                let r2 = dp.x * dp.x + dp.y * dp.y;
                                if r2 >= h2 {
                                    continue;
                                }
                                density += pbf_kernel_w(r2, h2);
                            }
                        }
                    }
                    densities[i] = density;
                }
                for (i, pos) in self.positions.iter().enumerate() {
                    let (bx, by) = bin_coord(*pos, bounds, cell, width, height);
                    let mut grad_i = Vec2::zero();
                    let mut sum_grad_sq = 0.0;
                    for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                        for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                            let idx = ny as usize * w + nx as usize;
                            for &j in &bins[idx] {
                                if j == i {
                                    continue;
                                }
                                let dp = *pos - self.positions[j];
                                let r2 = dp.x * dp.x + dp.y * dp.y;
                                if r2 >= h2 {
                                    continue;
                                }
                                let grad = pbf_kernel_grad(dp, r2, h2);
                                sum_grad_sq += grad.x * grad.x + grad.y * grad.y;
                                grad_i = grad_i + grad;
                            }
                        }
                    }
                    sum_grad_sq += grad_i.x * grad_i.x + grad_i.y * grad_i.y;
                    let c = densities[i] * inv_rest - 1.0;
                    lambdas[i] = if sum_grad_sq > 0.0 {
                        -c / (sum_grad_sq + eps)
                    } else {
                        0.0
                    };
                }
                for (i, pos) in self.positions.iter().enumerate() {
                    let (bx, by) = bin_coord(*pos, bounds, cell, width, height);
                    let mut delta = Vec2::zero();
                    for ny in (by - 1).max(0)..=(by + 1).min(h as i32 - 1) {
                        for nx in (bx - 1).max(0)..=(bx + 1).min(w as i32 - 1) {
                            let idx = ny as usize * w + nx as usize;
                            for &j in &bins[idx] {
                                if j == i {
                                    continue;
                                }
                                let dp = *pos - self.positions[j];
                                let r2 = dp.x * dp.x + dp.y * dp.y;
                                if r2 >= h2 {
                                    continue;
                                }
                                let grad = pbf_kernel_grad(dp, r2, h2);
                                let scorr = if scorr_k > 0.0 {
                                    pbf_scorr(r2, h2, scorr_q2, scorr_k, scorr_n)
                                } else {
                                    0.0
                                };
                                let scale = lambdas[i] + lambdas[j] + scorr;
                                delta = delta + grad.scale(scale);
                            }
                        }
                    }
                    deltas[i] = delta.scale(inv_rest);
                }
                for i in 0..n {
                    let mut pos = self.positions[i] + deltas[i];
                    let mut vel = self.velocities[i];
                    apply_bounds(&mut pos, &mut vel, bounds, 0.0);
                    self.positions[i] = pos;
                    self.velocities[i] = vel;
                }
            }
        }
        let inv_dt = 1.0 / dt;
        if do_parallel {
            self.positions
                .par_iter_mut()
                .zip(self.velocities.par_iter_mut())
                .enumerate()
                .for_each(|(i, (pos, vel))| {
                    let mut next = *pos;
                    let mut next_vel = (next - prev_positions[i]).scale(inv_dt);
                    apply_bounds(&mut next, &mut next_vel, bounds, bounce);
                    *pos = next;
                    *vel = next_vel;
                });
        } else {
            for i in 0..n {
                let mut pos = self.positions[i];
                let mut vel = (pos - prev_positions[i]).scale(inv_dt);
                apply_bounds(&mut pos, &mut vel, bounds, bounce);
                self.positions[i] = pos;
                self.velocities[i] = vel;
            }
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

    fn sph_density_accum(&self, grid: MacGrid2, radius: f32) -> Vec<f32> {
        let cell_grid = grid.cell_grid();
        let mut accum = vec![0.0; cell_grid.size()];
        if self.is_empty() {
            return accum;
        }
        let radius = radius.max(1e-6);
        let radius_sq = radius * radius;
        let inv_radius_sq = 1.0 / radius_sq;
        let dx = cell_grid.dx();
        let width = cell_grid.width() as i32;
        let height = cell_grid.height() as i32;
        let kernel_cells = (radius / dx).ceil() as i32;
        let stride = cell_grid.width();
        for pos in &self.positions {
            let cx = (pos.x / dx - 0.5).floor() as i32;
            let cy = (pos.y / dx - 0.5).floor() as i32;
            let min_x = (cx - kernel_cells).max(0);
            let max_x = (cx + kernel_cells).min(width - 1);
            let min_y = (cy - kernel_cells).max(0);
            let max_y = (cy + kernel_cells).min(height - 1);
            for y in min_y..=max_y {
                let py = (y as f32 + 0.5) * dx;
                let dy = py - pos.y;
                for x in min_x..=max_x {
                    let px = (x as f32 + 0.5) * dx;
                    let dxp = px - pos.x;
                    let r2 = dxp * dxp + dy * dy;
                    if r2 >= radius_sq {
                        continue;
                    }
                    let t = 1.0 - r2 * inv_radius_sq;
                    let w = t * t * t;
                    let idx = y as usize * stride + x as usize;
                    accum[idx] += w;
                }
            }
        }
        accum
    }
}

fn pbf_kernel_w(r2: f32, h2: f32) -> f32 {
    let t = 1.0 - r2 / h2;
    t * t * t
}

fn pbf_kernel_grad(dp: Vec2, r2: f32, h2: f32) -> Vec2 {
    if r2 <= 1e-12 {
        return Vec2::zero();
    }
    let t = 1.0 - r2 / h2;
    let scale = -6.0 * t * t / h2;
    dp.scale(scale)
}

fn pbf_scorr(r2: f32, h2: f32, q2: f32, k: f32, n: f32) -> f32 {
    if q2 <= 0.0 {
        return 0.0;
    }
    let w = pbf_kernel_w(r2, h2);
    let wq = pbf_kernel_w(q2, h2);
    if wq <= 0.0 {
        return 0.0;
    }
    let ratio = (w / wq).clamp(0.0, 1.0);
    -k * ratio.powf(n)
}

fn build_bins(
    positions: &[Vec2],
    bounds: ParticleBounds,
    cell: f32,
) -> (Vec<Vec<usize>>, usize, usize, f32, i32, i32) {
    let cell = cell.max(1e-6);
    let width = ((bounds.max.x - bounds.min.x) / cell).floor() as i32 + 1;
    let height = ((bounds.max.y - bounds.min.y) / cell).floor() as i32 + 1;
    if width <= 0 || height <= 0 {
        return (Vec::new(), 0, 0, cell, width, height);
    }
    let w = width as usize;
    let h = height as usize;
    let mut bins = vec![Vec::new(); w * h];
    for (i, pos) in positions.iter().enumerate() {
        let (bx, by) = bin_coord(*pos, bounds, cell, width, height);
        bins[by as usize * w + bx as usize].push(i);
    }
    (bins, w, h, cell, width, height)
}

fn bin_coord(pos: Vec2, bounds: ParticleBounds, cell: f32, width: i32, height: i32) -> (i32, i32) {
    let mut bx = ((pos.x - bounds.min.x) / cell).floor() as i32;
    let mut by = ((pos.y - bounds.min.y) / cell).floor() as i32;
    if bx < 0 || by < 0 || bx >= width || by >= height {
        bx = bx.clamp(0, width - 1);
        by = by.clamp(0, height - 1);
    }
    (bx, by)
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

fn for_each_staggered_neighbor(
    grid: StaggeredGrid2,
    pos: Vec2,
    mut f: impl FnMut(usize, usize, f32, Vec2),
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
            let node = Vec2::new(
                origin.0 + ix as f32 * dx,
                origin.1 + iy as f32 * dx,
            );
            f(ix, iy, w, node);
        }
    }
}

fn apic_affine_row(field: &StaggeredField2, pos: Vec2) -> Vec2 {
    let grid = field.grid();
    let mut m00 = 0.0;
    let mut m01 = 0.0;
    let mut m11 = 0.0;
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    for_each_staggered_neighbor(grid, pos, |ix, iy, w, node| {
        let d = Vec2::new(node.x - pos.x, node.y - pos.y);
        let dx = d.x;
        let dy = d.y;
        m00 += w * dx * dx;
        m01 += w * dx * dy;
        m11 += w * dy * dy;
        let val = field.get(ix, iy);
        b0 += w * val * dx;
        b1 += w * val * dy;
    });
    let det = m00 * m11 - m01 * m01;
    if det.abs() <= 1e-6 {
        return Vec2::zero();
    }
    let inv00 = m11 / det;
    let inv01 = -m01 / det;
    let inv11 = m00 / det;
    Vec2::new(b0 * inv00 + b1 * inv01, b0 * inv01 + b1 * inv11)
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
    fn gpu_pack_unpack_updates_particles() {
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(1.0, 2.0), Vec2::new(0.5, -0.5), ParticlePhase::Liquid);
        system.push(Vec2::new(3.0, 4.0), Vec2::new(-1.0, 2.0), ParticlePhase::Liquid);
        let mut packed = system.pack_gpu();
        packed[0].pos = [9.0, 8.0];
        packed[0].vel = [2.0, -3.0];
        packed[1].pos = [-1.0, 0.25];
        packed[1].vel = [0.0, 1.5];
        let ok = system.unpack_gpu(&packed);
        assert!(ok);
        assert_eq!(system.positions()[0], Vec2::new(9.0, 8.0));
        assert_eq!(system.velocities()[0], Vec2::new(2.0, -3.0));
        assert_eq!(system.positions()[1], Vec2::new(-1.0, 0.25));
        assert_eq!(system.velocities()[1], Vec2::new(0.0, 1.5));
    }

    #[test]
    fn gpu_unpack_rejects_mismatched_length() {
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0), ParticlePhase::Liquid);
        let data = vec![
            GpuParticle {
                pos: [0.0, 0.0],
                vel: [1.0, 1.0],
            },
            GpuParticle {
                pos: [1.0, 2.0],
                vel: [3.0, 4.0],
            },
        ];
        assert!(!system.unpack_gpu(&data));
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
    fn apic_matches_pic_for_zero_affine() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let mut system = ParticleSystem::new();
        let pos = Vec2::new(3.2, 4.1);
        let vel = Vec2::new(0.7, -0.3);
        system.push(pos, vel, ParticlePhase::Liquid);
        let pic = system.to_grid(grid);
        let apic = system.to_grid_apic(grid);
        let pic_sample = pic.sample_linear((pos.x, pos.y));
        let apic_sample = apic.sample_linear((pos.x, pos.y));
        assert_close(pic_sample.x, apic_sample.x, 1e-4);
        assert_close(pic_sample.y, apic_sample.y, 1e-4);
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

    #[test]
    fn separation_pushes_particles_apart() {
        let grid = MacGrid2::new(10, 10, 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(2.0, 2.0), Vec2::zero(), ParticlePhase::Liquid);
        system.push(Vec2::new(2.1, 2.0), Vec2::zero(), ParticlePhase::Liquid);
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        system.separate(0.5, bounds, 1);
        let dx = system.positions()[0].x - system.positions()[1].x;
        let dy = system.positions()[0].y - system.positions()[1].y;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist >= 0.45);
    }

    #[test]
    fn sph_density_normalizes_to_rest() {
        let grid = MacGrid2::new(12, 12, 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(6.5, 6.5), Vec2::zero(), ParticlePhase::Liquid);
        let radius = 2.0;
        let rest = system.rest_sph_density(grid, radius);
        let field = system.to_sph_density_with_rest(grid, radius, 1.0, rest);
        let (_sum, _min, max_value, _nonfinite) = field.stats();
        assert_close(max_value, 1.0, 1e-4);
    }

    #[test]
    fn level_set_is_negative_near_particles() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(4.0, 4.0), Vec2::zero(), ParticlePhase::Liquid);
        let phi = system.to_level_set(grid, bounds, 0.75, 3.0);
        let center = phi.get(3, 3);
        assert!(center < 0.0);
    }

    #[test]
    fn pbf_project_keeps_particles_in_bounds() {
        let grid = MacGrid2::new(12, 12, 1.0);
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(2.5, 2.5), Vec2::zero(), ParticlePhase::Liquid);
        system.push(Vec2::new(2.6, 2.5), Vec2::zero(), ParticlePhase::Liquid);
        let rest = system.pbf_rest_density(bounds, 1.0);
        let prev = system.positions().to_vec();
        system.pbf_project(bounds, 1.0, rest, 2, 0.1, 0.0, 0.0, 4.0, 0.2, &prev);
        for pos in system.positions() {
            assert!(pos.x >= bounds.min.x && pos.x <= bounds.max.x);
            assert!(pos.y >= bounds.min.y && pos.y <= bounds.max.y);
        }
    }

    #[test]
    fn xsph_viscosity_smooths_velocity() {
        let grid = MacGrid2::new(6, 6, 1.0);
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(2.5, 2.5), Vec2::new(1.0, 0.0), ParticlePhase::Liquid);
        system.push(Vec2::new(2.9, 2.5), Vec2::new(-1.0, 0.0), ParticlePhase::Liquid);
        let before = system.velocities()[0] - system.velocities()[1];
        let before_sq = before.x * before.x + before.y * before.y;
        system.xsph_viscosity(bounds, 1.0, 0.5);
        let after = system.velocities()[0] - system.velocities()[1];
        let after_sq = after.x * after.x + after.y * after.y;
        assert!(after_sq < before_sq);
        for vel in system.velocities() {
            assert!(vel.x.is_finite() && vel.y.is_finite());
        }
    }

    #[test]
    fn resample_volume_adds_and_removes() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let bounds = ParticleBounds::from_grid(grid, 0.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let occ_low = Field2::new(grid.cell_grid(), 0.0);
        let occ_high = Field2::new(grid.cell_grid(), 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(1.5, 1.5), Vec2::zero(), ParticlePhase::Liquid);
        system.push(Vec2::new(2.5, 1.5), Vec2::zero(), ParticlePhase::Liquid);
        let (added, removed) =
            system.resample_volume(grid, bounds, &occ_low, &velocity, 6, 0.0, 0.6, 0.9, 10, 0.1, 1);
        assert_eq!(removed, 0);
        assert_eq!(added, 4);
        assert_eq!(system.len(), 6);
        let (added, removed) = system.resample_volume(
            grid,
            bounds,
            &occ_high,
            &velocity,
            3,
            0.0,
            0.2,
            0.3,
            10,
            0.1,
            2,
        );
        assert_eq!(added, 0);
        assert_eq!(removed, 3);
        assert_eq!(system.len(), 3);
    }

    #[test]
    fn air_drag_damps_sparse_particles() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(1.5, 1.5), Vec2::new(1.0, 0.0), ParticlePhase::Liquid);
        let occ_low = Field2::new(grid.cell_grid(), 0.0);
        let occ_high = Field2::new(grid.cell_grid(), 1.0);
        system.apply_air_drag(&occ_low, 0.1, 2.0, 0.25);
        assert!(system.velocities[0].x < 1.0);
        system.velocities[0] = Vec2::new(1.0, 0.0);
        system.apply_air_drag(&occ_high, 0.1, 2.0, 0.25);
        assert_close(system.velocities[0].x, 1.0, 1e-6);
    }

    #[test]
    fn circle_obstacle_pushes_particles_out() {
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), ParticlePhase::Liquid);
        system.apply_circle_obstacle(Vec2::zero(), 1.0, 0.0);
        let pos = system.positions()[0];
        let dist = (pos.x * pos.x + pos.y * pos.y).sqrt();
        assert!(dist >= 1.0 - 1e-5);
    }

    #[test]
    fn set_velocity_field_assigns_values() {
        let mut system = ParticleSystem::new();
        system.push(Vec2::new(1.0, 2.0), Vec2::zero(), ParticlePhase::Liquid);
        system.push(Vec2::new(2.0, 3.0), Vec2::zero(), ParticlePhase::Liquid);
        system.set_velocity_field(|pos| Vec2::new(pos.x, pos.y));
        assert_close(system.velocities()[0].x, 1.0, 1e-6);
        assert_close(system.velocities()[0].y, 2.0, 1e-6);
        assert_close(system.velocities()[1].x, 2.0, 1e-6);
        assert_close(system.velocities()[1].y, 3.0, 1e-6);
    }
}
