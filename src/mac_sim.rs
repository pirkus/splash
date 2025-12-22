use crate::{
    apply_domain_boundaries, apply_solid_boundaries, BoundaryConfig, CellFlags, CellType, Field2,
    Grid2, MacGrid2, MacVelocity2, StaggeredField2, Vec2, vec_field::VecField2,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdvectionScheme {
    SemiLagrangian,
    Bfecc,
}

#[derive(Clone, Copy, Debug)]
pub struct MacSimParams {
    pub dt: f32,
    pub cfl: f32,
    pub diffusion: f32,
    pub viscosity: f32,
    pub pressure_iters: usize,
    pub pressure_tol: f32,
    pub advection: AdvectionScheme,
    pub boundaries: BoundaryConfig,
    pub body_force: Vec2,
    pub buoyancy: f32,
    pub ambient_density: f32,
    pub surface_tension: f32,
    pub free_surface: bool,
    pub density_threshold: f32,
    pub surface_band: f32,
    pub preserve_mass: bool,
}

#[derive(Clone, Debug)]
pub struct MacSimState {
    pub density: Field2,
    pub velocity: MacVelocity2,
    pub flags: CellFlags,
}

pub fn step(state: &MacSimState, params: MacSimParams) -> MacSimState {
    let target_mass = if params.free_surface && params.preserve_mass {
        state.density.sum()
    } else {
        0.0
    };
    let flag_density = if params.free_surface {
        sharpen_density(&state.density, params.density_threshold, params.surface_band)
    } else {
        state.density.clone()
    };
    let flags = if params.free_surface {
        flags_from_density(&flag_density, &state.flags, params.density_threshold)
    } else {
        state.flags.clone()
    };
    let bounded = apply_domain_boundaries(&state.velocity, params.boundaries);
    let bounded = apply_solid_boundaries(&bounded, &flags);
    let bounded = if params.free_surface {
        apply_fluid_mask(&bounded, &flags)
    } else {
        bounded
    };
    let dt = clamp_dt(params.dt, params.cfl, &bounded);
    let forced = apply_external_forces(&bounded, &state.density, params, dt);
    let forced = apply_domain_boundaries(&forced, params.boundaries);
    let forced = apply_solid_boundaries(&forced, &flags);
    let forced = if params.free_surface {
        apply_fluid_mask(&forced, &flags)
    } else {
        forced
    };
    let advected_velocity = match params.advection {
        AdvectionScheme::SemiLagrangian => advect_velocity(&forced, &forced, dt),
        AdvectionScheme::Bfecc => advect_velocity_bfecc(&forced, &forced, dt),
    };
    let advected_velocity = if params.free_surface {
        apply_fluid_mask(&advected_velocity, &flags)
    } else {
        advected_velocity
    };
    let diffused_velocity = diffuse_velocity(&advected_velocity, params.viscosity, dt);
    let projected_velocity =
        project_with_flags(&diffused_velocity, &flags, params.pressure_iters, params.pressure_tol);
    let projected_velocity = apply_domain_boundaries(&projected_velocity, params.boundaries);
    let projected_velocity = apply_solid_boundaries(&projected_velocity, &flags);
    let projected_velocity = if params.free_surface {
        apply_fluid_mask(&projected_velocity, &flags)
    } else {
        projected_velocity
    };
    let advected_density = match params.advection {
        AdvectionScheme::SemiLagrangian => advect_scalar(&state.density, &projected_velocity, dt),
        AdvectionScheme::Bfecc => advect_scalar_bfecc(&state.density, &projected_velocity, dt),
    };
    let diffused_density = diffuse_scalar(&advected_density, params.diffusion, dt);
    let continuous_density = if params.free_surface {
        let clamped = clamp_scalar(&diffused_density, 0.0, 1.0);
        if params.surface_band > 0.0 && params.preserve_mass {
            rescale_density_to_mass(&clamped, target_mass)
        } else {
            clamped
        }
    } else {
        diffused_density
    };
    let next_flags = if params.free_surface {
        let next_flag_density =
            sharpen_density(&continuous_density, params.density_threshold, params.surface_band);
        flags_from_density(&next_flag_density, &flags, params.density_threshold)
    } else {
        flags
    };
    MacSimState {
        density: continuous_density,
        velocity: projected_velocity,
        flags: next_flags,
    }
}

fn clamp_dt(dt: f32, cfl: f32, velocity: &MacVelocity2) -> f32 {
    if dt <= 0.0 || cfl <= 0.0 {
        return dt;
    }
    let max_vel = velocity.max_abs();
    if max_vel == 0.0 {
        return dt;
    }
    let dx = velocity.grid().dx();
    let limit = cfl * dx / max_vel;
    dt.min(limit)
}

pub fn flags_from_density(
    density: &Field2,
    base_flags: &CellFlags,
    threshold: f32,
) -> CellFlags {
    let grid = density.grid();
    CellFlags::from_fn(grid, |x, y| match base_flags.get(x, y) {
        CellType::Solid => CellType::Solid,
        _ => {
            if density.get(x, y) > threshold {
                CellType::Fluid
            } else {
                CellType::Air
            }
        }
    })
}

pub fn apply_external_forces(
    velocity: &MacVelocity2,
    density: &Field2,
    params: MacSimParams,
    dt: f32,
) -> MacVelocity2 {
    let with_body = add_body_force(velocity, params.body_force, dt);
    let with_buoyancy = add_buoyancy(
        &with_body,
        density,
        params.buoyancy,
        params.ambient_density,
        dt,
    );
    if params.surface_tension == 0.0 || dt == 0.0 {
        with_buoyancy
    } else {
        add_surface_tension(&with_buoyancy, density, params.surface_tension, dt)
    }
}

pub fn add_body_force(velocity: &MacVelocity2, force: Vec2, dt: f32) -> MacVelocity2 {
    if dt == 0.0 || (force.x == 0.0 && force.y == 0.0) {
        return velocity.clone();
    }
    let u = velocity
        .u()
        .map_with_index(|_, _, value| value + force.x * dt);
    let v = velocity
        .v()
        .map_with_index(|_, _, value| value + force.y * dt);
    MacVelocity2::from_components(velocity.grid(), u, v)
}

pub fn add_buoyancy(
    velocity: &MacVelocity2,
    density: &Field2,
    buoyancy: f32,
    ambient_density: f32,
    dt: f32,
) -> MacVelocity2 {
    if dt == 0.0 || buoyancy == 0.0 {
        return velocity.clone();
    }
    let v_grid = velocity.v().grid();
    let v = velocity.v().map_with_index(|x, y, value| {
        let pos = v_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        value + (rho - ambient_density) * buoyancy * dt
    });
    MacVelocity2::from_components(velocity.grid(), velocity.u().clone(), v)
}

pub fn add_surface_tension(
    velocity: &MacVelocity2,
    density: &Field2,
    surface_tension: f32,
    dt: f32,
) -> MacVelocity2 {
    if surface_tension == 0.0 || dt == 0.0 {
        return velocity.clone();
    }
    let grid = density.grid();
    let dx = grid.dx();
    let eps = 1e-6;
    let normals = VecField2::from_fn(grid, |x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let ddx = (density.sample_clamped(xi + 1, yi) - density.sample_clamped(xi - 1, yi))
            / (2.0 * dx);
        let ddy = (density.sample_clamped(xi, yi + 1) - density.sample_clamped(xi, yi - 1))
            / (2.0 * dx);
        let mag = (ddx * ddx + ddy * ddy).sqrt();
        if mag > eps {
            Vec2::new(ddx / mag, ddy / mag)
        } else {
            Vec2::zero()
        }
    });
    let curvature = Field2::from_fn(grid, |x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let nx_r = normals.u().sample_clamped(xi + 1, yi);
        let nx_l = normals.u().sample_clamped(xi - 1, yi);
        let ny_u = normals.v().sample_clamped(xi, yi + 1);
        let ny_d = normals.v().sample_clamped(xi, yi - 1);
        -((nx_r - nx_l) + (ny_u - ny_d)) / (2.0 * dx)
    });
    let grad_mag = Field2::from_fn(grid, |x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let ddx = (density.sample_clamped(xi + 1, yi) - density.sample_clamped(xi - 1, yi))
            / (2.0 * dx);
        let ddy = (density.sample_clamped(xi, yi + 1) - density.sample_clamped(xi, yi - 1))
            / (2.0 * dx);
        (ddx * ddx + ddy * ddy).sqrt()
    });
    let force = VecField2::from_fn(grid, |x, y| {
        let n = normals.get(x, y);
        let kappa = curvature.get(x, y);
        let delta = grad_mag.get(x, y);
        n.scale(surface_tension * kappa * delta)
    });
    let u = velocity.u().map_with_index(|x, y, value| {
        let pos = velocity.u().grid().index_position(x, y);
        value + force.sample_linear(pos).x * dt
    });
    let v = velocity.v().map_with_index(|x, y, value| {
        let pos = velocity.v().grid().index_position(x, y);
        value + force.sample_linear(pos).y * dt
    });
    MacVelocity2::from_components(velocity.grid(), u, v)
}

pub fn advect_scalar(field: &Field2, velocity: &MacVelocity2, dt: f32) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let grid = field.grid();
    field.map_with_index(|x, y, _| {
        let pos = grid.cell_center(x, y);
        let v = velocity.sample_linear(pos);
        field.sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    })
}

pub fn advect_scalar_bfecc(field: &Field2, velocity: &MacVelocity2, dt: f32) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let forward = advect_scalar(field, velocity, dt);
    let backward = advect_scalar(&forward, velocity, -dt);
    let correction = field.add_scaled(&backward, -1.0);
    let corrected = forward.add_scaled(&correction, 0.5);
    clamp_scalar_to_neighbors(field, &corrected)
}

pub fn rescale_density_to_mass(density: &Field2, target_mass: f32) -> Field2 {
    if target_mass <= 0.0 || !target_mass.is_finite() {
        return density.clone();
    }
    let current = density.sum();
    if current <= 0.0 || !current.is_finite() {
        return density.clone();
    }
    let scale = target_mass / current;
    density.map(|value| value * scale)
}

pub fn sharpen_density(density: &Field2, threshold: f32, band: f32) -> Field2 {
    if band <= 0.0 {
        return density.map(|value| if value >= threshold { 1.0 } else { 0.0 });
    }
    let edge0 = (threshold - band).max(0.0);
    let edge1 = (threshold + band).min(1.0);
    if edge1 <= edge0 {
        return density.map(|value| if value >= threshold { 1.0 } else { 0.0 });
    }
    let inv = 1.0 / (edge1 - edge0);
    density.map(|value| {
        let t = ((value - edge0) * inv).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    })
}

pub fn clamp_scalar(field: &Field2, min: f32, max: f32) -> Field2 {
    field.map(|value| {
        if value.is_finite() {
            value.clamp(min, max)
        } else {
            min
        }
    })
}

fn clamp_scalar_to_neighbors(source: &Field2, candidate: &Field2) -> Field2 {
    candidate.map_with_index(|x, y, value| {
        let xi = x as i32;
        let yi = y as i32;
        let mut min_value = source.sample_clamped(xi, yi);
        if !min_value.is_finite() {
            return 0.0;
        }
        let mut max_value = min_value;
        let neighbors = [
            (xi - 1, yi),
            (xi + 1, yi),
            (xi, yi - 1),
            (xi, yi + 1),
        ];
        for (nx, ny) in neighbors {
            let sample = source.sample_clamped(nx, ny);
            if sample.is_finite() {
                if sample < min_value {
                    min_value = sample;
                }
                if sample > max_value {
                    max_value = sample;
                }
            }
        }
        let sanitized = if value.is_finite() { value } else { min_value };
        sanitized.clamp(min_value, max_value)
    })
}

pub fn advect_velocity(field: &MacVelocity2, velocity: &MacVelocity2, dt: f32) -> MacVelocity2 {
    let u_grid = field.u().grid();
    let v_grid = field.v().grid();
    let u = field.u().map_with_index(|x, y, _| {
        let pos = u_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        field
            .u()
            .sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
    let v = field.v().map_with_index(|x, y, _| {
        let pos = v_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        field
            .v()
            .sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
    MacVelocity2::from_components(field.grid(), u, v)
}

pub fn advect_velocity_bfecc(field: &MacVelocity2, velocity: &MacVelocity2, dt: f32) -> MacVelocity2 {
    if dt == 0.0 {
        return field.clone();
    }
    let forward = advect_velocity(field, velocity, dt);
    let backward = advect_velocity(&forward, velocity, -dt);
    let correction = field.add_scaled(&backward, -1.0);
    let corrected = forward.add_scaled(&correction, 0.5);
    let u = clamp_staggered_to_neighbors(field.u(), corrected.u());
    let v = clamp_staggered_to_neighbors(field.v(), corrected.v());
    MacVelocity2::from_components(field.grid(), u, v)
}

fn clamp_staggered_to_neighbors(
    source: &StaggeredField2,
    candidate: &StaggeredField2,
) -> StaggeredField2 {
    candidate.map_with_index(|x, y, value| {
        let xi = x as i32;
        let yi = y as i32;
        let mut min_value = source.sample_clamped(xi, yi);
        if !min_value.is_finite() {
            return 0.0;
        }
        let mut max_value = min_value;
        let neighbors = [
            (xi - 1, yi),
            (xi + 1, yi),
            (xi, yi - 1),
            (xi, yi + 1),
        ];
        for (nx, ny) in neighbors {
            let sample = source.sample_clamped(nx, ny);
            if sample.is_finite() {
                if sample < min_value {
                    min_value = sample;
                }
                if sample > max_value {
                    max_value = sample;
                }
            }
        }
        let sanitized = if value.is_finite() { value } else { min_value };
        sanitized.clamp(min_value, max_value)
    })
}

pub fn apply_fluid_mask(velocity: &MacVelocity2, flags: &CellFlags) -> MacVelocity2 {
    let grid = velocity.grid();
    let w = grid.width();
    let h = grid.height();
    let u = velocity.u().map_with_index(|x, y, value| {
        let left = if x == 0 {
            CellType::Solid
        } else {
            flags.get(x - 1, y)
        };
        let right = if x == w {
            CellType::Solid
        } else {
            flags.get(x, y)
        };
        if left != CellType::Fluid && right != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    let v = velocity.v().map_with_index(|x, y, value| {
        let bottom = if y == 0 {
            CellType::Solid
        } else {
            flags.get(x, y - 1)
        };
        let top = if y == h {
            CellType::Solid
        } else {
            flags.get(x, y)
        };
        if bottom != CellType::Fluid && top != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    MacVelocity2::from_components(grid, u, v)
}

pub fn diffuse_scalar(field: &Field2, diffusion: f32, dt: f32) -> Field2 {
    if diffusion == 0.0 || dt == 0.0 {
        return field.clone();
    }
    field.add_scaled(&field.laplacian(), diffusion * dt)
}

pub fn diffuse_velocity(field: &MacVelocity2, viscosity: f32, dt: f32) -> MacVelocity2 {
    if viscosity == 0.0 || dt == 0.0 {
        return field.clone();
    }
    let u = diffuse_staggered(field.u(), viscosity, dt);
    let v = diffuse_staggered(field.v(), viscosity, dt);
    MacVelocity2::from_components(field.grid(), u, v)
}

pub fn diffuse_staggered(field: &StaggeredField2, diffusion: f32, dt: f32) -> StaggeredField2 {
    if diffusion == 0.0 || dt == 0.0 {
        return field.clone();
    }
    field.add_scaled(&field.laplacian(), diffusion * dt)
}

pub fn divergence(velocity: &MacVelocity2) -> Field2 {
    let grid = velocity.grid();
    let dx = grid.dx();
    let cell_grid = grid.cell_grid();
    Field2::from_fn(cell_grid, |x, y| {
        let u_r = velocity.u().get(x + 1, y);
        let u_l = velocity.u().get(x, y);
        let v_u = velocity.v().get(x, y + 1);
        let v_d = velocity.v().get(x, y);
        (u_r - u_l + v_u - v_d) / dx
    })
}

pub fn project_with_flags(
    velocity: &MacVelocity2,
    flags: &CellFlags,
    iterations: usize,
    tol: f32,
) -> MacVelocity2 {
    let div = divergence(velocity).map_with_index(|x, y, value| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    let pressure = solve_pressure_pcg_with_flags(&div, flags, iterations, tol);
    apply_pressure_gradient(velocity, &pressure)
}

pub fn solve_pressure_pcg_with_flags(
    divergence: &Field2,
    flags: &CellFlags,
    iterations: usize,
    tol: f32,
) -> Field2 {
    let grid = divergence.grid();
    let b = divergence.map(|value| -value);
    let inv_diag = build_inv_diag(flags, grid);
    let mut x = Field2::new(grid, 0.0);
    let mut r = b;
    let mut z = Field2::new(grid, 0.0);
    z.mul_pointwise_into(&r, &inv_diag);
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);
    if rz_old == 0.0 {
        return x;
    }
    let tol_sq = tol * tol;
    let mut ap = Field2::new(grid, 0.0);
    for _ in 0..iterations {
        apply_negative_laplacian_with_flags_into(&p, flags, &mut ap);
        let denom = p.dot(&ap);
        if denom.abs() < 1e-20 {
            break;
        }
        let alpha = rz_old / denom;
        x.add_scaled_in_place(&p, alpha);
        r.add_scaled_in_place(&ap, -alpha);
        if r.dot(&r) <= tol_sq {
            break;
        }
        z.mul_pointwise_into(&r, &inv_diag);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz_old;
        p.scale_in_place(beta);
        p.add_scaled_in_place(&z, 1.0);
        rz_old = rz_new;
    }
    x
}

fn apply_negative_laplacian_with_flags_into(field: &Field2, flags: &CellFlags, out: &mut Field2) {
    debug_assert_eq!(field.grid(), out.grid(), "field grid mismatch");
    let grid = field.grid();
    let dx2 = grid.dx() * grid.dx();
    out.fill_with_index(|x, y| {
        if flags.get(x, y) != CellType::Fluid {
            return 0.0;
        }
        let center = field.get(x, y);
        let (sum, count) = neighbor_sum_count(field, flags, x, y);
        if count == 0.0 {
            0.0
        } else {
            (count * center - sum) / dx2
        }
    });
}

fn build_inv_diag(flags: &CellFlags, grid: Grid2) -> Field2 {
    let dx2 = grid.dx() * grid.dx();
    Field2::from_fn(grid, |x, y| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            let count = neighbor_count(flags, grid, x, y);
            if count == 0.0 {
                0.0
            } else {
                dx2 / count
            }
        }
    })
}

fn neighbor_sum_count(field: &Field2, flags: &CellFlags, x: usize, y: usize) -> (f32, f32) {
    let grid = field.grid();
    let width = grid.width() as i32;
    let height = grid.height() as i32;
    let mut count = 0.0;
    let mut sum = 0.0;
    let coords = [
        (x as i32 - 1, y as i32),
        (x as i32 + 1, y as i32),
        (x as i32, y as i32 - 1),
        (x as i32, y as i32 + 1),
    ];
    for (nx, ny) in coords {
        if nx < 0 || ny < 0 || nx >= width || ny >= height {
            continue;
        }
        let nxu = nx as usize;
        let nyu = ny as usize;
        match flags.get(nxu, nyu) {
            CellType::Solid => continue,
            CellType::Air => {
                count += 1.0;
            }
            CellType::Fluid => {
                sum += field.get(nxu, nyu);
                count += 1.0;
            }
        }
    }
    (sum, count)
}

fn neighbor_count(flags: &CellFlags, grid: Grid2, x: usize, y: usize) -> f32 {
    let width = grid.width() as i32;
    let height = grid.height() as i32;
    let mut count = 0.0;
    let coords = [
        (x as i32 - 1, y as i32),
        (x as i32 + 1, y as i32),
        (x as i32, y as i32 - 1),
        (x as i32, y as i32 + 1),
    ];
    for (nx, ny) in coords {
        if nx < 0 || ny < 0 || nx >= width || ny >= height {
            continue;
        }
        let nxu = nx as usize;
        let nyu = ny as usize;
        match flags.get(nxu, nyu) {
            CellType::Solid => continue,
            CellType::Air | CellType::Fluid => {
                count += 1.0;
            }
        }
    }
    count
}

pub fn apply_pressure_gradient(velocity: &MacVelocity2, pressure: &Field2) -> MacVelocity2 {
    let grid = velocity.grid();
    let dx = grid.dx();
    let cell_grid = grid.cell_grid();
    let u = velocity.u().map_with_index(|x, y, value| {
        let left = if x == 0 { 0 } else { x - 1 };
        let right = x.min(cell_grid.width() - 1);
        let p_l = pressure.get(left, y);
        let p_r = pressure.get(right, y);
        value - (p_r - p_l) / dx
    });
    let v = velocity.v().map_with_index(|x, y, value| {
        let bottom = if y == 0 { 0 } else { y - 1 };
        let top = y.min(cell_grid.height() - 1);
        let p_b = pressure.get(x, bottom);
        let p_t = pressure.get(x, top);
        value - (p_t - p_b) / dx
    });
    MacVelocity2::from_components(grid, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundaryCondition, CellType};
    use std::f32::consts::PI;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} to be within {tol} of {b}"
        );
    }

    #[test]
    fn divergence_of_constant_velocity_is_zero() {
        let grid = MacGrid2::new(8, 6, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::new(1.0, -1.0));
        let div = divergence(&velocity);
        assert_close(div.abs_sum(), 0.0, 1e-6);
    }

    #[test]
    fn advect_scalar_keeps_constant_field() {
        let grid = MacGrid2::new(10, 10, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::new(2.0, -1.0));
        let density = Field2::new(grid.cell_grid(), 3.0);
        let advected = advect_scalar(&density, &velocity, 0.5);
        assert_close(advected.sum(), density.sum(), 1e-5);
    }

    #[test]
    fn rescale_density_preserves_target_mass() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| (x + y) as f32 * 0.1);
        let target = density.sum() * 0.5;
        let rescaled = rescale_density_to_mass(&density, target);
        assert_close(rescaled.sum(), target, 1e-6);
    }

    #[test]
    fn sharpen_density_thresholds_values() {
        let grid = MacGrid2::new(2, 2, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| {
            if x == 0 && y == 0 {
                0.2
            } else {
                0.8
            }
        });
        let sharpened = sharpen_density(&density, 0.5, 0.0);
        assert_close(sharpened.get(0, 0), 0.0, 1e-6);
        assert_close(sharpened.get(1, 1), 1.0, 1e-6);
    }

    #[test]
    fn step_preserves_constant_density_with_no_solids() {
        let grid = MacGrid2::new(6, 6, 1.0);
        let density = Field2::new(grid.cell_grid(), 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let params = MacSimParams {
            dt: 0.1,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.0,
            pressure_iters: 20,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: false,
            density_threshold: 0.5,
            surface_band: 0.0,
            preserve_mass: false,
        };
        let state = MacSimState {
            density,
            velocity,
            flags,
        };
        let next = step(&state, params);
        assert_close(next.density.sum(), state.density.sum(), 1e-6);
    }

    #[test]
    fn apply_pressure_gradient_handles_edges() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let pressure = Field2::from_fn(grid.cell_grid(), |x, y| (x + y) as f32);
        let projected = apply_pressure_gradient(&velocity, &pressure);
        assert_eq!(projected.u().grid(), grid.u_grid());
        assert_eq!(projected.v().grid(), grid.v_grid());
    }

    #[test]
    fn body_force_adds_uniform_acceleration() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let forced = add_body_force(&velocity, Vec2::new(0.0, 2.0), 0.5);
        assert_close(forced.v().get(1, 1), 1.0, 1e-6);
        assert_close(forced.u().get(1, 1), 0.0, 1e-6);
    }

    #[test]
    fn buoyancy_pushes_upward_with_density() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let density = Field2::new(grid.cell_grid(), 2.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let buoyant = add_buoyancy(&velocity, &density, 1.0, 1.0, 0.5);
        assert_close(buoyant.v().get(1, 1), 0.5, 1e-6);
    }

    #[test]
    fn lid_driven_cavity_sets_top_velocity() {
        let grid = MacGrid2::new(8, 8, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let state = MacSimState {
            density: Field2::new(grid.cell_grid(), 0.0),
            velocity,
            flags,
        };
        let params = MacSimParams {
            dt: 0.1,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.0,
            pressure_iters: 10,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig {
                left: BoundaryCondition::NoSlip,
                right: BoundaryCondition::NoSlip,
                bottom: BoundaryCondition::NoSlip,
                top: BoundaryCondition::Inflow(Vec2::new(1.0, 0.0)),
            },
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: false,
            density_threshold: 0.5,
            surface_band: 0.0,
            preserve_mass: false,
        };
        let next = step(&state, params);
        assert_close(next.velocity.u().get(1, 7), 1.0, 1e-6);
        assert_close(next.velocity.v().get(1, 8), 0.0, 1e-6);
    }

    #[test]
    fn lid_driven_cavity_drives_interior_flow() {
        let grid = MacGrid2::new(16, 16, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let state = MacSimState {
            density: Field2::new(grid.cell_grid(), 0.0),
            velocity,
            flags,
        };
        let params = MacSimParams {
            dt: 0.1,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.2,
            pressure_iters: 20,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig {
                left: BoundaryCondition::NoSlip,
                right: BoundaryCondition::NoSlip,
                bottom: BoundaryCondition::NoSlip,
                top: BoundaryCondition::Inflow(Vec2::new(1.0, 0.0)),
            },
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: false,
            density_threshold: 0.5,
            surface_band: 0.0,
            preserve_mass: false,
        };
        let evolved = (0..20).fold(state, |current, _| step(&current, params));
        let mid_x = grid.width() / 2;
        let u_interior = evolved.velocity.u().get(mid_x, grid.height() - 2);
        assert!(u_interior > 1e-4);
    }

    #[test]
    fn vortex_decay_reduces_energy() {
        let grid = MacGrid2::new(32, 32, 1.0);
        let dx = grid.dx();
        let lx = grid.width() as f32 * dx;
        let ly = grid.height() as f32 * dx;
        let u_grid = grid.u_grid();
        let v_grid = grid.v_grid();
        let u = StaggeredField2::from_fn(u_grid, |x, y| {
            let (px, py) = u_grid.index_position(x, y);
            let sx = (PI * px / lx).sin();
            let cy = (PI * py / ly).cos();
            (PI / ly) * sx * cy
        });
        let v = StaggeredField2::from_fn(v_grid, |x, y| {
            let (px, py) = v_grid.index_position(x, y);
            let cx = (PI * px / lx).cos();
            let sy = (PI * py / ly).sin();
            -(PI / lx) * cx * sy
        });
        let velocity = MacVelocity2::from_components(grid, u, v);
        let energy_before = velocity.energy();
        let diffused = diffuse_velocity(&velocity, 0.1, 0.1);
        let energy_after = diffused.energy();
        assert!(energy_after < energy_before);
    }

    #[test]
    fn cfl_clamps_dt() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let velocity = MacVelocity2::new(grid, Vec2::new(2.0, 0.0));
        let dt = clamp_dt(1.0, 0.5, &velocity);
        assert_close(dt, 0.25, 1e-6);
    }

    #[test]
    fn mass_conservation_without_sources_or_flow() {
        let grid = MacGrid2::new(32, 32, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| (x * y) as f32 * 0.0005);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let params = MacSimParams {
            dt: 0.1,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.0,
            pressure_iters: 10,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: false,
            density_threshold: 0.5,
            surface_band: 0.0,
            preserve_mass: false,
        };
        let state = MacSimState {
            density,
            velocity,
            flags,
        };
        let before = state.density.sum();
        let evolved = (0..10).fold(state, |current, _| step(&current, params));
        let after = evolved.density.sum();
        assert_close(after, before, 1e-6);
    }
}
