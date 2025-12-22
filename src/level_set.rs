use crate::{
    apply_domain_boundaries, apply_solid_boundaries,
    mac_sim::{
        add_body_force, advect_scalar, advect_scalar_bfecc, advect_velocity,
        advect_velocity_bfecc, apply_fluid_mask, diffuse_velocity, project_with_flags,
        AdvectionScheme,
    },
    vec_field::VecField2, BoundaryConfig, CellFlags, CellType, Field2, MacVelocity2,
    StaggeredField2, Vec2,
};

#[derive(Clone, Copy, Debug)]
pub struct LevelSetParams {
    pub dt: f32,
    pub cfl: f32,
    pub viscosity: f32,
    pub pressure_iters: usize,
    pub pressure_tol: f32,
    pub advection: AdvectionScheme,
    pub boundaries: BoundaryConfig,
    pub body_force: Vec2,
    pub surface_tension: f32,
    pub surface_tension_band: f32,
    pub reinit_iters: usize,
    pub reinit_dt: f32,
    pub extrapolation_iters: usize,
    pub preserve_volume: bool,
    pub target_volume: f32,
    pub volume_band: f32,
}

#[derive(Clone, Debug)]
pub struct LevelSetState {
    pub phi: Field2,
    pub velocity: MacVelocity2,
    pub flags: CellFlags,
}

pub fn level_set_step(state: &LevelSetState, params: LevelSetParams) -> LevelSetState {
    let flags = flags_from_phi(&state.phi, &state.flags);
    let bounded = apply_domain_boundaries(&state.velocity, params.boundaries);
    let bounded = apply_solid_boundaries(&bounded, &flags);
    let bounded = apply_fluid_mask(&bounded, &flags);
    let dt = clamp_dt(params.dt, params.cfl, &bounded);
    let forced = add_body_force(&bounded, params.body_force, dt);
    let forced = if params.surface_tension != 0.0 && dt != 0.0 {
        add_surface_tension_phi(
            &forced,
            &state.phi,
            params.surface_tension,
            params.surface_tension_band,
            dt,
        )
    } else {
        forced
    };
    let forced = apply_domain_boundaries(&forced, params.boundaries);
    let forced = apply_solid_boundaries(&forced, &flags);
    let forced = apply_fluid_mask(&forced, &flags);
    let advected_velocity = match params.advection {
        AdvectionScheme::SemiLagrangian => advect_velocity(&forced, &forced, dt),
        AdvectionScheme::Bfecc => advect_velocity_bfecc(&forced, &forced, dt),
    };
    let advected_velocity = apply_fluid_mask(&advected_velocity, &flags);
    let diffused_velocity = diffuse_velocity(&advected_velocity, params.viscosity, dt);
    let projected_velocity =
        project_with_flags(&diffused_velocity, &flags, params.pressure_iters, params.pressure_tol);
    let projected_velocity = apply_domain_boundaries(&projected_velocity, params.boundaries);
    let projected_velocity = apply_solid_boundaries(&projected_velocity, &flags);
    let projected_velocity = apply_fluid_mask(&projected_velocity, &flags);
    let advect_velocity = if params.extrapolation_iters > 0 {
        extrapolate_velocity(&projected_velocity, &flags, params.extrapolation_iters)
    } else {
        projected_velocity.clone()
    };
    let advected_phi = match params.advection {
        AdvectionScheme::SemiLagrangian => advect_scalar(&state.phi, &advect_velocity, dt),
        AdvectionScheme::Bfecc => advect_scalar_bfecc(&state.phi, &advect_velocity, dt),
    };
    let reinitialized_phi = reinitialize_phi(&advected_phi, params.reinit_iters, params.reinit_dt);
    let corrected_phi = if params.preserve_volume {
        correct_phi_volume(
            &reinitialized_phi,
            params.target_volume,
            params.volume_band,
        )
    } else {
        reinitialized_phi
    };
    let corrected_phi = enforce_phi_neumann(&corrected_phi);
    let next_flags = flags_from_phi(&corrected_phi, &state.flags);
    LevelSetState {
        phi: corrected_phi,
        velocity: projected_velocity,
        flags: next_flags,
    }
}

pub fn flags_from_phi(phi: &Field2, base_flags: &CellFlags) -> CellFlags {
    let grid = phi.grid();
    CellFlags::from_fn(grid, |x, y| match base_flags.get(x, y) {
        CellType::Solid => CellType::Solid,
        _ => {
            if phi.get(x, y) <= 0.0 {
                CellType::Fluid
            } else {
                CellType::Air
            }
        }
    })
}

pub fn phi_to_density(phi: &Field2, band: f32) -> Field2 {
    if band <= 0.0 {
        return phi.map(|value| if value <= 0.0 { 1.0 } else { 0.0 });
    }
    let edge0 = -band;
    let edge1 = band;
    phi.map(|value| 1.0 - smoothstep(edge0, edge1, value))
}

pub fn volume_from_phi(phi: &Field2, band: f32) -> f32 {
    if band <= 0.0 {
        return phi.sum_with(|value| if value <= 0.0 { 1.0 } else { 0.0 });
    }
    let edge0 = -band;
    let edge1 = band;
    phi.sum_with(|value| 1.0 - smoothstep(edge0, edge1, value))
}

pub fn reinitialize_phi(phi: &Field2, iterations: usize, dt: f32) -> Field2 {
    if iterations == 0 || dt == 0.0 {
        return phi.clone();
    }
    let phi0 = phi.clone();
    (0..iterations).fold(phi.clone(), |current, _| reinit_step(&current, &phi0, dt))
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

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn correct_phi_volume(phi: &Field2, target_volume: f32, band: f32) -> Field2 {
    if target_volume <= 0.0 || !target_volume.is_finite() {
        return phi.clone();
    }
    let (min_phi, max_phi) = phi.min_max();
    let mut low = -max_phi - band;
    let mut high = -min_phi + band;
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    let low_volume = volume_with_offset(phi, band, low);
    let high_volume = volume_with_offset(phi, band, high);
    if target_volume >= low_volume {
        return shift_phi(phi, low);
    }
    if target_volume <= high_volume {
        return shift_phi(phi, high);
    }
    let mut lo = low;
    let mut hi = high;
    let tol = target_volume * 1e-4 + 1e-4;
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        let mid_volume = volume_with_offset(phi, band, mid);
        if (mid_volume - target_volume).abs() <= tol {
            return shift_phi(phi, mid);
        }
        if mid_volume > target_volume {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    shift_phi(phi, 0.5 * (lo + hi))
}

fn shift_phi(phi: &Field2, offset: f32) -> Field2 {
    if offset == 0.0 {
        return phi.clone();
    }
    phi.map(|value| value + offset)
}

fn volume_with_offset(phi: &Field2, band: f32, offset: f32) -> f32 {
    if band <= 0.0 {
        return phi.sum_with(|value| if value + offset <= 0.0 { 1.0 } else { 0.0 });
    }
    let edge0 = -band;
    let edge1 = band;
    phi.sum_with(|value| 1.0 - smoothstep(edge0, edge1, value + offset))
}

fn smooth_sign(value: f32, dx: f32) -> f32 {
    value / (value * value + dx * dx).sqrt()
}

fn reinit_step(phi: &Field2, phi0: &Field2, dt: f32) -> Field2 {
    let grid = phi.grid();
    let dx = grid.dx();
    phi.map_with_index(|x, y, value| {
        let phi0_value = phi0.get(x, y);
        let s = smooth_sign(phi0_value, dx);
        if s == 0.0 {
            return value;
        }
        let grad = godunov_gradient(phi, x as i32, y as i32, s);
        value - dt * s * (grad - 1.0)
    })
}

fn godunov_gradient(phi: &Field2, x: i32, y: i32, sign: f32) -> f32 {
    let dx = phi.grid().dx();
    let phi_x_pos = (phi.sample_clamped(x + 1, y) - phi.sample_clamped(x, y)) / dx;
    let phi_x_neg = (phi.sample_clamped(x, y) - phi.sample_clamped(x - 1, y)) / dx;
    let phi_y_pos = (phi.sample_clamped(x, y + 1) - phi.sample_clamped(x, y)) / dx;
    let phi_y_neg = (phi.sample_clamped(x, y) - phi.sample_clamped(x, y - 1)) / dx;
    let (a, b) = if sign > 0.0 {
        let ax = phi_x_neg.max(0.0).powi(2) + phi_x_pos.min(0.0).powi(2);
        let ay = phi_y_neg.max(0.0).powi(2) + phi_y_pos.min(0.0).powi(2);
        (ax, ay)
    } else {
        let ax = phi_x_pos.max(0.0).powi(2) + phi_x_neg.min(0.0).powi(2);
        let ay = phi_y_pos.max(0.0).powi(2) + phi_y_neg.min(0.0).powi(2);
        (ax, ay)
    };
    (a + b).sqrt()
}

fn add_surface_tension_phi(
    velocity: &MacVelocity2,
    phi: &Field2,
    surface_tension: f32,
    band: f32,
    dt: f32,
) -> MacVelocity2 {
    if surface_tension == 0.0 || dt == 0.0 {
        return velocity.clone();
    }
    let grid = phi.grid();
    let dx = grid.dx();
    let eps = if band > 0.0 { band } else { 1.5 * dx };
    let normals = VecField2::from_fn(grid, |x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let grad = phi_gradient(phi, xi, yi);
        let mag = (grad.x * grad.x + grad.y * grad.y).sqrt();
        if mag > 1e-6 {
            Vec2::new(grad.x / mag, grad.y / mag)
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
    let delta = Field2::from_fn(grid, |x, y| smooth_delta(phi.get(x, y), eps));
    let force = VecField2::from_fn(grid, |x, y| {
        let n = normals.get(x, y);
        let kappa = curvature.get(x, y);
        let d = delta.get(x, y);
        n.scale(surface_tension * kappa * d)
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

fn smooth_delta(phi: f32, eps: f32) -> f32 {
    if eps <= 0.0 {
        return 0.0;
    }
    let abs_phi = phi.abs();
    if abs_phi >= eps {
        0.0
    } else {
        let t = std::f32::consts::PI * phi / eps;
        0.5 / eps * (1.0 + t.cos())
    }
}

fn phi_gradient(phi: &Field2, x: i32, y: i32) -> Vec2 {
    let dx = phi.grid().dx();
    let ddx = (phi.sample_clamped(x + 1, y) - phi.sample_clamped(x - 1, y)) / (2.0 * dx);
    let ddy = (phi.sample_clamped(x, y + 1) - phi.sample_clamped(x, y - 1)) / (2.0 * dx);
    Vec2::new(ddx, ddy)
}

fn extrapolate_velocity(
    velocity: &MacVelocity2,
    flags: &CellFlags,
    iterations: usize,
) -> MacVelocity2 {
    if iterations == 0 {
        return velocity.clone();
    }
    let grid = velocity.grid();
    let u_grid = velocity.u().grid();
    let v_grid = velocity.v().grid();
    let mut u_mask = build_u_face_mask(flags, grid.width());
    let mut v_mask = build_v_face_mask(flags, grid.height());
    let mut u_field = velocity.u().clone();
    let mut v_field = velocity.v().clone();
    for _ in 0..iterations {
        let next_u_mask = expand_mask(&u_mask, u_grid.width(), u_grid.height());
        let next_v_mask = expand_mask(&v_mask, v_grid.width(), v_grid.height());
        u_field = extrapolate_staggered(&u_field, &u_mask, &next_u_mask);
        v_field = extrapolate_staggered(&v_field, &v_mask, &next_v_mask);
        u_mask = next_u_mask;
        v_mask = next_v_mask;
    }
    MacVelocity2::from_components(grid, u_field, v_field)
}

fn build_u_face_mask(flags: &CellFlags, width: usize) -> Vec<bool> {
    let height = flags.grid().height();
    let u_width = width + 1;
    (0..u_width * height)
        .map(|i| {
            let x = i % u_width;
            let y = i / u_width;
            let left = if x == 0 {
                CellType::Air
            } else {
                flags.get(x - 1, y)
            };
            let right = if x == width {
                CellType::Air
            } else {
                flags.get(x, y)
            };
            matches!(left, CellType::Fluid) || matches!(right, CellType::Fluid)
        })
        .collect()
}

fn build_v_face_mask(flags: &CellFlags, height: usize) -> Vec<bool> {
    let width = flags.grid().width();
    let v_height = height + 1;
    (0..width * v_height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let bottom = if y == 0 {
                CellType::Air
            } else {
                flags.get(x, y - 1)
            };
            let top = if y == height {
                CellType::Air
            } else {
                flags.get(x, y)
            };
            matches!(bottom, CellType::Fluid) || matches!(top, CellType::Fluid)
        })
        .collect()
}

fn expand_mask(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    (0..mask.len())
        .map(|i| {
            if mask[i] {
                return true;
            }
            let x = i % width;
            let y = i / width;
            let mut any = false;
            if x > 0 && mask[i - 1] {
                any = true;
            }
            if x + 1 < width && mask[i + 1] {
                any = true;
            }
            if y > 0 && mask[i - width] {
                any = true;
            }
            if y + 1 < height && mask[i + width] {
                any = true;
            }
            any
        })
        .collect()
}

fn extrapolate_staggered(
    field: &StaggeredField2,
    mask: &[bool],
    next_mask: &[bool],
) -> StaggeredField2 {
    let grid = field.grid();
    let width = grid.width();
    let height = grid.height();
    field.map_with_index(|x, y, value| {
        let idx = y * width + x;
        if mask[idx] {
            return value;
        }
        if !next_mask[idx] {
            return value;
        }
        let mut sum = 0.0;
        let mut count = 0.0;
        if x > 0 {
            let nidx = idx - 1;
            if mask[nidx] {
                sum += field.get(x - 1, y);
                count += 1.0;
            }
        }
        if x + 1 < width {
            let nidx = idx + 1;
            if mask[nidx] {
                sum += field.get(x + 1, y);
                count += 1.0;
            }
        }
        if y > 0 {
            let nidx = idx - width;
            if mask[nidx] {
                sum += field.get(x, y - 1);
                count += 1.0;
            }
        }
        if y + 1 < height {
            let nidx = idx + width;
            if mask[nidx] {
                sum += field.get(x, y + 1);
                count += 1.0;
            }
        }
        if count > 0.0 {
            sum / count
        } else {
            value
        }
    })
}

fn enforce_phi_neumann(phi: &Field2) -> Field2 {
    let grid = phi.grid();
    let width = grid.width();
    let height = grid.height();
    Field2::from_fn(grid, |x, y| {
        let boundary = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
        if !boundary {
            return phi.get(x, y);
        }
        let ix = if x == 0 {
            1
        } else if x + 1 == width {
            width.saturating_sub(2)
        } else {
            x
        };
        let iy = if y == 0 {
            1
        } else if y + 1 == height {
            height.saturating_sub(2)
        } else {
            y
        };
        let ix = ix.min(width.saturating_sub(1));
        let iy = iy.min(height.saturating_sub(1));
        phi.get(ix, iy)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MacGrid2, Grid2};

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} to be within {tol} of {b}"
        );
    }

    #[test]
    fn flags_from_phi_respects_solids() {
        let grid = Grid2::new(2, 2, 1.0);
        let phi = Field2::from_fn(grid, |x, y| if x == 0 && y == 0 { -1.0 } else { 1.0 });
        let base = CellFlags::from_fn(grid, |x, y| {
            if x == 1 && y == 1 {
                CellType::Solid
            } else {
                CellType::Fluid
            }
        });
        let flags = flags_from_phi(&phi, &base);
        assert_eq!(flags.get(0, 0), CellType::Fluid);
        assert_eq!(flags.get(1, 0), CellType::Air);
        assert_eq!(flags.get(1, 1), CellType::Solid);
    }

    #[test]
    fn phi_to_density_blends_interface() {
        let grid = Grid2::new(3, 1, 1.0);
        let phi = Field2::from_fn(grid, |x, _| match x {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        });
        let density = phi_to_density(&phi, 1.0);
        assert_close(density.get(0, 0), 1.0, 1e-6);
        assert_close(density.get(1, 0), 0.5, 1e-6);
        assert_close(density.get(2, 0), 0.0, 1e-6);
    }

    #[test]
    fn reinitialize_phi_keeps_linear_distance() {
        let grid = Grid2::new(5, 5, 1.0);
        let phi = Field2::from_fn(grid, |x, _| x as f32 - 2.0);
        let reinit = reinitialize_phi(&phi, 5, 0.3);
        assert_close(reinit.get(3, 2), phi.get(3, 2), 1e-4);
        assert_close(reinit.get(1, 2), phi.get(1, 2), 1e-4);
    }

    #[test]
    fn level_set_step_keeps_phi_with_zero_dt() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let phi = Field2::new(grid.cell_grid(), -1.0);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let state = LevelSetState { phi, velocity, flags };
        let params = LevelSetParams {
            dt: 0.0,
            cfl: 0.5,
            viscosity: 0.0,
            pressure_iters: 5,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            surface_tension: 0.0,
            surface_tension_band: 0.0,
            reinit_iters: 0,
            reinit_dt: 0.0,
            extrapolation_iters: 0,
            preserve_volume: false,
            target_volume: 0.0,
            volume_band: 0.0,
        };
        let next = level_set_step(&state, params);
        assert_eq!(next.phi, state.phi);
    }

    #[test]
    fn level_set_step_preserves_volume_when_enabled() {
        let grid = MacGrid2::new(6, 6, 1.0);
        let base_phi = Field2::from_fn(grid.cell_grid(), |_, y| y as f32 - 2.5);
        let target = volume_from_phi(&base_phi, 1.0);
        let shifted_phi = base_phi.map(|value| value + 1.2);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let state = LevelSetState {
            phi: shifted_phi,
            velocity,
            flags,
        };
        let params = LevelSetParams {
            dt: 0.0,
            cfl: 0.5,
            viscosity: 0.0,
            pressure_iters: 5,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            surface_tension: 0.0,
            surface_tension_band: 0.0,
            reinit_iters: 0,
            reinit_dt: 0.0,
            extrapolation_iters: 0,
            preserve_volume: true,
            target_volume: target,
            volume_band: 1.0,
        };
        let next = level_set_step(&state, params);
        let volume = volume_from_phi(&next.phi, 1.0);
        assert_close(volume, target, 1e-2);
    }

    #[test]
    fn enforce_phi_neumann_copies_interior_to_boundary() {
        let grid = Grid2::new(4, 4, 1.0);
        let phi = Field2::from_fn(grid, |x, y| (x + y * 10) as f32);
        let enforced = enforce_phi_neumann(&phi);
        assert_close(enforced.get(0, 1), phi.get(1, 1), 1e-6);
        assert_close(enforced.get(3, 2), phi.get(2, 2), 1e-6);
        assert_close(enforced.get(2, 0), phi.get(2, 1), 1e-6);
        assert_close(enforced.get(1, 3), phi.get(1, 2), 1e-6);
    }
}
