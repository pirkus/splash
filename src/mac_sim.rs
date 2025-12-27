use crate::{
    apply_domain_boundaries, apply_solid_boundaries, mac::apply_domain_boundaries_into,
    mac::apply_solid_boundaries_into, BoundaryConfig, CellFlags, CellType, Field2, Grid2,
    MacGrid2, MacVelocity2, StaggeredField2, Vec2, vec_field::VecField2,
};
use std::sync::OnceLock;

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
    pub vorticity_strength: f32,
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

#[derive(Clone, Debug)]
pub struct MacSimWorkspace {
    velocity_a: MacVelocity2,
    velocity_b: MacVelocity2,
    velocity_scratch: MacVelocity2,
    density_a: Field2,
    density_b: Field2,
    density_scratch: Field2,
    divergence: Field2,
    pressure: Field2,
    pcg: PcgScratch,
    surface: SurfaceTensionScratch,
    vorticity: VorticityScratch,
}

#[derive(Clone, Debug)]
pub struct PcgScratch {
    inv_diag: Field2,
    r: Field2,
    z: Field2,
    p: Field2,
    ap: Field2,
}

#[derive(Clone, Debug)]
pub struct SurfaceTensionScratch {
    pub normals: VecField2,
    pub curvature: Field2,
    pub grad_mag: Field2,
    pub force: VecField2,
}

#[derive(Clone, Debug)]
pub(crate) struct VorticityScratch {
    omega: Field2,
    force: VecField2,
}

impl SurfaceTensionScratch {
    pub fn new(grid: Grid2) -> Self {
        Self {
            normals: VecField2::new(grid, Vec2::zero()),
            curvature: Field2::new(grid, 0.0),
            grad_mag: Field2::new(grid, 0.0),
            force: VecField2::new(grid, Vec2::zero()),
        }
    }
}

impl VorticityScratch {
    pub fn new(grid: Grid2) -> Self {
        Self {
            omega: Field2::new(grid, 0.0),
            force: VecField2::new(grid, Vec2::zero()),
        }
    }
}

impl PcgScratch {
    pub fn new(grid: Grid2) -> Self {
        Self {
            inv_diag: Field2::new(grid, 0.0),
            r: Field2::new(grid, 0.0),
            z: Field2::new(grid, 0.0),
            p: Field2::new(grid, 0.0),
            ap: Field2::new(grid, 0.0),
        }
    }
}

impl MacSimWorkspace {
    pub fn new(grid: MacGrid2) -> Self {
        let cell_grid = grid.cell_grid();
        let velocity_a = MacVelocity2::new(grid, Vec2::zero());
        let velocity_b = MacVelocity2::new(grid, Vec2::zero());
        let velocity_scratch = MacVelocity2::new(grid, Vec2::zero());
        let density_a = Field2::new(cell_grid, 0.0);
        let density_b = Field2::new(cell_grid, 0.0);
        let density_scratch = Field2::new(cell_grid, 0.0);
        let divergence = Field2::new(cell_grid, 0.0);
        let pressure = Field2::new(cell_grid, 0.0);
        let pcg = PcgScratch::new(cell_grid);
        let surface = SurfaceTensionScratch::new(cell_grid);
        let vorticity = VorticityScratch::new(cell_grid);
        Self {
            velocity_a,
            velocity_b,
            velocity_scratch,
            density_a,
            density_b,
            density_scratch,
            divergence,
            pressure,
            pcg,
            surface,
            vorticity,
        }
    }
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
    let mut dt = clamp_dt(params.dt, params.cfl, &bounded);
    let mut forced = apply_external_forces(&bounded, &state.density, params, dt);
    let dt_forced = clamp_dt(params.dt, params.cfl, &forced);
    if dt_forced < dt {
        dt = dt_forced;
        forced = apply_external_forces(&bounded, &state.density, params, dt);
    }
    let forced = apply_domain_boundaries(&forced, params.boundaries);
    let forced = apply_solid_boundaries(&forced, &flags);
    let forced = if params.free_surface {
        apply_fluid_mask(&forced, &flags)
    } else {
        forced
    };
    let advected_velocity = match params.advection {
        AdvectionScheme::SemiLagrangian => advect_velocity_with_flags(&forced, &forced, &flags, dt),
        AdvectionScheme::Bfecc => advect_velocity_bfecc_with_flags(&forced, &forced, &flags, dt),
    };
    let advected_velocity = if params.free_surface {
        apply_fluid_mask(&advected_velocity, &flags)
    } else {
        advected_velocity
    };
    let mut diffused_velocity = diffuse_velocity(&advected_velocity, params.viscosity, dt);
    if params.vorticity_strength != 0.0 && dt != 0.0 {
        let mut scratch = VorticityScratch::new(state.velocity.grid().cell_grid());
        add_vorticity_confinement_in_place(
            &mut diffused_velocity,
            params.vorticity_strength,
            dt,
            &mut scratch,
        );
    }
    let projected_velocity = project_with_flags(
        &diffused_velocity,
        &flags,
        params.pressure_iters,
        params.pressure_tol,
    );
    let projected_velocity = apply_domain_boundaries(&projected_velocity, params.boundaries);
    let projected_velocity = apply_solid_boundaries(&projected_velocity, &flags);
    let projected_velocity = if params.free_surface {
        apply_fluid_mask(&projected_velocity, &flags)
    } else {
        projected_velocity
    };
    let advect_velocity = if params.free_surface {
        extrapolate_velocity(&projected_velocity, &flags, 2)
    } else {
        projected_velocity.clone()
    };
    let dt_density = clamp_dt(dt, params.cfl, &advect_velocity);
    let advected_density = if params.free_surface {
        advect_scalar_conservative_with_flags(&state.density, &advect_velocity, &flags, dt_density)
    } else {
        match params.advection {
            AdvectionScheme::SemiLagrangian => advect_scalar_with_flags(
                &state.density,
                &advect_velocity,
                &flags,
                dt_density,
            ),
            AdvectionScheme::Bfecc => advect_scalar_bfecc_with_flags(
                &state.density,
                &advect_velocity,
                &flags,
                dt_density,
            ),
        }
    };
    let diffused_density = diffuse_scalar(&advected_density, params.diffusion, dt_density);
    let mut continuous_density = if params.free_surface {
        clamp_scalar(&diffused_density, 0.0, 1.0)
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
    if params.free_surface && (params.preserve_mass || clear_air_enabled()) {
        mask_density_to_fluid_in_place(&mut continuous_density, &next_flags);
    }
    if params.free_surface && params.surface_band > 0.0 && params.preserve_mass {
        rescale_density_to_mass_masked_in_place(
            &mut continuous_density,
            target_mass,
            mass_min_fill(params),
        );
    }
    MacSimState {
        density: continuous_density,
        velocity: projected_velocity,
        flags: next_flags,
    }
}

pub fn step_in_place(state: &mut MacSimState, params: MacSimParams, scratch: &mut MacSimWorkspace) {
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
    if params.free_surface {
        state.flags = flags_from_density(&flag_density, &state.flags, params.density_threshold);
    }
    scratch.velocity_a.clone_from(&state.velocity);
    apply_domain_boundaries_into(&mut scratch.velocity_b, &scratch.velocity_a, params.boundaries);
    apply_solid_boundaries_into(&mut scratch.velocity_a, &scratch.velocity_b, &state.flags);
    apply_fluid_mask_into(&mut scratch.velocity_b, &scratch.velocity_a, &state.flags);
    let mut dt = clamp_dt(params.dt, params.cfl, &scratch.velocity_b);
    scratch.velocity_scratch.clone_from(&scratch.velocity_b);
    add_body_force_into(&mut scratch.velocity_a, &scratch.velocity_b, params.body_force, dt);
    add_buoyancy_into(
        &mut scratch.velocity_b,
        &scratch.velocity_a,
        &state.density,
        params.buoyancy,
        params.ambient_density,
        dt,
    );
    if params.surface_tension != 0.0 && dt != 0.0 {
        add_surface_tension_into(
            &mut scratch.velocity_a,
            &scratch.velocity_b,
            &state.density,
            params.surface_tension,
            dt,
            &mut scratch.surface,
        );
    } else {
        scratch.velocity_a.clone_from(&scratch.velocity_b);
    }
    let dt_forced = clamp_dt(params.dt, params.cfl, &scratch.velocity_a);
    if dt_forced < dt {
        dt = dt_forced;
        add_body_force_into(
            &mut scratch.velocity_a,
            &scratch.velocity_scratch,
            params.body_force,
            dt,
        );
        add_buoyancy_into(
            &mut scratch.velocity_b,
            &scratch.velocity_a,
            &state.density,
            params.buoyancy,
            params.ambient_density,
            dt,
        );
        if params.surface_tension != 0.0 && dt != 0.0 {
            add_surface_tension_into(
                &mut scratch.velocity_a,
                &scratch.velocity_b,
                &state.density,
                params.surface_tension,
                dt,
                &mut scratch.surface,
            );
        } else {
            scratch.velocity_a.clone_from(&scratch.velocity_b);
        }
    }
    apply_domain_boundaries_into(&mut scratch.velocity_b, &scratch.velocity_a, params.boundaries);
    apply_solid_boundaries_into(&mut scratch.velocity_a, &scratch.velocity_b, &state.flags);
    apply_fluid_mask_into(&mut scratch.velocity_b, &scratch.velocity_a, &state.flags);
    match params.advection {
        AdvectionScheme::SemiLagrangian => {
            advect_velocity_into_with_flags(
                &mut scratch.velocity_a,
                &scratch.velocity_b,
                &scratch.velocity_b,
                &state.flags,
                dt,
            );
        }
        AdvectionScheme::Bfecc => {
            advect_velocity_bfecc_into_with_flags(
                &mut scratch.velocity_a,
                &mut scratch.velocity_scratch,
                &scratch.velocity_b,
                &scratch.velocity_b,
                &state.flags,
                dt,
            );
        }
    }
    apply_fluid_mask_into(&mut scratch.velocity_b, &scratch.velocity_a, &state.flags);
    diffuse_velocity_into(
        &mut scratch.velocity_a,
        &scratch.velocity_b,
        params.viscosity,
        dt,
    );
    add_vorticity_confinement_in_place(
        &mut scratch.velocity_a,
        params.vorticity_strength,
        dt,
        &mut scratch.vorticity,
    );
    project_with_flags_into(
        &mut scratch.velocity_b,
        &scratch.velocity_a,
        &state.flags,
        params.pressure_iters,
        params.pressure_tol,
        &mut scratch.divergence,
        &mut scratch.pressure,
        &mut scratch.pcg,
    );
    apply_domain_boundaries_into(&mut scratch.velocity_a, &scratch.velocity_b, params.boundaries);
    apply_solid_boundaries_into(&mut scratch.velocity_b, &scratch.velocity_a, &state.flags);
    apply_fluid_mask_into(&mut scratch.velocity_a, &scratch.velocity_b, &state.flags);
    state.velocity.clone_from(&scratch.velocity_a);
    let advect_velocity = if params.free_surface {
        extrapolate_velocity(&state.velocity, &state.flags, 2)
    } else {
        state.velocity.clone()
    };
    let dt_density = clamp_dt(dt, params.cfl, &advect_velocity);
    if params.free_surface {
        advect_scalar_conservative_into_with_flags(
            &mut scratch.density_a,
            &state.density,
            &advect_velocity,
            &state.flags,
            dt_density,
        );
    } else {
        match params.advection {
            AdvectionScheme::SemiLagrangian => {
                advect_scalar_into_with_flags(
                    &mut scratch.density_a,
                    &state.density,
                    &advect_velocity,
                    &state.flags,
                    dt_density,
                );
            }
            AdvectionScheme::Bfecc => {
                advect_scalar_bfecc_into_with_flags(
                    &mut scratch.density_a,
                    &mut scratch.density_scratch,
                    &state.density,
                    &advect_velocity,
                    &state.flags,
                    dt_density,
                );
            }
        }
    }
    diffuse_scalar_into(
        &mut scratch.density_b,
        &scratch.density_a,
        params.diffusion,
        dt_density,
    );
    if params.free_surface {
        clamp_scalar_in_place(&mut scratch.density_b, 0.0, 1.0);
    }
    state.density.clone_from(&scratch.density_b);
    if params.free_surface {
        let next_flag_density =
            sharpen_density(&state.density, params.density_threshold, params.surface_band);
        state.flags = flags_from_density(&next_flag_density, &state.flags, params.density_threshold);
        if params.preserve_mass || clear_air_enabled() {
            mask_density_to_fluid_in_place(&mut state.density, &state.flags);
        }
        if params.surface_band > 0.0 && params.preserve_mass {
            rescale_density_to_mass_masked_in_place(
                &mut state.density,
                target_mass,
                mass_min_fill(params),
            );
        }
    }
}

fn clamp_dt(dt: f32, cfl: f32, velocity: &MacVelocity2) -> f32 {
    if dt <= 0.0 || cfl <= 0.0 {
        return dt;
    }
    let max_vel = velocity.max_abs();
    if !max_vel.is_finite() {
        return 0.0;
    }
    if max_vel == 0.0 {
        return dt;
    }
    let dx = velocity.grid().dx();
    let limit = cfl * dx / max_vel;
    dt.min(limit)
}

pub(crate) fn clear_air_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIM_CLEAR_AIR")
            .ok()
            .map(|value| value != "0")
            .unwrap_or(true)
    })
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
    let with_body = if params.free_surface {
        add_body_force_masked(velocity, density, params.body_force, dt)
    } else {
        add_body_force(velocity, params.body_force, dt)
    };
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

pub fn add_body_force_masked(
    velocity: &MacVelocity2,
    density: &Field2,
    force: Vec2,
    dt: f32,
) -> MacVelocity2 {
    if dt == 0.0 || (force.x == 0.0 && force.y == 0.0) {
        return velocity.clone();
    }
    let u_grid = velocity.u().grid();
    let v_grid = velocity.v().grid();
    let u = velocity.u().map_with_index(|x, y, value| {
        let pos = u_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        value + force.x * dt * rho
    });
    let v = velocity.v().map_with_index(|x, y, value| {
        let pos = v_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        value + force.y * dt * rho
    });
    MacVelocity2::from_components(velocity.grid(), u, v)
}

pub fn add_body_force_into(out: &mut MacVelocity2, velocity: &MacVelocity2, force: Vec2, dt: f32) {
    if dt == 0.0 || (force.x == 0.0 && force.y == 0.0) {
        out.clone_from(velocity);
        return;
    }
    out.u_mut()
        .fill_with_index(|x, y| velocity.u().get(x, y) + force.x * dt);
    out.v_mut()
        .fill_with_index(|x, y| velocity.v().get(x, y) + force.y * dt);
}

pub fn add_body_force_into_masked(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    density: &Field2,
    force: Vec2,
    dt: f32,
) {
    if dt == 0.0 || (force.x == 0.0 && force.y == 0.0) {
        out.clone_from(velocity);
        return;
    }
    let u_grid = velocity.u().grid();
    let v_grid = velocity.v().grid();
    out.u_mut().fill_with_index(|x, y| {
        let pos = u_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        velocity.u().get(x, y) + force.x * dt * rho
    });
    out.v_mut().fill_with_index(|x, y| {
        let pos = v_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        velocity.v().get(x, y) + force.y * dt * rho
    });
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

pub fn add_buoyancy_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    density: &Field2,
    buoyancy: f32,
    ambient_density: f32,
    dt: f32,
) {
    if dt == 0.0 || buoyancy == 0.0 {
        out.clone_from(velocity);
        return;
    }
    let v_grid = velocity.v().grid();
    out.u_mut().clone_from(velocity.u());
    out.v_mut().fill_with_index(|x, y| {
        let pos = v_grid.index_position(x, y);
        let rho = density.sample_linear(pos);
        velocity.v().get(x, y) + (rho - ambient_density) * buoyancy * dt
    });
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

pub fn add_surface_tension_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    density: &Field2,
    surface_tension: f32,
    dt: f32,
    scratch: &mut SurfaceTensionScratch,
) {
    if surface_tension == 0.0 || dt == 0.0 {
        out.clone_from(velocity);
        return;
    }
    let grid = density.grid();
    let dx = grid.dx();
    let eps = 1e-6;
    scratch.normals.u_mut().fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let ddx = (density.sample_clamped(xi + 1, yi) - density.sample_clamped(xi - 1, yi))
            / (2.0 * dx);
        let ddy = (density.sample_clamped(xi, yi + 1) - density.sample_clamped(xi, yi - 1))
            / (2.0 * dx);
        let mag = (ddx * ddx + ddy * ddy).sqrt();
        if mag > eps {
            ddx / mag
        } else {
            0.0
        }
    });
    scratch.normals.v_mut().fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let ddx = (density.sample_clamped(xi + 1, yi) - density.sample_clamped(xi - 1, yi))
            / (2.0 * dx);
        let ddy = (density.sample_clamped(xi, yi + 1) - density.sample_clamped(xi, yi - 1))
            / (2.0 * dx);
        let mag = (ddx * ddx + ddy * ddy).sqrt();
        if mag > eps {
            ddy / mag
        } else {
            0.0
        }
    });
    scratch.curvature.fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let nx_r = scratch.normals.u().sample_clamped(xi + 1, yi);
        let nx_l = scratch.normals.u().sample_clamped(xi - 1, yi);
        let ny_u = scratch.normals.v().sample_clamped(xi, yi + 1);
        let ny_d = scratch.normals.v().sample_clamped(xi, yi - 1);
        -((nx_r - nx_l) + (ny_u - ny_d)) / (2.0 * dx)
    });
    scratch.grad_mag.fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let ddx = (density.sample_clamped(xi + 1, yi) - density.sample_clamped(xi - 1, yi))
            / (2.0 * dx);
        let ddy = (density.sample_clamped(xi, yi + 1) - density.sample_clamped(xi, yi - 1))
            / (2.0 * dx);
        (ddx * ddx + ddy * ddy).sqrt()
    });
    scratch.force.u_mut().fill_with_index(|x, y| {
        let n = scratch.normals.u().get(x, y);
        let kappa = scratch.curvature.get(x, y);
        let delta = scratch.grad_mag.get(x, y);
        surface_tension * kappa * delta * n
    });
    scratch.force.v_mut().fill_with_index(|x, y| {
        let n = scratch.normals.v().get(x, y);
        let kappa = scratch.curvature.get(x, y);
        let delta = scratch.grad_mag.get(x, y);
        surface_tension * kappa * delta * n
    });
    out.u_mut().fill_with_index(|x, y| {
        let pos = velocity.u().grid().index_position(x, y);
        velocity.u().get(x, y) + scratch.force.sample_linear(pos).x * dt
    });
    out.v_mut().fill_with_index(|x, y| {
        let pos = velocity.v().grid().index_position(x, y);
        velocity.v().get(x, y) + scratch.force.sample_linear(pos).y * dt
    });
}

fn compute_vorticity_into(out: &mut Field2, velocity: &MacVelocity2) {
    let grid = velocity.grid();
    let dx = grid.dx();
    let u = velocity.u();
    let v = velocity.v();
    out.fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let u_center = |cx: i32, cy: i32| {
            0.5 * (u.sample_clamped(cx, cy) + u.sample_clamped(cx + 1, cy))
        };
        let v_center = |cx: i32, cy: i32| {
            0.5 * (v.sample_clamped(cx, cy) + v.sample_clamped(cx, cy + 1))
        };
        let dv_dx = (v_center(xi + 1, yi) - v_center(xi - 1, yi)) / (2.0 * dx);
        let du_dy = (u_center(xi, yi + 1) - u_center(xi, yi - 1)) / (2.0 * dx);
        dv_dx - du_dy
    });
}

fn compute_vorticity_force_into(out: &mut VecField2, omega: &Field2, eps: f32) {
    let grid = omega.grid();
    let dx = grid.dx();
    let inv_2dx = 0.5 / dx;
    out.u_mut().fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let omega_c = omega.get(x, y);
        let grad_x = (omega.sample_clamped(xi + 1, yi).abs()
            - omega.sample_clamped(xi - 1, yi).abs())
            * inv_2dx;
        let grad_y = (omega.sample_clamped(xi, yi + 1).abs()
            - omega.sample_clamped(xi, yi - 1).abs())
            * inv_2dx;
        let mag = (grad_x * grad_x + grad_y * grad_y).sqrt();
        if mag > eps {
            (grad_y / mag) * omega_c
        } else {
            0.0
        }
    });
    out.v_mut().fill_with_index(|x, y| {
        let xi = x as i32;
        let yi = y as i32;
        let omega_c = omega.get(x, y);
        let grad_x = (omega.sample_clamped(xi + 1, yi).abs()
            - omega.sample_clamped(xi - 1, yi).abs())
            * inv_2dx;
        let grad_y = (omega.sample_clamped(xi, yi + 1).abs()
            - omega.sample_clamped(xi, yi - 1).abs())
            * inv_2dx;
        let mag = (grad_x * grad_x + grad_y * grad_y).sqrt();
        if mag > eps {
            (-grad_x / mag) * omega_c
        } else {
            0.0
        }
    });
}

pub(crate) fn add_vorticity_confinement_in_place(
    velocity: &mut MacVelocity2,
    strength: f32,
    dt: f32,
    scratch: &mut VorticityScratch,
) {
    if strength == 0.0 || dt == 0.0 {
        return;
    }
    let eps = 1e-6;
    compute_vorticity_into(&mut scratch.omega, velocity);
    compute_vorticity_force_into(&mut scratch.force, &scratch.omega, eps);
    let grid = velocity.grid();
    let w = grid.width() as i32;
    let h = grid.height() as i32;
    velocity.u_mut().update_with_index(|x, y, value| {
        let cx0 = (x as i32 - 1).clamp(0, w - 1) as usize;
        let cx1 = (x as i32).clamp(0, w - 1) as usize;
        let cy = (y as i32).clamp(0, h - 1) as usize;
        let fx = 0.5 * (scratch.force.get(cx0, cy).x + scratch.force.get(cx1, cy).x);
        value + strength * dt * fx
    });
    velocity.v_mut().update_with_index(|x, y, value| {
        let cx = (x as i32).clamp(0, w - 1) as usize;
        let cy0 = (y as i32 - 1).clamp(0, h - 1) as usize;
        let cy1 = (y as i32).clamp(0, h - 1) as usize;
        let fy = 0.5 * (scratch.force.get(cx, cy0).y + scratch.force.get(cx, cy1).y);
        value + strength * dt * fy
    });
}

fn cell_type_at_pos(flags: &CellFlags, pos: (f32, f32)) -> CellType {
    let grid = flags.grid();
    let dx = grid.dx();
    let max_x = grid.width() as f32 * dx;
    let max_y = grid.height() as f32 * dx;
    if pos.0 < 0.0 || pos.1 < 0.0 || pos.0 > max_x || pos.1 > max_y {
        return CellType::Solid;
    }
    let gx = pos.0 / dx - 0.5;
    let gy = pos.1 / dx - 0.5;
    let ix = gx.floor() as i32;
    let iy = gy.floor() as i32;
    let ix = ix.clamp(0, grid.width() as i32 - 1) as usize;
    let iy = iy.clamp(0, grid.height() as i32 - 1) as usize;
    flags.get(ix, iy)
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

pub fn advect_scalar_with_flags(
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let grid = field.grid();
    field.map_with_index(|x, y, value| {
        if flags.get(x, y) == CellType::Solid {
            return value;
        }
        let pos = grid.cell_center(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.sample_linear(back)
        }
    })
}

pub fn advect_scalar_into(out: &mut Field2, field: &Field2, velocity: &MacVelocity2, dt: f32) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let grid = field.grid();
    out.fill_with_index(|x, y| {
        let pos = grid.cell_center(x, y);
        let v = velocity.sample_linear(pos);
        field.sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
}

pub fn advect_scalar_into_with_flags(
    out: &mut Field2,
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let grid = field.grid();
    out.fill_with_index(|x, y| {
        let value = field.get(x, y);
        if flags.get(x, y) == CellType::Solid {
            return value;
        }
        let pos = grid.cell_center(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.sample_linear(back)
        }
    });
}

#[allow(dead_code)]
pub fn advect_scalar_conservative(field: &Field2, velocity: &MacVelocity2, dt: f32) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let mut out = Field2::new(field.grid(), 0.0);
    advect_scalar_conservative_into(&mut out, field, velocity, dt);
    out
}

pub fn advect_scalar_conservative_with_flags(
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let mut out = Field2::new(field.grid(), 0.0);
    advect_scalar_conservative_into_with_flags(&mut out, field, velocity, flags, dt);
    out
}

#[allow(dead_code)]
pub fn advect_scalar_conservative_into(
    out: &mut Field2,
    field: &Field2,
    velocity: &MacVelocity2,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let inv_dx = 1.0 / field.grid().dx();
    out.fill_with_index(|x, y| {
        let u_l = velocity.u().get(x, y);
        let u_r = velocity.u().get(x + 1, y);
        let v_b = velocity.v().get(x, y);
        let v_t = velocity.v().get(x, y + 1);
        let rho_l = if u_l > 0.0 {
            field.sample_clamped(x as i32 - 1, y as i32)
        } else {
            field.get(x, y)
        };
        let rho_r = if u_r > 0.0 {
            field.get(x, y)
        } else {
            field.sample_clamped(x as i32 + 1, y as i32)
        };
        let rho_b = if v_b > 0.0 {
            field.sample_clamped(x as i32, y as i32 - 1)
        } else {
            field.get(x, y)
        };
        let rho_t = if v_t > 0.0 {
            field.get(x, y)
        } else {
            field.sample_clamped(x as i32, y as i32 + 1)
        };
        let flux_l = u_l * rho_l;
        let flux_r = u_r * rho_r;
        let flux_b = v_b * rho_b;
        let flux_t = v_t * rho_t;
        let delta = (flux_r - flux_l) + (flux_t - flux_b);
        field.get(x, y) - dt * inv_dx * delta
    });
}

pub fn advect_scalar_conservative_into_with_flags(
    out: &mut Field2,
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let grid = field.grid();
    let inv_dx = 1.0 / grid.dx();
    let w = grid.width() as i32;
    let h = grid.height() as i32;
    out.fill_with_index(|x, y| {
        if flags.get(x, y) == CellType::Solid {
            return field.get(x, y);
        }
        let xi = x as i32;
        let yi = y as i32;
        let left_cell = if xi == 0 {
            CellType::Solid
        } else {
            flags.get((xi - 1) as usize, y)
        };
        let right_cell = if xi + 1 >= w {
            CellType::Solid
        } else {
            flags.get((xi + 1) as usize, y)
        };
        let bottom_cell = if yi == 0 {
            CellType::Solid
        } else {
            flags.get(x, (yi - 1) as usize)
        };
        let top_cell = if yi + 1 >= h {
            CellType::Solid
        } else {
            flags.get(x, (yi + 1) as usize)
        };
        let solid_l = left_cell == CellType::Solid || flags.get(x, y) == CellType::Solid;
        let solid_r = right_cell == CellType::Solid || flags.get(x, y) == CellType::Solid;
        let solid_b = bottom_cell == CellType::Solid || flags.get(x, y) == CellType::Solid;
        let solid_t = top_cell == CellType::Solid || flags.get(x, y) == CellType::Solid;
        let u_l = if solid_l { 0.0 } else { velocity.u().get(x, y) };
        let u_r = if solid_r { 0.0 } else { velocity.u().get(x + 1, y) };
        let v_b = if solid_b { 0.0 } else { velocity.v().get(x, y) };
        let v_t = if solid_t { 0.0 } else { velocity.v().get(x, y + 1) };
        let rho_l = if u_l > 0.0 {
            if left_cell == CellType::Solid {
                field.get(x, y)
            } else {
                field.sample_clamped(xi - 1, yi)
            }
        } else {
            field.get(x, y)
        };
        let rho_r = if u_r <= 0.0 && right_cell != CellType::Solid {
            field.sample_clamped(xi + 1, yi)
        } else {
            field.get(x, y)
        };
        let rho_b = if v_b > 0.0 {
            if bottom_cell == CellType::Solid {
                field.get(x, y)
            } else {
                field.sample_clamped(xi, yi - 1)
            }
        } else {
            field.get(x, y)
        };
        let rho_t = if v_t <= 0.0 && top_cell != CellType::Solid {
            field.sample_clamped(xi, yi + 1)
        } else {
            field.get(x, y)
        };
        let flux_l = if solid_l { 0.0 } else { u_l * rho_l };
        let flux_r = if solid_r { 0.0 } else { u_r * rho_r };
        let flux_b = if solid_b { 0.0 } else { v_b * rho_b };
        let flux_t = if solid_t { 0.0 } else { v_t * rho_t };
        let delta = (flux_r - flux_l) + (flux_t - flux_b);
        field.get(x, y) - dt * inv_dx * delta
    });
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

pub fn advect_scalar_bfecc_with_flags(
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) -> Field2 {
    if dt == 0.0 {
        return field.clone();
    }
    let forward = advect_scalar_with_flags(field, velocity, flags, dt);
    let backward = advect_scalar_with_flags(&forward, velocity, flags, -dt);
    let correction = field.add_scaled(&backward, -1.0);
    let corrected = forward.add_scaled(&correction, 0.5);
    clamp_scalar_to_neighbors_with_flags(field, &corrected, flags)
}

pub fn advect_scalar_bfecc_into(
    out: &mut Field2,
    scratch: &mut Field2,
    field: &Field2,
    velocity: &MacVelocity2,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    advect_scalar_into(out, field, velocity, dt);
    advect_scalar_into(scratch, out, velocity, -dt);
    clamp_scalar_to_neighbors_in_place(out, field, scratch);
}

pub fn advect_scalar_bfecc_into_with_flags(
    out: &mut Field2,
    scratch: &mut Field2,
    field: &Field2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    advect_scalar_into_with_flags(out, field, velocity, flags, dt);
    advect_scalar_into_with_flags(scratch, out, velocity, flags, -dt);
    clamp_scalar_to_neighbors_in_place_with_flags(out, field, scratch, flags);
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn rescale_density_to_mass_in_place(density: &mut Field2, target_mass: f32) {
    if target_mass <= 0.0 || !target_mass.is_finite() {
        return;
    }
    let current = density.sum();
    if current <= 0.0 || !current.is_finite() {
        return;
    }
    let scale = target_mass / current;
    density.scale_in_place(scale);
}

fn rescale_density_to_mass_masked_in_place(
    density: &mut Field2,
    target_mass: f32,
    min_fill: f32,
) {
    if target_mass <= 0.0 || !target_mass.is_finite() {
        return;
    }
    let current = density.sum();
    if current <= 0.0 || !current.is_finite() {
        return;
    }
    let error = target_mass - current;
    if error.abs() < 1e-3 {
        return;
    }
    if error > 0.0 {
        let capacity = density.sum_with(|value| {
            if value.is_finite() && value > min_fill {
                (1.0 - value).max(0.0)
            } else {
                0.0
            }
        });
        if capacity <= 0.0 || !capacity.is_finite() {
            return;
        }
        let scale = (error / capacity).min(1.0);
        density.update_with_index(|_x, _y, value| {
            if !value.is_finite() || value >= 1.0 || value <= min_fill {
                value.clamp(0.0, 1.0)
            } else {
                (value + (1.0 - value) * scale).clamp(0.0, 1.0)
            }
        });
    } else {
        let removable = density.sum_with(|value| {
            if value.is_finite() && value > min_fill {
                value.max(0.0)
            } else {
                0.0
            }
        });
        if removable <= 0.0 || !removable.is_finite() {
            return;
        }
        let scale = (-error / removable).min(1.0);
        density.update_with_index(|_x, _y, value| {
            if !value.is_finite() || value <= 0.0 || value <= min_fill {
                value.clamp(0.0, 1.0)
            } else {
                (value - value * scale).clamp(0.0, 1.0)
            }
        });
    }
}

fn mass_min_fill(params: MacSimParams) -> f32 {
    (params.density_threshold * 0.6).clamp(0.1, 0.6)
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

pub fn clamp_scalar_in_place(field: &mut Field2, min: f32, max: f32) {
    field.update_with_index(|_x, _y, value| {
        if value.is_finite() {
            value.clamp(min, max)
        } else {
            min
        }
    });
}

pub(crate) fn mask_density_to_fluid_in_place(density: &mut Field2, flags: &CellFlags) {
    density.update_with_index(|x, y, value| {
        if flags.get(x, y) == CellType::Fluid && value.is_finite() {
            value
        } else {
            0.0
        }
    });
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

fn clamp_scalar_to_neighbors_with_flags(
    source: &Field2,
    candidate: &Field2,
    flags: &CellFlags,
) -> Field2 {
    let grid = source.grid();
    let width = grid.width() as i32;
    let height = grid.height() as i32;
    candidate.map_with_index(|x, y, value| {
        if flags.get(x, y) == CellType::Solid {
            return source.get(x, y);
        }
        let xi = x as i32;
        let yi = y as i32;
        let mut min_value = source.get(x, y);
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
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            if flags.get(ux, uy) == CellType::Solid {
                continue;
            }
            let sample = source.get(ux, uy);
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

fn clamp_scalar_to_neighbors_in_place(out: &mut Field2, source: &Field2, backward: &Field2) {
    out.update_with_index(|x, y, forward| {
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
        let backward_value = backward.get(x, y);
        let corrected = forward + 0.5 * (source.get(x, y) - backward_value);
        let sanitized = if corrected.is_finite() { corrected } else { min_value };
        sanitized.clamp(min_value, max_value)
    });
}

fn clamp_scalar_to_neighbors_in_place_with_flags(
    out: &mut Field2,
    source: &Field2,
    backward: &Field2,
    flags: &CellFlags,
) {
    let grid = source.grid();
    let width = grid.width() as i32;
    let height = grid.height() as i32;
    out.update_with_index(|x, y, forward| {
        if flags.get(x, y) == CellType::Solid {
            return source.get(x, y);
        }
        let xi = x as i32;
        let yi = y as i32;
        let mut min_value = source.get(x, y);
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
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            if flags.get(ux, uy) == CellType::Solid {
                continue;
            }
            let sample = source.get(ux, uy);
            if sample.is_finite() {
                if sample < min_value {
                    min_value = sample;
                }
                if sample > max_value {
                    max_value = sample;
                }
            }
        }
        let backward_value = backward.get(x, y);
        let corrected = forward + 0.5 * (source.get(x, y) - backward_value);
        let sanitized = if corrected.is_finite() { corrected } else { min_value };
        sanitized.clamp(min_value, max_value)
    });
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

pub fn advect_velocity_with_flags(
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) -> MacVelocity2 {
    if dt == 0.0 {
        return field.clone();
    }
    let u_grid = field.u().grid();
    let v_grid = field.v().grid();
    let u = field.u().map_with_index(|x, y, value| {
        let pos = u_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.u().sample_linear(back)
        }
    });
    let v = field.v().map_with_index(|x, y, value| {
        let pos = v_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.v().sample_linear(back)
        }
    });
    MacVelocity2::from_components(field.grid(), u, v)
}

pub fn advect_velocity_into(
    out: &mut MacVelocity2,
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let u_grid = field.u().grid();
    let v_grid = field.v().grid();
    out.u_mut().fill_with_index(|x, y| {
        let pos = u_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        field
            .u()
            .sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
    out.v_mut().fill_with_index(|x, y| {
        let pos = v_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        field
            .v()
            .sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
}

pub fn advect_velocity_into_with_flags(
    out: &mut MacVelocity2,
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let u_grid = field.u().grid();
    let v_grid = field.v().grid();
    out.u_mut().fill_with_index(|x, y| {
        let value = field.u().get(x, y);
        let pos = u_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.u().sample_linear(back)
        }
    });
    out.v_mut().fill_with_index(|x, y| {
        let value = field.v().get(x, y);
        let pos = v_grid.index_position(x, y);
        let v = velocity.sample_linear(pos);
        let back = (pos.0 - v.x * dt, pos.1 - v.y * dt);
        if cell_type_at_pos(flags, back) == CellType::Solid {
            value
        } else {
            field.v().sample_linear(back)
        }
    });
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

pub fn advect_velocity_bfecc_with_flags(
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) -> MacVelocity2 {
    if dt == 0.0 {
        return field.clone();
    }
    let forward = advect_velocity_with_flags(field, velocity, flags, dt);
    let backward = advect_velocity_with_flags(&forward, velocity, flags, -dt);
    let correction = field.add_scaled(&backward, -1.0);
    let corrected = forward.add_scaled(&correction, 0.5);
    let u = clamp_staggered_to_neighbors(field.u(), corrected.u());
    let v = clamp_staggered_to_neighbors(field.v(), corrected.v());
    MacVelocity2::from_components(field.grid(), u, v)
}

pub fn advect_velocity_bfecc_into(
    out: &mut MacVelocity2,
    scratch: &mut MacVelocity2,
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    advect_velocity_into(out, field, velocity, dt);
    advect_velocity_into(scratch, out, velocity, -dt);
    clamp_staggered_to_neighbors_in_place(out.u_mut(), field.u(), scratch.u());
    clamp_staggered_to_neighbors_in_place(out.v_mut(), field.v(), scratch.v());
}

pub fn advect_velocity_bfecc_into_with_flags(
    out: &mut MacVelocity2,
    scratch: &mut MacVelocity2,
    field: &MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    dt: f32,
) {
    if dt == 0.0 {
        out.clone_from(field);
        return;
    }
    advect_velocity_into_with_flags(out, field, velocity, flags, dt);
    advect_velocity_into_with_flags(scratch, out, velocity, flags, -dt);
    clamp_staggered_to_neighbors_in_place(out.u_mut(), field.u(), scratch.u());
    clamp_staggered_to_neighbors_in_place(out.v_mut(), field.v(), scratch.v());
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

fn clamp_staggered_to_neighbors_in_place(
    out: &mut StaggeredField2,
    source: &StaggeredField2,
    backward: &StaggeredField2,
) {
    out.update_with_index(|x, y, forward| {
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
        let backward_value = backward.get(x, y);
        let corrected = forward + 0.5 * (source.get(x, y) - backward_value);
        let sanitized = if corrected.is_finite() { corrected } else { min_value };
        sanitized.clamp(min_value, max_value)
    });
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

pub fn apply_fluid_mask_into(velocity: &mut MacVelocity2, source: &MacVelocity2, flags: &CellFlags) {
    let grid = source.grid();
    let w = grid.width();
    let h = grid.height();
    velocity.u_mut().fill_with_index(|x, y| {
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
            source.u().get(x, y)
        }
    });
    velocity.v_mut().fill_with_index(|x, y| {
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
            source.v().get(x, y)
        }
    });
}

pub fn extrapolate_velocity(
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

pub fn diffuse_scalar(field: &Field2, diffusion: f32, dt: f32) -> Field2 {
    if diffusion == 0.0 || dt == 0.0 {
        return field.clone();
    }
    field.add_scaled(&field.laplacian(), diffusion * dt)
}

pub fn diffuse_scalar_into(out: &mut Field2, field: &Field2, diffusion: f32, dt: f32) {
    if diffusion == 0.0 || dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let dx2 = field.grid().dx() * field.grid().dx();
    let scale = diffusion * dt / dx2;
    out.fill_with_index(|x, y| {
        let left = field.sample_clamped(x as i32 - 1, y as i32);
        let right = field.sample_clamped(x as i32 + 1, y as i32);
        let down = field.sample_clamped(x as i32, y as i32 - 1);
        let up = field.sample_clamped(x as i32, y as i32 + 1);
        let center = field.get(x, y);
        center + (left + right + up + down - 4.0 * center) * scale
    });
}

pub fn diffuse_velocity(field: &MacVelocity2, viscosity: f32, dt: f32) -> MacVelocity2 {
    if viscosity == 0.0 || dt == 0.0 {
        return field.clone();
    }
    let u = diffuse_staggered(field.u(), viscosity, dt);
    let v = diffuse_staggered(field.v(), viscosity, dt);
    MacVelocity2::from_components(field.grid(), u, v)
}

pub fn diffuse_velocity_into(
    out: &mut MacVelocity2,
    field: &MacVelocity2,
    viscosity: f32,
    dt: f32,
) {
    if viscosity == 0.0 || dt == 0.0 {
        out.clone_from(field);
        return;
    }
    diffuse_staggered_into(out.u_mut(), field.u(), viscosity, dt);
    diffuse_staggered_into(out.v_mut(), field.v(), viscosity, dt);
}

pub fn diffuse_staggered(field: &StaggeredField2, diffusion: f32, dt: f32) -> StaggeredField2 {
    if diffusion == 0.0 || dt == 0.0 {
        return field.clone();
    }
    field.add_scaled(&field.laplacian(), diffusion * dt)
}

pub fn diffuse_staggered_into(
    out: &mut StaggeredField2,
    field: &StaggeredField2,
    diffusion: f32,
    dt: f32,
) {
    if diffusion == 0.0 || dt == 0.0 {
        out.clone_from(field);
        return;
    }
    let dx2 = field.grid().dx() * field.grid().dx();
    let scale = diffusion * dt / dx2;
    out.fill_with_index(|x, y| {
        let left = field.sample_clamped(x as i32 - 1, y as i32);
        let right = field.sample_clamped(x as i32 + 1, y as i32);
        let down = field.sample_clamped(x as i32, y as i32 - 1);
        let up = field.sample_clamped(x as i32, y as i32 + 1);
        let center = field.get(x, y);
        center + (left + right + up + down - 4.0 * center) * scale
    });
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

pub fn divergence_into(out: &mut Field2, velocity: &MacVelocity2) {
    let grid = velocity.grid();
    let dx = grid.dx();
    out.fill_with_index(|x, y| {
        let u_r = velocity.u().get(x + 1, y);
        let u_l = velocity.u().get(x, y);
        let v_u = velocity.v().get(x, y + 1);
        let v_d = velocity.v().get(x, y);
        (u_r - u_l + v_u - v_d) / dx
    });
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

#[allow(clippy::too_many_arguments)]
pub fn project_with_flags_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    iterations: usize,
    tol: f32,
    divergence: &mut Field2,
    pressure: &mut Field2,
    scratch: &mut PcgScratch,
) {
    divergence_into(divergence, velocity);
    divergence.update_with_index(|x, y, value| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    solve_pressure_pcg_with_flags_into(pressure, divergence, flags, iterations, tol, scratch);
    apply_pressure_gradient_into(out, velocity, pressure);
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
    let mut rz_old = z.mul_pointwise_into_and_dot_left(&r, &inv_diag);
    let mut p = z.clone();
    if rz_old == 0.0 {
        return x;
    }
    let tol_sq = tol * tol;
    let mut ap = Field2::new(grid, 0.0);
    for _ in 0..iterations {
        let denom = apply_negative_laplacian_with_flags_into_and_dot(&p, flags, &mut ap);
        if denom.abs() < 1e-20 {
            break;
        }
        let alpha = rz_old / denom;
        x.add_scaled_in_place(&p, alpha);
        let r_norm_sq = r.add_scaled_in_place_and_sum_sq(&ap, -alpha);
        if r_norm_sq <= tol_sq {
            break;
        }
        let rz_new = z.mul_pointwise_into_and_dot_left(&r, &inv_diag);
        let beta = rz_new / rz_old;
        p.scale_and_add_in_place(beta, &z);
        rz_old = rz_new;
    }
    x
}

pub fn solve_pressure_pcg_with_flags_into(
    out: &mut Field2,
    divergence: &Field2,
    flags: &CellFlags,
    iterations: usize,
    tol: f32,
    scratch: &mut PcgScratch,
) {
    let grid = divergence.grid();
    out.fill_with_index(|_, _| 0.0);
    scratch.inv_diag.fill_with_index(|x, y| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            let count = neighbor_count(flags, grid, x, y);
            if count == 0.0 {
                0.0
            } else {
                grid.dx() * grid.dx() / count
            }
        }
    });
    scratch.r.fill_with_index(|x, y| -divergence.get(x, y));
    let mut rz_old = scratch
        .z
        .mul_pointwise_into_and_dot_left(&scratch.r, &scratch.inv_diag);
    scratch.p.clone_from(&scratch.z);
    if rz_old == 0.0 {
        return;
    }
    let tol_sq = tol * tol;
    for _ in 0..iterations {
        let denom = apply_negative_laplacian_with_flags_into_and_dot(
            &scratch.p,
            flags,
            &mut scratch.ap,
        );
        if denom.abs() < 1e-20 {
            break;
        }
        let alpha = rz_old / denom;
        out.add_scaled_in_place(&scratch.p, alpha);
        let r_norm_sq = scratch.r.add_scaled_in_place_and_sum_sq(&scratch.ap, -alpha);
        if r_norm_sq <= tol_sq {
            break;
        }
        let rz_new = scratch
            .z
            .mul_pointwise_into_and_dot_left(&scratch.r, &scratch.inv_diag);
        let beta = rz_new / rz_old;
        scratch.p.scale_and_add_in_place(beta, &scratch.z);
        rz_old = rz_new;
    }
}

pub fn solve_pressure_pcg_with_flags_warm_start_into(
    out: &mut Field2,
    divergence: &Field2,
    flags: &CellFlags,
    iterations: usize,
    tol: f32,
    scratch: &mut PcgScratch,
) {
    if iterations == 0 {
        return;
    }
    let grid = divergence.grid();
    scratch.inv_diag.fill_with_index(|x, y| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            let count = neighbor_count(flags, grid, x, y);
            if count == 0.0 {
                0.0
            } else {
                grid.dx() * grid.dx() / count
            }
        }
    });
    let _ = apply_negative_laplacian_with_flags_into_and_dot(out, flags, &mut scratch.ap);
    scratch.r.fill_with_index(|x, y| -divergence.get(x, y) - scratch.ap.get(x, y));
    let mut rz_old = scratch
        .z
        .mul_pointwise_into_and_dot_left(&scratch.r, &scratch.inv_diag);
    scratch.p.clone_from(&scratch.z);
    if rz_old == 0.0 {
        return;
    }
    let tol_sq = tol * tol;
    for _ in 0..iterations {
        let denom = apply_negative_laplacian_with_flags_into_and_dot(
            &scratch.p,
            flags,
            &mut scratch.ap,
        );
        if denom.abs() < 1e-20 {
            break;
        }
        let alpha = rz_old / denom;
        out.add_scaled_in_place(&scratch.p, alpha);
        let r_norm_sq = scratch.r.add_scaled_in_place_and_sum_sq(&scratch.ap, -alpha);
        if r_norm_sq <= tol_sq {
            break;
        }
        let rz_new = scratch
            .z
            .mul_pointwise_into_and_dot_left(&scratch.r, &scratch.inv_diag);
        let beta = rz_new / rz_old;
        scratch.p.scale_and_add_in_place(beta, &scratch.z);
        rz_old = rz_new;
    }
}

fn apply_negative_laplacian_with_flags_into_and_dot(
    field: &Field2,
    flags: &CellFlags,
    out: &mut Field2,
) -> f32 {
    debug_assert_eq!(field.grid(), out.grid(), "field grid mismatch");
    let grid = field.grid();
    let dx2 = grid.dx() * grid.dx();
    out.fill_with_index_and_dot(field, |x, y| {
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
    })
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

pub fn apply_pressure_gradient_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    pressure: &Field2,
) {
    let grid = velocity.grid();
    let dx = grid.dx();
    let cell_grid = grid.cell_grid();
    out.u_mut().fill_with_index(|x, y| {
        let left = if x == 0 { 0 } else { x - 1 };
        let right = x.min(cell_grid.width() - 1);
        let p_l = pressure.get(left, y);
        let p_r = pressure.get(right, y);
        velocity.u().get(x, y) - (p_r - p_l) / dx
    });
    out.v_mut().fill_with_index(|x, y| {
        let bottom = if y == 0 { 0 } else { y - 1 };
        let top = y.min(cell_grid.height() - 1);
        let p_b = pressure.get(x, bottom);
        let p_t = pressure.get(x, top);
        velocity.v().get(x, y) - (p_t - p_b) / dx
    });
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

    fn pool_density(grid: MacGrid2, height_fraction: f32) -> Field2 {
        let cutoff = (grid.height() as f32 * height_fraction).round() as usize;
        Field2::from_fn(grid.cell_grid(), |_x, y| if y < cutoff { 1.0 } else { 0.0 })
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
    fn advect_scalar_with_flags_avoids_solids() {
        let grid = MacGrid2::new(4, 4, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| {
            if x == 1 && y == 1 {
                1.0
            } else {
                0.0
            }
        });
        let flags = CellFlags::from_fn(grid.cell_grid(), |x, y| {
            if x == 2 && y == 1 {
                CellType::Solid
            } else {
                CellType::Fluid
            }
        });
        let velocity = MacVelocity2::new(grid, Vec2::new(-1.0, 0.0));
        let advected = advect_scalar_with_flags(&density, &velocity, &flags, 1.0);
        assert_close(advected.get(1, 1), 1.0, 1e-6);
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
    fn mask_density_to_fluid_clears_air_cells() {
        let grid = MacGrid2::new(3, 3, 1.0);
        let mut density = Field2::new(grid.cell_grid(), 1.0);
        let flags = CellFlags::from_fn(grid.cell_grid(), |x, y| {
            if x == 1 && y == 1 {
                CellType::Air
            } else {
                CellType::Fluid
            }
        });
        mask_density_to_fluid_in_place(&mut density, &flags);
        assert_close(density.get(1, 1), 0.0, 1e-6);
        assert_close(density.get(0, 0), 1.0, 1e-6);
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
            vorticity_strength: 0.0,
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
            vorticity_strength: 0.0,
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
            vorticity_strength: 0.0,
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
    fn vorticity_confinement_modifies_velocity() {
        let grid = MacGrid2::new(16, 16, 1.0);
        let center = Vec2::new(
            grid.width() as f32 * 0.5 * grid.dx(),
            grid.height() as f32 * 0.5 * grid.dx(),
        );
        let u_grid = grid.u_grid();
        let v_grid = grid.v_grid();
        let mut velocity = MacVelocity2::from_components(
            grid,
            StaggeredField2::from_fn(u_grid, |x, y| {
                let (px, py) = u_grid.index_position(x, y);
                let dx = px - center.x;
                let dy = py - center.y;
                let falloff = (-(dx * dx + dy * dy) * 0.02).exp();
                -dy * falloff
            }),
            StaggeredField2::from_fn(v_grid, |x, y| {
                let (px, py) = v_grid.index_position(x, y);
                let dx = px - center.x;
                let dy = py - center.y;
                let falloff = (-(dx * dx + dy * dy) * 0.02).exp();
                dx * falloff
            }),
        );
        let before_u = velocity.u().clone();
        let before_v = velocity.v().clone();
        let mut scratch = VorticityScratch::new(grid.cell_grid());
        add_vorticity_confinement_in_place(&mut velocity, 2.0, 0.1, &mut scratch);
        let mut max_delta: f32 = 0.0;
        for y in 0..u_grid.height() {
            for x in 0..u_grid.width() {
                let delta = (velocity.u().get(x, y) - before_u.get(x, y)).abs();
                max_delta = max_delta.max(delta);
            }
        }
        for y in 0..v_grid.height() {
            for x in 0..v_grid.width() {
                let delta = (velocity.v().get(x, y) - before_v.get(x, y)).abs();
                max_delta = max_delta.max(delta);
            }
        }
        assert!(max_delta > 1e-6);
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
            vorticity_strength: 0.0,
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

    #[test]
    fn free_surface_preserves_mass_without_forces() {
        let grid = MacGrid2::new(32, 32, 1.0);
        let density = pool_density(grid, 0.5);
        let velocity = MacVelocity2::new(grid, Vec2::zero());
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let params = MacSimParams {
            dt: 0.1,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.0,
            vorticity_strength: 0.0,
            pressure_iters: 10,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: true,
            density_threshold: 0.5,
            surface_band: 0.1,
            preserve_mass: true,
        };
        let mut state = MacSimState {
            density,
            velocity,
            flags,
        };
        let before = state.density.sum();
        let mut workspace = MacSimWorkspace::new(grid);
        for _ in 0..15 {
            step_in_place(&mut state, params, &mut workspace);
        }
        let after = state.density.sum();
        assert_close(after, before, 1e-2);
    }

    #[test]
    fn free_surface_stays_finite_over_steps() {
        let grid = MacGrid2::new(24, 24, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| {
            let dx = x as f32 - 12.0;
            let dy = y as f32 - 12.0;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 6.0 { 1.0 } else { 0.0 }
        });
        let velocity = MacVelocity2::new(grid, Vec2::new(1.25, -0.75));
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let params = MacSimParams {
            dt: 0.08,
            cfl: 0.5,
            diffusion: 0.0,
            viscosity: 0.0,
            vorticity_strength: 0.0,
            pressure_iters: 10,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::Bfecc,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::zero(),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: true,
            density_threshold: 0.5,
            surface_band: 0.1,
            preserve_mass: false,
        };
        let mut state = MacSimState {
            density,
            velocity,
            flags,
        };
        let mut workspace = MacSimWorkspace::new(grid);
        for _ in 0..20 {
            step_in_place(&mut state, params, &mut workspace);
        }
        let (_sum, min_value, max_value, non_finite) = state.density.stats();
        assert_eq!(non_finite, 0);
        assert!(min_value >= -1e-4);
        assert!(max_value <= 1.0001);
    }

    #[test]
    fn step_in_place_matches_step() {
        let grid = MacGrid2::new(8, 6, 1.0);
        let density = Field2::from_fn(grid.cell_grid(), |x, y| (x + y) as f32 * 0.05);
        let velocity = MacVelocity2::new(grid, Vec2::new(0.5, -0.25));
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let params = MacSimParams {
            dt: 0.05,
            cfl: 0.5,
            diffusion: 0.01,
            viscosity: 0.02,
            vorticity_strength: 0.0,
            pressure_iters: 15,
            pressure_tol: 1e-5,
            advection: AdvectionScheme::SemiLagrangian,
            boundaries: BoundaryConfig::no_slip(),
            body_force: Vec2::new(0.0, -1.0),
            buoyancy: 0.0,
            ambient_density: 0.0,
            surface_tension: 0.0,
            free_surface: false,
            density_threshold: 0.5,
            surface_band: 0.0,
            preserve_mass: false,
        };
        let state = MacSimState {
            density: density.clone(),
            velocity: velocity.clone(),
            flags,
        };
        let mut in_place = MacSimState {
            density,
            velocity,
            flags: CellFlags::new(grid.cell_grid(), CellType::Fluid),
        };
        let mut workspace = MacSimWorkspace::new(grid);
        step_in_place(&mut in_place, params, &mut workspace);
        let stepped = step(&state, params);
        let density_diff = Field2::from_fn(grid.cell_grid(), |x, y| {
            (in_place.density.get(x, y) - stepped.density.get(x, y)).abs()
        })
        .sum();
        assert!(density_diff < 1e-5);
        let u_diff = in_place
            .velocity
            .u()
            .map_with_index(|x, y, _| {
                in_place.velocity.u().get(x, y) - stepped.velocity.u().get(x, y)
            })
            .abs_sum();
        let v_diff = in_place
            .velocity
            .v()
            .map_with_index(|x, y, _| {
                in_place.velocity.v().get(x, y) - stepped.velocity.v().get(x, y)
            })
            .abs_sum();
        assert!(u_diff + v_diff < 1e-5);
    }
}
