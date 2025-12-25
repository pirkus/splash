use crate::{
    apply_domain_boundaries, apply_solid_boundaries, mac::apply_domain_boundaries_into,
    mac::apply_solid_boundaries_into, mac_sim::{
        add_body_force, add_body_force_into, add_body_force_into_masked, add_body_force_masked,
        add_buoyancy, add_buoyancy_into, add_surface_tension, add_surface_tension_into,
        add_vorticity_confinement_in_place, advect_scalar_bfecc_into_with_flags,
        advect_scalar_bfecc_with_flags, advect_scalar_conservative_into_with_flags,
        advect_scalar_into_with_flags, advect_scalar_with_flags,
        advect_velocity_bfecc_into_with_flags, advect_velocity_bfecc_with_flags,
        advect_velocity_into_with_flags, advect_velocity_with_flags,
        apply_fluid_mask, apply_fluid_mask_into, clear_air_enabled, extrapolate_velocity,
        apply_pressure_gradient_into,
        diffuse_scalar, diffuse_scalar_into, diffuse_velocity, diffuse_velocity_into,
        divergence_into, flags_from_density, sharpen_density,
        solve_pressure_pcg_with_flags_warm_start_into, AdvectionScheme, MacSimParams, MacSimState,
        PcgScratch, SurfaceTensionScratch, VorticityScratch,
    },
    CellFlags, CellType, Field2, Grid2, MacGrid2, MacVelocity2, Vec2,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub struct MultigridParams {
    pub levels: usize,
    pub cycles: usize,
    pub pre_smooth: usize,
    pub post_smooth: usize,
    pub coarse_smooth: usize,
    pub omega: f32,
    pub tol: f32,
    pub final_smooth: usize,
    pub pcg_iters: usize,
    pub pcg_tol: f32,
}

impl Default for MultigridParams {
    fn default() -> Self {
        Self {
            levels: 0,
            cycles: 2,
            pre_smooth: 2,
            post_smooth: 2,
            coarse_smooth: 20,
            omega: 0.8,
            tol: 0.0,
            final_smooth: 0,
            pcg_iters: 0,
            pcg_tol: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MacSimMgWorkspace {
    velocity_a: MacVelocity2,
    velocity_b: MacVelocity2,
    velocity_scratch: MacVelocity2,
    density_a: Field2,
    density_b: Field2,
    density_scratch: Field2,
    divergence: Field2,
    surface: SurfaceTensionScratch,
    vorticity: VorticityScratch,
    mg: MultigridWorkspace,
    pcg: PcgScratch,
}

impl MacSimMgWorkspace {
    pub fn new(grid: MacGrid2) -> Self {
        let cell_grid = grid.cell_grid();
        let velocity_a = MacVelocity2::new(grid, Vec2::zero());
        let velocity_b = MacVelocity2::new(grid, Vec2::zero());
        let velocity_scratch = MacVelocity2::new(grid, Vec2::zero());
        let density_a = Field2::new(cell_grid, 0.0);
        let density_b = Field2::new(cell_grid, 0.0);
        let density_scratch = Field2::new(cell_grid, 0.0);
        let divergence = Field2::new(cell_grid, 0.0);
        let surface = SurfaceTensionScratch::new(cell_grid);
        let vorticity = VorticityScratch::new(cell_grid);
        let mg = MultigridWorkspace::new(cell_grid, 0);
        let pcg = PcgScratch::new(cell_grid);
        Self {
            velocity_a,
            velocity_b,
            velocity_scratch,
            density_a,
            density_b,
            density_scratch,
            divergence,
            surface,
            vorticity,
            mg,
            pcg,
        }
    }
}

pub fn step_mg(state: &MacSimState, params: MacSimParams, mg_params: MultigridParams) -> MacSimState {
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
    let projected_velocity =
        project_with_flags_mg(&diffused_velocity, &flags, mg_params);
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
    let advected_density = match params.advection {
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
    };
    let diffused_density = diffuse_scalar(&advected_density, params.diffusion, dt_density);
    let mut continuous_density = if params.free_surface {
        let mut clamped = clamp_scalar(&diffused_density, 0.0, 1.0);
        if params.preserve_mass {
            let trail = trail_tuning(params);
            decay_low_density_in_place(&mut clamped, trail.cutoff, trail.decay);
            rescale_density_to_mass_masked_in_place(&mut clamped, target_mass, trail.min_fill);
        }
        clamped
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
    if params.free_surface && params.preserve_mass {
        let trail = trail_tuning(params);
        rescale_density_to_mass_masked_in_place(&mut continuous_density, target_mass, trail.min_fill);
    }
    MacSimState {
        density: continuous_density,
        velocity: projected_velocity,
        flags: next_flags,
    }
}

pub fn step_in_place_mg(
    state: &mut MacSimState,
    params: MacSimParams,
    mg_params: MultigridParams,
    scratch: &mut MacSimMgWorkspace,
) {
    static STEP_COUNT: AtomicUsize = AtomicUsize::new(0);
    let step_idx = STEP_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
    static LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
    let step_index = LOG_COUNT.fetch_add(1, Ordering::SeqCst);
    let debug = cfg!(debug_assertions) && step_index < 4;
    if debug {
        println!("mg:step={}", step_index + 1);
        log_density_stats("mg:init", &state.density);
        log_flag_counts("mg:init_flags", &state.flags);
    }
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
    if debug {
        log_flag_counts("mg:flags", &state.flags);
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
        if params.free_surface {
            add_body_force_into_masked(
                &mut scratch.velocity_a,
                &scratch.velocity_scratch,
                &state.density,
                params.body_force,
                dt,
            );
        } else {
            add_body_force_into(
                &mut scratch.velocity_a,
                &scratch.velocity_scratch,
                params.body_force,
                dt,
            );
        }
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
    if debug {
        println!(
            "mg:pre_project max_abs={:.3}",
            scratch.velocity_a.max_abs()
        );
    }
    project_with_flags_mg_into(
        &mut scratch.velocity_b,
        &scratch.velocity_a,
        &state.flags,
        mg_params,
        &mut scratch.divergence,
        &mut scratch.mg,
        &mut scratch.pcg,
    );
    if debug {
        log_density_stats("mg:divergence", &scratch.divergence);
        log_density_stats("mg:pressure", &scratch.mg.levels[0].pressure);
    }
    apply_domain_boundaries_into(&mut scratch.velocity_a, &scratch.velocity_b, params.boundaries);
    apply_solid_boundaries_into(&mut scratch.velocity_b, &scratch.velocity_a, &state.flags);
    apply_fluid_mask_into(&mut scratch.velocity_a, &scratch.velocity_b, &state.flags);
    state.velocity.clone_from(&scratch.velocity_a);
    if debug {
        println!(
            "mg:velocity max_abs={:.3}",
            state.velocity.max_abs()
        );
    }
    let advect_velocity = if params.free_surface {
        extrapolate_velocity(&state.velocity, &state.flags, 2)
    } else {
        state.velocity.clone()
    };
    let dt_density = clamp_dt(dt, params.cfl, &advect_velocity);
    if debug {
        println!("mg:dt density={:.4} vel_dt={:.4}", dt_density, dt);
    }
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
    if debug {
        log_density_stats("mg:after_advect", &scratch.density_a);
        log_density_stats("mg:after_diffuse", &scratch.density_b);
    }
    if params.free_surface {
        clamp_scalar_in_place(&mut scratch.density_b, 0.0, 1.0);
        if params.preserve_mass {
            let trail = trail_tuning(params);
            decay_low_density_in_place(&mut scratch.density_b, trail.cutoff, trail.decay);
            rescale_density_to_mass_masked_in_place(
                &mut scratch.density_b,
                target_mass,
                trail.min_fill,
            );
        }
    }
    let (delta_sum, delta_max) = if trail_debug_enabled() {
        density_delta_stats(&state.density, &scratch.density_b)
    } else {
        (0.0, 0.0)
    };
    state.density.clone_from(&scratch.density_b);
    if params.free_surface {
        sharpen_density_into(
            &mut scratch.density_scratch,
            &state.density,
            params.density_threshold,
            params.surface_band,
        );
        state.flags =
            flags_from_density(&scratch.density_scratch, &state.flags, params.density_threshold);
        if params.preserve_mass || clear_air_enabled() {
            mask_density_to_fluid_in_place(&mut state.density, &state.flags);
        }
        if params.preserve_mass {
            let trail = trail_tuning(params);
            rescale_density_to_mass_masked_in_place(&mut state.density, target_mass, trail.min_fill);
        }
        if trail_debug_enabled() && step_idx.is_multiple_of(10) {
            let trail = trail_tuning(params);
            log_trail_stats("mg:trail", step_idx, &state.density, trail.cutoff);
            log_trail_params("mg:trail_cfg", step_idx, trail);
            log_density_bins("mg:bins", step_idx, &state.density);
            log_air_density("mg:air", step_idx, &state.density, &state.flags, trail.cutoff);
            log_delta_stats(
                "mg:delta",
                step_idx,
                delta_sum,
                delta_max,
                state.velocity.max_abs(),
            );
        }
    }
    if debug {
        log_density_stats("mg:after_clamp", &state.density);
        log_flag_counts("mg:final_flags", &state.flags);
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

fn apply_external_forces(
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

fn clamp_scalar(field: &Field2, min: f32, max: f32) -> Field2 {
    field.map(|value| {
        if value.is_finite() {
            value.clamp(min, max)
        } else {
            min
        }
    })
}

fn clamp_scalar_in_place(field: &mut Field2, min: f32, max: f32) {
    field.update_with_index(|_x, _y, value| {
        if value.is_finite() {
            value.clamp(min, max)
        } else {
            min
        }
    });
}

fn mask_density_to_fluid_in_place(density: &mut Field2, flags: &CellFlags) {
    density.update_with_index(|x, y, value| {
        if flags.get(x, y) == CellType::Fluid && value.is_finite() {
            value
        } else {
            0.0
        }
    });
}

fn decay_low_density_in_place(density: &mut Field2, cutoff: f32, decay: f32) {
    let decay = decay.clamp(0.0, 1.0);
    density.update_with_index(|_x, _y, value| {
        if !value.is_finite() {
            0.0
        } else if value < cutoff {
            value * decay
        } else {
            value
        }
    });
}

#[derive(Clone, Copy, Debug)]
struct TrailTuning {
    cutoff: f32,
    decay: f32,
    min_fill: f32,
}

fn trail_tuning(params: MacSimParams) -> TrailTuning {
    let env = trail_env();
    let cutoff = env
        .cutoff
        .unwrap_or_else(|| (params.density_threshold * 0.4).clamp(0.05, 0.4))
        .clamp(0.0, 1.0);
    let decay = env.decay.unwrap_or(0.85).clamp(0.0, 1.0);
    let min_fill = env
        .min_fill
        .unwrap_or_else(|| (params.density_threshold * 0.6).clamp(0.1, 0.6))
        .clamp(0.0, 1.0);
    TrailTuning {
        cutoff,
        decay,
        min_fill,
    }
}

fn trail_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIM_DEBUG_TRAIL")
            .ok()
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

#[derive(Clone, Copy, Debug)]
struct TrailEnv {
    cutoff: Option<f32>,
    decay: Option<f32>,
    min_fill: Option<f32>,
}

fn trail_env() -> &'static TrailEnv {
    static ENV: OnceLock<TrailEnv> = OnceLock::new();
    ENV.get_or_init(|| TrailEnv {
        cutoff: env_f32("SIM_TRAIL_CUTOFF"),
        decay: env_f32("SIM_TRAIL_DECAY"),
        min_fill: env_f32("SIM_TRAIL_MIN_FILL"),
    })
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
}

fn log_trail_stats(label: &str, step: usize, density: &Field2, cutoff: f32) {
    let (sum, min_value, max_value, non_finite) = density.stats();
    let (low_sum, low_count, high_sum, high_count) = trail_stats(density, cutoff);
    println!(
        "{label} step={} cutoff={:.3} sum={:.2} min={:.3} max={:.3} low_sum={:.2} low_count={} high_sum={:.2} high_count={} nonfinite={}",
        step,
        cutoff,
        sum,
        min_value,
        max_value,
        low_sum,
        low_count,
        high_sum,
        high_count,
        non_finite
    );
}

fn trail_stats(density: &Field2, cutoff: f32) -> (f32, usize, f32, usize) {
    let grid = density.grid();
    let mut low_sum = 0.0;
    let mut low_count = 0;
    let mut high_sum = 0.0;
    let mut high_count = 0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let value = density.get(x, y);
            if !value.is_finite() {
                continue;
            }
            if value < cutoff {
                low_sum += value;
                low_count += 1;
            } else {
                high_sum += value;
                high_count += 1;
            }
        }
    }
    (low_sum, low_count, high_sum, high_count)
}

fn log_trail_params(label: &str, step: usize, trail: TrailTuning) {
    println!(
        "{label} step={} cutoff={:.3} decay={:.3} min_fill={:.3}",
        step, trail.cutoff, trail.decay, trail.min_fill
    );
}

fn log_density_bins(label: &str, step: usize, density: &Field2) {
    let (bins, counts) = density_bins(density);
    println!(
        "{label} step={} b0={:.2} b1={:.2} b2={:.2} b3={:.2} b4={:.2} c0={} c1={} c2={} c3={} c4={}",
        step,
        bins[0],
        bins[1],
        bins[2],
        bins[3],
        bins[4],
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4]
    );
}

fn density_bins(density: &Field2) -> ([f32; 5], [usize; 5]) {
    let grid = density.grid();
    let mut sums = [0.0; 5];
    let mut counts = [0_usize; 5];
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let value = density.get(x, y);
            if !value.is_finite() {
                continue;
            }
            let idx = if value < 0.2 {
                0
            } else if value < 0.4 {
                1
            } else if value < 0.6 {
                2
            } else if value < 0.8 {
                3
            } else {
                4
            };
            sums[idx] += value;
            counts[idx] += 1;
        }
    }
    (sums, counts)
}

fn log_air_density(
    label: &str,
    step: usize,
    density: &Field2,
    flags: &CellFlags,
    cutoff: f32,
) {
    let (air_sum, air_count, trail_sum, trail_count) = air_density_stats(density, flags, cutoff);
    println!(
        "{label} step={} air_sum={:.2} air_count={} trail_sum={:.2} trail_count={}",
        step, air_sum, air_count, trail_sum, trail_count
    );
}

fn air_density_stats(
    density: &Field2,
    flags: &CellFlags,
    cutoff: f32,
) -> (f32, usize, f32, usize) {
    let grid = density.grid();
    let mut air_sum = 0.0;
    let mut air_count = 0;
    let mut trail_sum = 0.0;
    let mut trail_count = 0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let value = density.get(x, y);
            if !value.is_finite() {
                continue;
            }
            if flags.get(x, y) == CellType::Air {
                air_sum += value;
                air_count += 1;
                if value < cutoff {
                    trail_sum += value;
                    trail_count += 1;
                }
            }
        }
    }
    (air_sum, air_count, trail_sum, trail_count)
}

fn density_delta_stats(prev: &Field2, next: &Field2) -> (f32, f32) {
    debug_assert_eq!(prev.grid(), next.grid(), "density grid mismatch");
    let grid = prev.grid();
    let mut sum = 0.0;
    let mut max = 0.0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let a = prev.get(x, y);
            let b = next.get(x, y);
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let diff = (a - b).abs();
            sum += diff;
            if diff > max {
                max = diff;
            }
        }
    }
    (sum, max)
}

fn log_delta_stats(label: &str, step: usize, delta_sum: f32, delta_max: f32, vel_max: f32) {
    println!(
        "{label} step={} delta_sum={:.4} delta_max={:.4} vel_max={:.4}",
        step, delta_sum, delta_max, vel_max
    );
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
        let scale = ((-error) / removable).min(1.0);
        density.update_with_index(|_x, _y, value| {
            if !value.is_finite() || value <= min_fill {
                value.clamp(0.0, 1.0)
            } else {
                (value - value * scale).clamp(0.0, 1.0)
            }
        });
    }
}

fn project_with_flags_mg(
    velocity: &MacVelocity2,
    flags: &CellFlags,
    mg_params: MultigridParams,
) -> MacVelocity2 {
    let mut divergence = Field2::new(flags.grid(), 0.0);
    divergence_into(&mut divergence, velocity);
    divergence.update_with_index(|x, y, value| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    let mut mg = MultigridWorkspace::new(flags.grid(), mg_params.levels);
    let mut pcg = PcgScratch::new(flags.grid());
    solve_pressure_multigrid(&mut mg, &divergence, flags, mg_params, &mut pcg);
    let mut projected = MacVelocity2::new(velocity.grid(), Vec2::zero());
    apply_pressure_gradient_into(
        &mut projected,
        velocity,
        &mg.levels[0].pressure,
    );
    projected
}

fn project_with_flags_mg_into(
    out: &mut MacVelocity2,
    velocity: &MacVelocity2,
    flags: &CellFlags,
    mg_params: MultigridParams,
    divergence: &mut Field2,
    mg: &mut MultigridWorkspace,
    pcg: &mut PcgScratch,
) {
    const PRESSURE_MAX_ABS: f32 = 1.0e4;
    static MG_DISABLED: AtomicBool = AtomicBool::new(false);
    divergence_into(divergence, velocity);
    divergence.update_with_index(|x, y, value| {
        if flags.get(x, y) != CellType::Fluid {
            0.0
        } else {
            value
        }
    });
    let mut use_fallback = MG_DISABLED.load(Ordering::SeqCst);
    if !use_fallback {
        solve_pressure_multigrid(mg, divergence, flags, mg_params, pcg);
        let (sum, min_value, max_value, non_finite) = mg.levels[0].pressure.stats();
        let max_abs = min_value.abs().max(max_value.abs());
        use_fallback = non_finite > 0 || !max_abs.is_finite() || max_abs > PRESSURE_MAX_ABS;
        if use_fallback {
            MG_DISABLED.store(true, Ordering::SeqCst);
            if cfg!(debug_assertions) {
                println!(
                    "mg:pressure fallback sum={:.2} min={:.4} max={:.4} nonfinite={}",
                    sum, min_value, max_value, non_finite
                );
            }
        }
    }
    if use_fallback {
        mg.levels[0].pressure.fill_with_index(|_, _| 0.0);
        let iters = mg_params.pcg_iters.max(80);
        let tol = if mg_params.pcg_tol > 0.0 {
            mg_params.pcg_tol
        } else {
            1e-4
        };
        solve_pressure_pcg_with_flags_warm_start_into(
            &mut mg.levels[0].pressure,
            divergence,
            flags,
            iters,
            tol,
            pcg,
        );
    }
    apply_pressure_gradient_into(out, velocity, &mg.levels[0].pressure);
}

#[derive(Clone, Debug)]
struct MultigridWorkspace {
    levels: Vec<MgLevel>,
}

#[derive(Clone, Debug)]
struct MgLevel {
    grid: Grid2,
    flags: CellFlags,
    pressure: Field2,
    rhs: Field2,
    residual: Field2,
    temp: Field2,
}

impl MultigridWorkspace {
    fn new(grid: Grid2, levels: usize) -> Self {
        let mut level_grids = Vec::new();
        level_grids.push(grid);
        let max_levels = max_levels_for_grid(grid);
        let target_levels = if levels == 0 {
            max_levels
        } else {
            levels.min(max_levels)
        };
        while level_grids.len() < target_levels {
            let prev = *level_grids.last().expect("missing grid");
            let next = coarsen_grid(prev);
            level_grids.push(next);
        }
        let levels = level_grids
            .into_iter()
            .map(|grid| MgLevel {
                grid,
                flags: CellFlags::new(grid, CellType::Fluid),
                pressure: Field2::new(grid, 0.0),
                rhs: Field2::new(grid, 0.0),
                residual: Field2::new(grid, 0.0),
                temp: Field2::new(grid, 0.0),
            })
            .collect();
        Self { levels }
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn update_flags(&mut self, fine_flags: &CellFlags, levels: usize) {
        if levels == 0 {
            return;
        }
        self.levels[0].flags.clone_from(fine_flags);
        for level in 1..levels {
            let (fine, coarse) = {
                let (left, right) = self.levels.split_at_mut(level);
                (&left[level - 1], &mut right[0])
            };
            coarsen_flags(&fine.flags, &mut coarse.flags);
        }
    }
}

fn solve_pressure_multigrid(
    mg: &mut MultigridWorkspace,
    divergence: &Field2,
    flags: &CellFlags,
    params: MultigridParams,
    pcg: &mut PcgScratch,
) {
    let levels = effective_levels(mg, params);
    if levels == 0 {
        return;
    }
    mg.update_flags(flags, levels);
    let fine = &mut mg.levels[0];
    fine.rhs.fill_with_index(|x, y| {
        if fine.flags.get(x, y) == CellType::Fluid {
            -divergence.get(x, y)
        } else {
            0.0
        }
    });
    fine.pressure.fill_with_index(|_, _| 0.0);
    let cycles = params.cycles.max(1);
    for _ in 0..cycles {
        v_cycle(mg, 0, levels, params);
        if params.tol > 0.0 {
            let fine = &mut mg.levels[0];
            compute_residual(fine);
            let avg = residual_average(fine);
            if avg <= params.tol {
                break;
            }
        }
    }
    if params.final_smooth > 0 {
        smooth_jacobi(&mut mg.levels[0], params.omega, params.final_smooth);
    }
    if params.pcg_iters > 0 {
        solve_pressure_pcg_with_flags_warm_start_into(
            &mut mg.levels[0].pressure,
            divergence,
            flags,
            params.pcg_iters,
            params.pcg_tol,
            pcg,
        );
    }
}

fn effective_levels(mg: &MultigridWorkspace, params: MultigridParams) -> usize {
    let max_levels = mg.level_count();
    if params.levels == 0 {
        max_levels
    } else {
        params.levels.min(max_levels)
    }
}

fn v_cycle(mg: &mut MultigridWorkspace, level: usize, levels: usize, params: MultigridParams) {
    let is_coarse = level + 1 >= levels;
    if is_coarse {
        smooth_jacobi(&mut mg.levels[level], params.omega, params.coarse_smooth);
        return;
    }
    smooth_jacobi(&mut mg.levels[level], params.omega, params.pre_smooth);
    compute_residual(&mut mg.levels[level]);
    {
        let (left, right) = mg.levels.split_at_mut(level + 1);
        let fine = &left[level];
        let coarse = &mut right[0];
        restrict_residual(fine, coarse);
        coarse.pressure.fill_with_index(|_, _| 0.0);
    }
    v_cycle(mg, level + 1, levels, params);
    {
        let (left, right) = mg.levels.split_at_mut(level + 1);
        let fine = &mut left[level];
        let coarse = &right[0];
        prolongate_and_add(coarse, fine);
    }
    smooth_jacobi(&mut mg.levels[level], params.omega, params.post_smooth);
}

fn smooth_jacobi(level: &mut MgLevel, omega: f32, iterations: usize) {
    if iterations == 0 {
        return;
    }
    let grid = level.grid;
    let dx2 = grid.dx() * grid.dx();
    for _ in 0..iterations {
        let pressure = &level.pressure;
        let rhs = &level.rhs;
        let flags = &level.flags;
        level.temp.fill_with_index(|x, y| {
            if flags.get(x, y) != CellType::Fluid {
                return 0.0;
            }
            let (sum, count) = neighbor_sum_count(pressure, flags, x, y);
            if count == 0.0 {
                0.0
            } else {
                (sum + rhs.get(x, y) * dx2) / count
            }
        });
        if (omega - 1.0).abs() > 1e-6 {
            level.temp.update_with_index(|x, y, value| {
                let old = pressure.get(x, y);
                (1.0 - omega) * old + omega * value
            });
        }
        std::mem::swap(&mut level.pressure, &mut level.temp);
    }
}

fn compute_residual(level: &mut MgLevel) {
    let grid = level.grid;
    let dx2 = grid.dx() * grid.dx();
    let pressure = &level.pressure;
    let rhs = &level.rhs;
    let flags = &level.flags;
    level.residual.fill_with_index(|x, y| {
        if flags.get(x, y) != CellType::Fluid {
            return 0.0;
        }
        let (sum, count) = neighbor_sum_count(pressure, flags, x, y);
        if count == 0.0 {
            0.0
        } else {
            let ax = (count * pressure.get(x, y) - sum) / dx2;
            rhs.get(x, y) - ax
        }
    });
}

fn restrict_residual(fine: &MgLevel, coarse: &mut MgLevel) {
    let fine_grid = fine.grid;
    let fine_w = fine_grid.width();
    let fine_h = fine_grid.height();
    coarse.rhs.fill_with_index(|cx, cy| {
        if coarse.flags.get(cx, cy) != CellType::Fluid {
            return 0.0;
        }
        let fx = cx * 2;
        let fy = cy * 2;
        let mut sum = 0.0;
        let mut samples = 0.0;
        for dy in 0..2 {
            for dx in 0..2 {
                let x = fx + dx;
                let y = fy + dy;
                if x < fine_w
                    && y < fine_h
                    && fine.flags.get(x, y) == CellType::Fluid
                {
                    sum += fine.residual.get(x, y);
                    samples += 1.0;
                }
            }
        }
        if samples == 0.0 {
            0.0
        } else {
            sum / samples
        }
    });
}

fn prolongate_and_add(coarse: &MgLevel, fine: &mut MgLevel) {
    let coarse_w = coarse.grid.width();
    let coarse_h = coarse.grid.height();
    fine.temp.fill_with_index(|x, y| {
        if fine.flags.get(x, y) != CellType::Fluid {
            return fine.pressure.get(x, y);
        }
        let gx = x as f32 * 0.5;
        let gy = y as f32 * 0.5;
        let cx = gx.floor() as usize;
        let cy = gy.floor() as usize;
        let tx = gx - cx as f32;
        let ty = gy - cy as f32;
        let c00 = coarse_value(coarse, cx, cy, coarse_w, coarse_h);
        let c10 = coarse_value(coarse, cx + 1, cy, coarse_w, coarse_h);
        let c01 = coarse_value(coarse, cx, cy + 1, coarse_w, coarse_h);
        let c11 = coarse_value(coarse, cx + 1, cy + 1, coarse_w, coarse_h);
        let interp_x0 = c00 + (c10 - c00) * tx;
        let interp_x1 = c01 + (c11 - c01) * tx;
        let correction = interp_x0 + (interp_x1 - interp_x0) * ty;
        fine.pressure.get(x, y) + correction
    });
    std::mem::swap(&mut fine.pressure, &mut fine.temp);
}

fn coarse_value(
    coarse: &MgLevel,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> f32 {
    let cx = x.min(width.saturating_sub(1));
    let cy = y.min(height.saturating_sub(1));
    if coarse.flags.get(cx, cy) != CellType::Fluid {
        0.0
    } else {
        coarse.pressure.get(cx, cy)
    }
}

fn residual_average(level: &MgLevel) -> f32 {
    let width = level.grid.width();
    let height = level.grid.height();
    let mut sum = 0.0;
    let mut count = 0.0;
    for y in 0..height {
        for x in 0..width {
            if level.flags.get(x, y) == CellType::Fluid {
                sum += level.residual.get(x, y).abs();
                count += 1.0;
            }
        }
    }
    if count == 0.0 {
        0.0
    } else {
        sum / count
    }
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

fn coarsen_grid(grid: Grid2) -> Grid2 {
    let width = grid.width().div_ceil(2).max(2);
    let height = grid.height().div_ceil(2).max(2);
    Grid2::new(width, height, grid.dx() * 2.0)
}

fn max_levels_for_grid(grid: Grid2) -> usize {
    let mut width = grid.width();
    let mut height = grid.height();
    let mut levels = 1;
    while width > 2 && height > 2 {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
        levels += 1;
    }
    levels
}

fn coarsen_flags(fine: &CellFlags, coarse: &mut CellFlags) {
    let fine_grid = fine.grid();
    let fine_w = fine_grid.width();
    let fine_h = fine_grid.height();
    coarse.fill_with_index(|x, y| {
        let fx = x * 2;
        let fy = y * 2;
        let mut has_solid = false;
        let mut has_fluid = false;
        for dy in 0..2 {
            for dx in 0..2 {
                let nx = fx + dx;
                let ny = fy + dy;
                if nx >= fine_w || ny >= fine_h {
                    continue;
                }
                match fine.get(nx, ny) {
                    CellType::Solid => has_solid = true,
                    CellType::Fluid => has_fluid = true,
                    CellType::Air => {}
                }
            }
        }
        if has_solid {
            CellType::Solid
        } else if has_fluid {
            CellType::Fluid
        } else {
            CellType::Air
        }
    });
}

fn log_density_stats(label: &str, field: &Field2) {
    let (sum, min_value, max_value, non_finite) = field.stats();
    println!(
        "{label} sum={:.2} min={:.4} max={:.4} nonfinite={}",
        sum, min_value, max_value, non_finite
    );
}

fn log_flag_counts(label: &str, flags: &CellFlags) {
    let grid = flags.grid();
    let mut fluid = 0;
    let mut air = 0;
    let mut solid = 0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            match flags.get(x, y) {
                CellType::Fluid => fluid += 1,
                CellType::Air => air += 1,
                CellType::Solid => solid += 1,
            }
        }
    }
    println!(
        "{label} fluid={} air={} solid={}",
        fluid, air, solid
    );
}

fn sharpen_density_into(out: &mut Field2, density: &Field2, threshold: f32, band: f32) {
    debug_assert_eq!(out.grid(), density.grid(), "density grid mismatch");
    if band <= 0.0 {
        out.fill_with_index(|x, y| {
            if density.get(x, y) >= threshold {
                1.0
            } else {
                0.0
            }
        });
        return;
    }
    let edge0 = (threshold - band).max(0.0);
    let edge1 = (threshold + band).min(1.0);
    if edge1 <= edge0 {
        out.fill_with_index(|x, y| {
            if density.get(x, y) >= threshold {
                1.0
            } else {
                0.0
            }
        });
        return;
    }
    let inv = 1.0 / (edge1 - edge0);
    out.fill_with_index(|x, y| {
        let value = density.get(x, y);
        let t = ((value - edge0) * inv).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundaryConfig, StaggeredField2};

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} to be within {tol} of {b}"
        );
    }

    #[test]
    fn multigrid_projection_reduces_divergence() {
        let grid = MacGrid2::new(16, 16, 1.0);
        let u = StaggeredField2::from_fn(grid.u_grid(), |x, _y| x as f32 * 0.05);
        let v = StaggeredField2::from_fn(grid.v_grid(), |_x, y| y as f32 * 0.05);
        let velocity = MacVelocity2::from_components(grid, u, v);
        let flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
        let mut divergence = Field2::new(grid.cell_grid(), 0.0);
        divergence_into(&mut divergence, &velocity);
        let before = divergence.abs_sum();
        let mg_params = MultigridParams {
            cycles: 3,
            tol: 0.0,
            final_smooth: 0,
            pcg_iters: 0,
            pcg_tol: 0.0,
            ..MultigridParams::default()
        };
        let mut mg = MultigridWorkspace::new(grid.cell_grid(), mg_params.levels);
        let mut pcg = PcgScratch::new(grid.cell_grid());
        solve_pressure_multigrid(&mut mg, &divergence, &flags, mg_params, &mut pcg);
        let mut projected = MacVelocity2::new(grid, Vec2::zero());
        apply_pressure_gradient_into(&mut projected, &velocity, &mg.levels[0].pressure);
        divergence_into(&mut divergence, &projected);
        let after = divergence.abs_sum();
        assert!(after < before);
    }

    #[test]
    fn free_surface_preserves_mass_with_multigrid() {
        let grid = MacGrid2::new(32, 32, 1.0);
        let cutoff = (grid.height() as f32 * 0.5).round() as usize;
        let density = Field2::from_fn(grid.cell_grid(), |_x, y| {
            if y < cutoff { 1.0 } else { 0.0 }
        });
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
        let mg_params = MultigridParams {
            cycles: 3,
            tol: 0.0,
            final_smooth: 0,
            pcg_iters: 0,
            pcg_tol: 0.0,
            ..MultigridParams::default()
        };
        let mut state = MacSimState {
            density,
            velocity,
            flags,
        };
        let before = state.density.sum();
        let mut workspace = MacSimMgWorkspace::new(grid);
        for _ in 0..12 {
            step_in_place_mg(&mut state, params, mg_params, &mut workspace);
        }
        let after = state.density.sum();
        assert_close(after, before, 1e-2);
    }
}
