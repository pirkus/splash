use anyhow::Result;
use nav_stokes_sim::{
    advect_level_set_surface_in_place, flags_from_density, flags_from_phi, level_set_step_in_place,
    overlay_text, phi_to_density, divergence, step_in_place, step_in_place_mg, volume_from_phi,
    AdvectionScheme, BoundaryConfig, CellFlags, CellType, Field2, LevelSetParams, LevelSetState,
    LevelSetWorkspace, MacGrid2, MacSimMgWorkspace, MacSimParams, MacSimState, MacSimWorkspace,
    MacVelocity2, MultigridParams, StaggeredField2, Vec2, VulkanApp, GLYPH_HEIGHT, LINE_SPACING,
};
use std::sync::OnceLock;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum SimMode {
    Density,
    LevelSet,
    DensityMg,
}

impl SimMode {
    fn toggle(self) -> Self {
        match self {
            SimMode::Density => SimMode::LevelSet,
            SimMode::LevelSet => SimMode::DensityMg,
            SimMode::DensityMg => SimMode::Density,
        }
    }

    fn tag(self) -> char {
        match self {
            SimMode::Density => '1',
            SimMode::LevelSet => '2',
            SimMode::DensityMg => '3',
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct InitConfig {
    base_height: f32,
    drop_radius: f32,
    drop_center: (f32, f32),
    drop_speed: f32,
}

impl InitConfig {
    fn new(grid: MacGrid2) -> Self {
        let size = grid.width().min(grid.height()) as f32;
        let base_height = grid.height() as f32 * 0.5;
        let drop_radius = size * 0.08;
        let drop_center = (
            grid.width() as f32 * 0.5,
            base_height + drop_radius * 1.9,
        );
        let drop_speed = env_f32("SIM_DROP_SPEED").unwrap_or(-10.0);
        Self {
            base_height,
            drop_radius,
            drop_center,
            drop_speed,
        }
    }
}

fn initial_phi(grid: MacGrid2, config: InitConfig) -> Field2 {
    Field2::from_fn(grid.cell_grid(), |x, y| {
        let y_f = y as f32 + 0.5;
        let plane_phi = y_f - config.base_height;
        let dx = (x as f32 + 0.5) - config.drop_center.0;
        let dy = y_f - config.drop_center.1;
        let drop_phi = (dx * dx + dy * dy).sqrt() - config.drop_radius;
        plane_phi.min(drop_phi)
    })
}

fn initial_velocity(grid: MacGrid2, config: InitConfig) -> MacVelocity2 {
    let radius = config.drop_radius;
    let center = config.drop_center;
    let base_height = config.base_height;
    let drop_speed = config.drop_speed;
    let u = StaggeredField2::from_fn(grid.u_grid(), |_x, _y| 0.0);
    let v = StaggeredField2::from_fn(grid.v_grid(), |x, y| {
        let (px, py) = grid.v_grid().index_position(x, y);
        if py <= base_height {
            return 0.0;
        }
        let dx = px - center.0;
        let dy = py - center.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist >= radius {
            0.0
        } else {
            let t = ((radius - dist) / radius).clamp(0.0, 1.0);
            drop_speed * t * t
        }
    });
    MacVelocity2::from_components(grid, u, v)
}

fn initial_dye(grid: MacGrid2, config: InitConfig, band: f32) -> Field2 {
    Field2::from_fn(grid.cell_grid(), |x, y| {
        let dx = (x as f32 + 0.5) - config.drop_center.0;
        let dy = (y as f32 + 0.5) - config.drop_center.1;
        let dist = (dx * dx + dy * dy).sqrt();
        let value = config.drop_radius - dist;
        if band <= 0.0 {
            if value >= 0.0 { 1.0 } else { 0.0 }
        } else {
            let edge0 = -band;
            let edge1 = band;
            smoothstep(edge0, edge1, value)
        }
    })
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn init_density_state(grid: MacGrid2, config: InitConfig, threshold: f32) -> MacSimState {
    let phi = initial_phi(grid, config);
    let density = phi_to_density(&phi, 0.0);
    let velocity = initial_velocity(grid, config);
    let base_flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
    let flags = flags_from_density(&density, &base_flags, threshold);
    MacSimState {
        density,
        velocity,
        flags,
    }
}

fn init_level_set_state(grid: MacGrid2, config: InitConfig) -> LevelSetState {
    let phi = initial_phi(grid, config);
    let velocity = initial_velocity(grid, config);
    let base_flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
    let flags = flags_from_phi(&phi, &base_flags);
    LevelSetState { phi, velocity, flags }
}

#[derive(Clone, Copy, Debug)]
struct DyeParams {
    diffusion: f32,
    decay: f32,
    air_decay: f32,
    base: f32,
    reinject: f32,
}

fn dye_params() -> DyeParams {
    static PARAMS: OnceLock<DyeParams> = OnceLock::new();
    *PARAMS.get_or_init(|| DyeParams {
        diffusion: env_f32("SIM_DYE_DIFFUSION").unwrap_or(0.0),
        decay: env_f32("SIM_DYE_DECAY").unwrap_or(1.0),
        air_decay: env_f32("SIM_DYE_AIR_DECAY").unwrap_or(1.0),
        base: env_f32("SIM_DYE_BASE").unwrap_or(0.45),
        reinject: env_f32("SIM_DYE_REINJECT").unwrap_or(0.0),
    })
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
}

fn start_mode() -> SimMode {
    match std::env::var("SIM_START_MODE").ok().as_deref() {
        Some("1") => SimMode::Density,
        Some("2") => SimMode::LevelSet,
        Some("3") => SimMode::DensityMg,
        _ => SimMode::DensityMg,
    }
}
 

fn headless_enabled() -> bool {
    if std::env::var("SIM_HEADLESS")
        .ok()
        .map(|value| value != "0")
        .unwrap_or(false)
    {
        return true;
    }
    std::env::args().any(|arg| arg == "--headless")
}

fn headless_frames() -> u64 {
    std::env::var("SIM_HEADLESS_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(120)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum DumpField {
    Density,
    Dye,
    Both,
}

fn dump_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIM_DUMP_MATRIX")
            .ok()
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

fn dump_stride() -> usize {
    static STRIDE: OnceLock<usize> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("SIM_DUMP_STRIDE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1)
    })
}

fn dump_field_mode() -> DumpField {
    static MODE: OnceLock<DumpField> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("SIM_DUMP_FIELD")
            .ok()
            .map(|val| val.trim().to_ascii_lowercase());
        match value.as_deref() {
            Some("density") | Some("height") => DumpField::Density,
            Some("dye") => DumpField::Dye,
            Some("both") | None => DumpField::Both,
            _ => DumpField::Both,
        }
    })
}

fn dump_matrix(label: &str, frame: u64, field: &Field2, stride: usize) {
    let grid = field.grid();
    println!(
        "{label} frame={} width={} height={} stride={}",
        frame,
        grid.width(),
        grid.height(),
        stride
    );
    let stride = stride.max(1);
    let mut line = String::new();
    for y in (0..grid.height()).step_by(stride) {
        line.clear();
        for x in (0..grid.width()).step_by(stride) {
            let value = field.get(x, y);
            let _ = std::fmt::Write::write_fmt(&mut line, format_args!("{:.3} ", value));
        }
        println!("{}", line.trim_end());
    }
}

fn advect_dye_into(out: &mut Field2, dye: &Field2, velocity: &MacVelocity2, dt: f32) {
    if dt == 0.0 {
        out.clone_from(dye);
        return;
    }
    let grid = dye.grid();
    out.fill_with_index(|x, y| {
        let pos = grid.cell_center(x, y);
        let v = velocity.sample_linear(pos);
        dye.sample_linear((pos.0 - v.x * dt, pos.1 - v.y * dt))
    });
}

fn diffuse_dye_into(out: &mut Field2, dye: &Field2, diffusion: f32, dt: f32) {
    if diffusion == 0.0 || dt == 0.0 {
        out.clone_from(dye);
        return;
    }
    let dx2 = dye.grid().dx() * dye.grid().dx();
    let scale = diffusion * dt / dx2;
    out.fill_with_index(|x, y| {
        let left = dye.sample_clamped(x as i32 - 1, y as i32);
        let right = dye.sample_clamped(x as i32 + 1, y as i32);
        let down = dye.sample_clamped(x as i32, y as i32 - 1);
        let up = dye.sample_clamped(x as i32, y as i32 + 1);
        let center = dye.get(x, y);
        center + (left + right + up + down - 4.0 * center) * scale
    });
}

fn clamp_field_in_place(field: &mut Field2, min: f32, max: f32) {
    field.update_with_index(|_x, _y, value| {
        if value.is_finite() {
            value.clamp(min, max)
        } else {
            min
        }
    });
}

fn decay_dye_in_air_in_place(dye: &mut Field2, flags: &CellFlags, decay: f32) {
    let decay = decay.clamp(0.0, 1.0);
    dye.update_with_index(|x, y, value| {
        if !value.is_finite() {
            0.0
        } else if flags.get(x, y) == CellType::Air {
            value * decay
        } else {
            value
        }
    });
}

fn mask_dye_to_density_in_place(dye: &mut Field2, density: &Field2) {
    dye.update_with_index(|x, y, value| {
        let rho = density.get(x, y);
        if !value.is_finite() || !rho.is_finite() {
            0.0
        } else {
            value * rho
        }
    });
}

fn preserve_dye_mass_in_place(dye: &mut Field2, target_mass: f32) {
    if target_mass <= 0.0 {
        return;
    }
    let current = dye.sum();
    if current <= 0.0 || current >= target_mass {
        return;
    }
    let max_scale = 4.0;
    let scale = (target_mass / current).min(max_scale);
    dye.scale_in_place(scale);
    clamp_field_in_place(dye, 0.0, 1.0);
}

fn update_dye_in_place(
    dye: &mut Field2,
    scratch: &mut Field2,
    velocity: &MacVelocity2,
    density: &Field2,
    flags: &CellFlags,
    dt: f32,
    params: DyeParams,
) {
    let target_mass = dye.sum();
    advect_dye_into(scratch, dye, velocity, dt);
    diffuse_dye_into(dye, scratch, params.diffusion, dt);
    if params.decay < 1.0 {
        dye.scale_in_place(params.decay);
    }
    if params.air_decay < 1.0 {
        decay_dye_in_air_in_place(dye, flags, params.air_decay);
    }
    mask_dye_to_density_in_place(dye, density);
    if params.reinject > 0.0 {
        reinforce_dye_in_place(dye, density, params.reinject);
    }
    clamp_field_in_place(dye, 0.0, 1.0);
    preserve_dye_mass_in_place(dye, target_mass);
}

fn dye_center(dye: &Field2) -> Option<(f32, f32)> {
    let grid = dye.grid();
    let mut sum = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let value = dye.get(x, y);
            if !value.is_finite() || value <= 0.0 {
                continue;
            }
            sum += value;
            let (cx, cy) = grid.cell_center(x, y);
            sum_x += value * cx;
            sum_y += value * cy;
        }
    }
    if sum > 0.0 {
        Some((sum_x / sum, sum_y / sum))
    } else {
        None
    }
}

fn droplet_center(density: &Field2, threshold: f32, base_height: f32) -> Option<(f32, f32)> {
    let grid = density.grid();
    let mut sum = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let value = density.get(x, y);
            if !value.is_finite() || value < threshold {
                continue;
            }
            let (cx, cy) = grid.cell_center(x, y);
            if cy <= base_height {
                continue;
            }
            sum += value;
            sum_x += value * cx;
            sum_y += value * cy;
        }
    }
    if sum > 0.0 {
        Some((sum_x / sum, sum_y / sum))
    } else {
        None
    }
}

fn surface_height_avg(density: &Field2, threshold: f32) -> Option<f32> {
    let grid = density.grid();
    let mut sum = 0.0;
    let mut count = 0.0;
    for x in 0..grid.width() {
        for y in (0..grid.height()).rev() {
            if density.get(x, y) >= threshold {
                sum += y as f32;
                count += 1.0;
                break;
            }
        }
    }
    if count > 0.0 {
        Some(sum / count)
    } else {
        None
    }
}

fn droplet_vy_stats(
    density: &Field2,
    velocity: &MacVelocity2,
    threshold: f32,
    base_height: f32,
) -> Option<(f32, f32, usize)> {
    let grid = density.grid();
    let mut sum = 0.0;
    let mut max_v: f32 = 0.0;
    let mut count = 0usize;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if density.get(x, y) < threshold {
                continue;
            }
            let (cx, cy) = grid.cell_center(x, y);
            if cy <= base_height {
                continue;
            }
            let v = velocity.sample_linear((cx, cy));
            sum += v.y;
            max_v = max_v.max(v.y.abs());
            count += 1;
        }
    }
    if count > 0 {
        Some((sum / count as f32, max_v, count))
    } else {
        None
    }
}

fn density_to_luma(density: &Field2, out: &mut Vec<u8>) {
    let grid = density.grid();
    let width = grid.width();
    let height = grid.height();
    let len = width * height;
    out.resize(len, 0);
    for y in 0..height {
        for x in 0..width {
            let value = density.get(x, y);
            let t = value.clamp(0.0, 1.0);
            let c = (t * 255.0) as u8;
            let idx = y * width + x;
            out[idx] = c;
        }
    }
}

fn dye_to_luma(dye: &Field2, density: &Field2, base: f32, threshold: f32, out: &mut Vec<u8>) {
    let grid = dye.grid();
    let width = grid.width();
    let height = grid.height();
    let len = width * height;
    out.resize(len, 0);
    let base = base.clamp(0.0, 1.0);
    let dye_weight = (1.0 - base).clamp(0.0, 1.0);
    for y in 0..height {
        for x in 0..width {
            let d = dye.get(x, y);
            let rho = density.get(x, y);
            let water = if rho >= threshold { base } else { 0.0 };
            let value = (water + dye_weight * d).clamp(0.0, 1.0);
            let c = (value * 255.0) as u8;
            let idx = y * width + x;
            out[idx] = c;
        }
    }
}

fn reinforce_dye_in_place(dye: &mut Field2, density: &Field2, reinject: f32) {
    let reinject = reinject.clamp(0.0, 1.0);
    if reinject == 0.0 {
        return;
    }
    dye.update_with_index(|x, y, value| {
        let rho = density.get(x, y);
        if !value.is_finite() || !rho.is_finite() {
            0.0
        } else {
            let target = (rho * reinject).clamp(0.0, 1.0);
            if value < target {
                target
            } else {
                value
            }
        }
    });
}

fn display_phi(phi: &Field2, band: f32) -> Field2 {
    phi_to_density(phi, band)
}

fn effective_dt(dt: f32, cfl: f32, velocity: &MacVelocity2) -> f32 {
    if dt <= 0.0 || cfl <= 0.0 {
        return dt;
    }
    let max_vel = velocity.max_abs();
    if max_vel == 0.0 {
        return dt;
    }
    let dx = velocity.grid().dx();
    dt.min(cfl * dx / max_vel)
}

#[allow(clippy::too_many_arguments)]
fn overlay_hud(
    texture: &mut [u8],
    width: usize,
    height: usize,
    mode: SimMode,
    dt: f32,
    cfl: f32,
    iters: usize,
    fps: f32,
    sim_ms: f32,
) {
    let line_height = GLYPH_HEIGHT + LINE_SPACING;
    let line_1 = format!("{} DT {:.3}", mode.tag(), dt);
    let line_2 = format!("CFL {:.2}", cfl);
    let line_3 = format!("IT {}", iters);
    let line_4 = format!("FPS {:.1}", fps);
    let line_5 = format!("SIM {:.2}", sim_ms);
    let mut y = 6;
    for line in [line_1, line_2, line_3, line_4, line_5] {
        overlay_text(texture, width, height, 6, y, &line, 240, true);
        y = y.saturating_add(line_height);
    }
}

fn main() -> Result<()> {
    let tex_width = 256;
    let tex_height = 256;
    let headless = headless_enabled();
    let grid = MacGrid2::new(tex_width, tex_height, 1.0);
    let init = InitConfig::new(grid);
    println!("init drop_speed={:.2}", init.drop_speed);
    let density_threshold = 0.35;
    let density_band = 0.08;
    let surface_band = grid.dx() * 1.2;
    let dye_band = density_band;
    let mut density_state = init_density_state(grid, init, density_threshold);
    let mut density_mg_state = init_density_state(grid, init, density_threshold);
    let mut level_state = init_level_set_state(grid, init);
    let mut density_phi = initial_phi(grid, init);
    let mut density_mg_phi = initial_phi(grid, init);
    let density_phi_volume = volume_from_phi(&density_phi, surface_band);
    let density_mg_phi_volume = density_phi_volume;
    let mut density_dye = initial_dye(grid, init, dye_band);
    let mut density_mg_dye = initial_dye(grid, init, dye_band);
    let mut density_dye_scratch = Field2::new(grid.cell_grid(), 0.0);
    let mut density_mg_dye_scratch = Field2::new(grid.cell_grid(), 0.0);
    let dye_params = dye_params();
    println!(
        "dye params diffusion={:.6} decay={:.6} air_decay={:.6} base={:.3} reinject={:.3}",
        dye_params.diffusion,
        dye_params.decay,
        dye_params.air_decay,
        dye_params.base,
        dye_params.reinject
    );
    density_state.density = phi_to_density(&density_phi, density_band);
    density_state.flags = flags_from_phi(&density_phi, &density_state.flags);
    density_mg_state.density = phi_to_density(&density_mg_phi, density_band);
    density_mg_state.flags = flags_from_phi(&density_mg_phi, &density_mg_state.flags);
    let mut density_workspace = MacSimWorkspace::new(grid);
    let mut density_mg_workspace = MacSimMgWorkspace::new(grid);
    let mut level_workspace = LevelSetWorkspace::new(grid);
    let target_volume = volume_from_phi(&level_state.phi, surface_band);
    let vorticity_strength = env_f32("SIM_VORTICITY").unwrap_or(0.0);
    let density_params = MacSimParams {
        dt: 0.04,
        cfl: 0.5,
        diffusion: 0.0,
        viscosity: 0.004,
        vorticity_strength,
        pressure_iters: 60,
        pressure_tol: 2e-4,
        advection: AdvectionScheme::Bfecc,
        boundaries: BoundaryConfig::no_slip(),
        body_force: Vec2::new(0.0, -16.0),
        buoyancy: 0.0,
        ambient_density: 0.0,
        surface_tension: 0.12,
        free_surface: true,
        density_threshold,
        surface_band: density_band,
        preserve_mass: true,
    };
    let density_mg_params = MacSimParams {
        preserve_mass: true,
        ..density_params
    };
    let level_params = LevelSetParams {
        dt: 0.04,
        cfl: 0.5,
        viscosity: 0.012,
        pressure_iters: 60,
        pressure_tol: 2e-4,
        advection: AdvectionScheme::Bfecc,
        boundaries: BoundaryConfig::no_slip(),
        body_force: Vec2::new(0.0, -12.0),
        surface_tension: 0.26,
        surface_tension_band: surface_band,
        reinit_iters: 5,
        reinit_dt: 0.35,
        extrapolation_iters: 2,
        preserve_volume: true,
        target_volume,
        volume_band: surface_band,
    };
    let density_surface_params = LevelSetParams {
        dt: density_params.dt,
        cfl: density_params.cfl,
        viscosity: 0.0,
        pressure_iters: 0,
        pressure_tol: 0.0,
        advection: density_params.advection,
        boundaries: density_params.boundaries,
        body_force: Vec2::zero(),
        surface_tension: 0.0,
        surface_tension_band: surface_band,
        reinit_iters: level_params.reinit_iters,
        reinit_dt: level_params.reinit_dt,
        extrapolation_iters: level_params.extrapolation_iters,
        preserve_volume: true,
        target_volume: density_phi_volume,
        volume_band: surface_band,
    };
    let density_mg_surface_params = LevelSetParams {
        target_volume: density_mg_phi_volume,
        ..density_surface_params
    };
    let mg_params = MultigridParams {
        cycles: 4,
        pre_smooth: 2,
        post_smooth: 2,
        coarse_smooth: 20,
        omega: 0.8,
        levels: 0,
        tol: 0.0,
        final_smooth: 0,
        pcg_iters: 0,
        pcg_tol: 0.0,
    };
    let mut mode = start_mode();
    let mut texture = Vec::new();
    let density_steps_per_frame = 4;
    let level_steps_per_frame = 4;
    let mut frame_count: u64 = 0;
    let mut avg_frame_ms = 0.0;
    let mut avg_sim_ms = 0.0;
    let mut step_frame = move |mode: SimMode,
                          frame_count: &mut u64,
                          avg_frame_ms: &mut f32,
                          avg_sim_ms: &mut f32,
                          app: Option<&mut VulkanApp>| {
        let frame_start = Instant::now();
        let sim_start = Instant::now();
        let before_stats = if *frame_count == 0 {
            Some(match mode {
                SimMode::Density => density_state.density.stats(),
                SimMode::LevelSet => level_state.phi.stats(),
                SimMode::DensityMg => density_mg_state.density.stats(),
            })
        } else {
            None
        };
        let (dt_display, cfl_display, iters_display) = match mode {
            SimMode::Density => {
                for _ in 0..density_steps_per_frame {
                    let phi_flags = flags_from_phi(&density_phi, &density_state.flags);
                    density_state.flags = phi_flags.clone();
                    density_state.density = phi_to_density(&density_phi, density_band);
                    step_in_place(&mut density_state, density_params, &mut density_workspace);
                    advect_level_set_surface_in_place(
                        &mut density_phi,
                        &density_state.velocity,
                        &phi_flags,
                        density_surface_params,
                        &mut level_workspace,
                    );
                }
                density_state.density = phi_to_density(&density_phi, density_band);
                density_state.flags = flags_from_phi(&density_phi, &density_state.flags);
                let dt_dye = effective_dt(
                    density_params.dt,
                    density_params.cfl,
                    &density_state.velocity,
                );
                update_dye_in_place(
                    &mut density_dye,
                    &mut density_dye_scratch,
                    &density_state.velocity,
                    &density_state.density,
                    &density_state.flags,
                    dt_dye,
                    dye_params,
                );
                dye_to_luma(
                    &density_dye,
                    &density_state.density,
                    dye_params.base,
                    density_threshold,
                    &mut texture,
                );
                let dt_display = effective_dt(
                    density_params.dt,
                    density_params.cfl,
                    &density_state.velocity,
                );
                (
                    dt_display,
                    density_params.cfl,
                    density_params.pressure_iters,
                )
            }
            SimMode::LevelSet => {
                for _ in 0..level_steps_per_frame {
                    level_set_step_in_place(&mut level_state, level_params, &mut level_workspace);
                }
                let display = display_phi(&level_state.phi, surface_band);
                density_to_luma(&display, &mut texture);
                let dt_display = effective_dt(
                    level_params.dt,
                    level_params.cfl,
                    &level_state.velocity,
                );
                (
                    dt_display,
                    level_params.cfl,
                    level_params.pressure_iters,
                )
            }
            SimMode::DensityMg => {
                for step_idx in 0..density_steps_per_frame {
                    let phi_flags = flags_from_phi(&density_mg_phi, &density_mg_state.flags);
                    density_mg_state.flags = phi_flags.clone();
                    density_mg_state.density = phi_to_density(&density_mg_phi, density_band);
                    step_in_place_mg(
                        &mut density_mg_state,
                        density_mg_params,
                        mg_params,
                        &mut density_mg_workspace,
                    );
                    advect_level_set_surface_in_place(
                        &mut density_mg_phi,
                        &density_mg_state.velocity,
                        &phi_flags,
                        density_mg_surface_params,
                        &mut level_workspace,
                    );
                    if cfg!(debug_assertions) && *frame_count == 0 {
                        let (sum, min_value, max_value, non_finite) =
                            density_mg_state.density.stats();
                        println!(
                            "mg:frame0 step={} sum={:.2} min={:.3} max={:.3} nonfinite={}",
                            step_idx + 1,
                            sum,
                            min_value,
                            max_value,
                            non_finite
                        );
                    }
                }
                density_mg_state.density = phi_to_density(&density_mg_phi, density_band);
                density_mg_state.flags = flags_from_phi(&density_mg_phi, &density_mg_state.flags);
                let dt_dye = effective_dt(
                    density_mg_params.dt,
                    density_mg_params.cfl,
                    &density_mg_state.velocity,
                );
                update_dye_in_place(
                    &mut density_mg_dye,
                    &mut density_mg_dye_scratch,
                    &density_mg_state.velocity,
                    &density_mg_state.density,
                    &density_mg_state.flags,
                    dt_dye,
                    dye_params,
                );
                dye_to_luma(
                    &density_mg_dye,
                    &density_mg_state.density,
                    dye_params.base,
                    density_threshold,
                    &mut texture,
                );
                let dt_display = effective_dt(
                    density_mg_params.dt,
                    density_mg_params.cfl,
                    &density_mg_state.velocity,
                );
                (dt_display, density_mg_params.cfl, mg_params.cycles)
            }
        };
        if let Some((sum_before, min_before, max_before, non_finite_before)) = before_stats {
            let (sum_after, min_after, max_after, non_finite_after) = match mode {
                SimMode::Density => density_state.density.stats(),
                SimMode::LevelSet => level_state.phi.stats(),
                SimMode::DensityMg => density_mg_state.density.stats(),
            };
            println!(
                "init mode={} sum={:.2}->{:.2} min={:.3}->{:.3} max={:.3}->{:.3} nonfinite={} -> {}",
                mode.tag(),
                sum_before,
                sum_after,
                min_before,
                min_after,
                max_before,
                max_after,
                non_finite_before,
                non_finite_after
            );
        }
        let sim_ms = sim_start.elapsed().as_secs_f32() * 1000.0;
        let frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        let alpha = 0.1;
        if *frame_count == 0 {
            *avg_frame_ms = frame_ms;
            *avg_sim_ms = sim_ms;
        } else {
            *avg_frame_ms = *avg_frame_ms * (1.0 - alpha) + frame_ms * alpha;
            *avg_sim_ms = *avg_sim_ms * (1.0 - alpha) + sim_ms * alpha;
        }
        let fps = if *avg_frame_ms > 0.0 {
            1000.0 / *avg_frame_ms
        } else {
            0.0
        };
        overlay_hud(
            &mut texture,
            tex_width,
            tex_height,
            mode,
            dt_display,
            cfl_display,
            iters_display,
            fps,
            *avg_sim_ms,
        );
        *frame_count += 1;
                if (*frame_count).is_multiple_of(10) {
                    let (mass, div_sum, vel_max, dye_stats, dye_center) = match mode {
                        SimMode::Density => (
                            density_state.density.sum(),
                            divergence(&density_state.velocity).abs_sum(),
                            density_state.velocity.max_abs(),
                            Some(density_dye.stats()),
                            dye_center(&density_dye),
                        ),
                        SimMode::LevelSet => (
                            volume_from_phi(&level_state.phi, surface_band),
                            divergence(&level_state.velocity).abs_sum(),
                            level_state.velocity.max_abs(),
                            None,
                            None,
                        ),
                        SimMode::DensityMg => (
                            density_mg_state.density.sum(),
                            divergence(&density_mg_state.velocity).abs_sum(),
                            density_mg_state.velocity.max_abs(),
                            Some(density_mg_dye.stats()),
                            dye_center(&density_mg_dye),
                        ),
                    };
            let surf_height = match mode {
                SimMode::Density => {
                    surface_height_avg(&density_state.density, density_threshold)
                }
                SimMode::DensityMg => {
                    surface_height_avg(&density_mg_state.density, density_threshold)
                }
                SimMode::LevelSet => None,
            };
            let surf_label = surf_height
                .map(|value| format!(" surf_y={:.2}", value))
                .unwrap_or_default();
            let droplet_stats = match mode {
                SimMode::Density => droplet_vy_stats(
                    &density_state.density,
                    &density_state.velocity,
                    density_threshold,
                    init.base_height,
                ),
                SimMode::DensityMg => droplet_vy_stats(
                    &density_mg_state.density,
                    &density_mg_state.velocity,
                    density_threshold,
                    init.base_height,
                ),
                SimMode::LevelSet => None,
            };
            let droplet_center = match mode {
                SimMode::Density => {
                    droplet_center(&density_state.density, density_threshold, init.base_height)
                }
                SimMode::DensityMg => {
                    droplet_center(&density_mg_state.density, density_threshold, init.base_height)
                }
                SimMode::LevelSet => None,
            };
            let droplet_label = droplet_stats
                .map(|(avg_vy, max_vy, count)| {
                    format!(
                        " drop_vy={:.2} drop_vmax={:.2} drop_n={}",
                        avg_vy, max_vy, count
                    )
                })
                .unwrap_or_default();
            let droplet_center_label = droplet_center
                .map(|(cx, cy)| format!(" drop_center=({:.2},{:.2})", cx, cy))
                .unwrap_or_default();
            if let Some((dye_sum, dye_min, dye_max, dye_nonfinite)) = dye_stats {
                if let Some((cx, cy)) = dye_center {
                    println!(
                        "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2} dye_sum={:.2} dye_min={:.3} dye_max={:.3} dye_nf={} dye_center=({:.2},{:.2}){}{}{}",
                        *frame_count,
                        mode.tag(),
                        *avg_sim_ms,
                        *avg_frame_ms,
                        fps,
                        mass,
                        div_sum,
                        vel_max,
                        dye_sum,
                        dye_min,
                        dye_max,
                        dye_nonfinite,
                        cx,
                        cy,
                        surf_label,
                        droplet_label,
                        droplet_center_label
                    );
                } else {
                    println!(
                        "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2} dye_sum={:.2} dye_min={:.3} dye_max={:.3} dye_nf={}{}{}{}",
                        *frame_count,
                        mode.tag(),
                        *avg_sim_ms,
                        *avg_frame_ms,
                        fps,
                        mass,
                        div_sum,
                        vel_max,
                        dye_sum,
                        dye_min,
                        dye_max,
                        dye_nonfinite,
                        surf_label,
                        droplet_label,
                        droplet_center_label
                    );
                }
            } else {
                println!(
                    "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2}{}{}{}",
                    *frame_count,
                    mode.tag(),
                    *avg_sim_ms,
                    *avg_frame_ms,
                    fps,
                    mass,
                    div_sum,
                    vel_max,
                    surf_label,
                    droplet_label,
                    droplet_center_label
                );
            }
                    if dump_enabled() {
                        let stride = dump_stride();
                        let dump_mode = dump_field_mode();
                        match mode {
                            SimMode::Density => {
                        if dump_mode == DumpField::Density || dump_mode == DumpField::Both {
                            dump_matrix(
                                "matrix density",
                                *frame_count,
                                &density_state.density,
                                stride,
                            );
                        }
                        if dump_mode == DumpField::Dye || dump_mode == DumpField::Both {
                            dump_matrix("matrix dye", *frame_count, &density_dye, stride);
                        }
                            }
                            SimMode::LevelSet => {
                                dump_matrix("matrix phi", *frame_count, &level_state.phi, stride);
                            }
                            SimMode::DensityMg => {
                        if dump_mode == DumpField::Density || dump_mode == DumpField::Both {
                            dump_matrix(
                                "matrix density_mg",
                                *frame_count,
                                &density_mg_state.density,
                                stride,
                            );
                        }
                        if dump_mode == DumpField::Dye || dump_mode == DumpField::Both {
                            dump_matrix(
                                "matrix dye_mg",
                                *frame_count,
                                &density_mg_dye,
                                stride,
                            );
                                }
                            }
                        }
                    }
                }
        if let Some(app) = app {
            if let Err(err) = app.update_texture(&texture) {
                eprintln!("texture upload error: {err:#}");
            }
            if let Err(err) = app.render() {
                eprintln!("render error: {err:#}");
            }
        }
    };

    if headless {
        let frames = headless_frames();
        println!("headless mode frames={}", frames);
        for _ in 0..frames {
            step_frame(
                mode,
                &mut frame_count,
                &mut avg_frame_ms,
                &mut avg_sim_ms,
                None,
            );
        }
        return Ok(());
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Navier-Stokes Sim")
        .with_inner_size(winit::dpi::LogicalSize::new(960.0, 720.0))
        .build(&event_loop)?;
    let mut app = VulkanApp::new(&window, (tex_width as u32, tex_height as u32))?;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if input.state == winit::event::ElementState::Pressed
                    && let Some(key) = input.virtual_keycode
                {
                    match key {
                        winit::event::VirtualKeyCode::Space => {
                            mode = mode.toggle();
                        }
                        winit::event::VirtualKeyCode::Key1 => {
                            mode = SimMode::Density;
                        }
                        winit::event::VirtualKeyCode::Key2 => {
                            mode = SimMode::LevelSet;
                        }
                        winit::event::VirtualKeyCode::Key3 => {
                            mode = SimMode::DensityMg;
                        }
                        _ => {}
                    }
                }
            }
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                step_frame(
                    mode,
                    &mut frame_count,
                    &mut avg_frame_ms,
                    &mut avg_sim_ms,
                    Some(&mut app),
                );
            }
            _ => {}
        }
    });
}
