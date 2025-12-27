use anyhow::Result;
use nav_stokes_sim::{
    advect_level_set_surface_in_place, apply_domain_boundaries, apply_solid_boundaries,
    flags_from_density, flags_from_phi,
    correct_phi_volume, level_set_step_in_place, overlay_text, phi_to_density, reinitialize_phi,
    divergence, extrapolate_velocity, project_with_flags, step_in_place, step_in_place_mg,
    volume_from_phi, AdvectionScheme, BoundaryCondition, BoundaryConfig, CellFlags, CellType,
    Field2, LevelSetParams, LevelSetState, LevelSetWorkspace, MacGrid2, MacSimMgWorkspace,
    MacSimParams, MacSimState, MacSimWorkspace, MacVelocity2, MultigridParams, ParticleBounds,
    ParticlePhase, ParticleSystem, StaggeredField2, Vec2, VulkanApp, GLYPH_HEIGHT, LINE_SPACING,
};
use std::cell::RefCell;
use std::rc::Rc;
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
    Particles,
    ParticlesGpu,
}

impl SimMode {
    fn toggle(self) -> Self {
        match self {
            SimMode::Density => SimMode::LevelSet,
            SimMode::LevelSet => SimMode::DensityMg,
            SimMode::DensityMg => SimMode::Particles,
            SimMode::Particles => SimMode::ParticlesGpu,
            SimMode::ParticlesGpu => SimMode::Density,
        }
    }

    fn tag(self) -> char {
        match self {
            SimMode::Density => '1',
            SimMode::LevelSet => '2',
            SimMode::DensityMg => '3',
            SimMode::Particles => '4',
            SimMode::ParticlesGpu => '5',
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

#[derive(Clone, Copy, Debug)]
struct Obstacle {
    center: Vec2,
    radius: f32,
    bounce: f32,
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
        let drop_speed = env_f32("SIM_DROP_SPEED").unwrap_or(-20.0);
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

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

fn start_mode() -> SimMode {
    match std::env::var("SIM_START_MODE").ok().as_deref() {
        Some("1") => SimMode::Density,
        Some("2") => SimMode::LevelSet,
        Some("3") => SimMode::DensityMg,
        Some("4") => SimMode::Particles,
        Some("5") => SimMode::ParticlesGpu,
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum ValidationCase {
    Drop,
    DamBreak,
    Volume,
    Slosh,
    Obstacle,
}

impl ValidationCase {
    fn label(self) -> &'static str {
        match self {
            ValidationCase::Drop => "drop",
            ValidationCase::DamBreak => "dam_break",
            ValidationCase::Volume => "volume",
            ValidationCase::Slosh => "slosh",
            ValidationCase::Obstacle => "obstacle",
        }
    }
}

fn validation_case() -> Option<ValidationCase> {
    match std::env::var("SIM_VALIDATE_CASE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("drop") => Some(ValidationCase::Drop),
        Some("dam") | Some("dam_break") | Some("dambreak") => Some(ValidationCase::DamBreak),
        Some("volume") | Some("conservation") => Some(ValidationCase::Volume),
        Some("slosh") | Some("sloshing") => Some(ValidationCase::Slosh),
        Some("obstacle") | Some("wake") => Some(ValidationCase::Obstacle),
        _ => None,
    }
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

#[derive(Clone, Debug)]
struct ValidationMetrics {
    case: ValidationCase,
    frames: u64,
    max_div: f32,
    min_vol_raw: f32,
    max_vol_raw: f32,
    specks_sum: usize,
    specks_max: usize,
    samples: usize,
}

impl ValidationMetrics {
    fn new(case: ValidationCase) -> Self {
        Self {
            case,
            frames: 0,
            max_div: 0.0,
            min_vol_raw: f32::INFINITY,
            max_vol_raw: 0.0,
            specks_sum: 0,
            specks_max: 0,
            samples: 0,
        }
    }
}

fn count_surface_specks(field: &Field2, threshold: f32, start_y: usize) -> usize {
    let grid = field.grid();
    let width = grid.width();
    let height = grid.height();
    let start_y = start_y.min(height);
    let mut count = 0usize;
    for y in start_y..height {
        for x in 0..width {
            if field.get(x, y) > threshold {
                count += 1;
            }
        }
    }
    count
}

fn apply_surface_hysteresis(
    field: &mut Field2,
    strong: f32,
    weak: f32,
    min_neighbors: usize,
) {
    if strong <= 0.0 || weak <= 0.0 {
        return;
    }
    let strong = strong.clamp(0.0, 1.0).max(weak);
    let weak = weak.clamp(0.0, strong);
    let grid = field.grid();
    let width = grid.width();
    let height = grid.height();
    let size = width * height;
    let mut strong_mask = vec![false; size];
    let mut weak_mask = vec![false; size];
    for y in 0..height {
        for x in 0..width {
            let v = field.get(x, y);
            let idx = y * width + x;
            if v >= strong {
                strong_mask[idx] = true;
                weak_mask[idx] = true;
            } else if v >= weak {
                weak_mask[idx] = true;
            }
        }
    }
    let mut keep = vec![false; size];
    for i in 0..size {
        if strong_mask[i] {
            keep[i] = true;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if keep[idx] || !weak_mask[idx] {
                continue;
            }
            let mut has_strong = false;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let nidx = ny * width + nx;
                    if strong_mask[nidx] {
                        has_strong = true;
                        break;
                    }
                }
                if has_strong {
                    break;
                }
            }
            if has_strong {
                keep[idx] = true;
            }
        }
    }
    if min_neighbors > 0 {
        let mut filtered = keep.clone();
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !keep[idx] {
                    continue;
                }
                let mut neighbors = 0usize;
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(height - 1);
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(width - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        if nx == x && ny == y {
                            continue;
                        }
                        if keep[ny * width + nx] {
                            neighbors += 1;
                        }
                    }
                }
                if neighbors < min_neighbors {
                    filtered[idx] = false;
                }
            }
        }
        keep = filtered;
    }
    let original = field.clone();
    field.fill_with_index(|x, y| {
        let idx = y * width + x;
        if keep[idx] {
            original.get(x, y)
        } else {
            0.0
        }
    });
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

fn particle_droplet_center(
    particles: &ParticleSystem,
    base_height: f32,
) -> Option<(f32, f32)> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0usize;
    for pos in particles.positions() {
        if pos.y <= base_height {
            continue;
        }
        sum_x += pos.x;
        sum_y += pos.y;
        count += 1;
    }
    if count > 0 {
        Some((sum_x / count as f32, sum_y / count as f32))
    } else {
        None
    }
}

fn particle_droplet_vy_stats(
    particles: &ParticleSystem,
    base_height: f32,
) -> Option<(f32, f32, usize)> {
    let mut sum_vy = 0.0;
    let mut max_v: f32 = 0.0;
    let mut count = 0usize;
    for (pos, vel) in particles.positions().iter().zip(particles.velocities()) {
        if pos.y <= base_height {
            continue;
        }
        sum_vy += vel.y;
        max_v = max_v.max(vel.y.abs());
        count += 1;
    }
    if count > 0 {
        Some((sum_vy / count as f32, max_v, count))
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

fn particle_level_set_from_particles(
    particles: &ParticleSystem,
    grid: MacGrid2,
    bounds: ParticleBounds,
    particle_radius: f32,
    max_distance: f32,
    reinit_iters: usize,
    reinit_dt: f32,
    volume_blend: f32,
    volume_weight_band: f32,
    target_volume: Option<(f32, f32)>,
) -> Field2 {
    let mut phi = particles.to_level_set(grid, bounds, particle_radius, max_distance);
    if reinit_iters > 0 && reinit_dt > 0.0 {
        phi = reinitialize_phi(&phi, reinit_iters, reinit_dt);
    }
    let volume_blend = volume_blend.clamp(0.0, 1.0);
    if let Some((volume, band)) = target_volume {
        if volume_blend >= 1.0 {
            phi = correct_phi_volume(&phi, volume, band);
        } else if volume_blend > 0.0 {
            let corrected = correct_phi_volume(&phi, volume, band);
            let weight_band = if volume_weight_band > 0.0 {
                volume_weight_band
            } else {
                band.max(1e-6)
            };
            phi = phi.zip_with(&corrected, |a, b| {
                let w = ((weight_band - a.abs()) / weight_band).clamp(0.0, 1.0);
                a + (b - a) * volume_blend * w
            });
        }
    }
    phi
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
    extra_lines: &[String],
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
    for line in extra_lines {
        overlay_text(texture, width, height, 6, y, line, 240, true);
        y = y.saturating_add(line_height);
    }
}

fn main() -> Result<()> {
    let grid_size = env_usize("SIM_GRID_SIZE").filter(|value| *value > 0);
    let tex_width = env_usize("SIM_GRID_WIDTH")
        .filter(|value| *value > 0)
        .or(grid_size)
        .unwrap_or(256);
    let tex_height = env_usize("SIM_GRID_HEIGHT")
        .filter(|value| *value > 0)
        .or(grid_size)
        .unwrap_or(256);
    let headless = headless_enabled();
    let validation = validation_case();
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
    let particle_spacing = env_f32("SIM_PARTICLE_SPACING").unwrap_or(grid.dx() * 0.75);
    let particle_jitter = 0.35;
    let pool_height_fraction = (init.base_height / (grid.height() as f32 * grid.dx())).clamp(0.1, 0.9);
    let dam_width_fraction = env_f32("SIM_DAM_WIDTH_FRAC").unwrap_or(0.4).clamp(0.05, 0.95);
    let dam_height_fraction =
        env_f32("SIM_DAM_HEIGHT_FRAC").unwrap_or(pool_height_fraction).clamp(0.05, 0.95);
    let width = grid.width() as f32 * grid.dx();
    let height = grid.height() as f32 * grid.dx();
    let min_dim = width.min(height);
    let obstacle = if validation == Some(ValidationCase::Obstacle) {
        let obs_radius = env_f32("SIM_OBS_RADIUS").unwrap_or(min_dim * 0.08);
        let obs_center_x = env_f32("SIM_OBS_CENTER_X").unwrap_or(0.35).clamp(0.05, 0.95);
        let obs_center_y = env_f32("SIM_OBS_CENTER_Y").unwrap_or(0.5).clamp(0.05, 0.95);
        let obs_bounce = env_f32("SIM_OBS_BOUNCE").unwrap_or(0.0);
        Some(Obstacle {
            center: Vec2::new(width * obs_center_x, height * obs_center_y),
            radius: obs_radius,
            bounce: obs_bounce,
        })
    } else {
        None
    };
    let speck_start_y = if validation == Some(ValidationCase::DamBreak) {
        (dam_height_fraction * grid.height() as f32) as usize
    } else {
        (pool_height_fraction * grid.height() as f32) as usize
    };
    let mut particles = match validation {
        Some(ValidationCase::DamBreak) => {
            let max = Vec2::new(width * dam_width_fraction, height * dam_height_fraction);
            ParticleSystem::seed_rect(
                Vec2::zero(),
                max,
                particle_spacing,
                particle_jitter,
                ParticlePhase::Liquid,
            )
        }
        _ => ParticleSystem::seed_pool(
            grid,
            pool_height_fraction,
            particle_spacing,
            particle_jitter,
            ParticlePhase::Liquid,
        ),
    };
    let spawn_droplet = match validation {
        Some(ValidationCase::DamBreak)
        | Some(ValidationCase::Volume)
        | Some(ValidationCase::Slosh)
        | Some(ValidationCase::Obstacle) => false,
        _ => true,
    };
    if validation == Some(ValidationCase::Slosh) {
        let slosh_speed = env_f32("SIM_SLOSH_SPEED").unwrap_or(3.0);
        let pool_height = height * pool_height_fraction;
        particles.set_velocity_field(|pos| {
            let t = (pos.y / pool_height).clamp(0.0, 1.0);
            let u = slosh_speed * (t * std::f32::consts::FRAC_PI_2).sin();
            Vec2::new(u, 0.0)
        });
    }
    if validation == Some(ValidationCase::Obstacle) {
        let flow_speed = env_f32("SIM_OBS_FLOW").unwrap_or(3.0);
        particles.set_velocity_field(|_pos| Vec2::new(flow_speed, 0.0));
    }
    if spawn_droplet {
        let mut droplet = ParticleSystem::seed_disk(
            Vec2::new(init.drop_center.0, init.drop_center.1),
            init.drop_radius * 0.9,
            particle_spacing,
            particle_jitter,
            ParticlePhase::Liquid,
        );
        droplet.add_velocity(Vec2::new(0.0, init.drop_speed));
        particles.append(droplet);
    }
    let particle_bounds = ParticleBounds::from_grid(grid, grid.dx() * 0.5);
    let mut particle_grid = MacVelocity2::new(grid, Vec2::zero());
    let mut particle_prev_grid = MacVelocity2::new(grid, Vec2::zero());
    let particle_base_flags = if let Some(obs) = obstacle {
        CellFlags::from_fn(grid.cell_grid(), |x, y| {
            let (cx, cy) = grid.cell_center(x, y);
            let dx = cx - obs.center.x;
            let dy = cy - obs.center.y;
            if dx * dx + dy * dy <= obs.radius * obs.radius {
                CellType::Solid
            } else {
                CellType::Fluid
            }
        })
    } else {
        CellFlags::new(grid.cell_grid(), CellType::Fluid)
    };
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
    let particle_dt = env_f32("SIM_PARTICLE_DT").unwrap_or(0.03);
    let mut particle_gravity = env_f32("SIM_PARTICLE_GRAVITY").unwrap_or(-9.8);
    if matches!(validation, Some(ValidationCase::Volume | ValidationCase::Obstacle))
        && std::env::var("SIM_PARTICLE_GRAVITY").ok().is_none()
    {
        particle_gravity = 0.0;
    }
    let particle_flip_ratio = env_f32("SIM_FLIP_RATIO").unwrap_or(0.05);
    let particle_bounce = env_f32("SIM_PARTICLE_BOUNCE").unwrap_or(0.0);
    let particle_pressure_iters = env_usize("SIM_PARTICLE_PRESSURE_ITERS").unwrap_or(60);
    let particle_pressure_tol = env_f32("SIM_PARTICLE_PRESSURE_TOL").unwrap_or(1e-4);
    let particle_extrap_iters = env_usize("SIM_PARTICLE_EXTRAP_ITERS").unwrap_or(2);
    let particle_separation = env_f32("SIM_PARTICLE_SEPARATION")
        .unwrap_or(particle_spacing * 1.0);
    let particle_separation_iters =
        env_usize("SIM_PARTICLE_SEPARATION_ITERS").unwrap_or(2);
    let particle_separation_every =
        env_usize("SIM_PARTICLE_SEPARATION_EVERY").unwrap_or(2);
    let particle_pls_reinit_iters =
        env_usize("SIM_PARTICLE_PLS_REINIT_ITERS").unwrap_or(3);
    let particle_pls_reinit_dt =
        env_f32("SIM_PARTICLE_PLS_REINIT_DT").unwrap_or(0.3);
    let particle_volume_blend = env_f32("SIM_PARTICLE_VOLUME_BLEND").unwrap_or(0.0);
    let particle_ls_radius = env_f32("SIM_PARTICLE_LS_RADIUS").unwrap_or(particle_spacing * 0.6);
    let particle_ls_max = env_f32("SIM_PARTICLE_LS_MAX").unwrap_or(particle_ls_radius * 4.0);
    let particle_surface_band =
        env_f32("SIM_PARTICLE_SURFACE_BAND").unwrap_or(particle_ls_radius * 0.75);
    let particle_volume_band =
        env_f32("SIM_PARTICLE_VOLUME_BAND").unwrap_or(particle_ls_radius * 2.0);
    let particle_occ_scale = env_f32("SIM_PARTICLE_OCC_SCALE").unwrap_or(1.0);
    let particle_occ_threshold = env_f32("SIM_PARTICLE_OCC_THRESHOLD").unwrap_or(0.32);
    let particle_flag_threshold = env_f32("SIM_PARTICLE_FLAG_THRESHOLD")
        .unwrap_or(particle_occ_threshold * 0.5)
        .min(particle_occ_threshold);
    let particle_rest_density = particles.rest_density(grid);
    let particle_render_volume = env_usize("SIM_PARTICLE_RENDER_VOLUME").unwrap_or(1) != 0;
    let particle_pbf_radius = env_f32("SIM_PARTICLE_PBF_RADIUS")
        .unwrap_or(particle_spacing * 1.3);
    let particle_render_mode = env_string("SIM_PARTICLE_RENDER")
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "phi".to_string());
    let particle_render_radius =
        env_f32("SIM_PARTICLE_RENDER_RADIUS").unwrap_or(particle_pbf_radius);
    let particle_render_scale = env_f32("SIM_PARTICLE_RENDER_SCALE").unwrap_or(1.0);
    let particle_pbf_iters = env_usize("SIM_PARTICLE_PBF_ITERS").unwrap_or(6);
    let particle_pbf_every = env_usize("SIM_PARTICLE_PBF_EVERY").unwrap_or(1);
    let particle_pbf_scorr_k = env_f32("SIM_PARTICLE_PBF_SCORR_K").unwrap_or(0.001);
    let particle_pbf_scorr_n = env_f32("SIM_PARTICLE_PBF_SCORR_N").unwrap_or(4.0);
    let particle_pbf_scorr_q = env_f32("SIM_PARTICLE_PBF_SCORR_Q").unwrap_or(0.2);
    let particle_pbf_rest = particles.pbf_rest_density(particle_bounds, particle_pbf_radius);
    let particle_render_rest = particles.rest_sph_density(grid, particle_render_radius);
    let particle_apic = env_usize("SIM_PARTICLE_APIC").unwrap_or(1) != 0;
    let particle_apic_clamp = env_f32("SIM_PARTICLE_APIC_CLAMP").unwrap_or(6.0);
    let particle_xsph = env_f32("SIM_PARTICLE_XSPH").unwrap_or(0.02);
    let particle_xsph_radius =
        env_f32("SIM_PARTICLE_XSPH_RADIUS").unwrap_or(particle_pbf_radius);
    let particle_resample = env_usize("SIM_PARTICLE_RESAMPLE").unwrap_or(1) != 0;
    let particle_resample_every = env_usize("SIM_PARTICLE_RESAMPLE_EVERY").unwrap_or(2).max(1);
    let particle_resample_low =
        env_f32("SIM_PARTICLE_RESAMPLE_LOW").unwrap_or(particle_occ_threshold * 0.7);
    let particle_resample_high =
        env_f32("SIM_PARTICLE_RESAMPLE_HIGH").unwrap_or(particle_occ_threshold * 1.4);
    let particle_resample_max = env_usize("SIM_PARTICLE_RESAMPLE_MAX")
        .unwrap_or((grid.width() / 2).max(64));
    let particle_resample_jitter =
        env_f32("SIM_PARTICLE_RESAMPLE_JITTER").unwrap_or(particle_spacing * 0.35);
    let particle_resample_seed = env_usize("SIM_PARTICLE_RESAMPLE_SEED").unwrap_or(0) as u32;
    let particle_target_count = particles.len();
    let particle_post_project = env_usize("SIM_PARTICLE_POST_PBF_PROJECT").unwrap_or(1) != 0;
    let particle_post_project_iters = env_usize("SIM_PARTICLE_POST_PBF_ITERS").unwrap_or(30);
    let particle_surface_clean = env_usize("SIM_PARTICLE_SURFACE_CLEAN").unwrap_or(1) != 0;
    let particle_surface_strong =
        env_f32("SIM_PARTICLE_SURFACE_STRONG").unwrap_or(0.5);
    let particle_surface_weak = env_f32("SIM_PARTICLE_SURFACE_WEAK").unwrap_or(0.35);
    let particle_surface_neighbors =
        env_usize("SIM_PARTICLE_SURFACE_NEIGHBORS").unwrap_or(4);
    let particle_air_drag = env_f32("SIM_PARTICLE_AIR_DRAG").unwrap_or(1.5);
    let particle_air_threshold =
        env_f32("SIM_PARTICLE_AIR_THRESHOLD").unwrap_or(particle_occ_threshold * 0.6);
    let particle_cfl = env_f32("SIM_PARTICLE_CFL").unwrap_or(0.5);
    let particle_max_vel = env_f32("SIM_PARTICLE_MAX_VEL").unwrap_or(200.0);
    let particle_gpu_p2g = env_usize("SIM_GPU_P2G").unwrap_or(1) != 0;
    let particle_gpu_p2g_scale = env_f32("SIM_GPU_P2G_SCALE").unwrap_or(1024.0);
    let validate_speck_threshold =
        env_f32("SIM_VALIDATE_SPECK_THRESHOLD").unwrap_or(particle_occ_threshold);
    let particle_boundary_mode = env_string("SIM_PARTICLE_BOUNDARY")
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "slip".to_string());
    let particle_hud_lines = vec![
        format!("RENDER {}", particle_render_mode.to_ascii_uppercase()),
        format!("BOUND {}", particle_boundary_mode.to_ascii_uppercase()),
        format!("APIC {}", if particle_apic { "ON" } else { "OFF" }),
        format!("GPU P2G {}", if particle_gpu_p2g { "ON" } else { "OFF" }),
    ];
    let particle_boundary_config = match particle_boundary_mode.as_str() {
        "no_slip" | "noslip" => BoundaryConfig::no_slip(),
        _ => BoundaryConfig {
            left: BoundaryCondition::Slip,
            right: BoundaryCondition::Slip,
            bottom: BoundaryCondition::Slip,
            top: BoundaryCondition::Slip,
        },
    };
    let (mut particle_density, particle_target_volume) = {
        let phi = particle_level_set_from_particles(
            &particles,
            grid,
            particle_bounds,
            particle_ls_radius,
            particle_ls_max,
            particle_pls_reinit_iters,
            particle_pls_reinit_dt,
            particle_volume_blend,
            particle_volume_band,
            None,
        );
        let density = display_phi(&phi, particle_surface_band);
        let volume = volume_from_phi(&phi, particle_surface_band);
        (density, volume)
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
    let mut mode = if validation.is_some() {
        SimMode::Particles
    } else {
        start_mode()
    };
    let mut texture = Vec::new();
    let density_steps_per_frame = 4;
    let level_steps_per_frame = 4;
    let particle_steps_per_frame = env_usize("SIM_PARTICLE_STEPS").unwrap_or(2).max(1);
    let mut frame_count: u64 = 0;
    let mut avg_frame_ms = 0.0;
    let mut avg_sim_ms = 0.0;
    let mut particle_step_counter: u64 = 0;
    let mut particle_volume_raw = 0.0;
    let mut particle_volume_rendered = 0.0;
    let mut particle_dt_effective = particle_dt;
    let mut particle_resample_added = 0usize;
    let mut particle_resample_removed = 0usize;
    let validation_metrics = Rc::new(RefCell::new(validation.map(ValidationMetrics::new)));
    let metrics_handle = validation_metrics.clone();
    let mut step_frame = move |mode: SimMode,
                          frame_count: &mut u64,
                          avg_frame_ms: &mut f32,
                          avg_sim_ms: &mut f32,
                          mut app: Option<&mut VulkanApp>| {
        let frame_start = Instant::now();
        let sim_start = Instant::now();
        let before_stats = if *frame_count == 0 {
            Some(match mode {
                SimMode::Density => density_state.density.stats(),
                SimMode::LevelSet => level_state.phi.stats(),
                SimMode::DensityMg => density_mg_state.density.stats(),
                SimMode::Particles | SimMode::ParticlesGpu => particle_density.stats(),
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
            SimMode::Particles | SimMode::ParticlesGpu => {
                let use_gpu = mode == SimMode::ParticlesGpu;
                particle_resample_added = 0;
                particle_resample_removed = 0;
                for _ in 0..particle_steps_per_frame {
                    let mut dt_step = particle_dt;
                    if particle_cfl > 0.0 {
                        let mut max_vel = particle_grid.max_abs();
                        if !max_vel.is_finite() {
                            max_vel = 0.0;
                        }
                        let particle_vel = particles.max_speed();
                        if particle_vel.is_finite() {
                            max_vel = max_vel.max(particle_vel);
                        }
                        if max_vel > 0.0 {
                            let dx = particle_grid.grid().dx();
                            dt_step = dt_step.min(particle_cfl * dx / max_vel);
                        }
                    }
                    particle_dt_effective = dt_step;
                    particles.add_velocity(Vec2::new(0.0, particle_gravity * dt_step));
                    let particle_occ = particles.to_density_with_rest(
                        grid,
                        particle_occ_scale,
                        particle_rest_density,
                    );
                    let particle_flags = CellFlags::from_fn(grid.cell_grid(), |x, y| {
                        match particle_base_flags.get(x, y) {
                            CellType::Solid => CellType::Solid,
                            _ => {
                                let occ = particle_occ.get(x, y);
                                if occ >= particle_flag_threshold {
                                    CellType::Fluid
                                } else {
                                    CellType::Air
                                }
                            }
                        }
                    });
                    let mut gpu_grid = None;
                    if use_gpu && particle_gpu_p2g {
                        let gpu_result = app
                            .as_deref_mut()
                            .map(|app| {
                                app.particle_grid_gpu(&particles, grid, particle_gpu_p2g_scale)
                            })
                            .unwrap_or_else(|| {
                                Err(anyhow::anyhow!(
                                    "gpu p2g requires a renderer for mode 5"
                                ))
                            });
                        match gpu_result {
                            Ok(grid_vel) => gpu_grid = Some(grid_vel),
                            Err(err) => {
                                eprintln!("gpu p2g error: {err:#}");
                            }
                        }
                    }
                    particle_grid = if let Some(grid_vel) = gpu_grid {
                        grid_vel
                    } else if particle_apic {
                        particles.to_grid_apic(grid)
                    } else {
                        particles.to_grid(grid)
                    };
                    particle_grid =
                        apply_domain_boundaries(&particle_grid, particle_boundary_config);
                    particle_grid = apply_solid_boundaries(&particle_grid, &particle_flags);
                    particle_prev_grid.clone_from(&particle_grid);
                    particle_grid = project_with_flags(
                        &particle_grid,
                        &particle_flags,
                        particle_pressure_iters,
                        particle_pressure_tol,
                    );
                    if particle_extrap_iters > 0 {
                        particle_grid = extrapolate_velocity(
                            &particle_grid,
                            &particle_flags,
                            particle_extrap_iters,
                        );
                    }
                    particle_grid =
                        apply_domain_boundaries(&particle_grid, particle_boundary_config);
                    particle_grid = apply_solid_boundaries(&particle_grid, &particle_flags);
                    if particle_apic {
                        particles.update_velocities_from_grid_apic(
                            &particle_grid,
                            Some(&particle_prev_grid),
                            particle_flip_ratio,
                        );
                        particles.clamp_affines(particle_apic_clamp);
                    } else {
                        particles.update_velocities_from_grid(
                            &particle_grid,
                            Some(&particle_prev_grid),
                            particle_flip_ratio,
                        );
                    }
                    let do_pbf = particle_pbf_iters > 0
                        && particle_pbf_every > 0
                        && (particle_step_counter % particle_pbf_every as u64) == 0;
                    let prev_positions = if do_pbf {
                        Some(particles.positions().to_vec())
                    } else {
                        None
                    };
                    if use_gpu {
                        let gpu_result = app
                            .as_deref_mut()
                            .map(|app| {
                                app.update_particles_gpu(
                                    &mut particles,
                                    grid,
                                    particle_bounds,
                                    dt_step,
                                    particle_bounce,
                                )
                            })
                            .unwrap_or_else(|| {
                                Err(anyhow::anyhow!("gpu particle step requires a renderer"))
                            });
                        if let Err(err) = gpu_result {
                            eprintln!("gpu particle step error: {err:#}");
                            particles.advect_rk2_grid(
                                &particle_grid,
                                dt_step,
                                particle_bounds,
                                particle_bounce,
                            );
                        }
                    } else {
                        particles.advect_rk2_grid(
                            &particle_grid,
                            dt_step,
                            particle_bounds,
                            particle_bounce,
                        );
                    }
                    if do_pbf {
                        if let Some(prev_positions) = prev_positions.as_deref() {
                            particles.pbf_project(
                                particle_bounds,
                                particle_pbf_radius,
                                particle_pbf_rest,
                                particle_pbf_iters,
                                dt_step,
                                particle_bounce,
                                particle_pbf_scorr_k,
                                particle_pbf_scorr_n,
                                particle_pbf_scorr_q,
                                prev_positions,
                            );
                        }
                    } else if particle_separation_iters > 0
                        && particle_separation_every > 0
                        && (particle_step_counter % particle_separation_every as u64) == 0
                    {
                        particles.separate(
                            particle_separation,
                            particle_bounds,
                            particle_separation_iters,
                        );
                    }
                    let do_resample = particle_resample
                        && (particle_step_counter % particle_resample_every as u64) == 0;
                    let mut occ_resample = None;
                    if do_resample {
                        let occ_field = particles.to_density_with_rest(
                            grid,
                            particle_occ_scale,
                            particle_rest_density,
                        );
                        let (added, removed) = particles.resample_volume(
                            grid,
                            particle_bounds,
                            &occ_field,
                            &particle_grid,
                            particle_target_count,
                            particle_flag_threshold,
                            particle_resample_low,
                            particle_resample_high,
                            particle_resample_max,
                            particle_resample_jitter,
                            particle_resample_seed ^ particle_step_counter as u32,
                        );
                        particle_resample_added += added;
                        particle_resample_removed += removed;
                        occ_resample = Some(occ_field);
                    }
                    if let Some(obs) = obstacle {
                        particles.apply_circle_obstacle(obs.center, obs.radius, obs.bounce);
                    }
                    if particle_post_project {
                        particle_grid = if particle_apic {
                            particles.to_grid_apic(grid)
                        } else {
                            particles.to_grid(grid)
                        };
                        particle_grid =
                            apply_domain_boundaries(&particle_grid, particle_boundary_config);
                        particle_grid = apply_solid_boundaries(&particle_grid, &particle_flags);
                        particle_grid = project_with_flags(
                            &particle_grid,
                            &particle_flags,
                            particle_post_project_iters,
                            particle_pressure_tol,
                        );
                        if particle_extrap_iters > 0 {
                            particle_grid = extrapolate_velocity(
                                &particle_grid,
                                &particle_flags,
                                particle_extrap_iters,
                            );
                        }
                        particle_grid =
                            apply_domain_boundaries(&particle_grid, particle_boundary_config);
                        particle_grid = apply_solid_boundaries(&particle_grid, &particle_flags);
                        if particle_apic {
                            particles.update_velocities_from_grid_apic(&particle_grid, None, 0.0);
                            particles.clamp_affines(particle_apic_clamp);
                        } else {
                            particles.update_velocities_from_grid(&particle_grid, None, 0.0);
                        }
                    }
                    if particle_air_drag > 0.0 {
                        let occ_drag = if let Some(ref occ) = occ_resample {
                            occ
                        } else {
                            occ_resample.get_or_insert_with(|| {
                                particles.to_density_with_rest(
                                    grid,
                                    particle_occ_scale,
                                    particle_rest_density,
                                )
                            })
                        };
                        particles.apply_air_drag(
                            occ_drag,
                            particle_air_threshold,
                            particle_air_drag,
                            dt_step,
                        );
                    }
                    if particle_xsph > 0.0 {
                        particles.xsph_viscosity(
                            particle_bounds,
                            particle_xsph_radius,
                            particle_xsph,
                        );
                    }
                    if particle_max_vel > 0.0 {
                        particles.clamp_speeds(particle_max_vel);
                    }
                    particle_step_counter = particle_step_counter.wrapping_add(1);
                }
                let particle_phi = particle_level_set_from_particles(
                    &particles,
                    grid,
                    particle_bounds,
                    particle_ls_radius,
                    particle_ls_max,
                    particle_pls_reinit_iters,
                    particle_pls_reinit_dt,
                    particle_volume_blend,
                    particle_volume_band,
                    Some((particle_target_volume, particle_surface_band)),
                );
                particle_volume_raw = volume_from_phi(&particle_phi, particle_surface_band);
                if particle_render_mode == "sph" {
                    let mut density = particles.to_sph_density_with_rest(
                        grid,
                        particle_render_radius,
                        particle_render_scale,
                        particle_render_rest,
                    );
                    let particle_occ = particles.to_density_with_rest(
                        grid,
                        particle_occ_scale,
                        particle_rest_density,
                    );
                    density.update_with_index(|x, y, value| {
                        if particle_occ.get(x, y) >= particle_occ_threshold {
                            value
                        } else {
                            0.0
                        }
                    });
                    let current_sum = density.sum();
                    if current_sum > 0.0 && particle_target_volume > 0.0 {
                        density.scale_in_place(particle_target_volume / current_sum);
                    }
                    clamp_field_in_place(&mut density, 0.0, 1.0);
                    if particle_surface_clean {
                        apply_surface_hysteresis(
                            &mut density,
                            particle_surface_strong,
                            particle_surface_weak,
                            particle_surface_neighbors,
                        );
                    }
                    let filtered_sum = density.sum();
                    if filtered_sum > 0.0 && particle_target_volume > 0.0 {
                        density.scale_in_place(particle_target_volume / filtered_sum);
                        clamp_field_in_place(&mut density, 0.0, 1.0);
                    }
                    particle_volume_rendered = density.sum();
                    particle_density = density;
                } else {
                    let render_phi = if particle_render_volume {
                        let corrected = correct_phi_volume(
                            &particle_phi,
                            particle_target_volume,
                            particle_surface_band,
                        );
                        if particle_volume_band > 0.0 {
                            corrected.zip_with(&particle_phi, |corr, orig| {
                                if orig.abs() <= particle_volume_band {
                                    corr
                                } else {
                                    orig
                                }
                            })
                        } else {
                            corrected
                        }
                    } else {
                        particle_phi
                    };
                    particle_volume_rendered = volume_from_phi(&render_phi, particle_surface_band);
                    particle_density = display_phi(&render_phi, particle_surface_band);
                }
                if let Some(metrics) = metrics_handle.borrow_mut().as_mut() {
                    let div = divergence(&particle_grid).abs_sum();
                    metrics.frames += 1;
                    metrics.max_div = metrics.max_div.max(div);
                    metrics.min_vol_raw = metrics.min_vol_raw.min(particle_volume_raw);
                    metrics.max_vol_raw = metrics.max_vol_raw.max(particle_volume_raw);
                    let speck_field = particles.to_density_with_rest(
                        grid,
                        particle_occ_scale,
                        particle_rest_density,
                    );
                    let specks =
                        count_surface_specks(&speck_field, validate_speck_threshold, speck_start_y);
                    metrics.specks_sum += specks;
                    metrics.specks_max = metrics.specks_max.max(specks);
                    metrics.samples += 1;
                }
                density_to_luma(&particle_density, &mut texture);
                (particle_dt_effective, 0.0, 0)
            }
        };
        if let Some((sum_before, min_before, max_before, non_finite_before)) = before_stats {
            let (sum_after, min_after, max_after, non_finite_after) = match mode {
                SimMode::Density => density_state.density.stats(),
                SimMode::LevelSet => level_state.phi.stats(),
                SimMode::DensityMg => density_mg_state.density.stats(),
                SimMode::Particles | SimMode::ParticlesGpu => particle_density.stats(),
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
            if matches!(mode, SimMode::Particles | SimMode::ParticlesGpu) {
                &particle_hud_lines
            } else {
                &[]
            },
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
                        SimMode::Particles | SimMode::ParticlesGpu => (
                            particles.len() as f32,
                            divergence(&particle_grid).abs_sum(),
                            particle_grid.max_abs(),
                            None,
                            None,
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
                SimMode::Particles | SimMode::ParticlesGpu => None,
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
                SimMode::Particles | SimMode::ParticlesGpu => {
                    particle_droplet_vy_stats(&particles, init.base_height)
                }
            };
            let droplet_center = match mode {
                SimMode::Density => {
                    droplet_center(&density_state.density, density_threshold, init.base_height)
                }
                SimMode::DensityMg => {
                    droplet_center(&density_mg_state.density, density_threshold, init.base_height)
                }
                SimMode::LevelSet => None,
                SimMode::Particles | SimMode::ParticlesGpu => {
                    particle_droplet_center(&particles, init.base_height)
                }
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
            let particle_volume_label = match mode {
                SimMode::Particles | SimMode::ParticlesGpu => format!(
                    " vol_raw={:.2} vol_render={:.2}",
                    particle_volume_raw, particle_volume_rendered
                ),
                _ => String::new(),
            };
            let particle_resample_label = match mode {
                SimMode::Particles | SimMode::ParticlesGpu => {
                    if particle_resample_added > 0 || particle_resample_removed > 0 {
                        format!(
                            " resample=+{}-{}",
                            particle_resample_added, particle_resample_removed
                        )
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            };
            if let Some((dye_sum, dye_min, dye_max, dye_nonfinite)) = dye_stats {
                if let Some((cx, cy)) = dye_center {
                    println!(
                        "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2} dye_sum={:.2} dye_min={:.3} dye_max={:.3} dye_nf={} dye_center=({:.2},{:.2}){}{}{}{}{}",
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
                        droplet_center_label,
                        particle_volume_label,
                        particle_resample_label
                    );
                } else {
                    println!(
                        "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2} dye_sum={:.2} dye_min={:.3} dye_max={:.3} dye_nf={}{}{}{}{}{}",
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
                        droplet_center_label,
                        particle_volume_label,
                        particle_resample_label
                    );
                }
            } else {
                println!(
                    "frame {} mode={} sim_ms={:.2} frame_ms={:.2} fps={:.1} mass={:.2} div={:.2} vel_max={:.2}{}{}{}{}{}",
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
                    droplet_center_label,
                    particle_volume_label,
                    particle_resample_label
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
                            SimMode::Particles | SimMode::ParticlesGpu => {
                                if dump_mode == DumpField::Density || dump_mode == DumpField::Both {
                                    dump_matrix(
                                        "matrix particles",
                                        *frame_count,
                                        &particle_density,
                                        stride,
                                    );
                                }
                            }
                        }
                    }
                }
        if let Some(app) = app.as_deref_mut() {
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
        if let Some(metrics) = validation_metrics.borrow_mut().take() {
            let samples = metrics.samples.max(1) as f32;
            let specks_avg = metrics.specks_sum as f32 / samples;
            let volume_loss = if particle_target_volume > 0.0 {
                ((particle_target_volume - metrics.min_vol_raw) / particle_target_volume).max(0.0)
            } else {
                0.0
            };
            println!(
                "validation case={} frames={} vol_raw_min={:.2} vol_loss_pct={:.2} max_div={:.2} specks_avg={:.2} specks_max={}",
                metrics.case.label(),
                metrics.frames,
                metrics.min_vol_raw,
                volume_loss * 100.0,
                metrics.max_div,
                specks_avg,
                metrics.specks_max
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
                        winit::event::VirtualKeyCode::Key4 => {
                            mode = SimMode::Particles;
                        }
                        winit::event::VirtualKeyCode::Key5 => {
                            mode = SimMode::ParticlesGpu;
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
