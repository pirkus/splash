use anyhow::Result;
use nav_stokes_sim::{
    flags_from_density, flags_from_phi, level_set_step, overlay_text, phi_to_density,
    sharpen_density, volume_from_phi, AdvectionScheme, BoundaryConfig, CellFlags, CellType, Field2,
    LevelSetParams, LevelSetState, MacGrid2, MacSimParams, MacSimState, MacVelocity2, Vec2,
    VulkanApp, GLYPH_HEIGHT, LINE_SPACING,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum SimMode {
    Density,
    LevelSet,
}

impl SimMode {
    fn toggle(self) -> Self {
        match self {
            SimMode::Density => SimMode::LevelSet,
            SimMode::LevelSet => SimMode::Density,
        }
    }

    fn tag(self) -> char {
        match self {
            SimMode::Density => '1',
            SimMode::LevelSet => '2',
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct InitConfig {
    base_height: f32,
    drop_radius: f32,
    drop_center: (f32, f32),
}

impl InitConfig {
    fn new(grid: MacGrid2) -> Self {
        let size = grid.width().min(grid.height()) as f32;
        let base_height = grid.height() as f32 * 0.5;
        let drop_radius = size * 0.05;
        let drop_center = (
            grid.width() as f32 * 0.5,
            base_height + drop_radius * 2.2,
        );
        Self {
            base_height,
            drop_radius,
            drop_center,
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

fn init_density_state(grid: MacGrid2, config: InitConfig, threshold: f32) -> MacSimState {
    let phi = initial_phi(grid, config);
    let density = phi_to_density(&phi, 0.0);
    let velocity = MacVelocity2::new(grid, Vec2::zero());
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
    let velocity = MacVelocity2::new(grid, Vec2::zero());
    let base_flags = CellFlags::new(grid.cell_grid(), CellType::Fluid);
    let flags = flags_from_phi(&phi, &base_flags);
    LevelSetState { phi, velocity, flags }
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

fn display_density(density: &Field2, threshold: f32, band: f32) -> Field2 {
    sharpen_density(density, threshold, band)
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

fn overlay_hud(
    texture: &mut [u8],
    width: usize,
    height: usize,
    mode: SimMode,
    dt: f32,
    cfl: f32,
    iters: usize,
) {
    let line_height = GLYPH_HEIGHT + LINE_SPACING;
    let line_1 = format!("{} DT {:.3}", mode.tag(), dt);
    let line_2 = format!("CFL {:.2}", cfl);
    let line_3 = format!("IT {}", iters);
    let mut y = 6;
    for line in [line_1, line_2, line_3] {
        overlay_text(texture, width, height, 6, y, &line, 240, true);
        y = y.saturating_add(line_height);
    }
}

fn main() -> Result<()> {
    let tex_width = 256;
    let tex_height = 256;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Navier-Stokes Sim")
        .with_inner_size(winit::dpi::LogicalSize::new(960.0, 720.0))
        .build(&event_loop)?;
    let mut app = VulkanApp::new(&window, (tex_width as u32, tex_height as u32))?;
    let grid = MacGrid2::new(tex_width, tex_height, 1.0);
    let init = InitConfig::new(grid);
    let density_threshold = 0.5;
    let density_band = 0.05;
    let surface_band = grid.dx() * 1.2;
    let mut density_state = init_density_state(grid, init, density_threshold);
    let mut level_state = init_level_set_state(grid, init);
    let target_volume = volume_from_phi(&level_state.phi, surface_band);
    let density_params = MacSimParams {
        dt: 0.04,
        cfl: 0.5,
        diffusion: 0.002,
        viscosity: 0.012,
        pressure_iters: 60,
        pressure_tol: 2e-4,
        advection: AdvectionScheme::Bfecc,
        boundaries: BoundaryConfig::no_slip(),
        body_force: Vec2::new(0.0, -12.0),
        buoyancy: 0.0,
        ambient_density: 0.0,
        surface_tension: 0.28,
        free_surface: true,
        density_threshold,
        surface_band: density_band,
        preserve_mass: false,
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
    let mut mode = SimMode::LevelSet;
    let mut texture = Vec::new();
    let density_steps_per_frame = 4;
    let level_steps_per_frame = 4;
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
                if input.state == winit::event::ElementState::Pressed {
                    if let Some(key) = input.virtual_keycode {
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
                            _ => {}
                        }
                    }
                }
            }
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                match mode {
                    SimMode::Density => {
                        let mut next = density_state.clone();
                        for _ in 0..density_steps_per_frame {
                            next = nav_stokes_sim::step(&next, density_params);
                        }
                        density_state = next;
                        let display =
                            display_density(&density_state.density, density_threshold, density_band);
                        density_to_luma(&display, &mut texture);
                        let dt = effective_dt(
                            density_params.dt,
                            density_params.cfl,
                            &density_state.velocity,
                        );
                        overlay_hud(
                            &mut texture,
                            tex_width,
                            tex_height,
                            mode,
                            dt,
                            density_params.cfl,
                            density_params.pressure_iters,
                        );
                    }
                    SimMode::LevelSet => {
                        let mut next = level_state.clone();
                        for _ in 0..level_steps_per_frame {
                            next = level_set_step(&next, level_params);
                        }
                        level_state = next;
                        let display = display_phi(&level_state.phi, surface_band);
                        density_to_luma(&display, &mut texture);
                        let dt = effective_dt(
                            level_params.dt,
                            level_params.cfl,
                            &level_state.velocity,
                        );
                        overlay_hud(
                            &mut texture,
                            tex_width,
                            tex_height,
                            mode,
                            dt,
                            level_params.cfl,
                            level_params.pressure_iters,
                        );
                    }
                }
                if let Err(err) = app.update_texture(&texture) {
                    eprintln!("texture upload error: {err:#}");
                }
                if let Err(err) = app.render() {
                    eprintln!("render error: {err:#}");
                }
            }
            _ => {}
        }
    });
}
