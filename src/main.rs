// FAST SPH Real-time Renderer - Optimized for speed!
// Run with: cargo run --release

use sph_water_sim::sph::{Integrator, ParticleSystem};
use pixels::{Pixels, SurfaceTexture};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent, VirtualKeyCode};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use std::io::Write;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 1200;

fn main() {
    println!("=== FAST SPH Water Simulation ===\n");

    let integrator = parse_integrator();
    
    // Domain setup
    let domain_width = 0.2;  // 20cm
    let domain_height = 0.3; // 30cm
    
    let mut system = ParticleSystem::new(domain_width, domain_height);
    
    // OPTIMIZED SPH parameters for speed
    system.smoothing_length = 0.008;  // 8mm (LARGER = fewer neighbors = faster!)
    system.rest_density = 1000.0;
    system.stiffness = 5000.0;
    system.viscosity = 0.001;  // Much lower! Real water viscosity
    system.gravity = 9.81;
    
    // FEWER PARTICLES for speed
    let particle_spacing = system.smoothing_length * 0.7;  // ~5.6mm spacing
    let particle_mass = particle_spacing * particle_spacing * system.rest_density;
    
    println!("🚀 SPEED OPTIMIZED:");
    println!("  Particle spacing: {:.1}mm (fewer particles)", particle_spacing * 1000.0);
    println!("  Smoothing length: {:.1}mm (larger search radius)", system.smoothing_length * 1000.0);
    println!("  Integrator: {}", integrator_label(integrator));
    
    // Add water at bottom (1/3 of domain)
    let water_height = domain_height / 3.0;
    system.add_particles_in_box(
        0.01,
        0.01,
        domain_width - 0.01,
        water_height,
        particle_spacing,
        particle_mass,
    );
    
    println!("  Initial particles: {} (vs ~1800 before)", system.num_particles());
    
    // MINIMAL settling - just enough to avoid explosion
    println!("\n⚡ Fast initialization...");
    system.initialize_hydrostatic_pressure();
    
    print!("  Quick settle...");
    std::io::stdout().flush().unwrap();
    system.settle(0.1, 0.7);  // Only 0.1s!
    println!(" Done!");
    
    // Add droplet
    system.add_circular_droplet(
        domain_width / 2.0,
        0.22,
        0.02,               // 2cm radius
        particle_spacing,
        particle_mass,
    );
    
    println!("  Total particles: {} ⚡", system.num_particles());
    println!();
    println!("Expected performance: ~60 FPS (real-time!)");
    println!();
    println!("Controls:");
    println!("  ESC/Q: Quit");
    println!("  SPACE: Pause/Resume");
    println!();
    
    // Setup window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("FAST SPH - Real-time Water Simulation")
        .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    
    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let mut pixels = Pixels::new(WINDOW_WIDTH, WINDOW_HEIGHT, surface_texture).unwrap();
    
    let mut time = 0.0;
    let mut frame = 0;
    let mut paused = false;
    let initial_mass = system.total_mass();
    let start = std::time::Instant::now();
    
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    use winit::event::ElementState;
                    if let Some(keycode) = input.virtual_keycode {
                        if input.state == ElementState::Pressed {
                            match keycode {
                                VirtualKeyCode::Escape | VirtualKeyCode::Q => {
                                    control_flow.set_exit();
                                }
                                VirtualKeyCode::Space => {
                                    paused = !paused;
                                    println!("{}", if paused { "⏸ Paused" } else { "▶ Resumed" });
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                if !paused {
                    // Run simulation - SINGLE step per frame for speed!
                    let dt = system.step_with_cfl(0.3, integrator);  // CFL = 0.3
                    time += dt;
                }
                
                // Render
                render(&system, pixels.frame_mut(), domain_width, domain_height);
                
                if let Err(e) = pixels.render() {
                    eprintln!("Render error: {}", e);
                    control_flow.set_exit();
                }
                
                frame += 1;
                
                // Stats every second
                if frame % 60 == 0 {
                    let elapsed = start.elapsed().as_secs_f32();
                    let fps = frame as f32 / elapsed;
                    let sim_ratio = time / elapsed;
                    let current_mass = system.total_mass();
                    let retention = current_mass / initial_mass * 100.0;
                    
                    println!("t={:.2}s | FPS: {:.1} | Sim: {:.2}× real-time | Mass: {:.1}% | Particles: {}", 
                             time, fps, sim_ratio, retention, system.num_particles());
                }
                
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn render(system: &ParticleSystem, frame: &mut [u8], domain_width: f32, domain_height: f32) {
    // Clear to white
    for pixel in frame.chunks_exact_mut(4) {
        pixel[0] = 240;
        pixel[1] = 240;
        pixel[2] = 240;
        pixel[3] = 255;
    }
    
    // Draw particles (larger for visibility with fewer particles)
    let scale_x = WINDOW_WIDTH as f32 / domain_width;
    let scale_y = WINDOW_HEIGHT as f32 / domain_height;
    let particle_radius = 3;  // Larger visualization
    
    for p in &system.particles {
        let px = (p.position.x * scale_x) as i32;
        let py = (WINDOW_HEIGHT as i32) - (p.position.y * scale_y) as i32;
        
        // Draw particle as circle
        for dy in -particle_radius..=particle_radius {
            for dx in -particle_radius..=particle_radius {
                if dx*dx + dy*dy <= particle_radius*particle_radius {
                    let x = (px + dx).clamp(0, WINDOW_WIDTH as i32 - 1) as u32;
                    let y = (py + dy).clamp(0, WINDOW_HEIGHT as i32 - 1) as u32;
                    let idx = ((y * WINDOW_WIDTH + x) * 4) as usize;
                    
                    if idx + 3 < frame.len() {
                        // Blue water
                        frame[idx] = 30;
                        frame[idx + 1] = 100;
                        frame[idx + 2] = 255;
                        frame[idx + 3] = 255;
                    }
                }
            }
        }
    }
}

fn parse_integrator() -> Integrator {
    let mut integrator = Integrator::Verlet;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--integrator" => {
                if let Some(value) = args.next() {
                    if let Some(parsed) = parse_integrator_value(&value) {
                        integrator = parsed;
                    }
                }
            }
            _ => {
                if let Some(value) = arg.strip_prefix("--integrator=") {
                    if let Some(parsed) = parse_integrator_value(value) {
                        integrator = parsed;
                    }
                }
            }
        }
    }

    integrator
}

fn parse_integrator_value(value: &str) -> Option<Integrator> {
    match value {
        "verlet" => Some(Integrator::Verlet),
        "euler" => Some(Integrator::Euler),
        _ => None,
    }
}

fn integrator_label(integrator: Integrator) -> &'static str {
    match integrator {
        Integrator::Verlet => "verlet",
        Integrator::Euler => "euler",
    }
}
