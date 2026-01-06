// SPH particle data structures
// This replaces the grid-based approach

use std::io::Write;

#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len > 1e-8 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Self::zero()
        }
    }

    pub fn dot(&self, other: &Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, scalar: f32) -> Vec2 {
        Vec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, other: Vec2) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// SPH Particle
#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub mass: f32,
    pub density: f32,
    pub pressure: f32,
    pub force: Vec2,  // Accumulated force
}

impl Particle {
    pub fn new(position: Vec2, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec2::zero(),
            mass,
            density: 1000.0,  // Water density
            pressure: 0.0,
            force: Vec2::zero(),
        }
    }
}

/// SPH Particle System
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub domain_width: f32,
    pub domain_height: f32,
    
    // SPH parameters
    pub smoothing_length: f32,  // h
    pub rest_density: f32,       // ρ₀
    pub stiffness: f32,          // k (for pressure)
    pub viscosity: f32,          // μ
    pub gravity: f32,
    
    // Spatial hashing for neighbor search
    cell_size: f32,
    cells: Vec<Vec<usize>>,  // Cell -> list of particle indices
    nx_cells: usize,
    ny_cells: usize,
}

impl ParticleSystem {
    pub fn new(domain_width: f32, domain_height: f32) -> Self {
        let smoothing_length = 0.01;  // 1cm smoothing radius
        let cell_size = 2.0 * smoothing_length;
        let nx_cells = (domain_width / cell_size).ceil() as usize;
        let ny_cells = (domain_height / cell_size).ceil() as usize;
        let num_cells = nx_cells * ny_cells;
        
        Self {
            particles: Vec::new(),
            domain_width,
            domain_height,
            smoothing_length,
            rest_density: 1000.0,
            stiffness: 1000.0,
            viscosity: 0.001,
            gravity: 9.81,
            cell_size,
            cells: vec![Vec::new(); num_cells],
            nx_cells,
            ny_cells,
        }
    }

    /// Get cell index for a position
    fn get_cell_index(&self, pos: &Vec2) -> Option<usize> {
        let i = (pos.x / self.cell_size) as i32;
        let j = (pos.y / self.cell_size) as i32;
        
        if i >= 0 && i < self.nx_cells as i32 && j >= 0 && j < self.ny_cells as i32 {
            Some((j as usize) * self.nx_cells + (i as usize))
        } else {
            None
        }
    }

    /// Build spatial hash grid for neighbor search
    pub fn build_spatial_hash(&mut self) {
        // Clear cells
        for cell in &mut self.cells {
            cell.clear();
        }
        
        // Assign particles to cells
        for (idx, particle) in self.particles.iter().enumerate() {
            if let Some(cell_idx) = self.get_cell_index(&particle.position) {
                self.cells[cell_idx].push(idx);
            }
        }
    }

    /// Get neighbors within smoothing radius
    pub fn get_neighbors(&self, particle_idx: usize, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let pos = self.particles[particle_idx].position;
        
        // Check surrounding cells
        let i = (pos.x / self.cell_size) as i32;
        let j = (pos.y / self.cell_size) as i32;
        
        for dy in -1..=1 {
            for dx in -1..=1 {
                let ni = i + dx;
                let nj = j + dy;
                
                if ni >= 0 && ni < self.nx_cells as i32 && nj >= 0 && nj < self.ny_cells as i32 {
                    let cell_idx = (nj as usize) * self.nx_cells + (ni as usize);
                    
                    for &neighbor_idx in &self.cells[cell_idx] {
                        if neighbor_idx != particle_idx {
                            let diff = self.particles[neighbor_idx].position - pos;
                            if diff.length_sq() < radius * radius {
                                neighbors.push(neighbor_idx);
                            }
                        }
                    }
                }
            }
        }
        
        neighbors
    }

    /// Initialize particles in a rectangular region
    pub fn add_particles_in_box(&mut self, x_min: f32, y_min: f32, x_max: f32, y_max: f32, spacing: f32, particle_mass: f32) {
        let mut x = x_min + spacing * 0.5;
        while x < x_max {
            let mut y = y_min + spacing * 0.5;
            while y < y_max {
                let pos = Vec2::new(x, y);
                self.particles.push(Particle::new(pos, particle_mass));
                y += spacing;
            }
            x += spacing;
        }
        println!("Created {} particles", self.particles.len());
    }

    /// Initialize a circular droplet of particles
    pub fn add_circular_droplet(&mut self, center_x: f32, center_y: f32, radius: f32, spacing: f32, particle_mass: f32) {
        let x_min = center_x - radius;
        let x_max = center_x + radius;
        let y_min = center_y - radius;
        let y_max = center_y + radius;
        
        let mut x = x_min + spacing * 0.5;
        while x < x_max {
            let mut y = y_min + spacing * 0.5;
            while y < y_max {
                let dx = x - center_x;
                let dy = y - center_y;
                if dx * dx + dy * dy < radius * radius {
                    let pos = Vec2::new(x, y);
                    self.particles.push(Particle::new(pos, particle_mass));
                }
                y += spacing;
            }
            x += spacing;
        }
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Initialize hydrostatic pressure distribution
    /// P = ρ₀ * g * h (pressure increases with depth)
    pub fn initialize_hydrostatic_pressure(&mut self) {
        // First, compute SPH densities properly
        self.build_spatial_hash();
        let h = self.smoothing_length;
        
        for i in 0..self.particles.len() {
            let neighbors = self.get_neighbors(i, 2.0 * h);
            let pos_i = self.particles[i].position;
            
            // Compute SPH density
            let mut density = self.particles[i].mass * crate::sph::cubic_spline_kernel(0.0, h);
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let r = (pos_i - pos_j).length();
                let w = crate::sph::cubic_spline_kernel(r, h);
                density += self.particles[j].mass * w;
            }
            self.particles[i].density = density.clamp(self.rest_density * 0.3, self.rest_density * 3.0);
        }
        
        // Now compute pressure from SPH density using equation of state
        // This ensures consistency with simulation pressure computation
        let rho0 = self.rest_density;
        let k = self.stiffness;
        let gamma = 7.0;
        
        for particle in &mut self.particles {
            let ratio = particle.density / rho0;
            if ratio > 1.0 {
                particle.pressure = k * (ratio.powf(gamma) - 1.0);
            } else {
                particle.pressure = k * (ratio - 1.0);
            }
            particle.pressure = particle.pressure.clamp(0.0, k * 10.0);
        }
    }

    /// Let particles settle to hydrostatic equilibrium
    /// Run a few simulation steps with high damping
    pub fn settle(&mut self, duration: f32, damping: f32) {
        let steps = (duration / 0.0001).round() as usize;
        let dt = duration / steps as f32;
        
        println!("    Settling: {} steps with damping={:.2}", steps, damping);
        
        for step in 0..steps {
            // Progress indicator every 1000 steps
            if step % 1000 == 0 && step > 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
            
            // Full force computation
            self.build_spatial_hash();
            
            let h = self.smoothing_length;
            let mut forces = vec![Vec2::zero(); self.particles.len()];
            
            // Compute densities
            for i in 0..self.particles.len() {
                let neighbors = self.get_neighbors(i, 2.0 * h);
                let pos_i = self.particles[i].position;
                
                let mut density = self.particles[i].mass * crate::sph::cubic_spline_kernel(0.0, h);
                for &j in &neighbors {
                    let pos_j = self.particles[j].position;
                    let r = (pos_i - pos_j).length();
                    let w = crate::sph::cubic_spline_kernel(r, h);
                    density += self.particles[j].mass * w;
                }
                self.particles[i].density = density.clamp(self.rest_density * 0.3, self.rest_density * 3.0);
            }
            
            // Compute pressures
            let rho0 = self.rest_density;
            let k = self.stiffness;
            for particle in &mut self.particles {
                let ratio = particle.density / rho0;
                if ratio > 1.0 {
                    particle.pressure = k * (ratio.powf(7.0) - 1.0);
                } else {
                    particle.pressure = k * (ratio - 1.0);
                }
                particle.pressure = particle.pressure.clamp(0.0, k * 10.0);
            }
            
            // Compute forces
            for (i, force) in forces.iter_mut().enumerate().take(self.particles.len()) {
                // Gravity
                force.y = -self.particles[i].mass * self.gravity;
                
                // Pressure force - use local copies to avoid borrow issues
                let neighbors = self.get_neighbors(i, 2.0 * h);
                let pos_i = self.particles[i].position;
                let p_i = self.particles[i].pressure;
                let rho_i = self.particles[i].density;
                let m_i = self.particles[i].mass;
                
                let particles_copy: Vec<_> = neighbors.iter().map(|&j| {
                    (self.particles[j].position, self.particles[j].pressure, 
                     self.particles[j].density, self.particles[j].mass)
                }).collect();
                
                for (pos_j, p_j, rho_j, m_j) in particles_copy {
                    let r_vec = (pos_i.x - pos_j.x, pos_i.y - pos_j.y);
                    let r = (r_vec.0 * r_vec.0 + r_vec.1 * r_vec.1).sqrt();
                    if r < 0.0001 { continue; }
                    
                    let grad_w = crate::sph::spiky_kernel_gradient(r_vec, h);
                    let pressure_term = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
                    force.x += -m_i * m_j * pressure_term * grad_w.0;
                    force.y += -m_i * m_j * pressure_term * grad_w.1;
                }
            }
            
            // Integrate with damping - ALLOW particles to move to find equilibrium!
            let margin = 0.001;
            for (i, force) in forces.iter().enumerate() {
                let mass = self.particles[i].mass;
                let mut vel = self.particles[i].velocity;
                let mut pos = self.particles[i].position;
                
                let acc = *force * (1.0 / mass);
                vel += acc * dt;
                vel = vel * damping;  // Damping
                pos += vel * dt;  // Allow movement!
                
                // Keep in bounds
                pos.x = pos.x.clamp(margin, self.domain_width - margin);
                pos.y = pos.y.clamp(margin, self.domain_height - margin);
                
                self.particles[i].velocity = vel;
                self.particles[i].position = pos;
            }
        }
        
        println!();  // Newline after progress dots
        
        // Zero out velocities after finding equilibrium positions
        for particle in &mut self.particles {
            particle.velocity = Vec2::zero();
        }
    }
}
