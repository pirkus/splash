// SPH force computation
// Pressure, viscosity, and gravity forces

use super::particles::{ParticleSystem, Vec2};
use super::kernels::{spiky_kernel_gradient, viscosity_laplacian};

impl ParticleSystem {
    /// Compute density for all particles using SPH interpolation
    pub fn compute_densities(&mut self) {
        let h = self.smoothing_length;
        let mut neighbors = Vec::new();
        
        for i in 0..self.particles.len() {
            self.get_neighbors_into(i, 2.0 * h, &mut neighbors);
            let pos_i = self.particles[i].position;
            
            // Self-contribution
            let mut density = self.particles[i].mass * super::kernels::cubic_spline_kernel(0.0, h);
            
            // Neighbor contributions
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let r = (pos_i - pos_j).length();
                let w = super::kernels::cubic_spline_kernel(r, h);
                density += self.particles[j].mass * w;
            }
            
            // Clamp density to reasonable range (prevent instability)
            let min_density = self.rest_density * 0.3;  // At least 30% of rest density
            let max_density = self.rest_density * 3.0;  // At most 3× rest density
            self.particles[i].density = density.clamp(min_density, max_density);
        }
    }

    /// Compute pressure from density using equation of state
    /// P = k * ((ρ / ρ₀)^γ - 1)
    pub fn compute_pressures(&mut self) {
        let rho0 = self.rest_density;
        let k = self.stiffness;
        let gamma = 7.0;  // Water-like
        
        for particle in &mut self.particles {
            let ratio = particle.density / rho0;
            
            // Softer pressure response for stability
            if ratio > 1.0 {
                // Compression: use equation of state
                particle.pressure = k * (ratio.powf(gamma) - 1.0);
            } else {
                // Tension: use linear response (more stable)
                particle.pressure = k * (ratio - 1.0);
            }
            
            // Clamp to reasonable range
            particle.pressure = particle.pressure.clamp(0.0, k * 10.0);
        }
    }

    /// Compute pressure forces using symmetric formulation
    /// F_pressure = -m * Σ(m_j * (P_i/ρ_i² + P_j/ρ_j²) * ∇W_ij)
    pub fn compute_pressure_forces(&mut self) {
        let h = self.smoothing_length;
        let mut forces = vec![Vec2::zero(); self.particles.len()];
        let mut neighbors = Vec::new();
        
        for (i, force) in forces.iter_mut().enumerate().take(self.particles.len()) {
            self.get_neighbors_into(i, h, &mut neighbors);
            let pos_i = self.particles[i].position;
            let rho_i = self.particles[i].density;
            let p_i = self.particles[i].pressure;
            let m_i = self.particles[i].mass;
            
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let rho_j = self.particles[j].density;
                let p_j = self.particles[j].pressure;
                let m_j = self.particles[j].mass;
                
                let r_vec = (pos_i.x - pos_j.x, pos_i.y - pos_j.y);
                let r = (r_vec.0 * r_vec.0 + r_vec.1 * r_vec.1).sqrt();
                
                // Prevent division by zero and extreme forces
                if r < 0.0001 {
                    continue;
                }
                
                let grad_w = spiky_kernel_gradient(r_vec, h);
                
                // Symmetric pressure force with safety clamping
                let pressure_term = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
                let force_mag = -m_i * m_j * pressure_term;
                
                // Limit maximum force to prevent explosions
                let max_force = 100.0 * m_i;  // Maximum force per particle pair
                let force_x = (force_mag * grad_w.0).clamp(-max_force, max_force);
                let force_y = (force_mag * grad_w.1).clamp(-max_force, max_force);
                
                force.x += force_x;
                force.y += force_y;
            }
        }
        
        // Apply forces
        for (i, force) in forces.iter().enumerate() {
            self.particles[i].force = *force;
        }
    }

    /// Compute viscosity forces
    /// F_viscosity = μ * Σ(m_j * (v_j - v_i) / ρ_j * ∇²W_ij)
    pub fn compute_viscosity_forces(&mut self) {
        let h = self.smoothing_length;
        let mu = self.viscosity;
        let mut forces = vec![Vec2::zero(); self.particles.len()];
        let mut neighbors = Vec::new();
        
        for (i, force) in forces.iter_mut().enumerate().take(self.particles.len()) {
            self.get_neighbors_into(i, h, &mut neighbors);
            let pos_i = self.particles[i].position;
            let vel_i = self.particles[i].velocity;
            let m_i = self.particles[i].mass;
            
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let vel_j = self.particles[j].velocity;
                let rho_j = self.particles[j].density;
                let m_j = self.particles[j].mass;
                
                let r = (pos_i - pos_j).length();
                let lap_w = viscosity_laplacian(r, h);
                
                let vel_diff = vel_j - vel_i;
                let visc_force = mu * m_i * m_j * (1.0 / rho_j) * lap_w;
                
                force.x += visc_force * vel_diff.x;
                force.y += visc_force * vel_diff.y;
            }
        }
        
        // Add to existing forces
        for (i, force) in forces.iter().enumerate() {
            self.particles[i].force += *force;
        }
    }

    /// Apply gravity
    pub fn compute_gravity_forces(&mut self) {
        for particle in &mut self.particles {
            let gravity_force = particle.mass * self.gravity;
            particle.force.y -= gravity_force;
        }
    }

    /// Apply boundary forces (penalty method)
    pub fn apply_boundary_forces(&mut self, boundary_stiffness: f32, boundary_damping: f32) {
        let margin = self.smoothing_length;
        let damping = boundary_damping.clamp(0.0, 1.0);
        
        for particle in &mut self.particles {
            // Left boundary
            if particle.position.x < margin {
                let penetration = margin - particle.position.x;
                let force = boundary_stiffness * penetration;
                particle.force.x += force;
                // Damping for wall collision
                if particle.velocity.x < 0.0 {
                    particle.velocity.x *= damping;
                }
            }
            
            // Right boundary
            if particle.position.x > self.domain_width - margin {
                let penetration = particle.position.x - (self.domain_width - margin);
                let force = boundary_stiffness * penetration;
                particle.force.x -= force;
                // Damping for wall collision
                if particle.velocity.x > 0.0 {
                    particle.velocity.x *= damping;
                }
            }
            
            // Bottom boundary
            if particle.position.y < margin {
                let penetration = margin - particle.position.y;
                let force = boundary_stiffness * penetration;
                particle.force.y += force;
                // Damping for bottom collision
                if particle.velocity.y < 0.0 {
                    particle.velocity.y *= damping;
                }
            }
            
            // Top boundary (open, no force, but prevent escape)
            if particle.position.y > self.domain_height - margin {
                let penetration = particle.position.y - (self.domain_height - margin);
                let force = boundary_stiffness * penetration;
                particle.force.y -= force;
                if particle.velocity.y > 0.0 {
                    particle.velocity.y *= damping;
                }
            }
        }
    }

    /// Compute all SPH forces
    pub fn compute_forces(&mut self) {
        // Clear forces
        for particle in &mut self.particles {
            particle.force = Vec2::zero();
        }
        
        // Rebuild spatial hash
        self.build_spatial_hash();
        
        // Compute densities and pressures
        self.compute_densities();
        self.compute_pressures();
        
        // Compute forces
        self.compute_pressure_forces();
        self.compute_viscosity_forces();
        self.compute_gravity_forces();
        
        // Softer boundary forces for stability
        let boundary_stiffness = self.stiffness * 0.8;
        let boundary_damping = 0.5;
        self.apply_boundary_forces(boundary_stiffness, boundary_damping);
        
        // Limit total force magnitude to prevent explosions
        let max_force = 1000.0;  // Maximum total force per particle
        for particle in &mut self.particles {
            let force_mag = particle.force.length();
            if force_mag > max_force {
                particle.force = particle.force * (max_force / force_mag);
            }
        }
    }
}
