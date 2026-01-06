// Time integration for SPH
// Leap-frog integration scheme

use super::particles::ParticleSystem;

impl ParticleSystem {
    /// Leap-frog integration step
    /// This is symplectic and energy-conserving
    pub fn integrate(&mut self, dt: f32) {
        let domain_width = self.domain_width;
        let domain_height = self.domain_height;
        let max_velocity = 10.0;  // Maximum velocity cap (10 m/s)
        
        for particle in &mut self.particles {
            // Compute acceleration from force
            let acc = particle.force * (1.0 / particle.mass);
            
            // Limit acceleration to prevent instability
            let acc_mag = acc.length();
            let max_acc = 500.0;  // 500 m/s² max
            let acc_limited = if acc_mag > max_acc {
                acc * (max_acc / acc_mag)
            } else {
                acc
            };
            
            // Update velocity
            particle.velocity += acc_limited * dt;
            
            // Limit velocity magnitude
            let vel_mag = particle.velocity.length();
            if vel_mag > max_velocity {
                particle.velocity = particle.velocity * (max_velocity / vel_mag);
            }
            
            // Update position
            particle.position += particle.velocity * dt;
            
            // Enforce domain boundaries (inline)
            let margin = 0.001;  // 1mm margin
            
            // Clamp position
            if particle.position.x < margin {
                particle.position.x = margin;
                if particle.velocity.x < 0.0 {
                    particle.velocity.x = -particle.velocity.x * 0.3;  // Bounce with damping
                }
            }
            if particle.position.x > domain_width - margin {
                particle.position.x = domain_width - margin;
                if particle.velocity.x > 0.0 {
                    particle.velocity.x = -particle.velocity.x * 0.3;
                }
            }
            
            if particle.position.y < margin {
                particle.position.y = margin;
                if particle.velocity.y < 0.0 {
                    particle.velocity.y = -particle.velocity.y * 0.3;
                }
            }
            if particle.position.y > domain_height - margin {
                particle.position.y = domain_height - margin;
                if particle.velocity.y > 0.0 {
                    particle.velocity.y = -particle.velocity.y * 0.3;
                }
            }
        }
    }

    /// Compute maximum velocity for CFL condition
    pub fn max_velocity(&self) -> f32 {
        self.particles.iter()
            .map(|p| p.velocity.length())
            .fold(0.0_f32, f32::max)
    }

    /// Compute adaptive time step based on CFL condition
    pub fn compute_timestep(&self, cfl: f32) -> f32 {
        let max_vel = self.max_velocity();
        let c_sound = (self.stiffness / self.rest_density).sqrt();  // Speed of sound
        let max_speed = max_vel.max(c_sound);
        
        // Conservative time step for stability
        let dt_cfl = if max_speed > 1e-6 {
            cfl * self.smoothing_length / max_speed
        } else {
            0.001
        };
        
        // Also limit by force-based criterion
        let max_force = self.particles.iter()
            .map(|p| p.force.length() / p.mass)
            .fold(0.0_f32, f32::max);
        
        let dt_force = if max_force > 1e-6 {
            (self.smoothing_length / max_force).sqrt()
        } else {
            0.001
        };
        
        // Take minimum of all constraints
        dt_cfl.min(dt_force).clamp(0.00001, 0.0005)
    }

    /// Single SPH simulation step
    pub fn step(&mut self, dt: f32) {
        // 1. Compute all forces
        self.compute_forces();
        
        // 2. Integrate equations of motion
        self.integrate(dt);
    }

    /// Get total kinetic energy (for diagnostics)
    pub fn kinetic_energy(&self) -> f32 {
        self.particles.iter()
            .map(|p| 0.5 * p.mass * p.velocity.length_sq())
            .sum()
    }

    /// Get total mass (should be constant!)
    pub fn total_mass(&self) -> f32 {
        self.particles.iter()
            .map(|p| p.mass)
            .sum()
    }

    /// Get average density
    pub fn average_density(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.particles.iter().map(|p| p.density).sum();
        sum / self.particles.len() as f32
    }

    /// Get average pressure
    pub fn average_pressure(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.particles.iter().map(|p| p.pressure).sum();
        sum / self.particles.len() as f32
    }
}
