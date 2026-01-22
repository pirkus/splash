// Time integration for SPH
// Velocity Verlet (default) and semi-implicit Euler options

use super::particles::{Particle, ParticleSystem, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    Verlet,
    Euler,
}

impl ParticleSystem {
    fn clamp_particle_to_domain(
        particle: &mut Particle,
        domain_width: f32,
        domain_height: f32,
        margin: f32,
    ) {
        if particle.position.x < margin {
            particle.position.x = margin;
            if particle.velocity.x < 0.0 {
                particle.velocity.x = 0.0;
            }
        }
        if particle.position.x > domain_width - margin {
            particle.position.x = domain_width - margin;
            if particle.velocity.x > 0.0 {
                particle.velocity.x = 0.0;
            }
        }

        if particle.position.y < margin {
            particle.position.y = margin;
            if particle.velocity.y < 0.0 {
                particle.velocity.y = 0.0;
            }
        }
        if particle.position.y > domain_height - margin {
            particle.position.y = domain_height - margin;
            if particle.velocity.y > 0.0 {
                particle.velocity.y = 0.0;
            }
        }
    }

    fn limit_vector(vec: Vec2, max: f32) -> Vec2 {
        let len = vec.length();
        if len > max {
            vec * (max / len)
        } else {
            vec
        }
    }

    fn integrate_euler(&mut self, dt: f32) {
        let max_velocity = 10.0;
        let max_acc = 500.0;
        let domain_width = self.domain_width;
        let domain_height = self.domain_height;
        let margin = self.smoothing_length;

        for particle in &mut self.particles {
            let acc = particle.force * (1.0 / particle.mass);
            let acc = Self::limit_vector(acc, max_acc);

            particle.velocity += acc * dt;
            particle.velocity = Self::limit_vector(particle.velocity, max_velocity);
            particle.position += particle.velocity * dt;

            Self::clamp_particle_to_domain(particle, domain_width, domain_height, margin);
        }
    }

    pub fn integrate_verlet(&mut self, dt: f32) {
        let max_velocity = 10.0;
        let max_acc = 500.0;
        let half_dt = 0.5 * dt;
        let domain_width = self.domain_width;
        let domain_height = self.domain_height;
        let margin = self.smoothing_length;

        for particle in &mut self.particles {
            let acc = particle.force * (1.0 / particle.mass);
            let acc = Self::limit_vector(acc, max_acc);

            particle.velocity += acc * half_dt;
            particle.velocity = Self::limit_vector(particle.velocity, max_velocity);
            particle.position += particle.velocity * dt;

            Self::clamp_particle_to_domain(particle, domain_width, domain_height, margin);
        }

        self.compute_forces();

        for particle in &mut self.particles {
            let acc = particle.force * (1.0 / particle.mass);
            let acc = Self::limit_vector(acc, max_acc);

            particle.velocity += acc * half_dt;
            particle.velocity = Self::limit_vector(particle.velocity, max_velocity);
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
        let gamma = 7.0;
        let c_sound = (gamma * self.stiffness / self.rest_density).sqrt();  // Speed of sound
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

    /// Single SPH simulation step with adaptive CFL timestep
    pub fn step_with_cfl(&mut self, cfl: f32, integrator: Integrator) -> f32 {
        self.compute_forces();
        let dt = self.compute_timestep(cfl);
        self.step_with_integrator(dt, integrator);
        dt
    }

    /// Single SPH simulation step
    pub fn step(&mut self, dt: f32) {
        self.compute_forces();
        self.step_with_integrator(dt, Integrator::Verlet);
    }

    /// Single SPH simulation step with chosen integrator
    pub fn step_with_integrator(&mut self, dt: f32, integrator: Integrator) {
        match integrator {
            Integrator::Verlet => self.integrate_verlet(dt),
            Integrator::Euler => self.integrate_euler(dt),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::particles::{Particle, Vec2};

    #[test]
    fn test_timestep_decreases_with_stiffness_and_velocity() {
        let mut system = ParticleSystem::new(0.2, 0.3);
        system.smoothing_length = 0.001;
        system.rest_density = 1000.0;
        system.stiffness = 1000.0;

        let mut particle = Particle::new(Vec2::new(0.1, 0.1), 1.0);
        particle.velocity = Vec2::new(0.0, 0.0);
        system.particles.push(particle);

        let dt_base = system.compute_timestep(0.3);

        system.stiffness = 4000.0;
        let dt_stiffer = system.compute_timestep(0.3);
        assert!(dt_stiffer < dt_base);

        system.stiffness = 1000.0;
        system.particles[0].velocity = Vec2::new(10.0, 0.0);
        let dt_faster = system.compute_timestep(0.3);
        assert!(dt_faster < dt_base);
    }
}
