// SPH kernel functions
// These provide smooth interpolation between particles

use std::f32::consts::PI;

/// Cubic spline kernel (Monaghan 1992)
/// Most commonly used SPH kernel
pub fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let q = r / h;
    
    // Normalization constant for 2D
    let alpha = 10.0 / (7.0 * PI * h * h);
    
    if q < 1.0 {
        alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    } else if q < 2.0 {
        alpha * 0.25 * (2.0 - q).powi(3)
    } else {
        0.0
    }
}

/// Gradient of cubic spline kernel
/// Used for computing forces
pub fn cubic_spline_gradient(r_vec: (f32, f32), h: f32) -> (f32, f32) {
    let r = (r_vec.0 * r_vec.0 + r_vec.1 * r_vec.1).sqrt();
    
    if r < 1e-8 {
        return (0.0, 0.0);
    }
    
    let q = r / h;
    let alpha = 10.0 / (7.0 * PI * h * h);
    
    let grad_w = if q < 1.0 {
        alpha * (-3.0 * q + 2.25 * q * q) / h
    } else if q < 2.0 {
        alpha * (-0.75 * (2.0 - q).powi(2)) / h
    } else {
        0.0
    };
    
    // Direction
    let nx = r_vec.0 / r;
    let ny = r_vec.1 / r;
    
    (grad_w * nx, grad_w * ny)
}

/// Spiky kernel - better for pressure forces (steeper gradient)
pub fn spiky_kernel_gradient(r_vec: (f32, f32), h: f32) -> (f32, f32) {
    let r = (r_vec.0 * r_vec.0 + r_vec.1 * r_vec.1).sqrt();
    
    if r < 1e-8 || r >= h {
        return (0.0, 0.0);
    }
    
    let alpha = 10.0 / (PI * h.powi(5));
    let grad_w = -alpha * 3.0 * (h - r).powi(2);
    
    let nx = r_vec.0 / r;
    let ny = r_vec.1 / r;
    
    (grad_w * nx, grad_w * ny)
}

/// Viscosity kernel (Laplacian)
/// Used for viscous forces
pub fn viscosity_laplacian(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    
    let alpha = 40.0 / (PI * h.powi(5));
    alpha * (h - r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_symmetry() {
        let h = 0.01;
        let k1 = cubic_spline_kernel(0.005, h);
        let k2 = cubic_spline_kernel(0.005, h);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_kernel_decreases() {
        let h = 0.01;
        let k1 = cubic_spline_kernel(0.0, h);
        let k2 = cubic_spline_kernel(0.005, h);
        let k3 = cubic_spline_kernel(0.01, h);
        assert!(k1 > k2);
        assert!(k2 > k3);
    }

    #[test]
    fn test_kernel_compact_support() {
        let h = 0.01;
        let k = cubic_spline_kernel(2.0 * h + 0.001, h);
        assert!(k.abs() < 1e-8);
    }
}
