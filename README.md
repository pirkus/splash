# SPH Water Simulation - Complete Mathematical Guide

Real-time 2D water simulation using Smoothed Particle Hydrodynamics (SPH).

## Quick Start

```bash
cargo run --release
```

**Controls:** ESC/Q = Quit, SPACE = Pause

---

## Table of Contents

1. [Governing Equations](#1-governing-equations)
2. [SPH Discretization](#2-sph-discretization)
3. [Kernel Functions](#3-kernel-functions)
4. [Density Computation](#4-density-computation)
5. [Pressure Forces](#5-pressure-forces)
6. [Viscosity Forces](#6-viscosity-forces)
7. [Time Integration](#7-time-integration)
8. [Boundary Conditions](#8-boundary-conditions)
9. [Numerical Stability](#9-numerical-stability)
10. [Implementation](#10-implementation)

---

## 1. Governing Equations

### The Navier-Stokes Equations

Incompressible fluid flow is governed by:

**Continuity (mass conservation):**
```
∂ρ/∂t + ∇·(ρv) = 0
```

**Momentum (Newton's 2nd law for fluids):**
```
ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + ρg
```

Where:
- `ρ(x,t)` = density [kg/m³]
- `v(x,t)` = velocity field [m/s]
- `p(x,t)` = pressure [Pa]
- `μ` = dynamic viscosity [Pa·s]
- `g` = gravitational acceleration [m/s²]

**Physical interpretation:**
- **Continuity**: Mass is conserved (water doesn't vanish)
- **Momentum**: Force = mass × acceleration, for every fluid particle
- **Incompressibility**: `∇·v = 0` (density stays constant)

For SPH, we use **weakly compressible** flow, allowing small density variations.

---

## 2. SPH Discretization

### The Core Idea

Replace continuous fields with **discrete particles**. Any scalar field `A(x)` is approximated by:

```
A(x) ≈ Σⱼ (mⱼ/ρⱼ) A(xⱼ) W(x - xⱼ, h)
```

Where:
- `mⱼ` = mass of particle j
- `ρⱼ` = density of particle j
- `W(r, h)` = **smoothing kernel** (interpolation function)
- `h` = smoothing length (interaction radius)
- Sum over all particles within distance `2h`

**Physical meaning:** Each particle "spreads" its properties over a region of size `h`.

### Gradient Approximation

The spatial gradient is:

```
∇A(xᵢ) ≈ Σⱼ (mⱼ/ρⱼ) A(xⱼ) ∇W(xᵢ - xⱼ, h)
```

Or using the **symmetric form** (better for momentum conservation):

```
∇A(xᵢ) ≈ ρᵢ Σⱼ mⱼ (Aᵢ/ρᵢ² + Aⱼ/ρⱼ²) ∇W(xᵢ - xⱼ, h)
```

### Laplacian Approximation

For viscosity, we need `∇²v`:

```
∇²v(xᵢ) ≈ Σⱼ (mⱼ/ρⱼ) (vⱼ - vᵢ) ∇²W(xᵢ - xⱼ, h)
```

---

## 3. Kernel Functions

### Requirements for Valid Kernels

A smoothing kernel `W(r, h)` must satisfy:

1. **Normalization:** `∫ W(r, h) dV = 1`
2. **Compact support:** `W(r, h) = 0` for `r > 2h`
3. **Dirac delta property:** `lim(h→0) W(r, h) = δ(r)`
4. **Smoothness:** At least C² continuous

### Cubic Spline Kernel (Monaghan & Lattanzio, 1985)

In 2D, let `q = r/h`:

```
W(r, h) = (10/(7πh²)) × {
    1 - (3/2)q² + (3/4)q³      if 0 ≤ q < 1
    (1/4)(2-q)³                if 1 ≤ q < 2
    0                          if q ≥ 2
}
```

**Gradient:**
```
∇W = (∂W/∂r) · (r⃗/r)

∂W/∂r = (10/(7πh³)) × {
    -3q + (9/4)q²       if 0 ≤ q < 1
    -(3/4)(2-q)²        if 1 ≤ q < 2
    0                   if q ≥ 2
}
```

**Properties:**
- Bell-shaped (maximum at r=0)
- Smooth transitions
- Zero beyond 2h (local interactions only)

### Spiky Kernel (for pressure)

```
W(r, h) = (15/(πh⁶)) × (h - r)³    for r < h
        = 0                         for r ≥ h
```

**Gradient magnitude:**
```
|∇W| = (45/(πh⁶)) × (h - r)²
```

This kernel has a **sharp gradient** near the center, making it good for pressure forces (strong repulsion when particles get close).

### Viscosity Laplacian Kernel

```
∇²W_visc(r, h) = (45/(πh⁶)) × (h - r)
```

This is always **positive**, ensuring numerical stability for viscosity.

---

## 4. Density Computation

### SPH Density Formula

```
ρᵢ = Σⱼ mⱼ W(|xᵢ - xⱼ|, h)
```

**Physical interpretation:**
- Each particle contributes mass to neighbors within radius 2h
- Denser regions = more particles nearby
- Sparser regions = fewer particles

**Properties:**
- **Exact mass conservation:** `Σᵢ mᵢ = const` (particles can't disappear!)
- **Self-contribution:** Particle i contributes to its own density
- **Local:** Only neighbors within 2h matter

**Implementation:**
```rust
for i in 0..n {
    density[i] = 0.0;
    for j in get_neighbors(i, 2*h) {
        let r = |pos[i] - pos[j]|;
        density[i] += mass[j] * W(r, h);
    }
}
```

---

## 5. Pressure Forces

### Equation of State (Tait)

Instead of solving `∇·v = 0`, we use an **equation of state**:

```
p = k((ρ/ρ₀)^γ - 1)
```

Where:
- `k` = stiffness constant [Pa]
- `ρ₀` = rest density (1000 kg/m³ for water)
- `γ = 7` (adiabatic index for water)

**Physical meaning:** Pressure increases **rapidly** when density deviates from ρ₀, enforcing near-incompressibility.

**Example:** If `ρ = 1.1ρ₀` (10% compression):
```
p = 5000 × ((1.1)^7 - 1) ≈ 4775 Pa
```

### Pressure Force Formula

From momentum equation: `F = -∇p`

Using symmetric SPH discretization:

```
(dv/dt)ᵢ|_pressure = -Σⱼ mⱼ (pᵢ/ρᵢ² + pⱼ/ρⱼ²) ∇W(xᵢ - xⱼ, h)
```

**Why symmetric?**
- Ensures `Fᵢⱼ = -Fⱼᵢ` (Newton's 3rd law)
- Conserves total momentum: `Σᵢ mᵢvᵢ = const`
- More stable numerically

**Implementation:**
```rust
for i in 0..n {
    for j in neighbors(i) {
        let grad_W = ∇W(pos[i] - pos[j], h);
        let term = mass[j] * (pressure[i]/density[i]² + pressure[j]/density[j]²);
        force[i] -= term * grad_W;
    }
}
```

### Stiffness Parameter

The "speed of sound" in the fluid is:

```
c_s = √(k·γ/ρ₀)
```

Typical values:
- `k = 5000` → `c_s ≈ 5.9 m/s` (stable, slightly compressible)
- `k = 50000` → `c_s ≈ 18.7 m/s` (stiffer, requires smaller timestep)

---

## 6. Viscosity Forces

### Physical Viscosity

From Navier-Stokes: `F_visc = μ∇²v`

Where `μ` is **dynamic viscosity** [Pa·s].

For water at 20°C: `μ ≈ 0.001 Pa·s`

### SPH Viscosity Formula

```
(dv/dt)ᵢ|_viscosity = ν Σⱼ (mⱼ/ρⱼ) (vⱼ - vᵢ) ∇²W_visc(xᵢ - xⱼ, h)
```

Where `ν = μ/ρ₀` is **kinematic viscosity** [m²/s].

**Physical interpretation:**
- If neighbor j is faster → pull i forward (positive force)
- If neighbor j is slower → pull i backward (negative force)
- **Smooths out velocity differences** between neighbors

**Effect:**
- Splashes settle down naturally
- Smooth, laminar flow
- Energy dissipation (no perpetual motion)

**Implementation:**
```rust
for i in 0..n {
    for j in neighbors(i) {
        let r = |pos[i] - pos[j]|;
        let lap_W = ∇²W_visc(r, h);
        let v_diff = vel[j] - vel[i];
        force[i] += viscosity * mass[j] / density[j] * v_diff * lap_W;
    }
}
```

---

## 7. Time Integration

### Leap-Frog (Verlet) Scheme

Second-order accurate, symplectic integrator:

```
v^(n+1/2) = v^(n-1/2) + a^n · Δt
x^(n+1) = x^n + v^(n+1/2) · Δt
```

**Properties:**
- Time-reversible
- Energy-conserving (symplectic)
- More stable than Euler
- Velocity and position are half-step offset

**Initialization (first step):**
```
v^(1/2) = v^0 + a^0 · (Δt/2)
```

### Adaptive Timestep (CFL Condition)

To maintain stability:

```
Δt ≤ C · min(Δt_courant, Δt_force, Δt_visc)
```

Where `C ≈ 0.2-0.4` is the CFL number.

**Courant condition (advection):**
```
Δt_courant = h / (v_max + c_s)
```

Ensures particles don't move more than one smoothing length per step.

**Force condition:**
```
Δt_force = √(h / a_max)
```

Prevents explosive accelerations.

**Viscosity condition:**
```
Δt_visc = h² / (2ν)
```

For explicit viscosity integration.

**Implementation:**
```rust
fn compute_timestep(&self, cfl: f32) -> f32 {
    let v_max = self.particles.iter().map(|p| p.velocity.length()).max();
    let a_max = self.particles.iter().map(|p| p.force.length() / p.mass).max();
    
    let c_s = (self.stiffness * 7.0 / self.rest_density).sqrt();
    let dt_courant = self.smoothing_length / (v_max + c_s);
    let dt_force = (self.smoothing_length / a_max).sqrt();
    
    cfl * dt_courant.min(dt_force).min(0.001)
}
```

---

## 8. Boundary Conditions

### Penalty Forces (Soft Boundaries)

For each particle near a wall (distance `d < ε`):

**Normal repulsion:**
```
F_boundary = k_boundary · (ε - d) · n̂
```

Where:
- `k_boundary` ≈ 5000-10000 (stiffness)
- `n̂` = outward normal
- `ε` = boundary thickness (~2h)

**Tangential damping (no-slip):**
```
v_tangential *= damping_factor  (e.g., 0.3)
```

**Implementation:**
```rust
if particle.x < margin {
    let penetration = margin - particle.x;
    force.x += boundary_stiffness * penetration;
    if velocity.x < 0.0 {
        velocity.x *= 0.3;  // Damp tangential velocity
    }
}
```

This ensures:
1. Particles can't escape domain
2. Tangential velocity is damped (viscous boundary layer)

---

## 9. Numerical Stability

### Challenges

1. **Particle clustering** - attractive forces can cause clumping
2. **Tensile instability** - negative pressure causes unphysical separation
3. **Energy blowup** - explicit integration accumulates errors

### Stabilization Techniques

#### 1. Density Clamping
```rust
density[i] = density[i].clamp(0.5 * rest_density, 2.0 * rest_density);
```

#### 2. Velocity Limiting
```rust
if velocity.length() > max_velocity {
    velocity = velocity.normalized() * max_velocity;
}
```

#### 3. Force Limiting
```rust
if force.length() > max_force {
    force = force.normalized() * max_force;
}
```

#### 4. Hydrostatic Initialization

Initialize with pressure matching hydrostatic equilibrium:
```
p(y) = ρ₀ · g · (y_top - y)
```

Then run a "settling phase" with high damping (0.7-0.9) for 0.1-0.3 seconds to remove initial oscillations.

#### 5. XSPH Velocity Smoothing

Replace particle velocity with neighbor-averaged velocity:
```
vᵢ* = vᵢ + ε Σⱼ (mⱼ/ρⱼ) (vⱼ - vᵢ) W(xᵢⱼ, h)
```

Where `ε ≈ 0.5`. This reduces particle disorder.

---

## 10. Implementation

### Spatial Hashing for Neighbor Search

Naive neighbor search is O(N²). We use **spatial hashing**:

1. Divide domain into grid cells of size `2h`
2. Hash particles to cells: `cell = (⌊x/2h⌋, ⌊y/2h⌋)`
3. For each particle, search only 9 adjacent cells

**Complexity:** O(N) average case

```rust
let cell_size = 2.0 * smoothing_length;
let grid_x = (pos.x / cell_size).floor() as i32;
let grid_y = (pos.y / cell_size).floor() as i32;

// Search 3×3 neighborhood
for dx in -1..=1 {
    for dy in -1..=1 {
        let key = (grid_x + dx, grid_y + dy);
        if let Some(particles) = hash_map.get(&key) {
            // Check distance for each particle in cell
        }
    }
}
```

### Physical Parameters (Water at 20°C)

| Parameter | Value | Unit |
|-----------|-------|------|
| Rest density (ρ₀) | 1000 | kg/m³ |
| Dynamic viscosity (μ) | 0.001 | Pa·s |
| Kinematic viscosity (ν) | 1×10⁻⁶ | m²/s |
| Gravity (g) | 9.81 | m/s² |

### Simulation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Domain size | 0.2m × 0.3m | Tabletop scale |
| Smoothing length (h) | 8mm | ~3 particles per radius |
| Particle spacing | 5.6mm | 0.7h (typical ratio) |
| Stiffness (k) | 5000 Pa | ~1% compressibility |
| CFL number | 0.3 | Stability margin |
| Timestep | 0.1-1ms | Adaptive (CFL) |

### Performance

- **~800 particles** (vs ~1800 with smaller spacing)
- **~60 FPS** on typical hardware
- **Real-time simulation** (sim time ≈ wall time)

**Optimizations:**
- Spatial hashing: O(N) neighbor search
- Larger smoothing length: fewer neighbors per particle
- Release mode compilation: 10-20× speedup
- Single timestep per frame: no subcycling

---

## Project Structure

```
sph-water-sim/
├── src/
│   ├── main.rs              # Rendering loop, window, input
│   ├── lib.rs               # Module declarations
│   └── sph/
│       ├── mod.rs           # SPH module
│       ├── particles.rs     # Particle struct, spatial hash
│       ├── kernels.rs       # W(r,h), ∇W, ∇²W
│       ├── forces.rs        # Pressure, viscosity, gravity
│       └── integration.rs   # Time stepping, CFL
├── Cargo.toml
├── run.sh
└── README.md
```

---

## Tuning Guide

### Adjust Realism vs. Performance

**More particles (slower, more detail):**
```rust
system.smoothing_length = 0.005;  // 5mm → ~2000 particles
```

**Fewer particles (faster):**
```rust
system.smoothing_length = 0.010;  // 10mm → ~400 particles
```

### Adjust Fluid Behavior

**Water-like (splashy):**
```rust
system.viscosity = 0.001;
system.stiffness = 5000.0;
```

**Honey-like (thick):**
```rust
system.viscosity = 0.1;      // 100× more viscous
system.stiffness = 10000.0;
```

**Jello-like (stiff):**
```rust
system.viscosity = 1.0;
system.stiffness = 20000.0;
```

### Adjust Compressibility

**More stiff (less compressible):**
```rust
system.stiffness = 50000.0;  // Requires smaller timestep
```

**Softer (more compressible):**
```rust
system.stiffness = 2000.0;   // More stable, faster
```

---

## References

### Foundational Papers

1. **Gingold & Monaghan (1977)** - "Smoothed particle hydrodynamics: theory and application to non-spherical stars"
2. **Monaghan (1992)** - "Smoothed Particle Hydrodynamics" (review paper)
3. **Monaghan (1994)** - "Simulating Free Surface Flows with SPH"
4. **Morris, Fox & Zhu (1997)** - "Modeling Low Reynolds Number Incompressible Flows Using SPH"
5. **Becker & Teschner (2007)** - "Weakly compressible SPH for free surface flows"
6. **Müller et al. (2003)** - "Particle-Based Fluid Simulation for Interactive Applications"

### Textbooks

- **Liu & Liu (2003):** *Smoothed Particle Hydrodynamics: A Meshfree Particle Method*
- **Violeau (2012):** *Fluid Mechanics and the SPH Method*

### Online Resources

- Matthias Müller's SIGGRAPH course: "Fluid Simulation (2007)"
- SPlisHSPlasH: Open-source SPH library (GitHub)
- Dan Koschier's tutorial: "Smoothed Particle Hydrodynamics Techniques"

---

## Why SPH?

### Advantages

- ✅ **Perfect mass conservation** - particles can't disappear
- ✅ **Lagrangian** - follows the flow naturally (no advection errors)
- ✅ **Meshfree** - no grid, free surfaces handled automatically
- ✅ **Intuitive** - "liquid = particles" matches physical intuition
- ✅ **Adaptive resolution** - naturally more particles where needed

### Disadvantages

- ❌ **Particle disorder** - requires stabilization
- ❌ **Tensile instability** - negative pressure causes issues
- ❌ **Neighbor search** - needs spatial data structure
- ❌ **Boundary conditions** - more complex than grid methods

### When to Use SPH

**Good for:**
- Free surface flows (splashes, waves, droplets)
- Large deformations
- Moving interfaces
- Real-time applications

**Not ideal for:**
- Very high Reynolds number (turbulence)
- Solid-fluid coupling (better methods exist)
- Extremely large simulations (grid methods scale better)

---

## License

MIT

---

## Quick Reference

### The Three Forces

1. **Pressure:** `F = -∇p` → `p = k((ρ/ρ₀)^7 - 1)` (Tait)
2. **Viscosity:** `F = μ∇²v` → smooths velocity differences
3. **Gravity:** `F = mg` → pulls downward

### SPH in One Formula

```
A(xᵢ) = Σⱼ (mⱼ/ρⱼ) A(xⱼ) W(xᵢ - xⱼ, h)
```

Everything else follows from this!

### Typical Timestep Loop

```rust
1. compute_densities()           // Σ m_j W(r, h)
2. compute_pressures()           // p = k((ρ/ρ₀)^γ - 1)
3. compute_pressure_forces()     // -Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
4. compute_viscosity_forces()    // ν Σ m_j/ρ_j (v_j - v_i) ∇²W
5. compute_gravity_forces()      // F = mg
6. integrate()                   // Leap-frog: v += a*dt, x += v*dt
7. enforce_boundaries()          // Keep particles in domain
```

---

**Ready to simulate? Run `./run.sh` and watch the water splash!** 💧🌊
