# Level-Set Simulation Step (Detailed)

This document describes the **Level-set mode** step-by-step, matching the logic in
`src/level_set.rs` (`level_set_step`). It focuses on the *why* and the *math* behind
each stage so you can reason about stability and visual behavior.
For the density-based pipeline, see `DENSITY_SIM_STEP.md`.

## Data and Grid Setup

- **Velocity** uses a **MAC grid** (staggered storage):
  - `u` lives on vertical faces (x-velocity).
  - `v` lives on horizontal faces (y-velocity).
  - This makes enforcing incompressibility (`∇·u = 0`) easier and more stable.
- **Surface** is a **level-set** scalar `phi` at cell centers:
  - `phi < 0` means **liquid**, `phi > 0` means **air**.
  - `|phi|` approximates distance to the surface when reinitialized.

### Why level-set?
Level-set gives a crisp interface that can naturally split/merge without explicit
surface tracking. The downside is volume loss from advection and reinit, so we
add a volume correction step.

## One Simulation Step (Level-Set Mode)

### 1) Build fluid/air flags from `phi`
- Purpose: mark which cells are liquid so the pressure solve and masking are applied
  only where fluid exists.
- Rule: `phi <= 0` => fluid, else air. Solids override everything.

### 2) Apply boundaries + fluid mask
- Apply domain boundary conditions (no-slip walls) on the staggered velocity.
- Apply solid boundaries (zero velocity on solid-adjacent faces).
- Apply a **fluid mask** to zero velocity faces where both adjacent cells are air.
- Purpose: keep the velocity valid at the interface and avoid pressure artifacts
  in empty regions.

### 3) CFL clamp for a stable timestep
- CFL condition: `dt <= cfl * dx / max(|u|)`.
- Purpose: avoid characteristic backtracing that jumps too far, which causes
  instability and smearing.

### 4) Add forces (gravity + surface tension)
- **Gravity**: constant acceleration added to velocity.
- **Surface tension** (optional):
  - Compute surface normals `n = ∇phi / |∇phi|`.
  - Curvature `kappa = -∇·n`.
  - Apply force `f = σ * kappa * δ(phi) * n`.
  - `δ(phi)` is a smoothed delta in a thin band around the surface.
- Purpose: gravity drives motion; surface tension smooths small surface ripples.

### 5) Advect velocity (Semi-Lagrangian or BFECC)
- Trace from each face center **backwards** along velocity to sample the old velocity.
- BFECC adds a forward/backward correction to reduce numerical diffusion.
- Purpose: transport momentum with the flow.

### 6) Diffuse velocity (viscosity)
- Apply a Laplacian to velocity to model viscous diffusion.
- Purpose: damp small-scale noise and model viscous fluids.

### 7) Pressure projection (incompressibility)
- Compute divergence `∇·u` on cells.
- Solve Poisson for pressure: `∇²p = (1/dt) * ∇·u` (solver uses PCG).
- Subtract pressure gradient: `u' = u - ∇p`.
- Purpose: enforce `∇·u = 0` so the fluid is incompressible.

### 8) Extrapolate velocity into air (optional)
- Iteratively extend nearby fluid velocities into adjacent air faces.
- Purpose: interface advection uses velocities just outside the liquid; extrapolation
  prevents surface tearing or stagnation.

### 9) Advect `phi`
- Backtrace cell centers through the velocity field and sample `phi`.
- Purpose: move the interface with the flow.

### 10) Reinitialize `phi`
- Solve a pseudo-time PDE to push `|∇phi|` toward `1`:
  - `phi_t + s(phi0)(|∇phi| - 1) = 0`
- `s(phi0)` is a smooth sign of the original `phi` to keep the zero level set fixed.
- Purpose: restore the signed-distance property so curvature and normals stay stable.

### 11) Volume correction (optional, enabled in viewer)
- Measure current fluid volume as the integral of a smooth Heaviside of `phi`.
- Find an offset `c` so that `phi + c` matches the **target volume** via bisection.
- Purpose: prevent steady water-level loss or gain across many steps.

### 12) Enforce zero-normal gradient at boundaries
- Copy interior `phi` to boundary cells (Neumann condition).
- Purpose: prevents artificial slopes at the left/right walls.

---

## Step Flow (Mermaid)

```mermaid
flowchart TD
    A[Start: phi, velocity] --> B[Build flags from phi]
    B --> C[Apply boundaries + fluid mask]
    C --> D[CFL clamp dt]
    D --> E[Add forces]
    E --> F[Advect velocity]
    F --> G[Diffuse velocity]
    G --> H[Project (pressure solve)]
    H --> I[Extrapolate velocity into air]
    I --> J[Advect phi]
    J --> K[Reinitialize phi]
    K --> L[Volume correction]
    L --> M[Neumann boundary for phi]
    M --> N[Output next state]
```

## Mental Model: Why This Works

- **MAC grid** makes incompressibility robust because pressure correction happens
  on cell centers while velocity is stored at faces.
- **Level-set surface** gives a sharp interface; reinit + volume correction
  keeps that interface well-behaved over time.
- **Projection** is the key to “liquid-like” motion: it removes non-physical
  divergence that causes sinking or expansion.

If you want a deeper dive into any sub-step (e.g., PCG details, curvature estimation,
or CFL constraints), call it out and I can expand that section.
