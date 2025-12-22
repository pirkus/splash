# Navier-Stokes Sim (Rust)

A compact 2D incompressible Navier-Stokes solver with a Vulkan-based windowed renderer.
It supports two free-surface models:
- Density-based MAC solver (fast to iterate, more diffusive).
- Level-set free surface (signed-distance surface tracking with reinit and volume correction).

The viewer runs a small "aquarium" scenario: a flat water line with a droplet falling in.

Docs:
- `LEVEL_SET_SIM_STEP.md` : detailed level-set step walkthrough.
- `DENSITY_SIM_STEP.md` : detailed density step walkthrough.

## Run

```bash
cargo run --release
```

Controls:
- `1` : density simulation
- `2` : level-set simulation
- `Space` : toggle simulation mode

The on-screen HUD shows:
- `1` or `2` to indicate the current sim
- `DT` (effective timestep after CFL clamp)
- `CFL` (target CFL)
- `IT` (pressure iterations)

## Simulation Notes

Core solver (shared):
- 2D MAC grid with staggered velocities and pressure projection.
- Advection via Semi-Lagrangian or BFECC (viewer uses BFECC).
- External forces: gravity (body force) and optional surface tension.

Surface models (two options):
- Density-based surface:
  - A scalar density field is advected/diffused and thresholded to define liquid vs air.
  - Simple and fast, but more diffusive and “smoke-like”.
- Level-set surface:
  - `phi` is a signed-distance field (negative = liquid, positive = air).
  - Reinitialization restores distance-like behavior.
  - Volume correction shifts `phi` to preserve liquid volume.
  - Boundary handling enforces a zero-normal gradient to avoid sloped walls.

Default mode: the viewer starts in Level-set mode (press `1` or `2` to switch).

## Per-Step Overview

Density-based step (MAC + density):
1. Apply domain/solid boundaries and optional fluid mask.
2. Clamp dt by CFL.
3. Add body forces (gravity, optional buoyancy/surface tension).
4. Advect velocity (Semi-Lagrangian or BFECC).
5. Diffuse velocity (viscosity).
6. Project (solve pressure, enforce divergence-free).
7. Advect density.
8. Diffuse density (optional).
9. Threshold density to update fluid/air flags.

Level-set step (MAC + phi):
1. Build fluid/air flags from `phi`.
2. Apply domain/solid boundaries and fluid mask.
3. Clamp dt by CFL.
4. Add body forces and surface tension (from `phi` curvature).
5. Advect velocity (Semi-Lagrangian or BFECC).
6. Diffuse velocity (viscosity).
7. Project (solve pressure, enforce divergence-free).
8. Extrapolate velocity into air (optional, improves surface motion).
9. Advect `phi`.
10. Reinitialize `phi` toward a signed-distance field.
11. Shift `phi` to preserve volume (optional).
12. Enforce zero-normal gradient at boundaries.

## Navier-Stokes in This Project

The incompressible Navier-Stokes equations model fluid motion:

- Momentum: `du/dt + u · ∇u = -∇p + ν ∇²u + f`
- Incompressibility: `∇ · u = 0`

Where `u` is velocity, `p` is pressure, `ν` is viscosity, and `f` is external force (gravity).

In this codebase:
- Advection handles the `u · ∇u` term (Semi-Lagrangian or BFECC).
- Viscosity applies the `ν ∇²u` diffusion term.
- External forces add `f` (gravity and optional surface tension).
- A pressure projection enforces `∇ · u = 0` by solving a Poisson equation.
- Free surfaces are modeled either by a density field or by a level-set `phi`.

## Project Layout

- `src/main.rs` : Vulkan viewer and simulation loop.
- `src/mac_sim.rs` : MAC solver (density-based).
- `src/level_set.rs` : Level-set free-surface solver.
- `src/hud.rs` : tiny HUD glyph renderer.
- `src/render/` : Vulkan setup and renderer.
- `PLAN.md` : staged project roadmap.
- `LEVEL_SET_SIM_STEP.md` : level-set step-by-step.
- `DENSITY_SIM_STEP.md` : density step-by-step.

## Tests

```bash
cargo test --lib
```
