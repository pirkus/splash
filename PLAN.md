# Navier-Stokes Simulation Plan
Status legend: [x] done, [~] in progress, [ ] not started

## Phase 1: Minimal 2D Incompressible Solver (Stokes / Low-Re)
1. [x] Define the numerical model (2D incompressible, constant density/viscosity, unit square domain).
2. [x] Pick a stable baseline discretization (MAC grid + finite differences). MAC grid integrated into the main solver.
3. [x] Implement core data structures (grid, fields, boundary flags). MAC grid + flags wired into the solver.
4. [x] Implement diffusion step (implicit or explicit with CFL limit). Current: explicit diffusion.
5. [x] Implement pressure projection (Poisson solve + velocity correction). Current: Jacobi.
6. [x] Add no-slip and inflow/outflow boundary conditions.
7. [x] Validate with simple cases (decay of a vortex, lid-driven cavity at low Re).

## Phase 1.5: Stabilization and Validation
1. [x] Add CFL-based `dt` clamp in the MAC step.
2. [x] Add boundary-aware pressure solve (solid-aware Poisson).
3. [x] Add regression for mass conservation (density sum stays stable with no sources).
4. [x] Add a short solver validation doc with expected qualitative behavior.

## Phase 1.9: Performance and Instrumentation
1. [ ] Parallelize core field ops with `rayon` while preserving functional style.
2. [ ] Add reusable solver workspace to reduce allocations in advection/diffusion.
3. [ ] Use adaptive pressure iterations based on residual instead of fixed `pressure_iters`.

## Phase 2: Full 2D Navier-Stokes
1. [x] Add advection (semi-Lagrangian or RK2/3 with backtracing). Current: semi-Lagrangian.
2. [x] Add external forces (body forces, buoyancy).
3. [x] Improve pressure solve (preconditioned conjugate gradient).
4. [x] Add basic visualization (velocity magnitude, vorticity).
5. [~] Add a Vulkan-based windowed 2D renderer (live field display with pan/zoom, plus a simple interaction: mouse drag injects velocity and click adds a density puff). Current: window + swapchain + textured fullscreen quad with live density.
   - [x] Upload scalar fields as textures and render via fullscreen quad.
   - [ ] Add a debug overlay (FPS, dt, CFL, solver iterations).
   - [ ] Add input mapping and a configurable brush (radius, strength, density amount).
   - [ ] Add deterministic interaction capture/replay for debugging.
   - [ ] Add render tests (offscreen render + image checksum in CI).
6. [ ] Add timestep control (CFL-based adaptive dt).

## Phase 3: Better Accuracy and Stability
1. [ ] Switch to higher-order advection (BFECC or MacCormack).
2. [ ] Add vorticity confinement (optional for visual detail).
3. [ ] Implement multigrid pressure solve.
4. [ ] Add boundary-aware advection (solid obstacles).
5. [ ] Regression tests for conservation and stability.

## Phase 4: Advanced Features
1. [ ] Add 3D support (staggered grid, memory management).
2. [ ] Add a Vulkan-based 3D renderer (slice views + volume preview).
   - [ ] Upload 3D fields (3D texture or texture array) and render slices.
   - [ ] Add a debug overlay (FPS, dt, CFL, solver iterations).
   - [ ] Add input mapping and a configurable brush (radius, strength, density amount).
   - [ ] Add deterministic interaction capture/replay for debugging.
   - [ ] Add render tests (offscreen render + image checksum in CI).
3. [ ] Add variable density/viscosity (Boussinesq / two-phase).
4. [ ] Add turbulence models (LES / Smagorinsky).
5. [ ] Add particle tracing (marker particles, streaklines).
6. [ ] GPU acceleration (compute shaders or CUDA/OpenCL).

## Phase 5: Production-Ready Simulation
1. [ ] Scene configuration (YAML/JSON inputs).
2. [ ] Data output formats (VTI, NPZ, or OpenVDB).
3. [ ] Performance profiling and optimization.
4. [ ] Reproducibility (seed control, deterministic stepping).
5. [ ] Documentation + examples.
