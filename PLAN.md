# Navier-Stokes Simulation Plan
Status legend: [x] done, [~] in progress, [ ] not started

## Phase 1: Minimal 2D Incompressible Solver (Stokes / Low-Re)
1. [x] Define the numerical model (2D incompressible, constant density/viscosity, grid in cell units).
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

## Phase 1.9: Performance and Instrumentation
1. [x] Parallelize core field ops with `rayon` while preserving functional style.
2. [x] Add reusable solver workspace to reduce allocations in advection/diffusion.
3. [x] Use adaptive pressure iterations based on residual instead of fixed `pressure_iters`.

## Phase 2: Full 2D Navier-Stokes
1. [x] Add advection (semi-Lagrangian or RK2/3 with backtracing). Current: semi-Lagrangian.
2. [x] Add external forces (body forces, buoyancy).
3. [x] Improve pressure solve (preconditioned conjugate gradient).
4. [ ] Add basic visualization (velocity magnitude, vorticity).
5. [~] Add a Vulkan-based windowed 2D renderer (live field display with pan/zoom, plus a simple interaction: mouse drag injects velocity and click adds a density puff). Current: window + swapchain + textured fullscreen quad with live density.
   - [x] Upload scalar fields as textures and render via fullscreen quad.
   - [x] Add a debug overlay (FPS, dt, CFL, solver iterations).
   - [ ] Add input mapping and a configurable brush (radius, strength, density amount).
   - [ ] Add deterministic interaction capture/replay for debugging.
   - [ ] Add render tests (offscreen render + image checksum in CI).
6. [x] Add timestep control (CFL-based adaptive dt).

## Phase 3: Better Accuracy and Stability
1. [x] Switch to higher-order advection (BFECC or MacCormack).
2. [x] Add vorticity confinement (optional for visual detail).
3. [x] Implement multigrid pressure solve.
4. [x] Add boundary-aware advection (solid obstacles).
5. [x] Regression tests for conservation and stability.

## Phase 4: FLIP/APIC CPU Water Solver (2D)
1. [x] Add particle storage (positions, velocities, phase tags) and seeded initial pool.
2. [x] Implement PIC/FLIP/APIC transfers (P2G, G2P) on MAC grid.
3. [x] Integrate particles (RK2/RK3 advection) with boundary handling.
4. [x] Pressure projection on the grid (reuse existing solver, tuned for liquid region).
5. [x] Surface reconstruction for rendering (particle level set via SPH density + reinit).
6. [ ] Add drop injection and interaction controls (click to add droplets).
7. [x] Add validation cases (dam break, drop impact, volume conservation).

## Phase 5: FLIP/APIC GPU Compute
1. [~] GPU particle storage + integration (Vulkan compute Mode 5, Euler step + CPU readback).
2. [~] GPU grid transfer kernels (P2G with int atomics + CPU readback; G2P pending).
3. [ ] GPU pressure solve (Jacobi/PCG/multigrid on device).
4. [ ] GPU surface reconstruction (density/level set and smoothing).
5. [ ] GPU visualization path (render directly from GPU buffers).
6. [ ] GPU APIC affine transfer (affine storage + APIC P2G/G2P on device).
7. [ ] GPU PBF constraints (position-based fluid solve on device).
8. [ ] GPU level-set reinit + volume correction (match Mode 4 surface).
9. [ ] GPU validation cases (drop/dam/volume metrics, parity with CPU).
10. [ ] GPU debug readback toggles (matrix dumps, metrics snapshots).
11. [ ] GPU/Vulkan interop path (zero-copy buffers for renderer).

## Phase 6: Advanced Features
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

## Phase 7: Production-Ready Simulation
1. [ ] Scene configuration (YAML/JSON inputs).
2. [ ] Data output formats (VTI, NPZ, or OpenVDB).
3. [ ] Performance profiling and optimization.
4. [ ] Reproducibility (seed control, deterministic stepping).
5. [ ] Documentation + examples.
