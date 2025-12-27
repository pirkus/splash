mod field;
mod grid;
mod hud;
mod level_set;
mod mac;
mod mac_sim;
mod mac_sim_mg;
mod particles;
mod render;
mod vec2;
mod vec_field;

pub use field::Field2;
pub use grid::Grid2;
pub use hud::{overlay_text, GLYPH_HEIGHT, GLYPH_SPACING, GLYPH_WIDTH, LINE_SPACING};
pub use level_set::{
    advect_level_set_surface_in_place, flags_from_phi, level_set_step, level_set_step_in_place,
    correct_phi_volume, phi_to_density, reinitialize_phi, volume_from_phi, LevelSetParams,
    LevelSetState, LevelSetWorkspace,
};
pub use mac_sim::{
    divergence, extrapolate_velocity, flags_from_density, project_with_flags, sharpen_density,
    step, step_in_place, AdvectionScheme, MacSimParams, MacSimState, MacSimWorkspace,
};
pub use mac_sim_mg::{step_in_place_mg, step_mg, MacSimMgWorkspace, MultigridParams};
pub use particles::{ParticleBounds, ParticlePhase, ParticleSystem};
pub use vec2::Vec2;
pub use mac::{
    apply_domain_boundaries, apply_solid_boundaries, BoundaryCondition, BoundaryConfig, CellFlags,
    CellType, MacGrid2, MacVelocity2, StaggeredField2, StaggeredGrid2,
};
pub use render::VulkanApp;
