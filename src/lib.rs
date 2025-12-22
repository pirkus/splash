mod field;
mod grid;
mod hud;
mod level_set;
mod mac;
mod mac_sim;
mod render;
mod vec2;
mod vec_field;

pub use field::Field2;
pub use grid::Grid2;
pub use hud::{overlay_text, GLYPH_HEIGHT, GLYPH_SPACING, GLYPH_WIDTH, LINE_SPACING};
pub use level_set::{
    flags_from_phi, level_set_step, phi_to_density, reinitialize_phi, volume_from_phi,
    LevelSetParams, LevelSetState,
};
pub use mac_sim::{
    flags_from_density, sharpen_density, step, AdvectionScheme, MacSimParams, MacSimState,
};
pub use vec2::Vec2;
pub use mac::{
    apply_domain_boundaries, apply_solid_boundaries, BoundaryCondition, BoundaryConfig, CellFlags,
    CellType, MacGrid2, MacVelocity2, StaggeredField2, StaggeredGrid2,
};
pub use render::VulkanApp;
