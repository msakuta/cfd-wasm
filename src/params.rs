use super::state::BoundaryCondition;

#[derive(Copy, Clone)]
pub(crate) struct Params {
    pub delta_time: f64,
    pub skip_frames: u32,
    pub mouse_pos: [i32; 2],
    pub visc: f64,
    pub diff: f64,
    pub density: f64,
    pub decay: f64,
    pub mouse_flow_speed: f64,
    pub diffuse_iter: usize,
    pub project_iter: usize,
    pub temperature: bool,
    pub half_heat_source: bool,
    pub heat_exchange_rate: f64,
    pub heat_buoyancy: f64,
    pub mouse_flow: bool,
    pub gamma: f32,
    pub show_velocity: bool,
    pub show_velocity_field: bool,
    pub obstacle: bool,
    pub dye_from_obstacle: bool,
    pub particles: bool,
    pub particle_trails: usize,
    pub redistribute_particles: bool,
    pub boundary_y: BoundaryCondition,
    pub boundary_x: BoundaryCondition,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            delta_time: 1.,
            skip_frames: 1,
            mouse_pos: [0, 0],
            visc: 0.01,
            diff: 0., // Diffusion seems ok with 0, since viscousity and Gauss-Seidel blends up anyway.
            density: 0.5,
            decay: 0.01,
            mouse_flow_speed: 0.02,
            diffuse_iter: 4,
            project_iter: 20,
            temperature: false,
            half_heat_source: false,
            heat_exchange_rate: 0.2,
            heat_buoyancy: 0.05,
            mouse_flow: true,
            gamma: 1.0,
            show_velocity: true,
            show_velocity_field: false,
            obstacle: false,
            dye_from_obstacle: true,
            particles: true,
            particle_trails: 0,
            redistribute_particles: true,
            boundary_x: BoundaryCondition::Wrap,
            boundary_y: BoundaryCondition::Wrap,
        }
    }
}
