use super::AngleInstancedArrays;
use crate::shader_bundle::ShaderBundle;
use web_sys::{WebGlBuffer, WebGlTexture};

pub(crate) struct Assets {
    pub instanced_arrays_ext: Option<AngleInstancedArrays>,
    pub flow_tex: Option<WebGlTexture>,
    pub particle_tex: Option<WebGlTexture>,
    pub rect_shader: Option<ShaderBundle>,
    pub particle_shader: Option<ShaderBundle>,
    pub particle_instancing_shader: Option<ShaderBundle>,
    pub arrow_shader: Option<ShaderBundle>,
    pub trail_shader: Option<ShaderBundle>,
    pub trail_buffer: Option<WebGlBuffer>,
    pub rect_buffer: Option<WebGlBuffer>,
    pub arrow_buffer: Option<WebGlBuffer>,
    pub particle_buffer: Option<WebGlBuffer>,
}

impl Default for Assets {
    fn default() -> Self {
        Self {
            instanced_arrays_ext: None,
            flow_tex: None,
            particle_tex: None,
            rect_shader: None,
            particle_shader: None,
            particle_instancing_shader: None,
            arrow_shader: None,
            trail_shader: None,
            trail_buffer: None,
            rect_buffer: None,
            arrow_buffer: None,
            particle_buffer: None,
        }
    }
}
