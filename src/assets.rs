use super::{
    gl_util::{
        compile_shader, gen_flow_texture, gen_particle_texture, link_program, vertex_buffer_data,
    },
    shape::Shape,
    state::particles::{PARTICLE_COUNT, PARTICLE_MAX_TRAIL_LEN},
    wasm_util::{console_log, AngleInstancedArrays},
};
use crate::shader_bundle::ShaderBundle;
use slice_of_array::SliceFlatExt;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGlBuffer, WebGlRenderingContext as GL, WebGlTexture};

pub(crate) struct Assets {
    pub instanced_arrays_ext: Option<AngleInstancedArrays>,
    pub flow_tex: Option<WebGlTexture>,
    pub particle_tex: Option<WebGlTexture>,
    pub rect_shader: Option<ShaderBundle>,
    pub particle_shader: Option<ShaderBundle>,
    pub particle_instancing_shader: Option<ShaderBundle>,
    pub arrow_shader: Option<ShaderBundle>,
    pub trail_shader: Option<ShaderBundle>,
    pub contour_shader: Option<ShaderBundle>,

    pub trail_buffer: Option<WebGlBuffer>,
    pub rect_buffer: Option<WebGlBuffer>,
    pub arrow_buffer: Option<WebGlBuffer>,
    pub particle_buffer: Option<WebGlBuffer>,
    pub contour_buffer: Option<WebGlBuffer>,
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
            contour_shader: None,
            trail_buffer: None,
            rect_buffer: None,
            arrow_buffer: None,
            particle_buffer: None,
            contour_buffer: None,
        }
    }
}

impl Assets {
    pub(crate) fn start_gl(&mut self, gl: &GL, shape: &Shape) -> Result<(), JsValue> {
        self.flow_tex = Some(gen_flow_texture(gl, &shape)?);

        self.particle_tex = Some(gen_particle_texture(gl)?);

        self.instanced_arrays_ext = gl
            .get_extension("ANGLE_instanced_arrays")
            .unwrap_or(None)
            .map(|v| v.unchecked_into::<AngleInstancedArrays>());
        console_log!(
            "WebGL Instanced arrays is {}",
            if self.instanced_arrays_ext.is_some() {
                "available"
            } else {
                "not available"
            }
        );

        let vert_shader = compile_shader(
            &gl,
            GL::VERTEX_SHADER,
            r#"
            attribute vec2 vertexData;
            uniform mat4 transform;
            uniform mat3 texTransform;
            varying vec2 texCoords;
            void main() {
                gl_Position = transform * vec4(vertexData.xy, 0.0, 1.0);

                texCoords = (texTransform * vec3((vertexData.xy + 1.) * 0.5, 1.)).xy;
            }
        "#,
        )?;
        let frag_shader = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            varying vec2 texCoords;

            uniform sampler2D texture;
            uniform float alpha;
            uniform float gamma;

            void main() {
                vec4 texColor = texture2D( texture, vec2(texCoords.x, texCoords.y) );
                gl_FragColor = vec4(pow(texColor.rgb, vec3(gamma)), texColor.a * alpha);
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader)?;
        gl.use_program(Some(&program));

        let shader = ShaderBundle::new(&gl, program);

        gl.active_texture(GL::TEXTURE0);

        gl.uniform1i(shader.texture_loc.as_ref(), 0);
        gl.uniform1f(shader.alpha_loc.as_ref(), 1.);

        gl.enable(GL::BLEND);
        gl.blend_equation(GL::FUNC_ADD);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);

        self.rect_shader = Some(shader);

        let frag_shader_add = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            varying vec2 texCoords;

            uniform sampler2D texture;
            uniform float alpha;

            void main() {
                vec4 texColor = texture2D( texture, vec2(texCoords.x, texCoords.y) );
                gl_FragColor = vec4(texColor.rgb, texColor.r * alpha);
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader_add)?;
        let shader = ShaderBundle::new(&gl, program);
        gl.uniform1f(shader.gamma_loc.as_ref(), 0.5);
        self.particle_shader = Some(shader);

        let vert_shader_instancing = compile_shader(
            &gl,
            GL::VERTEX_SHADER,
            r#"
            attribute vec2 vertexData;
            attribute vec2 position;
            attribute float alpha;
            uniform mat4 transform;
            uniform mat3 texTransform;
            varying vec2 texCoords;
            varying float alphaVar;

            void main() {
                mat4 centerize = mat4(
                    4, 0, 0, 0,
                    0, -4, 0, 0,
                    0, 0, 4, 0,
                    -1, 1, -1, 1);
                gl_Position = centerize * (transform * vec4(vertexData.xy, 0.0, 1.0) + vec4(position.xy, 0.0, 1.0));
                texCoords = (texTransform * vec3((vertexData.xy + 1.) * 0.5, 1.)).xy;
                alphaVar = alpha;
            }
        "#,
        )?;
        let frag_shader_instancing = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            varying vec2 texCoords;
            varying float alphaVar;

            uniform sampler2D texture;

            void main() {
                vec4 texColor = texture2D( texture, vec2(texCoords.x, texCoords.y) );
                gl_FragColor = vec4(texColor.rgb, texColor.r * alphaVar);
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader_instancing, &frag_shader_instancing)?;
        let shader = ShaderBundle::new(&gl, program);
        self.particle_instancing_shader = Some(shader);

        let frag_shader_flat = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            uniform float alpha;

            void main() {
                gl_FragColor = vec4(0.5, 0.5, 1., alpha);
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader_flat)?;
        let shader = ShaderBundle::new(&gl, program);
        gl.uniform1f(shader.alpha_loc.as_ref(), 0.5);
        self.arrow_shader = Some(shader);

        let frag_shader_color = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            uniform vec4 color;

            void main() {
                gl_FragColor = color;
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader_color)?;
        let shader = ShaderBundle::new(&gl, program);
        gl.uniform4f(shader.color_loc.as_ref(), 1.0, 1.0, 1.0, 0.5);
        self.contour_shader = Some(shader);

        let vert_shader = compile_shader(
            &gl,
            GL::VERTEX_SHADER,
            r#"
            attribute vec4 vertexData;
            uniform mat4 transform;
            uniform mat3 texTransform;
            varying vec2 texCoords;
            void main() {
                gl_Position = vec4(vertexData.xy, 0.0, 1.0);

                texCoords = (texTransform * vec3(vertexData.zw, 1.)).xy;
            }
        "#,
        )?;
        let frag_shader = compile_shader(
            &gl,
            GL::FRAGMENT_SHADER,
            r#"
            precision mediump float;

            varying vec2 texCoords;

            uniform sampler2D texture;

            void main() {
                vec4 texColor = texture2D( texture, vec2(texCoords.x, texCoords.y) );
                gl_FragColor = texColor;
            }
        "#,
        )?;
        let program = link_program(&gl, &vert_shader, &frag_shader)?;
        gl.use_program(Some(&program));
        self.trail_shader = Some(ShaderBundle::new(&gl, program));

        gl.active_texture(GL::TEXTURE0);
        gl.uniform1i(
            self.trail_shader
                .as_ref()
                .and_then(|s| s.texture_loc.as_ref()),
            0,
        );

        let create_buffer = |data| -> Result<_, JsValue> {
            let buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);
            gl.bind_buffer(GL::ARRAY_BUFFER, buffer.as_ref());
            vertex_buffer_data(&gl, data);
            Ok(buffer)
        };

        self.trail_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);

        self.rect_buffer = create_buffer(&[1., 1., -1., 1., -1., -1., 1., -1.])?;

        self.arrow_buffer = create_buffer(&[1., 0., -1., -0.2, -1., 0.2])?;

        self.particle_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);
        gl.bind_buffer(GL::ARRAY_BUFFER, self.particle_buffer.as_ref());
        gl.buffer_data_with_i32(
            GL::ARRAY_BUFFER,
            (PARTICLE_COUNT * 3 * std::mem::size_of::<f32>() * (1 + PARTICLE_MAX_TRAIL_LEN)) as i32,
            GL::DYNAMIC_DRAW,
        );

        self.contour_buffer = create_buffer(
            &[
                [1., 1., -1., 1., -1., -1., 1., -1.],
                [1., 0.5, -1., 0.5, -1., -0.5, 1., -0.5],
                [0.5, 1., -0.5, 1.0, -0.5, -1., 0.5, -1.],
            ]
            .flat(),
        )?;

        gl.clear_color(0.0, 0.2, 0.5, 1.0);

        Ok(())
    }
}
