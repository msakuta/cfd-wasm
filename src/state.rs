//! Definition of simulation state and its update logic

mod cfd;
pub(crate) mod particles;
mod renderer;

use cgmath::{Matrix3, Matrix4, Rad, Vector3};
use wasm_bindgen::prelude::*;
use web_sys::WebGlRenderingContext as GL;

use crate::{
    assets::Assets,
    bit_util::ceil_pow2,
    gl_util::enable_buffer,
    params::Params,
    shape::{Idx, Shape},
    xor128::Xor128,
};

use self::{
    cfd::obstacle_position,
    particles::{new_particles, Particle},
};

pub use self::renderer::{cfd_canvas, cfd_webgl};

pub(crate) use self::cfd::{add_density, add_velo, BoundaryCondition};

pub(crate) struct State {
    density: Vec<f64>,
    density2: Vec<f64>,
    temperature: Option<Vec<f64>>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    vx0: Vec<f64>,
    vy0: Vec<f64>,
    work: Vec<f64>,
    work2: Vec<f64>,
    shape: Shape,
    params: Params,
    particles: Vec<Particle>,
    /// In-memory buffer to avoid reallocating huge memory every tick
    particle_buf: Vec<f32>,
    /// Actually used length of particle_buf, which can be less than buffer's length
    particle_buf_active_len: usize,
    xor128: Xor128,
    assets: Assets,
}

impl State {
    fn new(width: usize, height: usize) -> Self {
        let mut xor128 = Xor128::new(123);

        let params = Params::default();

        let shape = (width as isize, height as isize);

        let particles = new_particles(&mut xor128, shape);

        Self {
            density: vec![0f64; width * height],
            density2: vec![0f64; width * height],
            temperature: None,
            vx: vec![0f64; width * height],
            vy: vec![0f64; width * height],
            vx0: vec![0f64; width * height],
            vy0: vec![0f64; width * height],
            work: vec![0f64; width * height],
            work2: vec![0f64; width * height],
            shape,
            params,
            particles,
            particle_buf: vec![],
            particle_buf_active_len: 0,
            xor128,
            assets: Assets::default(),
        }
    }

    /// Destructively calculate speed (length of velocity vector) field using State's working memory
    fn calc_velo(&mut self) -> &Vec<f64> {
        self.work
            .iter_mut()
            .zip(self.vx.iter().zip(self.vy.iter()))
            .for_each(|(dest, (x, y))| *dest = (x * x + y * y).sqrt());
        &self.work
    }

    fn render_fluid(&mut self, data: &mut Vec<u8>) {
        const U_MAX: f64 = 1.0;
        const V_MAX: f64 = 1.0;
        const W_MAX: f64 = 0.01;

        self.calc_velo();
        let (u, v, w) = (
            if self.params.temperature {
                self.temperature.as_ref().unwrap_or(&self.density)
            } else {
                &self.density
            },
            &self.density2,
            &self.work,
        );
        let shape = &self.shape;
        for y in 0..self.shape.1 {
            for x in 0..self.shape.0 {
                data[shape.idx(x, y) * 4] = ((u[shape.idx(x, y)]) / U_MAX * 127.) as u8;
                data[shape.idx(x, y) * 4 + 1] = ((v[shape.idx(x, y)]) / V_MAX * 127.) as u8;
                data[shape.idx(x, y) * 4 + 2] = if self.params.show_velocity {
                    ((w[shape.idx(x, y)]) / W_MAX * 127.) as u8
                } else {
                    0
                };
                data[shape.idx(x, y) * 4 + 3] = 255;
            }
        }

        let (center, radius) = obstacle_position(shape);

        if self.params.obstacle {
            for j in center.1 - radius..center.1 + radius {
                for i in center.0 - radius..center.0 + radius {
                    let dist2 = (j - center.1) * (j - center.1) + (i - center.0) * (i - center.0);
                    if dist2 < radius * radius {
                        data[shape.idx(i, j) * 4] = 127;
                        data[shape.idx(i, j) * 4 + 1] = 127;
                        data[shape.idx(i, j) * 4 + 2] = 0;
                        data[shape.idx(i, j) * 4 + 3] = 255;
                    }
                }
            }
        }
    }

    fn render_velocity_field(&self, ctx: &web_sys::CanvasRenderingContext2d) {
        if self.params.show_velocity_field {
            const CELL_SIZE: isize = 10;
            const CELL_SIZE_F: f64 = CELL_SIZE as f64;
            const VELOCITY_SCALE: f64 = 1e3;
            const MAX_VELOCITY: f64 = 1e-2;
            let x_cells = self.shape.0 / CELL_SIZE;
            let y_cells = self.shape.1 / CELL_SIZE;
            ctx.set_stroke_style(&JsValue::from_str("#7f7fff"));
            ctx.set_line_width(1.);
            for j in 0..y_cells {
                for i in 0..x_cells {
                    let (x, y) = (i as f64, j as f64);
                    let idx = self
                        .shape
                        .idx(i * CELL_SIZE + CELL_SIZE / 2, j * CELL_SIZE + CELL_SIZE / 2);
                    let (mut vx, mut vy) = (self.vx[idx], self.vy[idx]);
                    let length2 = vx * vx + vy * vy;
                    if MAX_VELOCITY * MAX_VELOCITY < length2 {
                        let length = length2.sqrt();
                        vx *= MAX_VELOCITY / length;
                        vy *= MAX_VELOCITY / length;
                    }
                    ctx.begin_path();
                    ctx.move_to(
                        x * CELL_SIZE_F + CELL_SIZE_F / 2. + vx * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2. + vy * VELOCITY_SCALE,
                    );
                    ctx.line_to(
                        x * CELL_SIZE_F + CELL_SIZE_F / 2.
                            - vx * VELOCITY_SCALE
                            - vy * 0.2 * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2. - vy * VELOCITY_SCALE
                            + vx * 0.2 * VELOCITY_SCALE,
                    );
                    ctx.line_to(
                        x * CELL_SIZE_F + CELL_SIZE_F / 2. - vx * VELOCITY_SCALE
                            + vy * 0.2 * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2.
                            - vy * VELOCITY_SCALE
                            - vx * 0.2 * VELOCITY_SCALE,
                    );
                    ctx.close_path();
                    ctx.stroke();
                }
            }
        }
    }

    fn render_velocity_field_gl(&self, gl: &GL) -> Result<(), JsValue> {
        if self.params.show_velocity_field {
            const CELL_SIZE: isize = 10;
            const CELL_SIZE_F: f64 = CELL_SIZE as f64;
            const VELOCITY_SCALE: f64 = 1e3;
            const MAX_VELOCITY: f64 = 1e-2;
            let x_cells = self.shape.0 / CELL_SIZE;
            let y_cells = self.shape.1 / CELL_SIZE;

            let shader = self
                .assets
                .arrow_shader
                .as_ref()
                .ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
            gl.use_program(Some(&shader.program));

            gl.uniform1f(shader.alpha_loc.as_ref(), 0.5);

            let centerize = Matrix4::from_nonuniform_scale(2., -2., 2.)
                * Matrix4::from_translation(Vector3::new(-0.5, -0.5, -0.5));

            enable_buffer(gl, &self.assets.arrow_buffer, 2, shader.vertex_position);
            for j in 0..y_cells {
                for i in 0..x_cells {
                    let (x, y) = (i as f64, j as f64);
                    let idx = self
                        .shape
                        .idx(i * CELL_SIZE + CELL_SIZE / 2, j * CELL_SIZE + CELL_SIZE / 2);
                    let (vx, vy) = (self.vx[idx], self.vy[idx]);
                    let length2 = vx * vx + vy * vy;
                    let length = VELOCITY_SCALE
                        * if MAX_VELOCITY * MAX_VELOCITY < length2 {
                            MAX_VELOCITY
                        } else {
                            length2.sqrt()
                        };

                    let scale = Matrix4::from_nonuniform_scale(
                        length as f32 / self.shape.0 as f32,
                        -length as f32 / self.shape.1 as f32,
                        1.,
                    );

                    let rotation = Matrix4::from_angle_z(Rad(-vy.atan2(vx) as f32));

                    let translation = Matrix4::from_translation(Vector3::new(
                        (x * CELL_SIZE_F) as f32 / self.shape.0 as f32,
                        (y * CELL_SIZE_F) as f32 / self.shape.1 as f32,
                        0.,
                    ));

                    gl.uniform_matrix4fv_with_f32_array(
                        shader.transform_loc.as_ref(),
                        false,
                        <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(
                            &(centerize * translation * scale * rotation),
                        ),
                    );

                    gl.draw_arrays(GL::TRIANGLE_FAN, 0, 3);
                }
            }
        }
        Ok(())
    }

    /// WebGL specific initializations
    pub fn start_gl(&mut self, gl: &GL) -> Result<(), JsValue> {
        self.assets.start_gl(gl, &self.shape)
    }

    pub fn draw_tex(
        &self,
        gl: &GL,
        // texture: &WebGlTexture,
    ) -> Result<(), JsValue> {
        let shader = self
            .assets
            .rect_shader
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Failed to load rect_shader"))?;
        gl.use_program(Some(&shader.program));

        gl.uniform1f(shader.gamma_loc.as_ref(), self.params.gamma);

        gl.uniform_matrix4fv_with_f32_array(
            shader.transform_loc.as_ref(),
            false,
            <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&Matrix4::from_nonuniform_scale(
                1., -1., 1.,
            )),
        );

        gl.uniform_matrix3fv_with_f32_array(
            shader.tex_transform_loc.as_ref(),
            false,
            <Matrix3<f32> as AsRef<[f32; 9]>>::as_ref(
                // &Matrix3::from_scale(1.)
                &Matrix3::from_nonuniform_scale(
                    self.shape.0 as f32 / ceil_pow2(self.shape.0) as f32,
                    self.shape.1 as f32 / ceil_pow2(self.shape.1) as f32,
                ),
            ),
        );

        enable_buffer(gl, &self.assets.rect_buffer, 2, shader.vertex_position);
        gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);

        Ok(())
    }

    fn put_image_gl(&self, gl: &GL, data: &[u8]) -> Result<(), JsValue> {
        gl.use_program(Some(
            &self
                .assets
                .rect_shader
                .as_ref()
                .ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?
                .program,
        ));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.assets.flow_tex.as_ref());

        let level = 0;
        let src_format = GL::RGBA;
        let src_type = GL::UNSIGNED_BYTE;
        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            level,
            0,
            0,
            self.shape.0 as i32,
            self.shape.1 as i32,
            src_format,
            src_type,
            Some(data),
        )?;

        self.draw_tex(gl)?;

        if self.params.particles {
            if self.assets.instanced_arrays_ext.is_some() {
                self.render_particles_gl_instancing(gl)?;
            } else {
                self.render_particles_gl(gl)?;
            }
        }

        Ok(())
    }

    fn render_contours(&self, data: &mut [u8]) {
        use crate::marching_squares::{line, pick_bits};
        let shape = &self.shape;
        let temperature = if let Some(ref temperature) = self.temperature {
            temperature
        } else {
            return;
        };
        const LEVELS: usize = 8;
        for level in 1..LEVELS {
            let threshold = level as f64 / LEVELS as f64;
            let red = (threshold * 127. + 128.) as u8;
            let blue = ((1. - threshold) * 127. + 128.) as u8;
            for y in 0..shape.1 {
                for x in 0..shape.0 {
                    if let Some(_) = line(pick_bits(
                        temperature,
                        *shape,
                        (x as isize, y as isize),
                        threshold,
                    )) {
                        data[shape.idx(x, y) * 4] = red;
                        data[shape.idx(x, y) * 4 + 1] = 255;
                        data[shape.idx(x, y) * 4 + 2] = blue;
                        data[shape.idx(x, y) * 4 + 3] = 255;
                    }
                }
            }
        }
    }
}
