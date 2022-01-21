mod assets;
mod cfd;
mod gl_util;
mod params;
mod particles;
mod renderer;
mod shader_bundle;
mod xor128;

use cgmath::{Matrix3, Matrix4, Rad, Vector3};
use wasm_bindgen::prelude::*;
use web_sys::WebGlRenderingContext as GL;

use crate::{
    assets::Assets,
    cfd::{advect, decay, diffuse, obstacle_position, project},
    gl_util::enable_buffer,
    params::Params,
    particles::{new_particles, Particle},
    xor128::Xor128,
};

pub use renderer::{cfd_canvas, cfd_webgl};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    unsafe fn log(s: &str);
}

macro_rules! console_log {
    ($fmt:expr, $($arg1:expr),*) => {
        crate::log(&format!($fmt, $($arg1),+))
    };
    ($fmt:expr) => {
        crate::log($fmt)
    }
}

pub(crate) use console_log;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = ANGLEInstancedArrays)]
    type AngleInstancedArrays;

    #[wasm_bindgen(method, getter, js_name = VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE)]
    fn vertex_attrib_array_divisor_angle(this: &AngleInstancedArrays) -> i32;

    #[wasm_bindgen(method, catch, js_name = drawArraysInstancedANGLE)]
    fn draw_arrays_instanced_angle(
        this: &AngleInstancedArrays,
        mode: u32,
        first: i32,
        count: i32,
        primcount: i32,
    ) -> Result<(), JsValue>;

    // TODO offset should be i64
    #[wasm_bindgen(method, catch, js_name = drawElementsInstancedANGLE)]
    fn draw_elements_instanced_angle(
        this: &AngleInstancedArrays,
        mode: u32,
        count: i32,
        type_: u32,
        offset: i32,
        primcount: i32,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen(method, js_name = vertexAttribDivisorANGLE)]
    fn vertex_attrib_divisor_angle(this: &AngleInstancedArrays, index: u32, divisor: u32);
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn _document() -> web_sys::Document {
    window()
        .document()
        .expect("should have a document on window")
}

fn _body() -> web_sys::HtmlElement {
    _document().body().expect("document should have a body")
}

fn ceil_pow2(i: isize) -> isize {
    let mut bit = 0;
    while (1 << bit) < i {
        bit += 1;
    }
    1 << bit
}

trait Idx {
    fn idx(&self, x: isize, y: isize) -> usize;
}

type Shape = (isize, isize);

impl Idx for Shape {
    fn idx(&self, x: isize, y: isize) -> usize {
        let (width, height) = self;
        ((x + width) % width + (y + height) % height * width) as usize
    }
}

#[derive(Copy, Clone, PartialEq)]
enum BoundaryCondition {
    Wrap,
    Fixed,
    Flow(f64),
}

struct State {
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

    fn fluid_step(&mut self) {
        let visc = self.params.visc;
        let diff = self.params.diff;
        let dt = self.params.delta_time;
        let diffuse_iter = self.params.diffuse_iter;
        let project_iter = self.params.project_iter;
        let shape = self.shape;

        if self.params.temperature {
            if self.temperature.is_none() {
                self.temperature = Some(vec![0.5f64; (shape.0 * shape.1) as usize]);
            }
        }

        self.vx0.copy_from_slice(&self.vx);
        self.vy0.copy_from_slice(&self.vy);

        // console_log!("diffusion: {} viscousity: {}", diff, visc);

        diffuse(
            1,
            &mut self.vx0,
            &self.vx,
            visc,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );
        diffuse(
            2,
            &mut self.vy0,
            &self.vy,
            visc,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );

        // let (prev_div, prev_max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        project(
            &mut self.vx0,
            &mut self.vy0,
            &mut self.work,
            &mut self.work2,
            project_iter,
            shape,
            &self.params,
        );
        // let (after_div, max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        advect(
            1,
            &mut self.vx,
            &self.vx0,
            &self.vx0,
            &self.vy0,
            dt,
            shape,
            &self.params,
        );
        advect(
            2,
            &mut self.vy,
            &self.vy0,
            &self.vx0,
            &self.vy0,
            dt,
            shape,
            &self.params,
        );

        if let (true, Some(temperature)) = (self.params.temperature, &mut self.temperature) {
            let buoyancy = self.params.heat_buoyancy;
            for i in 0..shape.0 {
                for j in 1..shape.1 - 1 {
                    self.vy[shape.idx(i, j)] += buoyancy
                        * (temperature[shape.idx(i, j + 1)]
                            + temperature[shape.idx(i, j - 1)]
                            + temperature[shape.idx(i + 1, j)]
                            + temperature[shape.idx(i - 1, j)]
                            - 4. * temperature[shape.idx(i, j)]);
                }
            }

            for i in 0..shape.0 {
                if !self.params.half_heat_source || i < shape.0 / 2 {
                    temperature[shape.idx(i, 1)] +=
                        (0. - temperature[shape.idx(i, 1)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 2)] +=
                        (0. - temperature[shape.idx(i, 2)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 3)] +=
                        (0. - temperature[shape.idx(i, 3)]) * self.params.heat_exchange_rate;
                }
                if !self.params.half_heat_source || shape.0 / 2 <= i {
                    temperature[shape.idx(i, shape.1 - 4)] += (1.
                        - temperature[shape.idx(i, shape.1 - 3)])
                        * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1 - 3)] += (1.
                        - temperature[shape.idx(i, shape.1 - 2)])
                        * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1 - 2)] += (1.
                        - temperature[shape.idx(i, shape.1 - 1)])
                        * self.params.heat_exchange_rate;
                }
            }

            let mut work = std::mem::take(&mut self.work);
            work.copy_from_slice(temperature);
            diffuse(
                0,
                &mut work,
                temperature,
                diff,
                dt,
                diffuse_iter,
                shape,
                &self.params,
            );
            advect(
                0,
                temperature,
                &work,
                &self.vx0,
                &self.vy0,
                dt,
                shape,
                &self.params,
            );
            self.work = work;
        }

        // let (prev_div, prev_max_div) = sum_divergence(vx, vy, (self.width, self.height));
        project(
            &mut self.vx,
            &mut self.vy,
            &mut self.work,
            &mut self.work2,
            project_iter,
            shape,
            &self.params,
        );
        // let (after_div, max_div) = sum_divergence(vx, vy, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        diffuse(
            0,
            &mut self.work,
            &self.density,
            diff,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );
        advect(
            0,
            &mut self.density,
            &self.work,
            &self.vx,
            &self.vy,
            dt,
            shape,
            &self.params,
        );
        decay(&mut self.density, 1. - self.params.decay);

        diffuse(
            0,
            &mut self.work,
            &self.density2,
            diff,
            dt,
            1,
            shape,
            &self.params,
        );
        advect(
            0,
            &mut self.density2,
            &self.work,
            &self.vx,
            &self.vy,
            dt,
            shape,
            &self.params,
        );
        decay(&mut self.density2, 1. - self.params.decay);

        if self.params.obstacle && self.params.dye_from_obstacle {
            let (center, radius) = obstacle_position(&shape);
            let radius = radius + 1;
            for j in center.1 - radius..center.1 + radius {
                for i in center.0 - radius..center.0 + radius {
                    let dist2 = (j - center.1) * (j - center.1) + (i - center.0) * (i - center.0);
                    if dist2 < radius * radius {
                        if j < center.1 {
                            self.density[shape.idx(i, j)] = 1.;
                        }
                        if center.1 < j {
                            self.density2[shape.idx(i, j)] = 1.;
                        }
                    }
                }
            }
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

    fn reset_particles(&mut self) {
        // Technically, this is a new allocation that we would want to avoid, but it happens only
        // when the user presses reset button.
        self.particles = new_particles(&mut self.xor128, self.shape);
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
}
