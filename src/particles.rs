use cgmath::{Matrix3, Matrix4, Vector3};
use wasm_bindgen::prelude::*;
use web_sys::WebGlRenderingContext as GL;

use super::{xor128::Xor128, Idx, Shape, State};
use crate::gl_util::{enable_buffer, vertex_buffer_sub_data};

pub(crate) const PARTICLE_COUNT: usize = 1000;
pub(crate) const PARTICLE_SIZE: f32 = 0.75;
pub(crate) const PARTICLE_MAX_TRAIL_LEN: usize = 10;

pub(crate) struct Particle {
    position: (f64, f64),
    history: Vec<(f64, f64)>,
    history_buf: Vec<(f32, f32)>,
}

pub(super) fn new_particles(xor128: &mut Xor128, shape: Shape) -> Vec<Particle> {
    (0..PARTICLE_COUNT)
        .map(|_| Particle {
            position: (
                (xor128.nexti() as isize % shape.0) as f64,
                (xor128.nexti() as isize % shape.1) as f64,
            ),
            history: vec![],
            history_buf: vec![],
        })
        .collect::<Vec<_>>()
}

impl State {
    pub(super) fn particle_position_array(&self) -> Vec<f64> {
        self.particles.iter().fold(vec![], |mut acc, p| {
            acc.push(p.position.0);
            acc.push(p.position.1);
            acc
        })
    }

    /// Redistirbute particles from concentrated area to sparse area, because the simulation tend to
    /// gather particles to certain location.
    ///
    /// The way it works is that it collects the local density information by splitting the entire space into grid cells
    /// and count particles in each cell, and then move particles from the cell with the highest density to
    /// less dense than a threshold cell.
    fn particle_redistribute(&mut self) {
        const GRID_SIZE: usize = 10;
        const GRID_ISIZE: isize = GRID_SIZE as isize;
        // Because expected density for each cell depends on the size of the simulation space and particle count, we
        // calculate the threshold from the parameters. Probably we should adjust GRID_SIZE instead.
        let min_density: usize =
            PARTICLE_COUNT / (self.shape.0 * self.shape.1 / GRID_ISIZE / GRID_ISIZE) as usize / 2;

        fn grid_position(particle: &Particle) -> (usize, usize) {
            let x = particle.position.0 as usize / GRID_SIZE;
            let y = particle.position.1 as usize / GRID_SIZE;
            (x, y)
        }

        let grid_columns = self.shape.0 as usize / GRID_SIZE;
        let grid_rows = self.shape.1 as usize / GRID_SIZE;
        let mut grid = vec![0usize; grid_columns * grid_rows];
        for particle in &self.particles {
            let (x, y) = grid_position(particle);
            grid[x + y * grid_columns] += 1;
        }
        // let mut global_moved = 0;
        // Need to avoid iterating over grid object in order to avoid borrow checker
        for i in 0..grid.len() {
            let mut moved = 0;
            while grid[i] < min_density && moved < min_density {
                let dst_cell_position = (i % grid_columns, i / grid_columns);
                let mut added_from = None;
                if let Some(max_cell) = grid.iter().enumerate().max_by_key(|v| v.1) {
                    let src_cell_position = (max_cell.0 % grid_columns, max_cell.0 / grid_columns);
                    if let Some(particle) = self
                        .particles
                        .iter_mut()
                        .find(|particle| src_cell_position == grid_position(particle))
                    {
                        particle.position = (
                            (self.xor128.nexti() as usize % GRID_SIZE
                                + dst_cell_position.0 * GRID_SIZE)
                                as f64,
                            (self.xor128.nexti() as usize % GRID_SIZE
                                + dst_cell_position.1 * GRID_SIZE)
                                as f64,
                        );
                        particle.history.clear();
                        particle.history_buf.clear();
                        added_from = Some(max_cell.0);
                    }
                }

                if let Some(j) = added_from {
                    grid[i] += 1;
                    grid[j] -= 1;
                    // global_moved += 1;
                    moved += 1;
                } else {
                    break;
                }
            }
        }
        // console_log!("grid {} x {}, max {:?} min {:?} min_density: {}, global_moved {}",
        //     grid_rows, grid_columns, grid.iter().max(), grid.iter().min(), min_density, global_moved);
    }

    pub(super) fn particle_step(&mut self, use_webgl: bool) {
        let desired_len = self.particles.len() * (self.params.particle_trails + 1) * 3;
        if self.particle_buf.len() < desired_len {
            self.particle_buf.resize(desired_len, 0.);
        }

        if self.params.redistribute_particles {
            self.particle_redistribute();
        }

        let mut idx = 0;
        for particle in &mut self.particles {
            let pvx = self.vx[self
                .shape
                .idx(particle.position.0 as isize, particle.position.1 as isize)];
            let pvy = self.vy[self
                .shape
                .idx(particle.position.0 as isize, particle.position.1 as isize)];
            let dtx = self.params.delta_time * (self.shape.0 - 2) as f64;
            let dty = self.params.delta_time * (self.shape.1 - 2) as f64;

            if 0 < self.params.particle_trails {
                if use_webgl && self.assets.instanced_arrays_ext.is_some() {
                    while self.params.particle_trails <= particle.history_buf.len() {
                        particle.history_buf.remove(0);
                    }

                    particle.history_buf.push((
                        particle.position.0 as f32 / self.shape.0 as f32,
                        particle.position.1 as f32 / self.shape.1 as f32,
                    ));
                } else {
                    while self.params.particle_trails <= particle.history.len() {
                        particle.history.remove(0);
                    }

                    particle.history.push(particle.position);
                }
            }

            if use_webgl && self.assets.instanced_arrays_ext.is_some() {
                let (x, y) = (particle.position.0, particle.position.1);
                self.particle_buf[idx * 3] = x as f32 / self.shape.0 as f32;
                self.particle_buf[idx * 3 + 1] = y as f32 / self.shape.1 as f32;
                self.particle_buf[idx * 3 + 2] = 1.;
                idx += 1;

                let history_len = particle.history_buf.len();
                for (i, position) in particle.history_buf.iter().enumerate() {
                    self.particle_buf[(idx + i) * 3] = position.0;
                    self.particle_buf[(idx + i) * 3 + 1] = position.1;
                    self.particle_buf[(idx + i) * 3 + 2] = i as f32 / history_len as f32;
                }
                idx += particle.history_buf.len();
            }

            // For some reason, updating particle.position after writing into particle_buf seems correct,
            // but I thought it should be the other way around.
            let (fwidth, fheight) = (self.shape.0 as f64, self.shape.1 as f64);
            particle.position.0 = (particle.position.0 + dtx * pvx + fwidth) % fwidth;
            particle.position.1 = (particle.position.1 + dty * pvy + fheight) % fheight;
        }

        if self.assets.instanced_arrays_ext.is_some() {
            self.particle_buf_active_len = idx;
        }
    }

    pub(super) fn render_particles(&self, data: &mut [u8]) {
        let shape = &self.shape;
        for particle in &self.particles {
            let (x, y) = (particle.position.0 as isize, particle.position.1 as isize);
            data[shape.idx(x, y) * 4] = 255;
            data[shape.idx(x, y) * 4 + 1] = 255;
            data[shape.idx(x, y) * 4 + 2] = 255;
            data[shape.idx(x, y) * 4 + 3] = 255;
            if 0 < self.params.particle_trails {
                for (i, position) in particle.history.iter().enumerate() {
                    let (x, y) = (position.0 as isize, position.1 as isize);
                    let inten = 255 * i / 10;
                    for j in 0..3 {
                        data[shape.idx(x, y) * 4 + j] = (inten
                            + data[shape.idx(x, y) * 4 + j] as usize * (255 - inten) / 255)
                            as u8;
                    }
                    data[shape.idx(x, y) * 4 + 3] = 255;
                }
            }
        }
    }

    /// Render particles without instancing. It's slow because it has to make a lot of WebGL calls.
    pub(super) fn render_particles_gl(&self, gl: &GL) -> Result<(), JsValue> {
        let shader = self
            .assets
            .particle_shader
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
        gl.use_program(Some(&shader.program));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.assets.particle_tex.as_ref());

        gl.uniform_matrix3fv_with_f32_array(
            shader.tex_transform_loc.as_ref(),
            false,
            <Matrix3<f32> as AsRef<[f32; 9]>>::as_ref(&Matrix3::from_scale(1.)),
        );

        let scale = Matrix4::from_nonuniform_scale(
            PARTICLE_SIZE / self.shape.0 as f32,
            -PARTICLE_SIZE / self.shape.1 as f32,
            1.,
        );
        let centerize = Matrix4::from_nonuniform_scale(2., -2., 2.)
            * Matrix4::from_translation(Vector3::new(-0.5, -0.5, -0.5));

        enable_buffer(gl, &self.assets.rect_buffer, 2, shader.vertex_position);
        for particle in &self.particles {
            let (x, y) = (particle.position.0, particle.position.1);
            let translation = Matrix4::from_translation(Vector3::new(
                x as f32 / self.shape.0 as f32,
                y as f32 / self.shape.1 as f32,
                0.,
            ));
            gl.uniform_matrix4fv_with_f32_array(
                shader.transform_loc.as_ref(),
                false,
                <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&(centerize * translation * scale)),
            );
            gl.uniform1f(shader.alpha_loc.as_ref(), 1.);

            gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);

            if 0 < self.params.particle_trails {
                for (i, position) in particle.history.iter().enumerate() {
                    let (x, y) = (position.0, position.1);
                    let inten = i as f32 / self.params.particle_trails as f32;
                    let translation = Matrix4::from_translation(Vector3::new(
                        x as f32 / self.shape.0 as f32,
                        y as f32 / self.shape.1 as f32,
                        0.,
                    ));
                    gl.uniform_matrix4fv_with_f32_array(
                        shader.transform_loc.as_ref(),
                        false,
                        <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(
                            &(centerize * translation * scale),
                        ),
                    );
                    gl.uniform1f(shader.alpha_loc.as_ref(), inten);
                    gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);
                }
            }
        }
        Ok(())
    }

    /// Render particles if the device supports instancing. It is much faster with fewer calls to the API.
    /// Note that there are no loops at all in this function.
    pub(super) fn render_particles_gl_instancing(&self, gl: &GL) -> Result<(), JsValue> {
        let instanced_arrays_ext = self
            .assets
            .instanced_arrays_ext
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Instanced arrays not supported"))?;

        let shader = self
            .assets
            .particle_instancing_shader
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
        if shader.attrib_position_loc < 0 {
            return Err(JsValue::from_str("matrix location was not found"));
        }

        gl.use_program(Some(&shader.program));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.assets.particle_tex.as_ref());

        let scale = Matrix4::from_nonuniform_scale(
            PARTICLE_SIZE / self.shape.0 as f32,
            -PARTICLE_SIZE / self.shape.1 as f32,
            1.,
        );

        gl.uniform_matrix4fv_with_f32_array(
            shader.transform_loc.as_ref(),
            false,
            <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&(scale)),
        );

        gl.uniform_matrix3fv_with_f32_array(
            shader.tex_transform_loc.as_ref(),
            false,
            <Matrix3<f32> as AsRef<[f32; 9]>>::as_ref(&Matrix3::from_scale(1.)),
        );

        gl.bind_buffer(GL::ARRAY_BUFFER, self.assets.particle_buffer.as_ref());
        vertex_buffer_sub_data(gl, &self.particle_buf);

        let stride = 3 * 4;
        gl.vertex_attrib_pointer_with_i32(
            shader.attrib_position_loc as u32,
            2,
            GL::FLOAT,
            false,
            stride,
            0,
        );
        gl.vertex_attrib_pointer_with_i32(
            shader.attrib_alpha_loc as u32,
            1,
            GL::FLOAT,
            false,
            stride,
            2 * 4,
        );

        instanced_arrays_ext.vertex_attrib_divisor_angle(shader.attrib_position_loc as u32, 1);
        gl.enable_vertex_attrib_array(shader.attrib_position_loc as u32);
        instanced_arrays_ext.vertex_attrib_divisor_angle(shader.attrib_alpha_loc as u32, 1);
        gl.enable_vertex_attrib_array(shader.attrib_alpha_loc as u32);

        instanced_arrays_ext.draw_arrays_instanced_angle(
            GL::TRIANGLE_FAN,
            0,                                   // offset
            4,                                   // num vertices per instance
            self.particle_buf_active_len as i32, // num instances
        )?;

        Ok(())
    }
}
