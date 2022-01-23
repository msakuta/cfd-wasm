use super::State;
use crate::{
    gl_util::enable_buffer,
    marching_squares::{line, pick_bits},
    shape::Idx,
};
use cgmath::{Matrix4, Vector3};
use wasm_bindgen::prelude::*;
use web_sys::WebGlRenderingContext as GL;

impl State {
    pub(super) fn render_contours_gl(&self, gl: &GL) -> Result<(), JsValue> {
        if !self.params.contour_lines {
            return Ok(());
        }

        if let Some(ref temperature) = self.temperature {
            let shader = self
                .assets
                .contour_shader
                .as_ref()
                .ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
            gl.use_program(Some(&shader.program));

            let centerize = Matrix4::from_nonuniform_scale(2., -2., 2.)
                * Matrix4::from_translation(Vector3::new(-0.5, -0.5, -0.5));

            let shape = self.shape;

            let scale = Matrix4::from_nonuniform_scale(
                0.5 / self.shape.0 as f32,
                0.5 / self.shape.1 as f32,
                1.,
            );

            enable_buffer(gl, &self.assets.rect_buffer, 2, shader.vertex_position);
            const LEVELS: usize = 8;
            for level in 1..LEVELS {
                let threshold = level as f64 / LEVELS as f64;
                let red = threshold * 0.5 + 0.5;
                let blue = (1. - threshold) * 0.5 + 0.5;

                gl.uniform4f(shader.color_loc.as_ref(), red as f32, 1.0, blue as f32, 0.5);

                for y in 0..shape.1 - 1 {
                    for x in 0..shape.0 - 1 {
                        if let Some(_) = line(pick_bits(
                            temperature,
                            shape,
                            (x as isize, y as isize),
                            threshold,
                        )) {
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

                            gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn render_contours(&self, data: &mut [u8]) {
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
            for y in 0..shape.1 - 1 {
                for x in 0..shape.0 - 1 {
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
