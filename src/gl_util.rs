use super::{ceil_pow2, shape::Shape};
use wasm_bindgen::prelude::*;
use web_sys::{WebGlBuffer, WebGlProgram, WebGlRenderingContext as GL, WebGlShader, WebGlTexture};

pub(crate) fn compile_shader(
    context: &GL,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub(crate) fn link_program(
    context: &GL,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}

pub(crate) fn vertex_buffer_data(context: &GL, vertices: &[f32]) {
    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(vertices);

        context.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &vert_array, GL::STATIC_DRAW);
    };
}

pub(crate) fn vertex_buffer_sub_data(context: &GL, vertices: &[f32]) {
    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(vertices);

        context.buffer_sub_data_with_i32_and_array_buffer_view(GL::ARRAY_BUFFER, 0, &vert_array);
    };
}

pub(crate) fn enable_buffer(
    gl: &GL,
    buffer: &Option<WebGlBuffer>,
    elements: i32,
    vertex_position: u32,
) {
    gl.bind_buffer(GL::ARRAY_BUFFER, buffer.as_ref());
    gl.vertex_attrib_pointer_with_i32(vertex_position, elements, GL::FLOAT, false, 0, 0);
    gl.enable_vertex_attrib_array(vertex_position);
}

/// Create a texture buffer, which could be filled by later texSubImage calls.
pub(crate) fn gen_flow_texture(context: &GL, shape: &Shape) -> Result<WebGlTexture, JsValue> {
    let texture = context.create_texture().unwrap();
    context.bind_texture(GL::TEXTURE_2D, Some(&texture));

    context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        GL::TEXTURE_2D,
        0,
        GL::RGBA as i32,
        ceil_pow2(shape.0) as i32,
        ceil_pow2(shape.1) as i32,
        0,
        GL::RGBA,
        GL::UNSIGNED_BYTE,
        None,
    )?;

    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::REPEAT as i32);
    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::REPEAT as i32);
    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR as i32);

    Ok(texture)
}

/// Procedurally create a texture with round border
pub(crate) fn gen_particle_texture(context: &GL) -> Result<WebGlTexture, JsValue> {
    let texture = context.create_texture().unwrap();
    context.bind_texture(GL::TEXTURE_2D, Some(&texture));

    const PARTICLE_TEXTURE_SIZE: usize = 8;
    const PARTICLE_TEXTURE_HALF_SIZE: usize = PARTICLE_TEXTURE_SIZE / 2;
    let mut image = [0u8; PARTICLE_TEXTURE_SIZE * PARTICLE_TEXTURE_SIZE];
    for i in 0..PARTICLE_TEXTURE_SIZE {
        for j in 0..PARTICLE_TEXTURE_SIZE {
            let x = (i as i32 - PARTICLE_TEXTURE_HALF_SIZE as i32) as f32
                / PARTICLE_TEXTURE_HALF_SIZE as f32;
            let y = (j as i32 - PARTICLE_TEXTURE_HALF_SIZE as i32) as f32
                / PARTICLE_TEXTURE_HALF_SIZE as f32;
            image[i * PARTICLE_TEXTURE_SIZE + j] =
                ((1. - (x * x + y * y).sqrt()).max(0.) * 255.) as u8;
        }
    }

    context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        GL::TEXTURE_2D,
        0,
        GL::LUMINANCE as i32,
        PARTICLE_TEXTURE_SIZE as i32,
        PARTICLE_TEXTURE_SIZE as i32,
        0,
        GL::LUMINANCE,
        GL::UNSIGNED_BYTE,
        Some(&image),
    )?;

    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::REPEAT as i32);
    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::REPEAT as i32);
    context.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR as i32);

    Ok(texture)
}
