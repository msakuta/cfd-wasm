extern crate console_error_panic_hook;
extern crate libm;
use std::panic;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, WebGlBuffer, WebGlProgram,
    WebGlRenderingContext as GL, WebGlShader, WebGlTexture};
use cgmath::{Matrix3, Matrix4, Vector3, Rad};

use shader_bundle::ShaderBundle;

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


mod shader_bundle;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = ANGLEInstancedArrays)]
    type AngleInstancedArrays;

    #[wasm_bindgen(method, getter, js_name = VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE)]
    fn vertex_attrib_array_divisor_angle(this: &AngleInstancedArrays) -> i32;

    #[wasm_bindgen(method, catch, js_name = drawArraysInstancedANGLE)]
    fn draw_arrays_instanced_angle(this: &AngleInstancedArrays, mode: u32, first: i32, count: i32, primcount: i32) -> Result<(), JsValue>;

    // TODO offset should be i64
    #[wasm_bindgen(method, catch, js_name = drawElementsInstancedANGLE)]
    fn draw_elements_instanced_angle(this: &AngleInstancedArrays, mode: u32, count: i32, type_: u32, offset: i32, primcount: i32) -> Result<(), JsValue>;

    #[wasm_bindgen(method, js_name = vertexAttribDivisorANGLE)]
    fn vertex_attrib_divisor_angle(this: &AngleInstancedArrays, index: u32, divisor: u32);
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn request_animation_frame(f: &Closure<dyn FnMut() -> Result<(), JsValue>>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

fn _document() -> web_sys::Document {
    window()
        .document()
        .expect("should have a document on window")
}

fn _body() -> web_sys::HtmlElement {
    _document().body().expect("document should have a body")
}

#[derive(Clone, Copy)]
struct Xor128{
    x: u32
}

impl Xor128{
    fn new(seed: u32) -> Self{
        let mut ret = Xor128{x: 2463534242};
        if 0 < seed{
            ret.x ^= (seed & 0xffffffff) >> 0;
            ret.nexti();
        }
        ret.nexti();
        ret
    }

    fn nexti(&mut self) -> u32{
        // We must bitmask and logical shift to simulate 32bit unsigned integer's behavior.
        // The optimizer is likely to actually make it uint32 internally (hopefully).
        // T = (I + L^a)(I + R^b)(I + L^c)
        // a = 13, b = 17, c = 5
        let x1 = ((self.x ^ (self.x << 13)) & 0xffffffff) >> 0;
        let x2 = ((x1 ^ (x1 >> 17)) & 0xffffffff) >> 0;
        self.x = ((x2 ^ (x2 << 5)) & 0xffffffff) >> 0;
        self.x
    }
}

const CHECKER_CELL: usize = 16;

fn fill_checker(density: &mut [f64], (width, height): (usize, usize)) {
    for y2 in 0..height / CHECKER_CELL {
        for x2 in 0..width / CHECKER_CELL {
            if (x2 + y2) % 2 == 0 {
                for y in y2 * CHECKER_CELL..(y2 + 1) * CHECKER_CELL {
                    for x in x2 * CHECKER_CELL..(x2 + 1) * CHECKER_CELL {
                        density[x * width + y] = 1.;
                    }
                }
            }
        }
    }
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

fn add_density(density: &mut [f64], x: isize, y: isize, amount: f64, shape: Shape) {
    density[shape.idx(x, y)] += amount;
}

fn add_velo(vx: &mut [f64], vy: &mut [f64], index: usize, amount: [f64; 2]) {
    vx[index] += amount[0];
    vy[index] += amount[1];
}

fn obstacle_position(shape: &Shape) -> ((isize, isize), isize) {
    ((shape.0 / 4, shape.1 / 2), shape.0 / 16)
}

fn set_bnd(b: i32, x: &mut [f64], shape: Shape, params: &Params) {
    let (center, radius) = obstacle_position(&shape);

    // Edge cases
    if params.boundary_y == BoundaryCondition::Fixed {
        for i in 1..shape.0 - 1 {
            x[shape.idx(i, 0  )] = if b == 2 { -x[shape.idx(i, 1  )] } else { x[shape.idx(i, 1  )] };
            x[shape.idx(i, shape.1-1)] = if b == 2 { -x[shape.idx(i, shape.1-2)] } else { x[shape.idx(i, shape.1-2)] };
        }
    } else if let BoundaryCondition::Flow(f) = params.boundary_y {
        for i in 1..shape.0 - 1 {
            x[shape.idx(i, 0  )] = if b == 2 { f } else { x[shape.idx(i, 1  )] };
            x[shape.idx(i, shape.1-1)] = x[shape.idx(i, shape.1-2)];
        }
    }
    if params.boundary_x == BoundaryCondition::Fixed {
        for j in 1..shape.1 - 1 {
            x[shape.idx(0  , j)] = if b == 1 { -x[shape.idx(1  , j)] } else { x[shape.idx(1  , j)] };
            x[shape.idx(shape.0-1, j)] = if b == 1 { -x[shape.idx(shape.0-2, j)] } else { x[shape.idx(shape.0-2, j)] };
        }
    } else if let BoundaryCondition::Flow(f) = params.boundary_x {
        for j in 1..shape.1 - 1 {
            x[shape.idx(0  , j)] = if b == 1 {
                if j < center.1 { f * 0.9 } else { f }
            } else {
                x[shape.idx(1  , j)]
            };
            x[shape.idx(shape.0-1, j)] = x[shape.idx(shape.0-2, j)];
        }
    }

    if params.obstacle {
        for j in center.1-radius..center.1+radius {
            for i in center.0-radius..center.0+radius {
                let dist2 = (j - center.1) * (j - center.1) + (i - center.0) * (i - center.0);
                if dist2 < radius * radius {
                    x[shape.idx(i, j)] = 0.;
                }
            }
        }
    }

    // Corner cases (literally)
    x[shape.idx(0, 0)]             = 0.5 * (x[shape.idx(1, 0)] + x[shape.idx(0, 1)]);
    x[shape.idx(0, shape.1-1)]     = 0.5 * (x[shape.idx(1, shape.1-1)] + x[shape.idx(0, shape.1-2)]);
    x[shape.idx(shape.0-1, 0)]     = 0.5 * (x[shape.idx(shape.0-2, 0)] + x[shape.idx(shape.0-1, 1)]);
    x[shape.idx(shape.0-1, shape.1-1)] = 0.5 * (x[shape.idx(shape.0-2, shape.1-1)]
                                                   + x[shape.idx(shape.0-1, shape.1-2)]);
}

fn get_range(shape: Shape, params: &Params) -> (isize, isize, isize, isize) {
    let (i0, i1) = match params.boundary_x {
        BoundaryCondition::Fixed => (1, shape.0-1),
        BoundaryCondition::Wrap => (0, shape.0),
        BoundaryCondition::Flow(_) => (1, shape.0),
    };
    let (j0, j1) = match params.boundary_y {
        BoundaryCondition::Fixed => (1, shape.1-1),
        BoundaryCondition::Wrap => (0, shape.1),
        BoundaryCondition::Flow(_) => (1, shape.1),
    };
    (i0, i1, j0, j1)
}

/// Solve linear system of equasions using Gauss-Seidel relaxation
///
/// @param b ignored
/// @param x Target field to be solved
fn lin_solve(b: i32, x: &mut [f64], x0: &[f64], a: f64, c: f64, iter: usize, shape: Shape, params: &Params) {
    let c_recip = 1.0 / c;
    let (ib, ie, jb, je) = get_range(shape, params);
    for _ in 0..iter {
        for j in jb..je {
            for i in ib..ie {
                x[shape.idx(i, j)] = (x0[shape.idx(i, j)]
                    + a*(x[shape.idx(i+1, j  )]
                        +x[shape.idx(i-1, j  )]
                        +x[shape.idx(i  , j+1)]
                        +x[shape.idx(i  , j-1)]
                    )) * c_recip;
            }
        }
        set_bnd(b, x, shape, params);
    }
}

fn diffuse(b: i32, x: &mut [f64], x0: &[f64], diff: f64, dt: f64, iter: usize, shape: Shape, params: &Params) {
    let a = dt * diff * (shape.0 - 2) as f64 * (shape.1 - 2) as f64;
    lin_solve(b, x, x0, a, 1. + 4. * a, iter, shape, params);
}

fn advect(b: i32, d: &mut [f64], d0: &[f64], vx: &[f64], vy: &[f64], dt: f64, shape: Shape, params: &Params) {
    let dtx = dt * (shape.0 - 2) as f64;
    let dty = dt * (shape.1 - 2) as f64;
    let (ib, ie, jb, je) = get_range(shape, params);

    for j in jb..je {
        let jfloat = j as f64;
        for i in ib..ie {
            let ifloat = i as f64;
            let mut x    = ifloat - dtx * vx[shape.idx(i, j)];
            if params.boundary_x == BoundaryCondition::Fixed {
                if x < 0.5 { x = 0.5 };
                if x > shape.0 as f64 + 0.5 { x = shape.0 as f64 + 0.5 };
            }
            let i0 = x.floor();
            let i1 = i0 + 1.0;
            let mut y = jfloat - dty * vy[shape.idx(i, j)];
            if params.boundary_y == BoundaryCondition::Fixed {
                if y < 0.5 { y = 0.5 };
                if y > shape.1 as f64 + 0.5 { y = shape.1 as f64 + 0.5 };
            }
            let j0 = y.floor();
            let j1 = j0 + 1.0; 
            
            let s1 = x - i0; 
            let s0 = 1.0 - s1; 
            let t1 = y - j0; 
            let t0 = 1.0 - t1;
            
            let i0i = i0 as isize;
            let i1i = i1 as isize;
            let j0i = j0 as isize;
            let j1i = j1 as isize;
            
            d[shape.idx(i, j)] = 
                s0 * ( t0 * (d0[shape.idx(i0i, j0i)])
                    +( t1 * (d0[shape.idx(i0i, j1i)])))
            +s1 * ( t0 * (d0[shape.idx(i1i, j0i)])
                    +( t1 * (d0[shape.idx(i1i, j1i)])));
        }
    }
    set_bnd(b, d, shape, params);
}

fn divergence(vx: &[f64], vy: &[f64], shape: Shape, mut proc: impl FnMut(Shape, f64)) {
    for j in 0..shape.1 {
        for i in 0..shape.0 {
            proc((i, j),
                vx[shape.idx(i+1, j  )]
                -vx[shape.idx(i-1, j  )]
                +vy[shape.idx(i  , j+1)]
                -vy[shape.idx(i  , j-1)])
        }
    }
}

fn sum_divergence(vx: &[f64], vy: &[f64], shape: Shape) -> (f64, f64) {
    let mut sum = 0.;
    let mut max = 0.;
    divergence(vx, vy, shape, |_, div| {
        sum += div;
        max = div.max(max);
    });
    (sum, max)
}

fn project(vx: &mut [f64], vy: &mut [f64], p: &mut [f64], div: &mut [f64], iter: usize, shape: Shape, params: &Params) {
    let (ib, ie, jb, je) = get_range(shape, params);
    for j in jb..je {
        for i in ib..ie {
            div[shape.idx(i, j)] = -0.5*(
                    vx[shape.idx(i+1, j  )]
                    -vx[shape.idx(i-1, j  )]
                    +vy[shape.idx(i  , j+1)]
                    -vy[shape.idx(i  , j-1)]
                ) / shape.0 as f64;
            p[shape.idx(i, j)] = 0.;
        }
    }
    set_bnd(0, div, shape, params);
    set_bnd(0, p, shape, params);
    lin_solve(0, p, div, 1., 4., iter, shape, params);

    for j in jb..je {
        for i in ib..ie {
            vx[shape.idx(i, j)] -= 0.5 * (  p[shape.idx(i+1, j)]
                                            -p[shape.idx(i-1, j)]) * shape.0 as f64;
            vy[shape.idx(i, j)] -= 0.5 * (  p[shape.idx(i, j+1)]
                                            -p[shape.idx(i, j-1)]) * shape.1 as f64;
        }
    }
    set_bnd(1, vx, shape, params);
    set_bnd(2, vy, shape, params);
}

fn decay(s: &mut [f64], decay_rate: f64) {
    for v in s {
        *v *= decay_rate;
    }
}

struct Particle {
    position: (f64, f64),
    history: Vec<(f64, f64)>,
}

fn new_particles(xor128: &mut Xor128, shape: Shape) -> Vec<Particle> {
    (0..1000).map(|_| Particle {
        position: ((xor128.nexti() as isize % shape.0) as f64, (xor128.nexti() as isize % shape.1) as f64),
        history: vec![],
    }).collect::<Vec<_>>()
}

/// Create a texture buffer, which could be filled by later texSubImage calls.
fn gen_flow_texture(context: &GL, shape: &Shape) -> Result<WebGlTexture, JsValue> {
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
fn gen_particle_texture(context: &GL) -> Result<WebGlTexture, JsValue> {
    let texture = context.create_texture().unwrap();
    context.bind_texture(GL::TEXTURE_2D, Some(&texture));

    const PARTICLE_TEXTURE_SIZE: usize = 8;
    const PARTICLE_TEXTURE_HALF_SIZE: usize = PARTICLE_TEXTURE_SIZE / 2;
    let mut image = [0u8; PARTICLE_TEXTURE_SIZE * PARTICLE_TEXTURE_SIZE];
    for i in 0..PARTICLE_TEXTURE_SIZE {
        for j in 0..PARTICLE_TEXTURE_SIZE {
            let x = (i as i32 - PARTICLE_TEXTURE_HALF_SIZE as i32) as f32 / PARTICLE_TEXTURE_HALF_SIZE as f32;
            let y = (j as i32 - PARTICLE_TEXTURE_HALF_SIZE as i32) as f32 / PARTICLE_TEXTURE_HALF_SIZE as f32;
            image[i * PARTICLE_TEXTURE_SIZE + j] = ((1. - (x * x + y * y).sqrt()).max(0.) * 255.) as u8;
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


#[derive(Copy, Clone, PartialEq)]
enum BoundaryCondition {
    Wrap,
    Fixed,
    Flow(f64),
}

#[derive(Copy, Clone)]
struct Params{
    delta_time: f64,
    skip_frames: u32,
    mouse_pos: [i32; 2],
    visc: f64,
    diff: f64,
    density: f64,
    decay: f64,
    mouse_flow_speed: f64,
    diffuse_iter: usize,
    project_iter: usize,
    temperature: bool,
    half_heat_source: bool,
    heat_exchange_rate: f64,
    heat_buoyancy: f64,
    mouse_flow: bool,
    show_velocity: bool,
    show_velocity_field: bool,
    obstacle: bool,
    dye_from_obstacle: bool,
    particles: bool,
    particle_trails: usize,
    boundary_y: BoundaryCondition,
    boundary_x: BoundaryCondition,
}

const PARTICLE_SIZE: f32 = 0.75;
const PARTICLE_MAX_TRAIL_LEN: usize = 10;

struct Assets {
    instanced_arrays_ext: Option<AngleInstancedArrays>,
    flow_tex: Option<WebGlTexture>,
    particle_tex: Option<WebGlTexture>,
    rect_shader: Option<ShaderBundle>,
    particle_shader: Option<ShaderBundle>,
    arrow_shader: Option<ShaderBundle>,
    trail_shader: Option<ShaderBundle>,
    pub trail_buffer: Option<WebGlBuffer>,
    pub rect_buffer: Option<WebGlBuffer>,
    arrow_buffer: Option<WebGlBuffer>,
    particle_buffer: Option<WebGlBuffer>,
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

        let params = Params {
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
            show_velocity: true,
            show_velocity_field: false,
            obstacle: false,
            dye_from_obstacle: true,
            particles: true,
            particle_trails: 0,
            boundary_x: BoundaryCondition::Wrap,
            boundary_y: BoundaryCondition::Wrap,
        };

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
            assets: Assets {
                instanced_arrays_ext: None,
                flow_tex: None,
                particle_tex: None,
                rect_shader: None,
                particle_shader: None,
                arrow_shader: None,
                trail_shader: None,
                trail_buffer: None,
                rect_buffer: None,
                arrow_buffer: None,
                particle_buffer: None,
            }
        }
    }

    fn fluid_step(&mut self) {
        let visc     = self.params.visc;
        let diff     = self.params.diff;
        let dt       = self.params.delta_time;
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

        diffuse(1, &mut self.vx0, &self.vx, visc, dt, diffuse_iter, shape, &self.params);
        diffuse(2, &mut self.vy0, &self.vy, visc, dt, diffuse_iter, shape, &self.params);

        // let (prev_div, prev_max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        project(&mut self.vx0, &mut self.vy0, &mut self.work, &mut self.work2, project_iter, shape, &self.params);
        // let (after_div, max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        advect(1, &mut self.vx, &self.vx0, &self.vx0, &self.vy0, dt, shape, &self.params);
        advect(2, &mut self.vy, &self.vy0, &self.vx0, &self.vy0, dt, shape, &self.params);

        if let (true, Some(temperature)) = (self.params.temperature, &mut self.temperature) {
            let buoyancy = self.params.heat_buoyancy;
            for i in 0..shape.0 {
                for j in 1..shape.1-1 {
                    self.vy[shape.idx(i, j)] += buoyancy * (
                        temperature[shape.idx(i, j + 1)] + temperature[shape.idx(i, j - 1)]
                        +temperature[shape.idx(i + 1, j)] + temperature[shape.idx(i - 1, j)]
                        -4. * temperature[shape.idx(i, j)]);
                }
            }

            for i in 0..shape.0 {
                if !self.params.half_heat_source || i < shape.0 / 2 {
                    temperature[shape.idx(i, 1)] += (0. - temperature[shape.idx(i, 1)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 2)] += (0. - temperature[shape.idx(i, 2)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 3)] += (0. - temperature[shape.idx(i, 3)]) * self.params.heat_exchange_rate;
                }
                if !self.params.half_heat_source || shape.0 / 2 <= i {
                    temperature[shape.idx(i, shape.1-4)] += (1. - temperature[shape.idx(i, shape.1-3)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1-3)] += (1. - temperature[shape.idx(i, shape.1-2)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1-2)] += (1. - temperature[shape.idx(i, shape.1-1)]) * self.params.heat_exchange_rate;
                }
            }

            let mut work = std::mem::take(&mut self.work);
            work.copy_from_slice(temperature);
            diffuse(0, &mut work, temperature, diff, dt, diffuse_iter, shape, &self.params);
            advect(0, temperature, &work, &self.vx0, &self.vy0, dt, shape, &self.params);
            self.work = work;
        }

        // let (prev_div, prev_max_div) = sum_divergence(vx, vy, (self.width, self.height));
        project(&mut self.vx, &mut self.vy, &mut self.work, &mut self.work2, project_iter, shape, &self.params);
        // let (after_div, max_div) = sum_divergence(vx, vy, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        diffuse(0, &mut self.work, &self.density, diff, dt, diffuse_iter, shape, &self.params);
        advect(0, &mut self.density, &self.work, &self.vx, &self.vy, dt, shape, &self.params);
        decay(&mut self.density, 1. - self.params.decay);

        diffuse(0, &mut self.work, &self.density2, diff, dt, 1, shape, &self.params);
        advect(0, &mut self.density2, &self.work, &self.vx, &self.vy, dt, shape, &self.params);
        decay(&mut self.density2, 1. - self.params.decay);

        if self.params.obstacle && self.params.dye_from_obstacle {
            let (center, radius) = obstacle_position(&shape);
            let radius = radius + 1;
            for j in center.1-radius..center.1+radius {
                for i in center.0-radius..center.0+radius {
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

    fn particle_step(&mut self) {

        let desired_len = self.particles.len() * (self.params.particle_trails + 1) * 2;
        if self.particle_buf.len() < desired_len {
            self.particle_buf.resize(desired_len, 0.);
        }

        let mut idx = 0;
        for particle in &mut self.particles {
            let pvx = self.vx[self.shape.idx(particle.position.0 as isize, particle.position.1 as isize)];
            let pvy = self.vy[self.shape.idx(particle.position.0 as isize, particle.position.1 as isize)];
            let dtx = self.params.delta_time * (self.shape.0 - 2) as f64;
            let dty = self.params.delta_time * (self.shape.1 - 2) as f64;

            if 0 < self.params.particle_trails {
                while self.params.particle_trails < particle.history.len() {
                    particle.history.remove(0);
                }

                particle.history.push(particle.position);
            }

            if self.assets.instanced_arrays_ext.is_some() {
                let (x, y) = (particle.position.0, particle.position.1);
                self.particle_buf[idx * 2    ] = x as f32 / self.shape.0 as f32;
                self.particle_buf[idx * 2 + 1] = y as f32 / self.shape.1 as f32;
                idx += 1;

                for (t, (x, y)) in particle.history.iter().enumerate() {
                    if self.params.particle_trails <= t {
                        break;
                    }
                    self.particle_buf[idx * 2    ] = *x as f32 / self.shape.0 as f32;
                    self.particle_buf[idx * 2 + 1] = *y as f32 / self.shape.1 as f32;
                    idx += 1;
                }
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
        let (u, v, w) = (if self.params.temperature {
            self.temperature.as_ref().unwrap_or(&self.density)
        } else {
            &self.density
        }, &self.density2, &self.work);
        let shape = &self.shape;
        for y in 0..self.shape.1 {
            for x in 0..self.shape.0 {
                data[shape.idx(x, y) * 4    ] = ((u[shape.idx(x, y)]) / U_MAX * 127.) as u8;
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
            for j in center.1-radius..center.1+radius {
                for i in center.0-radius..center.0+radius {
                    let dist2 = (j - center.1) * (j - center.1) + (i - center.0) * (i - center.0);
                    if dist2 < radius * radius {
                        data[shape.idx(i, j) * 4    ] = 127;
                        data[shape.idx(i, j) * 4 + 1] = 127;
                        data[shape.idx(i, j) * 4 + 2] = 0;
                        data[shape.idx(i, j) * 4 + 3] = 255;
                    }
                }
            }
        }
    }

    fn render_particles(&self, data: &mut [u8]) {
        let shape = &self.shape;
        for particle in &self.particles {
            let (x, y) = (particle.position.0 as isize, particle.position.1 as isize);
            data[shape.idx(x, y) * 4    ] = 255;
            data[shape.idx(x, y) * 4 + 1] = 255;
            data[shape.idx(x, y) * 4 + 2] = 255;
            data[shape.idx(x, y) * 4 + 3] = 255;
            if 0 < self.params.particle_trails {
                for (i, position) in particle.history.iter().enumerate() {
                    let (x, y) = (position.0 as isize, position.1 as isize);
                    let inten = 255 * i / 10;
                    for j in 0..3 {
                        data[shape.idx(x, y) * 4 + j] = (inten +
                            data[shape.idx(x, y) * 4 + j] as usize * (255 - inten) / 255) as u8;
                    }
                    data[shape.idx(x, y) * 4 + 3] = 255;
                }
            }
        }
    }

    fn render_particles_gl(&self, gl: &GL) -> Result<(), JsValue> {
        let shader = self.assets.particle_shader.as_ref().ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
        gl.use_program(Some(&shader.program));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.assets.particle_tex.as_ref());

        gl.uniform_matrix3fv_with_f32_array(
            shader.tex_transform_loc.as_ref(),
            false,
            <Matrix3<f32> as AsRef<[f32; 9]>>::as_ref(&Matrix3::from_scale(1.))
        );

        let scale = Matrix4::from_nonuniform_scale(PARTICLE_SIZE / self.shape.0 as f32, -PARTICLE_SIZE / self.shape.1 as f32, 1.);
        let centerize = Matrix4::from_nonuniform_scale(2., -2., 2.)
            * Matrix4::from_translation(Vector3::new(-0.5, -0.5, -0.5));

        enable_buffer(gl, &self.assets.rect_buffer, 2, shader.vertex_position);
        for particle in &self.particles {
            let (x, y) = (particle.position.0, particle.position.1);
            let translation = Matrix4::from_translation(
                Vector3::new(x as f32 / self.shape.0 as f32, y as f32 / self.shape.1 as f32, 0.));
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
                    let translation = Matrix4::from_translation(
                        Vector3::new(x as f32 / self.shape.0 as f32, y as f32 / self.shape.1 as f32, 0.));
                    gl.uniform_matrix4fv_with_f32_array(
                        shader.transform_loc.as_ref(),
                        false,
                        <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&(centerize * translation * scale)),
                    );
                    gl.uniform1f(shader.alpha_loc.as_ref(), inten);
                    gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);
                }
            }
        }
        Ok(())
    }

    fn render_particles_gl_instancing(&self, gl: &GL) -> Result<(), JsValue> {
        let instanced_arrays_ext = self.assets.instanced_arrays_ext.as_ref()
            .ok_or_else(|| JsValue::from_str("Instanced arrays not supported"))?;


        let shader = self.assets.particle_shader.as_ref().ok_or_else(|| JsValue::from_str("Could not find rect_shader"))?;
        if shader.position_loc < 0 {
            return Err(JsValue::from_str("matrix location was not found"));
        }

        gl.use_program(Some(&shader.program));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.assets.particle_tex.as_ref());

        let scale = Matrix4::from_nonuniform_scale(
            PARTICLE_SIZE / self.shape.0 as f32,
            -PARTICLE_SIZE / self.shape.1 as f32,
            1.
        );

        gl.uniform_matrix4fv_with_f32_array(
            shader.transform_loc.as_ref(),
            false,
            <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&(scale))
        );

        gl.uniform_matrix3fv_with_f32_array(
            shader.tex_transform_loc.as_ref(),
            false,
            <Matrix3<f32> as AsRef<[f32; 9]>>::as_ref(&Matrix3::from_scale(1.))
        );

        gl.uniform1f(shader.alpha_loc.as_ref(), 0.5);

        gl.bind_buffer(GL::ARRAY_BUFFER, self.assets.particle_buffer.as_ref());
        vertex_buffer_sub_data(gl, &self.particle_buf);

        enable_buffer(gl, &self.assets.particle_buffer, 2, shader.position_loc as u32);

        instanced_arrays_ext.vertex_attrib_divisor_angle(shader.position_loc as u32, 1);

        instanced_arrays_ext.draw_arrays_instanced_angle(
            GL::TRIANGLE_FAN,
            0,   // offset
            4,   // num vertices per instance
            self.particle_buf_active_len as i32,  // num instances
        )?;

        Ok(())
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
                    let idx = self.shape.idx(i * CELL_SIZE + CELL_SIZE / 2, j * CELL_SIZE + CELL_SIZE / 2);
                    let (mut vx, mut vy) = (self.vx[idx], self.vy[idx]);
                    let length2 = vx * vx + vy * vy;
                    if MAX_VELOCITY * MAX_VELOCITY < length2 {
                        let length = length2.sqrt();
                        vx *= MAX_VELOCITY / length;
                        vy *= MAX_VELOCITY / length;
                    }
                    ctx.begin_path();
                    ctx.move_to(x * CELL_SIZE_F + CELL_SIZE_F / 2. + vx * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2. + vy * VELOCITY_SCALE);
                    ctx.line_to(x * CELL_SIZE_F + CELL_SIZE_F / 2. - vx * VELOCITY_SCALE - vy * 0.2 * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2. - vy * VELOCITY_SCALE + vx * 0.2 * VELOCITY_SCALE);
                    ctx.line_to(x * CELL_SIZE_F + CELL_SIZE_F / 2. - vx * VELOCITY_SCALE + vy * 0.2 * VELOCITY_SCALE,
                        y * CELL_SIZE_F + CELL_SIZE_F / 2. - vy * VELOCITY_SCALE - vx * 0.2 * VELOCITY_SCALE);
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

            let shader = self.assets.arrow_shader.as_ref().ok_or_else(
                || JsValue::from_str("Could not find rect_shader"))?;
            gl.use_program(Some(&shader.program));

            gl.uniform1f(shader.alpha_loc.as_ref(), 0.5);

            let centerize = Matrix4::from_nonuniform_scale(2., -2., 2.)
                * Matrix4::from_translation(Vector3::new(-0.5, -0.5, -0.5));

            enable_buffer(gl, &self.assets.arrow_buffer, 2, shader.vertex_position);
            for j in 0..y_cells {
                for i in 0..x_cells {
                    let (x, y) = (i as f64, j as f64);
                    let idx = self.shape.idx(i * CELL_SIZE + CELL_SIZE / 2, j * CELL_SIZE + CELL_SIZE / 2);
                    let (vx, vy) = (self.vx[idx], self.vy[idx]);
                    let length2 = vx * vx + vy * vy;
                    let length = VELOCITY_SCALE * if MAX_VELOCITY * MAX_VELOCITY < length2 {
                        MAX_VELOCITY
                    } else {
                        length2.sqrt()
                    };

                    let scale = Matrix4::from_nonuniform_scale(
                        length as f32 / self.shape.0 as f32,
                        -length as f32 / self.shape.1 as f32,
                        1.
                    );

                    let rotation = Matrix4::from_angle_z(Rad(-vy.atan2(vx) as f32));

                    let translation = Matrix4::from_translation(
                        Vector3::new(
                            (x * CELL_SIZE_F) as f32 / self.shape.0 as f32,
                            (y * CELL_SIZE_F) as f32 / self.shape.1 as f32, 0.));

                    gl.uniform_matrix4fv_with_f32_array(
                        shader.transform_loc.as_ref(),
                        false,
                        <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&(centerize * translation * scale * rotation)),
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

        self.assets.flow_tex = Some(gen_flow_texture(gl, &self.shape)?);

        self.assets.particle_tex = Some(gen_particle_texture(gl)?);

        self.assets.instanced_arrays_ext = Some(gl.get_extension("ANGLE_instanced_arrays")
            .unwrap().unwrap().unchecked_into::<AngleInstancedArrays>());

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

            void main() {
                vec4 texColor = texture2D( texture, vec2(texCoords.x, texCoords.y) );
                gl_FragColor = vec4(texColor.rgb, texColor.a * alpha);
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

        self.assets.rect_shader = Some(shader);

        let vert_shader_instancing = compile_shader(
            &gl,
            GL::VERTEX_SHADER,
            r#"
            attribute vec2 vertexData;
            attribute vec2 position;
            uniform mat4 transform;
            uniform mat3 texTransform;
            varying vec2 texCoords;

            void main() {
                mat4 centerize = mat4(
                    4, 0, 0, 0,
                    0, -4, 0, 0,
                    0, 0, 4, 0,
                    -1, 1, -1, 1);
                gl_Position = centerize * (transform * vec4(vertexData.xy, 0.0, 1.0) + vec4(position.xy, 0.0, 1.0));
                texCoords = (texTransform * vec3((vertexData.xy + 1.) * 0.5, 1.)).xy;
            }
        "#,
        )?;
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
        let program = link_program(&gl, &vert_shader_instancing, &frag_shader_add)?;
        let shader = ShaderBundle::new(&gl, program);
        gl.uniform1f(shader.alpha_loc.as_ref(), 1.);
        self.assets.particle_shader = Some(shader);

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
        self.assets.arrow_shader = Some(shader);


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
        self.assets.trail_shader = Some(ShaderBundle::new(&gl, program));

        gl.active_texture(GL::TEXTURE0);
        gl.uniform1i(
            self.assets
                .trail_shader
                .as_ref()
                .and_then(|s| s.texture_loc.as_ref()),
            0,
        );

        self.assets.trail_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);

        self.assets.rect_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);
        gl.bind_buffer(GL::ARRAY_BUFFER, self.assets.rect_buffer.as_ref());
        let rect_vertices: [f32; 8] = [1., 1., -1., 1., -1., -1., 1., -1.];
        vertex_buffer_data(&gl, &rect_vertices);

        self.assets.arrow_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);
        gl.bind_buffer(GL::ARRAY_BUFFER, self.assets.arrow_buffer.as_ref());
        let arrow_vertices: [f32; 6] = [1., 0., -1., -0.2, -1., 0.2];
        vertex_buffer_data(&gl, &arrow_vertices);

        self.assets.particle_buffer = Some(gl.create_buffer().ok_or("failed to create buffer")?);
        gl.bind_buffer(GL::ARRAY_BUFFER, self.assets.particle_buffer.as_ref());
        gl.buffer_data_with_i32(
            GL::ARRAY_BUFFER,
            (self.particles.len() * 2 * std::mem::size_of::<f32>() * (1 + PARTICLE_MAX_TRAIL_LEN)) as i32,
            GL::DYNAMIC_DRAW
        );

        gl.clear_color(0.0, 0.2, 0.5, 1.0);

        Ok(())
    }

    pub fn draw_tex(
        &self,
        gl: &GL,
        // texture: &WebGlTexture,
    ) -> Result<(), JsValue> {
        let shader = self.assets.rect_shader.as_ref()
            .ok_or_else(|| JsValue::from_str("Failed to load rect_shader"))?;
        gl.use_program(Some(&shader.program));

        gl.uniform_matrix4fv_with_f32_array(
            shader.transform_loc.as_ref(),
            false,
            <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(&Matrix4::from_nonuniform_scale(1., -1., 1.)),
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
            )
        );

        enable_buffer(gl, &self.assets.rect_buffer, 2, shader.vertex_position);
        gl.draw_arrays(GL::TRIANGLE_FAN, 0, 4);

        Ok(())
    }

    fn put_image_gl(&self, gl: &GL, data: &[u8]) -> Result<(), JsValue> {
        gl.use_program(Some(&self.assets.rect_shader.as_ref().ok_or_else(
            || JsValue::from_str("Could not find rect_shader"))?.program));

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

trait Renderer {
    fn start(&self, _state: &mut State) -> Result<(), JsValue> {
        Ok(())
    }
    fn render(&self, state: &State, data: &mut [u8]) -> Result<(), JsValue>;
}

struct CanvasRenderer {
    ctx: CanvasRenderingContext2d,
}

impl Renderer for CanvasRenderer {
    fn render(&self, state: &State, data: &mut [u8]) -> Result<(), JsValue> {
        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
            wasm_bindgen::Clamped(data), state.shape.0 as u32, state.shape.1 as u32).unwrap();
        if state.params.particles {
            state.render_particles(data);
        }
        self.ctx.put_image_data(&image_data, 0., 0.)?;
        state.render_velocity_field(&self.ctx);
        Ok(())
    }
}

struct WebGLRenderer {
    gl: GL,
}

impl Renderer for WebGLRenderer {
    fn start(&self, state: &mut State) -> Result<(), JsValue> {
        state.start_gl(&self.gl)
    }
    fn render(&self, state: &State, data: &mut [u8]) -> Result<(), JsValue> {
        self.gl.clear(GL::COLOR_BUFFER_BIT);
        state.put_image_gl(&self.gl, &data)?;
        state.render_velocity_field_gl(&self.gl)?;
        Ok(())
    }
}

#[wasm_bindgen]
pub fn cfd_canvas(width: usize, height: usize, ctx: web_sys::CanvasRenderingContext2d, callback: js_sys::Function) -> Result<(), JsValue> {
    cfd_temp(width, height, CanvasRenderer{ ctx }, callback)
}

#[wasm_bindgen]
pub fn cfd_webgl(width: usize, height: usize, gl: GL, callback: js_sys::Function) -> Result<(), JsValue> {
    cfd_temp(width, height, WebGLRenderer{ gl }, callback)
}

fn cfd_temp(width: usize, height: usize, renderer: impl Renderer + 'static, callback: js_sys::Function) -> Result<(), JsValue> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    let mut data = vec![0u8; 4 * width * height];

    let mut state = State::new(width, height);

    renderer.start(&mut state)?;

    state.render_fluid(&mut data);

    let func = Rc::new(RefCell::new(None));
    let g = func.clone();

    let mut i = 0;

    console_log!("Starting frames");

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || -> Result<(), JsValue> {
        let Params{mouse_pos, ..} = state.params;
        // console_log!("Rendering frame {}, mouse_pos: {}, {} delta_time: {}, skip_frames: {}, f: {}, k: {}, ru: {}, rv: {}",
        //     i, mouse_pos[0], mouse_pos[1], delta_time, state.params.skip_frames, f, k, state.params.ru, state.params.rv);

        i += 1;

        // let velo = state.calc_velo(&state.vx, &state.vy);
        // let mut div = vec![0f64; width * height];
        // divergence(&Vx, &Vy, (width, height), |(x, y), v| div[ix(x as i32, y as i32)] = v.abs());

        let average = state.vx.iter().zip(state.vy.iter())
            .fold(0f64, |acc, v| acc.max((v.0 * v.0 + v.1 * v.1).sqrt()));

        // console_log!("frame {}, density sum {:.5e}, cen: {:.5e} maxvelo: {:.5e} mouse {:?}",
        //     i, average,
        //     state.density[ix(mouse_pos[0], mouse_pos[1])], velo.iter().fold(0., |acc: f64, v| acc.max(*v)),
        //     mouse_pos);

        if state.params.mouse_flow {
            let density_phase = 0.5 * (i as f64 * 0.02352 * std::f64::consts::PI).cos() + 0.5;
            let density = if state.params.temperature {
                state.temperature.as_mut().unwrap_or(&mut state.density)
            } else {
                &mut state.density
            };
            add_density(density, mouse_pos[0] as isize, mouse_pos[1] as isize,
                density_phase * state.params.density, state.shape);
            let density2_phase = 0.5 * ((i as f64 * 0.02352 + 1.) * std::f64::consts::PI).cos() + 0.5;
            add_density(&mut state.density2, mouse_pos[0] as isize, mouse_pos[1] as isize,
                density2_phase * state.params.density, state.shape);
            // let angle_rad = (i as f64 * 0.002 * std::f64::consts::PI) * 2. * std::f64::consts::PI;
            let mut hasher = Xor128::new((i / 16) as u32);
            let angle_rad = ((hasher.nexti() as f64 / 0xffffffffu32 as f64) * 2. * std::f64::consts::PI) * 2. * std::f64::consts::PI;
            add_velo(&mut state.vx, &mut state.vy, state.shape.idx(mouse_pos[0] as isize, mouse_pos[1] as isize),
                [state.params.mouse_flow_speed * angle_rad.cos(), state.params.mouse_flow_speed * angle_rad.sin()]);
        }

        for _ in 0..state.params.skip_frames {
            state.fluid_step();
            state.particle_step();
        }

        state.render_fluid(&mut data);

        renderer.render(&state, &mut data)?;

        let particle_vec = state.particles.iter()
        .fold(vec![], |mut acc, p| {
            acc.push(p.position.0);
            acc.push(p.position.1);
            acc
        });
        let buf = js_sys::Float64Array::from(&particle_vec as &[f64]);

        let callback_ret = callback.call1(&window(), &JsValue::from(buf)).unwrap_or(JsValue::from(true));
        let terminate_requested = js_sys::Reflect::get(&callback_ret, &JsValue::from("terminate"))
            .unwrap_or_else(|_| JsValue::from(true));
        if terminate_requested.is_truthy() {
            return Ok(());
        }
        let assign_state = |name: &str, setter: &mut dyn FnMut(f64)| {
            if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from(name)) {
                if let Some(the_val) = new_val.as_f64() {
                    // console_log!("received {} value: {}", name, the_val);
                    // let mut params_ref = (*params).borrow_mut();
                    // setter(&mut params_ref, the_val);
                    setter(the_val);
                }
            }
        };
        let assign_usize = |name: &str, setter: &mut dyn FnMut(usize)| {
            if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from(name)) {
                if let Some(the_val) = new_val.as_f64() {
                    // console_log!("received {} value: {}", name, the_val);
                    // let mut params_ref = (*params).borrow_mut();
                    // setter(&mut params_ref, the_val);
                    setter(the_val as usize);
                }
            }
        };
        let assign_check = |name: &str, setter: &mut dyn FnMut(bool)| {
            if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from(name)) {
                if let Some(the_val) = new_val.as_bool() {
                    // console_log!("received {} value: {}", name, the_val);
                    // let mut params_ref = (*params).borrow_mut();
                    // setter(&mut params_ref, the_val);
                    setter(the_val);
                }
            }
        };
        let assign_boundary = |name: &str, setter: &mut dyn FnMut(BoundaryCondition)| {
            use BoundaryCondition::{Wrap, Fixed, Flow};

            if let (Ok(new_val), Ok(flow)) = (
                js_sys::Reflect::get(&callback_ret, &JsValue::from(name)),
                js_sys::Reflect::get(&callback_ret, &JsValue::from("boundaryFlowSpeed"))) {
                if let (Some(s), Some(flow)) = (new_val.as_string(), flow.as_f64()) {
                    match &s as &str {
                        "Fixed" => setter(Fixed),
                        "Wrap" => setter(Wrap),
                        "Flow" => setter(Flow(flow)),
                        _ => return Err(JsValue::from_str("Unrecognized boundary condition type")),
                    }
                } else {
                    return Err(JsValue::from_str("Unrecognized boundary condition type"));
                }
            }
            Ok(())
        };

        assign_state("deltaTime", &mut |value| state.params.delta_time = value);
        assign_state("skipFrames", &mut |value| state.params.skip_frames = value as u32);
        assign_state("visc", &mut |value| state.params.visc = value);
        assign_state("diff", &mut |value| state.params.diff = value);
        assign_state("density", &mut |value| state.params.density = value);
        assign_state("decay", &mut |value| state.params.decay = value);
        assign_state("mouseFlowSpeed", &mut |value| state.params.mouse_flow_speed = value);
        assign_usize("diffIter", &mut |value| state.params.diffuse_iter = value);
        assign_usize("projIter", &mut |value| state.params.project_iter = value);
        assign_check("temperature", &mut |value| state.params.temperature = value);
        assign_check("halfHeatSource", &mut |value| state.params.half_heat_source = value);
        assign_state("heatExchangeRate", &mut |value| state.params.heat_exchange_rate = value);
        assign_state("heatBuoyancy", &mut |value| state.params.heat_buoyancy = value);
        assign_check("mouseFlow", &mut |value| state.params.mouse_flow = value);
        assign_check("showVelocity", &mut |value| state.params.show_velocity = value);
        assign_check("showVelocityField", &mut |value| state.params.show_velocity_field = value);
        assign_check("obstacle", &mut |value| state.params.obstacle = value);
        assign_check("dyeFromObstacle", &mut |value| state.params.dye_from_obstacle = value);
        assign_check("particles", &mut |value| state.params.particles = value);
        assign_usize("particleTrails", &mut |value| state.params.particle_trails = value);
        assign_boundary("boundaryX", &mut |value| state.params.boundary_x = value)?;
        assign_boundary("boundaryY", &mut |value| state.params.boundary_y = value)?;
        if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from("mousePos")) {
            for i in 0..2 {
                if let Ok(the_val) = js_sys::Reflect::get_u32(&new_val, i) {
                    if let Some(value) = the_val.as_f64() {
                        // console_log!("received mouse_pos[{}] value: {}", i, value);
                        state.params.mouse_pos[i as usize] = value as i32;
                    }
                }
            }
        }
        if let Ok(val) = js_sys::Reflect::get(&callback_ret, &JsValue::from("resetParticles")) {
            if let Some(true) = val.as_bool() {
                state.reset_particles();
            }
        }

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(func.borrow().as_ref().unwrap());

        Ok(())
    }) as Box<dyn FnMut() -> Result<(), JsValue>>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}

fn compile_shader(context: &GL, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
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

fn link_program(
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

pub fn vertex_buffer_data(context: &GL, vertices: &[f32]) {
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


pub fn vertex_buffer_sub_data(context: &GL, vertices: &[f32]) {
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

pub fn enable_buffer(gl: &GL, buffer: &Option<WebGlBuffer>, elements: i32, vertex_position: u32) {
    gl.bind_buffer(GL::ARRAY_BUFFER, buffer.as_ref());
    gl.vertex_attrib_pointer_with_i32(vertex_position, elements, GL::FLOAT, false, 0, 0);
    gl.enable_vertex_attrib_array(vertex_position);
}
