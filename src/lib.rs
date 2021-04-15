extern crate console_error_panic_hook;
extern crate libm;
use std::panic;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    unsafe fn log(s: &str);
}

macro_rules! console_log {
    ($fmt:expr, $($arg1:expr),*) => {
        log(&format!($fmt, $($arg1),+))
    };
    ($fmt:expr) => {
        log($fmt)
    }
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
            x[shape.idx(0  , j)] = if b == 1 { f } else { x[shape.idx(1  , j)] };
            x[shape.idx(shape.0-1, j)] = x[shape.idx(shape.0-2, j)];
        }
    }

    let (center, radius) = obstacle_position(&shape);

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

type Particle = (f64, f64);

fn new_particles(xor128: &mut Xor128, shape: Shape) -> Vec<Particle> {
    (0..1000).map(|_| {
        ((xor128.nexti() as isize % shape.0) as f64, (xor128.nexti() as isize % shape.1) as f64)
    }).collect::<Vec<_>>()
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
    mouse_flow: bool,
    obstacle: bool,
    dye_from_obstacle: bool,
    boundary_y: BoundaryCondition,
    boundary_x: BoundaryCondition,
}

struct State {
    density: Vec<f64>,
    density2: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    vx0: Vec<f64>,
    vy0: Vec<f64>,
    work: Vec<f64>,
    work2: Vec<f64>,
    shape: Shape,
    params: Params,
    particles: Vec<(f64, f64)>,
    xor128: Xor128,
}

impl State {
    fn new(width: usize, height: usize) -> Self {
        let mut xor128 = Xor128::new(123);

        let params = Params{
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
            mouse_flow: true,
            obstacle: false,
            dye_from_obstacle: true,
            boundary_x: BoundaryCondition::Wrap,
            boundary_y: BoundaryCondition::Wrap,
        };

        let shape = (width as isize, height as isize);

        let particles = new_particles(&mut xor128, shape);

        Self {
            density: vec![0f64; width * height],
            density2: vec![0f64; width * height],
            vx: vec![0f64; width * height],
            vy: vec![0f64; width * height],
            vx0: vec![0f64; width * height],
            vy0: vec![0f64; width * height],
            work: vec![0f64; width * height],
            work2: vec![0f64; width * height],
            shape,
            params,
            particles,
            xor128,
        }
    }

    fn fluid_step(&mut self) {
        let visc     = self.params.visc;
        let diff     = self.params.diff;
        let dt       = self.params.delta_time;
        let diffuse_iter = self.params.diffuse_iter;
        let project_iter = self.params.project_iter;
        let shape = self.shape;

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
        for particle in &mut self.particles {
            let pvx = self.vx[self.shape.idx(particle.0 as isize, particle.1 as isize)];
            let pvy = self.vy[self.shape.idx(particle.0 as isize, particle.1 as isize)];
            let dtx = self.params.delta_time * (self.shape.0 - 2) as f64;
            let dty = self.params.delta_time * (self.shape.1 - 2) as f64;

            let (fwidth, fheight) = (self.shape.0 as f64, self.shape.1 as f64);
            particle.0 = (particle.0 + dtx * pvx + fwidth) % fwidth;
            particle.1 = (particle.1 + dty * pvy + fheight) % fheight;
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

    fn render(&mut self, data: &mut Vec<u8>) {
        const U_MAX: f64 = 1.0;
        const V_MAX: f64 = 1.0;
        const W_MAX: f64 = 0.01;

        self.calc_velo();
        let (u, v, w) = (&self.density, &self.density2, &self.work);
        let shape = &self.shape;
        for y in 0..self.shape.1 {
            for x in 0..self.shape.0 {
                data[shape.idx(x, y) * 4    ] = ((u[shape.idx(x, y)]) / U_MAX * 127.) as u8;
                data[shape.idx(x, y) * 4 + 1] = ((v[shape.idx(x, y)]) / V_MAX * 127.) as u8;
                data[shape.idx(x, y) * 4 + 2] = ((w[shape.idx(x, y)]) / W_MAX * 127.) as u8;
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

        for particle in &self.particles {
            let (x, y) = (particle.0 as isize, particle.1 as isize);
            data[shape.idx(x, y) * 4    ] = 255;
            data[shape.idx(x, y) * 4 + 1] = 255;
            data[shape.idx(x, y) * 4 + 2] = 255;
            data[shape.idx(x, y) * 4 + 3] = 255;
        }
    }

    fn reset_particles(&mut self) {
        // Technically, this is a new allocation that we would want to avoid, but it happens only
        // when the user presses reset button.
        self.particles = new_particles(&mut self.xor128, self.shape);
    }
}

#[wasm_bindgen]
pub fn turing(width: usize, height: usize, callback: js_sys::Function) -> Result<(), JsValue> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    let mut data = vec![0u8; 4 * width * height];

    let mut state = State::new(width, height);

    state.render(&mut data);

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

        let average = state.density.iter().fold(0., |acc, v| acc + v);

        // console_log!("frame {}, density sum {:.5e}, cen: {:.5e} maxvelo: {:.5e} mouse {:?}",
        //     i, average,
        //     state.density[ix(mouse_pos[0], mouse_pos[1])], velo.iter().fold(0., |acc: f64, v| acc.max(*v)),
        //     mouse_pos);

        if state.params.mouse_flow {
            let density_phase = 0.5 * (i as f64 * 0.02352 * std::f64::consts::PI).cos() + 0.5;
            add_density(&mut state.density, mouse_pos[0] as isize, mouse_pos[1] as isize,
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

        state.render(&mut data);

        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&mut data), width as u32, height as u32).unwrap();

        let callback_ret = callback.call2(&window(), &JsValue::from(image_data),
            &JsValue::from(average)).unwrap_or(JsValue::from(true));
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
        assign_check("mouseFlow", &mut |value| state.params.mouse_flow = value);
        assign_check("obstacle", &mut |value| state.params.obstacle = value);
        assign_check("dyeFromObstacle", &mut |value| state.params.dye_from_obstacle = value);
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
