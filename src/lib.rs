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

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
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

#[allow(non_upper_case_globals)]
#[wasm_bindgen]
pub fn turing(width: usize, height: usize, callback: js_sys::Function) -> Result<(), JsValue> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let mut u = vec![0f64; 4 * width * height];
    let mut v = vec![0f64; 4 * width * height];
    let mut s = vec![0f64; 4 * width * height];
    let mut density = vec![0f64; 4 * width * height];

    let mut Vx = vec![0f64; 4 * width * height];
    let mut Vy = vec![0f64; 4 * width * height];

    let mut Vx0 = vec![0f64; 4 * width * height];
    let mut Vy0 = vec![0f64; 4 * width * height];
    let mut data = vec![0u8; 4 * width * height];

    let mut xor128 = Xor128::new(123);

    const uMax: f64 = 1.0;
    const vMax: f64 = 1.0;

    for y in 0..height {
        for x in 0..width {
            u[x + y * width] = (xor128.nexti() as f64 / 0xffffffffu32 as f64 - 0.5) * uMax;
            if (x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2) < 100 {
                u[x + y * width] = uMax;
            }
            // v[x + y * width] = (xor128.nexti() as f64 / 0xffffffffu32 as f64 - 0.5) * vMax;
                // + libm::sin(x as f64 * 0.05 * libm::acos(0.)) * uMax;
            // v[x + y * width] = libm::sin((x as f64 * 0.05 + 0.5) * libm::acos(0.)) * 0.5 * uMax;
        }
    }

    let render = |height, width, data: &mut Vec<u8>, u: &[f64], v: &[f64]| {
        for y in 0..height {
            for x in 0..width {
                data[(x + y * width) * 4    ] = ((u[x + y * width] % uMax) / uMax * 127.) as u8;
                data[(x + y * width) * 4 + 1] = ((v[x + y * width]) / vMax * 127.) as u8;
                data[(x + y * width) * 4 + 2] = 0;
                data[(x + y * width) * 4 + 3] = 255;
            }
        }
    };

    #[derive(Copy, Clone)]
    struct Params{
        delta_time: f64,
        skip_frames: u32,
        mouse_pos: [i32; 2],
        f: f64,
        k: f64,
        ru: f64,
        rv: f64,
    }

    let params = //Rc::new(RefCell::new(
        Params{
        delta_time: 1.,
        skip_frames: 1,
        mouse_pos: [0, 0],
        f: 0.03,
        k: 0.056,
        ru: 0.07,
        rv: 0.056,
    };


    // fn diffuse(width: usize, height: usize, u_next: &mut Vec<f64>, u: &Vec<f64>, d: f64){
    //     let [iwidth, iheight] = [width as isize, height as isize];
    //     for y in 0..iheight {
    //         for x in 0..iwidth {
    //             u_next[x as usize + y as usize * width] += {
    //                 let fu = |x, y| {
    //                     u[((x + iwidth) % iwidth + (y + iheight) % iheight * iwidth) as usize]
    //                 };
    //                 d * (fu(x + 1, y) + fu(x - 1, y)
    //                     + fu(x, y - 1) + fu(x, y + 1) - 4. * fu(x, y))
    //             };
    //         }
    //     }
    // }

    fn add_density(density: &mut [f64], x: usize, y: usize, amount: f64, width: usize, height: usize) {
        let ix = |x, y| {
            ((x + width) % width + (y + height) % height * width) as usize
        };
        density[ix(x, y)] += amount;
    }

    fn add_velo(vx: &mut [f64], vy: &mut [f64], index: usize, amount: [f64; 2]) {
        vx[index] += amount[0];
        vy[index] += amount[1];
    }

    let set_bnd = |b: i32, x: &mut [f64], width: usize, height: usize| {
        let [iwidth, iheight] = [width as isize, height as isize];
        let ix = |x, y| {
            ((x + width) % width + (y + height) % height * width) as usize
        };
        // Edge cases
        for i in 1..width - 1 {
            x[ix(i, 0  )] = if b == 2 { -x[ix(i, 1  )] } else { x[ix(i, 1  )] };
            x[ix(i, height-1)] = if b == 2 { -x[ix(i, height-2)] } else { x[ix(i, height-2)] };
        }
        for j in 1..height - 1 {
            x[ix(0  , j)] = if b == 1 { -x[ix(1  , j)] } else { x[ix(1  , j)] };
            x[ix(width-1, j)] = if b == 1 { -x[ix(width-2, j)] } else { x[ix(width-2, j)] };
        }

        // Corner cases (literally)
        // x[ix!(0, 0)]       = 0.33f * (x[ix!(1, 0)]
        //                             + x[ix!(0, 1)]);
        // x[ix!(0, height-1)]     = 0.33f * (x[ix!(1, height-1, 0)]
        //                             + x[ix!(0, N-2, 0)]
        //                             + x[ix!(0, N-1, 1)]);
        // x[ix!(0, 0)]     = 0.33f * (x[ix!(1, 0, N-1)]
        //                             + x[ix!(0, 1, N-1)]
        //                             + x[ix!(0, 0, N)]);
        // x[ix!(0, N-1, N-1)]   = 0.33f * (x[ix!(1, N-1, N-1)]
        //                             + x[ix!(0, N-2, N-1)]
        //                             + x[ix!(0, N-1, N-2)]);
        // x[ix!(N-1, 0, 0)]     = 0.33f * (x[ix!(N-2, 0, 0)]
        //                             + x[ix!(N-1, 1, 0)]
        //                             + x[ix!(N-1, 0, 1)]);
        // x[ix!(N-1, N-1, 0)]   = 0.33f * (x[ix!(N-2, N-1, 0)]
        //                             + x[ix!(N-1, N-2, 0)]
        //                             + x[ix!(N-1, N-1, 1)]);
        // x[ix!(N-1, 0, N-1)]   = 0.33f * (x[ix!(N-2, 0, N-1)]
        //                             + x[ix!(N-1, 1, N-1)]
        //                             + x[ix!(N-1, 0, N-2)]);
        // x[ix!(N-1, N-1, N-1)] = 0.33f * (x[ix!(N-2, N-1, N-1)]
        //                             + x[ix!(N-1, N-2, N-1)]
        //                             + x[ix!(N-1, N-1, N-2)]);
    };

    fn lin_solve(b: i32, x: &mut [f64], x0: &[f64], a: f64, c: f64, iter: usize, width: usize, height: usize) {
        let (iwidth, iheight) = (width as isize, height as isize);
        let ix = |x, y| {
            ((x + iwidth) % iwidth + (y + iheight) % iheight * iwidth) as usize
        };
        let c_recip = 1.0 / c;
        for _ in 0..iter {
            for j in 1..iheight - 1 {
                for i in 1..iwidth - 1 {
                    x[ix(i, j)] = (x0[ix(i, j)]
                            + a*(    x0[ix(i+1, j  )]
                                    +x0[ix(i-1, j  )]
                                    +x0[ix(i  , j+1)]
                                    +x0[ix(i  , j-1)]
                            )) * c_recip;
                }
            }
            // set_bnd(b, x, N);
        }
    }

    fn diffuse(b: i32, x: &mut [f64], x0: &[f64], diff: f64, dt: f64, iter: usize, width: usize, height: usize) {
        let a = dt * diff * (width - 2) as f64 * (height - 2) as f64;
        lin_solve(b, x, x0, a, 1. + 4. * a, iter, width, height);
    }

    fn advect(b: i32, d: &mut [f64], d0: &[f64], velocX: &[f64], velocY: &[f64], dt: f64, width: usize, height: usize) {
        let (iwidth, iheight) = (width as isize, height as isize);
        let ix = |x, y| {
            ((x + width) % width + (y + height) % height * width) as usize
        };
    
        let dtx = dt * (width - 2) as f64;
        let dty = dt * (height - 2) as f64;

        let fwidth = width as f64;
        let fheight = height as f64;

        for j in 1..height - 1 { 
            let jfloat = j as f64;
            for i in 1..width - 1 {
                let ifloat = i as f64;
                let mut x    = ifloat - dtx * velocX[ix(i, j)];
                let mut y    = jfloat - dty * velocY[ix(i, j)];
                
                if x < 0.5 { x = 0.5 }
                if x > fwidth + 0.5 { x = fwidth + 0.5 };
                let i0 = x.floor();
                let i1 = i0 + 1.0;
                if y < 0.5 { y = 0.5 }
                if y > fheight + 0.5 { y = fheight + 0.5 };
                let j0 = y.floor();
                let j1 = j0 + 1.0; 
                
                let s1 = x - i0; 
                let s0 = 1.0 - s1; 
                let t1 = y - j0; 
                let t0 = 1.0 - t1;
                
                let i0i = i0 as usize;
                let i1i = i1 as usize;
                let j0i = j0 as usize;
                let j1i = j1 as usize;
                
                d[ix(i, j)] = 
                    s0 * ( t0 * (d0[ix(i0i, j0i)])
                        +( t1 * (d0[ix(i0i, j1i)])))
                +s1 * ( t0 * (d0[ix(i1i, j0i)])
                        +( t1 * (d0[ix(i1i, j1i)])));
            }
        }
        // set_bnd(b, d, N);
    }

    struct State {
        s: Vec<f64>,
        density: Vec<f64>,
        width: usize,
        height: usize,
        params: Params,
    }

    impl State {


        fn fluid_step(&mut self, vx: &mut [f64], vy: &mut [f64]) {
            let visc     = 0.5;
            let diff     = 0.125;
            let dt       = self.params.delta_time;
            let mut vx0     = vx.to_vec();
            let mut vy0     = vy.to_vec();
            let s       = &mut self.s;
            let density = &mut self.density;
            
            diffuse(1, &mut vx0, vx, visc, dt, 1, self.width, self.height);
            diffuse(2, &mut vy0, vy, visc, dt, 1, self.width, self.height);
            
            // project(vx0, Vy0, Vz0, Vx, Vy, 4, N);
            
            advect(1, vx, &vx0, &vx0, &vy0, dt, self.width, self.height);
            advect(2, vy, &vy0, &vx0, &vy0, dt, self.width, self.height);
            
            // project(Vx, Vy, Vz, vx0, Vy0, 4, N);
            
            diffuse(0, s, density, diff, dt, 1, self.width, self.height);
            advect(0, density, s, vx, vy, dt, self.width, self.height);
        }
    }

    let mut state = State{
        s,
        density,
        width,
        height,
        params,
    };

    fn react_u(width: usize, height: usize, u_next: &mut Vec<f64>, u: &Vec<f64>, v: &Vec<f64>, params: &Params){
        for y in 0..height {
            for x in 0..width {
                let u_p = u[x + y * width];
                let v_p = v[x + y * width];
                u_next[x + y * width] += params.delta_time * (u_p * u_p * v_p - (params.f + params.k) * u_p);
            }
        }
    }

    fn react_v(width: usize, height: usize, v_next: &mut Vec<f64>, u: &Vec<f64>, v: &Vec<f64>, params: &Params){
        for y in 0..height {
            for x in 0..width {
                let u_p = u[x + y * width];
                let v_p = v[x + y * width];
                v_next[x + y * width] += params.delta_time * (-u_p * u_p * v_p + params.f * (1. - v_p));
            }
        }
    }

    fn _clip(width: usize, height: usize, u: &mut Vec<f64>, max: f64, min: f64){
        for y in 0..height {
            for x in 0..width {
                u[x + y * width] = libm::fmax(libm::fmin(u[x + y * width], max), min);
            }
        }
    }

    render(height, width, &mut data, &mut u, &mut v);

    let func = Rc::new(RefCell::new(None));
    let g = func.clone();

    let mut i = 0;

    console_log!("Starting frames");

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let Params{delta_time, f, k, mouse_pos, ..} = state.params;
        // console_log!("Rendering frame {}, mouse_pos: {}, {} delta_time: {}, skip_frames: {}, f: {}, k: {}, ru: {}, rv: {}",
        //     i, mouse_pos[0], mouse_pos[1], delta_time, state.params.skip_frames, f, k, state.params.ru, state.params.rv);

        i += 1;

        let (iwidth, iheight) = (width as isize, height as isize);
        let ix = |x, y| {
            ((x as isize + iwidth) % iwidth + (y as isize + iheight) % iheight * iwidth) as usize
        };

        let velo = Vx.iter().zip(Vy.iter()).map(|(x, y)| (x * x + y * y).sqrt()).collect::<Vec<_>>();

        console_log!("frame {}, density sum {:.5e}, cen: {:.5e} maxvelo: {:.5e} mouse {:?}",
            i, state.density.iter().fold(0., |acc, v| acc + v),
            state.density[ix(mouse_pos[0], mouse_pos[1])], velo.iter().fold(0., |acc: f64, v| acc.max(*v)), mouse_pos);

        add_density(&mut state.density, mouse_pos[0] as usize, mouse_pos[1] as usize, 100. * state.params.ru, state.width, state.height);
        add_velo(&mut Vx, &mut Vy, ix(mouse_pos[0], mouse_pos[1]), [1., 1.]);

        for _ in 0..state.params.skip_frames {
            state.fluid_step(&mut Vx, &mut Vy);
            // let mut u_next = u.clone();
            // let mut v_next = v.clone();
            // diffuse(width, height, &mut u_next, &u, &params_val.rv * delta_time);
            // diffuse(width, height, &mut v_next, &v, &params_val.ru * delta_time);
            // react_u(width, height, &mut u_next, &u, &v, &params_val);
            // react_v(width, height, &mut v_next, &u, &v, &params_val);
            // clip(width, height, &mut u, uMax, 0.);
            // clip(width, height, &mut v, vMax, 0.);
            // u = u_next;
            // v = v_next;
        }

        // let [xx, yy] = mouse_pos;
        // for x in xx-1..xx+2 {
        //     for y in yy-1..yy+2 {
        //         if 0 <= x && x < width as i32 && 0 <= y && y < height as i32 {
        //             u[x as usize + y as usize * width] = uMax;
        //         }
        //     }
        // }

        let average = {
            let mut accum = 0.;
            for y in 1..height-1 {
                for x in 1..width-1 {
                    accum += u[x + y * width];
                }
            }
            accum / (width * height) as f64
        };

        render(height, width, &mut data, &state.density, &velo);

        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&mut data), width as u32, height as u32).unwrap();

        let callback_ret = callback.call2(&window(), &JsValue::from(image_data),
            &JsValue::from(average)).unwrap_or(JsValue::from(true));
        let terminate_requested = js_sys::Reflect::get(&callback_ret, &JsValue::from("terminate"))
            .unwrap_or_else(|_| JsValue::from(true));
        if terminate_requested.is_truthy() {
            return
        }
        let mut assign_state = |name: &str, setter: fn(params: &mut Params, val: f64)| {
            if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from(name)) {
                if let Some(the_val) = new_val.as_f64() {
                    // console_log!("received {} value: {}", name, the_val);
                    // let mut params_ref = (*params).borrow_mut();
                    // setter(&mut params_ref, the_val);
                    setter(&mut state.params, the_val);
                }
            }
        };
        assign_state("deltaTime", |params, value| params.delta_time = value);
        assign_state("skipFrames", |params, value| params.skip_frames = value as u32);
        assign_state("f", |params, value| params.f = value);
        assign_state("k", |params, value| params.k = value);
        assign_state("ru", |params, value| params.ru = value);
        assign_state("rv", |params, value| params.rv = value);
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

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(func.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
