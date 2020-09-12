extern crate image;
#[macro_use]
extern crate console_error_panic_hook;
extern crate libm;
use std::panic;
use std::cmp;

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

fn document() -> web_sys::Document {
    window()
        .document()
        .expect("should have a document on window")
}

fn body() -> web_sys::HtmlElement {
    document().body().expect("document should have a body")
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

    let mut render = |height, width, data: &mut Vec<u8>, u: &mut Vec<f64>, v: &mut Vec<f64>| {
        for y in 0..height {
            for x in 0..width {
                data[(x + y * width) * 4    ] = ((u[x + y * width] + uMax) / 2. / uMax * 127.) as u8;
                data[(x + y * width) * 4 + 1] = ((v[x + y * width] + uMax) / 2. / vMax * 127.) as u8;
                data[(x + y * width) * 4 + 2] = 0;
                data[(x + y * width) * 4 + 3] = 255;
            }
        }
    };

    #[derive(Copy, Clone)]
    struct Params{
        delta_time: f64,
        skip_frames: u32,
        f: f64,
        k: f64,
        ru: f64,
        rv: f64,
    }

    let mut params = Rc::new(RefCell::new(Params{
        delta_time: 1.,
        skip_frames: 1,
        f: 0.03,
        k: 0.056,
        ru: 0.07,
        rv: 0.056,
    }));

    fn diffuse(width: usize, height: usize, u_next: &mut Vec<f64>, u: &Vec<f64>, d: f64){
        let [iwidth, iheight] = [width as isize, height as isize];
        for y in 0..iheight {
            for x in 0..iwidth {
                u_next[x as usize + y as usize * width] += {
                    let fu = |x, y| {
                        u[((x + iwidth) % iwidth + (y + iheight) % iheight * iwidth) as usize]
                    };
                    d * (fu(x + 1, y) + fu(x - 1, y)
                        + fu(x, y - 1) + fu(x, y + 1) - 4. * fu(x, y))
                };
            }
        }
    }

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

    fn clip(width: usize, height: usize, u: &mut Vec<f64>, max: f64, min: f64){
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

    let mouse_pos = Rc::new(RefCell::new((0, 0)));

    let canvas = document().get_element_by_id("canvas").unwrap().dyn_into::<web_sys::HtmlCanvasElement>()?;
    {
        let closure = Closure::wrap(Box::new((|mut mouse_pos: Rc<RefCell<(i32, i32)>>| move |event: web_sys::MouseEvent| {
            if let Ok(mut mut_mouse) = mouse_pos.try_borrow_mut() {
                *mut_mouse = (event.offset_x(), event.offset_y());
                console_log!("mousemove detected: {} {}", mut_mouse.0, mut_mouse.1);
            }
            else{
                console_log!("mousemove detected but failed to update");
            }
        })(mouse_pos.clone())) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    };

    console_log!("Starting frames");

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mouse_ref = *(*mouse_pos).borrow();
        let params_val = *(*params).borrow();
        let Params{delta_time, f, k, ..} = params_val;
        console_log!("Rendering frame {}, mouse_pos: {}, {} delta_time: {}, skip_frames: {}, f: {}, k: {}, ru: {}, rv: {}",
            i, mouse_ref.0, mouse_ref.0, delta_time, params_val.skip_frames, f, k, params_val.ru, params_val.rv);

        i += 1;

        for _ in 0..params_val.skip_frames {
            let mut u_next = u.clone();
            let mut v_next = v.clone();
            diffuse(width, height, &mut u_next, &u, &params_val.rv * delta_time);
            diffuse(width, height, &mut v_next, &v, &params_val.ru * delta_time);
            react_u(width, height, &mut u_next, &u, &v, &params_val);
            react_v(width, height, &mut v_next, &u, &v, &params_val);
            clip(width, height, &mut u, uMax, 0.);
            clip(width, height, &mut v, vMax, 0.);
            u = u_next;
            v = v_next;
        }

        let (xx, yy) = mouse_ref;
        for x in xx-1..xx+2 {
            for y in yy-1..yy+2 {
                if 0 <= x && x < width as i32 && 0 <= y && y < height as i32 {
                    u[x as usize + y as usize * width] = uMax;
                }
            }
        }

        let average = {
            let mut accum = 0.;
            for y in 1..height-1 {
                for x in 1..width-1 {
                    accum += u[x + y * width];
                }
            }
            accum / (width * height) as f64
        };

        render(height, width, &mut data, &mut u, &mut v);

        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&mut data), width as u32, height as u32).unwrap();

        let callback_ret = callback.call2(&window(), &JsValue::from(image_data),
            &JsValue::from(average)).unwrap_or(JsValue::from(true));
        let terminate_requested = js_sys::Reflect::get(&callback_ret, &JsValue::from("terminate"))
            .unwrap_or_else(|_| JsValue::from(true));
        if terminate_requested.is_truthy() {
            return
        }
        let assign_state = |name: &str, setter: fn(params: &mut Params, val: f64)| {
            if let Ok(new_val) = js_sys::Reflect::get(&callback_ret, &JsValue::from(name)) {
                if let Some(the_val) = new_val.as_f64() {
                    console_log!("received {} value: {}", name, the_val);
                    let mut params_ref = (*params).borrow_mut();
                    setter(&mut params_ref, the_val);
                }
            }
        };
        assign_state("deltaTime", |params, value| params.delta_time = value);
        assign_state("skipFrames", |params, value| params.skip_frames = value as u32);
        assign_state("f", |params, value| params.f = value);
        assign_state("k", |params, value| params.k = value);
        assign_state("ru", |params, value| params.ru = value);
        assign_state("rv", |params, value| params.rv = value);

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(func.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
