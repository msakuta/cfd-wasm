extern crate image;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;
extern crate console_error_panic_hook;
extern crate libm;
use std::panic;
use std::cmp;

use std::collections::HashMap;
use std::sync::Arc;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;
use render::{RenderColor,
    UVMap,
    RenderMaterial, RenderPattern,
    RenderObject, RenderSphere, RenderFloor,
    RenderEnv,
    render, hermite_interpolate};
use vec3::Vec3;
use quat::Quat;
use rand::prelude::*;

mod render;
mod vec3;
mod quat;
mod modutil;
mod pixelutil;

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

#[wasm_bindgen]
pub fn helloworld() -> String {
    String::from("Hello world from Rust!")
}

fn bgcolor(ren: &RenderEnv, direction: &Vec3) -> RenderColor{
    use std::f32::consts::PI;
    let phi = direction.z.atan2(direction.x);
    let the = direction.y.asin();
    let d = (50. * PI + phi * 10. * PI) % (2. * PI) - PI;
    let dd = (50. * PI + the * 10. * PI) % (2. * PI) - PI;
    let ret = RenderColor::new(
        0.5 / (15. * (d * d * dd * dd) + 1.),
        0.25 - direction.y / 4.,
        0.25 - direction.y / 4.,
    );
    let dot = ren.light.dot(direction);

    if dot > 0.9 {
        if 0.9995 < dot {
            RenderColor::new(2., 2., 2.)
        }
        else {
            let ret2 = if 0.995 < dot {
                let dd = (dot - 0.995) * 150.;
                RenderColor::new(ret.r + dd, ret.g + dd, ret.b + dd)
            } else { ret };
            let dot2 = dot - 0.9;
            RenderColor::new(ret2.r + dot2 * 5., ret2.g + dot2 * 5., ret2.b)
        }
    }
    else {
        ret
    }
    // else PointMandel(dir->x * 2., dir->z * 2., 32, ret);
}

#[wasm_bindgen]
pub fn render_func(context: &CanvasRenderingContext2d, width: usize, height: usize, pos: Vec<f32>, pyr: Vec<f32>) -> Result<(), JsValue> {
    let xmax = width;
    let ymax = height;
    let xfov = 1.;
    let yfov = ymax as f32 / xmax as f32;
    let thread_count = 1;
    use std::f32::consts::PI;

    let mut materials: HashMap<String, Arc<RenderMaterial>> = HashMap::new();

    let floor_material = Arc::new(RenderMaterial::new("floor".to_string(),
        RenderColor::new(1.0, 1.0, 0.0), RenderColor::new(0.0, 0.0, 0.0),  0, 0., 0.0)
        .pattern(RenderPattern::RepeatedGradation)
        .pattern_scale(300.)
        .pattern_angle_scale(0.2)
        .texture_ok("bar.png"));
    materials.insert("floor".to_string(), floor_material);

    let mirror_material = Arc::new(RenderMaterial::new("mirror".to_string(),
        RenderColor::new(0.0, 0.0, 0.0), RenderColor::new(1.0, 1.0, 1.0), 24, 0., 0.0)
        .frac(RenderColor::new(1., 1., 1.)));

    let red_material = Arc::new(RenderMaterial::new("red".to_string(),
        RenderColor::new(0.8, 0.0, 0.0), RenderColor::new(0.0, 0.0, 0.0), 24, 0., 0.0)
        .glow_dist(5.));

    let transparent_material = Arc::new(RenderMaterial::new("transparent".to_string(),
        RenderColor::new(0.0, 0.0, 0.0), RenderColor::new(0.0, 0.0, 0.0),  0, 1., 1.5)
        .frac(RenderColor::new(1.49998, 1.49999, 1.5)));


    let objects: Vec<RenderObject> = vec!{
        /* Plane */
            RenderObject::Floor(
                RenderFloor::new_raw(materials.get("floor").unwrap().clone(),       Vec3::new(  0.0, -300.0,  0.0),  Vec3::new(0., 1., 0.))
                .uvmap(UVMap::ZX),
            ),
            // RenderFloor::new (floor_material,       Vec3::new(-300.0,   0.0,  0.0),  Vec3::new(1., 0., 0.)),
        /* Spheres */
            RenderSphere::new(mirror_material.clone(), 80.0, Vec3::new(   0.0, -30.0,172.0)),
            RenderSphere::new(mirror_material, 80.0, Vec3::new(   -200.0, -30.0,172.0)),
            RenderSphere::new(red_material, 80.0, Vec3::new(-200.0,-200.0,172.0)),
        // /*	{80.0F,  70.0F,-200.0F,150.0F, 0.0F, 0.0F, 0.8F, 0.0F, 0.0F, 0.0F, 0.0F,24, 1., 1., {1.}},*/
            RenderSphere::new(transparent_material, 100.0, Vec3::new(  70.0,-200.0,150.0)),
        /*	{000.F, 0.F, 0.F, 1500.F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,24, 0, 0},*/
        /*	{100.F, -70.F, -150.F, 160.F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,24, .5F, .2F},*/
        };

    let mut data = vec![0u8; 4 * width * height];

    for y in 0..height {
        for x in 0..width {
            data[(x + y * width) * 4    ] = ((x) * 255 / width) as u8;
            data[(x + y * width) * 4 + 1] = ((y) * 255 / height) as u8;
            data[(x + y * width) * 4 + 2] = ((x + y) % 32 + 32) as u8;
            data[(x + y * width) * 4 + 3] = 255;
        }
    }

    let mut putpoint = |x: i32, y: i32, fc: &RenderColor| {
        data[(x as usize + y as usize * width) * 4    ] = (fc.r * 255.).min(255.) as u8;
        data[(x as usize + y as usize * width) * 4 + 1] = (fc.g * 255.).min(255.) as u8;
        data[(x as usize + y as usize * width) * 4 + 2] = (fc.b * 255.).min(255.) as u8;
    };

    let mut ren: RenderEnv = RenderEnv::new(
        if 3 == pos.len() {
            Vec3::new(pos[0], pos[1], pos[2])
        }
        else {
            Vec3::new(0., -150., -300.)
        }, /* cam */
        if 3 == pyr.len() {
            Vec3::new(pyr[0], pyr[1], pyr[2])
        }
        else {
            Vec3::new(0., -PI / 2., -PI / 2.)
        }, /* pyr */
        xmax as i32,
        ymax as i32, /* xres, yres */
        xfov,
        yfov, /* xfov, yfov*/
        //pointproc: putpoint, /* pointproc */
        bgcolor, /* bgproc */
    )
    .materials(materials)
    .objects(objects)
    .light(Vec3::new(50., 60., -50.))
    .use_raymarching(false)
    .glow_effect(None);
    log(&format!("pyr: {}, {}, {}", ren.camera.pyr.x, ren.camera.pyr.y, ren.camera.pyr.z));

    render(&ren, &mut putpoint, thread_count);

    log(&format!("data: {}, {}", data[0], data.len()));

    let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&mut data), width as u32, height as u32)?;
    context.put_image_data(&image_data, 0., 0.);

    Ok(())
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

#[wasm_bindgen]
pub fn deserialize_string(save_data: &str, width: usize, height: usize, callback: js_sys::Function) -> Result<(), JsValue>{
    let xmax = width;
    let ymax = height;
    let xfov = 1.;
    let yfov = ymax as f32 / xmax as f32;
    use std::f32::consts::PI;

    let mut ren: RenderEnv = RenderEnv::new(
        Vec3::new(0., -150., -300.), /* cam */
        Vec3::new(0., -PI / 2., -PI / 2.), /* pyr */
        xmax as i32,
        ymax as i32, /* xres, yres */
        xfov,
        yfov, /* xfov, yfov*/
        bgcolor, /* bgproc */
    );
    ren.deserialize(&save_data).map_err(|e| JsValue::from(
        "Deserialize error: ".to_string() + &e.s))?;
    log(&format!("deserialized materials: {}, objects: {}, camera_motion: {}", ren.materials.len(), ren.objects.len(), ren.camera_motion.0.len()));

    let mut data = vec![0u8; 4 * width * height];

    for y in 0..height {
        for x in 0..width {
            data[(x + y * width) * 4    ] = ((x) * 255 / width) as u8;
            data[(x + y * width) * 4 + 1] = ((y) * 255 / height) as u8;
            data[(x + y * width) * 4 + 2] = ((x + y) % 32 + 32) as u8;
            data[(x + y * width) * 4 + 3] = 255;
        }
    }

    let func = Rc::new(RefCell::new(None));
    let g = func.clone();

    let mut prev_camera = ren.camera;
    let mut prev_velocity = Vec3::zero();
    let total_frames = ren.camera_motion.0.iter().fold(0., |acc, m| acc + m.duration);
    let mut accum_frame = 0;
    let frame_step = 0.5;
    let mut i = 0.;
    let mut frame_num = 0;
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        i += frame_step;
        while i >= ren.camera_motion.0[frame_num].duration {
            prev_camera = ren.camera_motion.0[frame_num].camera;
            prev_velocity = ren.camera_motion.0[frame_num].velocity;
            i -= ren.camera_motion.0[frame_num].duration;
            frame_num = (frame_num + 1) % ren.camera_motion.0.len();
            console_log!("keyframe switched: {}, i became {}", frame_num, i);
        }
        let frame = &ren.camera_motion.0[frame_num];

        let v0 = prev_velocity;
        let v1 = frame.velocity;
        console_log!("keyframe {} / {}, v0: {},{},{}", frame_num, ren.camera_motion.0.len(), v0.x, v0.y, v0.z);
        if i >= frame.duration {
            // Drop our handle to this closure so that it will get cleaned
            // up once we return.
            let _ = func.borrow_mut().take();
            return
        }
        let f = i as f32 / frame.duration;
        console_log!("Rendering frame {} / {}, v0: {},{}", accum_frame, total_frames, v0.x, v0.y);

        let text = format!("time: {}, frame: {}", i, frame_num);
        document().get_element_by_id("label").unwrap().set_inner_html(&text);

        let mut putpoint = |x: i32, y: i32, fc: &RenderColor| {
            data[(x as usize + y as usize * width) * 4    ] = (fc.r * 255.).min(255.) as u8;
            data[(x as usize + y as usize * width) * 4 + 1] = (fc.g * 255.).min(255.) as u8;
            data[(x as usize + y as usize * width) * 4 + 2] = (fc.b * 255.).min(255.) as u8;
        };

        ren.camera.position = hermite_interpolate(f, &prev_camera.position, &frame.camera.position,
            &v0, &v1);
        ren.camera.rotation = if let Some(target) = frame.camera_target() {
            let delta = target - ren.camera.position;
            let pitch = (delta.y).atan2((delta.x * delta.x + delta.z * delta.z).sqrt());
            let yaw = -delta.z.atan2(delta.x);
            Quat::rotation(yaw, 0., 1., 0.)
            * Quat::rotation(pitch, 0., 0., 1.)
            * Quat::rotation(-std::f32::consts::PI / 2., 1., 0., 0.)
        }
        else{
            prev_camera.rotation.slerp(&frame.camera.rotation, f)
        };

        render(&ren, &mut putpoint, 1);

        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&mut data), width as u32, height as u32).unwrap();

        let terminate_requested = callback.call1(&window(), &JsValue::from(image_data)).unwrap_or(JsValue::from(true));
        if terminate_requested.is_truthy() {
            return
        }

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(func.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
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

    const factor: f64 = 1.;
    const epsilon: f64 = 1.;
    const Au: f64 = 1.2;
    const Bu: f64 = 0.5;
    const Cu: f64 = 0. * 0.04;
    const Du: f64 = 0.056 * factor / epsilon / epsilon;
    const Av: f64 = 0. * -0.1;
    const Bv: f64 = 0.08;
    const Cv: f64 = 0. * 0.15;
    const Dv: f64 = 0.09 * factor / epsilon / epsilon;
    const F: f64 = 0.03;
    const k: f64 = 0.056;
    const skip: usize = 1;

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

    fn react_u(width: usize, height: usize, u_next: &mut Vec<f64>, u: &Vec<f64>, v: &Vec<f64>, a: f64, b: f64, c: f64){
        for y in 1..height-1 {
            for x in 1..width-1 {
                let u_p = u[x + y * width];
                let v_p = v[x + y * width];
                u_next[x + y * width] += factor * (u_p * u_p * v_p - (F + k) * u_p);
            }
        }
    }

    fn react_v(width: usize, height: usize, v_next: &mut Vec<f64>, u: &Vec<f64>, v: &Vec<f64>, a: f64, b: f64, c: f64){
        for y in 1..height-1 {
            for x in 1..width-1 {
                let u_p = u[x + y * width];
                let v_p = v[x + y * width];
                v_next[x + y * width] += factor * (-u_p * u_p * v_p + F * (1. - v_p));
            }
        }
    }

    fn clip(width: usize, height: usize, u: &mut Vec<f64>, max: f64, min: f64){
        for y in 1..height-1 {
            for x in 1..width-1 {
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
        console_log!("Rendering frame {}, mouse_pos: {}, {}", i, mouse_ref.0, mouse_ref.0);

        i += 1;

        for _ in 0..skip {
            let mut u_next = u.clone();
            let mut v_next = v.clone();
            diffuse(width, height, &mut u_next, &u, Du);
            diffuse(width, height, &mut v_next, &v, Dv);
            react_u(width, height, &mut u_next, &u, &v, Au, Bu, Cu);
            react_v(width, height, &mut v_next, &u, &v, Au, Bu, Cu);
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

        let terminate_requested = callback.call2(&window(), &JsValue::from(image_data),
            &JsValue::from(average)).unwrap_or(JsValue::from(true));
        if terminate_requested.is_truthy() {
            return
        }

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(func.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
