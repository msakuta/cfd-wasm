use std::panic;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, WebGlRenderingContext as GL};

use crate::{
    cfd::{add_density, add_velo, BoundaryCondition},
    console_log,
    params::Params,
    window,
    xor128::Xor128,
    Idx, State,
};

fn request_animation_frame(f: &Closure<dyn FnMut() -> Result<(), JsValue>>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

trait Renderer {
    fn is_webgl(&self) -> bool;
    fn start(&self, _state: &mut State) -> Result<(), JsValue> {
        Ok(())
    }
    fn render(&self, state: &State, data: &mut [u8]) -> Result<(), JsValue>;
}

struct CanvasRenderer {
    ctx: CanvasRenderingContext2d,
}

impl Renderer for CanvasRenderer {
    fn is_webgl(&self) -> bool {
        false
    }
    fn render(&self, state: &State, data: &mut [u8]) -> Result<(), JsValue> {
        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
            wasm_bindgen::Clamped(data),
            state.shape.0 as u32,
            state.shape.1 as u32,
        )
        .unwrap();
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
    fn is_webgl(&self) -> bool {
        true
    }
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
pub fn cfd_canvas(
    width: usize,
    height: usize,
    ctx: web_sys::CanvasRenderingContext2d,
    callback: js_sys::Function,
) -> Result<(), JsValue> {
    cfd_temp(width, height, CanvasRenderer { ctx }, callback)
}

#[wasm_bindgen]
pub fn cfd_webgl(
    width: usize,
    height: usize,
    gl: GL,
    callback: js_sys::Function,
) -> Result<(), JsValue> {
    cfd_temp(width, height, WebGLRenderer { gl }, callback)
}

fn cfd_temp(
    width: usize,
    height: usize,
    renderer: impl Renderer + 'static,
    callback: js_sys::Function,
) -> Result<(), JsValue> {
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
        let Params { mouse_pos, .. } = state.params;
        // console_log!("Rendering frame {}, mouse_pos: {}, {} delta_time: {}, skip_frames: {}, f: {}, k: {}, ru: {}, rv: {}",
        //     i, mouse_pos[0], mouse_pos[1], delta_time, state.params.skip_frames, f, k, state.params.ru, state.params.rv);

        i += 1;

        // let velo = state.calc_velo(&state.vx, &state.vy);
        // let mut div = vec![0f64; width * height];
        // divergence(&Vx, &Vy, (width, height), |(x, y), v| div[ix(x as i32, y as i32)] = v.abs());

        // let average = state.vx.iter().zip(state.vy.iter())
        //     .fold(0f64, |acc, v| acc.max((v.0 * v.0 + v.1 * v.1).sqrt()));

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
            add_density(
                density,
                mouse_pos[0] as isize,
                mouse_pos[1] as isize,
                density_phase * state.params.density,
                state.shape,
            );
            let density2_phase =
                0.5 * ((i as f64 * 0.02352 + 1.) * std::f64::consts::PI).cos() + 0.5;
            add_density(
                &mut state.density2,
                mouse_pos[0] as isize,
                mouse_pos[1] as isize,
                density2_phase * state.params.density,
                state.shape,
            );
            // let angle_rad = (i as f64 * 0.002 * std::f64::consts::PI) * 2. * std::f64::consts::PI;
            let mut hasher = Xor128::new((i / 16) as u32);
            let angle_rad =
                ((hasher.nexti() as f64 / 0xffffffffu32 as f64) * 2. * std::f64::consts::PI)
                    * 2.
                    * std::f64::consts::PI;
            add_velo(
                &mut state.vx,
                &mut state.vy,
                state
                    .shape
                    .idx(mouse_pos[0] as isize, mouse_pos[1] as isize),
                [
                    state.params.mouse_flow_speed * angle_rad.cos(),
                    state.params.mouse_flow_speed * angle_rad.sin(),
                ],
            );
        }

        for _ in 0..state.params.skip_frames {
            state.fluid_step();
            state.particle_step(renderer.is_webgl());
        }

        state.render_fluid(&mut data);

        renderer.render(&state, &mut data)?;

        let particle_vec = state.particle_position_array();
        let buf = js_sys::Float64Array::from(&particle_vec as &[f64]);

        let callback_ret = callback
            .call1(&window(), &JsValue::from(buf))
            .unwrap_or(JsValue::from(true));
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
            use BoundaryCondition::{Fixed, Flow, Wrap};

            if let (Ok(new_val), Ok(flow)) = (
                js_sys::Reflect::get(&callback_ret, &JsValue::from(name)),
                js_sys::Reflect::get(&callback_ret, &JsValue::from("boundaryFlowSpeed")),
            ) {
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
        assign_state("skipFrames", &mut |value| {
            state.params.skip_frames = value as u32
        });
        assign_state("visc", &mut |value| state.params.visc = value);
        assign_state("diff", &mut |value| state.params.diff = value);
        assign_state("density", &mut |value| state.params.density = value);
        assign_state("decay", &mut |value| state.params.decay = value);
        assign_state("mouseFlowSpeed", &mut |value| {
            state.params.mouse_flow_speed = value
        });
        assign_usize("diffIter", &mut |value| state.params.diffuse_iter = value);
        assign_usize("projIter", &mut |value| state.params.project_iter = value);
        assign_check("temperature", &mut |value| state.params.temperature = value);
        assign_check("halfHeatSource", &mut |value| {
            state.params.half_heat_source = value
        });
        assign_state("heatExchangeRate", &mut |value| {
            state.params.heat_exchange_rate = value
        });
        assign_state("heatBuoyancy", &mut |value| {
            state.params.heat_buoyancy = value
        });
        assign_check("mouseFlow", &mut |value| state.params.mouse_flow = value);
        assign_state("gamma", &mut |value| state.params.gamma = value as f32);
        assign_check("showVelocity", &mut |value| {
            state.params.show_velocity = value
        });
        assign_check("showVelocityField", &mut |value| {
            state.params.show_velocity_field = value
        });
        assign_check("obstacle", &mut |value| state.params.obstacle = value);
        assign_check("dyeFromObstacle", &mut |value| {
            state.params.dye_from_obstacle = value
        });
        assign_check("particles", &mut |value| state.params.particles = value);
        assign_usize("particleTrails", &mut |value| {
            state.params.particle_trails = value
        });
        assign_check("redistributeParticles", &mut |value| {
            state.params.redistribute_particles = value
        });
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
    })
        as Box<dyn FnMut() -> Result<(), JsValue>>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
