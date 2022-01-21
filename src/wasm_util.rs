use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub(crate) unsafe fn log(s: &str);
}

macro_rules! console_log {
    ($fmt:expr, $($arg1:expr),*) => {
        crate::wasm_util::log(&format!($fmt, $($arg1),+))
    };
    ($fmt:expr) => {
        crate::wasm_util::log($fmt)
    }
}

pub(crate) use console_log;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = ANGLEInstancedArrays)]
    pub(crate) type AngleInstancedArrays;

    #[wasm_bindgen(method, getter, js_name = VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE)]
    pub(crate) fn vertex_attrib_array_divisor_angle(this: &AngleInstancedArrays) -> i32;

    #[wasm_bindgen(method, catch, js_name = drawArraysInstancedANGLE)]
    pub(crate) fn draw_arrays_instanced_angle(
        this: &AngleInstancedArrays,
        mode: u32,
        first: i32,
        count: i32,
        primcount: i32,
    ) -> Result<(), JsValue>;

    // TODO offset should be i64
    #[wasm_bindgen(method, catch, js_name = drawElementsInstancedANGLE)]
    pub(crate) fn draw_elements_instanced_angle(
        this: &AngleInstancedArrays,
        mode: u32,
        count: i32,
        type_: u32,
        offset: i32,
        primcount: i32,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen(method, js_name = vertexAttribDivisorANGLE)]
    pub(crate) fn vertex_attrib_divisor_angle(
        this: &AngleInstancedArrays,
        index: u32,
        divisor: u32,
    );
}

pub(crate) fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}
