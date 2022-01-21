mod assets;
mod bit_util;
mod gl_util;
mod marching_squares;
mod params;
mod shader_bundle;
mod shape;
mod state;
mod wasm_util;
mod xor128;

/// Published entry points for JS runtime
/// They don't really have to be 'pub'-ed here, since wasm-bindgen macro does all the job,
/// but rust-analyzer is happier with this.
pub use state::{cfd_canvas, cfd_webgl};
