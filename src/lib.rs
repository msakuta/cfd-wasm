mod assets;
mod gl_util;
mod params;
mod shader_bundle;
mod shape;
mod state;
mod wasm_util;
mod xor128;

pub use state::{cfd_canvas, cfd_webgl};

fn ceil_pow2(i: isize) -> isize {
    let mut bit = 0;
    while (1 << bit) < i {
        bit += 1;
    }
    1 << bit
}
