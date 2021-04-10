[![Rust-wasm](https://github.com/msakuta/cfd-wasm/actions/workflows/rust-wasm.yml/badge.svg)](https://github.com/msakuta/cfd-wasm/actions/workflows/rust-wasm.yml)

# cfd-wasm

Computational Fluid Dynamics simulation with Webassembly and Rust.

Try it now with your browser!

https://msakuta.github.io/cfd-wasm/

## Screencasts

![screencast](https://msakuta.github.io/images/showcase/rd-system-wasm.gif)

## How to build and run

Install

* Cargo >1.40
* npm

Install wasm-pack command line tool with

    cargo install wasm-pack

Build the project

    wasm-pack build --target web

Serve the web server

    npx serve .

Browse http://localhost:5000/

## Controls

You can move your mouse cursor on the simulation to disturb the field. It may create a new pattern, depending on the parameters.

You can control the parameters with sliders on the bottom half of the screen.

"Time step" and "Skip frames" both contributes to the speed of the simulation.

## Parameter presets

Reaction-diffusion system is very sensitive to parameters. Interesting patterns show up only in very certain combination of parameters.
So there are buttons to set the parameters to one of these interesting sets.

## The tutorial that I used

https://aralroca.com/blog/first-steps-webassembly-rust


## Reference

https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
