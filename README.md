[![Rust-wasm](https://github.com/msakuta/cfd-wasm/actions/workflows/rust-wasm.yml/badge.svg)](https://github.com/msakuta/cfd-wasm/actions/workflows/rust-wasm.yml)

# cfd-wasm

Computational Fluid Dynamics simulation with Webassembly and Rust.

Try it now on your browser!

https://msakuta.github.io/cfd-wasm/

## Screenshots

![screenshot](images/screenshot00.jpg)

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

You can move your mouse cursor on the simulation to disturb the field.
A dye of red and green color will emit from the mouse cursor in random
directions.

You can control the parameters with sliders on the bottom half of the screen.

"Time step" and "Skip frames" both contributes to the speed of the simulation.

### Gauss-Seidel method parameters

The accuracy of the simulation is determined by 2 parameters:

* Gauss-Seidel iter for diffusion (default=4)
* Gauss-Seidel iter for projection (default=20)

Increasing these values will improve accuracy, but it would take more computation per step.
Personally, I feel Ok with low value with diffusion, but projection is better to keep it high
for nice swirly fluid behavior.

## A bit of History

This project is a fork of [rd-system-wasm](https://github.com/msakuta/rd-system-wasm) because of obvious similarity
in simulation model.

However, the solver is very different. This project uses more stable solver (see the references),
so it wouldn't diverge even if you put extreme parameters.

## The tutorial that I used

https://aralroca.com/blog/first-steps-webassembly-rust


## Reference

https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

https://youtu.be/qsYE1wMEMPA
