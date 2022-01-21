//! Computational Fluid Dynamics methods

use super::{Idx, Params, Shape, State};

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum BoundaryCondition {
    Wrap,
    Fixed,
    Flow(f64),
}

pub(crate) fn add_density(density: &mut [f64], x: isize, y: isize, amount: f64, shape: Shape) {
    density[shape.idx(x, y)] += amount;
}

pub(crate) fn add_velo(vx: &mut [f64], vy: &mut [f64], index: usize, amount: [f64; 2]) {
    vx[index] += amount[0];
    vy[index] += amount[1];
}

pub(crate) fn obstacle_position(shape: &Shape) -> ((isize, isize), isize) {
    ((shape.0 / 4, shape.1 / 2), shape.0 / 16)
}

fn set_bnd(b: i32, x: &mut [f64], shape: Shape, params: &Params) {
    let (center, radius) = obstacle_position(&shape);

    // Edge cases
    if params.boundary_y == BoundaryCondition::Fixed {
        for i in 1..shape.0 - 1 {
            x[shape.idx(i, 0)] = if b == 2 {
                -x[shape.idx(i, 1)]
            } else {
                x[shape.idx(i, 1)]
            };
            x[shape.idx(i, shape.1 - 1)] = if b == 2 {
                -x[shape.idx(i, shape.1 - 2)]
            } else {
                x[shape.idx(i, shape.1 - 2)]
            };
        }
    } else if let BoundaryCondition::Flow(f) = params.boundary_y {
        for i in 1..shape.0 - 1 {
            x[shape.idx(i, 0)] = if b == 2 { f } else { x[shape.idx(i, 1)] };
            x[shape.idx(i, shape.1 - 1)] = x[shape.idx(i, shape.1 - 2)];
        }
    }
    if params.boundary_x == BoundaryCondition::Fixed {
        for j in 1..shape.1 - 1 {
            x[shape.idx(0, j)] = if b == 1 {
                -x[shape.idx(1, j)]
            } else {
                x[shape.idx(1, j)]
            };
            x[shape.idx(shape.0 - 1, j)] = if b == 1 {
                -x[shape.idx(shape.0 - 2, j)]
            } else {
                x[shape.idx(shape.0 - 2, j)]
            };
        }
    } else if let BoundaryCondition::Flow(f) = params.boundary_x {
        for j in 1..shape.1 - 1 {
            x[shape.idx(0, j)] = if b == 1 {
                if j < center.1 {
                    f * 0.9
                } else {
                    f
                }
            } else {
                x[shape.idx(1, j)]
            };
            x[shape.idx(shape.0 - 1, j)] = x[shape.idx(shape.0 - 2, j)];
        }
    }

    if params.obstacle {
        for j in center.1 - radius..center.1 + radius {
            for i in center.0 - radius..center.0 + radius {
                let dist2 = (j - center.1) * (j - center.1) + (i - center.0) * (i - center.0);
                if dist2 < radius * radius {
                    x[shape.idx(i, j)] = 0.;
                }
            }
        }
    }

    // Corner cases (literally)
    x[shape.idx(0, 0)] = 0.5 * (x[shape.idx(1, 0)] + x[shape.idx(0, 1)]);
    x[shape.idx(0, shape.1 - 1)] =
        0.5 * (x[shape.idx(1, shape.1 - 1)] + x[shape.idx(0, shape.1 - 2)]);
    x[shape.idx(shape.0 - 1, 0)] =
        0.5 * (x[shape.idx(shape.0 - 2, 0)] + x[shape.idx(shape.0 - 1, 1)]);
    x[shape.idx(shape.0 - 1, shape.1 - 1)] =
        0.5 * (x[shape.idx(shape.0 - 2, shape.1 - 1)] + x[shape.idx(shape.0 - 1, shape.1 - 2)]);
}

fn get_range(shape: Shape, params: &Params) -> (isize, isize, isize, isize) {
    let (i0, i1) = match params.boundary_x {
        BoundaryCondition::Fixed => (1, shape.0 - 1),
        BoundaryCondition::Wrap => (0, shape.0),
        BoundaryCondition::Flow(_) => (1, shape.0),
    };
    let (j0, j1) = match params.boundary_y {
        BoundaryCondition::Fixed => (1, shape.1 - 1),
        BoundaryCondition::Wrap => (0, shape.1),
        BoundaryCondition::Flow(_) => (1, shape.1),
    };
    (i0, i1, j0, j1)
}

/// Solve linear system of equasions using Gauss-Seidel relaxation
///
/// @param b ignored
/// @param x Target field to be solved
fn lin_solve(
    b: i32,
    x: &mut [f64],
    x0: &[f64],
    a: f64,
    c: f64,
    iter: usize,
    shape: Shape,
    params: &Params,
) {
    let c_recip = 1.0 / c;
    let (ib, ie, jb, je) = get_range(shape, params);
    for _ in 0..iter {
        for j in jb..je {
            for i in ib..ie {
                x[shape.idx(i, j)] = (x0[shape.idx(i, j)]
                    + a * (x[shape.idx(i + 1, j)]
                        + x[shape.idx(i - 1, j)]
                        + x[shape.idx(i, j + 1)]
                        + x[shape.idx(i, j - 1)]))
                    * c_recip;
            }
        }
        set_bnd(b, x, shape, params);
    }
}

fn diffuse(
    b: i32,
    x: &mut [f64],
    x0: &[f64],
    diff: f64,
    dt: f64,
    iter: usize,
    shape: Shape,
    params: &Params,
) {
    let a = dt * diff * (shape.0 - 2) as f64 * (shape.1 - 2) as f64;
    lin_solve(b, x, x0, a, 1. + 4. * a, iter, shape, params);
}

fn advect(
    b: i32,
    d: &mut [f64],
    d0: &[f64],
    vx: &[f64],
    vy: &[f64],
    dt: f64,
    shape: Shape,
    params: &Params,
) {
    let dtx = dt * (shape.0 - 2) as f64;
    let dty = dt * (shape.1 - 2) as f64;
    let (ib, ie, jb, je) = get_range(shape, params);

    for j in jb..je {
        let jfloat = j as f64;
        for i in ib..ie {
            let ifloat = i as f64;
            let mut x = ifloat - dtx * vx[shape.idx(i, j)];
            if params.boundary_x == BoundaryCondition::Fixed {
                if x < 0.5 {
                    x = 0.5
                };
                if x > shape.0 as f64 + 0.5 {
                    x = shape.0 as f64 + 0.5
                };
            }
            let i0 = x.floor();
            let i1 = i0 + 1.0;
            let mut y = jfloat - dty * vy[shape.idx(i, j)];
            if params.boundary_y == BoundaryCondition::Fixed {
                if y < 0.5 {
                    y = 0.5
                };
                if y > shape.1 as f64 + 0.5 {
                    y = shape.1 as f64 + 0.5
                };
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

            d[shape.idx(i, j)] = s0
                * (t0 * (d0[shape.idx(i0i, j0i)]) + (t1 * (d0[shape.idx(i0i, j1i)])))
                + s1 * (t0 * (d0[shape.idx(i1i, j0i)]) + (t1 * (d0[shape.idx(i1i, j1i)])));
        }
    }
    set_bnd(b, d, shape, params);
}

fn project(
    vx: &mut [f64],
    vy: &mut [f64],
    p: &mut [f64],
    div: &mut [f64],
    iter: usize,
    shape: Shape,
    params: &Params,
) {
    let (ib, ie, jb, je) = get_range(shape, params);
    for j in jb..je {
        for i in ib..ie {
            div[shape.idx(i, j)] = -0.5
                * (vx[shape.idx(i + 1, j)] - vx[shape.idx(i - 1, j)] + vy[shape.idx(i, j + 1)]
                    - vy[shape.idx(i, j - 1)])
                / shape.0 as f64;
            p[shape.idx(i, j)] = 0.;
        }
    }
    set_bnd(0, div, shape, params);
    set_bnd(0, p, shape, params);
    lin_solve(0, p, div, 1., 4., iter, shape, params);

    for j in jb..je {
        for i in ib..ie {
            vx[shape.idx(i, j)] -=
                0.5 * (p[shape.idx(i + 1, j)] - p[shape.idx(i - 1, j)]) * shape.0 as f64;
            vy[shape.idx(i, j)] -=
                0.5 * (p[shape.idx(i, j + 1)] - p[shape.idx(i, j - 1)]) * shape.1 as f64;
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

impl State {
    pub(super) fn fluid_step(&mut self) {
        let visc = self.params.visc;
        let diff = self.params.diff;
        let dt = self.params.delta_time;
        let diffuse_iter = self.params.diffuse_iter;
        let project_iter = self.params.project_iter;
        let shape = self.shape;

        if self.params.temperature {
            if self.temperature.is_none() {
                self.temperature = Some(vec![0.5f64; (shape.0 * shape.1) as usize]);
            }
        }

        self.vx0.copy_from_slice(&self.vx);
        self.vy0.copy_from_slice(&self.vy);

        // console_log!("diffusion: {} viscousity: {}", diff, visc);

        diffuse(
            1,
            &mut self.vx0,
            &self.vx,
            visc,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );
        diffuse(
            2,
            &mut self.vy0,
            &self.vy,
            visc,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );

        // let (prev_div, prev_max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        project(
            &mut self.vx0,
            &mut self.vy0,
            &mut self.work,
            &mut self.work2,
            project_iter,
            shape,
            &self.params,
        );
        // let (after_div, max_div) = sum_divergence(&mut vx0, &mut vy0, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        advect(
            1,
            &mut self.vx,
            &self.vx0,
            &self.vx0,
            &self.vy0,
            dt,
            shape,
            &self.params,
        );
        advect(
            2,
            &mut self.vy,
            &self.vy0,
            &self.vx0,
            &self.vy0,
            dt,
            shape,
            &self.params,
        );

        if let (true, Some(temperature)) = (self.params.temperature, &mut self.temperature) {
            let buoyancy = self.params.heat_buoyancy;
            for i in 0..shape.0 {
                for j in 1..shape.1 - 1 {
                    self.vy[shape.idx(i, j)] += buoyancy
                        * (temperature[shape.idx(i, j + 1)]
                            + temperature[shape.idx(i, j - 1)]
                            + temperature[shape.idx(i + 1, j)]
                            + temperature[shape.idx(i - 1, j)]
                            - 4. * temperature[shape.idx(i, j)]);
                }
            }

            for i in 0..shape.0 {
                if !self.params.half_heat_source || i < shape.0 / 2 {
                    temperature[shape.idx(i, 1)] +=
                        (0. - temperature[shape.idx(i, 1)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 2)] +=
                        (0. - temperature[shape.idx(i, 2)]) * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, 3)] +=
                        (0. - temperature[shape.idx(i, 3)]) * self.params.heat_exchange_rate;
                }
                if !self.params.half_heat_source || shape.0 / 2 <= i {
                    temperature[shape.idx(i, shape.1 - 4)] += (1.
                        - temperature[shape.idx(i, shape.1 - 3)])
                        * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1 - 3)] += (1.
                        - temperature[shape.idx(i, shape.1 - 2)])
                        * self.params.heat_exchange_rate;
                    temperature[shape.idx(i, shape.1 - 2)] += (1.
                        - temperature[shape.idx(i, shape.1 - 1)])
                        * self.params.heat_exchange_rate;
                }
            }

            let mut work = std::mem::take(&mut self.work);
            work.copy_from_slice(temperature);
            diffuse(
                0,
                &mut work,
                temperature,
                diff,
                dt,
                diffuse_iter,
                shape,
                &self.params,
            );
            advect(
                0,
                temperature,
                &work,
                &self.vx0,
                &self.vy0,
                dt,
                shape,
                &self.params,
            );
            self.work = work;
        }

        // let (prev_div, prev_max_div) = sum_divergence(vx, vy, (self.width, self.height));
        project(
            &mut self.vx,
            &mut self.vy,
            &mut self.work,
            &mut self.work2,
            project_iter,
            shape,
            &self.params,
        );
        // let (after_div, max_div) = sum_divergence(vx, vy, (self.width, self.height));
        // console_log!("prev_div: {:.5e} max: {:.5e} after_div: {:.5e} max_div: {:.5e}", prev_div, prev_max_div, after_div, max_div);

        diffuse(
            0,
            &mut self.work,
            &self.density,
            diff,
            dt,
            diffuse_iter,
            shape,
            &self.params,
        );
        advect(
            0,
            &mut self.density,
            &self.work,
            &self.vx,
            &self.vy,
            dt,
            shape,
            &self.params,
        );
        decay(&mut self.density, 1. - self.params.decay);

        diffuse(
            0,
            &mut self.work,
            &self.density2,
            diff,
            dt,
            1,
            shape,
            &self.params,
        );
        advect(
            0,
            &mut self.density2,
            &self.work,
            &self.vx,
            &self.vy,
            dt,
            shape,
            &self.params,
        );
        decay(&mut self.density2, 1. - self.params.decay);

        if self.params.obstacle && self.params.dye_from_obstacle {
            let (center, radius) = obstacle_position(&shape);
            let radius = radius + 1;
            for j in center.1 - radius..center.1 + radius {
                for i in center.0 - radius..center.0 + radius {
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
}
