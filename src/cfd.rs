//! Computational Fluid Dynamics methods

use super::{BoundaryCondition, Idx, Params, Shape};

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

pub(crate) fn diffuse(
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

pub(crate) fn advect(
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

pub(crate) fn project(
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

pub(crate) fn decay(s: &mut [f64], decay_rate: f64) {
    for v in s {
        *v *= decay_rate;
    }
}
