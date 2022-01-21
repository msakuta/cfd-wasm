use crate::shape::{Idx, Shape};

pub(crate) fn pick_bits(f: &[f64], shape: Shape, pos: (isize, isize), threshold: f64) -> u8 {
    (f[shape.idx(pos.0, pos.1)] > threshold) as u8
        | (((f[shape.idx(pos.0 + 1, pos.1)] > threshold) as u8) << 1)
        | (((f[shape.idx(pos.0 + 1, pos.1 + 1)] > threshold) as u8) << 2)
        | (((f[shape.idx(pos.0, pos.1 + 1)] > threshold) as u8) << 3)
}

pub(crate) fn line(idx: u8) -> Option<[[f64; 2]; 2]> {
    match idx {
        0 => None,
        1 => Some([[0.5, 0.], [0., 0.5]]),
        2 => Some([[0.5, 0.], [1., 0.5]]),
        3 => Some([[0., 0.5], [1., 0.5]]),
        4 => Some([[1., 0.5], [0.5, 1.]]),
        5 => Some([[1., 0.5], [0.5, 1.]]),
        6 => Some([[0.5, 0.], [0.5, 1.]]),
        7 => Some([[0.5, 1.], [0., 0.5]]),
        8 => Some([[0.5, 1.], [0., 0.5]]),
        9 => Some([[0.5, 0.], [0.5, 1.]]),
        10 => Some([[1., 0.5], [0.5, 1.]]),
        11 => Some([[1., 0.5], [0.5, 1.]]),
        12 => Some([[0., 0.5], [1., 0.5]]),
        13 => Some([[0.5, 0.], [1., 0.5]]),
        14 => Some([[0.5, 0.], [1., 0.5]]),
        15 => None,
        _ => panic!("index must be in 0-15!"),
    }
}

#[test]
fn test_bits() {
    assert_eq!(pick_bits(&[0., 0., 0., 0.], (2, 2), (0, 0), 0.5), 0);
    assert_eq!(pick_bits(&[1., 0., 0., 0.], (2, 2), (0, 0), 0.5), 1);
    assert_eq!(pick_bits(&[0., 1., 0., 0.], (2, 2), (0, 0), 0.5), 2);
    assert_eq!(pick_bits(&[0., 0., 1., 0.], (2, 2), (0, 0), 0.5), 8);
    assert_eq!(pick_bits(&[0., 0., 0., 1.], (2, 2), (0, 0), 0.5), 4);
}
