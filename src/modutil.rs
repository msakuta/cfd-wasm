

pub fn fmod(f: f32, freq: f32) -> f32{
    f - (f / freq).floor() * freq
}
pub fn imod(f: i32, freq: i32) -> i32{
    f - (f as f32 / freq as f32).floor() as i32 * freq
}
pub fn umod(f: u32, freq: u32) -> u32{
    f - (f as f32 / freq as f32).floor() as u32 * freq
}
pub fn fimod(f: f32, freq: f32) -> (f32, u32){
    let fm = fmod(f, freq);
    let fi = fm.floor();
    (fm - fi, imod(fm as i32, freq as i32) as u32)
}



#[test]
fn test_fmod() {
    assert_eq!(fmod(2.5, 2.5), 0.);
    assert_eq!(fmod(2.5, 5.), 2.5);
    assert_eq!(fmod(1.25, 2.), 1.25);
    assert_eq!(fmod(5.0, 2.5), 0.0);
    assert_eq!(fmod(-2.75, 5.5), 2.75);
}

#[test]
fn test_imod() {
    assert_eq!(imod(3, 5), 3);
    assert_eq!(imod(5, 3), 2);
    assert_eq!(imod(-2, 3), 1);
    assert_eq!(imod(-5, 7), 2);
}

#[test]
fn test_fimod() {
    fn assert_near(a: f32, b: f32){
        assert!((a - b).abs() < 1e-6);
    }

    fn assert_near2(a: (f32, u32), b: (f32, u32)){
        assert_near(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    assert_near2(fimod(3.2, 5.), (0.2, 3));
    assert_near2(fimod(5.7, 3.), (0.7, 2));
    assert_near2(fimod(-2.5, 3.), (0.5, 0));
    assert_near2(fimod(-5.9, 7.), (0.1, 1));
}
#[test]
fn test_umod() {
    assert_eq!(umod(3, 5), 3);
    assert_eq!(umod(5, 3), 2);
    assert_eq!(umod(4, 3), 1);
    assert_eq!(umod(9, 7), 2);
}
