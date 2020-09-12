
pub type Pixel = image::Rgb<u8>;
pub type PixelF = image::Rgb<f32>;

pub fn add_pixel(a: PixelF, b: &PixelF) -> PixelF{
    image::Rgb::<f32>([a[0] as f32 + b[0] as f32, a[1] as f32 + b[1] as f32, a[2] as f32 + b[2] as f32])
}
pub fn scale_pixel(s: f32, a: Pixel) -> PixelF{
    image::Rgb::<f32>([s * a[0] as f32, s * a[1] as f32, s * a[2] as f32])
}

#[test]
fn test_add_pixel() {
    assert_eq!(add_pixel(image::Rgb::<f32>([1., 2., 3.]), &image::Rgb::<f32>([10., 20., 30.])), image::Rgb::<f32>([11., 22., 33.]));
    assert_eq!(add_pixel(image::Rgb::<f32>([10., 20., 30.]), &image::Rgb::<f32>([1., 2., 3.])), image::Rgb::<f32>([11., 22., 33.]));
}

#[test]
fn test_scale_pixel() {
    assert_eq!(scale_pixel(3.5, image::Rgb::<u8>([1, 2, 3])), image::Rgb::<f32>([3.5, 7.0, 10.5]));
    assert_eq!(add_pixel(image::Rgb::<f32>([10., 20., 30.]), &scale_pixel(2., image::Rgb::<u8>([1, 2, 3]))), image::Rgb::<f32>([12., 24., 36.]));
}
