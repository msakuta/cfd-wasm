pub(crate) trait Idx {
    fn idx(&self, x: isize, y: isize) -> usize;
}

pub(crate) type Shape = (isize, isize);

impl Idx for Shape {
    fn idx(&self, x: isize, y: isize) -> usize {
        let (width, height) = self;
        ((x + width) % width + (y + height) % height * width) as usize
    }
}
