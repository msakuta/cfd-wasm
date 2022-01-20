#[derive(Clone, Copy)]
pub(crate) struct Xor128 {
    x: u32,
}

impl Xor128 {
    pub(crate) fn new(seed: u32) -> Self {
        let mut ret = Xor128 { x: 2463534242 };
        if 0 < seed {
            ret.x ^= (seed & 0xffffffff) >> 0;
            ret.nexti();
        }
        ret.nexti();
        ret
    }

    pub(crate) fn nexti(&mut self) -> u32 {
        // We must bitmask and logical shift to simulate 32bit unsigned integer's behavior.
        // The optimizer is likely to actually make it uint32 internally (hopefully).
        // T = (I + L^a)(I + R^b)(I + L^c)
        // a = 13, b = 17, c = 5
        let x1 = ((self.x ^ (self.x << 13)) & 0xffffffff) >> 0;
        let x2 = ((x1 ^ (x1 >> 17)) & 0xffffffff) >> 0;
        self.x = ((x2 ^ (x2 << 5)) & 0xffffffff) >> 0;
        self.x
    }
}
