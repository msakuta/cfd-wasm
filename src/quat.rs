
use std::ops::{Add, AddAssign, Sub, Mul};
use std::convert::From;
use crate::vec3::Vec3;

#[derive(Clone, Debug, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quat{
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32,
}

impl Quat{
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat{
        Quat{x, y, z, w}
    }

    pub fn zero() -> Self{
        Self::new(0., 0., 0., 0.)
    }

    pub fn add(&self, o: &Self) -> Self{
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)
    }

    pub fn dot(&self, b: &Self) -> f32 {
        self.x*b.x
         + self.y*b.y
         + self.z*b.z
         + self.w*b.w
    }

    pub fn squared_len(&self) -> f32{
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn len(&self) -> f32 {
        self.squared_len().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.len();
        Self::new(self.x / len, self.y / len, self.z / len, self.w / len)
    }

    pub fn normalize(&mut self){
        let len = self.len();
        self.scale(1. / len);
    }

    pub fn scaled(&self, o: f32) -> Self{
        Self::new(self.x * o, self.y * o, self.z * o, self.w * o)
    }

    pub fn scale(&mut self, o: f32){
        self.x *= o;
        self.y *= o;
        self.z *= o;
        self.w *= o;
    }

    pub fn conjugated(&self) -> Self{
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    pub fn mul(&self, o: &Quat) -> Quat{
        let qa = self;
        let qb = o;
        Quat::new(qa.y*qb.z-qa.z*qb.y+qa.x*qb.w+qa.w*qb.x,
            qa.z*qb.x-qa.x*qb.z+qa.y*qb.w+qa.w*qb.y,
            qa.x*qb.y-qa.y*qb.x+qa.z*qb.w+qa.w*qb.z,
            -qa.x*qb.x-qa.y*qb.y-qa.z*qb.z+qa.w*qb.w)
    }

    pub fn transform(&self, v: &Vec3) -> Vec3{
        let qc = self.conjugated();
        let q: Quat = Quat::from(*v);
        let qr = self.mul(&q);
        let qret = qr * qc;
        Vec3::new(qret.x, qret.y, qret.z)
    }

    pub fn rotate(&self, v: &Vec3) -> Self{
        let q = Quat::new(v.x, v.y, v.z, 0.);
        Quat::mul(&q, &self).add(self).normalized()
    }

    /// \brief Returns rotation represented as a quaternion, defined by angle and vector.
    ///
    /// Note that the vector must be normalized.
    /// \param p Angle in radians
    /// \param sx,sy,sz Components of axis vector, must be normalized
    pub fn rotation(p: f32, sx: f32, sy: f32, sz: f32) -> Quat{
        let len = (p / 2.).sin();
        Quat::new(len * sx, len * sy, len * sz, (p / 2.).cos())
    }

    pub fn slerp(&self, o: &Self, t: f32) -> Self{
        let qr = self.dot(o);
        let ss = 1.0 - qr * qr;

        fn sqrtepsilon() -> f32{
            (1e-10f32).sqrt()
        }

        if ss <= sqrtepsilon() || self == o {
            *self
        }
        else {
            let sp = ss.sqrt();

            let ph = qr.acos();
            let pt = ph * t;
            let mut t1 = pt.sin() / sp;
            let t0 = (ph - pt).sin() / sp;

            // Long path case
            if qr < 0. {
                t1 *= -1.;
            }

            Self::new(
                self.x * t0 + o.x * t1,
                self.y * t0 + o.y * t1,
                self.z * t0 + o.z * t1,
                self.w * t0 + o.w * t1)
        }
    }

    pub fn from_pyr(pyr: &Vec3) -> Self{
        let mx = Self::rotation(pyr.z, 1., 0., 0.);
        let my = Self::rotation(pyr.y, 0., 0., 1.);
        let mp = Self::rotation(pyr.x, 0., 1., 0.);
        mx * my * mp
    }
}

// It's a shame that we cannot omit '&' in front of Vec3 object
// if we want to use multiplication operator (*).
// Another option is to call like v1.mul(v2), but it's ugly too.
impl Mul<f32> for &Quat{
    type Output = Quat;

    fn mul(self, o: f32) -> Quat{
        Quat::new(self.x * o, self.y * o, self.z * o, self.w * o)
    }
}

impl Mul for Quat{
    type Output = Quat;

    fn mul(self, o: Quat) -> Quat{
        Quat::mul(&self, &o)
    }
}

impl Add for &Quat{
    type Output = Quat;

    fn add(self, o: Self) -> Quat{
        self.add(o)
    }
}

impl AddAssign for Quat{
    fn add_assign(&mut self, o: Quat){
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
        self.w += o.w;
    }
}

impl Sub for &Quat{
    type Output = Quat;

    fn sub(self, o: Self) -> Quat{
        Quat::new(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)
    }
}

impl From<Vec3> for Quat{
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z, 0.)
    }
}

