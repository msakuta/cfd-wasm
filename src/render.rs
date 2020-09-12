
use crate::vec3::Vec3;
use crate::quat::Quat;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;
use image::DynamicImage;
use std::io;
use crate::modutil::*;
use crate::pixelutil::*;

pub const MAX_REFLECTIONS: i32 = 3;
pub const MAX_REFRACTIONS: i32 = 10;


const OUTONLY: u32 = 1;
const INONLY: u32 = 1<<1;
const RIGNORE: u32 = 1<<2;
const GIGNORE: u32 = 1<<3;
const BIGNORE: u32 = 1<<4;
// const RONLY: u32 = (GIGNORE|BIGNORE);
// const GONLY: u32 = (RIGNORE|BIGNORE);
// const BONLY: u32 = (RIGNORE|GIGNORE);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RenderColor{
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RenderColor{
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self{r, g, b}
    }

    pub fn zero() -> Self {
        Self{r: 0., g: 0., b: 0.}
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RenderPattern{
    Solid, Checkerboard, RepeatedGradation
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum UVMap{
    XY, YZ, ZX, LL,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum TextureFilter{
    Nearest, Bilinear
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMaterialSerial{
    name: String,
    diffuse: RenderColor, /* Diffuse(R,G,B) */
    specular: RenderColor,/* Specular(R,G,B) */
    pn: i32,			/* Phong model index */
    t: f32, /* transparency, unit length per decay */
    n: f32, /* refraction constant */
    glow_dist: f32,
    frac: RenderColor, /* refraction per spectrum */
    pattern: RenderPattern,
    pattern_scale: f32,
    pattern_angle_scale: f32,
    texture_name: String,
    texture_filter: TextureFilter,
}

pub struct RenderMaterial{
    name: String,
    diffuse: RenderColor, /* Diffuse(R,G,B) */
    specular: RenderColor,/* Specular(R,G,B) */
    pn: i32,			/* Phong model index */
    t: f32, /* transparency, unit length per decay */
    n: f32, /* refraction constant */
    glow_dist: f32,
    frac: RenderColor, /* refraction per spectrum */
    pattern: RenderPattern,
    pattern_scale: f32,
    pattern_angle_scale: f32,
    texture_name: String,
    texture: Option<DynamicImage>,
    texture_filter: TextureFilter,
}

trait RenderMaterialInterface{
    fn get_phong_number(&self) -> i32;
    fn get_transparency(&self) -> f32;
    fn get_refraction_index(&self) -> f32;
    fn lookup_texture(&self, uv: (f32, f32)) -> RenderColor;
}

impl RenderMaterial{
    pub fn new(
        name: String,
        diffuse: RenderColor,
        specular: RenderColor,
        pn: i32,
        t: f32,
        n: f32)
     -> RenderMaterial{
         RenderMaterial{
             name,
             diffuse,
             specular,
             pn,
             t,
             n,
             glow_dist: 0.,
             frac: RenderColor::new(1., 1., 1.),
             pattern: RenderPattern::Solid,
             pattern_scale: 1.,
             pattern_angle_scale: 1.,
             texture_name: String::new(),
             texture: None,
             texture_filter: TextureFilter::Nearest,
         }
    }

    #[allow(dead_code)]
    pub fn get_name(&self) -> &str{
        &self.name
    }

    pub fn glow_dist(mut self, v: f32) -> Self{
        self.glow_dist = v;
        self
    }

    pub fn frac(mut self, frac: RenderColor) -> Self{
        self.frac = frac;
        self
    }

    pub fn pattern(mut self, pattern: RenderPattern) -> Self{
        self.pattern = pattern;
        self
    }

    pub fn pattern_scale(mut self, pattern_scale: f32) -> Self{
        self.pattern_scale = pattern_scale;
        self
    }

    pub fn pattern_angle_scale(mut self, pattern_angle_scale: f32) -> Self{
        self.pattern_angle_scale = pattern_angle_scale;
        self
    }

    #[allow(dead_code)]
    /// Error when open image failed
    pub fn texture(mut self, texture_name: &str) -> Result<Self, io::Error>{
        self.texture_name = String::from(texture_name);
        self.texture = Some(image::open(texture_name).or_else(
            |_|Err(io::Error::new(io::ErrorKind::Other, "texture image file load failed")))?);
        Ok(self)
    }

    /// Ignore quietly when open image failed
    pub fn texture_ok(mut self, texture_name: &str) -> Self{
        self.texture_name = String::from(texture_name);
        self.texture = image::open(texture_name).ok();
        self
    }

    fn serialize(&self) -> RenderMaterialSerial{
        RenderMaterialSerial{
            name: self.name.clone(),
            diffuse: self.diffuse,
            specular: self.specular,
            pn: self.pn,
            t: self.t,
            n: self.n,
            glow_dist: self.glow_dist,
            frac: self.frac,
            pattern: self.pattern,
            pattern_scale: self.pattern_scale,
            pattern_angle_scale: self.pattern_angle_scale,
            texture_name: self.texture_name.clone(),
            texture_filter: self.texture_filter,
        }
    }

    fn deserialize(obj: &RenderMaterialSerial) -> Result<RenderMaterial, DeserializeError>{
        Ok(RenderMaterial{
            name: obj.name.clone(),
            diffuse: obj.diffuse,
            specular: obj.specular,
            pn: obj.pn,
            t: obj.t,
            n: obj.n,
            glow_dist: obj.glow_dist,
            frac: obj.frac,
            pattern: obj.pattern,
            pattern_scale: obj.pattern_scale,
            pattern_angle_scale: obj.pattern_angle_scale,
            texture_name: obj.texture_name.clone(),
            texture: image::open(&obj.texture_name).ok(),
            texture_filter: obj.texture_filter,
        })
    }

    fn get_uv(&self, pos: &Vec3, uvmap: UVMap) -> (f32, f32) {
        match uvmap {
            UVMap::XY => (pos.x / self.pattern_scale, pos.y / self.pattern_scale),
            UVMap::YZ => (pos.y / self.pattern_scale, pos.z / self.pattern_scale),
            UVMap::ZX => (pos.z / self.pattern_scale, pos.x / self.pattern_scale),
            UVMap::LL => {
                let (dx, dz) = (pos.x, pos.z);
                (pos.z.atan2(pos.x) / self.pattern_angle_scale,
                    ((dx * dx + dz * dz).sqrt().atan2(pos.y)) / self.pattern_angle_scale)
            },
        }
    }
}

impl RenderMaterialInterface for RenderMaterial{
    fn get_phong_number(&self) -> i32{
        self.pn
    }

    fn get_transparency(&self) -> f32{
        self.t
    }

    fn get_refraction_index(&self) -> f32{
        self.n
    }

    fn lookup_texture(&self, uv: (f32, f32)) -> RenderColor{
        let (u, v) = uv;
        if let Some(image::ImageRgb8(ref texture)) = self.texture {
            match self.texture_filter {
                TextureFilter::Nearest => {
                    let pixel = *texture.get_pixel(
                        imod((u * texture.width() as f32) as i32, texture.width() as i32) as u32,
                        imod((v * texture.height() as f32) as i32, texture.height() as i32) as u32);
                    return RenderColor{r: pixel[0] as f32 / 256., g: pixel[1] as f32 / 256., b: pixel[2] as f32 / 256.};
                },
                TextureFilter::Bilinear => {
                    let (fu, iu) = fimod(u * texture.width() as f32, texture.width() as f32);
                    let (fv, iv) = fimod(v * texture.height() as f32, texture.height() as f32);
                    let zero: PixelF = image::Rgb::<f32>([0f32; 3]);
                    let pixel = [
                        scale_pixel((1. - fu) * (1. - fv), *texture.get_pixel(iu, iv)),
                        scale_pixel((1. - fu) * fv, *texture.get_pixel(iu, umod(iv + 1, texture.height()))),
                        scale_pixel(fu * (1. - fv),  *texture.get_pixel(umod(iu + 1, texture.width()), iv)),
                        scale_pixel(fu * fv, *texture.get_pixel(umod(iu + 1, texture.width()), umod(iv + 1, texture.height()))),
                    ].iter().fold(zero, add_pixel);
                    return RenderColor{r: pixel[0] as f32 / 256., g: pixel[1] as f32 / 256., b: pixel[2] as f32 / 256.};
                }
            }
        }
        match self.pattern {
            RenderPattern::Solid => self.diffuse,
            RenderPattern::Checkerboard => {
                let ix = u.floor() as i32;
                let iy = v.floor() as i32;
                if (ix + iy) % 2 == 0 {
                    RenderColor::new(0., 0., 0.)
                } else {
                    self.diffuse
                }
            }
            RenderPattern::RepeatedGradation => {
                RenderColor::new(
                    self.diffuse.r * fmod(u, 1.),
                    self.diffuse.g * fmod(v, 1.),
                    self.diffuse.b
                )
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RenderSphereSerial{
    material: String,
    r: f32,
    org: Vec3,
    uvmap: UVMap,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RenderFloorSerial{
    material: String,
    org: Vec3,		/* Center */
    face_normal: Vec3,
    uvmap: UVMap,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum RenderObjectSerial{
    Sphere(RenderSphereSerial),
    Floor(RenderFloorSerial),
}

pub struct DeserializeError{
    pub s: String,
}

impl DeserializeError{
    fn new(s: &str) -> Self{
        DeserializeError{s: s.to_string()}
    }
}

impl Into<std::io::Error> for DeserializeError{
    fn into(self) -> std::io::Error{
        std::io::Error::new(std::io::ErrorKind::Other, "Deserialize error: ".to_string() + &self.s)
    }
}

impl From<serde_yaml::Error> for DeserializeError{
    fn from(_e: serde_yaml::Error) -> DeserializeError{
        DeserializeError{s: "serde_yaml::Error".to_string()}
    }
}

pub trait RenderObjectInterface{
    fn get_material(&self) -> &RenderMaterial;
    fn get_diffuse(&self, position: &Vec3) -> RenderColor;
    fn get_specular(&self, position: &Vec3) -> RenderColor;
    fn get_normal(&self, position: &Vec3) -> Vec3;
    fn raycast(&self, vi: &Vec3, eye: &Vec3, ray_length: f32, flags: u32) -> f32;
    fn distance(&self, vi: &Vec3) -> f32;
    fn serialize(&self) -> RenderObjectSerial;
}

#[derive(Clone)]
pub struct RenderSphere{
    material: Arc<RenderMaterial>,
    r: f32,			/* Radius */
    org: Vec3,		/* Center */
    uvmap: UVMap,
}

impl RenderSphere{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        material: Arc<RenderMaterial>,
        r: f32,
        org: Vec3,
    ) -> RenderObject {
        RenderObject::Sphere(RenderSphere::new_raw(
            material,
            r,
            org,
        ))
    }

    fn new_raw(
        material: Arc<RenderMaterial>,
        r: f32,
        org: Vec3,
    ) -> RenderSphere {
        RenderSphere{
            material,
            r,
            org,
            uvmap: UVMap::XY,
        }
    }

    fn uvmap(mut self, v: UVMap) -> Self{
        self.uvmap = v;
        self
    }

    fn deserialize(ren: &RenderEnv, serial: &RenderSphereSerial) -> Result<RenderObject, DeserializeError>{
        Ok(RenderObject::Sphere(
            Self::new_raw(ren.materials.get(&serial.material)
            .ok_or_else(|| DeserializeError::new(&format!("RenderSphere couldn't find material {}", serial.material)))?
            .clone(),
            serial.r, serial.org)
            .uvmap(serial.uvmap)))
    }
}

impl RenderObjectInterface for RenderSphere{
    fn get_material(&self) -> &RenderMaterial{
        &self.material
    }

    fn get_diffuse(&self, position: &Vec3) -> RenderColor{
        self.material.lookup_texture(self.material.get_uv(&(*position - self.org), self.uvmap))
    }

    fn get_specular(&self, _position: &Vec3) -> RenderColor{
        self.material.specular
    }

    fn get_normal(&self, position: &Vec3) -> Vec3{
        (*position - self.org).normalized()
    }

    fn raycast(&self, vi: &Vec3, eye: &Vec3, ray_length: f32, flags: u32) -> f32{
        let obj = self;
        /* calculate vector from eye position to the object's center. */
        let wpt = *vi - obj.org;

        /* scalar product of the ray and the vector. */
        let b = 2.0f32 * eye.dot(&wpt);

        /* ??? */
        let c = wpt.dot(&wpt) - obj.r * obj.r;

        /* discriminant?? */
        let d2 = b * b - 4.0f32 * c;
        if d2 >= std::f32::EPSILON {
            let d = d2.sqrt();
            let t0 = (-b - d) as f32 / 2.0f32;
            if 0 == (flags & OUTONLY) && t0 >= 0.0f32 && t0 < ray_length {
                return t0;
            }
            else if 0 == (flags & INONLY) && 0f32 < (t0 + d) && t0 + d < ray_length {
                return t0 + d;
            }
        }

        ray_length
    }

    fn distance(&self, vi: &Vec3) -> f32{
        ((self.org - *vi).len() - self.r).max(0.)
    }

    fn serialize(&self) -> RenderObjectSerial{
        RenderObjectSerial::Sphere(RenderSphereSerial{
            material: self.material.name.clone(),
            org: self.org,
            r: self.r,
            uvmap: self.uvmap,
        })
    }
}

#[derive(Clone)]
pub struct RenderFloor{
    material: Arc<RenderMaterial>,
    org: Vec3,		/* Center */
    face_normal: Vec3,
    uvmap: UVMap,
}

impl RenderFloor{
    #[allow(dead_code)]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        material: Arc<RenderMaterial>,
        org: Vec3,
        face_normal: Vec3,
    ) -> RenderObject {
        RenderObject::Floor(RenderFloor::new_raw(
            material,
            org,
            face_normal,
        ))
    }

    pub fn new_raw(
        material: Arc<RenderMaterial>,
        org: Vec3,
        face_normal: Vec3,
    ) -> RenderFloor {
        RenderFloor{
            material,
            org,
            face_normal,
            uvmap: UVMap::XY,
        }
    }

    pub fn uvmap(mut self, uvmap: UVMap) -> Self{
        self.uvmap = uvmap;
        self
    }

    fn deserialize(ren: &RenderEnv, serial: &RenderFloorSerial) -> Result<RenderObject, DeserializeError>{
        Ok(RenderObject::Floor(Self::new_raw(ren.materials.get(&serial.material)
            .ok_or_else(|| DeserializeError::new(&format!("RenderFloor couldn't find material {}", serial.material)))?
            .clone(),
            serial.org, serial.face_normal)
            .uvmap(serial.uvmap)))
    }
}

impl RenderObjectInterface for RenderFloor{
    fn get_material(&self) -> &RenderMaterial{
        &self.material
    }

    fn get_diffuse(&self, position: &Vec3) -> RenderColor{
        self.material.lookup_texture(self.material.get_uv(&(position - &self.org), self.uvmap))
    }

    fn get_specular(&self, _position: &Vec3) -> RenderColor{
        self.material.specular
    }

    fn get_normal(&self, _: &Vec3) -> Vec3{
        self.face_normal
    }

    fn raycast(&self, vi: &Vec3, eye: &Vec3, ray_length: f32, _flags: u32) -> f32{
        let wpt = vi - &self.org;
        let w = self.face_normal.dot(eye);
        if /*fabs(w) > 1.0e-7*/ w <= 0. {
            let t0 = (-self.face_normal.dot(&wpt)) / w;
            if t0 >= 0. && t0 < ray_length {
                return t0;
            }
        }
        ray_length
    }

    fn distance(&self, vi: &Vec3) -> f32{
        (vi - &self.org).dot(&self.face_normal).max(0.)
    }

    fn serialize(&self) -> RenderObjectSerial{
        RenderObjectSerial::Floor(RenderFloorSerial{
            material: self.material.name.clone(),
            org: self.org,
            face_normal: self.face_normal,
            uvmap: self.uvmap,
        })
    }
}

#[derive(Clone)]
pub enum RenderObject{
    Sphere(RenderSphere),
    Floor(RenderFloor)
}

impl RenderObject{
    pub fn get_interface(&self) -> &dyn RenderObjectInterface{
        match self {
            RenderObject::Sphere(ref obj) => obj as &dyn RenderObjectInterface,
            RenderObject::Floor(ref obj) => obj as &dyn RenderObjectInterface,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct CameraSerial{
    position: Vec3,
    pyr: Vec3,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct CameraKeyframeSerial{
    camera: CameraSerial,
    velocity: Vec3,
    camera_target: Option<Vec3>,
    duration: f32,
}

#[derive(Clone, Serialize, Deserialize)]
struct CameraMotionSerial(Vec<CameraKeyframeSerial>);

#[derive(Clone, Copy)]
pub struct Camera{
    pub position: Vec3,
    pub pyr: Vec3,
    pub rotation: Quat,
}

impl From<CameraSerial> for Camera{
    fn from(o: CameraSerial) -> Camera{
        Camera{
            position: o.position,
            pyr: o.pyr,
            rotation: Quat::from_pyr(&o.pyr),
        }
    }
}

#[derive(Clone)]
pub struct CameraKeyframe{
    pub camera: Camera,
    pub velocity: Vec3,
    _camera_target: Option<Vec3>,
    pub duration: f32,
}

impl CameraKeyframe{
    pub fn camera_target(&self) -> Option<Vec3>{
        self._camera_target
    }
}

#[derive(Clone)]
pub struct CameraMotion(pub Vec<CameraKeyframe>);

#[derive(Clone)]
pub struct RenderEnv{
    pub camera: Camera, /* camera position */
    pub camera_motion: CameraMotion,
    pub xres: i32,
    pub yres: i32,
    pub xfov: f32,
    pub yfov: f32,
    // Materials are stored in a string map, whose key is a string.
    // A material is stored in Arc in order to share between global material list
    // and each object. I'm not sure if it's better than embedding into each object.
    // We wanted to but cannot use reference (borrow checker gets mad about enums)
    // nor Rc (multithreading gets mad).
    pub materials: HashMap<String, Arc<RenderMaterial>>,
    pub objects: Vec<RenderObject>,
    pub light: Vec3,
    pub bgproc: fn(ren: &RenderEnv, pos: &Vec3) -> RenderColor,
    pub use_raymarching: bool,
    glow_effect: Option<f32>,
    pub max_reflections: i32,
    pub max_refractions: i32,
}

#[derive(Serialize, Deserialize)]
struct Scene{
    camera: CameraSerial,
    camera_motion: CameraMotionSerial,
    light: Vec3,
    max_reflections: i32,
    max_refractions: i32,
    materials: HashMap<String, RenderMaterialSerial>,
    objects: Vec<RenderObjectSerial>,
}


impl RenderEnv{
    pub fn new(
        cam: Vec3, /* camera position */
        pyr: Vec3, /* camera direction in pitch yaw roll form */
        xres: i32,
        yres: i32,
        xfov: f32,
        yfov: f32,
        bgproc: fn(ren: &RenderEnv, pos: &Vec3) -> RenderColor
    ) -> Self{
        RenderEnv{
            camera: Camera{
                position: cam,
                pyr,
                rotation: Quat::from_pyr(&pyr),
            },
            camera_motion: CameraMotion(vec![]),
            xres,
            yres,
            xfov,
            yfov,
            materials: HashMap::new(),
            objects: Vec::new(),
            light: Vec3::new(0., 0., 1.),
            bgproc,
            use_raymarching: false,
            glow_effect: None,
            max_reflections: MAX_REFLECTIONS,
            max_refractions: MAX_REFRACTIONS,
        }
    }

    pub fn materials(mut self, materials: HashMap<String, Arc<RenderMaterial>>) -> Self{
        self.materials = materials;
        self
    }

    pub fn objects(mut self, objects: Vec<RenderObject>) -> Self{
        self.objects = objects;
        self
    }

    fn set_light(&mut self, light: Vec3){
        self.light = light.normalized();
    }

    pub fn light(mut self, light: Vec3) -> Self{
        self.set_light(light);
        self
    }

    pub fn use_raymarching(mut self, f: bool) -> Self{
        self.use_raymarching = f;
        self
    }

    pub fn glow_effect(mut self, v: Option<f32>) -> Self{
        self.glow_effect = v;
        self
    }

    pub fn serialize(&self) -> Result<String, std::io::Error>{
        let mut sceneobj = Scene{
            camera: CameraSerial{
                position: self.camera.position,
                pyr: self.camera.pyr
            },
            camera_motion: CameraMotionSerial(vec![]),
            light: self.light,
            max_reflections: MAX_REFLECTIONS,
            max_refractions: MAX_REFRACTIONS,
            materials: HashMap::new(),
            objects: self.objects.iter().map(|o| o.get_interface().serialize()).collect(),
        };
        for object in &self.objects {
            let material = object.get_interface().get_material();
            sceneobj.materials.insert(material.name.clone(), material.serialize());
        }
        Ok(serde_yaml::to_string(&sceneobj)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?)
        // println!("{}", scene);
    }

    pub fn deserialize(&mut self, s: &str) -> Result<(), DeserializeError>{
        let sceneobj = serde_yaml::from_str::<Scene>(s)?;
        let mm: Result<HashMap<_, _>, DeserializeError> = sceneobj.materials.into_iter().map(
            |m| Ok((m.0, Arc::new(RenderMaterial::deserialize(&m.1)?)))).collect();
        self.camera = Camera::from(sceneobj.camera);
        self.set_light(sceneobj.light);
        self.camera_motion = CameraMotion(sceneobj.camera_motion.0.iter().map(|o|
            CameraKeyframe{
                camera: Camera::from(o.camera),
                velocity: o.velocity,
                _camera_target: o.camera_target,
                duration: o.duration,
            }).collect());
        self.max_reflections = sceneobj.max_reflections;
        self.max_refractions = sceneobj.max_refractions;
        self.materials = mm?;
        self.objects.clear();
        for object in sceneobj.objects {
            match object {
                RenderObjectSerial::Sphere(ref sobj) =>
                    self.objects.push(RenderSphere::deserialize(self, sobj)?),
                RenderObjectSerial::Floor(ref sobj) =>
                    self.objects.push(RenderFloor::deserialize(self, sobj)?),
            }
        }
        Ok(())
    }
}


pub fn render(ren: &RenderEnv, pointproc: &mut impl FnMut(i32, i32, &RenderColor),
    thread_count: i32) {


    let process_line = |iy: i32, point_middle: &mut dyn FnMut(i32, i32, RenderColor)| {
        for ix in 0..ren.xres {
            let mut vi = ren.camera.position;
            let mut eye: Vec3 = Vec3::new( /* cast ray direction vector? */
                1.,
                (ix - ren.xres / 2) as f32 * 2. * ren.xfov / ren.xres as f32,
                -(iy - ren.yres / 2) as f32 * 2. * ren.yfov / ren.yres as f32,
            );
            eye = ren.camera.rotation.transform(&eye).normalized();

            point_middle(ix, iy,
                if ren.use_raymarching { raymarch } else { raytrace }
                (ren, &mut vi, &mut eye, 0, None, 0) );
        }
    };

    if thread_count == 1 {
        let mut point_middle = |ix: i32, iy: i32, col: RenderColor| {
            pointproc(ix, iy, &col);
        };
        for iy in 0..ren.yres {
            process_line(iy, &mut point_middle);
        }
    }
    else{
        println!("Splitting scanlines; {} threads", thread_count);
        for m_y in 0..ren.yres {
            process_line(m_y, &mut |ix: i32, _iy: i32, col: RenderColor| {
                pointproc(ix, _iy, &col);
            });
        }
    }
}

// This warning is stupid, these variables are intermediate variables for the
// function, so having long name wouldn't help to understand.  Anyone who needs
// to understand what this function does needs to look into Hermite interpolation
// and fully understand it anyways.
#[allow(clippy::many_single_char_names)]
fn hermite_interpolate_f32(t: f32, x0: f32, x1: f32, v0: f32, v1: f32) -> f32{
    let h = 1.;
    let d = x0;
    let c = v0;
    let r = x1 - x0 - h * v0;
    let s = v1 - v0;
    let a = (h * s - 2. * r) / h / h / h;
    let b = (-h * s + 3. * r) / h / h;
    a * t * t * t + b * t * t + c * t + d
}

pub fn hermite_interpolate(t: f32, x0: &Vec3, x1: &Vec3, v0: &Vec3, v1: &Vec3) -> Vec3{
    Vec3::new(
        hermite_interpolate_f32(t, x0.x, x1.x, v0.x, v1.x),
        hermite_interpolate_f32(t, x0.y, x1.y, v0.y, v1.y),
        hermite_interpolate_f32(t, x0.z, x1.z, v0.z, v1.z))
}

pub fn render_frames(ren: &mut RenderEnv, width: usize, height: usize,
    frame_proc: &mut impl FnMut(i32, &Vec<u8>), thread_count: i32)
{
    let mut prev_camera = ren.camera;
    let mut prev_velocity = Vec3::zero();
    let total_frames = ren.camera_motion.0.iter().fold(0., |acc, m| acc + m.duration);
    let mut accum_frame = 0;
    let frame_step = 0.5;
    for (n, frame) in ren.camera_motion.0.iter().enumerate() {
        let v0 = prev_velocity;
        let v1 = frame.velocity;
        println!("keyframe {} / {}, v0: {},{},{}", n, ren.camera_motion.0.len(), v0.x, v0.y, v0.z);
        for i in 0..(frame.duration / frame_step) as i32 {
            let f = i as f32 / (frame.duration / frame_step);
            println!("Rendering frame {} / {}, v0: {},{}", accum_frame, total_frames, v0.x, v0.y);
            ren.camera.position = hermite_interpolate(f, &prev_camera.position, &frame.camera.position,
                &v0, &v1);
            ren.camera.rotation = if let Some(target) = frame._camera_target {
                let delta = target - ren.camera.position;
                let pitch = (delta.y).atan2((delta.x * delta.x + delta.z * delta.z).sqrt());
                let yaw = -delta.z.atan2(delta.x);
                Quat::rotation(yaw, 0., 1., 0.)
                * Quat::rotation(pitch, 0., 0., 1.)
                * Quat::rotation(-std::f32::consts::PI / 2., 1., 0., 0.)
            }
            else{
                prev_camera.rotation.slerp(&frame.camera.rotation, f)
            };
            let data = {
                let mut data = vec![0u8; 3 * width * height];
                let mut putpoint = |x: i32, y: i32, fc: &RenderColor| {
                    data[(x as usize + y as usize * width) * 3    ] = (fc.r * 255.).min(255.) as u8;
                    data[(x as usize + y as usize * width) * 3 + 1] = (fc.g * 255.).min(255.) as u8;
                    data[(x as usize + y as usize * width) * 3 + 2] = (fc.b * 255.).min(255.) as u8;
                };

                render(ren, &mut putpoint, thread_count);
                data
            };
            frame_proc(accum_frame, &data);
            accum_frame += 1;
            // }
        }
        prev_camera = frame.camera;
        prev_velocity = frame.velocity;
    }
}


/* find first object the ray hits */
/// @returns time at which ray intersects with a shape and its object id.
fn raycast(ren: &RenderEnv, vi: &Vec3, eye: &Vec3,
    ig: Option<&RenderObject>, flags: u32) -> (f32, usize)
{
    let mut t = std::f32::INFINITY;
    let mut ret_idx = 0;

	for (idx, obj) in ren.objects.iter().enumerate() {
        if let Some(ignore_obj) = ig {
            if ignore_obj as *const _ == obj as *const _ {
                continue;
            }
        }

        let obj_t = obj.get_interface().raycast(vi, eye, t, flags);
        if obj_t < t {
            t = obj_t;
            ret_idx = idx;
        }
    }

	(t, ret_idx)
}

fn shading(ren: &RenderEnv,
            idx: usize,
            n: &Vec3,
            pt: &Vec3,
            eye: &Vec3,
            nest: i32) -> RenderColor
{
    let o = &ren.objects[idx].get_interface();

    // let mut lv: f32;
    let (diffuse_intensity, reflected_ray, reflection_intensity) = {
        /* scalar product of light normal and surface normal */
        let light_incidence = ren.light.dot(n);
        let ln2 = 2.0 * light_incidence;
        let reflected_ray_to_light_source = (n * ln2) - ren.light;

        let eps = std::f32::EPSILON;
        let pn = o.get_material().get_phong_number();
        (
            light_incidence.max(0.),
            *pt + (ren.light * eps),
            if 0 != pn {
                let reflection_incidence = -reflected_ray_to_light_source.dot(eye);
                if reflection_incidence > 0.0 { reflection_incidence.powi(pn) }
                else        { 0.0 }
            }
            else { 0. }
        )
    };

    /* shadow trace */
    let (k1, k2) = {
        let ray: Vec3 = ren.light;
        let k1 = 0.2;
        if ren.use_raymarching {
            let RaymarchSingleResult{
                iter, travel_dist, ..} = raymarch_single(ren, &reflected_ray, &ray, Some(&ren.objects[idx]));
            if FAR_AWAY <= travel_dist || MAX_ITER <= iter || 0. < ren.objects[idx].get_interface().get_material().get_transparency() {
                ((k1 + diffuse_intensity).min(1.), reflection_intensity)
            }
            else {
                (k1, 0.)
            }
        }
        else {
            let (t, i) = raycast(ren, &reflected_ray, &ray, Some(&ren.objects[idx]), 0);
            if t >= std::f32::INFINITY || 0. < ren.objects[i].get_interface().get_material().get_transparency() {
                ((k1 + diffuse_intensity).min(1.), reflection_intensity)
            }
            else {
                (k1, 0.)
            }
        }
    };

	/* face texturing */
		let kd = o.get_diffuse(pt);
	// else{
	// 	kd.fred = ren.objects[idx].kdr;
	// 	kd.fgreen = ren.objects[idx].kdg;
	// 	kd.fblue = ren.objects[idx].kdb;
	// }

	/* refraction! */
	if nest < ren.max_refractions && 0. < o.get_material().get_transparency() {
		let sp = eye.dot(&n);
		let f = o.get_material().get_transparency();

		let fc2 = {
            let frac = o.get_material().get_refraction_index();
			let reference = sp * (if sp > 0. { frac } else { 1. / frac } - 1.);
            let mut ray = (*eye + (n * reference)).normalized();
            let eps = std::f32::EPSILON;
			let mut pt3 = *pt + (ray * eps);
            (if ren.use_raymarching { raymarch }
                else { raytrace })(ren, &mut pt3, &mut ray, nest,
                Some(&ren.objects[idx]), if sp < 0. { OUTONLY } else { INONLY })
		};
/*		t = raycast(ren, &reflectedRay, &ray, &i, &ren->objects[idx], OUTONLY);
		if(t < INFINITY)
		{
			Vec3 n2;
			f = exp(-t / o->t);
			normal(ren, i, &reflectedRay, &n2);
			shading(ren, i, &n2, &reflectedRay, &ray, &fc2, nest+1);
		}
		else{
			f = 0;
			ren->bgproc(&ray, &fc2);
		}*/
        RenderColor{
            r: (kd.r * k1 + k2) * (1. - f) + fc2.r * f,
            g: (kd.g * k1 + k2) * (1. - f) + fc2.g * f,
            b: (kd.b * k1 + k2) * (1. - f) + fc2.b * f,
        }
	}
	else{
		RenderColor{
            r: kd.r * k1 + k2,
            g: kd.g * k1 + k2,
            b: kd.b * k1 + k2,
        }
	}
}


fn raytrace(ren: &RenderEnv, vi: &mut Vec3, eye: &mut Vec3,
    mut lev: i32, init_ig: Option<&RenderObject>, mut flags: u32) -> RenderColor
{
    let mut fcs = RenderColor::new(1., 1., 1.);

	let mut ret_color = RenderColor::new(0., 0., 0.);
/*	bgcolor(eye, pColor);*/

    let mut ig: Option<&RenderObject> = init_ig;
	loop {
		lev += 1;
		let (t, idx) = raycast(ren, vi, eye, ig, flags);
		if t < std::f32::INFINITY {
/*			t -= EPS;*/

            /* shared point */
            // What a terrible formula... it's almost impractical to use it everywhere.
            let pt = (*eye * t) + *vi;

            let o = &ren.objects[idx].get_interface();
            let n = o.get_normal(&pt);
            let face_color = shading(ren, idx,&n,&pt,eye, lev);
            // if idx == 2 {
            //     println!("Hit {}: eye: {:?} normal: {:?} shading: {:?}", idx, eye, n, face_color);
            // }

            let ks = o.get_specular(&pt);

            if 0 == (RIGNORE & flags) { ret_color.r += face_color.r * fcs.r; fcs.r *= ks.r; }
            if 0 == (GIGNORE & flags) { ret_color.g += face_color.g * fcs.g; fcs.g *= ks.g; }
            if 0 == (BIGNORE & flags) { ret_color.b += face_color.b * fcs.b; fcs.b *= ks.b; }
            if idx == 0 {
                break;
            }

			if (fcs.r + fcs.g + fcs.b) <= 0.1 {
				break;
            }

			if lev >= ren.max_reflections {
                break;
            }

			*vi = pt;
			let en2 = -2.0 * eye.dot(&n);
			*eye += n * en2;

			if n.dot(&eye) < 0. {
                flags &= !INONLY;
                flags |= OUTONLY;
            }
			else {
                flags &= !OUTONLY;
                flags |= INONLY;
            }

            ig = Some(&ren.objects[idx]);
        }
        else{
            let fc2 = (ren.bgproc)(ren, eye);
            ret_color.r	+= fc2.r * fcs.r;
            ret_color.g	+= fc2.g * fcs.g;
            ret_color.b	+= fc2.b * fcs.b;
        }
        if !(t < std::f32::INFINITY && lev < ren.max_reflections) {
            break;
        }
	}

    ret_color
}

fn distance_estimate(ren: &RenderEnv, vi: &Vec3,
    ig: Option<&RenderObject>) -> (f32, usize, f32)
{
    let mut closest_dist = std::f32::INFINITY;
    let mut ret_idx = 0;
    let mut glowing_dist = std::f32::INFINITY;

    for (idx, obj) in ren.objects.iter().enumerate() {
        if let Some(ignore_obj) = ig {
            if ignore_obj as *const _ == obj as *const _ {
                continue;
            }
        }

        let dist = obj.get_interface().distance(vi);
        if dist < closest_dist {
            closest_dist = dist;
            ret_idx = idx;
        }

        let glow = dist * obj.get_interface().get_material().glow_dist;
        if 0. < glow && glow < glowing_dist {
            glowing_dist = glow;
        }
    }

    (closest_dist, ret_idx, glowing_dist)
}

const RAYMARCH_EPS: f32 = 1e-3;
const FAR_AWAY: f32 = 1e4;
const MAX_ITER: usize = 10000;

struct RaymarchSingleResult{
    final_dist: f32,
    idx: usize,
    pos: Vec3,
    iter: usize,
    travel_dist: f32,
    min_dist: f32,
}

fn raymarch_single(ren: &RenderEnv, init_pos: &Vec3, eye: &Vec3, ig: Option<&RenderObject>)
    -> RaymarchSingleResult
{
    let mut iter = 0;
    let mut travel_dist = 0.;
    let mut pos = *init_pos;
    let mut min_dist = std::f32::INFINITY;
    loop {
        let (dist, idx, glowing_dist) = distance_estimate(ren, &pos, ig);
        pos = (*eye * dist) + pos;
        travel_dist += dist;
        iter += 1;
        // let glowing_dist = ren.objects[idx].get_interface().get_glowing_dist();
        if glowing_dist < min_dist {
            min_dist = glowing_dist;
        }
        // println!("raymarch {:?} iter: {} pos: {:?}, dist: {}", eye, iter, pos, dist);
        if dist < RAYMARCH_EPS || FAR_AWAY < dist || MAX_ITER < iter {
            return RaymarchSingleResult{
                final_dist: dist,
                idx,
                pos,
                iter,
                travel_dist,
                min_dist,
            }
        }
    }
}

fn raymarch(ren: &RenderEnv, vi: &mut Vec3, eye: &mut Vec3,
    mut lev: i32, init_ig: Option<&RenderObject>, mut flags: u32) -> RenderColor
{
    // println!("using raymarch {:?}", eye);
    let mut fcs = RenderColor::new(1., 1., 1.);
    let mut pos = *vi;

    let mut ret_color = RenderColor::new(0., 0., 0.);
    let mut min_min_dist = std::f32::INFINITY;
/*	bgcolor(eye, pColor);*/

    let mut ig: Option<&RenderObject> = init_ig;
    loop {
        lev += 1;
        let RaymarchSingleResult{
            final_dist, idx, pos: pt, iter, min_dist, ..} = raymarch_single(ren, &pos, eye, ig);
        if min_dist < min_min_dist {
            min_min_dist = min_dist;
        }
        if MAX_ITER < iter {
            // println!("Max iter reached: {:?} dist: {} idx: {}", eye, dist, idx);
        }
        if final_dist < RAYMARCH_EPS {
/*			t -= EPS;*/

            /* safe point */
            // What a terrible formula... it's almost impractical to use it everywhere.

            let o = &ren.objects[idx].get_interface();
            let n = o.get_normal(&pt);
            // let face_color = RenderColor::new(travel_dist / 100. % 1., 0., 0.);
            let face_color = shading(ren, idx,&n,&pt,eye, lev);
            // if idx == 2 {
                // println!("Hit {}: eye: {:?} normal: {:?} shading: {:?}", idx, eye, n, face_color);
            // }

            let ks = o.get_specular(&pt);

            if 0 == (RIGNORE & flags) { ret_color.r += face_color.r * fcs.r; fcs.r *= ks.r; }
            if 0 == (GIGNORE & flags) { ret_color.g += face_color.g * fcs.g; fcs.g *= ks.g; }
            if 0 == (BIGNORE & flags) { ret_color.b += face_color.b * fcs.b; fcs.b *= ks.b; }
            if idx == 0 {
                break;
            }

			if (fcs.r + fcs.g + fcs.b) <= 0.1 {
				break;
            }

			if lev >= MAX_REFLECTIONS {
                break;
            }

			pos = pt;
			let en2 = -2.0 * eye.dot(&n);
			*eye += n * en2;

			if n.dot(&eye) < 0. {
                flags &= !INONLY;
                flags |= OUTONLY;
            }
			else {
                flags &= !OUTONLY;
                flags |= INONLY;
            }

            ig = Some(&ren.objects[idx]);
        }
        else{
            let fc2 = (ren.bgproc)(ren, eye);
            ret_color.r	+= fc2.r * fcs.r;
            ret_color.g	+= fc2.g * fcs.g;
            ret_color.b	+= fc2.b * fcs.b;
        }
        if MAX_REFLECTIONS <= lev {
            break;
        }
	}
    // println!("raymarch loop end {:?}", eye);

    if let Some(glow_effect) = ren.glow_effect {
        let factor = if min_min_dist == std::f32::INFINITY { 1. }
            else { 1. + (0. + glow_effect * (0.99f32).powf(min_min_dist)) };
        RenderColor::new(
            factor * ret_color.r,
            factor * ret_color.g,
            factor * ret_color.b)
    }
    else{
        ret_color
    }
}

