//! `avo` and `pvo` on a latitude-longitude grid, where the map factor is
//! 1/cos(lat) along x and 1 along y (MAPFAC_MX, MAPFAC_MY).  The known
//! answer: a solid-body rotation of the atmosphere about any axis r0 with
//! angular speed w has relative vorticity 2 w (r0 . r) at every point r.

mod common;

use wrf_core::diag::vorticity::{absolute_vorticity_staggered, absolute_vorticity_staggered_xy};
use wrf_core::{getvar, ComputeOpts, WrfFile};

const EARTH_RADIUS: f64 = 6_370_000.0;
const W: f64 = 5.0e-6; // rad/s: about 30 m/s at the rotation's equator

/// The axis of the test rotation: latitude 30, longitude 20 (unit vector).
fn axis() -> [f64; 3] {
    unit(30.0, 20.0)
}

fn unit(lat: f64, lon: f64) -> [f64; 3] {
    let (p, l) = (lat.to_radians(), lon.to_radians());
    [p.cos() * l.cos(), p.cos() * l.sin(), p.sin()]
}

/// (u, v) of the rotation w * axis x (R r) at (lat, lon).
fn rotation_wind(lat: f64, lon: f64) -> (f64, f64) {
    let [a, b, c] = axis();
    let (p, l) = (lat.to_radians(), lon.to_radians());
    let r = unit(lat, lon);
    let cross = [b * r[2] - c * r[1], c * r[0] - a * r[2], a * r[1] - b * r[0]];
    let east = [-l.sin(), l.cos(), 0.0];
    let north = [-p.sin() * l.cos(), -p.sin() * l.sin(), p.cos()];
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    (EARTH_RADIUS * W * dot(cross, east), EARTH_RADIUS * W * dot(cross, north))
}

fn rotation_vorticity(lat: f64, lon: f64) -> f64 {
    let r0 = axis();
    let r = unit(lat, lon);
    2.0 * W * (r0[0] * r[0] + r0[1] * r[1] + r0[2] * r[2])
}

/// A regular grid: mass rows at lat0 + j d, columns at lon0 + i d; u faces
/// half a step west, v faces half a step south.
struct Grid {
    nx: usize,
    ny: usize,
    lat0: f64,
    lon0: f64,
    d: f64,
}

impl Grid {
    fn lat(&self, j: f64) -> f64 { self.lat0 + j * self.d }
    fn lon(&self, i: f64) -> f64 { self.lon0 + i * self.d }
    fn spacing_m(&self) -> f64 { EARTH_RADIUS * self.d.to_radians() }
    fn u(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for j in 0..self.ny { for i in 0..=self.nx {
            out.push(rotation_wind(self.lat(j as f64), self.lon(i as f64 - 0.5)).0);
        } }
        out
    }
    fn v(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for j in 0..=self.ny { for i in 0..self.nx {
            out.push(rotation_wind(self.lat(j as f64 - 0.5), self.lon(i as f64)).1);
        } }
        out
    }
    fn mass(&self, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let mut out = Vec::new();
        for j in 0..self.ny { for i in 0..self.nx { out.push(f(self.lat(j as f64), self.lon(i as f64))); } }
        out
    }
}

fn grid() -> Grid {
    // 1 degree from 85 S to 85 N, 60 degrees of longitude.
    Grid { nx: 61, ny: 171, lat0: -85.0, lon0: -10.0, d: 1.0 }
}

#[test]
fn latlon_rotation_is_the_spherical_curl_at_every_latitude() {
    let g = grid();
    let (nx, ny) = (g.nx, g.ny);
    let mx = g.mass(|lat, _| 1.0 / lat.to_radians().cos());
    let my = vec![1.0; nx * ny];
    let mut mxu = Vec::new();
    for j in 0..ny { for _ in 0..=nx { mxu.push(1.0 / g.lat(j as f64).to_radians().cos()); } }
    let myv = vec![1.0; (ny + 1) * nx];
    let cor = vec![0.0; nx * ny];
    let ds = g.spacing_m();
    let avo = absolute_vorticity_staggered_xy(&g.u(), &g.v(), &mxu, &myv, &mx, &my, &cor, nx, ny, 1, ds, ds).unwrap();
    // The isotropic form with the unit factor a file without MAPFAC_M falls back to.
    let iso = absolute_vorticity_staggered(
        &g.u(), &g.v(), &vec![1.0; ny * (nx + 1)], &vec![1.0; (ny + 1) * nx], &vec![1.0; nx * ny],
        &cor, nx, ny, 1, ds, ds).unwrap();
    let mut worst_iso_high_lat = 0.0f64;
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let (lat, lon) = (g.lat(j as f64), g.lon(i as f64));
            let exact = rotation_vorticity(lat, lon);
            let got = avo[j * nx + i];
            // Second-order differences on a 1 degree grid; the metric term
            // grows toward the pole, so the bound does too.
            let bound = 2.0e-4 * 2.0 * W / lat.to_radians().cos().powi(2);
            assert!((got - exact).abs() < bound, "lat {lat} lon {lon}: {got:e} vs {exact:e}");
            if (70.0..=80.0).contains(&lat.abs()) {
                worst_iso_high_lat = worst_iso_high_lat.max((iso[j * nx + i] - exact).abs() / (2.0 * W));
            }
        }
    }
    // Without the x/y factors the answer is off by order one at 70-80 degrees.
    assert!(worst_iso_high_lat > 0.5, "isotropic error only {worst_iso_high_lat}");
}

/// A small lat-lon file as the global model writes it: MAPFAC_MX and
/// MAPFAC_MY only, plus F, U, V, T, P, PB, XLAT, XLONG.
fn latlon_file(tag: &str, g: &Grid) -> WrfFile {
    let ds = g.spacing_m();
    let mut file = common::WrfLike::new(g.nx, g.ny, 3, ds, ds);
    let u = g.u();
    let v = g.v();
    let nx = g.nx;
    file.u(|_, j, i| u[j * (nx + 1) + i])
        .v(|_, j, i| v[j * nx + i])
        .mass3("T", |k, _, _| 2.0 * k as f64)
        .mass3("P", |_, _, _| 0.0)
        .mass3("PB", |k, _, _| 50_000.0 - 5_000.0 * k as f64)
        .plane("XLAT", |j, _| g.lat(j as f64))
        .plane("XLONG", |_, i| g.lon(i as f64))
        .plane("MAPFAC_MX", |j, _| 1.0 / g.lat(j as f64).to_radians().cos())
        .plane("MAPFAC_MY", |_, _| 1.0)
        .plane("F", |j, _| 2.0 * 7.2921159e-5 * g.lat(j as f64).to_radians().sin());
    let path = file.write(tag);
    let opened = WrfFile::open(&path).expect("open test file");
    std::fs::remove_file(&path).ok();
    opened
}

#[test]
fn avo_of_a_latlon_file_is_the_rotation_plus_f() {
    let g = Grid { nx: 24, ny: 81, lat0: -80.0, lon0: 0.0, d: 2.0 };
    let file = latlon_file("avo-latlon", &g);
    let out = getvar(&file, "avo", Some(0), &ComputeOpts::default()).unwrap();
    assert_eq!(out.units, "s-1");
    let nxy = g.nx * g.ny;
    for k in 0..3 {
        for j in 1..g.ny - 1 {
            for i in 1..g.nx - 1 {
                let lat = g.lat(j as f64);
                let f = 2.0 * 7.2921159e-5 * lat.to_radians().sin();
                let exact = rotation_vorticity(lat, g.lon(i as f64)) + f;
                let got = out.data[k * nxy + j * g.nx + i];
                // 2 degree spacing and winds stored as float32.
                let bound = 1.0e-3 * 2.0 * W / lat.to_radians().cos().powi(2);
                assert!((got - exact).abs() < bound, "k {k} lat {lat}: {got:e} vs {exact:e}");
            }
        }
    }
}

#[test]
fn pvo_of_a_latlon_file_carries_the_spherical_vorticity() {
    let g = Grid { nx: 24, ny: 81, lat0: -80.0, lon0: 0.0, d: 2.0 };
    let file = latlon_file("pvo-latlon", &g);
    let out = getvar(&file, "pvo", Some(0), &ComputeOpts::default()).unwrap();
    // theta rises 2 K per level while pressure falls 50 hPa: dtheta/dp = -4e-4 K/Pa.
    let (grav, dthdp) = (9.80665, -4.0e-4);
    let nxy = g.nx * g.ny;
    for j in 2..g.ny - 2 {
        for i in 2..g.nx - 2 {
            let lat = g.lat(j as f64);
            let f = 2.0 * 7.2921159e-5 * lat.to_radians().sin();
            let zeta = rotation_vorticity(lat, g.lon(i as f64));
            let expected = -grav * (zeta + f) * dthdp * 1e6;
            let got = out.data[nxy + j * g.nx + i];
            let bound = 2.0e-3 * grav * 2.0 * W * dthdp.abs() * 1e6 / lat.to_radians().cos().powi(2);
            assert!((got - expected).abs() < bound, "lat {lat}: {got} vs {expected}");
        }
    }
}

#[test]
fn a_conformal_file_with_equal_x_and_y_factors_keeps_the_isotropic_answer() {
    // A Lambert file that also writes MAPFAC_MX == MAPFAC_MY == MAPFAC_M must
    // give the same bits as without them.
    let (nx, ny, nz) = (6usize, 5usize, 2usize);
    let build = |tag: &str, with_xy: bool| {
        let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
        let m = |j: usize, i: usize| 0.98 + 0.001 * j as f64 + 0.0005 * i as f64;
        file.u(|k, j, i| 5.0 + k as f64 - 0.3 * j as f64 + 0.1 * (i * j) as f64)
            .v(|k, j, i| -2.0 + 0.2 * i as f64 + 0.05 * (k * j) as f64)
            .mass3("T", |k, _, _| 2.0 * k as f64)
            .mass3("P", |_, _, _| 0.0)
            .mass3("PB", |k, _, _| 90_000.0 - 5_000.0 * k as f64)
            .plane("XLAT", |j, _| 40.0 + 0.03 * j as f64)
            .plane("XLONG", |_, i| -100.0 + 0.03 * i as f64)
            .plane("F", |j, _| 1.0e-4 + 1.0e-7 * j as f64)
            .plane("MAPFAC_M", m)
            .plane_u("MAPFAC_U", |j, i| m(j, i.min(nx - 1)))
            .plane_v("MAPFAC_V", |j, i| m(j.min(ny - 1), i));
        if with_xy {
            file.plane("MAPFAC_MX", m).plane("MAPFAC_MY", m);
        }
        let path = file.write(tag);
        let opened = WrfFile::open(&path).expect("open test file");
        std::fs::remove_file(&path).ok();
        opened
    };
    let plain = build("conformal-plain", false);
    let with_xy = build("conformal-xy", true);
    for name in ["avo", "pvo"] {
        let a = getvar(&plain, name, Some(0), &ComputeOpts::default()).unwrap();
        let b = getvar(&with_xy, name, Some(0), &ComputeOpts::default()).unwrap();
        assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{name} changed");
    }
}
