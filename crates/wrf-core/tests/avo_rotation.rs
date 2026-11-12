//! `avo` and `pvo` against wrf-python's definition on fields whose answer
//! is known: solid-body rotation has relative vorticity 2 omega everywhere.
//! Up to 0.2.39 both bodies passed (v, u) to `vorticity(u, v)` and returned
//! du/dx - dv/dy, which is zero here, so every closed low vanished from the
//! vorticity maps.

mod common;

use wrf_core::diag::vorticity::absolute_vorticity_staggered;
use wrf_core::{getvar, ComputeOpts, WrfFile};

#[test]
fn solid_rotation_has_vorticity_two_omega_on_the_staggered_grid() {
    let (nx, ny, nz) = (7usize, 5usize, 2usize);
    let (dx, dy) = (3000.0, 3000.0);
    let omega = 1.0e-4;
    // Mass point (i, j) sits at x = i dx, y = j dy; u faces at x = (i - 1/2) dx,
    // v faces at y = (j - 1/2) dy.
    let mut u = Vec::new();
    let mut v = Vec::new();
    for _ in 0..nz {
        for j in 0..ny {
            for _i in 0..=nx {
                u.push(-omega * (j as f64 * dy));
            }
        }
        for _j in 0..=ny {
            for i in 0..nx {
                v.push(omega * (i as f64 * dx));
            }
        }
    }
    let msfu = vec![1.0; ny * (nx + 1)];
    let msfv = vec![1.0; (ny + 1) * nx];
    let msfm = vec![1.0; nx * ny];
    let cor = vec![1.0e-4; nx * ny];
    let avo = absolute_vorticity_staggered(&u, &v, &msfu, &msfv, &msfm, &cor, nx, ny, nz, dx, dy).unwrap();
    for (index, value) in avo.iter().enumerate() {
        assert!((value - (2.0 * omega + 1.0e-4)).abs() < 1e-12, "cell {index}: {value}");
    }
    // A cyclone and an anticyclone differ in sign: reversing the rotation
    // reverses the relative part.
    let u_anti: Vec<f64> = u.iter().map(|x| -x).collect();
    let v_anti: Vec<f64> = v.iter().map(|x| -x).collect();
    let anti = absolute_vorticity_staggered(&u_anti, &v_anti, &msfu, &msfv, &msfm, &cor, nx, ny, nz, dx, dy).unwrap();
    assert!(anti.iter().all(|value| (value - (-2.0 * omega + 1.0e-4)).abs() < 1e-12));
}

#[test]
fn map_factors_enter_as_wrf_python_applies_them() {
    // Uniform map factor m: winds divided by m, differences times m^2,
    // so the relative vorticity scales by m.
    let (nx, ny, nz) = (4usize, 4usize, 1usize);
    let omega = 2.0e-4;
    let m = 1.05;
    let mut u = Vec::new();
    let mut v = Vec::new();
    for j in 0..ny { for _ in 0..=nx { u.push(-omega * j as f64 * 1000.0); } }
    for _ in 0..=ny { for i in 0..nx { v.push(omega * i as f64 * 1000.0); } }
    let avo = absolute_vorticity_staggered(
        &u, &v, &vec![m; ny * (nx + 1)], &vec![m; (ny + 1) * nx], &vec![m; nx * ny],
        &vec![0.0; nx * ny], nx, ny, nz, 1000.0, 1000.0).unwrap();
    assert!(avo.iter().all(|value| (value - 2.0 * omega * m).abs() < 1e-12));
}

const OMEGA_ROT: f64 = 1.0e-4;
const DX: f64 = 3000.0;

/// A small file in solid-body rotation (relative vorticity 2 OMEGA_ROT),
/// with theta rising 2 K and pressure falling 50 hPa per level, so that
/// dtheta/dp = -4e-4 K/Pa at every point.
fn rotating_file(tag: &str, sign: f64) -> WrfFile {
    let (nx, ny, nz) = (6usize, 5usize, 4usize);
    let mut file = common::WrfLike::new(nx, ny, nz, DX, DX);
    file.u(|_, j, _| -sign * OMEGA_ROT * j as f64 * DX)
        .v(|_, _, i| sign * OMEGA_ROT * i as f64 * DX)
        .mass3("T", |k, _, _| 2.0 * k as f64)
        .mass3("P", |_, _, _| 0.0)
        .mass3("PB", |k, _, _| 100_000.0 - 5_000.0 * k as f64)
        .plane("XLAT", |_, _| 45.0)
        .plane("XLONG", |_, i| -100.0 + i as f64 * 0.03)
        .plane("F", |_, _| 1.0e-4);
    let path = file.write(tag);
    let opened = WrfFile::open(&path).expect("open test file");
    std::fs::remove_file(&path).ok();
    opened
}

#[test]
fn avo_of_a_rotating_file_is_two_omega_plus_f() {
    for (tag, sign) in [("avo-cyclone", 1.0), ("avo-anticyclone", -1.0)] {
        let file = rotating_file(tag, sign);
        let out = getvar(&file, "avo", Some(0), &ComputeOpts::default()).unwrap();
        assert_eq!(out.units, "s-1");
        assert_eq!(out.shape, vec![4, 5, 6]);
        let expected = sign * 2.0 * OMEGA_ROT + 1.0e-4;
        for (index, value) in out.data.iter().enumerate() {
            assert!((value - expected).abs() < 1e-10, "{tag} cell {index}: {value} vs {expected}");
        }
    }
}

#[test]
fn pvo_of_a_rotating_file_carries_the_relative_vorticity() {
    // PVO = -g (zeta + f) dtheta/dp, in PVU.  With zeta lost (the swapped
    // arguments) the cyclone and the anticyclone would both give the
    // f-only value; with it they differ by 2 x 2 omega.
    let g = 9.80665;
    let f = 2.0 * 7.2921159e-5 * 45f64.to_radians().sin();
    let dthdp = -4.0e-4;
    let f_only = -g * f * dthdp * 1e6;
    for (tag, sign) in [("pvo-cyclone", 1.0), ("pvo-anticyclone", -1.0)] {
        let file = rotating_file(tag, sign);
        let out = getvar(&file, "pvo", Some(0), &ComputeOpts::default()).unwrap();
        let expected = -g * (sign * 2.0 * OMEGA_ROT + f) * dthdp * 1e6;
        for (index, value) in out.data.iter().enumerate() {
            assert!((value - expected).abs() < 1e-5 * expected.abs(), "{tag} cell {index}: {value} vs {expected}");
            assert!((value - f_only).abs() > 0.1, "{tag} cell {index}: relative vorticity missing ({value})");
        }
    }
}
