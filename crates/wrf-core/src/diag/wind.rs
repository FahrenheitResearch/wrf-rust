//! Wind diagnostic variables:
//! ua, va, wa, wspd, wdir, uvmet, uvmet10, wspd10, wdir10

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;
use crate::grid::par_to_vec;
use crate::units::PAR_MIN_LEN;
use rayon::prelude::*;

/// U-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_ua(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.u_destag(t).map(|v| par_to_vec(&v))
}

/// V-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_va(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.v_destag(t).map(|v| par_to_vec(&v))
}

/// W-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_wa(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.w_destag(t).map(|v| par_to_vec(&v))
}

/// Latitude (degrees). `[ny, nx]`
pub fn compute_lat(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlat(t).map(|v| par_to_vec(&v))
}

/// Longitude (degrees). `[ny, nx]`
pub fn compute_lon(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlong(t).map(|v| par_to_vec(&v))
}

/// Rotate grid-relative (u, v) to earth-relative using SINALPHA/COSALPHA.
fn rotate_to_earth(
    u: &[f64],
    v: &[f64],
    sina: &[f64],
    cosa: &[f64],
    nxy: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut u_earth = vec![0.0; u.len()];
    let mut v_earth = vec![0.0; v.len()];
    rotate_to_earth_into(u, v, sina, cosa, nxy, &mut u_earth, &mut v_earth);
    (u_earth, v_earth)
}

/// `rotate_to_earth` into given output slices (uvmet writes both halves of one array).
fn rotate_to_earth_into(
    u: &[f64],
    v: &[f64],
    sina: &[f64],
    cosa: &[f64],
    nxy: usize,
    u_earth: &mut [f64],
    v_earth: &mut [f64],
) {
    u_earth
        .par_iter_mut()
        .zip(v_earth.par_iter_mut())
        .enumerate()
        .with_min_len(PAR_MIN_LEN)
        .for_each(|(idx, (ue, ve))| {
            let ij = idx % nxy;
            let ca = cosa[ij];
            let sa = sina[ij];
            *ue = u[idx] * ca - v[idx] * sa;
            *ve = u[idx] * sa + v[idx] * ca;
        });
}

/// Wind speed from (u, v). `[nz, ny, nx]`
pub fn compute_wspd(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    Ok(u.par_iter()
        .zip(v.par_iter())
        .with_min_len(PAR_MIN_LEN)
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// Wind direction (degrees, meteorological convention). `[nz, ny, nx]`
pub fn compute_wdir(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}

/// Earth-rotated U/V wind. Returns interleaved `[u_earth..., v_earth...]` (2 * nxyz).
pub fn compute_uvmet(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let n = u.len();
    let mut out = vec![0.0; 2 * n];
    let (ue, ve) = out.split_at_mut(n);
    rotate_to_earth_into(&u, &v, &sina, &cosa, f.nxy(), ue, ve);
    Ok(out)
}

/// 10-m earth-rotated U/V wind. Returns `[u_earth..., v_earth...]` (2 * nxy).
pub fn compute_uvmet10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let n = u10.len();
    let mut out = vec![0.0; 2 * n];
    let (ue, ve) = out.split_at_mut(n);
    rotate_to_earth_into(&u10, &v10, &sina, &cosa, f.nxy(), ue, ve);
    Ok(out)
}

/// 10-m wind speed (m/s). `[ny, nx]`
pub fn compute_wspd10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    Ok(u.par_iter()
        .zip(v.par_iter())
        .with_min_len(PAR_MIN_LEN)
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// 10-m wind direction (degrees). `[ny, nx]`
pub fn compute_wdir10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}
