//! Pressure, height, and geopotential diagnostic variables:
//! pressure, height, height_agl, zstag, geopt, geopt_stag, terrain, slp, omega

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const G: f64 = 9.80665;
const RD: f64 = 287.058;

/// Full pressure (Pa). `[nz, ny, nx]`
pub fn compute_pressure(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_pressure(t)
}

/// Height MSL (m). `[nz, ny, nx]`
pub fn compute_height(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.height_msl(t)
}

/// Height AGL (m). `[nz, ny, nx]`
pub fn compute_height_agl(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.height_agl(t)
}

/// Height on staggered Z levels (m). `[nz_stag, ny, nx]`
pub fn compute_zstag(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let geopt_stag = f.geopotential_stag(t)?;
    Ok(geopt_stag.iter().map(|v| v / G).collect())
}

/// Full geopotential (m^2/s^2), destaggered. `[nz, ny, nx]`
pub fn compute_geopt(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_geopotential(t)
}

/// Geopotential on staggered Z levels. `[nz_stag, ny, nx]`
pub fn compute_geopt_stag(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.geopotential_stag(t)
}

/// Terrain height (m). `[ny, nx]`
pub fn compute_terrain(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.terrain(t)
}

/// Sea-level pressure (hPa). `[ny, nx]`
///
/// Uses the WRF method: Shuell's approach (Shuell 1995) with smoothed lapse rate.
pub fn compute_slp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;
    let h = f.height_msl(t)?;
    let ter = f.terrain(t)?;

    let nxy = f.nxy();
    let _nz = f.nz;

    let mut slp = vec![0.0f64; nxy];

    slp.par_iter_mut().enumerate().for_each(|(ij, slp_val)| {
        // Use lowest model level
        let p_sfc = pres[ij]; // k=0 level
        let t_sfc = tk[ij];
        let qv_sfc = qv[ij].max(0.0);
        let z_sfc = h[ij];
        let z_ter = ter[ij];

        // Virtual temperature at surface
        let tv_sfc = t_sfc * (1.0 + 0.61 * qv_sfc);

        // Use next level up for lapse rate
        let _p1 = pres[nxy + ij]; // k=1
        let t1 = tk[nxy + ij];
        let qv1 = qv[nxy + ij].max(0.0);
        let tv1 = t1 * (1.0 + 0.61 * qv1);
        let z1 = h[nxy + ij];

        // Temperature lapse rate (K/m)
        let gamma = if (z1 - z_sfc).abs() > 1.0 {
            ((tv_sfc - tv1) / (z1 - z_sfc)).clamp(0.001, 0.0065)
        } else {
            0.0065
        };

        // Standard barometric formula from surface to sea level
        let dz = z_ter;
        if dz.abs() < 1.0 {
            *slp_val = p_sfc / 100.0;
        } else {
            // Hypsometric equation with constant lapse rate
            let t_sl = tv_sfc + gamma * dz;
            let exponent = G / (RD * gamma);
            *slp_val = (p_sfc * (t_sl / tv_sfc).powf(exponent)) / 100.0;
        }
    });

    Ok(slp)
}

/// Omega: vertical velocity in pressure coordinates (Pa/s). `[nz, ny, nx]`
///
/// omega = -rho * g * w = -(p / (Rd * Tv)) * g * w
pub fn compute_omega(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let w = f.w_destag(t)?;
    let pres = f.full_pressure(t)?;
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;

    Ok(w.par_iter()
        .zip(pres.par_iter())
        .zip(tk.par_iter())
        .zip(qv.par_iter())
        .map(|(((w, p), t_k), q)| {
            let tv = t_k * (1.0 + 0.61 * q.max(0.0));
            let rho = p / (RD * tv);
            -rho * G * w
        })
        .collect())
}
