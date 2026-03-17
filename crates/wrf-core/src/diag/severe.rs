//! Severe weather composite diagnostic variables:
//! stp, scp, ehi, critical_angle, ship, bri

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Significant Tornado Parameter -- fixed layer (dimensionless). `[ny, nx]`
///
/// Thompson et al. (2003) formulation with proper term limits.
/// Uses SURFACE-BASED parcel for CAPE and LCL, 0-1 km SRH, 0-6 km shear.
///   STP = cape_term * lcl_term * srh_term * shear_term
///
/// SRH is computed through the canonical compute_srh_field path (earth-rotated + 10m prepend).
pub fn compute_stp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    // Surface-based CAPE + LCL
    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (sbcape, _, lcl, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "sb",
    );

    // 0-1 km SRH via canonical path (earth-rotated winds + 10m prepend)
    let srh1 = crate::diag::srh::compute_srh_field(f, t, 1000.0, None)?;

    // 0-6 km shear magnitude
    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(stp_fixed_from_components(&sbcape, &lcl, &srh1, &shear6))
}

/// Effective-layer Significant Tornado Parameter (dimensionless). `[ny, nx]`
///
/// Uses MIXED-LAYER parcel for CAPE, LCL, and CIN.
/// Uses effective inflow layer SRH and effective bulk wind difference (EBWD).
/// Includes CIN term: (200 + mlCIN) / 150.
///
/// STP_eff = (mlCAPE/1500) * ((2000-mlLCL)/1000) * (ESRH/150) * (EBWD/20) * ((200+mlCIN)/150)
///
/// Effective SRH uses earth-rotated winds with 10m prepend via compute_effective_srh.
pub fn compute_stp_effective(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    // Mixed-layer CAPE, CIN, and LCL
    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (mlcape, mlcin, lcl, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "ml",
    );

    // 0-6 km shear magnitude (EBWD approximation)
    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    // Effective-layer SRH via canonical path (earth-rotated winds + 10m prepend)
    let eff_srh = crate::diag::srh::compute_effective_srh(f, t, opts)?;

    Ok(stp_eff_from_components(&mlcape, &lcl, &mlcin, &eff_srh, &shear6))
}

/// Generic STP dispatcher: uses opts.layer_type to choose fixed or effective.
///
/// - `"effective"` -> `compute_stp_effective`
/// - anything else (default) -> `compute_stp` (fixed layer)
pub fn compute_stp_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    match opts.layer_type.as_deref() {
        Some("effective") => compute_stp_effective(f, t, opts),
        _ => compute_stp(f, t, opts),
    }
}

/// Fixed-layer STP: 4-term formula (no CIN).
///   STP = (sbCAPE/1500) * ((2000-LCL)/1000) * (SRH/150) * (shear/20)
fn stp_fixed_from_components(cape: &[f64], lcl: &[f64], srh: &[f64], shear: &[f64]) -> Vec<f64> {
    cape.par_iter()
        .zip(lcl.par_iter())
        .zip(srh.par_iter())
        .zip(shear.par_iter())
        .map(|(((c, l), s), sh)| {
            let cape_term = (c / 1500.0).max(0.0);
            let lcl_term = if *l >= 2000.0 { 0.0 } else if *l <= 1000.0 { 1.0 } else { (2000.0 - l) / 1000.0 };
            let srh_term = (s / 150.0).max(0.0);
            let shear_term = if *sh < 12.5 { 0.0 } else if *sh >= 30.0 { 1.5 } else { sh / 20.0 };
            cape_term * lcl_term * srh_term * shear_term
        })
        .collect()
}

/// Effective-layer STP: 5-term formula with CIN.
///   STP_eff = (mlCAPE/1500) * ((2000-mlLCL)/1000) * (ESRH/150) * (EBWD/20) * ((200+mlCIN)/150)
fn stp_eff_from_components(cape: &[f64], lcl: &[f64], cin: &[f64], srh: &[f64], shear: &[f64]) -> Vec<f64> {
    cape.par_iter()
        .zip(lcl.par_iter())
        .zip(cin.par_iter())
        .zip(srh.par_iter())
        .zip(shear.par_iter())
        .map(|((((c, l), ci), s), sh)| {
            let cape_term = (c / 1500.0).max(0.0);
            let lcl_term = if *l >= 2000.0 { 0.0 } else if *l <= 1000.0 { 1.0 } else { (2000.0 - l) / 1000.0 };
            let srh_term = (s / 150.0).max(0.0);
            let shear_term = if *sh < 12.5 { 0.0 } else if *sh >= 30.0 { 1.5 } else { sh / 20.0 };
            // CIN term: (200 + mlCIN) / 150, clamped to [0, 1]
            // CIN is negative, so 200 + CIN shrinks toward 0 as CIN gets more negative
            let cin_term = ((200.0 + ci) / 150.0).clamp(0.0, 1.0);
            cape_term * lcl_term * srh_term * shear_term * cin_term
        })
        .collect()
}

/// Supercell Composite Parameter (dimensionless). `[ny, nx]`
///
/// SRH is computed through the canonical compute_srh_field path (earth-rotated + 10m prepend).
pub fn compute_scp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (mucape, _, _, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "mu",
    );

    // 0-3 km SRH via canonical path (earth-rotated winds + 10m prepend)
    let srh3 = crate::diag::srh::compute_srh_field(f, t, 3000.0, None)?;
    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(crate::met::composite::compute_scp(&mucape, &srh3, &shear6))
}

/// Energy-Helicity Index (dimensionless). `[ny, nx]`
///
/// EHI = (CAPE * SRH) / 160000
///
/// SRH depth is configurable via `opts.depth_m` (default 1000 m for 0-1 km EHI).
/// SRH is computed through the canonical compute_srh_field path (earth-rotated + 10m prepend).
pub fn compute_ehi(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (sbcape, _, _, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "sb",
    );

    // SRH via canonical path (earth-rotated winds + 10m prepend)
    let srh_depth = opts.depth_m.unwrap_or(1000.0);
    let srh = crate::diag::srh::compute_srh_field(f, t, srh_depth, None)?;

    // EHI = (CAPE * SRH) / 160000
    Ok(sbcape
        .par_iter()
        .zip(srh.par_iter())
        .map(|(cape, s)| (cape * s) / 160000.0)
        .collect())
}

/// Critical angle (degrees). `[ny, nx]`
pub fn compute_critical_angle(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut result = vec![0.0f64; nxy];
    result.par_iter_mut().enumerate().for_each(|(ij, val)| {
        let mut u_prof = Vec::with_capacity(nz);
        let mut v_prof = Vec::with_capacity(nz);
        let mut h_prof = Vec::with_capacity(nz);

        for k in 0..nz {
            let idx = k * nxy + ij;
            u_prof.push(u[idx]);
            v_prof.push(v[idx]);
            h_prof.push(h_agl[idx]);
        }

        // Get Bunkers RM storm motion
        let ((sm_u, sm_v), _, _) =
            crate::met::wind::bunkers_storm_motion(&u_prof, &v_prof, &h_prof);

        // Interpolate wind at 500m
        let (u_500, v_500) = interp_wind_at_height(&u_prof, &v_prof, &h_prof, 500.0);

        *val = crate::met::wind::critical_angle(sm_u, sm_v, u_prof[0], v_prof[0], u_500, v_500);
    });

    Ok(result)
}

/// Significant Hail Parameter (dimensionless). `[ny, nx]`
pub fn compute_ship(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (mucape, _, _, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "mu",
    );

    // SHIP needs MUCAPE, column-integrated water, and T500
    // Find T at ~500 hPa for each column
    let mut t500 = vec![0.0f64; nxy];
    t500.par_iter_mut().enumerate().for_each(|(ij, t500_val)| {
        for k in 0..nz - 1 {
            let idx = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;
            if pres_hpa[idx] >= 500.0 && pres_hpa[idx1] < 500.0 {
                let frac = (500.0 - pres_hpa[idx1]) / (pres_hpa[idx] - pres_hpa[idx1]);
                *t500_val = tc[idx1] + frac * (tc[idx] - tc[idx1]);
                break;
            }
        }
    });

    // Simple SHIP approximation: MUCAPE * |mixing_ratio| * lapse_rate * (-T500) * shear / denom
    // Simplified SHIP version
    Ok(mucape
        .par_iter()
        .zip(t500.par_iter())
        .map(|(cape, t5)| {
            // Simplified SHIP
            let t500_factor = (-t5).max(0.0) / 30.0;
            (cape / 1500.0 * t500_factor).max(0.0)
        })
        .collect())
}

/// Bulk Richardson Number (dimensionless). `[ny, nx]`
pub fn compute_bri(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;   // Pa -- compute_cape_cin converts internally
    let t2 = f.t2_for_opts(t, opts)?;       // K  -- compute_cape_cin converts internally
    let q2 = f.q2_for_opts(t, opts)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    // Pass raw Pa/K values -- compute_cape_cin converts internally
    let (sbcape, _, _, _) = crate::met::composite::compute_cape_cin(
        &pres, &tc, &qv, &h_agl, &psfc, &t2, &q2,
        nx, ny, nz, "sb",
    );

    let shear6 = crate::met::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    // BRN = CAPE / (0.5 * shear^2)
    Ok(sbcape
        .par_iter()
        .zip(shear6.par_iter())
        .map(|(cape, shr)| {
            let denom = 0.5 * shr * shr;
            if denom > 0.1 { cape / denom } else { 0.0 }
        })
        .collect())
}

// ── Helpers ──

/// Linear interpolation of wind at a target height.
fn interp_wind_at_height(
    u_prof: &[f64],
    v_prof: &[f64],
    h_prof: &[f64],
    target_h: f64,
) -> (f64, f64) {
    for k in 0..h_prof.len() - 1 {
        if h_prof[k] <= target_h && h_prof[k + 1] > target_h {
            let frac = (target_h - h_prof[k]) / (h_prof[k + 1] - h_prof[k]);
            let u = u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]);
            let v = v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]);
            return (u, v);
        }
    }
    // Fallback: nearest level
    if target_h <= h_prof[0] {
        (u_prof[0], v_prof[0])
    } else {
        let last = h_prof.len() - 1;
        (u_prof[last], v_prof[last])
    }
}
