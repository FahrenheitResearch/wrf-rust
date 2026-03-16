//! Severe weather composite diagnostic variables:
//! stp, scp, ehi, critical_angle, ship, bri

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Significant Tornado Parameter (dimensionless). `[ny, nx]`
pub fn compute_stp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    // STP = (mlCAPE/1500) * ((2000-LCL)/1000) * (SRH_1km/150) * (shear_6km/20)
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();
    let q2 = f.q2(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();

    // MLCAPE + LCL
    let (mlcape, _, lcl, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
        nx, ny, nz, "ml",
    );

    // 0-1 km SRH
    let srh1 = wx_math::composite::compute_srh(&u, &v, &h_agl, nx, ny, nz, 1000.0);

    // 0-6 km shear
    let shear6 = wx_math::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(wx_math::composite::compute_stp(&mlcape, &lcl, &srh1, &shear6))
}

/// Supercell Composite Parameter (dimensionless). `[ny, nx]`
pub fn compute_scp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa: Vec<f64> = f.full_pressure(t)?.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();
    let q2 = f.q2(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    let (mucape, _, _, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
        nx, ny, nz, "mu",
    );

    let srh3 = wx_math::composite::compute_srh(&u, &v, &h_agl, nx, ny, nz, 3000.0);
    let shear6 = wx_math::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(wx_math::composite::compute_scp(&mucape, &srh3, &shear6))
}

/// Energy-Helicity Index (dimensionless). `[ny, nx]`
pub fn compute_ehi(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa: Vec<f64> = f.full_pressure(t)?.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();
    let q2 = f.q2(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    let (sbcape, _, _, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
        nx, ny, nz, "sb",
    );

    let srh1 = wx_math::composite::compute_srh(&u, &v, &h_agl, nx, ny, nz, 1000.0);

    Ok(wx_math::composite::compute_ehi(&sbcape, &srh1))
}

/// Critical angle (degrees). `[ny, nx]`
pub fn compute_critical_angle(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
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
            metrust::calc::bunkers_storm_motion(&u_prof, &v_prof, &h_prof);

        // Interpolate wind at 500m
        let (u_500, v_500) = interp_wind_at_height(&u_prof, &v_prof, &h_prof, 500.0);

        *val = metrust::calc::critical_angle(sm_u, sm_v, u_prof[0], v_prof[0], u_500, v_500);
    });

    Ok(result)
}

/// Significant Hail Parameter (dimensionless). `[ny, nx]`
pub fn compute_ship(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa: Vec<f64> = f.full_pressure(t)?.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();
    let q2 = f.q2(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let (mucape, _, _, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
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
    // Using wx_math if available, otherwise simplified version
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
pub fn compute_bri(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa: Vec<f64> = f.full_pressure(t)?.iter().map(|p| p / 100.0).collect();
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();
    let q2 = f.q2(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    let (sbcape, _, _, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
        nx, ny, nz, "sb",
    );

    let shear6 = wx_math::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

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
