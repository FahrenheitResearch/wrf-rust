//! Severe weather composite diagnostic variables:
//! stp, scp, ehi, critical_angle, ship, bri

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Significant Tornado Parameter -- fixed layer (dimensionless). `[ny, nx]`
///
/// Thompson et al. (2003) formulation with proper term limits:
///   STP = cape_term * lcl_term * srh_term * shear_term
pub fn compute_stp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
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

    // 0-6 km shear magnitude
    let shear6 = wx_math::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    Ok(stp_from_components(&mlcape, &lcl, &srh1, &shear6))
}

/// Effective-layer Significant Tornado Parameter (dimensionless). `[ny, nx]`
///
/// Uses the effective inflow layer (CAPE >= 100, CIN >= -250) for CAPE and SRH
/// instead of fixed mixed-layer / 0-1 km values.
pub fn compute_stp_effective(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
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
    let nxy = nx * ny;

    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();

    // LCL from mixed-layer parcel (needed for LCL term regardless of layer type)
    let (_, _, lcl, _) = wx_math::composite::compute_cape_cin(
        &pres_hpa, &tc, &qv, &h_agl, &psfc, &t2_c, &q2,
        nx, ny, nz, "ml",
    );

    // 0-6 km shear magnitude (fixed layer, same as standard STP)
    let shear6 = wx_math::composite::compute_shear(&u, &v, &h_agl, nx, ny, nz, 0.0, 6000.0);

    // Effective-layer CAPE and SRH: computed column-by-column
    let mut eff_cape = vec![0.0f64; nxy];
    let mut eff_srh = vec![0.0f64; nxy];

    eff_cape
        .par_iter_mut()
        .zip(eff_srh.par_iter_mut())
        .enumerate()
        .for_each(|(ij, (cape_val, srh_val))| {
            // Extract column profiles
            let mut p_prof = Vec::with_capacity(nz);
            let mut t_prof = Vec::with_capacity(nz);
            let mut td_prof = Vec::with_capacity(nz);
            let mut h_prof = Vec::with_capacity(nz);
            let mut u_prof = Vec::with_capacity(nz);
            let mut v_prof = Vec::with_capacity(nz);

            for k in 0..nz {
                let idx = k * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
                h_prof.push(h_agl[idx]);
                u_prof.push(u[idx]);
                v_prof.push(v[idx]);
            }

            // Find the effective inflow layer: lowest contiguous layer where
            // parcel CAPE >= 100 J/kg AND CIN >= -250 J/kg
            let mut eff_bot: Option<usize> = None;
            let mut eff_top: Option<usize> = None;

            for k in 0..nz {
                if p_prof.len() - k < 2 {
                    break;
                }
                let (c, ci, _, _) = wx_math::thermo::cape_cin_core(
                    &p_prof[k..], &t_prof[k..], &td_prof[k..], &h_prof[k..],
                    p_prof[k], t_prof[k], td_prof[k],
                    "sb", 100.0, 300.0, None,
                );
                if c >= 100.0 && ci >= -250.0 {
                    if eff_bot.is_none() {
                        eff_bot = Some(k);
                    }
                    eff_top = Some(k);
                } else if eff_bot.is_some() {
                    // End of contiguous effective layer
                    break;
                }
            }

            if let (Some(bot), Some(top)) = (eff_bot, eff_top) {
                // MUCAPE within the effective layer: find parcel with max CAPE
                let mut max_cape = 0.0f64;
                for k in bot..=top {
                    if p_prof.len() - k < 2 {
                        break;
                    }
                    let (c, _, _, _) = wx_math::thermo::cape_cin_core(
                        &p_prof[k..], &t_prof[k..], &td_prof[k..], &h_prof[k..],
                        p_prof[k], t_prof[k], td_prof[k],
                        "sb", 100.0, 300.0, None,
                    );
                    if c > max_cape {
                        max_cape = c;
                    }
                }
                *cape_val = max_cape;

                // Effective-layer SRH: from effective bottom height to effective top height
                let eff_depth = h_prof[top] - h_prof[bot];
                if eff_depth > 0.0 {
                    let ((sm_u, sm_v), _, _) =
                        metrust::calc::bunkers_storm_motion(&u_prof, &v_prof, &h_prof);
                    let (_, _, total) = metrust::calc::storm_relative_helicity(
                        &u_prof[bot..], &v_prof[bot..], &h_prof[bot..],
                        eff_depth, sm_u, sm_v,
                    );
                    *srh_val = total;
                }
            }
        });

    Ok(stp_from_components(&eff_cape, &lcl, &eff_srh, &shear6))
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

/// Thompson et al. (2003) STP from pre-computed component arrays.
///
/// Applies the standard term limits:
///   cape_term  = (CAPE / 1500).max(0)
///   lcl_term   = ((2000 - LCL) / 1000), capped [0, 1]
///   srh_term   = (SRH / 150).max(0)
///   shear_term = (shear / 20), capped [0, 1.5], zeroed when shear < 12.5
fn stp_from_components(cape: &[f64], lcl: &[f64], srh: &[f64], shear: &[f64]) -> Vec<f64> {
    cape.par_iter()
        .zip(lcl.par_iter())
        .zip(srh.par_iter())
        .zip(shear.par_iter())
        .map(|(((c, l), s), sh)| {
            let cape_term = (c / 1500.0).max(0.0);

            let lcl_term = if *l >= 2000.0 {
                0.0
            } else if *l <= 1000.0 {
                1.0
            } else {
                (2000.0 - l) / 1000.0
            };

            let srh_term = (s / 150.0).max(0.0);

            let shear_term = if *sh < 12.5 {
                0.0
            } else if *sh >= 30.0 {
                1.5
            } else {
                sh / 20.0
            };

            cape_term * lcl_term * srh_term * shear_term
        })
        .collect()
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
///
/// EHI = (CAPE * SRH) / 160000
///
/// SRH depth is configurable via `opts.depth_m` (default 1000 m for 0-1 km EHI).
pub fn compute_ehi(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
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

    let srh_depth = opts.depth_m.unwrap_or(1000.0);
    let srh = wx_math::composite::compute_srh(&u, &v, &h_agl, nx, ny, nz, srh_depth);

    // EHI = (CAPE * SRH) / 160000
    Ok(sbcape
        .par_iter()
        .zip(srh.par_iter())
        .map(|(cape, s)| (cape * s) / 160000.0)
        .collect())
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
