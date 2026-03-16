//! CAPE diagnostic variables: sbcape, mlcape, mucape, cape2d, cape3d, lcl, lfc, el
//!
//! Uses wx_math::composite::compute_cape_cin() for parallel grid computation
//! and wx_math::thermo::cape_cin_core() for column-by-column fallback.

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Helper: extract the 3-D + 2-D fields needed for CAPE, call compute_cape_cin.
/// Returns (cape_2d, cin_2d, lcl_2d, lfc_2d).
fn compute_cape_fields(
    f: &WrfFile,
    t: usize,
    parcel_type: &str,
    top_m: Option<f64>,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pres = f.full_pressure(t)?; // Pa -> need hPa for wx-math
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2(t)?;
    let q2 = f.q2(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Convert pressure from Pa to hPa for wx-math
    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();
    // Convert surface pressure from Pa to hPa
    let psfc_hpa: Vec<f64> = psfc.iter().map(|p| p / 100.0).collect();
    // Convert T2 from K to C
    let t2_c: Vec<f64> = t2.iter().map(|t| t - 273.15).collect();

    // Use wx_math's grid-parallel CAPE computation if no top_m specified
    if top_m.is_none() {
        let (cape, cin, lcl, lfc) = wx_math::composite::compute_cape_cin(
            &pres_hpa, &tc, &qv, &h_agl, &psfc_hpa, &t2_c, &q2,
            nx, ny, nz, parcel_type,
        );
        return Ok((cape, cin, lcl, lfc));
    }

    // With top_m: use column-by-column cape_cin_core
    let mut cape = vec![0.0f64; nxy];
    let mut cin = vec![0.0f64; nxy];
    let mut lcl = vec![0.0f64; nxy];
    let mut lfc = vec![0.0f64; nxy];

    cape.par_iter_mut()
        .zip(cin.par_iter_mut())
        .zip(lcl.par_iter_mut())
        .zip(lfc.par_iter_mut())
        .enumerate()
        .for_each(|(ij, (((cape_v, cin_v), lcl_v), lfc_v))| {
            // Extract column profiles
            let mut p_prof = Vec::with_capacity(nz);
            let mut t_prof = Vec::with_capacity(nz);
            let mut td_prof = Vec::with_capacity(nz);
            let mut h_prof = Vec::with_capacity(nz);

            for k in 0..nz {
                let idx = k * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                // Compute Td from q and p
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                let td = (243.5 * ln_e) / (17.67 - ln_e);
                td_prof.push(td);
                h_prof.push(h_agl[idx]);
            }

            let (c, ci, l, lf) = wx_math::thermo::cape_cin_core(
                &p_prof, &t_prof, &td_prof, &h_prof,
                psfc_hpa[ij], t2_c[ij], td_prof.first().copied().unwrap_or(0.0),
                parcel_type, 100.0, 300.0, top_m,
            );

            *cape_v = c;
            *cin_v = ci;
            *lcl_v = l;
            *lfc_v = lf;
        });

    Ok((cape, cin, lcl, lfc))
}

fn resolve_parcel_type(opts: &ComputeOpts, default: &str) -> String {
    opts.parcel_type.as_deref().unwrap_or(default).to_lowercase()
}

// ── Public compute functions ──

pub fn compute_sbcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "sb", opts.top_m)?;
    Ok(cape)
}

pub fn compute_sbcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "sb", opts.top_m)?;
    Ok(cin)
}

pub fn compute_mlcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "ml", opts.top_m)?;
    Ok(cape)
}

pub fn compute_mlcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "ml", opts.top_m)?;
    Ok(cin)
}

pub fn compute_mucape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "mu", opts.top_m)?;
    Ok(cape)
}

pub fn compute_mucin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "mu", opts.top_m)?;
    Ok(cin)
}

pub fn compute_lcl(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, lcl, _) = compute_cape_fields(f, t, &pt, opts.top_m)?;
    Ok(lcl)
}

pub fn compute_lfc(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, _, lfc) = compute_cape_fields(f, t, &pt, opts.top_m)?;
    Ok(lfc)
}

pub fn compute_el(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    // EL not directly returned by compute_cape_cin, compute column-by-column
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let _pt = resolve_parcel_type(opts, "sb");

    let mut el = vec![0.0f64; nxy];
    el.par_iter_mut().enumerate().for_each(|(ij, el_v)| {
        let mut p_prof = Vec::with_capacity(nz);
        let mut t_prof = Vec::with_capacity(nz);
        let mut td_prof = Vec::with_capacity(nz);
        let mut h_prof = Vec::with_capacity(nz);

        for k in 0..nz {
            let idx = k * nxy + ij;
            p_prof.push(pres_hpa[idx]);
            t_prof.push(tc[idx]);
            let q = qv[idx].max(1e-10);
            let e = q * pres_hpa[idx] / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
            h_prof.push(h_agl[idx]);
        }

        // Use wx_math el function — returns Option<(p_el, t_el)>
        if let Some((el_pres, _t_el)) = wx_math::thermo::el(&p_prof, &t_prof, &td_prof) {
            if el_pres > 0.0 {
                *el_v = wx_math::thermo::get_height_at_pres(el_pres, &p_prof, &h_prof);
            }
        }
    });

    Ok(el)
}

/// cape2d: backward-compatible with wrf-python. Returns `[cape, cin, lcl, lfc]` interleaved (4 * nxy).
pub fn compute_cape2d(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "ml");
    let (cape, cin, lcl, lfc) = compute_cape_fields(f, t, &pt, opts.top_m)?;
    let mut out = cape;
    out.extend(cin);
    out.extend(lcl);
    out.extend(lfc);
    Ok(out)
}

/// cape3d: column CAPE at every level (3-D). Uses the column method for each parcel starting level.
pub fn compute_cape3d(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let _psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let _t2_c: Vec<f64> = f.t2(t)?.iter().map(|t| t - 273.15).collect();

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // For each column, compute CAPE for a parcel starting at each level
    let mut cape3d = vec![0.0f64; nz * nxy];

    cape3d.par_chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        for ij in 0..nxy {
            let mut p_prof = Vec::with_capacity(nz - k);
            let mut t_prof = Vec::with_capacity(nz - k);
            let mut td_prof = Vec::with_capacity(nz - k);
            let mut h_prof = Vec::with_capacity(nz - k);

            for kk in k..nz {
                let idx = kk * nxy + ij;
                p_prof.push(pres_hpa[idx]);
                t_prof.push(tc[idx]);
                let q = qv[idx].max(1e-10);
                let e = q * pres_hpa[idx] / (0.622 + q);
                let ln_e = (e / 6.112).max(1e-10).ln();
                td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
                h_prof.push(h_agl[idx]);
            }

            if p_prof.len() >= 2 {
                let (c, _, _, _) = wx_math::thermo::cape_cin_core(
                    &p_prof, &t_prof, &td_prof, &h_prof,
                    p_prof[0], t_prof[0], td_prof[0],
                    "sb", 100.0, 300.0, None,
                );
                plane[ij] = c;
            }
        }
    });

    Ok(cape3d)
}
