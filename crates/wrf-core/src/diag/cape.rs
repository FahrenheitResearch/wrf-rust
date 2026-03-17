//! CAPE diagnostic variables: sbcape, mlcape, mucape, cape2d, cape3d, lcl, lfc, el
//!
//! Uses crate::met::composite::compute_cape_cin() for parallel grid computation
//! and crate::met::thermo::cape_cin_core() for column-by-column fallback.

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
    lake_interp: Option<f64>,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pres = f.full_pressure(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = match lake_interp {
        Some(a) if a > 0.0 => f.t2_lake_corrected(t, a)?,
        _ => f.t2(t)?,
    };
    let q2 = match lake_interp {
        Some(a) if a > 0.0 => f.q2_lake_corrected(t, a)?,
        _ => f.q2(t)?,
    };

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Convert pressure from Pa to hPa for thermo functions
    let pres_hpa: Vec<f64> = pres.iter().map(|p| p / 100.0).collect();
    // Convert surface pressure from Pa to hPa
    let psfc_hpa: Vec<f64> = psfc.iter().map(|p| p / 100.0).collect();
    // Convert T2 from K to C
    let t2_c: Vec<f64> = t2.iter().map(|t| t - 273.15).collect();

    // Use crate::met's grid-parallel CAPE computation if no top_m specified
    if top_m.is_none() {
        let (cape, cin, lcl, lfc) = crate::met::composite::compute_cape_cin(
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

            let (c, ci, l, lf) = crate::met::thermo::cape_cin_core(
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
    let (cape, _, _, _) = compute_cape_fields(f, t, "sb", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_sbcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "sb", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_mlcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "ml", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_mlcin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "ml", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_mucape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = compute_cape_fields(f, t, "mu", opts.top_m, opts.lake_interp)?;
    Ok(cape)
}

pub fn compute_mucin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = compute_cape_fields(f, t, "mu", opts.top_m, opts.lake_interp)?;
    Ok(cin)
}

pub fn compute_lcl(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, lcl, _) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
    Ok(lcl)
}

pub fn compute_lfc(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "sb");
    let (_, _, _, lfc) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
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

        // Use crate::met el function — returns Option<(p_el, t_el)>
        if let Some((el_pres, _t_el)) = crate::met::thermo::el(&p_prof, &t_prof, &td_prof) {
            if el_pres > 0.0 {
                *el_v = crate::met::thermo::get_height_at_pres(el_pres, &p_prof, &h_prof);
            }
        }
    });

    Ok(el)
}

/// cape2d: backward-compatible with wrf-python. Returns `[cape, cin, lcl, lfc]` interleaved (4 * nxy).
pub fn compute_cape2d(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let pt = resolve_parcel_type(opts, "ml");
    let (cape, cin, lcl, lfc) = compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)?;
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
                let (c, _, _, _) = crate::met::thermo::cape_cin_core(
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

// ── Custom parcel helper ──

/// Compute CAPE fields using a custom parcel (user-specified pressure, temperature, dewpoint).
/// Column-by-column, using "sb" parcel type but substituting custom values for psfc/t2m/td2m.
/// Returns (cape_2d, cin_2d, lcl_2d, lfc_2d).
fn compute_cape_fields_custom(
    f: &WrfFile,
    t: usize,
    parcel_p_hpa: f64,
    parcel_t_c: f64,
    parcel_td_c: f64,
    top_m: Option<f64>,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

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

            // Use "sb" parcel type but pass custom parcel values as psfc/t2m/td2m
            let (c, ci, l, lf) = crate::met::thermo::cape_cin_core(
                &p_prof, &t_prof, &td_prof, &h_prof,
                parcel_p_hpa, parcel_t_c, parcel_td_c,
                "sb", 100.0, 300.0, top_m,
            );

            *cape_v = c;
            *cin_v = ci;
            *lcl_v = l;
            *lfc_v = lf;
        });

    Ok((cape, cin, lcl, lfc))
}

/// Determine whether opts specifies a custom parcel (all three of pressure, temperature,
/// and dewpoint are set).
fn has_custom_parcel(opts: &ComputeOpts) -> bool {
    opts.parcel_pressure.is_some()
        && opts.parcel_temperature.is_some()
        && opts.parcel_dewpoint.is_some()
}

/// Resolve CAPE fields for generic functions: custom parcel, named parcel type, or default "sb".
fn resolve_cape_fields(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if has_custom_parcel(opts) {
        let p = opts.parcel_pressure.unwrap();
        let tc = opts.parcel_temperature.unwrap();
        let td = opts.parcel_dewpoint.unwrap();
        compute_cape_fields_custom(f, t, p, tc, td, opts.top_m)
    } else {
        let pt = resolve_parcel_type(opts, "sb");
        compute_cape_fields(f, t, &pt, opts.top_m, opts.lake_interp)
    }
}

// ── Generic public compute functions ──

/// Generic CAPE computation.
///
/// - If `opts.parcel_pressure`, `opts.parcel_temperature`, and `opts.parcel_dewpoint` are all
///   set, uses a custom parcel lifted from the specified conditions.
/// - Otherwise falls back to `opts.parcel_type` ("sb", "ml", "mu"), defaulting to "sb".
/// - Supports `opts.top_m` for truncated CAPE (e.g., 0-3 km CAPE).
pub fn compute_cape_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (cape, _, _, _) = resolve_cape_fields(f, t, opts)?;
    Ok(cape)
}

/// Generic CIN computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_cin_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, cin, _, _) = resolve_cape_fields(f, t, opts)?;
    Ok(cin)
}

/// Generic LCL height computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_lcl_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, lcl, _) = resolve_cape_fields(f, t, opts)?;
    Ok(lcl)
}

/// Generic LFC height computation. Same parcel logic as `compute_cape_generic`.
pub fn compute_lfc_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, lfc) = resolve_cape_fields(f, t, opts)?;
    Ok(lfc)
}

/// Generic EL height computation. Same parcel logic as `compute_cape_generic`.
///
/// For custom parcels, column-by-column EL is computed via `crate::met::thermo::el` using the
/// custom parcel's temperature and dewpoint as the surface values in the profile.
pub fn compute_el_generic(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    if has_custom_parcel(opts) {
        let parcel_p = opts.parcel_pressure.unwrap();
        let parcel_t = opts.parcel_temperature.unwrap();
        let parcel_td = opts.parcel_dewpoint.unwrap();

        let pres_hpa = f.pressure_hpa(t)?;
        let tc = f.temperature_c(t)?;
        let qv = f.qvapor(t)?;
        let h_agl = f.height_agl(t)?;

        let nx = f.nx;
        let ny = f.ny;
        let nz = f.nz;
        let nxy = nx * ny;

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

            // Build a modified profile starting from the custom parcel level:
            // insert the custom parcel at the base and use el() to find the EL.
            let mut mod_p = vec![parcel_p];
            let mut mod_t = vec![parcel_t];
            let mut mod_td = vec![parcel_td];

            for k in 0..nz {
                if p_prof[k] < parcel_p {
                    mod_p.push(p_prof[k]);
                    mod_t.push(t_prof[k]);
                    mod_td.push(td_prof[k]);
                }
            }

            if mod_p.len() >= 2 {
                if let Some((el_pres, _)) = crate::met::thermo::el(&mod_p, &mod_t, &mod_td) {
                    if el_pres > 0.0 {
                        *el_v = crate::met::thermo::get_height_at_pres(el_pres, &p_prof, &h_prof);
                    }
                }
            }
        });

        Ok(el)
    } else {
        // Delegate to existing EL function for named parcel types
        compute_el(f, t, opts)
    }
}

// ── Effective inflow layer ──

/// Compute the effective inflow layer bounds for each grid column.
///
/// Returns a flat vector of length `2 * nxy` with interleaved
/// `[eff_base_height, eff_top_height]` in meters AGL.
///
/// Effective inflow base: lowest model level where a parcel lifted from that level
/// produces CAPE >= 100 J/kg AND CIN >= -250 J/kg.
///
/// Effective inflow top: equilibrium level of the most-unstable parcel within the
/// effective layer (approximated as the highest level still meeting the CAPE/CIN criteria).
pub fn compute_effective_inflow_layer(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut result = vec![0.0f64; 2 * nxy];

    result.par_chunks_mut(2).enumerate().for_each(|(ij, out)| {
        // Extract column profiles
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

        let mut eff_base: Option<usize> = None;
        let mut eff_top: Option<usize> = None;
        let mut mu_cape = 0.0f64;
        let mut mu_level: Option<usize> = None;

        // Scan upward from surface
        for k in 0..nz {
            if p_prof.len() - k < 2 {
                break;
            }

            // Lift parcel from level k
            let (c, ci, _, _) = crate::met::thermo::cape_cin_core(
                &p_prof[k..], &t_prof[k..], &td_prof[k..], &h_prof[k..],
                p_prof[k], t_prof[k], td_prof[k],
                "sb", 100.0, 300.0, None,
            );

            if c >= 100.0 && ci >= -250.0 {
                if eff_base.is_none() {
                    eff_base = Some(k);
                }
                eff_top = Some(k);

                if c > mu_cape {
                    mu_cape = c;
                    mu_level = Some(k);
                }
            } else if eff_base.is_some() {
                // Once we leave the effective layer, stop scanning
                break;
            }
        }

        if let (Some(base_k), Some(_top_k)) = (eff_base, eff_top) {
            out[0] = h_prof[base_k]; // effective inflow base height AGL

            // Effective inflow top = EL of the most-unstable parcel in the layer
            if let Some(mu_k) = mu_level {
                let mu_p_prof = &p_prof[mu_k..];
                let mu_t_prof = &t_prof[mu_k..];
                let mu_td_prof = &td_prof[mu_k..];
                let _mu_h_prof = &h_prof[mu_k..];

                if let Some((el_pres, _)) =
                    crate::met::thermo::el(mu_p_prof, mu_t_prof, mu_td_prof)
                {
                    if el_pres > 0.0 {
                        out[1] = crate::met::thermo::get_height_at_pres(
                            el_pres, &p_prof, &h_prof,
                        );
                    }
                }
                // If el() returned None, out[1] stays 0.0
            }
        }
        // If no effective layer found, both out[0] and out[1] remain 0.0
    });

    Ok(result)
}

/// Compute the CAPE of the most-unstable parcel within the effective inflow layer.
///
/// For each column, scans upward from the surface computing CAPE/CIN for a parcel
/// lifted from each model level. The effective inflow layer is defined as the
/// contiguous region where CAPE >= 100 J/kg AND CIN >= -250 J/kg. Returns the CAPE
/// of the most-unstable parcel within that layer. If no effective layer is found,
/// returns 0.0 for that column.
pub fn compute_effective_inflow_cape(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut result = vec![0.0f64; nxy];

    result.par_iter_mut().enumerate().for_each(|(ij, out)| {
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

        let mut found_base = false;
        let mut mu_cape = 0.0f64;

        for k in 0..nz {
            if p_prof.len() - k < 2 {
                break;
            }

            let (c, ci, _, _) = crate::met::thermo::cape_cin_core(
                &p_prof[k..], &t_prof[k..], &td_prof[k..], &h_prof[k..],
                p_prof[k], t_prof[k], td_prof[k],
                "sb", 100.0, 300.0, None,
            );

            if c >= 100.0 && ci >= -250.0 {
                found_base = true;
                if c > mu_cape {
                    mu_cape = c;
                }
            } else if found_base {
                break;
            }
        }

        *out = mu_cape;
    });

    Ok(result)
}
