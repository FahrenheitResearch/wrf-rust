//! Storm-relative helicity and bulk shear diagnostics.
//!
//! Uses proper Bunkers storm motion (crate::met), NOT wrf-python's
//! broken 0.75*(3-10km mean wind) rotated 30 degrees.

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const SURFACE_LAYER_HEIGHT_M: f64 = 0.0;

/// Canonical SRH entry point for all grid-based SRH computations.
///
/// Applies earth-rotation (SINALPHA/COSALPHA) to both 3-D and 10-m winds,
/// prepends U10/V10 as the surface layer, and then computes SRH via Bunkers RM
/// (or a caller-supplied storm motion).  All SRH consumers -- including
/// STP, SCP, EHI, and effective SRH -- should funnel through this function
/// so that every path sees the same wind preparation.
pub fn compute_srh_field(
    f: &WrfFile,
    t: usize,
    depth_m: f64,
    storm_motion: Option<&crate::compute::StormMotion>,
) -> WrfResult<Vec<f64>> {
    // Use earth-rotated winds for SRH (matches SHARPpy/MetPy convention)
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nxy = nx * ny;

    // Rotate 3D winds to earth coordinates
    let mut u = vec![0.0f64; u_grid.len()];
    let mut v = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    // Rotate 10m winds to earth coordinates
    let mut u10 = vec![0.0f64; nxy];
    let mut v10 = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let nz = f.nz;

    if let Some(storm_motion) = storm_motion {
        // Custom storm motion: compute column-by-column
        let mut srh = vec![0.0f64; nxy];
        srh.iter_mut().enumerate().for_each(|(ij, srh_val)| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10[ij]);
            v_prof.push(v10[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u[idx]);
                v_prof.push(v[idx]);
                h_prof.push(h_agl[idx]);
            }

            let (sm_u, sm_v) = storm_motion.at(ij);
            let (_, _, total) = crate::met::wind::storm_relative_helicity(
                &u_prof, &v_prof, &h_prof, depth_m, sm_u, sm_v,
            );
            *srh_val = total;
        });
        Ok(srh)
    } else {
        // Default: use grid-parallel SRH with Bunkers
        // Prepend 10m winds as the surface layer for each column.
        let nz_aug = nz + 1;
        let mut u_aug = Vec::with_capacity(nz_aug * nxy);
        let mut v_aug = Vec::with_capacity(nz_aug * nxy);
        let mut h_aug = Vec::with_capacity(nz_aug * nxy);
        let mut p_aug = Vec::with_capacity(nz_aug * nxy);

        // Level 0: 10m winds anchored to the surface, with surface pressure.
        for ij in 0..nxy {
            u_aug.push(u10[ij]);
            v_aug.push(v10[ij]);
            h_aug.push(SURFACE_LAYER_HEIGHT_M);
            p_aug.push(psfc_hpa[ij]);
        }
        // Levels 1..nz: model levels
        for k in 0..nz {
            let off = k * nxy;
            for ij in 0..nxy {
                u_aug.push(u[off + ij]);
                v_aug.push(v[off + ij]);
                h_aug.push(h_agl[off + ij]);
                p_aug.push(pres_hpa[off + ij]);
            }
        }

        Ok(crate::met::composite::compute_srh_with_pressure(
            &u_aug, &v_aug, &h_aug, &p_aug, nx, ny, nz_aug, depth_m,
        ))
    }
}

/// Helper: compute bulk shear magnitude for a given layer.
fn compute_shear_field(f: &WrfFile, t: usize, bottom_m: f64, top_m: f64) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let _nxy = nx * ny;

    Ok(crate::met::composite::compute_shear(
        &u, &v, &h_agl, nx, ny, nz, bottom_m, top_m,
    ))
}

/// Helper: compute Bunkers storm motion for each column.
/// Returns (rm_u, rm_v, lm_u, lm_v, mean_u, mean_v) as 6 interleaved nxy fields.
///
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
fn compute_bunkers_columns(
    f: &WrfFile,
    t: usize,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Rotate 3D winds to earth coordinates
    let mut u = vec![0.0f64; u_grid.len()];
    let mut v = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    // Rotate 10m winds to earth coordinates
    let mut u10 = vec![0.0f64; nxy];
    let mut v10 = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let mut rm_u = vec![0.0f64; nxy];
    let mut rm_v = vec![0.0f64; nxy];
    let mut lm_u = vec![0.0f64; nxy];
    let mut lm_v = vec![0.0f64; nxy];
    let mut mn_u = vec![0.0f64; nxy];
    let mut mn_v = vec![0.0f64; nxy];

    // Process columns in parallel
    let results: Vec<_> = (0..nxy)
        .into_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10[ij]);
            v_prof.push(v10[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u[idx]);
                v_prof.push(v[idx]);
                h_prof.push(h_agl[idx]);
            }

            let ((ru, rv), (lu, lv), (mu, mv)) =
                crate::met::wind::bunkers_storm_motion(&u_prof, &v_prof, &h_prof);
            (ij, ru, rv, lu, lv, mu, mv)
        })
        .collect();

    for (ij, ru, rv, lu, lv, mu, mv) in results {
        rm_u[ij] = ru;
        rm_v[ij] = rv;
        lm_u[ij] = lu;
        lm_v[ij] = lv;
        mn_u[ij] = mu;
        mn_v[ij] = mv;
    }

    Ok((rm_u, rm_v, lm_u, lm_v, mn_u, mn_v))
}

// ── Public compute functions ──

/// 0-1 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh1(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(f, t, 1000.0, opts.storm_motion.as_ref())
}

/// 0-3 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh3(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(f, t, 3000.0, opts.storm_motion.as_ref())
}

/// SRH with configurable depth (default 3000m). `[ny, nx]`
pub fn compute_srh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let depth = opts.depth_m.unwrap_or(3000.0);
    compute_srh_field(f, t, depth, opts.storm_motion.as_ref())
}

/// 0-1 km bulk wind shear magnitude (m/s). `[ny, nx]`
pub fn compute_shear_0_1km(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_shear_field(f, t, 0.0, 1000.0)
}

/// 0-6 km bulk wind shear magnitude (m/s). `[ny, nx]`
pub fn compute_shear_0_6km(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_shear_field(f, t, 0.0, 6000.0)
}

/// Bunkers right-mover storm motion (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_bunkers_rm(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (rm_u, rm_v, _, _, _, _) = compute_bunkers_columns(f, t)?;
    let mut out = rm_u;
    out.extend(rm_v);
    Ok(out)
}

/// Bunkers left-mover storm motion (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_bunkers_lm(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, lm_u, lm_v, _, _) = compute_bunkers_columns(f, t)?;
    let mut out = lm_u;
    out.extend(lm_v);
    Ok(out)
}

/// 0-6 km mean wind (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_mean_wind_0_6km(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, mn_u, mn_v) = compute_bunkers_columns(f, t)?;
    let mut out = mn_u;
    out.extend(mn_v);
    Ok(out)
}

/// Effective inflow layer SRH (m^2/s^2). `[ny, nx]`
///
/// Finds the effective inflow layer where CAPE >= 100 J/kg and CIN >= -250 J/kg,
/// then computes storm-relative helicity over that layer using Bunkers storm motion
/// (or custom motion if `opts.storm_motion` is set).
///
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
pub fn compute_effective_srh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    // Earth-rotated 3D winds
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Rotate 3D winds to earth coordinates
    let mut u = vec![0.0f64; u_grid.len()];
    let mut v = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    // Rotate 10m winds to earth coordinates
    let mut u10 = vec![0.0f64; nxy];
    let mut v10 = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let custom_sm = opts.storm_motion.as_ref();

    let mut srh = vec![0.0f64; nxy];
    srh.iter_mut().enumerate().for_each(|(ij, srh_val)| {
        // Build augmented column profiles with a surface-layer prepend.
        let mut p_prof = Vec::with_capacity(nz + 1);
        let mut t_prof = Vec::with_capacity(nz + 1);
        let mut td_prof = Vec::with_capacity(nz + 1);
        let mut h_prof = Vec::with_capacity(nz + 1);
        let mut u_prof = Vec::with_capacity(nz + 1);
        let mut v_prof = Vec::with_capacity(nz + 1);

        // Prepend 10m level: use surface pressure and lowest-level T/Td as approximation.
        let idx0 = ij; // k=0 level
        let q0 = qv[idx0].max(1e-10);
        let e0 = q0 * pres_hpa[idx0] / (0.622 + q0);
        let ln_e0 = (e0 / 6.112).max(1e-10).ln();
        let td0 = (243.5 * ln_e0) / (17.67 - ln_e0);

        p_prof.push(pres_hpa[idx0]); // approximate surface pressure
        t_prof.push(tc[idx0]);
        td_prof.push(td0);
        h_prof.push(SURFACE_LAYER_HEIGHT_M);
        u_prof.push(u10[ij]);
        v_prof.push(v10[ij]);

        for k in 0..nz {
            let idx = k * nxy + ij;
            p_prof.push(pres_hpa[idx]);
            t_prof.push(tc[idx]);
            // Dewpoint from mixing ratio
            let q = qv[idx].max(1e-10);
            let e = q * pres_hpa[idx] / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            td_prof.push((243.5 * ln_e) / (17.67 - ln_e));
            h_prof.push(h_agl[idx]);
            u_prof.push(u[idx]);
            v_prof.push(v[idx]);
        }

        // Find effective inflow layer bounds by testing CAPE/CIN at each level
        let nz_aug = nz + 1;
        let mut eff_base: Option<usize> = None;
        let mut eff_top: usize = 0;

        for k in 0..nz_aug {
            if nz_aug - k < 2 {
                break;
            }

            // Compute CAPE/CIN for a parcel lifted from level k
            let (cape_k, cin_k, _, _) = crate::met::thermo::cape_cin_core(
                &p_prof[k..],
                &t_prof[k..],
                &td_prof[k..],
                &h_prof[k..],
                p_prof[k],
                t_prof[k],
                td_prof[k],
                "sb",
                100.0,
                300.0,
                None,
            );

            if cape_k >= 100.0 && cin_k >= -250.0 {
                if eff_base.is_none() {
                    eff_base = Some(k);
                }
                eff_top = k;
            } else if eff_base.is_some() {
                // Effective layer must be continuous; stop at first failure
                break;
            }
        }

        // If no effective layer found, SRH = 0
        let base_k = match eff_base {
            Some(k) => k,
            None => return,
        };

        let eff_base_h = h_prof[base_k];
        let eff_top_h = h_prof[eff_top];

        if eff_top_h <= eff_base_h {
            return;
        }

        // Trim profiles to start from effective base
        let u_eff: Vec<f64> = u_prof[base_k..].to_vec();
        let v_eff: Vec<f64> = v_prof[base_k..].to_vec();
        let h_eff: Vec<f64> = h_prof[base_k..].to_vec();

        // Get storm motion (from full augmented profile)
        let (sm_u, sm_v) = if let Some(sm) = custom_sm {
            sm.at(ij)
        } else {
            let ((ru, rv), _, _) =
                crate::met::wind::bunkers_storm_motion(&u_prof, &v_prof, &h_prof);
            (ru, rv)
        };

        // storm_relative_helicity interprets depth_m as absolute AGL,
        // so pass eff_top_h directly
        let (_, _, total) = crate::met::wind::storm_relative_helicity(
            &u_eff, &v_eff, &h_eff, eff_top_h, sm_u, sm_v,
        );
        *srh_val = total;
    });

    Ok(srh)
}

/// Configurable bulk wind shear magnitude (m/s). `[ny, nx]`
///
/// Uses `opts.bottom_m` (default 0) and `opts.top_m` (default 6000) for the layer.
pub fn compute_bulk_shear(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let bottom = opts.bottom_m.unwrap_or(0.0);
    let top = opts.top_m.unwrap_or(6000.0);
    compute_shear_field(f, t, bottom, top)
}

/// Configurable mean wind (m/s). Returns `[u_mean, v_mean]` interleaved (2 * nxy). `[ny, nx]` per component.
///
/// Uses `opts.bottom_m` (default 0) and `opts.top_m` (default 6000) for the layer.
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
pub fn compute_mean_wind(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    // Rotate 3D winds to earth coordinates
    let mut u = vec![0.0f64; u_grid.len()];
    let mut v = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    // Rotate 10m winds to earth coordinates
    let mut u10 = vec![0.0f64; nxy];
    let mut v10 = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let bottom = opts.bottom_m.unwrap_or(0.0);
    let top = opts.top_m.unwrap_or(6000.0);

    let mut mean_u = vec![0.0f64; nxy];
    let mut mean_v = vec![0.0f64; nxy];

    let results: Vec<_> = (0..nxy)
        .into_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10[ij]);
            v_prof.push(v10[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u[idx]);
                v_prof.push(v[idx]);
                h_prof.push(h_agl[idx]);
            }

            let (mu, mv) = crate::met::wind::mean_wind(&u_prof, &v_prof, &h_prof, bottom, top);
            (ij, mu, mv)
        })
        .collect();

    for (ij, mu, mv) in results {
        mean_u[ij] = mu;
        mean_v[ij] = mv;
    }

    let mut out = mean_u;
    out.extend(mean_v);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::SURFACE_LAYER_HEIGHT_M;

    #[test]
    fn surface_augmentation_anchors_10m_winds_at_zero_agl() {
        assert_eq!(SURFACE_LAYER_HEIGHT_M, 0.0);
    }
}
