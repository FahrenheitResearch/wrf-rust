//! Storm-relative helicity and bulk shear diagnostics.
//!
//! The default public SRH variables retain BowEcho/WRF-Runner's Bunkers
//! semantics. `srh_wrfpython` is a separate, explicitly named compatibility
//! path for NCAR wrf-python's legacy RIP `DCALRELHL` algorithm.

use crate::compute::{ComputeOpts, StormMotionMethod};
use crate::diag::cape::{build_surface_augmented_thermo_column, effective_inflow_layer_grid};
use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;
use rayon::prelude::*;

const SURFACE_LAYER_HEIGHT_M: f64 = 0.0;
const BUNKERS_STACK_FIELDS: usize = 6;
const WRFPYTHON_GRAVITY_M_S2: f64 = 9.81;
const WRFPYTHON_MEAN_BOTTOM_M: f64 = 3_000.0;
const WRFPYTHON_MEAN_TOP_M: f64 = 10_000.0;
const WRFPYTHON_STORM_SPEED_FACTOR: f64 = 0.75;
const WRFPYTHON_STORM_TURN_DEG: f64 = 30.0;

fn resolved_storm_motion_method(opts: &ComputeOpts) -> StormMotionMethod {
    opts.storm_motion_method
        .unwrap_or(StormMotionMethod::PressureWeighted)
}

fn bunkers_cache_key(method: StormMotionMethod) -> &'static str {
    match method {
        StormMotionMethod::PressureWeighted => "bunkers_stack_pw",
        StormMotionMethod::NonPressureWeighted => "bunkers_stack_npw",
    }
}

fn pack_bunkers_stack(
    rm_u: &[f64],
    rm_v: &[f64],
    lm_u: &[f64],
    lm_v: &[f64],
    mn_u: &[f64],
    mn_v: &[f64],
) -> Vec<f64> {
    let nxy = rm_u.len();
    let mut stacked = Vec::with_capacity(BUNKERS_STACK_FIELDS * nxy);
    stacked.extend_from_slice(rm_u);
    stacked.extend_from_slice(rm_v);
    stacked.extend_from_slice(lm_u);
    stacked.extend_from_slice(lm_v);
    stacked.extend_from_slice(mn_u);
    stacked.extend_from_slice(mn_v);
    stacked
}

fn unpack_bunkers_stack(
    stacked: &[f64],
    nxy: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if stacked.len() != BUNKERS_STACK_FIELDS * nxy {
        return None;
    }

    Some((
        stacked[0..nxy].to_vec(),
        stacked[nxy..2 * nxy].to_vec(),
        stacked[2 * nxy..3 * nxy].to_vec(),
        stacked[3 * nxy..4 * nxy].to_vec(),
        stacked[4 * nxy..5 * nxy].to_vec(),
        stacked[5 * nxy..6 * nxy].to_vec(),
    ))
}

/// Locate the model levels used by NCAR wrf-python's RIP `DCALRELHL` routine.
///
/// Heights are surface-to-top mass-level heights MSL. The returned indices are
/// the first level above 3 km AGL, the first level above 10 km AGL (or the
/// penultimate model level), and the first level above the requested SRH top.
/// The strict `>` comparisons and the early stop at 10 km reproduce the
/// Fortran loop rather than interpolating exact layer endpoints.
fn wrfpython_rip_level_bounds(
    height_msl: &[f64],
    terrain_m: f64,
    top_m: f64,
) -> Option<(usize, usize, usize)> {
    let n = height_msl.len();
    if n < 3 || !terrain_m.is_finite() || !top_m.is_finite() {
        return None;
    }

    let heights_are_usable = height_msl
        .iter()
        .enumerate()
        .all(|(k, height)| height.is_finite() && (k == 0 || *height > height_msl[k - 1]));
    if !heights_are_usable {
        return None;
    }

    let mut level_3km = None;
    let mut level_10km = None;
    let mut level_top = None;

    // The upstream Fortran receives a top-to-surface array and scans it in
    // reverse. WRF data here are surface-to-top, so this is the same traversal.
    // The last mass level is excluded exactly as in `DO k = mkzh, 2, -1`.
    for k in 0..n - 1 {
        let height_agl = height_msl[k] - terrain_m;
        if height_agl > WRFPYTHON_MEAN_TOP_M {
            level_10km = Some(k);
            break;
        }
        if height_agl > top_m && level_top.is_none() {
            level_top = Some(k);
        }
        if height_agl > WRFPYTHON_MEAN_BOTTOM_M && level_3km.is_none() {
            level_3km = Some(k);
        }
    }

    let level_10km = level_10km.unwrap_or(n - 2);
    let level_3km = level_3km?;
    let level_top = level_top?;
    (level_3km <= level_10km).then_some((level_3km, level_10km, level_top))
}

/// Legacy RIP storm motion used by NCAR wrf-python `DCALRELHL`.
fn wrfpython_rip_storm_motion(
    u_prof: &[f64],
    v_prof: &[f64],
    height_msl: &[f64],
    level_3km: usize,
    level_10km: usize,
    latitude_deg: f64,
) -> Option<(f64, f64)> {
    let n = u_prof.len();
    if n < 3
        || v_prof.len() != n
        || height_msl.len() != n
        || level_3km > level_10km
        || level_10km + 1 >= n
        || !latitude_deg.is_finite()
    {
        return None;
    }

    let mut depth_sum = 0.0;
    let mut u_sum = 0.0;
    let mut v_sum = 0.0;
    for k in level_3km..=level_10km {
        let depth = height_msl[k + 1] - height_msl[k];
        depth_sum += depth;
        u_sum += 0.5 * depth * (u_prof[k + 1] + u_prof[k]);
        v_sum += 0.5 * depth * (v_prof[k + 1] + v_prof[k]);
    }
    if !depth_sum.is_finite() || depth_sum <= 0.0 {
        return None;
    }

    let mean_u = u_sum / depth_sum;
    let mean_v = v_sum / depth_sum;
    if !mean_u.is_finite() || !mean_v.is_finite() {
        return None;
    }

    // Preserve the meteorological-direction conversion and branch order from
    // wrf_relhl.f90 instead of replacing it with a modern Bunkers deviation.
    let mean_speed = mean_u.hypot(mean_v);
    let mean_direction_deg = if mean_u == 0.0 && mean_v == 0.0 {
        0.0
    } else {
        180.0 / std::f64::consts::PI * (std::f64::consts::PI + mean_u.atan2(mean_v))
    };
    let storm_speed = WRFPYTHON_STORM_SPEED_FACTOR * mean_speed;
    let mut storm_direction_deg = if latitude_deg >= 0.0 {
        mean_direction_deg + WRFPYTHON_STORM_TURN_DEG
    } else {
        mean_direction_deg - WRFPYTHON_STORM_TURN_DEG
    };
    if storm_direction_deg > 360.0 {
        storm_direction_deg -= 360.0;
    }

    let storm_direction_rad = storm_direction_deg * std::f64::consts::PI / 180.0;
    Some((
        -storm_speed * storm_direction_rad.sin(),
        -storm_speed * storm_direction_rad.cos(),
    ))
}

/// One-column port of NCAR wrf-python 1.3.4.1 `DCALRELHL`.
fn wrfpython_rip_srh_column(
    u_prof: &[f64],
    v_prof: &[f64],
    height_msl: &[f64],
    terrain_m: f64,
    latitude_deg: f64,
    top_m: f64,
) -> f64 {
    let n = u_prof.len();
    if n < 3
        || v_prof.len() != n
        || height_msl.len() != n
        || u_prof.iter().any(|value| !value.is_finite())
        || v_prof.iter().any(|value| !value.is_finite())
    {
        return 0.0;
    }

    let (level_3km, level_10km, level_top) =
        match wrfpython_rip_level_bounds(height_msl, terrain_m, top_m) {
            Some(bounds) => bounds,
            None => return 0.0,
        };
    let (storm_u, storm_v) = match wrfpython_rip_storm_motion(
        u_prof,
        v_prof,
        height_msl,
        level_3km,
        level_10km,
        latitude_deg,
    ) {
        Some(motion) => motion,
        None => return 0.0,
    };

    // Upstream starts with the second-lowest mass level and includes the first
    // model level strictly above `top_m`; it does not prepend U10 or interpolate.
    if level_top == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 1..=level_top {
        let contribution = (u_prof[k] - storm_u) * (v_prof[k] - v_prof[k - 1])
            - (v_prof[k] - storm_v) * (u_prof[k] - u_prof[k - 1]);
        sum += contribution;
    }
    -sum
}

/// Strict NCAR wrf-python/RIP SRH compatibility path. `[ny, nx]`
///
/// This intentionally does **not** share the Bunkers preparation path:
/// wrf-python uses grid-relative destaggered mass-level winds, no U10 surface
/// anchor, geopotential height divided by its 9.81 m/s^2 constant, a discrete
/// 3--10-km mean, 0.75 storm speed, and a latitude-dependent 30-degree turn.
/// `depth_m` defaults to 3000 m. `storm_motion` and `storm_motion_method` are
/// rejected because accepting them would no longer be strict compatibility.
pub fn compute_srh_wrfpython(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    if opts.storm_motion.is_some() || opts.storm_motion_method.is_some() {
        return Err(WrfError::InvalidParam(
            "srh_wrfpython has fixed NCAR RIP storm-motion semantics; use srh for custom or Bunkers motion"
                .to_string(),
        ));
    }

    let top_m = opts.depth_m.unwrap_or(3_000.0);
    if !top_m.is_finite() || top_m <= 0.0 || top_m >= WRFPYTHON_MEAN_TOP_M {
        return Err(WrfError::InvalidParam(format!(
            "srh_wrfpython depth_m must be finite and in (0, 10000) m, got {top_m}"
        )));
    }

    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let geopotential = f.full_geopotential(t)?;
    let terrain = f.terrain(t)?;
    let latitude = f.xlat(t)?;
    let nz = f.nz;
    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map_init(
            || {
                (
                    Vec::with_capacity(nz),
                    Vec::with_capacity(nz),
                    Vec::with_capacity(nz),
                )
            },
            |(u_prof, v_prof, height_prof), ij| {
                u_prof.clear();
                v_prof.clear();
                height_prof.clear();
                for k in 0..nz {
                    let idx = k * nxy + ij;
                    u_prof.push(u[idx]);
                    v_prof.push(v[idx]);
                    height_prof.push(geopotential[idx] / WRFPYTHON_GRAVITY_M_S2);
                }
                wrfpython_rip_srh_column(
                    u_prof,
                    v_prof,
                    height_prof,
                    terrain[ij],
                    latitude[ij],
                    top_m,
                )
            },
        )
        .collect())
}

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
    storm_motion_method: Option<StormMotionMethod>,
) -> WrfResult<Vec<f64>> {
    // Use earth-rotated winds for SRH (matches SHARPpy/MetPy convention)
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let h_agl = f.height_agl(t)?;
    let pres_hpa = f.pressure_hpa(t)?;
    let psfc_hpa: Vec<f64> = f.psfc(t)?.iter().map(|p| p / 100.0).collect();
    let latitude = f.xlat(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nxy = nx * ny;

    let nz = f.nz;

    if let Some(storm_motion) = storm_motion {
        // Custom storm motion: compute column-by-column
        let mut srh = vec![0.0f64; nxy];
        srh.iter_mut().enumerate().for_each(|(ij, srh_val)| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
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
            u_aug.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_aug.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_aug.push(SURFACE_LAYER_HEIGHT_M);
            p_aug.push(psfc_hpa[ij]);
        }
        // Levels 1..nz: model levels
        for k in 0..nz {
            let off = k * nxy;
            for ij in 0..nxy {
                u_aug.push(u_grid[off + ij] * cosa[ij] - v_grid[off + ij] * sina[ij]);
                v_aug.push(u_grid[off + ij] * sina[ij] + v_grid[off + ij] * cosa[ij]);
                h_aug.push(h_agl[off + ij]);
                p_aug.push(pres_hpa[off + ij]);
            }
        }

        if matches!(
            storm_motion_method,
            Some(StormMotionMethod::NonPressureWeighted)
        ) {
            Ok(
                crate::met::composite::compute_srh_with_npw_bunkers_and_latitude(
                    &u_aug, &v_aug, &h_aug, &p_aug, &latitude, nx, ny, nz_aug, depth_m,
                ),
            )
        } else {
            Ok(
                crate::met::composite::compute_srh_with_pressure_and_latitude(
                    &u_aug, &v_aug, &h_aug, &p_aug, &latitude, nx, ny, nz_aug, depth_m,
                ),
            )
        }
    }
}

fn surface_augmented_shear_from_profile(
    u_prof: &[f64],
    v_prof: &[f64],
    h_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> f64 {
    let (du, dv) = crate::met::wind::bulk_shear(u_prof, v_prof, h_prof, bottom_m, top_m);
    du.hypot(dv)
}

/// Compute bulk shear after anchoring the column with the 10-m wind at 0 m AGL.
pub(crate) fn compute_shear_field(
    f: &WrfFile,
    t: usize,
    bottom_m: f64,
    top_m: f64,
) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nz = f.nz;
    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map_init(
            || {
                (
                    Vec::with_capacity(nz + 1),
                    Vec::with_capacity(nz + 1),
                    Vec::with_capacity(nz + 1),
                )
            },
            |(u_prof, v_prof, h_prof), ij| {
                u_prof.clear();
                v_prof.clear();
                h_prof.clear();
                u_prof.push(u10[ij]);
                v_prof.push(v10[ij]);
                h_prof.push(SURFACE_LAYER_HEIGHT_M);

                for k in 0..nz {
                    let idx = k * nxy + ij;
                    u_prof.push(u[idx]);
                    v_prof.push(v[idx]);
                    h_prof.push(h_agl[idx]);
                }

                if h_prof.len() > 2 && h_prof[1] > h_prof[h_prof.len() - 1] {
                    u_prof[1..].reverse();
                    v_prof[1..].reverse();
                    h_prof[1..].reverse();
                }

                surface_augmented_shear_from_profile(u_prof, v_prof, h_prof, bottom_m, top_m)
            },
        )
        .collect())
}

/// Helper: compute Bunkers storm motion for each column.
/// Returns (rm_u, rm_v, lm_u, lm_v, mean_u, mean_v) as 6 interleaved nxy fields.
///
/// Uses earth-rotated winds with 10m prepend, matching compute_srh_field.
fn compute_bunkers_columns(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let storm_motion_method = resolved_storm_motion_method(opts);
    let nxy = f.nx * f.ny;
    let cache_key = bunkers_cache_key(storm_motion_method);
    if let Some(stacked) = f.cached_field(cache_key) {
        if let Some(fields) = unpack_bunkers_stack(&stacked, nxy) {
            return Ok(fields);
        }
    }

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
    let nz = f.nz;
    let nxy = nx * ny;

    let mut rm_u = vec![0.0f64; nxy];
    let mut rm_v = vec![0.0f64; nxy];
    let mut lm_u = vec![0.0f64; nxy];
    let mut lm_v = vec![0.0f64; nxy];
    let mut mn_u = vec![0.0f64; nxy];
    let mut mn_v = vec![0.0f64; nxy];

    let results: Vec<_> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            let mut p_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);
            p_prof.push(psfc_hpa[ij]);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
                h_prof.push(h_agl[idx]);
                p_prof.push(pres_hpa[idx]);
            }

            let ((ru, rv), (lu, lv), (mu, mv)) = match storm_motion_method {
                StormMotionMethod::PressureWeighted => {
                    crate::met::composite::pressure_weighted_bunkers_storm_motion(
                        &h_prof, &u_prof, &v_prof, &p_prof,
                    )
                }
                StormMotionMethod::NonPressureWeighted => {
                    crate::met::wind::bunkers_storm_motion_npw_pressure_resampled(
                        &u_prof, &v_prof, &h_prof, &p_prof,
                    )
                }
            };
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

    let stacked = pack_bunkers_stack(&rm_u, &rm_v, &lm_u, &lm_v, &mn_u, &mn_v);
    f.store_cached_field(cache_key.to_string(), stacked);
    Ok((rm_u, rm_v, lm_u, lm_v, mn_u, mn_v))
}

// ── Public compute functions ──

/// 0-1 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh1(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(
        f,
        t,
        1000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
}

/// 0-3 km SRH (m^2/s^2). `[ny, nx]`
pub fn compute_srh3(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_srh_field(
        f,
        t,
        3000.0,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
}

/// SRH with configurable depth (default 3000m). `[ny, nx]`
pub fn compute_srh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let depth = opts.depth_m.unwrap_or(3000.0);
    compute_srh_field(
        f,
        t,
        depth,
        opts.storm_motion.as_ref(),
        opts.storm_motion_method,
    )
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
pub fn compute_bunkers_rm(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (rm_u, rm_v, _, _, _, _) = compute_bunkers_columns(f, t, opts)?;
    let mut out = rm_u;
    out.extend(rm_v);
    Ok(out)
}

/// Bunkers left-mover storm motion (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_bunkers_lm(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, lm_u, lm_v, _, _) = compute_bunkers_columns(f, t, opts)?;
    let mut out = lm_u;
    out.extend(lm_v);
    Ok(out)
}

/// 0-6 km mean wind (m/s). Returns `[u, v]` interleaved (2 * nxy).
pub fn compute_mean_wind_0_6km(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, mn_u, mn_v) = compute_bunkers_columns(f, t, opts)?;
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
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;
    let latitude = f.xlat(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let effective_layers = effective_inflow_layer_grid(f, t, opts)?;

    let custom_sm = opts.storm_motion.as_ref();
    let storm_motion_method = resolved_storm_motion_method(opts);
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (p_prof, _, _, h_prof) = build_surface_augmented_thermo_column(
                &pres_hpa, &tc, &qv, &h_agl, psfc[ij], t2[ij], q2[ij], nz, nxy, ij,
            );
            let layer = match effective_layers.layer(ij) {
                Some(layer) => layer,
                None => return 0.0,
            };

            if layer.top_h <= layer.base_h {
                return 0.0;
            }

            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
            }

            let (sm_u, sm_v) = if let Some(sm) = custom_sm {
                sm.at(ij)
            } else {
                let (right_mover, left_mover) = match storm_motion_method {
                    StormMotionMethod::PressureWeighted => {
                        let (right, left, _) =
                            crate::met::composite::pressure_weighted_bunkers_storm_motion(
                                &h_prof, &u_prof, &v_prof, &p_prof,
                            );
                        (right, left)
                    }
                    StormMotionMethod::NonPressureWeighted => {
                        let (right, left, _) =
                            crate::met::wind::bunkers_storm_motion_npw_pressure_resampled(
                                &u_prof, &v_prof, &h_prof, &p_prof,
                            );
                        (right, left)
                    }
                };
                crate::met::wind::cyclonic_bunkers_motion(latitude[ij], right_mover, left_mover)
            };

            let (_, _, total) = crate::met::wind::storm_relative_helicity(
                &u_prof[layer.base_idx..],
                &v_prof[layer.base_idx..],
                &h_prof[layer.base_idx..],
                layer.top_h,
                sm_u,
                sm_v,
            );
            total
        })
        .collect())
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

    let bottom = opts.bottom_m.unwrap_or(0.0);
    let top = opts.top_m.unwrap_or(6000.0);

    let mut mean_u = vec![0.0f64; nxy];
    let mut mean_v = vec![0.0f64; nxy];

    let results: Vec<_> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            // Prepend 10m wind as the surface layer.
            let mut u_prof = Vec::with_capacity(nz + 1);
            let mut v_prof = Vec::with_capacity(nz + 1);
            let mut h_prof = Vec::with_capacity(nz + 1);
            u_prof.push(u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij]);
            v_prof.push(u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij]);
            h_prof.push(SURFACE_LAYER_HEIGHT_M);

            for k in 0..nz {
                let idx = k * nxy + ij;
                u_prof.push(u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij]);
                v_prof.push(u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij]);
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
    use super::{
        bunkers_cache_key, pack_bunkers_stack, surface_augmented_shear_from_profile,
        unpack_bunkers_stack, wrfpython_rip_level_bounds, wrfpython_rip_srh_column,
        wrfpython_rip_storm_motion, StormMotionMethod, SURFACE_LAYER_HEIGHT_M,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn surface_augmentation_anchors_10m_winds_at_zero_agl() {
        assert_eq!(SURFACE_LAYER_HEIGHT_M, 0.0);
    }

    #[test]
    fn fixed_layer_shear_uses_the_10m_surface_wind() {
        let u = [0.0, 5.0, 20.0];
        let v = [0.0, 0.0, 0.0];
        let h = [0.0, 25.0, 1_000.0];

        let shear = surface_augmented_shear_from_profile(&u, &v, &h, 0.0, 1_000.0);

        assert!((shear - 20.0).abs() < 1.0e-12);
    }

    #[test]
    fn wrfpython_rip_motion_uses_three_to_ten_km_mean_and_hemisphere_turn() {
        let heights = [
            100.0, 1_000.0, 3_001.0, 5_000.0, 8_000.0, 10_001.0, 12_000.0,
        ];
        let u = [10.0; 7];
        let v = [0.0; 7];
        let (level_3km, level_10km, _) =
            wrfpython_rip_level_bounds(&heights, 0.0, 3_000.0).unwrap();

        let north =
            wrfpython_rip_storm_motion(&u, &v, &heights, level_3km, level_10km, 35.0).unwrap();
        let south =
            wrfpython_rip_storm_motion(&u, &v, &heights, level_3km, level_10km, -35.0).unwrap();

        assert_close(north.0, 6.495_190_528_383_29);
        assert_close(north.1, -3.75);
        assert_close(south.0, 6.495_190_528_383_29);
        assert_close(south.1, 3.75);
    }

    #[test]
    fn wrfpython_rip_srh_reproduces_signed_mirrored_golden_columns() {
        let heights = [
            100.0, 1_000.0, 3_001.0, 5_000.0, 8_000.0, 10_001.0, 12_000.0,
        ];
        let u = [0.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let v_north = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v_south = [0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let north = wrfpython_rip_srh_column(&u, &v_north, &heights, 0.0, 35.0, 3_000.0);
        let south = wrfpython_rip_srh_column(&u, &v_south, &heights, 0.0, -35.0, 3_000.0);

        assert_close(north, 87.5);
        assert_close(south, -87.5);
    }

    #[test]
    fn wrfpython_rip_srh_uses_first_model_level_strictly_above_top() {
        let heights = [
            100.0, 1_000.0, 3_000.0, 3_500.0, 8_000.0, 10_001.0, 12_000.0,
        ];
        let (_, _, level_top) = wrfpython_rip_level_bounds(&heights, 0.0, 3_000.0).unwrap();

        assert_eq!(level_top, 3);
    }

    #[test]
    fn bunkers_stack_round_trip_preserves_field_order() {
        let stacked = pack_bunkers_stack(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
        );
        let unpacked = unpack_bunkers_stack(&stacked, 2).unwrap();
        assert_eq!(unpacked.0, vec![1.0, 2.0]);
        assert_eq!(unpacked.1, vec![3.0, 4.0]);
        assert_eq!(unpacked.2, vec![5.0, 6.0]);
        assert_eq!(unpacked.3, vec![7.0, 8.0]);
        assert_eq!(unpacked.4, vec![9.0, 10.0]);
        assert_eq!(unpacked.5, vec![11.0, 12.0]);
    }

    #[test]
    fn bunkers_cache_key_tracks_method() {
        assert_eq!(
            bunkers_cache_key(StormMotionMethod::PressureWeighted),
            "bunkers_stack_pw"
        );
        assert_eq!(
            bunkers_cache_key(StormMotionMethod::NonPressureWeighted),
            "bunkers_stack_npw"
        );
    }
}
