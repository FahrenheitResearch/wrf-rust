//! Extra diagnostic variables:
//! lapse rates, freezing level, wet-bulb zero, theta_w, fire indices,
//! and sounding-derived scalar diagnostics.

use crate::compute::ComputeOpts;
use crate::diag::cape::build_surface_augmented_thermo_column;
use crate::error::WrfResult;
use crate::file::WrfFile;
use rayon::prelude::*;

/// 700-500 hPa lapse rate (°C/km). `[ny, nx]`
///
/// Delegates to the generic compute_lapse_rate with pressure bounds.
/// Supports use_virtual and lake_interp via opts.
pub fn compute_lapse_rate_700_500(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let mut lr_opts = opts.clone();
    lr_opts.bottom_p = Some(700.0);
    lr_opts.top_p = Some(500.0);
    compute_lapse_rate(f, t, &lr_opts)
}

/// 0-3 km AGL lapse rate (°C/km). `[ny, nx]`
///
/// Delegates to the generic compute_lapse_rate with height bounds.
/// Uses T2 at surface, supports use_virtual and lake_interp via opts.
pub fn compute_lapse_rate_0_3km(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let mut lr_opts = opts.clone();
    lr_opts.bottom_m = Some(0.0);
    lr_opts.top_m = Some(3000.0);
    compute_lapse_rate(f, t, &lr_opts)
}

/// Freezing level height AGL (m). `[ny, nx]`
pub fn compute_freezing_level(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tc = f.temperature_c(t)?;
    let h_agl = f.height_agl(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut fzlev = vec![0.0f64; nxy];
    fzlev.iter_mut().enumerate().for_each(|(ij, fz_val)| {
        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;
            // Find first crossing of 0°C going upward
            if tc[idx0] >= 0.0 && tc[idx1] < 0.0 {
                let frac = (0.0 - tc[idx0]) / (tc[idx1] - tc[idx0]);
                *fz_val = h_agl[idx0] + frac * (h_agl[idx1] - h_agl[idx0]);
                return;
            }
        }
        // If surface is already below freezing, freezing level is 0
        if tc[ij] < 0.0 {
            *fz_val = 0.0;
        }
    });

    Ok(fzlev)
}

/// Wet-bulb zero height AGL (m). `[ny, nx]`
pub fn compute_wet_bulb_0(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut wb0 = vec![0.0f64; nxy];
    wb0.iter_mut().enumerate().for_each(|(ij, wb0_val)| {
        // Compute wet-bulb at each level, find first crossing below 0°C
        let mut prev_twb = f64::NAN;
        let mut prev_h = 0.0;

        for k in 0..nz {
            let idx = k * nxy + ij;
            let q = qv[idx].max(1e-10);
            let e = q * p_hpa[idx] / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            let td_c = (243.5 * ln_e) / (17.67 - ln_e);

            let twb = crate::met::thermo::wet_bulb_temperature(p_hpa[idx], tc[idx], td_c);

            if k > 0 && prev_twb >= 0.0 && twb < 0.0 {
                let frac = (0.0 - prev_twb) / (twb - prev_twb);
                *wb0_val = prev_h + frac * (h_agl[idx] - prev_h);
                return;
            }
            prev_twb = twb;
            prev_h = h_agl[idx];
        }
    });

    Ok(wb0)
}

/// Wet-bulb potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta_w(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| {
            let q = q.max(1e-10);
            let e = q * p / (0.622 + q);
            let ln_e = (e / 6.112).max(1e-10).ln();
            let td_c = (243.5 * ln_e) / (17.67 - ln_e);
            crate::met::thermo::wet_bulb_potential_temperature(*p, *t_c, td_c) + 273.15
        })
        .collect())
}

/// Fosberg Fire Weather Index (dimensionless). `[ny, nx]`
/// Uses 2-m temperature, 2-m RH, and 10-m wind speed.
pub fn compute_fosberg(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2(t)?;
    let q2 = f.q2(t)?;
    let psfc = f.psfc(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let t_k = t2[ij];
            let t_c = t_k - 273.15;
            let t_f = t_c * 9.0 / 5.0 + 32.0;
            let p_hpa = psfc[ij] / 100.0;
            let q = q2[ij].max(0.0);
            let e = q * p_hpa / (0.622 + q);
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            let rh = (e / es * 100.0).clamp(0.0, 100.0);
            let wspd_mph = (u10[ij].powi(2) + v10[ij].powi(2)).sqrt() / 0.44704;

            crate::met::composite::fosberg_fire_weather_index(t_f, rh, wspd_mph)
        })
        .collect())
}

/// Haines Index (1-3, low/moderate/high). `[ny, nx]`
pub fn compute_haines(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    let nxy = f.nxy();
    let nz = f.nz;

    let mut haines = vec![0.0f64; nxy];
    haines.iter_mut().enumerate().for_each(|(ij, h_val)| {
        // Find T and Td at 950, 850 hPa
        let mut t950 = 0.0f64;
        let mut t850 = 0.0f64;
        let mut td850 = 0.0f64;

        for k in 0..nz - 1 {
            let idx = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            for &target_p in &[950.0, 850.0] {
                if p_hpa[idx] >= target_p && p_hpa[idx1] < target_p {
                    let frac = (target_p - p_hpa[idx1]) / (p_hpa[idx] - p_hpa[idx1]);
                    let t_interp = tc[idx1] + frac * (tc[idx] - tc[idx1]);
                    let q_interp = qv[idx1] + frac * (qv[idx] - qv[idx1]);
                    let q = q_interp.max(1e-10);
                    let e = q * target_p / (0.622 + q);
                    let ln_e = (e / 6.112).max(1e-10).ln();
                    let td = (243.5 * ln_e) / (17.67 - ln_e);

                    if target_p == 950.0 {
                        t950 = t_interp;
                    } else {
                        t850 = t_interp;
                        td850 = td;
                    }
                }
            }
        }

        *h_val = crate::met::composite::haines_index(t950, t850, td850) as f64;
    });

    Ok(haines)
}

/// Hot-Dry-Windy Index (dimensionless). `[ny, nx]`
pub fn compute_hdw(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2(t)?;
    let q2 = f.q2(t)?;
    let psfc = f.psfc(t)?;
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;

    let nxy = f.nxy();

    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let t_c = t2[ij] - 273.15;
            let p_hpa = psfc[ij] / 100.0;
            let q = q2[ij].max(0.0);
            let e = q * p_hpa / (0.622 + q);
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            let rh = (e / es * 100.0).clamp(0.0, 100.0);
            let wspd_ms = (u10[ij].powi(2) + v10[ij].powi(2)).sqrt();

            // VPD = es - e (in hPa)
            let vpd = (es - e).max(0.0);

            crate::met::composite::hot_dry_windy(t_c, rh, wspd_ms, vpd)
        })
        .collect())
}

/// Generic configurable lapse rate (°C/km). `[ny, nx]`
///
/// Supports two modes:
/// - **Height mode** (default): `bottom_m` / `top_m` in meters AGL. Defaults to 0-3 km.
/// - **Pressure mode**: `bottom_p` / `top_p` in hPa (e.g. 700, 500). When either
///   pressure bound is set, height bounds are ignored.
///
/// If `opts.use_virtual` is `Some(true)`, virtual temperature
/// Tv = T * (1 + 0.61 * qv) is used instead of absolute temperature.
pub fn compute_lapse_rate(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let use_virtual = opts.use_virtual == Some(true);
    let use_pressure = opts.bottom_p.is_some() || opts.top_p.is_some();

    let tc = f.temperature_c(t)?;
    let h_agl = f.height_agl(t)?;
    let qv = if use_virtual {
        Some(f.qvapor(t)?)
    } else {
        None
    };
    let pres_hpa = if use_pressure {
        Some(f.pressure_hpa(t)?)
    } else {
        None
    };

    // For surface (bottom_m=0 or not set), use 2m data with lake_interp support
    let bottom_is_surface = !use_pressure && opts.bottom_m.unwrap_or(0.0) < 10.0;
    let t2_c = if bottom_is_surface {
        let t2_k = match opts.lake_interp {
            Some(a) if a > 0.0 => f.t2_lake_corrected(t, a)?,
            _ => f.t2(t)?.to_vec(),
        };
        Some(t2_k.iter().map(|v| v - 273.15).collect::<Vec<f64>>())
    } else {
        None
    };
    let q2 = if bottom_is_surface && use_virtual {
        Some(match opts.lake_interp {
            Some(a) if a > 0.0 => f.q2_lake_corrected(t, a)?,
            _ => f.q2(t)?.to_vec(),
        })
    } else {
        None
    };

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut lr = vec![0.0f64; nxy];
    lr.iter_mut().enumerate().for_each(|(ij, lr_val)| {
        // Helper: temperature (or virtual temperature) at a 3-D index
        let temp_at = |idx: usize| -> f64 {
            let t_c = tc[idx];
            if use_virtual {
                let q = qv.as_ref().unwrap()[idx].max(0.0);
                let t_k = t_c + 273.15;
                let tv_k = t_k * (1.0 + 0.61 * q);
                tv_k - 273.15
            } else {
                t_c
            }
        };

        if use_pressure {
            // Pressure mode: interpolate T and H at given pressure levels
            let p = pres_hpa.as_ref().unwrap();
            let p_bot = opts.bottom_p.unwrap_or(700.0); // hPa (higher pressure = lower)
            let p_top = opts.top_p.unwrap_or(500.0); // hPa (lower pressure = higher)

            let interp_at_p = |target_p: f64| -> (f64, f64) {
                // Pressure decreases with height; scan upward
                for k in 0..nz - 1 {
                    let idx0 = k * nxy + ij;
                    let idx1 = (k + 1) * nxy + ij;
                    let p0 = p[idx0];
                    let p1 = p[idx1];
                    if p0 >= target_p && p1 < target_p {
                        let frac = (target_p - p1) / (p0 - p1);
                        let t_interp = temp_at(idx1) + frac * (temp_at(idx0) - temp_at(idx1));
                        let h_interp = h_agl[idx1] + frac * (h_agl[idx0] - h_agl[idx1]);
                        return (t_interp, h_interp);
                    }
                }
                // Fallback: nearest level
                (temp_at(ij), h_agl[ij])
            };

            let (t_bot, h_bot) = interp_at_p(p_bot);
            let (t_top, h_top) = interp_at_p(p_top);
            let depth_km = (h_top - h_bot).abs() / 1000.0;
            if depth_km > 0.01 {
                *lr_val = -(t_top - t_bot) / depth_km;
            }
        } else {
            // Height mode
            let bottom_m = opts.bottom_m.unwrap_or(0.0);
            let top_m = opts.top_m.unwrap_or(3000.0);
            let depth_km = (top_m - bottom_m) / 1000.0;

            let interp_at_h = |target_h: f64| -> f64 {
                // Use 2m data for surface (target_h < 10m)
                if target_h < 10.0 {
                    if let Some(ref t2) = t2_c {
                        let t_sfc = t2[ij];
                        if use_virtual {
                            if let Some(ref q) = q2 {
                                let t_k = t_sfc + 273.15;
                                let tv_k = t_k * (1.0 + 0.61 * q[ij].max(0.0));
                                return tv_k - 273.15;
                            }
                        }
                        return t_sfc;
                    }
                }
                if h_agl[ij] >= target_h {
                    return temp_at(ij);
                }
                for k in 1..nz {
                    let idx = k * nxy + ij;
                    if h_agl[idx] >= target_h {
                        let idx_prev = (k - 1) * nxy + ij;
                        let h_prev = h_agl[idx_prev];
                        let frac = (target_h - h_prev) / (h_agl[idx] - h_prev);
                        return temp_at(idx_prev) + frac * (temp_at(idx) - temp_at(idx_prev));
                    }
                }
                temp_at((nz - 1) * nxy + ij)
            };

            let t_bot = interp_at_h(bottom_m);
            let t_top = interp_at_h(top_m);
            if depth_km > 0.01 {
                *lr_val = -(t_top - t_bot) / depth_km;
            }
        }
    });

    Ok(lr)
}

fn compute_thermo_profile_scalar<F>(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
    func: F,
) -> WrfResult<Vec<f64>>
where
    F: Fn(&[f64], &[f64], &[f64], &[f64]) -> f64 + Sync,
{
    let pres_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;
    let h_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;

    let nz = f.nz;
    let nxy = f.nxy();
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (p, t, td, h) = build_surface_augmented_thermo_column(
                &pres_hpa, &tc, &qv, &h_agl, psfc[ij], t2[ij], q2[ij], nz, nxy, ij,
            );
            func(&p, &t, &td, &h)
        })
        .collect())
}

/// K Index. `[ny, nx]`
pub fn compute_k_index(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        let t850 = interp_at_pressure(p, temp, 850.0);
        let t700 = interp_at_pressure(p, temp, 700.0);
        let t500 = interp_at_pressure(p, temp, 500.0);
        let td850 = interp_at_pressure(p, td, 850.0);
        let td700 = interp_at_pressure(p, td, 700.0);
        crate::met::composite::k_index(t850, t700, t500, td850, td700)
    })
}

/// Total Totals Index. `[ny, nx]`
pub fn compute_total_totals(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        let t850 = interp_at_pressure(p, temp, 850.0);
        let t500 = interp_at_pressure(p, temp, 500.0);
        let td850 = interp_at_pressure(p, td, 850.0);
        crate::met::composite::total_totals(t850, t500, td850)
    })
}

/// Mean mixing ratio in the lowest 100 hPa (g/kg). `[ny, nx]`
pub fn compute_mean_mixr(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, _temp, td, _| {
        mean_mixratio(p, td, p[0], p[0] - 100.0).unwrap_or(f64::NAN)
    })
}

/// Mean relative humidity in the lowest 100 hPa (%). `[ny, nx]`
pub fn compute_low_rh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        mean_rh_pressure(p, temp, td, p[0], p[0] - 100.0).unwrap_or(f64::NAN)
    })
}

/// Mean relative humidity from 700-500 hPa (%). `[ny, nx]`
pub fn compute_mid_rh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        mean_rh_pressure(p, temp, td, 700.0, 500.0).unwrap_or(f64::NAN)
    })
}

/// Mean relative humidity through the dendritic growth zone (%). `[ny, nx]`
pub fn compute_dgz_rh(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        let p_minus_12 = pressure_at_temperature(p, temp, -12.0);
        let p_minus_17 = pressure_at_temperature(p, temp, -17.0);
        if p_minus_12.is_finite() && p_minus_17.is_finite() {
            mean_rh_pressure(
                p,
                temp,
                td,
                p_minus_12.max(p_minus_17),
                p_minus_12.min(p_minus_17),
            )
            .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    })
}

/// LCL temperature (C). `[ny, nx]`
pub fn compute_lcl_temp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        let (_, lcl_t) = crate::met::thermo::drylift(p[0], temp[0], td[0]);
        lcl_t
    })
}

/// Convective temperature (C), CCL-based SHARPpy-style estimate. `[ny, nx]`
pub fn compute_convective_temp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, _| {
        convective_temp_profile(p, temp, td).unwrap_or(f64::NAN)
    })
}

/// Forecast maximum surface temperature (C). `[ny, nx]`
pub fn compute_max_temp(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, _td, _| {
        let ptop = p[0] - 100.0;
        let t_top = interp_at_pressure(p, temp, ptop);
        if !t_top.is_finite() || ptop <= 0.0 {
            return f64::NAN;
        }
        ((t_top + crate::met::thermo::ZEROCNK + 2.0) * (p[0] / ptop).powf(crate::met::thermo::ROCP))
            - crate::met::thermo::ZEROCNK
    })
}

/// Downdraft CAPE (J/kg). `[ny, nx]`
pub fn compute_dcape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    compute_thermo_profile_scalar(f, t, opts, |p, temp, td, h| {
        dcape_profile(p, temp, td, h).unwrap_or(0.0)
    })
}

fn interp_at_pressure(p: &[f64], values: &[f64], target_p: f64) -> f64 {
    if p.is_empty() || values.len() != p.len() || !target_p.is_finite() {
        return f64::NAN;
    }
    for i in 0..p.len().saturating_sub(1) {
        let p0 = p[i];
        let p1 = p[i + 1];
        if !p0.is_finite()
            || !p1.is_finite()
            || !values[i].is_finite()
            || !values[i + 1].is_finite()
        {
            continue;
        }
        if (target_p <= p0 && target_p >= p1) || (target_p >= p0 && target_p <= p1) {
            if (p1 - p0).abs() < 1.0e-9 {
                return values[i];
            }
            let frac = (target_p - p0) / (p1 - p0);
            return values[i] + frac * (values[i + 1] - values[i]);
        }
    }
    f64::NAN
}

fn pressure_at_temperature(p: &[f64], temp: &[f64], target_t: f64) -> f64 {
    if p.is_empty() || temp.len() != p.len() || !target_t.is_finite() {
        return f64::NAN;
    }
    for i in 0..temp.len().saturating_sub(1) {
        let t0 = temp[i];
        let t1 = temp[i + 1];
        if !t0.is_finite() || !t1.is_finite() || !p[i].is_finite() || !p[i + 1].is_finite() {
            continue;
        }
        if (target_t <= t0 && target_t >= t1) || (target_t >= t0 && target_t <= t1) {
            if (t1 - t0).abs() < 1.0e-9 {
                return p[i];
            }
            let frac = (target_t - t0) / (t1 - t0);
            return p[i] + frac * (p[i + 1] - p[i]);
        }
    }
    f64::NAN
}

fn mean_mixratio(p: &[f64], td: &[f64], pbot: f64, ptop: f64) -> Option<f64> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot <= ptop {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    let steps = (pbot - ptop).ceil() as usize + 1;
    for step in 0..steps {
        let sample_p = (pbot - step as f64).max(ptop);
        let sample_td = interp_at_pressure(p, td, sample_p);
        let mixr = crate::met::thermo::mixratio(sample_p, sample_td);
        if mixr.is_finite() {
            sum += mixr;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn mean_rh_pressure(p: &[f64], temp: &[f64], td: &[f64], pbot: f64, ptop: f64) -> Option<f64> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot <= ptop {
        return None;
    }
    let mut weighted = 0.0;
    let mut total = 0.0;
    let steps = (pbot - ptop).ceil() as usize + 1;
    for step in 0..steps {
        let sample_p = (pbot - step as f64).max(ptop);
        let sample_t = interp_at_pressure(p, temp, sample_p);
        let sample_td = interp_at_pressure(p, td, sample_p);
        let rh = relative_humidity(sample_t, sample_td);
        if rh.is_finite() {
            weighted += rh * sample_p;
            total += sample_p;
        }
    }
    (total > 0.0).then_some(weighted / total)
}

fn relative_humidity(temp_c: f64, td_c: f64) -> f64 {
    if !temp_c.is_finite() || !td_c.is_finite() {
        return f64::NAN;
    }
    let e = 6.112 * (17.67 * td_c / (td_c + 243.5)).exp();
    let es = 6.112 * (17.67 * temp_c / (temp_c + 243.5)).exp();
    (e / es * 100.0).clamp(0.0, 100.0)
}

fn convective_temp_profile(p: &[f64], temp: &[f64], td: &[f64]) -> Option<f64> {
    let sfc_p = *p.first()?;
    let mmr = mean_mixratio(p, td, sfc_p, sfc_p - 100.0)?;
    let steps = (sfc_p - 100.0).max(0.0).ceil() as usize;
    for step in 0..steps {
        let sample_p = sfc_p - step as f64;
        let sample_t = interp_at_pressure(p, temp, sample_p);
        let sat_mixr = crate::met::thermo::mixratio(sample_p, sample_t);
        if sat_mixr.is_finite() && sat_mixr <= mmr {
            let theta_ccl = (sample_t + crate::met::thermo::ZEROCNK)
                * (1000.0 / sample_p).powf(crate::met::thermo::ROCP);
            return Some(
                theta_ccl * (sfc_p / 1000.0).powf(crate::met::thermo::ROCP)
                    - crate::met::thermo::ZEROCNK,
            );
        }
    }
    None
}

fn dcape_profile(p: &[f64], temp: &[f64], td: &[f64], h: &[f64]) -> Option<f64> {
    if p.len() < 2 || temp.len() != p.len() || td.len() != p.len() || h.len() != p.len() {
        return None;
    }
    let sfc_p = p[0];
    let mut min_thetae = f64::INFINITY;
    let mut min_p = f64::NAN;
    for &level_p in p {
        if !level_p.is_finite() || level_p < sfc_p - 400.0 {
            continue;
        }
        if let Some(thetae) = mean_thetae(p, temp, td, level_p, level_p - 100.0) {
            if thetae < min_thetae {
                min_thetae = thetae;
                min_p = level_p - 50.0;
            }
        }
    }
    if !min_p.is_finite() {
        return None;
    }
    let mut i = (0..p.len()).rev().find(|&idx| p[idx] >= min_p)?;
    let mut parcel_t = crate::met::thermo::wet_bulb_temperature(
        min_p,
        interp_at_pressure(p, temp, min_p),
        interp_at_pressure(p, td, min_p),
    );
    let mut parcel_p = min_p;
    let mut env_t = interp_at_pressure(p, temp, parcel_p);
    let mut height = interp_at_pressure(p, h, parcel_p);
    let mut energy = 0.0;

    while i > 0 {
        i -= 1;
        let next_p = p[i];
        let next_env_t = temp[i];
        let next_h = h[i];
        let next_parcel_t = wet_lift(parcel_p, parcel_t, next_p);
        if env_t.is_finite() && next_env_t.is_finite() && height.is_finite() && next_h.is_finite() {
            let deficit_1 = (parcel_t - env_t) / (env_t + crate::met::thermo::ZEROCNK);
            let deficit_2 =
                (next_parcel_t - next_env_t) / (next_env_t + crate::met::thermo::ZEROCNK);
            energy += crate::met::thermo::G * (deficit_1 + deficit_2) * 0.5 * (next_h - height);
        }
        parcel_p = next_p;
        parcel_t = next_parcel_t;
        env_t = next_env_t;
        height = next_h;
    }
    Some(energy.max(0.0))
}

fn mean_thetae(p: &[f64], temp: &[f64], td: &[f64], pbot: f64, ptop: f64) -> Option<f64> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot <= ptop {
        return None;
    }
    let steps = (pbot - ptop).ceil() as usize + 1;
    let mut sum = 0.0;
    let mut count = 0usize;
    for step in 0..steps {
        let sample_p = (pbot - step as f64).max(ptop);
        let sample_t = interp_at_pressure(p, temp, sample_p);
        let sample_td = interp_at_pressure(p, td, sample_p);
        let thetae = crate::met::thermo::thetae(sample_p, sample_t, sample_td);
        if thetae.is_finite() {
            sum += thetae;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn wet_lift(p: f64, temp_c: f64, target_p: f64) -> f64 {
    let theta_c = (temp_c + crate::met::thermo::ZEROCNK)
        * (1000.0 / p).powf(crate::met::thermo::ROCP)
        - crate::met::thermo::ZEROCNK;
    let theta_m = theta_c - crate::met::thermo::wobf(theta_c) + crate::met::thermo::wobf(temp_c);
    crate::met::thermo::satlift(target_p, theta_m)
}
