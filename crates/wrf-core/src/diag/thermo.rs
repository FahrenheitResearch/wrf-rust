//! Thermodynamic diagnostic variables:
//! temp, tc, theta, theta_e, tv, twb, td, rh

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// 2-m temperature (K). `[ny, nx]`
pub fn compute_t2(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.t2_for_opts(t, opts)
}

/// 2-m virtual temperature (K). `[ny, nx]`
/// Tv = T2 * (1 + 0.61 * Q2). Supports lake_interp.
pub fn compute_tv2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    Ok(t2
        .iter()
        .zip(q2.iter())
        .map(|(tk, q)| tk * (1.0 + 0.61 * q.max(0.0)))
        .collect())
}

/// Temperature (K). `[nz, ny, nx]`
pub fn compute_temp(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.temperature(t).map(|v| v.to_vec())
}

/// Temperature (°C). `[nz, ny, nx]`
pub fn compute_tc(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.temperature_c(t).map(|v| v.to_vec())
}

/// Potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.full_theta(t).map(|v| v.to_vec())
}

/// Equivalent potential temperature (K). `[nz, ny, nx]`
pub fn compute_theta_e(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    // Keep the native theta-e vapor-pressure RH path; the registered Td/Twb
    // diagnostics use wrf-python's separate mixing-ratio conversion below.
    let result: Vec<f64> = p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| theta_e_from_model_state(*p, *t_c, *q))
        .collect();
    Ok(result)
}

/// Virtual temperature (K). `[nz, ny, nx]`
pub fn compute_tv(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tk = f.temperature(t)?;
    let qv = f.qvapor(t)?;
    // Tv = T * (1 + 0.61 * qv)
    Ok(tk
        .iter()
        .zip(qv.iter())
        .map(|(t, q)| t * (1.0 + 0.61 * q.max(0.0)))
        .collect())
}

/// Wet-bulb temperature (K). `[nz, ny, nx]`
pub fn compute_twb(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| {
            let td_c = dewpoint_3d_from_model_state(*q, *p);
            crate::met::thermo::wet_bulb_temperature(*p, *t_c, td_c) + 273.15
        })
        .collect())
}

/// Dewpoint temperature (°C). `[nz, ny, nx]`
pub fn compute_td(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(qv.iter())
        .map(|(p, q)| dewpoint_3d_from_model_state(*q, *p))
        .collect())
}

/// Relative humidity (%). `[nz, ny, nx]`
pub fn compute_rh(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let p_hpa = f.pressure_hpa(t)?;
    let tc = f.temperature_c(t)?;
    let qv = f.qvapor(t)?;

    Ok(p_hpa
        .iter()
        .zip(tc.iter())
        .zip(qv.iter())
        .map(|((p, t_c), q)| wrf_relative_humidity_from_mixing_ratio(*q, *p, *t_c))
        .collect())
}

// ── Helpers ──

/// Compute dewpoint (°C) from mixing ratio (kg/kg) and pressure (hPa).
fn dewpoint_3d_from_model_state(q_kgkg: f64, p_hpa: f64) -> f64 {
    crate::met::thermo::dewpoint_from_mixing_ratio(q_kgkg, p_hpa)
}

/// Compute wrf-python relative humidity (%) from water-vapor mixing ratio.
///
/// This follows NCAR's `DCOMPUTERH`: `qvs = eps*es/(p-(1-eps)*es)`,
/// then clamps `qv/qvs` to [0, 1].
/// Reference: <https://github.com/NCAR/wrf-python/blob/31c923335227b22fa656fd589a5342b91103e939/fortran/wrf_user.f90#L703-L730>
pub(crate) fn wrf_relative_humidity_from_mixing_ratio(q_kgkg: f64, p_hpa: f64, t_c: f64) -> f64 {
    let q = q_kgkg.max(0.0);
    let es_hpa = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
    let qvs = 0.622 * es_hpa / (p_hpa - (1.0 - 0.622) * es_hpa);
    100.0 * (q / qvs).clamp(0.0, 1.0)
}

// Retain the vapor-pressure RH convention used by the native theta-e path.
// Changing this helper would alter theta-e in addition to the registered RH
// diagnostics addressed by the wrf-python parity correction above.
fn vapor_pressure_relative_humidity_from_mixing_ratio(q_kgkg: f64, p_hpa: f64, t_c: f64) -> f64 {
    let q = q_kgkg.max(0.0);
    let e_hpa = q * p_hpa / (0.622 + q);
    let es_hpa = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
    (e_hpa / es_hpa * 100.0).clamp(0.0, 100.0)
}

fn theta_e_from_model_state(p_hpa: f64, t_c: f64, q_kgkg: f64) -> f64 {
    let td_c = crate::met::thermo::dewpoint_from_rh(
        t_c,
        vapor_pressure_relative_humidity_from_mixing_ratio(q_kgkg, p_hpa, t_c),
    );
    crate::met::thermo::equivalent_potential_temperature(p_hpa, t_c, td_c)
}

#[cfg(test)]
mod tests {
    use super::{
        dewpoint_3d_from_model_state, theta_e_from_model_state,
        wrf_relative_humidity_from_mixing_ratio,
    };

    #[test]
    fn three_dimensional_dewpoint_uses_wrf_vapor_pressure_floor() {
        let dewpoint = dewpoint_3d_from_model_state(2.0e-6, 100.0);

        assert!((dewpoint + 80.447_858_788_617_48).abs() < 1.0e-12);
    }

    #[test]
    fn relative_humidity_matches_ncar_qv_over_qvs() {
        let rh = wrf_relative_humidity_from_mixing_ratio(0.014, 1_000.0, 30.0);

        assert!((rh - 52.164_479_671_648_93).abs() < 1.0e-12);
    }

    #[test]
    fn relative_humidity_clamps_dry_and_supersaturated_inputs() {
        let es = 6.112 * (17.67_f64 * 30.0 / (30.0 + 243.5)).exp();
        let qvs = 0.622 * es / (1_000.0 - (1.0 - 0.622) * es);

        assert_eq!(
            wrf_relative_humidity_from_mixing_ratio(-0.001, 1_000.0, 30.0),
            0.0
        );
        assert_eq!(
            wrf_relative_humidity_from_mixing_ratio(2.0 * qvs, 1_000.0, 30.0),
            100.0
        );
    }

    #[test]
    fn theta_e_registered_units_are_kelvin_not_offset_twice() {
        let theta_e = theta_e_from_model_state(1_000.0, 30.0, 0.014);

        assert!((theta_e - 344.9322).abs() < 0.001);
        assert!((250.0..450.0).contains(&theta_e));
    }
}
