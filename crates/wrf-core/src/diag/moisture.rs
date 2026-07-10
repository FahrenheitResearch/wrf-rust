//! Moisture diagnostic variables:
//! pw, rh2m, dp2m, mixing_ratio, specific_humidity

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Precipitable water (mm). `[ny, nx]`
pub fn compute_pw(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let qv = f.qvapor(t)?;
    let pres = f.full_pressure(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    Ok(crate::met::composite::compute_pw(&qv, &pres, nx, ny, nz))
}

/// 2-m relative humidity (%). `[ny, nx]`
pub fn compute_rh2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let psfc = f.psfc(t)?; // Pa

    Ok(t2
        .iter()
        .zip(q2.iter())
        .zip(psfc.iter())
        .map(|((t_k, q), p_pa)| {
            crate::diag::thermo::wrf_relative_humidity_from_mixing_ratio(
                *q,
                *p_pa / 100.0,
                *t_k - 273.15,
            )
        })
        .collect())
}

/// 2-m dewpoint (°C). `[ny, nx]`
pub fn compute_dp2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let q2 = f.q2_for_opts(t, opts)?;
    let psfc = f.psfc(t)?;

    Ok(q2
        .iter()
        .zip(psfc.iter())
        .map(|(q, p_pa)| dewpoint_2m_from_model_state(*q, *p_pa))
        .collect())
}

fn dewpoint_2m_from_model_state(q_kgkg: f64, p_pa: f64) -> f64 {
    crate::met::thermo::dewpoint_from_mixing_ratio(q_kgkg, p_pa / 100.0)
}

/// Water vapor mixing ratio (kg/kg). `[nz, ny, nx]`
pub fn compute_mixing_ratio(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.qvapor(t).map(|v| v.to_vec())
}

/// Specific humidity (kg/kg). `[nz, ny, nx]`
/// q = qv / (1 + qv)
pub fn compute_specific_humidity(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let qv = f.qvapor(t)?;
    Ok(qv.iter().map(|q| q / (1.0 + q)).collect())
}

#[cfg(test)]
mod tests {
    use super::dewpoint_2m_from_model_state;

    #[test]
    fn two_meter_dewpoint_uses_wrf_vapor_pressure_floor() {
        let dewpoint = dewpoint_2m_from_model_state(0.0, 100_000.0);

        assert!((dewpoint + 80.447_858_788_617_48).abs() < 1.0e-12);
    }
}
