//! Wind diagnostic variables:
//! ua, va, wa, wspd, wdir, uvmet, uvmet10, wspd10, wdir10

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// U-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_ua(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.u_destag(t).map(|v| v.to_vec())
}

/// V-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_va(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.v_destag(t).map(|v| v.to_vec())
}

/// W-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_wa(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.w_destag(t).map(|v| v.to_vec())
}

/// Latitude (degrees). `[ny, nx]`
pub fn compute_lat(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlat(t).map(|v| v.to_vec())
}

/// Longitude (degrees). `[ny, nx]`
pub fn compute_lon(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlong(t).map(|v| v.to_vec())
}

/// Rotate one grid-relative wind vector to earth-relative coordinates.
///
/// WPS writes `SINALPHA` with the Northern Hemisphere rotation sense. NCAR
/// wrf-python's `DCOMPUTEUVMET` reverses that angle south of the equator, so
/// the sine term must be negated when `latitude_deg < 0`.
#[inline]
pub(crate) fn rotate_grid_wind_to_earth(
    u: f64,
    v: f64,
    sina: f64,
    cosa: f64,
    latitude_deg: f64,
) -> (f64, f64) {
    let signed_sina = if latitude_deg < 0.0 { -sina } else { sina };
    (u * cosa - v * signed_sina, u * signed_sina + v * cosa)
}

/// Rotate grid-relative (u, v) to earth-relative using SINALPHA/COSALPHA.
fn rotate_to_earth(
    u: &[f64],
    v: &[f64],
    sina: &[f64],
    cosa: &[f64],
    latitude: &[f64],
    nxy: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut u_earth = vec![0.0; u.len()];
    let mut v_earth = vec![0.0; v.len()];

    u_earth
        .iter_mut()
        .zip(v_earth.iter_mut())
        .enumerate()
        .for_each(|(idx, (ue, ve))| {
            let ij = idx % nxy;
            (*ue, *ve) =
                rotate_grid_wind_to_earth(u[idx], v[idx], sina[ij], cosa[ij], latitude[ij]);
        });

    (u_earth, v_earth)
}

/// Wind speed from (u, v). `[nz, ny, nx]`
pub fn compute_wspd(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    Ok(u.iter()
        .zip(v.iter())
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// Wind direction (degrees, meteorological convention). `[nz, ny, nx]`
pub fn compute_wdir(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let latitude = f.xlat(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, &latitude, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}

/// Earth-rotated U/V wind. Returns interleaved `[u_earth..., v_earth...]` (2 * nxyz).
pub fn compute_uvmet(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let latitude = f.xlat(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, &latitude, f.nxy());

    let mut out = ue;
    out.extend(ve);
    Ok(out)
}

/// 10-m earth-rotated U/V wind. Returns `[u_earth..., v_earth...]` (2 * nxy).
pub fn compute_uvmet10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let latitude = f.xlat(t)?;

    let (ue, ve) = rotate_to_earth(&u10, &v10, &sina, &cosa, &latitude, f.nxy());

    let mut out = ue;
    out.extend(ve);
    Ok(out)
}

/// 10-m wind speed (m/s). `[ny, nx]`
pub fn compute_wspd10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    Ok(u.iter()
        .zip(v.iter())
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// 10-m wind direction (degrees). `[ny, nx]`
pub fn compute_wdir10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;
    let latitude = f.xlat(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, &latitude, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::rotate_grid_wind_to_earth;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn northern_hemisphere_rotation_preserves_wps_sine_sense() {
        let angle = 7.0_f64.to_radians();
        let (u_earth, v_earth) =
            rotate_grid_wind_to_earth(10.0, 0.0, angle.sin(), angle.cos(), 35.0);

        assert_close(u_earth, 10.0 * angle.cos());
        assert_close(v_earth, 10.0 * angle.sin());
    }

    #[test]
    fn southern_hemisphere_rotation_reverses_wps_sine_sense() {
        let angle = 7.0_f64.to_radians();
        let (u_earth, v_earth) =
            rotate_grid_wind_to_earth(10.0, 0.0, angle.sin(), angle.cos(), -35.0);

        assert_close(u_earth, 10.0 * angle.cos());
        assert_close(v_earth, -10.0 * angle.sin());
    }
}
