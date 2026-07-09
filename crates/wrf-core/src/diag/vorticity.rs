//! Vorticity diagnostic variables: avo, pvo

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const OMEGA: f64 = 7.2921159e-5; // Earth's angular velocity (rad/s)
const AVO_DISPLAY_SCALE: f64 = 1.0e5; // wrf-python reports 10^-5 s^-1

fn relative_vorticity_from_uv(
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    crate::met::dynamics::vorticity(u, v, nx, ny, dx, dy)
}

/// Absolute vorticity (10^-5 s^-1). `[nz, ny, nx]`
///
/// AVO = relative_vorticity + coriolis_parameter
/// = (dv/dx - du/dy) + 2*Omega*sin(lat)
pub fn compute_avo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let lat = f.xlat(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    let mut avo = vec![0.0f64; nz * nxy];

    avo.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];

        // Compute relative vorticity using central differences
        let rel_vort = relative_vorticity_from_uv(u_plane, v_plane, nx, ny, dx, dy);

        for ij in 0..nxy {
            let f_cor = 2.0 * OMEGA * (lat[ij].to_radians()).sin();
            plane[ij] = (rel_vort[ij] + f_cor) * AVO_DISPLAY_SCALE;
        }
    });

    Ok(avo)
}

/// Potential vorticity (PVU, 1 PVU = 10^-6 K m^2 kg^-1 s^-1). `[nz, ny, nx]`
///
/// PVO = -g * (f + zeta) * (dtheta/dp)
///
/// This is a hydrostatic vertical-stretching approximation. Full WRF-Python
/// parity also requires the horizontal potential-temperature-gradient
/// (baroclinic) terms and WRF map-metric factors; those remain future work.
pub fn compute_pvo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let theta = f.full_theta(t)?;
    let pres = f.full_pressure(t)?;
    let lat = f.xlat(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    let g = 9.80665;

    let mut pvo = vec![0.0f64; nz * nxy];

    pvo.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];

        let rel_vort = relative_vorticity_from_uv(u_plane, v_plane, nx, ny, dx, dy);

        for ij in 0..nxy {
            let f_cor = 2.0 * OMEGA * (lat[ij].to_radians()).sin();
            let abs_vort = rel_vort[ij] + f_cor;

            // dtheta/dp using centered differences in the vertical
            let dtheta_dp = if k == 0 {
                let idx0 = ij;
                let idx1 = nxy + ij;
                (theta[idx1] - theta[idx0]) / (pres[idx1] - pres[idx0])
            } else if k == nz - 1 {
                let idx0 = (k - 1) * nxy + ij;
                let idx1 = k * nxy + ij;
                (theta[idx1] - theta[idx0]) / (pres[idx1] - pres[idx0])
            } else {
                let idx_b = (k - 1) * nxy + ij;
                let idx_t = (k + 1) * nxy + ij;
                (theta[idx_t] - theta[idx_b]) / (pres[idx_t] - pres[idx_b])
            };

            // PVO = -g * abs_vort * dtheta/dp, convert to PVU (*1e6)
            plane[ij] = -g * abs_vort * dtheta_dp * 1e6;
        }
    });

    Ok(pvo)
}

#[cfg(test)]
mod tests {
    use super::relative_vorticity_from_uv;

    const NX: usize = 5;
    const NY: usize = 4;
    const DX: f64 = 2_000.0;
    const DY: f64 = 3_000.0;

    fn analytic_wind(mut wind_at: impl FnMut(f64, f64) -> (f64, f64)) -> (Vec<f64>, Vec<f64>) {
        let mut u = Vec::with_capacity(NX * NY);
        let mut v = Vec::with_capacity(NX * NY);
        for j in 0..NY {
            for i in 0..NX {
                let (u_value, v_value) = wind_at(i as f64 * DX, j as f64 * DY);
                u.push(u_value);
                v.push(v_value);
            }
        }
        (u, v)
    }

    #[test]
    fn solid_body_rotation_has_twice_the_angular_velocity() {
        let angular_velocity = 1.5e-4;
        let (u, v) = analytic_wind(|x, y| (-angular_velocity * y, angular_velocity * x));

        let vorticity = relative_vorticity_from_uv(&u, &v, NX, NY, DX, DY);

        for value in vorticity {
            assert!((value - 2.0 * angular_velocity).abs() < 1.0e-12);
        }
    }

    #[test]
    fn pure_deformation_has_zero_vorticity() {
        let deformation_rate = 2.0e-4;
        let (u, v) = analytic_wind(|x, y| (deformation_rate * x, -deformation_rate * y));

        let vorticity = relative_vorticity_from_uv(&u, &v, NX, NY, DX, DY);

        for value in vorticity {
            assert!(value.abs() < 1.0e-12);
        }
    }
}
