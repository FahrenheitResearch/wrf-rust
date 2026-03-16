//! Radar diagnostic variables: dbz, maxdbz
//!
//! Simulated reflectivity from hydrometeor mixing ratios using the
//! Smith (1984) / Koch et al. Z-R relationship.

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

#[allow(dead_code)]
const RHO_WATER: f64 = 1000.0; // kg/m^3
const RD: f64 = 287.058;

/// Simulated reflectivity (dBZ). `[nz, ny, nx]`
///
/// Uses Smith (1984) formulation: Z = f(qr, qs, qg, rho_air)
pub fn compute_dbz(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tk = f.temperature(t)?;
    let pres = f.full_pressure(t)?;
    let qv = f.qvapor(t)?;

    // Try to read hydrometeor fields; default to zero if absent
    let qr = f.read_var("QRAIN", t).unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let qs = f.read_var("QSNOW", t).unwrap_or_else(|_| vec![0.0; f.nxyz()]);
    let qg = f.read_var("QGRAUP", t).unwrap_or_else(|_| vec![0.0; f.nxyz()]);

    let n = f.nxyz();

    let dbz: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let t_k = tk[i];
            let qv_i = qv[i].max(0.0);
            let tv = t_k * (1.0 + 0.61 * qv_i);
            let rho = pres[i] / (RD * tv);

            let qr_i = qr[i].max(0.0);
            let qs_i = qs[i].max(0.0);
            let qg_i = qg[i].max(0.0);

            // Reflectivity factor Z (mm^6/m^3)
            // Rain: Z_r = 720 * N0r^(-7/4) * (rho * qr)^(7/4)
            // With N0r = 8e6 m^-4
            let z_rain = if qr_i > 1e-8 {
                let rho_qr = rho * qr_i * 1000.0; // g/m^3
                43.1 * rho_qr.powf(1.75)
            } else {
                0.0
            };

            // Snow: Z_s = factor * (rho * qs)^1.75 (using rho_s = 100 kg/m^3)
            let z_snow = if qs_i > 1e-8 {
                let rho_qs = rho * qs_i * 1000.0;
                48.0 * rho_qs.powf(1.75)
            } else {
                0.0
            };

            // Graupel: Z_g = factor * (rho * qg)^1.75
            let z_graup = if qg_i > 1e-8 {
                let rho_qg = rho * qg_i * 1000.0;
                94.5 * rho_qg.powf(1.75)
            } else {
                0.0
            };

            let z_total = z_rain + z_snow + z_graup;
            if z_total > 0.0 {
                10.0 * z_total.log10()
            } else {
                -30.0 // below noise floor
            }
        })
        .collect();

    Ok(dbz)
}

/// Maximum (composite) reflectivity (dBZ). `[ny, nx]`
pub fn compute_maxdbz(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let dbz_3d = compute_dbz(f, t, opts)?;
    let nxy = f.nxy();
    let nz = f.nz;

    let mut maxdbz = vec![f64::NEG_INFINITY; nxy];
    for k in 0..nz {
        let offset = k * nxy;
        for ij in 0..nxy {
            let val = dbz_3d[offset + ij];
            if val > maxdbz[ij] {
                maxdbz[ij] = val;
            }
        }
    }
    Ok(maxdbz)
}
