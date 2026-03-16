//! Updraft helicity diagnostic.

use rayon::prelude::*;

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Updraft helicity (m^2/s^2). `[ny, nx]`
///
/// UH = integral from z_bot to z_top of (w * zeta_z) dz
/// Default layer: 2-5 km AGL.
pub fn compute_uhel(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let w = f.w_destag(t)?;
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let h_agl = f.height_agl(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    // Integration layer bounds (default 2-5 km AGL), configurable
    let z_bot = opts.bottom_m.unwrap_or(2000.0);
    let z_top = opts.top_m.unwrap_or(5000.0);

    // Compute relative vorticity at each level
    let mut vort_3d = vec![0.0f64; nz * nxy];
    vort_3d.par_chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];
        let vort = wx_math::dynamics::vorticity(v_plane, u_plane, nx, ny, dx, dy);
        plane.copy_from_slice(&vort);
    });

    // Integrate w * vort over the layer for each column
    let mut uhel = vec![0.0f64; nxy];
    uhel.par_iter_mut().enumerate().for_each(|(ij, uh_val)| {
        let mut integral = 0.0f64;
        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            let h0 = h_agl[idx0];
            let h1 = h_agl[idx1];

            // Check if this layer intersects the integration bounds
            if h1 < z_bot || h0 > z_top {
                continue;
            }

            let dz = (h1.min(z_top) - h0.max(z_bot)).max(0.0);
            if dz <= 0.0 {
                continue;
            }

            let w_avg = 0.5 * (w[idx0] + w[idx1]);
            let vort_avg = 0.5 * (vort_3d[idx0] + vort_3d[idx1]);

            // Only count updraft
            if w_avg > 0.0 {
                integral += w_avg * vort_avg * dz;
            }
        }
        *uh_val = integral;
    });

    Ok(uhel)
}
