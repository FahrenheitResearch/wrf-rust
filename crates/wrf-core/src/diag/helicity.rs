//! Updraft helicity diagnostic.
//!
//! Follows the wrf-python Fortran subroutine `DCALCUH` (`calc_uh.f90`) for its
//! derivative stencil, updraft gate, integration, and output halo. The layer
//! height reference intentionally differs; see [`compute_uhel`] for details.
//!
//! 1. Compute tem1(k) = w_destag(k) * vorticity(k) at each scalar level,
//!    where vorticity uses centered differences divided by the map scale
//!    factor (MAPFAC_M).
//! 2. For each column, compute the column-mean w over [z_bot, z_top].
//! 3. If the column-mean w > 0, integrate tem1 over the layer using the
//!    trapezoidal rule.  Otherwise UH = 0 for that column.

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Linearly interpolate value at height `z` between two levels.
/// `z0`/`z1` are heights at the two levels, `v0`/`v1` are the field values.
#[inline]
fn lerp_at(z: f64, z0: f64, z1: f64, v0: f64, v1: f64) -> f64 {
    if (z1 - z0).abs() < 1e-6 {
        0.5 * (v0 + v1)
    } else {
        let frac = (z - z0) / (z1 - z0);
        v0 + frac * (v1 - v0)
    }
}

/// Vertical vorticity on the exact horizontal/vertical stencil used by
/// NCAR wrf-python 1.3.4.1 `DCALCUH`.
///
/// The Fortran kernel initializes the work array to zero and only fills
/// 1-based `k=2..nz-2`, `j=2..ny-1`, and `i=2..nx-1`. Keeping those untouched
/// cells at zero is observable in the two-cell output halo and must not be
/// replaced with one-sided derivatives when parity with `uhel` is requested.
///
/// For `DCALCUH` parity, the centered derivatives deliberately preserve the
/// kernel's single `/MAPFAC_M` divisor. They are not rewritten to use the
/// staggered-map-factor and squared-mass-factor metric form used by WRF's
/// separate AVO/PVO kernel. See the pinned
/// [DCALCUH formula](https://github.com/NCAR/wrf-python/blob/31c923335227b22fa656fd589a5342b91103e939/fortran/calc_uh.f90#L67-L74).
fn dcalcuh_vorticity(
    u: &[f64],
    v: &[f64],
    mapfct: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let nxy = nx * ny;
    let mut vorticity = vec![0.0; nz * nxy];
    if nx < 3 || ny < 3 || nz < 3 {
        return vorticity;
    }

    let twodx = 2.0 * dx;
    let twody = 2.0 * dy;
    for k in 1..nz.saturating_sub(2) {
        let offset = k * nxy;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let ij = j * nx + i;
                let map_factor = mapfct[ij];
                let dvdx = (v[offset + ij + 1] - v[offset + ij - 1]) / (twodx * map_factor);
                let dudy = (u[offset + ij + nx] - u[offset + ij - nx]) / (twody * map_factor);
                vorticity[offset + ij] = dvdx - dudy;
            }
        }
    }

    vorticity
}

#[inline]
fn dcalcuh_output_column(i: usize, j: usize, nx: usize, ny: usize) -> bool {
    // Fortran: DO i=2,nx-2 and DO j=2,ny-2 (1-based, inclusive).
    i >= 1 && i < nx.saturating_sub(2) && j >= 1 && j < ny.saturating_sub(2)
}

/// Updraft helicity (m^2/s^2). `[ny, nx]`
///
/// UH = integral from z_bot to z_top of (w * zeta_z) dz
/// Default layer: 2-5 km AGL.
///
/// Follows the wrf-python Fortran `DCALCUH` for:
/// - Vorticity uses centered differences divided by map scale factor and the
///   same zero-initialized boundary/vertical stencil as `DCALCUH`.
/// - A column-mean w is computed first; only columns with positive mean w
///   contribute to UH (matching the Fortran's updraft check).
/// - The integrand is w*vort (pre-multiplied), integrated with the
///   trapezoidal rule.
///
/// # Vertical layer reference
///
/// `bottom_m` and `top_m` are true terrain-relative AGL bounds in wrf-rust:
/// the integration coordinate is mass-level geopotential height minus terrain.
/// The pinned NCAR wrapper instead passes staggered geopotential height to
/// `DCALCUH`, whose kernel adds the requested bounds to `zp(i,j,2)`, the first
/// staggered W level above terrain. See the pinned
/// [wrapper](https://github.com/NCAR/wrf-python/blob/31c923335227b22fa656fd589a5342b91103e939/src/wrf/g_helicity.py#L183-L208)
/// and [kernel bounds](https://github.com/NCAR/wrf-python/blob/31c923335227b22fa656fd589a5342b91103e939/fortran/calc_uh.f90#L82-L90).
/// Consequently, wrf-rust does not claim exact `DCALCUH` vertical-bound parity.
/// A representative 25--60 m first-level offset shifts NCAR's nominal 2--5 km
/// layer upward by the same amount. For illustration, an integrand proportional
/// to height changes by about 0.7--1.7% under that shift; a constant integrand
/// is unchanged, and real-profile sensitivity depends on vertical structure.
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

    // Try to read map scale factor (MAPFAC_M).  Fall back to 1.0 everywhere
    // if the variable is not present in the file.
    let mapfct: Vec<f64> = f.read_var("MAPFAC_M", t).unwrap_or_else(|_| vec![1.0; nxy]);

    // Exact zero-initialized DCALCUH derivative stencil. Reference:
    // https://github.com/NCAR/wrf-python/blob/31c923335227b22fa656fd589a5342b91103e939/fortran/calc_uh.f90#L55-L78
    let vort_3d = dcalcuh_vorticity(&u, &v, &mapfct, nx, ny, nz, dx, dy);

    // Pre-multiply: tem1(k) = w_destag(k) * vorticity(k)
    let mut tem1 = vec![0.0f64; nz * nxy];
    tem1.iter_mut().enumerate().for_each(|(idx, val)| {
        *val = w[idx] * vort_3d[idx];
    });

    // Integrate per column, checking column-mean w first (Fortran DCALCUH logic).
    let mut uhel = vec![0.0f64; nxy];
    uhel.iter_mut().enumerate().for_each(|(ij, uh_val)| {
        let i = ij % nx;
        let j = ij / nx;
        if !dcalcuh_output_column(i, j, nx, ny) {
            return;
        }

        // --- Step 1: compute column-mean w over [z_bot, z_top] ---
        let mut w_sum = 0.0f64;
        let mut depth = 0.0f64;

        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            let h0 = h_agl[idx0];
            let h1 = h_agl[idx1];

            // Skip layers entirely outside the integration bounds
            if h1 <= z_bot || h0 >= z_top {
                continue;
            }

            let z_lo = h0.max(z_bot);
            let z_hi = h1.min(z_top);
            let dz = z_hi - z_lo;
            if dz <= 0.0 {
                continue;
            }

            // Interpolate w to clamped endpoints
            let w_lo = if z_lo > h0 {
                lerp_at(z_lo, h0, h1, w[idx0], w[idx1])
            } else {
                w[idx0]
            };
            let w_hi = if z_hi < h1 {
                lerp_at(z_hi, h0, h1, w[idx0], w[idx1])
            } else {
                w[idx1]
            };

            w_sum += 0.5 * (w_lo + w_hi) * dz;
            depth += dz;
        }

        // Column-mean w check: only positive-mean columns contribute UH
        if depth <= 0.0 {
            return;
        }
        let w_mean = w_sum / depth;
        if w_mean <= 0.0 {
            return;
        }

        // --- Step 2: integrate tem1 (= w*vort) over [z_bot, z_top] ---
        let mut integral = 0.0f64;

        for k in 0..nz - 1 {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;

            let h0 = h_agl[idx0];
            let h1 = h_agl[idx1];

            if h1 <= z_bot || h0 >= z_top {
                continue;
            }

            let z_lo = h0.max(z_bot);
            let z_hi = h1.min(z_top);
            let dz = z_hi - z_lo;
            if dz <= 0.0 {
                continue;
            }

            // Interpolate tem1 to clamped endpoints
            let t0 = tem1[idx0];
            let t1 = tem1[idx1];

            let tem1_lo = if z_lo > h0 {
                lerp_at(z_lo, h0, h1, t0, t1)
            } else {
                t0
            };
            let tem1_hi = if z_hi < h1 {
                lerp_at(z_hi, h0, h1, t0, t1)
            } else {
                t1
            };

            integral += 0.5 * (tem1_lo + tem1_hi) * dz;
        }

        *uh_val = integral;
    });

    Ok(uhel)
}

#[cfg(test)]
mod tests {
    use super::{dcalcuh_output_column, dcalcuh_vorticity};

    #[test]
    fn dcalcuh_keeps_fortran_work_array_boundaries_zero() {
        let (nx, ny, nz) = (5, 5, 5);
        let nxy = nx * ny;
        let u = vec![0.0; nz * nxy];
        let mut v = vec![0.0; nz * nxy];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    v[k * nxy + j * nx + i] = i as f64;
                }
            }
        }

        let vorticity = dcalcuh_vorticity(&u, &v, &vec![1.0; nxy], nx, ny, nz, 1.0, 1.0);

        assert_eq!(vorticity[2 * nx + 2], 0.0, "lowest scalar level is zero");
        assert_eq!(vorticity[nxy + 2 * nx], 0.0, "west boundary is zero");
        assert_eq!(vorticity[nxy + 2 * nx + 2], 1.0);
        assert_eq!(
            vorticity[(nz - 2) * nxy + 2 * nx + 2],
            0.0,
            "top two scalar levels are zero"
        );
    }

    #[test]
    fn dcalcuh_output_uses_the_asymmetric_two_cell_fortran_halo() {
        let expected = [
            [false, false, false, false, false],
            [false, true, true, false, false],
            [false, true, true, false, false],
            [false, false, false, false, false],
            [false, false, false, false, false],
        ];
        for (j, row) in expected.iter().enumerate() {
            for (i, expected_value) in row.iter().enumerate() {
                assert_eq!(dcalcuh_output_column(i, j, 5, 5), *expected_value);
            }
        }
    }
}
