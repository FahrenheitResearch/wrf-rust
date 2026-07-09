//! WRF absolute- and potential-vorticity diagnostics.
//!
//! The kernels below reproduce NCAR `wrf-python` 1.3.4.1 commit
//! `31c923335227b22fa656fd589a5342b91103e939` `DCOMPUTEABSVORT` and
//! `DCOMPUTEPV` from `fortran/wrf_pvo.f90`. The WRF staggered winds and
//! stagger-specific map factors are deliberately retained: destaggering first
//! is not algebraically equivalent on a projected grid.

use crate::compute::ComputeOpts;
use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;

const WRF_G: f64 = 9.81;
const AVO_DISPLAY_SCALE: f64 = 1.0e5;
const PVO_FIRST_SCALE: f64 = 10_000.0;
const PVO_SECOND_SCALE: f64 = 100.0;

#[derive(Clone, Copy, Debug)]
struct GridGeometry {
    nx: usize,
    ny: usize,
    nz: usize,
    nxp1: usize,
    nyp1: usize,
    nxy: usize,
    scalar_len: usize,
    u_len: usize,
    v_len: usize,
    dx: f64,
    dy: f64,
}

impl GridGeometry {
    fn from_file(f: &WrfFile) -> WrfResult<Self> {
        // wrf-python reads the attributes for each diagnostic.  Do not use
        // WrfFile's open-time fallback spacing here: missing DX/DY must be an
        // explicit error for a map-metric diagnostic.
        let dx = f.global_attr_f64("DX")?;
        let dy = f.global_attr_f64("DY")?;
        Self::new(f.nx, f.ny, f.nz, dx, dy)
    }

    fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64) -> WrfResult<Self> {
        if nx < 2 || ny < 2 {
            return Err(WrfError::DimMismatch(format!(
                "AVO/PVO require at least a 2x2 horizontal mass grid, got {ny}x{nx}"
            )));
        }
        if nz == 0 {
            return Err(WrfError::DimMismatch(
                "AVO/PVO require at least one bottom_top level".to_string(),
            ));
        }
        if !dx.is_finite() || dx <= 0.0 {
            return Err(WrfError::InvalidParam(format!(
                "DX must be finite and positive for AVO/PVO, got {dx}"
            )));
        }
        if !dy.is_finite() || dy <= 0.0 {
            return Err(WrfError::InvalidParam(format!(
                "DY must be finite and positive for AVO/PVO, got {dy}"
            )));
        }

        let nxp1 = nx.checked_add(1).ok_or_else(grid_size_overflow)?;
        let nyp1 = ny.checked_add(1).ok_or_else(grid_size_overflow)?;
        let nxy = nx.checked_mul(ny).ok_or_else(grid_size_overflow)?;
        let scalar_len = nxy.checked_mul(nz).ok_or_else(grid_size_overflow)?;
        let u_plane_len = nxp1.checked_mul(ny).ok_or_else(grid_size_overflow)?;
        let v_plane_len = nx.checked_mul(nyp1).ok_or_else(grid_size_overflow)?;
        let u_len = u_plane_len
            .checked_mul(nz)
            .ok_or_else(grid_size_overflow)?;
        let v_len = v_plane_len
            .checked_mul(nz)
            .ok_or_else(grid_size_overflow)?;

        Ok(Self {
            nx,
            ny,
            nz,
            nxp1,
            nyp1,
            nxy,
            scalar_len,
            u_len,
            v_len,
            dx,
            dy,
        })
    }

    fn require_pvo_levels(self) -> WrfResult<Self> {
        if self.nz < 2 {
            return Err(WrfError::DimMismatch(format!(
                "PVO requires at least two bottom_top levels, got {}",
                self.nz
            )));
        }
        Ok(self)
    }

    #[inline(always)]
    fn scalar_index(self, k: usize, j: usize, i: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    #[inline(always)]
    fn u_index(self, k: usize, j: usize, i: usize) -> usize {
        (k * self.ny + j) * self.nxp1 + i
    }

    #[inline(always)]
    fn v_index(self, k: usize, j: usize, i: usize) -> usize {
        (k * self.nyp1 + j) * self.nx + i
    }

    #[inline(always)]
    fn mapfac_u_index(self, j: usize, i: usize) -> usize {
        j * self.nxp1 + i
    }

    #[inline(always)]
    fn mapfac_v_index(self, j: usize, i: usize) -> usize {
        j * self.nx + i
    }

    #[inline(always)]
    fn mass_index(self, j: usize, i: usize) -> usize {
        j * self.nx + i
    }
}

fn grid_size_overflow() -> WrfError {
    WrfError::DimMismatch("AVO/PVO grid dimensions overflow usize".to_string())
}

struct VorticityFields {
    u: Vec<f64>,
    v: Vec<f64>,
    mapfac_u: Vec<f64>,
    mapfac_v: Vec<f64>,
    mapfac_m: Vec<f64>,
    coriolis: Vec<f64>,
}

impl VorticityFields {
    fn validate(&self, grid: GridGeometry) -> WrfResult<()> {
        expect_len("U", self.u.len(), grid.u_len)?;
        expect_len("V", self.v.len(), grid.v_len)?;
        expect_len("MAPFAC_U", self.mapfac_u.len(), grid.ny * grid.nxp1)?;
        expect_len("MAPFAC_V", self.mapfac_v.len(), grid.nyp1 * grid.nx)?;
        expect_len("MAPFAC_M", self.mapfac_m.len(), grid.nxy)?;
        expect_len("F", self.coriolis.len(), grid.nxy)?;

        validate_map_factors("MAPFAC_U", &self.mapfac_u)?;
        validate_map_factors("MAPFAC_V", &self.mapfac_v)?;
        validate_map_factors("MAPFAC_M", &self.mapfac_m)?;

        if let Some((index, value)) = self
            .coriolis
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(WrfError::InvalidParam(format!(
                "F[{index}] must be finite for AVO/PVO, got {value}"
            )));
        }

        Ok(())
    }
}

fn expect_len(name: &str, actual: usize, expected: usize) -> WrfResult<()> {
    if actual != expected {
        return Err(WrfError::DimMismatch(format!(
            "{name} has {actual} values for AVO/PVO, expected {expected}"
        )));
    }
    Ok(())
}

fn validate_map_factors(name: &str, values: &[f64]) -> WrfResult<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(WrfError::InvalidParam(format!(
            "{name}[{index}] must be finite and positive for AVO/PVO, got {value}"
        )));
    }
    Ok(())
}

fn expect_var_shape(f: &WrfFile, name: &str, expected: &[usize]) -> WrfResult<()> {
    let actual = f.var_shape_no_time(name)?;
    if actual.as_slice() != expected {
        return Err(WrfError::DimMismatch(format!(
            "{name} has shape {actual:?} for AVO/PVO, expected {expected:?}"
        )));
    }
    Ok(())
}

fn read_vorticity_fields(
    f: &WrfFile,
    t: usize,
    grid: GridGeometry,
) -> WrfResult<VorticityFields> {
    expect_var_shape(f, "U", &[grid.nz, grid.ny, grid.nxp1])?;
    expect_var_shape(f, "V", &[grid.nz, grid.nyp1, grid.nx])?;
    expect_var_shape(f, "MAPFAC_U", &[grid.ny, grid.nxp1])?;
    expect_var_shape(f, "MAPFAC_V", &[grid.nyp1, grid.nx])?;
    expect_var_shape(f, "MAPFAC_M", &[grid.ny, grid.nx])?;
    expect_var_shape(f, "F", &[grid.ny, grid.nx])?;

    let fields = VorticityFields {
        u: f.read_var("U", t)?,
        v: f.read_var("V", t)?,
        mapfac_u: f.read_var("MAPFAC_U", t)?,
        mapfac_v: f.read_var("MAPFAC_V", t)?,
        mapfac_m: f.read_var("MAPFAC_M", t)?,
        coriolis: f.read_var("F", t)?,
    };
    Ok(fields)
}

/// Return WRF's map-metric absolute vorticity at a mass point in s^-1.
///
/// This retains the exact staggered-grid averaging and clamped one-sided
/// boundary indices used by `DCOMPUTEABSVORT`/`DCOMPUTEPV`.
#[inline]
fn absolute_vorticity_at(
    fields: &VorticityFields,
    grid: GridGeometry,
    k: usize,
    j: usize,
    i: usize,
) -> f64 {
    let jp1 = (j + 1).min(grid.ny - 1);
    let jm1 = j.saturating_sub(1);
    let ip1 = (i + 1).min(grid.nx - 1);
    let im1 = i.saturating_sub(1);
    let dsx = (ip1 - im1) as f64 * grid.dx;
    let dsy = (jp1 - jm1) as f64 * grid.dy;
    let mass_index = grid.mass_index(j, i);
    let mm = fields.mapfac_m[mass_index] * fields.mapfac_m[mass_index];

    let dudy = 0.5
        * (fields.u[grid.u_index(k, jp1, i)]
            / fields.mapfac_u[grid.mapfac_u_index(jp1, i)]
            + fields.u[grid.u_index(k, jp1, i + 1)]
                / fields.mapfac_u[grid.mapfac_u_index(jp1, i + 1)]
            - fields.u[grid.u_index(k, jm1, i)]
                / fields.mapfac_u[grid.mapfac_u_index(jm1, i)]
            - fields.u[grid.u_index(k, jm1, i + 1)]
                / fields.mapfac_u[grid.mapfac_u_index(jm1, i + 1)])
        / dsy
        * mm;

    let dvdx = 0.5
        * (fields.v[grid.v_index(k, j, ip1)]
            / fields.mapfac_v[grid.mapfac_v_index(j, ip1)]
            + fields.v[grid.v_index(k, j + 1, ip1)]
                / fields.mapfac_v[grid.mapfac_v_index(j + 1, ip1)]
            - fields.v[grid.v_index(k, j, im1)]
                / fields.mapfac_v[grid.mapfac_v_index(j, im1)]
            - fields.v[grid.v_index(k, j + 1, im1)]
                / fields.mapfac_v[grid.mapfac_v_index(j + 1, im1)])
        / dsx
        * mm;

    dvdx - dudy + fields.coriolis[mass_index]
}

fn wrf_absolute_vorticity(
    fields: &VorticityFields,
    grid: GridGeometry,
) -> WrfResult<Vec<f64>> {
    fields.validate(grid)?;
    let mut output = vec![0.0; grid.scalar_len];

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let index = grid.scalar_index(k, j, i);
                output[index] =
                    absolute_vorticity_at(fields, grid, k, j, i) * AVO_DISPLAY_SCALE;
            }
        }
    }

    Ok(output)
}

fn wrf_potential_vorticity(
    fields: &VorticityFields,
    theta: &[f64],
    pressure: &[f64],
    grid: GridGeometry,
) -> WrfResult<Vec<f64>> {
    let grid = grid.require_pvo_levels()?;
    fields.validate(grid)?;
    expect_len("full potential temperature", theta.len(), grid.scalar_len)?;
    expect_len("full pressure", pressure.len(), grid.scalar_len)?;

    let mut output = vec![0.0; grid.scalar_len];

    for k in 0..grid.nz {
        let kp1 = (k + 1).min(grid.nz - 1);
        let km1 = k.saturating_sub(1);

        for j in 0..grid.ny {
            let jp1 = (j + 1).min(grid.ny - 1);
            let jm1 = j.saturating_sub(1);
            let dsy = (jp1 - jm1) as f64 * grid.dy;

            for i in 0..grid.nx {
                let ip1 = (i + 1).min(grid.nx - 1);
                let im1 = i.saturating_sub(1);
                let dsx = (ip1 - im1) as f64 * grid.dx;
                let index = grid.scalar_index(k, j, i);
                let upper_index = grid.scalar_index(kp1, j, i);
                let lower_index = grid.scalar_index(km1, j, i);
                let dp = pressure[upper_index] - pressure[lower_index];

                if !dp.is_finite() || dp.abs() <= f64::EPSILON {
                    return Err(WrfError::InvalidParam(format!(
                        "PVO pressure difference at (k={k}, j={j}, i={i}) must be finite and non-zero, got {dp} Pa"
                    )));
                }

                let dudp = 0.5
                    * (fields.u[grid.u_index(kp1, j, i)]
                        + fields.u[grid.u_index(kp1, j, i + 1)]
                        - fields.u[grid.u_index(km1, j, i)]
                        - fields.u[grid.u_index(km1, j, i + 1)])
                    / dp;
                let dvdp = 0.5
                    * (fields.v[grid.v_index(kp1, j, i)]
                        + fields.v[grid.v_index(kp1, j + 1, i)]
                        - fields.v[grid.v_index(km1, j, i)]
                        - fields.v[grid.v_index(km1, j + 1, i)])
                    / dp;
                let dthdp = (theta[upper_index] - theta[lower_index]) / dp;
                let mapfac_m = fields.mapfac_m[grid.mass_index(j, i)];
                let dthdx = (theta[grid.scalar_index(k, j, ip1)]
                    - theta[grid.scalar_index(k, j, im1)])
                    / dsx
                    * mapfac_m;
                let dthdy = (theta[grid.scalar_index(k, jp1, i)]
                    - theta[grid.scalar_index(k, jm1, i)])
                    / dsy
                    * mapfac_m;
                let avort = absolute_vorticity_at(fields, grid, k, j, i);

                // Preserve the two scale operations in the pinned Fortran:
                // first *10000, then *100, yielding PVU.
                let pv = -WRF_G
                    * (dthdp * avort - dvdp * dthdx + dudp * dthdy)
                    * PVO_FIRST_SCALE;
                output[index] = pv * PVO_SECOND_SCALE;
            }
        }
    }

    Ok(output)
}

/// Absolute vorticity (10^-5 s^-1). Shape: `[nz, ny, nx]`.
///
/// This is a direct Rust translation of wrf-python 1.3.4.1
/// `DCOMPUTEABSVORT`: raw staggered `U`/`V`, `MAPFAC_U`/`MAPFAC_V`, squared
/// `MAPFAC_M`, raw WRF `F`, and the Fortran boundary formulas are preserved.
pub fn compute_avo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let grid = GridGeometry::from_file(f)?;
    let fields = read_vorticity_fields(f, t, grid)?;
    wrf_absolute_vorticity(&fields, grid)
}

/// Potential vorticity (PVU). Shape: `[nz, ny, nx]`.
///
/// This is a direct Rust translation of wrf-python 1.3.4.1 `DCOMPUTEPV`,
/// including the vertical stretching and both horizontal baroclinic terms.
/// Pressure and potential temperature remain on WRF mass levels; only `U` and
/// `V` use their native horizontal staggering.
pub fn compute_pvo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let grid = GridGeometry::from_file(f)?.require_pvo_levels()?;
    let fields = read_vorticity_fields(f, t, grid)?;

    expect_var_shape(f, "T", &[grid.nz, grid.ny, grid.nx])?;
    expect_var_shape(f, "P", &[grid.nz, grid.ny, grid.nx])?;
    expect_var_shape(f, "PB", &[grid.nz, grid.ny, grid.nx])?;

    let theta = f.full_theta(t)?;
    let pressure = f.full_pressure(t)?;
    wrf_potential_vorticity(&fields, &theta, &pressure, grid)
}

#[cfg(test)]
mod tests {
    use super::{
        wrf_absolute_vorticity, wrf_potential_vorticity, GridGeometry, VorticityFields,
        AVO_DISPLAY_SCALE, PVO_FIRST_SCALE, PVO_SECOND_SCALE, WRF_G,
    };
    use crate::error::WrfError;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.16e}, expected {expected:.16e}, tolerance {tolerance:.3e}"
        );
    }

    fn staggered_winds(
        grid: GridGeometry,
        mut u_at: impl FnMut(usize, f64, f64) -> f64,
        mut v_at: impl FnMut(usize, f64, f64) -> f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut u = Vec::with_capacity(grid.u_len);
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nxp1 {
                    u.push(u_at(k, i as f64 * grid.dx, j as f64 * grid.dy));
                }
            }
        }

        let mut v = Vec::with_capacity(grid.v_len);
        for k in 0..grid.nz {
            for j in 0..grid.nyp1 {
                for i in 0..grid.nx {
                    v.push(v_at(k, i as f64 * grid.dx, j as f64 * grid.dy));
                }
            }
        }
        (u, v)
    }

    fn unit_metric_fields(
        grid: GridGeometry,
        u: Vec<f64>,
        v: Vec<f64>,
        coriolis: Vec<f64>,
    ) -> VorticityFields {
        VorticityFields {
            u,
            v,
            mapfac_u: vec![1.0; grid.ny * grid.nxp1],
            mapfac_v: vec![1.0; grid.nyp1 * grid.nx],
            mapfac_m: vec![1.0; grid.nxy],
            coriolis,
        }
    }

    #[test]
    fn avo_solid_body_rotation_matches_all_fortran_boundaries() {
        let grid = GridGeometry::new(5, 4, 2, 2_000.0, 3_000.0).unwrap();
        let angular_velocity = 1.5e-4;
        let coriolis = 8.0e-5;
        let (u, v) = staggered_winds(
            grid,
            |_, _, y| -angular_velocity * y,
            |_, x, _| angular_velocity * x,
        );
        let fields = unit_metric_fields(grid, u, v, vec![coriolis; grid.nxy]);

        let output = wrf_absolute_vorticity(&fields, grid).unwrap();
        let expected = (2.0 * angular_velocity + coriolis) * AVO_DISPLAY_SCALE;
        for value in output {
            assert_close(value, expected, 1.0e-12);
        }
    }

    #[test]
    fn avo_uses_stagger_specific_and_squared_mass_map_factors() {
        let grid = GridGeometry::new(4, 3, 1, 1_500.0, 2_500.0).unwrap();
        let dudy = -1.3e-4;
        let dvdx = 2.1e-4;
        let mapfac_u: Vec<f64> = (0..grid.ny * grid.nxp1)
            .map(|index| 0.8 + index as f64 * 0.013)
            .collect();
        let mapfac_v: Vec<f64> = (0..grid.nyp1 * grid.nx)
            .map(|index| 1.4 - index as f64 * 0.011)
            .collect();
        let mapfac_m: Vec<f64> = (0..grid.nxy)
            .map(|index| 0.9 + index as f64 * 0.021)
            .collect();
        let coriolis: Vec<f64> = (0..grid.nxy)
            .map(|index| 3.0e-5 + index as f64 * 7.0e-7)
            .collect();

        let mut u = vec![0.0; grid.u_len];
        for j in 0..grid.ny {
            let y = j as f64 * grid.dy;
            for i in 0..grid.nxp1 {
                let map_index = grid.mapfac_u_index(j, i);
                u[grid.u_index(0, j, i)] = mapfac_u[map_index] * dudy * y;
            }
        }
        let mut v = vec![0.0; grid.v_len];
        for j in 0..grid.nyp1 {
            for i in 0..grid.nx {
                let x = i as f64 * grid.dx;
                let map_index = grid.mapfac_v_index(j, i);
                v[grid.v_index(0, j, i)] = mapfac_v[map_index] * dvdx * x;
            }
        }

        let fields = VorticityFields {
            u,
            v,
            mapfac_u,
            mapfac_v,
            mapfac_m: mapfac_m.clone(),
            coriolis: coriolis.clone(),
        };
        let output = wrf_absolute_vorticity(&fields, grid).unwrap();

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let mass_index = grid.mass_index(j, i);
                let expected = ((dvdx - dudy) * mapfac_m[mass_index].powi(2)
                    + coriolis[mass_index])
                    * AVO_DISPLAY_SCALE;
                assert_close(output[mass_index], expected, 1.0e-11);
            }
        }
    }

    #[test]
    fn pvo_includes_stretching_and_both_baroclinic_terms() {
        let grid = GridGeometry::new(4, 3, 3, 1_500.0, 2_500.0).unwrap();
        let pressure_levels = [90_000.0, 80_000.0, 65_000.0];
        let dtheta_dp = -4.0e-4;
        let dtheta_dx = 2.5e-4;
        let dtheta_dy = 1.5e-4;
        let dudp = 6.0e-5;
        let dvdp = -4.0e-5;
        let dudy = -1.2e-4;
        let dvdx = 1.7e-4;

        let (mut u, mut v) = staggered_winds(
            grid,
            |k, _, y| 10.0 + dudp * (pressure_levels[k] - 80_000.0) + dudy * y,
            |k, x, _| -3.0 + dvdp * (pressure_levels[k] - 80_000.0) + dvdx * x,
        );
        let mapfac_u: Vec<f64> = (0..grid.ny * grid.nxp1)
            .map(|index| 0.85 + index as f64 * 0.011)
            .collect();
        let mapfac_v: Vec<f64> = (0..grid.nyp1 * grid.nx)
            .map(|index| 1.25 - index as f64 * 0.009)
            .collect();
        let mapfac_m: Vec<f64> = (0..grid.nxy)
            .map(|index| 1.05 + index as f64 * 0.017)
            .collect();
        let coriolis: Vec<f64> = (0..grid.nxy)
            .map(|index| 7.0e-5 + index as f64 * 4.0e-7)
            .collect();

        // Make U/MAPFAC_U and V/MAPFAC_V the analytic horizontal winds.
        // The vertical terms in DCOMPUTEPV deliberately use raw U/V, so their
        // expected pressure derivatives retain the adjacent face metrics.
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nxp1 {
                    u[grid.u_index(k, j, i)] *= mapfac_u[grid.mapfac_u_index(j, i)];
                }
            }
            for j in 0..grid.nyp1 {
                for i in 0..grid.nx {
                    v[grid.v_index(k, j, i)] *= mapfac_v[grid.mapfac_v_index(j, i)];
                }
            }
        }
        let fields = VorticityFields {
            u,
            v,
            mapfac_u: mapfac_u.clone(),
            mapfac_v: mapfac_v.clone(),
            mapfac_m: mapfac_m.clone(),
            coriolis: coriolis.clone(),
        };

        let mut pressure = vec![0.0; grid.scalar_len];
        let mut theta = vec![0.0; grid.scalar_len];
        for (k, level_pressure) in pressure_levels.iter().copied().enumerate() {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let index = grid.scalar_index(k, j, i);
                    pressure[index] = level_pressure;
                    theta[index] = 300.0
                        + dtheta_dp * (level_pressure - 80_000.0)
                        + dtheta_dx * i as f64 * grid.dx
                        + dtheta_dy * j as f64 * grid.dy;
                }
            }
        }

        let output = wrf_potential_vorticity(&fields, &theta, &pressure, grid).unwrap();
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let mass_index = grid.mass_index(j, i);
                    let mapfac = mapfac_m[mass_index];
                    let avort =
                        (dvdx - dudy) * mapfac.powi(2) + coriolis[mass_index];
                    let local_dudp = 0.5
                        * dudp
                        * (mapfac_u[grid.mapfac_u_index(j, i)]
                            + mapfac_u[grid.mapfac_u_index(j, i + 1)]);
                    let local_dvdp = 0.5
                        * dvdp
                        * (mapfac_v[grid.mapfac_v_index(j, i)]
                            + mapfac_v[grid.mapfac_v_index(j + 1, i)]);
                    let full_ertel_term = dtheta_dp * avort
                        - local_dvdp * dtheta_dx * mapfac
                        + local_dudp * dtheta_dy * mapfac;
                    let expected = -WRF_G * full_ertel_term * PVO_FIRST_SCALE
                        * PVO_SECOND_SCALE;
                    let index = grid.scalar_index(k, j, i);
                    assert_close(output[index], expected, 1.0e-8);

                    let stretching_only = -WRF_G * dtheta_dp * avort * PVO_FIRST_SCALE
                        * PVO_SECOND_SCALE;
                    assert!((output[index] - stretching_only).abs() > 1.0e-3);
                }
            }
        }
    }

    #[test]
    fn malformed_map_factor_is_a_clear_error() {
        let grid = GridGeometry::new(2, 2, 1, 1_000.0, 1_000.0).unwrap();
        let (u, v) = staggered_winds(grid, |_, _, _| 0.0, |_, _, _| 0.0);
        let mut fields = unit_metric_fields(grid, u, v, vec![0.0; grid.nxy]);
        fields.mapfac_u[1] = 0.0;

        let error = wrf_absolute_vorticity(&fields, grid).unwrap_err();
        assert!(matches!(error, WrfError::InvalidParam(_)));
        assert!(error.to_string().contains("MAPFAC_U[1]"));
    }

    #[test]
    fn zero_vertical_pressure_difference_is_a_clear_error() {
        let grid = GridGeometry::new(2, 2, 2, 1_000.0, 1_000.0).unwrap();
        let (u, v) = staggered_winds(grid, |_, _, _| 0.0, |_, _, _| 0.0);
        let fields = unit_metric_fields(grid, u, v, vec![8.0e-5; grid.nxy]);
        let theta = vec![300.0; grid.scalar_len];
        let pressure = vec![80_000.0; grid.scalar_len];

        let error = wrf_potential_vorticity(&fields, &theta, &pressure, grid).unwrap_err();
        assert!(matches!(error, WrfError::InvalidParam(_)));
        assert!(error.to_string().contains("pressure difference"));
    }

    #[test]
    fn malformed_staggered_length_is_a_clear_error() {
        let grid = GridGeometry::new(3, 2, 1, 1_000.0, 1_000.0).unwrap();
        let (u, v) = staggered_winds(grid, |_, _, _| 0.0, |_, _, _| 0.0);
        let mut fields = unit_metric_fields(grid, u, v, vec![0.0; grid.nxy]);
        fields.v.pop();

        let error = wrf_absolute_vorticity(&fields, grid).unwrap_err();
        assert!(matches!(error, WrfError::DimMismatch(_)));
        assert!(error.to_string().contains("V has"));
    }
}
