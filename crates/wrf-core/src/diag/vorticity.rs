//! Vorticity diagnostic variables: avo, pvo

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const OMEGA: f64 = 7.2921159e-5; // Earth's angular velocity (rad/s)

/// Absolute vorticity (s^-1). `[nz, ny, nx]`
///
/// wrf-python's `avo` (fortran/wrf_pvo.f90 DCOMPUTEABSVORT, Apache-2.0),
/// in s^-1 rather than its 10^-5 s^-1: on the native staggered winds, each
/// face wind divided by its own map factor, centred differences (one-sided
/// at the domain edge, as wrf-python clamps), times the mass-point map
/// factor squared, plus the file's Coriolis parameter F (2 Omega sin(lat)
/// when the file carries no F):
///
///   dvdx = 0.5 (v(i+1,j)/mv + v(i+1,j+1)/mv - v(i-1,j)/mv - v(i-1,j+1)/mv) / dsx * m^2
///   dudy = 0.5 (u(i,j+1)/mu + u(i+1,j+1)/mu - u(i,j-1)/mu - u(i+1,j-1)/mu) / dsy * m^2
///   avo  = dvdx - dudy + F
///
/// A latitude-longitude grid is not conformal: its map factor is
/// 1/cos(lat) along x and 1 along y, which one isotropic factor cannot
/// carry (a global file may carry only MAPFAC_MX and MAPFAC_MY).  When the
/// file's MAPFAC_MX and MAPFAC_MY differ, the same stencil takes the
/// orthogonal-grid form instead, which is the spherical curl there:
///
///   avo = mx my ( d(v/my)/dx - d(u/mx)/dy ) + F
///
/// with u over MAPFAC_UX and v over MAPFAC_VY (or MAPFAC_MX / MAPFAC_MY
/// averaged onto the faces when the file carries no face factors).
/// Lambert, Mercator and polar-stereographic files keep the form above.
///
/// Breakage it fixes (up to 0.2.39): the earlier body passed (v, u) to
/// `met::dynamics::vorticity(u, v, ..)`, so it returned du/dx - dv/dy --
/// zero for solid rotation -- and every closed low vanished from the
/// 500 hPa vorticity map while jet shear zones stayed (correlation 0.23
/// with HRRR's ABSV at the same analysis at 500 hPa, 0.06 at 850 hPa, with
/// the winds themselves matching to 0.3 kt; 0.81 and 0.51 after the fix).
pub fn compute_avo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.read_var("U", t)?;
    let v = f.read_var("V", t)?;
    let (nx, ny, nz) = (f.nx, f.ny, f.nz);
    let nxy = nx * ny;
    let cor = match f.read_var("F", t) {
        Ok(cor) if cor.len() == nxy => cor,
        _ => f.xlat(t)?.iter().map(|lat| 2.0 * OMEGA * lat.to_radians().sin()).collect(),
    };
    if let Some((mx, my)) = anisotropic_map_factors(f, t, nxy) {
        let mxu = f
            .read_var("MAPFAC_UX", t)
            .ok()
            .filter(|m| m.len() == ny * (nx + 1))
            .unwrap_or_else(|| mass_to_u_faces(&mx, nx, ny));
        let myv = f
            .read_var("MAPFAC_VY", t)
            .ok()
            .filter(|m| m.len() == (ny + 1) * nx)
            .unwrap_or_else(|| mass_to_v_faces(&my, nx, ny));
        return absolute_vorticity_staggered_xy(
            &u, &v, &mxu, &myv, &mx, &my, &cor, nx, ny, nz, f.dx, f.dy,
        );
    }
    let ones_u = vec![1.0; (nx + 1) * ny];
    let ones_v = vec![1.0; nx * (ny + 1)];
    let ones_m = vec![1.0; nxy];
    let msfu = f.read_var("MAPFAC_U", t).unwrap_or(ones_u);
    let msfv = f.read_var("MAPFAC_V", t).unwrap_or(ones_v);
    let msfm = f.read_var("MAPFAC_M", t).unwrap_or(ones_m);
    absolute_vorticity_staggered(&u, &v, &msfu, &msfv, &msfm, &cor, nx, ny, nz, f.dx, f.dy)
}

/// `(MAPFAC_MX, MAPFAC_MY)` when the file carries both and they differ
/// (a latitude-longitude grid); `None` for a conformal grid, whose two
/// factors are the same, and for a file without them.
fn anisotropic_map_factors(f: &WrfFile, t: usize, nxy: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    if !f.has_var("MAPFAC_MX") || !f.has_var("MAPFAC_MY") {
        return None;
    }
    let mx = f.read_var("MAPFAC_MX", t).ok().filter(|m| m.len() == nxy)?;
    let my = f.read_var("MAPFAC_MY", t).ok().filter(|m| m.len() == nxy)?;
    let differ = mx.iter().zip(&my).any(|(a, b)| (a - b).abs() > 1.0e-6 * b.abs());
    differ.then_some((mx, my))
}

/// A mass-point plane `[ny][nx]` averaged onto the u faces `[ny][nx+1]`
/// (the edge faces take the edge column).
fn mass_to_u_faces(m: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(ny * (nx + 1));
    for j in 0..ny {
        let row = &m[j * nx..(j + 1) * nx];
        out.push(row[0]);
        out.extend(row.windows(2).map(|w| 0.5 * (w[0] + w[1])));
        out.push(row[nx - 1]);
    }
    out
}

/// A mass-point plane `[ny][nx]` averaged onto the v faces `[ny+1][nx]`
/// (the edge faces take the edge row).
fn mass_to_v_faces(m: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity((ny + 1) * nx);
    out.extend_from_slice(&m[..nx]);
    for j in 1..ny {
        out.extend((0..nx).map(|i| 0.5 * (m[(j - 1) * nx + i] + m[j * nx + i])));
    }
    out.extend_from_slice(&m[(ny - 1) * nx..]);
    out
}

/// The arithmetic of [`compute_avo`] on plain arrays, row-major
/// `[k][j][i]` with j south to north: `u` is `[nz][ny][nx+1]`, `v`
/// `[nz][ny+1][nx]`, `msfu` `[ny][nx+1]`, `msfv` `[ny+1][nx]`, `msfm` and
/// `cor` `[ny][nx]`.
#[allow(clippy::too_many_arguments)]
pub fn absolute_vorticity_staggered(
    u: &[f64],
    v: &[f64],
    msfu: &[f64],
    msfv: &[f64],
    msfm: &[f64],
    cor: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
) -> WrfResult<Vec<f64>> {
    check_staggered_shapes(u, v, msfu, msfv, &[msfm, cor], nx, ny, nz)?;
    let mm: Vec<f64> = msfm.iter().map(|m| m * m).collect();
    Ok(staggered_kernel(u, v, msfu, msfv, &mm, cor, nx, ny, nz, dx, dy))
}

/// The latitude-longitude form of [`compute_avo`] on plain arrays:
/// `mx my (d(v/my)/dx - d(u/mx)/dy) + cor`.  `mxu` is the x map factor on
/// the u faces `[ny][nx+1]`, `myv` the y map factor on the v faces
/// `[ny+1][nx]`; `mx`, `my` and `cor` are mass-point planes `[ny][nx]`.
#[allow(clippy::too_many_arguments)]
pub fn absolute_vorticity_staggered_xy(
    u: &[f64],
    v: &[f64],
    mxu: &[f64],
    myv: &[f64],
    mx: &[f64],
    my: &[f64],
    cor: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
) -> WrfResult<Vec<f64>> {
    check_staggered_shapes(u, v, mxu, myv, &[mx, my, cor], nx, ny, nz)?;
    let mm: Vec<f64> = mx.iter().zip(my).map(|(a, b)| a * b).collect();
    Ok(staggered_kernel(u, v, mxu, myv, &mm, cor, nx, ny, nz, dx, dy))
}

#[allow(clippy::too_many_arguments)]
fn check_staggered_shapes(
    u: &[f64],
    v: &[f64],
    ufac: &[f64],
    vfac: &[f64],
    mass_planes: &[&[f64]],
    nx: usize,
    ny: usize,
    nz: usize,
) -> WrfResult<()> {
    let nxu = nx + 1;
    let nxy = nx * ny;
    if u.len() != nz * ny * nxu || v.len() != nz * (ny + 1) * nx || ufac.len() != ny * nxu
        || vfac.len() != (ny + 1) * nx || mass_planes.iter().any(|p| p.len() != nxy)
        || nx < 2 || ny < 2
    {
        return Err(crate::error::WrfError::DimMismatch(format!(
            "avo needs U [{nz},{ny},{nxu}], V [{nz},{},{nx}] and their map factors",
            ny + 1
        )));
    }
    Ok(())
}

/// DCOMPUTEABSVORT's loop: `ufac` divides u on its faces, `vfac` v on its
/// faces, `mm` multiplies both differences at the mass point.
#[allow(clippy::too_many_arguments)]
fn staggered_kernel(
    u: &[f64],
    v: &[f64],
    ufac: &[f64],
    vfac: &[f64],
    mm: &[f64],
    cor: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let nxu = nx + 1;
    let nxy = nx * ny;
    let mut av = vec![0.0f64; nz * nxy];
    for k in 0..nz {
        let uk = &u[k * ny * nxu..(k + 1) * ny * nxu];
        let vk = &v[k * (ny + 1) * nx..(k + 1) * (ny + 1) * nx];
        for j in 0..ny {
            let jp1 = (j + 1).min(ny - 1);
            let jm1 = j.saturating_sub(1);
            let dsy = (jp1 - jm1) as f64 * dy;
            for i in 0..nx {
                let ip1 = (i + 1).min(nx - 1);
                let im1 = i.saturating_sub(1);
                let dsx = (ip1 - im1) as f64 * dx;
                let mm = mm[j * nx + i];
                let uu = |jj: usize, ii: usize| uk[jj * nxu + ii] / ufac[jj * nxu + ii];
                let vv = |jj: usize, ii: usize| vk[jj * nx + ii] / vfac[jj * nx + ii];
                let dudy = 0.5 * (uu(jp1, i) + uu(jp1, i + 1) - uu(jm1, i) - uu(jm1, i + 1)) / dsy * mm;
                let dvdx = 0.5 * (vv(j, ip1) + vv(j + 1, ip1) - vv(j, im1) - vv(j + 1, im1)) / dsx * mm;
                av[k * nxy + j * nx + i] = dvdx - dudy + cor[j * nx + i];
            }
        }
    }
    av
}

/// Relative vorticity of one mass-point level on an orthogonal grid,
/// `mx my (d(v/my)/dx - d(u/mx)/dy)`: the spherical curl on a
/// latitude-longitude grid.  [`compute_pvo`] uses it when the file's map
/// factors are anisotropic.
#[allow(clippy::too_many_arguments)]
fn relative_vorticity_xy(
    u: &[f64],
    v: &[f64],
    mx: &[f64],
    my: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let v_over_my: Vec<f64> = v.iter().zip(my).map(|(v, m)| v / m).collect();
    let u_over_mx: Vec<f64> = u.iter().zip(mx).map(|(u, m)| u / m).collect();
    let dvdx = crate::met::dynamics::gradient_x(&v_over_my, nx, ny, dx);
    let dudy = crate::met::dynamics::gradient_y(&u_over_mx, nx, ny, dy);
    (0..nx * ny)
        .map(|ij| mx[ij] * my[ij] * (dvdx[ij] - dudy[ij]))
        .collect()
}

/// Potential vorticity (PVU, 1 PVU = 10^-6 K m^2 kg^-1 s^-1). `[nz, ny, nx]`
///
/// PVO = -g * (f + zeta) * (dtheta/dp)
///
/// zeta carries no map factor on a conformal grid (as up to 0.2.39); on a
/// latitude-longitude file it is `mx my (d(v/my)/dx - d(u/mx)/dy)`, as in
/// [`compute_avo`].
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
    let latlon = anisotropic_map_factors(f, t, nxy);

    let mut pvo = vec![0.0f64; nz * nxy];

    pvo.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];

        // (u, v) in that order: vorticity(u, v) is dv/dx - du/dy.  The
        // earlier (v, u) gave du/dx - dv/dy (see compute_avo).  A
        // latitude-longitude file takes its x and y map factors, as avo does.
        let rel_vort = match &latlon {
            Some((mx, my)) => relative_vorticity_xy(u_plane, v_plane, mx, my, nx, ny, dx, dy),
            None => crate::met::dynamics::vorticity(u_plane, v_plane, nx, ny, dx, dy),
        };

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
