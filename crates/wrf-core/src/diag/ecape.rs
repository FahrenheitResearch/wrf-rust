//! ECAPE-family diagnostics backed by ecape-rs.
//!
//! Uses earth-rotated winds plus a synthetic surface layer from PSFC/T2/Q2/U10/V10
//! so parcel selection and storm motion see the same near-surface context as the
//! rest of wrf-rust's severe-weather diagnostics.

use ecape_rs::{
    calc_ecape_parcel, CapeType, ParcelOptions, StormMotionType as EcapeStormMotionType,
};
use rayon::prelude::*;

use crate::compute::{ComputeOpts, StormMotion};
use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;

#[derive(Debug, Clone, Copy, Default)]
struct EcapeSummary {
    ecape: f64,
    ncape: f64,
    cape: f64,
    cin: f64,
    lfc: f64,
    el: f64,
}

const ECAPE_STACK_FIELDS: usize = 6;

fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
    let td_c = crate::met::composite::dewpoint_from_q(q_kgkg, p_pa / 100.0);
    (td_c + 273.15).min(temp_k)
}

fn validate_ecape_opts(opts: &ComputeOpts) -> WrfResult<()> {
    if opts.parcel_pressure.is_some()
        || opts.parcel_temperature.is_some()
        || opts.parcel_dewpoint.is_some()
    {
        return Err(WrfError::InvalidParam(
            "ecape variables do not support custom parcel thermodynamics; use parcel_type=\"sb\", \"ml\", or \"mu\"".into(),
        ));
    }

    if opts.storm_motion_method.is_some() && opts.storm_motion.is_none() {
        return Err(WrfError::InvalidParam(
            "ecape variables do not use storm_motion_method; use storm_motion for an explicit vector or storm_motion_type=\"bunkers_rm\"|\"bunkers_lm\"|\"mean_wind\"".into(),
        ));
    }

    if opts.top_m.is_some()
        || opts.bottom_m.is_some()
        || opts.depth_m.is_some()
        || opts.bottom_p.is_some()
        || opts.top_p.is_some()
    {
        return Err(WrfError::InvalidParam(
            "ecape variables do not support CAPE/SRH layer-bound options like top_m, bottom_m, depth_m, bottom_p, or top_p".into(),
        ));
    }

    Ok(())
}

fn resolve_parcel_type(opts: &ComputeOpts) -> WrfResult<CapeType> {
    match opts
        .parcel_type
        .as_deref()
        .unwrap_or("sb")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "sb" | "surface_based" | "surface" => Ok(CapeType::SurfaceBased),
        "ml" | "mixed_layer" => Ok(CapeType::MixedLayer),
        "mu" | "most_unstable" => Ok(CapeType::MostUnstable),
        other => Err(WrfError::InvalidParam(format!(
            "unsupported ECAPE parcel_type '{other}'; expected 'sb', 'ml', or 'mu'"
        ))),
    }
}

fn resolve_storm_motion_type(opts: &ComputeOpts) -> WrfResult<EcapeStormMotionType> {
    match opts
        .storm_motion_type
        .as_deref()
        .unwrap_or("right_moving")
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "bunkers_rm" | "right_moving" | "right" | "rm" => {
            Ok(EcapeStormMotionType::RightMoving)
        }
        "bunkers_lm" | "left_moving" | "left" | "lm" => Ok(EcapeStormMotionType::LeftMoving),
        "mean_wind" | "mean" => Ok(EcapeStormMotionType::MeanWind),
        "user_defined" | "custom" => Ok(EcapeStormMotionType::UserDefined),
        other => Err(WrfError::InvalidParam(format!(
            "unsupported ECAPE storm_motion_type '{other}'; expected 'bunkers_rm', 'bunkers_lm', or 'mean_wind'"
        ))),
    }
}

fn build_parcel_options(
    opts: &ComputeOpts,
    ij: usize,
    parcel_type: CapeType,
    storm_motion_type: EcapeStormMotionType,
) -> ParcelOptions {
    let mut parcel_opts = ParcelOptions {
        cape_type: parcel_type,
        storm_motion_type,
        entrainment_rate: opts.entrainment_rate,
        pseudoadiabatic: opts.pseudoadiabatic,
        ..ParcelOptions::default()
    };

    if let Some(storm_motion) = opts.storm_motion.as_ref() {
        let (u, v) = storm_motion.at(ij);
        parcel_opts.storm_motion_type = EcapeStormMotionType::UserDefined;
        parcel_opts.storm_motion_u_ms = Some(u);
        parcel_opts.storm_motion_v_ms = Some(v);
    }

    parcel_opts
}

fn push_level(
    pressure_pa: &mut Vec<f64>,
    height_m: &mut Vec<f64>,
    temp_k: &mut Vec<f64>,
    dewpoint_k: &mut Vec<f64>,
    u_ms: &mut Vec<f64>,
    v_ms: &mut Vec<f64>,
    p: f64,
    z: f64,
    t: f64,
    td: f64,
    u: f64,
    v: f64,
) {
    if !p.is_finite()
        || !z.is_finite()
        || !t.is_finite()
        || !td.is_finite()
        || !u.is_finite()
        || !v.is_finite()
    {
        return;
    }

    if let (Some(&last_p), Some(&last_z)) = (pressure_pa.last(), height_m.last()) {
        if p >= last_p || z <= last_z {
            return;
        }
    }

    pressure_pa.push(p);
    height_m.push(z);
    temp_k.push(t);
    dewpoint_k.push(td.min(t));
    u_ms.push(u);
    v_ms.push(v);
}

fn build_surface_augmented_ecape_column(
    pressure_3d: &[f64],
    temperature_k_3d: &[f64],
    qv_3d: &[f64],
    height_agl_3d: &[f64],
    u_earth_3d: &[f64],
    v_earth_3d: &[f64],
    psfc_pa: f64,
    t2_k: f64,
    q2_kgkg: f64,
    u10_ms: f64,
    v10_ms: f64,
    nz: usize,
    nxy: usize,
    ij: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pressure_pa = Vec::with_capacity(nz + 1);
    let mut height_m = Vec::with_capacity(nz + 1);
    let mut temp_k = Vec::with_capacity(nz + 1);
    let mut dewpoint_k = Vec::with_capacity(nz + 1);
    let mut u_ms = Vec::with_capacity(nz + 1);
    let mut v_ms = Vec::with_capacity(nz + 1);

    push_level(
        &mut pressure_pa,
        &mut height_m,
        &mut temp_k,
        &mut dewpoint_k,
        &mut u_ms,
        &mut v_ms,
        psfc_pa,
        0.0,
        t2_k,
        dewpoint_k_from_q(q2_kgkg, psfc_pa, t2_k),
        u10_ms,
        v10_ms,
    );

    for k in 0..nz {
        let idx = k * nxy + ij;
        push_level(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut dewpoint_k,
            &mut u_ms,
            &mut v_ms,
            pressure_3d[idx],
            height_agl_3d[idx],
            temperature_k_3d[idx],
            dewpoint_k_from_q(qv_3d[idx], pressure_3d[idx], temperature_k_3d[idx]),
            u_earth_3d[idx],
            v_earth_3d[idx],
        );
    }

    (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms)
}

fn ecape_cache_key(opts: &ComputeOpts) -> Option<String> {
    let parcel_type = opts
        .parcel_type
        .as_deref()
        .unwrap_or("sb")
        .trim()
        .to_ascii_lowercase();
    let storm_motion_type = opts
        .storm_motion_type
        .as_deref()
        .unwrap_or("right_moving")
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_");
    let entrainment = opts
        .entrainment_rate
        .map(|v| format!("{:016x}", v.to_bits()))
        .unwrap_or_else(|| "auto".to_string());
    let pseudoadiabatic = match opts.pseudoadiabatic {
        Some(true) => "1",
        Some(false) => "0",
        None => "default",
    };
    let lake_interp = opts
        .lake_interp
        .map(|v| format!("{:016x}", v.to_bits()))
        .unwrap_or_else(|| "none".to_string());
    let storm_motion = match opts.storm_motion.as_ref() {
        None => "default".to_string(),
        Some(StormMotion::Uniform { u, v }) => {
            format!("uniform_{:016x}_{:016x}", u.to_bits(), v.to_bits())
        }
        Some(StormMotion::Grid { .. }) => return None,
    };

    Some(format!(
        "ecape_stack_{parcel_type}_{storm_motion_type}_{entrainment}_{pseudoadiabatic}_{lake_interp}_{storm_motion}"
    ))
}

fn pack_ecape_stack(
    ecape: &[f64],
    ncape: &[f64],
    cape: &[f64],
    cin: &[f64],
    lfc: &[f64],
    el: &[f64],
) -> Vec<f64> {
    let nxy = ecape.len();
    let mut stacked = Vec::with_capacity(ECAPE_STACK_FIELDS * nxy);
    stacked.extend_from_slice(ecape);
    stacked.extend_from_slice(ncape);
    stacked.extend_from_slice(cape);
    stacked.extend_from_slice(cin);
    stacked.extend_from_slice(lfc);
    stacked.extend_from_slice(el);
    stacked
}

fn unpack_ecape_stack(
    stacked: &[f64],
    nxy: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if stacked.len() != ECAPE_STACK_FIELDS * nxy {
        return None;
    }

    Some((
        stacked[0..nxy].to_vec(),
        stacked[nxy..2 * nxy].to_vec(),
        stacked[2 * nxy..3 * nxy].to_vec(),
        stacked[3 * nxy..4 * nxy].to_vec(),
        stacked[4 * nxy..5 * nxy].to_vec(),
        stacked[5 * nxy..6 * nxy].to_vec(),
    ))
}

fn compute_ecape_fields(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_ecape_opts(opts)?;

    let nxy = f.nx * f.ny;
    if let Some(key) = ecape_cache_key(opts) {
        if let Some(stacked) = f.cached_field(&key) {
            if let Some(fields) = unpack_ecape_stack(&stacked, nxy) {
                return Ok(fields);
            }
        }
    }

    let pressure = f.full_pressure(t)?;
    let temperature = f.temperature(t)?;
    let qvapor = f.qvapor(t)?;
    let height_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut u_earth = vec![0.0f64; u_grid.len()];
    let mut v_earth = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u_earth[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v_earth[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    let mut u10_earth = vec![0.0f64; nxy];
    let mut v10_earth = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10_earth[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10_earth[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let parcel_type = resolve_parcel_type(opts)?;
    let storm_motion_type = resolve_storm_motion_type(opts)?;

    let results: Vec<EcapeSummary> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                build_surface_augmented_ecape_column(
                    &pressure,
                    &temperature,
                    &qvapor,
                    &height_agl,
                    &u_earth,
                    &v_earth,
                    psfc[ij],
                    t2[ij],
                    q2[ij],
                    u10_earth[ij],
                    v10_earth[ij],
                    nz,
                    nxy,
                    ij,
                );

            if pressure_pa.len() < 2 {
                return EcapeSummary::default();
            }

            let parcel_opts = build_parcel_options(opts, ij, parcel_type, storm_motion_type);
            match calc_ecape_parcel(
                &height_m,
                &pressure_pa,
                &temp_k,
                &dewpoint_k,
                &u_ms,
                &v_ms,
                &parcel_opts,
            ) {
                Ok(result) => EcapeSummary {
                    ecape: result.ecape_jkg,
                    ncape: result.ncape_jkg,
                    cape: result.cape_jkg,
                    cin: result.cin_jkg,
                    lfc: result.lfc_m.unwrap_or(0.0),
                    el: result.el_m.unwrap_or(0.0),
                },
                Err(_) => EcapeSummary::default(),
            }
        })
        .collect();

    let mut ecape = Vec::with_capacity(nxy);
    let mut ncape = Vec::with_capacity(nxy);
    let mut cape = Vec::with_capacity(nxy);
    let mut cin = Vec::with_capacity(nxy);
    let mut lfc = Vec::with_capacity(nxy);
    let mut el = Vec::with_capacity(nxy);

    for result in results {
        ecape.push(result.ecape);
        ncape.push(result.ncape);
        cape.push(result.cape);
        cin.push(result.cin);
        lfc.push(result.lfc);
        el.push(result.el);
    }

    if let Some(key) = ecape_cache_key(opts) {
        let stacked = pack_ecape_stack(&ecape, &ncape, &cape, &cin, &lfc, &el);
        f.store_cached_field(key, stacked);
    }

    Ok((ecape, ncape, cape, cin, lfc, el))
}

pub fn compute_ecape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (ecape, _, _, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(ecape)
}

pub fn compute_ncape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, ncape, _, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(ncape)
}

pub fn compute_ecape_cape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, cape, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(cape)
}

pub fn compute_ecape_cin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, cin, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(cin)
}

pub fn compute_ecape_lfc(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, lfc, _) = compute_ecape_fields(f, t, opts)?;
    Ok(lfc)
}

pub fn compute_ecape_el(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, _, el) = compute_ecape_fields(f, t, opts)?;
    Ok(el)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::StormMotion;

    fn base_opts() -> ComputeOpts {
        ComputeOpts::default()
    }

    #[test]
    fn rejects_custom_parcel_thermodynamics() {
        let mut opts = base_opts();
        opts.parcel_pressure = Some(900.0);
        opts.parcel_temperature = Some(20.0);
        opts.parcel_dewpoint = Some(15.0);

        let err = validate_ecape_opts(&opts).unwrap_err();
        assert!(matches!(err, WrfError::InvalidParam(_)));
    }

    #[test]
    fn resolves_supported_parcel_types() {
        let mut opts = base_opts();
        assert!(matches!(
            resolve_parcel_type(&opts).unwrap(),
            CapeType::SurfaceBased
        ));

        opts.parcel_type = Some("ml".into());
        assert!(matches!(
            resolve_parcel_type(&opts).unwrap(),
            CapeType::MixedLayer
        ));

        opts.parcel_type = Some("mu".into());
        assert!(matches!(
            resolve_parcel_type(&opts).unwrap(),
            CapeType::MostUnstable
        ));
    }

    #[test]
    fn explicit_storm_motion_becomes_user_defined() {
        let mut opts = base_opts();
        opts.storm_motion = Some(StormMotion::Uniform { u: 12.0, v: 8.0 });

        let parcel_opts = build_parcel_options(
            &opts,
            0,
            CapeType::SurfaceBased,
            EcapeStormMotionType::RightMoving,
        );

        assert!(matches!(
            parcel_opts.storm_motion_type,
            EcapeStormMotionType::UserDefined
        ));
        assert_eq!(parcel_opts.storm_motion_u_ms, Some(12.0));
        assert_eq!(parcel_opts.storm_motion_v_ms, Some(8.0));
    }

    #[test]
    fn grid_storm_motion_skips_ecape_cache() {
        let mut opts = base_opts();
        opts.storm_motion = Some(StormMotion::Grid {
            u: vec![1.0, 2.0],
            v: vec![3.0, 4.0],
        });

        assert!(ecape_cache_key(&opts).is_none());
    }

    #[test]
    fn ecape_stack_round_trip_preserves_field_order() {
        let stacked = pack_ecape_stack(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
        );

        let unpacked = unpack_ecape_stack(&stacked, 2).unwrap();
        assert_eq!(unpacked.0, vec![1.0, 2.0]);
        assert_eq!(unpacked.1, vec![3.0, 4.0]);
        assert_eq!(unpacked.2, vec![5.0, 6.0]);
        assert_eq!(unpacked.3, vec![7.0, 8.0]);
        assert_eq!(unpacked.4, vec![9.0, 10.0]);
        assert_eq!(unpacked.5, vec![11.0, 12.0]);
    }
}
