//! WRF sounding extraction and validated sounding column types.
//!
//! This crate owns point and box sounding selection. It does not own map
//! rendering or WRF product recipes.

mod native_table;

use std::path::Path;

use ecape_rs::{calc_ecape_parcel, CapeType, ParcelOptions, StormMotionType};
use serde::{Deserialize, Serialize};
use sharprs::profile::StationInfo;
use sharprs::render::{compute_all_params, render_full_sounding, ComputedParams};
use sharprs::Profile as SharprsProfile;
use thiserror::Error;
use wrf_core::{getvar, ComputeOpts, VarOutput, WrfFile};

const MS_TO_KTS: f64 = 1.943_844_492_440_604_6;
const KTS_TO_MS: f64 = 0.514_444_444_444_444_5;
const PRESSURE_MONOTONIC_TOLERANCE_HPA: f64 = 1.0e-6;
const HEIGHT_MONOTONIC_TOLERANCE_M: f64 = 1.0e-6;
const DEWPOINT_TEMPERATURE_TOLERANCE_C: f64 = 0.1;

#[derive(Debug, Error)]
pub enum SoundingError {
    #[error("point i/j ({i}, {j}) is outside grid nx={nx}, ny={ny}")]
    PointOutOfBounds {
        i: usize,
        j: usize,
        nx: usize,
        ny: usize,
    },
    #[error("box selection did not include any grid points")]
    EmptyBoxSelection,
    #[error("field `{field}` needs at least {expected_at_least} values, got {actual}")]
    InvalidLength {
        field: &'static str,
        expected_at_least: usize,
        actual: usize,
    },
    #[error("field `{field}` length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("field `{field}` contains invalid data: {reason}")]
    InvalidValue { field: &'static str, reason: String },
    #[error("field `{field}` expected shape [nz, ny, nx], got {shape:?}")]
    NotThreeDimensional { field: String, shape: Vec<usize> },
    #[error("field `{field}` expected shape [ny, nx], got {shape:?}")]
    NotTwoDimensional { field: String, shape: Vec<usize> },
    #[error(transparent)]
    Wrf(#[from] wrf_core::WrfError),
    #[error(transparent)]
    SharprsProfile(#[from] sharprs::profile::ProfileError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}

pub type SoundingResult<T> = Result<T, SoundingError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SoundingSelection {
    PointLatLon {
        lat: f64,
        lon: f64,
    },
    PointIj {
        i: usize,
        j: usize,
    },
    BoxLatLon {
        south: f64,
        west: f64,
        north: f64,
        east: f64,
        method: BoxSoundingMethod,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoxSoundingMethod {
    MeanProfile,
    MedianProfile,
    MostUnstableColumn,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SoundingMetadata {
    pub station_id: String,
    pub valid_time: String,
    pub latitude_deg: Option<f64>,
    pub longitude_deg: Option<f64>,
    pub elevation_m: Option<f64>,
    pub sample_method: Option<String>,
    pub selected_i: Option<usize>,
    pub selected_j: Option<usize>,
    pub box_point_count: Option<usize>,
    #[serde(default)]
    pub box_radius_lat_deg: Option<f64>,
    #[serde(default)]
    pub box_radius_lon_deg: Option<f64>,
}

impl SoundingMetadata {
    pub fn to_station_info(&self) -> StationInfo {
        StationInfo {
            station_id: self.station_id.clone(),
            latitude: self.latitude_deg.unwrap_or(f64::NAN),
            longitude: self.longitude_deg.unwrap_or(f64::NAN),
            elevation: self.elevation_m.unwrap_or(f64::NAN),
            datetime: self.valid_time.clone(),
        }
    }

    pub fn from_station_info(station: &StationInfo) -> Self {
        Self {
            station_id: station.station_id.clone(),
            valid_time: station.datetime.clone(),
            latitude_deg: finite_or_none(station.latitude),
            longitude_deg: finite_or_none(station.longitude),
            elevation_m: finite_or_none(station.elevation),
            sample_method: None,
            selected_i: None,
            selected_j: None,
            box_point_count: None,
            box_radius_lat_deg: None,
            box_radius_lon_deg: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundingColumn {
    pub pressure_hpa: Vec<f64>,
    pub height_m_msl: Vec<f64>,
    pub temperature_c: Vec<f64>,
    pub dewpoint_c: Vec<f64>,
    pub u_ms: Vec<f64>,
    pub v_ms: Vec<f64>,
    #[serde(default)]
    pub omega_pa_s: Vec<f64>,
    #[serde(default)]
    pub metadata: SoundingMetadata,
}

impl SoundingColumn {
    pub fn len(&self) -> usize {
        self.pressure_hpa.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pressure_hpa.is_empty()
    }

    pub fn validate(&self) -> SoundingResult<()> {
        let expected = self.pressure_hpa.len();
        if expected < 2 {
            return Err(SoundingError::InvalidLength {
                field: "pressure_hpa",
                expected_at_least: 2,
                actual: expected,
            });
        }

        validate_len("height_m_msl", self.height_m_msl.len(), expected)?;
        validate_len("temperature_c", self.temperature_c.len(), expected)?;
        validate_len("dewpoint_c", self.dewpoint_c.len(), expected)?;
        validate_len("u_ms", self.u_ms.len(), expected)?;
        validate_len("v_ms", self.v_ms.len(), expected)?;
        if !self.omega_pa_s.is_empty() {
            validate_len("omega_pa_s", self.omega_pa_s.len(), expected)?;
        }

        validate_finite("pressure_hpa", &self.pressure_hpa)?;
        validate_finite("height_m_msl", &self.height_m_msl)?;
        validate_finite("temperature_c", &self.temperature_c)?;
        validate_finite("dewpoint_c", &self.dewpoint_c)?;
        validate_finite("u_ms", &self.u_ms)?;
        validate_finite("v_ms", &self.v_ms)?;
        if !self.omega_pa_s.is_empty() {
            validate_finite("omega_pa_s", &self.omega_pa_s)?;
        }

        validate_monotonic_non_increasing(
            "pressure_hpa",
            &self.pressure_hpa,
            PRESSURE_MONOTONIC_TOLERANCE_HPA,
        )?;
        validate_monotonic_non_decreasing(
            "height_m_msl",
            &self.height_m_msl,
            HEIGHT_MONOTONIC_TOLERANCE_M,
        )?;
        validate_dewpoint_not_above_temperature(
            &self.temperature_c,
            &self.dewpoint_c,
            DEWPOINT_TEMPERATURE_TOLERANCE_C,
        )?;
        Ok(())
    }

    pub fn to_sharprs_profile(&self) -> SoundingResult<SharprsProfile> {
        self.validate()?;

        let u_kts: Vec<f64> = self.u_ms.iter().map(|value| value * MS_TO_KTS).collect();
        let v_kts: Vec<f64> = self.v_ms.iter().map(|value| value * MS_TO_KTS).collect();

        Ok(SharprsProfile::from_uv(
            &self.pressure_hpa,
            &self.height_m_msl,
            &self.temperature_c,
            &self.dewpoint_c,
            &u_kts,
            &v_kts,
            &self.omega_pa_s,
            self.metadata.to_station_info(),
        )?)
    }

    pub fn from_sharprs_profile(profile: &SharprsProfile) -> Self {
        Self {
            pressure_hpa: profile.pres.clone(),
            height_m_msl: profile.hght.clone(),
            temperature_c: profile.tmpc.clone(),
            dewpoint_c: profile.dwpc.clone(),
            u_ms: profile.u.iter().map(|value| value * KTS_TO_MS).collect(),
            v_ms: profile.v.iter().map(|value| value * KTS_TO_MS).collect(),
            omega_pa_s: if profile.omeg.iter().any(|value| value.is_finite()) {
                profile.omeg.clone()
            } else {
                Vec::new()
            },
            metadata: SoundingMetadata::from_station_info(&profile.station),
        }
    }
}

#[derive(Debug)]
pub struct NativeSounding {
    pub profile: SharprsProfile,
    pub params: ComputedParams,
    pub verified_ecape: VerifiedEcapeParcels,
    pub metadata: SoundingMetadata,
}

impl NativeSounding {
    pub fn from_column(column: &SoundingColumn) -> SoundingResult<Self> {
        let profile = column.to_sharprs_profile()?;
        let params = compute_all_params(&profile);
        let verified_ecape = verified_ecape_params(&profile);
        Ok(Self {
            profile,
            params,
            verified_ecape,
            metadata: column.metadata.clone(),
        })
    }

    pub fn render_full_png(&self) -> Vec<u8> {
        let base = render_full_sounding(&self.profile, &self.params);
        native_table::replace_title_and_table(
            &base,
            &self.profile,
            &self.params,
            &self.verified_ecape,
            &self.metadata,
        )
        .unwrap_or(base)
    }

    pub fn write_full_png<P: AsRef<Path>>(&self, path: P) -> SoundingResult<()> {
        std::fs::write(path, self.render_full_png())?;
        Ok(())
    }
}

pub fn render_full_sounding_png(column: &SoundingColumn) -> SoundingResult<Vec<u8>> {
    Ok(NativeSounding::from_column(column)?.render_full_png())
}

pub fn write_full_sounding_png<P: AsRef<Path>>(
    column: &SoundingColumn,
    path: P,
) -> SoundingResult<()> {
    NativeSounding::from_column(column)?.write_full_png(path)
}

pub fn extract_and_write_sounding_png<P: AsRef<Path>>(
    file: &WrfFile,
    selection: SoundingSelection,
    timeidx: Option<usize>,
    path: P,
) -> SoundingResult<()> {
    let column = extract_sounding(file, selection, timeidx)?;
    write_full_sounding_png(&column, path)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerifiedEcapeParcelParams {
    pub ecape: f64,
    pub ncape: f64,
    pub cape: f64,
    pub cinh: f64,
    pub cape_3km: f64,
    pub cape_6km: f64,
    pub lfc_m: f64,
    pub el_m: f64,
}

impl VerifiedEcapeParcelParams {
    pub const fn missing() -> Self {
        Self {
            ecape: f64::NAN,
            ncape: f64::NAN,
            cape: f64::NAN,
            cinh: f64::NAN,
            cape_3km: f64::NAN,
            cape_6km: f64::NAN,
            lfc_m: f64::NAN,
            el_m: f64::NAN,
        }
    }
}

impl Default for VerifiedEcapeParcelParams {
    fn default() -> Self {
        Self::missing()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerifiedEcapeParcels {
    pub surface_based: VerifiedEcapeParcelParams,
    pub mixed_layer: VerifiedEcapeParcelParams,
    pub most_unstable: VerifiedEcapeParcelParams,
}

impl VerifiedEcapeParcels {
    pub const fn missing() -> Self {
        Self {
            surface_based: VerifiedEcapeParcelParams::missing(),
            mixed_layer: VerifiedEcapeParcelParams::missing(),
            most_unstable: VerifiedEcapeParcelParams::missing(),
        }
    }
}

impl Default for VerifiedEcapeParcels {
    fn default() -> Self {
        Self::missing()
    }
}

struct EcapeProfileInputs {
    pressure_pa: Vec<f64>,
    height_m: Vec<f64>,
    temperature_k: Vec<f64>,
    dewpoint_k: Vec<f64>,
    u_ms: Vec<f64>,
    v_ms: Vec<f64>,
    surface_height_m: f64,
}

fn verified_ecape_params(profile: &SharprsProfile) -> VerifiedEcapeParcels {
    let Some(inputs) = EcapeProfileInputs::from_sharprs(profile) else {
        return VerifiedEcapeParcels::missing();
    };

    VerifiedEcapeParcels {
        surface_based: verified_ecape_parcel(&inputs, CapeType::SurfaceBased),
        mixed_layer: verified_ecape_parcel(&inputs, CapeType::MixedLayer),
        most_unstable: verified_ecape_parcel(&inputs, CapeType::MostUnstable),
    }
}

impl EcapeProfileInputs {
    fn from_sharprs(profile: &SharprsProfile) -> Option<Self> {
        let mut pressure_pa = Vec::with_capacity(profile.num_levels());
        let mut height_m = Vec::with_capacity(profile.num_levels());
        let mut temperature_k = Vec::with_capacity(profile.num_levels());
        let mut dewpoint_k = Vec::with_capacity(profile.num_levels());
        let mut u_ms = Vec::with_capacity(profile.num_levels());
        let mut v_ms = Vec::with_capacity(profile.num_levels());

        for i in 0..profile.num_levels() {
            let p_pa = profile.pres[i] * 100.0;
            let h_m = profile.hght[i];
            let t_k = profile.tmpc[i] + 273.15;
            let td_k = profile.dwpc[i] + 273.15;
            let u = profile.u[i] * KTS_TO_MS;
            let v = profile.v[i] * KTS_TO_MS;
            if p_pa.is_finite()
                && p_pa > 0.0
                && h_m.is_finite()
                && t_k.is_finite()
                && t_k > 0.0
                && td_k.is_finite()
                && td_k > 0.0
                && u.is_finite()
                && v.is_finite()
            {
                pressure_pa.push(p_pa);
                height_m.push(h_m);
                temperature_k.push(t_k);
                dewpoint_k.push(td_k);
                u_ms.push(u);
                v_ms.push(v);
            }
        }

        if pressure_pa.len() < 3 {
            return None;
        }

        Some(Self {
            surface_height_m: *height_m.first()?,
            pressure_pa,
            height_m,
            temperature_k,
            dewpoint_k,
            u_ms,
            v_ms,
        })
    }
}

fn verified_ecape_parcel(
    inputs: &EcapeProfileInputs,
    cape_type: CapeType,
) -> VerifiedEcapeParcelParams {
    let base_options = ParcelOptions {
        cape_type,
        storm_motion_type: StormMotionType::RightMoving,
        pseudoadiabatic: Some(true),
        ..ParcelOptions::default()
    };
    let entraining = calc_ecape_parcel(
        &inputs.height_m,
        &inputs.pressure_pa,
        &inputs.temperature_k,
        &inputs.dewpoint_k,
        &inputs.u_ms,
        &inputs.v_ms,
        &base_options,
    );

    let undiluted_options = ParcelOptions {
        entrainment_rate: Some(0.0),
        ..base_options
    };
    let undiluted = calc_ecape_parcel(
        &inputs.height_m,
        &inputs.pressure_pa,
        &inputs.temperature_k,
        &inputs.dewpoint_k,
        &inputs.u_ms,
        &inputs.v_ms,
        &undiluted_options,
    );

    let (Ok(entraining), Ok(undiluted)) = (entraining, undiluted) else {
        return VerifiedEcapeParcelParams::missing();
    };

    VerifiedEcapeParcelParams {
        ecape: entraining.ecape_jkg,
        ncape: entraining.ncape_jkg,
        cape: undiluted.cape_jkg,
        cinh: undiluted.cin_jkg,
        cape_3km: positive_buoyancy_to_depth(&undiluted, 3000.0),
        cape_6km: positive_buoyancy_to_depth(&undiluted, 6000.0),
        lfc_m: undiluted
            .lfc_m
            .map(|height| height - inputs.surface_height_m)
            .unwrap_or(f64::NAN),
        el_m: undiluted
            .el_m
            .map(|height| height - inputs.surface_height_m)
            .unwrap_or(f64::NAN),
    }
}

fn positive_buoyancy_to_depth(result: &ecape_rs::EcapeParcelResult, depth_m: f64) -> f64 {
    let profile = &result.parcel_profile;
    if profile.height_m.len() < 2 || profile.buoyancy_ms2.len() != profile.height_m.len() {
        return if result.cape_jkg == 0.0 {
            0.0
        } else {
            f64::NAN
        };
    }

    let bottom = profile.height_m[0];
    let top = bottom + depth_m;
    let mut energy = 0.0;
    for i in 0..profile.height_m.len() - 1 {
        let z0 = profile.height_m[i];
        let z1 = profile.height_m[i + 1];
        if !z0.is_finite() || !z1.is_finite() || z1 <= z0 || z0 >= top {
            continue;
        }
        let b0 = profile.buoyancy_ms2[i];
        let b1 = profile.buoyancy_ms2[i + 1];
        if !b0.is_finite() || !b1.is_finite() {
            continue;
        }

        let seg_top = z1.min(top);
        let frac = ((seg_top - z0) / (z1 - z0)).clamp(0.0, 1.0);
        let b_seg_top = b0 + frac * (b1 - b0);
        energy += positive_linear_area(z0, b0, seg_top, b_seg_top);
    }
    energy
}

fn positive_linear_area(z0: f64, b0: f64, z1: f64, b1: f64) -> f64 {
    let dz = z1 - z0;
    if dz <= 0.0 {
        return 0.0;
    }
    if b0 <= 0.0 && b1 <= 0.0 {
        0.0
    } else if b0 >= 0.0 && b1 >= 0.0 {
        0.5 * (b0 + b1) * dz
    } else if b0 < 0.0 {
        let frac = (-b0 / (b1 - b0)).clamp(0.0, 1.0);
        let positive_dz = dz * (1.0 - frac);
        0.5 * b1.max(0.0) * positive_dz
    } else {
        let frac = (b0 / (b0 - b1)).clamp(0.0, 1.0);
        let positive_dz = dz * frac;
        0.5 * b0.max(0.0) * positive_dz
    }
}

pub fn extract_sounding(
    file: &WrfFile,
    selection: SoundingSelection,
    timeidx: Option<usize>,
) -> SoundingResult<SoundingColumn> {
    let t = timeidx.unwrap_or(0);
    match selection {
        SoundingSelection::PointLatLon { lat, lon } => {
            let (i, j) = nearest_grid_point(file, t, lat, lon)?;
            let mut column = extract_point_ij(file, i, j, t)?;
            column.metadata.sample_method = Some("nearest".to_string());
            Ok(column)
        }
        SoundingSelection::PointIj { i, j } => extract_point_ij(file, i, j, t),
        SoundingSelection::BoxLatLon {
            south,
            west,
            north,
            east,
            method,
        } => extract_box_latlon(file, south, west, north, east, method, t),
    }
}

pub fn extract_point_ij(
    file: &WrfFile,
    i: usize,
    j: usize,
    timeidx: usize,
) -> SoundingResult<SoundingColumn> {
    ensure_point(file, i, j)?;
    let indices = vec![(i, j)];
    let mut column = assemble_column(file, &indices, AggregateMethod::Mean, timeidx)?;
    column.metadata.selected_i = Some(i);
    column.metadata.selected_j = Some(j);
    column.metadata.sample_method = Some("point_ij".to_string());
    column.validate()?;
    Ok(column)
}

pub fn extract_box_latlon(
    file: &WrfFile,
    south: f64,
    west: f64,
    north: f64,
    east: f64,
    method: BoxSoundingMethod,
    timeidx: usize,
) -> SoundingResult<SoundingColumn> {
    let indices = box_indices(file, timeidx, south, west, north, east)?;
    let mut column = match method {
        BoxSoundingMethod::MeanProfile => {
            assemble_column(file, &indices, AggregateMethod::Mean, timeidx)?
        }
        BoxSoundingMethod::MedianProfile => {
            assemble_column(file, &indices, AggregateMethod::Median, timeidx)?
        }
        BoxSoundingMethod::MostUnstableColumn => {
            let (i, j) = most_unstable_index(file, &indices, timeidx)?;
            extract_point_ij(file, i, j, timeidx)?
        }
    };
    column.metadata.sample_method = Some(format!("{method:?}"));
    column.metadata.box_point_count = Some(indices.len());
    column.metadata.box_radius_lat_deg = Some((north - south).abs() * 0.5);
    column.metadata.box_radius_lon_deg = Some((east - west).abs() * 0.5);
    column.validate()?;
    Ok(column)
}

fn assemble_column(
    file: &WrfFile,
    indices: &[(usize, usize)],
    method: AggregateMethod,
    timeidx: usize,
) -> SoundingResult<SoundingColumn> {
    let pressure = getvar(file, "pressure", Some(timeidx), &ComputeOpts::default())?;
    let height = getvar(file, "height", Some(timeidx), &ComputeOpts::default())?;
    let temperature = getvar(
        file,
        "temp",
        Some(timeidx),
        &ComputeOpts {
            units: Some("degC".to_string()),
            ..Default::default()
        },
    )?;
    let dewpoint = getvar(
        file,
        "td",
        Some(timeidx),
        &ComputeOpts {
            units: Some("degC".to_string()),
            ..Default::default()
        },
    )?;
    let u = getvar(file, "ua", Some(timeidx), &ComputeOpts::default())?;
    let v = getvar(file, "va", Some(timeidx), &ComputeOpts::default())?;
    let omega = getvar(file, "omega", Some(timeidx), &ComputeOpts::default()).ok();

    let metadata = metadata_for_indices(file, indices, timeidx)?;
    Ok(SoundingColumn {
        pressure_hpa: aggregate_profile(&pressure, file, indices, method, "pressure")?,
        height_m_msl: aggregate_profile(&height, file, indices, method, "height")?,
        temperature_c: aggregate_profile(&temperature, file, indices, method, "temp")?,
        dewpoint_c: aggregate_profile(&dewpoint, file, indices, method, "td")?,
        u_ms: aggregate_profile(&u, file, indices, method, "ua")?,
        v_ms: aggregate_profile(&v, file, indices, method, "va")?,
        omega_pa_s: match omega {
            Some(output) => aggregate_profile(&output, file, indices, method, "omega")?,
            None => Vec::new(),
        },
        metadata,
    })
}

fn aggregate_profile(
    output: &VarOutput,
    file: &WrfFile,
    indices: &[(usize, usize)],
    method: AggregateMethod,
    field: &str,
) -> SoundingResult<Vec<f64>> {
    ensure_3d(output, file, field)?;
    let mut profile = Vec::with_capacity(file.nz);
    for k in 0..file.nz {
        let mut values = Vec::with_capacity(indices.len());
        for &(i, j) in indices {
            values.push(output.data[k * file.ny * file.nx + j * file.nx + i]);
        }
        profile.push(match method {
            AggregateMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AggregateMethod::Median => median(&mut values),
        });
    }
    Ok(profile)
}

fn ensure_3d(output: &VarOutput, file: &WrfFile, field: &str) -> SoundingResult<()> {
    if output.shape != [file.nz, file.ny, file.nx] {
        return Err(SoundingError::NotThreeDimensional {
            field: field.to_string(),
            shape: output.shape.clone(),
        });
    }
    Ok(())
}

fn ensure_2d(output: &VarOutput, file: &WrfFile, field: &str) -> SoundingResult<()> {
    if output.shape != [file.ny, file.nx] {
        return Err(SoundingError::NotTwoDimensional {
            field: field.to_string(),
            shape: output.shape.clone(),
        });
    }
    Ok(())
}

fn nearest_grid_point(
    file: &WrfFile,
    timeidx: usize,
    lat: f64,
    lon: f64,
) -> SoundingResult<(usize, usize)> {
    let lats = file.xlat(timeidx)?;
    let lons = file.xlong(timeidx)?;
    let mut best = None;
    for j in 0..file.ny {
        for i in 0..file.nx {
            let idx = j * file.nx + i;
            let dlat = lats[idx] - lat;
            let dlon = lons[idx] - lon;
            let dist2 = dlat * dlat + dlon * dlon;
            if best.map(|(_, best_dist)| dist2 < best_dist).unwrap_or(true) {
                best = Some(((i, j), dist2));
            }
        }
    }
    Ok(best.expect("non-empty WRF grid").0)
}

fn box_indices(
    file: &WrfFile,
    timeidx: usize,
    south: f64,
    west: f64,
    north: f64,
    east: f64,
) -> SoundingResult<Vec<(usize, usize)>> {
    let lats = file.xlat(timeidx)?;
    let lons = file.xlong(timeidx)?;
    let min_lat = south.min(north);
    let max_lat = south.max(north);
    let crosses_dateline = west > east;
    let mut indices = Vec::new();
    for j in 0..file.ny {
        for i in 0..file.nx {
            let idx = j * file.nx + i;
            let lat = lats[idx];
            let lon = lons[idx];
            let lon_ok = if crosses_dateline {
                lon >= west || lon <= east
            } else {
                lon >= west && lon <= east
            };
            if lat >= min_lat && lat <= max_lat && lon_ok {
                indices.push((i, j));
            }
        }
    }
    if indices.is_empty() {
        return Err(SoundingError::EmptyBoxSelection);
    }
    Ok(indices)
}

fn most_unstable_index(
    file: &WrfFile,
    indices: &[(usize, usize)],
    timeidx: usize,
) -> SoundingResult<(usize, usize)> {
    let mucape = getvar(file, "mucape", Some(timeidx), &ComputeOpts::default())?;
    ensure_2d(&mucape, file, "mucape")?;
    let mut best = None;
    for &(i, j) in indices {
        let value = mucape.data[j * file.nx + i];
        if !value.is_finite() {
            continue;
        }
        if best
            .map(|(_, best_value)| value > best_value)
            .unwrap_or(true)
        {
            best = Some(((i, j), value));
        }
    }
    Ok(best.map(|(ij, _)| ij).unwrap_or(indices[0]))
}

fn metadata_for_indices(
    file: &WrfFile,
    indices: &[(usize, usize)],
    timeidx: usize,
) -> SoundingResult<SoundingMetadata> {
    let lat = file.xlat(timeidx)?;
    let lon = file.xlong(timeidx)?;
    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;
    for &(i, j) in indices {
        let idx = j * file.nx + i;
        lat_sum += lat[idx];
        lon_sum += lon[idx];
    }
    let count = indices.len() as f64;
    let valid_time = file
        .times()
        .ok()
        .and_then(|times| times.get(timeidx).cloned())
        .unwrap_or_else(|| format!("timeidx={timeidx}"));
    Ok(SoundingMetadata {
        station_id: "WRF".to_string(),
        valid_time,
        latitude_deg: Some(lat_sum / count),
        longitude_deg: Some(lon_sum / count),
        elevation_m: None,
        sample_method: None,
        selected_i: None,
        selected_j: None,
        box_point_count: Some(indices.len()),
        box_radius_lat_deg: None,
        box_radius_lon_deg: None,
    })
}

fn ensure_point(file: &WrfFile, i: usize, j: usize) -> SoundingResult<()> {
    if i >= file.nx || j >= file.ny {
        return Err(SoundingError::PointOutOfBounds {
            i,
            j,
            nx: file.nx,
            ny: file.ny,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum AggregateMethod {
    Mean,
    Median,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> SoundingResult<()> {
    if actual != expected {
        return Err(SoundingError::LengthMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, values: &[f64]) -> SoundingResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(SoundingError::InvalidValue {
            field,
            reason: format!("non-finite value at index {index}"),
        });
    }
    Ok(())
}

fn validate_monotonic_non_increasing(
    field: &'static str,
    values: &[f64],
    tolerance: f64,
) -> SoundingResult<()> {
    for (index, pair) in values.windows(2).enumerate() {
        if pair[1] > pair[0] + tolerance {
            return Err(SoundingError::InvalidValue {
                field,
                reason: format!(
                    "value increased from {} to {} between levels {} and {}",
                    pair[0],
                    pair[1],
                    index,
                    index + 1
                ),
            });
        }
    }
    Ok(())
}

fn validate_monotonic_non_decreasing(
    field: &'static str,
    values: &[f64],
    tolerance: f64,
) -> SoundingResult<()> {
    for (index, pair) in values.windows(2).enumerate() {
        if pair[1] + tolerance < pair[0] {
            return Err(SoundingError::InvalidValue {
                field,
                reason: format!(
                    "value decreased from {} to {} between levels {} and {}",
                    pair[0],
                    pair[1],
                    index,
                    index + 1
                ),
            });
        }
    }
    Ok(())
}

fn validate_dewpoint_not_above_temperature(
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    tolerance: f64,
) -> SoundingResult<()> {
    for (index, (&t, &td)) in temperature_c.iter().zip(dewpoint_c.iter()).enumerate() {
        if td > t + tolerance {
            return Err(SoundingError::InvalidValue {
                field: "dewpoint_c",
                reason: format!("dewpoint {td} C exceeds temperature {t} C at level {index}"),
            });
        }
    }
    Ok(())
}

fn finite_or_none(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_column() -> SoundingColumn {
        SoundingColumn {
            pressure_hpa: vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0],
            height_m_msl: vec![150.0, 800.0, 1500.0, 3100.0, 5600.0, 9200.0],
            temperature_c: vec![24.0, 20.0, 16.0, 4.0, -12.0, -38.0],
            dewpoint_c: vec![20.0, 17.0, 12.0, -2.0, -24.0, -48.0],
            u_ms: vec![4.0, 6.0, 9.0, 14.0, 22.0, 34.0],
            v_ms: vec![1.0, 3.0, 6.0, 10.0, 18.0, 26.0],
            omega_pa_s: Vec::new(),
            metadata: SoundingMetadata {
                station_id: "WRF".to_string(),
                valid_time: "1974-04-03T22:00:00Z".to_string(),
                latitude_deg: Some(33.5),
                longitude_deg: Some(-88.5),
                elevation_m: Some(150.0),
                sample_method: Some("test".to_string()),
                selected_i: Some(10),
                selected_j: Some(20),
                box_point_count: Some(1),
                box_radius_lat_deg: None,
                box_radius_lon_deg: None,
            },
        }
    }

    #[test]
    fn sounding_validation_rejects_bad_lengths() {
        let column = SoundingColumn {
            pressure_hpa: vec![1000.0, 900.0],
            height_m_msl: vec![0.0],
            temperature_c: vec![20.0, 10.0],
            dewpoint_c: vec![15.0, 5.0],
            u_ms: vec![1.0, 2.0],
            v_ms: vec![1.0, 2.0],
            omega_pa_s: Vec::new(),
            metadata: SoundingMetadata::default(),
        };
        assert!(matches!(
            column.validate(),
            Err(SoundingError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn median_handles_even_counts() {
        let mut values = vec![4.0, 1.0, 2.0, 3.0];
        assert_eq!(median(&mut values), 2.5);
    }

    #[test]
    fn converts_column_into_sharprs_profile() {
        let column = sample_column();
        let profile = column.to_sharprs_profile().unwrap();
        assert_eq!(profile.num_levels(), column.len());
        assert_eq!(profile.station.station_id, "WRF");
        assert!((profile.u[0] - column.u_ms[0] * MS_TO_KTS).abs() < 1.0e-9);
    }

    #[test]
    fn roundtrip_from_sharprs_profile_preserves_metadata_and_levels() {
        let column = sample_column();
        let profile = column.to_sharprs_profile().unwrap();
        let roundtrip = SoundingColumn::from_sharprs_profile(&profile);
        assert_eq!(roundtrip.len(), column.len());
        assert_eq!(roundtrip.metadata.station_id, column.metadata.station_id);
        assert_eq!(roundtrip.metadata.valid_time, column.metadata.valid_time);
        assert!((roundtrip.u_ms[0] - column.u_ms[0]).abs() < 1.0e-9);
    }

    #[test]
    fn renders_full_sounding_png_bytes() {
        let bytes = render_full_sounding_png(&sample_column()).unwrap();
        assert!(bytes.len() > 1000);
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }
}
