//! WRF sounding extraction and validated sounding column types.
//!
//! This crate owns point and box sounding selection. It does not own map
//! rendering or WRF product recipes.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use wrf_core::{getvar, ComputeOpts, VarOutput, WrfFile};

const PRESSURE_MONOTONIC_TOLERANCE_HPA: f64 = 1.0e-6;
const HEIGHT_MONOTONIC_TOLERANCE_M: f64 = 1.0e-6;
const DEWPOINT_TEMPERATURE_TOLERANCE_C: f64 = 1.0e-6;

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
            extract_point_ij(file, i, j, t)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
