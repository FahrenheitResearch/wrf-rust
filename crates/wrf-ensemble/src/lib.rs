//! WRF ensemble reductions.
//!
//! This crate stays above `wrf-core`: each member is opened as a normal WRF
//! file, products are computed through `wrf-products`, then same-grid fields
//! are reduced across members.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use wrf_core::WrfFile;
use wrf_products::{build_product_request, ProductError, WrfProduct};
use wrf_render::{
    render_image_with_style, save_rgba_png_profile_with_options, Color, ColorScale,
    DiscreteColorScale, ExtendMode, Field2D, MapRenderRequest, PngWriteOptions, ProductKey,
    RustwxRenderError, OPERATIONAL_FAST,
};

const GRID_TOLERANCE_DEG: f32 = 1.0e-4;

#[derive(Debug, Error)]
pub enum EnsembleError {
    #[error("ensemble has no members")]
    EmptyEnsemble,
    #[error("manifest member `{member}` is missing path `{path}`")]
    MissingMember { member: String, path: String },
    #[error("glob pattern `{pattern}` matched no WRF files")]
    EmptyGlob { pattern: String },
    #[error("failed to read glob pattern `{pattern}`: {source}")]
    GlobPattern {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },
    #[error("failed to read glob entry from `{pattern}`: {source}")]
    GlobEntry {
        pattern: String,
        #[source]
        source: glob::GlobError,
    },
    #[error("member `{member}` grid does not match the first ensemble member")]
    GridMismatch { member: String },
    #[error(
        "member `{member}` valid time `{valid_time}` does not match first member `{expected}`"
    )]
    ValidTimeMismatch {
        member: String,
        expected: String,
        valid_time: String,
    },
    #[error("stat `{stat}` requires a value")]
    MissingStatValue { stat: &'static str },
    #[error("percentile must be between 0 and 100, got {0}")]
    InvalidPercentile(f64),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Wrf(#[from] wrf_core::WrfError),
    #[error(transparent)]
    Product(#[from] ProductError),
    #[error(transparent)]
    Render(#[from] RustwxRenderError),
}

pub type EnsembleResult<T> = Result<T, EnsembleError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleMember {
    pub id: String,
    pub path: PathBuf,
}

impl EnsembleMember {
    pub fn new<S: Into<String>, P: Into<PathBuf>>(id: S, path: P) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
        }
    }

    pub fn from_path<P: Into<PathBuf>>(path: P) -> Self {
        let path = path.into();
        let id = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .or_else(|| path.file_name().and_then(|name| name.to_str()))
            .unwrap_or("member")
            .to_string();
        Self { id, path }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleManifest {
    pub members: Vec<ManifestMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ManifestMember {
    Path(PathBuf),
    Object {
        id: Option<String>,
        path: PathBuf,
        valid_time: Option<String>,
    },
}

impl ManifestMember {
    fn into_member(self, manifest_dir: &Path) -> EnsembleMember {
        match self {
            Self::Path(path) => {
                EnsembleMember::from_path(resolve_manifest_path(manifest_dir, path))
            }
            Self::Object { id, path, .. } => {
                let path = resolve_manifest_path(manifest_dir, path);
                match id {
                    Some(id) => EnsembleMember::new(id, path),
                    None => EnsembleMember::from_path(path),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrfEnsemble {
    members: Vec<EnsembleMember>,
}

impl WrfEnsemble {
    pub fn new(members: Vec<EnsembleMember>) -> EnsembleResult<Self> {
        if members.is_empty() {
            return Err(EnsembleError::EmptyEnsemble);
        }
        Ok(Self { members })
    }

    pub fn from_paths<I, P>(paths: I) -> EnsembleResult<Self>
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let members = paths
            .into_iter()
            .map(EnsembleMember::from_path)
            .collect::<Vec<_>>();
        Self::new(members)
    }

    pub fn from_glob(pattern: &str) -> EnsembleResult<Self> {
        let mut paths = Vec::new();
        let entries = glob::glob(pattern).map_err(|source| EnsembleError::GlobPattern {
            pattern: pattern.to_string(),
            source,
        })?;
        for entry in entries {
            paths.push(entry.map_err(|source| EnsembleError::GlobEntry {
                pattern: pattern.to_string(),
                source,
            })?);
        }
        paths.sort();
        if paths.is_empty() {
            return Err(EnsembleError::EmptyGlob {
                pattern: pattern.to_string(),
            });
        }
        Self::from_paths(paths)
    }

    pub fn from_manifest<P: AsRef<Path>>(path: P) -> EnsembleResult<Self> {
        let path = path.as_ref();
        let manifest: EnsembleManifest = serde_json::from_slice(&std::fs::read(path)?)?;
        let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let members = manifest
            .members
            .into_iter()
            .map(|member| member.into_member(manifest_dir))
            .collect::<Vec<_>>();

        for member in &members {
            if !member.path.exists() {
                return Err(EnsembleError::MissingMember {
                    member: member.id.clone(),
                    path: member.path.display().to_string(),
                });
            }
        }

        Self::new(members)
    }

    pub fn members(&self) -> &[EnsembleMember] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn build_product_request(
        &self,
        product: WrfProduct,
        stat: EnsembleStat,
        timeidx: Option<usize>,
    ) -> EnsembleResult<MapRenderRequest> {
        let mut member_fields = Vec::with_capacity(self.members.len());
        let mut template = None;
        let mut expected_valid_time: Option<String> = None;

        for member in &self.members {
            let file = WrfFile::open(&member.path)?;
            let valid_time = valid_time_for_member(&file, timeidx)?;
            if let (Some(expected), Some(valid_time)) = (&expected_valid_time, &valid_time) {
                if expected != valid_time {
                    return Err(EnsembleError::ValidTimeMismatch {
                        member: member.id.clone(),
                        expected: expected.clone(),
                        valid_time: valid_time.clone(),
                    });
                }
            } else if expected_valid_time.is_none() {
                expected_valid_time = valid_time;
            }
            let request = build_product_request(&file, product, timeidx)?;
            if let Some(first) = member_fields.first() {
                validate_same_grid(first, &request.field, &member.id)?;
            }
            member_fields.push(request.field.clone());
            template.get_or_insert(request);
        }

        let mut request = template.ok_or(EnsembleError::EmptyEnsemble)?;
        let reduced = reduce_fields(product, &member_fields, stat)?;

        request.field = reduced;
        request.scale = scale_for_stat(&request.scale, stat);
        request.title = Some(format!(
            "{} Ensemble {}",
            product.recipe().title_template,
            stat.title_label()
        ));
        request.subtitle_center = Some(format!(
            "{} members | probability denominator: finite members",
            self.members.len()
        ));
        request.subtitle_right = expected_valid_time
            .map(|valid| format!("valid {valid} | source: wrf ensemble"))
            .or_else(|| Some("source: wrf ensemble".to_string()));

        // First pass only reduces the filled scalar product. Keep overlays off
        // until contour/vector fields are reduced explicitly too.
        request.contours.clear();
        request.wind_barbs.clear();
        Ok(request)
    }

    pub fn render_product_png<P: AsRef<Path>>(
        &self,
        product: WrfProduct,
        stat: EnsembleStat,
        timeidx: Option<usize>,
        path: P,
    ) -> EnsembleResult<()> {
        let request = self.build_product_request(product, stat, timeidx)?;
        let image = render_image_with_style(&request, OPERATIONAL_FAST)?;
        let path = path.as_ref();
        save_rgba_png_profile_with_options(&image, path, &PngWriteOptions::default())?;
        write_ensemble_sidecar(path, &ensemble_sidecar(self, product, stat, &request))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleRenderSidecar {
    pub package_name: String,
    pub package_version: String,
    pub product_id: String,
    pub stat: String,
    pub units: String,
    pub members: Vec<EnsembleMember>,
    pub member_count: usize,
    pub valid_time_alignment: String,
    pub same_grid_validation: String,
    pub probability_denominator: String,
    pub provenance: String,
}

fn ensemble_sidecar(
    ensemble: &WrfEnsemble,
    product: WrfProduct,
    stat: EnsembleStat,
    request: &MapRenderRequest,
) -> EnsembleRenderSidecar {
    EnsembleRenderSidecar {
        package_name: "wrf-ensemble".to_string(),
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        product_id: product.canonical_name().to_string(),
        stat: stat.slug(),
        units: request.field.units.clone(),
        members: ensemble.members.clone(),
        member_count: ensemble.members.len(),
        valid_time_alignment: "strict_same_timeidx_valid_time".to_string(),
        same_grid_validation: "strict_lat_lon_grid_match".to_string(),
        probability_denominator: "finite_members_per_grid_cell".to_string(),
        provenance: "wrf-ensemble manifest/glob -> wrf-products product -> product-first reduction"
            .to_string(),
    }
}

fn write_ensemble_sidecar(path: &Path, sidecar: &EnsembleRenderSidecar) -> EnsembleResult<()> {
    let sidecar_path = path.with_extension("json");
    let data = serde_json::to_vec_pretty(sidecar)?;
    std::fs::write(sidecar_path, data)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleStat {
    Mean,
    StdDev,
    Min,
    Max,
    Percentile(f64),
    ProbabilityAbove(f64),
    ProbabilityAtOrAbove(f64),
    ProbabilityBelow(f64),
    ProbabilityAtOrBelow(f64),
}

impl EnsembleStat {
    pub fn slug(self) -> String {
        match self {
            Self::Mean => "mean".to_string(),
            Self::StdDev => "stddev".to_string(),
            Self::Min => "min".to_string(),
            Self::Max => "max".to_string(),
            Self::Percentile(p) => format!("p{}", format_value(p)),
            Self::ProbabilityAbove(v) => format!("prob_gt_{}", format_value(v)),
            Self::ProbabilityAtOrAbove(v) => format!("prob_ge_{}", format_value(v)),
            Self::ProbabilityBelow(v) => format!("prob_lt_{}", format_value(v)),
            Self::ProbabilityAtOrBelow(v) => format!("prob_le_{}", format_value(v)),
        }
    }

    pub fn title_label(self) -> String {
        match self {
            Self::Mean => "Mean".to_string(),
            Self::StdDev => "Spread".to_string(),
            Self::Min => "Min".to_string(),
            Self::Max => "Max".to_string(),
            Self::Percentile(p) => format!("P{}", format_tickish(p)),
            Self::ProbabilityAbove(v) => format!("Prob > {}", format_tickish(v)),
            Self::ProbabilityAtOrAbove(v) => format!("Prob >= {}", format_tickish(v)),
            Self::ProbabilityBelow(v) => format!("Prob < {}", format_tickish(v)),
            Self::ProbabilityAtOrBelow(v) => format!("Prob <= {}", format_tickish(v)),
        }
    }

    fn needs_value(name: &str) -> bool {
        matches!(
            normalize_stat_name(name).as_str(),
            "p" | "percentile"
                | "prob_gt"
                | "gt"
                | "prob_ge"
                | "ge"
                | "prob_lt"
                | "lt"
                | "prob_le"
                | "le"
        )
    }
}

pub fn parse_ensemble_stat(name: &str, value: Option<f64>) -> EnsembleResult<EnsembleStat> {
    let normalized = normalize_stat_name(name);
    let require_value = |stat| value.ok_or(EnsembleError::MissingStatValue { stat });
    match normalized.as_str() {
        "mean" | "avg" | "average" => Ok(EnsembleStat::Mean),
        "stddev" | "std" | "spread" => Ok(EnsembleStat::StdDev),
        "min" | "minimum" => Ok(EnsembleStat::Min),
        "max" | "maximum" => Ok(EnsembleStat::Max),
        "p" | "percentile" => {
            let p = require_value("percentile")?;
            if !(0.0..=100.0).contains(&p) {
                return Err(EnsembleError::InvalidPercentile(p));
            }
            Ok(EnsembleStat::Percentile(p))
        }
        "prob_gt" | "gt" => Ok(EnsembleStat::ProbabilityAbove(require_value("prob_gt")?)),
        "prob_ge" | "ge" => Ok(EnsembleStat::ProbabilityAtOrAbove(require_value(
            "prob_ge",
        )?)),
        "prob_lt" | "lt" => Ok(EnsembleStat::ProbabilityBelow(require_value("prob_lt")?)),
        "prob_le" | "le" => Ok(EnsembleStat::ProbabilityAtOrBelow(require_value(
            "prob_le",
        )?)),
        _ => Err(EnsembleError::MissingStatValue {
            stat: "known stat name",
        }),
    }
}

pub fn stat_requires_value(name: &str) -> bool {
    EnsembleStat::needs_value(name)
}

fn reduce_fields(
    product: WrfProduct,
    fields: &[Field2D],
    stat: EnsembleStat,
) -> EnsembleResult<Field2D> {
    let first = fields.first().ok_or(EnsembleError::EmptyEnsemble)?;
    let values = reduce_values(fields, stat)?;
    let units = if is_probability(stat) {
        "%"
    } else {
        first.units.as_str()
    };
    Ok(Field2D::new(
        ProductKey::named(format!(
            "ensemble_{}_{}",
            product.canonical_name(),
            stat.slug()
        )),
        units,
        first.grid.clone(),
        values,
    )?)
}

fn reduce_values(fields: &[Field2D], stat: EnsembleStat) -> EnsembleResult<Vec<f32>> {
    let len = fields
        .first()
        .ok_or(EnsembleError::EmptyEnsemble)?
        .values
        .len();
    let mut out = Vec::with_capacity(len);
    let mut scratch = Vec::with_capacity(fields.len());

    for idx in 0..len {
        scratch.clear();
        for field in fields {
            let value = field.values[idx];
            if value.is_finite() {
                scratch.push(value as f64);
            }
        }
        out.push(reduce_column(&mut scratch, stat) as f32);
    }

    Ok(out)
}

fn reduce_column(values: &mut [f64], stat: EnsembleStat) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    match stat {
        EnsembleStat::Mean => values.iter().sum::<f64>() / values.len() as f64,
        EnsembleStat::StdDev => {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            variance.sqrt()
        }
        EnsembleStat::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        EnsembleStat::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        EnsembleStat::Percentile(p) => percentile(values, p),
        EnsembleStat::ProbabilityAbove(threshold) => probability(values, |value| value > threshold),
        EnsembleStat::ProbabilityAtOrAbove(threshold) => {
            probability(values, |value| value >= threshold)
        }
        EnsembleStat::ProbabilityBelow(threshold) => probability(values, |value| value < threshold),
        EnsembleStat::ProbabilityAtOrBelow(threshold) => {
            probability(values, |value| value <= threshold)
        }
    }
}

fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.len() == 1 {
        return values[0];
    }
    let rank = (percentile / 100.0).clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let weight = rank - lo as f64;
        values[lo] + (values[hi] - values[lo]) * weight
    }
}

fn probability(values: &[f64], predicate: impl Fn(f64) -> bool) -> f64 {
    let hits = values.iter().filter(|value| predicate(**value)).count();
    100.0 * hits as f64 / values.len() as f64
}

fn validate_same_grid(first: &Field2D, next: &Field2D, member: &str) -> EnsembleResult<()> {
    if first.grid.shape != next.grid.shape
        || first.grid.lat_deg.len() != next.grid.lat_deg.len()
        || first.grid.lon_deg.len() != next.grid.lon_deg.len()
    {
        return Err(EnsembleError::GridMismatch {
            member: member.to_string(),
        });
    }

    let same_lats = first
        .grid
        .lat_deg
        .iter()
        .zip(&next.grid.lat_deg)
        .all(|(a, b)| (*a - *b).abs() <= GRID_TOLERANCE_DEG);
    let same_lons = first
        .grid
        .lon_deg
        .iter()
        .zip(&next.grid.lon_deg)
        .all(|(a, b)| (*a - *b).abs() <= GRID_TOLERANCE_DEG);
    if same_lats && same_lons {
        Ok(())
    } else {
        Err(EnsembleError::GridMismatch {
            member: member.to_string(),
        })
    }
}

fn scale_for_stat(base: &ColorScale, stat: EnsembleStat) -> ColorScale {
    if is_probability(stat) {
        return ColorScale::Discrete(probability_scale());
    }
    base.clone()
}

fn probability_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: (0..=10).map(|idx| idx as f64 * 10.0).collect(),
        colors: vec![
            Color::rgba(247, 247, 247, 255),
            Color::rgba(222, 235, 247, 255),
            Color::rgba(198, 219, 239, 255),
            Color::rgba(158, 202, 225, 255),
            Color::rgba(107, 174, 214, 255),
            Color::rgba(66, 146, 198, 255),
            Color::rgba(254, 217, 118, 255),
            Color::rgba(254, 178, 76, 255),
            Color::rgba(253, 141, 60, 255),
            Color::rgba(227, 26, 28, 255),
        ],
        extend: ExtendMode::Neither,
        mask_below: None,
    }
}

fn is_probability(stat: EnsembleStat) -> bool {
    matches!(
        stat,
        EnsembleStat::ProbabilityAbove(_)
            | EnsembleStat::ProbabilityAtOrAbove(_)
            | EnsembleStat::ProbabilityBelow(_)
            | EnsembleStat::ProbabilityAtOrBelow(_)
    )
}

fn valid_time_for_member(file: &WrfFile, timeidx: Option<usize>) -> EnsembleResult<Option<String>> {
    let t = timeidx.unwrap_or(0);
    let times = file.times()?;
    Ok(times.get(t).map(|value| value.trim().to_string()))
}

fn resolve_manifest_path(manifest_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    }
}

fn normalize_stat_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn format_value(value: f64) -> String {
    format_tickish(value).replace('.', "p").replace('-', "m")
}

fn format_tickish(value: f64) -> String {
    if (value - value.round()).abs() < 1.0e-9 {
        format!("{}", value.round() as i64)
    } else {
        let text = format!("{value:.3}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wrf_render::{GridShape, LatLonGrid};

    fn sample_field(values: Vec<f32>) -> Field2D {
        Field2D::new(
            ProductKey::named("sample"),
            "units",
            LatLonGrid::new(
                GridShape::new(2, 2).unwrap(),
                vec![30.0, 30.0, 31.0, 31.0],
                vec![-100.0, -99.0, -100.0, -99.0],
            )
            .unwrap(),
            values,
        )
        .unwrap()
    }

    #[test]
    fn mean_reduces_each_grid_cell() {
        let fields = vec![
            sample_field(vec![1.0, 2.0, 3.0, 4.0]),
            sample_field(vec![3.0, 4.0, 5.0, 6.0]),
        ];

        let reduced = reduce_values(&fields, EnsembleStat::Mean).unwrap();

        assert_eq!(reduced, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn probability_ignores_nan_members() {
        let fields = vec![
            sample_field(vec![1.0, f32::NAN, 3.0, 4.0]),
            sample_field(vec![2.0, 4.0, 1.0, 6.0]),
        ];

        let reduced = reduce_values(&fields, EnsembleStat::ProbabilityAtOrAbove(2.0)).unwrap();

        assert_eq!(reduced, vec![50.0, 100.0, 50.0, 100.0]);
    }

    #[test]
    fn percentile_interpolates_between_sorted_members() {
        let fields = vec![
            sample_field(vec![0.0, 0.0, 0.0, 0.0]),
            sample_field(vec![10.0, 10.0, 10.0, 10.0]),
            sample_field(vec![20.0, 20.0, 20.0, 20.0]),
        ];

        let reduced = reduce_values(&fields, EnsembleStat::Percentile(25.0)).unwrap();

        assert_eq!(reduced, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn grid_mismatch_is_rejected() {
        let first = sample_field(vec![1.0, 2.0, 3.0, 4.0]);
        let mut second = sample_field(vec![1.0, 2.0, 3.0, 4.0]);
        second.grid.lon_deg[2] += 1.0;

        let err = validate_same_grid(&first, &second, "m02").unwrap_err();

        assert!(matches!(err, EnsembleError::GridMismatch { .. }));
    }

    #[test]
    fn parse_probability_stat_requires_threshold() {
        assert!(parse_ensemble_stat("prob_ge", None).is_err());
        assert_eq!(
            parse_ensemble_stat("prob_ge", Some(1.0)).unwrap(),
            EnsembleStat::ProbabilityAtOrAbove(1.0)
        );
    }

    #[test]
    fn manifest_paths_are_resolved_relative_to_manifest() {
        let resolved = resolve_manifest_path(
            Path::new("case_root"),
            PathBuf::from("members/m01/wrfout_d01"),
        );

        assert_eq!(
            resolved,
            PathBuf::from("case_root").join("members/m01/wrfout_d01")
        );
    }

    #[test]
    fn manifest_missing_members_fail_before_rendering() {
        let dir =
            std::env::temp_dir().join(format!("wrf_ensemble_manifest_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("ensemble.json");
        std::fs::write(
            &manifest,
            r#"{"members":[{"id":"m01","path":"missing_wrfout"}]}"#,
        )
        .unwrap();

        let err = WrfEnsemble::from_manifest(&manifest).unwrap_err();
        assert!(matches!(err, EnsembleError::MissingMember { .. }));

        let _ = std::fs::remove_file(manifest);
        let _ = std::fs::remove_dir(dir);
    }
}
