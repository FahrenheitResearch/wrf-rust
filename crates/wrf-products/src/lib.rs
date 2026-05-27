//! WRF product recipes.
//!
//! This crate is glue. It maps product names to `wrf-core::getvar` calls and
//! `wrf-render` requests, but it does not duplicate diagnostic science.

use std::fs;
use std::path::{Path, PathBuf};

use rustwx_core::{
    ProductKeyMetadata, ProductLineage, ProductMaturity, ProductProvenance, ProductSemanticFlag,
    ProductWindowSpec, StatisticalProcess,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wrf_core::{
    getvar, met::composite::interp_to_height_level, ComputeOpts, VarOutput, WrfFile, WrfProjection,
};
use wrf_render::{
    build_projected_map_with_options,
    map_frame_aspect_ratio_for_mode_with_domain_frame_style_and_colorbar_orientation,
    render_image_with_style, save_rgba_png_profile_with_options, srh_scale_levels,
    stp_scale_levels, BasemapDetail, Color, ColorScale, ColorbarOrientation, ContourStyle,
    DiscreteColorScale, DomainFrame, DomainFrameSource, ExtendMode, Field2D, GeographicBounds,
    GridShape, LatLonGrid, LegendControls, LegendMode, LevelDensity, MapRenderRequest,
    OverlayLegend, OverlayLegendItem, PngWriteOptions, ProductKey, ProductVisualMode,
    ProjectedMapBuildOptions, ProjectionSpec, RasterSampleMode, RenderDensity, RgbaGridField,
    RustwxRenderError, StaticPlotStyle, WeatherPalette, WindBarbStyle, OPERATIONAL_FAST,
};

const DEFAULT_PRODUCT_WIDTH: u32 = 1600;
const DEFAULT_PRODUCT_HEIGHT: u32 = 1200;
const UH_TRACK_THRESHOLD: f32 = 50.0;
const UH_TRACK_BINS: [f32; 4] = [50.0, 100.0, 200.0, 300.0];
const UH_TRACK_FILL_ALPHA: u8 = 58;
const UH_TRACK_FILL_COLORS: [Color; 4] = [
    Color::rgba(255, 241, 118, UH_TRACK_FILL_ALPHA),
    Color::rgba(255, 169, 77, 82),
    Color::rgba(239, 83, 80, 104),
    Color::rgba(171, 71, 188, 132),
];
const SURFACE_BARB_SPACING_PX: f64 = 58.0;
const UPPER_AIR_BARB_SPACING_PX: f64 = 64.0;
const OPERATIONAL_BARB_GRID_STRIDE: usize = 1;
const OPERATIONAL_BARB_HALO_WIDTH_PX: u32 = 2;
const OPERATIONAL_BARB_WIDTH_PX: u32 = 1;
const OPERATIONAL_BARB_LENGTH_PX: f64 = 20.0;
const EARTH_ROTATED_U10_VAR: &str = "uvmet10_u";
const EARTH_ROTATED_V10_VAR: &str = "uvmet10_v";
const NATIVE_UPDRAFT_HELICITY_MAX_VAR: &str = "UP_HELI_MAX";
const NATIVE_OR_COMPUTED_UH_VAR: &str = "native_or_computed_uhel";

#[derive(Debug, Error)]
pub enum ProductError {
    #[error("unknown WRF product `{0}`")]
    UnknownProduct(String),
    #[error("product `{product}` expected a 2-D field, got shape {shape:?}")]
    NotTwoDimensional { product: String, shape: Vec<usize> },
    #[error("product `{product}` expected a 3-D field, got shape {shape:?}")]
    NotThreeDimensional { product: String, shape: Vec<usize> },
    #[error("pressure level {level_hpa} hPa produced no valid samples for `{product}`")]
    EmptyPressureLevel { product: String, level_hpa: f64 },
    #[error("projected map build failed: {0}")]
    Projection(String),
    #[error("history directory `{path}` could not be read: {message}")]
    HistoryDirRead { path: PathBuf, message: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Wrf(#[from] wrf_core::WrfError),
    #[error(transparent)]
    Render(#[from] RustwxRenderError),
}

pub type ProductResult<T> = Result<T, ProductError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductPalette {
    Cape,
    SurfaceBasedCape,
    MixedLayerCape,
    MostUnstableCape,
    Ecape,
    ThreeCape,
    DeepLayerCape,
    EffectiveCape,
    Ncape,
    Cin,
    Srh,
    Stp,
    EffectiveStp,
    FixedLayerStp,
    Scp,
    Ehi,
    Tehi,
    Tts,
    Vtp,
    Uh,
    Dcape,
    Ship,
    Dcp,
    Wndg,
    Esp,
    Mmp,
    CriticalAngle,
    LapseRate,
    Vorticity,
    PotentialVorticity,
    Omega,
    Reflectivity,
    SimIr,
    Temperature,
    UpperAirTemperature,
    SurfaceTemperature,
    Dewpoint,
    UpperAirDewpoint,
    SurfaceDewpoint,
    WetBulbPotentialTemperature,
    RelativeHumidity,
    SurfaceRelativeHumidity,
    UpperAirRelativeHumidity,
    SevereMoisture,
    CloudFraction,
    WindSpeed,
    LayerMeanWind,
    UpperAirWind,
    JetSpeed,
    SurfaceWind,
    BulkShear,
    WindComponent,
    Precipitation,
    AccumulatedPrecipitation,
    Pwat,
    Terrain,
    LclLfcHeight,
    EquilibriumLevel,
    LclTemperature,
    FreezingLevel,
    PblHeight,
    ConvectiveIndex,
    TotalTotals,
    MeanMixr,
    FireWeather,
    Haines,
    Hdw,
    RichardsonNumber,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskPolicy {
    None,
    Below(f64),
}

impl MaskPolicy {
    fn threshold(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Below(value) => Some(value),
        }
    }
}

impl ProductPalette {
    pub fn default_levels(self) -> Vec<f32> {
        match self {
            Self::Cape | Self::SurfaceBasedCape | Self::MixedLayerCape | Self::MostUnstableCape => {
                range_step(0.0, 8100.0, 100.0)
            }
            Self::Ecape => range_i32(0, 5000, 50),
            Self::ThreeCape => three_cape_levels(),
            Self::DeepLayerCape | Self::EffectiveCape => six_cape_levels(),
            Self::Ncape => range_i32(0, 2000, 50),
            Self::Cin => range_i32(-300, 0, 25),
            Self::Srh => srh_scale_levels()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            Self::Stp | Self::EffectiveStp | Self::FixedLayerStp => stp_scale_levels()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            Self::Scp => scp_levels(),
            Self::Ehi => range_step(0.0, 24.0, 0.2),
            Self::Tehi | Self::Tts | Self::Vtp => stp_scale_levels()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            Self::Uh => range_step(0.0, 400.0, 5.0),
            Self::Dcape => range_i32(0, 2500, 100),
            Self::Ship => vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            Self::Dcp | Self::Wndg => range_i32(0, 10, 1),
            Self::Esp => range_i32(0, 20, 1),
            Self::Mmp => (0..=20).map(|value| value as f32 * 0.05).collect(),
            Self::CriticalAngle => range_i32(0, 180, 15),
            Self::LapseRate => range_step(2.0, 10.0, 0.1),
            Self::Vorticity => range_step(0.0, 0.0005, 0.000025),
            Self::PotentialVorticity => range_i32(-2, 10, 1),
            Self::Omega => vec![
                -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
            ],
            Self::Reflectivity => range_step(5.0, 75.0, 5.0),
            Self::SimIr => range_step(-90.0, 50.0, 1.0),
            Self::Temperature => range_step(-40.0, 50.0, 5.0),
            Self::UpperAirTemperature => range_i32(-80, 40, 1),
            Self::SurfaceTemperature => range_step(-60.0, 120.0, 1.0),
            Self::Dewpoint | Self::SurfaceDewpoint => range_step(-40.0, 90.0, 1.0),
            Self::UpperAirDewpoint => range_i32(-45, 30, 1),
            Self::WetBulbPotentialTemperature => range_i32(-10, 45, 1),
            Self::RelativeHumidity => range_step(0.0, 100.0, 10.0),
            Self::SurfaceRelativeHumidity => range_i32(0, 100, 5),
            Self::UpperAirRelativeHumidity => range_i32(0, 100, 5),
            Self::SevereMoisture => range_i32(0, 100, 5),
            Self::CloudFraction => range_i32(0, 100, 5),
            Self::WindSpeed => range_step(0.0, 120.0, 10.0),
            Self::LayerMeanWind => range_i32(0, 120, 5),
            Self::UpperAirWind => range_i32(0, 160, 5),
            Self::JetSpeed => range_i32(40, 200, 5),
            Self::SurfaceWind => wind_10m_levels(),
            Self::BulkShear => bulk_shear_levels(),
            Self::WindComponent => wind_component_levels(),
            Self::Precipitation | Self::AccumulatedPrecipitation => precip_accum_levels(),
            Self::Pwat => pwat_inches_levels(),
            Self::Terrain => range_i32(0, 4000, 250),
            Self::LclLfcHeight => range_i32(0, 8000, 500),
            Self::EquilibriumLevel => range_i32(4000, 18000, 1000),
            Self::LclTemperature => range_i32(-30, 30, 2),
            Self::FreezingLevel => range_i32(0, 6000, 250),
            Self::PblHeight => range_i32(0, 5000, 250),
            Self::ConvectiveIndex => range_i32(0, 50, 2),
            Self::TotalTotals => range_i32(30, 70, 2),
            Self::MeanMixr => range_i32(0, 25, 1),
            Self::FireWeather => range_i32(0, 100, 5),
            Self::Haines => vec![2.0, 3.0, 4.0, 5.0, 6.0],
            Self::Hdw => range_i32(0, 1000, 50),
            Self::RichardsonNumber => range_i32(0, 100, 5),
        }
    }

    pub fn scale(self, levels: Vec<f32>, extend: ExtendMode) -> ColorScale {
        self.scale_with_policy(levels, extend, self.mask_policy())
    }

    fn scale_with_policy(
        self,
        levels: Vec<f32>,
        extend: ExtendMode,
        mask_policy: MaskPolicy,
    ) -> ColorScale {
        let levels = levels.into_iter().map(|value| value as f64).collect();
        let discrete = DiscreteColorScale {
            levels,
            colors: self.colors(),
            extend,
            mask_below: mask_policy.threshold(),
        };
        ColorScale::Discrete(discrete)
    }

    fn default_extend(self) -> ExtendMode {
        match self {
            Self::Cin => ExtendMode::Min,
            Self::Vorticity
            | Self::PotentialVorticity
            | Self::Omega
            | Self::Temperature
            | Self::UpperAirTemperature
            | Self::SurfaceTemperature
            | Self::Dewpoint
            | Self::UpperAirDewpoint
            | Self::SurfaceDewpoint
            | Self::LclTemperature
            | Self::WetBulbPotentialTemperature
            | Self::WindComponent
            | Self::SimIr => ExtendMode::Both,
            _ => ExtendMode::Max,
        }
    }

    fn mask_policy(self) -> MaskPolicy {
        match self {
            Self::Cape
            | Self::SurfaceBasedCape
            | Self::MixedLayerCape
            | Self::MostUnstableCape
            | Self::Ecape
            | Self::ThreeCape
            | Self::DeepLayerCape
            | Self::EffectiveCape
            | Self::Ncape => MaskPolicy::Below(1.0),
            Self::Srh
            | Self::Stp
            | Self::EffectiveStp
            | Self::FixedLayerStp
            | Self::Scp
            | Self::Ehi
            | Self::Tehi
            | Self::Tts
            | Self::Vtp
            | Self::Uh
            | Self::Dcape
            | Self::Ship
            | Self::Dcp
            | Self::Wndg
            | Self::Esp
            | Self::Mmp => MaskPolicy::Below(0.01),
            Self::Reflectivity => MaskPolicy::Below(5.0),
            Self::SurfaceWind => MaskPolicy::Below(10.0),
            Self::Precipitation | Self::AccumulatedPrecipitation => MaskPolicy::Below(0.01),
            _ => MaskPolicy::None,
        }
    }

    fn colors(self) -> Vec<Color> {
        match self {
            Self::Cape => wrf_render::weather::weather_palette(WeatherPalette::Cape),
            Self::SurfaceBasedCape | Self::MixedLayerCape | Self::MostUnstableCape => {
                wrf_render::weather::weather_palette(WeatherPalette::Cape)
            }
            Self::Ecape => wrf_render::weather::weather_palette(WeatherPalette::Ecape),
            Self::ThreeCape => wrf_render::weather::weather_palette(WeatherPalette::ThreeCape),
            Self::DeepLayerCape | Self::EffectiveCape => {
                wrf_render::weather::weather_palette(WeatherPalette::Cape)
            }
            Self::Ncape => wrf_render::weather::weather_palette(WeatherPalette::Ncape),
            Self::Srh => wrf_render::weather::weather_palette(WeatherPalette::Srh),
            Self::Stp => wrf_render::weather::weather_palette(WeatherPalette::Stp),
            Self::EffectiveStp | Self::FixedLayerStp => {
                wrf_render::weather::weather_palette(WeatherPalette::Stp)
            }
            Self::Ehi => wrf_render::weather::weather_palette(WeatherPalette::Ehi),
            Self::Uh => wrf_render::weather::weather_palette(WeatherPalette::Uh),
            Self::LapseRate => wrf_render::weather::weather_palette(WeatherPalette::LapseRate),
            Self::Vorticity => wrf_render::weather::weather_palette(WeatherPalette::RelVort),
            Self::Reflectivity => {
                wrf_render::weather::weather_palette(WeatherPalette::Reflectivity)
            }
            Self::SimIr => wrf_render::weather::weather_palette(WeatherPalette::SimIr),
            Self::Temperature => wrf_render::weather::weather_palette(WeatherPalette::Temperature),
            Self::UpperAirTemperature => colors_from_hex(&[
                "#2d004b", "#54278f", "#756bb1", "#9e9ac8", "#dadaeb", "#f7f7f7", "#fee8c8",
                "#fdbb84", "#e34a33", "#b30000",
            ]),
            Self::SurfaceTemperature => surface_temperature_colors(),
            Self::Dewpoint => wrf_render::weather::weather_palette(WeatherPalette::Dewpoint),
            Self::UpperAirDewpoint => colors_from_hex(&[
                "#7f2704", "#a63603", "#d94801", "#fdae6b", "#fee6ce", "#f7f7f7", "#c7e9c0",
                "#74c476", "#238b45", "#08519c",
            ]),
            Self::SurfaceDewpoint => surface_dewpoint_colors(),
            Self::WetBulbPotentialTemperature => wet_bulb_potential_temperature_colors(),
            Self::RelativeHumidity => wrf_render::weather::weather_palette(WeatherPalette::Rh),
            Self::SurfaceRelativeHumidity => colors_from_hex(&[
                "#8c2d04", "#cc4c02", "#ec7014", "#fe9929", "#fec44f", "#fff7bc", "#e0f3db",
                "#a8ddb5", "#43a2ca", "#0868ac",
            ]),
            Self::UpperAirRelativeHumidity => colors_from_hex(&[
                "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#f7f7f7", "#c7eae5", "#80cdc1",
                "#35978f", "#01665e", "#003c30",
            ]),
            Self::SevereMoisture => colors_from_hex(&[
                "#7f2704", "#a63603", "#d94801", "#f16913", "#fdae6b", "#fdd0a2", "#deebf7",
                "#9ecae1", "#4292c6", "#08519c",
            ]),
            Self::WindSpeed => wrf_render::weather::weather_palette(WeatherPalette::Winds),
            Self::LayerMeanWind
            | Self::UpperAirWind
            | Self::JetSpeed
            | Self::SurfaceWind
            | Self::BulkShear => wrf_render::weather::weather_palette(WeatherPalette::Winds),
            Self::Precipitation => wrf_render::weather::weather_palette(WeatherPalette::Precip),
            Self::AccumulatedPrecipitation => {
                wrf_render::weather::weather_palette(WeatherPalette::Precip)
            }
            Self::WindComponent => wind_component_colors(),
            Self::Cin => wrf_render::weather::weather_palette(WeatherPalette::Cin),
            Self::Scp => wrf_render::weather::weather_palette(WeatherPalette::Scp),
            Self::Tehi
            | Self::Tts
            | Self::Vtp
            | Self::Ship
            | Self::Dcp
            | Self::Wndg
            | Self::Esp
            | Self::Mmp => wrf_render::weather::weather_palette(WeatherPalette::Stp),
            Self::Dcape => wrf_render::weather::weather_palette(WeatherPalette::Cape),
            Self::CriticalAngle => colors_from_hex(&[
                "#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c", "#7f0000",
            ]),
            Self::PotentialVorticity => colors_from_hex(&[
                "#313695", "#74add1", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43",
                "#d73027", "#a50026",
            ]),
            Self::Omega => colors_from_hex(&[
                "#762a83", "#9970ab", "#c2a5cf", "#e7d4e8", "#f7f7f7", "#d9f0d3", "#a6dba0",
                "#5aae61", "#1b7837",
            ]),
            Self::CloudFraction => colors_from_hex(&[
                "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5",
                "#08519c",
            ]),
            Self::Pwat => colors_from_hex(&[
                "#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494",
                "#54278f", "#9e0142",
            ]),
            Self::Terrain => colors_from_hex(&[
                "#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c", "#bdb76b", "#8c6d31",
                "#6b4f2a", "#f7f7f7",
            ]),
            Self::LclLfcHeight => colors_from_hex(&[
                "#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02",
                "#8c2d04",
            ]),
            Self::EquilibriumLevel => colors_from_hex(&[
                "#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d",
                "#810f7c", "#4d004b",
            ]),
            Self::LclTemperature => colors_from_hex(&[
                "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#f7f7f7", "#fee090",
                "#fdae61", "#f46d43", "#d73027", "#a50026",
            ]),
            Self::FreezingLevel => colors_from_hex(&[
                "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c",
                "#08306b",
            ]),
            Self::PblHeight => colors_from_hex(&[
                "#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#31a354", "#006837",
                "#004529",
            ]),
            Self::ConvectiveIndex => colors_from_hex(&[
                "#f7f7f7", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#fee08b", "#fdae61",
                "#f46d43", "#d73027",
            ]),
            Self::TotalTotals => colors_from_hex(&[
                "#f7f7f7", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#fee08b", "#fdae61",
                "#f46d43", "#d73027", "#a50026",
            ]),
            Self::MeanMixr => colors_from_hex(&[
                "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45",
                "#005a32", "#003c22",
            ]),
            Self::FireWeather => colors_from_hex(&[
                "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c",
                "#bd0026", "#800026",
            ]),
            Self::Haines => {
                colors_from_hex(&["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c"])
            }
            Self::Hdw => colors_from_hex(&[
                "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c",
                "#bd0026", "#800026", "#4d0018",
            ]),
            Self::RichardsonNumber => colors_from_hex(&[
                "#f7f7f7", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#fee08b", "#fdae61",
                "#f46d43", "#d73027",
            ]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrfProduct {
    Ecape,
    SbEcape,
    MlEcape,
    MuEcape,
    Ncape,
    EcapeCape,
    EcapeCin,
    EcapeLfc,
    EcapeEl,
    EcapeScp,
    EcapeEhi,
    Sbcape,
    Sbcin,
    Mlcape,
    Mlcin,
    Mucape,
    Mucin,
    Sb3Cape,
    Ml3Cape,
    Mu3Cape,
    Sb6Cape,
    Ml6Cape,
    Mu6Cape,
    EffectiveCape,
    EffectiveInflowBase,
    EffectiveInflowTop,
    Srh01,
    Srh03,
    EffectiveSrh,
    Shear01,
    StpEffective,
    StpFixed,
    Scp,
    Ehi,
    Tehi,
    Tts,
    VtpMod,
    CriticalAngle,
    Ship,
    Bri,
    Dcape,
    Dcp,
    Wndg,
    Esp,
    Mmp,
    Shear06,
    Ebwd,
    MeanWind06,
    Reflectivity,
    Reflectivity1km,
    ReflectivityUh,
    CloudTopTemp,
    CloudFracLow,
    CloudFracMid,
    CloudFracHigh,
    SlpWind10m,
    SurfaceWind10m,
    U10Component,
    V10Component,
    T2,
    Td2,
    Rh2,
    Pwat,
    PrecipAccum,
    UpdraftHelicity,
    Pblh,
    Terrain,
    Lcl,
    Lfc,
    El,
    LclTemp,
    KIndex,
    TotalTotals,
    MeanMixr,
    LowRh,
    MidRh,
    DgzRh,
    ConvTemp,
    MaxTemp,
    LapseRate700500,
    LapseRate03,
    FreezingLevel,
    WetBulbZero,
    Fosberg,
    Haines,
    Hdw,
    Height200Wind,
    Temp200Wind,
    Wind200,
    Height250Wind,
    Temp250Wind,
    Wind250,
    Height300Wind,
    Temp300Wind,
    Wind300,
    Height500Wind,
    Temp500Wind,
    Wind500,
    Vort500Wind,
    Pvo500,
    Omega500,
    ThetaW850,
    Temp700Wind,
    Height700Wind,
    Rh700Wind,
    Omega700Wind,
    Height850Wind,
    Temp850Wind,
    Td850Wind,
    Wind850,
}

impl WrfProduct {
    pub fn from_name(name: &str) -> Option<Self> {
        match normalize(name).as_str() {
            "ecape" | "entraining_cape" => Some(Self::Ecape),
            "sb_ecape" | "sbecape" | "surface_based_ecape" => Some(Self::SbEcape),
            "ml_ecape" | "mlecape" | "mixed_layer_ecape" => Some(Self::MlEcape),
            "mu_ecape" | "muecape" | "most_unstable_ecape" => Some(Self::MuEcape),
            "ncape" | "normalized_cape" => Some(Self::Ncape),
            "ecape_cape" | "entraining_parcel_cape" => Some(Self::EcapeCape),
            "ecape_cin" | "ecin" | "entraining_parcel_cin" => Some(Self::EcapeCin),
            "ecape_lfc" | "entraining_parcel_lfc" => Some(Self::EcapeLfc),
            "ecape_el" | "entraining_parcel_el" => Some(Self::EcapeEl),
            "ecape_scp" | "entraining_scp" | "experimental_ecape_scp" => Some(Self::EcapeScp),
            "ecape_ehi" | "entraining_ehi" | "experimental_ecape_ehi" => Some(Self::EcapeEhi),
            "sbcape" | "surface_based_cape" => Some(Self::Sbcape),
            "sbcin" | "surface_based_cin" => Some(Self::Sbcin),
            "mlcape" | "mixed_layer_cape" => Some(Self::Mlcape),
            "mlcin" | "mixed_layer_cin" => Some(Self::Mlcin),
            "mucape" | "most_unstable_cape" => Some(Self::Mucape),
            "mucin" | "most_unstable_cin" => Some(Self::Mucin),
            "sb3cape" | "surface_based_3cape" | "surface_based_0_3km_cape" => Some(Self::Sb3Cape),
            "ml3cape" | "3cape" | "mixed_layer_3cape" | "mixed_layer_0_3km_cape" => {
                Some(Self::Ml3Cape)
            }
            "mu3cape" | "most_unstable_3cape" | "most_unstable_0_3km_cape" => Some(Self::Mu3Cape),
            "sb6cape" | "surface_based_6cape" | "surface_based_0_6km_cape" => Some(Self::Sb6Cape),
            "ml6cape" | "6cape" | "mixed_layer_6cape" | "mixed_layer_0_6km_cape" => {
                Some(Self::Ml6Cape)
            }
            "mu6cape" | "most_unstable_6cape" | "most_unstable_0_6km_cape" => Some(Self::Mu6Cape),
            "effective_cape" | "eff_cape" | "effective_inflow_cape" => Some(Self::EffectiveCape),
            "effective_inflow_base" | "eff_inflow_base" | "effective_layer_base" => {
                Some(Self::EffectiveInflowBase)
            }
            "effective_inflow_top" | "eff_inflow_top" | "effective_layer_top" => {
                Some(Self::EffectiveInflowTop)
            }
            "srh1" | "srh01" | "srh_0_1km" | "srh01km" => Some(Self::Srh01),
            "srh3" | "srh03" | "srh_0_3km" | "srh03km" => Some(Self::Srh03),
            "effective_srh" | "eff_srh" | "srh_eff" => Some(Self::EffectiveSrh),
            "shear01" | "shear_0_1km" | "bulk_shear_0_1km" | "shr01" => Some(Self::Shear01),
            "stp" | "stp_effective" => Some(Self::StpEffective),
            "stp_fixed" => Some(Self::StpFixed),
            "scp" | "supercell_composite" | "supercell_composite_parameter" => Some(Self::Scp),
            "ehi" | "energy_helicity_index" => Some(Self::Ehi),
            "tehi" | "tornadic_ehi" | "tornadic_0_1km_ehi" => Some(Self::Tehi),
            "tts" | "tornadic_tilting_stretching" => Some(Self::Tts),
            "vtp" | "vtp_mod" | "modified_violent_tornado_parameter" => Some(Self::VtpMod),
            "critical_angle" | "crit_angle" => Some(Self::CriticalAngle),
            "ship" | "significant_hail_parameter" => Some(Self::Ship),
            "bri" | "bulk_richardson_number" => Some(Self::Bri),
            "dcape" | "downdraft_cape" => Some(Self::Dcape),
            "dcp" | "derecho_composite_parameter" => Some(Self::Dcp),
            "wndg" | "wind_damage_parameter" => Some(Self::Wndg),
            "esp" | "enhanced_stretching_potential" => Some(Self::Esp),
            "mmp" | "mcs_maintenance_probability" => Some(Self::Mmp),
            "shear06" | "shear_0_6km" | "bulk_shear_0_6km" => Some(Self::Shear06),
            "ebwd" | "effective_bulk_shear" | "effective_bulk_wind_difference" => Some(Self::Ebwd),
            "mean_wind06" | "mean_wind_0_6km" | "mean_wind_6km" => Some(Self::MeanWind06),
            "dbz" | "maxdbz" | "reflectivity" => Some(Self::Reflectivity),
            "dbz_1km"
            | "reflectivity_1km"
            | "1km_reflectivity"
            | "1km_agl_reflectivity"
            | "reflectivity_1km_agl" => Some(Self::Reflectivity1km),
            "reflectivity_uh"
            | "uh_reflectivity"
            | "refl_uh"
            | "dbz_uh"
            | "reflectivity_updraft_helicity"
            | "reflectivity_uh_combo" => Some(Self::ReflectivityUh),
            "ctt" | "cloud_top_temperature" => Some(Self::CloudTopTemp),
            "cloudfrac_low" | "low_cloud_fraction" | "low_cloudfrac" => Some(Self::CloudFracLow),
            "cloudfrac_mid" | "mid_cloud_fraction" | "mid_cloudfrac" => Some(Self::CloudFracMid),
            "cloudfrac_high" | "high_cloud_fraction" | "high_cloudfrac" => {
                Some(Self::CloudFracHigh)
            }
            "slp_wind10m" | "slp_wind" | "mslp_wind10m" => Some(Self::SlpWind10m),
            "surface_wind" | "surface_wind10m" | "wspd10" | "wind10m" => Some(Self::SurfaceWind10m),
            "u10" | "u10_component" | "u_10m" | "u_wind10m" | "u_component_10m" => {
                Some(Self::U10Component)
            }
            "v10" | "v10_component" | "v_10m" | "v_wind10m" | "v_component_10m" => {
                Some(Self::V10Component)
            }
            "t2" | "temperature_2m" | "2m_temperature" => Some(Self::T2),
            "td2" | "dp2m" | "dewpoint_2m" | "2m_dewpoint" => Some(Self::Td2),
            "rh2" | "rh2m" | "relative_humidity_2m" | "2m_rh" => Some(Self::Rh2),
            "pw" | "pwat" | "precipitable_water" => Some(Self::Pwat),
            "precip" | "qpf" | "rainnc" | "precip_accum" => Some(Self::PrecipAccum),
            "uhel" | "uh" | "updraft_helicity" => Some(Self::UpdraftHelicity),
            "pblh" | "pbl_height" => Some(Self::Pblh),
            "terrain" | "hgt" => Some(Self::Terrain),
            "lcl" | "lcl_height" => Some(Self::Lcl),
            "lfc" | "lfc_height" => Some(Self::Lfc),
            "el" | "el_height" | "equilibrium_level" => Some(Self::El),
            "lcl_temp" | "lcl_temperature" => Some(Self::LclTemp),
            "k_index" | "kindex" => Some(Self::KIndex),
            "total_totals" | "t_totals" | "tot_tots" => Some(Self::TotalTotals),
            "mean_mixr" | "mean_mixing_ratio" => Some(Self::MeanMixr),
            "low_rh" | "low_level_rh" | "mean_low_rh" => Some(Self::LowRh),
            "mid_rh" | "mid_level_rh" | "mean_mid_rh" => Some(Self::MidRh),
            "dgz_rh" | "dendritic_growth_zone_rh" => Some(Self::DgzRh),
            "convective_temp" | "conv_t" | "convt" => Some(Self::ConvTemp),
            "max_temp" | "forecast_max_temp" | "maxt" => Some(Self::MaxTemp),
            "lapse_rate_700_500" | "lr75" | "700_500_lapse_rate" => Some(Self::LapseRate700500),
            "lapse_rate_0_3km" | "lr03" | "0_3km_lapse_rate" => Some(Self::LapseRate03),
            "freezing_level" | "fzlev" => Some(Self::FreezingLevel),
            "wet_bulb_0" | "wet_bulb_zero" | "wb0" => Some(Self::WetBulbZero),
            "fosberg" | "fwi" => Some(Self::Fosberg),
            "haines" | "haines_index" => Some(Self::Haines),
            "hdw" | "hot_dry_windy" => Some(Self::Hdw),
            "height200" | "200mb_height" | "200_height_wind" | "height200_wind" => {
                Some(Self::Height200Wind)
            }
            "temp200" | "200mb_temp" | "200_temp_wind" | "temp200_wind" => Some(Self::Temp200Wind),
            "wind200" | "200mb_wind" | "200_wind" => Some(Self::Wind200),
            "height250" | "250mb_height" | "250_height_wind" | "height250_wind" => {
                Some(Self::Height250Wind)
            }
            "temp250" | "250mb_temp" | "250_temp_wind" | "temp250_wind" => Some(Self::Temp250Wind),
            "wind250" | "250mb_wind" | "250_wind" | "jet250" | "250mb_jet" | "250_jet" => {
                Some(Self::Wind250)
            }
            "height300" | "300mb_height" | "300_height_wind" | "height300_wind" => {
                Some(Self::Height300Wind)
            }
            "temp300" | "300mb_temp" | "300_temp_wind" | "temp300_wind" => Some(Self::Temp300Wind),
            "wind300" | "300mb_wind" | "300_wind" | "jet300" | "300mb_jet" | "300_jet" => {
                Some(Self::Wind300)
            }
            "height500" | "500mb_height" | "500_height_wind" | "height500_wind" => {
                Some(Self::Height500Wind)
            }
            "temp500" | "500mb_temp" | "500_temp_wind" | "temp500_wind" => Some(Self::Temp500Wind),
            "wind500" | "500mb_wind" | "500_wind" => Some(Self::Wind500),
            "vort500" | "vort500_wind" | "500mb_vort" | "500mb_vorticity" | "vorticity500_wind" => {
                Some(Self::Vort500Wind)
            }
            "pvo500" | "pv500" | "500mb_pv" | "500mb_potential_vorticity" => Some(Self::Pvo500),
            "omega500" | "500mb_omega" | "vertical_velocity500" => Some(Self::Omega500),
            "theta_w850"
            | "theta_w_850mb"
            | "850mb_theta_w"
            | "850mb_wet_bulb_potential_temperature" => Some(Self::ThetaW850),
            "temp700" | "700mb_temp" | "700_temp_wind" | "temp700_wind" => Some(Self::Temp700Wind),
            "height700" | "700mb_height" | "700_height_wind" | "height700_wind" => {
                Some(Self::Height700Wind)
            }
            "rh700" | "700mb_rh" | "700_relative_humidity" | "rh700_wind" => Some(Self::Rh700Wind),
            "omega700" | "700mb_omega" | "vertical_velocity700" | "omega700_wind" => {
                Some(Self::Omega700Wind)
            }
            "height850" | "850mb_height" | "850_height_wind" | "height850_wind" => {
                Some(Self::Height850Wind)
            }
            "temp850" | "850mb_temp" | "850_temp_wind" | "temp850_wind" => Some(Self::Temp850Wind),
            "td850" | "dewpoint850" | "850mb_dewpoint" | "850mb_td" | "td850_wind"
            | "dewpoint850_wind" => Some(Self::Td850Wind),
            "wind850" | "850mb_wind" | "850_wind" => Some(Self::Wind850),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Ecape => "ecape",
            Self::SbEcape => "sb_ecape",
            Self::MlEcape => "ml_ecape",
            Self::MuEcape => "mu_ecape",
            Self::Ncape => "ncape",
            Self::EcapeCape => "ecape_cape",
            Self::EcapeCin => "ecape_cin",
            Self::EcapeLfc => "ecape_lfc",
            Self::EcapeEl => "ecape_el",
            Self::EcapeScp => "ecape_scp",
            Self::EcapeEhi => "ecape_ehi",
            Self::Sbcape => "sbcape",
            Self::Sbcin => "sbcin",
            Self::Mlcape => "mlcape",
            Self::Mlcin => "mlcin",
            Self::Mucape => "mucape",
            Self::Mucin => "mucin",
            Self::Sb3Cape => "sb3cape",
            Self::Ml3Cape => "ml3cape",
            Self::Mu3Cape => "mu3cape",
            Self::Sb6Cape => "sb6cape",
            Self::Ml6Cape => "ml6cape",
            Self::Mu6Cape => "mu6cape",
            Self::EffectiveCape => "effective_cape",
            Self::EffectiveInflowBase => "effective_inflow_base",
            Self::EffectiveInflowTop => "effective_inflow_top",
            Self::Srh01 => "srh01",
            Self::Srh03 => "srh03",
            Self::EffectiveSrh => "effective_srh",
            Self::Shear01 => "shear01",
            Self::StpEffective => "stp_effective",
            Self::StpFixed => "stp_fixed",
            Self::Scp => "scp",
            Self::Ehi => "ehi",
            Self::Tehi => "tehi",
            Self::Tts => "tts",
            Self::VtpMod => "vtp_mod",
            Self::CriticalAngle => "critical_angle",
            Self::Ship => "ship",
            Self::Bri => "bri",
            Self::Dcape => "dcape",
            Self::Dcp => "dcp",
            Self::Wndg => "wndg",
            Self::Esp => "esp",
            Self::Mmp => "mmp",
            Self::Shear06 => "shear06",
            Self::Ebwd => "ebwd",
            Self::MeanWind06 => "mean_wind06",
            Self::Reflectivity => "reflectivity",
            Self::Reflectivity1km => "reflectivity_1km",
            Self::ReflectivityUh => "reflectivity_uh",
            Self::CloudTopTemp => "cloud_top_temperature",
            Self::CloudFracLow => "cloudfrac_low",
            Self::CloudFracMid => "cloudfrac_mid",
            Self::CloudFracHigh => "cloudfrac_high",
            Self::SlpWind10m => "slp_wind10m",
            Self::SurfaceWind10m => "surface_wind10m",
            Self::U10Component => "u10_component",
            Self::V10Component => "v10_component",
            Self::T2 => "t2",
            Self::Td2 => "td2",
            Self::Rh2 => "rh2",
            Self::Pwat => "pwat",
            Self::PrecipAccum => "precip_accum",
            Self::UpdraftHelicity => "updraft_helicity",
            Self::Pblh => "pblh",
            Self::Terrain => "terrain",
            Self::Lcl => "lcl",
            Self::Lfc => "lfc",
            Self::El => "el",
            Self::LclTemp => "lcl_temp",
            Self::KIndex => "k_index",
            Self::TotalTotals => "total_totals",
            Self::MeanMixr => "mean_mixr",
            Self::LowRh => "low_rh",
            Self::MidRh => "mid_rh",
            Self::DgzRh => "dgz_rh",
            Self::ConvTemp => "convective_temp",
            Self::MaxTemp => "max_temp",
            Self::LapseRate700500 => "lapse_rate_700_500",
            Self::LapseRate03 => "lapse_rate_0_3km",
            Self::FreezingLevel => "freezing_level",
            Self::WetBulbZero => "wet_bulb_zero",
            Self::Fosberg => "fosberg",
            Self::Haines => "haines",
            Self::Hdw => "hdw",
            Self::Height200Wind => "height200_wind",
            Self::Temp200Wind => "temp200_wind",
            Self::Wind200 => "wind200",
            Self::Height250Wind => "height250_wind",
            Self::Temp250Wind => "temp250_wind",
            Self::Wind250 => "wind250",
            Self::Height300Wind => "height300_wind",
            Self::Temp300Wind => "temp300_wind",
            Self::Wind300 => "wind300",
            Self::Height500Wind => "height500_wind",
            Self::Temp500Wind => "temp500_wind",
            Self::Wind500 => "wind500",
            Self::Vort500Wind => "vort500_wind",
            Self::Pvo500 => "pvo500",
            Self::Omega500 => "omega500",
            Self::ThetaW850 => "theta_w850",
            Self::Temp700Wind => "temp700_wind",
            Self::Height700Wind => "height700_wind",
            Self::Rh700Wind => "rh700_wind",
            Self::Omega700Wind => "omega700_wind",
            Self::Height850Wind => "height850_wind",
            Self::Temp850Wind => "temp850_wind",
            Self::Td850Wind => "td850_wind",
            Self::Wind850 => "wind850",
        }
    }

    pub fn visual_mode(self) -> ProductVisualMode {
        match self {
            Self::Height200Wind
            | Self::Temp200Wind
            | Self::Wind200
            | Self::Height250Wind
            | Self::Temp250Wind
            | Self::Wind250
            | Self::Height300Wind
            | Self::Temp300Wind
            | Self::Wind300
            | Self::Height500Wind
            | Self::Temp500Wind
            | Self::Wind500
            | Self::Vort500Wind
            | Self::Pvo500
            | Self::Omega500
            | Self::ThetaW850
            | Self::Temp700Wind
            | Self::Height700Wind
            | Self::Rh700Wind
            | Self::Omega700Wind
            | Self::Height850Wind
            | Self::Temp850Wind
            | Self::Td850Wind
            | Self::Wind850 => ProductVisualMode::UpperAirAnalysis,
            Self::Ecape
            | Self::SbEcape
            | Self::MlEcape
            | Self::MuEcape
            | Self::Ncape
            | Self::EcapeCape
            | Self::EcapeCin
            | Self::EcapeLfc
            | Self::EcapeEl
            | Self::EcapeScp
            | Self::EcapeEhi
            | Self::Sbcape
            | Self::Sbcin
            | Self::Mlcape
            | Self::Mlcin
            | Self::Mucape
            | Self::Mucin
            | Self::Sb3Cape
            | Self::Ml3Cape
            | Self::Mu3Cape
            | Self::Sb6Cape
            | Self::Ml6Cape
            | Self::Mu6Cape
            | Self::EffectiveCape
            | Self::EffectiveInflowBase
            | Self::EffectiveInflowTop
            | Self::Srh01
            | Self::Srh03
            | Self::EffectiveSrh
            | Self::Shear01
            | Self::Shear06
            | Self::Ebwd
            | Self::StpEffective
            | Self::StpFixed
            | Self::Scp
            | Self::Ehi
            | Self::Tehi
            | Self::Tts
            | Self::VtpMod
            | Self::CriticalAngle
            | Self::Ship
            | Self::Bri
            | Self::Dcape
            | Self::Dcp
            | Self::Wndg
            | Self::Esp
            | Self::Mmp
            | Self::UpdraftHelicity
            | Self::ReflectivityUh
            | Self::Lcl
            | Self::Lfc
            | Self::El
            | Self::LclTemp
            | Self::KIndex
            | Self::TotalTotals
            | Self::MeanMixr
            | Self::LowRh
            | Self::MidRh
            | Self::DgzRh
            | Self::ConvTemp
            | Self::MaxTemp
            | Self::LapseRate700500
            | Self::LapseRate03 => ProductVisualMode::SevereDiagnostic,
            _ => ProductVisualMode::FilledMeteorology,
        }
    }

    pub fn recipe(self) -> ProductRecipe {
        match self {
            Self::Ecape => ecape_parcel_recipe("sb", "Surface-Based ECAPE (SB)"),
            Self::SbEcape => ecape_parcel_recipe("sb", "Surface-Based ECAPE (SB)"),
            Self::MlEcape => ecape_parcel_recipe("ml", "Mixed-Layer ECAPE (ML)"),
            Self::MuEcape => ecape_parcel_recipe("mu", "Most-Unstable ECAPE (MU)"),
            Self::Ncape => ecape_component_recipe(
                "ncape",
                "J/kg",
                ProductPalette::Ncape,
                ProductPalette::Ncape.default_levels(),
                "Normalized CAPE (NCAPE, SB)",
            ),
            Self::EcapeCape => ecape_component_recipe(
                "ecape_cape",
                "J/kg",
                ProductPalette::Ecape,
                ProductPalette::Ecape.default_levels(),
                "ECAPE Parcel CAPE (SB)",
            ),
            Self::EcapeCin => ecape_component_recipe(
                "ecape_cin",
                "J/kg",
                ProductPalette::Cin,
                ProductPalette::Cin.default_levels(),
                "ECAPE CIN (SB)",
            ),
            Self::EcapeLfc => ecape_component_recipe(
                "ecape_lfc",
                "m",
                ProductPalette::LclLfcHeight,
                range_i32(0, 8000, 500),
                "ECAPE LFC Height (SB)",
            ),
            Self::EcapeEl => ecape_component_recipe(
                "ecape_el",
                "m",
                ProductPalette::EquilibriumLevel,
                range_i32(4000, 18000, 1000),
                "ECAPE Equilibrium Level Height (SB)",
            ),
            Self::EcapeScp => severe_recipe_with_levels(
                "ecape_scp",
                "",
                ProductPalette::Scp,
                scp_levels(),
                "ECAPE Supercell Composite Parameter (MU)",
            ),
            Self::EcapeEhi => severe_recipe(
                "ecape_ehi",
                "",
                ProductPalette::Ehi,
                "ECAPE Energy Helicity Index (SB)",
            ),
            Self::Sbcape => {
                severe_recipe("sbcape", "J/kg", ProductPalette::SurfaceBasedCape, "SBCAPE")
            }
            Self::Sbcin => cin_recipe("sbcin", "SBCIN"),
            Self::Mlcape => {
                severe_recipe("mlcape", "J/kg", ProductPalette::MixedLayerCape, "MLCAPE")
            }
            Self::Mlcin => cin_recipe("mlcin", "MLCIN"),
            Self::Mucape => {
                severe_recipe("mucape", "J/kg", ProductPalette::MostUnstableCape, "MUCAPE")
            }
            Self::Mucin => cin_recipe("mucin", "MUCIN"),
            Self::Sb3Cape => severe_recipe_with_levels(
                "sb3cape",
                "J/kg",
                ProductPalette::ThreeCape,
                ProductPalette::ThreeCape.default_levels(),
                "SB 0-3 km CAPE",
            ),
            Self::Ml3Cape => severe_recipe_with_levels(
                "ml3cape",
                "J/kg",
                ProductPalette::ThreeCape,
                ProductPalette::ThreeCape.default_levels(),
                "ML 0-3 km CAPE",
            ),
            Self::Mu3Cape => severe_recipe_with_levels(
                "mu3cape",
                "J/kg",
                ProductPalette::ThreeCape,
                ProductPalette::ThreeCape.default_levels(),
                "MU 0-3 km CAPE",
            ),
            Self::Sb6Cape => severe_recipe_with_levels(
                "sb6cape",
                "J/kg",
                ProductPalette::DeepLayerCape,
                ProductPalette::DeepLayerCape.default_levels(),
                "SB 0-6 km CAPE",
            ),
            Self::Ml6Cape => severe_recipe_with_levels(
                "ml6cape",
                "J/kg",
                ProductPalette::DeepLayerCape,
                ProductPalette::DeepLayerCape.default_levels(),
                "ML 0-6 km CAPE",
            ),
            Self::Mu6Cape => severe_recipe_with_levels(
                "mu6cape",
                "J/kg",
                ProductPalette::DeepLayerCape,
                ProductPalette::DeepLayerCape.default_levels(),
                "MU 0-6 km CAPE",
            ),
            Self::EffectiveCape => severe_recipe_with_levels(
                "effective_cape",
                "J/kg",
                ProductPalette::EffectiveCape,
                ProductPalette::EffectiveCape.default_levels(),
                "Effective-Layer CAPE",
            ),
            Self::EffectiveInflowBase => height_recipe(
                "effective_inflow_base",
                "m",
                range_i32(0, 5000, 250),
                "Effective Inflow Base Height AGL",
            ),
            Self::EffectiveInflowTop => height_recipe(
                "effective_inflow_top",
                "m",
                range_i32(0, 8000, 250),
                "Effective Inflow Top Height AGL",
            ),
            Self::Srh01 => severe_recipe(
                "srh1",
                "m2/s2",
                ProductPalette::Srh,
                "0-1 km Storm Relative Helicity (m2/s2)",
            ),
            Self::Srh03 => severe_recipe(
                "srh3",
                "m2/s2",
                ProductPalette::Srh,
                "0-3 km Storm Relative Helicity (m2/s2)",
            ),
            Self::EffectiveSrh => severe_recipe(
                "effective_srh",
                "m2/s2",
                ProductPalette::Srh,
                "Effective-Layer SRH (m2/s2)",
            ),
            Self::Shear01 => wind_layer_recipe("bulk_shear", 0.0, 1000.0, "0-1 km Bulk Shear (kt)"),
            Self::StpEffective => ProductRecipe {
                fill_var: "stp",
                fill_units: "",
                palette: ProductPalette::EffectiveStp,
                levels: ProductPalette::EffectiveStp.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Effective STP",
                opts: ComputeOptsPatch {
                    layer_type: Some("effective"),
                    ..Default::default()
                },
            },
            Self::StpFixed => ProductRecipe {
                fill_var: "stp",
                fill_units: "",
                palette: ProductPalette::FixedLayerStp,
                levels: ProductPalette::FixedLayerStp.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Fixed-Layer STP",
                opts: ComputeOptsPatch {
                    layer_type: Some("fixed"),
                    ..Default::default()
                },
            },
            Self::Scp => severe_recipe_with_levels(
                "scp",
                "",
                ProductPalette::Scp,
                scp_levels(),
                "Supercell Composite Parameter",
            ),
            Self::Ehi => severe_recipe(
                "ehi",
                "",
                ProductPalette::Ehi,
                "0-3km Energy Helicity Index",
            ),
            Self::Tehi => severe_recipe("tehi", "", ProductPalette::Tehi, "Tornadic 0-1 km EHI"),
            Self::Tts => severe_recipe(
                "tts",
                "",
                ProductPalette::Tts,
                "Tornadic Tilting and Stretching",
            ),
            Self::VtpMod => severe_recipe(
                "vtp_mod",
                "",
                ProductPalette::Vtp,
                "Modified Violent Tornado Parameter",
            ),
            Self::CriticalAngle => ProductRecipe {
                fill_var: "critical_angle",
                fill_units: "degrees",
                palette: ProductPalette::CriticalAngle,
                levels: ProductPalette::CriticalAngle.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Critical Angle",
                opts: ComputeOptsPatch::default(),
            },
            Self::Ship => severe_recipe(
                "ship",
                "",
                ProductPalette::Ship,
                "Significant Hail Parameter",
            ),
            Self::Bri => ProductRecipe {
                fill_var: "bri",
                fill_units: "",
                palette: ProductPalette::RichardsonNumber,
                levels: ProductPalette::RichardsonNumber.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Bulk Richardson Number",
                opts: ComputeOptsPatch::default(),
            },
            Self::Dcape => severe_recipe_with_levels(
                "dcape",
                "J/kg",
                ProductPalette::Dcape,
                ProductPalette::Dcape.default_levels(),
                "Downdraft CAPE (J/kg)",
            ),
            Self::Dcp => severe_recipe_with_levels(
                "dcp",
                "",
                ProductPalette::Dcp,
                ProductPalette::Dcp.default_levels(),
                "Derecho Composite Parameter",
            ),
            Self::Wndg => severe_recipe_with_levels(
                "wndg",
                "",
                ProductPalette::Wndg,
                ProductPalette::Wndg.default_levels(),
                "Wind Damage Parameter",
            ),
            Self::Esp => severe_recipe_with_levels(
                "esp",
                "",
                ProductPalette::Esp,
                ProductPalette::Esp.default_levels(),
                "Enhanced Stretching Potential",
            ),
            Self::Mmp => severe_recipe_with_levels(
                "mmp",
                "",
                ProductPalette::Mmp,
                ProductPalette::Mmp.default_levels(),
                "MCS Maintenance Probability",
            ),
            Self::Shear06 => ProductRecipe {
                fill_var: "bulk_shear",
                fill_units: "knots",
                palette: ProductPalette::BulkShear,
                levels: ProductPalette::BulkShear.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "0-6 km Bulk Shear (kt)",
                opts: ComputeOptsPatch {
                    bottom_m: Some(0.0),
                    top_m: Some(6000.0),
                    ..Default::default()
                },
            },
            Self::Ebwd => severe_recipe(
                "ebwd",
                "knots",
                ProductPalette::BulkShear,
                "Effective Bulk Wind Difference (kt)",
            ),
            Self::MeanWind06 => {
                wind_layer_recipe("mean_wind", 0.0, 6000.0, "0-6 km Mean Wind (kt)")
            }
            Self::Reflectivity => ProductRecipe {
                fill_var: "maxdbz",
                fill_units: "dBZ",
                palette: ProductPalette::Reflectivity,
                levels: ProductPalette::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Composite Reflectivity",
                opts: ComputeOptsPatch::default(),
            },
            Self::Reflectivity1km => ProductRecipe {
                fill_var: "dbz_1000m_agl",
                fill_units: "dBZ",
                palette: ProductPalette::Reflectivity,
                levels: ProductPalette::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "1km AGL Reflectivity",
                opts: ComputeOptsPatch::default(),
            },
            Self::ReflectivityUh => ProductRecipe {
                fill_var: "dbz_1000m_agl",
                fill_units: "dBZ",
                palette: ProductPalette::Reflectivity,
                levels: ProductPalette::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "1 km AGL Reflectivity + 1 h UH Swath (50/100/200/300 m2/s2)",
                opts: ComputeOptsPatch::default(),
            },
            Self::CloudTopTemp => ProductRecipe {
                fill_var: "ctt",
                fill_units: "degC",
                palette: ProductPalette::SimIr,
                levels: ProductPalette::SimIr.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Simulated IR Satellite (Brightness Temp degC)",
                opts: ComputeOptsPatch::default(),
            },
            Self::CloudFracLow => ProductRecipe {
                fill_var: "cloudfrac_low",
                fill_units: "%",
                palette: ProductPalette::CloudFraction,
                levels: ProductPalette::CloudFraction.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Low Cloud Fraction (%)",
                opts: ComputeOptsPatch::default(),
            },
            Self::CloudFracMid => ProductRecipe {
                fill_var: "cloudfrac_mid",
                fill_units: "%",
                palette: ProductPalette::CloudFraction,
                levels: ProductPalette::CloudFraction.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Mid Cloud Fraction (%)",
                opts: ComputeOptsPatch::default(),
            },
            Self::CloudFracHigh => ProductRecipe {
                fill_var: "cloudfrac_high",
                fill_units: "%",
                palette: ProductPalette::CloudFraction,
                levels: ProductPalette::CloudFraction.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "High Cloud Fraction (%)",
                opts: ComputeOptsPatch::default(),
            },
            Self::SlpWind10m => ProductRecipe {
                fill_var: "wspd10",
                fill_units: "knots",
                palette: ProductPalette::SurfaceWind,
                levels: wind_10m_levels(),
                contour_overlays: vec![slp_contours(slp_contour_levels())],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: EARTH_ROTATED_U10_VAR,
                    v_var: EARTH_ROTATED_V10_VAR,
                    units: "knots",
                    spacing_px: SURFACE_BARB_SPACING_PX,
                    color: Color::BLACK,
                    halo_color: Color::WHITE,
                    halo_width_px: OPERATIONAL_BARB_HALO_WIDTH_PX,
                    width_px: OPERATIONAL_BARB_WIDTH_PX,
                    length_px: OPERATIONAL_BARB_LENGTH_PX,
                }),
                title_template: "Surface MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::SurfaceWind10m => ProductRecipe {
                fill_var: "wspd10",
                fill_units: "knots",
                palette: ProductPalette::SurfaceWind,
                levels: wind_10m_levels(),
                contour_overlays: vec![slp_contours(slp_contour_levels())],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: EARTH_ROTATED_U10_VAR,
                    v_var: EARTH_ROTATED_V10_VAR,
                    units: "knots",
                    spacing_px: SURFACE_BARB_SPACING_PX,
                    color: Color::BLACK,
                    halo_color: Color::WHITE,
                    halo_width_px: OPERATIONAL_BARB_HALO_WIDTH_PX,
                    width_px: OPERATIONAL_BARB_WIDTH_PX,
                    length_px: OPERATIONAL_BARB_LENGTH_PX,
                }),
                title_template: "Surface MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::U10Component => ProductRecipe {
                fill_var: EARTH_ROTATED_U10_VAR,
                fill_units: "knots",
                palette: ProductPalette::WindComponent,
                levels: ProductPalette::WindComponent.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template:
                    "10m Earth-Relative U Wind Component (kt; green = backing, red = veering)",
                opts: ComputeOptsPatch::default(),
            },
            Self::V10Component => ProductRecipe {
                fill_var: EARTH_ROTATED_V10_VAR,
                fill_units: "knots",
                palette: ProductPalette::WindComponent,
                levels: ProductPalette::WindComponent.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template:
                    "10m Earth-Relative V Wind Component (kt; green = negative, red = positive)",
                opts: ComputeOptsPatch::default(),
            },
            Self::T2 => ProductRecipe {
                fill_var: "T2",
                fill_units: "degF",
                palette: ProductPalette::SurfaceTemperature,
                levels: ProductPalette::SurfaceTemperature.default_levels(),
                contour_overlays: vec![slp_contours(slp_contour_levels())],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: EARTH_ROTATED_U10_VAR,
                    v_var: EARTH_ROTATED_V10_VAR,
                    units: "knots",
                    spacing_px: SURFACE_BARB_SPACING_PX,
                    color: Color::BLACK,
                    halo_color: Color::WHITE,
                    halo_width_px: OPERATIONAL_BARB_HALO_WIDTH_PX,
                    width_px: OPERATIONAL_BARB_WIDTH_PX,
                    length_px: OPERATIONAL_BARB_LENGTH_PX,
                }),
                title_template: "Surface Temperature (degF), MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::Td2 => ProductRecipe {
                fill_var: "dp2m",
                fill_units: "degF",
                palette: ProductPalette::SurfaceDewpoint,
                levels: ProductPalette::SurfaceDewpoint.default_levels(),
                contour_overlays: vec![slp_contours(
                    (980..=1032).step_by(2).map(|value| value as f32).collect(),
                )],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: EARTH_ROTATED_U10_VAR,
                    v_var: EARTH_ROTATED_V10_VAR,
                    units: "knots",
                    spacing_px: SURFACE_BARB_SPACING_PX,
                    color: Color::BLACK,
                    halo_color: Color::WHITE,
                    halo_width_px: OPERATIONAL_BARB_HALO_WIDTH_PX,
                    width_px: OPERATIONAL_BARB_WIDTH_PX,
                    length_px: OPERATIONAL_BARB_LENGTH_PX,
                }),
                title_template: "Surface Dewpoint (degF), MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::Rh2 => ProductRecipe {
                fill_var: "rh2m",
                fill_units: "%",
                palette: ProductPalette::SurfaceRelativeHumidity,
                levels: ProductPalette::SurfaceRelativeHumidity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "2 m Relative Humidity",
                opts: ComputeOptsPatch::default(),
            },
            Self::Pwat => ProductRecipe {
                fill_var: "pw",
                fill_units: "in",
                palette: ProductPalette::Pwat,
                levels: ProductPalette::Pwat.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Precipitable Water (in)",
                opts: ComputeOptsPatch::default(),
            },
            Self::PrecipAccum => ProductRecipe {
                fill_var: "precip_accum",
                fill_units: "in",
                palette: ProductPalette::AccumulatedPrecipitation,
                levels: ProductPalette::AccumulatedPrecipitation.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Accumulated Precipitation (in)",
                opts: ComputeOptsPatch::default(),
            },
            Self::UpdraftHelicity => ProductRecipe {
                fill_var: NATIVE_OR_COMPUTED_UH_VAR,
                fill_units: "m2/s2",
                palette: ProductPalette::Uh,
                levels: vec![25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0],
                contour_overlays: vec![uh_contours(NATIVE_OR_COMPUTED_UH_VAR)],
                barb_overlay: None,
                title_template: "Updraft Helicity (m2/s2)",
                opts: ComputeOptsPatch::default(),
            },
            Self::Pblh => height_recipe("PBLH", "m", range_i32(0, 5000, 250), "PBL Height"),
            Self::Terrain => {
                height_recipe("terrain", "m", range_i32(0, 4000, 250), "Terrain Height")
            }
            Self::Lcl => height_recipe("lcl", "m", range_i32(0, 4000, 250), "LCL Height AGL"),
            Self::Lfc => height_recipe("lfc", "m", range_i32(0, 8000, 500), "LFC Height AGL"),
            Self::El => height_recipe(
                "el",
                "m",
                range_i32(4000, 18000, 1000),
                "Equilibrium Level Height AGL",
            ),
            Self::LclTemp => severe_recipe_with_levels(
                "lcl_temp",
                "degC",
                ProductPalette::LclTemperature,
                ProductPalette::LclTemperature.default_levels(),
                "LCL Temperature (degC)",
            ),
            Self::KIndex => severe_recipe_with_levels(
                "k_index",
                "",
                ProductPalette::ConvectiveIndex,
                ProductPalette::ConvectiveIndex.default_levels(),
                "K Index",
            ),
            Self::TotalTotals => severe_recipe_with_levels(
                "total_totals",
                "",
                ProductPalette::TotalTotals,
                ProductPalette::TotalTotals.default_levels(),
                "Total Totals Index",
            ),
            Self::MeanMixr => severe_recipe_with_levels(
                "mean_mixr",
                "g/kg",
                ProductPalette::MeanMixr,
                ProductPalette::MeanMixr.default_levels(),
                "Mean Mixing Ratio (lowest 100 hPa)",
            ),
            Self::LowRh => severe_recipe_with_levels(
                "low_rh",
                "%",
                ProductPalette::SevereMoisture,
                range_i32(0, 100, 5),
                "Low-Level Mean RH (%)",
            ),
            Self::MidRh => severe_recipe_with_levels(
                "mid_rh",
                "%",
                ProductPalette::SevereMoisture,
                range_i32(0, 100, 5),
                "Mid-Level Mean RH (%)",
            ),
            Self::DgzRh => severe_recipe_with_levels(
                "dgz_rh",
                "%",
                ProductPalette::SevereMoisture,
                range_i32(0, 100, 5),
                "Dendritic Growth Zone Mean RH (%)",
            ),
            Self::ConvTemp => ProductRecipe {
                fill_var: "convective_temp",
                fill_units: "degF",
                palette: ProductPalette::SurfaceTemperature,
                levels: ProductPalette::SurfaceTemperature.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Convective Temperature (degF)",
                opts: ComputeOptsPatch::default(),
            },
            Self::MaxTemp => ProductRecipe {
                fill_var: "max_temp",
                fill_units: "degF",
                palette: ProductPalette::SurfaceTemperature,
                levels: ProductPalette::SurfaceTemperature.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Forecast Maximum Temperature (degF)",
                opts: ComputeOptsPatch::default(),
            },
            Self::LapseRate700500 => ProductRecipe {
                fill_var: "lapse_rate_700_500",
                fill_units: "degC/km",
                palette: ProductPalette::LapseRate,
                levels: ProductPalette::LapseRate.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "700-500mb Lapse Rate (degC/km)",
                opts: ComputeOptsPatch::default(),
            },
            Self::LapseRate03 => ProductRecipe {
                fill_var: "lapse_rate_0_3km",
                fill_units: "degC/km",
                palette: ProductPalette::LapseRate,
                levels: vec![4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "0-3 km Lapse Rate (degC/km)",
                opts: ComputeOptsPatch::default(),
            },
            Self::FreezingLevel => height_recipe(
                "freezing_level",
                "m",
                range_i32(0, 6000, 250),
                "Freezing Level AGL",
            ),
            Self::WetBulbZero => height_recipe(
                "wet_bulb_0",
                "m",
                range_i32(0, 6000, 250),
                "Wet-Bulb Zero Height AGL",
            ),
            Self::Fosberg => ProductRecipe {
                fill_var: "fosberg",
                fill_units: "",
                palette: ProductPalette::FireWeather,
                levels: ProductPalette::FireWeather.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Fosberg Fire Weather Index",
                opts: ComputeOptsPatch::default(),
            },
            Self::Haines => ProductRecipe {
                fill_var: "haines",
                fill_units: "",
                palette: ProductPalette::Haines,
                levels: ProductPalette::Haines.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Haines Index",
                opts: ComputeOptsPatch::default(),
            },
            Self::Hdw => ProductRecipe {
                fill_var: "hdw",
                fill_units: "",
                palette: ProductPalette::Hdw,
                levels: ProductPalette::Hdw.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Hot-Dry-Windy Index",
                opts: ComputeOptsPatch::default(),
            },
            Self::Height200Wind => {
                pressure_height_wind_recipe(200, "200 mb Height (dam), Wind (kt)")
            }
            Self::Temp200Wind => {
                pressure_temp_wind_recipe(200, "200 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind200 => pressure_wind_recipe(200, "200 mb Height (dam), Wind (kt)"),
            Self::Height250Wind => {
                pressure_height_wind_recipe(250, "250 mb Height (dam), Wind (kt)")
            }
            Self::Temp250Wind => {
                pressure_temp_wind_recipe(250, "250 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind250 => pressure_jet_recipe(250, "250 mb Jet Streak (kt), Height (dam)"),
            Self::Height300Wind => {
                pressure_height_wind_recipe(300, "300 mb Height (dam), Wind (kt)")
            }
            Self::Temp300Wind => {
                pressure_temp_wind_recipe(300, "300 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind300 => pressure_jet_recipe(300, "300 mb Jet Streak (kt), Height (dam)"),
            Self::Height500Wind => {
                pressure_height_wind_recipe(500, "500 mb Height (dam), Wind (kt)")
            }
            Self::Temp500Wind => {
                pressure_temp_wind_recipe(500, "500 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind500 => pressure_wind_recipe(500, "500 mb Height (dam), Wind (kt)"),
            Self::Vort500Wind => upper_air_analysis_recipe(
                500,
                "avo_500mb",
                "s-1",
                ProductPalette::Vorticity,
                vec![
                    0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00040, 0.00050,
                ],
                "500 hPa Absolute Vorticity, Height (dam), and Wind (kt)",
            ),
            Self::Pvo500 => upper_air_analysis_recipe(
                500,
                "pvo_500mb",
                "PVU",
                ProductPalette::PotentialVorticity,
                ProductPalette::PotentialVorticity.default_levels(),
                "500 hPa Potential Vorticity (PVU), Height (dam), and Wind (kt)",
            ),
            Self::Omega500 => upper_air_analysis_recipe(
                500,
                "omega_500mb",
                "Pa/s",
                ProductPalette::Omega,
                ProductPalette::Omega.default_levels(),
                "500 hPa Omega (Pa/s), Height (dam), and Wind (kt)",
            ),
            Self::ThetaW850 => upper_air_analysis_recipe(
                850,
                "theta_w_850mb",
                "degC",
                ProductPalette::WetBulbPotentialTemperature,
                ProductPalette::WetBulbPotentialTemperature.default_levels(),
                "850 hPa Wet-Bulb Potential Temperature (degC), Height (dam), and Wind (kt)",
            ),
            Self::Temp700Wind => {
                pressure_temp_wind_recipe(700, "700 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Height700Wind => {
                pressure_height_wind_recipe(700, "700 mb Height (dam), Wind (kt)")
            }
            Self::Rh700Wind => upper_air_analysis_recipe(
                700,
                "rh_700mb",
                "%",
                ProductPalette::UpperAirRelativeHumidity,
                ProductPalette::UpperAirRelativeHumidity.default_levels(),
                "700 hPa Relative Humidity (%), Height (dam), and Wind (kt)",
            ),
            Self::Omega700Wind => upper_air_analysis_recipe(
                700,
                "omega_700mb",
                "Pa/s",
                ProductPalette::Omega,
                ProductPalette::Omega.default_levels(),
                "700 hPa Omega (Pa/s), Height (dam), and Wind (kt)",
            ),
            Self::Height850Wind => {
                pressure_height_wind_recipe(850, "850 mb Height (dam), Wind (kt)")
            }
            Self::Temp850Wind => {
                pressure_temp_wind_recipe(850, "850 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Td850Wind => upper_air_analysis_recipe(
                850,
                "td_850mb",
                "degC",
                ProductPalette::UpperAirDewpoint,
                pressure_dewpoint_levels(850),
                "850 hPa Dewpoint (degC), Height (dam), and Wind (kt)",
            ),
            Self::Wind850 => pressure_wind_recipe(850, "850 mb Height (dam), Wind (kt)"),
        }
    }
}

pub const DEFAULT_PRODUCT_SUITE: &[WrfProduct] = &[
    WrfProduct::SbEcape,
    WrfProduct::MlEcape,
    WrfProduct::MuEcape,
    WrfProduct::Ncape,
    WrfProduct::EcapeCape,
    WrfProduct::EcapeCin,
    WrfProduct::EcapeLfc,
    WrfProduct::EcapeEl,
    WrfProduct::EcapeScp,
    WrfProduct::EcapeEhi,
    WrfProduct::Sbcape,
    WrfProduct::Sbcin,
    WrfProduct::Mlcape,
    WrfProduct::Mlcin,
    WrfProduct::Mucape,
    WrfProduct::Mucin,
    WrfProduct::Sb3Cape,
    WrfProduct::Ml3Cape,
    WrfProduct::Mu3Cape,
    WrfProduct::Sb6Cape,
    WrfProduct::Ml6Cape,
    WrfProduct::Mu6Cape,
    WrfProduct::EffectiveCape,
    WrfProduct::EffectiveInflowBase,
    WrfProduct::EffectiveInflowTop,
    WrfProduct::Srh01,
    WrfProduct::Srh03,
    WrfProduct::EffectiveSrh,
    WrfProduct::Shear01,
    WrfProduct::Shear06,
    WrfProduct::Ebwd,
    WrfProduct::MeanWind06,
    WrfProduct::StpEffective,
    WrfProduct::StpFixed,
    WrfProduct::Scp,
    WrfProduct::Ehi,
    WrfProduct::Tehi,
    WrfProduct::Tts,
    WrfProduct::VtpMod,
    WrfProduct::CriticalAngle,
    WrfProduct::Ship,
    WrfProduct::Bri,
    WrfProduct::Dcape,
    WrfProduct::Dcp,
    WrfProduct::Wndg,
    WrfProduct::Esp,
    WrfProduct::Mmp,
    WrfProduct::Reflectivity,
    WrfProduct::Reflectivity1km,
    WrfProduct::ReflectivityUh,
    WrfProduct::UpdraftHelicity,
    WrfProduct::CloudFracLow,
    WrfProduct::CloudFracMid,
    WrfProduct::CloudFracHigh,
    WrfProduct::SlpWind10m,
    WrfProduct::SurfaceWind10m,
    WrfProduct::U10Component,
    WrfProduct::V10Component,
    WrfProduct::T2,
    WrfProduct::Td2,
    WrfProduct::Rh2,
    WrfProduct::Pwat,
    WrfProduct::PrecipAccum,
    WrfProduct::Pblh,
    WrfProduct::Terrain,
    WrfProduct::Lcl,
    WrfProduct::Lfc,
    WrfProduct::El,
    WrfProduct::LclTemp,
    WrfProduct::KIndex,
    WrfProduct::TotalTotals,
    WrfProduct::MeanMixr,
    WrfProduct::LowRh,
    WrfProduct::MidRh,
    WrfProduct::DgzRh,
    WrfProduct::ConvTemp,
    WrfProduct::MaxTemp,
    WrfProduct::LapseRate700500,
    WrfProduct::LapseRate03,
    WrfProduct::FreezingLevel,
    WrfProduct::WetBulbZero,
    WrfProduct::Fosberg,
    WrfProduct::Haines,
    WrfProduct::Hdw,
    WrfProduct::Height200Wind,
    WrfProduct::Temp200Wind,
    WrfProduct::Wind200,
    WrfProduct::Height250Wind,
    WrfProduct::Temp250Wind,
    WrfProduct::Wind250,
    WrfProduct::Height300Wind,
    WrfProduct::Temp300Wind,
    WrfProduct::Wind300,
    WrfProduct::Height500Wind,
    WrfProduct::Temp500Wind,
    WrfProduct::Wind500,
    WrfProduct::Vort500Wind,
    WrfProduct::Pvo500,
    WrfProduct::Omega500,
    WrfProduct::ThetaW850,
    WrfProduct::Temp700Wind,
    WrfProduct::Height700Wind,
    WrfProduct::Rh700Wind,
    WrfProduct::Omega700Wind,
    WrfProduct::Height850Wind,
    WrfProduct::Temp850Wind,
    WrfProduct::Td850Wind,
    WrfProduct::Wind850,
    WrfProduct::CloudTopTemp,
];

pub fn default_product_suite() -> &'static [WrfProduct] {
    DEFAULT_PRODUCT_SUITE
}

pub fn all_products() -> impl Iterator<Item = WrfProduct> {
    std::iter::once(WrfProduct::Ecape).chain(default_product_suite().iter().copied())
}

pub type ProductId = WrfProduct;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductInputSource {
    CurrentFile,
    ExplicitHistory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredInput {
    pub name: String,
    pub role: String,
    pub units: String,
    pub source: ProductInputSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryRequirementKind {
    None,
    CurrentFileWindow,
    ExplicitHistoryWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryRequirement {
    pub kind: HistoryRequirementKind,
    pub window_minutes: Option<u32>,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductSpec {
    pub id: String,
    pub aliases: Vec<String>,
    pub title: String,
    pub fill_variable: String,
    pub output_units: String,
    pub palette: ProductPalette,
    pub levels: Vec<f32>,
    pub legend_ticks: Option<Vec<f64>>,
    pub required_inputs: Vec<RequiredInput>,
    pub history: HistoryRequirement,
    pub frame_policy: ProductFramePolicy,
    pub visual_mode: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderSidecar {
    pub package_name: String,
    pub package_version: String,
    pub product_id: String,
    pub input_file: String,
    pub valid_time: Option<String>,
    pub init_time: Option<String>,
    pub units: String,
    pub source: String,
    pub provenance: String,
    pub required_inputs: Vec<RequiredInput>,
    pub history: HistoryRequirement,
    pub history_files: Vec<String>,
    pub history_dir: Option<String>,
    pub frame_policy: ProductFramePolicy,
}

#[derive(Debug, Clone)]
pub struct ProductRecipe {
    pub fill_var: &'static str,
    pub fill_units: &'static str,
    pub palette: ProductPalette,
    pub levels: Vec<f32>,
    pub contour_overlays: Vec<ContourRecipe>,
    pub barb_overlay: Option<WindBarbRecipe>,
    pub title_template: &'static str,
    pub opts: ComputeOptsPatch,
}

#[derive(Debug, Clone)]
pub struct ContourRecipe {
    pub var: &'static str,
    pub units: &'static str,
    pub levels: Vec<f32>,
    pub color: Color,
    pub width_px: u32,
    pub halo_color: Color,
    pub halo_width_px: u32,
    pub major_every: usize,
    pub major_width_px: u32,
    pub label_every: usize,
    pub labels: bool,
    pub show_extrema: bool,
    pub opts: ComputeOptsPatch,
}

#[derive(Debug, Clone)]
pub struct WindBarbRecipe {
    pub u_var: &'static str,
    pub v_var: &'static str,
    pub units: &'static str,
    pub spacing_px: f64,
    pub color: Color,
    pub halo_color: Color,
    pub halo_width_px: u32,
    pub width_px: u32,
    pub length_px: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProductOverlayRecipe {
    UhTrackSwath(UhTrackOverlayRecipe),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UhTrackOverlayRecipe {
    pub source_var: &'static str,
    pub units: &'static str,
    pub threshold_bins: Vec<f32>,
    pub fill_colors: Vec<Color>,
    pub edge_color: Color,
    pub edge_width_px: u32,
    pub edge_halo_color: Color,
    pub edge_halo_width_px: u32,
    pub lookback_minutes: u32,
    pub label: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductOverlayLegendEntry {
    pub label: String,
    pub fill_color: Color,
    pub outline_color: Color,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductOverlayLegendRecipe {
    pub title: String,
    pub entries: Vec<ProductOverlayLegendEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductVisualSourceRole {
    Fill,
    ContourOverlay,
    WindBarbOverlay,
    UhTrackOverlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductSourceKind {
    Native,
    NativeOrComputed,
    NativeOrInterpolated,
    Derived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductTemporalSemantics {
    Instant,
    Accumulation { window_minutes: Option<u32> },
    HistoryMaximum { lookback_minutes: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductSourceSemantics {
    pub role: ProductVisualSourceRole,
    pub var: &'static str,
    pub units: &'static str,
    pub source: ProductSourceKind,
    pub temporal: ProductTemporalSemantics,
    pub label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpperAirFillRole {
    Height,
    JetSpeed,
    WindSpeed,
    Temperature,
    Dewpoint,
    Vorticity,
    PotentialVorticity,
    Omega,
    RelativeHumidity,
    WetBulbPotentialTemperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpperAirTemplateRecipe {
    pub level_hpa: u16,
    pub fill_role: UpperAirFillRole,
    pub fill_var: &'static str,
    pub fill_units: &'static str,
    pub height_contour_var: &'static str,
    pub height_units: &'static str,
    pub wind_u_var: &'static str,
    pub wind_v_var: &'static str,
    pub wind_units: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductFramePolicy {
    FullDomain,
    FiniteData,
    FiniteDataWithOverlays,
    StormCentered,
    GeographicCrop,
}

#[derive(Debug, Clone)]
pub struct ProductVisualRecipe {
    pub palette: ProductPalette,
    pub levels: Vec<f32>,
    pub legend_ticks: Option<Vec<f64>>,
    pub legend_levels: Option<Vec<f64>>,
    pub colorbar_label: Option<&'static str>,
    pub mask_policy: MaskPolicy,
    pub extend: ExtendMode,
    pub contour_overlays: Vec<ContourRecipe>,
    pub barb_overlay: Option<WindBarbRecipe>,
    pub overlays: Vec<ProductOverlayRecipe>,
    pub overlay_legends: Vec<ProductOverlayLegendRecipe>,
    pub source_semantics: Vec<ProductSourceSemantics>,
    pub upper_air_template: Option<UpperAirTemplateRecipe>,
    pub frame_policy: ProductFramePolicy,
    pub provenance_label: &'static str,
}

impl ProductRecipe {
    pub fn visual_recipe(&self, product: WrfProduct) -> ProductVisualRecipe {
        let palette = self.palette;
        let legend_ticks = product_tick_values(product);
        let overlays = product_overlay_recipes(product);
        let overlay_legends = product_overlay_legends(&overlays);
        ProductVisualRecipe {
            palette,
            levels: self.levels.clone(),
            legend_levels: legend_ticks
                .as_ref()
                .filter(|levels| levels.len() >= 2)
                .cloned(),
            legend_ticks,
            colorbar_label: product_display_units(self.fill_units),
            mask_policy: palette.mask_policy(),
            extend: palette.default_extend(),
            contour_overlays: self.contour_overlays.clone(),
            barb_overlay: self.barb_overlay.clone(),
            overlays,
            overlay_legends,
            source_semantics: product_source_semantics(product, self),
            upper_air_template: upper_air_template_recipe(product, self),
            frame_policy: product_frame_policy(product),
            provenance_label: product_provenance_label(product),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProductRenderOptions {
    pub history_dir: Option<PathBuf>,
    pub history_files: Vec<PathBuf>,
    pub geographic_bounds: Option<GeographicBounds>,
    pub storm_center: Option<StormCenteredFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StormCenteredFrame {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub radius_km: f64,
}

impl StormCenteredFrame {
    pub fn new(lat_deg: f64, lon_deg: f64, radius_km: f64) -> Self {
        Self {
            lat_deg: if lat_deg.is_finite() {
                lat_deg.clamp(-89.9, 89.9)
            } else {
                0.0
            },
            lon_deg: normalize_render_option_longitude(lon_deg),
            radius_km: if radius_km.is_finite() && radius_km > 0.0 {
                radius_km
            } else {
                75.0
            },
        }
    }

    fn geographic_bounds(self) -> GeographicBounds {
        let lat_radius_deg = self.radius_km / 111.32;
        let lon_scale = (111.32 * self.lat_deg.to_radians().cos().abs()).max(15.0);
        let lon_radius_deg = self.radius_km / lon_scale;
        GeographicBounds::new(
            self.lon_deg - lon_radius_deg,
            self.lon_deg + lon_radius_deg,
            self.lat_deg - lat_radius_deg,
            self.lat_deg + lat_radius_deg,
        )
    }
}

impl ProductRenderOptions {
    pub fn single_file() -> Self {
        Self::default()
    }

    pub fn with_history_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.history_dir = Some(path.into());
        self
    }

    pub fn with_history_files<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.history_files = paths.into_iter().map(Into::into).collect();
        self
    }

    pub fn push_history_file<P: Into<PathBuf>>(&mut self, path: P) {
        self.history_files.push(path.into());
    }

    pub fn with_geographic_bounds(
        mut self,
        west_deg: f64,
        east_deg: f64,
        south_deg: f64,
        north_deg: f64,
    ) -> Self {
        self.geographic_bounds = Some(GeographicBounds::new(
            west_deg, east_deg, south_deg, north_deg,
        ));
        self
    }

    pub fn with_storm_center(mut self, lat_deg: f64, lon_deg: f64, radius_km: f64) -> Self {
        self.storm_center = Some(StormCenteredFrame::new(lat_deg, lon_deg, radius_km));
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductInputContract {
    pub product: WrfProduct,
    pub current_wrfout_required: bool,
    pub optional_history: Option<HistoryInputContract>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryInputContract {
    pub cli_flag: &'static str,
    pub lookback_minutes: u32,
    pub description: &'static str,
}

pub fn product_input_contract(product: WrfProduct) -> ProductInputContract {
    ProductInputContract {
        product,
        current_wrfout_required: true,
        optional_history: match product {
            WrfProduct::ReflectivityUh => Some(HistoryInputContract {
                cli_flag: "--history-dir",
                lookback_minutes: 60,
                description: "same-domain wrfout files with valid times in the previous 60 minutes; used only for the 1h UH swath",
            }),
            _ => None,
        },
    }
}

pub fn operational_product_presentation_style() -> StaticPlotStyle {
    OPERATIONAL_FAST
}

pub fn operational_product_colorbar_orientation() -> ColorbarOrientation {
    ColorbarOrientation::VerticalRight
}

pub fn operational_product_raster_sample_mode() -> RasterSampleMode {
    RasterSampleMode::Linear
}

pub fn operational_product_render_density() -> RenderDensity {
    RenderDensity {
        fill: LevelDensity::default(),
        palette_multiplier: 1,
    }
}

pub fn operational_product_legend_mode() -> LegendMode {
    LegendMode::Stepped
}

pub fn operational_product_legend_density() -> LevelDensity {
    LevelDensity::default()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductContourContractSummary {
    pub var: &'static str,
    pub units: &'static str,
    pub level_count: usize,
    pub first_level: Option<f64>,
    pub last_level: Option<f64>,
    pub minor_interval: Option<f64>,
    pub major_every: usize,
    pub label_every: usize,
    pub labels: bool,
    pub color: Color,
    pub halo_color: Color,
    pub width_px: u32,
    pub major_width_px: u32,
    pub halo_width_px: u32,
    pub show_extrema: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductBarbContractSummary {
    pub u_var: &'static str,
    pub v_var: &'static str,
    pub units: &'static str,
    pub stride_x: usize,
    pub stride_y: usize,
    pub spacing_px: f64,
    pub color: Color,
    pub halo_color: Color,
    pub width_px: u32,
    pub halo_width_px: u32,
    pub length_px: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductOverlayContractSummary {
    pub label: &'static str,
    pub source_var: &'static str,
    pub units: &'static str,
    pub threshold_bins: Vec<f64>,
    pub fill_count: usize,
    pub edge_color: Color,
    pub edge_width_px: u32,
    pub edge_halo_color: Color,
    pub edge_halo_width_px: u32,
    pub lookback_minutes: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProductVisualContractSummary {
    pub product: WrfProduct,
    pub title: &'static str,
    pub presentation_style: StaticPlotStyle,
    pub visual_mode: ProductVisualMode,
    pub colorbar_orientation: ColorbarOrientation,
    pub raster_sample_mode: RasterSampleMode,
    pub render_density: RenderDensity,
    pub legend_mode: LegendMode,
    pub legend_density: LevelDensity,
    pub fill_var: &'static str,
    pub fill_units: &'static str,
    pub palette: ProductPalette,
    pub extend_mode: ExtendMode,
    pub palette_color_count: usize,
    pub level_count: usize,
    pub first_level: Option<f64>,
    pub last_level: Option<f64>,
    pub level_interval: Option<f64>,
    pub legend_ticks: Option<Vec<f64>>,
    pub legend_thresholds: Option<Vec<f64>>,
    pub colorbar_tick_step: Option<f64>,
    pub colorbar_label: Option<&'static str>,
    pub mask_policy: MaskPolicy,
    pub frame_policy: ProductFramePolicy,
    pub frame_source: Option<DomainFrameSource>,
    pub frame_clear_outside: Option<bool>,
    pub frame_inset_px: Option<u32>,
    pub frame_outline_width_px: Option<u32>,
    pub frame_padding_fraction: f64,
    pub contour_count: usize,
    pub contours: Vec<ProductContourContractSummary>,
    pub barb_units: Option<&'static str>,
    pub barb_spacing_px: Option<f64>,
    pub barbs: Option<ProductBarbContractSummary>,
    pub overlay_count: usize,
    pub overlays: Vec<ProductOverlayContractSummary>,
    pub overlay_legend_titles: Vec<String>,
    pub provenance_label: &'static str,
    pub source_semantics: Vec<ProductSourceSemantics>,
    pub upper_air_template: Option<UpperAirTemplateRecipe>,
}

pub fn product_visual_contract_summary(product: WrfProduct) -> ProductVisualContractSummary {
    let recipe = product.recipe();
    let visual = recipe.visual_recipe(product);
    let frame = domain_frame_for_policy(visual.frame_policy);
    ProductVisualContractSummary {
        product,
        title: recipe.title_template,
        presentation_style: operational_product_presentation_style(),
        visual_mode: product.visual_mode(),
        colorbar_orientation: operational_product_colorbar_orientation(),
        raster_sample_mode: operational_product_raster_sample_mode(),
        render_density: operational_product_render_density(),
        legend_mode: operational_product_legend_mode(),
        legend_density: operational_product_legend_density(),
        fill_var: recipe.fill_var,
        fill_units: recipe.fill_units,
        palette: visual.palette,
        extend_mode: visual.extend,
        palette_color_count: visual.palette.colors().len(),
        level_count: visual.levels.len(),
        first_level: visual.levels.first().map(|value| *value as f64),
        last_level: visual.levels.last().map(|value| *value as f64),
        level_interval: uniform_level_interval(&visual.levels),
        legend_ticks: visual.legend_ticks.clone(),
        legend_thresholds: visual.legend_levels.clone(),
        colorbar_tick_step: product_tick_step(product),
        colorbar_label: visual.colorbar_label,
        mask_policy: visual.mask_policy,
        frame_policy: visual.frame_policy,
        frame_source: frame.map(|frame| frame.source),
        frame_clear_outside: frame.map(|frame| frame.clear_outside),
        frame_inset_px: frame.map(|frame| frame.inset_px),
        frame_outline_width_px: frame.map(|frame| frame.outline_width),
        frame_padding_fraction: frame_padding_fraction(visual.frame_policy),
        contour_count: visual.contour_overlays.len(),
        contours: visual
            .contour_overlays
            .iter()
            .map(contour_contract_summary)
            .collect(),
        barb_units: visual.barb_overlay.as_ref().map(|barbs| barbs.units),
        barb_spacing_px: visual.barb_overlay.as_ref().map(|barbs| barbs.spacing_px),
        barbs: visual.barb_overlay.as_ref().map(barb_contract_summary),
        overlay_count: visual.overlays.len(),
        overlays: visual
            .overlays
            .iter()
            .map(overlay_contract_summary)
            .collect(),
        overlay_legend_titles: visual
            .overlay_legends
            .iter()
            .map(|legend| legend.title.clone())
            .collect(),
        provenance_label: visual.provenance_label,
        source_semantics: visual.source_semantics.clone(),
        upper_air_template: visual.upper_air_template,
    }
}

fn contour_contract_summary(contour: &ContourRecipe) -> ProductContourContractSummary {
    ProductContourContractSummary {
        var: contour.var,
        units: contour.units,
        level_count: contour.levels.len(),
        first_level: contour.levels.first().map(|value| *value as f64),
        last_level: contour.levels.last().map(|value| *value as f64),
        minor_interval: uniform_level_interval(&contour.levels),
        major_every: contour.major_every,
        label_every: contour.label_every,
        labels: contour.labels,
        color: contour.color,
        halo_color: contour.halo_color,
        width_px: contour.width_px,
        major_width_px: contour.major_width_px,
        halo_width_px: contour.halo_width_px,
        show_extrema: contour.show_extrema,
    }
}

fn barb_contract_summary(barbs: &WindBarbRecipe) -> ProductBarbContractSummary {
    ProductBarbContractSummary {
        u_var: barbs.u_var,
        v_var: barbs.v_var,
        units: barbs.units,
        stride_x: OPERATIONAL_BARB_GRID_STRIDE,
        stride_y: OPERATIONAL_BARB_GRID_STRIDE,
        spacing_px: barbs.spacing_px,
        color: barbs.color,
        halo_color: barbs.halo_color,
        width_px: barbs.width_px,
        halo_width_px: barbs.halo_width_px,
        length_px: barbs.length_px,
    }
}

fn overlay_contract_summary(overlay: &ProductOverlayRecipe) -> ProductOverlayContractSummary {
    match overlay {
        ProductOverlayRecipe::UhTrackSwath(overlay) => ProductOverlayContractSummary {
            label: overlay.label,
            source_var: overlay.source_var,
            units: overlay.units,
            threshold_bins: overlay
                .threshold_bins
                .iter()
                .map(|value| *value as f64)
                .collect(),
            fill_count: overlay.fill_colors.len(),
            edge_color: overlay.edge_color,
            edge_width_px: overlay.edge_width_px,
            edge_halo_color: overlay.edge_halo_color,
            edge_halo_width_px: overlay.edge_halo_width_px,
            lookback_minutes: Some(overlay.lookback_minutes),
        },
    }
}

fn uniform_level_interval(levels: &[f32]) -> Option<f64> {
    let mut windows = levels.windows(2);
    let first = windows.next()?;
    let interval = (first[1] - first[0]) as f64;
    if !interval.is_finite() || interval <= 0.0 {
        return None;
    }
    let is_uniform = windows.all(|pair| {
        let delta = (pair[1] - pair[0]) as f64;
        (delta - interval).abs() <= interval.abs().max(1.0) * 1.0e-4
    });
    is_uniform.then_some(interval)
}

#[derive(Debug, Clone, Default)]
pub struct ComputeOptsPatch {
    pub units: Option<&'static str>,
    pub parcel_type: Option<&'static str>,
    pub storm_motion_type: Option<&'static str>,
    pub layer_type: Option<&'static str>,
    pub bottom_m: Option<f64>,
    pub top_m: Option<f64>,
}

impl ComputeOptsPatch {
    fn apply(&self, requested_units: &'static str) -> ComputeOpts {
        ComputeOpts {
            units: if requested_units.is_empty() {
                self.units.map(str::to_string)
            } else {
                Some(requested_units.to_string())
            },
            parcel_type: self.parcel_type.map(str::to_string),
            storm_motion_type: self.storm_motion_type.map(str::to_string),
            layer_type: self.layer_type.map(str::to_string),
            bottom_m: self.bottom_m,
            top_m: self.top_m,
            ..Default::default()
        }
    }
}

pub fn product_spec(product: ProductId) -> ProductSpec {
    let recipe = product.recipe();
    let visual = recipe.visual_recipe(product);
    ProductSpec {
        id: product.canonical_name().to_string(),
        aliases: product_aliases(product),
        title: recipe.title_template.to_string(),
        fill_variable: recipe.fill_var.to_string(),
        output_units: display_unit_label(recipe.fill_units).to_string(),
        palette: visual.palette,
        levels: visual.levels.clone(),
        legend_ticks: visual.legend_ticks.clone(),
        required_inputs: required_inputs_for_product(product, &recipe, &visual),
        history: history_requirement(product),
        frame_policy: visual.frame_policy,
        visual_mode: visual_mode_name(product.visual_mode()).to_string(),
    }
}

pub fn product_specs() -> Vec<ProductSpec> {
    all_products().map(product_spec).collect()
}

pub fn product_specs_for(products: &[ProductId]) -> Vec<ProductSpec> {
    products.iter().copied().map(product_spec).collect()
}

pub fn product_specs_json(products: &[ProductId]) -> ProductResult<String> {
    Ok(serde_json::to_string_pretty(&product_specs_for(products))?)
}

pub fn product_docs_markdown() -> String {
    let mut out = String::from(
        "| Product | Units | Palette | History | Required inputs |\n| --- | --- | --- | --- | --- |\n",
    );
    for spec in product_specs() {
        let inputs = spec
            .required_inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "| `{}` | `{}` | `{:?}` | `{:?}` | {} |\n",
            spec.id, spec.output_units, spec.palette, spec.history.kind, inputs
        ));
    }
    out
}

fn product_aliases(product: ProductId) -> Vec<String> {
    let aliases: &[&str] = match product {
        WrfProduct::SbEcape => &["sb_ecape", "sbecape", "surface_based_ecape"],
        WrfProduct::MlEcape => &["ml_ecape", "mlecape", "mixed_layer_ecape"],
        WrfProduct::MuEcape => &["mu_ecape", "muecape", "most_unstable_ecape"],
        WrfProduct::Sbcape => &["sbcape", "surface_based_cape"],
        WrfProduct::Mlcape => &["mlcape", "mixed_layer_cape"],
        WrfProduct::Mucape => &["mucape", "most_unstable_cape"],
        WrfProduct::Srh01 => &["srh01", "srh1", "srh_0_1km", "srh01km"],
        WrfProduct::Srh03 => &["srh03", "srh3", "srh_0_3km", "srh03km"],
        WrfProduct::StpEffective => &["stp_effective", "stp"],
        WrfProduct::StpFixed => &["stp_fixed"],
        WrfProduct::Scp => &["scp", "supercell_composite_parameter"],
        WrfProduct::Reflectivity => &["reflectivity", "dbz", "maxdbz"],
        WrfProduct::ReflectivityUh => &[
            "reflectivity_uh",
            "uh_reflectivity",
            "refl_uh",
            "dbz_uh",
            "reflectivity_uh_combo",
        ],
        WrfProduct::SurfaceWind10m => &["surface_wind10m", "surface_wind", "wspd10", "wind10m"],
        WrfProduct::SlpWind10m => &["slp_wind10m", "slp_wind", "mslp_wind10m"],
        WrfProduct::PrecipAccum => &["precip_accum", "precip", "qpf", "rainnc"],
        WrfProduct::Pwat => &["pw", "pwat", "precipitable_water"],
        _ => &[],
    };
    let mut result = Vec::with_capacity(aliases.len() + 1);
    result.push(product.canonical_name().to_string());
    for alias in aliases {
        if *alias != product.canonical_name() {
            result.push((*alias).to_string());
        }
    }
    result
}

fn required_inputs_for_product(
    _product: ProductId,
    recipe: &ProductRecipe,
    visual: &ProductVisualRecipe,
) -> Vec<RequiredInput> {
    let mut inputs = Vec::new();
    push_required_inputs(
        &mut inputs,
        recipe.fill_var,
        "fill",
        recipe.fill_units,
        ProductInputSource::CurrentFile,
    );
    for contour in &recipe.contour_overlays {
        push_required_inputs(
            &mut inputs,
            contour.var,
            "contour",
            contour.units,
            ProductInputSource::CurrentFile,
        );
    }
    if let Some(barbs) = &recipe.barb_overlay {
        push_required_inputs(
            &mut inputs,
            barbs.u_var,
            "barb_u",
            barbs.units,
            ProductInputSource::CurrentFile,
        );
        push_required_inputs(
            &mut inputs,
            barbs.v_var,
            "barb_v",
            barbs.units,
            ProductInputSource::CurrentFile,
        );
    }
    for overlay in &visual.overlays {
        match overlay {
            ProductOverlayRecipe::UhTrackSwath(overlay) => {
                push_required_inputs(
                    &mut inputs,
                    overlay.source_var,
                    "overlay",
                    overlay.units,
                    ProductInputSource::CurrentFile,
                );
            }
        }
    }
    dedup_required_inputs(inputs)
}

fn push_required_inputs(
    inputs: &mut Vec<RequiredInput>,
    var: &str,
    role: &str,
    units: &str,
    source: ProductInputSource,
) {
    match var {
        "precip_accum" => {
            push_required_input(inputs, "RAINC", role, "mm", source);
            push_required_input(inputs, "RAINNC", role, "mm", source);
        }
        "uhel_0_3km_1h_max" => {
            push_required_input(
                inputs,
                "UP_HELI_MAX|uhel",
                role,
                "m2/s2",
                ProductInputSource::CurrentFile,
            );
            push_required_input(
                inputs,
                "UP_HELI_MAX|uhel",
                "history_window",
                "m2/s2",
                ProductInputSource::ExplicitHistory,
            );
        }
        NATIVE_OR_COMPUTED_UH_VAR => {
            push_required_input(inputs, "UP_HELI_MAX|uhel", role, "m2/s2", source);
        }
        _ => {
            if let Some((source_var, _)) = parse_multiplane_2d_var(var) {
                push_required_input(inputs, source_var, role, units, source);
            } else if let Some((base_var, _)) = parse_height_level_var(var) {
                push_required_input(inputs, "height_agl", "vertical_coordinate", "m", source);
                push_required_input(inputs, base_var, role, units, source);
            } else if let Some((source_var, _, _)) = parse_multiplane_pressure_level_var(var) {
                push_required_input(inputs, "pressure", "vertical_coordinate", "hPa", source);
                push_required_input(inputs, source_var, role, units, source);
            } else if let Some((base_var, _)) = parse_pressure_level_var(var) {
                push_required_input(inputs, "pressure", "vertical_coordinate", "hPa", source);
                push_required_input(inputs, base_var, role, units, source);
            } else {
                push_required_input(inputs, var, role, units, source);
            }
        }
    }
}

fn push_required_input(
    inputs: &mut Vec<RequiredInput>,
    name: &str,
    role: &str,
    units: &str,
    source: ProductInputSource,
) {
    inputs.push(RequiredInput {
        name: name.to_string(),
        role: role.to_string(),
        units: display_unit_label(units).to_string(),
        source,
    });
}

fn dedup_required_inputs(inputs: Vec<RequiredInput>) -> Vec<RequiredInput> {
    let mut deduped = Vec::new();
    for input in inputs {
        if !deduped.iter().any(|existing: &RequiredInput| {
            existing.name == input.name
                && existing.role == input.role
                && existing.source == input.source
        }) {
            deduped.push(input);
        }
    }
    deduped
}

fn history_requirement(product: ProductId) -> HistoryRequirement {
    if product == WrfProduct::ReflectivityUh {
        HistoryRequirement {
            kind: HistoryRequirementKind::ExplicitHistoryWindow,
            window_minutes: Some(60),
            note: "single-file by default; pass explicit previous files or --history-dir for a complete one-hour UH window".to_string(),
        }
    } else {
        HistoryRequirement {
            kind: HistoryRequirementKind::None,
            window_minutes: None,
            note: "single current file".to_string(),
        }
    }
}

fn visual_mode_name(mode: ProductVisualMode) -> &'static str {
    match mode {
        ProductVisualMode::FilledMeteorology => "filled_meteorology",
        ProductVisualMode::UpperAirAnalysis => "upper_air_analysis",
        ProductVisualMode::OverlayAnalysis => "overlay_analysis",
        ProductVisualMode::SevereDiagnostic => "severe_diagnostic",
        ProductVisualMode::PanelMember => "panel_member",
        ProductVisualMode::ComparisonPanel => "comparison_panel",
    }
}

fn display_unit_label(units: &str) -> &str {
    match units.trim() {
        "" => "unitless",
        "knots" => "kt",
        "degrees" => "deg",
        _ => units,
    }
}

pub fn parse_product(name: &str) -> ProductResult<WrfProduct> {
    WrfProduct::from_name(name).ok_or_else(|| ProductError::UnknownProduct(name.to_string()))
}

pub fn build_product_request(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: Option<usize>,
) -> ProductResult<MapRenderRequest> {
    build_product_request_with_options(file, product, timeidx, &ProductRenderOptions::default())
}

pub fn build_product_request_with_options(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: Option<usize>,
    options: &ProductRenderOptions,
) -> ProductResult<MapRenderRequest> {
    let t = timeidx.unwrap_or(0);
    let recipe = product.recipe();
    let visual = recipe.visual_recipe(product);
    let fill = build_recipe_field(
        file,
        recipe.fill_var,
        recipe.fill_units,
        &recipe.opts,
        t,
        options,
    )?;
    let scale =
        visual
            .palette
            .scale_with_policy(visual.levels.clone(), visual.extend, visual.mask_policy);
    let mut request = MapRenderRequest::new(fill, scale);
    request.colorbar_label = visual.colorbar_label.map(str::to_string);
    request.title = Some(recipe.title_template.to_string());
    request.subtitle_left = wrf_time_subtitle(file, t);
    request.subtitle_center = Some(visual.provenance_label.to_string());
    request.subtitle_right = Some(wrf_source_subtitle(file));
    request.product_metadata = Some(product_request_metadata(product, &recipe, &visual));
    request.overlay_legends = render_overlay_legends(&visual.overlay_legends);
    let (width, height) = product_render_size();
    request.width = width;
    request.height = height;
    let frame_policy = requested_frame_policy(visual.frame_policy, options);
    request.domain_frame = domain_frame_for_policy(frame_policy);
    apply_operational_visual_controls(&mut request, product, &visual);
    apply_projected_map(file, t, &mut request, frame_policy, options)?;

    for overlay in &visual.overlays {
        match overlay {
            ProductOverlayRecipe::UhTrackSwath(overlay) => {
                let uh = build_recipe_field(
                    file,
                    overlay.source_var,
                    overlay.units,
                    &ComputeOptsPatch::default(),
                    t,
                    options,
                )?;
                let uh_track = build_uh_track_overlay_field(&uh, overlay)?;
                apply_reflectivity_uh_rgba(&visual, overlay, &uh_track, &mut request)?;
                request = request.with_contour_field(
                    &uh_track,
                    overlay
                        .threshold_bins
                        .iter()
                        .map(|value| *value as f64)
                        .collect(),
                    uh_track_outline_style(overlay),
                )?;
            }
        }
    }

    for contour in visual.contour_overlays {
        let field =
            build_recipe_field(file, contour.var, contour.units, &contour.opts, t, options)?;
        request = request.with_contour_field(
            &field,
            contour
                .levels
                .into_iter()
                .map(|value| value as f64)
                .collect(),
            ContourStyle {
                color: contour.color,
                width: contour.width_px,
                halo_color: contour.halo_color,
                halo_width: contour.halo_width_px,
                major_every: contour.major_every,
                major_width: contour.major_width_px,
                label_every: contour.label_every,
                labels: contour.labels,
                show_extrema: contour.show_extrema,
            },
        )?;
    }

    if let Some(barbs) = visual.barb_overlay {
        let u = build_recipe_field(
            file,
            barbs.u_var,
            barbs.units,
            &ComputeOptsPatch::default(),
            t,
            options,
        )?;
        let v = build_recipe_field(
            file,
            barbs.v_var,
            barbs.units,
            &ComputeOptsPatch::default(),
            t,
            options,
        )?;
        request = request.with_wind_barbs(&u, &v, operational_wind_barb_style(&barbs))?;
    }

    Ok(request)
}

fn operational_wind_barb_style(barbs: &WindBarbRecipe) -> WindBarbStyle {
    WindBarbStyle {
        stride_x: OPERATIONAL_BARB_GRID_STRIDE,
        stride_y: OPERATIONAL_BARB_GRID_STRIDE,
        spacing_px: barbs.spacing_px,
        halo_color: barbs.halo_color,
        halo_width: barbs.halo_width_px,
        color: barbs.color,
        width: barbs.width_px,
        length_px: barbs.length_px,
    }
}

fn apply_operational_visual_controls(
    request: &mut MapRenderRequest,
    product: WrfProduct,
    visual: &ProductVisualRecipe,
) {
    request.colorbar_orientation = operational_product_colorbar_orientation();
    request.cbar_tick_step = product_tick_step(product);
    request.cbar_ticks = visual.legend_ticks.clone();
    request.visual_mode = product.visual_mode();
    request.raster_sample_mode = operational_product_raster_sample_mode();
    request.render_density = operational_product_render_density();
    request.legend = LegendControls {
        density: operational_product_legend_density(),
        mode: operational_product_legend_mode(),
        levels: visual.legend_levels.clone(),
    };
}

#[cfg(test)]
fn request_uses_operational_visual_controls(
    request: &MapRenderRequest,
    product: WrfProduct,
    visual: &ProductVisualRecipe,
) -> bool {
    request.colorbar_orientation == operational_product_colorbar_orientation()
        && request.cbar_tick_step == product_tick_step(product)
        && request.cbar_ticks == visual.legend_ticks
        && request.visual_mode == product.visual_mode()
        && request.raster_sample_mode == operational_product_raster_sample_mode()
        && request.render_density == operational_product_render_density()
        && request.legend.density == operational_product_legend_density()
        && request.legend.mode == operational_product_legend_mode()
        && request.legend.levels == visual.legend_levels
}

fn model_data_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        ..DomainFrame::model_data_default()
    }
}

fn domain_frame_for_policy(policy: ProductFramePolicy) -> Option<DomainFrame> {
    let mut frame = model_data_domain_frame();
    match policy {
        ProductFramePolicy::FullDomain => Some(frame),
        ProductFramePolicy::FiniteData => {
            frame.source = DomainFrameSource::RasterAlpha;
            frame.legend_follows_frame = false;
            frame.chrome_follows_frame = false;
            Some(frame)
        }
        ProductFramePolicy::FiniteDataWithOverlays => {
            frame.source = DomainFrameSource::RasterAlpha;
            frame.clear_outside = false;
            frame.legend_follows_frame = false;
            frame.chrome_follows_frame = false;
            Some(frame)
        }
        ProductFramePolicy::StormCentered => {
            frame.clear_outside = false;
            frame.inset_px = 10;
            Some(frame)
        }
        ProductFramePolicy::GeographicCrop => Some(frame),
    }
}

fn requested_frame_policy(
    default_policy: ProductFramePolicy,
    options: &ProductRenderOptions,
) -> ProductFramePolicy {
    if options.geographic_bounds.is_some() {
        ProductFramePolicy::GeographicCrop
    } else if options.storm_center.is_some() {
        ProductFramePolicy::StormCentered
    } else {
        default_policy
    }
}

fn product_frame_policy(product: WrfProduct) -> ProductFramePolicy {
    match product {
        WrfProduct::ReflectivityUh => ProductFramePolicy::FiniteDataWithOverlays,
        WrfProduct::Reflectivity | WrfProduct::Reflectivity1km | WrfProduct::UpdraftHelicity => {
            ProductFramePolicy::FiniteData
        }
        _ => ProductFramePolicy::FullDomain,
    }
}

fn product_overlay_recipes(product: WrfProduct) -> Vec<ProductOverlayRecipe> {
    match product {
        WrfProduct::ReflectivityUh => {
            vec![ProductOverlayRecipe::UhTrackSwath(UhTrackOverlayRecipe {
                source_var: "uhel_0_3km_1h_max",
                units: "m2/s2",
                threshold_bins: UH_TRACK_BINS.to_vec(),
                fill_colors: UH_TRACK_FILL_COLORS.to_vec(),
                edge_color: Color::BLACK,
                edge_width_px: 2,
                edge_halo_color: Color::WHITE,
                edge_halo_width_px: 1,
                lookback_minutes: 60,
                label: "1 h 0-3 km UH swath",
            })]
        }
        _ => Vec::new(),
    }
}

fn product_overlay_legends(overlays: &[ProductOverlayRecipe]) -> Vec<ProductOverlayLegendRecipe> {
    overlays
        .iter()
        .filter_map(|overlay| match overlay {
            ProductOverlayRecipe::UhTrackSwath(overlay) => {
                let entries = overlay
                    .threshold_bins
                    .iter()
                    .zip(overlay.fill_colors.iter())
                    .map(|(&threshold, &fill_color)| ProductOverlayLegendEntry {
                        label: format!(">= {}", format_threshold_value(threshold)),
                        fill_color,
                        outline_color: overlay.edge_color,
                    })
                    .collect::<Vec<_>>();
                (!entries.is_empty()).then(|| ProductOverlayLegendRecipe {
                    title: format!(
                        "{} ({})",
                        overlay.label,
                        product_display_units(overlay.units).unwrap_or(overlay.units)
                    ),
                    entries,
                })
            }
        })
        .collect()
}

fn format_threshold_value(value: f32) -> String {
    if (value.fract()).abs() <= f32::EPSILON {
        format!("{}", value as i32)
    } else {
        let formatted = format!("{value:.1}");
        formatted
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn product_source_semantics(
    product: WrfProduct,
    recipe: &ProductRecipe,
) -> Vec<ProductSourceSemantics> {
    let mut sources = vec![ProductSourceSemantics {
        role: ProductVisualSourceRole::Fill,
        var: recipe.fill_var,
        units: recipe.fill_units,
        source: product_fill_source_kind(product),
        temporal: product_fill_temporal_semantics(product),
        label: product_fill_source_label(product),
    }];

    sources.extend(
        recipe
            .contour_overlays
            .iter()
            .map(|contour| ProductSourceSemantics {
                role: ProductVisualSourceRole::ContourOverlay,
                var: contour.var,
                units: contour.units,
                source: contour_source_kind(contour.var),
                temporal: ProductTemporalSemantics::Instant,
                label: contour_source_label(contour.var),
            }),
    );

    if let Some(barbs) = &recipe.barb_overlay {
        let kind = wind_barb_source_kind(product);
        sources.push(ProductSourceSemantics {
            role: ProductVisualSourceRole::WindBarbOverlay,
            var: barbs.u_var,
            units: barbs.units,
            source: kind,
            temporal: ProductTemporalSemantics::Instant,
            label: "earth-relative wind barb U component converted to knots",
        });
        sources.push(ProductSourceSemantics {
            role: ProductVisualSourceRole::WindBarbOverlay,
            var: barbs.v_var,
            units: barbs.units,
            source: kind,
            temporal: ProductTemporalSemantics::Instant,
            label: "earth-relative wind barb V component converted to knots",
        });
    }

    match product {
        WrfProduct::ReflectivityUh => {
            sources.push(ProductSourceSemantics {
                role: ProductVisualSourceRole::UhTrackOverlay,
                var: "uhel_0_3km_1h_max",
                units: "m2/s2",
                source: ProductSourceKind::NativeOrComputed,
                temporal: ProductTemporalSemantics::HistoryMaximum {
                    lookback_minutes: 60,
                },
                label: "0-3 km UH one-hour history maximum",
            });
        }
        _ => {}
    }

    sources
}

fn product_fill_source_kind(product: WrfProduct) -> ProductSourceKind {
    match product {
        WrfProduct::Reflectivity | WrfProduct::UpdraftHelicity => {
            ProductSourceKind::NativeOrComputed
        }
        WrfProduct::Reflectivity1km | WrfProduct::ReflectivityUh => {
            ProductSourceKind::NativeOrInterpolated
        }
        WrfProduct::PrecipAccum => ProductSourceKind::Derived,
        WrfProduct::CloudTopTemp
        | WrfProduct::CloudFracLow
        | WrfProduct::CloudFracMid
        | WrfProduct::CloudFracHigh
        | WrfProduct::SlpWind10m
        | WrfProduct::SurfaceWind10m
        | WrfProduct::U10Component
        | WrfProduct::V10Component
        | WrfProduct::T2
        | WrfProduct::Td2
        | WrfProduct::Rh2
        | WrfProduct::Pblh
        | WrfProduct::Terrain => ProductSourceKind::NativeOrComputed,
        _ if product.visual_mode() == ProductVisualMode::UpperAirAnalysis => {
            ProductSourceKind::NativeOrInterpolated
        }
        _ => ProductSourceKind::Derived,
    }
}

fn product_fill_temporal_semantics(product: WrfProduct) -> ProductTemporalSemantics {
    match product {
        WrfProduct::PrecipAccum => ProductTemporalSemantics::Accumulation {
            window_minutes: None,
        },
        _ => ProductTemporalSemantics::Instant,
    }
}

fn product_fill_source_label(product: WrfProduct) -> &'static str {
    match product {
        WrfProduct::Reflectivity => "instant composite reflectivity",
        WrfProduct::Reflectivity1km | WrfProduct::ReflectivityUh => "instant 1 km AGL reflectivity",
        WrfProduct::UpdraftHelicity => "instant 0-3 km updraft helicity",
        WrfProduct::PrecipAccum => "run-total accumulated precipitation",
        WrfProduct::Pwat => "column precipitable water converted to inches",
        _ if product.visual_mode() == ProductVisualMode::UpperAirAnalysis => {
            "pressure-level interpolated fill field"
        }
        _ if product_fill_source_kind(product) == ProductSourceKind::Derived => {
            "computed diagnostic fill field"
        }
        _ => "WRF-native or computed fill field",
    }
}

fn contour_source_kind(var: &str) -> ProductSourceKind {
    if var.contains("_mb") {
        ProductSourceKind::NativeOrInterpolated
    } else {
        ProductSourceKind::NativeOrComputed
    }
}

fn contour_source_label(var: &str) -> &'static str {
    if var == "slp" {
        "MSLP contour overlay"
    } else if var.contains("_mb") {
        "pressure-level contour overlay"
    } else {
        "contour overlay source field"
    }
}

fn wind_barb_source_kind(product: WrfProduct) -> ProductSourceKind {
    if product.visual_mode() == ProductVisualMode::UpperAirAnalysis {
        ProductSourceKind::NativeOrInterpolated
    } else {
        ProductSourceKind::NativeOrComputed
    }
}

fn product_display_units(fill_units: &'static str) -> Option<&'static str> {
    match fill_units.trim() {
        "" => None,
        "knots" => Some("kt"),
        "degrees" => Some("deg"),
        units => Some(units),
    }
}

fn render_overlay_legends(legends: &[ProductOverlayLegendRecipe]) -> Vec<OverlayLegend> {
    legends
        .iter()
        .map(|legend| {
            OverlayLegend::new(
                legend.title.clone(),
                legend
                    .entries
                    .iter()
                    .map(|entry| {
                        OverlayLegendItem::new(
                            entry.label.clone(),
                            entry.fill_color,
                            entry.outline_color,
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

fn product_request_metadata(
    product: WrfProduct,
    recipe: &ProductRecipe,
    visual: &ProductVisualRecipe,
) -> ProductKeyMetadata {
    ProductKeyMetadata::new(recipe.title_template)
        .with_native_units(recipe.fill_units)
        .with_category("wrf_product")
        .with_description(operational_request_description(product, visual))
        .with_provenance(product_request_provenance(visual))
}

fn operational_request_description(product: WrfProduct, visual: &ProductVisualRecipe) -> String {
    let mut parts = vec![visual.provenance_label.to_string()];
    if !visual.source_semantics.is_empty() {
        parts.push(
            visual
                .source_semantics
                .iter()
                .map(|source| source.label)
                .collect::<Vec<_>>()
                .join("; "),
        );
    }
    if let Some(label) = visual
        .overlays
        .iter()
        .map(ProductOverlayRecipe::label)
        .next()
    {
        parts.push(label.to_string());
    }
    parts.push(format!("canonical product `{}`", product.canonical_name()));
    parts.join(" | ")
}

fn product_request_provenance(visual: &ProductVisualRecipe) -> ProductProvenance {
    let mut provenance =
        ProductProvenance::new(product_lineage(visual), ProductMaturity::Operational);
    if visual.source_semantics.len() > 1 || !visual.overlays.is_empty() {
        provenance = provenance.with_flag(ProductSemanticFlag::Composite);
    }
    if let Some(window) = product_window_spec(visual) {
        provenance = provenance.with_window(window);
    }
    provenance
}

fn product_lineage(visual: &ProductVisualRecipe) -> ProductLineage {
    if visual.source_semantics.iter().any(|source| {
        matches!(
            source.temporal,
            ProductTemporalSemantics::HistoryMaximum { .. }
        )
    }) {
        ProductLineage::Windowed
    } else if visual.source_semantics.iter().any(|source| {
        matches!(
            source.temporal,
            ProductTemporalSemantics::Accumulation { .. }
        ) || matches!(source.source, ProductSourceKind::Derived)
    }) {
        ProductLineage::Derived
    } else if visual.source_semantics.len() > 1 || !visual.overlays.is_empty() {
        ProductLineage::Bundled
    } else {
        ProductLineage::Direct
    }
}

fn product_window_spec(visual: &ProductVisualRecipe) -> Option<ProductWindowSpec> {
    visual
        .source_semantics
        .iter()
        .find_map(|source| match source.temporal {
            ProductTemporalSemantics::HistoryMaximum { lookback_minutes } => {
                Some(ProductWindowSpec {
                    process: StatisticalProcess::Maximum,
                    duration_hours: minutes_to_hours(lookback_minutes),
                })
            }
            ProductTemporalSemantics::Accumulation { window_minutes } => Some(ProductWindowSpec {
                process: StatisticalProcess::Accumulation,
                duration_hours: window_minutes.and_then(minutes_to_hours),
            }),
            ProductTemporalSemantics::Instant => None,
        })
}

fn minutes_to_hours(minutes: u32) -> Option<u16> {
    if minutes == 0 {
        None
    } else {
        Some(minutes.div_ceil(60).min(u16::MAX as u32) as u16)
    }
}

impl ProductOverlayRecipe {
    fn label(&self) -> &'static str {
        match self {
            Self::UhTrackSwath(overlay) => overlay.label,
        }
    }
}

fn upper_air_template_recipe(
    product: WrfProduct,
    recipe: &ProductRecipe,
) -> Option<UpperAirTemplateRecipe> {
    let (level_hpa, fill_role) = upper_air_template_level_and_role(product)?;
    Some(UpperAirTemplateRecipe {
        level_hpa,
        fill_role,
        fill_var: recipe.fill_var,
        fill_units: recipe.fill_units,
        height_contour_var: pressure_field_name("height", level_hpa),
        height_units: "dam",
        wind_u_var: pressure_field_name("uvmet_u", level_hpa),
        wind_v_var: pressure_field_name("uvmet_v", level_hpa),
        wind_units: "knots",
    })
}

fn upper_air_template_level_and_role(product: WrfProduct) -> Option<(u16, UpperAirFillRole)> {
    Some(match product {
        WrfProduct::Height200Wind | WrfProduct::Wind200 => (200, UpperAirFillRole::WindSpeed),
        WrfProduct::Temp200Wind => (200, UpperAirFillRole::Temperature),
        WrfProduct::Height250Wind => (250, UpperAirFillRole::WindSpeed),
        WrfProduct::Temp250Wind => (250, UpperAirFillRole::Temperature),
        WrfProduct::Wind250 => (250, UpperAirFillRole::JetSpeed),
        WrfProduct::Height300Wind => (300, UpperAirFillRole::WindSpeed),
        WrfProduct::Temp300Wind => (300, UpperAirFillRole::Temperature),
        WrfProduct::Wind300 => (300, UpperAirFillRole::JetSpeed),
        WrfProduct::Height500Wind | WrfProduct::Wind500 => (500, UpperAirFillRole::WindSpeed),
        WrfProduct::Temp500Wind => (500, UpperAirFillRole::Temperature),
        WrfProduct::Vort500Wind => (500, UpperAirFillRole::Vorticity),
        WrfProduct::Pvo500 => (500, UpperAirFillRole::PotentialVorticity),
        WrfProduct::Omega500 => (500, UpperAirFillRole::Omega),
        WrfProduct::ThetaW850 => (850, UpperAirFillRole::WetBulbPotentialTemperature),
        WrfProduct::Temp700Wind => (700, UpperAirFillRole::Temperature),
        WrfProduct::Height700Wind => (700, UpperAirFillRole::WindSpeed),
        WrfProduct::Rh700Wind => (700, UpperAirFillRole::RelativeHumidity),
        WrfProduct::Omega700Wind => (700, UpperAirFillRole::Omega),
        WrfProduct::Height850Wind | WrfProduct::Wind850 => (850, UpperAirFillRole::WindSpeed),
        WrfProduct::Temp850Wind => (850, UpperAirFillRole::Temperature),
        WrfProduct::Td850Wind => (850, UpperAirFillRole::Dewpoint),
        _ => return None,
    })
}

fn product_provenance_label(product: WrfProduct) -> &'static str {
    match product {
        WrfProduct::ReflectivityUh => {
            "Instant 1 km AGL reflectivity + native/computed 0-3 km 1h UH swath bins"
        }
        WrfProduct::Reflectivity => "Instant native/computed composite reflectivity",
        WrfProduct::Reflectivity1km => "Instant native/interpolated 1 km AGL reflectivity",
        WrfProduct::UpdraftHelicity => "Instant native UP_HELI_MAX or computed 0-3 km UH",
        WrfProduct::PrecipAccum => "Accumulated RAINC + RAINNC converted to inches",
        WrfProduct::Ecape
        | WrfProduct::SbEcape
        | WrfProduct::MlEcape
        | WrfProduct::MuEcape
        | WrfProduct::Ncape
        | WrfProduct::EcapeCape
        | WrfProduct::EcapeCin
        | WrfProduct::EcapeLfc
        | WrfProduct::EcapeEl => "Computed entraining parcel buoyancy diagnostic",
        WrfProduct::Sbcape
        | WrfProduct::Sbcin
        | WrfProduct::Mlcape
        | WrfProduct::Mlcin
        | WrfProduct::Mucape
        | WrfProduct::Mucin
        | WrfProduct::Sb3Cape
        | WrfProduct::Ml3Cape
        | WrfProduct::Mu3Cape
        | WrfProduct::Sb6Cape
        | WrfProduct::Ml6Cape
        | WrfProduct::Mu6Cape
        | WrfProduct::EffectiveCape => "Computed parcel CAPE/CIN buoyancy diagnostic",
        WrfProduct::EffectiveInflowBase | WrfProduct::EffectiveInflowTop => {
            "Computed effective inflow layer bounds"
        }
        WrfProduct::Srh01 | WrfProduct::Srh03 | WrfProduct::EffectiveSrh => {
            "Storm-motion-relative helicity diagnostic"
        }
        WrfProduct::Shear01 | WrfProduct::Shear06 | WrfProduct::Ebwd | WrfProduct::MeanWind06 => {
            "Layer wind diagnostic converted to knots"
        }
        WrfProduct::StpEffective => "Effective-layer STP from CAPE, SRH, shear, LCL, and CIN",
        WrfProduct::StpFixed => "Fixed-layer STP from CAPE, SRH, shear, LCL, and CIN",
        WrfProduct::Scp | WrfProduct::EcapeScp => {
            "Supercell composite from CAPE, SRH, and deep-layer shear"
        }
        WrfProduct::Ehi | WrfProduct::EcapeEhi => {
            "Energy-helicity index from CAPE and storm-relative helicity"
        }
        WrfProduct::Tehi => "Tornadic EHI emphasizing 0-1 km streamwise vorticity",
        WrfProduct::Tts => "Tornadic tilting and stretching composite diagnostic",
        WrfProduct::VtpMod => "Violent tornado proxy composite diagnostic",
        WrfProduct::CriticalAngle => "Storm-relative inflow critical-angle diagnostic",
        WrfProduct::Ship => {
            "Significant hail composite from CAPE, shear, lapse rate, and freezing level"
        }
        WrfProduct::Dcp => "Derecho composite from instability, shear, and downdraft potential",
        WrfProduct::Wndg => "Wind-damage composite from CAPE, lapse rate, CIN, and low-level wind",
        WrfProduct::Esp => "Enhanced stretching potential from low-level CAPE and lapse rate",
        WrfProduct::Mmp => "MCS maintenance probability composite diagnostic",
        WrfProduct::Bri => "Bulk Richardson number from CAPE and deep-layer shear",
        WrfProduct::Dcape => "Downdraft CAPE from parcel descent profile",
        WrfProduct::CloudTopTemp => "Simulated IR cloud-top brightness temperature",
        WrfProduct::CloudFracLow | WrfProduct::CloudFracMid | WrfProduct::CloudFracHigh => {
            "WRF cloud fraction layer diagnostic"
        }
        WrfProduct::SlpWind10m | WrfProduct::SurfaceWind10m => {
            "10 m wind speed with MSLP contours and wind barbs"
        }
        WrfProduct::U10Component | WrfProduct::V10Component => {
            "10 m earth-relative signed wind component converted to knots"
        }
        WrfProduct::T2 => "2 m temperature with MSLP contours and 10 m wind barbs",
        WrfProduct::Td2 => "2 m dewpoint with MSLP contours and 10 m wind barbs",
        WrfProduct::Rh2 => "2 m relative humidity diagnostic",
        WrfProduct::Pwat => "Column precipitable water diagnostic converted to inches",
        WrfProduct::Pblh => "WRF planetary boundary layer height",
        WrfProduct::Terrain => "WRF model terrain height",
        WrfProduct::Lcl
        | WrfProduct::Lfc
        | WrfProduct::El
        | WrfProduct::LclTemp
        | WrfProduct::FreezingLevel
        | WrfProduct::WetBulbZero => "Computed thermodynamic level diagnostic",
        WrfProduct::KIndex | WrfProduct::TotalTotals | WrfProduct::MeanMixr => {
            "Thermodynamic instability/moisture index"
        }
        WrfProduct::LowRh | WrfProduct::MidRh | WrfProduct::DgzRh => {
            "Layer-mean relative humidity diagnostic"
        }
        WrfProduct::ConvTemp | WrfProduct::MaxTemp => "Surface temperature forecast diagnostic",
        WrfProduct::LapseRate700500 | WrfProduct::LapseRate03 => "Layer lapse-rate diagnostic",
        WrfProduct::Fosberg | WrfProduct::Haines | WrfProduct::Hdw => {
            "Fire-weather environment diagnostic"
        }
        WrfProduct::Wind250 | WrfProduct::Wind300 => {
            "Pressure-level jet-speed analysis with height contours and wind barbs"
        }
        WrfProduct::Height200Wind
        | WrfProduct::Temp200Wind
        | WrfProduct::Wind200
        | WrfProduct::Height250Wind
        | WrfProduct::Temp250Wind
        | WrfProduct::Height300Wind
        | WrfProduct::Temp300Wind
        | WrfProduct::Height500Wind
        | WrfProduct::Temp500Wind
        | WrfProduct::Wind500
        | WrfProduct::Vort500Wind
        | WrfProduct::Pvo500
        | WrfProduct::Omega500
        | WrfProduct::ThetaW850
        | WrfProduct::Temp700Wind
        | WrfProduct::Height700Wind
        | WrfProduct::Rh700Wind
        | WrfProduct::Omega700Wind
        | WrfProduct::Height850Wind
        | WrfProduct::Temp850Wind
        | WrfProduct::Td850Wind
        | WrfProduct::Wind850 => "Pressure-level interpolation with height contours and wind barbs",
    }
}

fn apply_reflectivity_uh_rgba(
    visual: &ProductVisualRecipe,
    overlay: &UhTrackOverlayRecipe,
    uh_track: &Field2D,
    request: &mut MapRenderRequest,
) -> ProductResult<()> {
    let scale = match visual.palette.scale_with_policy(
        visual.levels.clone(),
        visual.extend,
        visual.mask_policy,
    ) {
        ColorScale::Discrete(scale) => scale,
        _ => unreachable!("product palette scales are discrete"),
    };
    let pixels = request
        .field
        .values
        .iter()
        .zip(uh_track.values.iter())
        .map(|(&refl, &uh)| reflectivity_uh_pixel(&scale, overlay, refl, uh))
        .collect();
    request.set_rgba_grid(RgbaGridField::new(request.field.grid.clone(), pixels)?);
    Ok(())
}

fn build_uh_track_overlay_field(
    uh: &Field2D,
    overlay: &UhTrackOverlayRecipe,
) -> ProductResult<Field2D> {
    Ok(Field2D::new(
        ProductKey::named(format!("{}_track", overlay.source_var)),
        uh.units.clone(),
        uh.grid.clone(),
        uh.values.clone(),
    )?)
}

fn uh_track_outline_style(overlay: &UhTrackOverlayRecipe) -> ContourStyle {
    ContourStyle {
        color: overlay.edge_color,
        width: overlay.edge_width_px,
        halo_color: overlay.edge_halo_color,
        halo_width: overlay.edge_halo_width_px,
        major_every: 1,
        major_width: overlay.edge_width_px,
        label_every: usize::MAX,
        labels: false,
        show_extrema: false,
    }
}

fn reflectivity_uh_pixel(
    scale: &DiscreteColorScale,
    overlay: &UhTrackOverlayRecipe,
    refl: f32,
    uh: f32,
) -> Color {
    let track = uh_track_color(overlay, uh);
    let refl = reflectivity_pixel(scale, refl);
    if track.a == 0 {
        refl
    } else if refl.a == 0 {
        track
    } else {
        blend_over(refl, track)
    }
}

fn reflectivity_pixel(scale: &DiscreteColorScale, refl: f32) -> Color {
    if !refl.is_finite() || refl < 5.0 {
        Color::TRANSPARENT
    } else {
        sample_product_scale(scale, refl)
    }
}

fn sample_product_scale(scale: &DiscreteColorScale, value: f32) -> Color {
    let value = value as f64;
    if !value.is_finite() {
        return Color::TRANSPARENT;
    }
    if let Some(mask) = scale.mask_below {
        if value < mask {
            return Color::TRANSPARENT;
        }
    }
    if scale.levels.len() < 2 || scale.colors.is_empty() {
        return Color::TRANSPARENT;
    }
    let lo = scale.levels[0];
    let hi = *scale.levels.last().unwrap_or(&lo);
    if value < lo {
        return if matches!(scale.extend, ExtendMode::Min | ExtendMode::Both) {
            scale.colors[0]
        } else {
            Color::TRANSPARENT
        };
    }
    if value >= hi {
        return if matches!(scale.extend, ExtendMode::Max | ExtendMode::Both) {
            *scale.colors.last().unwrap_or(&Color::TRANSPARENT)
        } else {
            Color::TRANSPARENT
        };
    }
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    let idx = (t * scale.colors.len() as f64).floor() as usize;
    scale.colors[idx.min(scale.colors.len() - 1)]
}

fn uh_track_color(overlay: &UhTrackOverlayRecipe, uh: f32) -> Color {
    if !uh.is_finite() || uh < uh_track_threshold(overlay) {
        return Color::TRANSPARENT;
    }
    uh_track_fill_color(overlay, uh)
}

fn uh_track_threshold(overlay: &UhTrackOverlayRecipe) -> f32 {
    overlay
        .threshold_bins
        .first()
        .copied()
        .unwrap_or(UH_TRACK_THRESHOLD)
}

fn uh_track_fill_color(overlay: &UhTrackOverlayRecipe, uh: f32) -> Color {
    let mut color = overlay
        .fill_colors
        .first()
        .copied()
        .unwrap_or(Color::TRANSPARENT);
    for (idx, threshold) in overlay.threshold_bins.iter().enumerate() {
        if uh >= *threshold {
            if let Some(bin_color) = overlay.fill_colors.get(idx).copied() {
                color = bin_color;
            }
        }
    }
    color
}

fn blend_over(base: Color, overlay: Color) -> Color {
    let oa = overlay.a as f32 / 255.0;
    let ba = base.a as f32 / 255.0;
    let out_a = oa + ba * (1.0 - oa);
    if out_a <= f32::EPSILON {
        return Color::TRANSPARENT;
    }
    let blend = |o: u8, b: u8| {
        (((o as f32 * oa) + (b as f32 * ba * (1.0 - oa))) / out_a)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color::rgba(
        blend(overlay.r, base.r),
        blend(overlay.g, base.g),
        blend(overlay.b, base.b),
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn product_render_size() -> (u32, u32) {
    let width = read_dimension_env("WRF_RUST_PLOT_WIDTH").unwrap_or(DEFAULT_PRODUCT_WIDTH);
    let height = read_dimension_env("WRF_RUST_PLOT_HEIGHT").unwrap_or(DEFAULT_PRODUCT_HEIGHT);
    (width, height)
}

fn read_dimension_env(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value >= 640)
}

fn product_tick_step(product: WrfProduct) -> Option<f64> {
    match product {
        WrfProduct::Ecape
        | WrfProduct::SbEcape
        | WrfProduct::MlEcape
        | WrfProduct::MuEcape
        | WrfProduct::EcapeCape
        | WrfProduct::Sbcape
        | WrfProduct::Mlcape
        | WrfProduct::Mucape
        | WrfProduct::Sb6Cape
        | WrfProduct::Ml6Cape
        | WrfProduct::Mu6Cape
        | WrfProduct::Dcape
        | WrfProduct::EffectiveCape => Some(500.0),
        WrfProduct::Sb3Cape | WrfProduct::Ml3Cape | WrfProduct::Mu3Cape => Some(50.0),
        WrfProduct::Ncape => Some(250.0),
        WrfProduct::Srh01 | WrfProduct::Srh03 | WrfProduct::EffectiveSrh => Some(100.0),
        WrfProduct::EffectiveInflowBase | WrfProduct::EffectiveInflowTop => Some(500.0),
        WrfProduct::SlpWind10m | WrfProduct::SurfaceWind10m => Some(5.0),
        WrfProduct::Shear01
        | WrfProduct::Shear06
        | WrfProduct::Ebwd
        | WrfProduct::MeanWind06
        | WrfProduct::Height200Wind
        | WrfProduct::Wind200
        | WrfProduct::Height250Wind
        | WrfProduct::Wind250
        | WrfProduct::Height300Wind
        | WrfProduct::Wind300
        | WrfProduct::Height500Wind
        | WrfProduct::Wind500
        | WrfProduct::Height700Wind
        | WrfProduct::Height850Wind
        | WrfProduct::Wind850 => Some(5.0),
        WrfProduct::Temp200Wind
        | WrfProduct::Temp250Wind
        | WrfProduct::Temp300Wind
        | WrfProduct::Temp500Wind
        | WrfProduct::ThetaW850
        | WrfProduct::Temp700Wind
        | WrfProduct::Temp850Wind
        | WrfProduct::Td850Wind
        | WrfProduct::Reflectivity
        | WrfProduct::Reflectivity1km
        | WrfProduct::ReflectivityUh => Some(5.0),
        WrfProduct::Pvo500 => Some(1.0),
        WrfProduct::Omega500 | WrfProduct::Omega700Wind => Some(0.25),
        WrfProduct::CloudFracLow | WrfProduct::CloudFracMid | WrfProduct::CloudFracHigh => {
            Some(10.0)
        }
        WrfProduct::Td2 => Some(10.0),
        WrfProduct::T2 => Some(10.0),
        WrfProduct::LowRh | WrfProduct::MidRh | WrfProduct::DgzRh => Some(10.0),
        WrfProduct::MeanMixr => Some(2.0),
        WrfProduct::Mmp => Some(0.1),
        WrfProduct::Dcp | WrfProduct::Wndg | WrfProduct::Esp => Some(2.0),
        WrfProduct::KIndex | WrfProduct::TotalTotals => Some(5.0),
        WrfProduct::LapseRate700500 | WrfProduct::LapseRate03 => Some(1.0),
        _ => None,
    }
}

fn product_tick_values(product: WrfProduct) -> Option<Vec<f64>> {
    match product {
        WrfProduct::Ecape
        | WrfProduct::SbEcape
        | WrfProduct::MlEcape
        | WrfProduct::MuEcape
        | WrfProduct::EcapeCape => Some(vec![
            0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0,
        ]),
        WrfProduct::Sbcape | WrfProduct::Mlcape | WrfProduct::Mucape => Some(vec![
            0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
        ]),
        WrfProduct::EffectiveCape => Some(vec![
            0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
        ]),
        WrfProduct::Sb6Cape | WrfProduct::Ml6Cape | WrfProduct::Mu6Cape => Some(vec![
            0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
        ]),
        WrfProduct::Ncape => Some(vec![
            0.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0,
        ]),
        WrfProduct::Dcape => Some(vec![
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0,
        ]),
        WrfProduct::Sbcin | WrfProduct::Mlcin | WrfProduct::Mucin | WrfProduct::EcapeCin => {
            Some(vec![-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0])
        }
        WrfProduct::Sb3Cape | WrfProduct::Ml3Cape | WrfProduct::Mu3Cape => Some(vec![
            0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0,
        ]),
        WrfProduct::StpEffective | WrfProduct::StpFixed => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0,
        ]),
        WrfProduct::Scp | WrfProduct::EcapeScp => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        ]),
        WrfProduct::Dcp | WrfProduct::Wndg => {
            Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        }
        WrfProduct::CriticalAngle => {
            Some(vec![0.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0, 180.0])
        }
        WrfProduct::Ship => Some(vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]),
        WrfProduct::Esp => Some(vec![
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
        ]),
        WrfProduct::Mmp => Some(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
        WrfProduct::Ehi | WrfProduct::EcapeEhi => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            24.0,
        ]),
        WrfProduct::Tehi | WrfProduct::Tts | WrfProduct::VtpMod => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0,
        ]),
        WrfProduct::Srh01 | WrfProduct::Srh03 | WrfProduct::EffectiveSrh => Some(vec![
            0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0,
            700.0, 800.0, 900.0, 1000.0, 1250.0, 1500.0,
        ]),
        WrfProduct::CloudTopTemp => Some(vec![
            -90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, 0.0, 20.0, 40.0,
        ]),
        WrfProduct::CloudFracLow | WrfProduct::CloudFracMid | WrfProduct::CloudFracHigh => {
            Some(vec![0.0, 25.0, 50.0, 75.0, 100.0])
        }
        WrfProduct::Pwat => Some(vec![0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]),
        WrfProduct::LowRh | WrfProduct::MidRh | WrfProduct::DgzRh => Some(vec![
            0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
        ]),
        WrfProduct::Rh700Wind => Some(vec![0.0, 25.0, 50.0, 75.0, 100.0]),
        WrfProduct::Shear01 | WrfProduct::Shear06 | WrfProduct::Ebwd | WrfProduct::MeanWind06 => {
            Some(vec![
                0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0,
            ])
        }
        WrfProduct::SlpWind10m | WrfProduct::SurfaceWind10m => Some(vec![
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
        ]),
        WrfProduct::Height200Wind | WrfProduct::Wind200 => Some(vec![
            50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 200.0,
        ]),
        WrfProduct::Height250Wind => {
            Some(vec![50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 180.0])
        }
        WrfProduct::Wind250 => Some(vec![50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0]),
        WrfProduct::Height300Wind => {
            Some(vec![40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 170.0])
        }
        WrfProduct::Wind300 => Some(vec![40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]),
        WrfProduct::Height500Wind | WrfProduct::Wind500 => {
            Some(vec![20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
        }
        WrfProduct::Height700Wind => Some(vec![15.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0]),
        WrfProduct::Height850Wind | WrfProduct::Wind850 => {
            Some(vec![15.0, 25.0, 35.0, 50.0, 65.0, 80.0])
        }
        WrfProduct::T2 | WrfProduct::ConvTemp | WrfProduct::MaxTemp => Some(vec![
            -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ]),
        WrfProduct::Td2 => Some(vec![
            -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
        ]),
        WrfProduct::Rh2 => Some(vec![0.0, 25.0, 50.0, 75.0, 100.0]),
        WrfProduct::U10Component | WrfProduct::V10Component => Some(vec![
            -75.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0,
        ]),
        WrfProduct::PrecipAccum => Some(vec![
            0.01, 0.05, 0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 9.00, 15.00,
        ]),
        WrfProduct::UpdraftHelicity => {
            Some(vec![25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0])
        }
        WrfProduct::Reflectivity | WrfProduct::Reflectivity1km | WrfProduct::ReflectivityUh => {
            Some(vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0])
        }
        WrfProduct::Pblh => Some(vec![
            0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0,
        ]),
        WrfProduct::Terrain => Some(vec![
            0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
        ]),
        WrfProduct::Lcl => Some(vec![
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
        ]),
        WrfProduct::EffectiveInflowBase => Some(vec![
            0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0,
        ]),
        WrfProduct::Lfc | WrfProduct::EffectiveInflowTop | WrfProduct::EcapeLfc => Some(vec![
            0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0,
        ]),
        WrfProduct::El | WrfProduct::EcapeEl => Some(vec![
            4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0,
        ]),
        WrfProduct::FreezingLevel | WrfProduct::WetBulbZero => Some(vec![
            0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
        ]),
        WrfProduct::LapseRate700500 | WrfProduct::LapseRate03 => {
            Some(vec![4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        }
        WrfProduct::LclTemp => Some(vec![-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]),
        WrfProduct::Temp200Wind => Some(vec![
            -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0,
        ]),
        WrfProduct::Temp250Wind => Some(vec![
            -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0,
        ]),
        WrfProduct::Temp300Wind => Some(vec![
            -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0,
        ]),
        WrfProduct::Temp500Wind => Some(vec![-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 5.0]),
        WrfProduct::Temp700Wind => Some(vec![-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 25.0]),
        WrfProduct::Temp850Wind => Some(vec![-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 35.0]),
        WrfProduct::ThetaW850 => Some(vec![-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 45.0]),
        WrfProduct::Vort500Wind => Some(vec![0.00005, 0.00010, 0.00020, 0.00030, 0.00040, 0.00050]),
        WrfProduct::Pvo500 => Some(vec![-2.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        WrfProduct::Omega500 | WrfProduct::Omega700Wind => {
            Some(vec![-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0])
        }
        WrfProduct::Td850Wind => Some(vec![
            -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0,
        ]),
        WrfProduct::Fosberg => Some(vec![0.0, 25.0, 50.0, 75.0, 100.0]),
        WrfProduct::Haines => Some(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        WrfProduct::Hdw => Some(vec![0.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0]),
        WrfProduct::Bri => Some(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]),
        WrfProduct::KIndex => Some(vec![0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]),
        WrfProduct::TotalTotals => Some(vec![30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0]),
        WrfProduct::MeanMixr => Some(vec![0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0]),
    }
}

fn wrf_time_subtitle(file: &WrfFile, timeidx: usize) -> Option<String> {
    let times = file.times().ok()?;
    let valid = times.get(timeidx)?.trim();
    let valid_time = parse_wrf_timestamp(valid)?;
    let valid_label = format_wrf_time(valid)?;
    let init_time = wrf_init_time(file);
    if let Some(forecast) = init_time.and_then(|init| forecast_label(init, valid_time)) {
        Some(format!("WRF | {forecast} | {valid_label}"))
    } else {
        Some(format!("{valid_label} | WRF"))
    }
}

fn wrf_source_subtitle(file: &WrfFile) -> String {
    if let Some(init_time) = wrf_init_time(file) {
        format!("{} | source: wrfout", format_init_time(init_time))
    } else {
        "source: wrfout".to_string()
    }
}

fn format_wrf_time(value: &str) -> Option<String> {
    parse_wrf_timestamp(value).map(format_valid_time)
}

fn wrf_init_time(file: &WrfFile) -> Option<WrfTimestamp> {
    ["START_DATE", "SIMULATION_START_DATE"]
        .into_iter()
        .filter_map(|name| file.global_attr_str(name).ok())
        .filter_map(|value| parse_wrf_timestamp(value.trim()))
        .next()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WrfTimestamp {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

fn parse_wrf_timestamp(value: &str) -> Option<WrfTimestamp> {
    if value.len() < 13 {
        return None;
    }
    Some(WrfTimestamp {
        year: value.get(0..4)?.parse().ok()?,
        month: value.get(5..7)?.parse().ok()?,
        day: value.get(8..10)?.parse().ok()?,
        hour: value.get(11..13)?.parse().ok()?,
        minute: value.get(14..16).unwrap_or("00").parse().ok()?,
        second: value.get(17..19).unwrap_or("00").parse().ok()?,
    })
}

fn format_valid_time(time: WrfTimestamp) -> String {
    if time.minute == 0 {
        format!(
            "Valid {:04}-{:02}-{:02} {:02}Z",
            time.year, time.month, time.day, time.hour
        )
    } else {
        format!(
            "Valid {:04}-{:02}-{:02} {:02}:{:02}Z",
            time.year, time.month, time.day, time.hour, time.minute
        )
    }
}

fn format_init_time(time: WrfTimestamp) -> String {
    if time.minute == 0 {
        format!(
            "Init {:04}-{:02}-{:02} {:02}Z",
            time.year, time.month, time.day, time.hour
        )
    } else {
        format!(
            "Init {:04}-{:02}-{:02} {:02}:{:02}Z",
            time.year, time.month, time.day, time.hour, time.minute
        )
    }
}

fn forecast_label(init: WrfTimestamp, valid: WrfTimestamp) -> Option<String> {
    let delta_minutes = timestamp_minutes(valid) - timestamp_minutes(init);
    if delta_minutes < 0 {
        return None;
    }
    let hours = delta_minutes / 60;
    let minutes = delta_minutes % 60;
    Some(format!("F{hours:03}:{minutes:02}"))
}

fn timestamp_minutes(time: WrfTimestamp) -> i64 {
    civil_days(time.year, time.month, time.day) * 24 * 60
        + time.hour as i64 * 60
        + time.minute as i64
        + if time.second >= 30 { 1 } else { 0 }
}

fn civil_days(year: i32, month: u32, day: u32) -> i64 {
    let mut y = year as i64;
    let m = month as i64;
    let d = day as i64;
    y -= if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe
}

pub fn render_product_png<P: AsRef<Path>>(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: Option<usize>,
    path: P,
) -> ProductResult<()> {
    render_product_png_with_options(
        file,
        product,
        timeidx,
        path,
        &ProductRenderOptions::default(),
    )
}

pub fn render_product_png_with_options<P: AsRef<Path>>(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: Option<usize>,
    path: P,
    options: &ProductRenderOptions,
) -> ProductResult<()> {
    let t = timeidx.unwrap_or(0);
    let request = build_product_request_with_options(file, product, timeidx, options)?;
    let image = render_image_with_style(&request, operational_product_presentation_style())?;
    let path = path.as_ref();
    save_rgba_png_profile_with_options(&image, path, &PngWriteOptions::default())?;
    write_render_sidecar(path, &render_sidecar(file, product, t, &request, options))?;
    Ok(())
}

fn write_render_sidecar(path: &Path, sidecar: &RenderSidecar) -> ProductResult<()> {
    let sidecar_path = path.with_extension("json");
    let data = serde_json::to_vec_pretty(sidecar)?;
    fs::write(sidecar_path, data)?;
    Ok(())
}

fn render_sidecar(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: usize,
    request: &MapRenderRequest,
    options: &ProductRenderOptions,
) -> RenderSidecar {
    let spec = product_spec(product);
    RenderSidecar {
        package_name: "wrf-products".to_string(),
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        product_id: product.canonical_name().to_string(),
        input_file: file.path.display().to_string(),
        valid_time: raw_valid_time_for_index(file, timeidx),
        init_time: wrf_init_time(file).map(format_init_time),
        units: request.field.units.clone(),
        source: wrf_source_subtitle(file),
        provenance: request
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.description.clone())
            .unwrap_or_else(|| {
                "wrf-products recipe -> wrf-core getvar -> wrf-render OPERATIONAL_FAST".to_string()
            }),
        required_inputs: spec.required_inputs,
        history: spec.history,
        history_files: options
            .history_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        history_dir: options
            .history_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        frame_policy: spec.frame_policy,
    }
}

fn build_recipe_field(
    file: &WrfFile,
    var: &str,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
    options: &ProductRenderOptions,
) -> ProductResult<Field2D> {
    if var == "precip_accum" {
        return build_precip_accum_field(file, timeidx);
    }
    if var == "uhel_0_3km_1h_max" {
        return build_uhel_0_3km_1h_max_field(file, timeidx, options);
    }
    if var == NATIVE_OR_COMPUTED_UH_VAR {
        return build_native_or_computed_uhel_field(file, timeidx);
    }
    if let Some((source_var, plane_index)) = parse_multiplane_2d_var(var) {
        return build_multiplane_2d_field(
            file,
            var,
            source_var,
            plane_index,
            units,
            patch,
            timeidx,
        );
    }
    if let Some((base_var, height_m)) = parse_height_level_var(var) {
        return build_height_level_field(file, var, base_var, height_m, units, patch, timeidx);
    }
    if let Some((source_var, plane_index, level_hpa)) = parse_multiplane_pressure_level_var(var) {
        return build_multiplane_pressure_level_field(
            file,
            var,
            source_var,
            plane_index,
            level_hpa,
            units,
            patch,
            timeidx,
        );
    }
    if let Some((base_var, level_hpa)) = parse_pressure_level_var(var) {
        return build_pressure_level_field(file, var, base_var, level_hpa, units, patch, timeidx);
    }
    let mut output = getvar(file, var, Some(timeidx), &patch.apply(units))?;
    if is_mean_wind_vector_var(var) {
        output = collapse_vector_output_to_speed(output, file.ny, file.nx);
    }
    output_to_field(file, var, output, timeidx)
}

fn is_mean_wind_vector_var(var: &str) -> bool {
    matches!(var, "mean_wind" | "mean_wind_0_6km")
}

fn collapse_vector_output_to_speed(output: VarOutput, ny: usize, nx: usize) -> VarOutput {
    let nxy = ny * nx;
    if output.shape != [2, ny, nx] || output.data.len() != 2 * nxy {
        return output;
    }

    let (u, v) = output.data.split_at(nxy);
    let data = u.iter().zip(v.iter()).map(|(&u, &v)| u.hypot(v)).collect();

    VarOutput {
        data,
        shape: vec![ny, nx],
        units: output.units,
        description: format!("{} speed magnitude", output.description)
            .trim()
            .to_string(),
    }
}

fn build_multiplane_2d_field(
    file: &WrfFile,
    product_name: &str,
    source_var: &str,
    plane_index: usize,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
) -> ProductResult<Field2D> {
    let output = getvar(file, source_var, Some(timeidx), &patch.apply(units))?;
    let nxy = file.ny * file.nx;
    if output.shape.len() != 3
        || output.shape[1] != file.ny
        || output.shape[2] != file.nx
        || output.data.len() < (plane_index + 1) * nxy
    {
        return Err(ProductError::NotTwoDimensional {
            product: product_name.to_string(),
            shape: output.shape,
        });
    }

    let start = plane_index * nxy;
    let end = start + nxy;
    output_to_field(
        file,
        product_name,
        VarOutput {
            data: output.data[start..end].to_vec(),
            shape: vec![file.ny, file.nx],
            units: output.units,
            description: format!("{} plane {}", output.description, plane_index),
        },
        timeidx,
    )
}

fn build_pressure_level_field(
    file: &WrfFile,
    product_name: &str,
    base_var: &str,
    level_hpa: f64,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
) -> ProductResult<Field2D> {
    let pressure = getvar(file, "pressure", Some(timeidx), &ComputeOpts::default())?;
    require_3d("pressure", &pressure, file)?;

    let field = getvar(file, base_var, Some(timeidx), &patch.apply(units))?;
    require_3d(product_name, &field, file)?;

    let data = interpolate_to_pressure_level(
        &pressure.data,
        &field.data,
        file.nz,
        file.ny,
        file.nx,
        level_hpa,
    );
    if data.iter().all(|value| !value.is_finite()) {
        return Err(ProductError::EmptyPressureLevel {
            product: product_name.to_string(),
            level_hpa,
        });
    }

    output_to_field(
        file,
        product_name,
        VarOutput {
            data,
            shape: vec![file.ny, file.nx],
            units: field.units,
            description: format!("{base_var} interpolated to {level_hpa:.0} hPa"),
        },
        timeidx,
    )
}

fn build_multiplane_pressure_level_field(
    file: &WrfFile,
    product_name: &str,
    source_var: &str,
    plane_index: usize,
    level_hpa: f64,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
) -> ProductResult<Field2D> {
    let pressure = getvar(file, "pressure", Some(timeidx), &ComputeOpts::default())?;
    require_3d("pressure", &pressure, file)?;

    let field = getvar(file, source_var, Some(timeidx), &patch.apply(units))?;
    let nxy = file.ny * file.nx;
    let nxyz = file.nz * nxy;
    if field.shape != [2, file.nz, file.ny, file.nx] || field.data.len() < (plane_index + 1) * nxyz
    {
        return Err(ProductError::NotThreeDimensional {
            product: product_name.to_string(),
            shape: field.shape,
        });
    }

    let start = plane_index * nxyz;
    let end = start + nxyz;
    let data = interpolate_to_pressure_level(
        &pressure.data,
        &field.data[start..end],
        file.nz,
        file.ny,
        file.nx,
        level_hpa,
    );
    if data.iter().all(|value| !value.is_finite()) {
        return Err(ProductError::EmptyPressureLevel {
            product: product_name.to_string(),
            level_hpa,
        });
    }

    output_to_field(
        file,
        product_name,
        VarOutput {
            data,
            shape: vec![file.ny, file.nx],
            units: field.units,
            description: format!(
                "{} plane {} interpolated to {level_hpa:.0} hPa",
                field.description, plane_index
            ),
        },
        timeidx,
    )
}

fn build_height_level_field(
    file: &WrfFile,
    product_name: &str,
    base_var: &str,
    height_m: f64,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
) -> ProductResult<Field2D> {
    let height_agl = getvar(file, "height_agl", Some(timeidx), &ComputeOpts::default())?;
    require_3d("height_agl", &height_agl, file)?;

    let field = getvar(file, base_var, Some(timeidx), &patch.apply(units))?;
    require_3d(product_name, &field, file)?;

    let data = interp_to_height_level(
        &field.data,
        &height_agl.data,
        file.nx,
        file.ny,
        file.nz,
        height_m,
    );
    if data.iter().all(|value| !value.is_finite()) {
        return Err(ProductError::Projection(format!(
            "height level {height_m:.0} m AGL produced no valid samples for `{product_name}`"
        )));
    }

    output_to_field(
        file,
        product_name,
        VarOutput {
            data,
            shape: vec![file.ny, file.nx],
            units: field.units,
            description: format!("{base_var} interpolated to {height_m:.0} m AGL"),
        },
        timeidx,
    )
}

fn require_3d(product: &str, output: &VarOutput, file: &WrfFile) -> ProductResult<()> {
    if output.shape != [file.nz, file.ny, file.nx] {
        return Err(ProductError::NotThreeDimensional {
            product: product.to_string(),
            shape: output.shape.clone(),
        });
    }
    Ok(())
}

fn interpolate_to_pressure_level(
    pressure_hpa: &[f64],
    values: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    level_hpa: f64,
) -> Vec<f64> {
    let nxy = ny * nx;
    let target_ln = level_hpa.ln();
    let mut out = vec![f64::NAN; nxy];

    for ij in 0..nxy {
        for k in 0..nz.saturating_sub(1) {
            let idx0 = k * nxy + ij;
            let idx1 = (k + 1) * nxy + ij;
            let p0 = pressure_hpa[idx0];
            let p1 = pressure_hpa[idx1];
            let v0 = values[idx0];
            let v1 = values[idx1];
            if !p0.is_finite()
                || !p1.is_finite()
                || !v0.is_finite()
                || !v1.is_finite()
                || p0 <= 0.0
                || p1 <= 0.0
            {
                continue;
            }

            let brackets =
                (p0 >= level_hpa && p1 <= level_hpa) || (p0 <= level_hpa && p1 >= level_hpa);
            if brackets {
                let denom = p1.ln() - p0.ln();
                let weight = if denom.abs() < f64::EPSILON {
                    0.0
                } else {
                    (target_ln - p0.ln()) / denom
                };
                out[ij] = v0 + (v1 - v0) * weight.clamp(0.0, 1.0);
                break;
            }
        }
    }

    out
}

fn build_precip_accum_field(file: &WrfFile, timeidx: usize) -> ProductResult<Field2D> {
    let rainc = getvar(file, "RAINC", Some(timeidx), &ComputeOpts::default())?;
    let rainnc = getvar(file, "RAINNC", Some(timeidx), &ComputeOpts::default())?;
    if rainc.shape != [file.ny, file.nx] {
        return Err(ProductError::NotTwoDimensional {
            product: "RAINC".to_string(),
            shape: rainc.shape,
        });
    }
    if rainnc.shape != [file.ny, file.nx] {
        return Err(ProductError::NotTwoDimensional {
            product: "RAINNC".to_string(),
            shape: rainnc.shape,
        });
    }
    let data = rainc
        .data
        .into_iter()
        .zip(rainnc.data)
        .map(|(convective, grid_scale)| (convective + grid_scale) / 25.4)
        .collect();
    output_to_field(
        file,
        "precip_accum",
        VarOutput {
            data,
            shape: vec![file.ny, file.nx],
            units: "in".to_string(),
            description: "Accumulated precipitation in inches ((RAINC + RAINNC) / 25.4)"
                .to_string(),
        },
        timeidx,
    )
}

fn build_uhel_0_3km_1h_max_field(
    file: &WrfFile,
    timeidx: usize,
    options: &ProductRenderOptions,
) -> ProductResult<Field2D> {
    let mut max_values: Option<Vec<f64>> = None;

    for idx in one_hour_window_indices(file, timeidx) {
        accumulate_uhel_max(&mut max_values, file, idx)?;
    }

    if let Some(current) = valid_time_for_index(file, timeidx) {
        for path in &options.history_files {
            if normalize_path_for_compare(path) == normalize_path_for_compare(&file.path) {
                continue;
            }
            let sibling = WrfFile::open(path)?;
            ensure_history_grid_matches(file, &sibling, path)?;
            for idx in one_hour_window_indices_for_valid_time(&sibling, current) {
                accumulate_uhel_max(&mut max_values, &sibling, idx)?;
            }
        }
    }

    if let (Some(history_dir), Some(current)) = (
        options.history_dir.as_deref(),
        valid_time_for_index(file, timeidx),
    ) {
        for (path, indices) in history_dir_one_hour_window_paths(file, current, history_dir)? {
            let Ok(sibling) = WrfFile::open(&path) else {
                continue;
            };
            if sibling.nx != file.nx || sibling.ny != file.ny || sibling.nz != file.nz {
                continue;
            }
            for idx in indices {
                accumulate_uhel_max(&mut max_values, &sibling, idx)?;
            }
        }
    }

    output_to_field(
        file,
        "uhel_0_3km_1h_max",
        VarOutput {
            data: max_values.unwrap_or_else(|| vec![f64::NAN; file.ny * file.nx]),
            shape: vec![file.ny, file.nx],
            units: "m2/s2".to_string(),
            description: "One-hour max native or computed updraft helicity".to_string(),
        },
        timeidx,
    )
}

fn build_native_or_computed_uhel_field(file: &WrfFile, timeidx: usize) -> ProductResult<Field2D> {
    let output = native_or_computed_uhel_output(file, timeidx)?;
    output_to_field(file, "updraft_helicity", output, timeidx)
}

fn accumulate_uhel_max(
    max_values: &mut Option<Vec<f64>>,
    file: &WrfFile,
    timeidx: usize,
) -> ProductResult<()> {
    let output = native_or_computed_uhel_output(file, timeidx)?;
    if output.shape != [file.ny, file.nx] {
        return Err(ProductError::NotTwoDimensional {
            product: "uhel_0_3km_1h_max".to_string(),
            shape: output.shape,
        });
    }

    match max_values {
        Some(max_values) => {
            for (max_value, value) in max_values.iter_mut().zip(output.data) {
                if value.is_finite() && (!max_value.is_finite() || value > *max_value) {
                    *max_value = value;
                }
            }
        }
        None => *max_values = Some(output.data),
    }
    Ok(())
}

fn native_or_computed_uhel_output(file: &WrfFile, timeidx: usize) -> ProductResult<VarOutput> {
    if file.has_var(NATIVE_UPDRAFT_HELICITY_MAX_VAR) {
        return getvar(
            file,
            NATIVE_UPDRAFT_HELICITY_MAX_VAR,
            Some(timeidx),
            &ComputeOpts {
                units: Some("m2/s2".to_string()),
                ..Default::default()
            },
        )
        .map(|mut output| {
            output.description = "Native WRF maximum updraft helicity (UP_HELI_MAX)".to_string();
            output
        })
        .map_err(Into::into);
    }

    getvar(
        file,
        "uhel",
        Some(timeidx),
        &ComputeOpts {
            units: Some("m2/s2".to_string()),
            bottom_m: Some(0.0),
            top_m: Some(3000.0),
            ..Default::default()
        },
    )
    .map(|mut output| {
        output.description =
            "Computed 0-3 km updraft helicity fallback; native UP_HELI_MAX not present".to_string();
        output
    })
    .map_err(Into::into)
}

fn ensure_history_grid_matches(
    file: &WrfFile,
    history: &WrfFile,
    path: &Path,
) -> ProductResult<()> {
    if history.nx != file.nx || history.ny != file.ny || history.nz != file.nz {
        return Err(ProductError::Projection(format!(
            "history file `{}` grid does not match current file",
            path.display()
        )));
    }
    Ok(())
}

fn history_dir_one_hour_window_paths(
    file: &WrfFile,
    current: WrfTimestamp,
    history_dir: &Path,
) -> ProductResult<Vec<(PathBuf, Vec<usize>)>> {
    let current_minutes = timestamp_minutes(current);
    let current_path = normalize_path_for_compare(&file.path);
    let current_prefix = wrfout_filename_prefix(&file.path);
    let entries = fs::read_dir(history_dir).map_err(|err| ProductError::HistoryDirRead {
        path: history_dir.to_path_buf(),
        message: err.to_string(),
    })?;

    let mut paths = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        if normalize_path_for_compare(&path) == current_path {
            continue;
        }
        if wrfout_filename_prefix(&path) != current_prefix {
            continue;
        }
        let Ok(candidate) = WrfFile::open(&path) else {
            continue;
        };
        let indices: Vec<usize> = candidate
            .times()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                let time = parse_wrf_timestamp(value.trim())?;
                let minutes = timestamp_minutes(time);
                (minutes <= current_minutes && minutes >= current_minutes - 60).then_some(idx)
            })
            .collect();
        if !indices.is_empty() {
            paths.push((path, indices));
        }
    }
    paths.sort_by(|(left, _), (right, _)| left.cmp(right));
    Ok(paths)
}

fn one_hour_window_indices(file: &WrfFile, timeidx: usize) -> Vec<usize> {
    let end = timeidx.min(file.nt.saturating_sub(1));
    let fallback = || vec![end];
    let Some(current) = valid_time_for_index(file, end) else {
        return fallback();
    };
    let Ok(times) = file.times() else {
        return fallback();
    };
    let current_minutes = timestamp_minutes(current);
    let mut indices = Vec::new();
    for idx in 0..=end {
        let Some(time) = times.get(idx).and_then(|value| parse_wrf_timestamp(value)) else {
            continue;
        };
        let minutes = timestamp_minutes(time);
        if minutes <= current_minutes && minutes >= current_minutes - 60 {
            indices.push(idx);
        }
    }
    if indices.is_empty() {
        fallback()
    } else {
        indices
    }
}

fn one_hour_window_indices_for_valid_time(file: &WrfFile, current: WrfTimestamp) -> Vec<usize> {
    let Ok(times) = file.times() else {
        return Vec::new();
    };
    let current_minutes = timestamp_minutes(current);
    times
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let time = parse_wrf_timestamp(value.trim())?;
            let minutes = timestamp_minutes(time);
            (minutes <= current_minutes && minutes >= current_minutes - 60).then_some(idx)
        })
        .collect()
}

fn valid_time_for_index(file: &WrfFile, timeidx: usize) -> Option<WrfTimestamp> {
    file.times()
        .ok()?
        .get(timeidx)
        .and_then(|value| parse_wrf_timestamp(value.trim()))
}

fn raw_valid_time_for_index(file: &WrfFile, timeidx: usize) -> Option<String> {
    file.times()
        .ok()?
        .get(timeidx)
        .map(|value| value.trim().to_string())
}

fn normalize_path_for_compare(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn wrfout_filename_prefix(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    wrf_timestamp_offset(name).map(|offset| name[..offset].to_string())
}

fn wrf_timestamp_offset(name: &str) -> Option<usize> {
    (0..name.len()).find(|&offset| name.get(offset..).and_then(parse_wrf_timestamp).is_some())
}

fn output_to_field(
    file: &WrfFile,
    name: &str,
    output: VarOutput,
    timeidx: usize,
) -> ProductResult<Field2D> {
    if output.shape.len() != 2 || output.shape[0] != file.ny || output.shape[1] != file.nx {
        return Err(ProductError::NotTwoDimensional {
            product: name.to_string(),
            shape: output.shape,
        });
    }
    let grid = render_grid(file, timeidx)?;
    let values = output.data.into_iter().map(|value| value as f32).collect();
    Ok(Field2D::new(
        ProductKey::named(name),
        output.units,
        grid,
        values,
    )?)
}

fn render_grid(file: &WrfFile, timeidx: usize) -> ProductResult<LatLonGrid> {
    let shape = GridShape::new(file.nx, file.ny)?;
    let lat = file
        .xlat(timeidx)?
        .iter()
        .map(|value| *value as f32)
        .collect();
    let lon = file
        .xlong(timeidx)?
        .iter()
        .map(|value| *value as f32)
        .collect();
    Ok(LatLonGrid::new(shape, lat, lon)?)
}

fn apply_projected_map(
    file: &WrfFile,
    timeidx: usize,
    request: &mut MapRenderRequest,
    frame_policy: ProductFramePolicy,
    render_options: &ProductRenderOptions,
) -> ProductResult<()> {
    let grid = render_grid(file, timeidx)?;
    let bounds = latlon_bounds(&grid).ok_or_else(|| {
        ProductError::Projection("WRF file has no finite XLAT/XLONG points".to_string())
    })?;
    let target_ratio =
        map_frame_aspect_ratio_for_mode_with_domain_frame_style_and_colorbar_orientation(
            request.visual_mode,
            request.width,
            request.height,
            request.colorbar,
            request.title.is_some(),
            request.domain_frame.is_some(),
            operational_product_presentation_style(),
            request.colorbar_orientation,
        );
    let mut options =
        projected_map_options_for_frame(bounds, target_ratio, frame_policy, render_options);
    if let Ok(projection) = WrfProjection::from_file(file) {
        options = options.with_projection(convert_projection(projection));
    }
    let projected = build_projected_map_with_options(&grid.lat_deg, &grid.lon_deg, &options)
        .map_err(|err| ProductError::Projection(err.to_string()))?;
    request.apply_projected_map(&projected);
    Ok(())
}

fn projected_map_options_for_frame(
    full_bounds: (f64, f64, f64, f64),
    target_ratio: f64,
    frame_policy: ProductFramePolicy,
    render_options: &ProductRenderOptions,
) -> ProjectedMapBuildOptions {
    let crop_bounds = selected_frame_bounds(frame_policy, render_options);
    let mut options = if let Some(bounds) = crop_bounds {
        ProjectedMapBuildOptions::from_bounds(geographic_bounds_tuple(bounds), target_ratio)
    } else {
        ProjectedMapBuildOptions::full_domain(target_ratio)
    };
    let detail_bounds = crop_bounds
        .map(geographic_bounds_tuple)
        .unwrap_or(full_bounds);
    options = options.with_basemap_detail(basemap_detail_for_bounds(detail_bounds));
    options.domain.pad_fraction = frame_padding_fraction(frame_policy);
    options
}

fn selected_frame_bounds(
    frame_policy: ProductFramePolicy,
    render_options: &ProductRenderOptions,
) -> Option<GeographicBounds> {
    render_options
        .geographic_bounds
        .or_else(|| {
            render_options
                .storm_center
                .map(StormCenteredFrame::geographic_bounds)
        })
        .filter(|_| {
            matches!(
                frame_policy,
                ProductFramePolicy::GeographicCrop | ProductFramePolicy::StormCentered
            )
        })
}

fn geographic_bounds_tuple(bounds: GeographicBounds) -> (f64, f64, f64, f64) {
    (
        bounds.west_deg,
        bounds.east_deg,
        bounds.south_deg,
        bounds.north_deg,
    )
}

fn frame_padding_fraction(frame_policy: ProductFramePolicy) -> f64 {
    match frame_policy {
        ProductFramePolicy::FiniteDataWithOverlays => 0.04,
        ProductFramePolicy::StormCentered => 0.08,
        _ => 0.02,
    }
}

fn normalize_render_option_longitude(lon_deg: f64) -> f64 {
    if !lon_deg.is_finite() {
        return 0.0;
    }
    let mut lon = lon_deg % 360.0;
    if lon <= -180.0 {
        lon += 360.0;
    } else if lon > 180.0 {
        lon -= 360.0;
    }
    lon
}

fn latlon_bounds(grid: &LatLonGrid) -> Option<(f64, f64, f64, f64)> {
    let mut west = f64::INFINITY;
    let mut east = f64::NEG_INFINITY;
    let mut south = f64::INFINITY;
    let mut north = f64::NEG_INFINITY;
    for (&lat, &lon) in grid.lat_deg.iter().zip(grid.lon_deg.iter()) {
        let lat = lat as f64;
        let lon = lon as f64;
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        west = west.min(lon);
        east = east.max(lon);
        south = south.min(lat);
        north = north.max(lat);
    }
    west.is_finite().then_some((west, east, south, north))
}

fn basemap_detail_for_bounds(bounds: (f64, f64, f64, f64)) -> BasemapDetail {
    let lon_span = longitude_span_deg(bounds.0, bounds.1);
    let lat_span = (bounds.3 - bounds.2).abs();
    if lon_span > 90.0 || lat_span > 45.0 {
        BasemapDetail::Global
    } else if lon_span > 20.0 || lat_span > 15.0 {
        BasemapDetail::Broad
    } else {
        BasemapDetail::Regional
    }
}

fn longitude_span_deg(west_deg: f64, east_deg: f64) -> f64 {
    if !west_deg.is_finite() || !east_deg.is_finite() {
        return 360.0;
    }
    let raw_span = (east_deg - west_deg).abs();
    if raw_span >= 359.0 {
        return raw_span.min(360.0);
    }

    let west = normalize_render_option_longitude(west_deg);
    let east = normalize_render_option_longitude(east_deg);
    if west <= east {
        east - west
    } else {
        east + 360.0 - west
    }
}

fn convert_projection(projection: WrfProjection) -> ProjectionSpec {
    match projection {
        WrfProjection::Lambert {
            truelat1,
            truelat2,
            stand_lon,
            ..
        } => ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: truelat1,
            standard_parallel_2_deg: truelat2,
            central_meridian_deg: stand_lon,
        },
        WrfProjection::PolarStereographic {
            truelat1,
            stand_lon,
            ..
        } => ProjectionSpec::PolarStereographic {
            true_latitude_deg: truelat1,
            central_meridian_deg: stand_lon,
            south_pole_on_projection_plane: truelat1 < 0.0,
        },
        WrfProjection::Mercator {
            truelat1, cen_lon, ..
        } => ProjectionSpec::Mercator {
            latitude_of_true_scale_deg: truelat1,
            central_meridian_deg: cen_lon,
        },
        WrfProjection::LatLon { .. } => ProjectionSpec::Geographic,
    }
}

fn ecape_parcel_recipe(parcel_type: &'static str, title_template: &'static str) -> ProductRecipe {
    ProductRecipe {
        fill_var: "ecape",
        fill_units: "J/kg",
        palette: ProductPalette::Ecape,
        levels: ProductPalette::Ecape.default_levels(),
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch {
            parcel_type: Some(parcel_type),
            ..Default::default()
        },
    }
}

fn ecape_component_recipe(
    fill_var: &'static str,
    fill_units: &'static str,
    palette: ProductPalette,
    levels: Vec<f32>,
    title_template: &'static str,
) -> ProductRecipe {
    ProductRecipe {
        fill_var,
        fill_units,
        palette,
        levels,
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch {
            parcel_type: Some("sb"),
            ..Default::default()
        },
    }
}

fn severe_recipe(
    fill_var: &'static str,
    fill_units: &'static str,
    palette: ProductPalette,
    title_template: &'static str,
) -> ProductRecipe {
    severe_recipe_with_levels(
        fill_var,
        fill_units,
        palette,
        palette.default_levels(),
        title_template,
    )
}

fn severe_recipe_with_levels(
    fill_var: &'static str,
    fill_units: &'static str,
    palette: ProductPalette,
    levels: Vec<f32>,
    title_template: &'static str,
) -> ProductRecipe {
    ProductRecipe {
        fill_var,
        fill_units,
        palette,
        levels,
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn cin_recipe(fill_var: &'static str, title_template: &'static str) -> ProductRecipe {
    ProductRecipe {
        fill_var,
        fill_units: "J/kg",
        palette: ProductPalette::Cin,
        levels: ProductPalette::Cin.default_levels(),
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn wind_layer_recipe(
    fill_var: &'static str,
    bottom_m: f64,
    top_m: f64,
    title_template: &'static str,
) -> ProductRecipe {
    let palette = if fill_var == "bulk_shear" {
        ProductPalette::BulkShear
    } else if fill_var == "mean_wind" {
        ProductPalette::LayerMeanWind
    } else {
        ProductPalette::WindSpeed
    };
    ProductRecipe {
        fill_var,
        fill_units: "knots",
        palette,
        levels: palette.default_levels(),
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch {
            bottom_m: Some(bottom_m),
            top_m: Some(top_m),
            ..Default::default()
        },
    }
}

fn height_recipe(
    fill_var: &'static str,
    fill_units: &'static str,
    levels: Vec<f32>,
    title_template: &'static str,
) -> ProductRecipe {
    ProductRecipe {
        fill_var,
        fill_units,
        palette: height_palette_for(fill_var),
        levels,
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_height_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    upper_air_analysis_recipe(
        level_hpa,
        pressure_field_name("wspd", level_hpa),
        "knots",
        ProductPalette::UpperAirWind,
        pressure_wind_levels(level_hpa),
        title_template,
    )
}

fn pressure_jet_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    upper_air_analysis_recipe(
        level_hpa,
        pressure_field_name("wspd", level_hpa),
        "knots",
        ProductPalette::JetSpeed,
        pressure_jet_levels(level_hpa),
        title_template,
    )
}

fn pressure_temp_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    upper_air_analysis_recipe(
        level_hpa,
        pressure_field_name("tc", level_hpa),
        "degC",
        ProductPalette::UpperAirTemperature,
        pressure_temperature_levels(level_hpa),
        title_template,
    )
}

fn pressure_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    pressure_height_wind_recipe(level_hpa, title_template)
}

fn upper_air_analysis_recipe(
    level_hpa: u16,
    fill_var: &'static str,
    fill_units: &'static str,
    palette: ProductPalette,
    levels: Vec<f32>,
    title_template: &'static str,
) -> ProductRecipe {
    ProductRecipe {
        fill_var,
        fill_units,
        palette,
        levels,
        contour_overlays: vec![height_contours(
            pressure_field_name("height", level_hpa),
            pressure_height_contour_levels(level_hpa),
        )],
        barb_overlay: Some(pressure_barbs(level_hpa)),
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_temperature_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        200 => range_i32(-75, -35, 1),
        250 => range_i32(-70, -30, 1),
        300 => range_i32(-65, -20, 1),
        500 => range_i32(-50, 5, 1),
        700 => range_i32(-40, 25, 1),
        850 => range_i32(-30, 35, 1),
        _ => range_i32(-40, 35, 1),
    }
}

fn pressure_dewpoint_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        850 => range_i32(-30, 25, 1),
        700 => range_i32(-45, 10, 1),
        _ => range_i32(-40, 25, 1),
    }
}

fn pressure_wind_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        200 => range_i32(50, 200, 5),
        250 => range_i32(50, 180, 5),
        300 => range_i32(40, 170, 5),
        500 => range_i32(20, 140, 5),
        700 => range_i32(15, 100, 5),
        850 => range_i32(15, 80, 5),
        _ => ProductPalette::UpperAirWind.default_levels(),
    }
}

fn pressure_jet_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        250 => range_i32(50, 190, 5),
        300 => range_i32(40, 180, 5),
        _ => pressure_wind_levels(level_hpa),
    }
}

fn pressure_height_contour_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        200 => range_i32(1080, 1280, 4),
        250 => range_i32(960, 1120, 4),
        300 => range_i32(840, 1000, 4),
        500 => range_i32(480, 600, 3),
        700 => range_i32(240, 330, 3),
        850 => range_i32(120, 180, 3),
        _ => ProductPalette::Terrain.default_levels(),
    }
}

fn height_contours(var: &'static str, levels: Vec<f32>) -> ContourRecipe {
    ContourRecipe {
        var,
        units: "dam",
        levels,
        color: Color::BLACK,
        width_px: 1,
        halo_color: Color::WHITE,
        halo_width_px: 1,
        major_every: 2,
        major_width_px: 2,
        label_every: 2,
        labels: true,
        show_extrema: false,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_barbs(level_hpa: u16) -> WindBarbRecipe {
    WindBarbRecipe {
        u_var: pressure_field_name("uvmet_u", level_hpa),
        v_var: pressure_field_name("uvmet_v", level_hpa),
        units: "knots",
        spacing_px: UPPER_AIR_BARB_SPACING_PX,
        color: Color::BLACK,
        halo_color: Color::WHITE,
        halo_width_px: OPERATIONAL_BARB_HALO_WIDTH_PX,
        width_px: OPERATIONAL_BARB_WIDTH_PX,
        length_px: OPERATIONAL_BARB_LENGTH_PX,
    }
}

fn height_palette_for(fill_var: &str) -> ProductPalette {
    match fill_var {
        "terrain" => ProductPalette::Terrain,
        "PBLH" => ProductPalette::PblHeight,
        "freezing_level" | "wet_bulb_0" => ProductPalette::FreezingLevel,
        "el" | "ecape_el" => ProductPalette::EquilibriumLevel,
        _ => ProductPalette::LclLfcHeight,
    }
}

fn pressure_field_name(field: &str, level_hpa: u16) -> &'static str {
    match (field, level_hpa) {
        ("height", 200) => "height_200mb",
        ("height", 250) => "height_250mb",
        ("height", 300) => "height_300mb",
        ("height", 500) => "height_500mb",
        ("height", 700) => "height_700mb",
        ("height", 850) => "height_850mb",
        ("tc", 200) => "tc_200mb",
        ("tc", 250) => "tc_250mb",
        ("tc", 300) => "tc_300mb",
        ("tc", 500) => "tc_500mb",
        ("tc", 700) => "tc_700mb",
        ("tc", 850) => "tc_850mb",
        ("td", 850) => "td_850mb",
        ("wspd", 200) => "wspd_200mb",
        ("wspd", 250) => "wspd_250mb",
        ("wspd", 300) => "wspd_300mb",
        ("wspd", 500) => "wspd_500mb",
        ("wspd", 700) => "wspd_700mb",
        ("wspd", 850) => "wspd_850mb",
        ("ua", 200) => "ua_200mb",
        ("ua", 250) => "ua_250mb",
        ("ua", 300) => "ua_300mb",
        ("ua", 500) => "ua_500mb",
        ("ua", 700) => "ua_700mb",
        ("ua", 850) => "ua_850mb",
        ("va", 200) => "va_200mb",
        ("va", 250) => "va_250mb",
        ("va", 300) => "va_300mb",
        ("va", 500) => "va_500mb",
        ("va", 700) => "va_700mb",
        ("va", 850) => "va_850mb",
        ("uvmet_u", 200) => "uvmet_u_200mb",
        ("uvmet_u", 250) => "uvmet_u_250mb",
        ("uvmet_u", 300) => "uvmet_u_300mb",
        ("uvmet_u", 500) => "uvmet_u_500mb",
        ("uvmet_u", 700) => "uvmet_u_700mb",
        ("uvmet_u", 850) => "uvmet_u_850mb",
        ("uvmet_v", 200) => "uvmet_v_200mb",
        ("uvmet_v", 250) => "uvmet_v_250mb",
        ("uvmet_v", 300) => "uvmet_v_300mb",
        ("uvmet_v", 500) => "uvmet_v_500mb",
        ("uvmet_v", 700) => "uvmet_v_700mb",
        ("uvmet_v", 850) => "uvmet_v_850mb",
        ("rh", 700) => "rh_700mb",
        ("avo", 500) => "avo_500mb",
        ("pvo", 500) => "pvo_500mb",
        ("omega", 500) => "omega_500mb",
        ("omega", 700) => "omega_700mb",
        ("theta_w", 850) => "theta_w_850mb",
        _ => panic!("unsupported pressure product field {field}_{level_hpa}mb"),
    }
}

fn parse_multiplane_2d_var(var: &str) -> Option<(&'static str, usize)> {
    match var {
        "uvmet10_u" => Some(("uvmet10", 0)),
        "uvmet10_v" => Some(("uvmet10", 1)),
        "effective_inflow_base" => Some(("effective_inflow", 0)),
        "effective_inflow_top" => Some(("effective_inflow", 1)),
        "cloudfrac_low" => Some(("cloudfrac", 0)),
        "cloudfrac_mid" => Some(("cloudfrac", 1)),
        "cloudfrac_high" => Some(("cloudfrac", 2)),
        _ => None,
    }
}

fn parse_multiplane_pressure_level_var(var: &str) -> Option<(&'static str, usize, f64)> {
    let (base, level) = var.rsplit_once('_')?;
    let hpa = level.strip_suffix("mb")?.parse::<f64>().ok()?;
    match base {
        "uvmet_u" => Some(("uvmet", 0, hpa)),
        "uvmet_v" => Some(("uvmet", 1, hpa)),
        _ => None,
    }
}

fn parse_pressure_level_var(var: &str) -> Option<(&str, f64)> {
    let (base, level) = var.rsplit_once('_')?;
    let hpa = level.strip_suffix("mb")?.parse::<f64>().ok()?;
    Some((base, hpa))
}

fn parse_height_level_var(var: &str) -> Option<(&str, f64)> {
    let stem = var.strip_suffix("_agl")?;
    let (base, level) = stem.rsplit_once('_')?;
    let meters = level.strip_suffix('m')?.parse::<f64>().ok()?;
    Some((base, meters))
}

fn range_i32(start: i32, end: i32, step: i32) -> Vec<f32> {
    let mut values = Vec::new();
    let mut value = start;
    while value <= end {
        values.push(value as f32);
        value += step;
    }
    values
}

fn range_step(start: f32, end: f32, step: f32) -> Vec<f32> {
    let mut values = Vec::new();
    let mut value = start;
    while value <= end + step * 0.5 {
        values.push(value);
        value += step;
    }
    values
}

fn wind_component_levels() -> Vec<f32> {
    range_i32(-75, 75, 5)
}

fn bulk_shear_levels() -> Vec<f32> {
    range_i32(0, 120, 5)
}

fn three_cape_levels() -> Vec<f32> {
    let mut levels = range_step(0.0, 300.0, 5.0);
    levels.extend(range_step(320.0, 500.0, 20.0));
    levels
}

fn six_cape_levels() -> Vec<f32> {
    range_i32(0, 6000, 100)
}

fn wind_10m_levels() -> Vec<f32> {
    range_i32(10, 70, 1)
}

fn precip_accum_levels() -> Vec<f32> {
    vec![
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.85, 0.95, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00,
        3.50, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 12.00, 15.00,
    ]
}

fn pwat_inches_levels() -> Vec<f32> {
    [
        0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30,
        1.40, 1.50, 1.60, 1.70, 1.75, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.75, 3.00,
    ]
    .to_vec()
}

fn slp_contour_levels() -> Vec<f32> {
    (960..=1040).step_by(2).map(|value| value as f32).collect()
}

fn slp_contours(levels: Vec<f32>) -> ContourRecipe {
    ContourRecipe {
        var: "slp",
        units: "hPa",
        levels,
        color: Color::BLACK,
        width_px: 1,
        halo_color: Color::WHITE,
        halo_width_px: 1,
        major_every: 2,
        major_width_px: 2,
        label_every: 2,
        labels: true,
        show_extrema: true,
        opts: ComputeOptsPatch::default(),
    }
}

fn uh_contours(var: &'static str) -> ContourRecipe {
    ContourRecipe {
        var,
        units: "m2/s2",
        levels: vec![50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        color: Color::BLACK,
        width_px: 1,
        halo_color: Color::WHITE,
        halo_width_px: 1,
        major_every: 2,
        major_width_px: 2,
        label_every: usize::MAX,
        labels: false,
        show_extrema: false,
        opts: ComputeOptsPatch::default(),
    }
}

fn scp_levels() -> Vec<f32> {
    range_i32(0, 70, 1)
}

fn surface_temperature_colors() -> Vec<Color> {
    colors_from_hex(&[
        "#3b005c", "#5a189a", "#2647a3", "#1d72b8", "#38a9db", "#95d5f0", "#d7f2ff", "#e6f5df",
        "#d8ecb7", "#b7df72", "#f2df5b", "#f6b84c", "#f08a3c", "#df5433", "#c12e2a", "#8f1d2c",
        "#64113f", "#3d0b35", "#21111f",
    ])
}

fn surface_dewpoint_colors() -> Vec<Color> {
    let mut colors = wrf_render::weather::weather_palette(WeatherPalette::Dewpoint);
    if colors.len() <= 1 {
        return colors;
    }

    colors.remove(0);
    if let Some(last) = colors.last().copied() {
        colors.push(last);
    }
    colors
}

fn wet_bulb_potential_temperature_colors() -> Vec<Color> {
    colors_from_hex(&[
        "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#d9f0d3", "#a6d96a", "#66bd63",
        "#1a9850", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026", "#762a83",
    ])
}

fn wind_component_colors() -> Vec<Color> {
    vec![
        Color::rgba(126, 0, 135, 255),
        Color::rgba(28, 38, 142, 255),
        Color::rgba(0, 119, 191, 255),
        Color::rgba(46, 218, 230, 255),
        Color::rgba(174, 232, 219, 255),
        Color::rgba(4, 224, 46, 255),
        Color::rgba(0, 96, 20, 255),
        Color::rgba(58, 78, 58, 255),
        Color::rgba(128, 128, 128, 255),
        Color::rgba(82, 58, 52, 255),
        Color::rgba(117, 27, 19, 255),
        Color::rgba(208, 25, 13, 255),
        Color::rgba(255, 37, 20, 255),
        Color::rgba(255, 116, 142, 255),
        Color::rgba(255, 219, 197, 255),
        Color::rgba(211, 118, 24, 255),
        Color::rgba(103, 45, 7, 255),
        Color::rgba(31, 9, 2, 255),
    ]
}

fn colors_from_hex(values: &[&str]) -> Vec<Color> {
    values
        .iter()
        .map(|value| {
            let trimmed = value.trim_start_matches('#');
            let red = u8::from_str_radix(&trimmed[0..2], 16).expect("valid red component");
            let green = u8::from_str_radix(&trimmed[2..4], 16).expect("valid green component");
            let blue = u8::from_str_radix(&trimmed[4..6], 16).expect("valid blue component");
            Color::rgba(red, green, blue, 255)
        })
        .collect()
}

fn normalize(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_aliases() {
        assert_eq!(parse_product("entraining_cape").unwrap(), WrfProduct::Ecape);
        assert_eq!(
            parse_product("surface-based-ecape").unwrap(),
            WrfProduct::SbEcape
        );
        assert_eq!(parse_product("ML_ECAPE").unwrap(), WrfProduct::MlEcape);
        assert_eq!(
            parse_product("most unstable ecape").unwrap(),
            WrfProduct::MuEcape
        );
        assert_eq!(parse_product("ecin").unwrap(), WrfProduct::EcapeCin);
        assert_eq!(
            parse_product("entraining_scp").unwrap(),
            WrfProduct::EcapeScp
        );
        assert_eq!(parse_product("stp").unwrap(), WrfProduct::StpEffective);
        assert_eq!(parse_product("SRH_0_3km").unwrap(), WrfProduct::Srh03);
        assert_eq!(
            parse_product("mslp-wind10m").unwrap(),
            WrfProduct::SlpWind10m
        );
        assert_eq!(
            parse_product("reflectivity-uh-combo").unwrap(),
            WrfProduct::ReflectivityUh
        );
        assert_eq!(
            parse_product("u10-component").unwrap(),
            WrfProduct::U10Component
        );
        assert_eq!(parse_product("v_10m").unwrap(), WrfProduct::V10Component);
        assert_eq!(
            parse_product("effective-inflow-cape").unwrap(),
            WrfProduct::EffectiveCape
        );
        assert_eq!(
            parse_product("low-cloud-fraction").unwrap(),
            WrfProduct::CloudFracLow
        );
        assert_eq!(
            parse_product("theta_w_850mb").unwrap(),
            WrfProduct::ThetaW850
        );
        assert_eq!(
            parse_product("700mb_omega").unwrap(),
            WrfProduct::Omega700Wind
        );
        assert_eq!(parse_product("250mb_jet").unwrap(), WrfProduct::Wind250);
        assert_eq!(parse_product("jet300").unwrap(), WrfProduct::Wind300);
        assert_eq!(
            parse_product("850mb_dewpoint").unwrap(),
            WrfProduct::Td850Wind
        );
        assert_eq!(parse_product("3cape").unwrap(), WrfProduct::Ml3Cape);
        assert_eq!(parse_product("mu6cape").unwrap(), WrfProduct::Mu6Cape);
        assert_eq!(parse_product("dcape").unwrap(), WrfProduct::Dcape);
        assert_eq!(parse_product("dcp").unwrap(), WrfProduct::Dcp);
        assert_eq!(parse_product("wndg").unwrap(), WrfProduct::Wndg);
        assert_eq!(parse_product("mmp").unwrap(), WrfProduct::Mmp);
        assert_eq!(parse_product("tot_tots").unwrap(), WrfProduct::TotalTotals);
        assert_eq!(parse_product("dgz-rh").unwrap(), WrfProduct::DgzRh);
    }

    #[test]
    fn recipes_keep_science_in_wrf_core() {
        let recipe = WrfProduct::Shear06.recipe();
        assert_eq!(recipe.fill_var, "bulk_shear");
        assert_eq!(recipe.opts.bottom_m, Some(0.0));
        assert_eq!(recipe.opts.top_m, Some(6000.0));

        let uh = WrfProduct::UpdraftHelicity.recipe();
        assert_eq!(uh.fill_var, NATIVE_OR_COMPUTED_UH_VAR);
        assert_eq!(uh.fill_units, "m2/s2");

        let ml3cape = WrfProduct::Ml3Cape.recipe();
        assert_eq!(ml3cape.fill_var, "ml3cape");
        assert_eq!(ml3cape.fill_units, "J/kg");
        assert_eq!(ml3cape.palette, ProductPalette::ThreeCape);
        assert_eq!(ml3cape.levels.first().copied(), Some(0.0));
        assert_eq!(ml3cape.levels.last().copied(), Some(500.0));

        let ml6cape = WrfProduct::Ml6Cape.recipe();
        assert_eq!(ml6cape.fill_var, "ml6cape");
        assert_eq!(ml6cape.fill_units, "J/kg");
        assert_eq!(ml6cape.palette, ProductPalette::DeepLayerCape);
        assert_eq!(ml6cape.levels.last().copied(), Some(6000.0));

        let dcape = WrfProduct::Dcape.recipe();
        assert_eq!(dcape.fill_var, "dcape");
        assert_eq!(dcape.fill_units, "J/kg");

        let dcp = WrfProduct::Dcp.recipe();
        assert_eq!(dcp.fill_var, "dcp");
        assert_eq!(dcp.levels.last().copied(), Some(10.0));

        let mmp = WrfProduct::Mmp.recipe();
        assert_eq!(mmp.fill_var, "mmp");
        assert_eq!(mmp.levels.first().copied(), Some(0.0));
        assert_eq!(mmp.levels.last().copied(), Some(1.0));
    }

    #[test]
    fn subtitles_include_valid_year() {
        assert_eq!(
            format_wrf_time("1974-04-03_22:00:00").as_deref(),
            Some("Valid 1974-04-03 22Z")
        );
        assert_eq!(
            format_wrf_time("1974-04-03_22:06:00").as_deref(),
            Some("Valid 1974-04-03 22:06Z")
        );
    }

    #[test]
    fn wrfout_filename_prefix_supports_colon_and_underscore_times() {
        assert_eq!(
            wrfout_filename_prefix(Path::new("wrfout_d02_1974-04-03_22:00:00")).as_deref(),
            Some("wrfout_d02_")
        );
        assert_eq!(
            wrfout_filename_prefix(Path::new("wrfout_d02_1974-04-03_22_00_00")).as_deref(),
            Some("wrfout_d02_")
        );
    }

    #[test]
    fn forecast_label_formats_sim_elapsed_time() {
        let init = parse_wrf_timestamp("1974-04-03_12:00:00").unwrap();
        let valid = parse_wrf_timestamp("1974-04-03_22:00:00").unwrap();
        assert_eq!(forecast_label(init, valid).as_deref(), Some("F010:00"));

        let valid = parse_wrf_timestamp("1974-04-03_12:06:00").unwrap();
        assert_eq!(forecast_label(init, valid).as_deref(), Some("F000:06"));
    }

    #[test]
    fn init_time_formats_for_subtitle() {
        let init = parse_wrf_timestamp("1974-04-03_12:00:00").unwrap();
        assert_eq!(format_init_time(init), "Init 1974-04-03 12Z");
    }

    #[test]
    fn ecape_products_name_their_parcel_type() {
        let sb = WrfProduct::SbEcape.recipe();
        assert_eq!(WrfProduct::SbEcape.canonical_name(), "sb_ecape");
        assert_eq!(sb.fill_var, "ecape");
        assert_eq!(sb.title_template, "Surface-Based ECAPE (SB)");
        assert_eq!(sb.opts.parcel_type, Some("sb"));

        let ml = WrfProduct::MlEcape.recipe();
        assert_eq!(WrfProduct::MlEcape.canonical_name(), "ml_ecape");
        assert_eq!(ml.title_template, "Mixed-Layer ECAPE (ML)");
        assert_eq!(ml.opts.parcel_type, Some("ml"));

        let mu = WrfProduct::MuEcape.recipe();
        assert_eq!(WrfProduct::MuEcape.canonical_name(), "mu_ecape");
        assert_eq!(mu.title_template, "Most-Unstable ECAPE (MU)");
        assert_eq!(mu.opts.parcel_type, Some("mu"));
    }

    #[test]
    fn wind_component_products_use_signed_component_palette() {
        let u = WrfProduct::U10Component.recipe();
        assert_eq!(WrfProduct::U10Component.canonical_name(), "u10_component");
        assert_eq!(u.fill_var, EARTH_ROTATED_U10_VAR);
        assert_eq!(u.fill_units, "knots");
        assert_eq!(u.palette, ProductPalette::WindComponent);
        assert_eq!(u.levels.first().copied(), Some(-75.0));
        assert_eq!(u.levels.last().copied(), Some(75.0));
        assert!(u.barb_overlay.is_none());
        assert!(u.title_template.contains("Earth-Relative"));

        let v = WrfProduct::V10Component.recipe();
        assert_eq!(WrfProduct::V10Component.canonical_name(), "v10_component");
        assert_eq!(v.fill_var, EARTH_ROTATED_V10_VAR);
        assert_eq!(v.fill_units, "knots");
        assert_eq!(v.palette, ProductPalette::WindComponent);
        assert!(v.title_template.contains("Earth-Relative"));
    }

    #[test]
    fn shear_products_keep_legacy_operational_wind_colortable() {
        for product in [WrfProduct::Shear01, WrfProduct::Shear06, WrfProduct::Ebwd] {
            let recipe = product.recipe();
            assert_eq!(recipe.fill_units, "knots");
            assert_eq!(recipe.palette, ProductPalette::BulkShear);
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(120.0));
            assert_eq!(
                recipe.levels,
                ProductPalette::BulkShear.default_levels(),
                "{} should use the dense severe-shear scale",
                product.canonical_name()
            );
        }
        assert_eq!(
            WrfProduct::MeanWind06.recipe().palette,
            ProductPalette::LayerMeanWind
        );
        assert_eq!(
            WrfProduct::MeanWind06.recipe().palette.colors(),
            ProductPalette::WindSpeed.colors(),
            "0-6 km mean wind should keep the legacy wind color table"
        );
        assert_eq!(
            ProductPalette::BulkShear.colors(),
            wrf_render::weather::weather_palette(WeatherPalette::Winds)
        );
    }

    #[test]
    fn upper_air_wind_products_use_pressure_level_wind_palette() {
        for product in [
            WrfProduct::Height200Wind,
            WrfProduct::Wind200,
            WrfProduct::Height250Wind,
            WrfProduct::Height300Wind,
            WrfProduct::Height500Wind,
            WrfProduct::Wind500,
            WrfProduct::Height700Wind,
            WrfProduct::Height850Wind,
            WrfProduct::Wind850,
        ] {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(product);
            assert_eq!(recipe.fill_units, "knots");
            assert_eq!(
                recipe.palette,
                ProductPalette::UpperAirWind,
                "{} should use the pressure-level wind palette",
                product.canonical_name()
            );
            assert_eq!(
                visual.palette.colors(),
                ProductPalette::WindSpeed.colors(),
                "{} should keep the legacy wind color table",
                product.canonical_name()
            );
            assert_eq!(
                visual
                    .upper_air_template
                    .as_ref()
                    .map(|template| template.fill_role),
                Some(UpperAirFillRole::WindSpeed)
            );
        }
    }

    #[test]
    fn upper_air_rh_uses_pressure_level_humidity_palette() {
        let recipe = WrfProduct::Rh700Wind.recipe();
        let visual = recipe.visual_recipe(WrfProduct::Rh700Wind);
        assert_eq!(recipe.fill_var, "rh_700mb");
        assert_eq!(recipe.fill_units, "%");
        assert_eq!(
            recipe.palette,
            ProductPalette::UpperAirRelativeHumidity,
            "700 hPa RH should use its upper-air analysis palette"
        );
        assert_eq!(recipe.levels.first().copied(), Some(0.0));
        assert_eq!(recipe.levels.last().copied(), Some(100.0));
        assert_ne!(
            visual.palette.colors(),
            ProductPalette::RelativeHumidity.colors(),
            "700 hPa RH should not borrow the broad RH color table"
        );
        assert_eq!(
            visual
                .upper_air_template
                .as_ref()
                .map(|template| template.fill_role),
            Some(UpperAirFillRole::RelativeHumidity)
        );
        assert_eq!(
            visual.legend_levels.as_deref(),
            Some(&[0.0, 25.0, 50.0, 75.0, 100.0][..])
        );
    }

    #[test]
    fn surface_dewpoint_palette_is_shifted_one_degree_down() {
        let td2 = WrfProduct::Td2.recipe();
        assert_eq!(td2.palette, ProductPalette::SurfaceDewpoint);
        assert_eq!(td2.levels.first().copied(), Some(-40.0));
        assert_eq!(td2.levels.last().copied(), Some(90.0));

        let base = wrf_render::weather::weather_palette(WeatherPalette::Dewpoint);
        let shifted = surface_dewpoint_colors();
        assert_eq!(shifted.len(), base.len());
        assert_eq!(shifted[0], base[1]);
    }

    #[test]
    fn surface_relative_humidity_uses_dedicated_surface_scale() {
        let recipe = WrfProduct::Rh2.recipe();
        let visual = recipe.visual_recipe(WrfProduct::Rh2);
        assert_eq!(recipe.fill_var, "rh2m");
        assert_eq!(recipe.fill_units, "%");
        assert_eq!(recipe.palette, ProductPalette::SurfaceRelativeHumidity);
        assert_eq!(recipe.levels.first().copied(), Some(0.0));
        assert_eq!(recipe.levels.last().copied(), Some(100.0));
        assert_eq!(visual.mask_policy, MaskPolicy::None);
        assert_eq!(
            visual.legend_levels.as_deref(),
            Some(&[0.0, 25.0, 50.0, 75.0, 100.0][..])
        );
        assert_ne!(
            visual.palette.colors(),
            ProductPalette::RelativeHumidity.colors(),
            "2 m RH should not borrow the broad RH color table"
        );
    }

    #[test]
    fn surface_temperature_uses_full_fahrenheit_table_and_overlays() {
        let t2 = WrfProduct::T2.recipe();
        assert_eq!(t2.fill_var, "T2");
        assert_eq!(t2.fill_units, "degF");
        assert_eq!(t2.palette, ProductPalette::SurfaceTemperature);
        assert_eq!(t2.levels.first().copied(), Some(-60.0));
        assert_eq!(t2.levels.last().copied(), Some(120.0));
        assert_eq!(t2.contour_overlays[0].var, "slp");
        assert!(t2.barb_overlay.is_some());
        let generic = wrf_render::weather::weather_palette(WeatherPalette::Temperature);
        let surface = surface_temperature_colors();
        assert_ne!(surface, generic);
        assert!(surface.len() > 12);
        assert!(t2.title_template.contains("degF"));
        assert!(t2.title_template.is_ascii());
    }

    #[test]
    fn surface_wind_products_use_10m_operational_scale() {
        for product in [WrfProduct::SurfaceWind10m, WrfProduct::SlpWind10m] {
            let recipe = product.recipe();
            assert_eq!(recipe.fill_var, "wspd10");
            assert_eq!(recipe.fill_units, "knots");
            assert_eq!(recipe.palette, ProductPalette::SurfaceWind);
            assert_eq!(recipe.levels.first().copied(), Some(10.0));
            assert_eq!(recipe.levels.last().copied(), Some(70.0));
            assert_eq!(
                recipe.visual_recipe(product).mask_policy,
                MaskPolicy::Below(10.0)
            );
            assert_eq!(recipe.contour_overlays[0].var, "slp");
            let barbs = recipe.barb_overlay.as_ref().expect("surface barbs");
            assert_eq!(barbs.u_var, EARTH_ROTATED_U10_VAR);
            assert_eq!(barbs.v_var, EARTH_ROTATED_V10_VAR);
            assert_eq!(barbs.units, "knots");
            assert_eq!(barbs.spacing_px, SURFACE_BARB_SPACING_PX);
            assert_eq!(barbs.halo_color, Color::WHITE);
            assert_eq!(barbs.halo_width_px, OPERATIONAL_BARB_HALO_WIDTH_PX);
            assert_eq!(barbs.width_px, OPERATIONAL_BARB_WIDTH_PX);
            assert_eq!(barbs.length_px, OPERATIONAL_BARB_LENGTH_PX);
        }
        assert_eq!(
            ProductPalette::SurfaceWind.colors(),
            wrf_render::weather::weather_palette(WeatherPalette::Winds)
        );
    }

    #[test]
    fn wind_barb_and_wind_fill_recipes_are_knot_consistent() {
        for product in default_product_suite() {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(*product);
            if recipe.fill_units == "knots" {
                assert!(
                    recipe.title_template.contains("kt"),
                    "{} fills wind in knots but title does not label kt: {}",
                    product.canonical_name(),
                    recipe.title_template
                );
                assert_eq!(
                    visual.colorbar_label,
                    Some("kt"),
                    "{} should render wind colorbars with operational kt labels while requesting knots",
                    product.canonical_name()
                );
            }
            if let Some(barbs) = recipe.barb_overlay.as_ref() {
                assert_eq!(
                    barbs.units,
                    "knots",
                    "{} barb components should be requested in knots",
                    product.canonical_name()
                );
                let style = operational_wind_barb_style(barbs);
                assert_eq!(
                    (style.stride_x, style.stride_y),
                    (OPERATIONAL_BARB_GRID_STRIDE, OPERATIONAL_BARB_GRID_STRIDE),
                    "{} wind barbs should consider every grid anchor before pixel-space decimation",
                    product.canonical_name()
                );
                assert_eq!(style.spacing_px, barbs.spacing_px);
                assert_eq!(style.halo_width, barbs.halo_width_px);
                assert!(
                    recipe.title_template.contains("kt"),
                    "{} has wind barbs but title does not label kt: {}",
                    product.canonical_name(),
                    recipe.title_template
                );
                assert!(
                    barbs.spacing_px > 0.0
                        && barbs.halo_color.a > 0
                        && barbs.halo_width_px > 0
                        && barbs.width_px > 0
                        && barbs.length_px > 0.0,
                    "{} wind barbs should carry complete visible styling",
                    product.canonical_name()
                );
            }
        }
    }

    #[test]
    fn wind_barb_recipes_use_earth_relative_component_sources() {
        assert_eq!(
            parse_multiplane_2d_var(EARTH_ROTATED_U10_VAR),
            Some(("uvmet10", 0))
        );
        assert_eq!(
            parse_multiplane_2d_var(EARTH_ROTATED_V10_VAR),
            Some(("uvmet10", 1))
        );
        assert_eq!(
            parse_multiplane_pressure_level_var("uvmet_u_500mb"),
            Some(("uvmet", 0, 500.0))
        );
        assert_eq!(
            parse_multiplane_pressure_level_var("uvmet_v_700mb"),
            Some(("uvmet", 1, 700.0))
        );

        for product in default_product_suite() {
            let Some(barbs) = product.recipe().barb_overlay else {
                continue;
            };

            assert!(
                matches!(
                    barbs.u_var,
                    EARTH_ROTATED_U10_VAR
                        | "uvmet_u_200mb"
                        | "uvmet_u_250mb"
                        | "uvmet_u_300mb"
                        | "uvmet_u_500mb"
                        | "uvmet_u_700mb"
                        | "uvmet_u_850mb"
                ),
                "{} should use earth-relative U wind for barbs, got {}",
                product.canonical_name(),
                barbs.u_var
            );
            assert!(
                matches!(
                    barbs.v_var,
                    EARTH_ROTATED_V10_VAR
                        | "uvmet_v_200mb"
                        | "uvmet_v_250mb"
                        | "uvmet_v_300mb"
                        | "uvmet_v_500mb"
                        | "uvmet_v_700mb"
                        | "uvmet_v_850mb"
                ),
                "{} should use earth-relative V wind for barbs, got {}",
                product.canonical_name(),
                barbs.v_var
            );
            assert_eq!(barbs.units, "knots");
            assert!(
                !matches!(barbs.u_var, "U10" | "V10")
                    && !barbs.u_var.starts_with("ua_")
                    && !barbs.u_var.starts_with("va_")
                    && !matches!(barbs.v_var, "U10" | "V10")
                    && !barbs.v_var.starts_with("ua_")
                    && !barbs.v_var.starts_with("va_"),
                "{} should not request grid-axis wind-barb components",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn visual_recipes_keep_display_units_separate_from_compute_units() {
        let wind = WrfProduct::Wind250.recipe();
        let wind_visual = wind.visual_recipe(WrfProduct::Wind250);
        assert_eq!(wind.fill_units, "knots");
        assert_eq!(wind_visual.colorbar_label, Some("kt"));
        assert!(wind_visual
            .source_semantics
            .iter()
            .any(|source| source.units == "knots"));

        let angle = WrfProduct::CriticalAngle.recipe();
        assert_eq!(angle.fill_units, "degrees");
        assert_eq!(
            angle
                .visual_recipe(WrfProduct::CriticalAngle)
                .colorbar_label,
            Some("deg")
        );

        let stp = WrfProduct::StpEffective.recipe();
        assert_eq!(
            stp.visual_recipe(WrfProduct::StpEffective).colorbar_label,
            None
        );
    }

    #[test]
    fn stp_products_use_layer_specific_palettes() {
        let effective = WrfProduct::StpEffective.recipe();
        assert_eq!(effective.palette, ProductPalette::EffectiveStp);
        assert_eq!(
            effective.levels,
            ProductPalette::EffectiveStp.default_levels()
        );

        let fixed = WrfProduct::StpFixed.recipe();
        assert_eq!(fixed.palette, ProductPalette::FixedLayerStp);
        assert_eq!(fixed.levels, ProductPalette::FixedLayerStp.default_levels());

        for product in [WrfProduct::StpEffective, WrfProduct::StpFixed] {
            let visual = product.recipe().visual_recipe(product);
            assert_eq!(visual.mask_policy, MaskPolicy::Below(0.01));
            assert_eq!(
                visual.legend_levels.as_deref(),
                Some(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0][..])
            );
            assert_eq!(visual.legend_ticks, visual.legend_levels);
            assert_eq!(
                visual.palette.colors(),
                ProductPalette::Stp.colors(),
                "{} should keep the legacy meteorologist-tuned STP color table",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn upper_air_jet_templates_are_explicit() {
        let jet250 = WrfProduct::Wind250.recipe();
        assert_eq!(jet250.fill_var, "wspd_250mb");
        assert_eq!(jet250.fill_units, "knots");
        assert_eq!(jet250.palette, ProductPalette::JetSpeed);
        assert!(jet250.title_template.contains("Jet Streak"));
        assert_eq!(jet250.levels.first().copied(), Some(50.0));
        assert_eq!(jet250.levels.last().copied(), Some(190.0));
        assert_eq!(jet250.contour_overlays[0].var, "height_250mb");
        assert_eq!(jet250.barb_overlay.as_ref().unwrap().units, "knots");
        let jet250_visual = jet250.visual_recipe(WrfProduct::Wind250);
        assert_eq!(
            jet250_visual.upper_air_template,
            Some(UpperAirTemplateRecipe {
                level_hpa: 250,
                fill_role: UpperAirFillRole::JetSpeed,
                fill_var: "wspd_250mb",
                fill_units: "knots",
                height_contour_var: "height_250mb",
                height_units: "dam",
                wind_u_var: "uvmet_u_250mb",
                wind_v_var: "uvmet_v_250mb",
                wind_units: "knots",
            })
        );
        assert_eq!(
            product_tick_values(WrfProduct::Wind250).unwrap(),
            vec![50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0]
        );

        let jet300 = WrfProduct::Wind300.recipe();
        assert_eq!(jet300.fill_var, "wspd_300mb");
        assert_eq!(jet300.fill_units, "knots");
        assert_eq!(jet300.palette, ProductPalette::JetSpeed);
        assert!(jet300.title_template.contains("Jet Streak"));
        assert_eq!(jet300.levels.first().copied(), Some(40.0));
        assert_eq!(jet300.levels.last().copied(), Some(180.0));
        assert_eq!(jet300.contour_overlays[0].var, "height_300mb");
        assert_eq!(
            jet300
                .visual_recipe(WrfProduct::Wind300)
                .upper_air_template
                .unwrap()
                .fill_role,
            UpperAirFillRole::JetSpeed
        );
        assert_eq!(
            product_tick_values(WrfProduct::Wind300).unwrap(),
            vec![40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
        );
    }

    #[test]
    fn upper_air_visual_recipes_carry_template_contracts() {
        for product in default_product_suite() {
            let visual = product.recipe().visual_recipe(*product);
            if product.visual_mode() != ProductVisualMode::UpperAirAnalysis {
                assert!(
                    visual.upper_air_template.is_none(),
                    "{} should not carry an upper-air template",
                    product.canonical_name()
                );
                continue;
            }

            let template = visual.upper_air_template.unwrap_or_else(|| {
                panic!(
                    "{} upper-air product should name its pressure-level template",
                    product.canonical_name()
                )
            });
            assert_eq!(template.fill_var, product.recipe().fill_var);
            assert_eq!(template.fill_units, product.recipe().fill_units);
            assert_eq!(
                template.height_contour_var,
                product.recipe().contour_overlays[0].var
            );
            let barbs = product.recipe().barb_overlay.expect("upper-air barbs");
            assert_eq!(template.wind_u_var, barbs.u_var);
            assert_eq!(template.wind_v_var, barbs.v_var);
            assert_eq!(template.wind_units, "knots");
            assert_eq!(template.height_units, "dam");
        }

        let vort500 = WrfProduct::Vort500Wind
            .recipe()
            .visual_recipe(WrfProduct::Vort500Wind)
            .upper_air_template
            .unwrap();
        assert_eq!(vort500.level_hpa, 500);
        assert_eq!(vort500.fill_role, UpperAirFillRole::Vorticity);

        let rh700 = WrfProduct::Rh700Wind
            .recipe()
            .visual_recipe(WrfProduct::Rh700Wind)
            .upper_air_template
            .unwrap();
        assert_eq!(rh700.level_hpa, 700);
        assert_eq!(rh700.fill_role, UpperAirFillRole::RelativeHumidity);

        let td850 = WrfProduct::Td850Wind
            .recipe()
            .visual_recipe(WrfProduct::Td850Wind)
            .upper_air_template
            .unwrap();
        assert_eq!(td850.level_hpa, 850);
        assert_eq!(td850.fill_role, UpperAirFillRole::Dewpoint);
    }

    #[test]
    fn recipes_expose_operational_visual_contract() {
        let visual = WrfProduct::ReflectivityUh
            .recipe()
            .visual_recipe(WrfProduct::ReflectivityUh);
        assert_eq!(visual.palette, ProductPalette::Reflectivity);
        assert_eq!(visual.mask_policy, MaskPolicy::Below(5.0));
        assert_eq!(
            visual.frame_policy,
            ProductFramePolicy::FiniteDataWithOverlays
        );
        assert_eq!(
            visual.legend_ticks.as_deref(),
            Some(&[5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0][..])
        );
        assert!(visual.provenance_label.contains("UH swath bins"));
        assert_eq!(visual.overlays.len(), 1);
        let ProductOverlayRecipe::UhTrackSwath(overlay) = &visual.overlays[0];
        assert_eq!(overlay.source_var, "uhel_0_3km_1h_max");
        assert_eq!(overlay.units, "m2/s2");
        assert_eq!(overlay.threshold_bins, UH_TRACK_BINS.to_vec());
        assert_eq!(overlay.fill_colors, UH_TRACK_FILL_COLORS.to_vec());
        assert_eq!(overlay.edge_color, Color::BLACK);
        assert_eq!(overlay.edge_width_px, 2);
        assert_eq!(overlay.edge_halo_color, Color::WHITE);
        assert_eq!(overlay.edge_halo_width_px, 1);
        assert_eq!(overlay.lookback_minutes, 60);
        assert_eq!(visual.overlay_legends.len(), 1);
        assert_eq!(
            visual.overlay_legends[0].title,
            "1 h 0-3 km UH swath (m2/s2)"
        );
        assert_eq!(visual.overlay_legends[0].entries.len(), UH_TRACK_BINS.len());

        let upper = WrfProduct::Height500Wind
            .recipe()
            .visual_recipe(WrfProduct::Height500Wind);
        assert_eq!(upper.frame_policy, ProductFramePolicy::FullDomain);
        assert_eq!(upper.barb_overlay.as_ref().unwrap().units, "knots");
        assert_eq!(
            upper.barb_overlay.as_ref().unwrap().spacing_px,
            UPPER_AIR_BARB_SPACING_PX
        );
        assert_eq!(upper.upper_air_template.as_ref().unwrap().level_hpa, 500);
    }

    #[test]
    fn default_suite_provenance_labels_are_product_aware() {
        for product in default_product_suite() {
            let label = product_provenance_label(*product);
            assert_ne!(
                label,
                "WRF diagnostic recipe",
                "{} should not use a generic provenance subtitle",
                product.canonical_name()
            );
            assert!(
                label.len() >= 16 && label.contains(' '),
                "{} provenance label should describe the meteorological source/meaning: {label}",
                product.canonical_name()
            );
        }

        assert_eq!(
            product_provenance_label(WrfProduct::T2),
            "2 m temperature with MSLP contours and 10 m wind barbs"
        );
        assert_eq!(
            product_provenance_label(WrfProduct::Wind250),
            "Pressure-level jet-speed analysis with height contours and wind barbs"
        );
        assert_eq!(
            product_provenance_label(WrfProduct::VtpMod),
            "Violent tornado proxy composite diagnostic"
        );
        for product in [
            WrfProduct::StpEffective,
            WrfProduct::StpFixed,
            WrfProduct::Scp,
            WrfProduct::Ehi,
            WrfProduct::Tehi,
            WrfProduct::Tts,
            WrfProduct::VtpMod,
            WrfProduct::CriticalAngle,
            WrfProduct::Ship,
            WrfProduct::Dcp,
            WrfProduct::Wndg,
            WrfProduct::Esp,
            WrfProduct::Mmp,
        ] {
            assert_ne!(
                product_provenance_label(product),
                "Composite severe-weather parameter diagnostic",
                "{} should carry a product-specific severe provenance label",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn operational_product_labels_are_ascii_and_unit_explicit() {
        for product in default_product_suite() {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(*product);
            assert!(
                recipe.title_template.is_ascii(),
                "{} title should use portable ASCII labels, got {}",
                product.canonical_name(),
                recipe.title_template
            );
            assert!(
                visual.provenance_label.is_ascii(),
                "{} provenance should use portable ASCII labels, got {}",
                product.canonical_name(),
                visual.provenance_label
            );
            for source in &visual.source_semantics {
                assert!(
                    source.label.is_ascii() && source.units.is_ascii(),
                    "{} source labels/units should be ASCII: {:?}",
                    product.canonical_name(),
                    source
                );
            }
            for overlay in &visual.overlays {
                assert!(
                    overlay.label().is_ascii(),
                    "{} overlay label should be ASCII",
                    product.canonical_name()
                );
            }
            for legend in &visual.overlay_legends {
                assert!(
                    legend.title.is_ascii(),
                    "{} overlay legend title should be ASCII",
                    product.canonical_name()
                );
                for entry in &legend.entries {
                    assert!(
                        entry.label.is_ascii(),
                        "{} overlay legend labels should be ASCII",
                        product.canonical_name()
                    );
                }
            }
        }
    }

    #[test]
    fn render_options_can_request_geographic_or_storm_centered_frames() {
        let crop_options =
            ProductRenderOptions::default().with_geographic_bounds(-101.0, -96.0, 33.0, 37.0);
        assert_eq!(
            requested_frame_policy(ProductFramePolicy::FullDomain, &crop_options),
            ProductFramePolicy::GeographicCrop
        );
        let finite = domain_frame_for_policy(ProductFramePolicy::FiniteData).unwrap();
        assert_eq!(finite.source, DomainFrameSource::RasterAlpha);
        assert!(finite.clear_outside);
        assert!(!finite.legend_follows_frame);
        assert!(!finite.chrome_follows_frame);
        let finite_overlay =
            domain_frame_for_policy(ProductFramePolicy::FiniteDataWithOverlays).unwrap();
        assert_eq!(finite_overlay.source, DomainFrameSource::RasterAlpha);
        assert!(!finite_overlay.clear_outside);
        let crop_map = projected_map_options_for_frame(
            (-125.0, -66.0, 24.0, 50.0),
            1.5,
            ProductFramePolicy::GeographicCrop,
            &crop_options,
        );
        assert_eq!(
            crop_map.domain.frame_source,
            wrf_render::ProjectedFrameSource::GeographicBounds(GeographicBounds::new(
                -101.0, -96.0, 33.0, 37.0
            ))
        );
        assert_eq!(crop_map.domain.pad_fraction, 0.02);

        let storm_options = ProductRenderOptions::default().with_storm_center(35.0, -97.0, 120.0);
        assert_eq!(
            requested_frame_policy(ProductFramePolicy::FullDomain, &storm_options),
            ProductFramePolicy::StormCentered
        );
        let storm_map = projected_map_options_for_frame(
            (-125.0, -66.0, 24.0, 50.0),
            1.5,
            ProductFramePolicy::StormCentered,
            &storm_options,
        );
        assert!(matches!(
            storm_map.domain.frame_source,
            wrf_render::ProjectedFrameSource::GeographicBounds(_)
        ));
        assert_eq!(storm_map.domain.pad_fraction, 0.08);
    }

    #[test]
    fn dateline_crops_keep_regional_basemap_detail() {
        assert_eq!(
            longitude_span_deg(177.0, -178.5),
            4.5,
            "antimeridian crop should use wrapped longitude span"
        );
        assert_eq!(
            basemap_detail_for_bounds((177.0, -178.5, 67.2, 68.8)),
            BasemapDetail::Regional,
            "small high-latitude antimeridian crops should not fall back to global linework"
        );
        assert_eq!(
            basemap_detail_for_bounds((-125.0, -66.0, 24.0, 50.0)),
            BasemapDetail::Broad
        );
        assert_eq!(
            basemap_detail_for_bounds((-180.0, 180.0, -85.0, 85.0)),
            BasemapDetail::Global
        );
    }

    #[test]
    fn storm_center_frame_normalizes_longitude_and_radius() {
        let storm = StormCenteredFrame::new(92.0, 190.0, -5.0);
        assert_eq!(storm.lat_deg, 89.9);
        assert_eq!(storm.lon_deg, -170.0);
        assert_eq!(storm.radius_km, 75.0);

        let bounds = StormCenteredFrame::new(35.0, -97.0, 111.32).geographic_bounds();
        assert!(bounds.south_deg < 35.0);
        assert!(bounds.north_deg > 35.0);
        assert!(bounds.west_deg < -97.0);
        assert!(bounds.east_deg > -97.0);
    }

    #[test]
    fn contour_recipes_carry_operational_hierarchy() {
        let surface = WrfProduct::T2.recipe();
        let slp = &surface.contour_overlays[0];
        assert_eq!(slp.var, "slp");
        assert_eq!(slp.units, "hPa");
        assert_eq!(slp.width_px, 1);
        assert_eq!(slp.halo_width_px, 1);
        assert_eq!(slp.major_every, 2);
        assert_eq!(slp.major_width_px, 2);
        assert_eq!(slp.label_every, 2);
        assert!(slp.show_extrema);

        let upper = WrfProduct::Height500Wind.recipe();
        let height = &upper.contour_overlays[0];
        assert_eq!(height.var, "height_500mb");
        assert_eq!(height.units, "dam");
        assert_eq!(height.major_every, 2);
        assert_eq!(height.major_width_px, 2);
        assert_eq!(height.label_every, 2);
        assert!(!height.show_extrema);
    }

    #[test]
    fn severe_products_no_longer_borrow_generic_palettes() {
        assert_eq!(WrfProduct::SbEcape.recipe().palette, ProductPalette::Ecape);
        assert_eq!(
            WrfProduct::EffectiveCape.recipe().palette,
            ProductPalette::EffectiveCape
        );
        assert_eq!(
            WrfProduct::Sb6Cape.recipe().palette,
            ProductPalette::DeepLayerCape
        );
        assert_eq!(WrfProduct::Ncape.recipe().palette, ProductPalette::Ncape);
        assert_eq!(WrfProduct::Sbcin.recipe().palette, ProductPalette::Cin);
        assert_eq!(WrfProduct::Scp.recipe().palette, ProductPalette::Scp);
        assert_eq!(WrfProduct::Ship.recipe().palette, ProductPalette::Ship);
        assert_eq!(WrfProduct::Dcp.recipe().palette, ProductPalette::Dcp);
        assert_eq!(WrfProduct::Wndg.recipe().palette, ProductPalette::Wndg);
        assert_eq!(WrfProduct::Esp.recipe().palette, ProductPalette::Esp);
        assert_eq!(WrfProduct::Mmp.recipe().palette, ProductPalette::Mmp);
        assert_eq!(
            WrfProduct::CriticalAngle.recipe().palette,
            ProductPalette::CriticalAngle
        );
        assert_eq!(WrfProduct::Pwat.recipe().palette, ProductPalette::Pwat);
        assert_eq!(
            WrfProduct::CloudFracLow.recipe().palette,
            ProductPalette::CloudFraction
        );
        assert_eq!(
            WrfProduct::Pvo500.recipe().palette,
            ProductPalette::PotentialVorticity
        );
        assert_eq!(WrfProduct::Omega500.recipe().palette, ProductPalette::Omega);
        assert_eq!(
            WrfProduct::Rh700Wind.recipe().palette,
            ProductPalette::UpperAirRelativeHumidity
        );
        assert_eq!(
            WrfProduct::Rh2.recipe().palette,
            ProductPalette::SurfaceRelativeHumidity
        );
        assert_eq!(
            WrfProduct::MeanWind06.recipe().palette,
            ProductPalette::LayerMeanWind
        );
        for product in [WrfProduct::LowRh, WrfProduct::MidRh, WrfProduct::DgzRh] {
            assert_eq!(
                product.recipe().palette,
                ProductPalette::SevereMoisture,
                "{} should not borrow the broad RH palette for severe moisture diagnostics",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn default_suite_does_not_use_generic_catchall_palettes() {
        let mut offenders = Vec::new();
        for product in default_product_suite() {
            let palette = product.recipe().palette;
            if matches!(
                palette,
                ProductPalette::Cape
                    | ProductPalette::Stp
                    | ProductPalette::Precipitation
                    | ProductPalette::RelativeHumidity
                    | ProductPalette::WindSpeed
                    | ProductPalette::Temperature
                    | ProductPalette::Dewpoint
            ) {
                offenders.push(format!("{} -> {palette:?}", product.canonical_name()));
            }
        }

        assert!(
            offenders.is_empty(),
            "default suite products should use product-aware surface/upper-air/diagnostic palettes instead of generic catch-alls: {}",
            offenders.join(", ")
        );
    }

    #[test]
    fn named_goal2_families_use_product_specific_palettes() {
        let expected = [
            (WrfProduct::Sbcape, ProductPalette::SurfaceBasedCape),
            (WrfProduct::Mlcape, ProductPalette::MixedLayerCape),
            (WrfProduct::Mucape, ProductPalette::MostUnstableCape),
            (WrfProduct::SbEcape, ProductPalette::Ecape),
            (WrfProduct::Sb3Cape, ProductPalette::ThreeCape),
            (WrfProduct::Sb6Cape, ProductPalette::DeepLayerCape),
            (WrfProduct::EffectiveCape, ProductPalette::EffectiveCape),
            (WrfProduct::Ncape, ProductPalette::Ncape),
            (WrfProduct::Sbcin, ProductPalette::Cin),
            (WrfProduct::StpEffective, ProductPalette::EffectiveStp),
            (WrfProduct::StpFixed, ProductPalette::FixedLayerStp),
            (WrfProduct::Scp, ProductPalette::Scp),
            (WrfProduct::Ship, ProductPalette::Ship),
            (WrfProduct::Dcape, ProductPalette::Dcape),
            (WrfProduct::Dcp, ProductPalette::Dcp),
            (WrfProduct::Wndg, ProductPalette::Wndg),
            (WrfProduct::Esp, ProductPalette::Esp),
            (WrfProduct::Mmp, ProductPalette::Mmp),
            (WrfProduct::Tehi, ProductPalette::Tehi),
            (WrfProduct::Tts, ProductPalette::Tts),
            (WrfProduct::VtpMod, ProductPalette::Vtp),
            (WrfProduct::Pwat, ProductPalette::Pwat),
            (WrfProduct::CloudFracLow, ProductPalette::CloudFraction),
            (WrfProduct::CloudFracMid, ProductPalette::CloudFraction),
            (WrfProduct::CloudFracHigh, ProductPalette::CloudFraction),
            (WrfProduct::CriticalAngle, ProductPalette::CriticalAngle),
            (WrfProduct::Terrain, ProductPalette::Terrain),
            (WrfProduct::Lcl, ProductPalette::LclLfcHeight),
            (WrfProduct::Lfc, ProductPalette::LclLfcHeight),
            (WrfProduct::El, ProductPalette::EquilibriumLevel),
            (WrfProduct::LclTemp, ProductPalette::LclTemperature),
            (WrfProduct::FreezingLevel, ProductPalette::FreezingLevel),
            (WrfProduct::WetBulbZero, ProductPalette::FreezingLevel),
            (WrfProduct::Pblh, ProductPalette::PblHeight),
            (WrfProduct::DgzRh, ProductPalette::SevereMoisture),
            (WrfProduct::LapseRate700500, ProductPalette::LapseRate),
            (WrfProduct::LapseRate03, ProductPalette::LapseRate),
            (WrfProduct::Omega500, ProductPalette::Omega),
            (WrfProduct::Omega700Wind, ProductPalette::Omega),
            (
                WrfProduct::Rh700Wind,
                ProductPalette::UpperAirRelativeHumidity,
            ),
            (WrfProduct::Pvo500, ProductPalette::PotentialVorticity),
            (WrfProduct::MeanWind06, ProductPalette::LayerMeanWind),
            (WrfProduct::Reflectivity, ProductPalette::Reflectivity),
            (WrfProduct::Reflectivity1km, ProductPalette::Reflectivity),
            (WrfProduct::ReflectivityUh, ProductPalette::Reflectivity),
            (WrfProduct::UpdraftHelicity, ProductPalette::Uh),
            (WrfProduct::T2, ProductPalette::SurfaceTemperature),
            (WrfProduct::Td2, ProductPalette::SurfaceDewpoint),
            (WrfProduct::Rh2, ProductPalette::SurfaceRelativeHumidity),
            (WrfProduct::SurfaceWind10m, ProductPalette::SurfaceWind),
            (
                WrfProduct::PrecipAccum,
                ProductPalette::AccumulatedPrecipitation,
            ),
            (WrfProduct::Height500Wind, ProductPalette::UpperAirWind),
            (WrfProduct::Temp850Wind, ProductPalette::UpperAirTemperature),
            (WrfProduct::Td850Wind, ProductPalette::UpperAirDewpoint),
        ];

        for (product, palette) in expected {
            assert_eq!(
                product.recipe().palette,
                palette,
                "{} should use its operational product palette",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn ncape_uses_dedicated_normalized_cape_scale() {
        let recipe = WrfProduct::Ncape.recipe();
        assert_eq!(recipe.fill_var, "ncape");
        assert_eq!(recipe.fill_units, "J/kg");
        assert_eq!(recipe.palette, ProductPalette::Ncape);
        assert_eq!(recipe.levels.first().copied(), Some(0.0));
        assert_eq!(recipe.levels.last().copied(), Some(2000.0));

        let visual = recipe.visual_recipe(WrfProduct::Ncape);
        assert_eq!(visual.mask_policy, MaskPolicy::Below(1.0));
        assert_eq!(
            visual.legend_levels.as_deref(),
            Some(&[0.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0][..])
        );
        assert_eq!(visual.legend_ticks, visual.legend_levels);
        assert!(
            visual.levels.len() > visual.legend_levels.as_ref().unwrap().len(),
            "NCAPE should keep a dense fill scale separate from readable legend thresholds"
        );
        assert_ne!(
            visual.palette.colors(),
            ProductPalette::Cape.colors(),
            "NCAPE should not silently borrow the broad CAPE color table"
        );
    }

    #[test]
    fn cape_variants_use_dedicated_operational_scales() {
        let standard_cape = [
            (WrfProduct::Sbcape, ProductPalette::SurfaceBasedCape),
            (WrfProduct::Mlcape, ProductPalette::MixedLayerCape),
            (WrfProduct::Mucape, ProductPalette::MostUnstableCape),
        ];
        for (product, expected_palette) in standard_cape {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(product);
            assert_eq!(
                recipe.palette,
                expected_palette,
                "{} should use a parcel-specific CAPE palette",
                product.canonical_name()
            );
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(8100.0));
            assert_eq!(visual.mask_policy, MaskPolicy::Below(1.0));
            assert_eq!(
                visual.legend_levels.as_deref(),
                Some(
                    &[
                        0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                        8000.0,
                    ][..]
                )
            );
            assert_eq!(
                visual.palette.colors(),
                ProductPalette::Cape.colors(),
                "{} should keep the legacy CAPE color table",
                product.canonical_name()
            );
        }
        assert_eq!(
            ProductPalette::SurfaceBasedCape.colors(),
            ProductPalette::MixedLayerCape.colors()
        );
        assert_eq!(
            ProductPalette::SurfaceBasedCape.colors(),
            ProductPalette::MostUnstableCape.colors()
        );

        for product in [
            WrfProduct::Ecape,
            WrfProduct::SbEcape,
            WrfProduct::MlEcape,
            WrfProduct::MuEcape,
            WrfProduct::EcapeCape,
        ] {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(product);
            assert_eq!(
                recipe.palette,
                ProductPalette::Ecape,
                "{} should use the ECAPE palette",
                product.canonical_name()
            );
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(5000.0));
            assert_eq!(visual.mask_policy, MaskPolicy::Below(1.0));
            assert_eq!(
                visual.legend_levels.as_deref(),
                Some(
                    &[
                        0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
                        5000.0
                    ][..]
                )
            );
            assert_ne!(
                visual.palette.colors(),
                ProductPalette::Cape.colors(),
                "{} should not borrow the broad CAPE color table",
                product.canonical_name()
            );
        }

        for product in [
            WrfProduct::Sb6Cape,
            WrfProduct::Ml6Cape,
            WrfProduct::Mu6Cape,
        ] {
            let visual = product.recipe().visual_recipe(product);
            assert_eq!(
                visual.palette,
                ProductPalette::DeepLayerCape,
                "{} should use the deep-layer CAPE palette",
                product.canonical_name()
            );
            assert_eq!(visual.levels.last().copied(), Some(6000.0));
            assert_eq!(visual.palette.colors(), ProductPalette::Cape.colors());
        }

        let effective = WrfProduct::EffectiveCape
            .recipe()
            .visual_recipe(WrfProduct::EffectiveCape);
        assert_eq!(effective.palette, ProductPalette::EffectiveCape);
        assert_eq!(effective.levels.last().copied(), Some(6000.0));
        assert_eq!(effective.palette.colors(), ProductPalette::Cape.colors());
    }

    #[test]
    fn severe_moisture_products_use_dedicated_rh_scale() {
        for product in [WrfProduct::LowRh, WrfProduct::MidRh, WrfProduct::DgzRh] {
            let recipe = product.recipe();
            assert_eq!(recipe.fill_units, "%");
            assert_eq!(recipe.palette, ProductPalette::SevereMoisture);
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(100.0));

            let visual = recipe.visual_recipe(product);
            assert_eq!(visual.mask_policy, MaskPolicy::None);
            assert_eq!(
                visual.legend_levels.as_deref(),
                Some(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0][..])
            );
            assert_eq!(visual.legend_ticks, visual.legend_levels);
            assert_ne!(
                visual.palette.colors(),
                ProductPalette::RelativeHumidity.colors(),
                "{} should use severe-moisture colors instead of the generic RH color table",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn upper_air_temperature_and_dewpoint_use_pressure_level_palettes() {
        for product in [
            WrfProduct::Temp200Wind,
            WrfProduct::Temp250Wind,
            WrfProduct::Temp300Wind,
            WrfProduct::Temp500Wind,
            WrfProduct::Temp700Wind,
            WrfProduct::Temp850Wind,
        ] {
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(product);
            assert_eq!(
                recipe.palette,
                ProductPalette::UpperAirTemperature,
                "{} should use the pressure-level temperature palette",
                product.canonical_name()
            );
            assert_ne!(
                visual.palette.colors(),
                ProductPalette::Temperature.colors(),
                "{} should not borrow the generic temperature color table",
                product.canonical_name()
            );
            assert_eq!(
                visual
                    .upper_air_template
                    .as_ref()
                    .map(|template| template.fill_role),
                Some(UpperAirFillRole::Temperature)
            );
            assert_eq!(visual.mask_policy, MaskPolicy::None);
        }

        let td850 = WrfProduct::Td850Wind.recipe();
        let visual = td850.visual_recipe(WrfProduct::Td850Wind);
        assert_eq!(td850.palette, ProductPalette::UpperAirDewpoint);
        assert_ne!(
            visual.palette.colors(),
            ProductPalette::Dewpoint.colors(),
            "850 hPa dewpoint should not borrow the generic dewpoint color table"
        );
        assert_eq!(
            visual
                .upper_air_template
                .as_ref()
                .map(|template| template.fill_role),
            Some(UpperAirFillRole::Dewpoint)
        );
        assert_eq!(visual.mask_policy, MaskPolicy::None);
    }

    #[test]
    fn lcl_temperature_uses_dedicated_thermodynamic_level_scale() {
        let recipe = WrfProduct::LclTemp.recipe();
        let visual = recipe.visual_recipe(WrfProduct::LclTemp);
        assert_eq!(recipe.fill_var, "lcl_temp");
        assert_eq!(recipe.fill_units, "degC");
        assert_eq!(recipe.palette, ProductPalette::LclTemperature);
        assert_eq!(recipe.levels.first().copied(), Some(-30.0));
        assert_eq!(recipe.levels.last().copied(), Some(30.0));
        assert_eq!(visual.extend, ExtendMode::Both);
        assert_eq!(visual.mask_policy, MaskPolicy::None);
        assert_eq!(
            visual.legend_levels.as_deref(),
            Some(&[-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0][..])
        );
        assert_ne!(
            visual.palette.colors(),
            ProductPalette::Temperature.colors(),
            "LCL temperature should not borrow the broad temperature color table"
        );
    }

    #[test]
    fn pwat_uses_inches_for_operational_severe_weather_plots() {
        let recipe = WrfProduct::Pwat.recipe();
        assert_eq!(recipe.fill_var, "pw");
        assert_eq!(recipe.fill_units, "in");
        assert_eq!(recipe.title_template, "Precipitable Water (in)");
        assert_eq!(recipe.palette, ProductPalette::Pwat);
        assert_eq!(recipe.levels.first().copied(), Some(0.0));
        assert_eq!(recipe.levels.last().copied(), Some(3.0));

        let visual = recipe.visual_recipe(WrfProduct::Pwat);
        assert_eq!(
            visual.legend_levels.as_deref(),
            Some(&[0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0][..])
        );
        assert_eq!(
            visual.source_semantics.as_slice(),
            &[ProductSourceSemantics {
                role: ProductVisualSourceRole::Fill,
                var: "pw",
                units: "in",
                source: ProductSourceKind::Derived,
                temporal: ProductTemporalSemantics::Instant,
                label: "column precipitable water converted to inches",
            }]
        );
        assert!(product_provenance_label(WrfProduct::Pwat).contains("converted to inches"));
    }

    #[test]
    fn lapse_rate_products_carry_explicit_degc_per_km_units() {
        for product in [WrfProduct::LapseRate700500, WrfProduct::LapseRate03] {
            let recipe = product.recipe();
            assert_eq!(
                recipe.fill_units,
                "degC/km",
                "{} should request/display lapse-rate units",
                product.canonical_name()
            );
            assert!(
                recipe.title_template.contains("degC/km"),
                "{} title should label lapse-rate units",
                product.canonical_name()
            );

            let visual = recipe.visual_recipe(product);
            assert_eq!(
                visual.source_semantics.as_slice(),
                &[ProductSourceSemantics {
                    role: ProductVisualSourceRole::Fill,
                    var: recipe.fill_var,
                    units: "degC/km",
                    source: ProductSourceKind::Derived,
                    temporal: ProductTemporalSemantics::Instant,
                    label: "computed diagnostic fill field",
                }]
            );
        }
    }

    #[test]
    fn tornadic_parameter_products_have_threshold_legends() {
        for product in [WrfProduct::Tehi, WrfProduct::Tts, WrfProduct::VtpMod] {
            let visual = product.recipe().visual_recipe(product);
            assert_eq!(
                visual.levels.first().copied(),
                Some(0.0),
                "{} should start at zero",
                product.canonical_name()
            );
            assert!(
                visual
                    .levels
                    .last()
                    .is_some_and(|value| (*value - 20.0).abs() < 1.0e-3),
                "{} should end at its operational high-end threshold",
                product.canonical_name()
            );
            assert_eq!(
                visual.legend_ticks.as_deref(),
                Some(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0][..]),
                "{} should use the legacy STP-style threshold legend bins",
                product.canonical_name()
            );
            assert_eq!(visual.legend_levels, visual.legend_ticks);
            assert_eq!(
                visual.palette.colors(),
                ProductPalette::Stp.colors(),
                "{} should keep the legacy STP-family color table",
                product.canonical_name()
            );
            assert!(
                visual.levels.len() > visual.legend_levels.as_ref().unwrap().len(),
                "{} should keep dense fill levels separate from readable legend levels",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn canonical_gallery_families_render_nonblank_smoke_images() {
        let severe_nested_grid = synthetic_grid(10, 8, 34.5, -99.5, 0.07, 0.09);
        let conus_grid = synthetic_grid(18, 10, 25.0, -124.0, 2.4, 3.5);
        let ordinary_grid = synthetic_grid(10, 8, 31.0, -103.0, 0.35, 0.45);
        let small_grid = synthetic_grid(5, 5, 35.0, -98.0, 0.035, 0.035);
        let dateline_grid = synthetic_grid(10, 6, 67.0, 176.5, 0.35, 0.95);
        let mut signature_mismatches = Vec::new();
        let mut rendered_frame_policies = std::collections::HashSet::new();

        let cases = [
            SyntheticSmokeCase::new(WrfProduct::StpEffective, severe_nested_grid.clone())
                .with_options(
                    ProductRenderOptions::default().with_storm_center(34.75, -99.15, 35.0),
                )
                .expect_frame_policy(ProductFramePolicy::StormCentered)
                .expect_signature(0x489f521a48808dd8),
            SyntheticSmokeCase::new(WrfProduct::StpFixed, severe_nested_grid.clone())
                .expect_signature(0x75d7acec7581b9ae),
            SyntheticSmokeCase::new(WrfProduct::VtpMod, severe_nested_grid.clone())
                .expect_signature(0x1461dd62844ea297),
            SyntheticSmokeCase::new(WrfProduct::Tehi, severe_nested_grid.clone())
                .expect_signature(0x2b3f901ecadf8182),
            SyntheticSmokeCase::new(WrfProduct::Tts, severe_nested_grid.clone())
                .expect_signature(0x3cdda888cc9790ee),
            SyntheticSmokeCase::new(WrfProduct::Scp, severe_nested_grid.clone())
                .expect_signature(0x853a5a78165c209c),
            SyntheticSmokeCase::new(WrfProduct::Ship, severe_nested_grid.clone())
                .expect_signature(0x6f8b94741b96d59d),
            SyntheticSmokeCase::new(WrfProduct::Dcape, severe_nested_grid.clone())
                .expect_signature(0xde72e41582a87607),
            SyntheticSmokeCase::new(WrfProduct::Dcp, severe_nested_grid.clone())
                .expect_signature(0x416fc6aa89d54e87),
            SyntheticSmokeCase::new(WrfProduct::Wndg, severe_nested_grid.clone())
                .expect_signature(0x79ce718649c05e81),
            SyntheticSmokeCase::new(WrfProduct::Esp, severe_nested_grid.clone())
                .expect_signature(0xc9ddcd97f9d41969),
            SyntheticSmokeCase::new(WrfProduct::Mmp, severe_nested_grid.clone())
                .expect_signature(0x16a26eacddfde4dc),
            SyntheticSmokeCase::new(WrfProduct::CriticalAngle, severe_nested_grid.clone())
                .expect_signature(0x1bdd84de2a74874c),
            SyntheticSmokeCase::new(WrfProduct::Sbcin, severe_nested_grid.clone())
                .expect_signature(0x2a0bd4f6f626e284),
            SyntheticSmokeCase::new(WrfProduct::Sb3Cape, severe_nested_grid.clone())
                .expect_signature(0xf4967b111eae9797),
            SyntheticSmokeCase::new(WrfProduct::Ncape, severe_nested_grid.clone())
                .expect_signature(0x2f1bbaf5dc1d45d5),
            SyntheticSmokeCase::new(WrfProduct::SbEcape, severe_nested_grid.clone())
                .expect_signature(0xb5c1abcdaeefcfe9),
            SyntheticSmokeCase::new(WrfProduct::Sb6Cape, severe_nested_grid.clone())
                .expect_signature(0x05b1f763ee664989),
            SyntheticSmokeCase::new(WrfProduct::EffectiveCape, severe_nested_grid.clone())
                .expect_signature(0xdde949dea73fd6b5),
            SyntheticSmokeCase::new(WrfProduct::Srh01, severe_nested_grid.clone())
                .expect_signature(0xa58256be16e73057),
            SyntheticSmokeCase::new(WrfProduct::Ehi, severe_nested_grid.clone())
                .expect_signature(0x8a264525c65763eb),
            SyntheticSmokeCase::new(WrfProduct::Shear06, severe_nested_grid.clone())
                .expect_signature(0xa541ff470bbd9627),
            SyntheticSmokeCase::new(WrfProduct::MeanWind06, severe_nested_grid.clone())
                .expect_signature(0x521a7c858af97dab),
            SyntheticSmokeCase::new(WrfProduct::Bri, severe_nested_grid.clone())
                .expect_signature(0x513ea50a008006a0),
            SyntheticSmokeCase::new(WrfProduct::Sbcape, conus_grid.clone())
                .expect_signature(0xb9c6aee038594864),
            SyntheticSmokeCase::new(WrfProduct::Mlcape, conus_grid.clone())
                .expect_signature(0x1409a3ff54d9a632),
            SyntheticSmokeCase::new(WrfProduct::Mucape, conus_grid.clone())
                .expect_signature(0xab53264b2bdddcb2),
            SyntheticSmokeCase::new(WrfProduct::Wind250, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x79e8dbe370934a19),
            SyntheticSmokeCase::new(WrfProduct::Wind300, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x25d61db6d2a72d7c),
            SyntheticSmokeCase::new(WrfProduct::Height500Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x326bcb4deb5d20a9),
            SyntheticSmokeCase::new(WrfProduct::Temp500Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xecb8182dcf5957cc),
            SyntheticSmokeCase::new(WrfProduct::Vort500Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x30d969c7618536bf),
            SyntheticSmokeCase::new(WrfProduct::Pvo500, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x2caa5d0b39e30b59),
            SyntheticSmokeCase::new(WrfProduct::Omega500, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xc066ab6c0e702764),
            SyntheticSmokeCase::new(WrfProduct::Temp700Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xdd3dc4722c9f74a2),
            SyntheticSmokeCase::new(WrfProduct::Height700Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x967bc68d2109bd53),
            SyntheticSmokeCase::new(WrfProduct::Rh700Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x37f79679dcd2098b),
            SyntheticSmokeCase::new(WrfProduct::Omega700Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xc2fa6877dead8236),
            SyntheticSmokeCase::new(WrfProduct::ThetaW850, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x2f16bf60c472bb31),
            SyntheticSmokeCase::new(WrfProduct::Temp850Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xe05a8a9381b5a200),
            SyntheticSmokeCase::new(WrfProduct::Td850Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xf6dd52c039f60ef2),
            SyntheticSmokeCase::new(WrfProduct::Height850Wind, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xbf103923416924f8),
            SyntheticSmokeCase::new(WrfProduct::Wind850, conus_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xbf103923416924f8),
            SyntheticSmokeCase::new(WrfProduct::T2, ordinary_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0x71fb47ceae746c1f),
            SyntheticSmokeCase::new(WrfProduct::Td2, ordinary_grid.clone())
                .expect_contours()
                .expect_barbs()
                .expect_signature(0xa8abcd227dc2fa13),
            SyntheticSmokeCase::new(WrfProduct::Rh2, ordinary_grid.clone())
                .expect_signature(0xad05e4531f646e92),
            SyntheticSmokeCase::new(WrfProduct::U10Component, ordinary_grid.clone())
                .expect_signature(0xef8963df1bcea270),
            SyntheticSmokeCase::new(WrfProduct::ReflectivityUh, ordinary_grid.clone())
                .expect_rgba_grid()
                .expect_contours()
                .expect_frame_policy(ProductFramePolicy::FiniteDataWithOverlays)
                .expect_signature(0x99e7851469f89623),
            SyntheticSmokeCase::new(WrfProduct::Reflectivity1km, ordinary_grid.clone())
                .expect_frame_policy(ProductFramePolicy::FiniteData)
                .expect_signature(0x2b89ac55cfb81d27),
            SyntheticSmokeCase::new(WrfProduct::UpdraftHelicity, ordinary_grid.clone())
                .expect_contours()
                .expect_frame_policy(ProductFramePolicy::FiniteData)
                .expect_signature(0xaec735c02558f349),
            SyntheticSmokeCase::new(WrfProduct::PrecipAccum, ordinary_grid.clone())
                .expect_signature(0xecb4be8c61c658fa),
            SyntheticSmokeCase::new(WrfProduct::Terrain, ordinary_grid.clone())
                .expect_signature(0xafd6621a6737d9f9),
            SyntheticSmokeCase::new(WrfProduct::Pblh, ordinary_grid.clone())
                .expect_signature(0x3b5c0b46118753fb),
            SyntheticSmokeCase::new(WrfProduct::Lcl, ordinary_grid.clone())
                .expect_signature(0xf1b5fa5a72cdcfa7),
            SyntheticSmokeCase::new(WrfProduct::Lfc, ordinary_grid.clone())
                .expect_signature(0xad53d6047d7c0009),
            SyntheticSmokeCase::new(WrfProduct::El, ordinary_grid.clone())
                .expect_signature(0x2b7d772a575ea446),
            SyntheticSmokeCase::new(WrfProduct::LclTemp, ordinary_grid.clone())
                .expect_signature(0x0a6ebca1bea1f645),
            SyntheticSmokeCase::new(WrfProduct::FreezingLevel, ordinary_grid.clone())
                .expect_signature(0xb44955fc04b21a53),
            SyntheticSmokeCase::new(WrfProduct::WetBulbZero, ordinary_grid.clone())
                .expect_signature(0x322b16c028e1e7ef),
            SyntheticSmokeCase::new(WrfProduct::LapseRate700500, ordinary_grid.clone())
                .expect_signature(0x91c3378d5a2c7689),
            SyntheticSmokeCase::new(WrfProduct::LapseRate03, ordinary_grid.clone())
                .expect_signature(0x11a86cf8c7aed8c5),
            SyntheticSmokeCase::new(WrfProduct::KIndex, ordinary_grid.clone())
                .expect_signature(0x76b7301486b6198f),
            SyntheticSmokeCase::new(WrfProduct::TotalTotals, ordinary_grid.clone())
                .expect_signature(0x0f269dab948b5e65),
            SyntheticSmokeCase::new(WrfProduct::MeanMixr, ordinary_grid.clone())
                .expect_signature(0xa1989f98cefd1753),
            SyntheticSmokeCase::new(WrfProduct::DgzRh, ordinary_grid.clone())
                .expect_signature(0x0eb3fb58b435cd5e),
            SyntheticSmokeCase::new(WrfProduct::Fosberg, ordinary_grid.clone())
                .expect_signature(0xd34ecdf1a6ed6858),
            SyntheticSmokeCase::new(WrfProduct::Haines, ordinary_grid.clone())
                .expect_signature(0x7a4669c077f12efc),
            SyntheticSmokeCase::new(WrfProduct::Hdw, ordinary_grid.clone())
                .expect_signature(0x1adf81f1b5433c7e),
            SyntheticSmokeCase::new(WrfProduct::CloudTopTemp, ordinary_grid.clone())
                .expect_signature(0xd853f34eba0ed4a6),
            SyntheticSmokeCase::new(WrfProduct::CloudFracLow, ordinary_grid.clone())
                .expect_signature(0xa88cc5acb3899deb),
            SyntheticSmokeCase::new(WrfProduct::CloudFracMid, ordinary_grid.clone())
                .expect_signature(0xf38296f759da6349),
            SyntheticSmokeCase::new(WrfProduct::CloudFracHigh, ordinary_grid.clone())
                .expect_signature(0x6d0dd9e455cee8a3),
            SyntheticSmokeCase::new(WrfProduct::SurfaceWind10m, small_grid.clone())
                .expect_contours()
                .expect_barbs()
                .allow_no_basemap_linework()
                .expect_signature(0x43b3ffba5c2b4ab2),
            SyntheticSmokeCase::new(WrfProduct::Pwat, dateline_grid.clone())
                .with_options(
                    ProductRenderOptions::default()
                        .with_geographic_bounds(177.0, -178.5, 67.2, 68.8),
                )
                .expect_frame_policy(ProductFramePolicy::GeographicCrop)
                .expect_signature(0x79b92239b589c43a),
        ];

        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::StpEffective
                    && case.grid == severe_nested_grid
                    && case.expected_frame_policy == Some(ProductFramePolicy::StormCentered)
            }),
            "canonical smoke gallery should include a severe nested storm-centered case"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.product == WrfProduct::Sbcape && case.grid == conus_grid),
            "canonical smoke gallery should include a CONUS-domain severe fill case"
        );
        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::Height500Wind
                    && case.grid == conus_grid
                    && case.expect_contours
                    && case.expect_barbs
            }),
            "canonical smoke gallery should include an upper-air contour/barb case"
        );
        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::T2
                    && case.grid == ordinary_grid
                    && case.expect_contours
                    && case.expect_barbs
            }),
            "canonical smoke gallery should include a surface map with contours and barbs"
        );
        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::ReflectivityUh
                    && case.expect_rgba_grid
                    && case.expected_frame_policy
                        == Some(ProductFramePolicy::FiniteDataWithOverlays)
            }),
            "canonical smoke gallery should include reflectivity plus UH swath semantics"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.product == WrfProduct::PrecipAccum),
            "canonical smoke gallery should include precipitation"
        );
        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::SurfaceWind10m
                    && case.grid == small_grid
                    && case.expect_contours
                    && case.expect_barbs
            }),
            "canonical smoke gallery should include a small local domain"
        );
        assert!(
            cases.iter().any(|case| {
                case.product == WrfProduct::Pwat
                    && case.grid == dateline_grid
                    && case.expected_frame_policy == Some(ProductFramePolicy::GeographicCrop)
            }),
            "canonical smoke gallery should include a high-latitude antimeridian crop"
        );

        for case in cases {
            let metrics = render_synthetic_product_smoke(case.product, case.grid, &case.options);
            rendered_frame_policies.insert(metrics.resolved_frame_policy);
            let area = metrics.width as usize * metrics.height as usize;
            assert!(
                metrics.non_background > area / 8,
                "{} smoke render should be nonblank",
                case.product.canonical_name()
            );
            assert!(
                metrics.color_bins >= 6,
                "{} smoke render should include multiple visually distinct color bins, got {}",
                case.product.canonical_name(),
                metrics.color_bins
            );
            assert!(
                metrics.colored_pixels > area / 100,
                "{} smoke render should include a meaningful colored meteorological field, got {} colored pixels across {} color bins",
                case.product.canonical_name(),
                metrics.colored_pixels,
                metrics.color_bins
            );
            assert!(
                metrics.has_projected_domain,
                "{} smoke render should exercise projected map framing",
                case.product.canonical_name()
            );
            if let Some(expected) = case.expected_frame_policy {
                assert_eq!(
                    metrics.resolved_frame_policy,
                    expected,
                    "{} smoke render should exercise the expected frame policy",
                    case.product.canonical_name()
                );
            }
            if let Some(expected) = case.expected_signature {
                if metrics.regression_signature != expected {
                    signature_mismatches.push(format!(
                        "{} synthetic image regression signature changed: left {:#018x}, right {:#018x}",
                        case.product.canonical_name(),
                        metrics.regression_signature,
                        expected
                    ));
                }
            }
            assert!(
                metrics.operational_request_controls_present,
                "{} smoke render should use the same operational visual controls as product request rendering",
                case.product.canonical_name()
            );
            assert!(
                metrics.chrome_contrast_pixels > 20,
                "{} smoke render should draw readable operational chrome/linework, got {} contrasting neutral pixels",
                case.product.canonical_name(),
                metrics.chrome_contrast_pixels
            );
            assert!(
                metrics.legend_contrast_pixels > 10,
                "{} smoke render should draw a readable legend, got {} contrasting legend pixels",
                case.product.canonical_name(),
                metrics.legend_contrast_pixels
            );
            assert!(
                metrics.provenance_subtitle_present,
                "{} smoke render should carry the product-aware provenance label into subtitle chrome",
                case.product.canonical_name()
            );
            assert!(
                metrics.metadata_description_mentions_product,
                "{} smoke render should carry a product-aware metadata description",
                case.product.canonical_name()
            );
            assert!(
                metrics.typed_operational_provenance_present,
                "{} smoke render should carry typed operational provenance metadata",
                case.product.canonical_name()
            );
            if case.expect_basemap_linework {
                assert!(
                    metrics.projected_lines > 0,
                    "{} smoke render should include projected basemap linework",
                    case.product.canonical_name()
                );
            }
            if case.expect_contours {
                assert!(
                    metrics.contour_layers > 0,
                    "{} smoke render should include recipe contours",
                    case.product.canonical_name()
                );
            }
            if case.expect_barbs {
                assert!(
                    metrics.barb_layers > 0,
                    "{} smoke render should include recipe wind barbs",
                    case.product.canonical_name()
                );
                assert!(
                    metrics.barbs_use_operational_pixel_decimation,
                    "{} smoke render should use 1x1 barb anchoring with positive pixel spacing and halos",
                    case.product.canonical_name()
                );
            }
            if case.expect_contours || case.expect_barbs || case.expect_rgba_grid {
                assert!(
                    metrics.map_dark_pixels > 20,
                    "{} smoke render should show dark operational overlays/linework inside the map, got {} dark map pixels",
                    case.product.canonical_name(),
                    metrics.map_dark_pixels
                );
            }
            if case.expect_rgba_grid {
                assert!(
                    metrics.has_rgba_grid,
                    "{} smoke render should include the UH RGBA swath overlay",
                    case.product.canonical_name()
                );
                assert!(
                    metrics.overlay_legends > 0,
                    "{} smoke render should include a readable UH overlay legend",
                    case.product.canonical_name()
                );
            }
            if !case.product.recipe().fill_units.trim().is_empty() {
                assert!(
                    metrics.colorbar_label_present,
                    "{} smoke render should carry units into the colorbar label",
                    case.product.canonical_name()
                );
            }
        }

        assert!(
            signature_mismatches.is_empty(),
            "{}",
            signature_mismatches.join("\n")
        );
        for expected in [
            ProductFramePolicy::FullDomain,
            ProductFramePolicy::FiniteData,
            ProductFramePolicy::FiniteDataWithOverlays,
            ProductFramePolicy::StormCentered,
            ProductFramePolicy::GeographicCrop,
        ] {
            assert!(
                rendered_frame_policies.contains(&expected),
                "canonical smoke gallery should exercise {expected:?} frame policy"
            );
        }
    }

    #[test]
    fn canonical_gallery_exercises_every_default_suite_palette() {
        let source = include_str!("lib.rs");
        let smoke_variants: std::collections::HashSet<&str> = source
            .split("SyntheticSmokeCase::new(WrfProduct::")
            .skip(1)
            .filter_map(|tail| {
                tail.split(|ch: char| !ch.is_ascii_alphanumeric())
                    .next()
                    .filter(|name| !name.is_empty())
            })
            .collect();
        assert!(
            !smoke_variants.is_empty(),
            "canonical smoke gallery should declare explicit product cases"
        );

        let mut required_palettes = std::collections::HashSet::new();
        let mut smoked_palettes = std::collections::HashSet::new();
        let mut smoked_filled = false;
        let mut smoked_severe = false;
        let mut smoked_upper_air = false;
        for product in default_product_suite() {
            let visual = product.recipe().visual_recipe(*product);
            required_palettes.insert(visual.palette);
            if smoke_variants.contains(format!("{:?}", product).as_str()) {
                smoked_palettes.insert(visual.palette);
                match product.visual_mode() {
                    ProductVisualMode::FilledMeteorology => smoked_filled = true,
                    ProductVisualMode::SevereDiagnostic => smoked_severe = true,
                    ProductVisualMode::UpperAirAnalysis => smoked_upper_air = true,
                    _ => {}
                }
            }
        }

        let mut missing: Vec<_> = required_palettes
            .difference(&smoked_palettes)
            .map(|palette| format!("{palette:?}"))
            .collect();
        missing.sort();
        assert!(
            missing.is_empty(),
            "canonical smoke gallery should render at least one product for each default-suite palette, missing: {}",
            missing.join(", ")
        );
        assert!(
            smoked_filled && smoked_severe && smoked_upper_air,
            "canonical smoke gallery should exercise filled, severe, and upper-air visual modes"
        );
    }

    #[test]
    fn canonical_gallery_exercises_operational_recipe_features() {
        let source = include_str!("lib.rs");
        let smoke_variants: std::collections::HashSet<&str> = source
            .split("SyntheticSmokeCase::new(WrfProduct::")
            .skip(1)
            .filter_map(|tail| {
                tail.split(|ch: char| !ch.is_ascii_alphanumeric())
                    .next()
                    .filter(|name| !name.is_empty())
            })
            .collect();

        let mut has_masked_fill = false;
        let mut has_unmasked_fill = false;
        let mut has_contours = false;
        let mut has_barbs = false;
        let mut has_product_overlay = false;
        let mut has_overlay_legend = false;
        let mut has_threshold_legend = false;
        let mut has_display_units = false;
        let mut has_unitless_index = false;
        let mut has_fill_source = false;
        let mut has_contour_source = false;
        let mut has_wind_barb_source = false;
        let mut has_uh_track_source = false;
        let mut has_native_or_computed_source = false;
        let mut has_native_or_interpolated_source = false;
        let mut has_derived_source = false;
        let mut has_instant_source = false;
        let mut has_accumulation_source = false;
        let mut has_history_maximum_source = false;

        for product in default_product_suite() {
            if !smoke_variants.contains(format!("{:?}", product).as_str()) {
                continue;
            }
            let recipe = product.recipe();
            let visual = recipe.visual_recipe(*product);
            has_masked_fill |= !matches!(visual.mask_policy, MaskPolicy::None);
            has_unmasked_fill |= matches!(visual.mask_policy, MaskPolicy::None);
            has_contours |= !visual.contour_overlays.is_empty();
            has_barbs |= visual.barb_overlay.is_some();
            has_product_overlay |= !visual.overlays.is_empty();
            has_overlay_legend |= !visual.overlay_legends.is_empty();
            has_threshold_legend |= visual
                .legend_levels
                .as_ref()
                .is_some_and(|legend| legend.len() >= 2 && visual.levels.len() > legend.len());
            has_display_units |= visual.colorbar_label.is_some();
            has_unitless_index |= recipe.fill_units.is_empty() && visual.colorbar_label.is_none();
            for source in &visual.source_semantics {
                has_fill_source |= matches!(source.role, ProductVisualSourceRole::Fill);
                has_contour_source |=
                    matches!(source.role, ProductVisualSourceRole::ContourOverlay);
                has_wind_barb_source |=
                    matches!(source.role, ProductVisualSourceRole::WindBarbOverlay);
                has_uh_track_source |=
                    matches!(source.role, ProductVisualSourceRole::UhTrackOverlay);
                has_native_or_computed_source |=
                    matches!(source.source, ProductSourceKind::NativeOrComputed);
                has_native_or_interpolated_source |=
                    matches!(source.source, ProductSourceKind::NativeOrInterpolated);
                has_derived_source |= matches!(source.source, ProductSourceKind::Derived);
                has_instant_source |= matches!(source.temporal, ProductTemporalSemantics::Instant);
                has_accumulation_source |= matches!(
                    source.temporal,
                    ProductTemporalSemantics::Accumulation { .. }
                );
                has_history_maximum_source |= matches!(
                    source.temporal,
                    ProductTemporalSemantics::HistoryMaximum { .. }
                );
            }
        }

        assert!(
            has_masked_fill,
            "gallery should render a masked fill product"
        );
        assert!(
            has_unmasked_fill,
            "gallery should render an unmasked fill product"
        );
        assert!(has_contours, "gallery should render contour overlays");
        assert!(has_barbs, "gallery should render wind barbs");
        assert!(
            has_product_overlay,
            "gallery should render a product overlay"
        );
        assert!(
            has_overlay_legend,
            "gallery should render an overlay legend"
        );
        assert!(
            has_threshold_legend,
            "gallery should render a dense fill with sparse threshold legend"
        );
        assert!(
            has_display_units,
            "gallery should render unit-labelled products"
        );
        assert!(
            has_unitless_index,
            "gallery should render unitless severe indices"
        );
        assert!(
            has_fill_source,
            "gallery should represent typed fill source semantics"
        );
        assert!(
            has_contour_source,
            "gallery should represent typed contour source semantics"
        );
        assert!(
            has_wind_barb_source,
            "gallery should represent typed wind-barb source semantics"
        );
        assert!(
            has_uh_track_source,
            "gallery should represent typed UH-track source semantics"
        );
        assert!(
            has_native_or_computed_source,
            "gallery should represent native-or-computed source semantics"
        );
        assert!(
            has_native_or_interpolated_source,
            "gallery should represent native-or-interpolated source semantics"
        );
        assert!(
            has_derived_source,
            "gallery should represent derived source semantics"
        );
        assert!(
            has_instant_source,
            "gallery should represent instant temporal semantics"
        );
        assert!(
            has_accumulation_source,
            "gallery should represent accumulation temporal semantics"
        );
        assert!(
            has_history_maximum_source,
            "gallery should represent history-maximum temporal semantics"
        );
    }

    #[test]
    fn real_wrf_fixture_gallery_smoke_renders_operational_products_when_available(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(wrfout_path) = std::env::var_os("WRF_RUST_REALDATA_WRFOUT") else {
            eprintln!(
                "set WRF_RUST_REALDATA_WRFOUT=/path/to/wrfout to run the real-data product gallery smoke test"
            );
            return Ok(());
        };

        let file = WrfFile::open(PathBuf::from(&wrfout_path))?;
        assert!(
            file.nt > 0,
            "real-data gallery fixture should expose at least one output time"
        );
        let timeidx = std::env::var("WRF_RUST_REALDATA_TIMEIDX")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or_else(|| file.nt.saturating_sub(1));
        assert!(
            timeidx < file.nt,
            "WRF_RUST_REALDATA_TIMEIDX={timeidx} is out of range for fixture with {} times",
            file.nt
        );

        let products = std::env::var("WRF_RUST_REALDATA_PRODUCTS")
            .ok()
            .map(|csv| {
                csv.split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(parse_product)
                    .collect::<ProductResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_else(|| {
                vec![
                    WrfProduct::StpEffective,
                    WrfProduct::Height500Wind,
                    WrfProduct::Temp850Wind,
                    WrfProduct::SurfaceWind10m,
                    WrfProduct::T2,
                    WrfProduct::Td2,
                    WrfProduct::Pwat,
                    WrfProduct::PrecipAccum,
                    WrfProduct::ReflectivityUh,
                ]
            });
        assert!(
            !products.is_empty(),
            "real-data gallery product list should not be empty"
        );

        let mut options = ProductRenderOptions::default();
        if let Some(history_dir) = std::env::var_os("WRF_RUST_REALDATA_HISTORY_DIR") {
            options = options.with_history_dir(PathBuf::from(history_dir));
        }

        let output_dir = std::env::var_os("WRF_RUST_REALDATA_OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "wrf-products-real-gallery-{}-{}",
                    std::process::id(),
                    timeidx
                ))
            });
        fs::create_dir_all(&output_dir)?;

        for product in products {
            let request =
                build_product_request_with_options(&file, product, Some(timeidx), &options)
                    .map_err(|err| {
                        format!(
                            "failed to build {} from real-data fixture {}: {err}",
                            product.canonical_name(),
                            PathBuf::from(&wrfout_path).display()
                        )
                    })?;
            let visual = product.recipe().visual_recipe(product);
            assert_eq!(
                request.subtitle_center.as_deref(),
                Some(visual.provenance_label),
                "{} should carry a provenance subtitle",
                product.canonical_name()
            );
            assert!(
                request
                    .product_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.provenance.as_ref())
                    .is_some_and(|provenance| provenance.maturity == ProductMaturity::Operational),
                "{} should carry typed operational product provenance",
                product.canonical_name()
            );
            if product == WrfProduct::ReflectivityUh {
                assert!(
                    request.rgba_grid.is_some(),
                    "reflectivity+UH real-data render should carry the UH swath RGBA overlay"
                );
                assert!(
                    !request.overlay_legends.is_empty(),
                    "reflectivity+UH real-data render should label the UH swath thresholds"
                );
            }

            let image = render_image_with_style(&request, OPERATIONAL_FAST).map_err(|err| {
                format!(
                    "failed to render {} from real-data fixture {}: {err}",
                    product.canonical_name(),
                    PathBuf::from(&wrfout_path).display()
                )
            })?;
            let area = (image.width() as usize).saturating_mul(image.height() as usize);
            let non_background = image
                .pixels()
                .filter(|pixel| pixel.0 != [255, 255, 255, 255])
                .count();
            let fingerprint = synthetic_image_fingerprint(&image);
            assert!(
                non_background > area / 25,
                "{} real-data render should not be mostly blank, got {non_background} non-background pixels across {area}",
                product.canonical_name()
            );
            assert!(
                fingerprint.colored_pixels > area / 200,
                "{} real-data render should contain colored meteorological data, got {} colored pixels",
                product.canonical_name(),
                fingerprint.colored_pixels
            );
            assert!(
                fingerprint.color_bins >= 4,
                "{} real-data render should use a meaningful operational palette, got {} coarse bins",
                product.canonical_name(),
                fingerprint.color_bins
            );
            assert!(
                fingerprint.chrome_contrast_pixels > 20,
                "{} real-data render should have readable map chrome, got {} contrasting pixels",
                product.canonical_name(),
                fingerprint.chrome_contrast_pixels
            );
            assert!(
                fingerprint.legend_contrast_pixels > 10,
                "{} real-data render should have a readable legend, got {} contrasting pixels",
                product.canonical_name(),
                fingerprint.legend_contrast_pixels
            );

            let output = output_dir.join(format!("{}.png", product.canonical_name()));
            save_rgba_png_profile_with_options(&image, &output, &PngWriteOptions::default())?;
            eprintln!(
                "real-data gallery rendered {} -> {}",
                product.canonical_name(),
                output.display()
            );
        }

        Ok(())
    }

    #[test]
    fn srh_products_use_1500_scale() {
        for product in [
            WrfProduct::Srh01,
            WrfProduct::Srh03,
            WrfProduct::EffectiveSrh,
        ] {
            let recipe = product.recipe();
            assert_eq!(recipe.palette, ProductPalette::Srh);
            assert_eq!(recipe.fill_units, "m2/s2");
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(1500.0));
            assert!(recipe.title_template.contains("m2/s2"));
            assert!(recipe.title_template.is_ascii());
        }
    }

    #[test]
    fn custom_diagnostics_are_first_class_products() {
        let effective = WrfProduct::EffectiveCape.recipe();
        assert_eq!(effective.fill_var, "effective_cape");
        assert_eq!(effective.fill_units, "J/kg");
        assert_eq!(effective.palette, ProductPalette::EffectiveCape);

        let base = WrfProduct::EffectiveInflowBase.recipe();
        assert_eq!(base.fill_var, "effective_inflow_base");
        assert_eq!(base.fill_units, "m");

        let low_cloud = WrfProduct::CloudFracLow.recipe();
        assert_eq!(low_cloud.fill_var, "cloudfrac_low");
        assert_eq!(low_cloud.fill_units, "%");
        assert_eq!(low_cloud.levels.first().copied(), Some(0.0));
        assert_eq!(low_cloud.levels.last().copied(), Some(100.0));

        let theta_w = WrfProduct::ThetaW850.recipe();
        assert_eq!(theta_w.fill_var, "theta_w_850mb");
        assert_eq!(theta_w.fill_units, "degC");
        assert_eq!(theta_w.palette, ProductPalette::WetBulbPotentialTemperature);
        assert_eq!(theta_w.levels.first().copied(), Some(-10.0));
        assert_eq!(theta_w.levels.last().copied(), Some(45.0));
        assert_eq!(
            theta_w
                .visual_recipe(WrfProduct::ThetaW850)
                .upper_air_template
                .unwrap()
                .fill_role,
            UpperAirFillRole::WetBulbPotentialTemperature
        );

        let pvo = WrfProduct::Pvo500.recipe();
        assert_eq!(pvo.fill_var, "pvo_500mb");
        assert_eq!(pvo.fill_units, "PVU");
        assert_eq!(
            pvo.title_template,
            "500 hPa Potential Vorticity (PVU), Height (dam), and Wind (kt)"
        );

        let omega700 = WrfProduct::Omega700Wind.recipe();
        assert_eq!(omega700.fill_var, "omega_700mb");
        assert_eq!(omega700.fill_units, "Pa/s");
        assert_eq!(omega700.palette, ProductPalette::Omega);
        assert_eq!(omega700.contour_overlays[0].var, "height_700mb");
        let omega700_barbs = omega700.barb_overlay.as_ref().unwrap();
        assert_eq!(omega700_barbs.u_var, "uvmet_u_700mb");
        assert_eq!(omega700_barbs.v_var, "uvmet_v_700mb");

        let td850 = WrfProduct::Td850Wind.recipe();
        assert_eq!(td850.fill_var, "td_850mb");
        assert_eq!(td850.fill_units, "degC");
        assert_eq!(td850.palette, ProductPalette::UpperAirDewpoint);
        assert_eq!(td850.levels.first().copied(), Some(-30.0));
        assert_eq!(td850.levels.last().copied(), Some(25.0));
        assert_eq!(
            product_tick_values(WrfProduct::Td850Wind).unwrap(),
            vec![-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0,]
        );
    }

    #[test]
    fn reflectivity_uh_combo_uses_1km_reflectivity_and_windowed_uh() {
        let recipe = WrfProduct::ReflectivityUh.recipe();
        let visual = recipe.visual_recipe(WrfProduct::ReflectivityUh);
        assert_eq!(recipe.fill_var, "dbz_1000m_agl");
        assert_eq!(recipe.fill_units, "dBZ");
        assert_eq!(recipe.palette, ProductPalette::Reflectivity);
        assert!(recipe.title_template.contains("1 h UH Swath"));
        assert!(matches!(
            visual.overlays.as_slice(),
            [ProductOverlayRecipe::UhTrackSwath(UhTrackOverlayRecipe {
                source_var: "uhel_0_3km_1h_max",
                units: "m2/s2",
                lookback_minutes: 60,
                ..
            })]
        ));
        assert_eq!(visual.overlay_legends.len(), 1);
        assert_eq!(
            visual.overlay_legends[0]
                .entries
                .iter()
                .map(|entry| entry.label.as_str())
                .collect::<Vec<_>>(),
            vec![">= 50", ">= 100", ">= 200", ">= 300"]
        );
        assert_eq!(
            visual.source_semantics,
            vec![
                ProductSourceSemantics {
                    role: ProductVisualSourceRole::Fill,
                    var: "dbz_1000m_agl",
                    units: "dBZ",
                    source: ProductSourceKind::NativeOrInterpolated,
                    temporal: ProductTemporalSemantics::Instant,
                    label: "instant 1 km AGL reflectivity",
                },
                ProductSourceSemantics {
                    role: ProductVisualSourceRole::UhTrackOverlay,
                    var: "uhel_0_3km_1h_max",
                    units: "m2/s2",
                    source: ProductSourceKind::NativeOrComputed,
                    temporal: ProductTemporalSemantics::HistoryMaximum {
                        lookback_minutes: 60,
                    },
                    label: "0-3 km UH one-hour history maximum",
                },
            ]
        );
    }

    #[test]
    fn reflectivity_uh_history_inputs_are_explicitly_optional() {
        let contract = product_input_contract(WrfProduct::ReflectivityUh);
        assert!(contract.current_wrfout_required);
        let history = contract.optional_history.expect("history contract");
        assert_eq!(history.cli_flag, "--history-dir");
        assert_eq!(history.lookback_minutes, 60);

        let stp = product_input_contract(WrfProduct::StpEffective);
        assert!(stp.current_wrfout_required);
        assert!(stp.optional_history.is_none());
    }

    #[test]
    fn visual_contract_summary_exposes_operational_plot_manifest() {
        let reflectivity = product_visual_contract_summary(WrfProduct::ReflectivityUh);
        assert_eq!(reflectivity.presentation_style, OPERATIONAL_FAST);
        assert_eq!(
            operational_product_presentation_style(),
            StaticPlotStyle::CleanAtlasFast
        );
        assert_eq!(
            reflectivity.visual_mode,
            WrfProduct::ReflectivityUh.visual_mode()
        );
        assert_eq!(
            reflectivity.colorbar_orientation,
            operational_product_colorbar_orientation()
        );
        assert_eq!(
            reflectivity.raster_sample_mode,
            operational_product_raster_sample_mode()
        );
        assert_eq!(
            reflectivity.render_density,
            operational_product_render_density()
        );
        assert_eq!(reflectivity.legend_mode, operational_product_legend_mode());
        assert_eq!(
            reflectivity.legend_density,
            operational_product_legend_density()
        );
        assert_eq!(reflectivity.palette, ProductPalette::Reflectivity);
        assert_eq!(reflectivity.extend_mode, ExtendMode::Max);
        assert!(reflectivity.palette_color_count >= 2);
        assert_eq!(reflectivity.fill_var, "dbz_1000m_agl");
        assert_eq!(reflectivity.fill_units, "dBZ");
        assert_eq!(reflectivity.first_level, Some(5.0));
        assert_eq!(reflectivity.last_level, Some(75.0));
        assert_eq!(reflectivity.level_interval, Some(5.0));
        assert_eq!(reflectivity.colorbar_label, Some("dBZ"));
        assert_eq!(reflectivity.colorbar_tick_step, Some(5.0));
        assert_eq!(
            reflectivity.legend_ticks.as_deref(),
            Some(&[5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0][..])
        );
        assert_eq!(
            reflectivity.frame_policy,
            ProductFramePolicy::FiniteDataWithOverlays
        );
        assert_eq!(
            reflectivity.frame_source,
            Some(DomainFrameSource::RasterAlpha)
        );
        assert_eq!(reflectivity.frame_clear_outside, Some(false));
        assert_eq!(reflectivity.frame_padding_fraction, 0.04);
        assert_eq!(reflectivity.overlay_count, 1);
        assert_eq!(
            reflectivity.overlay_legend_titles,
            vec!["1 h 0-3 km UH swath (m2/s2)".to_string()]
        );
        assert_eq!(reflectivity.overlays.len(), 1);
        assert_eq!(reflectivity.overlays[0].label, "1 h 0-3 km UH swath");
        assert_eq!(reflectivity.overlays[0].source_var, "uhel_0_3km_1h_max");
        assert_eq!(reflectivity.overlays[0].lookback_minutes, Some(60));
        assert_eq!(
            reflectivity.overlays[0].threshold_bins,
            vec![50.0, 100.0, 200.0, 300.0]
        );
        assert_eq!(reflectivity.overlays[0].fill_count, 4);
        assert_eq!(reflectivity.overlays[0].edge_color, Color::BLACK);
        assert!(reflectivity.legend_thresholds.as_ref().is_some_and(
            |legend| legend.as_slice() == [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
        ));
        assert!(reflectivity
            .source_semantics
            .iter()
            .any(
                |source| source.role == ProductVisualSourceRole::UhTrackOverlay
                    && matches!(
                        source.temporal,
                        ProductTemporalSemantics::HistoryMaximum {
                            lookback_minutes: 60
                        }
                    )
            ));

        let upper = product_visual_contract_summary(WrfProduct::Height500Wind);
        let template = upper.upper_air_template.expect("upper-air template");
        assert_eq!(template.level_hpa, 500);
        assert_eq!(template.fill_role, UpperAirFillRole::WindSpeed);
        assert_eq!(upper.contour_count, 1);
        assert_eq!(upper.barb_units, Some("knots"));
        assert_eq!(upper.barb_spacing_px, Some(UPPER_AIR_BARB_SPACING_PX));
        assert_eq!(upper.contours.len(), 1);
        let height_contour = &upper.contours[0];
        assert_eq!(height_contour.var, "height_500mb");
        assert_eq!(height_contour.units, "dam");
        assert_eq!(height_contour.minor_interval, Some(3.0));
        assert_eq!(height_contour.major_every, 2);
        assert!(height_contour.labels);
        assert_eq!(height_contour.label_every, 2);
        assert_eq!(height_contour.color, Color::BLACK);
        assert_eq!(height_contour.halo_color, Color::WHITE);
        assert!(height_contour.major_width_px > height_contour.width_px);
        assert!(height_contour.halo_width_px > 0);
        let barbs = upper.barbs.as_ref().expect("upper-air barbs");
        assert_eq!(barbs.u_var, "uvmet_u_500mb");
        assert_eq!(barbs.v_var, "uvmet_v_500mb");
        assert_eq!(barbs.units, "knots");
        assert_eq!(barbs.stride_x, OPERATIONAL_BARB_GRID_STRIDE);
        assert_eq!(barbs.stride_y, OPERATIONAL_BARB_GRID_STRIDE);
        assert_eq!(barbs.spacing_px, UPPER_AIR_BARB_SPACING_PX);
        assert_eq!(barbs.color, Color::BLACK);
        assert_eq!(barbs.halo_color, Color::WHITE);
        assert!(barbs.halo_width_px > 0);

        let surface = product_visual_contract_summary(WrfProduct::T2);
        assert_eq!(surface.palette, ProductPalette::SurfaceTemperature);
        assert_eq!(surface.colorbar_label, Some("degF"));
        assert_eq!(surface.colorbar_tick_step, Some(10.0));
        assert_eq!(surface.contour_count, 1);
        assert_eq!(surface.contours[0].var, "slp");
        assert_eq!(surface.contours[0].units, "hPa");
        assert_eq!(surface.barb_units, Some("knots"));
        assert_eq!(
            surface.provenance_label,
            "2 m temperature with MSLP contours and 10 m wind barbs"
        );
    }

    #[test]
    fn default_suite_visual_contract_summaries_are_complete() {
        for product in default_product_suite() {
            let summary = product_visual_contract_summary(*product);
            assert_eq!(summary.product, *product);
            assert_eq!(
                summary.presentation_style,
                operational_product_presentation_style(),
                "{} should expose the same operational presentation style used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.visual_mode,
                product.visual_mode(),
                "{} should expose the same visual mode used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.colorbar_orientation,
                operational_product_colorbar_orientation(),
                "{} should expose the operational colorbar orientation used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.raster_sample_mode,
                operational_product_raster_sample_mode(),
                "{} should expose the operational raster sampling mode used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.render_density,
                operational_product_render_density(),
                "{} should expose the operational render density used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.legend_mode,
                operational_product_legend_mode(),
                "{} should expose the operational legend mode used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.legend_density,
                operational_product_legend_density(),
                "{} should expose the operational legend density used by rendering",
                product.canonical_name()
            );
            assert!(!summary.title.trim().is_empty());
            assert!(!summary.fill_var.trim().is_empty());
            let visual = product.recipe().visual_recipe(*product);
            let frame = domain_frame_for_policy(summary.frame_policy);
            assert_eq!(
                summary.extend_mode,
                visual.extend,
                "{} should expose the scale extend behavior used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.palette_color_count,
                visual.palette.colors().len(),
                "{} should expose the palette color count used by rendering",
                product.canonical_name()
            );
            assert!(
                summary.palette_color_count >= 2,
                "{} should expose at least two palette colors",
                product.canonical_name()
            );
            assert_eq!(
                summary.frame_source,
                frame.map(|frame| frame.source),
                "{} should expose the domain frame source used by rendering",
                product.canonical_name()
            );
            assert_eq!(
                summary.frame_clear_outside,
                frame.map(|frame| frame.clear_outside),
                "{} should expose whether the frame clears outside data",
                product.canonical_name()
            );
            assert_eq!(
                summary.frame_inset_px,
                frame.map(|frame| frame.inset_px),
                "{} should expose the domain frame inset",
                product.canonical_name()
            );
            assert_eq!(
                summary.frame_outline_width_px,
                frame.map(|frame| frame.outline_width),
                "{} should expose the domain frame outline width",
                product.canonical_name()
            );
            assert_eq!(
                summary.frame_padding_fraction,
                frame_padding_fraction(summary.frame_policy),
                "{} should expose projected frame padding",
                product.canonical_name()
            );
            assert!(summary.level_count >= 2);
            assert!(
                summary.first_level.is_some()
                    && summary.last_level.is_some()
                    && summary.first_level < summary.last_level,
                "{} should expose an ordered fill scale range",
                product.canonical_name()
            );
            assert!(
                summary
                    .legend_thresholds
                    .as_ref()
                    .is_some_and(|legend| (2..=20).contains(&legend.len())),
                "{} should expose readable operational legend thresholds",
                product.canonical_name()
            );
            assert_eq!(
                summary.legend_ticks,
                summary.legend_thresholds,
                "{} should expose the operational colorbar ticks used by rendering",
                product.canonical_name()
            );
            assert!(
                summary
                    .colorbar_tick_step
                    .map_or(true, |step| step.is_finite() && step > 0.0),
                "{} colorbar tick step should be positive when provided",
                product.canonical_name()
            );
            assert!(
                !summary.provenance_label.trim().is_empty(),
                "{} should expose a provenance label",
                product.canonical_name()
            );
            assert!(
                !summary.source_semantics.is_empty(),
                "{} should expose source semantics",
                product.canonical_name()
            );
            assert_eq!(
                summary.contour_count,
                summary.contours.len(),
                "{} contour summary count should match detailed contour contracts",
                product.canonical_name()
            );
            assert_eq!(
                summary.overlay_count,
                summary.overlays.len(),
                "{} overlay summary count should match detailed overlay contracts",
                product.canonical_name()
            );
            for contour in &summary.contours {
                assert!(
                    contour.level_count >= 2
                        && contour.first_level.is_some()
                        && contour.last_level.is_some()
                        && contour.major_width_px >= contour.width_px
                        && contour.halo_width_px > 0
                        && contour.color.a > 0
                        && contour.halo_color.a > 0
                        && color_contrast(contour.color, contour.halo_color) > 96,
                    "{} contour summaries should expose operational hierarchy, ranges, and contrast styling",
                    product.canonical_name()
                );
            }
            for overlay in &summary.overlays {
                assert!(
                    overlay.threshold_bins.len() >= 2
                        && overlay
                            .threshold_bins
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        && overlay.fill_count == overlay.threshold_bins.len()
                        && overlay.edge_color.a == 255
                        && overlay.edge_width_px > 0
                        && overlay.edge_halo_color.a > 0
                        && overlay.edge_halo_width_px > 0
                        && overlay.lookback_minutes.map_or(true, |minutes| minutes > 0),
                    "{} overlay summaries should expose bins, fills, outlines, and time semantics",
                    product.canonical_name()
                );
            }
            if let Some(barbs) = &summary.barbs {
                assert_eq!(summary.barb_units, Some(barbs.units));
                assert_eq!(summary.barb_spacing_px, Some(barbs.spacing_px));
                assert_eq!(barbs.units, "knots");
                assert_eq!(barbs.stride_x, OPERATIONAL_BARB_GRID_STRIDE);
                assert_eq!(barbs.stride_y, OPERATIONAL_BARB_GRID_STRIDE);
                assert!(
                    barbs.spacing_px > 0.0
                        && barbs.halo_width_px > 0
                        && barbs.color.a > 0
                        && barbs.halo_color.a > 0
                        && color_contrast(barbs.color, barbs.halo_color) > 96,
                    "{} barb summaries should expose pixel spacing and contrast styling",
                    product.canonical_name()
                );
            }
        }
    }

    #[test]
    fn reflectivity_products_name_temporal_and_source_semantics() {
        let reflectivity = WrfProduct::Reflectivity
            .recipe()
            .visual_recipe(WrfProduct::Reflectivity);
        assert_eq!(
            reflectivity.source_semantics.as_slice(),
            &[ProductSourceSemantics {
                role: ProductVisualSourceRole::Fill,
                var: "maxdbz",
                units: "dBZ",
                source: ProductSourceKind::NativeOrComputed,
                temporal: ProductTemporalSemantics::Instant,
                label: "instant composite reflectivity",
            }]
        );

        let reflectivity_1km = WrfProduct::Reflectivity1km
            .recipe()
            .visual_recipe(WrfProduct::Reflectivity1km);
        assert_eq!(
            reflectivity_1km.source_semantics.as_slice(),
            &[ProductSourceSemantics {
                role: ProductVisualSourceRole::Fill,
                var: "dbz_1000m_agl",
                units: "dBZ",
                source: ProductSourceKind::NativeOrInterpolated,
                temporal: ProductTemporalSemantics::Instant,
                label: "instant 1 km AGL reflectivity",
            }]
        );

        let uh = WrfProduct::UpdraftHelicity
            .recipe()
            .visual_recipe(WrfProduct::UpdraftHelicity);
        assert_eq!(
            uh.source_semantics.as_slice(),
            &[
                ProductSourceSemantics {
                    role: ProductVisualSourceRole::Fill,
                    var: NATIVE_OR_COMPUTED_UH_VAR,
                    units: "m2/s2",
                    source: ProductSourceKind::NativeOrComputed,
                    temporal: ProductTemporalSemantics::Instant,
                    label: "instant 0-3 km updraft helicity",
                },
                ProductSourceSemantics {
                    role: ProductVisualSourceRole::ContourOverlay,
                    var: NATIVE_OR_COMPUTED_UH_VAR,
                    units: "m2/s2",
                    source: ProductSourceKind::NativeOrComputed,
                    temporal: ProductTemporalSemantics::Instant,
                    label: "contour overlay source field",
                },
            ]
        );

        let precip = WrfProduct::PrecipAccum
            .recipe()
            .visual_recipe(WrfProduct::PrecipAccum);
        assert_eq!(
            precip.source_semantics.as_slice(),
            &[ProductSourceSemantics {
                role: ProductVisualSourceRole::Fill,
                var: "precip_accum",
                units: "in",
                source: ProductSourceKind::Derived,
                temporal: ProductTemporalSemantics::Accumulation {
                    window_minutes: None,
                },
                label: "run-total accumulated precipitation",
            }]
        );
    }

    #[test]
    fn operational_source_semantics_become_request_metadata() {
        let recipe = WrfProduct::ReflectivityUh.recipe();
        let visual = recipe.visual_recipe(WrfProduct::ReflectivityUh);
        let metadata = product_request_metadata(WrfProduct::ReflectivityUh, &recipe, &visual);
        let provenance = metadata.provenance.as_ref().expect("provenance");

        assert_eq!(metadata.display_name, recipe.title_template);
        assert_eq!(metadata.native_units.as_deref(), Some("dBZ"));
        assert_eq!(metadata.category.as_deref(), Some("wrf_product"));
        assert!(metadata
            .description
            .as_ref()
            .map(|description| description.contains("1 h 0-3 km UH swath"))
            .unwrap_or(false));
        assert_eq!(provenance.lineage, ProductLineage::Windowed);
        assert_eq!(provenance.maturity, ProductMaturity::Operational);
        assert!(provenance.flags.contains(&ProductSemanticFlag::Composite));
        assert_eq!(
            provenance.window,
            Some(ProductWindowSpec {
                process: StatisticalProcess::Maximum,
                duration_hours: Some(1),
            })
        );

        let precip_recipe = WrfProduct::PrecipAccum.recipe();
        let precip_visual = precip_recipe.visual_recipe(WrfProduct::PrecipAccum);
        let precip_metadata =
            product_request_metadata(WrfProduct::PrecipAccum, &precip_recipe, &precip_visual);
        let precip_provenance = precip_metadata.provenance.as_ref().unwrap();
        assert_eq!(precip_provenance.lineage, ProductLineage::Derived);
        assert_eq!(
            precip_provenance.window,
            Some(ProductWindowSpec {
                process: StatisticalProcess::Accumulation,
                duration_hours: None,
            })
        );
    }

    #[test]
    fn mean_wind_vector_output_collapses_to_speed_for_products() {
        let output = VarOutput {
            data: vec![3.0, 4.0, 0.0, 5.0, 4.0, 3.0, 12.0, 12.0],
            shape: vec![2, 2, 2],
            units: "knots".to_string(),
            description: "Mean wind".to_string(),
        };

        let speed = collapse_vector_output_to_speed(output, 2, 2);

        assert_eq!(speed.shape, vec![2, 2]);
        assert_eq!(speed.units, "knots");
        assert_eq!(speed.data, vec![5.0, 5.0, 12.0, 13.0]);
        assert!(speed.description.contains("speed magnitude"));
    }

    #[test]
    fn reflectivity_uh_combo_draws_uh_as_track_overlay() {
        let recipe = WrfProduct::ReflectivityUh.recipe();
        let visual = recipe.visual_recipe(WrfProduct::ReflectivityUh);
        let ProductOverlayRecipe::UhTrackSwath(overlay) = &visual.overlays[0];
        let ColorScale::Discrete(scale) = recipe.palette.scale(recipe.levels, ExtendMode::Both)
        else {
            panic!("expected discrete scale")
        };

        let clear_air = reflectivity_uh_pixel(&scale, overlay, 0.0, 10.0);
        assert_eq!(clear_air, Color::TRANSPARENT);

        let clear_air_track = reflectivity_uh_pixel(&scale, overlay, 0.0, 150.0);
        assert_ne!(clear_air_track, Color::TRANSPARENT);
        assert_eq!(clear_air_track, uh_track_fill_color(overlay, 150.0));

        let clear_air_track_fill = reflectivity_uh_pixel(&scale, overlay, 0.0, 75.0);
        assert_ne!(clear_air_track_fill, Color::TRANSPARENT);
        assert_eq!(clear_air_track_fill, uh_track_fill_color(overlay, 75.0));

        let stronger_track_fill = reflectivity_uh_pixel(&scale, overlay, 0.0, 225.0);
        assert!(stronger_track_fill.a > clear_air_track_fill.a);
        assert_ne!(
            (
                stronger_track_fill.r,
                stronger_track_fill.g,
                stronger_track_fill.b
            ),
            (
                clear_air_track_fill.r,
                clear_air_track_fill.g,
                clear_air_track_fill.b
            )
        );
        let strongest_track_fill = reflectivity_uh_pixel(&scale, overlay, 0.0, 325.0);
        assert!(strongest_track_fill.a > stronger_track_fill.a);
        assert_ne!(
            (
                strongest_track_fill.r,
                strongest_track_fill.g,
                strongest_track_fill.b
            ),
            (
                stronger_track_fill.r,
                stronger_track_fill.g,
                stronger_track_fill.b
            )
        );

        let storm_reflectivity = reflectivity_uh_pixel(&scale, overlay, 45.0, 150.0);
        let storm_reflectivity_base = sample_product_scale(&scale, 45.0);
        assert_ne!(storm_reflectivity, storm_reflectivity_base);
        assert!(storm_reflectivity.r > storm_reflectivity_base.r);
    }

    #[test]
    fn uh_track_swath_bins_are_colored_and_thresholded() {
        let visual = WrfProduct::ReflectivityUh
            .recipe()
            .visual_recipe(WrfProduct::ReflectivityUh);
        let ProductOverlayRecipe::UhTrackSwath(overlay) = &visual.overlays[0];

        assert_eq!(uh_track_color(overlay, 49.9), Color::TRANSPARENT);
        assert_eq!(
            uh_track_color(overlay, 75.0),
            uh_track_fill_color(overlay, 75.0)
        );
        let outline = uh_track_outline_style(overlay);
        assert_eq!(outline.color, Color::BLACK);
        assert_eq!(outline.width, overlay.edge_width_px);
        assert_eq!(outline.halo_color, Color::WHITE);
        assert_eq!(outline.halo_width, overlay.edge_halo_width_px);

        let bins = [
            uh_track_fill_color(overlay, 75.0),
            uh_track_fill_color(overlay, 150.0),
            uh_track_fill_color(overlay, 250.0),
            uh_track_fill_color(overlay, 350.0),
        ];

        assert_eq!(bins[0].a, UH_TRACK_FILL_ALPHA);
        for pair in bins.windows(2) {
            assert!(pair[1].a > pair[0].a);
            assert_ne!(
                (pair[0].r, pair[0].g, pair[0].b),
                (pair[1].r, pair[1].g, pair[1].b)
            );
        }
    }

    #[test]
    fn precip_accum_uses_inch_operational_scale() {
        let recipe = WrfProduct::PrecipAccum.recipe();
        assert_eq!(recipe.fill_units, "in");
        assert_eq!(recipe.palette, ProductPalette::AccumulatedPrecipitation);
        assert!(recipe.title_template.contains("(in)"));
        assert_eq!(recipe.levels.first().copied(), Some(0.01));
        assert_eq!(recipe.levels.last().copied(), Some(15.0));
        for tick in [
            0.01, 0.05, 0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 9.00, 15.00,
        ] {
            assert!(recipe
                .levels
                .iter()
                .any(|level| (*level - tick).abs() < f32::EPSILON));
        }
        assert_eq!(
            product_tick_values(WrfProduct::PrecipAccum).unwrap(),
            vec![
                0.01, 0.05, 0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 9.00,
                15.00,
            ]
        );
        let summary = product_visual_contract_summary(WrfProduct::PrecipAccum);
        let legend = summary
            .legend_thresholds
            .as_ref()
            .expect("precip legend thresholds");
        assert_eq!(summary.extend_mode, ExtendMode::Max);
        assert!(
            summary.palette_color_count > summary.level_count,
            "precip can keep a detailed internal palette without turning the legend into a dense ramp"
        );
        assert!(
            legend.len() < summary.level_count && legend.len() <= 14,
            "precip legend should stay threshold-driven and readable"
        );
    }

    #[test]
    fn visual_recipes_carry_threshold_legend_levels() {
        let cape = WrfProduct::Sbcape
            .recipe()
            .visual_recipe(WrfProduct::Sbcape);
        let cape_legend = cape.legend_levels.as_ref().unwrap();
        assert_eq!(cape.legend_ticks.as_ref(), Some(cape_legend));
        assert_eq!(
            cape_legend,
            &vec![
                0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
            ]
        );
        assert!(cape.levels.len() > cape_legend.len());

        let refl = WrfProduct::ReflectivityUh
            .recipe()
            .visual_recipe(WrfProduct::ReflectivityUh);
        assert_eq!(
            refl.legend_levels.as_deref(),
            Some(&[5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0][..])
        );
        assert!(refl.levels.len() > refl.legend_levels.as_ref().unwrap().len());
    }

    #[test]
    fn default_suite_visual_legends_are_explicit_sparse_and_in_range() {
        for product in default_product_suite() {
            let visual = product.recipe().visual_recipe(*product);
            let legend = visual.legend_levels.as_ref().unwrap_or_else(|| {
                panic!(
                    "{} should define explicit operational legend thresholds",
                    product.canonical_name()
                )
            });
            assert!(
                (2..=20).contains(&legend.len()),
                "{} should keep legend thresholds readable, got {}",
                product.canonical_name(),
                legend.len()
            );

            let lo = visual.levels.first().copied().unwrap_or_default() as f64;
            let hi = visual.levels.last().copied().unwrap_or_default() as f64;
            assert!(
                legend.first().copied().unwrap_or(lo) >= lo - 1.0e-6,
                "{} legend starts below the fill scale",
                product.canonical_name()
            );
            assert!(
                legend.last().copied().unwrap_or(hi) <= hi + 1.0e-3,
                "{} legend extends beyond the fill scale",
                product.canonical_name()
            );
        }
    }

    #[test]
    fn all_products_have_explicit_operational_visual_contracts() {
        let mut seen = std::collections::HashSet::new();

        for product in all_products() {
            assert!(
                seen.insert(product),
                "{} should appear only once in the visual contract audit",
                product.canonical_name()
            );

            let recipe = product.recipe();
            assert!(
                !recipe.fill_var.is_empty(),
                "{} should name a source diagnostic",
                product.canonical_name()
            );
            assert!(
                recipe.levels.len() >= 2 && recipe.levels.iter().all(|value| value.is_finite()),
                "{} should define finite fill levels",
                product.canonical_name()
            );
            assert!(
                recipe.levels.windows(2).all(|pair| pair[0] < pair[1]),
                "{} fill levels should be strictly increasing",
                product.canonical_name()
            );

            let visual = recipe.visual_recipe(product);
            assert!(
                !visual.source_semantics.is_empty(),
                "{} visual recipe should carry typed fill/source semantics",
                product.canonical_name()
            );
            assert_eq!(
                visual
                    .source_semantics
                    .iter()
                    .filter(|source| matches!(source.role, ProductVisualSourceRole::Fill))
                    .count(),
                1,
                "{} visual recipe should carry exactly one typed fill source",
                product.canonical_name()
            );
            assert!(
                recipe.contour_overlays.is_empty()
                    || visual.source_semantics.iter().any(|source| {
                        matches!(source.role, ProductVisualSourceRole::ContourOverlay)
                    }),
                "{} contour overlays should be represented in typed source semantics",
                product.canonical_name()
            );
            assert!(
                recipe.barb_overlay.is_none()
                    || visual
                        .source_semantics
                        .iter()
                        .filter(|source| {
                            matches!(source.role, ProductVisualSourceRole::WindBarbOverlay)
                        })
                        .count()
                        >= 2,
                "{} wind barbs should name both component sources",
                product.canonical_name()
            );
            assert!(
                visual.overlays.is_empty()
                    || visual.source_semantics.iter().any(|source| {
                        matches!(source.role, ProductVisualSourceRole::UhTrackOverlay)
                    }),
                "{} product overlays should be represented in typed source semantics",
                product.canonical_name()
            );
            let colors = visual.palette.colors();
            assert!(
                colors.len() >= 2 && colors.iter().any(|color| color.a > 0),
                "{} should resolve to a visible product palette",
                product.canonical_name()
            );

            let legend = visual.legend_levels.as_ref().unwrap_or_else(|| {
                panic!(
                    "{} should define explicit operational legend thresholds",
                    product.canonical_name()
                )
            });
            assert!(
                (2..=20).contains(&legend.len()),
                "{} should keep legend thresholds readable, got {}",
                product.canonical_name(),
                legend.len()
            );
            assert!(
                legend.iter().all(|value| value.is_finite())
                    && legend.windows(2).all(|pair| pair[0] < pair[1]),
                "{} legend thresholds should be finite and strictly increasing",
                product.canonical_name()
            );

            let lo = visual.levels.first().copied().unwrap_or_default() as f64;
            let hi = visual.levels.last().copied().unwrap_or_default() as f64;
            assert!(
                legend.first().copied().unwrap_or(lo) >= lo - 1.0e-6,
                "{} legend starts below the fill scale",
                product.canonical_name()
            );
            assert!(
                legend.last().copied().unwrap_or(hi) <= hi + 1.0e-3,
                "{} legend extends beyond the fill scale",
                product.canonical_name()
            );

            for contour in &visual.contour_overlays {
                assert!(
                    contour.levels.len() >= 2
                        && contour.levels.iter().all(|value| value.is_finite())
                        && contour.levels.windows(2).all(|pair| pair[0] < pair[1]),
                    "{} contour {} should carry ordered finite operational levels",
                    product.canonical_name(),
                    contour.var
                );
                assert!(
                    contour.halo_width_px > 0 && contour.major_width_px >= contour.width_px,
                    "{} contour {} should carry hierarchy and halo styling",
                    product.canonical_name(),
                    contour.var
                );
            }

            for overlay in &visual.overlays {
                match overlay {
                    ProductOverlayRecipe::UhTrackSwath(overlay) => {
                        assert!(
                            overlay.threshold_bins.len() >= 2
                                && overlay
                                    .threshold_bins
                                    .windows(2)
                                    .all(|pair| pair[0] < pair[1])
                                && overlay.threshold_bins.iter().all(|value| value.is_finite()),
                            "{} UH swath overlay should define ordered finite bins",
                            product.canonical_name()
                        );
                        assert_eq!(
                            overlay.threshold_bins.len(),
                            overlay.fill_colors.len(),
                            "{} UH swath overlay should pair every bin with a fill color",
                            product.canonical_name()
                        );
                        assert!(
                            overlay.fill_colors.iter().all(|color| color.a > 0)
                                && overlay.edge_color.a == 255
                                && overlay.edge_width_px > 0
                                && overlay.edge_halo_color.a > 0
                                && overlay.edge_halo_width_px > 0
                                && overlay.lookback_minutes > 0,
                            "{} UH swath overlay should define visible fills, crisp outlined edges, and a history window",
                            product.canonical_name()
                        );
                    }
                }
            }

            for source in &visual.source_semantics {
                assert!(
                    !source.var.is_empty() && !source.label.is_empty(),
                    "{} source semantics should name a variable and label",
                    product.canonical_name()
                );
                if matches!(source.role, ProductVisualSourceRole::Fill) {
                    assert_eq!(
                        source.var,
                        recipe.fill_var,
                        "{} fill source semantics should match the fill variable",
                        product.canonical_name()
                    );
                    assert_eq!(
                        source.units,
                        recipe.fill_units,
                        "{} fill source semantics should match the fill units",
                        product.canonical_name()
                    );
                }
                match source.temporal {
                    ProductTemporalSemantics::Instant => {}
                    ProductTemporalSemantics::Accumulation { window_minutes } => {
                        assert!(
                            window_minutes.map_or(true, |minutes| minutes > 0),
                            "{} accumulation source window should be positive when specified",
                            product.canonical_name()
                        );
                    }
                    ProductTemporalSemantics::HistoryMaximum { lookback_minutes } => {
                        assert!(
                            lookback_minutes > 0,
                            "{} history maximum lookback should be positive",
                            product.canonical_name()
                        );
                    }
                }
            }

            if let Some(barbs) = &visual.barb_overlay {
                assert_eq!(
                    barbs.units,
                    "knots",
                    "{} wind barbs should use knot components",
                    product.canonical_name()
                );
                assert!(
                    barbs.spacing_px > 0.0
                        && barbs.color.a > 0
                        && barbs.halo_color.a > 0
                        && barbs.halo_width_px > 0
                        && barbs.width_px > 0
                        && barbs.length_px > 0.0,
                    "{} wind barbs should define visible pixel-space spacing and contrast styling",
                    product.canonical_name()
                );
            }
        }

        assert_eq!(
            seen.len(),
            110,
            "update the operational visual contract audit when WrfProduct changes"
        );
    }

    #[test]
    fn scp_products_use_outbreak_scale() {
        let scp = WrfProduct::Scp.recipe();
        assert_eq!(scp.levels.first().copied(), Some(0.0));
        assert_eq!(scp.levels.last().copied(), Some(70.0));

        let ecape_scp = WrfProduct::EcapeScp.recipe();
        assert_eq!(ecape_scp.levels.first().copied(), Some(0.0));
        assert_eq!(ecape_scp.levels.last().copied(), Some(70.0));
    }

    #[derive(Debug, Clone)]
    struct SyntheticSmokeCase {
        product: WrfProduct,
        grid: LatLonGrid,
        options: ProductRenderOptions,
        expect_contours: bool,
        expect_barbs: bool,
        expect_rgba_grid: bool,
        expect_basemap_linework: bool,
        expected_frame_policy: Option<ProductFramePolicy>,
        expected_signature: Option<u64>,
    }

    impl SyntheticSmokeCase {
        fn new(product: WrfProduct, grid: LatLonGrid) -> Self {
            Self {
                product,
                grid,
                options: ProductRenderOptions::default(),
                expect_contours: false,
                expect_barbs: false,
                expect_rgba_grid: false,
                expect_basemap_linework: true,
                expected_frame_policy: None,
                expected_signature: None,
            }
        }

        fn expect_contours(mut self) -> Self {
            self.expect_contours = true;
            self
        }

        fn with_options(mut self, options: ProductRenderOptions) -> Self {
            self.options = options;
            self
        }

        fn expect_barbs(mut self) -> Self {
            self.expect_barbs = true;
            self
        }

        fn expect_rgba_grid(mut self) -> Self {
            self.expect_rgba_grid = true;
            self
        }

        fn allow_no_basemap_linework(mut self) -> Self {
            self.expect_basemap_linework = false;
            self
        }

        fn expect_frame_policy(mut self, frame_policy: ProductFramePolicy) -> Self {
            self.expected_frame_policy = Some(frame_policy);
            self
        }

        fn expect_signature(mut self, signature: u64) -> Self {
            self.expected_signature = Some(signature);
            self
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SyntheticSmokeMetrics {
        width: u32,
        height: u32,
        non_background: usize,
        colored_pixels: usize,
        color_bins: usize,
        map_dark_pixels: usize,
        chrome_contrast_pixels: usize,
        legend_contrast_pixels: usize,
        colorbar_label_present: bool,
        provenance_subtitle_present: bool,
        metadata_description_mentions_product: bool,
        typed_operational_provenance_present: bool,
        operational_request_controls_present: bool,
        regression_signature: u64,
        resolved_frame_policy: ProductFramePolicy,
        projected_lines: usize,
        has_projected_domain: bool,
        contour_layers: usize,
        barb_layers: usize,
        barbs_use_operational_pixel_decimation: bool,
        has_rgba_grid: bool,
        overlay_legends: usize,
    }

    fn render_synthetic_product_smoke(
        product: WrfProduct,
        grid: LatLonGrid,
        options: &ProductRenderOptions,
    ) -> SyntheticSmokeMetrics {
        let recipe = product.recipe();
        let visual = recipe.visual_recipe(product);
        let values = synthetic_values_for_levels(&visual.levels, grid.shape.len());
        let field = Field2D::new(
            ProductKey::named(product.canonical_name()),
            recipe.fill_units,
            grid,
            values,
        )
        .unwrap();
        let scale = visual.palette.scale_with_policy(
            visual.levels.clone(),
            visual.extend,
            visual.mask_policy,
        );
        let mut request = MapRenderRequest::new(field, scale);
        request.colorbar_label = visual.colorbar_label.map(str::to_string);
        request.width = 320;
        request.height = 240;
        request.title = Some(recipe.title_template.to_string());
        request.subtitle_center = Some(visual.provenance_label.to_string());
        request.product_metadata = Some(product_request_metadata(product, &recipe, &visual));
        request.overlay_legends = render_overlay_legends(&visual.overlay_legends);
        apply_operational_visual_controls(&mut request, product, &visual);
        let frame_policy = requested_frame_policy(visual.frame_policy, options);
        request.domain_frame = domain_frame_for_policy(frame_policy);
        apply_synthetic_projected_map(&mut request, frame_policy, options);

        for overlay in &visual.overlays {
            match overlay {
                ProductOverlayRecipe::UhTrackSwath(overlay) => {
                    let mut levels = Vec::with_capacity(overlay.threshold_bins.len() + 2);
                    levels.push(0.0);
                    levels.extend_from_slice(&overlay.threshold_bins);
                    levels.push(overlay.threshold_bins.last().copied().unwrap_or(300.0) + 100.0);
                    let uh = synthetic_field_for_levels(
                        overlay.source_var,
                        overlay.units,
                        request.field.grid.clone(),
                        &levels,
                    );
                    let uh_track = build_uh_track_overlay_field(&uh, overlay).unwrap();
                    apply_reflectivity_uh_rgba(&visual, overlay, &uh_track, &mut request).unwrap();
                    request
                        .add_contour_field(
                            &uh_track,
                            overlay
                                .threshold_bins
                                .iter()
                                .map(|value| *value as f64)
                                .collect(),
                            uh_track_outline_style(overlay),
                        )
                        .unwrap();
                }
            }
        }

        for contour in &visual.contour_overlays {
            let field = synthetic_field_for_levels(
                contour.var,
                contour.units,
                request.field.grid.clone(),
                &contour.levels,
            );
            request
                .add_contour_field(
                    &field,
                    contour.levels.iter().map(|value| *value as f64).collect(),
                    ContourStyle {
                        color: contour.color,
                        width: contour.width_px,
                        halo_color: contour.halo_color,
                        halo_width: contour.halo_width_px,
                        major_every: contour.major_every,
                        major_width: contour.major_width_px,
                        label_every: contour.label_every,
                        labels: contour.labels,
                        show_extrema: contour.show_extrema,
                    },
                )
                .unwrap();
        }

        if let Some(barbs) = &visual.barb_overlay {
            let u = synthetic_wind_component_field(
                barbs.u_var,
                barbs.units,
                request.field.grid.clone(),
                0.0,
            );
            let v = synthetic_wind_component_field(
                barbs.v_var,
                barbs.units,
                request.field.grid.clone(),
                90.0,
            );
            request
                .add_wind_barbs(&u, &v, operational_wind_barb_style(barbs))
                .unwrap();
        }

        let projected_lines = request.projected_lines.len();
        let has_projected_domain = request.projected_domain.is_some();
        let contour_layers = request.contours.len();
        let barb_layers = request.wind_barbs.len();
        let barbs_use_operational_pixel_decimation = request.wind_barbs.iter().all(|barbs| {
            barbs.stride_x == OPERATIONAL_BARB_GRID_STRIDE
                && barbs.stride_y == OPERATIONAL_BARB_GRID_STRIDE
                && barbs.spacing_px > 0.0
                && barbs.halo_color.a > 0
                && barbs.halo_width > 0
        });
        let has_rgba_grid = request.rgba_grid.is_some();
        let overlay_legends = request.overlay_legends.len();
        let operational_request_controls_present =
            request_uses_operational_visual_controls(&request, product, &visual);
        let colorbar_label_present = request
            .colorbar_label
            .as_deref()
            .is_some_and(|label| !label.trim().is_empty());
        let provenance_subtitle_present = request
            .subtitle_center
            .as_deref()
            .is_some_and(|subtitle| subtitle == visual.provenance_label);
        let metadata_description_mentions_product = request
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.description.as_deref())
            .is_some_and(|description| {
                description.contains(visual.provenance_label)
                    && description.contains(product.canonical_name())
            });
        let typed_operational_provenance_present = request
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
            .is_some_and(|provenance| provenance.maturity == ProductMaturity::Operational);
        let image = render_image_with_style(&request, OPERATIONAL_FAST).unwrap();
        let non_background = image
            .pixels()
            .filter(|pixel| pixel.0 != [255, 255, 255, 255])
            .count();
        let fingerprint = synthetic_image_fingerprint(&image);
        SyntheticSmokeMetrics {
            width: image.width(),
            height: image.height(),
            non_background,
            colored_pixels: fingerprint.colored_pixels,
            color_bins: fingerprint.color_bins,
            map_dark_pixels: fingerprint.map_dark_pixels,
            chrome_contrast_pixels: fingerprint.chrome_contrast_pixels,
            legend_contrast_pixels: fingerprint.legend_contrast_pixels,
            colorbar_label_present,
            provenance_subtitle_present,
            metadata_description_mentions_product,
            typed_operational_provenance_present,
            operational_request_controls_present,
            regression_signature: fingerprint.regression_signature,
            resolved_frame_policy: frame_policy,
            projected_lines,
            has_projected_domain,
            contour_layers,
            barb_layers,
            barbs_use_operational_pixel_decimation,
            has_rgba_grid,
            overlay_legends,
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct SyntheticImageFingerprint {
        colored_pixels: usize,
        color_bins: usize,
        map_dark_pixels: usize,
        chrome_contrast_pixels: usize,
        legend_contrast_pixels: usize,
        regression_signature: u64,
    }

    fn synthetic_image_fingerprint(image: &wrf_render::RgbaImage) -> SyntheticImageFingerprint {
        let mut color_bins = std::collections::BTreeSet::new();
        let mut colored_pixels = 0usize;
        let mut map_dark_pixels = 0usize;
        let mut chrome_contrast_pixels = 0usize;
        let mut legend_contrast_pixels = 0usize;
        let legend_left = image.width().saturating_sub(112);
        let map_top = 44;
        let map_bottom = image.height().saturating_sub(18);
        let chrome_background = image.get_pixel(0, 0).0;
        let legend_background = chrome_background;

        for (x, y, pixel) in image.enumerate_pixels() {
            let [r, g, b, a] = pixel.0;
            if a == 0 || [r, g, b] == [255, 255, 255] {
                continue;
            }
            color_bins.insert((r / 32, g / 32, b / 32));
            let max_channel = r.max(g).max(b);
            let min_channel = r.min(g).min(b);
            if max_channel.saturating_sub(min_channel) > 24 {
                colored_pixels += 1;
            }
            let contrasts_with_chrome = max_channel_delta(pixel.0, chrome_background) > 28;
            let contrasts_with_legend = max_channel_delta(pixel.0, legend_background) > 28;
            let neutral_contrast =
                max_channel < 170 && max_channel.saturating_sub(min_channel) <= 60;
            if x < legend_left && y <= map_bottom && neutral_contrast && contrasts_with_chrome {
                chrome_contrast_pixels += 1;
            } else if x >= legend_left && y >= map_top && y <= map_bottom && contrasts_with_legend {
                legend_contrast_pixels += 1;
            }
            if r < 90 && g < 90 && b < 90 {
                if x < legend_left && y >= map_top && y <= map_bottom {
                    map_dark_pixels += 1;
                }
            }
        }

        SyntheticImageFingerprint {
            colored_pixels,
            color_bins: color_bins.len(),
            map_dark_pixels,
            chrome_contrast_pixels,
            legend_contrast_pixels,
            regression_signature: synthetic_regression_signature(image),
        }
    }

    fn synthetic_regression_signature(image: &wrf_render::RgbaImage) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        const BLOCKS_X: u32 = 16;
        const BLOCKS_Y: u32 = 12;

        fn write_byte(hash: &mut u64, value: u8) {
            *hash ^= value as u64;
            *hash = hash.wrapping_mul(FNV_PRIME);
        }

        fn write_u32(hash: &mut u64, value: u32) {
            for byte in value.to_le_bytes() {
                write_byte(hash, byte);
            }
        }

        let mut hash = FNV_OFFSET;
        write_u32(&mut hash, image.width());
        write_u32(&mut hash, image.height());

        for block_y in 0..BLOCKS_Y {
            let y0 = block_y * image.height() / BLOCKS_Y;
            let y1 = ((block_y + 1) * image.height() / BLOCKS_Y).max(y0 + 1);
            for block_x in 0..BLOCKS_X {
                let x0 = block_x * image.width() / BLOCKS_X;
                let x1 = ((block_x + 1) * image.width() / BLOCKS_X).max(x0 + 1);
                let mut sums = [0u64; 4];
                let mut count = 0u64;
                for y in y0..y1.min(image.height()) {
                    for x in x0..x1.min(image.width()) {
                        let [r, g, b, a] = image.get_pixel(x, y).0;
                        sums[0] += r as u64;
                        sums[1] += g as u64;
                        sums[2] += b as u64;
                        sums[3] += a as u64;
                        count += 1;
                    }
                }
                let count = count.max(1);
                for sum in sums {
                    write_byte(&mut hash, ((sum / count) / 16) as u8);
                }
            }
        }

        hash
    }

    fn max_channel_delta(a: [u8; 4], b: [u8; 4]) -> u8 {
        a[0].abs_diff(b[0])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[2]))
    }

    fn color_contrast(a: Color, b: Color) -> u8 {
        max_channel_delta(color_channels(a), color_channels(b))
    }

    fn color_channels(color: Color) -> [u8; 4] {
        [color.r, color.g, color.b, color.a]
    }

    fn apply_synthetic_projected_map(
        request: &mut MapRenderRequest,
        frame_policy: ProductFramePolicy,
        options: &ProductRenderOptions,
    ) {
        let bounds = latlon_bounds(&request.field.grid).expect("synthetic grid bounds");
        let target_ratio =
            map_frame_aspect_ratio_for_mode_with_domain_frame_style_and_colorbar_orientation(
                request.visual_mode,
                request.width,
                request.height,
                request.colorbar,
                request.title.is_some(),
                request.domain_frame.is_some(),
                OPERATIONAL_FAST,
                request.colorbar_orientation,
            );
        let map_options =
            projected_map_options_for_frame(bounds, target_ratio, frame_policy, options);
        let projected = build_projected_map_with_options(
            &request.field.grid.lat_deg,
            &request.field.grid.lon_deg,
            &map_options,
        )
        .expect("synthetic projected map");
        request.apply_projected_map(&projected);
    }

    fn synthetic_field_for_levels(
        name: &'static str,
        units: &'static str,
        grid: LatLonGrid,
        levels: &[f32],
    ) -> Field2D {
        let values = synthetic_values_for_levels(levels, grid.shape.len());
        Field2D::new(ProductKey::named(name), units, grid, values).unwrap()
    }

    fn synthetic_wind_component_field(
        name: &'static str,
        units: &'static str,
        grid: LatLonGrid,
        phase_deg: f32,
    ) -> Field2D {
        let values = (0..grid.shape.len())
            .map(|idx| {
                let phase = phase_deg.to_radians() + idx as f32 * 0.45;
                22.0 + 18.0 * phase.sin()
            })
            .collect();
        Field2D::new(ProductKey::named(name), units, grid, values).unwrap()
    }

    fn synthetic_grid(
        nx: usize,
        ny: usize,
        lat0: f32,
        lon0: f32,
        dlat: f32,
        dlon: f32,
    ) -> LatLonGrid {
        let shape = GridShape::new(nx, ny).unwrap();
        let mut lat = Vec::with_capacity(shape.len());
        let mut lon = Vec::with_capacity(shape.len());
        for j in 0..ny {
            for i in 0..nx {
                lat.push(lat0 + j as f32 * dlat);
                let mut value = lon0 + i as f32 * dlon;
                if value > 180.0 {
                    value -= 360.0;
                }
                lon.push(value);
            }
        }
        LatLonGrid::new(shape, lat, lon).unwrap()
    }

    fn synthetic_values_for_levels(levels: &[f32], len: usize) -> Vec<f32> {
        let lo = levels.first().copied().unwrap_or(0.0);
        let hi = levels.last().copied().unwrap_or(lo + 1.0);
        let span = (hi - lo).abs().max(1.0);
        (0..len)
            .map(|idx| {
                let phase = (idx % 17) as f32 / 16.0;
                lo + span * phase
            })
            .collect()
    }
}
