//! WRF product recipes.
//!
//! This crate is glue. It maps product names to `wrf-core::getvar` calls and
//! `wrf-render` requests, but it does not duplicate diagnostic science.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use wrf_core::{
    getvar, met::composite::interp_to_height_level, ComputeOpts, VarOutput, WrfFile, WrfProjection,
};
use wrf_render::{
    build_projected_map_with_options,
    map_frame_aspect_ratio_for_mode_with_domain_frame_style_and_colorbar_orientation,
    palette_scale, render_image_with_style, save_rgba_png_profile_with_options, srh_scale_levels,
    stp_scale_levels, BasemapDetail, Color, ColorScale, ColorbarOrientation, ContourStyle,
    DiscreteColorScale, DomainFrame, ExtendMode, Field2D, GridShape, LatLonGrid, LegendControls,
    LegendMode, LevelDensity, MapRenderRequest, PngWriteOptions, ProductKey, ProductVisualMode,
    ProjectedMapBuildOptions, ProjectionSpec, RasterSampleMode, RenderDensity, RgbaGridField,
    RustwxRenderError, WeatherPalette, WindBarbStyle, OPERATIONAL_FAST,
};

const DEFAULT_PRODUCT_WIDTH: u32 = 1600;
const DEFAULT_PRODUCT_HEIGHT: u32 = 1200;
const UH_TRACK_THRESHOLD: f32 = 50.0;
const UH_TRACK_FILL_ALPHA: u8 = 38;
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
    #[error(transparent)]
    Wrf(#[from] wrf_core::WrfError),
    #[error(transparent)]
    Render(#[from] RustwxRenderError),
}

pub type ProductResult<T> = Result<T, ProductError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaletteId {
    Cape,
    Srh,
    Stp,
    Ehi,
    Uh,
    LapseRate,
    Vorticity,
    Reflectivity,
    SimIr,
    Temperature,
    SurfaceTemperature,
    Dewpoint,
    SurfaceDewpoint,
    RelativeHumidity,
    Wind,
    WindComponent,
    Precipitation,
    Grayscale,
}

impl PaletteId {
    pub fn default_levels(self) -> Vec<f32> {
        match self {
            Self::Cape => range_step(0.0, 8100.0, 100.0),
            Self::Srh => srh_scale_levels()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            Self::Stp => stp_scale_levels()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            Self::Ehi => range_step(0.0, 24.0, 0.2),
            Self::Uh => range_step(0.0, 400.0, 5.0),
            Self::LapseRate => range_step(2.0, 10.0, 0.1),
            Self::Vorticity => range_step(0.0, 0.0005, 0.000025),
            Self::Reflectivity => range_step(5.0, 75.0, 5.0),
            Self::SimIr => range_step(-90.0, 50.0, 1.0),
            Self::Temperature => range_step(-40.0, 50.0, 5.0),
            Self::SurfaceTemperature => range_step(-60.0, 120.0, 1.0),
            Self::Dewpoint | Self::SurfaceDewpoint => range_step(-40.0, 90.0, 1.0),
            Self::RelativeHumidity => range_step(0.0, 100.0, 10.0),
            Self::Wind => range_step(0.0, 120.0, 10.0),
            Self::WindComponent => wind_component_levels(),
            Self::Precipitation => vec![0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0],
            Self::Grayscale => range_step(0.0, 1.0, 0.1),
        }
    }

    fn scale(self, levels: Vec<f32>, extend: ExtendMode) -> ColorScale {
        let levels = levels.into_iter().map(|value| value as f64).collect();
        let mask_below = self.mask_below();
        let discrete = match self {
            Self::Cape => palette_scale(WeatherPalette::Cape, levels, extend, mask_below),
            Self::Srh => palette_scale(WeatherPalette::Srh, levels, extend, mask_below),
            Self::Stp => palette_scale(WeatherPalette::Stp, levels, extend, mask_below),
            Self::Ehi => palette_scale(WeatherPalette::Ehi, levels, extend, mask_below),
            Self::Uh => palette_scale(WeatherPalette::Uh, levels, extend, mask_below),
            Self::LapseRate => palette_scale(WeatherPalette::LapseRate, levels, extend, mask_below),
            Self::Vorticity => palette_scale(WeatherPalette::RelVort, levels, extend, mask_below),
            Self::Reflectivity => {
                palette_scale(WeatherPalette::Reflectivity, levels, extend, mask_below)
            }
            Self::SimIr => palette_scale(WeatherPalette::SimIr, levels, extend, mask_below),
            Self::Temperature | Self::SurfaceTemperature => {
                palette_scale(WeatherPalette::Temperature, levels, extend, mask_below)
            }
            Self::Dewpoint => palette_scale(WeatherPalette::Dewpoint, levels, extend, mask_below),
            Self::SurfaceDewpoint => DiscreteColorScale {
                levels,
                colors: surface_dewpoint_colors(),
                extend,
                mask_below,
            },
            Self::RelativeHumidity => palette_scale(WeatherPalette::Rh, levels, extend, mask_below),
            Self::Wind => palette_scale(WeatherPalette::Winds, levels, extend, mask_below),
            Self::WindComponent => DiscreteColorScale {
                levels,
                colors: wind_component_colors(),
                extend,
                mask_below: None,
            },
            Self::Precipitation => {
                palette_scale(WeatherPalette::Precip, levels, extend, mask_below)
            }
            Self::Grayscale => DiscreteColorScale {
                levels,
                colors: vec![
                    Color::rgba(250, 250, 250, 255),
                    Color::rgba(224, 224, 224, 255),
                    Color::rgba(190, 190, 190, 255),
                    Color::rgba(150, 150, 150, 255),
                    Color::rgba(112, 112, 112, 255),
                ],
                extend,
                mask_below: None,
            },
        };
        ColorScale::Discrete(discrete)
    }

    fn mask_below(self) -> Option<f64> {
        match self {
            Self::Cape => Some(1.0),
            Self::Srh | Self::Stp | Self::Ehi | Self::Uh | Self::Reflectivity => Some(0.01),
            Self::Precipitation => Some(0.001),
            _ => None,
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
    Shear06,
    Ebwd,
    MeanWind06,
    Reflectivity,
    Reflectivity1km,
    ReflectivityUh,
    CloudTopTemp,
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
    Omega500,
    Temp700Wind,
    Height700Wind,
    Rh700Wind,
    Height850Wind,
    Temp850Wind,
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
            "wind250" | "250mb_wind" | "250_wind" => Some(Self::Wind250),
            "height300" | "300mb_height" | "300_height_wind" | "height300_wind" => {
                Some(Self::Height300Wind)
            }
            "temp300" | "300mb_temp" | "300_temp_wind" | "temp300_wind" => Some(Self::Temp300Wind),
            "wind300" | "300mb_wind" | "300_wind" => Some(Self::Wind300),
            "height500" | "500mb_height" | "500_height_wind" | "height500_wind" => {
                Some(Self::Height500Wind)
            }
            "temp500" | "500mb_temp" | "500_temp_wind" | "temp500_wind" => Some(Self::Temp500Wind),
            "wind500" | "500mb_wind" | "500_wind" => Some(Self::Wind500),
            "vort500" | "vort500_wind" | "500mb_vort" | "500mb_vorticity" | "vorticity500_wind" => {
                Some(Self::Vort500Wind)
            }
            "omega500" | "500mb_omega" | "vertical_velocity500" => Some(Self::Omega500),
            "temp700" | "700mb_temp" | "700_temp_wind" | "temp700_wind" => Some(Self::Temp700Wind),
            "height700" | "700mb_height" | "700_height_wind" | "height700_wind" => {
                Some(Self::Height700Wind)
            }
            "rh700" | "700mb_rh" | "700_relative_humidity" | "rh700_wind" => Some(Self::Rh700Wind),
            "height850" | "850mb_height" | "850_height_wind" | "height850_wind" => {
                Some(Self::Height850Wind)
            }
            "temp850" | "850mb_temp" | "850_temp_wind" | "temp850_wind" => Some(Self::Temp850Wind),
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
            Self::Shear06 => "shear06",
            Self::Ebwd => "ebwd",
            Self::MeanWind06 => "mean_wind06",
            Self::Reflectivity => "reflectivity",
            Self::Reflectivity1km => "reflectivity_1km",
            Self::ReflectivityUh => "reflectivity_uh",
            Self::CloudTopTemp => "cloud_top_temperature",
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
            Self::Omega500 => "omega500",
            Self::Temp700Wind => "temp700_wind",
            Self::Height700Wind => "height700_wind",
            Self::Rh700Wind => "rh700_wind",
            Self::Height850Wind => "height850_wind",
            Self::Temp850Wind => "temp850_wind",
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
            | Self::Omega500
            | Self::Temp700Wind
            | Self::Height700Wind
            | Self::Rh700Wind
            | Self::Height850Wind
            | Self::Temp850Wind
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
            | Self::UpdraftHelicity
            | Self::ReflectivityUh
            | Self::Lcl
            | Self::Lfc
            | Self::El
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
                PaletteId::Cape,
                range_i32(0, 2000, 50),
                "Normalized CAPE (NCAPE, SB)",
            ),
            Self::EcapeCape => ecape_component_recipe(
                "ecape_cape",
                "J/kg",
                PaletteId::Cape,
                PaletteId::Cape.default_levels(),
                "ECAPE Parcel CAPE (SB)",
            ),
            Self::EcapeCin => ecape_component_recipe(
                "ecape_cin",
                "J/kg",
                PaletteId::Cape,
                range_i32(-300, 0, 25),
                "ECAPE CIN (SB)",
            ),
            Self::EcapeLfc => ecape_component_recipe(
                "ecape_lfc",
                "m",
                PaletteId::Grayscale,
                range_i32(0, 8000, 500),
                "ECAPE LFC Height (SB)",
            ),
            Self::EcapeEl => ecape_component_recipe(
                "ecape_el",
                "m",
                PaletteId::Grayscale,
                range_i32(4000, 18000, 1000),
                "ECAPE Equilibrium Level Height (SB)",
            ),
            Self::EcapeScp => severe_recipe_with_levels(
                "ecape_scp",
                "",
                PaletteId::Stp,
                scp_levels(),
                "ECAPE Supercell Composite Parameter (MU)",
            ),
            Self::EcapeEhi => severe_recipe(
                "ecape_ehi",
                "",
                PaletteId::Ehi,
                "ECAPE Energy Helicity Index (SB)",
            ),
            Self::Sbcape => severe_recipe("sbcape", "J/kg", PaletteId::Cape, "SBCAPE"),
            Self::Sbcin => cin_recipe("sbcin", "SBCIN"),
            Self::Mlcape => severe_recipe("mlcape", "J/kg", PaletteId::Cape, "MLCAPE"),
            Self::Mlcin => cin_recipe("mlcin", "MLCIN"),
            Self::Mucape => severe_recipe("mucape", "J/kg", PaletteId::Cape, "MUCAPE"),
            Self::Mucin => cin_recipe("mucin", "MUCIN"),
            Self::Srh01 => severe_recipe(
                "srh1",
                "m2/s2",
                PaletteId::Srh,
                "0-1km Storm Relative Helicity (m²/s²)",
            ),
            Self::Srh03 => severe_recipe(
                "srh3",
                "m2/s2",
                PaletteId::Srh,
                "0-3km Storm Relative Helicity (m²/s²)",
            ),
            Self::EffectiveSrh => severe_recipe(
                "effective_srh",
                "m2/s2",
                PaletteId::Srh,
                "Effective-Layer SRH (m²/s²)",
            ),
            Self::Shear01 => wind_layer_recipe("bulk_shear", 0.0, 1000.0, "0-1 km Bulk Shear"),
            Self::StpEffective => ProductRecipe {
                fill_var: "stp",
                fill_units: "",
                palette: PaletteId::Stp,
                levels: PaletteId::Stp.default_levels(),
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
                palette: PaletteId::Stp,
                levels: PaletteId::Stp.default_levels(),
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
                PaletteId::Stp,
                scp_levels(),
                "Supercell Composite Parameter",
            ),
            Self::Ehi => severe_recipe("ehi", "", PaletteId::Ehi, "0-3km Energy Helicity Index"),
            Self::Tehi => severe_recipe("tehi", "", PaletteId::Stp, "Tornadic 0-1 km EHI"),
            Self::Tts => {
                severe_recipe("tts", "", PaletteId::Stp, "Tornadic Tilting and Stretching")
            }
            Self::VtpMod => severe_recipe(
                "vtp_mod",
                "",
                PaletteId::Stp,
                "Modified Violent Tornado Parameter",
            ),
            Self::CriticalAngle => ProductRecipe {
                fill_var: "critical_angle",
                fill_units: "degrees",
                palette: PaletteId::Wind,
                levels: range_i32(0, 180, 15),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Critical Angle",
                opts: ComputeOptsPatch::default(),
            },
            Self::Ship => severe_recipe("ship", "", PaletteId::Stp, "Significant Hail Parameter"),
            Self::Bri => ProductRecipe {
                fill_var: "bri",
                fill_units: "",
                palette: PaletteId::Stp,
                levels: range_i32(0, 100, 5),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Bulk Richardson Number",
                opts: ComputeOptsPatch::default(),
            },
            Self::Shear06 => ProductRecipe {
                fill_var: "bulk_shear",
                fill_units: "knots",
                palette: PaletteId::Wind,
                levels: PaletteId::Wind.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "0-6 km Bulk Shear",
                opts: ComputeOptsPatch {
                    bottom_m: Some(0.0),
                    top_m: Some(6000.0),
                    ..Default::default()
                },
            },
            Self::Ebwd => severe_recipe(
                "ebwd",
                "knots",
                PaletteId::Wind,
                "Effective Bulk Wind Difference",
            ),
            Self::MeanWind06 => wind_layer_recipe("mean_wind", 0.0, 6000.0, "0-6 km Mean Wind"),
            Self::Reflectivity => ProductRecipe {
                fill_var: "maxdbz",
                fill_units: "dBZ",
                palette: PaletteId::Reflectivity,
                levels: PaletteId::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Composite Reflectivity",
                opts: ComputeOptsPatch::default(),
            },
            Self::Reflectivity1km => ProductRecipe {
                fill_var: "dbz_1000m_agl",
                fill_units: "dBZ",
                palette: PaletteId::Reflectivity,
                levels: PaletteId::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "1km AGL Reflectivity",
                opts: ComputeOptsPatch::default(),
            },
            Self::ReflectivityUh => ProductRecipe {
                fill_var: "dbz_1000m_agl",
                fill_units: "dBZ",
                palette: PaletteId::Reflectivity,
                levels: PaletteId::Reflectivity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "1km AGL Reflectivity (dBZ), 1h Max Updraft Helicity > 50 (m²/s²)",
                opts: ComputeOptsPatch::default(),
            },
            Self::CloudTopTemp => ProductRecipe {
                fill_var: "ctt",
                fill_units: "degC",
                palette: PaletteId::SimIr,
                levels: PaletteId::SimIr.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Simulated IR Satellite (Brightness Temp degC)",
                opts: ComputeOptsPatch::default(),
            },
            Self::SlpWind10m => ProductRecipe {
                fill_var: "wspd10",
                fill_units: "knots",
                palette: PaletteId::Wind,
                levels: wind_10m_levels(),
                contour_overlays: vec![ContourRecipe {
                    var: "slp",
                    units: "hPa",
                    levels: slp_contour_levels(),
                    color: Color::BLACK,
                    width_px: 1,
                    opts: ComputeOptsPatch::default(),
                }],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: "U10",
                    v_var: "V10",
                    units: "m/s",
                    stride_x: 16,
                    stride_y: 16,
                    color: Color::BLACK,
                }),
                title_template: "Surface MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::SurfaceWind10m => ProductRecipe {
                fill_var: "wspd10",
                fill_units: "knots",
                palette: PaletteId::Wind,
                levels: wind_10m_levels(),
                contour_overlays: vec![ContourRecipe {
                    var: "slp",
                    units: "hPa",
                    levels: slp_contour_levels(),
                    color: Color::BLACK,
                    width_px: 1,
                    opts: ComputeOptsPatch::default(),
                }],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: "U10",
                    v_var: "V10",
                    units: "knots",
                    stride_x: 16,
                    stride_y: 16,
                    color: Color::BLACK,
                }),
                title_template: "Surface MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::U10Component => ProductRecipe {
                fill_var: "U10",
                fill_units: "knots",
                palette: PaletteId::WindComponent,
                levels: PaletteId::WindComponent.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "10m U Wind Component (green = backing, red = veering)",
                opts: ComputeOptsPatch::default(),
            },
            Self::V10Component => ProductRecipe {
                fill_var: "V10",
                fill_units: "knots",
                palette: PaletteId::WindComponent,
                levels: PaletteId::WindComponent.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "10m V Wind Component (green = negative, red = positive)",
                opts: ComputeOptsPatch::default(),
            },
            Self::T2 => ProductRecipe {
                fill_var: "T2",
                fill_units: "degF",
                palette: PaletteId::SurfaceTemperature,
                levels: PaletteId::SurfaceTemperature.default_levels(),
                contour_overlays: vec![ContourRecipe {
                    var: "slp",
                    units: "hPa",
                    levels: slp_contour_levels(),
                    color: Color::BLACK,
                    width_px: 1,
                    opts: ComputeOptsPatch::default(),
                }],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: "U10",
                    v_var: "V10",
                    units: "knots",
                    stride_x: 16,
                    stride_y: 16,
                    color: Color::BLACK,
                }),
                title_template: "Surface Temperature (°F), MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::Td2 => ProductRecipe {
                fill_var: "dp2m",
                fill_units: "degF",
                palette: PaletteId::SurfaceDewpoint,
                levels: PaletteId::SurfaceDewpoint.default_levels(),
                contour_overlays: vec![ContourRecipe {
                    var: "slp",
                    units: "hPa",
                    levels: (980..=1032).step_by(2).map(|value| value as f32).collect(),
                    color: Color::BLACK,
                    width_px: 1,
                    opts: ComputeOptsPatch::default(),
                }],
                barb_overlay: Some(WindBarbRecipe {
                    u_var: "U10",
                    v_var: "V10",
                    units: "knots",
                    stride_x: 16,
                    stride_y: 16,
                    color: Color::BLACK,
                }),
                title_template: "Surface Dewpoint (°F), MSLP (mb), 10m AGL Wind (kt)",
                opts: ComputeOptsPatch::default(),
            },
            Self::Rh2 => ProductRecipe {
                fill_var: "rh2m",
                fill_units: "%",
                palette: PaletteId::RelativeHumidity,
                levels: PaletteId::RelativeHumidity.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "2 m Relative Humidity",
                opts: ComputeOptsPatch::default(),
            },
            Self::Pwat => ProductRecipe {
                fill_var: "pw",
                fill_units: "mm",
                palette: PaletteId::Precipitation,
                levels: range_i32(0, 75, 5),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Precipitable Water",
                opts: ComputeOptsPatch::default(),
            },
            Self::PrecipAccum => ProductRecipe {
                fill_var: "precip_accum",
                fill_units: "mm",
                palette: PaletteId::Precipitation,
                levels: PaletteId::Precipitation.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Accumulated Precipitation",
                opts: ComputeOptsPatch::default(),
            },
            Self::UpdraftHelicity => ProductRecipe {
                fill_var: NATIVE_OR_COMPUTED_UH_VAR,
                fill_units: "m2/s2",
                palette: PaletteId::Uh,
                levels: vec![25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0],
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Updraft Helicity (m²/s²)",
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
            Self::LapseRate700500 => ProductRecipe {
                fill_var: "lapse_rate_700_500",
                fill_units: "",
                palette: PaletteId::LapseRate,
                levels: PaletteId::LapseRate.default_levels(),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "700-500mb Lapse Rate (degC/km)",
                opts: ComputeOptsPatch::default(),
            },
            Self::LapseRate03 => ProductRecipe {
                fill_var: "lapse_rate_0_3km",
                fill_units: "",
                palette: PaletteId::LapseRate,
                levels: vec![4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "0-3 km Lapse Rate",
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
                palette: PaletteId::Stp,
                levels: range_i32(0, 100, 5),
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Fosberg Fire Weather Index",
                opts: ComputeOptsPatch::default(),
            },
            Self::Haines => ProductRecipe {
                fill_var: "haines",
                fill_units: "",
                palette: PaletteId::Stp,
                levels: vec![2.0, 3.0, 4.0, 5.0, 6.0],
                contour_overlays: Vec::new(),
                barb_overlay: None,
                title_template: "Haines Index",
                opts: ComputeOptsPatch::default(),
            },
            Self::Hdw => ProductRecipe {
                fill_var: "hdw",
                fill_units: "",
                palette: PaletteId::Wind,
                levels: range_i32(0, 1000, 50),
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
            Self::Wind250 => pressure_wind_recipe(250, "250 mb Height (dam), Wind (kt)"),
            Self::Height300Wind => {
                pressure_height_wind_recipe(300, "300 mb Height (dam), Wind (kt)")
            }
            Self::Temp300Wind => {
                pressure_temp_wind_recipe(300, "300 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind300 => pressure_wind_recipe(300, "300 mb Height (dam), Wind (kt)"),
            Self::Height500Wind => {
                pressure_height_wind_recipe(500, "500 mb Height (dam), Wind (kt)")
            }
            Self::Temp500Wind => {
                pressure_temp_wind_recipe(500, "500 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Wind500 => pressure_wind_recipe(500, "500 mb Height (dam), Wind (kt)"),
            Self::Vort500Wind => ProductRecipe {
                fill_var: "avo_500mb",
                fill_units: "s-1",
                palette: PaletteId::Vorticity,
                levels: vec![
                    0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00040, 0.00050,
                ],
                contour_overlays: vec![height_contours("height_500mb", range_i32(480, 600, 6))],
                barb_overlay: Some(pressure_barbs(500)),
                title_template: "500 hPa Absolute Vorticity, Height, and Wind",
                opts: ComputeOptsPatch::default(),
            },
            Self::Omega500 => ProductRecipe {
                fill_var: "omega_500mb",
                fill_units: "Pa/s",
                palette: PaletteId::Temperature,
                levels: vec![
                    -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
                ],
                contour_overlays: vec![height_contours("height_500mb", range_i32(480, 600, 6))],
                barb_overlay: Some(pressure_barbs(500)),
                title_template: "500 hPa Omega, Height, and Wind",
                opts: ComputeOptsPatch::default(),
            },
            Self::Temp700Wind => {
                pressure_temp_wind_recipe(700, "700 mb Temperature (degC), Height (dam), Wind (kt)")
            }
            Self::Height700Wind => {
                pressure_height_wind_recipe(700, "700 mb Height (dam), Wind (kt)")
            }
            Self::Rh700Wind => ProductRecipe {
                fill_var: "rh_700mb",
                fill_units: "%",
                palette: PaletteId::RelativeHumidity,
                levels: PaletteId::RelativeHumidity.default_levels(),
                contour_overlays: vec![height_contours("height_700mb", range_i32(240, 330, 6))],
                barb_overlay: Some(pressure_barbs(700)),
                title_template: "700 hPa Relative Humidity, Height, and Wind",
                opts: ComputeOptsPatch::default(),
            },
            Self::Height850Wind => {
                pressure_height_wind_recipe(850, "850 mb Height (dam), Wind (kt)")
            }
            Self::Temp850Wind => {
                pressure_temp_wind_recipe(850, "850 mb Temperature (degC), Height (dam), Wind (kt)")
            }
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
    WrfProduct::Srh01,
    WrfProduct::Srh03,
    WrfProduct::EffectiveSrh,
    WrfProduct::Shear01,
    WrfProduct::Shear06,
    WrfProduct::Ebwd,
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
    WrfProduct::Reflectivity,
    WrfProduct::Reflectivity1km,
    WrfProduct::ReflectivityUh,
    WrfProduct::UpdraftHelicity,
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
    WrfProduct::Lcl,
    WrfProduct::Lfc,
    WrfProduct::El,
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
    WrfProduct::Omega500,
    WrfProduct::Temp700Wind,
    WrfProduct::Height700Wind,
    WrfProduct::Rh700Wind,
    WrfProduct::Height850Wind,
    WrfProduct::Temp850Wind,
    WrfProduct::Wind850,
    WrfProduct::CloudTopTemp,
];

pub fn default_product_suite() -> &'static [WrfProduct] {
    DEFAULT_PRODUCT_SUITE
}

#[derive(Debug, Clone)]
pub struct ProductRecipe {
    pub fill_var: &'static str,
    pub fill_units: &'static str,
    pub palette: PaletteId,
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
    pub opts: ComputeOptsPatch,
}

#[derive(Debug, Clone)]
pub struct WindBarbRecipe {
    pub u_var: &'static str,
    pub v_var: &'static str,
    pub units: &'static str,
    pub stride_x: usize,
    pub stride_y: usize,
    pub color: Color,
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

pub fn parse_product(name: &str) -> ProductResult<WrfProduct> {
    WrfProduct::from_name(name).ok_or_else(|| ProductError::UnknownProduct(name.to_string()))
}

pub fn build_product_request(
    file: &WrfFile,
    product: WrfProduct,
    timeidx: Option<usize>,
) -> ProductResult<MapRenderRequest> {
    let t = timeidx.unwrap_or(0);
    let recipe = product.recipe();
    let fill = build_recipe_field(file, recipe.fill_var, recipe.fill_units, &recipe.opts, t)?;
    let scale = recipe
        .palette
        .scale(recipe.levels.clone(), ExtendMode::Both);
    let mut request = MapRenderRequest::new(fill, scale);
    request.title = Some(recipe.title_template.to_string());
    request.subtitle_left = wrf_time_subtitle(file, t);
    request.subtitle_right = Some(wrf_source_subtitle(file));
    let (width, height) = product_render_size();
    request.width = width;
    request.height = height;
    request.colorbar_orientation = ColorbarOrientation::VerticalRight;
    request.domain_frame = Some(model_data_domain_frame());
    request.cbar_tick_step = product_tick_step(product);
    request.cbar_ticks = product_tick_values(product);
    request.visual_mode = product.visual_mode();
    request.raster_sample_mode = RasterSampleMode::Linear;
    request.render_density = RenderDensity {
        fill: LevelDensity::default(),
        palette_multiplier: 1,
    };
    request.legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Stepped,
    };
    apply_projected_map(file, t, &mut request)?;

    if product == WrfProduct::ReflectivityUh {
        let uh = build_recipe_field(
            file,
            "uhel_0_3km_1h_max",
            "m2/s2",
            &ComputeOptsPatch::default(),
            t,
        )?;
        let uh_track = build_uh_track_overlay_field(&uh)?;
        apply_reflectivity_uh_rgba(&recipe, &uh_track, &mut request)?;
    }

    for contour in recipe.contour_overlays {
        let field = build_recipe_field(file, contour.var, contour.units, &contour.opts, t)?;
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
                labels: true,
                show_extrema: false,
            },
        )?;
    }

    if let Some(barbs) = recipe.barb_overlay {
        let u = build_recipe_field(
            file,
            barbs.u_var,
            barbs.units,
            &ComputeOptsPatch::default(),
            t,
        )?;
        let v = build_recipe_field(
            file,
            barbs.v_var,
            barbs.units,
            &ComputeOptsPatch::default(),
            t,
        )?;
        request = request.with_wind_barbs(
            &u,
            &v,
            WindBarbStyle {
                stride_x: barbs.stride_x,
                stride_y: barbs.stride_y,
                color: barbs.color,
                width: 1,
                length_px: 20.0,
            },
        )?;
    }

    Ok(request)
}

fn model_data_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        ..DomainFrame::model_data_default()
    }
}

fn apply_reflectivity_uh_rgba(
    recipe: &ProductRecipe,
    uh_track: &Field2D,
    request: &mut MapRenderRequest,
) -> ProductResult<()> {
    let scale = match recipe
        .palette
        .scale(recipe.levels.clone(), ExtendMode::Both)
    {
        ColorScale::Discrete(scale) => scale,
        _ => unreachable!("product palette scales are discrete"),
    };
    let nx = uh_track.grid.shape.nx;
    let ny = uh_track.grid.shape.ny;
    let pixels = request
        .field
        .values
        .iter()
        .zip(uh_track.values.iter())
        .enumerate()
        .map(|(idx, (&refl, &uh))| {
            let is_edge = is_uh_track_edge(&uh_track.values, nx, ny, idx);
            reflectivity_uh_pixel(&scale, refl, uh, is_edge)
        })
        .collect();
    request.set_rgba_grid(RgbaGridField::new(request.field.grid.clone(), pixels)?);
    Ok(())
}

fn build_uh_track_overlay_field(uh: &Field2D) -> ProductResult<Field2D> {
    Ok(Field2D::new(
        ProductKey::named("uhel_0_3km_1h_max_track"),
        uh.units.clone(),
        uh.grid.clone(),
        uh.values.clone(),
    )?)
}

fn reflectivity_uh_pixel(
    scale: &DiscreteColorScale,
    refl: f32,
    uh: f32,
    is_track_edge: bool,
) -> Color {
    let track = uh_track_color(uh, is_track_edge);
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

fn uh_track_color(uh: f32, is_track_edge: bool) -> Color {
    if !uh.is_finite() || uh < UH_TRACK_THRESHOLD {
        return Color::TRANSPARENT;
    }
    if is_track_edge {
        Color::rgba(0, 0, 0, 255)
    } else {
        Color::rgba(0, 0, 0, UH_TRACK_FILL_ALPHA)
    }
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

fn is_uh_track_edge(values: &[f32], nx: usize, ny: usize, idx: usize) -> bool {
    if nx == 0 || ny == 0 || idx >= values.len() || !is_active_uh_track(values[idx]) {
        return false;
    }
    let x = idx % nx;
    let y = idx / nx;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = x as isize + dx;
            let yy = y as isize + dy;
            if xx < 0 || yy < 0 || xx >= nx as isize || yy >= ny as isize {
                return true;
            }
            let neighbor_idx = yy as usize * nx + xx as usize;
            if neighbor_idx >= values.len() || !is_active_uh_track(values[neighbor_idx]) {
                return true;
            }
        }
    }
    false
}

fn is_active_uh_track(value: f32) -> bool {
    value.is_finite() && value >= UH_TRACK_THRESHOLD
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
        | WrfProduct::Mucape => Some(500.0),
        WrfProduct::Ncape => Some(250.0),
        WrfProduct::Srh01 | WrfProduct::Srh03 | WrfProduct::EffectiveSrh => Some(100.0),
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
        | WrfProduct::Temp700Wind
        | WrfProduct::Temp850Wind
        | WrfProduct::Reflectivity
        | WrfProduct::Reflectivity1km
        | WrfProduct::ReflectivityUh => Some(5.0),
        WrfProduct::Td2 => Some(10.0),
        WrfProduct::T2 => Some(10.0),
        WrfProduct::LapseRate700500 | WrfProduct::LapseRate03 => Some(1.0),
        _ => None,
    }
}

fn product_tick_values(product: WrfProduct) -> Option<Vec<f64>> {
    match product {
        WrfProduct::StpEffective | WrfProduct::StpFixed => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0,
        ]),
        WrfProduct::Scp | WrfProduct::EcapeScp => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        ]),
        WrfProduct::Ehi | WrfProduct::EcapeEhi | WrfProduct::Tehi | WrfProduct::Tts => Some(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            24.0,
        ]),
        WrfProduct::Srh01 | WrfProduct::Srh03 | WrfProduct::EffectiveSrh => Some(vec![
            0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0,
            700.0, 800.0, 900.0, 1000.0, 1250.0, 1500.0,
        ]),
        WrfProduct::CloudTopTemp => Some(vec![
            -90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, 0.0, 20.0, 40.0,
        ]),
        WrfProduct::SlpWind10m | WrfProduct::SurfaceWind10m => Some(vec![
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
        ]),
        WrfProduct::T2 => Some(vec![
            -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ]),
        WrfProduct::U10Component | WrfProduct::V10Component => Some(vec![
            -75.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0,
        ]),
        _ => None,
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
    let request = build_product_request(file, product, timeidx)?;
    let image = render_image_with_style(&request, OPERATIONAL_FAST)?;
    save_rgba_png_profile_with_options(&image, path, &PngWriteOptions::default())?;
    Ok(())
}

fn build_recipe_field(
    file: &WrfFile,
    var: &str,
    units: &'static str,
    patch: &ComputeOptsPatch,
    timeidx: usize,
) -> ProductResult<Field2D> {
    if var == "precip_accum" {
        return build_precip_accum_field(file, timeidx);
    }
    if var == "uhel_0_3km_1h_max" {
        return build_uhel_0_3km_1h_max_field(file, timeidx);
    }
    if var == NATIVE_OR_COMPUTED_UH_VAR {
        return build_native_or_computed_uhel_field(file, timeidx);
    }
    if let Some((base_var, height_m)) = parse_height_level_var(var) {
        return build_height_level_field(file, var, base_var, height_m, units, patch, timeidx);
    }
    if let Some((base_var, level_hpa)) = parse_pressure_level_var(var) {
        return build_pressure_level_field(file, var, base_var, level_hpa, units, patch, timeidx);
    }
    let output = getvar(file, var, Some(timeidx), &patch.apply(units))?;
    output_to_field(file, var, output, timeidx)
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
        .map(|(convective, grid_scale)| convective + grid_scale)
        .collect();
    output_to_field(
        file,
        "precip_accum",
        VarOutput {
            data,
            shape: vec![file.ny, file.nx],
            units: "mm".to_string(),
            description: "Accumulated precipitation (RAINC + RAINNC)".to_string(),
        },
        timeidx,
    )
}

fn build_uhel_0_3km_1h_max_field(file: &WrfFile, timeidx: usize) -> ProductResult<Field2D> {
    let mut max_values: Option<Vec<f64>> = None;

    for idx in one_hour_window_indices(file, timeidx) {
        accumulate_uhel_max(&mut max_values, file, idx)?;
    }

    if let Some(current) = valid_time_for_index(file, timeidx) {
        for (path, indices) in sibling_one_hour_window_paths(file, current) {
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

fn sibling_one_hour_window_paths(
    file: &WrfFile,
    current: WrfTimestamp,
) -> Vec<(PathBuf, Vec<usize>)> {
    let Some(parent) = file.path.parent() else {
        return Vec::new();
    };
    let current_minutes = timestamp_minutes(current);
    let current_path = normalize_path_for_compare(&file.path);
    let current_prefix = wrfout_filename_prefix(&file.path);
    let Ok(entries) = fs::read_dir(parent) else {
        return Vec::new();
    };

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
    paths
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

fn valid_time_for_index(file: &WrfFile, timeidx: usize) -> Option<WrfTimestamp> {
    file.times()
        .ok()?
        .get(timeidx)
        .and_then(|value| parse_wrf_timestamp(value.trim()))
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
            OPERATIONAL_FAST,
            request.colorbar_orientation,
        );
    let mut options = ProjectedMapBuildOptions::full_domain(target_ratio);
    if let Ok(projection) = WrfProjection::from_file(file) {
        options = options.with_projection(convert_projection(projection));
    }
    options = options.with_basemap_detail(basemap_detail_for_bounds(bounds));
    options.domain.pad_fraction = 0.02;
    let projected = build_projected_map_with_options(&grid.lat_deg, &grid.lon_deg, &options)
        .map_err(|err| ProductError::Projection(err.to_string()))?;
    request.apply_projected_map(&projected);
    Ok(())
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
    let lon_span = (bounds.1 - bounds.0).abs();
    let lat_span = (bounds.3 - bounds.2).abs();
    if lon_span > 90.0 || lat_span > 45.0 {
        BasemapDetail::Global
    } else if lon_span > 20.0 || lat_span > 15.0 {
        BasemapDetail::Broad
    } else {
        BasemapDetail::Regional
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
        palette: PaletteId::Cape,
        levels: PaletteId::Cape.default_levels(),
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
    palette: PaletteId,
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
    palette: PaletteId,
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
    palette: PaletteId,
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
        palette: PaletteId::Cape,
        levels: range_i32(-300, 0, 25),
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
    ProductRecipe {
        fill_var,
        fill_units: "knots",
        palette: PaletteId::Wind,
        levels: PaletteId::Wind.default_levels(),
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
        palette: PaletteId::Grayscale,
        levels,
        contour_overlays: Vec::new(),
        barb_overlay: None,
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_height_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    ProductRecipe {
        fill_var: pressure_field_name("wspd", level_hpa),
        fill_units: "knots",
        palette: PaletteId::Wind,
        levels: pressure_wind_levels(level_hpa),
        contour_overlays: vec![height_contours(
            pressure_field_name("height", level_hpa),
            pressure_height_contour_levels(level_hpa),
        )],
        barb_overlay: Some(pressure_barbs(level_hpa)),
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_temp_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    ProductRecipe {
        fill_var: pressure_field_name("tc", level_hpa),
        fill_units: "degC",
        palette: PaletteId::Temperature,
        levels: pressure_temperature_levels(level_hpa),
        contour_overlays: vec![height_contours(
            pressure_field_name("height", level_hpa),
            pressure_height_contour_levels(level_hpa),
        )],
        barb_overlay: Some(pressure_barbs(level_hpa)),
        title_template,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_wind_recipe(level_hpa: u16, title_template: &'static str) -> ProductRecipe {
    ProductRecipe {
        fill_var: pressure_field_name("wspd", level_hpa),
        fill_units: "knots",
        palette: PaletteId::Wind,
        levels: pressure_wind_levels(level_hpa),
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

fn pressure_wind_levels(level_hpa: u16) -> Vec<f32> {
    match level_hpa {
        200 => range_i32(50, 200, 5),
        250 => range_i32(50, 180, 5),
        300 => range_i32(40, 170, 5),
        500 => range_i32(20, 140, 5),
        700 => range_i32(15, 100, 5),
        850 => range_i32(15, 80, 5),
        _ => PaletteId::Wind.default_levels(),
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
        _ => PaletteId::Grayscale.default_levels(),
    }
}

fn height_contours(var: &'static str, levels: Vec<f32>) -> ContourRecipe {
    ContourRecipe {
        var,
        units: "dam",
        levels,
        color: Color::BLACK,
        width_px: 1,
        opts: ComputeOptsPatch::default(),
    }
}

fn pressure_barbs(level_hpa: u16) -> WindBarbRecipe {
    WindBarbRecipe {
        u_var: pressure_field_name("ua", level_hpa),
        v_var: pressure_field_name("va", level_hpa),
        units: "knots",
        stride_x: 10,
        stride_y: 10,
        color: Color::BLACK,
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
        ("rh", 700) => "rh_700mb",
        ("avo", 500) => "avo_500mb",
        ("omega", 500) => "omega_500mb",
        _ => panic!("unsupported pressure product field {field}_{level_hpa}mb"),
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

fn wind_10m_levels() -> Vec<f32> {
    range_i32(10, 70, 1)
}

fn slp_contour_levels() -> Vec<f32> {
    (960..=1040).step_by(2).map(|value| value as f32).collect()
}

fn scp_levels() -> Vec<f32> {
    range_i32(0, 70, 1)
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
        assert_eq!(u.fill_var, "U10");
        assert_eq!(u.fill_units, "knots");
        assert_eq!(u.palette, PaletteId::WindComponent);
        assert_eq!(u.levels.first().copied(), Some(-75.0));
        assert_eq!(u.levels.last().copied(), Some(75.0));
        assert!(u.barb_overlay.is_none());

        let v = WrfProduct::V10Component.recipe();
        assert_eq!(WrfProduct::V10Component.canonical_name(), "v10_component");
        assert_eq!(v.fill_var, "V10");
        assert_eq!(v.fill_units, "knots");
        assert_eq!(v.palette, PaletteId::WindComponent);
    }

    #[test]
    fn surface_dewpoint_palette_is_shifted_one_degree_down() {
        let td2 = WrfProduct::Td2.recipe();
        assert_eq!(td2.palette, PaletteId::SurfaceDewpoint);
        assert_eq!(td2.levels.first().copied(), Some(-40.0));
        assert_eq!(td2.levels.last().copied(), Some(90.0));

        let base = wrf_render::weather::weather_palette(WeatherPalette::Dewpoint);
        let shifted = surface_dewpoint_colors();
        assert_eq!(shifted.len(), base.len());
        assert_eq!(shifted[0], base[1]);
    }

    #[test]
    fn surface_temperature_uses_full_fahrenheit_table_and_overlays() {
        let t2 = WrfProduct::T2.recipe();
        assert_eq!(t2.fill_var, "T2");
        assert_eq!(t2.fill_units, "degF");
        assert_eq!(t2.palette, PaletteId::SurfaceTemperature);
        assert_eq!(t2.levels.first().copied(), Some(-60.0));
        assert_eq!(t2.levels.last().copied(), Some(120.0));
        assert_eq!(t2.contour_overlays[0].var, "slp");
        assert!(t2.barb_overlay.is_some());
        assert!(t2.title_template.contains("°F"));
    }

    #[test]
    fn surface_wind_products_use_10m_operational_scale() {
        for product in [WrfProduct::SurfaceWind10m, WrfProduct::SlpWind10m] {
            let recipe = product.recipe();
            assert_eq!(recipe.fill_var, "wspd10");
            assert_eq!(recipe.fill_units, "knots");
            assert_eq!(recipe.levels.first().copied(), Some(10.0));
            assert_eq!(recipe.levels.last().copied(), Some(70.0));
            assert_eq!(recipe.contour_overlays[0].var, "slp");
            assert!(recipe.barb_overlay.is_some());
        }
    }

    #[test]
    fn srh_products_use_1500_scale() {
        for product in [
            WrfProduct::Srh01,
            WrfProduct::Srh03,
            WrfProduct::EffectiveSrh,
        ] {
            let recipe = product.recipe();
            assert_eq!(recipe.palette, PaletteId::Srh);
            assert_eq!(recipe.fill_units, "m2/s2");
            assert_eq!(recipe.levels.first().copied(), Some(0.0));
            assert_eq!(recipe.levels.last().copied(), Some(1500.0));
        }
    }

    #[test]
    fn reflectivity_uh_combo_uses_1km_reflectivity_and_windowed_uh() {
        let recipe = WrfProduct::ReflectivityUh.recipe();
        assert_eq!(recipe.fill_var, "dbz_1000m_agl");
        assert_eq!(recipe.fill_units, "dBZ");
        assert_eq!(recipe.palette, PaletteId::Reflectivity);
        assert!(recipe.title_template.contains("1h Max Updraft Helicity"));
    }

    #[test]
    fn reflectivity_uh_combo_draws_uh_as_track_overlay() {
        let recipe = WrfProduct::ReflectivityUh.recipe();
        let ColorScale::Discrete(scale) = recipe.palette.scale(recipe.levels, ExtendMode::Both)
        else {
            panic!("expected discrete scale")
        };

        let clear_air = reflectivity_uh_pixel(&scale, 0.0, 10.0, false);
        assert_eq!(clear_air, Color::TRANSPARENT);

        let clear_air_track = reflectivity_uh_pixel(&scale, 0.0, 150.0, true);
        assert_ne!(clear_air_track, Color::TRANSPARENT);
        assert_eq!(clear_air_track, Color::rgba(0, 0, 0, 255));

        let clear_air_track_fill = reflectivity_uh_pixel(&scale, 0.0, 150.0, false);
        assert_ne!(clear_air_track_fill, Color::TRANSPARENT);
        assert_eq!(clear_air_track_fill.r, 0);
        assert_eq!(clear_air_track_fill.g, 0);
        assert_eq!(clear_air_track_fill.b, 0);
        assert_eq!(clear_air_track_fill.a, UH_TRACK_FILL_ALPHA);

        let storm_reflectivity = reflectivity_uh_pixel(&scale, 45.0, 150.0, false);
        let storm_reflectivity_base = sample_product_scale(&scale, 45.0);
        assert_ne!(storm_reflectivity, storm_reflectivity_base);
        assert!(storm_reflectivity.r < storm_reflectivity_base.r);
        assert!(storm_reflectivity.g < storm_reflectivity_base.g);
        assert!(storm_reflectivity.b < storm_reflectivity_base.b);

        let storm_track_edge = reflectivity_uh_pixel(&scale, 45.0, 150.0, true);
        assert_eq!(storm_track_edge, Color::rgba(0, 0, 0, 255));
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
}
