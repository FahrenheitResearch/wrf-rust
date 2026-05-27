use std::env;
use std::fs;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{
    default_product_suite, parse_product, product_input_contract, product_visual_contract_summary,
    render_product_png_with_options, ProductRenderOptions, WrfProduct,
};

const USAGE: &str = "usage: render_suite [--history-dir DIR] [--bounds west,east,south,north] [--storm-center lat,lon,radius-km] [--products csv] [--timeidx idx] [--print-required-inputs [csv]] <wrfout> <output-dir> [timeidx] [products_csv]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = SuiteArgs::parse(env::args().skip(1))?;
    let products = match args.products_csv.as_deref() {
        Some(csv) => parse_product_list(csv)?,
        None => default_product_suite().to_vec(),
    };

    if args.print_required_inputs {
        print_required_inputs(&products);
        return Ok(());
    }

    let input = args.input.ok_or(USAGE)?;
    let output_dir = args.output_dir.ok_or(USAGE)?;
    fs::create_dir_all(&output_dir)?;
    let file = WrfFile::open(PathBuf::from(input))?;
    let mut rendered = 0usize;
    let mut failed = 0usize;
    let mut options = ProductRenderOptions::default();
    if let Some(history_dir) = args.history_dir {
        options = options.with_history_dir(history_dir);
    }
    if let Some((west, east, south, north)) = args.geographic_bounds {
        options = options.with_geographic_bounds(west, east, south, north);
    }
    if let Some((lat, lon, radius_km)) = args.storm_center {
        options = options.with_storm_center(lat, lon, radius_km);
    }

    for product in products {
        let output = PathBuf::from(&output_dir).join(format!("{}.png", product.canonical_name()));
        match render_product_png_with_options(&file, product, args.timeidx, &output, &options) {
            Ok(()) => {
                rendered += 1;
                println!(
                    "rendered {} -> {}",
                    product.canonical_name(),
                    output.display()
                );
            }
            Err(err) => {
                failed += 1;
                eprintln!("failed {}: {err}", product.canonical_name());
            }
        }
    }

    println!("suite complete: {rendered} rendered, {failed} failed");
    if rendered == 0 {
        return Err("no products rendered".into());
    }
    Ok(())
}

#[derive(Debug, Default)]
struct SuiteArgs {
    input: Option<String>,
    output_dir: Option<String>,
    timeidx: Option<usize>,
    products_csv: Option<String>,
    history_dir: Option<PathBuf>,
    geographic_bounds: Option<(f64, f64, f64, f64)>,
    storm_center: Option<(f64, f64, f64)>,
    print_required_inputs: bool,
}

impl SuiteArgs {
    fn parse<I>(args: I) -> Result<Self, Box<dyn std::error::Error>>
    where
        I: IntoIterator<Item = String>,
    {
        let mut parsed = Self::default();
        let mut positionals = Vec::new();
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--history-dir" => {
                    parsed.history_dir = Some(PathBuf::from(iter.next().ok_or(USAGE)?));
                }
                "--bounds" => {
                    parsed.geographic_bounds = Some(parse_bounds(&iter.next().ok_or(USAGE)?)?);
                }
                "--storm-center" => {
                    parsed.storm_center = Some(parse_storm_center(&iter.next().ok_or(USAGE)?)?);
                }
                "--products" => {
                    parsed.products_csv = Some(iter.next().ok_or(USAGE)?);
                }
                "--timeidx" => {
                    parsed.timeidx = Some(iter.next().ok_or(USAGE)?.parse()?);
                }
                "--print-required-inputs" => {
                    parsed.print_required_inputs = true;
                }
                "--help" | "-h" => {
                    return Err(USAGE.into());
                }
                _ if arg.starts_with("--") => {
                    return Err(format!("unknown option `{arg}`\n{USAGE}").into());
                }
                _ => positionals.push(arg),
            }
        }

        if parsed.print_required_inputs {
            if parsed.products_csv.is_none() {
                parsed.products_csv = positionals.first().cloned();
            }
            return Ok(parsed);
        }

        parsed.input = positionals.first().cloned();
        parsed.output_dir = positionals.get(1).cloned();
        if parsed.timeidx.is_none() {
            parsed.timeidx = positionals.get(2).map(|value| value.parse()).transpose()?;
        }
        if parsed.products_csv.is_none() {
            parsed.products_csv = positionals.get(3).cloned();
        }
        Ok(parsed)
    }
}

fn parse_product_list(csv: &str) -> Result<Vec<WrfProduct>, Box<dyn std::error::Error>> {
    csv.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| Ok(parse_product(name)?))
        .collect()
}

fn parse_bounds(value: &str) -> Result<(f64, f64, f64, f64), Box<dyn std::error::Error>> {
    let parts = parse_csv_numbers(value, 4)?;
    Ok((parts[0], parts[1], parts[2], parts[3]))
}

fn parse_storm_center(value: &str) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let parts = parse_csv_numbers(value, 3)?;
    Ok((parts[0], parts[1], parts[2]))
}

fn parse_csv_numbers(value: &str, expected: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let parts: Vec<f64> = value
        .split(',')
        .map(str::trim)
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if parts.len() != expected {
        return Err(format!("expected {expected} comma-separated numbers, got `{value}`").into());
    }
    Ok(parts)
}

fn print_required_inputs(products: &[WrfProduct]) {
    println!("required: current wrfout file passed on the command line");
    println!("default: single-file rendering; no sibling wrfout files are scanned");
    for product in products {
        let contract = product_input_contract(*product);
        let visual = product_visual_contract_summary(*product);
        println!(
            "product: {} | presentation=OPERATIONAL_FAST/{:?} | mode={:?} | sample={:?} | colorbar_pos={:?} | legend_mode={:?} | density=fill{}x/palette{}x/legend{}x | palette={:?} colors={} extend={:?} | units={} | colorbar={} | levels={} range={} interval={} | ticks={} | tick_step={} | legend={} | mask={:?} | frame={:?} source={} clear={} inset={} outline={} pad={} | contours={} | barbs={} | overlays={} | provenance={}",
            product.canonical_name(),
            visual.presentation_style,
            visual.visual_mode,
            visual.raster_sample_mode,
            visual.colorbar_orientation,
            visual.legend_mode,
            visual.render_density.fill.multiplier,
            visual.render_density.palette_multiplier,
            visual.legend_density.multiplier,
            visual.palette,
            visual.palette_color_count,
            visual.extend_mode,
            display_units(visual.fill_units),
            visual.colorbar_label.unwrap_or("unitless"),
            visual.level_count,
            format_level_range(visual.first_level, visual.last_level),
            format_optional_number(visual.level_interval),
            format_legend_thresholds(visual.legend_ticks.as_deref()),
            format_optional_number(visual.colorbar_tick_step),
            format_legend_thresholds(visual.legend_thresholds.as_deref()),
            visual.mask_policy,
            visual.frame_policy,
            format_optional_debug(visual.frame_source),
            format_optional_bool(visual.frame_clear_outside),
            format_optional_u32(visual.frame_inset_px),
            format_optional_u32(visual.frame_outline_width_px),
            format_number(visual.frame_padding_fraction),
            visual.contour_count,
            format_barbs(visual.barb_units, visual.barb_spacing_px),
            visual.overlay_count,
            visual.provenance_label,
        );
        if let Some(template) = visual.upper_air_template {
            println!(
                "  upper-air: {} hPa fill={:?} height_contours={} wind_barbs={}/{} {}",
                template.level_hpa,
                template.fill_role,
                template.height_contour_var,
                template.wind_u_var,
                template.wind_v_var,
                template.wind_units
            );
        }
        for contour in &visual.contours {
            println!(
                "  contour: {} {} levels={} range={} interval={} major_every={} labels={} label_every={} color={} halo_color={} widths={}/{} halo={} extrema={}",
                contour.var,
                display_units(contour.units),
                contour.level_count,
                format_level_range(contour.first_level, contour.last_level),
                format_optional_number(contour.minor_interval),
                contour.major_every,
                contour.labels,
                contour.label_every,
                format_color(contour.color),
                format_color(contour.halo_color),
                contour.width_px,
                contour.major_width_px,
                contour.halo_width_px,
                contour.show_extrema
            );
        }
        if let Some(barbs) = &visual.barbs {
            println!(
                "  barbs: {}/{} {} stride={}x{} spacing={:.0}px length={:.0}px color={} halo_color={} width={} halo={}",
                barbs.u_var,
                barbs.v_var,
                barbs.units,
                barbs.stride_x,
                barbs.stride_y,
                barbs.spacing_px,
                barbs.length_px,
                format_color(barbs.color),
                format_color(barbs.halo_color),
                barbs.width_px,
                barbs.halo_width_px
            );
        }
        for overlay in &visual.overlays {
            println!(
                "  overlay: {} source={} units={} thresholds={} fills={} edge_alpha={} history={}",
                overlay.label,
                overlay.source_var,
                display_units(overlay.units),
                format_legend_thresholds(Some(&overlay.threshold_bins)),
                overlay.fill_count,
                overlay.edge_color.a,
                overlay
                    .lookback_minutes
                    .map(|minutes| format!("{minutes}min"))
                    .unwrap_or_else(|| "none".to_string())
            );
        }
        if !visual.overlay_legend_titles.is_empty() {
            println!(
                "  overlay legends: {}",
                visual.overlay_legend_titles.join(", ")
            );
        }
        if !visual.source_semantics.is_empty() {
            println!(
                "  sources: {}",
                visual
                    .source_semantics
                    .iter()
                    .map(|source| source.label)
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        if let Some(history) = contract.optional_history {
            println!(
                "optional: {} can use {} DIR containing {}",
                product.canonical_name(),
                history.cli_flag,
                history.description
            );
        }
    }
}

fn display_units(units: &str) -> &str {
    if units.trim().is_empty() {
        "unitless"
    } else {
        units
    }
}

fn format_barbs(units: Option<&str>, spacing_px: Option<f64>) -> String {
    match (units, spacing_px) {
        (Some(units), Some(spacing)) => format!("{units} @ {spacing:.0}px"),
        (Some(units), None) => units.to_string(),
        _ => "none".to_string(),
    }
}

fn format_legend_thresholds(legend: Option<&[f64]>) -> String {
    let Some(legend) = legend else {
        return "none".to_string();
    };
    legend
        .iter()
        .map(|value| {
            if value.fract().abs() < 1.0e-9 {
                format!("{value:.0}")
            } else {
                format!("{value:.2}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn format_level_range(first: Option<f64>, last: Option<f64>) -> String {
    match (first, last) {
        (Some(first), Some(last)) => {
            format!("{}..{}", format_number(first), format_number(last))
        }
        _ => "none".to_string(),
    }
}

fn format_optional_number(value: Option<f64>) -> String {
    value
        .map(format_number)
        .unwrap_or_else(|| "varies".to_string())
}

fn format_optional_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_optional_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_optional_debug<T: std::fmt::Debug>(value: Option<T>) -> String {
    value
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| "none".to_string())
}

fn format_color(color: wrf_render::Color) -> String {
    format!("rgba({},{},{},{})", color.r, color.g, color.b, color.a)
}

fn format_number(value: f64) -> String {
    if value.fract().abs() < 1.0e-9 {
        format!("{value:.0}")
    } else {
        format!("{value:.2}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}
