use std::env;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{parse_product, render_product_png_with_options, ProductRenderOptions};

const USAGE: &str =
    "usage: render_product [--history-dir DIR] [--bounds west,east,south,north] [--storm-center lat,lon,radius-km] <wrfout> <product> <output.png> [timeidx]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ProductArgs::parse(env::args().skip(1))?;
    let input = args.input.ok_or(USAGE)?;
    let product_name = args.product.ok_or(USAGE)?;
    let output = args.output.ok_or(USAGE)?;

    let product = parse_product(&product_name)?;
    let file = WrfFile::open(PathBuf::from(input))?;
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
    render_product_png_with_options(
        &file,
        product,
        args.timeidx,
        PathBuf::from(output),
        &options,
    )?;

    Ok(())
}

#[derive(Debug, Default)]
struct ProductArgs {
    input: Option<String>,
    product: Option<String>,
    output: Option<String>,
    timeidx: Option<usize>,
    history_dir: Option<PathBuf>,
    geographic_bounds: Option<(f64, f64, f64, f64)>,
    storm_center: Option<(f64, f64, f64)>,
}

impl ProductArgs {
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
                "--help" | "-h" => return Err(USAGE.into()),
                _ if arg.starts_with("--") => {
                    return Err(format!("unknown option `{arg}`\n{USAGE}").into());
                }
                _ => positionals.push(arg),
            }
        }

        parsed.input = positionals.first().cloned();
        parsed.product = positionals.get(1).cloned();
        parsed.output = positionals.get(2).cloned();
        parsed.timeidx = positionals.get(3).map(|value| value.parse()).transpose()?;
        Ok(parsed)
    }
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
