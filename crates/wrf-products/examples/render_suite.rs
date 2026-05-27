use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use wrf_core::WrfFile;
use wrf_products::{
    default_product_suite, parse_product, product_specs_json, render_product_png_with_options,
    ProductRenderOptions, WrfProduct,
};

const USAGE: &str = "usage: render_suite [--previous-file WRFOUT]... [--history-dir DIR] [--bounds west,east,south,north] [--storm-center lat,lon,radius-km] [--products csv] [--timeidx idx] [--print-required-inputs [csv]] <wrfout> <output-dir> [timeidx] [products_csv]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = SuiteArgs::parse(env::args().skip(1))?;
    let products = match args.products_csv.as_deref() {
        Some(csv) => parse_product_list(csv)?,
        None => default_product_suite().to_vec(),
    };

    if args.print_required_inputs {
        println!("{}", product_specs_json(&products)?);
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
    options = options.with_history_files(args.history_files);
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
    history_files: Vec<PathBuf>,
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
                "--previous-file" => {
                    parsed
                        .history_files
                        .push(PathBuf::from(iter.next().ok_or(USAGE)?));
                }
                "--history-dir" => {
                    let dir = PathBuf::from(iter.next().ok_or(USAGE)?);
                    parsed.history_files.extend(collect_history_dir(&dir)?);
                    parsed.history_dir = Some(dir);
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
        parsed.history_files.sort();
        parsed.history_files.dedup();
        Ok(parsed)
    }
}

fn collect_history_dir(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
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
