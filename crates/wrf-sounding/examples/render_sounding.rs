use std::env;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_sounding::{extract_and_write_sounding_png, BoxSoundingMethod, SoundingSelection};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 4 {
        print_usage();
        std::process::exit(2);
    }

    let wrfout = PathBuf::from(args.remove(0));
    let output = PathBuf::from(args.remove(0));
    let timeidx = args.remove(0).parse::<usize>()?;
    let selection = parse_selection(&args)?;

    let file = WrfFile::open(&wrfout)?;
    extract_and_write_sounding_png(&file, selection, Some(timeidx), &output)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn parse_selection(args: &[String]) -> Result<SoundingSelection, Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        Some("--ij") => {
            let value = args.get(1).ok_or("--ij requires i,j")?;
            let (i, j) = parse_pair_usize(value)?;
            Ok(SoundingSelection::PointIj { i, j })
        }
        Some("--latlon") => {
            let value = args.get(1).ok_or("--latlon requires lat,lon")?;
            let (lat, lon) = parse_pair_f64(value)?;
            Ok(SoundingSelection::PointLatLon { lat, lon })
        }
        Some("--box") => {
            let value = args.get(1).ok_or("--box requires south,west,north,east")?;
            let method = parse_method(args.get(3).map(String::as_str).unwrap_or("mean"))?;
            let parts = parse_four_f64(value)?;
            Ok(SoundingSelection::BoxLatLon {
                south: parts[0],
                west: parts[1],
                north: parts[2],
                east: parts[3],
                method,
            })
        }
        _ => Err("expected --ij, --latlon, or --box".into()),
    }
}

fn parse_method(value: &str) -> Result<BoxSoundingMethod, Box<dyn std::error::Error>> {
    match value {
        "mean" | "mean_profile" => Ok(BoxSoundingMethod::MeanProfile),
        "median" | "median_profile" => Ok(BoxSoundingMethod::MedianProfile),
        "most_unstable" | "most_unstable_column" | "mu" => {
            Ok(BoxSoundingMethod::MostUnstableColumn)
        }
        _ => Err(format!("unknown box sounding method `{value}`").into()),
    }
}

fn parse_pair_f64(value: &str) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err(format!("expected two comma-separated values, got `{value}`").into());
    }
    Ok((parts[0].parse()?, parts[1].parse()?))
}

fn parse_pair_usize(value: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err(format!("expected two comma-separated values, got `{value}`").into());
    }
    Ok((parts[0].parse()?, parts[1].parse()?))
}

fn parse_four_f64(value: &str) -> Result<[f64; 4], Box<dyn std::error::Error>> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err(format!("expected four comma-separated values, got `{value}`").into());
    }
    Ok([
        parts[0].parse()?,
        parts[1].parse()?,
        parts[2].parse()?,
        parts[3].parse()?,
    ])
}

fn print_usage() {
    eprintln!(
        "usage:\n  render_sounding <wrfout> <output.png> <timeidx> --latlon lat,lon\n  render_sounding <wrfout> <output.png> <timeidx> --ij i,j\n  render_sounding <wrfout> <output.png> <timeidx> --box south,west,north,east [--method mean|median|most_unstable]"
    );
}
