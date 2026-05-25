use std::env;
use std::fs;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{default_product_suite, parse_product, render_product_png, WrfProduct};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: render_suite <wrfout> <output-dir> [timeidx] [products_csv]")?;
    let output_dir = args
        .next()
        .ok_or("usage: render_suite <wrfout> <output-dir> [timeidx] [products_csv]")?;
    let timeidx = args.next().map(|value| value.parse()).transpose()?;
    let products = match args.next() {
        Some(csv) => parse_product_list(&csv)?,
        None => default_product_suite().to_vec(),
    };

    fs::create_dir_all(&output_dir)?;
    let file = WrfFile::open(PathBuf::from(input))?;
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for product in products {
        let output = PathBuf::from(&output_dir).join(format!("{}.png", product.canonical_name()));
        match render_product_png(&file, product, timeidx, &output) {
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

fn parse_product_list(csv: &str) -> Result<Vec<WrfProduct>, Box<dyn std::error::Error>> {
    csv.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| Ok(parse_product(name)?))
        .collect()
}
