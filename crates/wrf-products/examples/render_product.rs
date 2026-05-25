use std::env;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{parse_product, render_product_png};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: render_product <wrfout> <product> <output.png> [timeidx]")?;
    let product_name = args
        .next()
        .ok_or("usage: render_product <wrfout> <product> <output.png> [timeidx]")?;
    let output = args
        .next()
        .ok_or("usage: render_product <wrfout> <product> <output.png> [timeidx]")?;
    let timeidx = args.next().map(|value| value.parse()).transpose()?;

    let product = parse_product(&product_name)?;
    let file = WrfFile::open(PathBuf::from(input))?;
    render_product_png(&file, product, timeidx, PathBuf::from(output))?;

    Ok(())
}
