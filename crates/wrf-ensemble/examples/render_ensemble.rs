use std::env;
use std::path::PathBuf;

use wrf_ensemble::{parse_ensemble_stat, stat_requires_value, WrfEnsemble};
use wrf_products::parse_product;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let member_glob = args.next().ok_or(
        "usage: render_ensemble <member_glob> <product> <stat> <output.png> [timeidx] [value]",
    )?;
    let product_name = args.next().ok_or(
        "usage: render_ensemble <member_glob> <product> <stat> <output.png> [timeidx] [value]",
    )?;
    let stat_name = args.next().ok_or(
        "usage: render_ensemble <member_glob> <product> <stat> <output.png> [timeidx] [value]",
    )?;
    let output = args.next().ok_or(
        "usage: render_ensemble <member_glob> <product> <stat> <output.png> [timeidx] [value]",
    )?;
    let timeidx = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let value = args.next().map(|value| value.parse::<f64>()).transpose()?;

    if stat_requires_value(&stat_name) && value.is_none() {
        return Err(format!("stat `{stat_name}` requires a percentile or threshold value").into());
    }

    let product = parse_product(&product_name)?;
    let stat = parse_ensemble_stat(&stat_name, value)?;
    let ensemble = WrfEnsemble::from_glob(&member_glob)?;
    ensemble.render_product_png(product, stat, timeidx, PathBuf::from(&output))?;

    println!(
        "rendered {} {} from {} members -> {}",
        product.canonical_name(),
        stat.slug(),
        ensemble.len(),
        output
    );

    Ok(())
}
