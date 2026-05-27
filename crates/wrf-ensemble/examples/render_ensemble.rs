use std::env;
use std::path::{Path, PathBuf};

use wrf_ensemble::{parse_ensemble_stat, stat_requires_value, WrfEnsemble};
use wrf_products::parse_product;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let member_source = args.next().ok_or(
        "usage: render_ensemble <member_glob_or_manifest.json> <product> <stat> <output.png> [timeidx] [value]",
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
    let ensemble = open_ensemble(&member_source)?;
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

fn open_ensemble(source: &str) -> Result<WrfEnsemble, Box<dyn std::error::Error>> {
    let path = Path::new(source);
    if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
        Ok(WrfEnsemble::from_manifest(path)?)
    } else {
        Ok(WrfEnsemble::from_glob(source)?)
    }
}
