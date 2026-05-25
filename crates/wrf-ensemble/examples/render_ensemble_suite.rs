use std::env;
use std::fs;
use std::path::PathBuf;

use wrf_ensemble::{parse_ensemble_stat, stat_requires_value, EnsembleStat, WrfEnsemble};
use wrf_products::{default_product_suite, parse_product, WrfProduct};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let member_glob = args.next().ok_or(
        "usage: render_ensemble_suite <member_glob> <output-dir> [timeidx] [stats_csv] [products_csv]",
    )?;
    let output_dir = args.next().ok_or(
        "usage: render_ensemble_suite <member_glob> <output-dir> [timeidx] [stats_csv] [products_csv]",
    )?;
    let timeidx = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let stats = match args.next() {
        Some(csv) => parse_stats_csv(&csv)?,
        None => vec![EnsembleStat::Mean, EnsembleStat::StdDev],
    };
    let products = match args.next() {
        Some(csv) => parse_products_csv(&csv)?,
        None => default_product_suite().to_vec(),
    };

    fs::create_dir_all(&output_dir)?;
    let ensemble = WrfEnsemble::from_glob(&member_glob)?;
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for product in products {
        for stat in &stats {
            let output = PathBuf::from(&output_dir).join(format!(
                "{}_{}.png",
                product.canonical_name(),
                stat.slug()
            ));
            match ensemble.render_product_png(product, *stat, timeidx, &output) {
                Ok(()) => {
                    rendered += 1;
                    println!(
                        "rendered {} {} -> {}",
                        product.canonical_name(),
                        stat.slug(),
                        output.display()
                    );
                }
                Err(err) => {
                    failed += 1;
                    eprintln!("failed {} {}: {err}", product.canonical_name(), stat.slug());
                }
            }
        }
    }

    println!(
        "ensemble suite complete: {rendered} rendered, {failed} failed from {} members",
        ensemble.len()
    );
    if rendered == 0 {
        return Err("no ensemble products rendered".into());
    }
    Ok(())
}

fn parse_stats_csv(csv: &str) -> Result<Vec<EnsembleStat>, Box<dyn std::error::Error>> {
    csv.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(parse_stat_token)
        .collect()
}

fn parse_stat_token(token: &str) -> Result<EnsembleStat, Box<dyn std::error::Error>> {
    let (name, value) = if let Some((name, value)) = token.split_once(':') {
        (name, Some(value.parse::<f64>()?))
    } else {
        (token, None)
    };
    if stat_requires_value(name) && value.is_none() {
        return Err(format!("stat `{name}` requires `name:value` in stats_csv").into());
    }
    Ok(parse_ensemble_stat(name, value)?)
}

fn parse_products_csv(csv: &str) -> Result<Vec<WrfProduct>, Box<dyn std::error::Error>> {
    csv.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| Ok(parse_product(name)?))
        .collect()
}
