use std::env;
use std::path::PathBuf;

use wrf_core::{getvar, ComputeOpts, WrfFile};
use wrf_products::{parse_product, ProductRecipe};

const USAGE: &str = "usage: inspect_product_stats <wrfout> <product[,product...]> [timeidx]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args.next().ok_or(USAGE)?;
    let product_list = args.next().ok_or(USAGE)?;
    let timeidx = args.next().map(|value| value.parse()).transpose()?;

    let file = WrfFile::open(PathBuf::from(input))?;
    for product_name in product_list.split(',').filter(|name| !name.is_empty()) {
        let product = parse_product(product_name)?;
        let recipe = product.recipe();
        let opts = compute_opts_for_recipe(&recipe);
        let output = getvar(&file, recipe.fill_var, timeidx, &opts)?;
        let mut values: Vec<f64> = output
            .data
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        values.sort_by(|a, b| a.total_cmp(b));
        if values.is_empty() {
            println!("{product_name}: no finite values");
            continue;
        }

        let min = values[0];
        let max = values[values.len() - 1];
        let p90 = percentile(&values, 0.90);
        let p95 = percentile(&values, 0.95);
        let p99 = percentile(&values, 0.99);
        println!(
            "{product_name}: var={} units={} shape={:?} min={min:.2} p90={p90:.2} p95={p95:.2} p99={p99:.2} max={max:.2}",
            recipe.fill_var, output.units, output.shape
        );
    }

    Ok(())
}

fn compute_opts_for_recipe(recipe: &ProductRecipe) -> ComputeOpts {
    ComputeOpts {
        units: if recipe.fill_units.is_empty() {
            recipe.opts.units.map(str::to_string)
        } else {
            Some(recipe.fill_units.to_string())
        },
        parcel_type: recipe.opts.parcel_type.map(str::to_string),
        storm_motion_type: recipe.opts.storm_motion_type.map(str::to_string),
        layer_type: recipe.opts.layer_type.map(str::to_string),
        bottom_m: recipe.opts.bottom_m,
        top_m: recipe.opts.top_m,
        ..Default::default()
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len().saturating_sub(1) as f64;
    let idx = (last * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
