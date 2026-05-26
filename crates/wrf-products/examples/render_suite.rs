use std::env;
use std::fs;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{
    default_product_suite, parse_product, product_input_contract, render_product_png_with_options,
    ProductRenderOptions, WrfProduct,
};

const USAGE: &str = "usage: render_suite [--history-dir DIR] [--products csv] [--timeidx idx] [--print-required-inputs [csv]] <wrfout> <output-dir> [timeidx] [products_csv]";

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
    let options = ProductRenderOptions {
        history_dir: args.history_dir,
    };

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

fn print_required_inputs(products: &[WrfProduct]) {
    println!("required: current wrfout file passed on the command line");
    println!("default: single-file rendering; no sibling wrfout files are scanned");
    for product in products {
        let contract = product_input_contract(*product);
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
