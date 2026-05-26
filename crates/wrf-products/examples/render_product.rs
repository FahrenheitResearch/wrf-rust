use std::env;
use std::path::PathBuf;

use wrf_core::WrfFile;
use wrf_products::{parse_product, render_product_png_with_options, ProductRenderOptions};

const USAGE: &str =
    "usage: render_product [--history-dir DIR] <wrfout> <product> <output.png> [timeidx]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ProductArgs::parse(env::args().skip(1))?;
    let input = args.input.ok_or(USAGE)?;
    let product_name = args.product.ok_or(USAGE)?;
    let output = args.output.ok_or(USAGE)?;

    let product = parse_product(&product_name)?;
    let file = WrfFile::open(PathBuf::from(input))?;
    let options = ProductRenderOptions {
        history_dir: args.history_dir,
    };
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
