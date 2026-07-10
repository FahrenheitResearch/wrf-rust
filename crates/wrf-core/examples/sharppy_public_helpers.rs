//! CLI bridge used only by the pinned SHARPpy differential acceptance harness.
//!
//! Each branch calls a public `wrf-core` helper. The Python side owns the
//! reference cases and verifies that the echoed case and diagnostic match.

use std::env;
use std::process;

const PROTOCOL: &str = "wrf-core-sharppy-public-helpers-v1";

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("{}", message.as_ref());
    process::exit(2);
}

fn parse_values(raw: &[String], expected: usize) -> Result<Vec<f64>, String> {
    if raw.len() != expected {
        return Err(format!(
            "expected {expected} numeric arguments, received {}",
            raw.len()
        ));
    }

    raw.iter()
        .enumerate()
        .map(|(index, value)| {
            let parsed = value
                .parse::<f64>()
                .map_err(|error| format!("argument {index} is not f64: {error}"))?;
            if parsed.is_finite() {
                Ok(parsed)
            } else {
                Err(format!("argument {index} must be finite"))
            }
        })
        .collect()
}

fn only_value(values: Vec<f64>) -> Result<f64, String> {
    if values.len() != 1 {
        return Err(format!(
            "public helper returned {} values instead of one",
            values.len()
        ));
    }
    let value = values[0];
    if value.is_finite() {
        Ok(value)
    } else {
        Err("public helper returned a non-finite value".to_string())
    }
}

fn evaluate(diagnostic: &str, raw: &[String]) -> Result<f64, String> {
    match diagnostic {
        "fixed_stp" => {
            let value = parse_values(raw, 4)?;
            only_value(wrf_core::met::composite::compute_stp(
                &[value[0]],
                &[value[1]],
                &[value[2]],
                &[value[3]],
            ))
        }
        "scp_neutral_cin" => {
            let value = parse_values(raw, 4)?;
            only_value(wrf_core::met::composite::supercell_composite_parameter(
                &[value[0]],
                &[value[1]],
                &[value[2]],
                &[value[3]],
                1,
                1,
            ))
        }
        "ship" => {
            let value = parse_values(raw, 6)?;
            only_value(wrf_core::met::composite::significant_hail_parameter(
                &[value[0]],
                &[value[1]],
                &[value[2]],
                &[value[3]],
                &[value[4]],
                &[value[5]],
                1,
                1,
            ))
        }
        "dcp_mean_wind" => {
            let value = parse_values(raw, 4)?;
            only_value(
                wrf_core::met::composite::derecho_composite_parameter_from_mean_wind(
                    &[value[0]],
                    &[value[1]],
                    &[value[2]],
                    &[value[3]],
                    1,
                    1,
                ),
            )
        }
        "critical_angle" => {
            let value = parse_values(raw, 6)?;
            let u_shear = value[4] - value[2];
            let v_shear = value[5] - value[3];
            only_value(wrf_core::met::composite::critical_angle(
                &[value[0]],
                &[value[1]],
                &[value[2]],
                &[value[3]],
                &[u_shear],
                &[v_shear],
                1,
                1,
            ))
        }
        _ => Err(format!("unknown diagnostic {diagnostic:?}")),
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.as_slice() == ["--provenance"] {
        println!("{PROTOCOL}");
        return;
    }
    if args.len() < 2 {
        fail("usage: sharppy_public_helpers <case-id> <diagnostic> <values...>");
    }

    let case_id = &args[0];
    let diagnostic = &args[1];
    if case_id.is_empty() || case_id.contains(['\t', '\r', '\n']) {
        fail("case id must be non-empty and contain no control separators");
    }
    if diagnostic.contains(['\t', '\r', '\n']) {
        fail("diagnostic must contain no control separators");
    }

    let value = evaluate(diagnostic, &args[2..]).unwrap_or_else(|error| fail(error));
    println!("{PROTOCOL}\t{case_id}\t{diagnostic}\t{value:.17}");
}
