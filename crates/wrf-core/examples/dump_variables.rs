//! Dump the diagnostic variable registry as JSON for the documentation site.
//!
//! Usage:
//!   cargo run -p wrf-core --example dump_variables -- [git-sha] > docs/data/variables.json
//!
//! The data fields (name, aliases, description, default units, dimensionality)
//! are taken directly from `wrf_core::variables::VARS` -- nothing is parsed.
//! The section label for each variable is recovered from the `// ── ... ──`
//! group comments in `variables.rs`; the scan is cross-checked so that every
//! registry entry maps to exactly one section, and the program exits nonzero
//! on any mismatch between the scan and the live registry.

use wrf_core::variables::{VarDim, VARS};

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Map every registry `name:` to the `// ── section ──` comment above it,
/// by scanning the same source file the registry is compiled from.
fn section_map() -> Vec<(String, String)> {
    let src = include_str!("../src/variables.rs");
    let mut current = String::new();
    let mut map = Vec::new();
    let mut in_vars = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("pub static VARS") {
            in_vars = true;
            continue;
        }
        if !in_vars {
            continue;
        }
        if t == "];" {
            break;
        }
        if let Some(rest) = t.strip_prefix("// \u{2500}\u{2500}") {
            let label = rest.trim_end_matches('\u{2500}').trim();
            // Drop the internal "Phase N:" bookkeeping prefix if present.
            let label = match label.find(": ") {
                Some(i) if label.starts_with("Phase") => &label[i + 2..],
                _ => label,
            };
            current = label.to_string();
        } else if let Some(rest) = t.strip_prefix("name: \"") {
            if let Some(end) = rest.find('"') {
                if current.is_empty() {
                    eprintln!("error: variable before first section comment");
                    std::process::exit(1);
                }
                map.push((rest[..end].to_string(), current.clone()));
            }
        }
    }
    map
}

fn main() {
    let sha = std::env::args().nth(1).unwrap_or_else(|| "unknown".into());
    let sections = section_map();

    // Cross-check: the source scan must yield exactly the registry, in order.
    if sections.len() != VARS.len() {
        eprintln!(
            "error: section scan found {} names but registry has {}",
            sections.len(),
            VARS.len()
        );
        std::process::exit(1);
    }
    for (i, v) in VARS.iter().enumerate() {
        if sections[i].0 != v.name {
            eprintln!(
                "error: scan/registry order mismatch at {}: {} vs {}",
                i, sections[i].0, v.name
            );
            std::process::exit(1);
        }
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"generated_from\": \"crates/wrf-core/src/variables.rs\",\n");
    out.push_str(&format!("  \"git_commit\": \"{}\",\n", json_escape(&sha)));
    out.push_str(&format!("  \"count\": {},\n", VARS.len()));
    out.push_str("  \"variables\": [\n");
    for (i, v) in VARS.iter().enumerate() {
        let aliases = v
            .aliases
            .iter()
            .map(|a| format!("\"{}\"", json_escape(a)))
            .collect::<Vec<_>>()
            .join(", ");
        let dim = match v.dim {
            VarDim::TwoD => "2D (ny, nx)",
            VarDim::ThreeD => "3D (nz, ny, nx)",
        };
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"aliases\": [{}], \"description\": \"{}\", \"units\": \"{}\", \"dim\": \"{}\", \"group\": \"{}\"}}{}\n",
            json_escape(v.name),
            aliases,
            json_escape(v.description),
            json_escape(v.default_units),
            dim,
            json_escape(&sections[i].1),
            if i + 1 == VARS.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n}\n");
    print!("{out}");
}
