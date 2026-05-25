# Agent Guide

`wrf-rust` is the canonical WRF repository. Keep edits WRF-focused and respect
the crate boundaries in `docs/ARCHITECTURE.md`.

## Where To Edit

- WRF file access, diagnostics, units, and `getvar`: `crates/wrf-core`
- Contour topology: `crates/wrf-contour`
- Rendering fields that have already been computed: `crates/wrf-render`
- Product recipes and `getvar` to render-request glue: `crates/wrf-products`
- Point and box soundings: `crates/wrf-sounding`
- Local run/file/cache discovery: `crates/wrf-store`
- Python wrappers and compatibility API: `python/wrf` and `src`

## What Not To Do

- Do not make `wrf-core` depend on rendering, stores, Python, or product recipes.
- Do not make `wrf-render` open WRF files.
- Do not copy the whole `rustwx` workspace into this repo.
- Do not delete or rewrite `rustwx` as the migration strategy.
- Do not duplicate diagnostic science in `wrf-products`.

## Porting Checklist

1. Identify the smallest WRF-relevant capability.
2. Port the data types into the smallest `wrf-*` crate that owns that boundary.
3. Keep broad model orchestration out of `wrf-rust`.
4. Add or update tests for the new boundary.
5. Run `cargo fmt` and targeted `cargo check` before handing off.
