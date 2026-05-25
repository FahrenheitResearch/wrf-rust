# wrf-rust Architecture

`wrf-rust` is the canonical WRF product repository. It owns WRF file access,
diagnostics, WRF-native rendering, WRF product recipes, WRF sounding extraction,
and local WRF run discovery.

`rustwx` remains the broad multi-model research and incubation workspace. Port
capabilities into `wrf-rust` only when they become WRF-specific product surface.
Do not turn `rustwx` into the WRF repository by deletion.

## Crate Boundaries

### wrf-core

Owns:

- WRF file reading
- Raw WRF variable access
- Destaggering
- Diagnostic calculations
- Units
- Projection and grid metadata
- `getvar`-compatible behavior

Does not own plotting, file catalogs, cache retention, rendered product recipes,
or sounding image layout.

### wrf-contour

Owns renderer-agnostic contour and filled-band topology. It understands scalar
grids, levels, and topology extraction. It does not know about WRF files,
products, palettes, Python, or image output.

### wrf-render

Owns native image rendering primitives for WRF fields:

- Render-ready 2-D WRF fields
- Palettes and levels
- Filled raster maps
- Contour and wind-barb overlay request types
- PNG output

`wrf-render` does not open WRF files and does not calculate meteorological
diagnostics. Callers must supply field values, grid metadata, and optional
lat/lon/projection metadata.

### wrf-products

Owns product recipes and glue between WRF diagnostics and rendering:

- Product-name parsing
- Mapping product names to `wrf-core::getvar` calls
- Product-specific units, palettes, levels, titles, contours, and wind overlays
- Building render-ready `WrfField2D` values from a `wrf-core::WrfFile`

`wrf-products` can call `wrf-core` and `wrf-render`. It should not duplicate
science formulas.

### wrf-ensemble

Owns WRF member-set reductions:

- Ensemble member discovery from explicit paths or glob patterns
- Same-grid validation before reduction
- Product-level reductions after each member is computed through `wrf-products`
- Mean, spread, min/max, percentile, and probability fields
- Ensemble render requests and PNG output

`wrf-ensemble` does not calculate single-member diagnostics directly and does
not silently regrid. If member grids differ, it must fail until an explicit
regridding/intersection policy is added.

### wrf-sounding

Owns WRF sounding selection and profile assembly:

- Point lat/lon selection
- Point i/j selection
- Box selection
- Mean-profile, median-profile, and most-unstable-column selection semantics
- Validated sounding-column data structures

Rendering backends such as `sharprs` can be added here, but WRF profile
extraction should remain separate from map products.

### wrf-store

Owns local WRF run and file discovery:

- Run roots
- Domain and valid-time file lookup
- Manifest paths
- Rendered product artifact paths
- Retention candidates

`wrf-store` is deliberately boring. It does not calculate meteorological fields,
render images, or choose product semantics.

## Python Layer

The Python `wrf` module remains the user-facing API. Python should become a thin
wrapper over Rust implementations when native paths exist, while compatibility
fallbacks can remain for notebooks and transitional plotting workflows.

Preferred production flow:

```text
Python -> Rust binding -> wrf-products -> wrf-core -> wrf-render -> PNG
```

For sounding products:

```text
Python -> Rust binding -> wrf-sounding -> sounding renderer -> PNG
```

## Porting Rule

Move capabilities from `rustwx`, not whole subsystems. If a port requires broad
HRRR/GFS/ECMWF/RAP/NAM/RRFS orchestration to come with it, the boundary is too
wide for `wrf-rust`.
