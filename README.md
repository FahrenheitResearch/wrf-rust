# wrf-rust

`wrf-rust` is a WRF post-processing library and Python package backed by Rust.
It reads WRF output files, computes common meteorological diagnostics, and now
contains a native Rust rendering/product workflow for operational-style plots.

The project has two user-facing surfaces:

- Python package `wrf-rust`, imported as `wrf`, for `WrfFile`, `getvar()`,
  NumPy arrays, and compatibility with existing analysis workflows.
- Rust crates under `crates/` for WRF-native diagnostics, rendering, products,
  soundings, local run discovery, and ensemble reductions.

The goal is simple: keep WRF science, plotting, and ensemble work focused in
one WRF-specific repo without pulling in the whole multi-model research stack.

## Status

This repository is active research software. The Python `getvar()` workflow is
the stable user API. The native Rust renderer, product recipes, soundings,
store, and ensemble support are under active development on feature branches
before they are cut into a release.

Current defaults favor the pure Rust reader path, so the core library does not
need a system NetCDF installation for normal wheel builds. A system NetCDF
backend is still available as an optional Cargo feature for development and
comparison.

## Install

For normal Python use:

```bash
pip install wrf-rust
```

The Python package installs a module named `wrf`:

```python
from wrf import WrfFile, getvar

f = WrfFile("wrfout_d01_2024-06-01_00:00:00")
slp = getvar(f, "slp", units="hPa")
sbcape = getvar(f, "sbcape")
```

For local development:

```bash
git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust
pip install maturin
maturin develop --release
```

To build the Python extension through the pure Rust reader path:

```bash
maturin develop --release --cargo-extra-args="--no-default-features --features pure-rust-reader"
```

## Quick Python Examples

Open a WRF file:

```python
from wrf import WrfFile, getvar, ALL_TIMES

f = WrfFile("wrfout_d02_1974-04-03_22_00_00")

print(f.nx, f.ny, f.nz, f.nt)
print(f.times())
```

Read standard diagnostics:

```python
temp_c = getvar(f, "temp", units="degC")
slp_mb = getvar(f, "slp", units="hPa")
wspd_kt = getvar(f, "wspd", units="kt")
td2_f = getvar(f, "td2", units="degF")
```

Compute severe-weather fields:

```python
sbcape = getvar(f, "sbcape")
mlcape = getvar(f, "mlcape")
mucape = getvar(f, "mucape")
srh1 = getvar(f, "srh1")
srh3 = getvar(f, "srh3")
stp = getvar(f, "stp", layer_type="effective")
scp = getvar(f, "scp")
```

Work with custom layers and storm motion:

```python
shear = getvar(f, "bulk_shear", bottom_m=0, top_m=6000)
lr = getvar(f, "lapse_rate", bottom_p=700, top_p=500)
srh = getvar(f, "srh", depth_m=1500, storm_motion=(12.0, 8.0))
```

Read all times from a file:

```python
slp_all = getvar(f, "slp", timeidx=ALL_TIMES, units="hPa")
```

Raw WRF variables are also available by name:

```python
rain = getvar(f, "RAINNC", units="in")
u10 = getvar(f, "U10", units="kt")
v10 = getvar(f, "V10", units="kt")
```

`netCDF4.Dataset` and xarray-like inputs are accepted when a source filepath is
available, but `wrf-rust` reopens the file natively. On Windows, close an open
`netCDF4.Dataset` before passing the same file to `wrf-rust`.

## Native Rust Plotting

The new plotting path is Rust-native:

```text
wrf-core getvar() -> wrf-products recipe -> wrf-render PNG
```

The renderer does not open WRF files. It renders fields, grids, overlays,
palettes, colorbars, and basemap layers supplied by callers. `wrf-products` is
the WRF-specific glue that turns a product name into `getvar()` calls plus a
render request.

Render one product:

```bash
cargo run -p wrf-products --example render_product -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  stp_effective \
  stp_effective.png \
  0
```

Render the default product suite:

```bash
cargo run -p wrf-products --example render_suite -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/products \
  0
```

Render selected products:

```bash
cargo run -p wrf-products --example render_suite -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/severe \
  0 \
  sbcape,mlcape,srh03,stp_effective,scp,reflectivity
```

Render with an explicit history directory for products that can use neighboring
WRF output files:

```bash
cargo run -p wrf-products --example render_suite -- \
  --history-dir /path/to/wrfout/history \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/severe \
  0 \
  reflectivity_uh
```

Ask the renderer what a product list needs before staging files:

```bash
cargo run -p wrf-products --example render_suite -- \
  --print-required-inputs reflectivity_uh,stp_effective,precip_accum
```

Product rendering uses the `OPERATIONAL_FAST` presentation profile from the
Rust rendering stack: projected data frame, operational-style colorbars,
discrete weather scales, basemap linework, contours, and wind barbs where the
product calls for them.

### Rendering Input Contract

Single-file rendering is the default and supported mode. `render_suite` and
`render_product` open only the wrfout file passed on the command line unless an
explicit input option is provided. They do not implicitly scan the current file's
directory for sibling wrfout files.

`reflectivity_uh` renders from the current wrfout by default. If the file
contains multiple `Time` records, those records may be used to build the 1-hour
UH max track within that single file. To include neighboring wrfout files in the
track, pass `--history-dir DIR`; only same-domain wrfout files in that explicit
directory with valid times in the previous 60 minutes are considered.

## Native Soundings

`wrf-sounding` extracts point and box soundings from a WRF file and renders them
through the vendored `sharprs` native Rust sounding renderer. This path does not
use Matplotlib, MetPy, or Python plotting.

Render a point sounding:

```bash
cargo run -p wrf-sounding --example render_sounding -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/sounding.png \
  0 \
  --latlon 32.98,-88.59
```

Render by grid point:

```bash
cargo run -p wrf-sounding --example render_sounding -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/sounding.png \
  0 \
  --ij 120,95
```

Render a box sounding:

```bash
cargo run -p wrf-sounding --example render_sounding -- \
  /path/to/wrfout_d02_1974-04-03_22_00_00 \
  output/box_sounding.png \
  0 \
  --box 32.5,-89.2,33.5,-88.0 --method most_unstable
```

Box methods are `mean`, `median`, and `most_unstable`.

Python users can call the same Rust renderer:

```python
from wrf import WrfFile, render_sounding

f = WrfFile("wrfout_d02_1974-04-03_22_00_00")
render_sounding(f, "sounding.png", timeidx=0, latlon=(32.98, -88.59))
render_sounding(
    f,
    "box_sounding.png",
    timeidx=0,
    box=(32.5, -89.2, 33.5, -88.0),
    method="most_unstable",
)
```

## Native Products

`wrf-products` currently includes product recipes for:

- ECAPE family: `ecape`, `sb_ecape`, `ml_ecape`, `mu_ecape`, `ncape`,
  `ecape_cape`, `ecape_cin`, `ecape_lfc`, `ecape_el`, `ecape_scp`,
  `ecape_ehi`
- CAPE/CIN: `sbcape`, `sbcin`, `mlcape`, `mlcin`, `mucape`, `mucin`,
  `effective_cape`, `effective_inflow_base`, `effective_inflow_top`
- Severe parameters: `srh01`, `srh03`, `effective_srh`, `shear01`,
  `shear06`, `ebwd`, `stp_effective`, `stp_fixed`, `scp`, `ehi`, `tehi`,
  `tts`, `vtp_mod`, `critical_angle`, `ship`, `bri`
- Radar/cloud/precip: `reflectivity`, `reflectivity_1km`,
  `reflectivity_uh`, `cloud_top_temperature`, `precip_accum`,
  `updraft_helicity`, `cloudfrac_low`, `cloudfrac_mid`, `cloudfrac_high`
- Surface fields: `slp_wind10m`, `surface_wind10m`, `u10_component`,
  `v10_component`, `t2`, `td2`, `rh2`, `pwat`, `pblh`, `terrain`
- Thermodynamic levels: `lcl`, `lfc`, `el`, `lapse_rate_700_500`,
  `lapse_rate_0_3km`, `freezing_level`, `wet_bulb_zero`
- Fire weather: `fosberg`, `haines`, `hdw`
- Upper air: `height200_wind`, `temp200_wind`, `wind200`,
  `height250_wind`, `temp250_wind`, `wind250`, `height300_wind`,
  `temp300_wind`, `wind300`, `height500_wind`, `temp500_wind`,
  `wind500`, `vort500_wind`, `pvo500`, `omega500`, `temp700_wind`,
  `height700_wind`, `rh700_wind`, `height850_wind`, `theta_w850`,
  `temp850_wind`, `wind850`

Product names accept several common aliases; the canonical names above are what
examples and output filenames use.

## Ensemble Workflows

`wrf-ensemble` runs the same product recipes across same-grid WRF members and
reduces the resulting fields.

Render one ensemble product:

```bash
cargo run -p wrf-ensemble --example render_ensemble -- \
  "members/*/wrfout_d02_1974-04-03_22_00_00" \
  stp_effective \
  mean \
  output/ensemble_stp_mean.png \
  0
```

Render a suite:

```bash
cargo run -p wrf-ensemble --example render_ensemble_suite -- \
  "members/*/wrfout_d02_1974-04-03_22_00_00" \
  output/ensemble \
  0 \
  mean,stddev,p:90,prob_ge:1,prob_ge:3 \
  stp_effective,scp,sbcape,reflectivity
```

Supported statistics:

- `mean`
- `stddev`
- `min`
- `max`
- `p:NN` / `percentile:NN`
- `prob_gt:VALUE`
- `prob_ge:VALUE`
- `prob_lt:VALUE`
- `prob_le:VALUE`

All members must be on the same WRF grid for a given reduction.

## Crate Layout

- `crates/wrf-core`: WRF file access, raw variables, diagnostics, units,
  grid/projection metadata, and `getvar()` behavior.
- `crates/wrf-contour`: contour topology and geometry.
- `crates/wrf-render`: native rendering primitives, palettes, overlays,
  projected maps, colorbars, text, wind barbs, and PNG output.
- `crates/wrf-products`: WRF product recipes that connect `wrf-core` fields to
  `wrf-render` requests.
- `crates/wrf-ensemble`: ensemble member discovery, product reductions, and
  ensemble rendering.
- `crates/wrf-sounding`: point and box sounding extraction.
- `crates/wrf-store`: local run/file catalog helpers.
- `crates/rustwx-core`: small shared support types ported from `rustwx` where
  WRF rendering needs them.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership boundaries and
[docs/MIGRATION_FROM_RUSTWX.md](docs/MIGRATION_FROM_RUSTWX.md) for what was
ported from `rustwx`.

## Diagnostics

The Python `getvar()` API and `wrf-core` support WRF raw variables plus common
derived diagnostics. Major groups include:

- Temperature, pressure, height, terrain, humidity, moisture, and wind.
- CAPE/CIN with surface-based, mixed-layer, most-unstable, generic, and custom
  parcel paths.
- ECAPE-family diagnostics with parcel selection and storm-motion options.
- Storm-relative helicity, Bunkers storm motion, bulk shear, mean wind, lapse
  rates, and severe composite parameters.
- Reflectivity, cloud-top temperature, updraft helicity, vorticity, and fire
  weather indices.

Common options:

| Option | Purpose |
| --- | --- |
| `units` | Convert output units where supported. |
| `timeidx` | Select a time index; use `ALL_TIMES` for all times. |
| `parcel_type` | Select `"sb"`, `"ml"`, or `"mu"` for parcel-aware diagnostics. |
| `bottom_m`, `top_m` | Layer bounds in meters AGL. |
| `bottom_p`, `top_p` | Layer bounds in hPa. |
| `depth_m` | SRH/EHI layer depth in meters AGL. |
| `storm_motion` | Custom storm motion `(u, v)` in m/s. |
| `storm_motion_type` | ECAPE storm-motion type such as `"bunkers_rm"`. |
| `layer_type` | STP layer type, usually `"fixed"` or `"effective"`. |
| `lake_interp` | Interpolate 2 m fields over small water bodies. |

## Unit Strings

Unit strings are case-insensitive.

| Category | Strings |
| --- | --- |
| Temperature | `K`, `degC`, `C`, `celsius`, `degF`, `F`, `fahrenheit` |
| Pressure | `Pa`, `hPa`, `mb`, `mbar`, `inHg` |
| Speed | `m/s`, `knots`, `kt`, `kts`, `mph`, `kph`, `km/h` |
| Length/height | `m`, `ft`, `km`, `mi`, `dam` |
| Precip/depth | `mm`, `in`, `inches` |
| Moisture | `kg/kg`, `g/kg` |

## Community WRF Guide

The repository also includes a WRF setup guide for Windows/WSL 2, real-data
initialization, domain sizing, and troubleshooting:

- [docs/index.html](docs/index.html)
- [docs/starter-files/README.md](docs/starter-files/README.md)
- [skills/wrf-community-onboarding/SKILL.md](skills/wrf-community-onboarding/SKILL.md)

## Development Checks

Useful checks before opening a PR:

```bash
cargo fmt --all
cargo test -p wrf-products
cargo test -p wrf-render
cargo check --workspace
```

For Python extension development:

```bash
maturin develop --release
python -m wrf info /path/to/wrfout
python -m wrf stats /path/to/wrfout sbcape slp temp
```

The Python CLI still contains simple matplotlib-based quick-look commands for
compatibility. Production WRF product rendering belongs in the Rust
`wrf-products`/`wrf-render` path.

## Acknowledgments

Special thanks to Solarpower07 for the Solar7 color tables, product styling,
and guidance on severe-weather diagnostics, storm motion, SRH, ECAPE, and WRF
plot presentation. The native WRF rendering work in this repo is better because
of that input.

## License

MIT
