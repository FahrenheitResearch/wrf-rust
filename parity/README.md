# wrf-python differential parity

This directory is the first-stage acceptance framework for treating `wrf-rust`
as a broad replacement for NCAR `wrf-python`. It deliberately separates three
questions that are easy to conflate:

1. Does the public API have compatible names, options, units, axes, component
   ordering, and missing-value behavior?
2. Where NCAR `wrf-python` is a valid oracle, are fields numerically within an
   explicit tolerance?
3. Where `wrf-rust` intentionally goes beyond or corrects `wrf-python`, does it
   match the appropriate scientific authority and its real consumer contracts?

The answer to (3) must never be manufactured by loosening a wrf-python
tolerance. Such entries are marked `contract_only` until an independent
profile/recipe adapter is added.

## Files

- `contracts-v1.json` is the resolved scientific field matrix. Every entry
  names the Rust variable and aliases, options/defaults, upstream extractor,
  units, axes, component order, missing-value policy, tolerance, and known
  deviations.
- `wrf-runner-contracts-v1.json` is a separate regression contract derived
  from the real `WRF-Runner` `New-PC-Updates` consumer. It distinguishes strict
  upstream behavior from deliberate wrf-rust extensions.
- `fixtures-v1.json` registers external fixtures by exact byte length and
  SHA-256. WRF files are never checked into this repository.
- `extract_wrfpython.py` and `extract_wrfrust.py` create implementation bundles.
- `compare.py` checks bundle provenance, shapes, missing masks, and numerical
  tolerances and can emit a machine-readable report.
- `probe_wrf_runner.py` exercises the candidate-only WRF-Runner API and shape
  contract, including scalar/2-D `interplevel`, handle-based coordinate helpers,
  projection construction, and its literal `getvar` calls.

The previously collected `tmp-wrfpy-audit/g_cape.py` source was used to retain
the documented upstream `cape_2d` order (`MCAPE`, `MCIN`, `LCL`, `LFC`). That
scratch directory is not a runtime input and is intentionally not copied here.

## Separate environments are mandatory

Both projects import as `wrf`; loading them together is ambiguous. The two
extractors therefore reject an environment containing the other distribution.

Create the reference environment on Linux (or another platform where the
Fortran-backed package is known to work):

```text
conda env create -f parity/environment-wrfpython.yml
conda activate wrf-python-reference
```

Build and install the candidate `wrf-rust` wheel in a different Linux
environment. Do not build it in the reference environment. The comparator only
needs NumPy and can use `environment-compare.yml`.

The pin is intentional: reference bundles record package/runtime versions, but
pinning also prevents a future environment solve from silently changing the
oracle. `wrf-python` normally needs a working Fortran/NetCDF toolchain if a
prebuilt conda package is unavailable. Python 3.14 is not a supported reference
target for this pinned stack.

## Reproducible workflow

Place the registered fixture outside Git and point the catalog's environment
variable at it:

```text
export WRF_PARITY_FIXTURE_D03=/data/wrfout_d03_2023-01-05_09_00_00
```

The extractor re-hashes all 2,388,958,814 bytes and refuses a filename-only or
size-only match.

Reference environment:

```text
python parity/extract_wrfpython.py \
  --fixture-id enderlin-d03-20230105-0900 \
  --output parity-results/enderlin-wrfpython.npz
```

Candidate environment:

```text
python parity/extract_wrfrust.py \
  --fixture-id enderlin-d03-20230105-0900 \
  --output parity-results/enderlin-wrfrust.npz
```

Comparator environment:

```text
python parity/compare.py \
  --reference parity-results/enderlin-wrfpython.npz \
  --candidate parity-results/enderlin-wrfrust.npz \
  --json-output parity-results/enderlin-report.json
```

Each `.npz` embeds fixture SHA-256, canonical contract SHA-256, selected IDs,
time index, implementation and dependency versions, module path, platform, and
per-array shape. Every bundle and JSON report gets a `.sha256` sidecar. NumPy
object arrays and pickle loading are disabled.

Use `--variables` or `--families` to make a targeted bundle. Both extractors
must use the identical selection, fixture, time index, and contract document.
`diagnostic` differences are reported but do not fail by default; add
`--strict-diagnostic` only after that contract is scientifically reconciled.

## WRF-Runner acceptance

The consumer manifest is based on commit
`fe2e5405ce818824a50c4bc4ef2b506ec07dc41e`; hashes of `plot_helper.py`,
`plot_functions.py`, and `run_viewer.py` make drift visible. Run its P0 probe in
the candidate environment:

```text
python parity/probe_wrf_runner.py \
  --fixture-id enderlin-d03-20230105-0900 \
  --consumer-root /path/to/WRF-Runner \
  --output parity-results/wrf-runner-p0.json
```

This is a compatibility gate, not the upstream oracle. WRF-Runner deliberately
uses wrf-rust extensions including `WrfFile`, handle-based `get_cartopy` and
`latlon_coords`, parcel-specific/truncated CAPE, lake interpolation, Bunkers
SRH, and effective-layer products.

Its production performance also cannot be inferred solely from a single-process
79-product benchmark. `generate_plots_for_timestep` submits products to separate
processes, each constructing a fresh `WrfFile`; per-handle CAPE/EIL caches are
not shared. A release benchmark must include the real product batch or a
faithful multi-process call-sequence driver.

## Resolved stage-one parity

- AVO and PVO directly mirror pinned wrf-python 1.3.4.1 commit
  `31c923335227b22fa656fd589a5342b91103e939`
  `DCOMPUTEABSVORT`/`DCOMPUTEPV` kernels: raw C-grid winds, stagger-specific
  and mass map factors, raw `F`, clamped boundary stencils, the pinned 9.81
  m/s^2 gravity constant, and all three Ertel-PV terms. They are required
  comparisons rather than documented approximations.
- `interplevel` now supports arbitrary left dimensions, the leading
  multiproduct dimension used by vector diagnostics, scalar and 1-D level
  requests, shared or left-dependent target surfaces, caller-selected missing
  values, `squeeze`, and optional xarray metadata. Its bracket scan, strict
  bounds, dtype, and dimension order follow wrf-python 1.3.4.1 commit
  `31c923335227b22fa656fd589a5342b91103e939` while retaining WRF-Runner's
  2-D target-surface use case. Without optional xarray, the shim deliberately
  still applies `squeeze` and keeps NaN in masked output buffers so existing
  WRF-Runner `numpy.asarray`/`numpy.array` call paths remain safe.

## Known stage-one gaps

- `theta_w`, SB/ML parcel diagnostics, shear, Bunkers motion, SHIP, STP, and
  SCP need an independent profile or operational-recipe reference adapter.
- NCAR SRH and WRF-Runner SRH remain distinct contracts. `srh_wrfpython`
  explicitly ports RIP `DCALRELHL`; the existing `srh`/`srh1`/`srh3` names
  retain WRF-Runner's Bunkers behavior. Southern-Hemisphere and threshold-level
  fixtures are still needed to exercise the strict path end to end.
- `ll_to_xy` and `xy_to_ll` now use the pinned wrf-python 1.3.4.1 analytic
  Lambert, polar-stereographic, Mercator, regular-lat/lon, and rotated-lat/lon
  equations. They preserve scalar/sequence and optional xarray metadata
  conventions, normalize antimeridian longitudes, support mass/U/V projection
  origins, and extrapolate outside-domain points instead of clamping them.
- For unrotated `MAP_PROJ=6`, longitude wrapping intentionally closes an
  upstream edge-case bug: pinned `DLLTOIJ` returns x=-358 rather than x=2 for
  a one-degree grid from 179E to 179W. All other projection values are pinned
  numerically to the compiled 1.3.4.1 reference routine.
- Moving nests are detected from time-varying stagger-specific reference
  coordinates and rejected explicitly. Time-dependent moving-domain output,
  multi-file `cat`/`join` dimensions, and mapping inputs remain stage-two gaps.
- Projection constructors now use WRF's spherical globe and pinned Lambert,
  Mercator, polar, regular, and rotated-lat/lon parameters. Moving-domain,
  multi-file mapping, and complete `CoordPair` utility semantics remain
  unsupported.
- Multiple representative fixtures are still needed: high terrain, lakes,
  moving nests, Southern Hemisphere, dateline/global grids, high surface
  pressure, shallow caps, and multiple buoyant layers.
