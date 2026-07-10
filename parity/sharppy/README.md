# SHARPpy science differential acceptance

This harness compares selected public `wrf-core` severe-weather helpers to
live calls into the PyPI package `SHARPpy==1.4.0a5`. That release is the package
from official tag
[`v1.4.0a5`](https://github.com/sharppy/SHARPpy/tree/a5405e255ab696c32db578dff2c4f83699ec717e),
commit `a5405e255ab696c32db578dff2c4f83699ec717e`.

The Rust bridge is `crates/wrf-core/examples/sharppy_public_helpers.rs`. It
links the real `wrf-core` crate and calls its public APIs; the placeholder code
under `vendor/sharprs` is not built or imported. `run.py` owns the reference
profiles, invokes live SHARPpy functions, passes identical scalar or
profile-derived inputs to the Rust bridge, compares results, and writes a JSON
report plus a SHA-256 sidecar.

## Clean reference environment

SHARPpy 1.4.0a5 metadata pins `numpy==1.15.*`, which has no wheel for modern
Python. Install the verified universal SHARPpy wheel without its obsolete
dependency metadata, then install the NumPy version used for acceptance
explicitly:

```text
python -m venv .venv-sharppy-a5
```

On Linux or macOS:

```text
source .venv-sharppy-a5/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.2.6
python -m pip install --no-deps --require-hashes \
  -r parity/sharppy/requirements-sharppy-a5.txt
```

On Windows PowerShell:

```text
.venv-sharppy-a5\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.2.6
python -m pip install --no-deps --require-hashes `
  -r parity/sharppy/requirements-sharppy-a5.txt
```

The requirements file pins the PyPI wheel SHA-256 to
`13582f88ba1932b842cbf3ceb6f5f1ddadc17b0b2fd9172a3fc74ed0bcadb868`.
The runner additionally rejects the wrong distribution or module version,
imports outside the installed distribution, and any mismatch in normalized
SHA-256 for the official tag's `params.py`, `winds.py`, `thermo.py`,
`utils.py`, `profile.py`, or `interp.py`.

The acceptance paths use only NumPy and `sharppy.sharptab`; the GUI and network
dependencies declared by the historical package are not needed.

## Build and run

From the repository root, with the reference environment active:

```text
cargo build -p wrf-core --example sharppy_public_helpers
python parity/sharppy/run.py \
  --report parity-results/sharppy-a5-report.json
```

On Windows, the same commands work in PowerShell; use a backtick instead of a
backslash if splitting the Python command across lines.

Success requires exit status zero and a report containing:

```json
{
  "status": "passed",
  "failure_count": 0,
  "expected_case_count": 23,
  "executed_case_count": 23,
  "passed_case_count": 23
}
```

The runner writes `parity-results/sharppy-a5-report.json.sha256` beside the
report. `parity-results/` is ignored because reports contain runtime paths,
platform details, the candidate Git commit, and a generation timestamp.

The gate fails closed if the SHARPpy version or source provenance differs, the
frozen case IDs/order/input manifest changes, the Rust binary is stale, the
Rust protocol echoes a different case or diagnostic, a case is missing or
non-finite, or a comparison exceeds its declared tolerance.

Candidate provenance is scoped independently from the wrf-python contract
catalog: every tracked `crates/wrf-core/src/**/*.rs` input plus the workspace
and crate `Cargo.toml` files, `Cargo.lock`, and the Rust bridge itself must be
tracked and clean. The binary must be newer than all of those inputs. The report records
normalized SHA-256 for every build input (including `met/composite.rs`,
`met/wind.rs`, `diag/severe.rs`, and `variables.rs`) and separately hashes the
Python runner. Changes to `parity/contracts-v1.json` are outside this dedicated
gate and cannot be mistaken for candidate science provenance.

For an intentional case edit, print the new canonical manifest hash with:

```text
python parity/sharppy/run.py --print-case-manifest-sha256
```

Review the complete manifest diff before updating the frozen hash in `run.py`.

## Live differential coverage

The 23 live cases cover:

- fixed STP LCL bounds and 0-6 km shear gate/cap edges;
- SHARPpy's three-term SCP EBWD gate/ramp/cap, compared with the public Rust
  four-term helper at neutral `MUCIN=-40 J/kg`;
- SHIP parcel mixing-ratio, shear, and T500 clamps plus all three sequential
  low-end corrections;
- DCP's 0-6 km mean-wind input and knot normalizations; and
- critical-angle inflow sign, calm/nonzero surface wind, and signed-component
  profiles.

The SHIP adapter derives MU-parcel mixing ratio through SHARPpy's own
`temp_at_mixrat`/`mixratio` functions. DCP and critical angle use real SHARPpy
profile objects. All reference values come from live `params`/`winds` calls,
not copied equations or golden constants.

## Definitions newer than SHARPpy a5

SHARPpy 1.4.0a5 cannot be the oracle for definitions that it does not contain.
The report records these as Rust-only coverage rather than manufacturing a
differential value:

- Current-SPC SCP MUCIN scaling is covered by
  `met::composite::tests::exported_scp_helper_retains_the_spc_mucin_scaling`
  and
  `diag::severe::tests::registered_scp_cape_tuple_seam_selects_the_mucin_component`,
  while `variables::tests::scp_registry_points_to_the_current_spc_compute_path`
  pins the registry wiring,
  with the [SPC SCP definition](https://www.spc.noaa.gov/exper/mesoanalysis/help/help_scp.html).
- TEHI is covered by `diag::severe::tests::tehi_matches_spc_beta_formula`,
  `tehi_sets_ml3cape_term_to_one_only_above_mlcape_threshold`, and
  `tehi_high_ml3cape_cap_is_overridden_above_mlcape_threshold`, with the
  [SPC TEHI definition](https://www.spc.noaa.gov/exper/mesoanalysis/help/help_tehi.html).
- Published VTP versus this repository's deliberately distinct `vtp_mod` is
  covered by `vtp_mod_is_distinct_from_published_vtp_for_an_illustrative_profile`
  plus separate `published_vtp_reference_pins_paper_bounds_and_layers` and
  `current_spc_vtp_reference_pins_literal_html_rules` oracles. The split
  preserves the source disagreement: the paper caps the 3CAPE factor at 2,
  while current SPC's literal prose sets the lapse-rate factor to 2 above
  100 J/kg 3CAPE. Five `vtp_mod` factor-bound tests pin the deliberately
  different compatibility product, citing
  [Hampshire et al. (2018)](https://doi.org/10.15191/nwajom.2018.0601) and the
  [current SPC VTP definition](https://www.spc.noaa.gov/exper/mesoanalysis/help/help_vtp.html).

These exclusions are explicit acceptance boundaries, not skipped failures.
