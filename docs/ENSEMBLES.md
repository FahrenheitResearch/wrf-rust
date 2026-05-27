# WRF Ensembles

`wrf-ensemble` reduces WRF products across member files and renders the result
with the same Rust `OPERATIONAL_FAST` product renderer used for deterministic
plots.

Current scope:

- manifest-first member lists, with glob input retained as a convenience
- same-grid members only
- strict valid-time alignment for the requested `timeidx`
- scalar filled product reductions
- overlays are disabled on ensemble products until contour/vector reductions are
  explicit
- no silent regridding
- probability denominators are finite members per grid cell

## Single Product

Prefer a manifest so member identity is explicit and paths are stable relative
to the manifest file:

```json
{
  "members": [
    {"id": "m01", "path": "members/m01/wrfout_d01_2026-05-25_00_00_00"},
    {"id": "m02", "path": "members/m02/wrfout_d01_2026-05-25_00_00_00"}
  ]
}
```

```powershell
cargo run -p wrf-ensemble --example render_ensemble -- `
  C:/runs/ensemble.json `
  mlcape mean C:/runs/plots/mlcape_mean.png 0
```

Probability and percentile stats need a value:

```powershell
cargo run -p wrf-ensemble --example render_ensemble -- `
  "C:/runs/members/*/wrfout_d01_2026-05-25_00_00_00" `
  stp_effective prob_ge C:/runs/plots/stp_prob_ge_1.png 0 1

cargo run -p wrf-ensemble --example render_ensemble -- `
  "C:/runs/members/*/wrfout_d01_2026-05-25_00_00_00" `
  precip_accum percentile C:/runs/plots/precip_p90.png 0 90
```

## Suite

Default suite renders mean and spread for every default WRF product:

```powershell
cargo run -p wrf-ensemble --example render_ensemble_suite -- `
  C:/runs/ensemble.json `
  C:/runs/plots/ensemble 0
```

Use comma-separated stats and products to keep tonight's runs focused:

```powershell
cargo run -p wrf-ensemble --example render_ensemble_suite -- `
  C:/runs/ensemble.json `
  C:/runs/plots/ensemble 0 `
  "mean,stddev,prob_ge:1,p:90" `
  "mlcape,stp_effective,srh03,fosberg,reflectivity"
```

Each ensemble PNG gets a JSON sidecar with product id, stat, members, member
count, strict grid/valid-time policy, probability denominator policy, output
units, and provenance.

Supported stats:

- `mean`
- `stddev` or `spread`
- `min`
- `max`
- `percentile:<p>` or `p:<p>` in suite CSV
- `prob_gt:<threshold>`
- `prob_ge:<threshold>`
- `prob_lt:<threshold>`
- `prob_le:<threshold>`
