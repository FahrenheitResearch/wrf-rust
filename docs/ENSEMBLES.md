# WRF Ensembles

`wrf-ensemble` reduces WRF products across member files and renders the result
with the same Rust `OPERATIONAL_FAST` product renderer used for deterministic
plots.

Current scope:

- same-grid members only
- scalar filled product reductions
- overlays are disabled on ensemble products until contour/vector reductions are
  explicit
- no silent regridding

## Single Product

```powershell
cargo run -p wrf-ensemble --example render_ensemble -- `
  "C:/runs/members/*/wrfout_d01_2026-05-25_00_00_00" `
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
  "C:/runs/members/*/wrfout_d01_2026-05-25_00_00_00" `
  C:/runs/plots/ensemble 0
```

Use comma-separated stats and products to keep tonight's runs focused:

```powershell
cargo run -p wrf-ensemble --example render_ensemble_suite -- `
  "C:/runs/members/*/wrfout_d01_2026-05-25_00_00_00" `
  C:/runs/plots/ensemble 0 `
  "mean,stddev,prob_ge:1,p:90" `
  "mlcape,stp_effective,srh03,fosberg,reflectivity"
```

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
