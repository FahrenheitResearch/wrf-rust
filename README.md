# wrf-rust

Rust-powered WRF post-processing with Python bindings. 83 diagnostic variables, built-in plotting, and parallel computation.

## Install

```bash
pip install wrf-rust
```

Pre-built wheels for Python 3.10-3.13 on Linux, macOS, and Windows. No Rust toolchain, no system libraries, no conda required.

## Community Guide

The repo now includes a full WRF community setup guide for Windows, WSL 2, real-data initialization, domain sizing, and troubleshooting:

- Manual-first Pages guide: [docs/index.html](docs/index.html)
- Optional Codex skill: [skills/wrf-community-onboarding/SKILL.md](skills/wrf-community-onboarding/SKILL.md)

## Usage

```python
from wrf import WrfFile, getvar

f = WrfFile("wrfout_d01_2024-06-01_00:00:00")

# Basic fields
temp = getvar(f, "temp", units="degC")
slp  = getvar(f, "slp",  units="hPa")
wspd = getvar(f, "wspd", units="knots")

# CAPE with parcel selection
sbcape = getvar(f, "sbcape")
mlcape = getvar(f, "mlcape")
mucape = getvar(f, "mucape")
sb3cap = getvar(f, "sbcape", top_m=3000)           # 0-3 km CAPE

# Custom parcel
cape = getvar(f, "cape", parcel_pressure=850,
              parcel_temperature=20, parcel_dewpoint=15)

# SRH with Bunkers storm motion
srh1 = getvar(f, "srh1")                            # 0-1 km
srh3 = getvar(f, "srh3")                            # 0-3 km
srh  = getvar(f, "srh", depth_m=1500, storm_motion=(12, 8))

# Effective inflow layer
eff_srh  = getvar(f, "effective_srh")
eff_cape = getvar(f, "effective_cape")

# Severe composites
stp     = getvar(f, "stp")                           # fixed-layer
stp_eff = getvar(f, "stp", layer_type="effective")   # effective-layer
scp     = getvar(f, "scp")
ehi     = getvar(f, "ehi", depth_m=3000)             # 0-3 km EHI

# Configurable layers
shear = getvar(f, "bulk_shear", bottom_m=1000, top_m=6000)
mw    = getvar(f, "mean_wind",  bottom_m=0, top_m=6000)
lr    = getvar(f, "lapse_rate", bottom_p=700, top_p=500)
lr_v  = getvar(f, "lapse_rate", bottom_m=0, top_m=3000, use_virtual=True)

# Lake interpolation (removes 2m artifacts over water bodies)
cape = getvar(f, "sbcape", lake_interp=1000)         # interp lakes < 1000 km2

# All timesteps
slp_all = getvar(f, "slp", timeidx=None)             # shape (nt, ny, nx)
```

Also accepts `netCDF4.Dataset` directly:

```python
from netCDF4 import Dataset
slp = getvar(Dataset("wrfout_d01..."), "slp")
```

Note: dataset inputs are reopened by filepath under the hood. On Windows,
do not keep a `netCDF4.Dataset` open while calling wrf-rust on that same
file, especially in subprocesses. Close the dataset first or pass a file
path / `WrfFile` instead.

## Plotting

```python
from wrf import plot_field, plot_wind, plot_skewt, panel

plot_field(f, "sbcape")                               # auto colormap + cartopy
plot_field(f, "sbcape", style="solar7")               # Solarpower07 colormaps
plot_wind(f)                                           # wind barbs
plot_skewt(f, point=(35.0, -97.5))                    # Skew-T with hodograph
panel(f, ["sbcape", "srh1", "stp", "shear_0_6km"])   # multi-panel

# Multi-timestep with consistent scale + GIF
from wrf.plot import render_timesteps
render_timesteps(f, "sbcape", timesteps=[0,1,2,3],
                 gif=True, fixed_scale=True)
```

## CLI

```bash
python -m wrf info  wrfout_d01_2024-06-01_00:00:00
python -m wrf stats wrfout_d01_2024-06-01_00:00:00 sbcape slp temp
python -m wrf plot  wrfout_d01_2024-06-01_00:00:00 slp -o slp.png
python -m wrf panel wrfout_d01_2024-06-01_00:00:00 sbcape srh1 stp -o severe.png
```

## Benchmark vs wrf-python

Tested on a 199x199x79 WRF grid. All values match exactly (rel_err=0.00).

| Variable | wrf-python | wrf-rust | Speedup |
|---|---|---|---|
| Temperature | 0.268s | 0.004s | **76x** |
| Potential temp | 0.088s | 0.003s | **26x** |
| Abs. vorticity | 0.173s | 0.015s | **11x** |
| Relative humidity | 0.353s | 0.097s | **4x** |
| Precipitable water | 0.224s | 0.077s | **3x** |
| Reflectivity | 0.494s | 0.234s | **2x** |
| SLP | 0.517s | 0.497s | 1x |
| Wind destagger | 0.089s | 0.081s | 1x |

Plus 17 variables wrf-python doesn't have (STP, SCP, EHI, critical angle, shear, Bunkers, lapse rates, fire indices, effective inflow layer).

## Variables

83 diagnostic variables. All support `units=` parameter.

### Thermodynamics

`temp` `tc` `theta` `theta_e` `theta_w` `tv` `twb` `td` `rh`

### Pressure & height

`pressure` `slp` `height` `height_agl` `terrain` `geopt` `omega`

### Moisture

`pw` `rh2m` `dp2m` `mixing_ratio` `specific_humidity`

### CAPE & convection

`sbcape` `sbcin` `mlcape` `mlcin` `mucape` `mucin` `cape` `cin` `lcl` `lfc` `el` `effective_cape` `effective_inflow` `cape2d` `cape3d`

All CAPE variables support `top_m` for truncated integration (e.g. `top_m=3000` for 3CAPE). Generic `cape`/`cin` accept `parcel_type` or custom parcel (`parcel_pressure`, `parcel_temperature`, `parcel_dewpoint`).

### Wind

`ua` `va` `wa` `wspd` `wdir` `wspd10` `wdir10` `uvmet` `uvmet10`

### SRH & shear

`srh1` `srh3` `srh` `effective_srh` `shear_0_1km` `shear_0_6km` `bulk_shear` `mean_wind` `bunkers_rm` `bunkers_lm` `mean_wind_0_6km`

SRH uses Bunkers Internal Dynamics method. All accept `storm_motion=(u,v)`.

### Severe composites

`stp` `stp_fixed` `stp_effective` `scp` `ehi` `critical_angle` `ship` `bri`

STP supports `layer_type="effective"` for the 5-term formula with MLCIN.

### Radar & cloud

`dbz` `maxdbz` `ctt` `cloudfrac` `uhel`

### Vorticity

`avo` `pvo`

### Lapse rates & levels

`lapse_rate_700_500` `lapse_rate_0_3km` `lapse_rate` `freezing_level` `wet_bulb_0`

Generic `lapse_rate` accepts `bottom_m`/`top_m` or `bottom_p`/`top_p`, plus `use_virtual=True`.

### Fire weather

`fosberg` `haines` `hdw`

## Parameters

| Parameter | Description |
|---|---|
| `units` | Output unit conversion (every variable) |
| `parcel_type` | `"sb"`, `"ml"`, `"mu"` for CAPE |
| `parcel_pressure/temperature/dewpoint` | Custom parcel (hPa, degC, degC) |
| `top_m` / `bottom_m` | Layer bounds in m AGL |
| `top_p` / `bottom_p` | Layer bounds in hPa |
| `depth_m` | SRH/EHI depth (m AGL) |
| `storm_motion` | Custom (u, v) in m/s |
| `layer_type` | `"fixed"` or `"effective"` for STP |
| `use_virtual` | Virtual temperature for lapse rates |
| `lake_interp` | Interpolate 2m fields over lakes < N km2 |

## Unit strings

Every registered variable supports `units=`. Raw WRF variables (RAINNC, T2, PSFC, etc.) also support conversion when their default unit is known. Case-insensitive.

| Category | Strings | Example |
|---|---|---|
| Temperature | `K`, `degC`, `C`, `celsius`, `degF`, `F`, `fahrenheit` | `units="degF"` |
| Pressure | `Pa`, `hPa`, `mb`, `mbar`, `inHg` | `units="hPa"` |
| Speed | `m/s`, `knots`, `kt`, `kts`, `mph`, `kph`, `km/h` | `units="knots"` |
| Length/Height | `m`, `ft`, `km`, `mi`, `dam` | `units="dam"` |
| Depth | `mm`, `in`, `inches` | `units="in"` |
| Moisture | `kg/kg`, `g/kg` | `units="g/kg"` |

### Raw WRF variables

Any variable name not in the computed registry is read directly from the file. Common ones:

| Variable | Default unit | Description |
|---|---|---|
| `RAINNC` | mm | Grid-scale accumulated precipitation |
| `RAINC` | mm | Convective accumulated precipitation |
| `T2` | K | 2-m temperature (also available as `t2`) |
| `PSFC` | Pa | Surface pressure |
| `TSK` | K | Skin temperature |
| `SST` | K | Sea surface temperature |
| `PBLH` | m | PBL height |
| `HFX` | W/m2 | Sensible heat flux |
| `LH` | W/m2 | Latent heat flux |
| `SWDOWN` | W/m2 | Downwelling shortwave |
| `GLW` | W/m2 | Downwelling longwave |
| `OLR` | W/m2 | Outgoing longwave |
| `UST` | m/s | Friction velocity |
| `SNOWH` | m | Snow depth |
| `LU_INDEX` | -- | Land use category |
| `U10` / `V10` | m/s | 10-m wind components |

```python
# These all work:
rain = getvar(f, "RAINNC", units="in")     # accumulated precip in inches
tsk  = getvar(f, "TSK", units="degF")      # skin temp in Fahrenheit
pblh = getvar(f, "PBLH", units="ft")       # PBL height in feet
```

## Building from source

Only needed for development. Users should `pip install wrf-rust`.

```bash
git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust
pip install maturin
maturin develop --release
```

## License

MIT
