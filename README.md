# wrf-rust

Rust-powered WRF post-processing. Replaces [wrf-python](https://github.com/NCAR/wrf-python) with correct meteorology, 73 diagnostic variables, universal unit conversion, and parallel computation.

**This is not a drop-in replacement for wrf-python.** It is an improved API that fixes fundamental scientific errors in wrf-python's CAPE and SRH calculations, adds missing variables that operational meteorologists actually need, and runs significantly faster.

## Why not wrf-python?

wrf-python has been the standard tool for WRF post-processing since 2017, but it has real problems that affect research and operational output:

### CAPE is wrong

wrf-python's `cape_2d` computes CAPE using a hybrid parcel: it averages the lowest 500m, then searches for the maximum theta-e level. This is **not** SBCAPE, MLCAPE, or MUCAPE. There is no way to select parcel type. The result doesn't match SPC mesoanalysis, SHARPpy, or any standard operational definition.

```python
# wrf-python: what parcel is this? nobody knows
cape = getvar(ncfile, "cape_2d")  # hybrid parcel, not operationally standard

# wrf-rust: explicit parcel selection, matches SPC/SHARPpy definitions
sbcape = getvar(f, "sbcape")  # surface-based
mlcape = getvar(f, "mlcape")  # 100 hPa mixed-layer
mucape = getvar(f, "mucape")  # most unstable
```

### SRH is wrong

wrf-python computes storm-relative helicity using `0.75 * (3-10 km mean wind)` rotated 30 degrees as the storm motion estimate. This is **not Bunkers**. It systematically overestimates SRH compared to the Bunkers Internal Dynamics method (Bunkers et al. 2000) used by SPC, SHARPpy, and MetPy.

```python
# wrf-python: non-Bunkers storm motion, overestimates SRH
srh = getvar(ncfile, "srh")

# wrf-rust: proper Bunkers (0-6km mean wind + 7.5 m/s perpendicular deviation)
srh1 = getvar(f, "srh1")                          # 0-1 km, Bunkers right-mover
srh3 = getvar(f, "srh3")                          # 0-3 km, Bunkers right-mover
srh  = getvar(f, "srh", storm_motion=(10, 5))     # custom storm motion
```

### Other issues wrf-rust fixes

| Problem | wrf-python | wrf-rust |
|---|---|---|
| Unit conversion | Inconsistent -- some variables support `units=`, many don't (cape, dbz, rh, avo, pvo, omega, pw, geopt...) | Every variable supports `units=` |
| Grid operations | Single-column functions, loop over grid yourself | Full 2D grid by default, rayon-parallel |
| NaN handling | Fortran routines silently produce garbage from NaN input | Input validation, clean NaN propagation |
| Missing variables | No STP, SCP, EHI, critical angle, shear, Bunkers, lapse rates, fire indices... | 73 variables including all SPC severe params |
| Speed | Python/Fortran interop overhead | Native Rust, parallel across grid columns |

## Quick start

### Install from source (requires Rust toolchain + system libnetcdf)

```bash
# Requires: Rust, maturin, libnetcdf + libhdf5 (e.g. from conda)
conda activate wrfplot  # or any env with libnetcdf
pip install maturin
git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust
maturin develop --release
```

### Usage

```python
from wrf import WrfFile, getvar, ALL_TIMES

# Open a WRF output file
f = WrfFile("wrfout_d01_2024-06-01_00:00:00")

# Basic fields
temp  = getvar(f, "temp", timeidx=0, units="degC")   # 3D temperature
slp   = getvar(f, "slp",  timeidx=0, units="hPa")    # sea-level pressure
wspd  = getvar(f, "wspd", timeidx=0, units="knots")   # 3D wind speed

# Convective parameters (the whole point)
sbcape = getvar(f, "sbcape", timeidx=0)   # surface-based CAPE (J/kg)
mlcape = getvar(f, "mlcape", timeidx=0)   # mixed-layer CAPE
mucape = getvar(f, "mucape", timeidx=0)   # most-unstable CAPE
mlcin  = getvar(f, "mlcin",  timeidx=0)   # mixed-layer CIN
lcl    = getvar(f, "lcl",    timeidx=0)   # LCL height (m AGL)

# Generic CAPE with parcel selection
cape  = getvar(f, "cape", parcel_type="ml")         # same as mlcape
cape3 = getvar(f, "cape", parcel_type="sb", top_m=3000)  # 0-3 km CAPE

# Custom parcel: specify starting conditions directly
cape_custom = getvar(f, "cape",
                     parcel_pressure=900,       # hPa
                     parcel_temperature=25,     # deg C
                     parcel_dewpoint=18)        # deg C

# Storm-relative helicity with proper Bunkers
srh1 = getvar(f, "srh1", timeidx=0)   # 0-1 km SRH
srh3 = getvar(f, "srh3", timeidx=0)   # 0-3 km SRH

# Custom storm motion
srh_custom = getvar(f, "srh", timeidx=0,
                    depth_m=3000, storm_motion=(12.0, 8.0))

# Effective inflow layer SRH (no more manual layer-finding!)
eff_srh = getvar(f, "effective_srh")

# Severe composites
stp     = getvar(f, "stp")                           # fixed-layer STP (default)
stp_eff = getvar(f, "stp", layer_type="effective")   # effective-layer STP
scp     = getvar(f, "scp")
ehi_1km = getvar(f, "ehi")                           # 0-1 km EHI (default)
ehi_3km = getvar(f, "ehi", depth_m=3000)             # 0-3 km EHI

# Configurable shear and mean wind
shr_0_3 = getvar(f, "bulk_shear", bottom_m=0, top_m=3000)
shr_1_6 = getvar(f, "bulk_shear", bottom_m=1000, top_m=6000)
mw      = getvar(f, "mean_wind",  bottom_m=0, top_m=6000)

# Configurable lapse rates
lr_03 = getvar(f, "lapse_rate", bottom_m=0, top_m=3000)
lr_36 = getvar(f, "lapse_rate", bottom_m=3000, top_m=6000)
lr_vt = getvar(f, "lapse_rate", bottom_m=0, top_m=3000, use_virtual=True)

# Updraft helicity with custom layer
uh_03 = getvar(f, "uhel", bottom_m=0, top_m=3000)   # 0-3 km UH

# All timesteps at once
slp_all = getvar(f, "slp", timeidx=ALL_TIMES)  # shape (nt, ny, nx)

# Method syntax works too
temp = f.getvar("temp", units="degF")
```

### Works with netCDF4.Dataset

If you have existing code using `netCDF4.Dataset`, wrf-rust accepts it directly:

```python
from netCDF4 import Dataset
from wrf import getvar

nc = Dataset("wrfout_d01_2024-06-01_00:00:00")
slp = getvar(nc, "slp", timeidx=0)  # auto-wraps the Dataset
```

### Discover available variables

```python
from wrf import available_variables
available_variables()   # prints formatted table

from wrf import list_variables
vars = list_variables() # returns list of dicts
```

## Variable reference

73 diagnostic variables organized by category. Every variable supports the `units=` kwarg.

### Grid & coordinates

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `pressure` | `pres`, `p` | Full model pressure (P+PB) | Pa | 3D |
| `height` | `z`, `height_msl` | Model height MSL | m | 3D |
| `height_agl` | `z_agl` | Model height AGL | m | 3D |
| `zstag` | `height_stag` | Height on staggered Z levels | m | 3D |
| `geopt` | `geopotential` | Full geopotential (PH+PHB) | m2/s2 | 3D |
| `geopt_stag` | `geopotential_stag` | Geopotential on staggered Z | m2/s2 | 3D |
| `terrain` | `ter`, `hgt` | Terrain height | m | 2D |
| `lat` | `xlat`, `latitude` | Latitude | degrees | 2D |
| `lon` | `xlong`, `longitude` | Longitude | degrees | 2D |

### Temperature & thermodynamics

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `temp` | `tk` | Temperature | K | 3D |
| `tc` | `temp_c` | Temperature | degC | 3D |
| `theta` | `th` | Potential temperature | K | 3D |
| `theta_e` | `eth` | Equivalent potential temperature | K | 3D |
| `theta_w` | | Wet-bulb potential temperature | K | 3D |
| `tv` | `virtual_temperature` | Virtual temperature | K | 3D |
| `twb` | `wet_bulb` | Wet-bulb temperature | K | 3D |
| `td` | `dp`, `dewpoint` | Dewpoint temperature | degC | 3D |
| `rh` | `relative_humidity` | Relative humidity | % | 3D |

### Pressure & vertical motion

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `slp` | `mslp`, `sea_level_pressure` | Sea-level pressure | hPa | 2D |
| `omega` | `vertical_velocity_pressure` | Vertical velocity (pressure coords) | Pa/s | 3D |

### Moisture

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `pw` | `precipitable_water` | Precipitable water | mm | 2D |
| `rh2m` | `rh2` | 2-m relative humidity | % | 2D |
| `dp2m` | `td2`, `td2m` | 2-m dewpoint temperature | degC | 2D |
| `mixing_ratio` | `qvapor` | Water vapor mixing ratio | kg/kg | 3D |
| `specific_humidity` | `q` | Specific humidity | kg/kg | 3D |

### CAPE & convective parameters

These use proper parcel definitions matching SPC/SHARPpy. Powered by [wx-math](https://github.com/FahrenheitResearch/rustmet) `compute_cape_cin()` with rayon parallelism.

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `sbcape` | `surface_based_cape` | Surface-based CAPE | J/kg | 2D |
| `sbcin` | `surface_based_cin` | Surface-based CIN | J/kg | 2D |
| `mlcape` | `mixed_layer_cape` | Mixed-layer CAPE (100 hPa depth) | J/kg | 2D |
| `mlcin` | `mixed_layer_cin` | Mixed-layer CIN | J/kg | 2D |
| `mucape` | `most_unstable_cape` | Most-unstable CAPE (300 hPa search) | J/kg | 2D |
| `mucin` | `most_unstable_cin` | Most-unstable CIN | J/kg | 2D |
| `lcl` | `lcl_height` | LCL height AGL | m | 2D |
| `lfc` | `lfc_height` | LFC height AGL | m | 2D |
| `el` | `equilibrium_level` | Equilibrium level height AGL | m | 2D |
| `cape` | | Generic CAPE (`parcel_type` or custom parcel) | J/kg | 2D |
| `cin` | | Generic CIN (`parcel_type` or custom parcel) | J/kg | 2D |
| `effective_cape` | `eff_cape` | MUCAPE within effective inflow layer | J/kg | 2D |
| `effective_inflow` | `eff_inflow` | Effective inflow layer base/top heights | m | 2D* |
| `cape2d` | | CAPE/CIN/LCL/LFC (backward-compat) | J/kg | 2D* |
| `cape3d` | | 3-D CAPE field | J/kg | 3D |

Use `top_m=3000` with any CAPE variable for 0-3 km CAPE (3CAPE).

The generic `cape`/`cin`/`lcl`/`lfc`/`el` variables accept:
- `parcel_type="sb"`, `"ml"`, or `"mu"` (default: `"sb"`)
- Custom parcel via `parcel_pressure` (hPa), `parcel_temperature` (degC), `parcel_dewpoint` (degC)
- `top_m` for truncated CAPE (e.g. `top_m=3000` for 3CAPE)

### Wind

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `ua` | `u_destag` | U-wind (destaggered) | m/s | 3D |
| `va` | `v_destag` | V-wind (destaggered) | m/s | 3D |
| `wa` | `w_destag` | W-wind (destaggered) | m/s | 3D |
| `wspd` | `wind_speed` | Wind speed | m/s | 3D |
| `wdir` | `wind_direction` | Wind direction (earth-relative) | degrees | 3D |
| `uvmet` | `earth_rotated_wind` | U/V rotated to earth coordinates | m/s | 3D* |
| `uvmet10` | `earth_rotated_wind_10m` | 10-m U/V earth-relative | m/s | 2D* |
| `wspd10` | `wind_speed_10m` | 10-m wind speed | m/s | 2D |
| `wdir10` | `wind_direction_10m` | 10-m wind direction | degrees | 2D |

### Storm-relative helicity & shear

Uses Bunkers Internal Dynamics method (Bunkers et al. 2000), not wrf-python's non-standard approach. Supports custom storm motion via `storm_motion=(u, v)`.

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `srh1` | `srh_0_1km` | 0-1 km SRH (Bunkers RM) | m2/s2 | 2D |
| `srh3` | `srh_0_3km` | 0-3 km SRH (Bunkers RM) | m2/s2 | 2D |
| `srh` | `storm_relative_helicity` | SRH (configurable `depth_m`) | m2/s2 | 2D |
| `effective_srh` | `srh_eff` | SRH over effective inflow layer | m2/s2 | 2D |
| `shear_0_1km` | `shr01` | 0-1 km bulk wind shear | m/s | 2D |
| `shear_0_6km` | `shr06` | 0-6 km bulk wind shear | m/s | 2D |
| `bulk_shear` | `shear` | Bulk shear (configurable `bottom_m`/`top_m`) | m/s | 2D |
| `bunkers_rm` | `bunkers_right` | Bunkers right-mover motion | m/s | 2D* |
| `bunkers_lm` | `bunkers_left` | Bunkers left-mover motion | m/s | 2D* |
| `mean_wind_0_6km` | `mean_wind_6km` | 0-6 km mean wind | m/s | 2D* |
| `mean_wind` | | Mean wind (configurable `bottom_m`/`top_m`) | m/s | 2D* |

### Severe weather composites

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `stp` | `significant_tornado_parameter` | STP -- fixed (default) or effective via `layer_type` | dimensionless | 2D |
| `stp_fixed` | | STP fixed-layer only | dimensionless | 2D |
| `stp_effective` | `stp_eff` | STP effective-layer only | dimensionless | 2D |
| `scp` | `supercell_composite_parameter` | SCP (Thompson et al. 2004) | dimensionless | 2D |
| `ehi` | `energy_helicity_index` | EHI (configurable SRH `depth_m`, default 0-1 km) | dimensionless | 2D |
| `critical_angle` | `crit_angle` | Critical angle (Esterheld & Giuliano 2008) | degrees | 2D |
| `ship` | `significant_hail_parameter` | Significant Hail Parameter | dimensionless | 2D |
| `bri` | `bulk_richardson_number` | Bulk Richardson Number | dimensionless | 2D |

### Radar & cloud

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `dbz` | `reflectivity` | Simulated reflectivity | dBZ | 3D |
| `maxdbz` | `composite_reflectivity` | Maximum (composite) reflectivity | dBZ | 2D |
| `ctt` | `cloud_top_temperature` | Cloud-top temperature | degC | 2D |
| `cloudfrac` | `cloud_fraction` | Cloud fraction (low/mid/high) | % | 2D* |
| `uhel` | `updraft_helicity` | Updraft helicity (configurable `bottom_m`/`top_m`, default 2-5 km) | m2/s2 | 2D |

### Vorticity

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `avo` | `absolute_vorticity` | Absolute vorticity | s-1 | 3D |
| `pvo` | `potential_vorticity` | Potential vorticity | PVU | 3D |

### Lapse rates & levels

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `lapse_rate_700_500` | `lr75` | 700-500 hPa lapse rate | degC/km | 2D |
| `lapse_rate_0_3km` | `lr03` | 0-3 km AGL lapse rate | degC/km | 2D |
| `lapse_rate` | `lr` | Lapse rate (configurable `bottom_m`/`top_m`, `use_virtual`) | degC/km | 2D |
| `freezing_level` | `fzlev` | Freezing level height AGL | m | 2D |
| `wet_bulb_0` | `wb0` | Wet-bulb zero height AGL | m | 2D |

### Fire weather

| Variable | Aliases | Description | Default units | Shape |
|---|---|---|---|---|
| `fosberg` | `fwi` | Fosberg Fire Weather Index | dimensionless | 2D |
| `haines` | `haines_index` | Haines Index | dimensionless | 2D |
| `hdw` | `hot_dry_windy` | Hot-Dry-Windy Index | dimensionless | 2D |

Variables marked with * return multi-field arrays (e.g., `uvmet` returns U and V stacked, `cape2d` returns CAPE/CIN/LCL/LFC stacked).

## Configurable parameters

Most hardcoded convenience variables (like `srh1`, `shear_0_6km`, `lapse_rate_0_3km`) have generic counterparts that accept configurable bounds. The philosophy: **hardcoded names for common cases, configurable names for everything else.**

| Parameter | Type | Used by | Description |
|---|---|---|---|
| `parcel_type` | `str` | `cape`, `cin`, `lcl`, `lfc`, `el`, `sbcape`... | Parcel selection: `"sb"`, `"ml"`, `"mu"` |
| `parcel_pressure` | `float` | `cape`, `cin`, `lcl`, `lfc`, `el` | Custom parcel starting pressure (hPa) |
| `parcel_temperature` | `float` | `cape`, `cin`, `lcl`, `lfc`, `el` | Custom parcel starting temperature (degC) |
| `parcel_dewpoint` | `float` | `cape`, `cin`, `lcl`, `lfc`, `el` | Custom parcel starting dewpoint (degC) |
| `top_m` | `float` | `cape`, `cin`, `bulk_shear`, `mean_wind`, `lapse_rate`, `uhel` | Top of integration layer (m AGL) |
| `bottom_m` | `float` | `bulk_shear`, `mean_wind`, `lapse_rate`, `uhel` | Bottom of layer (m AGL) |
| `depth_m` | `float` | `srh`, `ehi` | SRH integration depth (m AGL) |
| `storm_motion` | `(float, float)` | `srh`, `srh1`, `srh3`, `effective_srh` | Custom storm motion (u, v) in m/s |
| `layer_type` | `str` | `stp` | `"fixed"` (default) or `"effective"` |
| `use_virtual` | `bool` | `lapse_rate` | Use virtual temperature instead of absolute |

### Custom parcel example

```python
# Lift a parcel from 850 hPa with T=20C, Td=15C
cape_850 = getvar(f, "cape",
                  parcel_pressure=850,
                  parcel_temperature=20,
                  parcel_dewpoint=15)

# Same parcel, but only integrate to 3 km (3CAPE)
cape_850_3km = getvar(f, "cape",
                      parcel_pressure=850,
                      parcel_temperature=20,
                      parcel_dewpoint=15,
                      top_m=3000)
```

### Effective inflow layer

The effective inflow layer is the contiguous layer where parcels have CAPE >= 100 J/kg and CIN >= -250 J/kg. This is the layer that actually feeds a storm.

```python
# Effective inflow layer bounds (base height, top height in m AGL)
eff_layer = getvar(f, "effective_inflow")  # shape (2, ny, nx)

# MUCAPE within the effective layer
eff_cape = getvar(f, "effective_cape")

# SRH over the effective layer (replaces the need for manual layer-finding)
eff_srh = getvar(f, "effective_srh")

# Effective-layer STP (uses effective CAPE + effective SRH)
stp_eff = getvar(f, "stp", layer_type="effective")
```

## Unit conversion

Every variable supports the `units=` parameter. Unit strings are case-insensitive.

| Category | Accepted strings |
|---|---|
| Temperature | `K`, `degC`, `C`, `degF`, `F`, `celsius`, `fahrenheit`, `kelvin` |
| Pressure | `Pa`, `hPa`, `mb`, `mbar`, `inHg` |
| Speed | `m/s`, `knots`, `kt`, `kts`, `mph`, `kph`, `km/h` |
| Length/Height | `m`, `ft`, `km`, `mi`, `feet`, `miles` |
| Moisture | `kg/kg`, `g/kg`, `%` |
| Depth | `mm`, `in`, `inches` |

```python
temp = getvar(f, "temp", units="degF")        # Fahrenheit
slp  = getvar(f, "slp",  units="inHg")        # inches of mercury
wspd = getvar(f, "wspd", units="knots")        # knots
hgt  = getvar(f, "height_agl", units="ft")     # feet
pw   = getvar(f, "pw",   units="in")           # inches
```

## Migrating from wrf-python

### Side-by-side comparison

```python
# ──── wrf-python ────
from netCDF4 import Dataset
import wrf

ncfile = Dataset("wrfout_d01_2024-06-01_00:00:00")
slp   = wrf.getvar(ncfile, "slp",    timeidx=0)
tk    = wrf.getvar(ncfile, "tk",     timeidx=0)
cape  = wrf.getvar(ncfile, "cape_2d", timeidx=0)  # what parcel?
srh   = wrf.getvar(ncfile, "srh",    timeidx=0)   # not Bunkers

# ──── wrf-rust ────
from wrf import WrfFile, getvar

f     = WrfFile("wrfout_d01_2024-06-01_00:00:00")
slp   = getvar(f, "slp",    timeidx=0)
tk    = getvar(f, "temp",   timeidx=0)             # or "tk" works
sbcape = getvar(f, "sbcape", timeidx=0)            # explicit parcel
srh1  = getvar(f, "srh1",   timeidx=0)             # proper Bunkers
```

### Key differences

| | wrf-python | wrf-rust |
|---|---|---|
| File handle | `netCDF4.Dataset` | `WrfFile` (also accepts `Dataset`) |
| Returns | `xarray.DataArray` with coords | `numpy.ndarray` |
| CAPE | `cape_2d` (ambiguous parcel) | `sbcape`, `mlcape`, `mucape` |
| SRH | Non-Bunkers storm motion | Bunkers (default) or custom |
| Units | Inconsistent support | Universal `units=` on everything |
| All times | `timeidx=wrf.ALL_TIMES` | `timeidx=ALL_TIMES` (or `None`) |
| Variable names | `tk`, `eth`, `cape_2d` | `temp`, `theta_e`, `sbcape` (old names work as aliases) |

### What's different about the output

wrf-rust returns **numpy arrays**, not xarray DataArrays. If you need lat/lon coordinates for plotting:

```python
lat = getvar(f, "lat", timeidx=0)
lon = getvar(f, "lon", timeidx=0)

import matplotlib.pyplot as plt
plt.contourf(lon, lat, slp)
```

### Name mapping

Most wrf-python variable names work as aliases:

| wrf-python | wrf-rust primary | Status |
|---|---|---|
| `tk` | `temp` | Alias works |
| `tc` | `tc` | Same |
| `th` | `theta` | Alias works |
| `eth` | `theta_e` | Alias works |
| `cape_2d` | `cape2d` | Use `sbcape`/`mlcape`/`mucape` instead |
| `cape_3d` | `cape3d` | Same |
| `slp` | `slp` | Same |
| `rh` | `rh` | Same |
| `ua` | `ua` | Same |
| `va` | `va` | Same |
| `wa` | `wa` | Same |
| `dbz` | `dbz` | Same |
| `pw` | `pw` | Same |
| `avo` | `avo` | Same |
| `pvo` | `pvo` | Same |

## Architecture

```
wrf-rust/
  crates/wrf-core/         Pure Rust library (no Python dependency)
    src/
      file.rs              WrfFile: NetCDF reading + field caching
      grid.rs              Arakawa C-grid destaggering (rayon-parallel)
      extract.rs           Perturbation decomposition (P+PB, PH+PHB, T+300)
      variables.rs         Variable registry (name -> compute function)
      compute.rs           getvar dispatch + unit conversion
      units.rs             Unit parsing and conversion
      projection.rs        WRF map projection extraction
      multi.rs             Multi-file time concatenation
      diag/                12 diagnostic modules
        thermo.rs          temp, theta, theta_e, tv, twb, td, rh
        pressure.rs        slp, pressure, height, omega, geopotential
        wind.rs            ua/va/wa, rotation, wspd/wdir
        cape.rs            SBCAPE/MLCAPE/MUCAPE via wx-math
        srh.rs             SRH via Bunkers, shear, storm motion
        moisture.rs        pw, rh2m, dp2m, mixing ratio
        severe.rs          STP, SCP, EHI, critical angle, SHIP, BRN
        radar.rs           Simulated reflectivity (Smith 1984)
        cloud.rs           Cloud-top temperature, cloud fraction
        vorticity.rs       Absolute and potential vorticity
        helicity.rs        Updraft helicity
        extra.rs           Lapse rates, freezing level, fire indices
  src/                     PyO3 bindings
    py_file.rs             WrfFile Python class
    py_getvar.rs           getvar() + list_variables()
  python/wrf/              Python wrapper
    __init__.py            User-facing API, Dataset compat, ALL_TIMES
```

### Dependencies

- **[wx-math](https://github.com/FahrenheitResearch/rustmet)** -- Core meteorological math: `cape_cin_core`, `compute_cape_cin`, `compute_srh`, `bunkers_storm_motion`, thermodynamics, stability indices
- **[metrust](https://github.com/FahrenheitResearch/metrust)** -- Higher-level wrappers: `storm_relative_helicity`, `significant_tornado_parameter`, `supercell_composite_parameter`, `critical_angle`
- **[netcdf](https://crates.io/crates/netcdf)** -- NetCDF4 file reading (wraps system libnetcdf)
- **[rayon](https://crates.io/crates/rayon)** -- Data-parallel computation across grid columns
- **[PyO3](https://pyo3.rs/) + [numpy](https://crates.io/crates/numpy)** -- Python bindings
- **[maturin](https://maturin.rs/)** -- Build system for Python packages with Rust

## Building from source

### Prerequisites

1. **Rust toolchain** -- [rustup.rs](https://rustup.rs/)
2. **System libnetcdf + libhdf5** -- easiest via conda:
   ```bash
   conda install -c conda-forge libnetcdf hdf5
   ```
3. **maturin** -- `pip install maturin`
4. **wx-math and metrust** -- clone the dependency repos:
   ```bash
   git clone https://github.com/FahrenheitResearch/rustmet.git ~/rustmet
   git clone https://github.com/FahrenheitResearch/metrust.git ~/metrust
   ```

### Build

```bash
git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust

# Update .cargo/config.toml with your HDF5/NetCDF paths if different
# Update crates/wrf-core/Cargo.toml with your wx-math/metrust paths if different

# Development build
maturin develop

# Release build (optimized)
maturin develop --release

# Check Rust compilation only
cargo check
```

### Windows notes

The `.cargo/config.toml` sets HDF5/NetCDF paths for the conda `wrfplot` environment. You also need the DLLs on your PATH at runtime:

```bash
set PATH=%PATH%;C:\Users\drew\miniforge3\envs\wrfplot\Library\bin
```

## References

- Bunkers, M.J., B.A. Klimowski, J.W. Zeitler, R.L. Thompson, and M.L. Weisman, 2000: Predicting supercell motion using a new hodograph technique. *Wea. Forecasting*, **15**, 61-79.
- Thompson, R.L., R. Edwards, J.A. Hart, K.L. Elmore, and P. Markowski, 2003: Close proximity soundings within supercell environments obtained from the Rapid Update Cycle. *Wea. Forecasting*, **18**, 1243-1261.
- Thompson, R.L., C.M. Mead, and R. Edwards, 2004: Effective storm-relative helicity and bulk shear in supercell thunderstorm environments. Preprints, *22nd Conf. on Severe Local Storms*, Hyannis, MA.
- Esterheld, J.M. and D.J. Giuliano, 2008: Discriminating between tornadic and non-tornadic supercells: a new hodograph technique. *E-Journal of Severe Storms Meteorology*, **3(2)**.
- Smith, P.L., 1984: Equivalent radar reflectivity factors for snow and ice particles. *J. Climate Appl. Meteor.*, **23**, 1258-1260.

## License

MIT
