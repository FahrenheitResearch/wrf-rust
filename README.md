# wrf-rust

Rust-powered WRF post-processing with Python bindings. 80+ diagnostic variables, configurable CAPE/SRH/shear/lapse rates, built-in plotting, and parallel computation via [rayon](https://crates.io/crates/rayon).

Built on top of [wx-math](https://github.com/FahrenheitResearch/rustmet) and [metrust](https://github.com/FahrenheitResearch/metrust) for meteorological calculations.

## Install

Requires Rust, maturin, and system libnetcdf/libhdf5 (easiest from conda).

```bash
conda install -c conda-forge libnetcdf hdf5
pip install maturin

git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust
maturin develop --release
```

## Usage

```python
from wrf import WrfFile, getvar

f = WrfFile("wrfout_d01_2024-06-01_00:00:00")

# Basic fields
temp = getvar(f, "temp", units="degC")
slp  = getvar(f, "slp",  units="hPa")
wspd = getvar(f, "wspd", units="knots")

# CAPE -- select your parcel type
sbcape = getvar(f, "sbcape")                             # surface-based
mlcape = getvar(f, "mlcape")                             # 100 hPa mixed-layer
mucape = getvar(f, "mucape")                             # most unstable
sb3cap = getvar(f, "sbcape", top_m=3000)                 # 0-3 km CAPE

# Custom parcel
cape = getvar(f, "cape", parcel_pressure=850,
              parcel_temperature=20, parcel_dewpoint=15)

# SRH with Bunkers storm motion
srh1 = getvar(f, "srh1")                                 # 0-1 km
srh3 = getvar(f, "srh3")                                 # 0-3 km
srh  = getvar(f, "srh", depth_m=1500, storm_motion=(12, 8))

# Effective inflow layer
eff_srh  = getvar(f, "effective_srh")
eff_cape = getvar(f, "effective_cape")

# Severe composites
stp     = getvar(f, "stp")                                # fixed-layer
stp_eff = getvar(f, "stp", layer_type="effective")        # effective-layer
scp     = getvar(f, "scp")
ehi     = getvar(f, "ehi", depth_m=3000)                  # 0-3 km EHI

# Configurable layers
shear = getvar(f, "bulk_shear", bottom_m=1000, top_m=6000)
mw    = getvar(f, "mean_wind",  bottom_m=0, top_m=6000)
lr    = getvar(f, "lapse_rate", bottom_p=700, top_p=500)  # pressure-based
lr_v  = getvar(f, "lapse_rate", bottom_m=0, top_m=3000, use_virtual=True)

# All timesteps
slp_all = getvar(f, "slp", timeidx=None)                  # shape (nt, ny, nx)
```

Also accepts `netCDF4.Dataset` directly:

```python
from netCDF4 import Dataset
slp = getvar(Dataset("wrfout_d01..."), "slp")
```

## Plotting

```python
from wrf import plot_field, plot_wind, plot_skewt, panel

plot_field(f, "sbcape")                          # auto colormap + cartopy map
plot_field(f, "slp", units="hPa")                # unit conversion
plot_wind(f)                                      # wind barbs over speed fill
plot_skewt(f, point=(35.0, -97.5))               # Skew-T with hodograph
panel(f, ["sbcape", "srh1", "stp", "shear_0_6km"])  # multi-panel
```

Multi-timestep rendering with consistent scales and GIF output:

```python
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

## Variables

80+ diagnostic variables. All support the `units=` parameter.

### Thermodynamics

| Variable | Aliases | Units | Description |
|---|---|---|---|
| `temp` | `tk` | K | Temperature |
| `tc` | `temp_c` | degC | Temperature (Celsius) |
| `theta` | `th` | K | Potential temperature |
| `theta_e` | `eth` | K | Equivalent potential temperature |
| `theta_w` | | K | Wet-bulb potential temperature |
| `tv` | | K | Virtual temperature |
| `twb` | `wet_bulb` | K | Wet-bulb temperature |
| `td` | `dp`, `dewpoint` | degC | Dewpoint |
| `rh` | | % | Relative humidity |

### Pressure & height

| Variable | Aliases | Units | Description |
|---|---|---|---|
| `pressure` | `pres`, `p` | Pa | Full model pressure |
| `slp` | `mslp` | hPa | Sea-level pressure |
| `height` | `z` | m | Height MSL |
| `height_agl` | `z_agl` | m | Height AGL |
| `terrain` | `ter`, `hgt` | m | Terrain height |
| `geopt` | | m2/s2 | Geopotential |
| `omega` | | Pa/s | Vertical velocity (pressure coords) |

### Moisture

| Variable | Aliases | Units | Description |
|---|---|---|---|
| `pw` | `precipitable_water` | mm | Precipitable water |
| `rh2m` | `rh2` | % | 2-m relative humidity |
| `dp2m` | `td2` | degC | 2-m dewpoint |
| `mixing_ratio` | `qvapor` | kg/kg | Water vapor mixing ratio |
| `specific_humidity` | `q` | kg/kg | Specific humidity |

### CAPE & convection

All CAPE variables support `top_m` for truncated integration (e.g. `top_m=3000` for 3CAPE).

| Variable | Units | Description |
|---|---|---|
| `sbcape` / `sbcin` | J/kg | Surface-based CAPE/CIN |
| `mlcape` / `mlcin` | J/kg | Mixed-layer CAPE/CIN (100 hPa) |
| `mucape` / `mucin` | J/kg | Most-unstable CAPE/CIN |
| `cape` / `cin` | J/kg | Generic (accepts `parcel_type` or custom parcel) |
| `lcl` / `lfc` / `el` | m | LCL / LFC / EL height AGL |
| `effective_cape` | J/kg | MUCAPE within effective inflow layer |
| `effective_inflow` | m | Effective inflow base/top heights |

### Wind

| Variable | Aliases | Units | Description |
|---|---|---|---|
| `ua` / `va` / `wa` | | m/s | Destaggered U/V/W wind |
| `wspd` / `wdir` | | m/s, deg | Wind speed / direction |
| `wspd10` / `wdir10` | | m/s, deg | 10-m wind speed / direction |
| `uvmet` / `uvmet10` | | m/s | Earth-rotated wind components |

### SRH & shear

SRH uses Bunkers Internal Dynamics method. All accept `storm_motion=(u,v)` for custom motion.

| Variable | Units | Description |
|---|---|---|
| `srh1` / `srh3` | m2/s2 | 0-1 / 0-3 km SRH |
| `srh` | m2/s2 | Configurable depth via `depth_m` |
| `effective_srh` | m2/s2 | SRH over effective inflow layer |
| `shear_0_1km` / `shear_0_6km` | m/s | Fixed-layer bulk shear |
| `bulk_shear` | m/s | Configurable via `bottom_m` / `top_m` |
| `mean_wind` | m/s | Configurable via `bottom_m` / `top_m` |
| `bunkers_rm` / `bunkers_lm` | m/s | Bunkers right/left-mover motion |

### Severe composites

| Variable | Description |
|---|---|
| `stp` | Significant Tornado Parameter (fixed or `layer_type="effective"`) |
| `scp` | Supercell Composite Parameter |
| `ehi` | Energy-Helicity Index (configurable SRH depth via `depth_m`) |
| `critical_angle` | Critical angle |
| `ship` | Significant Hail Parameter |
| `bri` | Bulk Richardson Number |

### Radar & cloud

| Variable | Units | Description |
|---|---|---|
| `dbz` / `maxdbz` | dBZ | Simulated / composite reflectivity |
| `ctt` | degC | Cloud-top temperature |
| `cloudfrac` | % | Cloud fraction (low/mid/high) |
| `uhel` | m2/s2 | Updraft helicity (configurable `bottom_m`/`top_m`) |

### Vorticity

| Variable | Units | Description |
|---|---|---|
| `avo` | s-1 | Absolute vorticity |
| `pvo` | PVU | Potential vorticity |

### Lapse rates & levels

| Variable | Units | Description |
|---|---|---|
| `lapse_rate_700_500` | degC/km | 700-500 hPa lapse rate |
| `lapse_rate_0_3km` | degC/km | 0-3 km lapse rate |
| `lapse_rate` | degC/km | Configurable (`bottom_m`/`top_m` or `bottom_p`/`top_p`, `use_virtual`) |
| `freezing_level` | m | Freezing level AGL |
| `wet_bulb_0` | m | Wet-bulb zero height AGL |

### Fire weather

| Variable | Description |
|---|---|
| `fosberg` | Fosberg Fire Weather Index |
| `haines` | Haines Index |
| `hdw` | Hot-Dry-Windy Index |

## Configurable parameters

Many variables have both hardcoded convenience names and generic configurable versions.

| Parameter | Type | Description |
|---|---|---|
| `units` | `str` | Output unit conversion (works on every variable) |
| `parcel_type` | `str` | `"sb"`, `"ml"`, `"mu"` for CAPE variables |
| `parcel_pressure` | `float` | Custom parcel pressure (hPa) |
| `parcel_temperature` | `float` | Custom parcel temperature (degC) |
| `parcel_dewpoint` | `float` | Custom parcel dewpoint (degC) |
| `top_m` | `float` | Top of layer in m AGL |
| `bottom_m` | `float` | Bottom of layer in m AGL |
| `top_p` | `float` | Top of layer in hPa (lapse rates) |
| `bottom_p` | `float` | Bottom of layer in hPa (lapse rates) |
| `depth_m` | `float` | SRH/EHI integration depth (m AGL) |
| `storm_motion` | `(u, v)` | Custom storm motion in m/s |
| `layer_type` | `str` | `"fixed"` or `"effective"` for STP |
| `use_virtual` | `bool` | Virtual temperature for lapse rates |

## Unit conversion

Every variable supports `units=`. Case-insensitive.

| Category | Strings |
|---|---|
| Temperature | `K`, `degC`, `degF`, `celsius`, `fahrenheit` |
| Pressure | `Pa`, `hPa`, `mb`, `inHg` |
| Speed | `m/s`, `knots`, `kt`, `mph`, `kph` |
| Length | `m`, `ft`, `km`, `mi` |
| Moisture | `kg/kg`, `g/kg` |
| Depth | `mm`, `in` |

## Building from source

### Prerequisites

1. **Rust** -- [rustup.rs](https://rustup.rs/)
2. **libnetcdf + libhdf5** -- `conda install -c conda-forge libnetcdf hdf5`
3. **maturin** -- `pip install maturin`
4. **wx-math + metrust** -- the meteorological math libraries:
   ```bash
   git clone https://github.com/FahrenheitResearch/rustmet.git ~/rustmet
   git clone https://github.com/FahrenheitResearch/metrust.git ~/metrust
   ```

### Build

```bash
git clone https://github.com/FahrenheitResearch/wrf-rust.git
cd wrf-rust
maturin develop --release
```

Update `crates/wrf-core/Cargo.toml` if your wx-math/metrust paths differ. Update `.cargo/config.toml` for HDF5/NetCDF library paths.

### Windows

The `.cargo/config.toml` points to conda's `wrfplot` environment for HDF5/NetCDF. The Python module auto-discovers DLL paths from common conda locations on import.

## Architecture

```
wrf-rust/
  crates/wrf-core/         Pure Rust library
    src/
      file.rs              NetCDF reading + field caching
      grid.rs              Arakawa C-grid destaggering
      variables.rs         Variable registry (80+ entries)
      compute.rs           getvar dispatch + unit conversion
      diag/                12 diagnostic modules
        cape.rs            SB/ML/MU CAPE, custom parcels, effective inflow
        srh.rs             Bunkers SRH, effective SRH, shear, mean wind
        severe.rs          STP (fixed + effective), SCP, EHI, critical angle
        thermo.rs          Temperature variants, theta-e, dewpoint, RH
        pressure.rs        SLP, height, omega, geopotential
        wind.rs            Destaggering, rotation, speed/direction
        moisture.rs        PW, mixing ratio, 2m fields
        radar.rs           Simulated reflectivity
        cloud.rs           Cloud-top temp, cloud fraction
        vorticity.rs       Absolute and potential vorticity
        helicity.rs        Updraft helicity
        extra.rs           Lapse rates, freezing level, fire indices
  src/                     PyO3 bindings
  python/wrf/              Python API
    __init__.py            getvar, WrfFile, ALL_TIMES
    plot.py                Plotting (matplotlib + cartopy)
    cli.py                 CLI (python -m wrf)
```

## License

MIT
