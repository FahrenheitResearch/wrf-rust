# Build And Run

This file is the stable WRF/WPS workflow.

## Recommended philosophy

- Build CPU WRF first.
- Build WPS second.
- Run a tiny sanity case.
- Only then move to real-data prep and larger domains.

## Official anchor

The closest official workflow is the WRF compile tutorial:

- compile WRF first
- compile WPS second
- for WPS `v4.4+`, use `./configure --build-grib2-libs`

## Community fast path on Ubuntu or WSL

Install the usual build stack first:

```bash
sudo apt update
sudo apt install -y \
  build-essential gcc g++ gfortran make m4 perl csh tcsh flex bison \
  curl file git cmake \
  libnetcdf-dev libnetcdff-dev \
  libopenmpi-dev openmpi-bin \
  zlib1g-dev libpng-dev
```

If that path becomes weird or inconsistent, fall back to the stricter official-library build workflow from the WRF compile tutorial.

## Build order

### WRF

```bash
git clone https://github.com/wrf-model/WRF.git
cd WRF
export NETCDF=/usr
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
./configure
./compile em_real -j "$(nproc)"
```

Guide the user to choose:

- GNU serial for the smallest and simplest starting point
- or GNU dmpar if they already intend to use MPI

Do not rely on fixed menu numbers. They vary.

The success condition is that `main/wrf.exe` and `main/real.exe` exist.

### WPS

```bash
git clone https://github.com/wrf-model/WPS.git
cd WPS
./configure --build-grib2-libs
export WRF_DIR=/path/to/WRF
./compile
```

The success condition is:

- `geogrid.exe`
- `ungrib.exe`
- `metgrid.exe`

## Static geography is part of "WPS works"

Do not leave out the WPS geographical data.

- Download the official WPS static geography package.
- The highest-resolution mandatory package is the normal default and is about `29 GB` uncompressed.
- Point `geog_data_path` in `namelist.wps` at that directory before running `geogrid.exe`.

## Starter files in this repo

The repo ships a practical starter pack under `docs/starter-files/`.

Useful files:

- `namelist.wps.ecmwf-starter`
- `namelist.input.ecmwf-starter`
- `Vtable.ECMWF_opendata`

Use them as a strong starting point, but still edit:

- dates
- domain center and projection
- `geog_data_path`
- `num_metgrid_levels`
- `num_metgrid_soil_levels`

## Stable run sequence

1. Set up geog data and `namelist.wps`
2. Run `geogrid.exe`
3. Link meteorological GRIB input
4. Choose the correct Vtable
5. Run `ungrib.exe`
6. If the forcing uses multiple GRIB families, repeat `ungrib.exe` with distinct prefixes and list them in `fg_name`
7. Run `metgrid.exe`
8. Copy `met_em*` into the WRF run directory
9. Prepare `namelist.input`
10. Run `real.exe`
11. Confirm `wrfinput_d01` and `wrfbdy_d01` exist
12. Run `wrf.exe`

## ECMWF-specific WPS pattern

If a user is staging ECMWF fields separately, a practical community pattern is:

- pressure fields under a prefix like `FILE`
- surface fields under a prefix like `SFILE`
- soil fields under a prefix like `SOILFILE`
- `fg_name = 'SFILE', 'SOILFILE', 'FILE'` in `namelist.wps`

That pattern is useful, but remind users it is not magic by itself:

- they still need the correct Vtable and field coverage
- they may need to run `ungrib.exe` multiple times
- the exact prefixes can vary as long as they are used consistently

## ECMWF compression gotcha

If ECMWF GRIB2 files fail in `ungrib.exe`, check whether they need to be rewritten first with `cdo`.

Example:

```bash
cdo -f grb2 copy input.grib2 output.grib2
```

Community scripts often automate this because CCSDS-compressed files are a real WPS pain point.

## Sanity checks

After `real.exe`:

- `wrfinput_d01` exists
- `wrfbdy_d01` exists
- start and end dates match the forcing

After `wrf.exe`:

- `rsl.out.0000` ends with success
- CFL is not exploding
- output files are actually being written

## Use wrf-rust to validate output fast

```bash
pip install wrf-rust
python -m wrf info wrfout_d01_YYYY-MM-DD_HH:MM:SS
python -m wrf stats wrfout_d01_YYYY-MM-DD_HH:MM:SS sbcape slp temp
```

If the file opens and core fields look sane, the run is at least structurally alive.
