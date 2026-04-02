# Troubleshooting

Always branch by the earliest failing stage.

## 1. WRF compile failed

### Fastest checks

- Find the first real error, not the last hundred cascading ones.
- Confirm compiler and netCDF choices are consistent.
- Confirm WPS is not being debugged before WRF itself builds.
- If `netcdf.inc` is missing, make sure netCDF-Fortran is actually installed in the same prefix WRF is using.
- If you see unresolved `GOMP_*` or `__kmpc_*` symbols later in WPS, stop mixing an OpenMP-heavy WRF build into a casual WPS path without matching runtime linkage.

## 2. WPS compile failed

### Fastest checks

- WRF must already compile successfully.
- Use the same compiler family for WRF and WPS.
- Use the same netCDF install for both.
- Confirm `WRF_DIR` points at the compiled WRF tree.
- For WPS `v4.4+`, prefer `./configure --build-grib2-libs`.
- If WPS throws unresolved `GOMP_*` or `__kmpc_*` symbols, the simplest hobbyist fix is often a simpler serial WPS build path.

## 3. `geogrid.exe` fails

### Common signatures

- `Could not open .../WPS_GEOG/.../index`
- domain-dimension errors involving `parent_grid_ratio`

### Fastest checks

- confirm `geog_data_path` is correct and absolute
- confirm the WPS geog data actually exists there
- confirm nest dimensions and parent ratios are valid

## 4. `ungrib.exe` fails

### Common signatures

- `edition_num: unable to open GRIBFILE.AAA`
- `Problem opening file Vtable`
- `Data not found: YYYY-MM-DD_HH:MM:SS.0000`

### Fastest checks

- make sure GRIB files were linked with `link_grib.csh`
- make sure the correct file is symlinked to `Vtable`
- confirm the data times actually cover the requested `start_date` and `end_date`
- if the data are recent ECMWF GRIB2, try the official `grib_set ... packingType=grid_simple` conversion step

## 5. `metgrid.exe` fails

### Common signatures

- `The mandatory field TT was not found in any input data`
- `ERROR: Screwy NDATE: 0000-00-00_00:00:00`

### Fastest checks

- confirm the `fg_name` prefixes match what `ungrib.exe` actually wrote
- confirm `ungrib.exe` really produced the expected intermediate files
- confirm all domains have valid start/end dates in `namelist.wps`
- use `rd_intermediate.exe` if needed to inspect the intermediate files instead of guessing

## 6. `real.exe` did not produce `wrfinput_d01` or `wrfbdy_d01`

### Most likely causes

- `met_em*` files are missing
- dates in `namelist.input` do not match the forcing
- wrong run directory
- broken I/O or netCDF environment

### Fastest checks

- confirm `met_em*` exists
- confirm the run directory contains the needed tables and executables
- confirm the forcing window covers the requested run

## 7. `wrf.exe` crashes immediately

### Known local example

There is a local `ierr=-1021` failure opening `wrfinput_d01`.

### Most likely causes

- `real.exe` did not finish cleanly
- files are missing or unreadable
- run directory is wrong
- netCDF linkage is inconsistent
- memory or disk is already exhausted

## 8. `wrf.exe` runs, but CFL explodes

### Rule

If CFL is blowing up from the beginning and thousands of points exceed the limit, do not trust the output just because the model technically completed.

### Fast actions

- reduce `time_step`
- reduce domain size
- simplify the setup
- check vertical resolution and physics choices
- rerun a shorter window first

## 9. Disk filled up

### Common causes

- output interval too short
- too many diagnostics
- domain too large
- long run duration

### Fast fixes

- increase `history_interval`
- reduce run length
- reduce output fields
- confirm free space before launch

Local evidence includes a single idealized output file growing to about `56 GB`, so disk warnings should be aggressive.
