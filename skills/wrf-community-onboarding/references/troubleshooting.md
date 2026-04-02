# Troubleshooting

Always branch by the earliest failing stage.

## 1. WRF compile failed

### Fastest checks

- Find the first real error, not the last hundred cascading ones.
- Confirm compiler and netCDF choices are consistent.
- Confirm WPS is not being debugged before WRF itself builds.

## 2. WPS compile failed

### Fastest checks

- WRF must already compile successfully.
- Use the same compiler family for WRF and WPS.
- Use the same netCDF install for both.
- For WPS `v4.4+`, prefer `./configure --build-grib2-libs`.

## 3. `real.exe` did not produce `wrfinput_d01` or `wrfbdy_d01`

### Most likely causes

- `met_em*` files are missing
- dates in `namelist.input` do not match the forcing
- wrong run directory
- broken I/O or netCDF environment

### Fastest checks

- confirm `met_em*` exists
- confirm the run directory contains the needed tables and executables
- confirm the forcing window covers the requested run

## 4. `wrf.exe` crashes immediately

### Known local example

There is a local `ierr=-1021` failure opening `wrfinput_d01`.

### Most likely causes

- `real.exe` did not finish cleanly
- files are missing or unreadable
- run directory is wrong
- netCDF linkage is inconsistent
- memory or disk is already exhausted

## 5. `wrf.exe` runs, but CFL explodes

### Rule

If CFL is blowing up from the beginning and thousands of points exceed the limit, do not trust the output just because the model technically completed.

### Fast actions

- reduce `time_step`
- reduce domain size
- simplify the setup
- check vertical resolution and physics choices
- rerun a shorter window first

## 6. Disk filled up

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
