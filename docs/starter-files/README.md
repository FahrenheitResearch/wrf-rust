# WRF Community Starter Files

These files are the repo's practical starter pack for a single-domain, CPU-first, convection-allowing WRF workflow.

They are based on:

- a locally verified `3 km`, `200 x 200`, single-domain CPU baseline
- a community member's strong ECMWF/open-data workflow patterns
- the custom ECMWF open-data `Vtable` that matches that workflow better than the stock legacy ECMWF table

They are not meant to be copied blindly without edits.

Edit these first:

- dates in both namelists
- map center and projection values in `namelist.wps`
- `geog_data_path`
- `num_metgrid_levels` and `num_metgrid_soil_levels` in `namelist.input` if your `met_em*` files use different counts

## Files

- `namelist.wps.ecmwf-starter`
  - Single-domain `3 km` starter
  - Uses the community split-prefix `metgrid` pattern:
    - `FILE` for pressure-level fields
    - `SFILE` for surface fields
    - `SOILFILE` for soil fields
- `namelist.input.ecmwf-starter`
  - Single-domain `3 km` severe-weather starter
  - CPU-first and convection allowing
- `Vtable.ECMWF_opendata`
  - Custom `ungrib.exe` table for ECMWF open-data fields

## ECMWF split-prefix pattern

The starter `namelist.wps` is set up for a split ECMWF workflow:

1. Link the ECMWF pressure GRIB file(s), point `Vtable` at `Vtable.ECMWF_opendata`, set `prefix = 'FILE'`, run `ungrib.exe`
2. Link the ECMWF surface GRIB file(s), set `prefix = 'SFILE'`, run `ungrib.exe`
3. Link the ECMWF soil GRIB file(s), set `prefix = 'SOILFILE'`, run `ungrib.exe`
4. Restore `fg_name = 'SFILE', 'SOILFILE', 'FILE'`
5. Run `metgrid.exe`

If you are not using split ECMWF streams, simplify `fg_name` and `prefix` to the single family you actually have.

## Quick reminder on `num_metgrid_levels`

These values are source-dependent.

If `real.exe` complains, inspect a `met_em*` file and match the counts to it.

Example:

```bash
ncdump -h met_em.d01.YYYY-MM-DD_HH:00:00 | rg "NUM_METGRID|num_metgrid"
```
