# Init Data

Use this file when the user needs forcing guidance.

## Default recommendation order

1. ECMWF open data for hobbyist real-time Euro/IFS forcing
2. GFS for the universal fallback
3. HRRR or RAP for regional fast-refresh use cases

## ECMWF

### Important clarification

As of `2026-04-01`, the simplest current ECMWF real-time path is the Free & Open Data Portal:

- `https://data.ecmwf.int/forecasts/`

For the common hobbyist real-time open-data path, users usually do **not** need an API key.

### What users get wrong

They often mix together:

- ECMWF open real-time data
- CDS archive access
- ECMWF Web API / MARS access

Those are not the same thing.

### Practical guidance

- For current real-time forecasts, point users to the open portal first.
- If they want automation, tell them to look at `ecmwf-opendata`.
- If they ask where the API key goes, clarify that API keys are mainly for CDS/archive-style access, not the simple open-data portal workflow.
- If they find community automation snippets, warn them that many are fragments from a larger project, not standalone scripts.
- If they need a working `ungrib.exe` field map for ECMWF open data, point them at `docs/starter-files/Vtable.ECMWF_opendata` in this repo.

### Practical community pattern

One useful ECMWF + WPS pattern from community workflows:

- download pressure-level, surface, and soil data separately
- run `ungrib.exe` separately for each prefix family
- use prefixes like `FILE`, `SFILE`, and `SOILFILE`
- point `metgrid.exe` at them with `fg_name = 'SFILE', 'SOILFILE', 'FILE'`

This is a useful pattern to mention when a user is confused about how multiple ECMWF fields get staged into WPS.

### Compression gotcha

Some ECMWF GRIB2 files may arrive with CCSDS compression that can break `ungrib.exe`.

Practical workaround:

- use `cdo -f grb2 copy input.grib2 output.grib2` to rewrite/decompress the file first
- if a user sees `ungrib.exe` failing on otherwise valid ECMWF files, this is one of the first things to check

### Official 2024+ ECMWF conversion rule

WRF support documents an additional rule for newer ECMWF GRIB2:

```bash
grib_set -r -w packingType=grid_ccsds -s packingType=grid_simple input.grib2 output.grib2
```

Then use `Vtable.ECMWF` from `WPS v4.6+`.

This is the official path when recent ECMWF files do not work directly with older WPS assumptions.

### Good wording

"As of 2026-04-01, the easy hobbyist path for ECMWF real-time forcing is the open portal at `data.ecmwf.int/forecasts/`, and that path usually does not require an API key."

## GFS

- Global
- easy public fallback
- available through NOMADS
- good when ECMWF access is inconvenient or the user wants a simpler public path

Common files:

- `gfs.tCCz.pgrb2.0p25.fFFF`

## HRRR

- Best when the user is focused on the CONUS and wants fast-refresh guidance.
- Regional, not universal.

Common files:

- `hrrr.tCCz.wrfprsfFF.grib2`
- `hrrr.tCCz.wrfnatfFF.grib2`
- `hrrr.tCCz.wrfsfcfFF.grib2`

## RAP

- Useful for broader North American fast-refresh forcing when HRRR is not the right fit.

Common file family:

- `rap.tCCz.awp130pgrbfFF.grib2`

## Decision rule

- User says "Euro" or wants best global synoptic forcing -> ECMWF open data
- User wants easiest universal public fallback -> GFS
- User wants CONUS fast-refresh storm work -> HRRR
- User wants broader regional fast-refresh -> RAP

## Caveats

- HRRR and RAP are regional and time-sensitive.
- GFS is broad and simple but not the highest-end Euro choice.
- ECMWF access wording should always include the date because access policy has changed over time.
