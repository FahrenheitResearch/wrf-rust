# Python API Compatibility

`wrf-rust` is published as the `wrf-rust` package and imported as `wrf`.
The compatibility anchor is the Python API used by WRF users:

- `WrfFile`
- `WrfFile.reader_capabilities()`
- `getvar`
- `getvar_all_times`
- `ALL_TIMES`
- `list_variables`
- `available_variables`
- `interplevel`
- `latlon_coords`
- `ll_to_xy`
- plotting and sounding helpers exposed from `wrf.plot` and `wrf.explorer`

The native Rust crates are useful directly, but their rendering, product,
store, sounding, and ensemble APIs can still move faster than the Python API.

## Supported

- File paths and `WrfFile` handles.
- `getvar(file, name, timeidx=0)` for registered diagnostics and raw WRF
  variables present in the file.
- `timeidx=ALL_TIMES` and `getvar_all_times(...)`, stacked along a leading
  time axis.
- Negative integer `timeidx` values.
- `units=` conversion for supported unit families. Unit conversion errors are
  fatal; data are not silently relabeled.
- `interplevel(field_3d, vert_coord_3d, target_level)` for scalar or 2-D
  target levels.
- `latlon_coords` and `ll_to_xy` for single time indices.
- `method=None` and `method="cat"` for simple file sequences.

## Accepted But Limited

- `netCDF4.Dataset` and xarray-like inputs are accepted only when a source
  filepath is available. `wrf-rust` reopens that path natively; it does not read
  through the live Python dataset object.
- File sequences are concatenated in time for `method=None` and `method="cat"`.
  They are not a full `wrf-python` multi-file dataset abstraction.
- `method="join"` is accepted for a single file because it has no effect there.

## Ignored For Compatibility

- `meta=` is accepted and ignored. `wrf-rust` returns NumPy arrays rather than
  xarray metadata objects.
- `cache=` is accepted and ignored. Native intermediate caches are internal and
  scoped to the active diagnostic call.

## Experimental

- ECAPE-family diagnostics, lake interpolation options, and advanced severe
  composite parameters are available but are not exact `wrf-python` features.
- Python plotting helpers (`plot_field`, `plot_wind`, `plot_skewt`, `Explorer`,
  `cross_section`, `profile`, and `hovmoller`) are convenience APIs layered on
  top of the stable array-returning `getvar` surface.
- Rust-native product rendering, soundings, store discovery, and ensemble
  rendering are still evolving.

## Not Implemented

- `method="join"` for multi-file input.
- Returning xarray objects from `meta=True`.
- Reading directly from in-memory xarray/netCDF4 objects without reopening the
  source file path.
- Implicit dataset reopening policies beyond the explicit filepath behavior
  above.
