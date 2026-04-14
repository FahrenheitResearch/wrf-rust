"""
wrf-rust: Fast WRF post-processing powered by Rust.

Fixes wrf-python's broken CAPE (proper SBCAPE/MLCAPE/MUCAPE parcel selection),
wrong SRH (Bunkers storm motion), adds 65+ variables with universal unit
support, and runs 5-30x faster.

Usage:
    from wrf import WrfFile, getvar

    f = WrfFile("wrfout_d01_2024-01-01_00:00:00")
    temp = getvar(f, "temp", timeidx=0, units="degC")
    cape = getvar(f, "sbcape", timeidx=0)
    srh  = getvar(f, "srh1", timeidx=0)

    # All timesteps at once
    slp = getvar(f, "slp", timeidx=ALL_TIMES, units="hPa")

    # Works with netCDF4.Dataset too (auto-wraps by reopening the filepath)
    from netCDF4 import Dataset
    nc = Dataset("wrfout_d01_2024-01-01_00:00:00")
    temp = getvar(nc, "temp", timeidx=0)
"""

import os
import sys
import warnings

import numpy as np

# On Windows, NetCDF/HDF5 DLLs may not be on PATH. Try common conda locations.
if sys.platform == "win32":
    _dll_dirs = [
        os.environ.get("NETCDF_DIR", ""),
        os.environ.get("HDF5_DIR", ""),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin"),
    ]
    # Also check wrfplot env specifically
    _home = os.path.expanduser("~")
    for _base in ("miniforge3", "miniconda3", "anaconda3"):
        _dll_dirs.append(os.path.join(_home, _base, "envs", "wrfplot", "Library", "bin"))
        _dll_dirs.append(os.path.join(_home, _base, "Library", "bin"))
    for _d in _dll_dirs:
        _d = os.path.join(_d, "bin") if _d and not _d.endswith("bin") and os.path.isdir(os.path.join(_d, "bin")) else _d
        if _d and os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

from wrf._wrf import WrfFile as _WrfFile
from wrf._wrf import list_variables as _list_variables

__all__ = [
    "WrfFile",
    "getvar",
    "list_variables",
    "ALL_TIMES",
    "available_variables",
    "interplevel",
    "get_cartopy",
    "latlon_coords",
    "ll_to_xy",
]
__version__ = "0.2.29"

# ── Optional plotting imports (require matplotlib) ──
try:
    from wrf.plot import plot_field, plot_wind, plot_skewt, panel

    __all__ += ["plot_field", "plot_wind", "plot_skewt", "panel"]
except ImportError:
    pass

# ── Optional explorer imports (require ipywidgets) ──
try:
    from wrf.explorer import Explorer, cross_section, profile, hovmoller

    __all__ += ["Explorer", "cross_section", "profile", "hovmoller"]
except ImportError:
    pass

# ── Optional Solar7 imports (require matplotlib) ──
# Lazy: importing wrf.solar7 registers the colormaps with matplotlib.
# We expose the public API names but defer actual import until accessed.
def __getattr__(name):
    if name in ("SOLAR7_STYLES", "solar7_products"):
        from wrf import solar7
        val = getattr(solar7, name)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = val
        __all__.append(name)
        return val
    raise AttributeError(f"module 'wrf' has no attribute {name!r}")

# Sentinel for "all time steps"
ALL_TIMES = None
_WARNED_DATASET_REOPEN = False


def _warn_dataset_reopen():
    """Warn once when a dataset-like object will be reopened by path."""
    global _WARNED_DATASET_REOPEN
    if _WARNED_DATASET_REOPEN or sys.platform != "win32":
        return

    warnings.warn(
        "wrf-rust reopens netCDF4/xarray dataset inputs by filepath instead "
        "of reading through the existing dataset handle. On Windows this can "
        "hang if the source dataset is still open while wrf-rust reads the "
        "same file, especially in subprocesses. Close the dataset first, or "
        "pass a file path / WrfFile instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_DATASET_REOPEN = True


class WrfFile:
    """A WRF output file handle.

    Can be constructed from a file path or from an existing
    ``netCDF4.Dataset`` (the Dataset's filepath is re-opened by the
    Rust backend for zero-copy performance).

    On Windows, avoid passing a live ``netCDF4.Dataset`` into wrf-rust
    while that Dataset remains open on the same file. wrf-rust will
    reopen the path natively, which can block in subprocesses.

    Attributes:
        nx, ny, nz, nt: Grid dimensions.
        dx, dy: Grid spacing in meters.
    """

    def __init__(self, path_or_dataset):
        if isinstance(path_or_dataset, _WrfFile):
            self._inner = path_or_dataset
        elif isinstance(path_or_dataset, (str, os.PathLike)):
            self._inner = _WrfFile(os.fspath(path_or_dataset))
        elif hasattr(path_or_dataset, "filepath"):
            # netCDF4.Dataset -- extract the file path and open natively
            _warn_dataset_reopen()
            fp = path_or_dataset.filepath()
            self._inner = _WrfFile(fp)
        elif hasattr(path_or_dataset, "encoding") and "source" in getattr(
            path_or_dataset, "encoding", {}
        ):
            # xarray.Dataset
            _warn_dataset_reopen()
            self._inner = _WrfFile(path_or_dataset.encoding["source"])
        else:
            # Last resort: try treating it as a path
            self._inner = _WrfFile(str(path_or_dataset))

    # ── Grid properties ──
    @property
    def nx(self):
        return self._inner.nx

    @property
    def ny(self):
        return self._inner.ny

    @property
    def nz(self):
        return self._inner.nz

    @property
    def nt(self):
        return self._inner.nt

    @property
    def dx(self):
        return self._inner.dx

    @property
    def dy(self):
        return self._inner.dy

    @property
    def path(self):
        return self._inner.path

    def times(self):
        """Return list of time strings (e.g. '2024-01-01_00:00:00')."""
        return self._inner.times()

    def getvar(self, name, timeidx=0, **kwargs):
        """Shorthand for ``getvar(self, name, timeidx, **kwargs)``."""
        return getvar(self, name, timeidx=timeidx, **kwargs)

    def __repr__(self):
        return (
            f"WrfFile('{self.path}', "
            f"nx={self.nx}, ny={self.ny}, nz={self.nz}, nt={self.nt})"
        )


def _ensure_wrffile(f):
    """Coerce various inputs into a WrfFile."""
    if isinstance(f, WrfFile):
        return f
    return WrfFile(f)


_GETVAR_NAME_ALIASES = {
    "cape_2d": "cape2d",
    "cape_3d": "cape3d",
    "mdbz": "maxdbz",
    "helicity": "uhel",
}


def _normalize_var_name(name):
    name_str = str(name)
    return _GETVAR_NAME_ALIASES.get(name_str.lower(), name_str)


def _normalize_single_timeidx(wf, timeidx):
    if timeidx is ALL_TIMES:
        return ALL_TIMES

    if isinstance(timeidx, np.integer):
        timeidx = int(timeidx)

    if not isinstance(timeidx, int):
        return timeidx

    if timeidx < 0:
        timeidx += wf.nt

    if timeidx < 0 or timeidx >= wf.nt:
        raise IndexError(f"timeidx {timeidx} out of range for file with {wf.nt} times")

    return timeidx


def _normalize_sequence_timeidx(wrffiles, timeidx):
    total_nt = sum(wf.nt for wf in wrffiles)

    if isinstance(timeidx, np.integer):
        timeidx = int(timeidx)

    if not isinstance(timeidx, int):
        return timeidx

    if timeidx < 0:
        timeidx += total_nt

    if timeidx < 0 or timeidx >= total_nt:
        raise IndexError(
            f"timeidx {timeidx} out of range for file sequence with {total_nt} times"
        )

    return timeidx


def _extract_wrffile_sequence(wrffile):
    if isinstance(wrffile, (list, tuple)):
        if not wrffile:
            raise ValueError("wrffile sequence is empty")
        return [_ensure_wrffile(item) for item in wrffile]
    return None


def _get_times_result(wrffiles, timeidx):
    if len(wrffiles) == 1:
        times = np.asarray(wrffiles[0].times())
        resolved = _normalize_single_timeidx(wrffiles[0], timeidx)
    else:
        times = np.asarray(
            [t for wf in wrffiles for t in wf.times()]
        )
        resolved = _normalize_sequence_timeidx(wrffiles, timeidx)

    if resolved is ALL_TIMES:
        return times
    return times[resolved]


def _getvar_sequence_cat(wrffiles, name, timeidx, squeeze, kwargs):
    if name == "times":
        return _get_times_result(wrffiles, timeidx)

    resolved = _normalize_sequence_timeidx(wrffiles, timeidx)
    if resolved is ALL_TIMES:
        arrays = [
            getvar(wf, name, timeidx=ALL_TIMES, squeeze=False, **kwargs)
            for wf in wrffiles
        ]
        result = np.concatenate(arrays, axis=0)
        if squeeze and result.shape[0] == 1:
            result = result[0]
        return result

    idx = resolved
    for wf in wrffiles:
        if idx < wf.nt:
            return getvar(wf, name, timeidx=idx, squeeze=squeeze, **kwargs)
        idx -= wf.nt

    raise IndexError(f"timeidx {resolved} out of range for file sequence")


def getvar(
    wrffile,
    name,
    timeidx=0,
    units=None,
    parcel_type=None,
    storm_motion=None,
    storm_motion_method=None,
    top_m=None,
    bottom_m=None,
    depth_m=None,
    parcel_pressure=None,
    parcel_temperature=None,
    parcel_dewpoint=None,
    bottom_p=None,
    top_p=None,
    layer_type=None,
    use_virtual=None,
    lake_interp=None,
    use_varint=None,
    use_liqskin=None,
    method=None,
    cache=None,
    meta=None,
    squeeze=True,
):
    """Compute a diagnostic variable from a WRF file.

    Parameters
    ----------
    wrffile : WrfFile, str, netCDF4.Dataset, or sequence of them
        The WRF output file, or a list/tuple of files for simple
        wrf-python-style concatenation.
    name : str
        Variable name (e.g. "temp", "slp", "sbcape", "srh1").
        Use ``list_variables()`` to see all supported names.
    timeidx : int or ALL_TIMES
        Time index.  Use ``ALL_TIMES`` (or ``None``) to retrieve all
        time steps stacked along a leading axis.
    units : str, optional
        Convert output to these units (e.g. "degC", "hPa", "knots").
    parcel_type : str, optional
        Parcel selection for CAPE variables: "sb", "ml", or "mu".
    storm_motion : tuple, optional
        Custom storm motion in m/s for SRH-family diagnostics. Pass either
        a scalar ``(u, v)`` pair, a pair of 2-D component grids
        ``(u_grid, v_grid)`` with shape ``(ny, nx)``, or a stacked array
        with shape ``(2, ny, nx)``.
    storm_motion_method : str, optional
        Default Bunkers storm-motion algorithm when ``storm_motion`` is not
        supplied. Use ``"pressure_weighted"`` (default), ``"weighted"``,
        ``"non_pressure_weighted"``, ``"unweighted"``, or ``"classic"``.
    top_m : float, optional
        Top of layer in metres AGL. Used by CAPE (truncated integration),
        shear, mean wind, lapse rates, updraft helicity.
    bottom_m : float, optional
        Bottom of layer in metres AGL. Used by shear, mean wind, lapse
        rates, updraft helicity.
    depth_m : float, optional
        Layer depth in metres AGL for SRH (e.g. 1000 for 0-1 km).
    parcel_pressure : float, optional
        Custom parcel starting pressure in hPa. Use with
        ``parcel_temperature`` and ``parcel_dewpoint`` for the generic
        ``cape``/``cin``/``lcl``/``lfc``/``el`` variables.
    parcel_temperature : float, optional
        Custom parcel starting temperature in deg C.
    parcel_dewpoint : float, optional
        Custom parcel starting dewpoint in deg C.
    bottom_p : float, optional
        Bottom of layer in hPa for pressure-based lapse rates (e.g. 700).
    top_p : float, optional
        Top of layer in hPa for pressure-based lapse rates (e.g. 500).
    layer_type : str, optional
        ``"fixed"`` (default) or ``"effective"`` for STP, SRH.
    use_virtual : bool, optional
        If True, use virtual temperature for lapse rate computation.
    use_varint : bool, optional
        If True, use variable intercept parameters (Thompson microphysics)
        for reflectivity. Default False matches wrf-python.
    use_liqskin : bool, optional
        If True, use bright-band liquid-skin correction for reflectivity.
        Default False matches wrf-python.
    method : str, optional
        Multi-file aggregation method. ``None`` and ``"cat"`` are supported.
        ``"join"`` is accepted for single files but not implemented for
        multi-file input.
    cache : any, optional
        Accepted for compatibility with wrf-python. Ignored.
    meta : bool, optional
        Accepted for compatibility with wrf-python. wrf-rust returns NumPy
        arrays, so this flag is ignored.
    squeeze : bool
        If True (default), remove length-1 leading dimensions.

    Returns
    -------
    numpy.ndarray
        2-D ``(ny, nx)`` or 3-D ``(nz, ny, nx)`` array, or with a
        leading time axis when ``timeidx=ALL_TIMES``.
    """
    if method not in (None, "cat", "join"):
        raise ValueError("method must be one of None, 'cat', or 'join'")

    del cache, meta

    name = _normalize_var_name(name)
    wrffiles = _extract_wrffile_sequence(wrffile)

    kwargs = dict(
        units=units,
        parcel_type=parcel_type,
        storm_motion=storm_motion,
        storm_motion_method=storm_motion_method,
        top_m=top_m,
        bottom_m=bottom_m,
        depth_m=depth_m,
        parcel_pressure=parcel_pressure,
        parcel_temperature=parcel_temperature,
        parcel_dewpoint=parcel_dewpoint,
        bottom_p=bottom_p,
        top_p=top_p,
        layer_type=layer_type,
        use_virtual=use_virtual,
        lake_interp=lake_interp,
        use_varint=use_varint,
        use_liqskin=use_liqskin,
    )

    if wrffiles is not None:
        if len(wrffiles) == 1:
            wf = wrffiles[0]
        else:
            if method == "join":
                raise NotImplementedError(
                    "method='join' is not implemented for multi-file input"
                )
            return _getvar_sequence_cat(wrffiles, name, timeidx, squeeze, kwargs)
    else:
        wf = _ensure_wrffile(wrffile)

    resolved_timeidx = _normalize_single_timeidx(wf, timeidx)

    if name == "times":
        return _get_times_result([wf], resolved_timeidx)

    if resolved_timeidx is ALL_TIMES:
        result = wf._inner.getvar_all_times(name, **kwargs)
        if squeeze and result.shape[0] == 1:
            result = result[0]
        return result
    else:
        return wf._inner.getvar(name, timeidx=resolved_timeidx, **kwargs)


def list_variables():
    """Return a list of all supported variable names with descriptions.

    Returns
    -------
    list of dict
        Each entry has keys ``name``, ``description``, ``units``.
    """
    return [
        {"name": name, "description": desc, "units": u}
        for name, desc, u in _list_variables()
    ]


def available_variables():
    """Print a formatted table of all supported variables."""
    vars_ = _list_variables()
    # Column widths
    nw = max(len(v[0]) for v in vars_) + 2
    dw = max(len(v[1]) for v in vars_) + 2
    print(f"{'Variable':<{nw}} {'Description':<{dw}} Units")
    print(f"{'-' * nw} {'-' * dw} -----")
    for name, desc, units in vars_:
        print(f"{name:<{nw}} {desc:<{dw}} {units}")


# =========================================================================
# interplevel -- vertical interpolation of 3D fields
# =========================================================================

def interplevel(field_3d, vert_coord_3d, target_level):
    """Interpolate a 3D field to a horizontal level.

    Drop-in replacement for ``wrf.interplevel()`` from wrf-python.

    Automatically detects whether the vertical coordinate is pressure
    (decreasing with height -> log-pressure interpolation) or height
    (increasing with height -> linear interpolation).

    Parameters
    ----------
    field_3d : ndarray, shape (nz, ny, nx)
        The 3D field to interpolate.
    vert_coord_3d : ndarray, shape (nz, ny, nx)
        The vertical coordinate field.  Typically full pressure in hPa
        (decreasing upward) or height AGL in metres (increasing upward).
    target_level : float
        The target level value in the same units as *vert_coord_3d*.

    Returns
    -------
    ndarray, shape (ny, nx)
        The interpolated 2D field.

    Examples
    --------
    >>> from wrf import WrfFile, getvar, interplevel
    >>> f = WrfFile("wrfout_d01_2024-05-01_00:00:00")
    >>> p = getvar(f, "pressure", timeidx=0, units="hPa")
    >>> tk = getvar(f, "temp", timeidx=0, units="K")
    >>> t_500 = interplevel(tk, p, 500.0)
    """
    field_3d = np.asarray(field_3d, dtype=np.float64)
    vert_coord_3d = np.asarray(vert_coord_3d, dtype=np.float64)

    if field_3d.ndim != 3 or vert_coord_3d.ndim != 3:
        raise ValueError(
            "field_3d and vert_coord_3d must be 3-D arrays (nz, ny, nx)"
        )
    if field_3d.shape != vert_coord_3d.shape:
        raise ValueError(
            f"Shape mismatch: field_3d {field_3d.shape} vs "
            f"vert_coord_3d {vert_coord_3d.shape}"
        )

    nz, ny, nx = field_3d.shape

    # Support both scalar and 2D target levels
    target_arr = np.asarray(target_level, dtype=np.float64)
    if target_arr.ndim == 0:
        # Scalar: broadcast to 2D
        target_2d = np.full((ny, nx), float(target_arr))
    elif target_arr.ndim == 2:
        if target_arr.shape != (ny, nx):
            raise ValueError(
                f"2D target_level shape {target_arr.shape} doesn't match "
                f"field shape ({ny}, {nx})"
            )
        target_2d = target_arr
    else:
        raise ValueError(
            "target_level must be a scalar or 2D array (ny, nx)"
        )

    # Determine direction: if the coordinate generally decreases along the
    # first axis it is pressure-like (use log interpolation); otherwise it
    # is height-like (use linear interpolation).
    mid_j, mid_i = ny // 2, nx // 2
    is_pressure = vert_coord_3d[0, mid_j, mid_i] > vert_coord_3d[-1, mid_j, mid_i]

    result = np.full((ny, nx), np.nan, dtype=np.float64)

    if is_pressure:
        # Log-pressure interpolation
        log_vert = np.log(np.clip(vert_coord_3d, 1e-10, None))
        log_target = np.log(target_2d)

        for k in range(nz - 1):
            # Find grid cells where target is bracketed by levels k and k+1
            # (pressure decreases upward, so vert[k] >= target >= vert[k+1])
            above = vert_coord_3d[k, :, :] >= target_level
            below = vert_coord_3d[k + 1, :, :] <= target_level
            mask = above & below & np.isnan(result)

            if not np.any(mask):
                continue

            denom = log_vert[k + 1, :, :] - log_vert[k, :, :]
            # Avoid division by zero
            safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
            frac = (log_target - log_vert[k, :, :]) / safe_denom
            interped = field_3d[k, :, :] + frac * (
                field_3d[k + 1, :, :] - field_3d[k, :, :]
            )
            result = np.where(mask, interped, result)

        # Points still NaN are underground or above model top -- leave as NaN
    else:
        # Linear height interpolation
        for k in range(nz - 1):
            above = vert_coord_3d[k, :, :] <= target_2d
            below = vert_coord_3d[k + 1, :, :] >= target_2d
            mask = above & below & np.isnan(result)

            if not np.any(mask):
                continue

            denom = vert_coord_3d[k + 1, :, :] - vert_coord_3d[k, :, :]
            safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
            frac = (target_2d - vert_coord_3d[k, :, :]) / safe_denom
            interped = field_3d[k, :, :] + frac * (
                field_3d[k + 1, :, :] - field_3d[k, :, :]
            )
            result = np.where(mask, interped, result)

        # Points still NaN are underground or above model top -- leave as NaN

    return result


# =========================================================================
# get_cartopy -- CRS projection from WRF file
# =========================================================================

def get_cartopy(wrffile):
    """Get a cartopy CRS projection from a WRF file.

    Drop-in replacement for ``wrf.get_cartopy()`` from wrf-python.

    First tries to read ``MAP_PROJ``, ``TRUELAT1``, ``TRUELAT2``,
    ``STAND_LON``, ``CEN_LAT``, and ``CEN_LON`` global attributes via
    netCDF4 (if available).  Falls back to inferring the projection from
    the lat/lon arrays in the Rust WrfFile handle when netCDF4 is not
    installed.

    Parameters
    ----------
    wrffile : WrfFile, str, or netCDF4.Dataset
        The WRF output file.

    Returns
    -------
    cartopy.crs.Projection
        A cartopy CRS object suitable for ``ax = plt.axes(projection=crs)``.

    Raises
    ------
    ImportError
        If cartopy is not installed.
    ValueError
        If the map projection is not supported.
    """
    import cartopy.crs as ccrs

    wf = _ensure_wrffile(wrffile)

    # --- Try reading global attributes via netCDF4 (optional) ---
    try:
        from netCDF4 import Dataset as _NCDataset

        nc = _NCDataset(wf.path, "r")
        try:
            map_proj = int(nc.getncattr("MAP_PROJ"))
            truelat1 = float(nc.getncattr("TRUELAT1"))
            truelat2 = float(nc.getncattr("TRUELAT2"))
            stand_lon = float(nc.getncattr("STAND_LON"))
            cen_lat = float(nc.getncattr("CEN_LAT"))
            cen_lon = float(nc.getncattr("CEN_LON"))
        finally:
            nc.close()

        if map_proj == 1:
            return ccrs.LambertConformal(
                central_longitude=stand_lon,
                central_latitude=cen_lat,
                standard_parallels=(truelat1, truelat2),
            )
        elif map_proj == 2:
            return ccrs.Stereographic(
                central_latitude=cen_lat,
                central_longitude=stand_lon,
                true_scale_latitude=truelat1,
            )
        elif map_proj == 3:
            return ccrs.Mercator(
                central_longitude=cen_lon,
                latitude_true_scale=truelat1,
            )
        elif map_proj == 6:
            return ccrs.PlateCarree(central_longitude=cen_lon)
        else:
            raise ValueError(
                f"Unsupported WRF map projection MAP_PROJ={map_proj}. "
                f"Supported: 1 (Lambert), 2 (Polar Stereographic), "
                f"3 (Mercator), 6 (Lat-Lon)."
            )
    except ImportError:
        pass  # netCDF4 not available -- fall through to inference

    # --- Fallback: infer projection from lat/lon arrays ---
    lat, lon = latlon_coords(wf, timeidx=0)
    cen_lat = float(lat[lat.shape[0] // 2, lat.shape[1] // 2])
    cen_lon = float(lon[lon.shape[0] // 2, lon.shape[1] // 2])

    # Check if lat/lon form a regular grid (PlateCarree)
    lat_range = float(lat.max() - lat.min())
    lon_range = float(lon.max() - lon.min())

    # Heuristic: if the lat spacing along columns is very uniform, it is
    # likely a lat-lon grid. Otherwise assume Lambert Conformal which is
    # the most common WRF projection.
    lat_col = lat[:, lat.shape[1] // 2]
    if lat_col.shape[0] > 1:
        dlat = np.diff(lat_col)
        lat_uniform = float(np.std(dlat)) < 0.001 * float(np.mean(np.abs(dlat)) + 1e-10)
    else:
        lat_uniform = True

    lon_row = lon[lon.shape[0] // 2, :]
    if lon_row.shape[0] > 1:
        dlon = np.diff(lon_row)
        lon_uniform = float(np.std(dlon)) < 0.001 * float(np.mean(np.abs(dlon)) + 1e-10)
    else:
        lon_uniform = True

    if lat_uniform and lon_uniform:
        # Regular lat-lon grid
        return ccrs.PlateCarree(central_longitude=cen_lon)
    else:
        # Default to Lambert Conformal -- the most common WRF projection.
        # Use the domain center and reasonable standard parallels.
        return ccrs.LambertConformal(
            central_longitude=cen_lon,
            central_latitude=cen_lat,
            standard_parallels=(cen_lat - 5.0, cen_lat + 5.0),
        )


# =========================================================================
# latlon_coords -- latitude / longitude 2D arrays
# =========================================================================

def latlon_coords(wrffile, timeidx=0):
    """Return the 2D latitude and longitude arrays from a WRF file.

    Drop-in replacement for ``wrf.latlon_coords()`` from wrf-python.

    Parameters
    ----------
    wrffile : WrfFile, str, or netCDF4.Dataset
        The WRF output file.
    timeidx : int, optional
        Time index (default 0). XLAT/XLONG are time-invariant in most
        WRF configurations, but some moving-nest runs vary them.

    Returns
    -------
    (lat, lon) : tuple of ndarray, each shape (ny, nx)
        XLAT and XLONG arrays in degrees.
    """
    wf = _ensure_wrffile(wrffile)
    lat = wf._inner.getvar("lat", timeidx=timeidx)
    lon = wf._inner.getvar("lon", timeidx=timeidx)
    return lat, lon


# =========================================================================
# ll_to_xy -- lat/lon to grid indices (fractional)
# =========================================================================

def ll_to_xy(wrffile, latitude, longitude, timeidx=0):
    """Convert lat/lon to fractional grid (x, y) indices.

    Drop-in replacement for ``wrf.ll_to_xy()`` from wrf-python.

    Finds the position on the WRF grid corresponding to the given
    latitude/longitude by inverse-distance interpolation, returning
    fractional indices (floats) so callers can do sub-grid interpolation.

    Parameters
    ----------
    wrffile : WrfFile, str, or netCDF4.Dataset
        The WRF output file.
    latitude : float
        Target latitude in degrees.
    longitude : float
        Target longitude in degrees.
    timeidx : int, optional
        Time index (default 0).

    Returns
    -------
    (x, y) : tuple of float
        Fractional grid indices (x = west-east, y = south-north).
        Integer parts give the grid cell; fractional parts give position
        within the cell.
    """
    lat2d, lon2d = latlon_coords(wrffile, timeidx=timeidx)

    # Find the nearest grid point
    dist = (lat2d - latitude) ** 2 + (lon2d - longitude) ** 2
    jn, in_ = np.unravel_index(np.argmin(dist), dist.shape)

    # Refine to fractional indices using bilinear interpolation
    # Search in the 2x2 cell around the nearest point
    ny, nx = lat2d.shape
    best_x = float(in_)
    best_y = float(jn)

    for j0 in range(max(0, jn - 1), min(ny - 1, jn + 1)):
        for i0 in range(max(0, in_ - 1), min(nx - 1, in_ + 1)):
            # Corners of this cell
            lat00 = lat2d[j0, i0]
            lat10 = lat2d[j0, i0 + 1]
            lat01 = lat2d[j0 + 1, i0]
            lat11 = lat2d[j0 + 1, i0 + 1]
            lon00 = lon2d[j0, i0]
            lon10 = lon2d[j0, i0 + 1]
            lon01 = lon2d[j0 + 1, i0]
            lon11 = lon2d[j0 + 1, i0 + 1]

            # Solve for (s, t) in [0,1]x[0,1] using iterative approach
            # Bilinear: lat = (1-s)(1-t)*lat00 + s(1-t)*lat10 + (1-s)t*lat01 + st*lat11
            # Start from center
            s, t = 0.5, 0.5
            for _ in range(10):
                lat_est = (1-s)*(1-t)*lat00 + s*(1-t)*lat10 + (1-s)*t*lat01 + s*t*lat11
                lon_est = (1-s)*(1-t)*lon00 + s*(1-t)*lon10 + (1-s)*t*lon01 + s*t*lon11

                dlat = latitude - lat_est
                dlon = longitude - lon_est

                # Jacobian approximation
                dlat_ds = -(1-t)*lat00 + (1-t)*lat10 - t*lat01 + t*lat11
                dlat_dt = -(1-s)*lat00 - s*lat10 + (1-s)*lat01 + s*lat11
                dlon_ds = -(1-t)*lon00 + (1-t)*lon10 - t*lon01 + t*lon11
                dlon_dt = -(1-s)*lon00 - s*lon10 + (1-s)*lon01 + s*lon11

                det = dlat_ds * dlon_dt - dlat_dt * dlon_ds
                if abs(det) < 1e-20:
                    break

                ds = (dlat * dlon_dt - dlon * dlat_dt) / det
                dt = (dlon * dlat_ds - dlat * dlon_ds) / det
                s += ds
                t += dt

            if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
                best_x = float(i0) + s
                best_y = float(j0) + t
                return best_x, best_y

    return best_x, best_y
