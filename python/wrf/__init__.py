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

    # Works with netCDF4.Dataset too (auto-wraps it)
    from netCDF4 import Dataset
    nc = Dataset("wrfout_d01_2024-01-01_00:00:00")
    temp = getvar(nc, "temp", timeidx=0)
"""

import os
import sys

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
]
__version__ = "0.2.3"

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


class WrfFile:
    """A WRF output file handle.

    Can be constructed from a file path or from an existing
    ``netCDF4.Dataset`` (the Dataset's filepath is re-opened by the
    Rust backend for zero-copy performance).

    Attributes:
        nx, ny, nz, nt: Grid dimensions.
        dx, dy: Grid spacing in meters.
    """

    def __init__(self, path_or_dataset):
        if isinstance(path_or_dataset, _WrfFile):
            self._inner = path_or_dataset
        elif isinstance(path_or_dataset, str):
            self._inner = _WrfFile(path_or_dataset)
        elif hasattr(path_or_dataset, "filepath"):
            # netCDF4.Dataset -- extract the file path and open natively
            fp = path_or_dataset.filepath()
            self._inner = _WrfFile(fp)
        elif hasattr(path_or_dataset, "encoding") and "source" in getattr(
            path_or_dataset, "encoding", {}
        ):
            # xarray.Dataset
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


def getvar(
    wrffile,
    name,
    timeidx=0,
    units=None,
    parcel_type=None,
    storm_motion=None,
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
    squeeze=True,
):
    """Compute a diagnostic variable from a WRF file.

    Parameters
    ----------
    wrffile : WrfFile, str, or netCDF4.Dataset
        The WRF output file.
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
    storm_motion : tuple of (float, float), optional
        Custom storm motion (u, v) in m/s for SRH variables.
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
    squeeze : bool
        If True (default), remove length-1 leading dimensions.

    Returns
    -------
    numpy.ndarray
        2-D ``(ny, nx)`` or 3-D ``(nz, ny, nx)`` array, or with a
        leading time axis when ``timeidx=ALL_TIMES``.
    """
    wf = _ensure_wrffile(wrffile)

    kwargs = dict(
        units=units,
        parcel_type=parcel_type,
        storm_motion=storm_motion,
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
    )

    if timeidx is ALL_TIMES:
        arrays = []
        for t in range(wf.nt):
            arr = wf._inner.getvar(name, timeidx=t, **kwargs)
            arrays.append(arr)
        result = np.stack(arrays, axis=0)
        if squeeze and result.shape[0] == 1:
            result = result[0]
        return result
    else:
        return wf._inner.getvar(name, timeidx=timeidx, **kwargs)


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
    target_level = float(target_level)

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

    # Determine direction: if the coordinate generally decreases along the
    # first axis it is pressure-like (use log interpolation); otherwise it
    # is height-like (use linear interpolation).
    mid_j, mid_i = ny // 2, nx // 2
    is_pressure = vert_coord_3d[0, mid_j, mid_i] > vert_coord_3d[-1, mid_j, mid_i]

    result = np.full((ny, nx), np.nan, dtype=np.float64)

    if is_pressure:
        # Log-pressure interpolation
        log_vert = np.log(np.clip(vert_coord_3d, 1e-10, None))
        log_target = np.log(target_level)

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

        # Extrapolate for points still NaN: above the highest or below
        # the lowest pressure level.
        still_nan = np.isnan(result)
        if np.any(still_nan):
            # If target pressure > surface (below ground), use surface value
            below_sfc = still_nan & (vert_coord_3d[0, :, :] < target_level)
            result = np.where(below_sfc, field_3d[0, :, :], result)
            # If target pressure < model top, use top value
            above_top = still_nan & (vert_coord_3d[-1, :, :] > target_level)
            result = np.where(above_top, field_3d[-1, :, :], result)
    else:
        # Linear height interpolation
        for k in range(nz - 1):
            above = vert_coord_3d[k, :, :] <= target_level
            below = vert_coord_3d[k + 1, :, :] >= target_level
            mask = above & below & np.isnan(result)

            if not np.any(mask):
                continue

            denom = vert_coord_3d[k + 1, :, :] - vert_coord_3d[k, :, :]
            safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
            frac = (target_level - vert_coord_3d[k, :, :]) / safe_denom
            interped = field_3d[k, :, :] + frac * (
                field_3d[k + 1, :, :] - field_3d[k, :, :]
            )
            result = np.where(mask, interped, result)

        # Extrapolate boundary values
        still_nan = np.isnan(result)
        if np.any(still_nan):
            below_sfc = still_nan & (vert_coord_3d[0, :, :] > target_level)
            result = np.where(below_sfc, field_3d[0, :, :], result)
            above_top = still_nan & (vert_coord_3d[-1, :, :] < target_level)
            result = np.where(above_top, field_3d[-1, :, :], result)

    return result


# =========================================================================
# get_cartopy -- CRS projection from WRF file
# =========================================================================

def get_cartopy(wrffile):
    """Get a cartopy CRS projection from a WRF file.

    Drop-in replacement for ``wrf.get_cartopy()`` from wrf-python.

    Reads the ``MAP_PROJ``, ``TRUELAT1``, ``TRUELAT2``, ``STAND_LON``,
    ``CEN_LAT``, and ``CEN_LON`` global attributes from the WRF file
    and returns the corresponding ``cartopy.crs`` projection.

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
    from netCDF4 import Dataset

    # Get the file path so we can read global attributes via netCDF4
    wf = _ensure_wrffile(wrffile)
    nc = Dataset(wf.path, "r")
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
        # Lambert Conformal Conic
        return ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=cen_lat,
            standard_parallels=(truelat1, truelat2),
        )
    elif map_proj == 2:
        # Polar Stereographic
        return ccrs.Stereographic(
            central_latitude=cen_lat,
            central_longitude=stand_lon,
            true_scale_latitude=truelat1,
        )
    elif map_proj == 3:
        # Mercator
        return ccrs.Mercator(
            central_longitude=cen_lon,
            latitude_true_scale=truelat1,
        )
    elif map_proj == 6:
        # Lat-Lon (cylindrical equidistant)
        return ccrs.PlateCarree(central_longitude=cen_lon)
    else:
        raise ValueError(
            f"Unsupported WRF map projection MAP_PROJ={map_proj}. "
            f"Supported: 1 (Lambert), 2 (Polar Stereographic), "
            f"3 (Mercator), 6 (Lat-Lon)."
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
