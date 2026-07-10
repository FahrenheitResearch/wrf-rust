"""
wrf-rust: Fast WRF post-processing powered by Rust.

Adds parcel-explicit SBCAPE/MLCAPE/MUCAPE, modern Bunkers SRH for operational
workflows, and separately named ``cape2d_wrfpython``, ``cape3d_wrfpython``,
and ``srh_wrfpython`` compatibility paths for NCAR's legacy RIP algorithms,
plus broad diagnostic and unit support.

Usage:
    from wrf import WrfFile, getvar

    f = WrfFile("wrfout_d01_2024-01-01_00:00:00")
    temp = getvar(f, "temp", timeidx=0, units="degC")
    cape = getvar(f, "sbcape", timeidx=0)
    srh  = getvar(f, "srh1", timeidx=0)
    srh_ncar = getvar(f, "srh_wrfpython", timeidx=0, depth_m=3000)
    ecape = getvar(f, "ecape", timeidx=0, storm_motion_type="bunkers_rm")

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

from ._coord_transform import (
    ProjectionParams as _ProjectionParams,
    WRF_EARTH_RADIUS as _COORD_EARTH_RADIUS,
    ll_to_xy as _project_ll_to_xy,
    longitude_delta as _longitude_delta,
    xy_to_ll as _project_xy_to_ll,
)

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
    "xy_to_ll",
    "CoordPair",
]
__version__ = "0.4.0"

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


class CoordPair:
    """Coordinate-pair metadata compatible with wrf-python results."""

    __slots__ = ("x", "y", "lat", "lon")

    def __init__(self, x=None, y=None, lat=None, lon=None):
        self.x = x
        self.y = y
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        values = []
        if self.x is not None:
            values.extend((f"x={self.x}", f"y={self.y}"))
        if self.lat is not None:
            values.extend((f"lat={self.lat}", f"lon={self.lon}"))
        return f"CoordPair({', '.join(values)})"

    __str__ = __repr__


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
    storm_motion_type=None,
    entrainment_rate=None,
    pseudoadiabatic=None,
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
    ecape_strict=None,
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
        Parcel selection for CAPE and ECAPE-family variables: "sb", "ml",
        or "mu". The separately named ``*_wrfpython`` CAPE diagnostics use
        fixed NCAR/RIP semantics and reject parcel overrides.
    storm_motion : tuple, optional
        Custom storm motion in m/s for SRH-family diagnostics. Pass either
        a scalar ``(u, v)`` pair, a pair of 2-D component grids
        ``(u_grid, v_grid)`` with shape ``(ny, nx)``, or a stacked array
        with shape ``(2, ny, nx)``.
    storm_motion_method : str, optional
        Default Bunkers storm-motion algorithm when ``storm_motion`` is not
        supplied. Use ``"pressure_weighted"`` (default), ``"weighted"``,
        ``"non_pressure_weighted"``, ``"unweighted"``, or ``"classic"``.
    storm_motion_type : str, optional
        ECAPE storm-motion type. Common values are ``"bunkers_rm"``,
        ``"bunkers_lm"``, and ``"mean_wind"``.
    entrainment_rate : float, optional
        ECAPE entrainment rate forwarded to the core implementation.
        Ignored by non-ECAPE diagnostics.
    pseudoadiabatic : bool, optional
        ECAPE pseudoadiabatic toggle forwarded to the core implementation.
        Ignored by non-ECAPE diagnostics.
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
    ecape_strict : bool, optional
        If True, ECAPE-family variables raise an error with failed-column
        counts instead of silently zero-filling ECAPE columns that cannot be
        computed. Default behavior remains silent zero-fill.

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
        storm_motion_type=storm_motion_type,
        entrainment_rate=entrainment_rate,
        pseudoadiabatic=pseudoadiabatic,
        ecape_strict=ecape_strict,
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


from wrf.soundings import render_sounding

__all__ += ["render_sounding"]


# =========================================================================
# interplevel -- vertical interpolation of 3D fields
# =========================================================================

from wrf.interpolation import interplevel


# =========================================================================
# get_cartopy -- CRS projection from WRF file
# =========================================================================

_WRF_EARTH_RADIUS = 6_370_000.0
_WRF_PROJECTION_ATTRS = (
    "MAP_PROJ",
    "TRUELAT1",
    "TRUELAT2",
    "STAND_LON",
    "MOAD_CEN_LAT",
    "CEN_LAT",
    "CEN_LON",
    "POLE_LAT",
    "POLE_LON",
)


def _wrf_cartopy_globe(ccrs):
    """Return the spherical globe used by WRF's map projections."""
    return ccrs.Globe(
        ellipse=None,
        semimajor_axis=_WRF_EARTH_RADIUS,
        semiminor_axis=_WRF_EARTH_RADIUS,
        nadgrids="@null",
    )


def _read_wrf_projection_attrs(nc):
    """Read the available WRF projection globals from a NetCDF handle."""
    attrs = {}
    for name in _WRF_PROJECTION_ATTRS:
        try:
            value = nc.getncattr(name)
        except (AttributeError, KeyError):
            continue

        if np.ma.is_masked(value):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"WRF projection attribute {name} is not numeric: {value!r}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"WRF projection attribute {name} is not finite: {value!r}"
            )
        attrs[name] = value
    return attrs


def _cartopy_from_wrf_attrs(ccrs, attrs):
    """Construct Cartopy CRS parameters using NCAR wrf-python semantics.

    The compatibility contract is NCAR/wrf-python ``projection.py`` at
    commit 124a8336529af6397fe150e14bd436d923122cdd.
    """
    attrs = {str(key).upper(): value for key, value in attrs.items()}

    try:
        map_proj_value = float(attrs["MAP_PROJ"])
    except KeyError as exc:
        raise ValueError("WRF file is missing required MAP_PROJ attribute") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid WRF MAP_PROJ value: {attrs['MAP_PROJ']!r}"
        ) from exc
    if not np.isfinite(map_proj_value) or not map_proj_value.is_integer():
        raise ValueError(f"Invalid WRF MAP_PROJ value: {map_proj_value!r}")
    map_proj = int(map_proj_value)

    def optional(name, fallback=None):
        value = attrs.get(name, fallback)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"WRF projection attribute {name} is not numeric: {value!r}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"WRF projection attribute {name} is not finite: {value!r}"
            )
        return value

    def required(name, fallback=None):
        value = optional(name, fallback)
        if value is None:
            raise ValueError(
                f"WRF MAP_PROJ={map_proj} requires projection attribute {name}"
            )
        return value

    cen_lat = optional("CEN_LAT")
    cen_lon = optional("CEN_LON")
    moad_cen_lat = optional("MOAD_CEN_LAT", cen_lat)
    stand_lon = optional("STAND_LON", cen_lon)
    truelat1 = optional("TRUELAT1")
    truelat2 = optional("TRUELAT2")
    pole_lat = optional("POLE_LAT")
    pole_lon = optional("POLE_LON")
    globe = _wrf_cartopy_globe(ccrs)

    if map_proj == 1:
        stand_lon = required("STAND_LON", cen_lon)
        moad_cen_lat = required("MOAD_CEN_LAT", cen_lat)
        truelat1 = required("TRUELAT1")
        standard_parallels = [truelat1]
        # wrf-python treats an absent or out-of-range TRUELAT2 as missing.
        if truelat2 is not None and abs(truelat2) <= 90.0:
            standard_parallels.append(truelat2)
        cutoff = -30.0 if moad_cen_lat >= 0.0 else 30.0
        return ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=moad_cen_lat,
            standard_parallels=tuple(standard_parallels),
            globe=globe,
            cutoff=cutoff,
        )

    if map_proj == 2:
        stand_lon = required("STAND_LON", cen_lon)
        truelat1 = required("TRUELAT1")
        return ccrs.Stereographic(
            central_latitude=-90.0 if truelat1 < 0.0 else 90.0,
            central_longitude=stand_lon,
            true_scale_latitude=truelat1,
            globe=globe,
        )

    if map_proj == 3:
        # WRF and wrf-python default a missing Mercator standard longitude
        # to zero, not to the nested-domain center longitude.
        stand_lon = optional("STAND_LON", 0.0)
        kwargs = {
            "central_longitude": stand_lon,
            "globe": globe,
        }
        if truelat1 is not None and truelat1 != 0.0:
            kwargs["latitude_true_scale"] = truelat1
        return ccrs.Mercator(**kwargs)

    if map_proj == 6:
        stand_lon = required("STAND_LON", cen_lon)

        # WRF's default pole describes an ordinary, unrotated lat/lon grid.
        # Older files sometimes omit both default POLE_* attributes.
        if ((pole_lat == 90.0 and pole_lon == 0.0) or
                (pole_lat is None and pole_lon is None)):
            return ccrs.PlateCarree(
                central_longitude=stand_lon,
                globe=globe,
            )

        if pole_lat is None or pole_lon is None:
            raise ValueError(
                "Rotated WRF MAP_PROJ=6 requires both POLE_LAT and POLE_LON"
            )

        # Match NCAR/wrf-python's RotatedLatLon conversion. POLE_LON normally
        # distinguishes the northern (180) and southern (0) conventions.
        north = True
        if pole_lon == 0.0:
            north = False
        elif pole_lon != 180.0 and moad_cen_lat is not None:
            north = moad_cen_lat >= 0.0
        cart_pole_lat = pole_lat if north else -pole_lat
        cart_pole_lon = -stand_lon - 180.0 if north else -stand_lon
        return ccrs.RotatedPole(
            pole_longitude=cart_pole_lon,
            pole_latitude=cart_pole_lat,
            central_rotated_longitude=180.0 - pole_lon,
            globe=globe,
        )

    raise ValueError(
        f"Unsupported WRF map projection MAP_PROJ={map_proj}. "
        "Supported: 1 (Lambert), 2 (Polar Stereographic), "
        "3 (Mercator), 6 (Lat-Lon)."
    )


def get_cartopy(wrffile):
    """Get a cartopy CRS projection from a WRF file.

    Drop-in replacement for ``wrf.get_cartopy()`` from wrf-python.

    First reads the WRF global projection attributes via netCDF4 (if
    available), following NCAR wrf-python's projection parameter semantics.
    Falls back to inferring the projection from the lat/lon arrays in the
    Rust WrfFile handle when netCDF4 is not installed.

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
            attrs = _read_wrf_projection_attrs(nc)
        finally:
            nc.close()
        return _cartopy_from_wrf_attrs(ccrs, attrs)
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
        return ccrs.PlateCarree(
            central_longitude=cen_lon,
            globe=_wrf_cartopy_globe(ccrs),
        )
    else:
        # Default to Lambert Conformal -- the most common WRF projection.
        # Use the domain center and reasonable standard parallels.
        return ccrs.LambertConformal(
            central_longitude=cen_lon,
            central_latitude=cen_lat,
            standard_parallels=(cen_lat - 5.0, cen_lat + 5.0),
            globe=_wrf_cartopy_globe(ccrs),
            cutoff=-30.0 if cen_lat >= 0.0 else 30.0,
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
# ll_to_xy / xy_to_ll -- analytic WRF projection coordinates
# =========================================================================

_MISSING = object()


def _normalize_stagger(stagger):
    if stagger is None:
        return "m"
    normalized = str(stagger).lower()
    if normalized not in ("m", "u", "v"):
        raise ValueError("invalid 'stagger' value; expected None, 'm', 'u', or 'v'")
    return normalized


def _is_dataset_source(source):
    return hasattr(source, "variables") and (
        hasattr(source, "getncattr") or hasattr(source, "attrs")
    )


class _CoordinateSource:
    """Own an optional netCDF handle used to read projection metadata."""

    def __init__(self, wrffile):
        self.wrffile = wrffile
        self.source = None
        self._opened = None

    def __enter__(self):
        if _is_dataset_source(self.wrffile):
            self.source = self.wrffile
            return self.source

        wf = _ensure_wrffile(self.wrffile)
        native = wf._inner
        if hasattr(native, "_global_attr_f64") and hasattr(native, "_has_var"):
            self.source = native
            return self.source

        try:
            from netCDF4 import Dataset as _NCDataset
        except ImportError as exc:
            raise ImportError(
                "analytic WRF coordinate conversion needs projection globals; "
                "install netCDF4 or use a wrf-rust wheel exposing native metadata"
            ) from exc
        self._opened = _NCDataset(wf.path, "r")
        self.source = self._opened
        return self.source

    def __exit__(self, exc_type, exc_value, traceback):
        if self._opened is not None:
            self._opened.close()
        return False


def _source_numeric_attr(source, name, default=_MISSING):
    try:
        if hasattr(source, "_global_attr_f64"):
            value = source._global_attr_f64(name)
        elif hasattr(source, "getncattr"):
            value = source.getncattr(name)
        else:
            attrs = getattr(source, "attrs", {})
            if name not in attrs:
                raise KeyError(name)
            value = attrs[name]
    except (AttributeError, KeyError, RuntimeError, OSError):
        if default is not _MISSING:
            return default
        raise ValueError(f"WRF file is missing required projection attribute {name}") from None

    if np.ma.is_masked(value):
        if default is not _MISSING:
            return default
        raise ValueError(f"WRF projection attribute {name} is masked")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"WRF projection attribute {name} is not numeric: {value!r}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"WRF projection attribute {name} is not finite: {value!r}")
    return value


def _source_has_var(source, name):
    if hasattr(source, "_has_var"):
        return bool(source._has_var(name))
    return name in getattr(source, "variables", {})


def _coordinate_var_names(source, stagger):
    if stagger == "m":
        lat_candidates = ("XLAT", "XLAT_M")
        lon_candidates = ("XLONG", "XLONG_M")
    else:
        suffix = stagger.upper()
        lat_candidates = (f"XLAT_{suffix}",)
        lon_candidates = (f"XLONG_{suffix}",)

    lat_name = next((name for name in lat_candidates if _source_has_var(source, name)), None)
    lon_name = next((name for name in lon_candidates if _source_has_var(source, name)), None)
    if lat_name is None or lon_name is None:
        grid_name = {"m": "mass", "u": "U-staggered", "v": "V-staggered"}[stagger]
        raise ValueError(f"WRF file is missing {grid_name} latitude/longitude variables")
    return lat_name, lon_name


def _as_finite_reference_series(values, name):
    array = np.ma.asarray(values, dtype=np.float64)
    array = np.asarray(np.ma.filled(array, np.nan), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"WRF coordinate variable {name} has no finite reference point")
    return array


def _source_reference_series(source, name):
    if hasattr(source, "_has_var"):
        nt = int(source.nt)
        values = np.empty(nt, dtype=np.float64)
        for time_index in range(nt):
            coordinate = np.asarray(
                source.getvar(name, timeidx=time_index), dtype=np.float64
            )
            if coordinate.ndim != 2 or coordinate.size == 0:
                raise ValueError(
                    f"WRF coordinate variable {name} must be a non-empty 2-D field"
                )
            values[time_index] = coordinate[0, 0]
        return _as_finite_reference_series(values, name)

    variable = source.variables[name]
    shape = tuple(variable.shape)
    if len(shape) == 2:
        values = variable[0, 0]
    elif len(shape) == 3:
        values = variable[:, 0, 0]
    else:
        raise ValueError(
            f"WRF coordinate variable {name} has unsupported shape {shape}; "
            "expected (y, x) or (time, y, x)"
        )
    return _as_finite_reference_series(values, name)


def _projection_params(wrffile, timeidx, stagger):
    if isinstance(timeidx, np.integer):
        timeidx = int(timeidx)
    if timeidx is not None and not isinstance(timeidx, int):
        raise TypeError("'timeidx' must be an integer or None")
    if timeidx is not None and timeidx < 0:
        # This matches the pinned wrf-python coordinate implementation even
        # though several other wrf-python APIs accept negative time indices.
        raise ValueError("'timeidx' must be greater than or equal to 0")

    stagger = _normalize_stagger(stagger)
    with _CoordinateSource(wrffile) as source:
        map_proj_value = _source_numeric_attr(source, "MAP_PROJ")
        if not map_proj_value.is_integer():
            raise ValueError(f"WRF MAP_PROJ must be an integer, got {map_proj_value!r}")
        map_proj = int(map_proj_value)
        truelat1 = _source_numeric_attr(source, "TRUELAT1")
        truelat2 = _source_numeric_attr(source, "TRUELAT2")
        stand_lon = _source_numeric_attr(source, "STAND_LON")
        dx = _source_numeric_attr(source, "DX")
        dy = _source_numeric_attr(source, "DY")
        pole_lat = _source_numeric_attr(source, "POLE_LAT", 90.0)
        pole_lon = _source_numeric_attr(source, "POLE_LON", 0.0)

        lat_name, lon_name = _coordinate_var_names(source, stagger)
        ref_lats = _source_reference_series(source, lat_name)
        ref_lons = _source_reference_series(source, lon_name)

    if ref_lats.size != ref_lons.size:
        raise ValueError("WRF latitude/longitude reference series have different lengths")
    if ref_lats.size > 1:
        lat_moved = np.any(np.abs(ref_lats - ref_lats[0]) > 1.0e-10)
        lon_moved = any(
            abs(_longitude_delta(value, ref_lons[0])) > 1.0e-10
            for value in ref_lons[1:]
        )
        if lat_moved or lon_moved:
            raise NotImplementedError(
                "moving-domain WRF coordinate metadata is not yet supported; "
                "ll_to_xy/xy_to_ll will not silently use or clamp to time 0"
            )

    latinc = 0.0
    loninc = 0.0
    if map_proj == 6:
        latinc = dy * 360.0 / (2.0 * np.pi * _COORD_EARTH_RADIUS)
        loninc = dx * 360.0 / (2.0 * np.pi * _COORD_EARTH_RADIUS)

    return _ProjectionParams(
        map_proj=map_proj,
        truelat1=truelat1,
        truelat2=truelat2,
        stand_lon=stand_lon,
        ref_lat=float(ref_lats[0]),
        ref_lon=float(ref_lons[0]),
        dx=dx,
        dy=dy,
        pole_lat=pole_lat,
        pole_lon=pole_lon,
        latinc=latinc,
        loninc=loninc,
    )


def _coordinate_inputs(first, second, first_name, second_name):
    try:
        first_values = np.asarray(first, dtype=np.float64)
        second_values = np.asarray(second, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{first_name}' and '{second_name}' must be numeric") from exc

    first_scalar = first_values.ndim == 0
    second_scalar = second_values.ndim == 0
    if first_scalar != second_scalar:
        raise ValueError(f"'{first_name}' and '{second_name}' must be the same length")
    if first_scalar:
        return first_values.reshape(1), second_values.reshape(1), True

    first_values = first_values.ravel()
    second_values = second_values.ravel()
    if first_values.size != second_values.size:
        raise ValueError(f"'{first_name}' and '{second_name}' must be the same length")
    return first_values, second_values, False


def _with_coordinate_metadata(result, first, second, *, xy, meta, squeeze):
    if not meta:
        return result
    try:
        from xarray import DataArray
    except ImportError:
        return result

    data = result if result.ndim != 1 else result[:, np.newaxis]
    first_values = np.asarray(first).ravel()
    second_values = np.asarray(second).ravel()
    pairs = np.empty(first_values.size, dtype=object)
    if xy:
        for index, (latitude, longitude) in enumerate(
                zip(first_values, second_values)):
            pairs[index] = CoordPair(lat=latitude, lon=longitude)
        dims = ("x_y", "idx")
        coords = {
            "x_y": ["x", "y"],
            "latlon_coord": ("idx", pairs),
        }
        name = "xy"
    else:
        for index, (x_value, y_value) in enumerate(
                zip(first_values, second_values)):
            pairs[index] = CoordPair(x=x_value, y=y_value)
        dims = ("lat_lon", "idx")
        coords = {
            "lat_lon": ["lat", "lon"],
            "xy_coord": ("idx", pairs),
        }
        name = "latlon"

    output = DataArray(data, name=name, dims=dims, coords=coords)
    return output.squeeze() if squeeze else output


def ll_to_xy(wrfin, latitude, longitude, timeidx=0, squeeze=True,
             meta=True, stagger=None, as_int=True):
    """Return zero-based WRF x/y coordinates for latitude/longitude values.

    This follows NCAR wrf-python 1.3.4.1's analytic WRF projection equations.
    Scalar inputs produce a leading two-element x/y result and sequences are
    flattened to ``(2, npoints)``. U and V staggering select the corresponding
    ``XLAT_U/XLONG_U`` or ``XLAT_V/XLONG_V`` projection origin. Coordinates
    outside the domain are extrapolated, as in wrf-python, rather than clamped.
    """
    params = _projection_params(wrfin, timeidx, stagger)
    latitudes, longitudes, scalar = _coordinate_inputs(
        latitude, longitude, "latitude", "longitude"
    )
    result = np.empty((2, latitudes.size), dtype=np.float64)
    for index, (lat_value, lon_value) in enumerate(zip(latitudes, longitudes)):
        result[:, index] = _project_ll_to_xy(params, lat_value, lon_value)

    if scalar:
        result = result[:, 0]
    if as_int:
        result = np.rint(result).astype(int)
    return _with_coordinate_metadata(
        result,
        latitude,
        longitude,
        xy=True,
        meta=meta,
        squeeze=squeeze,
    )


def xy_to_ll(wrfin, x, y, timeidx=0, squeeze=True, meta=True,
             stagger=None):
    """Return latitude/longitude values for zero-based WRF x/y coordinates.

    The return convention matches wrf-python: the leading axis is latitude,
    longitude; scalar inputs produce shape ``(2,)`` and sequences produce
    ``(2, npoints)``. Antimeridian results are normalized to [-180, 180].
    """
    params = _projection_params(wrfin, timeidx, stagger)
    x_values, y_values, scalar = _coordinate_inputs(x, y, "x", "y")
    result = np.empty((2, x_values.size), dtype=np.float64)
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        result[:, index] = _project_xy_to_ll(params, x_value, y_value)

    if scalar:
        result = result[:, 0]
    return _with_coordinate_metadata(
        result,
        x,
        y,
        xy=False,
        meta=meta,
        squeeze=squeeze,
    )
