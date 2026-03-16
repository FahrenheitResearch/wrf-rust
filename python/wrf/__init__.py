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

import numpy as np

from wrf._wrf import WrfFile as _WrfFile
from wrf._wrf import list_variables as _list_variables

__all__ = [
    "WrfFile",
    "getvar",
    "list_variables",
    "ALL_TIMES",
    "available_variables",
]
__version__ = "0.1.0"

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
    layer_type=None,
    use_virtual=None,
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
        layer_type=layer_type,
        use_virtual=use_virtual,
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
