"""Interactive WRF output explorer for Jupyter notebooks.

Provides widget-based tools for browsing variables, time steps, and
vertical levels from WRF output files.  Falls back gracefully when
Jupyter dependencies (ipywidgets, matplotlib) are not installed.

Usage::

    from wrf import Explorer
    ex = Explorer("wrfout_d01_2024-06-01_00:00:00")
    ex.show()

    from wrf import cross_section, profile, hovmoller
    cross_section(f, "temp", timeidx=0, start=(35, -97), end=(36, -96))
    profile(f, ["temp", "td", "theta_e"], timeidx=0, point=(35.0, -97.5))
    hovmoller(f, "temp", lat=35.0, lon=-97.5)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from wrf import WrfFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_jupyter_deps():
    """Raise a clear message if ipywidgets/IPython are missing."""
    try:
        import ipywidgets  # noqa: F401
        import IPython.display  # noqa: F401
    except ImportError:
        raise ImportError(
            "The wrf explorer requires ipywidgets and IPython.\n"
            "Install them with:  pip install ipywidgets matplotlib"
        )


def _ensure_wrffile(f):
    """Coerce input to a WrfFile."""
    from wrf import WrfFile
    if isinstance(f, WrfFile):
        return f
    return WrfFile(f)


def _get_var_meta():
    """Return {name: {description, units}} for every registered variable."""
    from wrf import list_variables
    return {v["name"]: v for v in list_variables()}


def _find_nearest_ij(f, lat: float, lon: float) -> Tuple[int, int]:
    """Find the (j, i) grid indices nearest to a given lat/lon.

    Reads XLAT and XLONG via ``getvar`` and finds the minimum-distance
    point.  Returns ``(j, i)`` -- row, column order matching numpy
    array indexing.
    """
    from wrf import getvar
    xlat = getvar(f, "lat", timeidx=0)   # (ny, nx)
    xlon = getvar(f, "lon", timeidx=0)   # (ny, nx)
    dist = np.sqrt((xlat - lat) ** 2 + (xlon - lon) ** 2)
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    return int(j), int(i)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (degrees)."""
    R = 6371.0
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _bresenham_indices(j0, i0, j1, i1):
    """Return arrays of (j, i) indices along a line between two grid cells.

    Uses Bresenham's line algorithm for clean, evenly-spaced sampling
    through the grid.
    """
    dj = abs(j1 - j0)
    di = abs(i1 - i0)
    sj = 1 if j0 < j1 else -1
    si = 1 if i0 < i1 else -1
    err = di - dj
    js, iss = [j0], [i0]
    j, i = j0, i0
    while j != j1 or i != i1:
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
        js.append(j)
        iss.append(i)
    return np.array(js), np.array(iss)


# ---------------------------------------------------------------------------
# Default colormap / styling helpers
# ---------------------------------------------------------------------------

def _pick_cmap(varname: str):
    """Pick a reasonable default colormap for a WRF variable."""
    import matplotlib.pyplot as plt
    name_lower = varname.lower()

    # Temperature-family
    if any(k in name_lower for k in ("temp", "tc", "theta", "tv", "twb", "td",
                                      "ctt", "dewpoint", "wet_bulb")):
        return plt.cm.RdYlBu_r

    # Moisture
    if any(k in name_lower for k in ("rh", "pw", "mixing_ratio", "specific_humidity",
                                      "dp2m")):
        return plt.cm.BrBG

    # CAPE / instability
    if any(k in name_lower for k in ("cape", "cin", "stp", "scp", "ehi",
                                      "ship", "bri")):
        return plt.cm.hot_r

    # Wind
    if any(k in name_lower for k in ("wspd", "shear", "mean_wind")):
        return plt.cm.YlOrRd

    # Reflectivity
    if "dbz" in name_lower or "reflectivity" in name_lower:
        return plt.cm.gist_ncar

    # Helicity / vorticity
    if any(k in name_lower for k in ("srh", "helicity", "vorticity", "uhel")):
        return plt.cm.PuOr_r

    # Pressure
    if any(k in name_lower for k in ("slp", "pressure", "pres", "omega")):
        return plt.cm.coolwarm

    # Heights / levels
    if any(k in name_lower for k in ("height", "terrain", "z_agl", "lcl",
                                      "lfc", "el", "freezing", "wet_bulb_0")):
        return plt.cm.terrain

    # Cloud
    if "cloud" in name_lower:
        return plt.cm.gray_r

    # Fire weather
    if any(k in name_lower for k in ("fosberg", "haines", "hdw")):
        return plt.cm.YlOrRd

    return plt.cm.viridis


def _smart_title(varname: str, meta: dict, units_override: Optional[str],
                 timeidx: int, times: Optional[list] = None) -> str:
    """Build a descriptive plot title."""
    info = meta.get(varname, {})
    desc = info.get("description", varname)
    units = units_override or info.get("units", "")
    time_str = ""
    if times and 0 <= timeidx < len(times):
        time_str = f"  |  {times[timeidx]}"
    unit_str = f" [{units}]" if units else ""
    return f"{desc}{unit_str}{time_str}"


# ---------------------------------------------------------------------------
# Core plot function (used by all explorer tools)
# ---------------------------------------------------------------------------

def plot_field(data, title="", cmap=None, ax=None, colorbar=True,
               vmin=None, vmax=None, xlabel=None, ylabel=None,
               figsize=(10, 7)):
    """Plot a 2D field with filled contours.

    Parameters
    ----------
    data : ndarray
        2-D array to plot.
    title : str
        Plot title.
    cmap : colormap, optional
        Matplotlib colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; created if None.
    colorbar : bool
        Whether to add a colorbar.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.viridis

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888")
        ax.set_title(title)
        return fig

    if vmin is None:
        vmin = float(np.percentile(finite, 2))
    if vmax is None:
        vmax = float(np.percentile(finite, 98))
    if vmin == vmax:
        vmax = vmin + 1.0

    im = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if created_fig:
        fig.tight_layout()

    return fig


# ===================================================================
# Explorer class -- interactive widget-based variable browser
# ===================================================================

class Explorer:
    """Interactive WRF output explorer for Jupyter notebooks.

    Parameters
    ----------
    path_or_file : str or WrfFile
        Path to a WRF output file, or an already-opened ``WrfFile``.

    Example
    -------
    >>> from wrf import Explorer
    >>> ex = Explorer("wrfout_d01_2024-06-01_00:00:00")
    >>> ex.show()
    """

    def __init__(self, path_or_file):
        _check_jupyter_deps()

        self._f = _ensure_wrffile(path_or_file)
        self._meta = _get_var_meta()
        self._var_names = sorted(self._meta.keys())
        self._times = self._f.times()

        # Cache the last-fetched data to avoid redundant compute
        self._cache_key = None
        self._cache_data = None

        self._build_widgets()

    # ── Widget construction ──

    def _build_widgets(self):
        import ipywidgets as w

        # Variable dropdown
        options = []
        for name in self._var_names:
            info = self._meta[name]
            label = f"{name}  --  {info['description']}  [{info['units']}]"
            options.append((label, name))

        self._var_dd = w.Dropdown(
            options=options,
            value=self._var_names[0] if self._var_names else None,
            description="Variable:",
            style={"description_width": "70px"},
            layout=w.Layout(width="500px"),
        )

        # Time slider
        nt = max(self._f.nt, 1)
        self._time_slider = w.IntSlider(
            value=0, min=0, max=nt - 1, step=1,
            description="Time:",
            style={"description_width": "70px"},
            layout=w.Layout(width="400px"),
            continuous_update=False,
        )
        # Time label
        self._time_label = w.Label(
            value=self._times[0] if self._times else "",
            layout=w.Layout(width="200px"),
        )

        # Level slider (for 3D vars)
        nz = max(self._f.nz, 1)
        self._level_slider = w.IntSlider(
            value=0, min=0, max=nz - 1, step=1,
            description="Level:",
            style={"description_width": "70px"},
            layout=w.Layout(width="400px"),
            continuous_update=False,
        )
        self._level_box = w.HBox(
            [self._level_slider],
            layout=w.Layout(display="flex"),
        )

        # Units text
        self._units_input = w.Text(
            value="",
            placeholder="e.g. degC, hPa, knots",
            description="Units:",
            style={"description_width": "70px"},
            layout=w.Layout(width="300px"),
            continuous_update=False,
        )

        # Variable info
        self._info_label = w.HTML(
            value="",
            layout=w.Layout(width="500px"),
        )

        # Output area for the plot
        self._output = w.Output(
            layout=w.Layout(width="100%", min_height="400px"),
        )

        # Wire up observers
        self._var_dd.observe(self._on_var_change, names="value")
        self._time_slider.observe(self._on_param_change, names="value")
        self._level_slider.observe(self._on_param_change, names="value")
        self._units_input.observe(self._on_param_change, names="value")

        # Trigger initial state
        self._on_var_change(None)

    def _on_var_change(self, change):
        """Called when the variable dropdown changes."""
        name = self._var_dd.value
        if name is None:
            return

        info = self._meta.get(name, {})
        desc = info.get("description", "")
        units = info.get("units", "")

        # Probe the shape to determine 2D vs 3D
        is_3d = self._is_3d(name)

        if is_3d:
            self._level_box.layout.display = "flex"
        else:
            self._level_box.layout.display = "none"

        shape_str = f"({self._f.nz}, {self._f.ny}, {self._f.nx})" if is_3d else f"({self._f.ny}, {self._f.nx})"
        self._info_label.value = (
            f"<b>{name}</b>: {desc} &nbsp;|&nbsp; "
            f"Units: <code>{units}</code> &nbsp;|&nbsp; "
            f"Shape: <code>{shape_str}</code>"
        )

        # Reset units placeholder to default
        self._units_input.placeholder = f"default: {units}" if units else "e.g. degC, hPa"

        self._update_plot()

    def _on_param_change(self, change):
        """Called when time, level, or units change."""
        # Update time label
        tidx = self._time_slider.value
        if self._times and 0 <= tidx < len(self._times):
            self._time_label.value = self._times[tidx]
        self._update_plot()

    def _is_3d(self, name: str) -> bool:
        """Check if a variable produces 3D output by fetching one slice."""
        key = ("_is3d", name)
        if key == self._cache_key:
            return self._cache_data

        from wrf import getvar
        try:
            data = getvar(self._f, name, timeidx=0)
            result = data.ndim == 3
            # Also cache the actual data for timeidx=0 to reuse
        except Exception:
            result = False

        self._cache_key = key
        self._cache_data = result
        return result

    def _fetch(self, name: str, timeidx: int, units: Optional[str]):
        """Fetch data, using a simple cache."""
        cache_key = (name, timeidx, units)
        if cache_key == self._cache_key:
            return self._cache_data

        from wrf import getvar
        kwargs = {}
        if units:
            kwargs["units"] = units
        data = getvar(self._f, name, timeidx=timeidx, **kwargs)
        self._cache_key = cache_key
        self._cache_data = data
        return data

    def _update_plot(self):
        """Re-render the plot with current widget state."""
        import matplotlib.pyplot as plt

        name = self._var_dd.value
        if name is None:
            return

        tidx = self._time_slider.value
        units = self._units_input.value.strip() or None

        self._output.clear_output(wait=True)
        with self._output:
            try:
                data = self._fetch(name, tidx, units)

                # For 3D data, slice at the selected level
                display_data = data
                if data.ndim == 3:
                    lvl = self._level_slider.value
                    lvl = min(lvl, data.shape[0] - 1)
                    display_data = data[lvl]

                title = _smart_title(
                    name, self._meta, units, tidx, self._times,
                )
                if data.ndim == 3:
                    title += f"  |  Level {self._level_slider.value}"

                cmap = _pick_cmap(name)
                fig = plot_field(display_data, title=title, cmap=cmap)
                plt.show()

            except Exception as e:
                from IPython.display import display, HTML
                display(HTML(
                    f"<div style='color:#c00; padding:10px;'>"
                    f"<b>Error computing {name}:</b> {e}</div>"
                ))

    # ── Public API ──

    def show(self):
        """Display the explorer widget in the notebook."""
        import ipywidgets as w
        from IPython.display import display

        time_row = w.HBox([self._time_slider, self._time_label])

        controls = w.VBox([
            self._var_dd,
            time_row,
            self._level_box,
            self._units_input,
            self._info_label,
        ], layout=w.Layout(padding="8px 0"))

        full = w.VBox([controls, self._output])
        display(full)

    def _repr_mimebundle_(self, **kwargs):
        """Auto-display in Jupyter when the object is the last expression."""
        self.show()
        return {"text/plain": repr(self)}

    def __repr__(self):
        return (
            f"Explorer('{self._f.path}', "
            f"nx={self._f.nx}, ny={self._f.ny}, nz={self._f.nz}, nt={self._f.nt})"
        )


# ===================================================================
# cross_section -- vertical cross-section viewer
# ===================================================================

def cross_section(
    f,
    varname: str,
    timeidx: int = 0,
    start: Optional[Tuple[float, float]] = None,
    end: Optional[Tuple[float, float]] = None,
    units: Optional[str] = None,
    height_type: str = "agl",
    num_points: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
    ax=None,
):
    """Plot a vertical cross-section of a 3D WRF variable.

    Parameters
    ----------
    f : WrfFile or str
        WRF output file.
    varname : str
        3-D variable name (e.g. "temp", "rh", "theta_e").
    timeidx : int
        Time index.
    start : (lat, lon) or None
        Starting point.  If None in a Jupyter notebook, launches an
        interactive click-to-select UI.
    end : (lat, lon) or None
        Ending point.
    units : str, optional
        Convert the variable to these units.
    height_type : str
        ``"agl"`` for metres above ground, ``"pressure"`` for hPa.
    num_points : int, optional
        Number of sample points along the section.  Default is the
        line length in grid cells.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Matplotlib axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from wrf import getvar

    f = _ensure_wrffile(f)

    # Interactive mode if start/end not given
    if start is None or end is None:
        return _cross_section_interactive(f, varname, timeidx, units,
                                          height_type, figsize)

    # Get lat/lon grids and find nearest grid points
    j0, i0 = _find_nearest_ij(f, start[0], start[1])
    j1, i1 = _find_nearest_ij(f, end[0], end[1])

    # Sample indices along the line
    jj, ii = _bresenham_indices(j0, i0, j1, i1)
    npts = len(jj)

    # Fetch the 3D variable
    kwargs = {}
    if units:
        kwargs["units"] = units
    data3d = getvar(f, varname, timeidx=timeidx, **kwargs)
    if data3d.ndim != 3:
        raise ValueError(f"cross_section requires a 3D variable, but '{varname}' is 2D")

    nz = data3d.shape[0]

    # Build the cross-section array: (nz, npts)
    xsec = np.empty((nz, npts), dtype=np.float64)
    for k in range(nz):
        xsec[k, :] = data3d[k, jj, ii]

    # Height coordinate
    if height_type == "pressure":
        hcoord = getvar(f, "pressure", timeidx=timeidx, units="hPa")
        ydata = np.empty((nz, npts), dtype=np.float64)
        for k in range(nz):
            ydata[k, :] = hcoord[k, jj, ii]
        ylabel = "Pressure (hPa)"
        invert_y = True
    else:
        hcoord = getvar(f, "height_agl", timeidx=timeidx)
        ydata = np.empty((nz, npts), dtype=np.float64)
        for k in range(nz):
            ydata[k, :] = hcoord[k, jj, ii]
        ylabel = "Height AGL (m)"
        invert_y = False

    # Distance along the section (km)
    xlat = getvar(f, "lat", timeidx=0)
    xlon = getvar(f, "lon", timeidx=0)
    dist = np.zeros(npts, dtype=np.float64)
    for n in range(1, npts):
        dist[n] = dist[n - 1] + _haversine_km(
            xlat[jj[n - 1], ii[n - 1]], xlon[jj[n - 1], ii[n - 1]],
            xlat[jj[n], ii[n]], xlon[jj[n], ii[n]],
        )

    # Terrain profile
    terrain = getvar(f, "terrain", timeidx=timeidx)
    ter_line = terrain[jj, ii]

    # 2D distance grid for pcolormesh
    dist2d = np.broadcast_to(dist[np.newaxis, :], (nz, npts))

    # Plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    cmap = _pick_cmap(varname)
    finite = xsec[np.isfinite(xsec)]
    if len(finite) > 0:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0, 1

    im = ax.pcolormesh(dist2d, ydata, xsec, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="gouraud")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    # Terrain fill
    if not invert_y:
        ax.fill_between(dist, 0, ter_line, color="#8B7355", alpha=0.8, zorder=5)
        ax.set_ylim(bottom=0)
    else:
        ax.invert_yaxis()

    meta = _get_var_meta()
    info = meta.get(varname, {})
    desc = info.get("description", varname)
    unit_str = units or info.get("units", "")
    ax.set_title(f"{desc} [{unit_str}]  --  Cross Section", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(ylabel)

    if created_fig:
        fig.tight_layout()

    plt.show()
    return fig


def _cross_section_interactive(f, varname, timeidx, units, height_type, figsize):
    """Interactive click-to-select cross-section UI."""
    _check_jupyter_deps()

    import ipywidgets as w
    from IPython.display import display
    import matplotlib.pyplot as plt
    from wrf import getvar

    # Show a terrain/SLP field for click reference
    output = w.Output()

    lat_start = w.FloatText(description="Start lat:", style={"description_width": "70px"},
                            layout=w.Layout(width="200px"))
    lon_start = w.FloatText(description="Start lon:", style={"description_width": "70px"},
                            layout=w.Layout(width="200px"))
    lat_end = w.FloatText(description="End lat:", style={"description_width": "70px"},
                          layout=w.Layout(width="200px"))
    lon_end = w.FloatText(description="End lon:", style={"description_width": "70px"},
                          layout=w.Layout(width="200px"))

    go_btn = w.Button(description="Plot Cross Section",
                      button_style="primary",
                      layout=w.Layout(width="200px"))

    info = w.HTML(value=(
        "<p>Enter start and end coordinates (lat, lon) for the cross section, "
        "then click <b>Plot Cross Section</b>.</p>"
    ))

    def on_click(btn):
        output.clear_output(wait=True)
        with output:
            try:
                cross_section(
                    f, varname, timeidx=timeidx,
                    start=(lat_start.value, lon_start.value),
                    end=(lat_end.value, lon_end.value),
                    units=units, height_type=height_type, figsize=figsize,
                )
            except Exception as e:
                from IPython.display import HTML as IPHTML
                display(IPHTML(f"<div style='color:#c00'><b>Error:</b> {e}</div>"))

    go_btn.on_click(on_click)

    # Show terrain map for reference
    with output:
        try:
            ter = getvar(f, "terrain", timeidx=0)
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.pcolormesh(ter, cmap=plt.cm.terrain, shading="auto")
            ax.set_title("Terrain (reference) -- enter lat/lon below", fontsize=11)
            ax.set_aspect("equal")
            fig.tight_layout()
            plt.show()
        except Exception:
            pass

    start_row = w.HBox([lat_start, lon_start])
    end_row = w.HBox([lat_end, lon_end])
    controls = w.VBox([info, start_row, end_row, go_btn])
    display(w.VBox([controls, output]))


# ===================================================================
# profile -- single-column vertical profile
# ===================================================================

def profile(
    f,
    varnames: Union[str, Sequence[str]],
    timeidx: int = 0,
    point: Optional[Tuple[float, float]] = None,
    j: Optional[int] = None,
    i: Optional[int] = None,
    units: Optional[Union[str, dict]] = None,
    height_type: str = "agl",
    figsize: Tuple[int, int] = (6, 8),
    ax=None,
):
    """Plot one or more 3D variables as vertical profiles at a single point.

    Parameters
    ----------
    f : WrfFile or str
        WRF output file.
    varnames : str or list of str
        One or more 3-D variable names (e.g. ``["temp", "td", "theta_e"]``).
    timeidx : int
        Time index.
    point : (lat, lon), optional
        Latitude/longitude of the profile location.  The nearest grid
        point is used.  Specify either ``point`` or ``(j, i)``.
    j, i : int, optional
        Direct grid indices (row, column).
    units : str or dict, optional
        Unit conversion.  A single string applies to all variables; a
        dict maps variable names to units (e.g.
        ``{"temp": "degC", "td": "degC"}``).
    height_type : str
        ``"agl"`` for metres above ground (default) or ``"pressure"``
        for hPa.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Matplotlib axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from wrf import getvar

    f = _ensure_wrffile(f)

    if isinstance(varnames, str):
        varnames = [varnames]

    # Resolve grid point
    if point is not None:
        jj, ii = _find_nearest_ij(f, point[0], point[1])
    elif j is not None and i is not None:
        jj, ii = j, i
    else:
        raise ValueError("Provide either point=(lat, lon) or j=, i= grid indices")

    # Height coordinate
    if height_type == "pressure":
        hcoord = getvar(f, "pressure", timeidx=timeidx, units="hPa")
        heights = hcoord[:, jj, ii]
        ylabel = "Pressure (hPa)"
        invert_y = True
    else:
        hcoord = getvar(f, "height_agl", timeidx=timeidx)
        heights = hcoord[:, jj, ii]
        ylabel = "Height AGL (m)"
        invert_y = False

    # Set up plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    meta = _get_var_meta()
    colors = plt.cm.tab10.colors

    for idx, vname in enumerate(varnames):
        # Determine units for this variable
        if isinstance(units, dict):
            vunits = units.get(vname, None)
        elif isinstance(units, str):
            vunits = units
        else:
            vunits = None

        kwargs = {}
        if vunits:
            kwargs["units"] = vunits

        data3d = getvar(f, vname, timeidx=timeidx, **kwargs)
        if data3d.ndim != 3:
            warnings.warn(f"Skipping '{vname}': not a 3D variable")
            continue

        col_data = data3d[:, jj, ii]
        info = meta.get(vname, {})
        unit_str = vunits or info.get("units", "")
        label = f"{info.get('description', vname)} [{unit_str}]"

        color = colors[idx % len(colors)]
        ax.plot(col_data, heights, color=color, linewidth=1.8, label=label)

    if invert_y:
        ax.invert_yaxis()

    # Title
    if point is not None:
        loc_str = f"({point[0]:.2f}, {point[1]:.2f})"
    else:
        loc_str = f"(j={jj}, i={ii})"

    times = f.times()
    time_str = ""
    if times and 0 <= timeidx < len(times):
        time_str = f"  |  {times[timeidx]}"

    ax.set_title(f"Vertical Profile at {loc_str}{time_str}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variable value")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    plt.show()
    return fig


# ===================================================================
# hovmoller -- time-height or time-lat Hovmoller diagrams
# ===================================================================

def hovmoller(
    f,
    varname: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    j: Optional[int] = None,
    i: Optional[int] = None,
    units: Optional[str] = None,
    mode: str = "auto",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """Plot a Hovmoller (time-height or time-latitude) diagram.

    Parameters
    ----------
    f : WrfFile or str
        WRF output file.
    varname : str
        Variable name.
    lat, lon : float, optional
        For time-height mode: fix a point and show time vs. height.
    j, i : int, optional
        Direct grid indices instead of lat/lon.
    units : str, optional
        Unit conversion.
    mode : str
        ``"auto"`` (default): time-height if variable is 3D,
        time-lat if 2D.  Or force ``"time-height"`` / ``"time-lat"``.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from wrf import getvar, ALL_TIMES

    f = _ensure_wrffile(f)
    meta = _get_var_meta()
    times = f.times()
    nt = f.nt

    # Resolve grid point
    if lat is not None and lon is not None:
        jj, ii = _find_nearest_ij(f, lat, lon)
    elif j is not None and i is not None:
        jj, ii = j, i
    else:
        # Default to domain center
        jj, ii = f.ny // 2, f.nx // 2

    # Probe dimensionality
    kwargs = {}
    if units:
        kwargs["units"] = units
    probe = getvar(f, varname, timeidx=0, **kwargs)
    is_3d = probe.ndim == 3

    # Determine mode
    if mode == "auto":
        mode = "time-height" if is_3d else "time-lat"

    # Build the Hovmoller data
    if mode == "time-height":
        if not is_3d:
            raise ValueError(
                f"time-height Hovmoller requires a 3D variable, but '{varname}' is 2D. "
                f"Use mode='time-lat' instead."
            )
        nz = probe.shape[0]
        hov = np.empty((nt, nz), dtype=np.float64)
        heights = np.empty((nt, nz), dtype=np.float64)

        for t in range(nt):
            data3d = getvar(f, varname, timeidx=t, **kwargs)
            hov[t, :] = data3d[:, jj, ii]
            hagl = getvar(f, "height_agl", timeidx=t)
            heights[t, :] = hagl[:, jj, ii]

        # Average heights across time for a stable y-axis
        y = np.mean(heights, axis=0)
        ylabel = "Height AGL (m)"

        # X-axis is time index
        x = np.arange(nt)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

        cmap = _pick_cmap(varname)
        finite = hov[np.isfinite(hov)]
        if len(finite) > 0:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0, 1

        # Broadcast for pcolormesh: (nt, nz)
        x2d = np.broadcast_to(x[:, np.newaxis], (nt, nz))
        y2d = np.broadcast_to(y[np.newaxis, :], (nt, nz))

        im = ax.pcolormesh(x2d, y2d, hov, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="gouraud")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        # X tick labels
        if nt <= 24:
            ax.set_xticks(x)
            labels = [times[t] if t < len(times) else str(t) for t in x]
            # Shorten labels: show only HH:MM if dates are the same
            if len(set(l[:10] for l in labels if len(l) >= 10)) == 1:
                labels = [l[11:16] if len(l) >= 16 else l for l in labels]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        else:
            # Too many -- just use a subset
            step = max(1, nt // 12)
            ticks = x[::step]
            ax.set_xticks(ticks)
            labels = [times[t][11:16] if t < len(times) and len(times[t]) >= 16
                      else str(t) for t in ticks]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        info = meta.get(varname, {})
        desc = info.get("description", varname)
        unit_str = units or info.get("units", "")
        loc_str = f"({lat:.2f}, {lon:.2f})" if lat is not None else f"(j={jj}, i={ii})"
        ax.set_title(f"{desc} [{unit_str}] -- Time-Height at {loc_str}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)

    elif mode == "time-lat":
        # Average over longitude (or at a fixed lon column)
        if is_3d:
            # Use lowest level
            level = 0
            hov = np.empty((nt, f.ny), dtype=np.float64)
            for t in range(nt):
                data = getvar(f, varname, timeidx=t, **kwargs)
                hov[t, :] = data[level, :, ii]
        else:
            hov = np.empty((nt, f.ny), dtype=np.float64)
            for t in range(nt):
                data = getvar(f, varname, timeidx=t, **kwargs)
                hov[t, :] = data[:, ii]

        xlat = getvar(f, "lat", timeidx=0)
        lat_axis = xlat[:, ii]  # (ny,)
        x = np.arange(nt)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

        cmap = _pick_cmap(varname)
        finite = hov[np.isfinite(hov)]
        if len(finite) > 0:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0, 1

        x2d = np.broadcast_to(x[:, np.newaxis], hov.shape)
        y2d = np.broadcast_to(lat_axis[np.newaxis, :], hov.shape)

        im = ax.pcolormesh(x2d, y2d, hov, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="gouraud")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        # X tick labels
        if nt <= 24:
            ax.set_xticks(x)
            labels = [times[t] if t < len(times) else str(t) for t in x]
            if len(set(l[:10] for l in labels if len(l) >= 10)) == 1:
                labels = [l[11:16] if len(l) >= 16 else l for l in labels]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        else:
            step = max(1, nt // 12)
            ticks = x[::step]
            ax.set_xticks(ticks)
            labels = [times[t][11:16] if t < len(times) and len(times[t]) >= 16
                      else str(t) for t in ticks]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        info = meta.get(varname, {})
        desc = info.get("description", varname)
        unit_str = units or info.get("units", "")
        lvl_note = " (sfc level)" if is_3d else ""
        ax.set_title(f"{desc} [{unit_str}] -- Time-Latitude{lvl_note}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Latitude")

    else:
        raise ValueError(f"Unknown mode '{mode}'; use 'time-height' or 'time-lat'")

    if created_fig:
        fig.tight_layout()

    plt.show()
    return fig
