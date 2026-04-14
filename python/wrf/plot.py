"""
wrf.plot -- Quick, beautiful WRF visualization.

Zero-configuration plotting for operational meteorologists.  Every function
produces a good-looking result out of the box, but accepts overrides for
full customization.

    from wrf import WrfFile
    from wrf.plot import plot_field, plot_wind, plot_skewt, panel

    f = WrfFile("wrfout_d01_2024-05-20_18:00:00")
    plot_field(f, "sbcape")
    plot_wind(f)
    plot_skewt(f, timeidx=0, point=(35.2, -97.4))
    panel(f, ["sbcape", "srh1", "stp", "shear_0_6km"])

Cartopy is used when available; if not, plots fall back to plain lat/lon
axes.  Matplotlib is imported lazily so importing this module never crashes.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

def _import_mpl():
    """Return (matplotlib, pyplot) or raise a clear error."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for wrf.plot.  Install it with:\n"
            "    pip install matplotlib"
        )


def _import_cartopy():
    """Return (cartopy.crs, cartopy.feature) or (None, None)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        return None, None


def _import_metpy():
    """Return metpy.plots.SkewT or None."""
    try:
        from metpy.plots import SkewT
        return SkewT
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Private: WrfFile coercion
# ---------------------------------------------------------------------------

def _ensure_wrffile(f):
    """Coerce a path / Dataset / WrfFile into a WrfFile."""
    from wrf import WrfFile
    if isinstance(f, WrfFile):
        return f
    return WrfFile(f)


# ---------------------------------------------------------------------------
# Variable metadata: colormaps, contour levels, labels
# ---------------------------------------------------------------------------

# Keys are matched against the variable name (case-insensitive).  Matching
# is done by exact name first, then by substring/pattern.  Each entry:
#   cmap        - matplotlib colormap name
#   levels      - default contour levels (None = auto)
#   label       - colorbar label (None = use variable description + units)
#   center_zero - if True, center the colormap on 0
#   extend      - colorbar extend ('neither','both','min','max')

_VAR_STYLES: Dict[str, Dict[str, Any]] = {
    # ---- CAPE / CIN ----
    "sbcape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "mlcape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "mucape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "cape":   dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "effective_cape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "cape3d": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "ecape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "ncape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "ecape_cape": dict(cmap="YlOrRd", levels=np.arange(0, 4250, 250), extend="max"),
    "sbcin":  dict(cmap="Blues_r", levels=np.arange(-300, 1, 25), extend="min"),
    "mlcin":  dict(cmap="Blues_r", levels=np.arange(-300, 1, 25), extend="min"),
    "mucin":  dict(cmap="Blues_r", levels=np.arange(-300, 1, 25), extend="min"),
    "cin":    dict(cmap="Blues_r", levels=np.arange(-300, 1, 25), extend="min"),
    "ecape_cin": dict(cmap="Blues_r", levels=np.arange(-300, 1, 25), extend="min"),

    # ---- Sounding levels ----
    "lcl":    dict(cmap="YlGnBu_r", levels=np.arange(0, 4200, 200), extend="max"),
    "lfc":    dict(cmap="YlGnBu_r", levels=np.arange(0, 5500, 500), extend="max"),
    "el":     dict(cmap="YlOrRd", levels=np.arange(0, 16000, 1000), extend="max"),
    "ecape_lfc": dict(cmap="YlGnBu_r", levels=np.arange(0, 5500, 500), extend="max"),
    "ecape_el": dict(cmap="YlOrRd", levels=np.arange(0, 16000, 1000), extend="max"),

    # ---- Helicity / Shear ----
    "srh1":   dict(cmap="RdPu", levels=np.arange(0, 525, 25), extend="max"),
    "srh3":   dict(cmap="RdPu", levels=np.arange(0, 525, 25), extend="max"),
    "srh":    dict(cmap="RdPu", levels=np.arange(0, 525, 25), extend="max"),
    "effective_srh": dict(cmap="RdPu", levels=np.arange(0, 525, 25), extend="max"),
    "shear_0_1km": dict(cmap="YlOrRd", levels=np.arange(0, 42, 2), extend="max"),
    "shear_0_6km": dict(cmap="YlOrRd", levels=np.arange(0, 42, 2), extend="max"),
    "bulk_shear":  dict(cmap="YlOrRd", levels=np.arange(0, 42, 2), extend="max"),

    # ---- Severe composites ----
    "stp":           dict(cmap="Reds", levels=np.arange(0, 11, 1), extend="max"),
    "stp_fixed":     dict(cmap="Reds", levels=np.arange(0, 11, 1), extend="max"),
    "stp_effective": dict(cmap="Reds", levels=np.arange(0, 11, 1), extend="max"),
    "scp":           dict(cmap="Reds", levels=np.arange(0, 11, 1), extend="max"),
    "ehi":           dict(cmap="Reds", levels=np.arange(0, 5.5, 0.5), extend="max"),
    "ship":          dict(cmap="YlOrRd", levels=np.arange(0, 5.5, 0.5), extend="max"),
    "bri":           dict(cmap="YlOrRd", levels=np.arange(0, 110, 10), extend="max"),
    "critical_angle": dict(cmap="RdYlBu_r", levels=np.arange(0, 200, 10), extend="max"),

    # ---- Temperature ----
    "temp":   dict(cmap="Spectral_r", extend="both"),
    "tc":     dict(cmap="RdBu_r", center_zero=True, extend="both"),
    "theta":  dict(cmap="Spectral_r", extend="both"),
    "theta_e": dict(cmap="Spectral_r", extend="both"),
    "tv":     dict(cmap="Spectral_r", extend="both"),
    "twb":    dict(cmap="Spectral_r", extend="both"),
    "td":     dict(cmap="RdYlGn", extend="both"),
    "dp2m":   dict(cmap="RdYlGn", levels=np.arange(-20, 82, 2), extend="both"),
    "theta_w": dict(cmap="Spectral_r", extend="both"),

    # ---- Moisture ----
    "rh":     dict(cmap="YlGnBu", levels=np.arange(0, 105, 5), extend="neither"),
    "rh2m":   dict(cmap="YlGnBu", levels=np.arange(0, 105, 5), extend="neither"),
    "pw":     dict(cmap="YlGnBu", levels=np.arange(0, 66, 3), extend="max"),
    "mixing_ratio": dict(cmap="YlGnBu", extend="max"),
    "specific_humidity": dict(cmap="YlGnBu", extend="max"),

    # ---- Pressure ----
    "slp":    dict(cmap="viridis", levels=np.arange(980, 1042, 2), extend="both"),
    "pressure": dict(cmap="viridis", extend="both"),

    # ---- Wind ----
    "wspd":   dict(cmap="YlGnBu", levels=np.arange(0, 42, 2), extend="max"),
    "wspd10": dict(cmap="YlGnBu", levels=np.arange(0, 42, 2), extend="max"),
    "wdir":   dict(cmap="hsv", levels=np.arange(0, 370, 10), extend="neither"),
    "wdir10": dict(cmap="hsv", levels=np.arange(0, 370, 10), extend="neither"),

    # ---- Vorticity ----
    "avo":    dict(cmap="RdBu_r", center_zero=True, extend="both"),
    "pvo":    dict(cmap="RdBu_r", center_zero=True, extend="both"),

    # ---- Radar ----
    "dbz":    dict(cmap="_nws_reflectivity", levels=np.arange(5, 71, 2.5), extend="max", mask_below=5.0),
    "maxdbz": dict(cmap="_nws_reflectivity", levels=np.arange(5, 71, 2.5), extend="max", mask_below=5.0),

    # ---- Cloud ----
    "ctt":       dict(cmap="Greys_r", levels=np.arange(-80, 22, 2), extend="both"),
    "cloudfrac": dict(cmap="Greys", levels=np.arange(0, 105, 5), extend="neither"),

    # ---- Heights / Terrain ----
    "height":     dict(cmap="terrain", extend="both"),
    "height_agl": dict(cmap="terrain", extend="both"),
    "terrain":    dict(cmap="terrain", levels=np.arange(0, 4200, 200), extend="max"),
    "geopt":      dict(cmap="terrain", extend="both"),
    "freezing_level": dict(cmap="cool", levels=np.arange(0, 5500, 250), extend="max"),
    "wet_bulb_0":     dict(cmap="cool", levels=np.arange(0, 5500, 250), extend="max"),

    # ---- Lapse rates ----
    "lapse_rate_700_500": dict(cmap="RdYlBu_r", levels=np.arange(4, 10.5, 0.5), extend="both"),
    "lapse_rate_0_3km":   dict(cmap="RdYlBu_r", levels=np.arange(4, 10.5, 0.5), extend="both"),
    "lapse_rate":         dict(cmap="RdYlBu_r", levels=np.arange(4, 10.5, 0.5), extend="both"),

    # ---- Fire weather ----
    "fosberg": dict(cmap="YlOrRd", levels=np.arange(0, 80, 5), extend="max"),
    "haines":  dict(cmap="YlOrRd", levels=np.arange(2, 7, 1), extend="neither"),
    "hdw":     dict(cmap="YlOrRd", extend="max"),

    # ---- Updraft helicity ----
    "uhel":    dict(cmap="RdPu", levels=np.arange(0, 210, 10), extend="max"),

    # ---- Wind components ----
    "ua":      dict(cmap="RdBu_r", center_zero=True, extend="both"),
    "va":      dict(cmap="RdBu_r", center_zero=True, extend="both"),
    "wa":      dict(cmap="RdBu_r", center_zero=True, extend="both"),
    "omega":   dict(cmap="RdBu_r", center_zero=True, extend="both"),

    # ---- Effective inflow ----
    "effective_inflow": dict(cmap="YlOrRd", extend="max"),
}


def _get_style(varname: str) -> Dict[str, Any]:
    """Look up the styling dict for a variable, with fallbacks."""
    key = varname.lower()
    if key in _VAR_STYLES:
        return dict(_VAR_STYLES[key])  # copy

    # Substring matching for aliases
    for pattern, style in _VAR_STYLES.items():
        if pattern in key or key in pattern:
            return dict(style)

    # Ultimate fallback
    return dict(cmap="viridis", extend="both")


# ---------------------------------------------------------------------------
# NWS-style reflectivity colormap
# ---------------------------------------------------------------------------

def _make_nws_reflectivity_cmap():
    """Build a discrete colormap that mimics NWS reflectivity displays."""
    _, plt = _import_mpl()
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # (dbz_lower_bound, R, G, B) -- 16 bins from -10 to 75 dBZ
    _colors = [
        (0.40, 0.40, 0.40),   # -10 to -5  grey
        (0.60, 0.60, 0.60),   # -5  to  0  light grey
        (0.00, 0.93, 0.93),   #  0  to  5  light cyan
        (0.00, 0.63, 0.93),   #  5  to 10  medium cyan
        (0.00, 0.00, 0.93),   # 10  to 15  blue
        (0.00, 0.93, 0.00),   # 15  to 20  bright green
        (0.00, 0.73, 0.00),   # 20  to 25  green
        (0.00, 0.53, 0.00),   # 25  to 30  dark green
        (0.93, 0.93, 0.00),   # 30  to 35  yellow
        (0.87, 0.67, 0.00),   # 35  to 40  gold
        (0.93, 0.33, 0.00),   # 40  to 45  orange
        (0.93, 0.00, 0.00),   # 45  to 50  red
        (0.73, 0.00, 0.00),   # 50  to 55  dark red
        (0.93, 0.00, 0.93),   # 55  to 60  magenta
        (0.60, 0.33, 0.73),   # 60  to 65  purple
        (0.40, 0.20, 0.53),   # 65  to 70  dark purple
        (1.00, 1.00, 1.00),   # 70  to 75  white
    ]
    cmap = ListedColormap(_colors, name="NWSReflectivity")
    bounds = list(range(-10, 80, 5))
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _resolve_cmap(name: str, data: np.ndarray):
    """Return (cmap, norm_or_None).  Handles the special _nws_reflectivity tag."""
    _, plt = _import_mpl()

    if name == "_nws_reflectivity":
        return _make_nws_reflectivity_cmap()

    try:
        cmap = plt.get_cmap(name)
    except ValueError:
        cmap = plt.get_cmap("viridis")

    return cmap, None


# ---------------------------------------------------------------------------
# Smart auto-levels
# ---------------------------------------------------------------------------

def _auto_levels(data: np.ndarray, n: int = 15, center_zero: bool = False):
    """Pick nice contour levels from the data range."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.linspace(0, 1, n)

    vmin, vmax = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))

    if center_zero:
        absmax = max(abs(vmin), abs(vmax))
        if absmax == 0:
            absmax = 1.0
        return np.linspace(-absmax, absmax, n)

    if vmin == vmax:
        return np.linspace(vmin - 1, vmax + 1, n)

    # Nice step size
    raw_step = (vmax - vmin) / n
    mag = 10 ** np.floor(np.log10(max(abs(raw_step), 1e-30)))
    nice_steps = np.array([1, 2, 2.5, 5, 10])
    step = mag * nice_steps[np.argmin(np.abs(nice_steps * mag - raw_step))]
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    levels = np.arange(lo, hi + step * 0.5, step)
    if len(levels) < 3:
        levels = np.linspace(vmin, vmax, n)
    return levels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_time_label(wrffile, timeidx: int) -> str:
    """Get a human-readable time label like '2024-05-20 18:00 UTC'."""
    try:
        times = wrffile.times()
        if timeidx < len(times):
            t = times[timeidx]
            # WRF format: 2024-05-20_18:00:00
            return t.replace("_", " ") + " UTC"
    except Exception:
        pass
    return f"t={timeidx}"


def _get_var_description(varname: str) -> str:
    """Attempt to get a human-readable variable description."""
    try:
        from wrf import list_variables
        for v in list_variables():
            if v["name"] == varname.lower():
                return v["description"]
    except Exception:
        pass
    return varname


def _get_var_units(varname: str) -> str:
    """Get the default units for a variable."""
    try:
        from wrf import list_variables
        for v in list_variables():
            if v["name"] == varname.lower():
                return v["units"]
    except Exception:
        pass
    return ""


def _find_nearest_ij(lat2d: np.ndarray, lon2d: np.ndarray,
                     target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find the (i, j) grid indices nearest to a lat/lon point."""
    dist = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return int(idx[0]), int(idx[1])


def _thin_mask(ny: int, nx: int, stride: int):
    """Boolean mask selecting every stride-th point in both dimensions."""
    mask = np.zeros((ny, nx), dtype=bool)
    mask[::stride, ::stride] = True
    return mask


# ---------------------------------------------------------------------------
# Public API: plot_field
# ---------------------------------------------------------------------------

def plot_field(
    wrffile,
    varname: str,
    timeidx: int = 0,
    *,
    style: Optional[str] = None,
    cmap: Optional[str] = None,
    levels: Optional[np.ndarray] = None,
    units: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 9),
    projection=None,
    ax=None,
    colorbar: bool = True,
    contour_lines: bool = False,
    contour_line_var: Optional[str] = None,
    contour_line_levels: Optional[np.ndarray] = None,
    contour_line_color: str = "k",
    contour_line_kwargs: Optional[dict] = None,
    borders: bool = True,
    gridlines: bool = False,
    extend: Optional[str] = None,
    **getvar_kwargs,
):
    """Plot a 2-D WRF field on a map.

    This is the workhorse function.  It fetches the variable, picks a
    smart colormap and contour levels, draws coastlines and borders,
    and adds a colorbar -- all with zero configuration.

    Parameters
    ----------
    wrffile : WrfFile, str, or Dataset
        WRF output file.
    varname : str
        Variable name (e.g. "sbcape", "slp", "maxdbz").
    timeidx : int
        Time index.
    style : str, optional
        Styling preset.  ``"solar7"`` uses the Solarpower07 colormaps
        and contour levels from ``wrf.solar7``.  Overridden by explicit
        *cmap* / *levels* / *extend* arguments.
    cmap : str, optional
        Matplotlib colormap name.  Overrides the automatic pick.
    levels : array-like, optional
        Contour levels.  Overrides automatic levels.
    units : str, optional
        Unit conversion (e.g. "knots", "degF").  Passed to getvar().
    title : str, optional
        Custom title.  Default: "{description} -- {time}".
    figsize : tuple
        Figure size in inches.
    projection : cartopy CRS, optional
        Map projection.  Default: LambertConformal centered on the domain.
    ax : matplotlib Axes, optional
        Existing axes to draw on (skips figure creation).
    colorbar : bool
        Whether to add a colorbar.
    contour_lines : bool
        Overlay contour lines on top of the filled contours.
    contour_line_var : str, optional
        Different variable for contour lines (e.g. "slp" lines over a
        temperature fill).
    contour_line_levels : array-like, optional
        Levels for the contour lines.
    contour_line_color : str
        Color for contour lines.
    contour_line_kwargs : dict, optional
        Extra kwargs passed to ax.contour().
    borders : bool
        Draw state/country borders and coastlines.
    gridlines : bool
        Draw lat/lon gridlines.
    extend : str, optional
        Colorbar extend mode.
    **getvar_kwargs
        Passed through to getvar() (e.g. parcel_type, layer_type).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes (may be a GeoAxes)
    """
    _, plt = _import_mpl()
    ccrs, cfeature = _import_cartopy()

    wf = _ensure_wrffile(wrffile)
    from wrf import getvar

    # Fetch data
    gv_kwargs = dict(getvar_kwargs)
    if units is not None:
        gv_kwargs["units"] = units
    data = getvar(wf, varname, timeidx=timeidx, **gv_kwargs)

    # If 3-D, take the lowest level for a sensible 2-D plot
    if data.ndim == 3:
        data = data[0]

    lat = getvar(wf, "lat", timeidx=timeidx)
    lon = getvar(wf, "lon", timeidx=timeidx)

    # -- Style resolution --
    if style == "solar7":
        from wrf.solar7 import SOLAR7_STYLES
        _key = varname.lower()
        _s7 = SOLAR7_STYLES.get(_key, {})
        # Solar7 style is the base; explicit args override it
        style_dict = {**_get_style(varname), **_s7}
    else:
        style_dict = _get_style(varname)
    use_cmap = cmap or style_dict.get("cmap", "viridis")
    use_extend = extend or style_dict.get("extend", "both")
    center_zero = style_dict.get("center_zero", False)

    if levels is not None:
        use_levels = np.asarray(levels)
    elif "levels" in style_dict and style_dict["levels"] is not None:
        use_levels = style_dict["levels"]
    else:
        use_levels = _auto_levels(data, center_zero=center_zero)

    cmap_obj, norm = _resolve_cmap(use_cmap, data)

    # -- Determine units label --
    if units is not None:
        units_label = units
    else:
        units_label = _get_var_units(varname)

    # -- Figure / axes creation --
    created_fig = ax is None
    if ax is None:
        if ccrs is not None:
            if projection is None:
                # Center a Lambert Conformal on the domain
                clon = float(np.nanmean(lon))
                clat = float(np.nanmean(lat))
                projection = ccrs.LambertConformal(
                    central_longitude=clon, central_latitude=clat
                )
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={"projection": projection},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # -- Filled contour --
    is_geo = ccrs is not None and hasattr(ax, "projection")
    plot_kwargs = dict(cmap=cmap_obj, extend=use_extend)
    if norm is not None:
        plot_kwargs["norm"] = norm
    else:
        plot_kwargs["levels"] = use_levels

    transform = ccrs.PlateCarree() if is_geo else None
    if transform is not None:
        plot_kwargs["transform"] = transform

    # Mask values below threshold (e.g., reflectivity < 5 dBZ)
    mask_below = style_dict.get("mask_below")
    if mask_below is not None:
        data = np.where(data >= mask_below, data, np.nan)

    cf = ax.contourf(lon, lat, data, **plot_kwargs)

    # -- Contour lines on top --
    if contour_lines or contour_line_var is not None:
        cl_data = data
        if contour_line_var is not None:
            cl_data = getvar(wf, contour_line_var, timeidx=timeidx)
            if cl_data.ndim == 3:
                cl_data = cl_data[0]
        cl_levels = contour_line_levels
        if cl_levels is None:
            cl_style = _get_style(contour_line_var or varname)
            cl_levels = cl_style.get("levels")
            if cl_levels is None:
                cl_levels = _auto_levels(cl_data, n=10)
        cl_kw = dict(colors=contour_line_color, linewidths=0.8)
        if transform is not None:
            cl_kw["transform"] = transform
        if contour_line_kwargs:
            cl_kw.update(contour_line_kwargs)
        cs = ax.contour(lon, lat, cl_data, levels=cl_levels, **cl_kw)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%g")

    # -- Map decorations --
    if is_geo and borders:
        ax.coastlines(linewidth=1.0, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="#333333")
        try:
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="#555555")
        except Exception:
            pass

    if is_geo and gridlines:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, alpha=0.3,
            linestyle=":", color="#999999",
            x_inline=False, y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#555555"}
        gl.ylabel_style = {"size": 8, "color": "#555555"}

    # Auto-zoom to domain extent
    if is_geo:
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        pad_lon = (lon_max - lon_min) * 0.02
        pad_lat = (lat_max - lat_min) * 0.02
        ax.set_extent(
            [lon_min - pad_lon, lon_max + pad_lon,
             lat_min - pad_lat, lat_max + pad_lat],
            crs=transform,
        )

    # -- Colorbar --
    if colorbar:
        desc = _get_var_description(varname)
        cbar_label = f"{desc} ({units_label})" if units_label else desc
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal",
                          pad=0.06, shrink=0.8, aspect=40,
                          extendrect=True)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9)

    # -- Title --
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        desc = _get_var_description(varname)
        time_label = _get_time_label(wf, timeidx)
        ax.set_title(f"{desc}\n{time_label}", fontsize=13, fontweight="bold")

    if not is_geo:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Public API: plot_wind
# ---------------------------------------------------------------------------

def plot_wind(
    wrffile,
    timeidx: int = 0,
    *,
    background: str = "wspd10",
    barb_type: str = "barbs",
    thin: Optional[int] = None,
    barb_length: float = 6,
    barb_color: str = "k",
    figsize: Tuple[float, float] = (12, 9),
    projection=None,
    ax=None,
    cmap: Optional[str] = None,
    levels: Optional[np.ndarray] = None,
    units: Optional[str] = None,
    title: Optional[str] = None,
    borders: bool = True,
    gridlines: bool = False,
    colorbar: bool = True,
    **getvar_kwargs,
):
    """Plot wind barbs or vectors over a filled background field.

    Parameters
    ----------
    wrffile : WrfFile, str, or Dataset
        WRF output file.
    timeidx : int
        Time index.
    background : str
        Variable for the filled contour behind the barbs.
        Default: "wspd10".
    barb_type : str
        "barbs" for wind barbs, "quiver" for arrows.
    thin : int, optional
        Plot every Nth grid point.  Default: auto-calculated so barbs
        don't overlap.
    barb_length : float
        Size of wind barbs.
    barb_color : str
        Color of the barbs/arrows.
    figsize, projection, ax, cmap, levels, units, title, borders, gridlines
        Same as plot_field().
    **getvar_kwargs
        Passed to getvar() for the background field.

    Returns
    -------
    fig, ax
    """
    _, plt = _import_mpl()
    ccrs, cfeature = _import_cartopy()

    wf = _ensure_wrffile(wrffile)
    from wrf import getvar

    # -- Background fill --
    fig, ax = plot_field(
        wf, background, timeidx=timeidx,
        cmap=cmap, levels=levels, units=units,
        figsize=figsize, projection=projection, ax=ax,
        title=title or None, borders=borders, gridlines=gridlines,
        colorbar=colorbar, **getvar_kwargs,
    )

    # -- Get earth-rotated winds --
    # uvmet10 returns (2, ny, nx): [0]=U, [1]=V
    try:
        uvmet = getvar(wf, "uvmet10", timeidx=timeidx)
        u = uvmet[0]
        v = uvmet[1]
    except Exception:
        # Fallback: use raw destaggered winds at lowest level
        u = getvar(wf, "ua", timeidx=timeidx)
        v = getvar(wf, "va", timeidx=timeidx)
        if u.ndim == 3:
            u = u[0]
            v = v[0]

    lat = getvar(wf, "lat", timeidx=timeidx)
    lon = getvar(wf, "lon", timeidx=timeidx)

    # Convert to knots for proper barb flags
    u_kt = u * 1.94384
    v_kt = v * 1.94384

    # Auto-thin based on grid size
    if thin is None:
        ny, nx = lat.shape
        thin = max(1, min(ny, nx) // 30)

    mask = _thin_mask(*lat.shape, thin)
    is_geo = ccrs is not None and hasattr(ax, "projection")
    transform = ccrs.PlateCarree() if is_geo else None

    barb_kw = {}
    if transform is not None:
        barb_kw["transform"] = transform

    if barb_type == "barbs":
        ax.barbs(
            lon[mask], lat[mask], u_kt[mask], v_kt[mask],
            length=barb_length, color=barb_color,
            linewidth=0.5, **barb_kw,
        )
    else:
        ax.quiver(
            lon[mask], lat[mask], u[mask], v[mask],
            color=barb_color, scale=500, width=0.002,
            **barb_kw,
        )

    # Update title if not custom
    if title is None:
        time_label = _get_time_label(wf, timeidx)
        bg_desc = _get_var_description(background)
        ax.set_title(
            f"{bg_desc} + Wind Barbs\n{time_label}",
            fontsize=13, fontweight="bold",
        )

    return fig, ax


# ---------------------------------------------------------------------------
# Public API: plot_skewt
# ---------------------------------------------------------------------------

def plot_skewt(
    wrffile,
    timeidx: int = 0,
    *,
    point: Optional[Tuple[float, float]] = None,
    ij: Optional[Tuple[int, int]] = None,
    figsize: Tuple[float, float] = (9, 10),
    hodograph: bool = True,
    hodograph_height: float = 12000.0,
    parcel_type: str = "sb",
    title: Optional[str] = None,
    **getvar_kwargs,
):
    """Plot a Skew-T log-P sounding for a single column.

    Uses MetPy's SkewT if available, otherwise falls back to a
    manual Skew-T using plain matplotlib.

    Parameters
    ----------
    wrffile : WrfFile, str, or Dataset
    timeidx : int
    point : (lat, lon), optional
        Geographic location.  Finds nearest grid point.
    ij : (i, j), optional
        Direct grid indices (south_north, west_east).  One of
        ``point`` or ``ij`` is required.
    figsize : tuple
    hodograph : bool
        Add a hodograph inset.
    hodograph_height : float
        Top height (m AGL) for the hodograph.
    parcel_type : str
        Parcel for CAPE annotation ("sb", "ml", "mu").
    title : str, optional
    **getvar_kwargs
        Passed through to getvar().

    Returns
    -------
    fig, ax_skewt
    """
    _, plt = _import_mpl()

    wf = _ensure_wrffile(wrffile)
    from wrf import getvar

    if point is None and ij is None:
        raise ValueError("Provide either point=(lat, lon) or ij=(i, j)")

    lat2d = getvar(wf, "lat", timeidx=timeidx)
    lon2d = getvar(wf, "lon", timeidx=timeidx)

    if point is not None:
        i, j = _find_nearest_ij(lat2d, lon2d, point[0], point[1])
    else:
        i, j = ij

    actual_lat = float(lat2d[i, j])
    actual_lon = float(lon2d[i, j])

    # -- Extract column profiles --
    # Pressure (Pa -> hPa)
    pres_3d = getvar(wf, "pressure", timeidx=timeidx, units="hPa")
    pres = pres_3d[:, i, j]

    # Temperature (K -> degC)
    temp_3d = getvar(wf, "tc", timeidx=timeidx)
    temp = temp_3d[:, i, j]

    # Dewpoint (degC)
    td_3d = getvar(wf, "td", timeidx=timeidx)
    td = td_3d[:, i, j]

    # Height AGL (m)
    z_3d = getvar(wf, "height_agl", timeidx=timeidx)
    z = z_3d[:, i, j]

    # Wind (earth-rotated, m/s -> knots)
    try:
        uvmet_3d = getvar(wf, "uvmet", timeidx=timeidx)
        u = uvmet_3d[0, :, i, j] * 1.94384
        v = uvmet_3d[1, :, i, j] * 1.94384
    except Exception:
        u_3d = getvar(wf, "ua", timeidx=timeidx)
        v_3d = getvar(wf, "va", timeidx=timeidx)
        u = u_3d[:, i, j] * 1.94384
        v = v_3d[:, i, j] * 1.94384

    # -- Sounding parameters --
    gv_cape_kwargs = {k: v for k, v in getvar_kwargs.items()}
    try:
        cape_name = f"{parcel_type}cape"
        cin_name = f"{parcel_type}cin"
        cape_val = float(getvar(wf, cape_name, timeidx=timeidx)[i, j])
        cin_val = float(getvar(wf, cin_name, timeidx=timeidx)[i, j])
    except Exception:
        cape_val = np.nan
        cin_val = np.nan

    try:
        lcl_val = float(getvar(wf, "lcl", timeidx=timeidx, **gv_cape_kwargs)[i, j])
    except Exception:
        lcl_val = np.nan

    try:
        lfc_val = float(getvar(wf, "lfc", timeidx=timeidx, **gv_cape_kwargs)[i, j])
    except Exception:
        lfc_val = np.nan

    try:
        el_val = float(getvar(wf, "el", timeidx=timeidx, **gv_cape_kwargs)[i, j])
    except Exception:
        el_val = np.nan

    # -- Attempt MetPy SkewT --
    SkewT_cls = _import_metpy()

    if SkewT_cls is not None:
        return _plot_skewt_metpy(
            plt, SkewT_cls, pres, temp, td, u, v, z,
            cape_val, cin_val, lcl_val, lfc_val, el_val,
            actual_lat, actual_lon, wf, timeidx,
            figsize, hodograph, hodograph_height, parcel_type, title,
        )
    else:
        return _plot_skewt_manual(
            plt, pres, temp, td, u, v, z,
            cape_val, cin_val, lcl_val, lfc_val, el_val,
            actual_lat, actual_lon, wf, timeidx,
            figsize, hodograph, hodograph_height, parcel_type, title,
        )


def _skewt_annotations(ax, cape_val, cin_val, lcl_val, lfc_val, el_val,
                        parcel_type, x_pos=0.02, y_start=0.98):
    """Add CAPE/CIN/LCL/LFC/EL text annotations to the sounding."""
    lines = []
    pt = parcel_type.upper()
    if np.isfinite(cape_val):
        lines.append(f"{pt}CAPE: {cape_val:.0f} J/kg")
    if np.isfinite(cin_val):
        lines.append(f"{pt}CIN:  {cin_val:.0f} J/kg")
    if np.isfinite(lcl_val):
        lines.append(f"LCL:  {lcl_val:.0f} m AGL")
    if np.isfinite(lfc_val):
        lines.append(f"LFC:  {lfc_val:.0f} m AGL")
    if np.isfinite(el_val):
        lines.append(f"EL:   {el_val:.0f} m AGL")

    if lines:
        text = "\n".join(lines)
        ax.text(
            x_pos, y_start, text,
            transform=ax.transAxes, fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="grey"),
        )


def _plot_skewt_metpy(
    plt, SkewT_cls, pres, temp, td, u, v, z,
    cape_val, cin_val, lcl_val, lfc_val, el_val,
    lat, lon, wf, timeidx,
    figsize, hodograph, hodograph_height, parcel_type, title,
):
    """Skew-T via MetPy."""
    from metpy.units import units as mu

    fig = plt.figure(figsize=figsize)
    skew = SkewT_cls(fig, rotation=45)

    # Attach units for MetPy
    p_u = pres * mu.hPa
    T_u = temp * mu.degC
    Td_u = td * mu.degC

    skew.plot(p_u, T_u, "r", linewidth=1.5, label="Temperature")
    skew.plot(p_u, Td_u, "g", linewidth=1.5, label="Dewpoint")

    # Wind barbs (thin to ~40 levels max)
    nz = len(pres)
    stride = max(1, nz // 40)
    skew.plot_barbs(p_u[::stride], u[::stride] * mu.knots, v[::stride] * mu.knots)

    # Standard atmosphere references
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 50)

    try:
        skew.plot_dry_adiabats(linewidth=0.5, alpha=0.3)
        skew.plot_moist_adiabats(linewidth=0.5, alpha=0.3)
        skew.plot_mixing_lines(linewidth=0.5, alpha=0.3)
    except Exception:
        pass

    # Annotations
    _skewt_annotations(skew.ax, cape_val, cin_val, lcl_val, lfc_val, el_val,
                       parcel_type)

    # Hodograph inset
    if hodograph:
        try:
            from metpy.plots import Hodograph
            hodo_ax = fig.add_axes([0.65, 0.65, 0.28, 0.28])
            h = Hodograph(hodo_ax, component_range=40)
            h.add_grid(increment=10)
            mask_z = z <= hodograph_height
            # Color segments by height
            _plot_hodograph_segments(h, u[mask_z], v[mask_z], z[mask_z])
        except Exception:
            pass

    # Title
    if title is not None:
        skew.ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        time_label = _get_time_label(wf, timeidx)
        skew.ax.set_title(
            f"Sounding at ({lat:.2f}, {lon:.2f})\n{time_label}",
            fontsize=13, fontweight="bold",
        )

    skew.ax.legend(loc="upper left", fontsize=8)
    return fig, skew.ax


def _plot_hodograph_segments(h, u, v, z):
    """Color hodograph by height: 0-1 red, 1-3 green, 3-6 gold, 6+ blue."""
    colors = [
        (0, 1000, "red", "0-1 km"),
        (1000, 3000, "green", "1-3 km"),
        (3000, 6000, "goldenrod", "3-6 km"),
        (6000, 99999, "royalblue", "6+ km"),
    ]
    for zbot, ztop, color, label in colors:
        mask = (z >= zbot) & (z <= ztop)
        if np.sum(mask) > 1:
            h.ax.plot(u[mask], v[mask], color=color, linewidth=1.8, label=label)

    h.ax.legend(fontsize=6, loc="lower left")


def _plot_skewt_manual(
    plt, pres, temp, td, u, v, z,
    cape_val, cin_val, lcl_val, lfc_val, el_val,
    lat, lon, wf, timeidx,
    figsize, hodograph, hodograph_height, parcel_type, title,
):
    """Fallback Skew-T using plain matplotlib (no MetPy)."""
    from matplotlib.ticker import ScalarFormatter

    fig = plt.figure(figsize=figsize)

    if hodograph:
        ax = fig.add_axes([0.08, 0.08, 0.82, 0.82])
    else:
        ax = fig.add_subplot(111)

    # Skew-T: plot temperature vs log(pressure) with a skewing transform.
    # We implement skewing manually: x_plot = T + skew * log(p0/p)
    skew_factor = 35.0
    p_ref = 1000.0

    def skew_x(T, p):
        return T + skew_factor * np.log10(p_ref / p)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([1000, 850, 700, 500, 400, 300, 200, 100])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(1050, 100)
    ax.invert_yaxis()

    # Plot temperature and dewpoint
    ax.plot(skew_x(temp, pres), pres, "r", linewidth=1.5, label="Temperature")
    ax.plot(skew_x(td, pres), pres, "g", linewidth=1.5, label="Dewpoint")

    # Draw some reference lines: dry adiabats, isotherms
    for T0 in range(-40, 50, 10):
        p_range = np.linspace(100, 1050, 100)
        ax.plot(skew_x(np.full_like(p_range, T0), p_range), p_range,
                color="tan", linewidth=0.3, alpha=0.5)

    ax.set_xlabel("Temperature (degC)", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    # Annotations
    _skewt_annotations(ax, cape_val, cin_val, lcl_val, lfc_val, el_val,
                       parcel_type, x_pos=0.02, y_start=0.35)

    # Hodograph inset
    if hodograph:
        hodo_ax = fig.add_axes([0.65, 0.65, 0.28, 0.28])
        mask_z = z <= hodograph_height
        _plot_hodograph_manual(hodo_ax, u[mask_z], v[mask_z], z[mask_z])

    # Title
    if title is not None:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        time_label = _get_time_label(wf, timeidx)
        ax.set_title(
            f"Sounding at ({lat:.2f}, {lon:.2f})\n{time_label}",
            fontsize=13, fontweight="bold",
        )

    return fig, ax


def _plot_hodograph_manual(ax, u, v, z):
    """Simple hodograph inset using plain matplotlib."""
    colors = [
        (0, 1000, "red", "0-1 km"),
        (1000, 3000, "green", "1-3 km"),
        (3000, 6000, "goldenrod", "3-6 km"),
        (6000, 99999, "royalblue", "6+ km"),
    ]
    for zbot, ztop, color, label in colors:
        mask = (z >= zbot) & (z <= ztop)
        if np.sum(mask) > 1:
            ax.plot(u[mask], v[mask], color=color, linewidth=1.8, label=label)

    # Concentric range rings
    for r in [10, 20, 30, 40]:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="grey",
                linewidth=0.3, alpha=0.5)

    ax.axhline(0, color="grey", linewidth=0.3)
    ax.axvline(0, color="grey", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("U (kt)", fontsize=7)
    ax.set_ylabel("V (kt)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="lower left")
    ax.set_title("Hodograph", fontsize=8)


# ---------------------------------------------------------------------------
# Public API: panel
# ---------------------------------------------------------------------------

def panel(
    wrffile,
    varnames: Sequence[str],
    timeidx: int = 0,
    *,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    projection=None,
    suptitle: Optional[str] = None,
    borders: bool = True,
    gridlines: bool = False,
    **getvar_kwargs,
):
    """Create a multi-panel plot of several variables side by side.

    Parameters
    ----------
    wrffile : WrfFile, str, or Dataset
    varnames : list of str
        Variable names to plot, one per panel.
    timeidx : int
    ncols : int, optional
        Number of columns.  Default: auto (2 for <=4 vars, 3 otherwise).
    figsize : tuple, optional
        Overall figure size.  Default: auto-scaled.
    projection : cartopy CRS, optional
    suptitle : str, optional
        Overall figure title.
    borders, gridlines : bool
    **getvar_kwargs
        Passed to getvar() for every panel.

    Returns
    -------
    fig, axes  (axes is a flat list)
    """
    _, plt = _import_mpl()
    ccrs, cfeature = _import_cartopy()

    wf = _ensure_wrffile(wrffile)
    from wrf import getvar

    n = len(varnames)
    if n == 0:
        raise ValueError("varnames must contain at least one variable")

    if ncols is None:
        ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    # Determine projection once (shared across panels)
    if projection is None and ccrs is not None:
        lat = getvar(wf, "lat", timeidx=timeidx)
        lon = getvar(wf, "lon", timeidx=timeidx)
        clon = float(np.nanmean(lon))
        clat = float(np.nanmean(lat))
        projection = ccrs.LambertConformal(
            central_longitude=clon, central_latitude=clat
        )

    subplot_kw = {"projection": projection} if projection is not None else {}
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=figsize, subplot_kw=subplot_kw,
    )

    # Flatten axes for easy indexing
    if n == 1:
        axes_flat = [axes_grid]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(np.atleast_1d(axes_grid).flat)
    else:
        axes_flat = list(axes_grid.flat)

    for idx, varname in enumerate(varnames):
        plot_field(
            wf, varname, timeidx=timeidx,
            ax=axes_flat[idx], borders=borders, gridlines=gridlines,
            **getvar_kwargs,
        )

    # Hide unused panels
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.01)
    else:
        time_label = _get_time_label(wf, timeidx)
        fig.suptitle(time_label, fontsize=14, y=1.01)

    fig.tight_layout()
    return fig, axes_flat[:n]


# ---------------------------------------------------------------------------
# Convenience shortcuts
# ---------------------------------------------------------------------------

def show():
    """Show all open figures (shortcut for plt.show())."""
    _, plt = _import_mpl()
    plt.show()


def savefig(fig, path: str, dpi: int = 150, **kwargs):
    """Save a figure with good defaults for met imagery."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", **kwargs)


# ---------------------------------------------------------------------------
# Multi-timestep rendering + GIF
# ---------------------------------------------------------------------------

def render_timesteps(
    wrffile,
    varname: str,
    timesteps=None,
    outdir: str = ".",
    prefix: str = "",
    dpi: int = 150,
    gif: bool = False,
    gif_path: str | None = None,
    gif_fps: int = 4,
    fixed_scale: bool = True,
    progress_callback=None,
    **getvar_kwargs,
):
    """Render a variable across multiple timesteps with consistent scale.

    Parameters
    ----------
    wrffile : WrfFile or str
        The WRF file.
    varname : str
        Variable name.
    timesteps : list of int, or None
        Which timesteps to render. None = all.
    outdir : str
        Directory for output PNGs.
    prefix : str
        Filename prefix (default: varname).
    dpi : int
        Image resolution.
    gif : bool
        If True, also create an animated GIF from the frames.
    gif_path : str
        Path for the GIF file. Default: outdir/varname.gif.
    gif_fps : int
        Frames per second for the GIF.
    fixed_scale : bool
        If True (default), compute min/max across ALL timesteps first
        and use the same colorbar range for every frame.
    progress_callback : callable(current, total, varname) or None
        Called after each frame for progress reporting.
    **getvar_kwargs
        Passed to getvar (units, parcel_type, etc).

    Returns
    -------
    list of str
        Paths to the generated PNG files.
    """
    _, plt = _import_mpl()
    wf = _ensure_wrffile(wrffile)

    if timesteps is None:
        timesteps = list(range(wf.nt))

    from wrf import getvar

    gv_kwargs = dict(getvar_kwargs)
    if not prefix:
        prefix = varname

    # ── Pass 1: compute global scale if fixed_scale ──
    global_vmin = None
    global_vmax = None
    if fixed_scale and len(timesteps) > 1:
        all_mins = []
        all_maxs = []
        for t in timesteps:
            data = getvar(wf, varname, timeidx=t, **gv_kwargs)
            if data.ndim == 3:
                data = data[0]
            valid = data[np.isfinite(data)]
            if len(valid) > 0:
                all_mins.append(float(np.percentile(valid, 2)))
                all_maxs.append(float(np.percentile(valid, 98)))
        if all_mins:
            global_vmin = min(all_mins)
            global_vmax = max(all_maxs)

    # ── Pass 2: render each frame ──
    style = _get_style(varname)
    fixed_levels = None
    if global_vmin is not None and global_vmax is not None:
        # Build levels from global range
        if "levels" in style and style["levels"] is not None:
            fixed_levels = style["levels"]
        else:
            fixed_levels = np.linspace(global_vmin, global_vmax, 20)

    os.makedirs(outdir, exist_ok=True)
    png_paths = []

    for i, t in enumerate(timesteps):
        kwargs = dict(getvar_kwargs)
        if fixed_levels is not None:
            fig, ax = plot_field(wf, varname, timeidx=t, levels=fixed_levels, **kwargs)
        else:
            fig, ax = plot_field(wf, varname, timeidx=t, **kwargs)

        fname = f"{prefix}_t{t:04d}.png"
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        png_paths.append(fpath)

        if progress_callback:
            progress_callback(i + 1, len(timesteps), varname)

    # ── GIF ──
    if gif and len(png_paths) > 1:
        _make_gif(png_paths, gif_path or os.path.join(outdir, f"{prefix}.gif"), gif_fps)

    return png_paths


def _make_gif(png_paths: list[str], gif_path: str, fps: int = 4):
    """Create an animated GIF from a list of PNG files."""
    try:
        from PIL import Image
        frames = []
        for p in png_paths:
            img = Image.open(p).convert("RGBA")
            # Convert to RGB (GIF doesn't support alpha)
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            frames.append(bg.convert("RGB"))
        if frames:
            duration = int(1000 / fps)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )
    except ImportError:
        # Fallback: use imageio if available
        try:
            import imageio.v3 as iio
            images = [iio.imread(p) for p in png_paths]
            iio.imwrite(gif_path, images, duration=1000 // fps, loop=0)
        except ImportError:
            raise ImportError(
                "GIF generation requires either Pillow (pip install Pillow) "
                "or imageio (pip install imageio)"
            )
