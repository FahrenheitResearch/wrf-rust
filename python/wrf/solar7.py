"""
wrf.solar7 -- Solarpower07-style colormaps and product presets.

Ports every colormap and all 30+ product definitions from the Solarpower07
(wrf-solar) Rust rendering engine into matplotlib objects.  Colormaps are
registered with matplotlib on first import so they can be used by name::

    import matplotlib.pyplot as plt
    from wrf.solar7 import SOLAR7_STYLES, solar7_products

    # Use a colormap directly
    plt.imshow(data, cmap="solar7_temperature")

    # Use the style dict in plot_field
    from wrf.plot import plot_field
    plot_field(wrf_file, "sbcape", style="solar7")

All colormaps use ``LinearSegmentedColormap.from_list()`` with enough
sample points for smooth interpolation, except ``solar7_reflectivity``
which is a ``ListedColormap`` of 27 discrete colours.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Lazy matplotlib import
# ---------------------------------------------------------------------------

def _import_mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        return matplotlib, plt, mcolors
    except ImportError:
        raise ImportError(
            "matplotlib is required for wrf.solar7.  Install it with:\n"
            "    pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Colour hex helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str):
    """Convert '#rrggbb' to (r, g, b) floats in [0, 1]."""
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _lerp_colors(hex_list, N):
    """Linearly interpolate *N* RGB tuples across a list of hex anchors."""
    anchors = [_hex_to_rgb(c) for c in hex_list]
    if N <= 1:
        return [anchors[0]]
    result = []
    for i in range(N):
        t = i / (N - 1) * (len(anchors) - 1)
        lo = int(t)
        hi = min(lo + 1, len(anchors) - 1)
        frac = t - lo
        r = anchors[lo][0] + frac * (anchors[hi][0] - anchors[lo][0])
        g = anchors[lo][1] + frac * (anchors[hi][1] - anchors[lo][1])
        b = anchors[lo][2] + frac * (anchors[hi][2] - anchors[lo][2])
        result.append((r, g, b))
    return result


# ---------------------------------------------------------------------------
# Raw colour data from the Rust source
# ---------------------------------------------------------------------------

_WINDS_COLORS = [
    "#ffffff", "#87cefa", "#6a5acd", "#e696dc", "#c85abe",
    "#a01496", "#c80028", "#dc283c", "#f05050", "#faf064",
    "#dcbe46", "#be8c28", "#a05a0a",
]

_TEMPERATURE_COLORS = [
    "#2b5d7e", "#75a8b0", "#aee3dc", "#a0b8d6", "#968bc5",
    "#8243b2", "#a343b3", "#f7f7ff", "#a0b8d6", "#0f5575",
    "#6d8c77", "#f8eea2", "#aa714d", "#5f0000", "#852c40",
    "#b28f85", "#e7e0da", "#959391", "#454844",
]

_DEWPOINT_DRY = ["#996f4f", "#4d4236", "#f2f2d8"]
_DEWPOINT_MOIST_SEGS = [
    (["#e3f3e6", "#64c461"], 10),
    (["#32ae32", "#084d06"], 10),
    (["#66a3ad", "#12292a"], 10),
    (["#66679d", "#2b1e63"], 10),
    (["#714270", "#a27382"], 10),
]

_RH_SEG1_COLORS = ["#a5734d", "#382f28", "#6e6559", "#a59b8e", "#ddd1c3"]
_RH_SEG2_COLORS = ["#c8d7c0", "#004a2f"]
_RH_SEG3_COLORS = ["#004123", "#28588c"]

_RELVORT_COLORS = [
    "#323232", "#4d4d4d", "#707070", "#8a8a8a", "#a1a1a1",
    "#c0c0c0", "#d6d6d6", "#e5e5e5", "#ffffff", "#fdd244",
    "#fea000", "#f16702", "#da2422", "#ab029b", "#78008f",
    "#44008b", "#000160", "#244488", "#4f85b2", "#73cadb",
    "#91fffd",
]

# Simulated IR: purple-pink -> white -> rainbow -> grayscale
_SIM_IR_SEG_COOL = ["#8b008b", "#da70d6", "#ffb6c1", "#ffffff"]
_SIM_IR_SEG_WARM = [
    "#ffffff", "#ffff00", "#ffd700", "#ffa500", "#ff4500",
    "#ff0000", "#dc143c", "#b22222", "#8b0000", "#800000",
    "#006400", "#008000", "#228b22", "#32cd32", "#00ff00",
]
_SIM_IR_SEG_GRAY = [
    "#000000", "#1a1a1a", "#333333", "#4d4d4d", "#666666",
    "#808080", "#999999", "#b3b3b3", "#cccccc", "#e6e6e6",
    "#f0f0f0",
]

# Composite base segments
_COMPOSITE_SEGS = [
    ["#ffffff", "#696969"],  # seg0
    ["#37536a", "#a7c8ce"],  # seg1
    ["#e9dd96", "#e16f02"],  # seg2
    ["#dc4110", "#8b0950"],  # seg3
    ["#73088a", "#da99e7"],  # seg4
    ["#e9bec3", "#b2445a"],  # seg5
    ["#893d48", "#bc9195"],  # seg6
]

_COMPOSITE_QUANTS = {
    "cape":       [10, 10, 10, 10, 10, 10, 20],
    "three_cape": [10, 10, 10, 10, 10, 10, 40],
    "ehi":        [10, 10, 20, 20, 20, 40, 40],
    "srh":        [10, 10, 10, 10, 10, 10, 40],
    "stp":        [10, 10, 10, 10, 10, 10, 40],
    "lapse_rate": [40, 10, 10, 10, 10,  0,  0],
    "uh":         [10, 10, 10, 10, 20, 20,  0],
}

_REFLECTIVITY_COLORS = [
    "#ffffff", "#f2f6fc", "#d9e3f4", "#b0c6e6", "#8aa7da",
    "#648bcb", "#396dc1", "#1350b4", "#0d4f5d", "#43736f",
    "#77987b", "#a8bf8b", "#fdf273", "#f2d45a", "#eeb247",
    "#e1932d", "#d97517", "#cd5403", "#cd0002", "#a10206",
    "#75030b", "#9e37ab", "#83259d", "#601490", "#818181",
    "#b3b3b3", "#e8e8e8",
]

_GEOPOT_ANOMALY_COLORS = [
    "#c9f2fc", "#e684f4", "#732164", "#7b2b8d", "#8a41d6",
    "#253fba", "#7089cb", "#c0d5e8", "#ffffff", "#fbcfa1",
    "#fc984b", "#b83800", "#a3241a", "#5e1425", "#42293e",
    "#557b75", "#ddd5cf",
]

_PRECIP_SEGS = [
    (["#ffffff", "#ffffff"],                         1),
    (["#dcdcdc", "#bebebe", "#9e9e9e", "#818181"],   9),
    (["#b8f0c1", "#156471"],                        40),
    (["#164fba", "#d8edf5"],                        50),
    (["#cfbddd", "#a134b1"],                       100),
    (["#a43c32", "#dd9c98"],                       200),
    (["#f6f0a3", "#7e4b26", "#542f17"],           1100),
]


# ---------------------------------------------------------------------------
# Colormap builders
# ---------------------------------------------------------------------------

def _build_composite_cmap(name: str, quants):
    """Build a composite colormap from the shared base segments."""
    colors = []
    for seg_colors, n in zip(_COMPOSITE_SEGS, quants):
        if n > 0:
            colors.extend(_lerp_colors(seg_colors, n))
    return colors, name


def _build_all_colormaps():
    """Build every Solar7 colormap and return a dict of name -> cmap."""
    _, plt, mcolors = _import_mpl()
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    cmaps = {}

    # --- solar7_winds (60 segments) ---
    colors = _lerp_colors(_WINDS_COLORS, 60)
    cmaps["solar7_winds"] = LinearSegmentedColormap.from_list(
        "solar7_winds", colors, N=256)

    # --- solar7_temperature (180 segments, -60F to 120F) ---
    colors = _lerp_colors(_TEMPERATURE_COLORS, 180)
    cmaps["solar7_temperature"] = LinearSegmentedColormap.from_list(
        "solar7_temperature", colors, N=256)

    # --- solar7_dewpoint (80 dry + 50 moist = 130 segments) ---
    colors = _lerp_colors(_DEWPOINT_DRY, 80)
    for seg_hex, n in _DEWPOINT_MOIST_SEGS:
        colors.extend(_lerp_colors(seg_hex, n))
    cmaps["solar7_dewpoint"] = LinearSegmentedColormap.from_list(
        "solar7_dewpoint", colors, N=256)

    # --- solar7_rh (40 + 50 + 10 = 100 segments) ---
    colors = _lerp_colors(_RH_SEG1_COLORS, 40)
    colors.extend(_lerp_colors(_RH_SEG2_COLORS, 50))
    colors.extend(_lerp_colors(_RH_SEG3_COLORS, 10))
    cmaps["solar7_rh"] = LinearSegmentedColormap.from_list(
        "solar7_rh", colors, N=256)

    # --- solar7_relvort (100 segments) ---
    colors = _lerp_colors(_RELVORT_COLORS, 100)
    cmaps["solar7_relvort"] = LinearSegmentedColormap.from_list(
        "solar7_relvort", colors, N=256)

    # --- solar7_sim_ir (10 + 60 + 60 = 130 segments) ---
    colors = _lerp_colors(_SIM_IR_SEG_COOL, 10)
    colors.extend(_lerp_colors(_SIM_IR_SEG_WARM, 60))
    colors.extend(_lerp_colors(_SIM_IR_SEG_GRAY, 60))
    cmaps["solar7_sim_ir"] = LinearSegmentedColormap.from_list(
        "solar7_sim_ir", colors, N=256)

    # --- Composite colormaps ---
    for comp_name, quants in _COMPOSITE_QUANTS.items():
        cmap_name = f"solar7_{comp_name}"
        comp_colors, _ = _build_composite_cmap(cmap_name, quants)
        cmaps[cmap_name] = LinearSegmentedColormap.from_list(
            cmap_name, comp_colors, N=256)

    # --- solar7_reflectivity (27 discrete colours) ---
    refl_rgb = [_hex_to_rgb(c) for c in _REFLECTIVITY_COLORS]
    cmaps["solar7_reflectivity"] = ListedColormap(
        refl_rgb, name="solar7_reflectivity")

    # --- solar7_geopot_anomaly ---
    colors = _lerp_colors(_GEOPOT_ANOMALY_COLORS, 100)
    cmaps["solar7_geopot_anomaly"] = LinearSegmentedColormap.from_list(
        "solar7_geopot_anomaly", colors, N=256)

    # --- solar7_precip (composite from segments) ---
    colors = []
    for seg_hex, n in _PRECIP_SEGS:
        colors.extend(_lerp_colors(seg_hex, n))
    cmaps["solar7_precip"] = LinearSegmentedColormap.from_list(
        "solar7_precip", colors, N=256)

    return cmaps


# ---------------------------------------------------------------------------
# Registration -- happens once on first import
# ---------------------------------------------------------------------------

_REGISTERED = False

def _register_colormaps():
    """Register all Solar7 colormaps with matplotlib (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    _, plt, _ = _import_mpl()
    import matplotlib

    cmaps = _build_all_colormaps()
    for name, cmap in cmaps.items():
        try:
            matplotlib.colormaps.register(cmap, name=name)
        except (ValueError, AttributeError):
            # Already registered, or older matplotlib without .colormaps
            try:
                plt.register_cmap(name=name, cmap=cmap)
            except Exception:
                pass
    _REGISTERED = True


# Trigger registration at import time
_register_colormaps()


# ---------------------------------------------------------------------------
# SOLAR7_STYLES -- variable name -> (cmap, levels, extend)
# ---------------------------------------------------------------------------
#
# These map WRF variable names to the Solarpower07 colormap and contour
# levels that match the operational product definitions.  Used by
# plot_field(style="solar7").

SOLAR7_STYLES: Dict[str, Dict[str, Any]] = {
    # ---- Surface / upper winds ----
    "wspd":   dict(cmap="solar7_winds", levels=np.arange(0, 62, 2), extend="max"),
    "wspd10": dict(cmap="solar7_winds", levels=np.arange(0, 62, 2), extend="max"),
    "ua":     dict(cmap="solar7_winds", levels=np.arange(0, 120, 5), extend="max"),
    "va":     dict(cmap="solar7_winds", levels=np.arange(0, 120, 5), extend="max"),

    # ---- Temperature (degF native for the map, -60 to 120) ----
    "temp":    dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "tc":      dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "theta":   dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "theta_e": dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "tv":      dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "twb":     dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),
    "theta_w": dict(cmap="solar7_temperature", levels=np.arange(-60, 122, 2), extend="both"),

    # ---- Dewpoint ----
    "td":   dict(cmap="solar7_dewpoint", levels=np.arange(-40, 82, 2), extend="both"),
    "dp2m": dict(cmap="solar7_dewpoint", levels=np.arange(-40, 82, 2), extend="both"),

    # ---- Relative humidity ----
    "rh":   dict(cmap="solar7_rh", levels=np.arange(0, 105, 5), extend="neither"),
    "rh2m": dict(cmap="solar7_rh", levels=np.arange(0, 105, 5), extend="neither"),

    # ---- Moisture ----
    "pw":                dict(cmap="solar7_dewpoint", levels=np.arange(0, 66, 3), extend="max"),
    "mixing_ratio":      dict(cmap="solar7_dewpoint", extend="max"),
    "specific_humidity": dict(cmap="solar7_dewpoint", extend="max"),

    # ---- Vorticity ----
    "avo": dict(cmap="solar7_relvort", levels=np.linspace(-30, 30, 21), extend="both", center_zero=True),
    "pvo": dict(cmap="solar7_relvort", levels=np.linspace(-10, 10, 21), extend="both", center_zero=True),

    # ---- Simulated IR / cloud-top temp ----
    "ctt":       dict(cmap="solar7_sim_ir", levels=np.arange(-80, 22, 2), extend="both"),
    "cloudfrac": dict(cmap="solar7_sim_ir", levels=np.arange(0, 105, 5), extend="neither"),

    # ---- CAPE ----
    "sbcape":         dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "mlcape":         dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "mucape":         dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "cape":           dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "effective_cape": dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "cape3d":         dict(cmap="solar7_three_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "ecape":          dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "ncape":          dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),
    "ecape_cape":     dict(cmap="solar7_cape", levels=np.arange(0, 4250, 250), extend="max"),

    # ---- CIN ----
    "sbcin": dict(cmap="solar7_cape", levels=np.arange(-300, 1, 25), extend="min"),
    "mlcin": dict(cmap="solar7_cape", levels=np.arange(-300, 1, 25), extend="min"),
    "mucin": dict(cmap="solar7_cape", levels=np.arange(-300, 1, 25), extend="min"),
    "cin":   dict(cmap="solar7_cape", levels=np.arange(-300, 1, 25), extend="min"),
    "ecape_cin": dict(cmap="solar7_cape", levels=np.arange(-300, 1, 25), extend="min"),

    # ---- Sounding levels ----
    "lcl": dict(cmap="solar7_cape", levels=np.arange(0, 4200, 200), extend="max"),
    "lfc": dict(cmap="solar7_cape", levels=np.arange(0, 5500, 500), extend="max"),
    "el":  dict(cmap="solar7_cape", levels=np.arange(0, 16000, 1000), extend="max"),
    "ecape_lfc": dict(cmap="solar7_cape", levels=np.arange(0, 5500, 500), extend="max"),
    "ecape_el":  dict(cmap="solar7_cape", levels=np.arange(0, 16000, 1000), extend="max"),

    # ---- SRH ----
    "srh":            dict(cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max"),
    "srh1":           dict(cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max"),
    "srh3":           dict(cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max"),
    "effective_srh":  dict(cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max"),

    # ---- STP ----
    "stp":           dict(cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max"),
    "stp_fixed":     dict(cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max"),
    "stp_effective": dict(cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max"),
    "ecape_scp":     dict(cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max"),

    # ---- EHI ----
    "ehi": dict(cmap="solar7_ehi", levels=np.arange(0, 5.5, 0.5), extend="max"),
    "ecape_ehi": dict(cmap="solar7_ehi", levels=np.arange(0, 5.5, 0.5), extend="max"),

    # ---- Shear ----
    "shear_0_1km": dict(cmap="solar7_winds", levels=np.arange(0, 42, 2), extend="max"),
    "shear_0_6km": dict(cmap="solar7_winds", levels=np.arange(0, 42, 2), extend="max"),
    "bulk_shear":  dict(cmap="solar7_winds", levels=np.arange(0, 42, 2), extend="max"),

    # ---- Severe composites ----
    "scp":            dict(cmap="solar7_cape", levels=np.arange(0, 11, 1), extend="max"),
    "ship":           dict(cmap="solar7_cape", levels=np.arange(0, 5.5, 0.5), extend="max"),
    "bri":            dict(cmap="solar7_cape", levels=np.arange(0, 110, 10), extend="max"),
    "critical_angle": dict(cmap="solar7_relvort", levels=np.arange(0, 200, 10), extend="max"),

    # ---- Updraft helicity ----
    "uhel": dict(cmap="solar7_uh", levels=np.arange(0, 210, 10), extend="max"),

    # ---- Lapse rates ----
    "lapse_rate":         dict(cmap="solar7_lapse_rate", levels=np.arange(4, 10.5, 0.5), extend="both"),
    "lapse_rate_700_500": dict(cmap="solar7_lapse_rate", levels=np.arange(4, 10.5, 0.5), extend="both"),
    "lapse_rate_0_3km":   dict(cmap="solar7_lapse_rate", levels=np.arange(4, 10.5, 0.5), extend="both"),

    # ---- Radar / reflectivity ----
    "dbz":    dict(cmap="solar7_reflectivity", levels=np.arange(5, 71, 2.5), extend="max", mask_below=5.0),
    "maxdbz": dict(cmap="solar7_reflectivity", levels=np.arange(5, 71, 2.5), extend="max", mask_below=5.0),

    # ---- Geopotential anomaly ----
    "geopt": dict(cmap="solar7_geopot_anomaly", extend="both", center_zero=True),

    # ---- Heights / terrain ----
    "height":         dict(cmap="solar7_geopot_anomaly", extend="both"),
    "height_agl":     dict(cmap="solar7_geopot_anomaly", extend="both"),
    "terrain":        dict(cmap="solar7_geopot_anomaly", levels=np.arange(0, 4200, 200), extend="max"),
    "freezing_level": dict(cmap="solar7_temperature", levels=np.arange(0, 5500, 250), extend="max"),
    "wet_bulb_0":     dict(cmap="solar7_temperature", levels=np.arange(0, 5500, 250), extend="max"),

    # ---- Precipitation ----
    "precip":    dict(cmap="solar7_precip", levels=np.concatenate([
                     np.arange(0, 1, 0.1), np.arange(1, 5, 0.5), np.arange(5, 25, 1),
                     np.arange(25, 55, 5)]), extend="max"),
    "rain":      dict(cmap="solar7_precip", levels=np.concatenate([
                     np.arange(0, 1, 0.1), np.arange(1, 5, 0.5), np.arange(5, 25, 1),
                     np.arange(25, 55, 5)]), extend="max"),

    # ---- Pressure ----
    "slp":      dict(cmap="solar7_geopot_anomaly", levels=np.arange(980, 1042, 2), extend="both"),
    "pressure": dict(cmap="solar7_geopot_anomaly", extend="both"),

    # ---- Effective inflow ----
    "effective_inflow": dict(cmap="solar7_winds", extend="max"),

    # ---- Wind direction ----
    "wdir":   dict(cmap="solar7_winds", levels=np.arange(0, 370, 10), extend="neither"),
    "wdir10": dict(cmap="solar7_winds", levels=np.arange(0, 370, 10), extend="neither"),

    # ---- Fire weather ----
    "fosberg": dict(cmap="solar7_temperature", levels=np.arange(0, 80, 5), extend="max"),
    "haines":  dict(cmap="solar7_temperature", levels=np.arange(2, 7, 1), extend="neither"),
    "hdw":     dict(cmap="solar7_temperature", extend="max"),

    # ---- Omega / vertical velocity ----
    "wa":    dict(cmap="solar7_relvort", center_zero=True, extend="both"),
    "omega": dict(cmap="solar7_relvort", center_zero=True, extend="both"),
}


# ---------------------------------------------------------------------------
# Product definitions -- the full Solarpower07 product catalogue
# ---------------------------------------------------------------------------

def solar7_products() -> List[Dict[str, Any]]:
    """Return a list of all Solarpower07 product definitions.

    Each dict has:
        name        - short variable name for getvar()
        title       - human-readable product title
        cmap        - matplotlib colormap name (solar7_*)
        levels      - numpy array of contour levels (or None for auto)
        extend      - colorbar extend mode
        units       - preferred unit string (or None for default)
        category    - grouping for UI/iteration
        center_zero - True if the colormap should be centred on zero

    Usage::

        from wrf.solar7 import solar7_products
        from wrf.plot import plot_field
        from wrf import WrfFile

        wf = WrfFile("wrfout_d01_2024-05-20_18:00:00")
        for prod in solar7_products():
            try:
                plot_field(wf, prod["name"], style="solar7",
                           units=prod.get("units"), title=prod["title"])
            except Exception:
                pass
    """
    return [
        # --- Surface ---
        dict(name="wspd10", title="10-m Wind Speed", cmap="solar7_winds",
             levels=np.arange(0, 62, 2), extend="max", units="knots",
             category="surface"),
        dict(name="temp", title="2-m Temperature", cmap="solar7_temperature",
             levels=np.arange(-60, 122, 2), extend="both", units="degF",
             category="surface"),
        dict(name="td", title="2-m Dewpoint", cmap="solar7_dewpoint",
             levels=np.arange(-40, 82, 2), extend="both", units="degF",
             category="surface"),
        dict(name="dp2m", title="2-m Dewpoint (dp2m)", cmap="solar7_dewpoint",
             levels=np.arange(-40, 82, 2), extend="both", units="degF",
             category="surface"),
        dict(name="rh2m", title="2-m Relative Humidity", cmap="solar7_rh",
             levels=np.arange(0, 105, 5), extend="neither", units=None,
             category="surface"),
        dict(name="slp", title="Sea-Level Pressure", cmap="solar7_geopot_anomaly",
             levels=np.arange(980, 1042, 2), extend="both", units="hPa",
             category="surface"),
        dict(name="pw", title="Precipitable Water", cmap="solar7_dewpoint",
             levels=np.arange(0, 66, 3), extend="max", units=None,
             category="surface"),

        # --- Thermodynamic instability ---
        dict(name="sbcape", title="Surface-Based CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="mlcape", title="Mixed-Layer CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="mucape", title="Most-Unstable CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="ecape", title="Entraining CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="ncape", title="Normalized CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="ecape_cape", title="ECAPE Companion CAPE", cmap="solar7_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="cape3d", title="3-D CAPE", cmap="solar7_three_cape",
             levels=np.arange(0, 4250, 250), extend="max", units=None,
             category="instability"),
        dict(name="sbcin", title="Surface-Based CIN", cmap="solar7_cape",
             levels=np.arange(-300, 1, 25), extend="min", units=None,
             category="instability"),
        dict(name="mlcin", title="Mixed-Layer CIN", cmap="solar7_cape",
             levels=np.arange(-300, 1, 25), extend="min", units=None,
             category="instability"),
        dict(name="ecape_cin", title="ECAPE Companion CIN", cmap="solar7_cape",
             levels=np.arange(-300, 1, 25), extend="min", units=None,
             category="instability"),
        dict(name="lcl", title="Lifting Condensation Level", cmap="solar7_cape",
             levels=np.arange(0, 4200, 200), extend="max", units=None,
             category="instability"),
        dict(name="lfc", title="Level of Free Convection", cmap="solar7_cape",
             levels=np.arange(0, 5500, 500), extend="max", units=None,
             category="instability"),
        dict(name="ecape_lfc", title="ECAPE Companion LFC", cmap="solar7_cape",
             levels=np.arange(0, 5500, 500), extend="max", units=None,
             category="instability"),
        dict(name="el", title="Equilibrium Level", cmap="solar7_cape",
             levels=np.arange(0, 16000, 1000), extend="max", units=None,
             category="instability"),
        dict(name="ecape_el", title="ECAPE Companion EL", cmap="solar7_cape",
             levels=np.arange(0, 16000, 1000), extend="max", units=None,
             category="instability"),

        # --- Lapse rates ---
        dict(name="lapse_rate_700_500", title="700-500 hPa Lapse Rate",
             cmap="solar7_lapse_rate", levels=np.arange(4, 10.5, 0.5),
             extend="both", units=None, category="lapse_rates"),
        dict(name="lapse_rate_0_3km", title="0-3 km Lapse Rate",
             cmap="solar7_lapse_rate", levels=np.arange(4, 10.5, 0.5),
             extend="both", units=None, category="lapse_rates"),

        # --- Helicity / shear ---
        dict(name="srh1", title="0-1 km Storm-Relative Helicity",
             cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max",
             units=None, category="shear"),
        dict(name="srh3", title="0-3 km Storm-Relative Helicity",
             cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max",
             units=None, category="shear"),
        dict(name="effective_srh", title="Effective SRH",
             cmap="solar7_srh", levels=np.arange(0, 525, 25), extend="max",
             units=None, category="shear"),
        dict(name="shear_0_1km", title="0-1 km Bulk Shear",
             cmap="solar7_winds", levels=np.arange(0, 42, 2), extend="max",
             units="knots", category="shear"),
        dict(name="shear_0_6km", title="0-6 km Bulk Shear",
             cmap="solar7_winds", levels=np.arange(0, 42, 2), extend="max",
             units="knots", category="shear"),

        # --- Severe composites ---
        dict(name="stp", title="Significant Tornado Parameter",
             cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max",
             units=None, category="severe"),
        dict(name="stp_fixed", title="STP (Fixed Layer)",
             cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max",
             units=None, category="severe"),
        dict(name="stp_effective", title="STP (Effective Layer)",
             cmap="solar7_stp", levels=np.arange(0, 11, 1), extend="max",
             units=None, category="severe"),
        dict(name="scp", title="Supercell Composite Parameter",
             cmap="solar7_cape", levels=np.arange(0, 11, 1), extend="max",
             units=None, category="severe"),
        dict(name="ecape_scp", title="Experimental ECAPE SCP",
             cmap="solar7_cape", levels=np.arange(0, 11, 1), extend="max",
             units=None, category="severe"),
        dict(name="ehi", title="Energy-Helicity Index",
             cmap="solar7_ehi", levels=np.arange(0, 5.5, 0.5), extend="max",
             units=None, category="severe"),
        dict(name="ecape_ehi", title="Experimental ECAPE EHI",
             cmap="solar7_ehi", levels=np.arange(0, 5.5, 0.5), extend="max",
             units=None, category="severe"),
        dict(name="ship", title="Sig. Hail Parameter",
             cmap="solar7_cape", levels=np.arange(0, 5.5, 0.5), extend="max",
             units=None, category="severe"),
        dict(name="critical_angle", title="Critical Angle",
             cmap="solar7_relvort", levels=np.arange(0, 200, 10), extend="max",
             units=None, category="severe"),

        # --- Updraft helicity ---
        dict(name="uhel", title="Updraft Helicity",
             cmap="solar7_uh", levels=np.arange(0, 210, 10), extend="max",
             units=None, category="severe"),

        # --- Radar ---
        dict(name="dbz", title="Simulated Reflectivity (1 km AGL)",
             cmap="solar7_reflectivity", levels=np.arange(-10, 80, 5),
             extend="max", units=None, category="radar"),
        dict(name="maxdbz", title="Composite Reflectivity",
             cmap="solar7_reflectivity", levels=np.arange(-10, 80, 5),
             extend="max", units=None, category="radar"),

        # --- Cloud / IR ---
        dict(name="ctt", title="Cloud-Top Temperature (Simulated IR)",
             cmap="solar7_sim_ir", levels=np.arange(-80, 22, 2),
             extend="both", units="degC", category="satellite"),

        # --- Vorticity ---
        dict(name="avo", title="Absolute Vorticity",
             cmap="solar7_relvort", levels=np.linspace(-30, 30, 21),
             extend="both", units=None, category="dynamics",
             center_zero=True),
        dict(name="pvo", title="Potential Vorticity",
             cmap="solar7_relvort", levels=np.linspace(-10, 10, 21),
             extend="both", units=None, category="dynamics",
             center_zero=True),

        # --- Geopotential / heights ---
        dict(name="geopt", title="Geopotential Anomaly",
             cmap="solar7_geopot_anomaly", levels=None, extend="both",
             units=None, category="dynamics", center_zero=True),
        dict(name="height", title="Geopotential Height",
             cmap="solar7_geopot_anomaly", levels=None, extend="both",
             units=None, category="dynamics"),
        dict(name="terrain", title="Terrain Height",
             cmap="solar7_geopot_anomaly", levels=np.arange(0, 4200, 200),
             extend="max", units=None, category="terrain"),

        # --- Fire weather ---
        dict(name="fosberg", title="Fosberg Fire-Weather Index",
             cmap="solar7_temperature", levels=np.arange(0, 80, 5),
             extend="max", units=None, category="fire"),
        dict(name="haines", title="Haines Index",
             cmap="solar7_temperature", levels=np.arange(2, 7, 1),
             extend="neither", units=None, category="fire"),
    ]
