"""
wrf-rust CLI -- quick-look tool for WRF output files.

Usage:
    python -m wrf info  wrfout_d01_...
    python -m wrf plot  wrfout_d01_... slp --units hPa -o slp.png
    python -m wrf panel wrfout_d01_... sbcape srh1 stp mlcin -o severe.png
    python -m wrf stats wrfout_d01_... sbcape slp temp
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAP_PROJ_NAMES = {
    1: "Lambert Conformal",
    2: "Polar Stereographic",
    3: "Mercator",
    6: "Lat-Lon (Cylindrical Equidistant)",
}


def _open(path: str):
    """Open a WRF file and return a wrf.WrfFile."""
    from wrf import WrfFile

    return WrfFile(path)


def _projection_name(path: str) -> str:
    """Best-effort projection name from the raw NetCDF global attrs."""
    try:
        from netCDF4 import Dataset

        with Dataset(path, "r") as nc:
            map_proj = int(nc.getncattr("MAP_PROJ"))
        return _MAP_PROJ_NAMES.get(map_proj, f"Unknown (MAP_PROJ={map_proj})")
    except Exception:
        return "Unknown"


def _default_units(varname: str) -> str:
    """Look up default units for *varname* from the variable registry."""
    from wrf import list_variables

    for v in list_variables():
        if v["name"] == varname:
            return v["units"]
    return ""


def _bold(text: str) -> str:
    """Wrap *text* in ANSI bold if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text


def _dim(text: str) -> str:
    """Dim (grey) text on TTYs."""
    if sys.stdout.isatty():
        return f"\033[90m{text}\033[0m"
    return text


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> None:
    """Print file dimensions, grid spacing, projection, available times."""
    wf = _open(args.file)
    try:
        times = wf.times()
    except Exception:
        times = ["(unavailable)"]

    proj = _projection_name(args.file)

    time_range = f"{times[0]} to {times[-1]}" if len(times) > 1 else times[0]
    spacing = f"{wf.dx:g} m" if wf.dx == wf.dy else f"{wf.dx:g} x {wf.dy:g} m"

    print()
    print(f"  {_bold('WRF File:')}   {os.path.basename(wf.path)}")
    print(f"  {_bold('Path:')}       {wf.path}")
    print(f"  {_bold('Grid:')}       {wf.nx} x {wf.ny} x {wf.nz}  (nx x ny x nz)")
    print(f"  {_bold('Times:')}      {wf.nt} step{'s' if wf.nt != 1 else ''}")
    print(f"  {_bold('Spacing:')}    {spacing}")
    print(f"  {_bold('Projection:')} {proj}")
    print(f"  {_bold('Time range:')} {time_range}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    """Print min / max / mean / std for one or more variables."""
    from wrf import getvar

    wf = _open(args.file)
    tidx = args.timeidx if args.timeidx is not None else 0

    print()
    for varname in args.variables:
        try:
            data = getvar(wf, varname, timeidx=tidx)
        except Exception as exc:
            print(f"  {varname:<14s} error: {exc}")
            continue

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            print(f"  {varname:<14s} (all NaN / Inf)")
            continue

        units = _default_units(varname)
        mn, mx = float(finite.min()), float(finite.max())
        avg, std = float(finite.mean()), float(finite.std())

        print(
            f"  {varname:<14s}"
            f"min={mn:<12.1f}"
            f"max={mx:<12.1f}"
            f"mean={avg:<12.1f}"
            f"std={std:<12.1f}"
            f"{units}"
        )
    print()


# ---------------------------------------------------------------------------
# Subcommand: plot
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    """Import matplotlib and return (plt, mpl) or exit with a helpful error."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        return plt, matplotlib
    except ImportError:
        print(
            "Error: matplotlib is required for plotting.\n"
            "  pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_plot(args: argparse.Namespace) -> None:
    """Plot a single variable to screen or file."""
    from wrf import getvar

    plt, _ = _ensure_matplotlib()

    wf = _open(args.file)
    tidx = args.timeidx if args.timeidx is not None else 0

    kwargs = {}
    if args.units:
        kwargs["units"] = args.units

    data = getvar(wf, args.variable, timeidx=tidx, **kwargs)

    # If 3-D, take a single level
    if data.ndim == 3:
        level = args.level if args.level is not None else 0
        if level >= data.shape[0]:
            print(
                f"Error: level {level} out of range "
                f"(variable has {data.shape[0]} levels).",
                file=sys.stderr,
            )
            sys.exit(1)
        data = data[level]

    units_label = args.units or _default_units(args.variable)
    try:
        title_time = wf.times()[tidx] if tidx < wf.nt else ""
    except Exception:
        title_time = f"t={tidx}"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(data, cmap=args.cmap or "viridis", shading="auto")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    if units_label:
        cbar.set_label(units_label, fontsize=11)

    title = f"{args.variable}"
    if args.level is not None:
        title += f"  [level {args.level}]"
    if title_time:
        title += f"\n{title_time}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("West-East Grid Index")
    ax.set_ylabel("South-North Grid Index")

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        # Switch to interactive backend for display
        import matplotlib

        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Subcommand: panel
# ---------------------------------------------------------------------------

def cmd_panel(args: argparse.Namespace) -> None:
    """Plot multiple variables as a panel grid."""
    from wrf import getvar

    plt, _ = _ensure_matplotlib()

    wf = _open(args.file)
    tidx = args.timeidx if args.timeidx is not None else 0

    nvars = len(args.variables)
    ncols = min(args.cols, nvars)
    nrows = (nvars + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False
    )

    for idx, varname in enumerate(args.variables):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        try:
            data = getvar(wf, varname, timeidx=tidx)
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"{varname}\n{exc}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="red",
            )
            ax.set_title(varname, fontsize=11)
            continue

        # If 3-D, take lowest level
        if data.ndim == 3:
            data = data[0]

        units_label = _default_units(varname)

        im = ax.pcolormesh(data, cmap="viridis", shading="auto")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        if units_label:
            cbar.set_label(units_label, fontsize=9)

        ax.set_title(varname, fontsize=11)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(nvars, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    time_str = wf.times()[tidx] if tidx < wf.nt else ""
    fig.suptitle(
        f"{os.path.basename(wf.path)}  {time_str}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        import matplotlib

        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wrf",
        description="wrf-rust: quick-look tool for WRF output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python -m wrf info  wrfout_d01_2024-06-01_00:00:00
              python -m wrf plot  wrfout_d01_2024-06-01_00:00:00 slp --units hPa -o slp.png
              python -m wrf panel wrfout_d01_2024-06-01_00:00:00 sbcape srh1 stp -o severe.png
              python -m wrf stats wrfout_d01_2024-06-01_00:00:00 sbcape slp temp
        """),
    )

    sub = parser.add_subparsers(dest="command", title="commands")

    # ── info ──
    p_info = sub.add_parser("info", help="Print file metadata and grid info.")
    p_info.add_argument("file", help="Path to WRF output file.")

    # ── plot ──
    p_plot = sub.add_parser("plot", help="Plot a single variable to screen or file.")
    p_plot.add_argument("file", help="Path to WRF output file.")
    p_plot.add_argument("variable", help="Variable name (e.g. slp, sbcape, temp).")
    p_plot.add_argument("--timeidx", type=int, default=None, help="Time index (default: 0).")
    p_plot.add_argument("--units", type=str, default=None, help="Convert to these units (e.g. hPa, degC).")
    p_plot.add_argument("--level", type=int, default=None, help="Vertical level index for 3-D fields.")
    p_plot.add_argument("--cmap", type=str, default=None, help="Matplotlib colormap name (default: viridis).")
    p_plot.add_argument("-o", "--output", type=str, default=None, help="Save to file instead of showing on screen.")

    # ── panel ──
    p_panel = sub.add_parser("panel", help="Plot multiple variables as a grid.")
    p_panel.add_argument("file", help="Path to WRF output file.")
    p_panel.add_argument("variables", nargs="+", help="Variable names.")
    p_panel.add_argument("--timeidx", type=int, default=None, help="Time index (default: 0).")
    p_panel.add_argument("--cols", type=int, default=2, help="Number of columns in panel (default: 2).")
    p_panel.add_argument("-o", "--output", type=str, default=None, help="Save to file instead of showing on screen.")

    # ── stats ──
    p_stats = sub.add_parser("stats", help="Print min/max/mean/std for variables.")
    p_stats.add_argument("file", help="Path to WRF output file.")
    p_stats.add_argument("variables", nargs="+", help="Variable names.")
    p_stats.add_argument("--timeidx", type=int, default=None, help="Time index (default: 0).")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "info": cmd_info,
        "plot": cmd_plot,
        "panel": cmd_panel,
        "stats": cmd_stats,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
