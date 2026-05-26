"""Native Rust sounding rendering."""

from __future__ import annotations

import os

from wrf import WrfFile
from wrf._wrf import (
    render_sounding_box as _render_sounding_box,
    render_sounding_ij as _render_sounding_ij,
    render_sounding_latlon as _render_sounding_latlon,
)


def render_sounding(
    wrffile,
    output,
    *,
    timeidx=0,
    latlon=None,
    ij=None,
    box=None,
    method="mean",
):
    """Render a native sharprs sounding PNG from a WRF file.

    Exactly one of ``latlon=(lat, lon)``, ``ij=(i, j)``, or
    ``box=(south, west, north, east)`` must be supplied.
    """

    inner = _as_inner_file(wrffile)
    output = os.fspath(output)
    selections = sum(value is not None for value in (latlon, ij, box))
    if selections != 1:
        raise ValueError("pass exactly one of latlon=, ij=, or box=")

    if latlon is not None:
        lat, lon = latlon
        _render_sounding_latlon(inner, output, float(lat), float(lon), timeidx)
    elif ij is not None:
        i, j = ij
        _render_sounding_ij(inner, output, int(i), int(j), timeidx)
    else:
        south, west, north, east = box
        _render_sounding_box(
            inner,
            output,
            float(south),
            float(west),
            float(north),
            float(east),
            method,
            timeidx,
        )

    return output


def _as_inner_file(wrffile):
    if isinstance(wrffile, WrfFile):
        return wrffile._inner
    return WrfFile(wrffile)._inner
