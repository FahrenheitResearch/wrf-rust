#!/usr/bin/env python3
"""Exercise the concrete WRF-Runner/wrf-rust extension contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    ParityError,
    canonical_json_bytes,
    csv_values,
    distribution_version,
    load_fixture,
    load_json,
    print_error_and_exit,
    sha256_bytes,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "wrf-runner-contracts-v1.json")
    parser.add_argument("--fixtures", type=Path, default=here / "fixtures-v1.json")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--calls", action="append", help="comma-separated getvar call ids")
    parser.add_argument("--priority", choices=("P0", "P1", "all"), default="P0")
    parser.add_argument("--consumer-root", type=Path, help="optionally verify observed source hashes")
    parser.add_argument("--skip-getvar", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_shape(spec: list[Any], handle: Any) -> tuple[int, ...]:
    values = {"nx": handle.nx, "ny": handle.ny, "nz": handle.nz, "nt": handle.nt}
    return tuple(values[item] if isinstance(item, str) else int(item) for item in spec)


def qualified_type(value: Any) -> str:
    """Return a stable, human-readable concrete type name for reports."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def checked_row(identifier: str, checks: list[tuple[bool, str]], **details: Any) -> dict[str, Any]:
    """Build a probe row with every failed contract stated explicitly."""
    failures = [message for passed, message in checks if not bool(passed)]
    return {
        "id": identifier,
        "passed": not failures,
        "failures": failures,
        **details,
    }


def exception_row(identifier: str, exc: Exception, **details: Any) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    return {
        "id": identifier,
        "passed": False,
        "failures": [f"call raised {error}"],
        "error": error,
        **details,
    }


def is_plain_ndarray(value: Any) -> bool:
    """Require the exact ndarray class, excluding MaskedArray and subclasses."""
    return type(value) is np.ndarray


def is_float_pair(value: Any) -> bool:
    """Require the legacy WRF-Runner scalar coordinate container exactly."""
    return (
        type(value) is tuple
        and len(value) == 2
        and all(
            type(component) is float or isinstance(component, np.floating)
            for component in value
        )
    )


def longitude_midpoint(first: float, second: float) -> float:
    """Return the antimeridian-safe midpoint of two longitudes."""
    delta = (second - first + 180.0) % 360.0 - 180.0
    return (first + 0.5 * delta + 180.0) % 360.0 - 180.0


def verify_consumer_files(contract: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for source in contract["consumer"]["files"]:
        path = (root / source["path"]).resolve()
        actual = sha256_file(path)
        rows.append(
            {
                "path": source["path"],
                "expected_sha256": source["sha256"],
                "actual_sha256": actual,
                "passed": actual == source["sha256"],
            }
        )
    return rows


def probe_helpers(wrf: Any, handle: Any) -> list[dict[str, Any]]:
    rows = []

    coordinate = np.broadcast_to(
        np.array([1000.0, 500.0])[:, None, None], (2, 2, 2)
    )
    field = np.broadcast_to(np.array([0.0, 100.0])[:, None, None], (2, 2, 2))
    scalar_expected = 100.0 * np.log(750.0 / 1000.0) / np.log(500.0 / 1000.0)
    try:
        scalar_raw = wrf.interplevel(field, coordinate, 750.0)
        scalar_type = qualified_type(scalar_raw)
        scalar = np.asarray(scalar_raw)
        rows.append(
            checked_row(
                "interplevel_scalar",
                [
                    (
                        is_plain_ndarray(scalar_raw),
                        f"default return must be exact numpy.ndarray, got {scalar_type}",
                    ),
                    (scalar.shape == (2, 2), f"expected shape (2, 2), got {scalar.shape}"),
                    (
                        np.allclose(scalar, scalar_expected),
                        "descending-pressure interpolation must preserve the v0.2.35 log-pressure result",
                    ),
                ],
                expected_type="numpy.ndarray (exact; no subclasses)",
                actual_type=scalar_type,
                shape=list(scalar.shape),
                value=float(scalar[0, 0]) if scalar.shape == (2, 2) else None,
            )
        )
    except Exception as exc:
        rows.append(exception_row("interplevel_scalar", exc))

    coordinate_2d = np.broadcast_to(coordinate, (2, 2, 2))
    field_2d = np.broadcast_to(field, (2, 2, 2))
    targets = np.array([[900.0, 800.0], [700.0, 600.0]])
    expected = 100.0 * np.log(targets / 1000.0) / np.log(500.0 / 1000.0)
    try:
        level_2d_raw = wrf.interplevel(field_2d, coordinate_2d, targets)
        level_2d_type = qualified_type(level_2d_raw)
        level_2d = np.asarray(level_2d_raw)
        shape_passed = level_2d.shape == (2, 2)
        value_passed = shape_passed and np.allclose(level_2d, expected)
        rows.append(
            checked_row(
                "interplevel_2d_target",
                [
                    (
                        is_plain_ndarray(level_2d_raw),
                        f"default return must be exact numpy.ndarray, got {level_2d_type}",
                    ),
                    (shape_passed, f"expected shape (2, 2), got {level_2d.shape}"),
                    (
                        value_passed,
                        "2-D targets must preserve v0.2.35 log-pressure interpolation",
                    ),
                ],
                expected_type="numpy.ndarray (exact; no subclasses)",
                actual_type=level_2d_type,
                shape=list(level_2d.shape),
                max_absolute_error=(
                    float(np.max(np.abs(level_2d - expected))) if shape_passed else None
                ),
            )
        )
    except Exception as exc:
        rows.append(exception_row("interplevel_2d_target", exc))

    try:
        missing_raw = wrf.interplevel(field, coordinate, 200.0)
        missing_type = qualified_type(missing_raw)
        missing = np.asarray(missing_raw)
        rows.append(
            checked_row(
                "interplevel_outside_missing",
                [
                    (
                        is_plain_ndarray(missing_raw),
                        f"default return must be exact numpy.ndarray, got {missing_type}",
                    ),
                    (missing.shape == (2, 2), f"expected shape (2, 2), got {missing.shape}"),
                    (
                        missing.shape == (2, 2) and np.isnan(missing).all(),
                        "outside-domain values must be NaN in the plain ndarray buffer",
                    ),
                ],
                expected_type="numpy.ndarray (exact; no subclasses)",
                actual_type=missing_type,
                shape=list(missing.shape),
                all_nan=bool(np.isnan(missing).all()),
            )
        )
    except Exception as exc:
        rows.append(exception_row("interplevel_outside_missing", exc))

    rows.append(
        checked_row(
            "wrffile_handle",
            [
                (
                    all(getattr(handle, name) > 0 for name in ("nx", "ny", "nz", "nt")),
                    "nx, ny, nz, and nt must all be positive",
                )
            ],
            dimensions={name: getattr(handle, name) for name in ("nx", "ny", "nz", "nt")},
        )
    )

    lat = None
    lon = None
    try:
        latlon_raw = wrf.latlon_coords(handle)
        pair_type_passed = type(latlon_raw) is tuple and len(latlon_raw) == 2
        if pair_type_passed:
            lat_raw, lon_raw = latlon_raw
            component_types = [qualified_type(lat_raw), qualified_type(lon_raw)]
            component_type_passed = is_plain_ndarray(lat_raw) and is_plain_ndarray(lon_raw)
            lat = np.asarray(lat_raw)
            lon = np.asarray(lon_raw)
        else:
            component_types = []
            component_type_passed = False
            try:
                lat_raw, lon_raw = latlon_raw
                component_types = [qualified_type(lat_raw), qualified_type(lon_raw)]
                lat = np.asarray(lat_raw)
                lon = np.asarray(lon_raw)
            except Exception:
                lat = np.empty(0)
                lon = np.empty(0)
        shape_passed = lat.shape == (handle.ny, handle.nx) and lon.shape == lat.shape
        finite_passed = bool(np.isfinite(lat).any() and np.isfinite(lon).any())
        rows.append(
            checked_row(
                "latlon_coords",
                [
                    (
                        pair_type_passed,
                        f"default return must be exact builtins.tuple of length 2, got {qualified_type(latlon_raw)}",
                    ),
                    (
                        component_type_passed,
                        f"latitude and longitude must be exact numpy.ndarray values, got {component_types}",
                    ),
                    (shape_passed, f"expected both shapes {(handle.ny, handle.nx)}, got {lat.shape} and {lon.shape}"),
                    (finite_passed, "latitude and longitude must contain finite values"),
                ],
                expected_type="tuple[numpy.ndarray, numpy.ndarray] (exact)",
                actual_type=qualified_type(latlon_raw),
                component_types=component_types,
                shape=list(lat.shape),
            )
        )
    except Exception as exc:
        rows.append(exception_row("latlon_coords", exc))

    if lat is None or lon is None or lat.shape != (handle.ny, handle.nx) or lon.shape != lat.shape:
        rows.append(
            {
                "id": "ll_to_xy_center",
                "passed": False,
                "failures": ["latlon_coords did not provide usable coordinate arrays"],
            }
        )
        rows.append(
            {
                "id": "ll_to_xy_fractional_boundary",
                "passed": False,
                "failures": ["latlon_coords did not provide usable coordinate arrays"],
            }
        )
    else:
        j = handle.ny // 2
        i = handle.nx // 2
        try:
            xy_raw = wrf.ll_to_xy(handle, float(lat[j, i]), float(lon[j, i]))
            xy_type = qualified_type(xy_raw)
            xy_array = np.asarray(xy_raw, dtype=float)
            shape_passed = xy_array.shape == (2,)
            finite_passed = shape_passed and np.isfinite(xy_array).all()
            location_passed = finite_passed and np.allclose(
                xy_array, [i, j], rtol=0.0, atol=0.25
            )
            rows.append(
                checked_row(
                    "ll_to_xy_center",
                    [
                        (
                            is_float_pair(xy_raw),
                            f"scalar default must return a tuple of float scalars, got {xy_type} with component types "
                            f"{[qualified_type(value) for value in xy_raw] if type(xy_raw) is tuple else 'n/a'}",
                        ),
                        (shape_passed, f"expected two coordinate values, got shape {xy_array.shape}"),
                        (finite_passed, "x/y coordinates must be finite"),
                        (location_passed, f"grid-center coordinate should be near {(i, j)}, got {xy_array.tolist()}"),
                    ],
                    expected_type="tuple[float | numpy.floating, float | numpy.floating]",
                    actual_type=xy_type,
                    component_types=(
                        [qualified_type(value) for value in xy_raw]
                        if type(xy_raw) is tuple
                        else []
                    ),
                    xy=xy_array.tolist(),
                )
            )
        except Exception as exc:
            rows.append(exception_row("ll_to_xy_center", exc))

        if handle.nx < 2:
            rows.append(
                {
                    "id": "ll_to_xy_fractional_boundary",
                    "passed": True,
                    "failures": [],
                    "skipped": True,
                    "reason": "fixture has fewer than two x grid points",
                }
            )
        else:
            edge_j = handle.ny // 2
            edge_lat = 0.5 * (float(lat[edge_j, 0]) + float(lat[edge_j, 1]))
            edge_lon = longitude_midpoint(
                float(lon[edge_j, 0]), float(lon[edge_j, 1])
            )
            if not np.isfinite([edge_lat, edge_lon]).all():
                rows.append(
                    {
                        "id": "ll_to_xy_fractional_boundary",
                        "passed": True,
                        "failures": [],
                        "skipped": True,
                        "reason": "fixture boundary coordinates are not finite",
                    }
                )
            else:
                try:
                    edge_raw = wrf.ll_to_xy(handle, edge_lat, edge_lon)
                    edge_type = qualified_type(edge_raw)
                    edge_xy = np.asarray(edge_raw, dtype=float)
                    shape_passed = edge_xy.shape == (2,)
                    finite_passed = shape_passed and np.isfinite(edge_xy).all()
                    expected_xy = np.array([0.5, float(edge_j)])
                    location_passed = finite_passed and np.allclose(
                        edge_xy, expected_xy, rtol=0.0, atol=0.25
                    )
                    fractional_passed = (
                        finite_passed and 0.1 < float(edge_xy[0]) < 0.9
                    )
                    rows.append(
                        checked_row(
                            "ll_to_xy_fractional_boundary",
                            [
                                (
                                    is_float_pair(edge_raw),
                                    f"scalar default must return a tuple of float scalars, got {edge_type}",
                                ),
                                (shape_passed, f"expected two coordinate values, got shape {edge_xy.shape}"),
                                (finite_passed, "boundary x/y coordinates must be finite"),
                                (
                                    location_passed,
                                    f"first-cell midpoint should map near {expected_xy.tolist()}, got {edge_xy.tolist()}",
                                ),
                                (
                                    fractional_passed,
                                    "first-cell midpoint x must remain fractional rather than round or clamp to a boundary",
                                ),
                            ],
                            expected_type="tuple[float | numpy.floating, float | numpy.floating]",
                            actual_type=edge_type,
                            target_latlon=[edge_lat, edge_lon],
                            expected_xy=expected_xy.tolist(),
                            xy=edge_xy.tolist(),
                        )
                    )
                except Exception as exc:
                    rows.append(exception_row("ll_to_xy_fractional_boundary", exc))

    try:
        projection = wrf.get_cartopy(handle)
        projection_type = qualified_type(projection)
        projection_result = checked_row(
            "get_cartopy",
            [
                (
                    hasattr(projection, "transform_points"),
                    f"projection {projection_type} has no transform_points method",
                )
            ],
            actual_type=projection_type,
        )
    except ImportError as exc:
        projection_result = {
            "id": "get_cartopy",
            "passed": False,
            "failures": [f"optional cartopy dependency missing: {exc}"],
            "reason": f"optional cartopy dependency missing: {exc}",
        }
    rows.append(projection_result)
    return rows


def probe_getvar_call(wrf: Any, handle: Any, call: dict[str, Any]) -> dict[str, Any]:
    """Probe one literal WRF-Runner getvar call without hiding its return type."""
    try:
        raw = wrf.getvar(handle, call["name"], **call["options"])
        actual_type = qualified_type(raw)
        array = np.asarray(raw)
        expected_shape = resolve_shape(call["shape"], handle)
        return checked_row(
            call["id"],
            [
                (
                    is_plain_ndarray(raw),
                    f"default getvar return must be exact numpy.ndarray, got {actual_type}",
                ),
                (
                    array.shape == expected_shape,
                    f"expected shape {expected_shape}, got {array.shape}",
                ),
            ],
            expected_type="numpy.ndarray (exact; no subclasses)",
            actual_type=actual_type,
            shape=list(array.shape),
            expected_shape=list(expected_shape),
            finite_fraction=float(np.mean(np.isfinite(array))),
        )
    except Exception as exc:
        return exception_row(call["id"], exc)


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if contract.get("schema_version") != 1:
        raise ParityError("unsupported WRF-Runner contract schema")
    contract_hash = sha256_bytes(canonical_json_bytes(contract))
    _, fixture_path, fixture_hash = load_fixture(
        args.fixtures, args.fixture_id, args.fixture
    )
    if distribution_version("wrf-rust") is None:
        raise ParityError("probe_wrf_runner.py must run in a wrf-rust wheel environment")
    if distribution_version("wrf-python") is not None:
        raise ParityError("use a separate candidate environment without wrf-python installed")

    import wrf

    handle = wrf.WrfFile(fixture_path)
    helper_rows = probe_helpers(wrf, handle)

    requested = set(csv_values(args.calls))
    known = {call["id"] for call in contract["getvar_calls"]}
    unknown = sorted(requested - known)
    if unknown:
        raise ParityError(f"unknown WRF-Runner call ids: {', '.join(unknown)}")

    getvar_rows = []
    if not args.skip_getvar:
        for call in contract["getvar_calls"]:
            if requested and call["id"] not in requested:
                continue
            if not requested and args.priority != "all" and call["priority"] != args.priority:
                continue
            getvar_rows.append(probe_getvar_call(wrf, handle, call))

    source_rows = []
    if args.consumer_root is not None:
        source_rows = verify_consumer_files(contract, args.consumer_root.resolve())

    failures = sum(not row["passed"] for row in helper_rows + getvar_rows + source_rows)
    report = {
        "report_schema_version": 1,
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash,
        "fixture_id": args.fixture_id,
        "fixture_sha256": fixture_hash,
        "wrf_rust_version": distribution_version("wrf-rust"),
        "failures": failures,
        "helper_probes": helper_rows,
        "getvar_probes": getvar_rows,
        "consumer_source_probes": source_rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(f"wrote {output} ({digest}); failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ParityError, OSError, RuntimeError, ValueError) as exc:
        print_error_and_exit(exc)
