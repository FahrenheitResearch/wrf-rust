#!/usr/bin/env python3
"""Exercise the concrete WRF-Runner/wrf-rust extension contract."""

from __future__ import annotations

import argparse
import json
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
    scalar = np.asarray(wrf.interplevel(field, coordinate, 750.0))
    rows.append(
        {
            "id": "interplevel_scalar",
            "passed": scalar.shape == (2, 2) and np.allclose(scalar, 50.0),
            "shape": list(scalar.shape),
            "value": float(scalar[0, 0]),
        }
    )

    coordinate_2d = np.broadcast_to(coordinate, (2, 2, 2))
    field_2d = np.broadcast_to(field, (2, 2, 2))
    targets = np.array([[900.0, 800.0], [700.0, 600.0]])
    expected = np.array([[20.0, 40.0], [60.0, 80.0]])
    level_2d = np.asarray(wrf.interplevel(field_2d, coordinate_2d, targets))
    rows.append(
        {
            "id": "interplevel_2d_target",
            "passed": level_2d.shape == (2, 2) and np.allclose(level_2d, expected),
            "shape": list(level_2d.shape),
            "max_absolute_error": float(np.max(np.abs(level_2d - expected))),
        }
    )

    rows.append(
        {
            "id": "wrffile_handle",
            "passed": all(getattr(handle, name) > 0 for name in ("nx", "ny", "nz", "nt")),
            "dimensions": {name: getattr(handle, name) for name in ("nx", "ny", "nz", "nt")},
        }
    )

    lat, lon = wrf.latlon_coords(handle)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    latlon_pass = (
        lat.shape == (handle.ny, handle.nx)
        and lon.shape == lat.shape
        and np.isfinite(lat).any()
        and np.isfinite(lon).any()
    )
    rows.append(
        {"id": "latlon_coords", "passed": bool(latlon_pass), "shape": list(lat.shape)}
    )

    j = handle.ny // 2
    i = handle.nx // 2
    xy = wrf.ll_to_xy(handle, float(lat[j, i]), float(lon[j, i]))
    xy_array = np.asarray(xy, dtype=float)
    ll_pass = xy_array.shape == (2,) and abs(xy_array[0] - i) <= 1.0 and abs(xy_array[1] - j) <= 1.0
    rows.append(
        {"id": "ll_to_xy_center", "passed": bool(ll_pass), "xy": xy_array.tolist()}
    )

    try:
        projection = wrf.get_cartopy(handle)
        projection_result = {
            "id": "get_cartopy",
            "passed": hasattr(projection, "transform_points"),
            "type": f"{type(projection).__module__}.{type(projection).__name__}",
        }
    except ImportError as exc:
        projection_result = {
            "id": "get_cartopy",
            "passed": False,
            "reason": f"optional cartopy dependency missing: {exc}",
        }
    rows.append(projection_result)
    return rows


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
            try:
                array = np.asarray(wrf.getvar(handle, call["name"], **call["options"]))
                expected_shape = resolve_shape(call["shape"], handle)
                getvar_rows.append(
                    {
                        "id": call["id"],
                        "passed": array.shape == expected_shape,
                        "shape": list(array.shape),
                        "expected_shape": list(expected_shape),
                        "finite_fraction": float(np.mean(np.isfinite(array))),
                    }
                )
            except Exception as exc:
                getvar_rows.append(
                    {"id": call["id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"}
                )

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
