#!/usr/bin/env python3
"""Extract upstream wrf-python results into a provenance-bearing NPZ bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import (
    BUNDLE_SCHEMA_VERSION,
    ParityError,
    apply_reference_transform,
    csv_values,
    distribution_version,
    load_contract,
    load_fixture,
    print_error_and_exit,
    runtime_provenance,
    select_contracts,
    write_bundle,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "contracts-v1.json")
    parser.add_argument("--fixtures", type=Path, default=here / "fixtures-v1.json")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--timeidx", type=int)
    parser.add_argument("--variables", action="append", help="comma-separated contract ids")
    parser.add_argument("--families", action="append", help="comma-separated families")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract, contract_hash = load_contract(args.contract)
    selected = select_contracts(
        contract, csv_values(args.variables), csv_values(args.families)
    )
    fixture, fixture_path, fixture_hash = load_fixture(
        args.fixtures, args.fixture_id, args.fixture
    )
    timeidx = fixture.get("default_timeidx", 0) if args.timeidx is None else args.timeidx

    wrf_python_version = distribution_version("wrf-python")
    if wrf_python_version is None:
        raise ParityError(
            "this extractor must run in the upstream wrf-python environment; "
            "the wrf-python distribution is not installed"
        )
    if distribution_version("wrf-rust") is not None:
        raise ParityError(
            "wrf-rust is installed in the reference environment; use separate environments "
            "because both projects import as 'wrf'"
        )

    import netCDF4  # Imported only after the environment guard.
    import wrf

    arrays = {}
    records: dict[str, Any] = {}
    source_cache: dict[str, Any] = {}
    with netCDF4.Dataset(fixture_path, mode="r") as dataset:
        for item in selected:
            upstream = item["wrf_python"]
            if upstream["extractor"] == "none":
                records[item["id"]] = {
                    "status": "not_available",
                    "reason": upstream["reason"],
                }
                continue

            source_name = upstream["name"]
            options = upstream["options"]
            cache_key = json.dumps(
                [source_name, options], sort_keys=True, separators=(",", ":")
            )
            if cache_key not in source_cache:
                source_cache[cache_key] = wrf.getvar(
                    dataset,
                    source_name,
                    timeidx=timeidx,
                    meta=False,
                    **options,
                )
            array = apply_reference_transform(source_cache[cache_key], item)
            arrays[item["id"]] = array
            records[item["id"]] = {
                "status": "extracted",
                "source_name": source_name,
                "source_options": options,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }

    metadata = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "implementation": "wrf-python",
        "implementation_version": wrf_python_version,
        "module_path": str(Path(wrf.__file__).resolve()),
        "netcdf4_version": netCDF4.__version__,
        "fixture_id": args.fixture_id,
        "fixture_sha256": fixture_hash,
        "fixture_size_bytes": fixture_path.stat().st_size,
        "timeidx": timeidx,
        "contract_sha256": contract_hash,
        "contract_version": contract["contract_version"],
        "selected_contract_ids": [item["id"] for item in selected],
        "variables": records,
        "runtime": runtime_provenance(),
    }
    digest = write_bundle(args.output, arrays, metadata)
    print(f"wrote {args.output} ({digest})")


if __name__ == "__main__":
    try:
        main()
    except (ParityError, OSError, RuntimeError, ValueError) as exc:
        print_error_and_exit(exc)
