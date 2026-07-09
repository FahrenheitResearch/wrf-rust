#!/usr/bin/env python3
"""Extract wrf-rust results into a provenance-bearing NPZ bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
    BUNDLE_SCHEMA_VERSION,
    ParityError,
    csv_values,
    distribution_version,
    load_contract,
    load_fixture,
    normalized_array,
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

    wrf_rust_version = distribution_version("wrf-rust")
    if wrf_rust_version is None:
        raise ParityError(
            "this extractor must run in the wrf-rust environment; install the wheel under "
            "test before extracting"
        )
    if distribution_version("wrf-python") is not None:
        raise ParityError(
            "wrf-python is installed in the candidate environment; use separate environments "
            "because both projects import as 'wrf'"
        )

    import wrf

    if not hasattr(wrf, "WrfFile"):
        raise ParityError(f"imported module is not wrf-rust: {wrf.__file__}")

    handle = wrf.WrfFile(fixture_path)
    arrays = {}
    records: dict[str, Any] = {}
    for item in selected:
        candidate = item["wrf_rust"]
        array = normalized_array(
            wrf.getvar(
                handle,
                candidate["name"],
                timeidx=timeidx,
                **candidate["options"],
            )
        )
        arrays[item["id"]] = array
        records[item["id"]] = {
            "status": "extracted",
            "source_name": candidate["name"],
            "source_options": candidate["options"],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    metadata = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "implementation": "wrf-rust",
        "implementation_version": wrf_rust_version,
        "module_path": str(Path(wrf.__file__).resolve()),
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
