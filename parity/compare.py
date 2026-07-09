#!/usr/bin/env python3
"""Compare wrf-rust and wrf-python parity bundles against explicit contracts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    ParityError,
    canonical_json_bytes,
    csv_values,
    load_bundle,
    load_contract,
    print_error_and_exit,
    select_contracts,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "contracts-v1.json")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--variables", action="append", help="comma-separated contract ids")
    parser.add_argument("--families", action="append", help="comma-separated families")
    parser.add_argument("--strict-diagnostic", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _finite_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _float32_reference_allowance(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return the directional half-ULP rounding cell around each reference value."""
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = reference.astype(np.float32)
    decoded = encoded.astype(np.float64)
    representable = decoded == reference
    if not np.all(representable):
        mismatch_count = int(np.count_nonzero(~representable))
        raise ParityError(
            "comparison.reference_precision='float32' requires finite reference "
            "values to be exactly representable as float32; "
            f"{mismatch_count} of {reference.size} finite values are not"
        )

    # The spacing immediately below and above a floating-point value can differ
    # at binade boundaries (for example, at 1.0). Use the neighbor on the side
    # of the candidate instead of a symmetric epsilon. At +/-MAX, nextafter in
    # one direction is infinity, so the finite opposite spacing supplies the
    # continuation of the terminal binade.
    direction = np.where(candidate < reference, -np.inf, np.inf).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        neighbor = np.nextafter(encoded, direction)
        fallback_neighbor = np.nextafter(encoded, -direction)
    spacing = np.abs(neighbor.astype(np.float64) - reference)
    fallback_spacing = np.abs(fallback_neighbor.astype(np.float64) - reference)
    spacing = np.where(np.isfinite(spacing), spacing, fallback_spacing)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ParityError("cannot determine a finite float32 ULP for the reference")
    return spacing * 0.5


def compare_array(
    candidate: np.ndarray,
    reference: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
        "passed": False,
    }
    if candidate.shape != reference.shape:
        result["reason"] = "shape_mismatch"
        return result

    candidate_missing = ~np.isfinite(candidate)
    reference_missing = ~np.isfinite(reference)
    missing_mismatch = candidate_missing ^ reference_missing
    overlap = ~(candidate_missing | reference_missing)
    count = int(candidate.size)
    overlap_count = int(np.count_nonzero(overlap))
    mismatch_count = int(np.count_nonzero(missing_mismatch))
    result.update(
        {
            "count": count,
            "finite_overlap_count": overlap_count,
            "candidate_missing_count": int(np.count_nonzero(candidate_missing)),
            "reference_missing_count": int(np.count_nonzero(reference_missing)),
            "missing_mismatch_count": mismatch_count,
        }
    )

    missing_policy = contract["missing_values"]
    missing_pass = (
        mismatch_count == 0
        if missing_policy["mask_must_match"]
        else mismatch_count / max(count, 1) <= missing_policy["max_mask_mismatch_fraction"]
    )

    if overlap_count:
        cand = candidate[overlap]
        ref = reference[overlap]
        absolute = np.abs(cand - ref)
        comparison = contract["comparison"]
        tolerance = comparison["tolerance"]
        allowed = float(tolerance["atol"]) + float(tolerance["rtol"]) * np.abs(ref)
        reference_precision = comparison.get("reference_precision")
        if reference_precision == "float32":
            quantization_allowance = _float32_reference_allowance(cand, ref)
            allowed = allowed + quantization_allowance
            result.update(
                {
                    "reference_precision": reference_precision,
                    "max_reference_quantization_allowance": _finite_number(
                        np.max(quantization_allowance)
                    ),
                }
            )
        within = absolute <= allowed
        denominator = np.maximum(np.abs(ref), float(tolerance.get("relative_floor", 0.0)))
        relative = np.divide(
            absolute,
            denominator,
            out=np.zeros_like(absolute),
            where=denominator > 0.0,
        )
        numeric_pass = bool(np.all(within))
        result.update(
            {
                "within_tolerance_count": int(np.count_nonzero(within)),
                "within_tolerance_fraction": float(np.mean(within)),
                "max_absolute_error": _finite_number(np.max(absolute)),
                "mean_absolute_error": _finite_number(np.mean(absolute)),
                "root_mean_square_error": _finite_number(
                    np.sqrt(np.mean(np.square(absolute)))
                ),
                "max_relative_error": _finite_number(np.max(relative)),
            }
        )
    else:
        numeric_pass = bool(missing_policy["allow_all_missing"])
        result["reason"] = "no_finite_overlap"

    result["passed"] = bool(missing_pass and numeric_pass)
    if not missing_pass:
        result["reason"] = "missing_mask_mismatch"
    elif not numeric_pass and "reason" not in result:
        result["reason"] = "outside_tolerance"
    return result


def validate_bundle_pair(
    reference_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    contract_hash: str,
) -> list[str]:
    expected_implementations = ("wrf-python", "wrf-rust")
    actual = (
        reference_metadata.get("implementation"),
        candidate_metadata.get("implementation"),
    )
    if actual != expected_implementations:
        raise ParityError(
            f"expected reference/candidate implementations {expected_implementations}, got {actual}"
        )
    for key in ("fixture_id", "fixture_sha256", "timeidx"):
        if reference_metadata.get(key) != candidate_metadata.get(key):
            raise ParityError(f"bundle provenance differs for {key}")
    for name, metadata in (
        ("reference", reference_metadata),
        ("candidate", candidate_metadata),
    ):
        if metadata.get("contract_sha256") != contract_hash:
            raise ParityError(f"{name} was extracted with a different contract document")

    selections: list[list[str]] = []
    for name, metadata in (
        ("reference", reference_metadata),
        ("candidate", candidate_metadata),
    ):
        raw = metadata.get("selected_contract_ids")
        if not isinstance(raw, list) or not raw or not all(
            isinstance(identifier, str) and identifier for identifier in raw
        ):
            raise ParityError(f"{name} bundle has an invalid contract selection")
        if len(raw) != len(set(raw)):
            raise ParityError(f"{name} bundle contract selection contains duplicates")
        selections.append(raw)
    if selections[0] != selections[1]:
        raise ParityError("reference and candidate bundles selected different contracts")
    return selections[0]


def main() -> None:
    args = parse_args()
    contract, contract_hash = load_contract(args.contract)
    requested_variables = csv_values(args.variables)
    requested_families = csv_values(args.families)
    reference, ref_metadata, ref_hash = load_bundle(args.reference)
    candidate, cand_metadata, cand_hash = load_bundle(args.candidate)
    bundle_ids = validate_bundle_pair(ref_metadata, cand_metadata, contract_hash)

    if requested_variables or requested_families:
        selected = select_contracts(contract, requested_variables, requested_families)
        requested_ids = [item["id"] for item in selected]
        if requested_ids != bundle_ids:
            raise ParityError(
                "comparator selection does not match the contracts recorded in both bundles"
            )
    else:
        contracts_by_id = {item["id"]: item for item in contract["variables"]}
        unknown = [identifier for identifier in bundle_ids if identifier not in contracts_by_id]
        if unknown:
            raise ParityError(
                f"bundle contains unknown contract ids: {', '.join(unknown)}"
            )
        selected = [contracts_by_id[identifier] for identifier in bundle_ids]

    rows: list[dict[str, Any]] = []
    failures = 0
    for item in selected:
        identifier = item["id"]
        mode = item["comparison"]["mode"]
        if mode == "contract_only":
            rows.append({"id": identifier, "mode": mode, "status": "not_compared"})
            continue
        if identifier not in reference or identifier not in candidate:
            row = {
                "id": identifier,
                "mode": mode,
                "status": "missing_array",
                "reference_present": identifier in reference,
                "candidate_present": identifier in candidate,
            }
            rows.append(row)
            if mode == "required" or args.strict_diagnostic:
                failures += 1
            continue

        metrics = compare_array(candidate[identifier], reference[identifier], item)
        status = "pass" if metrics["passed"] else "difference"
        row = {"id": identifier, "mode": mode, "status": status, **metrics}
        rows.append(row)
        if not metrics["passed"] and (mode == "required" or args.strict_diagnostic):
            failures += 1

    report = {
        "report_schema_version": 1,
        "contract_sha256": contract_hash,
        "fixture_id": ref_metadata["fixture_id"],
        "fixture_sha256": ref_metadata["fixture_sha256"],
        "timeidx": ref_metadata["timeidx"],
        "reference_bundle_sha256": ref_hash,
        "candidate_bundle_sha256": cand_hash,
        "reference_version": ref_metadata["implementation_version"],
        "candidate_version": cand_metadata["implementation_version"],
        "strict_diagnostic": args.strict_diagnostic,
        "failures": failures,
        "variables": rows,
    }

    print("id                             mode           status       max_abs       within")
    print("------------------------------ -------------- ------------ ------------- --------")
    for row in rows:
        max_abs = row.get("max_absolute_error")
        fraction = row.get("within_tolerance_fraction")
        max_text = "-" if max_abs is None else f"{max_abs:.6g}"
        fraction_text = "-" if fraction is None else f"{fraction:.3%}"
        print(
            f"{row['id']:<30} {row['mode']:<14} {row['status']:<12} "
            f"{max_text:<13} {fraction_text}"
        )

    if args.json_output:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(report) + b"\n")
        digest = sha256_file(output)
        output.with_name(output.name + ".sha256").write_text(
            f"{digest}  {output.name}\n", encoding="ascii"
        )
        print(f"wrote {output} ({digest})")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ParityError, OSError, RuntimeError, ValueError) as exc:
        print_error_and_exit(exc)
