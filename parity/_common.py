"""Shared, implementation-neutral helpers for the differential parity tools."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BUNDLE_SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
METADATA_KEY = "__metadata__"


class ParityError(RuntimeError):
    """A user-actionable parity harness error."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityError(f"{path} must contain a JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise ParityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def validate_contract_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ParityError(
            "unsupported contract schema_version "
            f"{document.get('schema_version')!r}; expected {CONTRACT_SCHEMA_VERSION}"
        )

    variables = document.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ParityError("contract must contain a non-empty variables array")

    seen: set[str] = set()
    required = {
        "id",
        "family",
        "wrf_rust",
        "wrf_python",
        "units",
        "dimensions",
        "component_order",
        "missing_values",
        "comparison",
        "intentional_deviations",
    }
    valid_modes = {"required", "diagnostic", "contract_only"}
    valid_extractors = {"getvar", "component", "none"}
    valid_reference_precisions = {"float32"}

    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            raise ParityError(f"variables[{index}] must be an object")
        missing = sorted(required - variable.keys())
        if missing:
            raise ParityError(
                f"variables[{index}] is missing required fields: {', '.join(missing)}"
            )

        identifier = variable["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ParityError(f"variables[{index}].id must be a non-empty string")
        if identifier in seen:
            raise ParityError(f"duplicate variable id: {identifier}")
        seen.add(identifier)

        rust = variable["wrf_rust"]
        upstream = variable["wrf_python"]
        comparison = variable["comparison"]
        if not isinstance(rust, dict) or not isinstance(rust.get("name"), str):
            raise ParityError(f"{identifier}: wrf_rust.name is required")
        if not isinstance(rust.get("aliases"), list):
            raise ParityError(f"{identifier}: wrf_rust.aliases must be an array")
        if not isinstance(rust.get("options"), dict):
            raise ParityError(f"{identifier}: wrf_rust.options must be an object")
        if not isinstance(upstream, dict) or upstream.get("extractor") not in valid_extractors:
            raise ParityError(
                f"{identifier}: wrf_python.extractor must be one of "
                f"{sorted(valid_extractors)}"
            )
        if not isinstance(upstream.get("options"), dict):
            raise ParityError(f"{identifier}: wrf_python.options must be an object")
        if not isinstance(comparison, dict) or comparison.get("mode") not in valid_modes:
            raise ParityError(
                f"{identifier}: comparison.mode must be one of {sorted(valid_modes)}"
            )
        reference_precision = comparison.get("reference_precision")
        if (
            reference_precision is not None
            and (
                not isinstance(reference_precision, str)
                or reference_precision not in valid_reference_precisions
            )
        ):
            raise ParityError(
                f"{identifier}: comparison.reference_precision must be one of "
                f"{sorted(valid_reference_precisions)} when present"
            )
        tolerance = comparison.get("tolerance")
        if not isinstance(tolerance, dict):
            raise ParityError(f"{identifier}: comparison.tolerance must be an object")
        for key in ("atol", "rtol"):
            value = tolerance.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise ParityError(f"{identifier}: tolerance.{key} must be non-negative")

        if comparison["mode"] != "contract_only" and upstream["extractor"] == "none":
            raise ParityError(
                f"{identifier}: {comparison['mode']} comparisons need a wrf-python extractor"
            )
        if upstream["extractor"] == "component":
            component = upstream.get("component")
            if not isinstance(component, dict) or not isinstance(component.get("index"), int):
                raise ParityError(
                    f"{identifier}: component extractor requires component.index"
                )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_contract_defaults(document: dict[str, Any]) -> dict[str, Any]:
    defaults = document.get("variable_defaults", {})
    variables = document.get("variables")
    if not isinstance(defaults, dict) or not isinstance(variables, list):
        return document
    resolved = dict(document)
    resolved["variables"] = [
        _deep_merge(defaults, item) if isinstance(item, dict) else item
        for item in variables
    ]
    return resolved


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    source_document = load_json(path)
    document = resolve_contract_defaults(source_document)
    validate_contract_document(document)
    return document, sha256_bytes(canonical_json_bytes(source_document))


def select_contracts(
    document: dict[str, Any],
    identifiers: Iterable[str] | None,
    families: Iterable[str] | None,
) -> list[dict[str, Any]]:
    requested_ids = set(identifiers or ())
    requested_families = set(families or ())
    known_ids = {item["id"] for item in document["variables"]}
    unknown = sorted(requested_ids - known_ids)
    if unknown:
        raise ParityError(f"unknown contract ids: {', '.join(unknown)}")

    selected = []
    for item in document["variables"]:
        if requested_ids and item["id"] not in requested_ids:
            continue
        if requested_families and item["family"] not in requested_families:
            continue
        selected.append(item)
    if not selected:
        raise ParityError("contract selection is empty")
    return selected


def load_fixture(
    fixture_catalog_path: Path,
    fixture_id: str,
    explicit_path: Path | None,
) -> tuple[dict[str, Any], Path, str]:
    catalog = load_json(fixture_catalog_path)
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("fixtures"), list):
        raise ParityError("fixture catalog must use schema_version 1 and contain fixtures")
    matches = [item for item in catalog["fixtures"] if item.get("id") == fixture_id]
    if len(matches) != 1:
        raise ParityError(f"fixture id {fixture_id!r} is not uniquely registered")
    fixture = matches[0]

    if explicit_path is None:
        env_name = fixture.get("path_env")
        raw_path = os.environ.get(env_name, "") if isinstance(env_name, str) else ""
        if not raw_path:
            raise ParityError(
                f"pass --fixture or set {env_name} for registered fixture {fixture_id!r}"
            )
        path = Path(raw_path)
    else:
        path = explicit_path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ParityError(f"fixture is not a regular file: {path}")

    expected_size = fixture.get("size_bytes")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ParityError(
            f"fixture size mismatch for {fixture_id}: expected {expected_size}, got {actual_size}"
        )
    expected_hash = fixture.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ParityError(f"fixture {fixture_id!r} lacks a valid SHA-256")
    actual_hash = sha256_file(path)
    if actual_hash.lower() != expected_hash.lower():
        raise ParityError(
            f"fixture SHA-256 mismatch for {fixture_id}: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    return fixture, path, actual_hash


def normalized_array(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        value = np.ma.asarray(value, dtype=np.float64).filled(np.nan)
    array = np.asarray(value, dtype=np.float64)
    return np.ascontiguousarray(array)


def apply_reference_transform(value: Any, contract: dict[str, Any]) -> np.ndarray:
    array = normalized_array(value)
    upstream = contract["wrf_python"]
    if upstream["extractor"] == "component":
        component = upstream["component"]
        axis = int(component.get("axis", 0))
        index = int(component["index"])
        array = np.take(array, index, axis=axis)
    scale = upstream.get("scale", 1.0)
    offset = upstream.get("offset", 0.0)
    if scale != 1.0 or offset != 0.0:
        array = array * float(scale) + float(offset)
    return np.ascontiguousarray(array, dtype=np.float64)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def encode_metadata(metadata: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(canonical_json_bytes(metadata), dtype=np.uint8)


def decode_metadata(value: np.ndarray) -> dict[str, Any]:
    try:
        decoded = json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ParityError(f"bundle metadata is malformed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ParityError("bundle metadata must be a JSON object")
    return decoded


def write_bundle(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: normalized_array(value) for name, value in arrays.items()}
    if METADATA_KEY in payload:
        raise ParityError(f"reserved array key: {METADATA_KEY}")
    payload[METADATA_KEY] = encode_metadata(metadata)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temp_name = handle.name
        np.savez_compressed(temp_name, **payload)
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    path = path.expanduser().resolve()
    digest = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if METADATA_KEY not in bundle.files:
                raise ParityError(f"{path} has no {METADATA_KEY} entry")
            metadata = decode_metadata(bundle[METADATA_KEY])
            arrays = {
                name: np.asarray(bundle[name], dtype=np.float64)
                for name in bundle.files
                if name != METADATA_KEY
            }
    except (OSError, ValueError) as exc:
        raise ParityError(f"cannot load bundle {path}: {exc}") from exc
    if metadata.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ParityError(f"unsupported bundle schema in {path}")
    return arrays, metadata, digest


def csv_values(raw: list[str] | None) -> list[str]:
    values: list[str] = []
    for item in raw or ():
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return values


def print_error_and_exit(exc: Exception) -> None:
    print(f"parity error: {exc}", file=sys.stderr)
    raise SystemExit(2)
