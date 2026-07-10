"""Safe, reusable custom diagnostics for WRF output.

Formula Lab deliberately exposes a small expression language rather than
Python's :func:`eval`.  Formulas cannot access Python objects, files, the
network, or the shell. Compilation parses, validates, and plans the expression
once; WRF fields are resolved afresh for each evaluation so file/time metadata
cannot leak between runs.

Examples
--------
Compile once and evaluate many times::

    >>> from wrf import WrfFile, compile_formula
    >>> formula = compile_formula("sqrt(U10^2 + V10^2)")
    >>> formula.dependencies
    ('U10', 'V10')
    >>> wind = formula.evaluate(WrfFile("wrfout_d01_..."), timeidx=0)  # doctest: +SKIP

Inspecting ``formula.plan`` does not read or evaluate a WRF file.  It is the
intended integration point for equation editors that need to show inferred
units, dependencies, calculus conventions, warnings, and cost before running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import numbers
import os
from pathlib import Path
import re
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import _wrf as _native


# These are native exception types so callers can catch the same hierarchy
# whether a failure occurs while compiling or evaluating.
FormulaError = _native.FormulaError
FormulaSyntaxError = _native.FormulaSyntaxError
FormulaNameError = _native.FormulaNameError
FormulaUnitError = _native.FormulaUnitError
FormulaShapeError = _native.FormulaShapeError
FormulaResourceError = _native.FormulaResourceError
FormulaEvaluationError = _native.FormulaEvaluationError


RECIPE_SCHEMA = "wrf-formula/v1"
_MAX_RECIPE_BYTES = 1_048_576
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_PARAMETERS = 256
_MAX_PARAMETER_NAME_CHARS = 128
_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RECIPE_KEYS = frozenset(
    {
        "schema",
        "name",
        "version",
        "description",
        "authors",
        "references",
        "tags",
        "source",
        "parameters",
        "expected_output_units",
        "requirements",
        "evaluation_options",
        "resource_limits",
    }
)
_FORMULA_EVALUATION_LOCK = threading.RLock()
_BOUNDARY_POLICIES = frozenset({"one_sided_second_order", "missing", "error"})
_MISSING_POLICIES = frozenset({"propagate", "error", "ignore_in_reductions"})
_NON_FINITE_POLICIES = frozenset({"propagate", "error"})
_HOST_RESOURCE_LIMITS = {
    "max_source_bytes": 64 * 1024,
    "max_tokens": 16_384,
    "max_ast_nodes": 16_384,
    "max_ast_depth": 128,
    "max_identifier_bytes": 128,
    "max_function_arity": 16,
    "max_assignments": 1024,
    "max_dependencies": 1024,
    "max_output_elements": 128 * 1024 * 1024,
    "max_working_bytes": 1024 * 1024 * 1024,
    "max_total_allocated_bytes": 4 * 1024 * 1024 * 1024,
    "max_operations": 4_000_000_000,
}


class FormulaRecipeError(FormulaError):
    """A recipe is malformed before Formula Lab compilation begins."""


def _validate_optional_text(value, name, *, max_chars=16_384):
    if value is None:
        return None
    if not isinstance(value, str):
        raise FormulaRecipeError(f"recipe {name!r} must be a string or null")
    if len(value) > max_chars:
        raise FormulaRecipeError(
            f"recipe {name!r} exceeds the {max_chars}-character limit"
        )
    return value


def _validate_parameters(parameters, *, context="parameters"):
    """Return an immutable, deterministic ``str -> finite float`` mapping."""
    if parameters is None:
        return MappingProxyType({})
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{context} must be a mapping of names to numeric scalars")
    if len(parameters) > _MAX_PARAMETERS:
        raise FormulaResourceError(
            f"{context} contains {len(parameters)} entries; maximum is {_MAX_PARAMETERS}"
        )

    normalized = {}
    normalized_names = set()
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise TypeError(f"{context} names must be strings, got {type(name).__name__}")
        if len(name) > _MAX_PARAMETER_NAME_CHARS or not _PARAMETER_NAME.fullmatch(name):
            raise ValueError(
                f"invalid formula parameter name {name!r}; expected a Python-style identifier"
            )
        # bool is a numbers.Real, but treating True as 1.0 hides recipe mistakes.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"parameter {name!r} must be a real numeric scalar, "
                f"got {type(value).__name__}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"parameter {name!r} must be finite, got {value!r}")
        folded = name.lower()
        if folded in normalized_names:
            raise ValueError(
                f"duplicate formula parameter {name!r}; names are ASCII case-insensitive"
            )
        normalized_names.add(folded)
        normalized[name] = numeric
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class FormulaParameter:
    """A declared scalar parameter with units, bounds, and a safe default."""

    default: float
    units: str = "1"
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def __post_init__(self):
        default = _finite_parameter_number(self.default, "default")
        minimum = (
            None
            if self.minimum is None
            else _finite_parameter_number(self.minimum, "minimum")
        )
        maximum = (
            None
            if self.maximum is None
            else _finite_parameter_number(self.maximum, "maximum")
        )
        units = _validate_optional_text(self.units, "parameter units", max_chars=256)
        description = _validate_optional_text(
            self.description, "parameter description", max_chars=4096
        )
        if not units:
            raise FormulaRecipeError("parameter units must not be empty")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise FormulaRecipeError("parameter minimum must not exceed maximum")
        if minimum is not None and default < minimum:
            raise FormulaRecipeError("parameter default is below its minimum")
        if maximum is not None and default > maximum:
            raise FormulaRecipeError("parameter default is above its maximum")
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "description", description)

    def to_dict(self):
        return {
            "units": self.units,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "description": self.description,
        }

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            allowed = {"default", "units", "minimum", "maximum", "description"}
            unknown = set(value) - allowed
            if unknown:
                names = ", ".join(repr(name) for name in sorted(unknown, key=str))
                raise FormulaRecipeError(f"unknown parameter specification key(s): {names}")
            if "default" not in value:
                raise FormulaRecipeError("parameter specification requires 'default'")
            return cls(
                default=value["default"],
                units=value.get("units", "1"),
                minimum=value.get("minimum"),
                maximum=value.get("maximum"),
                description=value.get("description", ""),
            )
        return cls(default=value)


def _finite_parameter_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise FormulaRecipeError(f"parameter {name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise FormulaRecipeError(f"parameter {name} must be finite")
    return result


def _validate_parameter_specs(parameters):
    if parameters is None:
        raise FormulaRecipeError("recipe parameters must not be null")
    if not isinstance(parameters, Mapping):
        raise FormulaRecipeError("recipe parameters must be a mapping")
    if len(parameters) > _MAX_PARAMETERS:
        raise FormulaResourceError(
            f"recipe parameters contain {len(parameters)} entries; maximum is {_MAX_PARAMETERS}"
        )
    result = {}
    normalized_names = set()
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise FormulaRecipeError("recipe parameter names must be strings")
        if len(name) > _MAX_PARAMETER_NAME_CHARS or not _PARAMETER_NAME.fullmatch(name):
            raise FormulaRecipeError(f"invalid formula parameter name {name!r}")
        folded = name.lower()
        if folded in normalized_names:
            raise FormulaRecipeError(
                f"duplicate recipe parameter {name!r}; names are ASCII case-insensitive"
            )
        normalized_names.add(folded)
        result[name] = FormulaParameter.from_value(value)
    return MappingProxyType(dict(sorted(result.items())))


def _normalize_policy(value, allowed, name):
    if not isinstance(value, str):
        raise FormulaRecipeError(f"recipe {name!r} must be a string")
    normalized = value.strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise FormulaRecipeError(f"invalid {name} {value!r}; expected one of {choices}")
    return normalized


def _validate_unit_overrides(overrides):
    if overrides is None:
        raise FormulaRecipeError("variable_unit_overrides must not be null")
    if not isinstance(overrides, Mapping):
        raise FormulaRecipeError("variable_unit_overrides must be a mapping")
    if len(overrides) > 1024:
        raise FormulaResourceError("variable_unit_overrides exceeds 1024 entries")
    result = {}
    normalized_names = set()
    for name, units in overrides.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise FormulaRecipeError("unit-override field names must be nonempty strings")
        if not isinstance(units, str) or not units.strip() or len(units) > 256:
            raise FormulaRecipeError(f"unit override for {name!r} must be a nonempty string")
        folded = name.lower()
        if folded in normalized_names:
            raise FormulaRecipeError(
                f"duplicate unit override {name!r}; names are ASCII case-insensitive"
            )
        normalized_names.add(folded)
        result[name] = units.strip()
    return MappingProxyType(dict(sorted(result.items())))


def _text_tuple(values, name, *, max_items=1024, max_chars=16_384):
    if values is None:
        raise FormulaRecipeError(f"recipe {name!r} must not be null")
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise FormulaRecipeError(f"recipe {name!r} must be a list of strings")
    if len(values) > max_items:
        raise FormulaResourceError(f"recipe {name!r} exceeds {max_items} items")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise FormulaRecipeError(f"recipe {name!r} items must be strings")
        if len(value) > max_chars:
            raise FormulaResourceError(
                f"recipe {name!r} item exceeds {max_chars} characters"
            )
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FormulaReference:
    """A citation, DOI, and/or URL attached to a portable recipe."""

    citation: str = ""
    doi: str | None = None
    url: str | None = None

    def __post_init__(self):
        citation = _validate_optional_text(
            self.citation, "reference citation", max_chars=8192
        )
        if citation is None:
            raise FormulaRecipeError("reference citation must be a string")
        object.__setattr__(self, "citation", citation)
        object.__setattr__(
            self, "doi", _validate_optional_text(self.doi, "reference doi", max_chars=512)
        )
        object.__setattr__(
            self, "url", _validate_optional_text(self.url, "reference url", max_chars=2048)
        )
        if not self.citation and not self.doi and not self.url:
            raise FormulaRecipeError("a formula reference must contain citation, doi, or url")

    def to_dict(self):
        return {"citation": self.citation, "doi": self.doi, "url": self.url}

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise FormulaRecipeError("formula references must be objects")
        unknown = set(value) - {"citation", "doi", "url"}
        if unknown:
            raise FormulaRecipeError(
                f"unknown formula reference key(s): {sorted(unknown, key=str)}"
            )
        return cls(
            citation=value.get("citation", ""),
            doi=value.get("doi"),
            url=value.get("url"),
        )


@dataclass(frozen=True, slots=True)
class FormulaRequirements:
    """Scientific data-resolution requirements declared by a recipe."""

    fields: tuple[str, ...] = ()
    maximum_cadence_seconds: float | None = None
    maximum_horizontal_spacing_m: float | None = None
    minimum_vertical_levels: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        fields = _text_tuple(self.fields, "requirements.fields")
        for name in fields:
            if len(name) > _MAX_PARAMETER_NAME_CHARS or not _PARAMETER_NAME.fullmatch(name):
                raise FormulaRecipeError(
                    f"invalid requirements.fields identifier {name!r}"
                )
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "notes", _text_tuple(self.notes, "requirements.notes"))
        for name in ("maximum_cadence_seconds", "maximum_horizontal_spacing_m"):
            value = getattr(self, name)
            if value is not None:
                value = _finite_parameter_number(value, name)
                if value <= 0.0:
                    raise FormulaRecipeError(f"{name} must be positive")
                object.__setattr__(self, name, value)
        if self.minimum_vertical_levels is not None:
            value = self.minimum_vertical_levels
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or int(value) <= 0:
                raise FormulaRecipeError("minimum_vertical_levels must be a positive integer")
            object.__setattr__(self, "minimum_vertical_levels", int(value))

    def to_dict(self):
        return {
            "fields": list(self.fields),
            "maximum_cadence_seconds": self.maximum_cadence_seconds,
            "maximum_horizontal_spacing_m": self.maximum_horizontal_spacing_m,
            "minimum_vertical_levels": self.minimum_vertical_levels,
            "notes": list(self.notes),
        }

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            raise FormulaRecipeError("recipe requirements must not be null")
        if not isinstance(value, Mapping):
            raise FormulaRecipeError("recipe requirements must be an object")
        allowed = {
            "fields",
            "maximum_cadence_seconds",
            "maximum_horizontal_spacing_m",
            "minimum_vertical_levels",
            "notes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise FormulaRecipeError(
                f"unknown recipe requirement key(s): {sorted(unknown, key=str)}"
            )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FormulaEvaluationOptions:
    """Portable missing, boundary, and unit-resolution semantics."""

    boundary_policy: str = "one_sided_second_order"
    missing_policy: str = "propagate"
    non_finite_policy: str = "propagate"
    variable_unit_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "boundary_policy",
            _normalize_policy(self.boundary_policy, _BOUNDARY_POLICIES, "boundary_policy"),
        )
        object.__setattr__(
            self,
            "missing_policy",
            _normalize_policy(self.missing_policy, _MISSING_POLICIES, "missing_policy"),
        )
        object.__setattr__(
            self,
            "non_finite_policy",
            _normalize_policy(
                self.non_finite_policy, _NON_FINITE_POLICIES, "non_finite_policy"
            ),
        )
        object.__setattr__(
            self,
            "variable_unit_overrides",
            _validate_unit_overrides(self.variable_unit_overrides),
        )

    def to_dict(self):
        return {
            "boundary_policy": self.boundary_policy,
            "missing_policy": self.missing_policy,
            "non_finite_policy": self.non_finite_policy,
            "variable_unit_overrides": dict(self.variable_unit_overrides),
        }

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            raise FormulaRecipeError("evaluation_options must not be null")
        if not isinstance(value, Mapping):
            raise FormulaRecipeError("evaluation_options must be an object")
        unknown = set(value) - {
            "boundary_policy",
            "missing_policy",
            "non_finite_policy",
            "variable_unit_overrides",
        }
        if unknown:
            raise FormulaRecipeError(
                f"unknown evaluation option key(s): {sorted(unknown, key=str)}"
            )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FormulaResourceLimits:
    """A reproducible lower-only ceiling bounded by immutable host maxima."""

    max_source_bytes: int = _HOST_RESOURCE_LIMITS["max_source_bytes"]
    max_tokens: int = _HOST_RESOURCE_LIMITS["max_tokens"]
    max_ast_nodes: int = _HOST_RESOURCE_LIMITS["max_ast_nodes"]
    max_ast_depth: int = _HOST_RESOURCE_LIMITS["max_ast_depth"]
    max_identifier_bytes: int = _HOST_RESOURCE_LIMITS["max_identifier_bytes"]
    max_function_arity: int = _HOST_RESOURCE_LIMITS["max_function_arity"]
    max_assignments: int = _HOST_RESOURCE_LIMITS["max_assignments"]
    max_dependencies: int = _HOST_RESOURCE_LIMITS["max_dependencies"]
    max_output_elements: int = _HOST_RESOURCE_LIMITS["max_output_elements"]
    max_working_bytes: int = _HOST_RESOURCE_LIMITS["max_working_bytes"]
    max_total_allocated_bytes: int = _HOST_RESOURCE_LIMITS[
        "max_total_allocated_bytes"
    ]
    max_operations: int = _HOST_RESOURCE_LIMITS["max_operations"]

    def __post_init__(self):
        for name, maximum in _HOST_RESOURCE_LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise FormulaRecipeError(f"resource_limits.{name} must be an integer")
            value = int(value)
            if value <= 0 or value > maximum:
                raise FormulaResourceError(
                    f"resource_limits.{name}={value} must be in 1..={maximum}"
                )
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {name: getattr(self, name) for name in _HOST_RESOURCE_LIMITS}

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise FormulaRecipeError("resource_limits must be an object or null")
        unknown = set(value) - set(_HOST_RESOURCE_LIMITS)
        if unknown:
            raise FormulaRecipeError(
                f"unknown resource limit key(s): {sorted(unknown, key=str)}"
            )
        missing = set(_HOST_RESOURCE_LIMITS) - set(value)
        if missing:
            raise FormulaRecipeError(
                f"canonical resource_limits missing key(s): {sorted(missing)}"
            )
        return cls(**value)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FormulaRecipeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise FormulaRecipeError(f"non-finite JSON number {value!r} is not allowed")


@dataclass(frozen=True, slots=True)
class FormulaRecipe:
    """A JSON-serializable, data-only Formula Lab recipe.

    ``parameters`` contains safe finite scalar defaults.  It never contains
    Python callbacks or executable code.  Callers may override defaults when
    evaluating, subject to declarations and ranges enforced by the native
    compiler.
    """

    source: str
    name: str = "custom_formula"
    version: str = "1"
    description: str = ""
    authors: tuple[str, ...] = ()
    references: tuple[FormulaReference, ...] = ()
    tags: tuple[str, ...] = ()
    parameters: Mapping[str, FormulaParameter | float] = field(default_factory=dict)
    expected_output_units: str | None = None
    requirements: FormulaRequirements = field(default_factory=FormulaRequirements)
    evaluation_options: FormulaEvaluationOptions = field(
        default_factory=FormulaEvaluationOptions
    )
    resource_limits: FormulaResourceLimits | None = None

    def __post_init__(self):
        if not isinstance(self.source, str):
            raise FormulaRecipeError("recipe 'source' must be a string")
        if len(self.source) > _MAX_SOURCE_BYTES:
            raise FormulaResourceError(
                f"formula source exceeds the {_MAX_SOURCE_BYTES}-byte host ceiling"
            )
        if not self.source or self.source.isspace():
            raise FormulaRecipeError("recipe 'source' must not be empty")
        object.__setattr__(
            self, "name", _validate_optional_text(self.name, "name", max_chars=256)
        )
        object.__setattr__(
            self, "version", _validate_optional_text(self.version, "version", max_chars=128)
        )
        if not self.name or not self.name.strip():
            raise FormulaRecipeError("recipe name must not be empty")
        if not self.version or not self.version.strip():
            raise FormulaRecipeError("recipe version must not be empty")
        object.__setattr__(
            self,
            "description",
            _validate_optional_text(self.description, "description"),
        )
        if self.description is None:
            raise FormulaRecipeError("recipe description must be a string")
        object.__setattr__(self, "authors", _text_tuple(self.authors, "authors"))
        object.__setattr__(self, "tags", _text_tuple(self.tags, "tags"))
        if isinstance(self.references, (str, bytes)) or not isinstance(
            self.references, (list, tuple)
        ):
            raise FormulaRecipeError("recipe references must be a list")
        if len(self.references) > 1024:
            raise FormulaResourceError("recipe references exceed 1024 items")
        object.__setattr__(
            self,
            "references",
            tuple(FormulaReference.from_value(value) for value in self.references),
        )
        object.__setattr__(
            self,
            "parameters",
            _validate_parameter_specs(self.parameters),
        )
        object.__setattr__(
            self,
            "expected_output_units",
            _validate_optional_text(
                self.expected_output_units, "expected_output_units", max_chars=256
            ),
        )
        object.__setattr__(
            self,
            "requirements",
            FormulaRequirements.from_value(self.requirements),
        )
        object.__setattr__(
            self,
            "evaluation_options",
            FormulaEvaluationOptions.from_value(self.evaluation_options),
        )
        object.__setattr__(
            self,
            "resource_limits",
            FormulaResourceLimits.from_value(self.resource_limits),
        )
        source_bytes = len(self.source.encode("utf-8"))
        source_limit = (
            self.resource_limits.max_source_bytes
            if self.resource_limits is not None
            else _MAX_SOURCE_BYTES
        )
        if source_bytes > source_limit:
            raise FormulaResourceError(
                f"formula source is {source_bytes} UTF-8 bytes; maximum is {source_limit}"
            )
        metadata_items = (
            len(self.authors)
            + len(self.references)
            + len(self.tags)
            + len(self.parameters)
            + len(self.requirements.fields)
            + len(self.requirements.notes)
            + len(self.evaluation_options.variable_unit_overrides)
        )
        if metadata_items > 1024:
            raise FormulaResourceError(
                f"recipe has {metadata_items} metadata items; maximum is 1024"
            )
        metadata_bytes = sum(
            len(value.encode("utf-8"))
            for value in (
                self.name,
                self.version,
                self.description,
                self.source,
                RECIPE_SCHEMA,
                self.expected_output_units or "",
                *self.authors,
                *self.tags,
                *self.requirements.fields,
                *self.requirements.notes,
            )
        )
        metadata_bytes += sum(
            len(reference.citation.encode("utf-8"))
            + (len(reference.doi.encode("utf-8")) if reference.doi else 0)
            + (len(reference.url.encode("utf-8")) if reference.url else 0)
            for reference in self.references
        )
        metadata_bytes += sum(
            len(name.encode("utf-8"))
            + len(specification.units.encode("utf-8"))
            + len(specification.description.encode("utf-8"))
            for name, specification in self.parameters.items()
        )
        metadata_bytes += sum(
            len(name.encode("utf-8")) + len(units.encode("utf-8"))
            for name, units in self.evaluation_options.variable_unit_overrides.items()
        )
        if metadata_bytes > 256 * 1024:
            raise FormulaResourceError(
                f"recipe metadata/source is {metadata_bytes} bytes; maximum is 262144"
            )

    def to_dict(self):
        """Return a plain, deterministic dictionary suitable for JSON."""
        result = {
            "schema": RECIPE_SCHEMA,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "authors": list(self.authors),
            "references": [reference.to_dict() for reference in self.references],
            "tags": list(self.tags),
            "source": self.source,
            "parameters": [
                {"name": name, **specification.to_dict()}
                for name, specification in self.parameters.items()
            ],
            "expected_output_units": self.expected_output_units,
            "requirements": self.requirements.to_dict(),
            "evaluation_options": self.evaluation_options.to_dict(),
            "resource_limits": (
                None if self.resource_limits is None else self.resource_limits.to_dict()
            ),
        }
        return result

    def to_json(self, *, indent=2):
        """Serialize this recipe without custom object hooks or executable data."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        ) + ("\n" if indent is not None else "")

    def save(self, path, *, indent=2):
        """Atomically write a UTF-8 JSON recipe to ``path``."""
        destination = Path(path)
        payload = self.to_json(indent=indent)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @classmethod
    def from_dict(cls, value):
        """Validate and construct a recipe from a plain JSON-style mapping."""
        if not isinstance(value, Mapping):
            raise FormulaRecipeError("formula recipe must be a JSON object")
        unknown = set(value) - _RECIPE_KEYS
        if unknown:
            names = ", ".join(repr(name) for name in sorted(unknown, key=str))
            raise FormulaRecipeError(f"unknown formula recipe key(s): {names}")
        missing = {"schema", "name", "version", "source"} - set(value)
        if missing:
            raise FormulaRecipeError(
                "formula recipe is missing required key(s): "
                + ", ".join(sorted(missing))
            )
        schema = value["schema"]
        if schema != RECIPE_SCHEMA:
            raise FormulaRecipeError(
                f"unsupported formula recipe schema {schema!r}; expected {RECIPE_SCHEMA!r}"
            )
        parameter_values = value.get("parameters", [])
        if not isinstance(parameter_values, list):
            raise FormulaRecipeError("recipe parameters must use canonical list form")
        parameters = {}
        for item in parameter_values:
            if not isinstance(item, Mapping):
                raise FormulaRecipeError("each parameter specification must be an object")
            unknown = set(item) - {
                "name",
                "units",
                "default",
                "minimum",
                "maximum",
                "description",
            }
            missing_parameter_keys = {"name", "units", "default"} - set(item)
            if unknown or missing_parameter_keys:
                raise FormulaRecipeError(
                    "invalid canonical parameter specification; unknown keys "
                    f"{sorted(unknown, key=str)}, missing keys "
                    f"{sorted(missing_parameter_keys)}"
                )
            name = item["name"]
            if not isinstance(name, str):
                raise FormulaRecipeError("parameter specification name must be a string")
            if name in parameters:
                raise FormulaRecipeError(f"duplicate parameter specification {name!r}")
            parameters[name] = FormulaParameter.from_value(
                {key: item[key] for key in item if key != "name"}
            )
        references = value.get("references", [])
        if not isinstance(references, list):
            raise FormulaRecipeError("recipe references must be a list")
        evaluation_options_value = value.get("evaluation_options", {})
        if "evaluation_options" in value:
            if not isinstance(evaluation_options_value, Mapping):
                raise FormulaRecipeError("evaluation_options must be an object")
            missing_options = {
                "boundary_policy",
                "missing_policy",
                "non_finite_policy",
            } - set(evaluation_options_value)
            if missing_options:
                raise FormulaRecipeError(
                    "canonical evaluation_options missing key(s): "
                    + ", ".join(sorted(missing_options))
                )
        return cls(
            source=value["source"],
            name=value["name"],
            version=value["version"],
            description=value.get("description", ""),
            authors=_text_tuple(value.get("authors", []), "authors"),
            references=tuple(FormulaReference.from_value(item) for item in references),
            tags=_text_tuple(value.get("tags", []), "tags"),
            parameters=parameters,
            expected_output_units=value.get("expected_output_units"),
            requirements=FormulaRequirements.from_value(value.get("requirements", {})),
            evaluation_options=FormulaEvaluationOptions.from_value(
                evaluation_options_value
            ),
            resource_limits=FormulaResourceLimits.from_value(
                value.get("resource_limits")
            ),
        )

    @classmethod
    def from_json(cls, text):
        """Parse a bounded, strict JSON recipe.

        Duplicate keys and JavaScript-style ``NaN``/``Infinity`` constants
        are rejected so the same recipe has one deterministic meaning.
        """
        if not isinstance(text, str):
            raise TypeError("recipe JSON must be a string")
        if len(text) > _MAX_RECIPE_BYTES:
            raise FormulaResourceError(
                f"recipe JSON exceeds the {_MAX_RECIPE_BYTES}-byte limit"
            )
        if len(text.encode("utf-8")) > _MAX_RECIPE_BYTES:
            raise FormulaResourceError(
                f"recipe JSON exceeds the {_MAX_RECIPE_BYTES}-byte limit"
            )
        try:
            value = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except FormulaError:
            raise
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise FormulaRecipeError(f"invalid formula recipe JSON: {exc}") from exc
        return cls.from_dict(value)

    @classmethod
    def load(cls, path):
        """Load a bounded UTF-8 JSON recipe from a local file."""
        source = Path(path)
        with source.open("rb") as stream:
            payload = stream.read(_MAX_RECIPE_BYTES + 1)
        if len(payload) > _MAX_RECIPE_BYTES:
            raise FormulaResourceError(
                f"recipe file exceeds the {_MAX_RECIPE_BYTES}-byte limit"
            )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise FormulaRecipeError(f"recipe file is not valid UTF-8: {exc}") from exc
        return cls.from_json(text)


@dataclass(frozen=True, slots=True)
class FormulaPlan:
    """A compile-time explanation of a formula without reading WRF data."""

    source: str
    canonical_source: str
    dependencies: tuple[str, ...]
    functions: tuple[str, ...] = ()
    assignments: tuple[str, ...] = ()
    units: str | None = None
    shape: Any = None
    description: str | None = None
    calculus_convention: str | None = None
    estimated_operations_per_point: int | None = None
    estimated_working_bytes: int | None = None
    ast_nodes: int | None = None
    ast_depth: int | None = None
    requirements: tuple[str, ...] = ()
    recipe_requirements: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def _from_native(cls, source, native_plan):
        if native_plan is None:
            native_plan = {}
        if not isinstance(native_plan, Mapping):
            raise FormulaEvaluationError(
                "native Formula Lab returned an invalid compile plan"
            )
        canonical = native_plan.get("canonical_source", source)
        return cls(
            source=source,
            canonical_source=str(canonical),
            dependencies=tuple(str(v) for v in native_plan.get("dependencies", ())),
            functions=tuple(str(v) for v in native_plan.get("functions", ())),
            assignments=tuple(str(v) for v in native_plan.get("assignments", ())),
            units=_optional_native_text(native_plan.get("units")),
            shape=_freeze_native_value(native_plan.get("shape")),
            description=_optional_native_text(native_plan.get("description")),
            calculus_convention=_optional_native_text(
                native_plan.get("calculus_convention")
            ),
            estimated_operations_per_point=_optional_native_int(
                native_plan.get("estimated_operations_per_point")
            ),
            estimated_working_bytes=_optional_native_int(
                native_plan.get("estimated_working_bytes")
            ),
            ast_nodes=_optional_native_int(native_plan.get("ast_nodes")),
            ast_depth=_optional_native_int(native_plan.get("ast_depth")),
            requirements=tuple(str(v) for v in native_plan.get("requirements", ())),
            recipe_requirements=(
                None
                if native_plan.get("recipe_requirements") is None
                else _freeze_native_value(native_plan["recipe_requirements"])
            ),
            warnings=tuple(str(v) for v in native_plan.get("warnings", ())),
        )

    def to_dict(self):
        """Return JSON-friendly plan metadata for UIs and notebooks."""
        return {
            "source": self.source,
            "canonical_source": self.canonical_source,
            "dependencies": list(self.dependencies),
            "functions": list(self.functions),
            "assignments": list(self.assignments),
            "units": self.units,
            "shape": _json_native_value(self.shape),
            "description": self.description,
            "calculus_convention": self.calculus_convention,
            "estimated_operations_per_point": self.estimated_operations_per_point,
            "estimated_working_bytes": self.estimated_working_bytes,
            "ast_nodes": self.ast_nodes,
            "ast_depth": self.ast_depth,
            "requirements": _json_native_value(self.requirements),
            "recipe_requirements": _json_native_value(self.recipe_requirements),
            "warnings": _json_native_value(self.warnings),
        }

    def explain(self):
        """Return a concise, human-readable preflight explanation."""
        lines = [f"Formula: {self.canonical_source}"]
        lines.append(
            "Dependencies: " + (", ".join(self.dependencies) or "none")
        )
        if self.units:
            lines.append(f"Inferred units: {self.units}")
        if self.shape is not None:
            lines.append(f"Inferred shape: {self.shape}")
        if self.calculus_convention:
            lines.append(f"Calculus: {self.calculus_convention}")
        if self.estimated_operations_per_point is not None:
            lines.append(
                "Estimated scalar operations per point: "
                f"{self.estimated_operations_per_point}"
            )
        if self.estimated_working_bytes is not None:
            lines.append(
                f"Estimated working memory: {self.estimated_working_bytes} bytes"
            )
        if self.ast_nodes is not None:
            lines.append(f"Expression graph: {self.ast_nodes} nodes, depth {self.ast_depth}")
        if self.requirements:
            lines.append("Requirements: " + "; ".join(self.requirements))
        if self.recipe_requirements:
            lines.append(
                "Recipe requirements: "
                + json.dumps(
                    _json_native_value(self.recipe_requirements),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FormulaResult:
    """An ndarray plus the metadata needed to interpret and reproduce it."""

    data: np.ndarray
    units: str | None
    description: str | None
    formula_source: str
    canonical_source: str
    dependencies: tuple[str, ...]
    timeidx: int
    axes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise FormulaEvaluationError("native Formula Lab result is not an ndarray")
        if self.data.dtype != np.dtype(np.float64):
            raise FormulaEvaluationError(
                "native Formula Lab result must have float64 dtype, "
                f"got {self.data.dtype}"
            )
        if not self.data.flags.c_contiguous:
            raise FormulaEvaluationError("Formula Lab result is not C-contiguous")
        if len(self.axes) != self.data.ndim:
            raise FormulaEvaluationError(
                f"Formula Lab returned {len(self.axes)} axis labels for a "
                f"{self.data.ndim}-D array"
            )
        if not isinstance(self.provenance, Mapping):
            raise FormulaEvaluationError("Formula Lab provenance is not a mapping")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(
                {str(key): _freeze_native_value(value) for key, value in self.provenance.items()}
            ),
        )

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.data, dtype=dtype)
        if copy is True:
            return array.copy()
        if copy is False and not np.shares_memory(array, self.data):
            raise ValueError("dtype conversion requires a copy")
        return array


def _optional_native_text(value):
    return None if value is None else str(value)


def _optional_native_int(value):
    return None if value is None else int(value)


def _freeze_native_value(value):
    """Recursively detach plan/provenance metadata from mutable native containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_native_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_native_value(item) for item in value)
    return value


def _json_native_value(value):
    """Convert frozen metadata back to plain JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): _json_native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native_value(item) for item in value]
    return value


def _native_attr(value, name, default=None):
    attribute = getattr(value, name, default)
    return attribute() if callable(attribute) else attribute


class Formula:
    """A compiled, reusable Formula Lab diagnostic.

    Construct formulas with :func:`compile_formula`; direct construction is
    intentionally private so every instance has passed native syntax, name,
    parameter-unit, recipe, and resource validation. Field-dependent shape and
    unit checks occur when a concrete WRF time is evaluated.
    """

    __slots__ = ("_native", "_recipe", "_plan")

    def __init__(self, native, recipe, *, _token=None):
        if _token is not _FORMULA_TOKEN:
            raise TypeError("use compile_formula() to construct Formula objects")
        self._native = native
        self._recipe = recipe
        native_plan = _native_attr(native, "plan", {})
        self._plan = FormulaPlan._from_native(recipe.source, native_plan)

    @property
    def source(self):
        return self._recipe.source

    @property
    def canonical_source(self):
        native_value = _native_attr(self._native, "canonical_source", None)
        return str(native_value) if native_value is not None else self._plan.canonical_source

    @property
    def dependencies(self):
        native_value = _native_attr(self._native, "dependencies", None)
        if native_value is None:
            return self._plan.dependencies
        return tuple(str(value) for value in native_value)

    @property
    def plan(self):
        return self._plan

    def explain(self):
        """Return the compile plan as readable preflight text."""
        return self._plan.explain()

    def to_recipe(self):
        """Return the immutable recipe used to compile this formula."""
        return self._recipe

    def evaluate(
        self,
        wrffile,
        timeidx=0,
        parameters=None,
        *,
        return_metadata=False,
        boundary_policy=None,
        missing_policy=None,
        non_finite_policy=None,
        variable_unit_overrides=None,
    ):
        """Evaluate this compiled formula for one WRF time.

        The default return value is an owned, C-contiguous ``float64`` NumPy
        array.  Set ``return_metadata=True`` for :class:`FormulaResult`.
        Formula objects may be reused. Formula evaluations are serialized
        process-wide in this release while the WRF reader's cache lock order is
        being hardened for released-GIL concurrency.
        """
        # Lazy import avoids making the compatibility module depend on this
        # additive API during initialization.
        from . import ALL_TIMES, _ensure_wrffile, _normalize_single_timeidx

        if isinstance(wrffile, (list, tuple)):
            raise NotImplementedError(
                "Formula Lab currently requires one WRF file handle; multi-file "
                "temporal resolution needs the future streaming resolver"
            )
        wf = _ensure_wrffile(wrffile)
        if timeidx is not ALL_TIMES and (
            isinstance(timeidx, (bool, np.bool_))
            or not isinstance(timeidx, (int, np.integer))
        ):
            raise TypeError("timeidx must be an integer or ALL_TIMES")
        resolved_timeidx = _normalize_single_timeidx(wf, timeidx)
        if resolved_timeidx is ALL_TIMES:
            raise NotImplementedError(
                "Formula Lab currently evaluates one time at a time; iterate over "
                "range(wrffile.nt) to keep memory use explicit"
            )

        merged = {
            name: specification.default
            for name, specification in self._recipe.parameters.items()
        }
        for name, value in _validate_parameters(parameters).items():
            canonical = next(
                (
                    declared
                    for declared in self._recipe.parameters
                    if declared.lower() == name.lower()
                ),
                name,
            )
            merged[canonical] = value
        recipe_options = self._recipe.evaluation_options
        boundary_policy = _normalize_policy(
            recipe_options.boundary_policy if boundary_policy is None else boundary_policy,
            _BOUNDARY_POLICIES,
            "boundary_policy",
        )
        missing_policy = _normalize_policy(
            recipe_options.missing_policy if missing_policy is None else missing_policy,
            _MISSING_POLICIES,
            "missing_policy",
        )
        non_finite_policy = _normalize_policy(
            recipe_options.non_finite_policy
            if non_finite_policy is None
            else non_finite_policy,
            _NON_FINITE_POLICIES,
            "non_finite_policy",
        )
        overrides = dict(recipe_options.variable_unit_overrides)
        if variable_unit_overrides is not None:
            for name, units in _validate_unit_overrides(variable_unit_overrides).items():
                canonical = next(
                    (existing for existing in overrides if existing.lower() == name.lower()),
                    name,
                )
                overrides[canonical] = units
        # The WRF reader currently has an inverted cache/cache-time lock order.
        # Keep one process-wide evaluation lock and retain the GIL in the native
        # binding until that order is fixed. This also covers multiple Python
        # wrappers around the same native WrfFile handle.
        with _FORMULA_EVALUATION_LOCK:
            native_result = self._native.evaluate(
                wf._inner,
                timeidx=int(resolved_timeidx),
                parameters=merged,
                boundary_policy=boundary_policy,
                missing_policy=missing_policy,
                non_finite_policy=non_finite_policy,
                variable_unit_overrides=overrides,
            )
        result = self._coerce_result(native_result, int(resolved_timeidx))
        return result if return_metadata else result.data

    def _coerce_result(self, native_result, timeidx):
        if not isinstance(native_result, tuple) or len(native_result) != 2:
            raise FormulaEvaluationError(
                "native Formula Lab result must be an (ndarray, metadata) pair"
            )
        data, metadata = native_result
        if not isinstance(metadata, Mapping):
            raise FormulaEvaluationError("native Formula Lab metadata is not a mapping")
        axes = tuple(str(value) for value in metadata.get("axes", ()))
        provenance = metadata.get("provenance", {})
        return FormulaResult(
            data=data,
            units=_optional_native_text(metadata.get("units", self._plan.units)),
            description=_optional_native_text(
                metadata.get("description", self._recipe.description)
            ),
            formula_source=self.source,
            canonical_source=self.canonical_source,
            dependencies=self.dependencies,
            timeidx=timeidx,
            axes=axes,
            provenance=provenance,
        )

    def __repr__(self):
        return f"Formula({self.canonical_source!r})"


_FORMULA_TOKEN = object()


def compile_formula(source_or_recipe):
    """Compile source or a :class:`FormulaRecipe` without reading WRF data."""
    if isinstance(source_or_recipe, Formula):
        return source_or_recipe
    if isinstance(source_or_recipe, FormulaRecipe):
        recipe = source_or_recipe
    elif isinstance(source_or_recipe, str):
        recipe = FormulaRecipe(source_or_recipe)
    else:
        raise TypeError("compile_formula() expects str, FormulaRecipe, or Formula")

    native = _native._compile_formula(recipe.to_dict())
    return Formula(native, recipe, _token=_FORMULA_TOKEN)


def evaluate_formula(
    wrffile,
    source_or_formula,
    timeidx=0,
    parameters=None,
    *,
    return_metadata=False,
    boundary_policy=None,
    missing_policy=None,
    non_finite_policy=None,
    variable_unit_overrides=None,
):
    """Compile if necessary and evaluate a custom diagnostic for one time."""
    formula = compile_formula(source_or_formula)
    return formula.evaluate(
        wrffile,
        timeidx=timeidx,
        parameters=parameters,
        return_metadata=return_metadata,
        boundary_policy=boundary_policy,
        missing_policy=missing_policy,
        non_finite_policy=non_finite_policy,
        variable_unit_overrides=variable_unit_overrides,
    )


def load_formula_recipe(path):
    """Load a strict, data-only JSON recipe from ``path``."""
    return FormulaRecipe.load(path)


__all__ = [
    "Formula",
    "FormulaPlan",
    "FormulaParameter",
    "FormulaReference",
    "FormulaRequirements",
    "FormulaEvaluationOptions",
    "FormulaResourceLimits",
    "FormulaRecipe",
    "FormulaResult",
    "FormulaError",
    "FormulaSyntaxError",
    "FormulaNameError",
    "FormulaUnitError",
    "FormulaShapeError",
    "FormulaResourceError",
    "FormulaEvaluationError",
    "FormulaRecipeError",
    "compile_formula",
    "evaluate_formula",
    "load_formula_recipe",
]
