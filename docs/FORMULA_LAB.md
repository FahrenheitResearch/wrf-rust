# Formula Lab

Formula Lab is a sandboxed, unit-aware expression engine for creating custom
WRF diagnostics. It is intended for interactive equation editors, notebooks,
repeatable research recipes, and rapid implementation of local equations from
the literature.

It is not Python `eval`. A formula cannot import modules, call Python, open a
file, use the network or shell, recurse, or execute an unbounded loop. Formula
source is parsed into a bounded expression graph and evaluated by Rust.

## Quick start

Compile once and evaluate at many times:

```python
from wrf import WrfFile, compile_formula

run = WrfFile("wrfout_d01_2011-04-27_21_00_00")
wind10 = compile_formula("sqrt(U10^2 + V10^2)")

print(wind10.dependencies)
print(wind10.explain())

for timeidx in range(run.nt):
    speed = wind10.evaluate(run, timeidx=timeidx)
```

`Formula.evaluate()` returns an owned, C-contiguous `float64` NumPy array by
default. Ask for metadata when the array will be saved, plotted, or published:

```python
result = wind10.evaluate(run, timeidx=0, return_metadata=True)
print(result.data.shape)
print(result.units)
print(result.axes)
print(result.provenance)
```

For a one-off expression:

```python
from wrf import evaluate_formula

speed = evaluate_formula(run, "sqrt(U10^2 + V10^2)", timeidx=0)
```

Repeated work should use `compile_formula()` so parsing, validation, dependency
analysis, and the execution plan are reused. Fields are deliberately resolved
again for each file/time evaluation.

## Preflight before evaluation

Compilation does not read a WRF file. The resulting `FormulaPlan` is designed
for BowEcho-style equation boxes and notebook preflight:

```python
formula = compile_formula("sqrt(U10^2 + V10^2)")
plan = formula.plan

print(plan.dependencies)
print(plan.units)
print(plan.shape)
print(plan.calculus_convention)
print(plan.estimated_operations_per_point)
print(plan.estimated_working_bytes)
print(plan.requirements)
print(plan.warnings)
```

`formula.explain()` renders the same information as readable text. A user
interface should show warnings and requirements before offering to evaluate a
large domain.

Some metadata cannot be final until a concrete WRF grid is opened. In that
case the plan reports what is inferable at compile time and the result metadata
records the resolved grid, axes, units, and provenance.

## Expression reference

Statements use newline or `;` separators. Zero or more assignments may precede
one final expression:

```text
wind10 = grid_vector(U10, V10)
magnitude(wind10)
```

The language supports `+`, `-`, `*`, `/`, `^`, comparisons, `and`, `or`, and
`not`. Power binds more tightly than unary minus, and implicit multiplication
is deliberately rejected. Names are ASCII case-insensitive for lookup.

Available bounded functions are:

- Selection and algebra: `where`, `min`, `max`, `clamp`, `abs`, `sqrt`,
  `pow`, `floor`, `ceil`, `round`, and `is_finite`.
- Transcendentals: `exp`, `ln`, `log`, `log10`, `sin`, `cos`, `tan`, `asin`,
  `acos`, `atan`, and `atan2`, with dimensional restrictions.
- Units: `quantity(value, "unit")` attaches units to dimensionless values;
  `convert(value, "unit")` requests a compatible output representation.
- Vectors: `grid_vector`, `earth_vector`, `component`, `magnitude`, and `dot`.
- Calculus: 2-D mass-grid `grad`, `div`, `curl`, and map-aware
  `laplacian`; `ddx`/`ddy` also accept 3-D scalar fields and differentiate
  along terrain-following model levels; `ddz` differentiates vertical columns
  against physical height. General 3-D `grad`/`div`/`curl`/Laplacian are
  rejected until full terrain-coordinate metric terms are implemented.
- Vertical operations: `integrate_z(field, height, lower, upper)`,
  `mean_z(field, height, lower, upper)`, and
  `interpolate_z(field, height, target)`.
- Time: `dt(expression)` uses actual adjacent valid times. Interior points use
  a centered difference; the boundary policy controls one-sided, missing, or
  error behavior at the first and last output.
- Radar-safe conversion: `dbz_to_z` and `z_to_dbz`; logarithmic dBZ is not
  silently treated as an ordinary dimensionless field.

Vertical bounds and targets must be length quantities, for example
`quantity(1000, "m")`. Vertical coordinate arguments must be monotonic physical
height fields on the same mass-grid geometry as the data. A plan requiring
adjacent times or physical height reports that requirement before evaluation.

## Recipes

`FormulaRecipe` is a strict data-only container:

```python
from wrf import (
    FormulaEvaluationOptions,
    FormulaParameter,
    FormulaReference,
    FormulaRecipe,
    compile_formula,
)

recipe = FormulaRecipe(
    source="sqrt(U10^2 + V10^2)",
    name="wind10",
    version="1.0.0",
    description="10 m scalar wind speed",
    expected_output_units="m s-1",
    references=(FormulaReference(citation="project-methods.md#wind10"),),
    parameters={
        "scale": FormulaParameter(
            default=1.0,
            units="1",
            minimum=0.0,
            maximum=10.0,
            description="dimensionless multiplier",
        )
    },
    evaluation_options=FormulaEvaluationOptions(
        boundary_policy="one_sided_second_order",
        missing_policy="propagate",
        non_finite_policy="error",
    ),
)

recipe.save("wind10.wrf-formula.json")
formula = compile_formula(FormulaRecipe.load("wind10.wrf-formula.json"))
```

The JSON schema identifier is `wrf-formula/v1`. Loading rejects:

- unknown or duplicate keys;
- unsupported schema versions;
- `NaN` and infinity;
- nonnumeric, nonfinite, nested, or excessive parameter values;
- oversized source or recipe files.

Recipes contain no pickle payloads, Python callbacks, import paths, or custom
JSON hooks. Treat a recipe as research data, not an executable plugin.

### Canonical `wrf-formula/v1` schema

Rust serde and Python read and write the same representation. Unknown fields
are rejected at every object level. The checked-in
[`wind10.wrf-formula.json`](examples/wind10.wrf-formula.json) file is the
round-trip fixture and copyable starting point. Rust applications should use
`Recipe::from_json_bytes` or `Recipe::from_json_reader` for untrusted input;
the bounded loaders enforce the 1 MiB recipe ceiling and compile-validation.

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema` | string | Required literal `wrf-formula/v1`. |
| `name`, `version`, `source` | string | Required recipe identity and expression source. |
| `description` | string | Human-readable scientific description. |
| `authors`, `tags` | string arrays | Attribution and discovery metadata. |
| `references` | object array | Exact `{citation, doi, url}` records; nullable DOI/URL. |
| `parameters` | object array | Exact `{name, units, default, minimum, maximum, description}` records. |
| `expected_output_units` | string or null | Compatibility assertion plus explicit conversion into that display unit; never a relabel-only operation. |
| `requirements` | object | Fields, maximum cadence/spacing, minimum vertical levels, and notes. |
| `evaluation_options` | object | Boundary, missing/nonfinite, and raw-field unit policies. |
| `resource_limits` | object or null | Exact lower-only execution ceilings; null uses host defaults. |

`resource_limits` contains the exact engine fields `max_source_bytes`,
`max_tokens`, `max_ast_nodes`, `max_ast_depth`, `max_identifier_bytes`,
`max_function_arity`, `max_assignments`, `max_dependencies`,
`max_output_elements`, `max_working_bytes`, `max_total_allocated_bytes`, and
`max_operations`. A recipe may lower any ceiling for reproducibility. It may
never raise an immutable host ceiling; attempted increases fail compilation
rather than being silently clamped.

`max_working_bytes` is currently the ceiling for any one f64 field buffer or
allocation, not a measured peak-live working set. `max_total_allocated_bytes`
is the conservative cumulative allocation meter across the evaluation; it can
reject a computation even when temporary buffers would later be released.

The exact `requirements` keys are `fields`, `maximum_cadence_seconds`,
`maximum_horizontal_spacing_m`, `minimum_vertical_levels`, and `notes`.
“Maximum spacing” is intentional: smaller grid spacing means finer horizontal
resolution, avoiding the ambiguity of a phrase such as “minimum resolution.”
Formula Lab preflights these requirements against the concrete resolver before
evaluating and copies the enforced requirement object into both the plan and
result provenance.

Evaluation parameters are finite numeric scalars:

```python
field = formula.evaluate(
    run,
    timeidx=0,
    parameters={"threshold": 15.0},
)
```

`FormulaParameter` declarations carry units, finite defaults, optional minimum
and maximum values, and descriptions. A plain numeric recipe value is shorthand
for a dimensionless parameter with that default. The native compiler rejects
undeclared parameters and values outside their declared ranges. Python
validates parameter names, scalar types, finiteness, bounds, and count before
crossing the extension boundary.

Evaluation semantics are recipe data too. Supported policies are:

| Recipe field | Values | Default |
| --- | --- | --- |
| `evaluation_options.boundary_policy` | `one_sided_second_order`, `missing`, `error` | `one_sided_second_order` |
| `evaluation_options.missing_policy` | `propagate`, `error`, `ignore_in_reductions` | `propagate` |
| `evaluation_options.non_finite_policy` | `propagate`, `error` | `propagate` |

Raw WRF fields without usable unit metadata require an explicit declaration:

```python
recipe = FormulaRecipe(
    source="RAW_RESEARCH_FIELD / quantity(1, \"s\")",
    evaluation_options=FormulaEvaluationOptions(
        variable_unit_overrides={"RAW_RESEARCH_FIELD": "m"},
    ),
)
```

The policies and resolved overrides are recorded in result provenance. They can
also be overridden for one evaluation through corresponding keyword arguments
to `Formula.evaluate()` or `evaluate_formula()`; recipe defaults remain
unchanged.

## Calculus conventions: an important scientific distinction

Formula Lab's generic calculus operators use a **mass-grid diagnostic
convention**. Inputs must already be identically shaped mass-grid fields; raw
staggered `U`, `V`, or `W` components are rejected rather than silently
destaggered. Horizontal derivatives use physical grid spacing and WRF map
factors. `ddx`/`ddy` on 3-D data follow terrain-following model levels, while
`ddz` uses an explicit or resolved physical-height column. Runtime boundary and
missing-data policies live in the recipe and result provenance.

That is not automatically the same algorithm as NCAR's historical raw
Arakawa-C-grid diagnostic kernels.

In particular:

- `avo` uses raw staggered `U` and `V`, Coriolis `F`, `MAPFAC_U`, `MAPFAC_V`,
  and `MAPFAC_M`, plus the NCAR boundary formulas.
- `pvo` builds on those raw-grid absolute-vorticity and pressure/theta
  derivatives with its own staggering and boundary semantics.
- WRF/NCAR updraft-helicity diagnostics use their documented vertical
  interpolation and integration conventions.

Therefore a generic expression such as a vertical component of `curl(wind)`
must not be advertised as bitwise, strict, or scientifically interchangeable
with `getvar(run, "avo")`. Use the existing named `avo`, `pvo`, and
updraft-helicity diagnostics when strict compatibility is required. Use
Formula Lab calculus when the intended research equation is explicitly the
mass-grid operator shown by `formula.plan.calculus_convention`.

This distinction is deliberate: both approaches are useful, but silently
mixing them would make published results difficult to reproduce.

## Errors and source spans

Formula failures use a stable exception hierarchy:

```text
FormulaError
|- FormulaSyntaxError
|- FormulaNameError
|- FormulaUnitError
|- FormulaShapeError
|- FormulaResourceError
`- FormulaEvaluationError
```

Native exceptions expose `kind`, `span`, `start`, `end`, and `notes` when
available. `start` and `end` are half-open UTF-8 byte offsets into the submitted
source. An equation editor can use them to underline the exact token that
failed without parsing an English error string.

Unit and shape mismatches are errors rather than implicit broadcasting or
conversion guesses. Missing fields, invalid grid metadata, nonfinite input
policies, domain errors, and resource-limit failures are likewise returned as
exceptions; malformed formula input must never panic the process.

## Arrays, fields, and output geometry

Formula Lab does not accept arbitrary object arrays or ragged values through
its public parameter interface. WRF fields are resolved natively and validated
before calculation. Outputs are always owned/safely retained NumPy `float64`
arrays in C order:

- scalar result: shape `()`;
- scalar field: WRF field axes such as `(bottom_top, south_north, west_east)`;
- vector field: a leading component axis followed by the field axes.

`FormulaResult.axes` must contain exactly one name per array dimension.

## Current execution boundaries

The first Formula Lab execution contract evaluates one WRF time at a time.
This keeps I/O and memory behavior explicit and prevents an equation box from
silently materializing an entire simulation. Iterate over `range(run.nt)` for
multiple times. A list/tuple of files is not yet accepted; multi-file temporal
operators belong in the planned streaming resolver so `dt` cannot silently
cross a missing or mismatched-domain boundary.

A `Formula` may be reused across files and times. Formula evaluations are
currently serialized process-wide because the WRF reader maintains a
single-time shared cache whose native locks do not yet have one proven order.
The native binding also retains the Python GIL. Use separate processes for
application-level parallelism in this release; safe released-GIL evaluation is
a future performance change, not an API change.

## What belongs outside the expression language

Formula Lab is for bounded algebra, reductions, interpolation, and local
differential operators. These require dedicated, reviewed kernels rather than
text-box syntax:

- Poisson, elliptic, pressure-decomposition, and other iterative global
  solvers with explicit boundary conditions and tolerances;
- trajectories and time integration;
- storm segmentation and object tracking;
- arbitrary loops, recursion, native libraries, Python callbacks, or plugins.

Such kernels can later be exposed as named, versioned operators with declared
inputs and provenance. Keeping them outside the basic expression language is
what lets Formula Lab remain safe and scientifically inspectable.

## Research reproducibility checklist

When publishing or sharing a custom diagnostic, save:

1. the JSON recipe and reference/DOI;
2. the canonical source from `formula.canonical_source`;
3. the full `FormulaPlan` and its calculus convention;
4. result units, axes, and provenance;
5. wrf-rust/Formula Lab version;
6. WRF run configuration, output cadence, and input hashes;
7. validation cases or expected statistics.

A fast equation is not automatically a valid atmospheric diagnostic. Formula
Lab makes assumptions visible and repeatable; researchers remain responsible
for choosing equations, scales, output cadence, and interpretations that their
simulation can support.
