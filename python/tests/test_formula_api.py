"""Formula Lab's Python contract and adversarial boundary tests.

These tests intentionally use a fake compiled native object for evaluation so
wrapper failures can be isolated from WRF fixture data.  Native parser and
evaluator behavior is covered inside ``wrf-formula``.
"""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import wrf
from wrf.formula import (
    Formula,
    FormulaEvaluationError,
    FormulaEvaluationOptions,
    FormulaParameter,
    FormulaReference,
    FormulaRecipe,
    FormulaRecipeError,
    FormulaResourceError,
    FormulaResourceLimits,
    FormulaRequirements,
    FormulaResult,
    compile_formula,
    evaluate_formula,
)


class _FakeCompiled:
    canonical_source = "sqrt(U10 ^ 2 + V10 ^ 2)"
    dependencies = ("U10", "V10")

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def plan(self):
        return {
            "canonical_source": self.canonical_source,
            "dependencies": list(self.dependencies),
            "functions": ["sqrt"],
            "assignments": [],
            "units": "m s-1",
            "shape": [2, 3],
            "calculus_convention": "WRF mass grid, second-order finite differences",
            "estimated_operations_per_point": 6,
            "estimated_working_bytes": 96,
            "requirements": ["U10", "V10"],
            "recipe_requirements": {
                "fields": ["U10", "V10"],
                "maximum_cadence_seconds": None,
                "maximum_horizontal_spacing_m": None,
                "minimum_vertical_levels": None,
                "notes": [],
            },
            "warnings": [],
        }

    def evaluate(self, wrffile, *, timeidx, parameters, **options):
        self.calls.append((wrffile, timeidx, dict(parameters), dict(options)))
        if self._result is not None:
            return self._result
        return (
            np.arange(6, dtype=np.float64).reshape(2, 3),
            {
                "units": "m s-1",
                "description": "10 m wind speed",
                "axes": ["south_north", "west_east"],
                "provenance": {"engine": "wrf-formula", "version": 1},
            },
        )


class _BlockingCompiled(_FakeCompiled):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.guard = threading.Lock()

    def evaluate(self, wrffile, *, timeidx, parameters, **options):
        with self.guard:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.02)
        with self.guard:
            self.active -= 1
        return super().evaluate(
            wrffile, timeidx=timeidx, parameters=parameters, **options
        )


class _FakeInnerFile:
    nt = 3


def _fake_wrffile():
    # Avoid opening fixture data while retaining the public wrapper contract.
    result = object.__new__(wrf.WrfFile)
    result._inner = _FakeInnerFile()
    result._formula_lock = threading.RLock()
    return result


class FormulaRecipeTests(unittest.TestCase):
    def test_checked_in_canonical_fixture_round_trips_without_shape_change(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "examples"
            / "wind10.wrf-formula.json"
        )
        parsed = json.loads(path.read_text(encoding="utf-8"))
        recipe = FormulaRecipe.load(path)
        self.assertEqual(recipe.to_dict(), parsed)

    def test_recipe_round_trip_is_deterministic_and_data_only(self):
        recipe = FormulaRecipe(
            "sqrt(U10^2 + V10^2)",
            name="wind10",
            description="10 m wind speed",
            expected_output_units="m s-1",
            references=(FormulaReference(url="https://example.invalid/paper"),),
            parameters={"threshold": np.float32(10.0)},
        )

        decoded = FormulaRecipe.from_json(recipe.to_json())

        self.assertEqual(decoded, recipe)
        self.assertEqual(decoded.parameters["threshold"].default, 10.0)
        self.assertEqual(decoded.parameters["threshold"].units, "1")
        self.assertNotIn("__class__", recipe.to_dict())

    def test_recipe_save_is_loadable_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wind.wrf-formula.json"
            FormulaRecipe("U10").save(path)

            self.assertEqual(FormulaRecipe.load(path).source, "U10")
            self.assertEqual([item.name for item in path.parent.iterdir()], [path.name])

    def test_duplicate_and_unknown_json_keys_are_rejected(self):
        with self.assertRaisesRegex(FormulaRecipeError, "duplicate JSON key"):
            FormulaRecipe.from_json(
                '{"schema":"wrf-formula/v1","source":"U10","source":"V10"}'
            )
        with self.assertRaisesRegex(FormulaRecipeError, "unknown formula recipe"):
            FormulaRecipe.from_json('{"source":"U10","python":"import os"}')

    def test_legacy_parameter_map_is_rejected_in_canonical_json(self):
        payload = {
            "schema": "wrf-formula/v1",
            "name": "legacy",
            "version": "1",
            "source": "U10 * scale",
            "parameters": {"scale": {"default": 1, "units": "1"}},
        }
        with self.assertRaisesRegex(FormulaRecipeError, "canonical list form"):
            FormulaRecipe.from_json(json.dumps(payload))

    def test_loaded_canonical_json_requires_schema_and_list_metadata(self):
        with self.assertRaisesRegex(FormulaRecipeError, "missing required"):
            FormulaRecipe.from_json(
                json.dumps({"name": "x", "version": "1", "source": "U10"})
            )
        payload = {
            "schema": "wrf-formula/v1",
            "name": "x",
            "version": "1",
            "source": "U10",
            "authors": None,
        }
        with self.assertRaises(FormulaRecipeError):
            FormulaRecipe.from_json(json.dumps(payload))

    def test_load_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(FormulaRecipeError, "UTF-8"):
                FormulaRecipe.load(path)

    def test_nonfinite_json_constants_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(FormulaRecipeError):
                FormulaRecipe.from_json(f'{{"source":"U10","parameters":{{"x":{value}}}}}')

    def test_oversize_json_is_rejected_before_parsing(self):
        payload = '{"source":"' + ("x" * 1_048_577) + '"}'
        with self.assertRaises(FormulaResourceError):
            FormulaRecipe.from_json(payload)

    def test_source_limit_counts_utf8_bytes_not_characters(self):
        # 32,769 two-byte characters fit under a character-only limit but not
        # Formula Lab's native 64 KiB UTF-8 source limit.
        with self.assertRaises(FormulaResourceError):
            FormulaRecipe("é" * 32_769)

    def test_parameter_defaults_reject_bool_object_and_nonfinite_values(self):
        invalid = (True, object(), np.array([1.0]), np.nan, np.inf)
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(
                (TypeError, ValueError, FormulaRecipeError)
            ):
                FormulaRecipe("U10", parameters={"threshold": value})

    def test_typed_parameter_units_bounds_and_description_round_trip(self):
        recipe = FormulaRecipe(
            "U10 * scale",
            parameters={
                "scale": FormulaParameter(
                    default=1.0,
                    units="1",
                    minimum=0.0,
                    maximum=2.0,
                    description="dimensionless multiplier",
                )
            },
        )
        decoded = FormulaRecipe.from_json(recipe.to_json())
        self.assertEqual(decoded.parameters["scale"], recipe.parameters["scale"])

    def test_parameter_default_must_be_inside_bounds(self):
        with self.assertRaises(FormulaRecipeError):
            FormulaParameter(default=-1.0, minimum=0.0)

    def test_evaluation_policies_and_unit_overrides_round_trip(self):
        recipe = FormulaRecipe(
            "RAW_FIELD",
            evaluation_options=FormulaEvaluationOptions(
                boundary_policy="missing",
                missing_policy="error",
                non_finite_policy="error",
                variable_unit_overrides={"RAW_FIELD": "m s-1"},
            ),
        )
        decoded = FormulaRecipe.from_json(recipe.to_json())
        self.assertEqual(decoded.evaluation_options.boundary_policy, "missing")
        self.assertEqual(decoded.evaluation_options.missing_policy, "error")
        self.assertEqual(decoded.evaluation_options.non_finite_policy, "error")
        self.assertEqual(
            decoded.evaluation_options.variable_unit_overrides,
            {"RAW_FIELD": "m s-1"},
        )

    def test_requirements_and_lower_resource_limits_round_trip(self):
        recipe = FormulaRecipe(
            "W",
            requirements=FormulaRequirements(
                fields=("W", "height"),
                maximum_cadence_seconds=30,
                maximum_horizontal_spacing_m=500,
                minimum_vertical_levels=40,
                notes=("Do not interpret as tornado resolving",),
            ),
            resource_limits=FormulaResourceLimits(max_ast_nodes=512),
        )
        decoded = FormulaRecipe.from_json(recipe.to_json())
        self.assertEqual(decoded.requirements.maximum_cadence_seconds, 30.0)
        self.assertEqual(decoded.resource_limits.max_ast_nodes, 512)

    def test_resource_limits_cannot_raise_host_ceiling(self):
        with self.assertRaises(FormulaResourceError):
            FormulaResourceLimits(max_ast_nodes=16_385)

    def test_recipe_mapping_is_immutable(self):
        recipe = FormulaRecipe("U10", parameters={"x": 1})
        with self.assertRaises(TypeError):
            recipe.parameters["x"] = 2


class FormulaCompileTests(unittest.TestCase):
    def _compile(self, source_or_recipe, native=None):
        native = native or _FakeCompiled()
        with mock.patch.object(wrf.formula._native, "_compile_formula", return_value=native) as call:
            formula = compile_formula(source_or_recipe)
        return formula, native, call

    def test_compile_passes_validated_recipe_to_native(self):
        recipe = FormulaRecipe(
            "sqrt(U10^2 + V10^2)",
            name="wind10",
            parameters={"threshold": 10},
        )

        formula, _, call = self._compile(recipe)

        call.assert_called_once_with(recipe.to_dict())
        self.assertEqual(formula.source, recipe.source)
        self.assertEqual(formula.canonical_source, _FakeCompiled.canonical_source)
        self.assertEqual(formula.dependencies, ("U10", "V10"))
        self.assertEqual(formula.plan.shape, (2, 3))
        self.assertEqual(formula.plan.recipe_requirements["fields"], ("U10", "V10"))
        self.assertIn("Dependencies: U10, V10", formula.explain())
        self.assertIs(formula.to_recipe(), recipe)
        json.dumps(formula.plan.to_dict(), allow_nan=False)

    def test_formula_cannot_bypass_compiler_and_recompile_is_identity(self):
        with self.assertRaises(TypeError):
            Formula(None, FormulaRecipe("U10"))
        formula, _, _ = self._compile("U10")
        self.assertIs(compile_formula(formula), formula)

    def test_compile_rejects_arbitrary_objects(self):
        with self.assertRaises(TypeError):
            compile_formula({"source": "U10"})

    def test_plan_shape_is_detached_from_mutable_native_list(self):
        native = _FakeCompiled()
        formula, _, _ = self._compile("U10", native)
        self.assertIsInstance(formula.plan.shape, tuple)
        with self.assertRaises(TypeError):
            formula.plan.shape[0] = 9


class FormulaEvaluationTests(unittest.TestCase):
    def _formula(self, native=None, recipe=None):
        native = native or _FakeCompiled()
        recipe = recipe or FormulaRecipe(
            "sqrt(U10^2 + V10^2)", parameters={"scale": 2.0}
        )
        with mock.patch.object(wrf.formula._native, "_compile_formula", return_value=native):
            return compile_formula(recipe), native

    def test_default_result_is_float64_contiguous_ndarray(self):
        formula, native = self._formula()

        result = formula.evaluate(
            _fake_wrffile(), timeidx=1, parameters={"offset": np.float64(3)}
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(native.calls[0][1], 1)
        self.assertEqual(native.calls[0][2], {"offset": 3.0, "scale": 2.0})
        self.assertEqual(
            native.calls[0][3]["boundary_policy"], "one_sided_second_order"
        )

    def test_metadata_result_has_units_axes_and_immutable_provenance(self):
        formula, _ = self._formula()

        result = formula.evaluate(_fake_wrffile(), return_metadata=True)

        self.assertIsInstance(result, FormulaResult)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.units, "m s-1")
        self.assertEqual(result.axes, ("south_north", "west_east"))
        np.testing.assert_array_equal(np.asarray(result), result.data)
        with self.assertRaises(TypeError):
            result.provenance["engine"] = "other"

    def test_negative_time_is_normalized_and_all_times_is_explicitly_rejected(self):
        formula, native = self._formula()
        formula.evaluate(_fake_wrffile(), timeidx=-1)
        self.assertEqual(native.calls[0][1], 2)
        with self.assertRaises(NotImplementedError):
            formula.evaluate(_fake_wrffile(), timeidx=wrf.ALL_TIMES)
        with self.assertRaises(NotImplementedError):
            formula.evaluate([_fake_wrffile(), _fake_wrffile()], timeidx=0)

    def test_bool_float_string_and_array_time_indices_are_rejected(self):
        formula, _ = self._formula()
        for value in (True, 1.5, "1", np.array(1)):
            with self.subTest(value=repr(value)), self.assertRaises(TypeError):
                formula.evaluate(_fake_wrffile(), timeidx=value)

    def test_parameter_overrides_reject_unsafe_values(self):
        formula, _ = self._formula()
        for value in (False, object(), np.array([1.0]), np.nan, -np.inf):
            with self.subTest(value=repr(value)), self.assertRaises(
                (TypeError, ValueError)
            ):
                    formula.evaluate(_fake_wrffile(), parameters={"scale": value})
        with self.assertRaises(ValueError):
            formula.evaluate(_fake_wrffile(), parameters={"Scale": 1, "scale": 2})

    def test_parameter_override_uses_declared_case_insensitive_name(self):
        recipe = FormulaRecipe("U10 * Scale", parameters={"Scale": 1.0})
        formula, native = self._formula(recipe=recipe)
        formula.evaluate(_fake_wrffile(), parameters={"scale": 2.0})
        self.assertEqual(native.calls[0][2], {"Scale": 2.0})

    def test_runtime_policy_overrides_reach_native_and_are_validated(self):
        formula, native = self._formula()
        formula.evaluate(
            _fake_wrffile(),
            boundary_policy="missing",
            missing_policy="ignore_in_reductions",
            non_finite_policy="error",
            variable_unit_overrides={"RAW": "Pa"},
        )
        options = native.calls[0][3]
        self.assertEqual(options["boundary_policy"], "missing")
        self.assertEqual(options["missing_policy"], "ignore_in_reductions")
        self.assertEqual(options["non_finite_policy"], "error")
        self.assertEqual(options["variable_unit_overrides"], {"RAW": "Pa"})
        with self.assertRaises(FormulaRecipeError):
            formula.evaluate(_fake_wrffile(), boundary_policy="periodic")

    def test_native_validates_undeclared_parameter(self):
        # The wrapper must forward names rather than silently dropping them;
        # the compiled native recipe owns declaration/range enforcement.
        formula, native = self._formula()
        formula.evaluate(_fake_wrffile(), parameters={"undeclared": 1.0})
        self.assertIn("undeclared", native.calls[0][2])

    def test_bad_native_result_shapes_are_typed_evaluation_errors(self):
        cases = (
            np.ones((2, 3), dtype=np.float32),
            np.ones((3, 2), dtype=np.float64).T,
        )
        for array in cases:
            native = _FakeCompiled(
                (array, {"axes": ["y", "x"], "provenance": {}})
            )
            formula, _ = self._formula(native=native)
            with self.subTest(dtype=str(array.dtype), contiguous=array.flags.c_contiguous):
                with self.assertRaises(FormulaEvaluationError):
                    formula.evaluate(_fake_wrffile())

    def test_bad_axes_and_provenance_are_typed_evaluation_errors(self):
        array = np.ones((2, 3), dtype=np.float64)
        metadata_cases = (
            {"axes": ["y"], "provenance": {}},
            {"axes": ["y", "x"], "provenance": ["not", "a", "mapping"]},
        )
        for metadata in metadata_cases:
            formula, _ = self._formula(native=_FakeCompiled((array, metadata)))
            with self.assertRaises(FormulaEvaluationError):
                formula.evaluate(_fake_wrffile())

    def test_scalar_is_a_zero_dimensional_float64_array(self):
        native = _FakeCompiled(
            (
                np.asarray(3.0, dtype=np.float64),
                {"axes": [], "units": "1", "provenance": {}},
            )
        )
        formula, _ = self._formula(native=native)
        result = formula.evaluate(_fake_wrffile())
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, np.dtype(np.float64))

    def test_same_wrffile_evaluations_are_serialized(self):
        native = _BlockingCompiled()
        formula, _ = self._formula(native=native)
        wrffile = _fake_wrffile()
        failures = []

        def evaluate():
            try:
                formula.evaluate(wrffile)
            except Exception as exc:  # pragma: no cover - diagnostic capture
                failures.append(exc)

        threads = [threading.Thread(target=evaluate) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(failures, [])
        self.assertEqual(native.maximum_active, 1)

    def test_top_level_evaluate_compiles_or_reuses(self):
        native = _FakeCompiled()
        with mock.patch.object(
            wrf.formula._native, "_compile_formula", return_value=native
        ) as compile_call:
            first = evaluate_formula(_fake_wrffile(), "U10")
            formula = compile_formula("U10")
            second = evaluate_formula(_fake_wrffile(), formula)

        self.assertEqual(compile_call.call_count, 2)
        np.testing.assert_array_equal(first, second)


class FormulaPublicSurfaceTests(unittest.TestCase):
    def test_root_exports_are_present(self):
        expected = {
            "Formula",
            "FormulaPlan",
            "FormulaParameter",
            "FormulaReference",
            "FormulaRequirements",
            "FormulaEvaluationOptions",
            "FormulaResourceLimits",
            "FormulaRecipe",
            "FormulaResult",
            "compile_formula",
            "evaluate_formula",
            "load_formula_recipe",
        }
        self.assertTrue(expected.issubset(set(wrf.__all__)))
        for name in expected:
            self.assertTrue(hasattr(wrf, name), name)

    def test_native_exception_hierarchy_and_span_attributes(self):
        self.assertTrue(issubclass(wrf.FormulaSyntaxError, wrf.FormulaError))
        self.assertTrue(issubclass(wrf.FormulaNameError, wrf.FormulaError))
        # Native compilation is responsible for attaching span/start/end.
        try:
            compile_formula("(")
        except wrf.FormulaSyntaxError as exc:
            self.assertTrue(hasattr(exc, "span"))
            self.assertTrue(hasattr(exc, "start"))
            self.assertTrue(hasattr(exc, "end"))
        else:
            self.fail("invalid source did not raise FormulaSyntaxError")


if __name__ == "__main__":
    unittest.main()
