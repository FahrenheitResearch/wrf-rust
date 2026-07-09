"""Focused compatibility tests for the pure-Python interplevel shim."""

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "wrf" / "interpolation.py"
_SPEC = importlib.util.spec_from_file_location("wrf_interpolation_under_test", _MODULE_PATH)
_INTERPOLATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INTERPOLATION)
interplevel = _INTERPOLATION.interplevel


def _planes(values, ny=2, nx=2, dtype=np.float64):
    values = np.asarray(values, dtype=dtype)
    return np.broadcast_to(values[:, None, None], (values.size, ny, nx)).copy()


class InterplevelTests(unittest.TestCase):
    def test_public_signature_matches_wrf_python(self):
        signature = inspect.signature(interplevel)
        self.assertEqual(
            list(signature.parameters),
            ["field3d", "vert", "desiredlev", "missing", "squeeze", "meta"],
        )
        self.assertIs(signature.parameters["squeeze"].default, True)
        self.assertIs(signature.parameters["meta"].default, True)

    def test_pressure_interpolation_is_linear_not_logarithmic(self):
        coordinate = _planes([1000.0, 500.0])
        field = _planes([0.0, 100.0])

        result = interplevel(field, coordinate, 750.0, meta=False)

        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, 50.0, rtol=0.0, atol=1e-12)

    def test_height_interpolation_remains_linear(self):
        coordinate = _planes([0.0, 1000.0])
        field = _planes([280.0, 270.0])

        result = interplevel(field, coordinate, 250.0, meta=False)

        np.testing.assert_allclose(result, 277.5, rtol=0.0, atol=1e-12)

    def test_wrf_runner_two_dimensional_target_surface_is_preserved(self):
        coordinate = _planes([1000.0, 500.0])
        field = _planes([0.0, 100.0])
        target = np.array([[900.0, 800.0], [700.0, 600.0]])

        result = interplevel(field, coordinate, target, meta=False)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(
            result, [[20.0, 40.0], [60.0, 80.0]], rtol=0.0, atol=1e-12
        )

    def test_arbitrary_left_dimensions_and_level_order(self):
        coordinate_3d = _planes([1000.0, 800.0, 600.0, 400.0])
        coordinate = np.broadcast_to(coordinate_3d, (2, 3) + coordinate_3d.shape)
        offsets = np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1, 1) * 100.0
        field = ((1000.0 - coordinate) * 0.1 + offsets).astype(np.float32)

        result = interplevel(
            field,
            coordinate,
            [900.0, 500.0],
            squeeze=False,
            meta=False,
        )

        self.assertEqual(result.shape, (2, 3, 2, 2, 2))
        self.assertEqual(result.dtype, np.dtype(np.float32))
        for left_index in np.ndindex((2, 3)):
            offset = float(np.ravel_multi_index(left_index, (2, 3)) * 100)
            np.testing.assert_allclose(result[left_index + (0,)], offset + 10.0)
            np.testing.assert_allclose(result[left_index + (1,)], offset + 50.0)

    def test_leading_multiproduct_dimension_matches_wrf_python(self):
        coordinate_3d = _planes([1000.0, 500.0])
        coordinate = np.broadcast_to(coordinate_3d, (2,) + coordinate_3d.shape)
        left_offsets = np.arange(2).reshape(2, 1, 1, 1) * 10.0
        base_field = (1000.0 - coordinate) * 0.1 + left_offsets
        field = np.stack((base_field, base_field + 100.0), axis=0)

        result = interplevel(
            field,
            coordinate,
            [750.0],
            squeeze=False,
            meta=False,
        )

        self.assertEqual(result.shape, (2, 2, 1, 2, 2))
        for product in range(2):
            for left in range(2):
                expected = product * 100.0 + left * 10.0 + 25.0
                np.testing.assert_allclose(result[product, left, 0], expected)

    def test_multiproduct_without_left_dims_retains_pinned_float64_work_dtype(self):
        coordinate = _planes([1000.0, 500.0], dtype=np.float32)
        base_field = _planes([0.0, 100.0], dtype=np.float32)
        field = np.stack((base_field, base_field + np.float32(10.0)), axis=0)

        result = interplevel(
            field,
            coordinate,
            [750.0],
            squeeze=False,
            meta=False,
        )

        self.assertEqual(result.shape, (2, 1, 2, 2))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(result[0], 50.0)
        np.testing.assert_allclose(result[1], 60.0)

    def test_shared_and_left_dependent_target_surfaces(self):
        coordinate_3d = _planes([1000.0, 500.0])
        coordinate = np.broadcast_to(coordinate_3d, (2,) + coordinate_3d.shape)
        offsets = np.arange(2).reshape(2, 1, 1, 1) * 10.0
        field = (1000.0 - coordinate) * 0.1 + offsets
        shared_target = np.array([[900.0, 800.0], [700.0, 600.0]])

        shared = interplevel(
            field, coordinate, shared_target, squeeze=False, meta=False
        )
        self.assertEqual(shared.shape, (2, 2, 2))
        for left in range(2):
            np.testing.assert_allclose(
                shared[left], (1000.0 - shared_target) * 0.1 + left * 10.0
            )

        targets = np.stack((shared_target, shared_target - 50.0), axis=0)
        varying = interplevel(
            field, coordinate, targets, squeeze=False, meta=False
        )
        self.assertEqual(varying.shape, (2, 2, 2))
        for left in range(2):
            np.testing.assert_allclose(
                varying[left], (1000.0 - targets[left]) * 0.1 + left * 10.0
            )

        multiproduct_field = np.stack((field, field + 100.0), axis=0)
        multiproduct = interplevel(
            multiproduct_field,
            coordinate,
            targets,
            squeeze=False,
            meta=False,
        )
        self.assertEqual(multiproduct.shape, (2, 2, 2, 2))
        np.testing.assert_allclose(multiproduct[0], varying)
        np.testing.assert_allclose(multiproduct[1], varying + 100.0)

    def test_monotonic_direction_is_selected_for_each_left_slice(self):
        coordinate = np.empty((2, 2, 2, 2), dtype=np.float64)
        coordinate[0] = _planes([1000.0, 500.0])
        coordinate[1] = _planes([0.0, 1000.0])
        field = np.empty_like(coordinate)
        field[0] = _planes([0.0, 100.0])
        field[1] = _planes([10.0, 30.0])

        result = interplevel(
            field, coordinate, 750.0, squeeze=False, meta=False
        )

        self.assertEqual(result.shape, (2, 1, 2, 2))
        np.testing.assert_allclose(result[0, 0], 50.0)
        np.testing.assert_allclose(result[1, 0], 25.0)

    def test_nonmonotonic_column_uses_topmost_matching_layer(self):
        coordinate = _planes([0.0, 10.0, 5.0, 15.0])
        field = _planes([0.0, 100.0, 200.0, 300.0])

        result = interplevel(field, coordinate, 7.0, meta=False)

        # Both 0-10 and 5-15 bracket seven. DINTERP3DZ scans from model top,
        # so it chooses 5-15 and returns 220 rather than 70.
        np.testing.assert_allclose(result, 220.0)

    def test_exact_and_out_of_range_levels_use_caller_missing_value(self):
        coordinate = _planes([1000.0, 500.0])
        field = _planes([0.0, 100.0])

        result = interplevel(
            field,
            coordinate,
            [1000.0, 750.0, 200.0],
            missing=-9999.0,
            squeeze=False,
            meta=False,
        )

        self.assertTrue(np.ma.getmaskarray(result[0]).all())
        self.assertFalse(np.ma.getmaskarray(result[1]).any())
        self.assertTrue(np.ma.getmaskarray(result[2]).all())
        np.testing.assert_allclose(result[1], 50.0)
        self.assertEqual(float(result.fill_value), -9999.0)

    def test_squeeze_false_retains_scalar_level_and_left_axes(self):
        coordinate = np.array([1000.0, 500.0])[None, :, None, None]
        field = np.array([0.0, 100.0])[None, :, None, None]

        unsqueezed = interplevel(
            field, coordinate, 750.0, squeeze=False, meta=False
        )
        squeezed = interplevel(
            field, coordinate, 750.0, squeeze=True, meta=False
        )

        self.assertEqual(unsqueezed.shape, (1, 1, 1, 1))
        self.assertEqual(squeezed.shape, ())
        self.assertEqual(float(squeezed), 50.0)

    def test_masked_inputs_follow_pinned_wrapped_array_data(self):
        coordinate = np.ma.array(
            _planes([1000.0, 500.0]),
            mask=np.broadcast_to([False, True], (2, 2, 2)),
        )
        field = np.ma.array(
            _planes([0.0, 100.0]),
            mask=np.broadcast_to([False, True], (2, 2, 2)),
        )

        result = interplevel(field, coordinate, 750.0, meta=False)

        # The pinned decorators cast the MaskedArray and pass its underlying
        # data buffer to F2PY; the input mask itself is not an interpolation
        # mask. Output missing sentinels are still returned as a MaskedArray.
        self.assertIsInstance(result, np.ma.MaskedArray)
        np.testing.assert_allclose(result, 50.0)

    def test_invalid_shapes_fail_clearly(self):
        coordinate = _planes([1000.0, 500.0])
        field = _planes([0.0, 100.0])

        with self.subTest("too_few_dimensions"):
            with self.assertRaisesRegex(ValueError, "at least three dimensions"):
                interplevel(field[:, 0], coordinate[:, 0], 750.0, meta=False)

        with self.subTest("field_coordinate_mismatch"):
            with self.assertRaisesRegex(ValueError, "same shape"):
                interplevel(field[:, :, :1], coordinate, 750.0, meta=False)

        with self.subTest("surface_horizontal_mismatch"):
            with self.assertRaisesRegex(ValueError, "rightmost dimensions"):
                interplevel(field, coordinate, np.ones((3, 2)), meta=False)

        left_coordinate = np.broadcast_to(coordinate, (2,) + coordinate.shape)
        left_field = np.broadcast_to(field, left_coordinate.shape)
        with self.subTest("surface_left_mismatch"):
            with self.assertRaisesRegex(ValueError, "left dimensions"):
                interplevel(
                    left_field,
                    left_coordinate,
                    np.ones((3, 2, 2)),
                    meta=False,
                )

        with self.subTest("missing_not_scalar"):
            with self.assertRaisesRegex(TypeError, "missing must be a scalar"):
                interplevel(field, coordinate, 750.0, missing=[-1.0], meta=False)

    def test_numpy_meta_output_uses_xarray_when_available(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray is not installed")

        result = interplevel(
            _planes([0.0, 100.0]),
            _planes([1000.0, 500.0]),
            750.0,
        )

        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.name, "field3d_interp")
        self.assertEqual(result.shape, (2, 2))

    def test_xarray_metadata_dimensions_coordinates_and_attrs(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray is not installed")

        dims = ("Time", "bottom_top", "south_north", "west_east")
        coordinate_values = np.broadcast_to(
            np.array([1000.0, 500.0])[None, :, None, None], (1, 2, 2, 2)
        ).copy()
        field_values = np.broadcast_to(
            np.array([0.0, 100.0])[None, :, None, None], (1, 2, 2, 2)
        ).copy()
        coords = {
            "Time": [0],
            "bottom_top": [0, 1],
            "south_north": [10, 20],
            "west_east": [30, 40],
            "XLAT": (
                ("south_north", "west_east"),
                [[35.0, 35.0], [36.0, 36.0]],
            ),
        }
        field = xr.DataArray(
            field_values,
            name="tk",
            dims=dims,
            coords=coords,
            attrs={
                "units": "K",
                "description": "temperature",
                "MemoryOrder": "XYZ",
            },
        )
        coordinate = xr.DataArray(
            coordinate_values,
            name="pressure",
            dims=dims,
            coords=coords,
            attrs={"units": "hPa"},
        )

        result = interplevel(
            field,
            coordinate,
            [750.0, 600.0],
            missing=-9999.0,
            squeeze=False,
            meta=True,
        )

        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(
            result.dims, ("Time", "level", "south_north", "west_east")
        )
        self.assertEqual(result.name, "tk_interp")
        np.testing.assert_array_equal(result.coords["level"], [750.0, 600.0])
        self.assertIn("XLAT", result.coords)
        self.assertNotIn("bottom_top", result.coords)
        self.assertEqual(result.attrs["units"], "K")
        self.assertEqual(result.attrs["vert_units"], "hPa")
        self.assertEqual(result.attrs["missing_value"], -9999.0)
        self.assertEqual(result.attrs["_FillValue"], -9999.0)
        self.assertNotIn("description", result.attrs)
        self.assertNotIn("MemoryOrder", result.attrs)
        np.testing.assert_allclose(result[:, 0], 50.0)
        np.testing.assert_allclose(result[:, 1], 80.0)

        scalar = interplevel(field, coordinate, 750, squeeze=True, meta=True)
        self.assertEqual(scalar.dims, ("south_north", "west_east"))
        self.assertEqual(int(scalar.coords["level"]), 750)
        self.assertEqual(int(scalar.coords["Time"]), 0)

        no_meta = interplevel(field, coordinate, 750.0, meta=np.bool_(False))
        self.assertIsInstance(no_meta, np.ma.MaskedArray)

    def test_xarray_left_dependent_surface_has_level_coordinate(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray is not installed")

        dims = ("Time", "bottom_top", "south_north", "west_east")
        coordinate_values = np.broadcast_to(
            np.array([1000.0, 500.0])[None, :, None, None], (1, 2, 2, 2)
        ).copy()
        field_values = (1000.0 - coordinate_values) * 0.1
        field = xr.DataArray(field_values, name="field", dims=dims)
        coordinate = xr.DataArray(
            coordinate_values, dims=dims, attrs={"units": "hPa"}
        )
        targets = np.array([[[900.0, 800.0], [700.0, 600.0]]])

        result = interplevel(
            field, coordinate, targets, squeeze=False, meta=True
        )

        self.assertEqual(result.dims, ("Time", "south_north", "west_east"))
        self.assertEqual(
            result.coords["level"].dims,
            ("Time", "south_north", "west_east"),
        )
        np.testing.assert_allclose(result.coords["level"], targets)
        np.testing.assert_allclose(result, (1000.0 - targets) * 0.1)

    def test_meta_fallback_without_xarray_still_honors_squeeze(self):
        field = _planes([0.0, 100.0])
        coordinate = _planes([1000.0, 500.0])

        with mock.patch.dict(sys.modules, {"xarray": None}):
            result = interplevel(field, coordinate, 750.0, meta=True)

        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, 50.0)

        with mock.patch.dict(sys.modules, {"xarray": None}):
            missing = interplevel(field, coordinate, 200.0, meta=True)

        self.assertTrue(np.ma.getmaskarray(missing).all())
        self.assertTrue(np.isnan(np.asarray(missing)).all())


if __name__ == "__main__":
    unittest.main()
