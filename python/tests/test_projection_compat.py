"""Pure-Python compatibility tests for wrf-python coordinate helpers.

These tests stub the native extension, Cartopy, and netCDF4 so they can run
without compiling wrf-rust or requiring optional plotting dependencies.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np


def _load_wrf_module():
    native = ModuleType("wrf._wrf")

    class NativeWrfFile:
        def __init__(self, path):
            self.path = str(path)

    native.WrfFile = NativeWrfFile
    native.list_variables = lambda: []
    native.render_sounding_box = lambda *args, **kwargs: None
    native.render_sounding_ij = lambda *args, **kwargs: None
    native.render_sounding_latlon = lambda *args, **kwargs: None

    plot = ModuleType("wrf.plot")
    for name in ("plot_field", "plot_wind", "plot_skewt", "panel"):
        setattr(plot, name, lambda *args, **kwargs: None)

    explorer = ModuleType("wrf.explorer")
    for name in ("Explorer", "cross_section", "profile", "hovmoller"):
        setattr(explorer, name, lambda *args, **kwargs: None)

    sys.modules["wrf._wrf"] = native
    sys.modules["wrf.plot"] = plot
    sys.modules["wrf.explorer"] = explorer

    package_dir = Path(__file__).resolve().parents[1] / "wrf"
    spec = importlib.util.spec_from_file_location(
        "wrf",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wrf"] = module
    spec.loader.exec_module(module)
    return module


WRF = _load_wrf_module()


class _Capture:
    kind = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _capture_type(name):
    return type(name, (_Capture,), {"kind": name})


FAKE_CCRS = ModuleType("cartopy.crs")
for _name in (
    "Globe",
    "LambertConformal",
    "Stereographic",
    "Mercator",
    "PlateCarree",
    "RotatedPole",
):
    setattr(FAKE_CCRS, _name, _capture_type(_name))


def _assert_wrf_globe(testcase, projection):
    globe = projection.kwargs["globe"]
    testcase.assertEqual(globe.kind, "Globe")
    testcase.assertIsNone(globe.kwargs["ellipse"])
    testcase.assertEqual(globe.kwargs["semimajor_axis"], 6_370_000.0)
    testcase.assertEqual(globe.kwargs["semiminor_axis"], 6_370_000.0)
    testcase.assertEqual(globe.kwargs["nadgrids"], "@null")


class ProjectionCompatibilityTests(unittest.TestCase):
    def test_nested_lambert_uses_moad_center_and_wrf_sphere(self):
        # A nested domain's CEN_LAT is deliberately different from the
        # parent-domain projection origin, MOAD_CEN_LAT.
        attrs = {
            "MAP_PROJ": 1,
            "TRUELAT1": 30.0,
            "TRUELAT2": 60.0,
            "STAND_LON": -97.0,
            "MOAD_CEN_LAT": 38.5,
            "CEN_LAT": 42.25,
            "CEN_LON": -96.75,
            "POLE_LAT": 90.0,
            "POLE_LON": 0.0,
        }

        class Dataset:
            def __init__(self, path, mode):
                self.path = path
                self.mode = mode
                self.closed = False

            def getncattr(self, name):
                try:
                    return attrs[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def close(self):
                self.closed = True

        netcdf4 = ModuleType("netCDF4")
        netcdf4.Dataset = Dataset
        cartopy = ModuleType("cartopy")
        cartopy.__path__ = []
        cartopy.crs = FAKE_CCRS

        with mock.patch.dict(
                sys.modules,
                {
                    "cartopy": cartopy,
                    "cartopy.crs": FAKE_CCRS,
                    "netCDF4": netcdf4,
                }), mock.patch.object(
                    WRF,
                    "_ensure_wrffile",
                    return_value=SimpleNamespace(path="nested-wrfout.nc"),
                ):
            projection = WRF.get_cartopy(object())

        self.assertEqual(projection.kind, "LambertConformal")
        self.assertEqual(projection.kwargs["central_longitude"], -97.0)
        self.assertEqual(projection.kwargs["central_latitude"], 38.5)
        self.assertNotEqual(projection.kwargs["central_latitude"], 42.25)
        self.assertEqual(projection.kwargs["standard_parallels"], (30.0, 60.0))
        self.assertEqual(projection.kwargs["cutoff"], -30.0)
        _assert_wrf_globe(self, projection)

    def test_southern_lambert_uses_positive_cutoff(self):
        projection = WRF._cartopy_from_wrf_attrs(
            FAKE_CCRS,
            {
                "MAP_PROJ": 1,
                "TRUELAT1": -30.0,
                "TRUELAT2": -60.0,
                "STAND_LON": 135.0,
                "MOAD_CEN_LAT": -40.0,
            },
        )
        self.assertEqual(projection.kwargs["cutoff"], 30.0)

    def test_other_wrf_projections_use_their_standard_longitude(self):
        cases = (
            (
                {
                    "MAP_PROJ": 2,
                    "TRUELAT1": -60.0,
                    "STAND_LON": 120.0,
                    "CEN_LAT": -70.0,
                    "CEN_LON": 125.0,
                },
                "Stereographic",
                "central_longitude",
                120.0,
            ),
            (
                {
                    "MAP_PROJ": 3,
                    "TRUELAT1": 20.0,
                    "STAND_LON": 140.0,
                    "CEN_LON": 135.0,
                },
                "Mercator",
                "central_longitude",
                140.0,
            ),
            (
                {
                    "MAP_PROJ": 6,
                    "STAND_LON": 12.0,
                    "CEN_LON": 8.0,
                    "POLE_LAT": 90.0,
                    "POLE_LON": 0.0,
                },
                "PlateCarree",
                "central_longitude",
                12.0,
            ),
        )
        for attrs, kind, key, value in cases:
            with self.subTest(kind=kind):
                projection = WRF._cartopy_from_wrf_attrs(FAKE_CCRS, attrs)
                self.assertEqual(projection.kind, kind)
                self.assertEqual(projection.kwargs[key], value)
                _assert_wrf_globe(self, projection)

        polar = WRF._cartopy_from_wrf_attrs(FAKE_CCRS, cases[0][0])
        self.assertEqual(polar.kwargs["central_latitude"], -90.0)
        self.assertEqual(polar.kwargs["true_scale_latitude"], -60.0)

    def test_rotated_latlon_matches_wrf_python_parameter_conversion(self):
        projection = WRF._cartopy_from_wrf_attrs(
            FAKE_CCRS,
            {
                "MAP_PROJ": 6,
                "STAND_LON": 10.0,
                "MOAD_CEN_LAT": 35.0,
                "POLE_LAT": 45.0,
                "POLE_LON": 180.0,
            },
        )
        self.assertEqual(projection.kind, "RotatedPole")
        self.assertEqual(projection.kwargs["pole_longitude"], -190.0)
        self.assertEqual(projection.kwargs["pole_latitude"], 45.0)
        self.assertEqual(projection.kwargs["central_rotated_longitude"], 0.0)
        _assert_wrf_globe(self, projection)

    def test_malformed_projection_metadata_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "MAP_PROJ"):
            WRF._cartopy_from_wrf_attrs(FAKE_CCRS, {})
        with self.assertRaisesRegex(ValueError, "both POLE_LAT and POLE_LON"):
            WRF._cartopy_from_wrf_attrs(
                FAKE_CCRS,
                {
                    "MAP_PROJ": 6,
                    "STAND_LON": 10.0,
                    "POLE_LAT": 45.0,
                },
            )


class LlToXyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.lat = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        self.lon = np.array(
            [
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

    def test_scalar_defaults_to_integer_leading_axis_result(self):
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(self.lat, self.lon)):
            xy = WRF.ll_to_xy(object(), 0.7, 10.6)

        self.assertEqual(xy.shape, (2,))
        self.assertTrue(np.issubdtype(xy.dtype, np.integer))
        np.testing.assert_array_equal(xy, [1, 1])
        # WRF-Runner indexes scalar results exactly this way.
        self.assertEqual(int(xy[0]), 1)
        self.assertEqual(int(xy[1]), 1)

    def test_sequences_have_leading_coordinate_axis_and_fractional_option(self):
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(self.lat, self.lon)):
            xy = WRF.ll_to_xy(
                object(),
                [0.25, 1.75],
                [10.5, 11.25],
                as_int=False,
            )

        self.assertEqual(xy.shape, (2, 2))
        np.testing.assert_allclose(xy[0], [0.5, 1.25], atol=1e-12)
        np.testing.assert_allclose(xy[1], [0.25, 1.75], atol=1e-12)

    def test_nested_sequences_are_flattened_like_wrf_python(self):
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(self.lat, self.lon)):
            xy = WRF.ll_to_xy(
                object(),
                [[0.0, 0.5], [1.0, 1.5]],
                [[10.0, 10.5], [11.0, 11.5]],
                as_int=False,
            )

        self.assertEqual(xy.shape, (2, 4))

    def test_mismatched_sequences_and_unsupported_stagger_are_explicit(self):
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(self.lat, self.lon)):
            with self.assertRaisesRegex(ValueError, "same length"):
                WRF.ll_to_xy(object(), [0.0, 1.0], [10.0])
            with self.assertRaises(NotImplementedError):
                WRF.ll_to_xy(object(), 0.0, 10.0, stagger="u")


if __name__ == "__main__":
    unittest.main()
