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


_DEGREE_METERS = 2.0 * np.pi * 6_370_000.0 / 360.0


class _FakeDataset:
    def __init__(self, attrs, variables):
        self.attrs = dict(attrs)
        self.variables = dict(variables)

    def getncattr(self, name):
        try:
            return self.attrs[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _projection_dataset(attrs=None, ref_lat=0.0, ref_lon=10.0,
                        moving_to=None):
    projection_attrs = {
        "MAP_PROJ": 6,
        "TRUELAT1": 0.0,
        "TRUELAT2": 0.0,
        "STAND_LON": 0.0,
        "DX": _DEGREE_METERS,
        "DY": _DEGREE_METERS,
        "POLE_LAT": 90.0,
        "POLE_LON": 0.0,
    }
    if attrs:
        projection_attrs.update(attrs)

    if moving_to is None:
        mass_lat = np.full((1, 3, 3), ref_lat, dtype=np.float64)
        mass_lon = np.full((1, 3, 3), ref_lon, dtype=np.float64)
    else:
        moved_lat, moved_lon = moving_to
        mass_lat = np.empty((2, 3, 3), dtype=np.float64)
        mass_lon = np.empty((2, 3, 3), dtype=np.float64)
        mass_lat[0].fill(ref_lat)
        mass_lat[1].fill(moved_lat)
        mass_lon[0].fill(ref_lon)
        mass_lon[1].fill(moved_lon)

    variables = {
        "XLAT": mass_lat,
        "XLONG": mass_lon,
        "XLAT_U": np.full((1, 3, 4), ref_lat, dtype=np.float64),
        "XLONG_U": np.full((1, 3, 4), ref_lon + 0.5, dtype=np.float64),
        "XLAT_V": np.full((1, 4, 3), ref_lat + 0.5, dtype=np.float64),
        "XLONG_V": np.full((1, 4, 3), ref_lon, dtype=np.float64),
    }
    return _FakeDataset(projection_attrs, variables)


class CoordinateCompatibilityTests(unittest.TestCase):
    def test_ordinary_scalar_restores_legacy_fractional_tuple(self):
        latitudes = np.broadcast_to(
            np.asarray([0.0, 1.0, 2.0])[:, np.newaxis], (3, 3)
        )
        longitudes = np.broadcast_to(
            np.asarray([10.0, 11.0, 12.0])[np.newaxis, :], (3, 3)
        )
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(latitudes, longitudes)):
            xy = WRF.ll_to_xy(object(), 0.7, 10.6)

        self.assertIs(type(xy), tuple)
        self.assertEqual(len(xy), 2)
        self.assertTrue(all(isinstance(value, (float, np.floating)) for value in xy))
        np.testing.assert_allclose(xy, (0.6, 0.7), atol=1e-12)

    def test_ordinary_scalar_clamps_california_outside_point_to_edge(self):
        # These bounds and dimensions mirror the approved d03 parity fixture.
        # The Oklahoma target is east and south of that California domain.
        latitudes = np.broadcast_to(
            np.linspace(37.59369659423828, 39.44728469848633, 800)[:, None],
            (800, 800),
        )
        longitudes = np.broadcast_to(
            np.linspace(-122.66775512695312, -120.27229309082031, 800)[None, :],
            (800, 800),
        )
        with mock.patch.object(
                WRF, "latlon_coords", return_value=(latitudes, longitudes)):
            xy = WRF.ll_to_xy(object(), 35.0, -97.0)

        self.assertIs(type(xy), tuple)
        self.assertIs(type(xy[0]), float)
        self.assertIs(type(xy[1]), float)
        self.assertEqual(xy, (799.0, 0.0))

    def test_explicit_options_keep_analytic_integer_result(self):
        dataset = _projection_dataset()
        xy = WRF.ll_to_xy(
            wrfin=dataset, latitude=0.7, longitude=10.6, meta=False
        )

        self.assertEqual(xy.shape, (2,))
        self.assertTrue(np.issubdtype(xy.dtype, np.integer))
        np.testing.assert_array_equal(xy, [1, 1])

        latlon = WRF.xy_to_ll(wrfin=dataset, x=1.0, y=1.0, meta=False)
        np.testing.assert_allclose(latlon, [1.0, 11.0], atol=1e-12)

    def test_omitted_options_preserve_sequence_analytic_behavior(self):
        dataset = _projection_dataset()
        xy = WRF.ll_to_xy(dataset, [0.25, 0.75], [10.5, 10.75])

        values = np.asarray(xy)
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.issubdtype(values.dtype, np.integer))
        np.testing.assert_array_equal(values, [[0, 1], [0, 1]])

    def test_sequences_flatten_and_preserve_leading_coordinate_axis(self):
        dataset = _projection_dataset()
        xy = WRF.ll_to_xy(
            dataset,
            [[0.25, 0.75], [1.25, 1.75]],
            [[10.5, 10.75], [11.0, 11.25]],
            as_int=False,
            meta=False,
        )

        self.assertEqual(xy.shape, (2, 4))
        np.testing.assert_allclose(xy[0], [0.5, 0.75, 1.0, 1.25], atol=1e-12)
        np.testing.assert_allclose(xy[1], [0.25, 0.75, 1.25, 1.75], atol=1e-12)

        latlon = WRF.xy_to_ll(dataset, xy[0], xy[1], meta=False)
        self.assertEqual(latlon.shape, (2, 4))
        np.testing.assert_allclose(latlon[0], [0.25, 0.75, 1.25, 1.75])
        np.testing.assert_allclose(latlon[1], [10.5, 10.75, 11.0, 11.25])

    def test_antimeridian_uses_shortest_longitude_delta(self):
        dataset = _projection_dataset(ref_lon=179.0)
        xy = WRF.ll_to_xy(dataset, 1.0, -179.0, as_int=False, meta=False)
        np.testing.assert_allclose(xy, [2.0, 1.0], atol=1e-12)

        latlon = WRF.xy_to_ll(dataset, xy[0], xy[1], meta=False)
        np.testing.assert_allclose(latlon, [1.0, -179.0], atol=1e-12)

    def test_u_and_v_staggering_use_their_own_coordinate_origins(self):
        dataset = _projection_dataset()
        mass = WRF.ll_to_xy(
            dataset, 0.0, 10.5, stagger="m", as_int=False, meta=False
        )
        u_grid = WRF.ll_to_xy(
            dataset, 0.0, 10.5, stagger="u", as_int=False, meta=False
        )
        v_grid = WRF.ll_to_xy(
            dataset, 0.5, 10.0, stagger="v", as_int=False, meta=False
        )
        np.testing.assert_allclose(mass, [0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(u_grid, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(v_grid, [0.0, 0.0], atol=1e-12)

        u_origin = WRF.xy_to_ll(dataset, 0.0, 0.0, stagger="u", meta=False)
        v_origin = WRF.xy_to_ll(dataset, 0.0, 0.0, stagger="v", meta=False)
        np.testing.assert_allclose(u_origin, [0.0, 10.5], atol=1e-12)
        np.testing.assert_allclose(v_origin, [0.5, 10.0], atol=1e-12)

    def test_all_supported_projections_round_trip_fractional_coordinates(self):
        cases = (
            (
                "lambert-north",
                {
                    "MAP_PROJ": 1,
                    "TRUELAT1": 30.0,
                    "TRUELAT2": 60.0,
                    "STAND_LON": -97.0,
                    "DX": 12_000.0,
                    "DY": 12_000.0,
                },
                35.0,
                -105.0,
            ),
            (
                "lambert-south",
                {
                    "MAP_PROJ": 1,
                    "TRUELAT1": -30.0,
                    "TRUELAT2": -60.0,
                    "STAND_LON": 135.0,
                    "DX": 9_000.0,
                    "DY": 9_000.0,
                },
                -40.0,
                130.0,
            ),
            (
                "polar-south",
                {
                    "MAP_PROJ": 2,
                    "TRUELAT1": -60.0,
                    "TRUELAT2": -60.0,
                    "STAND_LON": 120.0,
                    "DX": 15_000.0,
                    "DY": 15_000.0,
                },
                -70.0,
                115.0,
            ),
            (
                "mercator-dateline",
                {
                    "MAP_PROJ": 3,
                    "TRUELAT1": 20.0,
                    "TRUELAT2": 20.0,
                    "STAND_LON": 180.0,
                    "DX": 10_000.0,
                    "DY": 10_000.0,
                },
                5.0,
                179.0,
            ),
            (
                "rotated-latlon",
                {
                    "MAP_PROJ": 6,
                    "TRUELAT1": 0.0,
                    "TRUELAT2": 0.0,
                    "STAND_LON": 10.0,
                    "POLE_LAT": 45.0,
                    "POLE_LON": 180.0,
                },
                35.0,
                170.0,
            ),
        )
        expected_xy = np.asarray([[0.0, 0.25, 2.5], [0.0, 1.5, -0.75]])
        for name, attrs, ref_lat, ref_lon in cases:
            with self.subTest(projection=name):
                dataset = _projection_dataset(
                    attrs=attrs, ref_lat=ref_lat, ref_lon=ref_lon
                )
                latlon = WRF.xy_to_ll(
                    dataset, expected_xy[0], expected_xy[1], meta=False
                )
                actual_xy = WRF.ll_to_xy(
                    dataset,
                    latlon[0],
                    latlon[1],
                    as_int=False,
                    meta=False,
                )
                np.testing.assert_allclose(actual_xy, expected_xy, atol=2e-9)

    def test_matches_pinned_wrfpython_numeric_reference(self):
        # Values generated by wrf-python 1.3.4.1's compiled DLLTOIJ/DIJTOLL
        # routines. These make the test independent of our own round trip.
        cases = (
            (
                {"MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
                 "STAND_LON": -97.0, "DX": 12_000.0, "DY": 12_000.0},
                35.0, -105.0, (36.25, -102.5),
                (19.33857315429392, 9.753658550444356),
                (34.94476941906187, -104.65637841480091),
            ),
            (
                {"MAP_PROJ": 1, "TRUELAT1": -30.0, "TRUELAT2": -60.0,
                 "STAND_LON": 135.0, "DX": 9_000.0, "DY": 9_000.0},
                -40.0, 130.0, (-38.5, 133.0),
                (27.067605929010885, 19.201060718721692),
                (-40.075131989992265, 130.2669794929356),
            ),
            (
                {"MAP_PROJ": 2, "TRUELAT1": -60.0, "TRUELAT2": -60.0,
                 "STAND_LON": 120.0, "DX": 15_000.0, "DY": 15_000.0},
                -70.0, 115.0, (-68.0, -179.0),
                (146.89973347619804, -64.51922458735578),
                (-70.13239726128698, 116.00123728491964),
            ),
            (
                {"MAP_PROJ": 3, "TRUELAT1": 20.0, "TRUELAT2": 20.0,
                 "STAND_LON": 180.0, "DX": 10_000.0, "DY": 10_000.0},
                5.0, 179.0, (7.0, -179.0),
                (20.894530261306567, 21.010713477520625),
                (4.928480170003198, 179.23929707619507),
            ),
            (
                {"MAP_PROJ": 6, "TRUELAT1": 0.0, "TRUELAT2": 0.0,
                 "STAND_LON": 10.0, "POLE_LAT": 45.0, "POLE_LON": 180.0},
                35.0, 170.0, (36.0, 172.0),
                (10.253320884232977, 0.8732490047699315),
                (34.2587018327836, 170.5640257276045),
            ),
        )
        for attrs, ref_lat, ref_lon, target, expected_xy, expected_sample in cases:
            with self.subTest(map_proj=attrs["MAP_PROJ"], ref_lat=ref_lat):
                dataset = _projection_dataset(
                    attrs=attrs, ref_lat=ref_lat, ref_lon=ref_lon
                )
                actual_xy = WRF.ll_to_xy(
                    dataset, *target, as_int=False, meta=False
                )
                actual_sample = WRF.xy_to_ll(dataset, 2.5, -0.75, meta=False)
                np.testing.assert_allclose(actual_xy, expected_xy, atol=2e-12)
                np.testing.assert_allclose(
                    actual_sample, expected_sample, atol=2e-12
                )

    def test_coordinates_outside_domain_are_extrapolated_not_clamped(self):
        dataset = _projection_dataset()
        xy = WRF.ll_to_xy(dataset, -4.0, 20.0, as_int=False, meta=False)
        np.testing.assert_allclose(xy, [10.0, -4.0], atol=1e-12)

    def test_moving_domain_fails_explicitly(self):
        dataset = _projection_dataset(moving_to=(0.25, 10.5))
        with self.assertRaisesRegex(NotImplementedError, "moving-domain"):
            WRF.ll_to_xy(dataset, 0.0, 10.0, meta=False)
        with self.assertRaisesRegex(NotImplementedError, "moving-domain"):
            WRF.xy_to_ll(dataset, 0.0, 0.0, meta=False)

    def test_bad_inputs_and_metadata_fail_clearly(self):
        dataset = _projection_dataset()
        with self.assertRaisesRegex(ValueError, "same length"):
            WRF.ll_to_xy(dataset, [0.0, 1.0], [10.0], meta=False)
        with self.assertRaisesRegex(ValueError, "stagger"):
            WRF.ll_to_xy(dataset, 0.0, 10.0, stagger="w", meta=False)
        with self.assertRaisesRegex(ValueError, "timeidx"):
            WRF.ll_to_xy(dataset, 0.0, 10.0, timeidx=-1, meta=False)

        missing_dx = _projection_dataset()
        del missing_dx.attrs["DX"]
        with self.assertRaisesRegex(ValueError, "DX"):
            WRF.ll_to_xy(missing_dx, 0.0, 10.0, meta=False)

    def test_optional_xarray_metadata_matches_wrf_python_conventions(self):
        try:
            import xarray  # noqa: F401
        except ImportError:
            self.skipTest("xarray is optional")

        dataset = _projection_dataset()
        xy = WRF.ll_to_xy(
            dataset, 0.25, 10.5, squeeze=False, meta=True, as_int=True
        )
        self.assertEqual(xy.name, "xy")
        self.assertEqual(xy.dims, ("x_y", "idx"))
        self.assertTrue(np.issubdtype(xy.dtype, np.integer))
        np.testing.assert_array_equal(xy.coords["x_y"], ["x", "y"])
        pair = xy.coords["latlon_coord"].values[0]
        self.assertIsInstance(pair, WRF.CoordPair)
        self.assertEqual((pair.lat, pair.lon), (0.25, 10.5))

        latlon = WRF.xy_to_ll(dataset, [0.5], [0.25], squeeze=False)
        self.assertEqual(latlon.name, "latlon")
        self.assertEqual(latlon.dims, ("lat_lon", "idx"))
        np.testing.assert_array_equal(latlon.coords["lat_lon"], ["lat", "lon"])
        pair = latlon.coords["xy_coord"].values[0]
        self.assertEqual((pair.x, pair.y), (0.5, 0.25))


if __name__ == "__main__":
    unittest.main()
