"""Regression tests for the WRF-Runner compatibility probe itself."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_wrf_runner as probe  # noqa: E402


class _Handle:
    nx = 2
    ny = 2
    nz = 2
    nt = 1


class _Projection:
    def transform_points(self):
        return None


class _GoodWrf:
    @staticmethod
    def interplevel(field, coordinate, target):
        target_array = np.asarray(target)
        if target_array.ndim == 0:
            if float(target_array) == 200.0:
                return np.full((2, 2), np.nan)
            value = 100.0 * np.log(float(target_array) / 1000.0) / np.log(0.5)
            return np.full((2, 2), value)
        return 100.0 * np.log(target_array / 1000.0) / np.log(0.5)

    @staticmethod
    def latlon_coords(_handle):
        lat = np.array([[0.0, 0.0], [1.0, 1.0]])
        lon = np.array([[0.0, 1.0], [0.0, 1.0]])
        return lat, lon

    @staticmethod
    def ll_to_xy(_handle, latitude, longitude):
        return float(longitude), float(latitude)

    @staticmethod
    def get_cartopy(_handle):
        return _Projection()

    @staticmethod
    def getvar(_handle, _name, **_options):
        return np.ones((2, 2))


class _ArraySubclass(np.ndarray):
    pass


class _BadReturnWrf(_GoodWrf):
    @staticmethod
    def interplevel(field, coordinate, target):
        return _GoodWrf.interplevel(field, coordinate, target).view(_ArraySubclass)

    @staticmethod
    def latlon_coords(handle):
        lat, lon = _GoodWrf.latlon_coords(handle)
        return [lat, lon]

    @staticmethod
    def ll_to_xy(_handle, latitude, longitude):
        return np.array([longitude, latitude])

    @staticmethod
    def getvar(_handle, _name, **_options):
        return np.ones((2, 2)).view(_ArraySubclass)


class ProbeContractTests(unittest.TestCase):
    def test_good_legacy_defaults_pass_without_coercion(self):
        rows = probe.probe_helpers(_GoodWrf, _Handle())
        failures = {row["id"]: row["failures"] for row in rows if not row["passed"]}
        self.assertEqual(failures, {})

        getvar = probe.probe_getvar_call(
            _GoodWrf,
            _Handle(),
            {"id": "sample", "name": "sample", "options": {}, "shape": [2, 2]},
        )
        self.assertTrue(getvar["passed"], getvar["failures"])
        self.assertEqual(getvar["actual_type"], "numpy.ndarray")

    def test_array_like_subclasses_and_sequence_pairs_fail_explicitly(self):
        rows = probe.probe_helpers(_BadReturnWrf, _Handle())
        by_id = {row["id"]: row for row in rows}

        for identifier in (
            "interplevel_scalar",
            "interplevel_2d_target",
            "interplevel_outside_missing",
            "latlon_coords",
            "ll_to_xy_center",
            "ll_to_xy_fractional_boundary",
        ):
            self.assertFalse(by_id[identifier]["passed"], identifier)
            self.assertTrue(by_id[identifier]["failures"], identifier)

        getvar = probe.probe_getvar_call(
            _BadReturnWrf,
            _Handle(),
            {"id": "sample", "name": "sample", "options": {}, "shape": [2, 2]},
        )
        self.assertFalse(getvar["passed"])
        self.assertIn("exact numpy.ndarray", " ".join(getvar["failures"]))

    def test_masked_array_is_not_a_plain_ndarray(self):
        self.assertFalse(probe.is_plain_ndarray(np.ma.masked_array([1.0])))
        self.assertTrue(probe.is_plain_ndarray(np.array([1.0])))

    def test_numpy_float_components_are_valid_legacy_tuple_scalars(self):
        self.assertTrue(probe.is_float_pair((np.float64(1.25), np.float32(2.5))))
        self.assertFalse(probe.is_float_pair(np.array([1.25, 2.5])))


if __name__ == "__main__":
    unittest.main()
