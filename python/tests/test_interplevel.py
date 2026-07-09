"""Compatibility tests for the Python interplevel shim."""

import unittest

import numpy as np

import wrf


class InterplevelTests(unittest.TestCase):
    def test_pressure_interpolation_is_linear_not_logarithmic(self):
        coordinate = np.array([1000.0, 500.0])[:, None, None]
        field = np.array([0.0, 100.0])[:, None, None]

        result = wrf.interplevel(field, coordinate, 750.0)

        np.testing.assert_allclose(result, [[50.0]], rtol=0.0, atol=1e-12)

    def test_two_dimensional_target_levels_are_supported(self):
        coordinate = np.broadcast_to(
            np.array([1000.0, 500.0])[:, None, None], (2, 2, 2)
        )
        field = np.broadcast_to(
            np.array([0.0, 100.0])[:, None, None], (2, 2, 2)
        )
        target = np.array([[900.0, 800.0], [700.0, 600.0]])

        result = wrf.interplevel(field, coordinate, target)

        np.testing.assert_allclose(
            result, [[20.0, 40.0], [60.0, 80.0]], rtol=0.0, atol=1e-12
        )

    def test_height_interpolation_remains_linear(self):
        coordinate = np.array([0.0, 1000.0])[:, None, None]
        field = np.array([280.0, 270.0])[:, None, None]

        result = wrf.interplevel(field, coordinate, 250.0)

        np.testing.assert_allclose(result, [[277.5]], rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
