"""Focused tests for contract validation and numerical parity policies."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


_PARITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PARITY_DIR))

from _common import ParityError, load_contract, validate_contract_document  # noqa: E402
from compare import compare_array  # noqa: E402


def _comparison_contract(
    *,
    reference_precision: str | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict:
    comparison = {
        "mode": "required",
        "tolerance": {"atol": atol, "rtol": rtol, "relative_floor": 0.0},
    }
    if reference_precision is not None:
        comparison["reference_precision"] = reference_precision
    return {
        "missing_values": {
            "mask_must_match": True,
            "max_mask_mismatch_fraction": 0.0,
            "allow_all_missing": False,
        },
        "comparison": comparison,
    }


class Float32ReferencePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.contract = _comparison_contract(reference_precision="float32")
        self.reference = np.array([1.0], dtype=np.float64)
        value = np.float32(1.0)
        lower = np.nextafter(value, np.float32(-np.inf)).astype(np.float64)
        upper = np.nextafter(value, np.float32(np.inf)).astype(np.float64)
        self.lower_spacing = 1.0 - float(lower)
        self.upper_spacing = float(upper) - 1.0

    def test_uses_neighbor_on_candidate_side_at_binade_boundary(self):
        self.assertGreater(self.upper_spacing, self.lower_spacing)

        above = np.array([1.0 + 0.49 * self.upper_spacing])
        below = np.array([1.0 - 0.49 * self.lower_spacing])

        above_result = compare_array(above, self.reference, self.contract)
        below_result = compare_array(below, self.reference, self.contract)

        self.assertTrue(above_result["passed"])
        self.assertTrue(below_result["passed"])
        self.assertEqual(above_result["reference_precision"], "float32")
        self.assertEqual(
            above_result["max_reference_quantization_allowance"],
            0.5 * self.upper_spacing,
        )
        self.assertEqual(
            below_result["max_reference_quantization_allowance"],
            0.5 * self.lower_spacing,
        )

    def test_rejects_candidate_beyond_directional_half_ulp(self):
        above = np.array([1.0 + 0.51 * self.upper_spacing])
        below = np.array([1.0 - 0.51 * self.lower_spacing])

        self.assertFalse(compare_array(above, self.reference, self.contract)["passed"])
        self.assertFalse(compare_array(below, self.reference, self.contract)["passed"])

    def test_rejects_reference_value_off_float32_lattice(self):
        unquantized = np.array([1.0 + 2.0**-30])

        with self.assertRaisesRegex(ParityError, "exactly representable as float32"):
            compare_array(unquantized, unquantized, self.contract)

    def test_terminal_float32_value_uses_finite_binade_spacing(self):
        reference = np.array([np.finfo(np.float32).max], dtype=np.float64)
        previous = np.nextafter(
            np.float32(reference[0]), np.float32(-np.inf)
        ).astype(np.float64)
        spacing = reference[0] - previous
        candidate = np.array([reference[0] + 0.49 * spacing])

        result = compare_array(candidate, reference, self.contract)

        self.assertTrue(result["passed"])
        self.assertTrue(
            np.isfinite(result["max_reference_quantization_allowance"])
        )
        self.assertEqual(
            result["max_reference_quantization_allowance"], 0.5 * spacing
        )

    def test_configured_tolerance_remains_separate_and_additive(self):
        contract = _comparison_contract(reference_precision="float32", atol=1e-8)
        candidate = np.array([1.0 + 0.5 * self.upper_spacing + 0.9e-8])

        self.assertTrue(compare_array(candidate, self.reference, contract)["passed"])


class ReferencePrecisionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, _ = load_contract(_PARITY_DIR / "contracts-v1.json")

    def test_avo_and_pvo_declare_float32_with_zero_algorithmic_tolerance(self):
        variables = {item["id"]: item for item in self.document["variables"]}

        for identifier in ("avo", "pvo"):
            with self.subTest(identifier=identifier):
                comparison = variables[identifier]["comparison"]
                self.assertEqual(comparison["reference_precision"], "float32")
                self.assertEqual(comparison["tolerance"]["atol"], 0.0)
                self.assertEqual(comparison["tolerance"]["rtol"], 0.0)

    def test_validator_rejects_unknown_reference_precision(self):
        document = deepcopy(self.document)
        document["variables"][0]["comparison"]["reference_precision"] = "float16"

        with self.assertRaisesRegex(ParityError, "comparison.reference_precision"):
            validate_contract_document(document)

    def test_validator_rejects_non_string_reference_precision(self):
        document = deepcopy(self.document)
        document["variables"][0]["comparison"]["reference_precision"] = ["float32"]

        with self.assertRaisesRegex(ParityError, "comparison.reference_precision"):
            validate_contract_document(document)


if __name__ == "__main__":
    unittest.main()
