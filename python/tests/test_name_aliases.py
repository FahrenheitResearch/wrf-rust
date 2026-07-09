"""Python-shim regressions that are not covered by wrf-core registry tests."""

import unittest

import wrf


class NameAliasTests(unittest.TestCase):
    def test_helicity_reaches_storm_relative_helicity_core_alias(self):
        self.assertEqual(wrf._normalize_var_name("helicity"), "helicity")
        self.assertEqual(wrf._normalize_var_name("HeLiCiTy"), "HeLiCiTy")

    def test_updraft_helicity_remains_explicit(self):
        self.assertEqual(wrf._normalize_var_name("uhel"), "uhel")
        self.assertEqual(wrf._normalize_var_name("updraft_helicity"), "updraft_helicity")

    def test_legacy_structural_aliases_remain_normalized(self):
        self.assertEqual(wrf._normalize_var_name("cape_2d"), "cape2d")
        self.assertEqual(wrf._normalize_var_name("cape_3d"), "cape3d")
        self.assertEqual(wrf._normalize_var_name("mdbz"), "maxdbz")


if __name__ == "__main__":
    unittest.main()
