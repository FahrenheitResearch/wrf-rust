#!/usr/bin/env python3
"""Pinned SHARPpy 1.4.0a5 differential acceptance runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA_VERSION = 1
HARNESS_ID = "wrf-rust-sharppy-differential"
RUST_PROTOCOL = "wrf-core-sharppy-public-helpers-v1"
EXPECTED_DISTRIBUTION = "SHARPpy"
EXPECTED_VERSION = "1.4.0a5"
OFFICIAL_TAG = "v1.4.0a5"
OFFICIAL_TAG_COMMIT = "a5405e255ab696c32db578dff2c4f83699ec717e"
PYPI_WHEEL_SHA256 = "13582f88ba1932b842cbf3ceb6f5f1ddadc17b0b2fd9172a3fc74ed0bcadb868"

EXPECTED_SOURCE_SHA256 = {
    "interp.py": "7fb249c953a30b734ae74342a1e874c2d6ecc38201a4c7885bb560e42825b366",
    "params.py": "fc6fb426fd230d1894e2f311ebc3d1bdeb98d9b2f2c5faa7850f6e38c7ed2a5a",
    "profile.py": "eab9e06592e6e9651b69d4975018a776305d99b798223155e49ef51ccecd5d49",
    "thermo.py": "9be540d23fc95dc45f7de1cb77ac374784272b5126f0a90594ca198cbe2ba315",
    "utils.py": "e35e7e8e780e51de52d467ac65131aa73e10f47b4832585de9b76547e0f7a32d",
    "winds.py": "8129d0eb5d9327edbc58de95ce252c11e13d8d6ee26ce20aecebe2ee0b02cbc4",
}

DEFAULT_TOLERANCE = {"atol": 1.0e-10, "rtol": 1.0e-12}


def _case(identifier: str, diagnostic: str, values: dict[str, float]) -> dict[str, Any]:
    return {
        "id": identifier,
        "diagnostic": diagnostic,
        "input": values,
        "tolerance": DEFAULT_TOLERANCE.copy(),
    }


CASES = [
    _case(
        "fixed_stp_lcl_below_unity",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 999.0, "srh01": 150.0, "bwd06_m_s": 20.0},
    ),
    _case(
        "fixed_stp_lcl_lower_boundary",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1000.0, "srh01": 150.0, "bwd06_m_s": 20.0},
    ),
    _case(
        "fixed_stp_lcl_mid_ramp",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1500.0, "srh01": 150.0, "bwd06_m_s": 20.0},
    ),
    _case(
        "fixed_stp_lcl_upper_boundary",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 2000.0, "srh01": 150.0, "bwd06_m_s": 20.0},
    ),
    _case(
        "fixed_stp_lcl_above_zero",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 2001.0, "srh01": 150.0, "bwd06_m_s": 20.0},
    ),
    _case(
        "fixed_stp_shear_below_gate",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1000.0, "srh01": 150.0, "bwd06_m_s": 12.4},
    ),
    _case(
        "fixed_stp_shear_lower_boundary",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1000.0, "srh01": 150.0, "bwd06_m_s": 12.5},
    ),
    _case(
        "fixed_stp_shear_cap_boundary",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1000.0, "srh01": 150.0, "bwd06_m_s": 30.0},
    ),
    _case(
        "fixed_stp_shear_above_cap",
        "fixed_stp",
        {"sbcape": 1500.0, "sblcl_m": 1000.0, "srh01": 150.0, "bwd06_m_s": 30.1},
    ),
    _case(
        "scp_ebwd_below_gate",
        "scp_neutral_cin",
        {"mucape": 1000.0, "effective_srh": 50.0, "ebwd_m_s": 9.0, "mucin": -40.0},
    ),
    _case(
        "scp_ebwd_lower_boundary",
        "scp_neutral_cin",
        {"mucape": 1000.0, "effective_srh": 50.0, "ebwd_m_s": 10.0, "mucin": -40.0},
    ),
    _case(
        "scp_ebwd_mid_ramp",
        "scp_neutral_cin",
        {"mucape": 1000.0, "effective_srh": 50.0, "ebwd_m_s": 15.0, "mucin": -40.0},
    ),
    _case(
        "scp_ebwd_cap_boundary",
        "scp_neutral_cin",
        {"mucape": 1000.0, "effective_srh": 50.0, "ebwd_m_s": 20.0, "mucin": -40.0},
    ),
    _case(
        "scp_ebwd_above_cap",
        "scp_neutral_cin",
        {"mucape": 1000.0, "effective_srh": 50.0, "ebwd_m_s": 21.0, "mucin": -40.0},
    ),
    _case(
        "ship_nominal",
        "ship",
        {
            "mucape": 2000.0,
            "mu_mixing_ratio_target_g_kg": 12.0,
            "lapse_rate_700_500_c_km": 7.0,
            "t500_c": -15.0,
            "shear_0_6km_m_s": 20.0,
            "freezing_level_agl_m": 3000.0,
        },
    ),
    _case(
        "ship_low_mr_warm_t500_high_shear_clamps",
        "ship",
        {
            "mucape": 2000.0,
            "mu_mixing_ratio_target_g_kg": 5.0,
            "lapse_rate_700_500_c_km": 7.0,
            "t500_c": -2.0,
            "shear_0_6km_m_s": 50.0,
            "freezing_level_agl_m": 3000.0,
        },
    ),
    _case(
        "ship_high_mr_low_shear_clamps",
        "ship",
        {
            "mucape": 2000.0,
            "mu_mixing_ratio_target_g_kg": 20.0,
            "lapse_rate_700_500_c_km": 7.0,
            "t500_c": -15.0,
            "shear_0_6km_m_s": 2.0,
            "freezing_level_agl_m": 3000.0,
        },
    ),
    _case(
        "ship_all_low_end_corrections",
        "ship",
        {
            "mucape": 650.0,
            "mu_mixing_ratio_target_g_kg": 12.0,
            "lapse_rate_700_500_c_km": 2.9,
            "t500_c": -10.0,
            "shear_0_6km_m_s": 20.0,
            "freezing_level_agl_m": 1200.0,
        },
    ),
    _case(
        "dcp_knot_normalization_unity",
        "dcp_mean_wind",
        {
            "dcape": 980.0,
            "mucape": 2000.0,
            "shear_0_6km_kt": 20.0,
            "mean_wind_0_6km_kt": 16.0,
        },
    ),
    _case(
        "dcp_mean_wind_half_scale",
        "dcp_mean_wind",
        {
            "dcape": 980.0,
            "mucape": 2000.0,
            "shear_0_6km_kt": 20.0,
            "mean_wind_0_6km_kt": 8.0,
        },
    ),
    _case(
        "critical_angle_calm_surface_sign",
        "critical_angle",
        {
            "storm_u_kt": 10.0,
            "storm_v_kt": 0.0,
            "surface_u_kt": 0.0,
            "surface_v_kt": 0.0,
            "wind_500m_u_kt": 10.0,
            "wind_500m_v_kt": 10.0,
        },
    ),
    _case(
        "critical_angle_nonzero_surface",
        "critical_angle",
        {
            "storm_u_kt": 10.0,
            "storm_v_kt": 0.0,
            "surface_u_kt": 5.0,
            "surface_v_kt": 5.0,
            "wind_500m_u_kt": 5.0,
            "wind_500m_v_kt": 15.0,
        },
    ),
    _case(
        "critical_angle_signed_components",
        "critical_angle",
        {
            "storm_u_kt": -10.0,
            "storm_v_kt": 6.0,
            "surface_u_kt": -5.0,
            "surface_v_kt": 2.0,
            "wind_500m_u_kt": 7.0,
            "wind_500m_v_kt": -8.0,
        },
    ),
]

EXPECTED_CASE_IDS = (
    "fixed_stp_lcl_below_unity",
    "fixed_stp_lcl_lower_boundary",
    "fixed_stp_lcl_mid_ramp",
    "fixed_stp_lcl_upper_boundary",
    "fixed_stp_lcl_above_zero",
    "fixed_stp_shear_below_gate",
    "fixed_stp_shear_lower_boundary",
    "fixed_stp_shear_cap_boundary",
    "fixed_stp_shear_above_cap",
    "scp_ebwd_below_gate",
    "scp_ebwd_lower_boundary",
    "scp_ebwd_mid_ramp",
    "scp_ebwd_cap_boundary",
    "scp_ebwd_above_cap",
    "ship_nominal",
    "ship_low_mr_warm_t500_high_shear_clamps",
    "ship_high_mr_low_shear_clamps",
    "ship_all_low_end_corrections",
    "dcp_knot_normalization_unity",
    "dcp_mean_wind_half_scale",
    "critical_angle_calm_surface_sign",
    "critical_angle_nonzero_surface",
    "critical_angle_signed_components",
)

EXPECTED_CASE_MANIFEST_SHA256 = (
    "0b2de8df1f083eeb03d9a8d8fb763d7cd9c302f1587b8d82738d6e99b173444d"
)

RUST_ONLY_COVERAGE = [
    {
        "scope": "current SPC SCP MUCIN magnitude factor",
        "reason": "SHARPpy 1.4.0a5 params.scp is the earlier three-term definition",
        "rust_tests": [
            "met::composite::tests::exported_scp_helper_retains_the_spc_mucin_scaling",
            "diag::severe::tests::registered_scp_cape_tuple_seam_selects_the_mucin_component",
            "variables::tests::scp_registry_points_to_the_current_spc_compute_path",
        ],
        "source": "https://www.spc.noaa.gov/exper/mesoanalysis/help/help_scp.html",
    },
    {
        "scope": "current SPC TEHI terms and threshold interpretation",
        "reason": "SHARPpy 1.4.0a5 has no TEHI implementation",
        "rust_tests": [
            "diag::severe::tests::tehi_matches_spc_beta_formula",
            "diag::severe::tests::tehi_sets_ml3cape_term_to_one_only_above_mlcape_threshold",
            "diag::severe::tests::tehi_high_ml3cape_cap_is_overridden_above_mlcape_threshold",
        ],
        "source": "https://www.spc.noaa.gov/exper/mesoanalysis/help/help_tehi.html",
    },
    {
        "scope": "published VTP versus repository-specific vtp_mod factors",
        "reason": "SHARPpy 1.4.0a5 has no VTP implementation",
        "rust_tests": [
            "diag::severe::tests::vtp_mod_is_distinct_from_published_vtp_for_an_illustrative_profile",
            "diag::severe::tests::vtp_mod_applies_ebwd_cutoff_and_cap",
            "diag::severe::tests::vtp_mod_applies_mllcl_cap_and_cutoff",
            "diag::severe::tests::vtp_mod_applies_mlcin_cutoff_and_cap",
            "diag::severe::tests::vtp_mod_applies_ml3cape_cap",
            "diag::severe::tests::vtp_mod_applies_lr700_500_floor_and_cap",
        ],
        "sources": [
            "https://doi.org/10.15191/nwajom.2018.0601",
            "https://www.spc.noaa.gov/exper/mesoanalysis/help/help_vtp.html",
        ],
    },
]


class HarnessError(RuntimeError):
    """Fail-closed acceptance error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file_normalized(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def case_manifest_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(CASES))


def validate_cases() -> str:
    identifiers = [case.get("id") for case in CASES]
    if identifiers != list(EXPECTED_CASE_IDS):
        raise HarnessError("case IDs/order do not match the frozen expected manifest")
    if len(set(identifiers)) != len(identifiers):
        raise HarnessError("case IDs are not unique")
    observed = case_manifest_sha256()
    if observed != EXPECTED_CASE_MANIFEST_SHA256:
        raise HarnessError(
            "case manifest provenance mismatch: "
            f"observed {observed}, expected {EXPECTED_CASE_MANIFEST_SHA256}"
        )
    return observed


def load_and_verify_sharppy() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np
        import sharppy
        from sharppy.sharptab import params, profile, thermo, utils, winds
    except Exception as error:  # pragma: no cover - exercised by bad environments
        raise HarnessError(
            f"cannot import SHARPpy reference environment: {error}"
        ) from error

    try:
        distribution = importlib.metadata.distribution(EXPECTED_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise HarnessError("SHARPpy distribution metadata is missing") from error

    distribution_version = distribution.version
    module_version = getattr(sharppy, "__version__", None)
    if distribution_version != EXPECTED_VERSION or module_version != EXPECTED_VERSION:
        raise HarnessError(
            "SHARPpy version mismatch: "
            f"distribution={distribution_version!r}, module={module_version!r}, "
            f"expected={EXPECTED_VERSION!r}"
        )

    distribution_root = Path(distribution.locate_file("")).resolve()
    module_path = Path(sharppy.__file__).resolve()
    try:
        module_path.relative_to(distribution_root)
    except ValueError as error:
        raise HarnessError(
            f"imported sharppy module {module_path} is outside installed distribution {distribution_root}"
        ) from error

    sharptab_root = Path(params.__file__).resolve().parent
    observed_hashes: dict[str, str] = {}
    for filename, expected_hash in EXPECTED_SOURCE_SHA256.items():
        path = sharptab_root / filename
        if not path.is_file():
            raise HarnessError(f"pinned SHARPpy source file is missing: {path}")
        observed_hash = sha256_file_normalized(path)
        observed_hashes[filename] = observed_hash
        if observed_hash != expected_hash:
            raise HarnessError(
                f"official-tag source mismatch for {filename}: "
                f"observed {observed_hash}, expected {expected_hash}"
            )

    modules = {
        "np": np,
        "params": params,
        "profile": profile,
        "thermo": thermo,
        "utils": utils,
        "winds": winds,
    }
    provenance = {
        "distribution": EXPECTED_DISTRIBUTION,
        "distribution_version": distribution_version,
        "module_version": module_version,
        "module_path": str(module_path),
        "official_tag": OFFICIAL_TAG,
        "official_tag_commit": OFFICIAL_TAG_COMMIT,
        "pypi_wheel_expected_sha256": PYPI_WHEEL_SHA256,
        "source_sha256": observed_hashes,
        "source_sha256_verified": True,
        "numpy_version": np.__version__,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    return modules, provenance


def _create_profile(modules: dict[str, Any], **values: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return modules["profile"].create_profile(
            profile="default", strictQC=False, **values
        )


def _dcp_profile(modules: dict[str, Any], values: dict[str, float]) -> Any:
    np = modules["np"]
    profile = _create_profile(
        modules,
        pres=np.array(
            [
                1000.0,
                950.0,
                900.0,
                800.0,
                700.0,
                600.0,
                500.0,
                400.0,
                300.0,
                250.0,
                200.0,
            ]
        ),
        hght=np.array(
            [
                0.0,
                500.0,
                1000.0,
                2000.0,
                3000.0,
                4200.0,
                5600.0,
                7200.0,
                9000.0,
                10300.0,
                11800.0,
            ]
        ),
        tmpc=np.array(
            [25.0, 22.0, 19.0, 12.0, 5.0, -3.0, -12.0, -23.0, -38.0, -47.0, -56.0]
        ),
        dwpc=np.array(
            [20.0, 18.0, 15.0, 8.0, 0.0, -8.0, -18.0, -30.0, -45.0, -54.0, -63.0]
        ),
        u=np.linspace(0.0, 30.0, 11),
        v=np.linspace(0.0, 10.0, 11),
    )
    profile.dcape = values["dcape"]
    profile.mupcl = SimpleNamespace(bplus=values["mucape"])
    profile.sfc_6km_shear = (values["shear_0_6km_kt"], 0.0)
    profile.mean_6km = (270.0, values["mean_wind_0_6km_kt"])
    return profile


def evaluate_reference(
    case: dict[str, Any], modules: dict[str, Any]
) -> tuple[float, list[float], dict[str, Any]]:
    diagnostic = case["diagnostic"]
    value = case["input"]
    params = modules["params"]
    thermo = modules["thermo"]
    utils = modules["utils"]
    winds = modules["winds"]
    np = modules["np"]

    if diagnostic == "fixed_stp":
        rust_args = [
            value["sbcape"],
            value["sblcl_m"],
            value["srh01"],
            value["bwd06_m_s"],
        ]
        reference = params.stp_fixed(*rust_args)
        return float(reference), rust_args, {}

    if diagnostic == "scp_neutral_cin":
        if value["mucin"] != -40.0:
            raise HarnessError(
                f"{case['id']}: SHARPpy three-term SCP requires neutral MUCIN=-40"
            )
        rust_args = [
            value["mucape"],
            value["effective_srh"],
            value["ebwd_m_s"],
            value["mucin"],
        ]
        reference = params.scp(*rust_args[:3])
        return (
            float(reference),
            rust_args,
            {"sharppy_terms": 3, "rust_mucin_factor": 1.0},
        )

    if diagnostic == "ship":
        parcel_pressure_hpa = 1000.0
        parcel_dewpoint_c = float(
            thermo.temp_at_mixrat(
                value["mu_mixing_ratio_target_g_kg"], parcel_pressure_hpa
            )
        )
        derived_mixing_ratio = float(
            thermo.mixratio(parcel_pressure_hpa, parcel_dewpoint_c)
        )
        shear_kt = float(utils.MS2KTS(value["shear_0_6km_m_s"]))
        derived_shear_m_s = float(utils.KTS2MS(utils.mag(shear_kt, 0.0)))
        parcel = SimpleNamespace(
            bplus=value["mucape"], pres=parcel_pressure_hpa, dwpc=parcel_dewpoint_c
        )
        ship_profile = SimpleNamespace(sfc_6km_shear=(shear_kt, 0.0))
        reference = params.ship(
            ship_profile,
            mupcl=parcel,
            frz_lvl=value["freezing_level_agl_m"],
            h5_temp=value["t500_c"],
            lr75=value["lapse_rate_700_500_c_km"],
        )
        rust_args = [
            value["mucape"],
            derived_shear_m_s,
            value["t500_c"],
            value["lapse_rate_700_500_c_km"],
            derived_mixing_ratio,
            value["freezing_level_agl_m"],
        ]
        derived = {
            "parcel_pressure_hpa": parcel_pressure_hpa,
            "parcel_dewpoint_c": parcel_dewpoint_c,
            "mu_mixing_ratio_g_kg": derived_mixing_ratio,
            "shear_0_6km_kt": shear_kt,
            "shear_0_6km_roundtrip_m_s": derived_shear_m_s,
        }
        return float(reference), rust_args, derived

    if diagnostic == "dcp_mean_wind":
        profile = _dcp_profile(modules, value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reference = params.dcp(profile)
        shear_m_s = float(utils.KTS2MS(value["shear_0_6km_kt"]))
        mean_wind_m_s = float(utils.KTS2MS(value["mean_wind_0_6km_kt"]))
        rust_args = [value["dcape"], value["mucape"], shear_m_s, mean_wind_m_s]
        derived = {
            "profile_shear_0_6km_kt": value["shear_0_6km_kt"],
            "profile_mean_wind_0_6km_kt": value["mean_wind_0_6km_kt"],
            "rust_shear_0_6km_m_s": shear_m_s,
            "rust_mean_wind_0_6km_m_s": mean_wind_m_s,
        }
        return float(reference), rust_args, derived

    if diagnostic == "critical_angle":
        profile = _create_profile(
            modules,
            pres=np.array([1000.0, 950.0, 900.0]),
            hght=np.array([0.0, 500.0, 1000.0]),
            tmpc=np.array([20.0, 17.0, 14.0]),
            dwpc=np.array([15.0, 12.0, 9.0]),
            u=np.array(
                [
                    value["surface_u_kt"],
                    value["wind_500m_u_kt"],
                    value["wind_500m_u_kt"] + 1.0,
                ]
            ),
            v=np.array(
                [
                    value["surface_v_kt"],
                    value["wind_500m_v_kt"],
                    value["wind_500m_v_kt"] + 1.0,
                ]
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reference = winds.critical_angle(
                profile, stu=value["storm_u_kt"], stv=value["storm_v_kt"]
            )
        rust_args = [
            value["storm_u_kt"],
            value["storm_v_kt"],
            value["surface_u_kt"],
            value["surface_v_kt"],
            value["wind_500m_u_kt"],
            value["wind_500m_v_kt"],
        ]
        derived = {
            "profile_pressure_hpa": [1000.0, 950.0, 900.0],
            "profile_height_m_agl": [0.0, 500.0, 1000.0],
            "component_units": "kt (angle is unit-invariant)",
        }
        return float(reference), rust_args, derived

    raise HarnessError(f"{case['id']}: unsupported diagnostic {diagnostic!r}")


def _git(repo_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def verify_rust_bridge(
    rust_binary: Path, rust_source: Path, repo_root: Path
) -> dict[str, Any]:
    if not rust_binary.is_file():
        raise HarnessError(f"Rust helper binary does not exist: {rust_binary}")
    if not rust_source.is_file():
        raise HarnessError(f"Rust helper source does not exist: {rust_source}")

    cargo_inputs = [
        repo_root / "Cargo.toml",
        repo_root / "Cargo.lock",
        repo_root / "crates" / "wrf-core" / "Cargo.toml",
    ]
    core_sources = sorted((repo_root / "crates" / "wrf-core" / "src").rglob("*.rs"))
    tracked_inputs = [rust_source, *cargo_inputs, *core_sources]
    build_inputs = tracked_inputs
    missing = [path for path in build_inputs if not path.is_file()]
    if missing:
        raise HarnessError(
            "candidate build inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )

    relative_tracked = [
        path.relative_to(repo_root).as_posix() for path in tracked_inputs
    ]
    tracked = _git(repo_root, ["ls-files", "--error-unmatch", "--", *relative_tracked])
    if tracked.returncode != 0 or tracked.stderr:
        raise HarnessError(
            "candidate Rust/Cargo inputs must all be tracked: "
            f"stdout={tracked.stdout!r}, stderr={tracked.stderr!r}"
        )
    tracked_paths = {
        line.strip().replace("\\", "/") for line in tracked.stdout.splitlines()
    }
    if tracked_paths != set(relative_tracked):
        missing_tracked = sorted(set(relative_tracked) - tracked_paths)
        raise HarnessError(
            "candidate Rust/Cargo inputs are untracked: " + ", ".join(missing_tracked)
        )

    status = _git(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *relative_tracked],
    )
    if status.returncode != 0 or status.stderr:
        raise HarnessError("cannot verify candidate Rust/Cargo worktree state")
    if status.stdout:
        raise HarnessError(
            "candidate Rust/Cargo inputs are dirty or untracked: "
            + status.stdout.replace("\n", "; ").strip()
        )

    newest_input = max(build_inputs, key=lambda path: path.stat().st_mtime_ns)
    if rust_binary.stat().st_mtime_ns < newest_input.stat().st_mtime_ns:
        raise HarnessError(
            "Rust helper binary is older than build input "
            f"{newest_input.relative_to(repo_root).as_posix()}; rebuild with cargo"
        )

    probe = subprocess.run(
        [str(rust_binary), "--provenance"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0 or probe.stderr or probe.stdout.strip() != RUST_PROTOCOL:
        raise HarnessError(
            "Rust helper provenance probe failed: "
            f"returncode={probe.returncode}, stdout={probe.stdout!r}, stderr={probe.stderr!r}"
        )

    git = _git(repo_root, ["rev-parse", "HEAD"])
    if git.returncode != 0 or git.stderr or len(git.stdout.strip()) != 40:
        raise HarnessError("cannot record candidate git provenance")

    input_hashes = {
        path.relative_to(repo_root).as_posix(): sha256_file_normalized(path)
        for path in build_inputs
    }

    return {
        "protocol": RUST_PROTOCOL,
        "binary_path": str(rust_binary.resolve()),
        "source_path": str(rust_source.resolve()),
        "source_sha256": sha256_file_normalized(rust_source),
        "git_head": git.stdout.strip(),
        "tracked_inputs_clean": True,
        "tracked_input_count": len(tracked_inputs),
        "build_input_count": len(build_inputs),
        "build_input_sha256": input_hashes,
    }


def run_rust_case(
    rust_binary: Path,
    repo_root: Path,
    case_id: str,
    diagnostic: str,
    values: list[float],
) -> float:
    command = [
        str(rust_binary),
        case_id,
        diagnostic,
        *(format(value, ".17g") for value in values),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or result.stderr:
        raise HarnessError(
            f"{case_id}: Rust helper failed with returncode={result.returncode}, "
            f"stderr={result.stderr!r}"
        )
    lines = result.stdout.splitlines()
    if len(lines) != 1:
        raise HarnessError(f"{case_id}: Rust helper must emit exactly one output line")
    fields = lines[0].split("\t")
    if fields[:3] != [RUST_PROTOCOL, case_id, diagnostic] or len(fields) != 4:
        raise HarnessError(
            f"{case_id}: Rust protocol/case mismatch in output {lines[0]!r}"
        )
    try:
        value = float(fields[3])
    except ValueError as error:
        raise HarnessError(f"{case_id}: Rust value is not numeric") from error
    if not math.isfinite(value):
        raise HarnessError(f"{case_id}: Rust value is not finite")
    return value


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    digest = sha256_bytes(encoded)
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_repo = script.parents[2]
    executable_suffix = ".exe" if os.name == "nt" else ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--rust-bin",
        type=Path,
        default=default_repo
        / "target"
        / "debug"
        / "examples"
        / f"sharppy_public_helpers{executable_suffix}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=default_repo / "parity-results" / "sharppy-a5-report.json",
    )
    parser.add_argument("--print-case-manifest-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_case_manifest_sha256:
        print(case_manifest_sha256())
        return 0

    repo_root = args.repo_root.resolve()
    rust_binary = args.rust_bin.resolve()
    rust_source = (
        repo_root / "crates" / "wrf-core" / "examples" / "sharppy_public_helpers.rs"
    )
    report_path = args.report.resolve()
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "harness": HARNESS_ID,
        "status": "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_case_count": len(EXPECTED_CASE_IDS),
        "executed_case_count": 0,
        "passed_case_count": 0,
        "failure_count": 0,
        "case_manifest_sha256": None,
        "provenance": {},
        "coverage": {
            "live_sharppy_diagnostics": [
                "fixed STP",
                "three-term SCP with neutral current-SPC MUCIN factor",
                "SHIP",
                "DCP",
                "critical angle",
            ],
            "rust_only": RUST_ONLY_COVERAGE,
        },
        "cases": rows,
        "failures": failures,
    }

    try:
        manifest_hash = validate_cases()
        report["case_manifest_sha256"] = manifest_hash
        modules, sharppy_provenance = load_and_verify_sharppy()
        rust_provenance = verify_rust_bridge(rust_binary, rust_source, repo_root)
        report["provenance"] = {
            "reference": sharppy_provenance,
            "candidate": rust_provenance,
            "harness": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file_normalized(Path(__file__).resolve()),
            },
        }

        for case in CASES:
            try:
                reference, rust_args, derived = evaluate_reference(case, modules)
                if not math.isfinite(reference):
                    raise HarnessError(
                        f"{case['id']}: SHARPpy returned a non-finite value"
                    )
                candidate = run_rust_case(
                    rust_binary,
                    repo_root,
                    case["id"],
                    case["diagnostic"],
                    rust_args,
                )
                absolute_error = abs(candidate - reference)
                tolerance = case["tolerance"]
                allowed_error = tolerance["atol"] + tolerance["rtol"] * abs(reference)
                passed = absolute_error <= allowed_error
                row = {
                    "id": case["id"],
                    "diagnostic": case["diagnostic"],
                    "input": case["input"],
                    "derived_input": derived,
                    "rust_arguments": rust_args,
                    "reference_value": reference,
                    "candidate_value": candidate,
                    "absolute_error": absolute_error,
                    "allowed_error": allowed_error,
                    "tolerance": tolerance,
                    "passed": passed,
                }
                rows.append(row)
                if not passed:
                    failures.append(
                        {
                            "kind": "numerical_mismatch",
                            "case_id": case["id"],
                            "message": (
                                f"absolute error {absolute_error} exceeds {allowed_error}"
                            ),
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "kind": "case_error",
                        "case_id": case["id"],
                        "message": f"{type(error).__name__}: {error}",
                    }
                )

        report["executed_case_count"] = len(rows)
        report["passed_case_count"] = sum(bool(row["passed"]) for row in rows)
        if len(rows) != len(CASES):
            failures.append(
                {
                    "kind": "case_count_mismatch",
                    "message": f"executed {len(rows)} rows for {len(CASES)} cases",
                }
            )
    except Exception as error:
        failures.append(
            {
                "kind": "harness_error",
                "message": f"{type(error).__name__}: {error}",
            }
        )

    report["failure_count"] = len(failures)
    if (
        not failures
        and report["executed_case_count"] == len(EXPECTED_CASE_IDS)
        and report["passed_case_count"] == len(EXPECTED_CASE_IDS)
    ):
        report["status"] = "passed"

    report_sha256 = write_report(report, report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failure_count": report["failure_count"],
                "passed_case_count": report["passed_case_count"],
                "report": str(report_path),
                "report_sha256": report_sha256,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
