#!/usr/bin/env python3
"""Validate the parity contract and print its canonical SHA-256."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ParityError, load_contract, print_error_and_exit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "contracts-v1.json",
    )
    args = parser.parse_args()
    contract, digest = load_contract(args.contract)
    print(
        f"contract {contract['contract_version']}: "
        f"{len(contract['variables'])} variables, sha256 {digest}"
    )


if __name__ == "__main__":
    try:
        main()
    except ParityError as exc:
        print_error_and_exit(exc)
