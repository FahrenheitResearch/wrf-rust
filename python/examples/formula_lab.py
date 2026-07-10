"""Compile, inspect, evaluate, and save a Formula Lab diagnostic.

Usage:
    python python/examples/formula_lab.py /path/to/wrfout [timeidx]
"""

from pathlib import Path
import sys

from wrf import FormulaReference, FormulaRecipe, WrfFile, compile_formula


def main(path, timeidx=0):
    recipe = FormulaRecipe(
        source="sqrt(U10^2 + V10^2)",
        name="wind10",
        description="10 m scalar wind speed",
        expected_output_units="m s-1",
        references=(
            FormulaReference(citation="WRF U10/V10 grid-relative wind components"),
        ),
    )
    formula = compile_formula(recipe)

    # Safe preflight: this does not open or evaluate the WRF file.
    print(formula.explain())

    wrffile = WrfFile(path)
    result = formula.evaluate(
        wrffile,
        timeidx=timeidx,
        return_metadata=True,
    )
    print(f"shape={result.shape} dtype={result.dtype} units={result.units}")
    print(f"min={result.data.min():.3f} max={result.data.max():.3f}")

    recipe_path = Path("wind10.wrf-formula.json")
    recipe.save(recipe_path)
    print(f"saved reproducible recipe to {recipe_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: formula_lab.py /path/to/wrfout [timeidx]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)
