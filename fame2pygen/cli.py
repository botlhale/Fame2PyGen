import argparse
import sys
from pathlib import Path
from .parser import parse_script
from .generators import build_generation_context, generate_formulas_module, generate_pipeline_script
from .version import __version__

MOCK_LIB_CODE = """# Optional mock polars_econ for dry runs
import polars as pl
from typing import List, Tuple

def chain(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, index_year: int):
    # Return first quantity as placeholder
    return price_quantity_pairs[0][1]

def convert(series: pl.DataFrame, date_col_name: str, as_freq: str, to_freq: str, technique: str, observed: str):
    original_col_name = next(c for c in series.columns if c != date_col_name)
    return series.sort(date_col_name).group_by_dynamic(date_col_name, every=to_freq).agg(pl.col(original_col_name).mean())

def fishvol(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int):
    return series_pairs[0][0]
"""

def read_inp(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [l.rstrip("\n") for l in text.splitlines()]
    return lines

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fame2pygen",
        description="Convert FAME (.inp) scripts to Polars-based Python modules."
    )
    parser.add_argument("input", help="Path to .inp file (or '-' for stdin)")
    parser.add_argument("--formulas-out", default="cformulas.py",
                        help="Output filename for formulas module (default: cformulas.py)")
    parser.add_argument("--pipeline-out", default="c_pipeline.py",
                        help="Output filename for pipeline script (default: c_pipeline.py)")
    parser.add_argument("--formulas-module-name", default=None,
                        help="Module name to use when importing formulas (default: derived from --formulas-out)")
    parser.add_argument("--write-mock-econ", action="store_true",
                        help="Also write a mock polars_econ.py in current directory.")
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.input == "-":
        lines = [l.rstrip("\n") for l in sys.stdin.readlines()]
    else:
        inp_path = Path(args.input)
        if not inp_path.exists():
            print(f"Input file not found: {inp_path}", file=sys.stderr)
            return 1
        lines = read_inp(inp_path)

    parsed = parse_script(lines)
    ctx = build_generation_context(parsed, lines)

    formulas_code = generate_formulas_module(ctx)
    formulas_path = Path(args.formulas_out)
    formulas_path.write_text(formulas_code, encoding="utf-8")

    formulas_module_name = args.formulas_module_name or formulas_path.stem

    pipeline_code = generate_pipeline_script(ctx, formulas_module_name)
    Path(args.pipeline_out).write_text(pipeline_code, encoding="utf-8")

    if args.write_mock_econ:
        Path("polars_econ.py").write_text(MOCK_LIB_CODE, encoding="utf-8")

    print(f"Generated: {formulas_path} and {args.pipeline_out}")
    if args.write_mock_econ:
        print("Generated mock polars_econ.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())