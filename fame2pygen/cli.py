import argparse
import sys
from pathlib import Path
from .parser import parse_script
from .generators import build_generation_context, generate_formulas_module, generate_pipeline_script
from .version import __version__
from .preprocess import preprocess_lines  # NEW

MOCK_LIB_CODE = """# Optional mock polars_econ for dry runs
import polars as pl
from typing import List, Tuple

def chain(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, index_year: int):
    return price_quantity_pairs[0][1]

def convert(series: pl.DataFrame, date_col_name: str, as_freq: str, to_freq: str, technique: str, observed: str):
    original_col = next(c for c in series.columns if c != date_col_name)
    return series.sort(date_col_name).group_by_dynamic(date_col_name, every=to_freq).agg(pl.col(original_col).mean())

def fishvol(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int):
    return series_pairs[0][0]
"""

def main(argv=None):
    parser = argparse.ArgumentParser(prog='fame2pygen', description='Convert FAME (.inp) scripts to Polars-based modules.')
    parser.add_argument('input', help='Path to .inp file or - for stdin')
    parser.add_argument('--formulas-out', default='cformulas.py')
    parser.add_argument('--pipeline-out', default='c_pipeline.py')
    parser.add_argument('--formulas-module-name', default=None, help='Override import module name for formulas file.')
    parser.add_argument('--write-mock-econ', action='store_true', help='Write mock polars_econ.py into CWD')
    parser.add_argument('--no-preprocess', action='store_true', help='Disable loop/list preprocessing (debug)')
    parser.add_argument('--version', action='store_true')
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.input == '-':
        raw_lines = [l.rstrip('\n') for l in sys.stdin]
    else:
        path = Path(args.input)
        if not path.exists():
            print(f'Input file not found: {path}', file=sys.stderr)
            return 1
        raw_lines = path.read_text(encoding='utf-8').splitlines()

    processed_lines = raw_lines if args.no_preprocess else preprocess_lines(raw_lines)

    parsed = parse_script(processed_lines)
    ctx = build_generation_context(parsed, processed_lines)

    formulas_code = generate_formulas_module(ctx)
    formulas_path = Path(args.formulas_out)
    formulas_path.write_text(formulas_code, encoding='utf-8')

    formulas_module_name = args.formulas_module_name or formulas_path.stem
    pipeline_code = generate_pipeline_script(ctx, formulas_module_name)
    Path(args.pipeline_out).write_text(pipeline_code, encoding='utf-8')

    if args.write_mock_econ:
        Path('polars_econ.py').write_text(MOCK_LIB_CODE, encoding='utf-8')

    print(f'Generated: {formulas_path} and {args.pipeline_out}')
    if args.write_mock_econ:
        print('Generated mock polars_econ.py')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())