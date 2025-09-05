# Optional mock polars_econ for dry runs
import polars as pl
from typing import List, Tuple

def chain(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, index_year: int):
    return price_quantity_pairs[0][1]

def convert(series: pl.DataFrame, date_col_name: str, as_freq: str, to_freq: str, technique: str, observed: str):
    original_col = next(c for c in series.columns if c != date_col_name)
    return series.sort(date_col_name).group_by_dynamic(date_col_name, every=to_freq).agg(pl.col(original_col).mean())

def fishvol(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int):
    return series_pairs[0][0]

def pct(expr: pl.Expr, offset: int = 1) -> pl.Expr:
    """Calculate percentage change with specified offset."""
    return ((expr / expr.shift(offset)) - 1) * 100
