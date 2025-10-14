# polars_econ_mock.py - minimal mock of polars_econ for development
import polars as pl
from typing import List, Tuple

def pct(expr, offset=1):
    return expr.shift(-offset).pct_change()

def chain(price_quantity_pairs, date_col, index_year):
    return pl.lit(1.0)  # mock implementation

def convert(series, date_col, as_freq, to_freq, technique, observed):
    return pl.lit(1.0)  # mock implementation

def fishvol(series_pairs, date_col, rebase_year):
    return pl.lit(1.0)  # mock implementation
