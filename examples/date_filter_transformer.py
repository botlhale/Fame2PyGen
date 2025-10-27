"""Auto-generated ts_transformer module - applies transformations from formulas"""
import polars as pl
from typing import List, Tuple
from datetime import date
from formulas import *

def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:
    """Apply transformations and return augmented DataFrame."""
    # Date filter: 01Feb2020 to 31Dec2020
    # --- Level 1: compute a@datefilter_01Feb2020_to_31Dec2020_2 ---
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER(pl.lit(100), "A", "01Feb2020", "31Dec2020").alias("A")
    ])
    # Date filter: 01Jan2021 to *
    # --- Level 1: compute a@datefilter_01Jan2021_to_*_4 ---
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER(pl.lit(250), "A", "01Jan2021", "*", preserve_existing=True).alias("A")
    ])
    return pdf
