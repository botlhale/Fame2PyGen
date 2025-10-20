"""Auto-generated ts_transformer module - applies transformations from formulas"""
import polars as pl
from typing import List, Tuple
from datetime import date
from formulas import *

def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:
    """Apply transformations and return augmented DataFrame."""
    # Date filter: * (all dates)
    # --- Level 1: compute v_base ---
    pdf = pdf.with_columns([
        pl.lit(100).alias("V_BASE")
    ])
    # Date filter: 2020-01-01 to 2020-12-31
    # --- Level 2: compute v_2020 ---
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER((pl.col("V_BASE") * pl.lit(2)), "V_2020", "2020-01-01", "2020-12-31").alias("V_2020")
    ])
    # Date filter: 2021-01-01 to 2021-12-31
    # --- Level 2: compute v_2021 ---
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER((pl.col("V_BASE") * pl.lit(3)), "V_2021", "2021-01-01", "2021-12-31").alias("V_2021")
    ])
    # Date filter: 2020-01-01 to 2020-12-31
    # --- Level 3: compute v_2020_adj ---
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER((pl.col("V_2020") + pl.lit(10)), "V_2020_ADJ", "2020-01-01", "2020-12-31").alias("V_2020_ADJ")
    ])
    # Date filter: * (all dates)
    # --- Level 3: compute v_combined ---
    pdf = pdf.with_columns([
        (pl.col("V_BASE") + pl.col("V_2020") + pl.col("V_2021")).alias("V_COMBINED")
    ])
    return pdf
