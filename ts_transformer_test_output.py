"""Auto-generated ts_transformer module - applies transformations from formulas"""
import polars as pl
from typing import List, Tuple
from datetime import date
from formulas import *

def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:
    """Apply transformations and return augmented DataFrame."""
    # Date filter: * (all dates)
    # --- Level 1: compute a.bot, a.bot@12jul1985, a.bot@13jul1985, b_c ---
    pdf = pdf.with_columns([
        pl.col("Z.SOME").alias("A.BOT"),
        ((pl.col("D.HAR").shift(1)/pl.col("D.HAR"))*(CONVERT(pl.col("DA_VAL"), "bus", "disc", "ave"))).alias('B_C')
    ])
    # Point-in-time assignments for A.BOT
    pdf = pdf.with_columns([
        pl.when(pl.col("DATE") == pl.lit("1985-07-12").cast(pl.Date))
    .then(pl.lit(130))
    .when(pl.col("DATE") == pl.lit("1985-07-13").cast(pl.Date))
    .then(pl.lit(901))
    .otherwise(pl.col("A.BOT"))
    .alias("A.BOT")
    ])
    return pdf
