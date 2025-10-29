"""Auto-generated ts_transformer module - applies transformations from formulas"""
import polars as pl
from typing import List, Tuple
from datetime import date
from formulas import *

def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:
    """Apply transformations and return augmented DataFrame."""
    # Date filter: * (all dates)
    # --- Level 1: compute a.bot, a.bot[12jul1985], a.bot[13jul1985], b_c ---
    pdf = pdf.with_columns([
        pl.col("Z.SOME").alias("A.BOT"),
        pl.lit(130).alias("A.BOT12JUL1985"),
        pl.lit(901).alias("A.BOT13JUL1985"),
        ((pl.col("D.HAR")[T-1]/pl.col("D.HAR"))*(convert(pl.col("DA_VAL"),pl.col("BUS"),pl.col("DISC"),pl.col("AVE")))).alias('B_C')
    ])
    return pdf
