"""Auto-generated ts_transformer module - applies transformations from formulas"""import polars as plfrom typing import List, Tuplefrom formulas import *
def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:    """Apply transformations and return augmented DataFrame."""    # --- Level 1: compute v1, vbot ---    pdf = pdf.with_columns([
        ADD_SERIES("V1", pl.col("V2"), pl.col("V3")),
        ASSIGN_SERIES("VBOT", 1)
    ])
    return pdf
