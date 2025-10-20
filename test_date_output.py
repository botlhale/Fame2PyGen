"""Auto-generated ts_transformer module - applies transformations from formulas"""import polars as plfrom typing import List, Tuplefrom formulas import *
def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:    """Apply transformations and return augmented DataFrame."""    # Date filter: 2020-01-01 to 2020-12-31    # --- Level 1: compute v1 ---    pdf = pdf.with_columns([
        ADD_SERIES("V1", pl.col("V2"), pl.col("V3"))
    ])
    # Date filter: * (all dates)    # --- Level 1: compute v4 ---    pdf = pdf.with_columns([
        ADD_SERIES("V4", pl.col("V5"), pl.col("V6"))
    ])
    return pdf
