"""Auto-generated ts_transformer module - applies transformations from formulas"""import polars as plfrom typing import List, Tuplefrom formulas import *
def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:    """Apply transformations and return augmented DataFrame."""    # Date filter: * (all dates)    # --- Level 1: compute v1 ---    pdf = pdf.with_columns([
        ASSIGN_SERIES("V1", 100)
    ])
    # Date filter: 2020-01-01 to 2020-12-31    # --- Level 2: compute v2 ---    pdf = pdf.with_columns([
        ADD_SERIES("V2", pl.col("V1"), 10)
    ])
    # Date filter: 2020-01-01 to 2020-12-31    # --- Level 3: compute v3 ---    pdf = pdf.with_columns([
        MUL_SERIES("V3", pl.col("V2"), 2)
    ])
    # Date filter: * (all dates)    # --- Level 4: compute v4 ---    pdf = pdf.with_columns([
        DIV_SERIES("V4", pl.col("V3"), 2)
    ])
    # Date filter: 2021-01-01 to 2021-06-30    # --- Level 5: compute v5 ---    pdf = pdf.with_columns([
        ADD_SERIES("V5", pl.col("V4"), 5)
    ])
    return pdf
