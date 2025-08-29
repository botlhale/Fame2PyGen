import polars as pl
from typing import List, Tuple

def A(b: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'a'.
Derived from FAME script line:
    a=b*2
"""
    res = (
        b*2
    )
    return res.alias("a")
