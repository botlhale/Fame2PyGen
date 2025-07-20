import polars as pl
from typing import List, Tuple

try:
    import polars_econ as ple
except ImportError:
    ple = None

def A() -> pl.Expr:
    """
    Computes the values for the a time series or variable using Polars expressions.
    Derived from FAME script(s):
        set a = v143*12

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V143").mul(12)
    )
    return res.alias("a")

def AA(a: pl.Series, a_: pl.Series) -> pl.Series:
    """
    Computes the values for the aa time series or variable using Polars expressions.
    Derived from FAME script(s):
        set aa = a$/a

    Returns:
        pl.Series: Polars Series to compute the time series or variable values.
    """
    res = (
        a_/a
    )
    return res

def A_() -> pl.Expr:
    """
    Computes the values for the a$ time series or variable using Polars expressions.
    Derived from FAME script(s):
        set a$ = v123*12

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V123").mul(12)
    )
    return res.alias("a_")

def BB(a: pl.Series, aa: pl.Series) -> pl.Series:
    """
    Computes the values for the bb time series or variable using Polars expressions.
    Derived from FAME script(s):
        set bb = aa+a

    Returns:
        pl.Series: Polars Series to compute the time series or variable values.
    """
    res = (
        aa+a
    )
    return res

def B_() -> pl.Expr:
    """
    Computes the values for the b$ time series or variable using Polars expressions.
    Derived from FAME script(s):
        set b$ = v1234*12

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V1234").mul(12)
    )
    return res.alias("b_")

def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:
    """Generic wrapper for $mchain using 'ple.chain'."""
    import polars_econ_mock as ple
    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))

def CONVERT(series: pl.DataFrame, as_freq: str, to_freq: str, technique: str, observed: str) -> pl.DataFrame:
    """Generic wrapper for convert using 'ple.convert'."""
    import polars_econ_mock as ple
    return ple.convert(series, "DATE", as_freq=as_freq, to_freq=to_freq, technique=technique, observed=observed)

def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:
    """Generic wrapper for $fishvol_rebase using 'ple.fishvol'."""
    import polars_econ_mock as ple
    return ple.fishvol(series_pairs, date_col, rebase_year)

def PA() -> pl.Expr:
    """
    Computes the values for the pa time series or variable using Polars expressions.
    Derived from FAME script(s):
        set pa = v143*4

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V143").mul(4)
    )
    return res.alias("pa")

def PAA(pa: pl.Series, pa_: pl.Series) -> pl.Series:
    """
    Computes the values for the paa time series or variable using Polars expressions.
    Derived from FAME script(s):
        set paa = pa$/pa

    Returns:
        pl.Series: Polars Series to compute the time series or variable values.
    """
    res = (
        pa_/pa
    )
    return res

def PA_() -> pl.Expr:
    """
    Computes the values for the pa$ time series or variable using Polars expressions.
    Derived from FAME script(s):
        set pa$ = v123*3

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V123").mul(3)
    )
    return res.alias("pa_")

def PBB(pa: pl.Series, paa: pl.Series) -> pl.Series:
    """
    Computes the values for the pbb time series or variable using Polars expressions.
    Derived from FAME script(s):
        set pbb = paa+pa

    Returns:
        pl.Series: Polars Series to compute the time series or variable values.
    """
    res = (
        paa+pa
    )
    return res

def PB_() -> pl.Expr:
    """
    Computes the values for the pb$ time series or variable using Polars expressions.
    Derived from FAME script(s):
        set pb$ = v1434*12

    Returns:
        pl.Expr: Polars expression to compute the time series or variable values.
    """
    res = (
        pl.col("V1434").mul(12)
    )
    return res.alias("pb_")
