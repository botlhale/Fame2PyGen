import polars as pl
from typing import List, Tuple

def A_(v123: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'a$'.
Derived from FAME script line:
    a$=v123*12
"""
    res = (
        v123*12
    )
    return res.alias("a_")

def A(v143: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'a'.
Derived from FAME script line:
    a=v143*12
"""
    res = (
        v143*12
    )
    return res.alias("a")

def PA_(v123: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'pa$'.
Derived from FAME script line:
    pa$=v123*3
"""
    res = (
        v123*3
    )
    return res.alias("pa_")

def PA(v143: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'pa'.
Derived from FAME script line:
    pa=v143*4
"""
    res = (
        v143*4
    )
    return res.alias("pa")

def AA(a: pl.Expr, a_: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'aa'.
Derived from FAME script line:
    aa=a$/a
"""
    res = (
        a_/a
    )
    return res.alias("aa")

def PAA(pa: pl.Expr, pa_: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'paa'.
Derived from FAME script line:
    paa=pa$/pa
"""
    res = (
        pa_/pa
    )
    return res.alias("paa")

def BB(a: pl.Expr, aa: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'bb'.
Derived from FAME script line:
    bb=aa+a
"""
    res = (
        aa+a
    )
    return res.alias("bb")

def PBB(pa: pl.Expr, paa: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'pbb'.
Derived from FAME script line:
    pbb=paa+pa
"""
    res = (
        paa+pa
    )
    return res.alias("pbb")

def D(bb: pl.Expr) -> pl.Expr:
    """
Computes values for FAME variable 'd'.
Derived from FAME script line:
    d={bb}
"""
    res = (
        bb
    )
    return res.alias("d")

def B_() -> pl.Expr:
    """
Computes values for FAME variable 'b$'.
Derived from FAME script line:
    b$=1234*12
"""
    res = (
        pl.lit(14808)
    )
    return res.alias("b_")

def PB_() -> pl.Expr:
    """
Computes values for FAME variable 'pb$'.
Derived from FAME script line:
    pb$=1434*12
"""
    res = (
        pl.lit(17208)
    )
    return res.alias("pb_")

def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:
    import polars_econ as ple
    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))

def CONVERT(series: 'pl.DataFrame', as_freq: str, to_freq: str, technique: str, observed: str) -> pl.DataFrame:
    import polars_econ as ple
    return ple.convert(series, 'DATE', as_freq=as_freq, to_freq=to_freq, technique=technique, observed=observed)

def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:
    import polars_econ as ple
    return ple.fishvol(series_pairs, date_col, rebase_year)
