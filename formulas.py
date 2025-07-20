import polars as pl
import ple

def CONVERT(df, series, freq, method, period):
    if freq == 'q':
        df_c = df.filter(pl.col("date").dt.is_quarter_end())
    elif freq == 'a':
        df_c = df.filter(pl.col("date").dt.is_year_end())
    else:
        df_c = df
    return ple.convert(df_c[series], freq, method, period)

def FISHVOL(df, vol_list, price_list, year=None):
    if year is not None:
        df = df.filter(pl.col("date") > f"{year}-01-01")
    pairs = [(df[v], df[p]) for v, p in zip(vol_list, price_list)]
    return ple.fishvol(pairs, year=year)

def CHAIN(df, series_list, base_year):
    series_objs = [df[col] for col in series_list]
    return ple.chain(series_objs, base_year)

def SUM_HORIZONTAL(df, cols):
    return pl.sum_horizontal([df[col] for col in cols])

def DECLARE_SERIES(df, name):
    return pl.lit(None, dtype=pl.Float64).alias(name)