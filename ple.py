import polars as pl

def fishvol(series_pairs, year=None):
    result = [vol.sum() for vol, price in series_pairs]
    return pl.Series("fishvol", result)

def convert(series, freq, method, period):
    return series

def chain(series_list, base_year):
    result = series_list[0]
    for s in series_list[1:]:
        result = result * s
    return result

def pct(series, lag=1):
    return series.pct_change(n=lag)

def overlay(series1, series2):
    return series1 + series2

def interp(series, method="linear"):
    return series.interpolate()