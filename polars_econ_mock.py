"""
┌───────────────────────────────────────────┐
│            Fame2PyGen                     │
│         FAME → Python                     │
│      Economic Mock Functions              │
└───────────────────────────────────────────┘
"""
import polars as pl
from typing import List, Tuple, Optional

def fishvol(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: Optional[pl.Expr] = None, rebase_year: Optional[int] = None) -> pl.Expr:
    """
    Mock implementation of Fisher volume index calculation.

    Args:
        series_pairs: List of tuples containing (quantity, price) expressions
        date_col: Date column expression (unused in mock)
        rebase_year: Year to rebase the index to (unused in mock)

    Returns:
        pl.Expr: Fisher volume index expression (mock: sum of quantities)
    """
    quantities = [pair[0] for pair in series_pairs]
    if len(quantities) == 1:
        return quantities[0]
    return pl.sum_horizontal(quantities)

def chain(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: Optional[pl.Expr] = None) -> pl.Expr:
    """
    Mock implementation of chain-linked operation over multiple (price, quantity) pairs.

    Args:
        price_quantity_pairs: List of (price_expr, quantity_expr) tuples
        date_col: Date column expression (unused in mock)

    Returns:
        pl.Expr: Combined expression (sum of price * quantity products)
    """
    products = [price * quantity for price, quantity in price_quantity_pairs]
    if len(products) == 1:
        return products[0]
    return pl.sum_horizontal(products)

def convert(series, date_col: Optional[str] = None, as_freq: Optional[str] = None, to_freq: Optional[str] = None, technique: Optional[str] = None, observed: Optional[str] = None):
    """
    Mock implementation of frequency conversion.

    Args:
        series: Input dataframe or lazyframe
        date_col: Name of date column
        as_freq: Source frequency
        to_freq: Target frequency
        technique: Conversion technique
        observed: Observation method

    Returns:
        Input unchanged.
    """
    return series

def pct(series: pl.Expr, lag: int = 1) -> pl.Expr:
    """
    Mock implementation of percentage change calculation.
    """
    return series.pct_change(lag)

def overlay(series1: pl.Expr, series2: pl.Expr) -> pl.Expr:
    """
    Mock implementation of overlay function.
    Overlays series2 onto series1, filling nulls in series1 with values from series2.
    """
    return series1.fill_null(series2)

def interp(series: pl.Expr, method: str = "linear") -> pl.Expr:
    """
    Mock implementation of interpolation.
    """
    return series.interpolate()

def mave(series: pl.Expr, window: int) -> pl.Expr:
    """
    Mock implementation of moving average.
    """
    return series.rolling_mean(window_size=window)
