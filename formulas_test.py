import polars as pl
from typing import List, Tuple

def APPLY_DATE_FILTER(expr: pl.Expr, col_name: str, start_date: str, end_date: str, date_col: str = 'DATE', preserve_existing: bool = False) -> pl.Expr:
    """Apply expression only to rows within date range, using null or existing values for other rows.
    
    Args:
        expr: Polars expression to apply
        col_name: Name of the column being updated
        start_date: Start date string (e.g., '2020-01-01' or '01Feb2020')
        end_date: End date string (e.g., '2020-12-31' or '*' for today)
        date_col: Name of the date column (default 'DATE')
        preserve_existing: If True, preserve existing column values outside date range (default False)
    
    Returns:
        Expression that applies to filtered rows only
    """
    import polars as pl
    from datetime import datetime, date as dt_date
    
    # Parse start date - support multiple formats
    def parse_date(date_str):
        # Try YYYY-MM-DD format
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            pass
        # Try ddMMMYYYY format (e.g., 01Feb2020)
        try:
            return datetime.strptime(date_str, '%d%b%Y').date()
        except ValueError:
            pass
        # Try YYYYQN format (e.g., 2020Q1)
        import re
        m = re.match(r'^(\d{4})Q([1-4])$', date_str, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            quarter = int(m.group(2))
            month = (quarter - 1) * 3 + 1
            return dt_date(year, month, 1)
        raise ValueError(f'Cannot parse date: {date_str}')
    
    start = parse_date(start_date)
    
    # Handle special end date '*' (up to today)
    if end_date == '*':
        end = dt_date.today()
    else:
        end = parse_date(end_date)
    
    # Apply expression only within date range
    # If preserve_existing is True, use existing column values outside the range
    # Otherwise, use null for rows outside the range
    otherwise_expr = pl.col(col_name) if preserve_existing else pl.lit(None)
    return pl.when(
        (pl.col(date_col) >= start) & (pl.col(date_col) <= end)
    ).then(expr).otherwise(otherwise_expr)

def CONVERT(series: pl.DataFrame, *args) -> pl.Expr:
    import polars_econ as ple
    # Handle both standard and custom convert signatures
    if len(args) == 4:
        # Standard: as_freq, to_freq, technique, observed
        return ple.convert(series, 'DATE', as_freq=args[0], to_freq=args[1], technique=args[2], observed=args[3])
    elif len(args) == 3:
        # Custom 3-param variant
        return ple.convert(series, 'DATE', *args)
    else:
        # Generic fallback - pass all args
        return ple.convert(series, 'DATE', *args)

