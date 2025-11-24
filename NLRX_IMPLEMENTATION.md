# NLRX Function Implementation Summary

## Overview
Added support for three new FAME functions to Fame2PyGen:
- `firstvalue(series)` - Extract the first non-null value from a series
- `lastvalue(series)` - Extract the last non-null value from a series  
- `nlrx(lambda, y, w1, w2, w3, w4, gss, gpr, ...)` - Non-linear relaxation function from polars_econ

## Changes Made

### 1. formulas_generator.py
- **Added to FUNCTION_NAMES**: `nlrx`, `firstvalue`, `lastvalue`
- **Added parsing support** in `parse_fame_formula()`:
  - `firstvalue(series)` pattern (lines ~785-790)
  - `lastvalue(series)` pattern (lines ~791-796)
  - `nlrx(...)` pattern with 8+ parameters (lines ~798-810)
- **Added function generation** in `generate_polars_functions()`:
  - Detection flags: `has_nlrx`, `has_firstvalue`, `has_lastvalue`
  - FIRSTVALUE helper: Returns `expr.drop_nulls().first()`
  - LASTVALUE helper: Returns `expr.drop_nulls().last()`
  - NLRX wrapper: Calls `ple.nlrx(df, lamb, y=..., w1=..., ...)`

### 2. fame2py_converter.py
- **Added handling in transformer generation**:
  - `firstvalue` type: Wraps result in `pl.lit()` to broadcast scalar (lines ~425-432)
  - `lastvalue` type: Wraps result in `pl.lit()` to broadcast scalar (lines ~434-441)
  - `nlrx` type: Direct DataFrame assignment since NLRX returns DataFrame (lines ~443-463)

### 3. polars_econ_mock.py
- **Added mock implementation** of `nlrx()`:
  - Signature: `def nlrx(df, lamb, *, y="y", w1="w1", w2="w2", w3="w3", w4="w4", gss="gss", gpr="gpr")`
  - Returns input DataFrame (mock behavior)

### 4. tests/test_nlrx.py (NEW)
- **20 comprehensive tests** covering:
  - Parsing of all three functions
  - Function generation in formulas.py
  - Code generation in ts_transformer.py
  - Integration with date filtering
  - Edge cases (case insensitivity, extra parameters, numeric literals)

### 5. examples/nlrx_example.py (NEW)
- Complete working example demonstrating the FAME pattern from the problem statement
- Shows generated code for all three functions
- Includes usage instructions

## Problem Statement Pattern

The implementation correctly handles this FAME scripting pattern:

```
start = firstvalue(a)
end = lastvalue(a)
lambda20 = 20
set <date start to end> b1 = 1
set <date start-7 to end> b1 = 0
set <date start to end> b2 = 0
set <date start to end> b3 = 0
set <date start to end> b4 = 0
set <date start to end> c = 0
set <date start to end> d = 0
a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d, begsa, endmona)
```

## Generated Python Code

### formulas.py
```python
def FIRSTVALUE(expr: pl.Expr) -> pl.Expr:
    """Get the first non-null value from a series."""
    return expr.drop_nulls().first()

def LASTVALUE(expr: pl.Expr) -> pl.Expr:
    """Get the last non-null value from a series."""
    return expr.drop_nulls().last()

def NLRX(df: pl.DataFrame, lamb: pl.Expr, y: pl.Expr, w1: pl.Expr, ...) -> pl.DataFrame:
    """Wrapper for polars_econ nlrx function."""
    import polars_econ as ple
    # Extract lambda value and column names
    lamb_val = df.select(lamb).item()
    y_col = get_col_name(y) or 'y'
    # ... (similar for other parameters)
    return ple.nlrx(df, lamb_val, y=y_col, w1=w1_col, ...)
```

### ts_transformer.py
```python
def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:
    # Get first and last values
    pdf = pdf.with_columns([
        pl.lit(FIRSTVALUE(pl.col("A"))).alias("START"),
        pl.lit(LASTVALUE(pl.col("A"))).alias("END"),
        pl.lit(20).alias("LAMBDA20")
    ])
    
    # Date-filtered assignments
    pdf = pdf.with_columns([
        APPLY_DATE_FILTER(pl.lit(1), "B1", "start", "end").alias("B1"),
        # ... (other filtered columns)
    ])
    
    # NLRX call
    pdf = NLRX(pdf, pl.col("LAMBDA20"), pl.col("A"), 
               pl.col("B1"), pl.col("B2"), ...)
    return pdf
```

## Key Design Decisions

1. **Scalar Broadcasting**: `firstvalue()` and `lastvalue()` return scalars, so we wrap them in `pl.lit()` to broadcast across all rows.

2. **DataFrame Handling**: `nlrx()` returns a DataFrame (not an expression), so we:
   - Flush pending column operations before calling NLRX
   - Directly reassign the result to `pdf`

3. **Parameter Extraction**: The NLRX wrapper extracts:
   - Lambda value from the expression or literal
   - Column names from `pl.col("NAME")` expressions using regex

4. **Flexibility**: The parser accepts 8+ parameters for nlrx, allowing for additional parameters beyond the minimum required.

## Testing

All 91 tests pass:
- 71 existing tests (no regressions)
- 20 new tests for nlrx functionality

Test coverage includes:
- Unit tests for parsing
- Function generation tests
- Integration tests with date filtering
- Code generation tests
- Edge case tests

## Compatibility

The implementation:
- Maintains backward compatibility with all existing features
- Follows the existing code patterns (e.g., similar to `convert()`, `fishvol()`)
- Integrates seamlessly with date filtering
- Preserves the dependency analysis and computation level ordering

## Files Modified

1. `fame2pygen/formulas_generator.py` - Parser and function generator
2. `fame2pygen/fame2py_converter.py` - Code generator
3. `fame2pygen/polars_econ_mock.py` - Mock implementation
4. `tests/test_nlrx.py` - New comprehensive tests
5. `examples/nlrx_example.py` - New usage example
