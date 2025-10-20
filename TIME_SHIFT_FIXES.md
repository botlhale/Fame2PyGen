# Time Shift/Lag/Lead Parsing Fixes

## Summary

This update fixes the parsing and Polars translation for FAME time-indexed expressions. The implementation ensures proper handling of time series operations with lag/lead notation (e.g., `var[t+1]`, `var[t-1]`) and the `pct()` function.

## Issues Fixed

### 1. Chain/MChain Parsing with "set" Prefix
**Problem**: Commands like `set abcd = $chain("a - b", "2020")` were not being recognized as chain operations.

**Solution**: Updated `parse_fame_formula` to call `_parse_chain_top_level` with the normalized string instead of the raw string, allowing proper detection after "set" prefix removal.

### 2. SHIFT_PCT Pattern Matching
**Problem**: The SHIFT_PCT pattern regex was matching against the full assignment including LHS (e.g., `v123s[t] = v123s[t+1]/...`), which failed because the LHS contained `[t]`.

**Solution**: Modified the parsing to extract and match only the RHS portion of the assignment, and properly extract the target variable from the LHS.

### 3. Convert Function Parameter Handling
**Problem**: The convert function regex only accepted exactly 4 parameters.

**Solution**: Updated the regex to accept 4 or more parameters and improved argument parsing to handle quoted strings correctly.

### 4. SHIFT_PCT_BACKWARDS Detection
**Problem**: The logic for detecting backwards SHIFT_PCT patterns (where `var[t] = var[t+1]/...`) was not working correctly.

**Solution**: Implemented proper detection that checks:
- LHS has `[t]` or `[t+0]` (current time)
- RHS has `[t+N]` with N > 0 (future time reference)
- This indicates a backwards calculation pattern

### 5. Dependency Cycle with Self-References
**Problem**: SHIFT_PCT patterns like `v123s[t] = v123s[t+1]/...` created self-referential dependencies causing cycle detection errors.

**Solution**: Modified dependency analysis to:
- Skip SHIFT_PCT type formulas entirely (they're handled specially)
- Skip self-references in time-indexed variables
- This prevents false cycle detection

### 6. Time-Indexed Token Assignment
**Problem**: Simple assignments like `v1 = v2[t+1]` were being incorrectly parsed as arithmetic operations because the `+` inside `[t+1]` was treated as an addition operator.

**Solution**: Updated the assign-series pattern to recognize time-indexed tokens (e.g., `v2[t+1]`) as single tokens.

### 7. Function Names as Column References
**Problem**: The `pct` function name was being converted to `pl.col("PCT")` instead of remaining as a function name.

**Solution**: Modified `_build_sub_map_and_placeholders` to skip tokenizing function names (pct, convert, fishvol_rebase, chain, mchain), allowing them to be processed correctly by the `render_polars_expr` function.

## Time Shift Conversion Table

| FAME Expression | Polars Expression | Description |
|-----------------|-------------------|-------------|
| `var[t]` | `pl.col("VAR")` | Current period |
| `var[t+1]` | `pl.col("VAR").shift(-1)` | Next period (lead) |
| `var[t+2]` | `pl.col("VAR").shift(-2)` | Two periods ahead |
| `var[t-1]` | `pl.col("VAR").shift(1)` | Previous period (lag) |
| `var[t-2]` | `pl.col("VAR").shift(2)` | Two periods back |

## Examples

### Simple Time Shift
```fame
v1 = v2[t+1]
```
Generates:
```python
ASSIGN_SERIES("V1", pl.col("V2").shift(-1))
```

### PCT with Time Shift
```fame
v5 = pct(v6[t+1])
```
Generates:
```python
(PCT(pl.col("V6").shift(-1))).alias('V5')
```

### SHIFT_PCT Backwards Pattern
```fame
set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))
```
Generates:
```python
pdf = SHIFT_PCT_BACKWARDS_MULTIPLE(pdf, "2016-12-31", "1981-03-31", 
    [('V123S', 'V1014S')], offsets=[1])
```

### Arithmetic with Time Shifts
```fame
v_sum = v_a[t+1] + v_b[t-1]
```
Generates:
```python
ASSIGN_SERIES("V_SUM", pl.col("V_A").shift(-1) + pl.col("V_B").shift(1))
```

## Test Coverage

All 17 existing tests pass, including:
- `test_parse_shift_pct_pattern` - Validates SHIFT_PCT parsing
- `test_parse_chain_operation` - Validates chain operation parsing
- `test_generate_functions` - Validates SHIFT_PCT_BACKWARDS generation
- `test_convert_function` - Validates convert function parsing
- `test_time_indexed_variables` - Validates time shift parsing
- `test_pct_function` - Validates PCT function usage

## Security

CodeQL analysis found 0 security vulnerabilities in the updated code.
