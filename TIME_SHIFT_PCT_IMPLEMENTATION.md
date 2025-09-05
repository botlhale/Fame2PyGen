# Time Shift and PCT Function Implementation

This document explains how the Fame2PyGen tool was extended to support time series references with time shifts (like `vbots[t+1]`) and the `pct()` function from the polars-econ library.

## Problem Statement

The requirement was to add functionality for FAME scripts like:
```
vbots = vbot
v23s = v23
set vbots[t] = vbots[t+1]/(1+(pct(v23s[t+1])/100))
```

This involves two key features:
1. **Time shifts**: `vbots[t+1]` represents accessing a time series at time t+1 (future value)
2. **PCT function**: `pct(expr, offset=1)` calculates percentage change from polars-econ library

## Implementation Approach

### 1. Parser Extensions (`parser.py`)

**New Helper Functions:**
- `contains_time_shift_references(rhs)`: Detects patterns like `var[t+1]`, `var[t-2]`
- `contains_pct_function(rhs)`: Detects `pct()` function calls
- `extract_time_shift_references(rhs)`: Extracts base variable names from time shift patterns
- `extract_pct_function_args(rhs)`: Extracts variable references from pct function arguments
- `parse_set_command(line)`: Handles "set variable = expression" syntax

**Updated Parsing Logic:**
- Extended variable pattern to support square brackets: `[a-zA-Z0-9_$.]+(?:\[[^\]]+\])?`
- Added time shift and pct function reference extraction to `parse_simple_command()`
- Filtered out 't' from references (part of time shift notation, not a variable)
- Added `parse_set_command` to PARSERS list

### 2. Generator Extensions (`generators.py`)

**New Conversion Functions:**
- `convert_time_shift_expression()`: Converts `var[t+1]` to `var.shift(-1)` 
- `convert_pct_function_calls()`: Converts `pct(expr)` to `PCT(expr)`

**Time Shift Logic:**
- `t+N` (future) → `.shift(-N)` (negative shift gets future values)
- `t-N` (past) → `.shift(N)` (positive shift gets past values)

**Updated Generation Pipeline:**
- Extended `fame_expr_to_polars()` to handle time shifts and pct functions
- Added `has_pct` flag to GenerationContext
- Added PCT function definition when pct functions are used

### 3. Model Extensions (`model.py`)

**GenerationContext:**
- Added `has_pct: bool = False` field to track pct function usage

### 4. CLI Extensions (`cli.py`)

**Mock Library:**
- Added `pct(expr, offset=1)` function to mock polars_econ library
- Implements percentage change: `((expr / expr.shift(offset)) - 1) * 100`

## Generated Code Structure

### Function Definitions

For the example script, generates:
```python
def VBOTST(v23s: pl.Expr, vbots: pl.Expr) -> pl.Expr:
    """
    Computes values for FAME variable 'vbots[t]'.
    Derived from FAME script line:
        vbots[t]=vbots[t+1]/(1+(pct(v23s[t+1])/100))
    """
    res = (
        vbots.shift(-1)/(1+(PCT(v23s.shift(-1))/100))
    )
    return res.alias("vbotst")

def PCT(expr: pl.Expr, offset: int = 1) -> pl.Expr:
    import polars_econ as ple
    return ple.pct(expr, offset)
```

### Time Shift Conversion Examples

| FAME Expression | Polars Expression | Explanation |
|-----------------|-------------------|-------------|
| `var[t+1]` | `var.shift(-1)` | Get next period value |
| `var[t-1]` | `var.shift(1)` | Get previous period value |
| `var[t+2]` | `var.shift(-2)` | Get value 2 periods ahead |
| `var[t-3]` | `var.shift(3)` | Get value 3 periods back |

### PCT Function Integration

- Detects `pct(variable)` or `pct(variable[t+offset])` patterns
- Converts to `PCT(expression)` wrapper function
- Wrapper imports and calls `polars_econ.pct(expr, offset)`
- Time shifts in pct arguments are handled properly

## Testing

Comprehensive test suite added in `tests/test_time_shift_pct.py`:

- **Parsing tests**: Verify correct extraction of time shift and pct references
- **Generation tests**: Verify correct Polars code generation
- **Integration tests**: End-to-end validation with complex expressions

All existing tests continue to pass, ensuring no regressions.

## Usage Examples

### Basic Time Shift
```
# FAME
result = var[t+1]

# Generated Polars
def RESULT(var: pl.Expr) -> pl.Expr:
    res = var.shift(-1)
    return res.alias("result")
```

### PCT Function
```
# FAME  
result = pct(variable)

# Generated Polars
def RESULT(variable: pl.Expr) -> pl.Expr:
    res = PCT(variable)
    return res.alias("result")

def PCT(expr: pl.Expr, offset: int = 1) -> pl.Expr:
    import polars_econ as ple
    return ple.pct(expr, offset)
```

### Complex Expression
```
# FAME
set vbots[t] = vbots[t+1]/(1+(pct(v23s[t+1])/100))

# Generated Polars
def VBOTST(v23s: pl.Expr, vbots: pl.Expr) -> pl.Expr:
    res = vbots.shift(-1)/(1+(PCT(v23s.shift(-1))/100))
    return res.alias("vbotst")
```

This implementation follows the same patterns as existing special functions (fishvol, chain, convert) and maintains consistency with the codebase architecture.