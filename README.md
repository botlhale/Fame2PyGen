<div align="center">
  <img src="logos/fame2pygen.png" alt="FAME 2 Python Logo" width="250"/>
</div>

# Fame2PyGen: Automated FAME to Python Model Converter

**Transform legacy FAME economic models into modern, high-performance Python code with Polars.**

[![PyPI version](https://badge.fury.io/py/fame2pygen.svg)](https://pypi.org/project/fame2pygen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Pitch: Accelerate Your Legacy Model Modernization

Fame2PyGen is an open-source tool designed to bridge the gap between legacy FAME (Forecasting Analysis of Models in Economics) scripting and modern Python data science. By automatically converting FAME commands into efficient Polars-based Python code, it reduces migration time from months to hours while maintaining accuracy and performance.

### Key Benefits
- **Speed**: Convert complex FAME models in minutes
- **Performance**: Leverage Polars' lightning-fast DataFrame operations
- **Maintainability**: Generate clean, readable Python code
- **Integration**: Seamlessly work with modern ML/AI pipelines

## How It Works

Fame2PyGen processes a list of FAME commands and generates three Python files:

1. **`formulas.py`**: A module containing helper functions that wrap calls to the `polars_econ` library (or a mock version)
2. **`ts_transformer.py`**: An executable pipeline function that applies transformations to a Polars DataFrame
3. **`polars_econ_mock.py`**: A mock implementation for testing without the full `polars_econ` library

### Supported FAME Patterns
- Simple assignments (`vbot = 1`)
- Arithmetic operations (`v1 = v2 + v3 - v4`)
- Time-indexed variables (`v1[t+1]`)
- Point-in-time (date-indexed) assignments (`gdp["2020-01-01"] = 1000`, `cpi["2020Q1"] = 105.5`)
- **Conditional expressions** (`result = if v1 gt 100 then v2 * 2 else nd`)
- Chain operations (`$chain("a-b", "2020")`)
- PCT functions (`pct(v1[t+1])`)
- Special SHIFT_PCT patterns (forward and backward calculations)
- Convert and Fishvol functions
- Date filtering commands (`date 2020-01-01 to 2020-12-31`, `date *`)
- Frequency commands (`freq m`, `freq q`, `freq b`, `freq bus`, etc.)

### Date Range Subsetting Support

Fame2PyGen supports FAME-style date filtering commands with **actual sub-DataFrame filtering**:

- **`date <start> to <end>`**: Sets a date range filter for subsequent operations. Operations will only affect rows within the specified date range, leaving other rows unchanged (set to null for new columns).
- **`date *`**: Disables date filtering, applying operations to all dates

The generator creates expressions that use Polars' conditional logic (`pl.when().then().otherwise()`) to apply operations only within specified date ranges. This enables multiple date windows to be processed independently within the same pipeline.

Example:
```python
commands = [
    "freq m",
    "v_base = 100",
    "date 2020-01-01 to 2020-12-31",
    "v_2020 = v_base * 2",  # Only affects dates in 2020
    "date 2021-01-01 to 2021-12-31",
    "v_2021 = v_base * 3",  # Only affects dates in 2021
    "date *",
    "v_all = v_base + v_2020 + v_2021",  # Affects all dates
]
```

**Generated Polars code:**
```python
pdf = pdf.with_columns([
    pl.lit(100).alias("V_BASE")
])
# Date filter: 2020-01-01 to 2020-12-31
pdf = pdf.with_columns([
    APPLY_DATE_FILTER((pl.col("V_BASE") * pl.lit(2)), "V_2020", "2020-01-01", "2020-12-31").alias("V_2020")
])
# Date filter: 2021-01-01 to 2021-12-31
pdf = pdf.with_columns([
    APPLY_DATE_FILTER((pl.col("V_BASE") * pl.lit(3)), "V_2021", "2021-01-01", "2021-12-31").alias("V_2021")
])
# Date filter: * (all dates)
pdf = pdf.with_columns([
    (pl.col("V_BASE") + pl.col("V_2020") + pl.col("V_2021")).alias("V_ALL")
])
```

The `APPLY_DATE_FILTER` helper function ensures that:
- Operations within a date range only modify rows where the date falls within that range
- Rows outside the range receive null values for new columns
- Multiple date ranges can be used sequentially to create complex temporal patterns

### Point-in-Time Assignment Support

Fame2PyGen supports FAME-style point-in-time (date-indexed) assignments for setting specific values at particular dates in a time series:

- **`var["YYYY-MM-DD"] = value`**: Assign a value to a specific date
- **`var["YYYYQN"] = value`**: Assign a value to a specific quarter (e.g., "2020Q1")

The generator translates these to Polars operations that filter and update specific rows based on the date.

**Examples:**
```python
commands = [
    'gdp["2020-01-01"] = 1000',              # Set gdp to 1000 on Jan 1, 2020
    'cpi["2020Q1"] = 105.5',                  # Set cpi to 105.5 for Q1 2020
    'adjusted["2020-01-01"] = gdp["2019-12-31"] * 1.05',  # Reference other dates
]
```

**Generated Polars code:**
```python
pdf = POINT_IN_TIME_ASSIGN(pdf, "GDP", "2020-01-01", pl.lit(1000))
pdf = POINT_IN_TIME_ASSIGN(pdf, "CPI", "2020Q1", pl.lit(105.5))
pdf = POINT_IN_TIME_ASSIGN(pdf, "ADJUSTED", "2020-01-01", 
    lambda df: df.filter(pl.col("DATE") == "2019-12-31").select(pl.col("GDP")).item() * 1.05)
```

### Frequency Support

Fame2PyGen supports FAME frequency commands for defining time series periodicity:

- **`freq a`**: Annual frequency
- **`freq q`**: Quarterly frequency
- **`freq m`**: Monthly frequency
- **`freq w`**: Weekly frequency
- **`freq d`**: Daily frequency
- **`freq b`** or **`freq bus`**: Business day frequency (working days, excluding weekends and holidays)

The frequency setting integrates with polars-econ for time series operations like `convert()`, ensuring proper handling of business day calendars and other time-based transformations.

Example:
```python
commands = [
    "freq b",  # Set business day frequency
    "v1 = convert(v2, 'm', 'b', 'avg', 'end')",  # Convert monthly to business day
]
```

### Conditional Expressions Support

Fame2PyGen supports FAME-style conditional logic with automatic translation to Polars `.when().then().otherwise()` expressions:

**Syntax**: `variable = if <condition> then <then_expr> else <else_expr>`

**Supported comparison operators**:
- `ge` - greater than or equal (>=)
- `gt` - greater than (>)
- `le` - less than or equal (<=)
- `lt` - less than (<)
- `eq` - equal (==)
- `ne` - not equal (!=)

**Special keywords (FAME null/missing values)**:
- `nd` - null/missing value (maps to `pl.lit(None)`)
- `na` - not available (maps to `pl.lit(None)`)
- `nc` - not computed (maps to `pl.lit(None)`)

**Examples:**
```python
commands = [
    "freq m",
    "base = 100",
    "threshold = 150",
    
    # Simple conditional
    "result1 = if base gt 100 then base * 2 else nd",
    
    # Conditional with multiple variables
    "result2 = if threshold ge 150 then base * 1.5 else base",
    
    # Complex expressions in branches
    "price = 50",
    "quantity = 10", 
    "total = if price lt 100 then price * quantity else price * quantity * 1.1",
]
```

**Generated Polars code:**
```python
pdf = pdf.with_columns([
    pl.when(pl.col("BASE") > 100).then(pl.col("BASE") * 2).otherwise(pl.lit(None)).alias("RESULT1"),
    pl.when(pl.col("THRESHOLD") >= 150).then(pl.col("BASE") * 1.5).otherwise(pl.col("BASE")).alias("RESULT2"),
    pl.when(pl.col("PRICE") < 100).then(pl.col("PRICE") * pl.col("QUANTITY")).otherwise(pl.col("PRICE") * pl.col("QUANTITY") * 1.1).alias("TOTAL")
])
```

**Note on FAME date functions**: Complex FAME date functions like `dateof()`, `make()`, `date()`, `contain()`, and `end()` are currently preserved in the condition as-is and may require manual review and implementation. The conditional structure itself is fully supported.

For a complete example, see [examples/conditional_expression_example.py](examples/conditional_expression_example.py).

### LSUM Function Support

Fame2PyGen supports the FAME `LSUM` (list sum) function for summing multiple arguments with null handling:

**Syntax**: `variable = LSUM(expr1, expr2, ..., exprN)`

The LSUM function:
- Sums all provided expressions
- Treats null values as 0 (null-safe addition)
- Commonly used with conditional expressions for handling missing values

**EXISTS Function**:
The `EXISTS(variable)` function checks if a variable has a non-null value at each row:
- Returns `True` where values are non-null
- Returns `False` where values are null

**Example - Complex nested IF with LSUM:**

This example shows a FAME formula that conditionally sums multiple series, handling missing values:

```
AA = IF T GT DATEOF(A.FINALDATE,*,BEFORE,ENDING) 
     THEN ND 
     ELSE LSUM(
         (if exists(BBA) then (if BBA EQ NA then 0 ELSE IF BBA EQ NC THEN 0 ELSE IF BBA EQ ND THEN 0 ELSE BBA) else 0),
         (if exists(BBB) then (if BBB EQ NA then 0 ELSE BBB) else 0)
     )
```

**Interpretation:**
1. **Outer IF**: Checks if current time (T) is greater than a computed date from A.FINALDATE
   - If TRUE → return ND (null)
   - If FALSE → compute LSUM

2. **LSUM arguments**: Each argument is a conditional expression that:
   - First checks if the variable exists (has a value)
   - If it exists, checks if the value is one of the special missing codes (NA, NC, ND) and returns 0
   - Otherwise returns the actual value
   - If the variable doesn't exist, returns 0

3. **Python/Polars translation**:
   - `LSUM(expr1, expr2, ...)` → `expr1.fill_null(0) + expr2.fill_null(0) + ...`
   - `EXISTS(col)` → `col.is_not_null()`
   - `EQ NA/NC/ND` → comparison with `pl.lit(None)` (is_null check)
   - Nested IFs → chained `pl.when().then().otherwise()`

**Generated Python code pattern:**
```python
# LSUM sums arguments with null → 0 handling
def LSUM(*args) -> pl.Expr:
    if not args:
        return pl.lit(0)
    result = args[0].fill_null(0)
    for arg in args[1:]:
        result = result + arg.fill_null(0)
    return result

# EXISTS checks for non-null values
def EXISTS(expr: pl.Expr) -> pl.Expr:
    return expr.is_not_null()

# Usage in transformer:
pdf = pdf.with_columns([
    pl.when(condition)
      .then(pl.lit(None))
      .otherwise(
          LSUM(
              pl.when(EXISTS(pl.col("BBA")))
                .then(
                    pl.when(pl.col("BBA").is_null())
                      .then(0)
                      .otherwise(pl.col("BBA"))
                )
                .otherwise(0),
              pl.when(EXISTS(pl.col("BBB")))
                .then(
                    pl.when(pl.col("BBB").is_null())
                      .then(0)
                      .otherwise(pl.col("BBB"))
                )
                .otherwise(0)
          )
      )
      .alias("AA")
])
```

**Simple LSUM example:**
```python
commands = [
    "freq m",
    "a = 10",
    "b = 20",
    "c = 30",
    "total = lsum(a, b, c)"  # Result: 60
]
```

**Note**: Complex nested conditionals inside LSUM arguments (like the example above) may require manual refinement after generation. The LSUM function itself and simple arguments are fully supported. For complex formulas, the generated code provides a good starting point that can be adjusted as needed.


### Usage Example

```python
from fame2pygen import generate_formulas_file, generate_test_script

fame_commands = [
    "freq m",
    "vbot = 1",
    "set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))",
]

# Generate files
generate_formulas_file(fame_commands)
generate_test_script(fame_commands)

# Use the generated ts_transformer function
import polars as pl
from ts_transformer import ts_transformer

df = pl.DataFrame({"DATE": [...], "V123S": [...], "V1014S": [...]})
result = ts_transformer(df)
```

## ⚠️ Important Cautions

**Fame2PyGen is a powerful automation tool, but it's not perfect.** The generated code may require minor manual adjustments:

- **Dependency Ordering**: While we strive for correct computation levels, complex models with intricate dependencies might need tweaks
- **Edge Cases**: Rare FAME patterns or custom functions may not be fully supported
- **Validation Required**: Always test the generated code against known FAME outputs
- **Performance Tuning**: Generated code might benefit from Polars optimizations

**Recommendation**: Treat Fame2PyGen as your first draft. Review, test, and refine the output before production use.

## Installation

```bash
pip install fame2pygen
```

Or clone and install:

```bash
git clone https://github.com/yourusername/Fame2PyGen.git
cd Fame2PyGen
pip install -e .
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Architecture

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).
