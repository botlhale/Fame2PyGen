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
