# Fame2PyGen Architecture

## Overview

Fame2PyGen consists of two main modules that work together to convert FAME scripts to Python:

1. **`formulas_generator.py`**: Formula parsing and rendering engine
2. **`fame2py_converter.py`**: Pipeline orchestration and code generation

## Core Components

### formulas_generator.py

#### Responsibilities
- **Text Normalization**: Cleans and standardizes FAME commands
- **Tokenization**: Parses FAME variables, time indexes, and expressions
- **Expression Rendering**: Converts FAME syntax to Polars expressions
- **Function Generation**: Dynamically creates helper functions based on detected patterns

#### Key Functions
- `parse_fame_formula()`: Parses individual FAME commands
- `render_polars_expr()`: Converts expressions to Polars syntax
- `generate_polars_functions()`: Creates helper function definitions

### fame2py_converter.py

#### Responsibilities
- **Dependency Analysis**: Builds a DAG of variable dependencies
- **Level Computation**: Determines execution order using topological sort
- **Code Generation**: Produces the main transformation pipeline

#### Key Functions
- `analyze_dependencies()`: Constructs dependency graph
- `get_computation_levels()`: Performs topological sort
- `generate_test_script()`: Creates the ts_transformer.py file

## Supported Patterns and Conversion Logic

| FAME Pattern | Conversion Logic | Output Example |
|--------------|------------------|----------------|
| Simple Assignment | Direct mapping | `ASSIGN_SERIES("VBOT", pl.lit(1))` |
| Arithmetic | Operator detection | `ADD_SERIES("V1", pl.col("V2"), pl.col("V3"))` |
| Time Indexing | Shift transformation | `pl.col("V1").shift(-1)` |
| Point-in-Time Assignment | Date filtering and update | `POINT_IN_TIME_ASSIGN(pdf, "GDP", "2020-01-01", pl.lit(1000))` |
| SHIFT_PCT Backwards | Batch processing | `SHIFT_PCT_BACKWARDS_MULTIPLE(...)` |
| Chain Operations | Function wrapping | `CHAIN(price_quantity_pairs=[...])` |
| Frequency Commands | Metadata tracking | `{"type": "freq", "freq": "b"}` |

### Frequency Support

Fame2PyGen supports all standard FAME frequency codes:

| FAME Code | Frequency | Description |
|-----------|-----------|-------------|
| `a` | Annual | Yearly data |
| `q` | Quarterly | Quarterly data |
| `m` | Monthly | Monthly data |
| `w` | Weekly | Weekly data |
| `d` | Daily | Daily data (all days) |
| `b` or `bus` | Business | Business days (excludes weekends/holidays) |

The frequency setting is parsed and made available to polars-econ functions (like `convert()`) which handle frequency-specific transformations including business day calendar adjustments.

### Point-in-Time Assignment Support

Fame2PyGen now supports FAME-style date-indexed assignments for updating specific dates in time series:

**Syntax Detection:**
- Pattern: `variable["date"] = expression` or `variable['date'] = expression`
- Supports formats: `"YYYY-MM-DD"` (e.g., "2020-01-01") and `"YYYYQN"` (e.g., "2020Q1")

**Code Generation:**
- Simple values: `POINT_IN_TIME_ASSIGN(pdf, "GDP", "2020-01-01", pl.lit(1000))`
- References to other dates: Uses lambda expressions to extract values at specific dates
- Example: `adjusted["2020-01-01"] = gdp["2019-12-31"] * 1.05` generates:
  ```python
  POINT_IN_TIME_ASSIGN(pdf, "ADJUSTED", "2020-01-01", 
      lambda df: df.filter(pl.col("DATE") == "2019-12-31").select(pl.col("GDP")).item() * 1.05)
  ```

**Helper Function:**
The `POINT_IN_TIME_ASSIGN` function:
1. Parses the date string (supporting multiple formats)
2. Filters the DataFrame for the target date
3. Evaluates the value expression (supports both `pl.Expr` and callable)
4. Updates the specific row via a join operation

## Dependency Analysis Details

The system uses a directed acyclic graph (DAG) to model dependencies:

- **Nodes**: Variables (lowercased)
- **Edges**: Dependencies (e.g., `v1` depends on `v2` if `v1 = v2 + v3`)
- **Time Indexing**: Handled by normalizing to base names and tracking offsets

Topological sort ensures variables are computed before they're referenced.

## Error Handling and Edge Cases

- **Cyclic Dependencies**: Detected and reported as exceptions
- **Unsupported Patterns**: Fall back to generic expression rendering
- **Time Index Mismatches**: Warned and handled via normalization

## Extensibility

To add support for new FAME patterns:
1. Update `parse_fame_formula()` with new regex patterns
2. Add conversion logic in `fame2py_converter.py`
3. Include helper functions if needed
4. Update tests and documentation

## Performance Considerations

- **Lazy Evaluation**: Polars expressions are lazy, enabling query optimization
- **Batch Processing**: Multiple SHIFT_PCT operations are batched for efficiency
- **Minimal Overhead**: Only generates necessary helper functions

## Testing

Unit tests in `tests/test_fame2pygen.py` cover:
- Pattern recognition
- Dependency ordering
- Code generation accuracy
- Edge cases and error conditions
