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
