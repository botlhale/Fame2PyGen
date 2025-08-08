<div align="center">
  <img src="logos/fame2pygen.png" alt="Fame2PyGen Logo" width="600"/>
</div>

# Fame2PyGen

Fame2PyGen is a toolset for automatically converting FAME-style formula scripts into executable, modular Python code using [Polars](https://pola.rs/) for DataFrame operations. It supports generic data transformation functions, auto-generated pipelines, and a mock FAME-style backend via `ple.py`.

## 🎨 Logo Options

We've created three logo concepts for the project:
- Text-based Logo: Clean, professional text design
- ASCII Art Logo: Visual representation with transformation arrows
- Modern Tech Logo: Contemporary graphical concept

See the `/logos` directory for complete designs and usage guidelines.

## ✨ Enhanced Existing Functions

### CHAIN Function (consolidated)
- chain(): Chain-linked operations over several (price, quantity) tuple pairs
- Input: list of (price_expr, quantity_expr) tuples
- Note: There is no separate CHAINSUM function; summation across pairs is handled within `chain()` by summing price*quantity components.

### FISHVOL Function
- fishvol_rebase(): Fisher volume operations (mock/enhanced placeholder in `ple.py`)
- Optional variable lists and explicit dependency lists supported at the pipeline level for ordering

### CONVERT Function
- convert(): Frequency conversion (mock implementation; production behavior to be specified)
- Optional dependency parameters can be used for pipeline-level ordering

## Project Structure

```
fame2pygen/
├── README.md
├── logos/                    # Logo designs and documentation
├── ple.py                    # Mock/enhanced backend functions (CHAIN/FISHVOL/CONVERT)
├── polars_econ_mock.py       # Mock FAME function implementations
├── formulagen.py             # Parsers for FAME-like scripts
├── write_formulagen.py       # Main generator using the parser
├── formulas.py               # Auto-generated: expression-based calculation functions
├── convpy4rmfame.py          # Auto-generated: computation pipeline
├── test_enhancements.py      # Test suite for branding and function presence
└── ...
```

### Expression-Based Formula Generation
- Functions return `pl.Expr` objects with proper aliasing
- Support for both expression and series-based operations
- Comprehensive docstring generation with FAME script derivation
- Type-safe function signatures

### Layered Computation Pipeline
- Structured computation using `with_columns()` approach
- Level-based dependency management (pipeline ordering)
- Frequency conversion handling (mock)
- Final output formatting with unpivot and column renaming (as needed)

### Extended FAME Function Support
- COPY: Series duplication
- PCT: Percentage change calculations with configurable lags
- INTERP: Linear and advanced interpolation methods
- OVERLAY: Series combination with null-filling
- MAVE: Moving averages with configurable windows
- MAVEC: Centered moving averages
- FISHVOL: Fisher volume index calculations
- CHAIN: Chain-linked index computations over multiple (price, quantity) pairs

## Quickstart: Using CHAIN

Below is a minimal example demonstrating how to use the consolidated `CHAIN` function over multiple (price, quantity) pairs:

```python
import polars as pl
import ple  # Fame2PyGen backend functions

# Sample input
df = pl.DataFrame({
    'date': pl.date_range(pl.date(2022, 1, 1), pl.date(2022, 12, 1), '1mo', eager=True),
    'prices_a': pl.Series(range(1, 12 + 1)),
    'quantities_a': pl.Series(range(10, 10 + 12)),
    'prices_b': pl.Series(range(2, 2 + 12)),
    'quantities_b': pl.Series(range(5, 5 + 12)),
})

# Build a chain-linked composite value across multiple pairs
chain_expr = ple.chain([
    (pl.col('prices_a'), pl.col('quantities_a')),
    (pl.col('prices_b'), pl.col('quantities_b')),
], date_col=pl.col('date'))  # date_col is optional in the mock placeholder

df = df.with_columns(chain_expr.alias('chain_value'))

print(df.select('date', 'chain_value'))
```

Notes:
- In this mock/enhanced placeholder, `chain()` computes a sum of products (price * quantity) across all pairs, per row.
- If you require base-year indexing or temporal chaining, you can extend `ple.chain()` or implement rebasing logic at the pipeline level.
