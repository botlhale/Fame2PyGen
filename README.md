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
- Chain operations (`$chain("a-b", "2020")`)
- PCT functions (`pct(v1[t+1])`)
- Special SHIFT_PCT patterns (forward and backward calculations)
- Convert and Fishvol functions
- Date filtering commands (`date 2020-01-01 to 2020-12-31`, `date *`)

### Date Filtering Support

Fame2PyGen now supports FAME-style date filtering commands:

- **`date <start> to <end>`**: Sets a date range filter for subsequent operations
- **`date *`**: Disables date filtering, applying operations to all dates

The generator tracks date filter state and adds comments to the generated code indicating which operations are affected by date filters. This allows for proper code review and future implementation of actual filtering logic.

Example:
```python
commands = [
    "freq m",
    "date 2020-01-01 to 2020-12-31",
    "v1 = v2 + v3",  # Only affects dates in 2020
    "date *",
    "v4 = v5 + v6",  # Affects all dates
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
