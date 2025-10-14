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
