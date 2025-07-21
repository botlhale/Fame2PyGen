<div align="center">
  <img src="logos/fame2pygen.png" alt="Fame2PyGen Logo" width="600"/>
</div>

# Fame2PyGen

**Fame2PyGen** is a comprehensive toolset for automatically converting FAME-style formula scripts into executable, modular Python code using [Polars](https://pola.rs/) for DataFrame operations. It supports generic data transformation functions, auto-generated pipelines, and extensive FAME function coverage via the `polars_econ_mock.py` module.

## 🎨 Logo Options

We've created three logo concepts for the project:

- **Text-based Logo**: Clean, professional text design
- **ASCII Art Logo**: Visual representation with transformation arrows  
- **Modern Tech Logo**: Contemporary graphical concept

See the `/logos` directory for complete designs and usage guidelines.

## Project Structure

```
fame2pygen/
├── README.md
├── logos/                    # Logo designs and documentation
│   ├── README.md
│   ├── logo1_textbased.md
│   ├── logo2_ascii.md
│   └── logo3_modern.md
├── ple.py                    # Enhanced mock backend for FAME-style calculations
├── polars_econ_mock.py      # Extended FAME function implementations
├── formulagen.py            # Enhanced parsers for FAME script commands
├── write_formulagen.py      # Main generator with extended function support
├── formulas.py              # Auto-generated: expression-based calculation functions
├── convpy4rmfame.py         # Auto-generated: layered computation pipeline
├── FAME_conversion_examples.md  # Comprehensive usage examples
└── test_enhancements.py     # Test suite for all functionality
```

## ✨ Enhanced Features

### Expression-Based Formula Generation
- Functions now return `pl.Expr` objects with proper aliasing
- Support for both expression and series-based operations
- Comprehensive docstring generation with FAME script derivation
- Type-safe function signatures

### Layered Computation Pipeline
- Structured computation using `with_columns()` approach
- Level-based dependency management
- Proper frequency conversion handling
- Final output formatting with unpivot and column renaming

### Extended FAME Function Support
- **COPY**: Series duplication
- **PCT**: Percentage change calculations with configurable lags
- **INTERP**: Linear and advanced interpolation methods
- **OVERLAY**: Series combination with null-filling
- **MAVE**: Moving averages with configurable windows
- **MAVEC**: Centered moving averages
- **FISHVOL**: Fisher volume index calculations
- **CHAIN**: Chain-linked index computations
- **CONVERT**: Frequency conversion with multiple techniques

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/botlhale/fame2pygen.git
cd fame2pygen
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed.  
Install [Polars](https://pola.rs/) using pip:

```bash
pip install polars
```

### 3. Test the enhanced functionality

```bash
python test_enhancements.py
```

### 4. Generate formulas and pipeline

Edit your FAME script inside `write_formulagen.py` (the `fame_script` variable).

Run:

```bash
python write_formulagen.py
```

This will generate:
- `formulas.py`: All expression-based calculation functions
- `convpy4rmfame.py`: The layered computation pipeline

### 5. Run the generated pipeline

```bash
python convpy4rmfame.py
```

You should see output like:

```
Computation finished successfully!
```

## 🚀 Usage Examples

### Basic FAME Script Conversion

**Original FAME:**
```fame
series gdp_real, price_index
set gdp_growth = pct(gdp_real, 4)
set smoothed_gdp = mave(gdp_real, 12)
set volume_index = $fishvol_rebase({gdp_real}, {price_index}, 2020)
```

**Generated Python:**
```python
pdf = pdf.with_columns([
    ple.pct(pl.col("gdp_real"), lag=4).alias("gdp_growth"),
    ple.mave(pl.col("gdp_real"), window=12).alias("smoothed_gdp"),
    FISHVOL(series_pairs=[(pl.col("gdp_real"), pl.col("price_index"))], 
            date_col=pl.col("DATE"), rebase_year=2020).alias("volume_index")
])
```

### Advanced Functions

```python
# Percentage changes with custom lags
pdf = pdf.with_columns([
    ple.pct(pl.col("quarterly_data"), lag=4).alias("yoy_growth"),
    ple.pct(pl.col("monthly_data"), lag=1).alias("mom_change")
])

# Moving averages (simple and centered)
pdf = pdf.with_columns([
    ple.mave(pl.col("volatile_series"), window=3).alias("smooth_3"),
    ple.mavec(pl.col("seasonal_data"), window=12).alias("deseasonalized")
])

# Data overlay and interpolation
pdf = pdf.with_columns([
    ple.overlay(pl.col("primary_data"), pl.col("backup_data")).alias("combined"),
    ple.interp(pl.col("sparse_data")).alias("filled_data")
])
```

See `FAME_conversion_examples.md` for comprehensive examples.

## 🧪 Testing

Run the test suite to verify all functionality:

```bash
python test_enhancements.py
```

The test suite covers:
- Logo creation and documentation
- Expression-based formula format
- Layered computation pipeline
- All extended FAME functions
- End-to-end integration

## 🔧 Customization

- **Custom FAME Scripts**: Edit `fame_script` in `write_formulagen.py`
- **Function Extensions**: Add new functions to `polars_econ_mock.py`
- **Parser Extensions**: Extend `formulagen.py` with new command parsers
- **Output Formatting**: Modify the final formatting section in generated pipelines

## 📚 Documentation

- `FAME_conversion_examples.md`: Complete function reference and examples
- `/logos/README.md`: Logo usage guidelines and variations
- Inline code documentation with FAME script derivations
- Type hints for all function signatures

## How It Works

1. **Enhanced Parsing**: `formulagen.py` recognizes extended FAME syntax
2. **Mock Backend**: `polars_econ_mock.py` provides Polars-native implementations
3. **Expression Generation**: `formulas.py` contains reusable expression builders
4. **Pipeline Assembly**: `convpy4rmfame.py` implements layered computation structure
5. **Integration**: Full end-to-end conversion from FAME scripts to executable Python

## 🎯 Benefits

- **Modern Stack**: Built on Polars for high-performance data processing
- **Type Safety**: Full type annotations and expression validation
- **Extensibility**: Easy to add new FAME functions and operations
- **Maintainability**: Clear separation between parsing, computation, and output
- **Performance**: Lazy evaluation and optimized computation graphs
- **Comprehensive**: Covers major FAME functionality with room for extension

## License

MIT  
Copyright (c) 13668754 Canada Inc
