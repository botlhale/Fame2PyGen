#!/usr/bin/env python3
"""
Example demonstrating date range subsetting with actual DataFrame filtering.

This example shows how date commands create sub-DataFrames that are processed
independently, enabling complex temporal patterns and multiple date windows.
"""

import polars as pl
from datetime import date
from fame2pygen.fame2py_converter import generate_test_script, generate_formulas_file

# Define FAME commands with multiple date ranges
fame_commands = [
    "freq m",
    
    # Base values (apply to all dates)
    "v_base = 100",
    
    # Operations for 2020 only
    "date 2020-01-01 to 2020-12-31",
    "v_2020 = v_base * 2",
    "v_2020_adj = v_2020 + 10",
    
    # Operations for 2021 only
    "date 2021-01-01 to 2021-12-31",
    "v_2021 = v_base * 3",
    
    # Reset to all dates for final computation
    "date *",
    "v_combined = v_base + v_2020 + v_2021",
]

# Generate the transformation code
print("Generating formulas.py...")
generate_formulas_file(fame_commands, "date_range_formulas.py")

print("Generating ts_transformer.py...")
generate_test_script(fame_commands, "date_range_transformer.py")

print("\nGenerated files:")
print("  - date_range_formulas.py: Helper functions")
print("  - date_range_transformer.py: Transformation pipeline")

print("\n" + "="*60)
print("Generated transformation code:")
print("="*60)

with open("date_range_transformer.py", "r") as f:
    print(f.read())

print("\n" + "="*60)
print("Testing with sample data:")
print("="*60)

# Import the generated transformer
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("formulas", "date_range_formulas.py")
formulas_module = importlib.util.module_from_spec(spec)
sys.modules["formulas"] = formulas_module
spec.loader.exec_module(formulas_module)

spec = importlib.util.spec_from_file_location("ts_transformer", "date_range_transformer.py")
ts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts_module)

# Create test DataFrame with dates spanning 2019-2022
test_df = pl.DataFrame({
    "DATE": [
        date(2019, 6, 1),
        date(2019, 12, 1),
        date(2020, 3, 1),
        date(2020, 6, 1),
        date(2020, 12, 1),
        date(2021, 3, 1),
        date(2021, 6, 1),
        date(2021, 12, 1),
        date(2022, 6, 1),
    ]
})

print("\nInput DataFrame:")
print(test_df)

# Apply transformations
result = ts_module.ts_transformer(test_df)

print("\nTransformed DataFrame:")
print(result)

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("- V_BASE: Should be 100 for all dates")
print("- V_2020: Should be 200 only for 2020 dates, null elsewhere")
print("- V_2020_ADJ: Should be 210 only for 2020 dates, null elsewhere")
print("- V_2021: Should be 300 only for 2021 dates, null elsewhere")
print("- V_COMBINED: Should be v_base + v_2020 + v_2021 for all dates")
print("  (null values in v_2020/v_2021 will result in null for v_combined)")

print("\n" + "="*60)
print("Notice how date range subsetting enables:")
print("="*60)
print("✓ Multiple independent date windows in a single pipeline")
print("✓ Operations that only affect specific time periods")
print("✓ Clean separation of temporal logic")
print("✓ Efficient processing without creating separate DataFrames")
print("="*60)
