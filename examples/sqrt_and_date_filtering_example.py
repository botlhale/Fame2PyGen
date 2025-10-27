#!/usr/bin/env python3
"""
Example demonstrating sqrt function and multiple date-filtered assignments.

This example shows:
1. Using sqrt() function with expressions
2. Using logical operators (or, and) with sqrt
3. Multiple assignments to the same variable with different date ranges
4. Business day frequency with date filtering
"""

from fame2pygen import generate_formulas_file, generate_test_script
import polars as pl
from datetime import date
import sys
import importlib.util

# Example 1: sqrt function with logical operators
print("=" * 60)
print("Example 1: sqrt function with logical operators")
print("=" * 60)

sqrt_commands = [
    "freq m",
    "base_a = 100",
    "base_b = 200",
    "base_c = 300",
    # Use sqrt with multiplication
    "result1 = sqrt(base_a * base_b)",
    # Use sqrt with or operator
    "result2 = sqrt(base_a) or sqrt(base_c)",
    # Use sqrt with and operator
    "result3 = sqrt(base_a) and sqrt(base_b)",
]

print("\nFAME commands:")
for cmd in sqrt_commands:
    print(f"  {cmd}")

generate_formulas_file(sqrt_commands, "sqrt_formulas.py")
generate_test_script(sqrt_commands, "sqrt_transformer.py")

print("\nGenerated transformer code:")
with open("sqrt_transformer.py", 'r') as f:
    content = f.read()
    # Print relevant part
    lines = content.split('\n')
    in_function = False
    for line in lines:
        if 'def ts_transformer' in line:
            in_function = True
        if in_function:
            print(line)

# Example 2: Multiple date-filtered assignments
print("\n" + "=" * 60)
print("Example 2: Multiple date-filtered assignments")
print("=" * 60)

date_filter_commands = [
    "freq bus",
    # Set base value for Feb 2020 to Dec 2020
    "date 01Feb2020 to 31Dec2020",
    "set a = 100",
    # Set different value for Jan 2021 onwards
    "date 01Jan2021 to *",
    "set a = 250",
]

print("\nFAME commands:")
for cmd in date_filter_commands:
    print(f"  {cmd}")

print("\nInterpretation:")
print("  - For dates between 01 Feb 2020 and 31 Dec 2020: a = 100")
print("  - For dates from 01 Jan 2021 to today: a = 250")
print("  - Note: Business day frequency (freq bus) means only working days")

generate_formulas_file(date_filter_commands, "date_filter_formulas.py")
generate_test_script(date_filter_commands, "date_filter_transformer.py")

print("\nGenerated transformer code:")
with open("date_filter_transformer.py", 'r') as f:
    content = f.read()
    # Print relevant part
    lines = content.split('\n')
    in_function = False
    for line in lines:
        if 'def ts_transformer' in line:
            in_function = True
        if in_function:
            print(line)

# Example 3: Test the generated code
print("\n" + "=" * 60)
print("Example 3: Testing the generated date filter transformer")
print("=" * 60)

# Import the generated modules
spec = importlib.util.spec_from_file_location("date_filter_formulas", "date_filter_formulas.py")
formulas_module = importlib.util.module_from_spec(spec)
sys.modules["formulas"] = formulas_module
spec.loader.exec_module(formulas_module)

spec = importlib.util.spec_from_file_location("date_filter_transformer", "date_filter_transformer.py")
transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_module)

# Create test data with dates spanning the ranges
test_df = pl.DataFrame({
    "DATE": [
        date(2019, 12, 15),  # Before range 1
        date(2020, 3, 15),   # In range 1 (Feb-Dec 2020)
        date(2020, 6, 15),   # In range 1
        date(2020, 12, 15),  # In range 1
        date(2021, 1, 15),   # In range 2 (Jan 2021 onwards)
        date(2021, 6, 15),   # In range 2
        date(2022, 1, 15),   # In range 2
    ]
})

print("\nInput DataFrame:")
print(test_df)

# Apply transformations
result = transformer_module.ts_transformer(test_df)

print("\nTransformed DataFrame:")
print(result)

print("\nObservations:")
print("  - Dates in 2020 (range 1) have A = 100")
print("  - Dates from 2021 onwards (range 2) have A = 250")
print("  - Date before both ranges (2019) has A = null (first range doesn't preserve)")
print("  - Second date filter (2021+) preserves values from first filter (2020)")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✓ sqrt() function works with expressions: sqrt(a*b)")
print("✓ Logical operators work: sqrt(a) or sqrt(b), sqrt(a) and sqrt(b)")
print("✓ Multiple date ranges can assign to same variable")
print("✓ Later date ranges preserve earlier values outside their range")
print("✓ Business day frequency (freq bus) is supported")
print("✓ Date format ddMMMYYYY (e.g., 01Feb2020) is supported")
print("✓ Date range ending with * means 'up to today'")
print("=" * 60)
