"""
Example demonstrating FAME-style conditional expressions in Fame2PyGen.

This example shows how to use conditional logic with comparison operators
and the 'nd' (null/missing) keyword.
"""

from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
import polars as pl
from datetime import date
import sys
import os

# FAME commands with conditional expressions
fame_commands = [
    "freq m",
    "base_value = 100",
    "threshold = 150",
    
    # Simple conditional: if base_value > 100, double it, else use nd (null)
    "result1 = if base_value gt 100 then base_value * 2 else nd",
    
    # Conditional with comparison operators
    "result2 = if threshold ge 150 then base_value * 1.5 else base_value",
    
    # Conditional with multiple variables
    "price = 50",
    "quantity = 10",
    "adjusted_total = if price lt 100 then price * quantity else price * quantity * 1.1",
    
    # Conditional with 'nd' in then clause
    "result3 = if base_value le 50 then nd else base_value * 3",
]

print("Generating Python code from FAME commands...")
print("\nFAME Commands:")
for cmd in fame_commands:
    print(f"  {cmd}")

# Generate the Python files
generate_formulas_file(fame_commands, "formulas.py")
generate_test_script(fame_commands, "ts_transformer.py")

print("\n✓ Generated formulas.py and ts_transformer.py")

# Import and use the generated transformer
sys.path.insert(0, os.getcwd())
from ts_transformer import ts_transformer

# Create sample data
sample_data = pl.DataFrame({
    "DATE": [
        date(2020, 1, 1),
        date(2020, 2, 1),
        date(2020, 3, 1),
    ]
})

print("\nInput DataFrame:")
print(sample_data)

# Apply transformations
result = ts_transformer(sample_data)

print("\nTransformed DataFrame:")
print(result)

print("\nExplanation of results:")
print("- BASE_VALUE = 100 (constant)")
print("- THRESHOLD = 150 (constant)")
print("- RESULT1: since base_value (100) is NOT > 100, result is nd (null)")
print("- RESULT2: since threshold (150) >= 150, result is base_value * 1.5 = 150")
print("- PRICE = 50, QUANTITY = 10")
print("- ADJUSTED_TOTAL: since price (50) < 100, result is price * quantity = 500")
print("- RESULT3: since base_value (100) is NOT <= 50, result is base_value * 3 = 300")

# Clean up generated files
os.remove("formulas.py")
os.remove("ts_transformer.py")
print("\n✓ Cleaned up generated files")
