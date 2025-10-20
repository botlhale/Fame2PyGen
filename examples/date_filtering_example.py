#!/usr/bin/env python3
"""
Example demonstrating date filtering support in Fame2PyGen.

This example shows how date commands are tracked and added as comments
in the generated transformation code.
"""

from fame2pygen.fame2py_converter import generate_test_script, generate_formulas_file

# Define FAME commands with date filtering
fame_commands = [
    "freq m",
    
    # Commands without date filter (apply to all dates)
    "v_base = 100",
    
    # Set date filter for 2020
    "date 2020-01-01 to 2020-12-31",
    "v_2020 = v_base * 1.1",  # Only affects 2020 dates
    "v_2020_adj = v_2020 + 10",  # Also only affects 2020 dates
    
    # Reset to all dates
    "date *",
    "v_all = v_2020_adj / 2",  # Affects all dates
    
    # Set date filter for 2021
    "date 2021-01-01 to 2021-12-31",
    "v_2021 = v_all * 1.2",  # Only affects 2021 dates
]

# Generate the transformation code
print("Generating formulas.py...")
generate_formulas_file(fame_commands, "date_formulas.py")

print("Generating ts_transformer.py...")
generate_test_script(fame_commands, "date_transformer.py")

print("\nGenerated files:")
print("  - date_formulas.py: Helper functions")
print("  - date_transformer.py: Transformation pipeline")

print("\n" + "="*60)
print("Generated transformation code:")
print("="*60)

with open("date_transformer.py", "r") as f:
    print(f.read())

print("\n" + "="*60)
print("Notice how date filter comments are added before each level!")
print("="*60)
