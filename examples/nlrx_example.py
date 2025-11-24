"""
Example demonstrating NLRX, FIRSTVALUE, and LASTVALUE functions.

This example shows how to use the new FAME functions that have been added:
- firstvalue(): Get the first non-null value from a series
- lastvalue(): Get the last non-null value from a series  
- nlrx(): Call polars_econ.nlrx for non-linear relaxation calculations

Based on the FAME scripting pattern:
  start = firstvalue(a)
  end = lastvalue(a)
  lambda20 = 20
  set <date start to end> b1 = 1
  set <date start-7 to end> b1 = 0
  set <date start to end> b2 = 0
  ...
  a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d, begsa, endmona)
"""

from fame2pygen import generate_formulas_file, generate_test_script
import tempfile
import os

# Define FAME commands using nlrx
fame_commands = [
    "freq m",  # Monthly frequency
    "start = firstvalue(a)",  # Get first value of series 'a'
    "end = lastvalue(a)",     # Get last value of series 'a'
    "lambda20 = 20",          # Lambda parameter for nlrx
    
    # Set up weight variables with date filtering
    "date start to end",
    "b1 = 1",
    "b2 = 0",
    "b3 = 0",
    "b4 = 0",
    "c = 0",
    "d = 0",
    
    # Reset date filter to all dates
    "date *",
    
    # Call nlrx function with all parameters
    # Note: In this simplified example, we're using 8 params (minimum required)
    # The actual polars_econ.nlrx may accept additional parameters
    "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)"
]

# Create temporary files for output
with tempfile.NamedTemporaryFile(mode='w', suffix='_formulas.py', delete=False) as f:
    formulas_file = f.name

with tempfile.NamedTemporaryFile(mode='w', suffix='_ts_transformer.py', delete=False) as f:
    transformer_file = f.name

try:
    # Generate the files
    print("Generating formulas.py and ts_transformer.py...")
    generate_formulas_file(fame_commands, formulas_file)
    generate_test_script(fame_commands, transformer_file)
    
    print(f"\nGenerated files:")
    print(f"  - {formulas_file}")
    print(f"  - {transformer_file}")
    
    # Display the generated formulas
    print("\n" + "=" * 80)
    print("GENERATED FORMULAS (formulas.py):")
    print("=" * 80)
    with open(formulas_file, 'r') as f:
        formulas_content = f.read()
        print(formulas_content)
    
    # Display the generated transformer
    print("\n" + "=" * 80)
    print("GENERATED TRANSFORMER (ts_transformer.py):")
    print("=" * 80)
    with open(transformer_file, 'r') as f:
        transformer_content = f.read()
        print(transformer_content)
    
    print("\n" + "=" * 80)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 80)
    print("1. FIRSTVALUE function - extracts first non-null value from a series")
    print("2. LASTVALUE function - extracts last non-null value from a series")
    print("3. NLRX function - wrapper for polars_econ.nlrx with proper parameter mapping")
    print("4. Date filtering - variables b1, b2, b3, b4, c, d are filtered by date range")
    print("5. DataFrame handling - NLRX returns a DataFrame which is properly assigned")
    
    print("\n" + "=" * 80)
    print("USAGE:")
    print("=" * 80)
    print("To use this in your code:")
    print("  import polars as pl")
    print("  from formulas import *")
    print("  from ts_transformer import ts_transformer")
    print("")
    print("  # Create your DataFrame with required columns")
    print("  df = pl.DataFrame({")
    print('      "DATE": [...],')
    print('      "A": [...],')
    print("      # ... other columns")
    print("  })")
    print("")
    print("  # Apply transformations")
    print("  result = ts_transformer(df)")
    print("  print(result)")

finally:
    # Cleanup
    if os.path.exists(formulas_file):
        os.unlink(formulas_file)
    if os.path.exists(transformer_file):
        os.unlink(transformer_file)
