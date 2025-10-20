"""
Business Day Frequency Example

This example demonstrates how to use business day frequency ('freq b' or 'freq bus')
with Fame2PyGen for time series analysis on working days only.

Business day frequency is commonly used in financial and economic data where weekends
and holidays should be excluded from calculations.
"""

from fame2pygen import generate_formulas_file, generate_test_script
import polars as pl
from datetime import date

# Example FAME commands using business day frequency
fame_commands = [
    # Set frequency to business days
    "freq b",
    
    # Simple assignment - works the same on any frequency
    "vbot = 1",
    
    # Convert monthly data to business day frequency
    # This might be used when you have monthly economic indicators
    # that you want to interpolate to daily business days
    "daily_gdp = convert(monthly_gdp, 'm', 'b', 'linear', 'end')",
    
    # Arithmetic operations work the same
    "growth = daily_gdp - daily_gdp[t-1]",
    
    # Percentage calculations
    "pct_change = pct(daily_gdp)",
]

# Alternative: using 'freq bus' (equivalent to 'freq b')
fame_commands_alt = [
    "freq bus",  # Alternative business day syntax
    "vbot = 1",
]

def main():
    """Generate Python files from FAME commands with business day frequency."""
    
    print("Generating formulas.py and ts_transformer.py...")
    print("=" * 60)
    
    # Generate the formula definitions
    generate_formulas_file(fame_commands, "formulas.py")
    print("✓ Generated formulas.py")
    
    # Generate the transformation pipeline
    generate_test_script(fame_commands, "ts_transformer.py")
    print("✓ Generated ts_transformer.py")
    
    print("\nExample usage:")
    print("-" * 60)
    
    # Show example of using the generated code
    example_code = """
import polars as pl
from ts_transformer import ts_transformer

# Create sample data with business days only
# In practice, you'd load this from your data source
df = pl.DataFrame({
    "DATE": pl.date_range(
        date(2023, 1, 1),
        date(2023, 12, 31),
        interval="1d",
        eager=True
    ),
    "MONTHLY_GDP": [100.0] * 365,  # Simplified example
})

# Filter to business days only (Monday-Friday)
# polars-econ would handle this automatically based on freq setting
df = df.filter(pl.col("DATE").dt.weekday() < 5)

# Apply transformations
result = ts_transformer(df)

print(result.head())
"""
    
    print(example_code)
    
    print("\n" + "=" * 60)
    print("Frequency Support in Fame2PyGen:")
    print("-" * 60)
    print("  freq a    - Annual")
    print("  freq q    - Quarterly")
    print("  freq m    - Monthly")
    print("  freq w    - Weekly")
    print("  freq d    - Daily")
    print("  freq b    - Business day (working days)")
    print("  freq bus  - Business day (alternative syntax)")
    print("=" * 60)
    
    print("\nNote: Business day frequency integrates with polars-econ")
    print("for proper handling of business day calendars in operations")
    print("like convert(), chain(), and other time series functions.")


if __name__ == "__main__":
    main()
