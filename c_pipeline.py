"""Auto-generated pipeline script."""
import polars as pl
from cformulas import *

# Example input DataFrame (replace with real data ingestion)
pdf = pl.DataFrame({
    "DATE": pl.date_range(pl.date(2018,1,1), pl.date(2023,12,31), "1mo", eager=True),
    "A_IN": [i*1.1 for i in range(72)],
    "B_IN": [i*1.2+5 for i in range(72)],
})

# Computation levels
# --- Level 1: a ---
pdf = pdf.with_columns([
    A(pl.col("B")).alias('A')
])

# Final formatting
final_result = (pdf.select(['DATE'] + ['A']).rename({'A': 'A'}).melt(id_vars='DATE', variable_name='TIME_SERIES_NAME', value_name='VALUE').sort(['TIME_SERIES_NAME','DATE']).with_columns(SOURCE_SCRIPT_NAME=pl.lit('FAME_INPUT')))
print(final_result)