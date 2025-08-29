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
# --- Level 1: a, a$, b$, pa, pa$, pb$ ---
pdf = pdf.with_columns([
    A(pl.col("V143")).alias('A')
    A_(pl.col("V123")).alias('A_')
    B_().alias('B_')
    PA(pl.col("V143")).alias('PA')
    PA_(pl.col("V123")).alias('PA_')
    PB_().alias('PB_')
])

# --- Level 2: aa, paa, xyz, xyz$ ---
pdf = pdf.with_columns([
    AA(pl.col("A"), pl.col("A_")).alias('AA')
    PAA(pl.col("PA"), pl.col("PA_")).alias('PAA')
    (FISHVOL(series_pairs=[(pl.col('A'), pl.col('PA'))], date_col=pl.col('DATE'), rebase_year=2017)*12).alias('XYZ')
    (FISHVOL(series_pairs=[(pl.col('B_'), pl.col('PB_'))], date_col=pl.col('DATE'), rebase_year=2017)*12).alias('XYZ_')
])

# --- Level 3: bb, pbb ---
pdf = pdf.with_columns([
    BB(pl.col("A"), pl.col("AA")).alias('BB')
    PBB(pl.col("PA"), pl.col("PAA")).alias('PBB')
])

# --- Level 4: abc, abc1, acc, d ---
pdf = pdf.with_columns([
    CHAIN(price_quantity_pairs=[(pl.col('PA_'), pl.col('A_')), (pl.col('PA'), pl.col('A')), (pl.col('PBB'), pl.col('BB'))], date_col=pl.col('DATE'), year='2025').alias('ABC')
    CHAIN(price_quantity_pairs=[(pl.col('PA_'), pl.col('A_')), (-pl.col('PA'), pl.col('A')), (pl.col('PBB'), pl.col('BB'))], date_col=pl.col('DATE'), year='2025').alias('ABC1')
    CHAIN(price_quantity_pairs=[(pl.col('PA_'), pl.col('A_')), (-pl.col('PBB'), pl.col('BB'))], date_col=pl.col('DATE'), year='2022').alias('ACC')
    D(pl.col("BB")).alias('D')
])

# --- Frequency Conversions ---
# Conversions targeting q
q_convert_df = CONVERT(pdf.select(['DATE','A_']), as_freq='1y', to_freq='1q', technique='discrete', observed='average').rename({'A_':'ZED'})
pdf = pdf.join_asof(q_convert_df, on='DATE')

# Final formatting
final_result = (pdf.select(['DATE'] + ['A_', 'A', 'PA_', 'PA', 'AA', 'PAA', 'BB', 'PBB', 'D', 'ABC', 'ZED', 'XYZ', 'B_', 'PB_', 'XYZ_', 'ABC1', 'ACC']).rename({'A_': 'A$', 'A': 'A', 'PA_': 'PA$', 'PA': 'PA', 'AA': 'AA', 'PAA': 'PAA', 'BB': 'BB', 'PBB': 'PBB', 'D': 'D', 'ABC': 'ABC', 'ZED': 'ZED', 'XYZ': 'XYZ', 'B_': 'B$', 'PB_': 'PB$', 'XYZ_': 'XYZ$', 'ABC1': 'ABC1', 'ACC': 'ACC'}).melt(id_vars='DATE', variable_name='TIME_SERIES_NAME', value_name='VALUE').sort(['TIME_SERIES_NAME','DATE']).with_columns(SOURCE_SCRIPT_NAME=pl.lit('FAME_INPUT')))
print(final_result)