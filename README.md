<div align="center">
  <img src="logos/fame2pygen.png" alt="Fame2PyGen Logo" width="600"/>
</div>

# Fame2PyGen

Fame2PyGen converts FAME-style formula scripts into modular Python using [Polars](https://pola.rs/). It generates:
- formulas.py: per-variable functions returning Polars expressions (pl.Expr)
- convpy4rmfame.py: an execution pipeline using with_columns()
- ple.py: mock/enhanced backend functions (CHAIN/FISHVOL/CONVERT)

Key behaviors:
- CHAIN is a consolidated operation that takes a list of (price_expr, quantity_expr) tuples (no separate CHAINSUM).
- FISHVOL sub-pipelines: once a FISHVOL appears, all computations that depend on it are executed in their own DataFrame filtered to DATE >= Jan 1 of the FISHVOL base year. The main DataFrame remains unfiltered.
- CONVERT sub-pipelines: quarterly or annual conversions and any dependents are executed in separate DataFrames to avoid altering the main DataFrame. 
- Final output: each sub-pipeline is melted to long format and then vertically concatenated.

## Quickstart: CHAIN and Sub-Pipelines

```python
import polars as pl
import ple

# Base frame (unfiltered)
df = pl.DataFrame({
    'date': pl.date_range(pl.date(2022, 1, 1), pl.date(2022, 12, 1), '1mo', eager=True),
    'prices_a': range(1, 13),
    'quantities_a': range(10, 22),
    'prices_b': range(2, 14),
    'quantities_b': range(5, 17),
})

# Main pipeline computations stay on df (no date filtering)
df = df.with_columns([
    (pl.col('quantities_a') * 2).alias('A'),
    (pl.col('quantities_b') + 3).alias('B'),
    ple.chain([
        (pl.col('prices_a'), pl.col('quantities_a')),
        (pl.col('prices_b'), pl.col('quantities_b')),
    ], date_col=pl.col('date')).alias('CHAIN_MAIN'),
])

# FISHVOL sub-pipeline: filter by base year start (e.g., 2020-01-01)
fishvol_df = df.filter(pl.col('date') >= pl.date(2020, 1, 1)).with_columns([
    ple.fishvol(series_pairs=[
        (pl.col('quantities_a'), pl.col('prices_a')),
        (pl.col('quantities_b'), pl.col('prices_b')),
    ], date_col=pl.col('date'), rebase_year=2020).alias('FISHVOL_IDX'),
])

# CONVERT sub-pipeline: quarterly; keep separate
convert_q_df = df.select(['date', 'A']).group_by_dynamic('date', every='1q').agg(pl.col('A').mean().alias('A_Q'))

# Melt and union at the end
main_long = df.select(['date', 'A', 'B', 'CHAIN_MAIN']).melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE')
fishvol_long = fishvol_df.select(['date', 'FISHVOL_IDX']).melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE')
convert_long = convert_q_df.melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE')

final = pl.concat([main_long, fishvol_long, convert_long], how='vertical_relaxed')
print(final)
```

Notes:
- The generator handles these sub-pipelines automatically based on the script (FISHVOL or CONVERT nodes and their dependency closures).
- If you need true chaining/rebasing logic, you can extend ple.chain and ple.fishvol accordingly.
