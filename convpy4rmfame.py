import polars as pl
import formulas
df = pl.DataFrame({
    'date': pl.date_range('2019-01-01', '2025-01-01', '1mo'),
    'v_a': pl.Series('v_a', range(1, 74)),
    'v_b': pl.Series('v_b', range(1, 74)),
    'p_a': pl.Series('p_a', range(1, 74)),
    'p_b': pl.Series('p_b', range(1, 74)),
    'gdp_m': pl.Series('gdp_m', range(1, 74)),
    'cpi_q': pl.Series('cpi_q', range(1, 74)),
    'gdp_q': pl.Series('gdp_q', range(1, 74)),
})
vols_g1 = ['v_a', 'v_b']
prices_g1 = ['p_a', 'p_b']
all_vols = ['v_a', 'v_b']
list_of_vol_aliases = [vols_g1]
# ---- DECLARE SERIES ----
gdp_q = formulas.DECLARE_SERIES(df, 'gdp_q')
cpi_q = formulas.DECLARE_SERIES(df, 'cpi_q')
vol_index_1 = formulas.DECLARE_SERIES(df, 'vol_index_1')
# ---- COMPUTATIONS ----
gdp_q = formulas.CONVERT(df, 'v_a', 'q', 'ave', 'end')
gdp_q = formulas.CONVERT(df, 'v_b', 'q', 'ave', 'end')
gdp_real = formulas.FISHVOL(df, vols_g1, prices_g1, year=2020)
vol_index_1 = formulas.SUM_HORIZONTAL(df, ['v_a', 'v_b'])
gdp_chained = formulas.CHAIN(df, ['gdp_q', 'cpi_q', 'pgdp_q', 'pcpi_q'], base_year=2022)
final_output = formulas.SUM_HORIZONTAL(df, ['gdp_chained', 'vol_index_1'])
print('Computation finished')