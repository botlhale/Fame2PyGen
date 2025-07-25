"""
┌─────────────────────────────────────┐
│            Fame2PyGen               │
│         FAME → Python               │
│    Auto-Generated Pipeline         │
└─────────────────────────────────────┘

This file was automatically generated by Fame2PyGen
Contains the conversion pipeline from FAME script to Python
"""
import polars as pl
from formulas import *
df = pl.DataFrame({
    'date': pl.date_range(pl.date(2019, 1, 1), pl.date(2025, 1, 1), '1mo', eager=True),
    'monthly_series': pl.Series('monthly_series', range(1, 74)),
    'prices': pl.Series('prices', range(1, 74)),
    'v123': pl.Series('v123', range(1, 74)),
    'v143': pl.Series('v143', range(1, 74)),
    'volumes': pl.Series('volumes', range(1, 74)),
})
# ---- DECLARE SERIES ----
# ---- COMPUTATIONS ----
# Mathematical expression: a$ = v123*12
df = df.with_columns([A_().alias('a$')])
# Mathematical expression: a = v143*12
df = df.with_columns([A().alias('a')])
# Mathematical expression: b = v143*2
df = df.with_columns([B().alias('b')])
# Mathematical expression: b$ = v123*6
df = df.with_columns([B_().alias('b$')])
# Mathematical expression: c$ = v123*5
df = df.with_columns([C_().alias('c$')])
# Mathematical expression: d = v123*1
df = df.with_columns([D().alias('d')])
# Mathematical expression: e = v123*2
df = df.with_columns([E().alias('e')])
# Mathematical expression: f = v123*3
df = df.with_columns([F().alias('f')])
# Mathematical expression: g = v123*4
df = df.with_columns([G().alias('g')])
# Mathematical expression: h = v123*5
df = df.with_columns([H().alias('h')])
# Mathematical expression: pa$ = v123*3
df = df.with_columns([PA_().alias('pa$')])
# Mathematical expression: pa = v143*4
df = df.with_columns([PA().alias('pa')])
# Mathematical expression: pb = v143*1
df = df.with_columns([PB().alias('pb')])
# Mathematical expression: pb$ = v123*1
df = df.with_columns([PB_().alias('pb$')])
# Mathematical expression: pc$ = v123*2
df = df.with_columns([PC_().alias('pc$')])
# Mathematical expression: pd = v123*3
df = df.with_columns([PD().alias('pd')])
# Mathematical expression: pe = v123*4
df = df.with_columns([PE().alias('pe')])
# Mathematical expression: pf = v123*5
df = df.with_columns([PF().alias('pf')])
# Mathematical expression: pg = v123*1
df = df.with_columns([PG().alias('pg')])
# Mathematical expression: ph = v123*2
df = df.with_columns([PH().alias('ph')])
# Enhanced fishvol function: vol_index = fishvol_enhanced(volumes, prices, year=2020, deps=['a', 'b'])
# Dependencies: ['a', 'b'] must be computed first
df = df.with_columns([FISHVOL_ENHANCED(['volumes'], ['prices'], pl.col('date'), 2020, ['a', 'b']).alias('vol_index')])
# Enhanced convert function: quarterly_data = convert_enhanced(monthly_series, q, average, end, deps=['c$'])
# Dependencies: ['c$'] must be computed first
df = df.with_columns([ple.convert_enhanced('monthly_series', 'q', 'average', 'end', ['c$']).alias('quarterly_data')])
# Mathematical expression: aa = a$/a
df = df.with_columns([(pl.col("a$")/pl.col("a")).alias('aa')])
# Mathematical expression: paa = pa$/pa
df = df.with_columns([(pl.col("pa$")/pl.col("pa")).alias('paa')])
# Mathematical expression: hxz = (b*12)/a
df = df.with_columns([((pl.col("b")*12)/pl.col("a")).alias('hxz')])
# Mathematical expression: abc$_d1 = a$+b$+c$+a
df = df.with_columns([(pl.col("a$")+pl.col("b$")+pl.col("c$")+pl.col("a")).alias('abc$_d1')])
# mchain function: c1 = chain(['a', 'b', 'c$', 'd', 'e', 'f', 'g', 'h'], base_year=2017)
df = df.with_columns([CHAIN([(pl.col('a'), pl.col('b')), (pl.col('c$'), pl.col('d')), (pl.col('e'), pl.col('f')), (pl.col('g'), pl.col('h'))], pl.col("date"), "2017").alias('c1')])
# Enhanced chainsum function: chain_total = chainsum(['b', 'c$', 'a'], base_year=2017, vars=['a', 'b', 'c$'])
df = df.with_columns([CHAINSUM([pl.col('b'), pl.col('c$'), pl.col('a')], pl.col('date'), '2017', ['a', 'b', 'c$']).alias('chain_total')])
# Mathematical expression: bb = aa+a
df = df.with_columns([(pl.col("aa")+pl.col("a")).alias('bb')])
# Mathematical expression: pbb = pa+paa
df = df.with_columns([(pl.col("pa")+pl.col("paa")).alias('pbb')])
print('Computation finished')