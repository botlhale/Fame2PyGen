# FAME Function Conversion Examples

<div align="center">
  <img src="logos/fame2pygen.png" alt="Fame2PyGen Logo" width="400"/>
</div>

This document shows examples of how various FAME functions are converted to Python using the Fame2PyGen system, including the **ENHANCED existing functions** with new chainsum operations and dependency support.

## Core FAME Functions Supported

### ENHANCED: Chain Sum Operations using Enhanced CHAIN Function

**FAME:**
```fame
# Chain sum with variable lists and dependencies using enhanced existing function
set aggregated_output = $chainsum("comp1 + comp2 + comp3", 2020, ["comp1", "comp2", "comp3"])
```
**Python Conversion:**
```python
from formulas import CHAIN

# Enhanced CHAIN function now supports both traditional chain and new chainsum operations
pdf = pdf.with_columns([
    CHAIN(expression_parts=[pl.col("comp1"), pl.col("comp2"), pl.col("comp3")], 
          date_col=pl.col("date"), index_year=2020, var_list=["comp1", "comp2", "comp3"],
          operation="chainsum").alias("aggregated_output")
])
```

### ENHANCED: FISHVOL Function with Dependencies

**FAME:**
```fame
# Enhanced fishvol with explicit dependencies using enhanced existing function
set volume_index = fishvol_rebase(volumes, prices, 2020, deps=["comp1", "aggregated_output"])
```
**Python Conversion:**
```python  
from formulas import FISHVOL

# Enhanced FISHVOL function now supports dependency parameters
pdf = pdf.with_columns([
    FISHVOL(vol_list=["volumes"], price_list=["prices"], date_col=pl.col("date"), 
            rebase_year=2020, dependencies=["comp1", "aggregated_output"]).alias("volume_index")
])
```

### ENHANCED: CONVERT Function with Dependencies

**FAME:**
```fame  
# Enhanced convert with dependency management using enhanced existing function
set quarterly_data = convert(monthly_series, q, average, end, deps=["volume_index"])
```
**Python Conversion:**
```python
from formulas import CONVERT

# Enhanced CONVERT function now supports dependency parameters
pdf = pdf.with_columns([
    CONVERT(source_var="monthly_series", freq="q", method="average", period="end", 
            dependencies=["volume_index"]).alias("quarterly_data")
])
```

### 1. Mathematical Operations
```fame
set result = series1 + series2
set result = series1 * 12
set result = series1 / series2
```
**Python Conversion:**
```python
# Direct operations using Polars expressions
pdf = pdf.with_columns([
    (pl.col("series1") + pl.col("series2")).alias("result"),
    (pl.col("series1") * 12).alias("result"),
    (pl.col("series1") / pl.col("series2")).alias("result")
])
```

### 2. COPY Function
**FAME:**
```fame
set backup_series = copy(original_series)
```
**Python Conversion:**
```python
from polars_econ_mock import copy

pdf = pdf.with_columns([
    copy(pl.col("original_series")).alias("backup_series")
])
```

### 3. PCT (Percentage Change)
**FAME:**
```fame
set growth_rate = pct(gdp_series, 4)  # 4-period lag
set monthly_change = pct(price_index)  # 1-period lag (default)
```
**Python Conversion:**
```python
from polars_econ_mock import pct

pdf = pdf.with_columns([
    pct(pl.col("gdp_series"), lag=4).alias("growth_rate"),
    pct(pl.col("price_index")).alias("monthly_change")
])
```

### 4. INTERP (Interpolation)
**FAME:**
```fame
set filled_series = interp(sparse_series, linear)
set cubic_interp = interp(data_series, cubic)
```
**Python Conversion:**
```python
from polars_econ_mock import interp

pdf = pdf.with_columns([
    interp(pl.col("sparse_series"), method="linear").alias("filled_series"),
    interp(pl.col("data_series"), method="cubic").alias("cubic_interp")
])
```

### 5. OVERLAY
**FAME:**
```fame
set combined_series = overlay(primary_series, backup_series)
```
**Python Conversion:**
```python
from polars_econ_mock import overlay

pdf = pdf.with_columns([
    overlay(pl.col("primary_series"), pl.col("backup_series")).alias("combined_series")
])
```

### 6. MAVE (Moving Average)
**FAME:**
```fame
set smooth_3 = mave(volatile_series, 3)
set trend_12 = mave(monthly_data, 12)
```
**Python Conversion:**
```python
from polars_econ_mock import mave

pdf = pdf.with_columns([
    mave(pl.col("volatile_series"), window=3).alias("smooth_3"),
    mave(pl.col("monthly_data"), window=12).alias("trend_12")
])
```

### 7. MAVEC (Centered Moving Average)
**FAME:**
```fame
set centered_avg = mavec(seasonal_series, 12)
set quarter_centered = mavec(quarterly_data, 4)
```
**Python Conversion:**
```python
from polars_econ_mock import mavec

pdf = pdf.with_columns([
    mavec(pl.col("seasonal_series"), window=12, center=True).alias("centered_avg"),
    mavec(pl.col("quarterly_data"), window=4, center=True).alias("quarter_centered")
])
```

### 8. CONVERT (Frequency Conversion)
**FAME:**
```fame
convert(monthly_series, quarterly, average, end)
convert(daily_data, monthly, sum, middle)
```
**Python Conversion:**
```python
from formulas import CONVERT

# Convert monthly to quarterly
quarterly_df = CONVERT(
    pdf.select(["DATE", "monthly_series"]),
    as_freq="1mo",
    to_freq="1q", 
    technique="average",
    observed="end"
)

# Convert daily to monthly  
monthly_df = CONVERT(
    pdf.select(["DATE", "daily_data"]),
    as_freq="1d",
    to_freq="1mo",
    technique="sum", 
    observed="middle"
)
```

### 9. FISHVOL (Fisher Volume Index)
**FAME:**
```fame
set volume_index = $fishvol_rebase({quantity1, quantity2}, {price1, price2}, 2020)
```
**Python Conversion:**
```python
from formulas import FISHVOL

pdf = pdf.with_columns([
    FISHVOL(series_pairs=[
        (pl.col("quantity1"), pl.col("price1")),
        (pl.col("quantity2"), pl.col("price2"))
    ], date_col=pl.col("DATE"), rebase_year=2020).alias("volume_index")
])
```

### 10. CHAIN (Chain-Linked Index)
**FAME:**
```fame
set chained_index = $mchain({price1, quantity1; price2, quantity2}, "2022")
```
**Python Conversion:**
```python
from formulas import CHAIN

pdf = pdf.with_columns([
    CHAIN(price_quantity_pairs=[
        (pl.col("price1"), pl.col("quantity1")),
        (pl.col("price2"), pl.col("quantity2"))
    ], date_col=pl.col("DATE"), year="2022").alias("chained_index")
])
```

## Complex Example: Complete FAME Script Conversion

**Original FAME Script:**
```fame
series gdp_nominal, gdp_real, price_index
freq quarterly

set gdp_growth = pct(gdp_real, 4)
set price_change = pct(price_index, 1)
set smoothed_gdp = mave(gdp_real, 4)
set interpolated_data = interp(gdp_nominal)
set combined_index = overlay(price_index, backup_prices)

set volume_measure = $fishvol_rebase({gdp_real}, {price_index}, 2020)
convert(monthly_source, quarterly, average, end)
```

**Generated Python Code:**
```python
import polars as pl
from formulas import *
import polars_econ_mock as ple

# Initial data setup
pdf = pl.DataFrame({
    "DATE": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 31), "1q"),
    "gdp_nominal": [100 + i*2.5 for i in range(24)],
    "gdp_real": [95 + i*1.8 for i in range(24)], 
    "price_index": [100 + i*0.5 for i in range(24)],
    "backup_prices": [99 + i*0.6 for i in range(24)]
})

# Computation pipeline
pdf = pdf.with_columns([
    # GDP growth (4-period lag)
    ple.pct(pl.col("gdp_real"), lag=4).alias("gdp_growth"),
    
    # Price change (1-period lag)
    ple.pct(pl.col("price_index"), lag=1).alias("price_change"),
    
    # 4-period moving average of GDP
    ple.mave(pl.col("gdp_real"), window=4).alias("smoothed_gdp"),
    
    # Linear interpolation of nominal GDP
    ple.interp(pl.col("gdp_nominal")).alias("interpolated_data"),
    
    # Overlay price index with backup
    ple.overlay(pl.col("price_index"), pl.col("backup_prices")).alias("combined_index"),
    
    # Fisher volume index
    FISHVOL(series_pairs=[
        (pl.col("gdp_real"), pl.col("price_index"))
    ], date_col=pl.col("DATE"), rebase_year=2020).alias("volume_measure")
])

# Frequency conversion would be handled separately
monthly_df = CONVERT(
    source_df.select(["DATE", "monthly_source"]),
    as_freq="1mo", 
    to_freq="1q",
    technique="average", 
    observed="end"
)

print("FAME conversion completed successfully!")
```

## Benefits of This Approach

1. **Type Safety**: Full Polars expression system with type checking
2. **Performance**: Lazy evaluation and optimized computation graphs
3. **Flexibility**: Easy to extend with custom functions
4. **Maintainability**: Clear mapping from FAME concepts to Python code
5. **Integration**: Works seamlessly with modern Python data stack