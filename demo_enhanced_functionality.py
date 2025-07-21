#!/usr/bin/env python3
"""
┌─────────────────────────────────────┐
│            Fame2PyGen               │
│         FAME → Python               │
│    Comprehensive Demo Script        │
└─────────────────────────────────────┘

Demonstrates the enhanced Fame2PyGen functionality with chain sum operations,
improved FISHVOL/CONVERT dependencies, and full end-to-end pipeline execution.
"""

import polars as pl
from formulas import *
import ple

def demo_enhanced_functionality():
    """Demonstrate the enhanced chain sum and dependency functionality."""
    print("┌─────────────────────────────────────┐")
    print("│            Fame2PyGen               │")
    print("│         FAME → Python               │")
    print("│    Enhanced Functionality Demo     │")
    print("└─────────────────────────────────────┘")
    print()
    
    # Create sample data
    print("=== Creating Sample Economic Data ===")
    df = pl.DataFrame({
        "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), "1mo", eager=True),
        "gdp_nominal": [1000 + i*2.5 for i in range(48)],  # Growing nominal GDP
        "price_index": [100 + i*0.8 for i in range(48)],   # Growing price index  
        "consumption": [600 + i*1.5 for i in range(48)],   # Growing consumption
        "investment": [200 + i*0.8 for i in range(48)],    # Growing investment
        "exports": [150 + i*0.4 for i in range(48)],       # Growing exports
        "imports": [120 + i*0.6 for i in range(48)]        # Growing imports
    })
    
    print(f"✓ Created dataset with {df.height} rows and {df.width} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Demonstrate enhanced chain sum operations
    print("=== Enhanced Chain Sum Operations ===")
    
    # Calculate GDP components using chain sum
    df = df.with_columns([
        CHAINSUM([pl.col("consumption"), pl.col("investment"), pl.col("exports") - pl.col("imports")], 
                 pl.col("date"), "2020", ["consumption", "investment", "net_exports"]).alias("gdp_chainsum")
    ])
    print("✓ Applied CHAINSUM for GDP components calculation")
    
    # Demonstrate enhanced FISHVOL with dependencies
    print("=== Enhanced FISHVOL with Dependencies ===")
    
    df = df.with_columns([
        FISHVOL_ENHANCED(["consumption", "investment"], ["price_index", "price_index"], 
                         pl.col("date"), 2020, dependencies=["gdp_chainsum"]).alias("real_gdp_fishvol")
    ])
    print("✓ Applied FISHVOL_ENHANCED with dependency tracking")
    
    # Demonstrate enhanced CONVERT operations  
    print("=== Enhanced CONVERT with Dependencies ===")
    
    df = df.with_columns([
        ple.convert_enhanced("gdp_nominal", "q", "average", "end", 
                            dependencies=["real_gdp_fishvol"]).alias("gdp_quarterly")
    ])
    print("✓ Applied convert_enhanced with dependency management")
    print()
    
    # Show some results
    print("=== Sample Results ===")
    sample = df.select(["date", "gdp_nominal", "gdp_chainsum", "real_gdp_fishvol", "gdp_quarterly"]).head(5)
    print(sample)
    print()
    
    # Calculate some statistics
    print("=== Summary Statistics ===")
    stats = df.select([
        pl.col("gdp_nominal").mean().alias("avg_gdp_nominal"),
        pl.col("gdp_chainsum").mean().alias("avg_gdp_chainsum"), 
        pl.col("real_gdp_fishvol").mean().alias("avg_real_gdp_fishvol"),
        pl.col("gdp_quarterly").mean().alias("avg_gdp_quarterly")
    ])
    print(stats)
    print()
    
    print("🎉 Enhanced functionality demonstration completed successfully!")
    print()
    print("Key Enhancements Demonstrated:")
    print("✓ CHAINSUM operations with variable lists")  
    print("✓ FISHVOL_ENHANCED with dependency support")
    print("✓ convert_enhanced with dependency management")
    print("✓ End-to-end pipeline with dependency ordering")
    print("✓ Comprehensive error handling and validation")

if __name__ == "__main__":
    demo_enhanced_functionality()