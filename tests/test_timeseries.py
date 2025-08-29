"""Tests for time series numeric reference functionality."""

from fame2pygen.parser import parse_script
from fame2pygen.generators import build_generation_context, generate_formulas_module

def test_timeseries_multiplication():
    """Test time series multiplication patterns like b$=1234*12"""
    script = ["b$=1234*12", "pb$=1434*12"]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 2
    assert parsed[0].target == "b$"
    assert "1234" in parsed[0].refs
    assert parsed[1].target == "pb$" 
    assert "1434" in parsed[1].refs
    
    # Check generation
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    assert 'pl.col("1234").mul(12)' in formulas
    assert 'pl.col("1434").mul(12)' in formulas

def test_timeseries_addition():
    """Test time series addition patterns like aaa=1233+2334+4827"""
    script = ["aaa=1233+2334+4827"]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 1
    assert parsed[0].target == "aaa"
    assert "1233" in parsed[0].refs
    assert "2334" in parsed[0].refs
    assert "4827" in parsed[0].refs
    
    # Check generation
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    assert 'pl.sum_horizontal([pl.col("1233"), pl.col("2334"), pl.col("4827")])' in formulas

def test_variable_addition_unchanged():
    """Test that variable addition patterns still work like vaaa=v1233+v2334+v4827"""
    script = ["vaaa=v1233+v2334+v4827"]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 1
    assert parsed[0].target == "vaaa"
    assert "v1233" in parsed[0].refs
    assert "v2334" in parsed[0].refs
    assert "v4827" in parsed[0].refs
    
    # Check generation - should NOT use sum_horizontal for variables
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    assert 'v1233+v2334+v4827' in formulas
    assert 'sum_horizontal' not in formulas

def test_literals_unchanged():
    """Test that simple numeric literals still work as before"""
    script = ["simple=42", "calc=2*3"]
    parsed = parse_script(script)
    
    # Check that simple numbers don't get treated as time series
    assert len(parsed) == 2
    assert len(parsed[0].refs) == 0  # No refs for simple literal
    assert len(parsed[1].refs) == 0  # No refs for simple calculation
    
    # Check generation - should use pl.lit()
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    assert 'pl.lit(42)' in formulas
    assert 'pl.lit(6)' in formulas

if __name__ == "__main__":
    test_timeseries_multiplication()
    test_timeseries_addition() 
    test_variable_addition_unchanged()
    test_literals_unchanged()
    print("All time series tests passed!")