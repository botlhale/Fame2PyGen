"""Tests for time shift and pct function functionality."""

from fame2pygen.parser import parse_script
from fame2pygen.generators import build_generation_context, generate_formulas_module

def test_time_shift_parsing():
    """Test parsing of time shift references like vbots[t+1]"""
    script = ["vbots = vbot", "set vbots[t] = vbots[t+1]"]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 2
    
    # First assignment should parse normally
    assert parsed[0].target == "vbots"
    assert "vbot" in parsed[0].refs
    
    # Second assignment should detect time shift
    assert parsed[1].target == "vbots[t]"
    assert "vbots" in parsed[1].refs
    assert parsed[1].rhs == "vbots[t+1]"

def test_pct_function_parsing():
    """Test parsing of pct function calls"""
    script = ["result = pct(v23s[t+1])"]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 1
    assert parsed[0].target == "result"
    assert "v23s" in parsed[0].refs
    assert "pct(v23s[t+1])" in parsed[0].rhs

def test_complex_expression_parsing():
    """Test parsing of complex expression with time shifts and pct function"""
    script = [
        "vbots = vbot",
        "v23s = v23", 
        "set vbots[t] = vbots[t+1]/(1+(pct(v23s[t+1])/100))"
    ]
    parsed = parse_script(script)
    
    # Check parsing
    assert len(parsed) == 3
    
    # Simple assignments
    assert parsed[0].target == "vbots"
    assert "vbot" in parsed[0].refs
    assert parsed[1].target == "v23s"
    assert "v23" in parsed[1].refs
    
    # Complex assignment
    assert parsed[2].target == "vbots[t]"
    assert "vbots" in parsed[2].refs
    assert "v23s" in parsed[2].refs

def test_time_shift_generation():
    """Test generation of time shift expressions"""
    script = ["result = var[t+1]"]
    parsed = parse_script(script)
    
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    
    # Should generate shift operation
    assert ".shift(-1)" in formulas

def test_pct_function_generation():
    """Test generation of pct function calls"""
    script = ["result = pct(variable)"]
    parsed = parse_script(script)
    
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    
    # Should include PCT function definition
    assert "def PCT(" in formulas
    assert "polars_econ as ple" in formulas
    assert "ple.pct(" in formulas

def test_pct_with_time_shift_generation():
    """Test generation of pct function with time shift"""
    script = ["result = pct(var[t+1])"]
    parsed = parse_script(script)
    
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    
    # Should generate both shift and PCT call
    assert "def PCT(" in formulas
    assert "shift(-1)" in formulas or ".shift(-1)" in formulas

def test_complex_expression_generation():
    """Test generation of the full complex expression"""
    script = [
        "vbots = vbot",
        "v23s = v23",
        "set vbots[t] = vbots[t+1]/(1+(pct(v23s[t+1])/100))"
    ]
    parsed = parse_script(script)
    
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    
    # Should include PCT function
    assert ctx.has_pct == True
    assert "def PCT(" in formulas

if __name__ == "__main__":
    test_time_shift_parsing()
    test_pct_function_parsing()
    test_complex_expression_parsing()
    test_time_shift_generation()
    test_pct_function_generation()
    test_pct_with_time_shift_generation()
    test_complex_expression_generation()
    print("All time shift and pct function tests passed!")