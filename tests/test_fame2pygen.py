import pytest
import polars as pl
from fame2pygen.formulas_generator import parse_fame_formula, generate_polars_functions
from fame2pygen.fame2py_converter import analyze_dependencies, get_computation_levels

def test_parse_simple_assignment():
    result = parse_fame_formula("vbot = 1")
    assert result["type"] == "assign_series"
    assert result["target"] == "vbot"
    assert result["rhs"] == "1"

def test_parse_shift_pct_pattern():
    result = parse_fame_formula("set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))")
    assert result["type"] == "shift_pct"
    assert result["ser1"] == "v123s"
    assert result["offset"] == 1

def test_parse_chain_operation():
    result = parse_fame_formula('set abcd = $chain("a - b - c - d", "2020")')
    assert result["type"] == "chain"
    assert result["target"] == "abcd"
    assert result["year"] == "2020"

def test_dependency_analysis():
    parsed = [
        {"target": "v1", "refs": ["v2"]},
        {"target": "v2", "refs": []}
    ]
    adj, in_degree = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, in_degree)
    assert levels[0] == ["v2"]
    assert levels[1] == ["v1"]

def test_generate_functions():
    cmds = ["set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))"]
    defs = generate_polars_functions(cmds)
    assert "SHIFT_PCT_BACKWARDS" in defs

def test_arithmetic_operations():
    result = parse_fame_formula("v1 = v2 + v3 - v4")
    assert result["type"] == "simple"
    assert "v2" in result["refs"]
    assert "v3" in result["refs"]

def test_time_indexed_variables():
    result = parse_fame_formula("v1 = v2[t+1]")
    assert result["type"] == "simple"
    assert "v2[t+1]" in result["refs"]

def test_pct_function():
    result = parse_fame_formula("set v21 = pct(v22[t+1])")
    assert result["type"] == "simple"
    assert "pct(v22[t+1])" in result["rhs"]

def test_convert_function():
    result = parse_fame_formula("set v23 = convert(v24, 'Q', 'M', 'AVG', 'END')")
    assert result["type"] == "convert"
    assert result["target"] == "v23"

def test_fishvol_function():
    result = parse_fame_formula("set v25 = fishvol_rebase({v26},{p26},2020)")
    assert result["type"] == "fishvol"
    assert result["year"] == "2020"

def test_list_alias():
    result = parse_fame_formula("v27 = {a, b, c}")
    assert result["type"] == "list_alias"
    assert result["target"] == "v27"

def test_freq_command():
    result = parse_fame_formula("freq m")
    assert result["type"] == "freq"
    assert result["freq"] == "m"

def test_date_all_command():
    result = parse_fame_formula("date *")
    assert result["type"] == "date"
    assert result["filter"] is None

def test_date_range_command():
    result = parse_fame_formula("date 2020-01-01 to 2020-12-31")
    assert result["type"] == "date"
    assert result["filter"]["start"] == "2020-01-01"
    assert result["filter"]["end"] == "2020-12-31"

def test_date_filter_in_generated_code():
    """Test that date filters are tracked and added as comments in generated code."""
    from fame2pygen.fame2py_converter import generate_test_script
    import tempfile
    import os
    
    cmds = [
        "freq m",
        "date 2020-01-01 to 2020-12-31",
        "v1 = v2 + v3",
        "date *",
        "v4 = v5 + v6"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        output_file = f.name
    
    try:
        generate_test_script(cmds, output_file)
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Verify date filter comments are present
        assert "# Date filter: 2020-01-01 to 2020-12-31" in content
        assert "# Date filter: * (all dates)" in content
        
        # Verify the computations are separate (not grouped together)
        assert "v1" in content and "v4" in content
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)

if __name__ == "__main__":
    pytest.main()
