import pytest
import polars as pl
from fame2pygen.formulas_generator import parse_fame_formula
from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
import tempfile
import os


def test_conditional_keywords_not_in_refs():
    """Test that conditional keywords (ge, gt, le, lt, eq, ne, nd, if, then, else) are not in refs."""
    result = parse_fame_formula('abc = if t ge 5 then a+b else nd')
    assert result is not None
    assert result["type"] == "conditional"
    # Check that conditional keywords are not in refs
    assert 'ge' not in result["refs"]
    assert 'if' not in result["refs"]
    assert 'then' not in result["refs"]
    assert 'else' not in result["refs"]
    assert 'nd' not in result["refs"]
    # Check that actual variables are in refs
    assert 'a' in result["refs"]
    assert 'b' in result["refs"]


def test_conditional_standalone_t_as_column():
    """Test that standalone 't' in conditionals is treated as DATE column reference."""
    cmds = ['freq m', 'abc = if t ge 5 then a+b else nd']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    
    try:
        generate_test_script(cmds, ts_file)
        with open(ts_file, 'r') as f:
            content = f.read()
        # Check that t is converted to pl.col("DATE") (FAME time variable)
        assert 'pl.col("DATE")' in content
        # Should not be pl.col("T")
        assert 'pl.col("T")' not in content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)


def test_nested_conditional_elseif():
    """Test nested conditionals (else if) are correctly handled."""
    cmds = ['freq m', 'abc = if t gt 10 then a else if t ge 5 then b else c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    
    try:
        generate_test_script(cmds, ts_file)
        with open(ts_file, 'r') as f:
            content = f.read()
        # Check for nested when/then/otherwise
        assert content.count('pl.when') == 2
        assert content.count('.then') == 2
        assert content.count('.otherwise') == 2
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)


def test_dot_notation_to_underscore():
    """Test that dot notation in variable names is preserved (Polars supports dots)."""
    result = parse_fame_formula('result = d.a + b.c')
    assert result is not None
    assert 'd.a' in result["refs"]
    assert 'b.c' in result["refs"]
    
    # Test transformer generation
    cmds = ['freq m', 'result = d.a + b.c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    
    try:
        generate_test_script(cmds, ts_file)
        with open(ts_file, 'r') as f:
            content = f.read()
        # Check that dots are preserved in uppercase
        assert 'D.A' in content
        assert 'B.C' in content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)


def test_add_series_function():
    """Test that addition uses ADD_SERIES function."""
    cmds = ['freq m', 'result = a + b + c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file has ADD_SERIES definition
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def ADD_SERIES" in formulas_content
        
        # Check transformer uses ADD_SERIES
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        assert 'ADD_SERIES("RESULT"' in ts_content
        assert 'pl.col("A")' in ts_content
        assert 'pl.col("B")' in ts_content
        assert 'pl.col("C")' in ts_content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


def test_sub_series_function():
    """Test that subtraction uses SUB_SERIES function."""
    cmds = ['freq m', 'result = a - b - c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file has SUB_SERIES definition
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def SUB_SERIES" in formulas_content
        
        # Check transformer uses SUB_SERIES
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        assert 'SUB_SERIES("RESULT"' in ts_content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


def test_mul_series_function():
    """Test that multiplication uses MUL_SERIES function."""
    cmds = ['freq m', 'result = a * b * c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file has MUL_SERIES definition
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def MUL_SERIES" in formulas_content
        
        # Check transformer uses MUL_SERIES
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        assert 'MUL_SERIES("RESULT"' in ts_content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


def test_div_series_function():
    """Test that division uses DIV_SERIES function."""
    cmds = ['freq m', 'result = a / b / c']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file has DIV_SERIES definition
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def DIV_SERIES" in formulas_content
        
        # Check transformer uses DIV_SERIES
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        assert 'DIV_SERIES("RESULT"' in ts_content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


def test_point_in_time_unquoted_date_ddmmmyyyy():
    """Test point-in-time assignment with unquoted date in ddMMMYYYY format."""
    result = parse_fame_formula('set a[12mar2020]=33')
    assert result is not None
    assert result["type"] == "point_in_time_assign"
    assert result["target"] == "a"
    assert result["date"] == "12mar2020"
    assert result["rhs"] == "33"


def test_point_in_time_unquoted_date_generation():
    """Test that point-in-time assignment with unquoted date generates correct code."""
    cmds = ['freq m', 'set a[12mar2020]=33']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file has POINT_IN_TIME_ASSIGN definition
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def POINT_IN_TIME_ASSIGN" in formulas_content
        
        # Check transformer uses chained when/then expressions
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        # Check for the when/then/otherwise chain pattern
        assert 'pl.when(pl.col("DATE") == pl.lit("2020-03-12").cast(pl.Date))' in ts_content
        assert '.then(pl.lit(33))' in ts_content
        assert '.alias("A")' in ts_content
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


def test_point_in_time_quarterly_unquoted():
    """Test point-in-time assignment with unquoted quarterly date."""
    result = parse_fame_formula('set cpi[2020Q1]=105.5')
    assert result is not None
    assert result["type"] == "point_in_time_assign"
    assert result["target"] == "cpi"
    assert result["date"] == "2020Q1"
    assert result["rhs"] == "105.5"


def test_comprehensive_example():
    """Test a comprehensive example with all features."""
    cmds = [
        'freq m',
        'base = 100',
        # Dot notation
        'result.a = d.a + b.c',
        # Point-in-time assignment
        'set v1[12mar2020]=33',
        # Conditional with standalone t
        'cond_result = if t ge 5 then result.a * 2 else nd',
        # Nested conditional
        'nested = if t gt 10 then base else if t ge 5 then base * 2 else base * 3',
        # Arithmetic operations
        'add_result = a + b + c',
        'sub_result = a - b',
        'mul_result = a * b',
        'div_result = a / b',
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        ts_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        formulas_file = f.name
    
    try:
        generate_formulas_file(cmds, formulas_file)
        generate_test_script(cmds, ts_file)
        
        # Check formulas file
        with open(formulas_file, 'r') as f:
            formulas_content = f.read()
        assert "def ADD_SERIES" in formulas_content
        assert "def SUB_SERIES" in formulas_content
        assert "def MUL_SERIES" in formulas_content
        assert "def DIV_SERIES" in formulas_content
        assert "def POINT_IN_TIME_ASSIGN" in formulas_content
        
        # Check transformer
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        
        # Check dot notation is preserved (uppercase with dots)
        assert 'D.A' in ts_content
        assert 'B.C' in ts_content
        assert 'RESULT.A' in ts_content
        
        # Check point-in-time assignment uses when/then chain
        assert 'pl.when(pl.col("DATE") == pl.lit("2020-03-12").cast(pl.Date))' in ts_content
        assert '.then(pl.lit(33))' in ts_content
        assert '.alias("V1")' in ts_content
        
        # Check conditionals - t should be mapped to DATE
        assert 'pl.when(pl.col("DATE") >= 5)' in ts_content
        assert 'pl.when(pl.col("DATE") > 10)' in ts_content
        
        # Check arithmetic functions
        assert 'ADD_SERIES("ADD_RESULT"' in ts_content
        assert 'SUB_SERIES("SUB_RESULT"' in ts_content
        assert 'MUL_SERIES("MUL_RESULT"' in ts_content
        assert 'DIV_SERIES("DIV_RESULT"' in ts_content
        
        # Verify code compiles
        compile(formulas_content, formulas_file, 'exec')
        compile(ts_content, ts_file, 'exec')
    finally:
        if os.path.exists(ts_file):
            os.unlink(ts_file)
        if os.path.exists(formulas_file):
            os.unlink(formulas_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
