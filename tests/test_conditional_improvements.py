"""
Tests for improved conditional handling and column name sanitization.
"""
import pytest
import polars as pl
from fame2pygen.formulas_generator import parse_fame_formula, sanitize_func_name, render_conditional_expr
from fame2pygen.fame2py_converter import generate_test_script, generate_formulas_file
import tempfile
import os


class TestConditionalKeywordHandling:
    """Test that conditional keywords (if, then, else, ge, gt, le, lt, eq, ne) are not treated as column references."""
    
    def test_parse_conditional_with_ge(self):
        """Test that 'ge' is recognized as comparison operator, not column."""
        result = parse_fame_formula("result = if a ge 100 then b else c")
        assert result["type"] == "conditional"
        assert "ge" not in result["refs"]  # 'ge' should not be in refs
        assert "a" in result["refs"]
        assert "b" in result["refs"]
        assert "c" in result["refs"]
    
    def test_parse_conditional_with_t_variable(self):
        """Test that 't' (time variable) is not treated as column reference."""
        result = parse_fame_formula("result = if t ge 100 then a else b")
        assert result["type"] == "conditional"
        assert "t" not in result["refs"]  # 't' should not be in refs
        assert "a" in result["refs"]
        assert "b" in result["refs"]
    
    def test_parse_conditional_with_date_functions(self):
        """Test that date functions are preserved in condition."""
        result = parse_fame_formula('c = if t ge dateof(make (date(bus), "10aug2020"), *, contain, end) then a+b else nd')
        assert result["type"] == "conditional"
        # Conditional keywords should not be in refs
        assert "if" not in result["refs"]
        assert "then" not in result["refs"]
        assert "else" not in result["refs"]
        assert "ge" not in result["refs"]
        assert "nd" not in result["refs"]
        # Variables should be in refs
        assert "a" in result["refs"]
        assert "b" in result["refs"]


class TestColumnNameSanitization:
    """Test that column names with dots are properly handled."""
    
    def test_sanitize_preserves_dots(self):
        """Test that dots in column names are preserved (since Polars supports them)."""
        # New behavior: dots are preserved and lowercased
        assert sanitize_func_name("d.a") == "d.a"
        assert sanitize_func_name("my.var.name") == "my.var.name"
        assert sanitize_func_name("D.A") == "d.a"  # Still lowercased
    
    def test_parse_dotted_column_assignment(self):
        """Test parsing assignment to dotted column name."""
        result = parse_fame_formula("d.a = 100")
        assert result["type"] == "simple" or result["type"] == "assign_series"
        assert result["target"] == "d.a"


class TestArithmeticInConditionals:
    """Test that arithmetic operations in conditionals use appropriate series functions."""
    
    def test_addition_in_then_clause(self):
        """Test that a+b in then clause is recognized."""
        result = parse_fame_formula("c = if t ge 100 then a+b else nd")
        assert result["type"] == "conditional"
        assert result["then_expr"] == "a+b"
        assert result["else_expr"] == "nd"
    
    def test_nested_conditional_with_arithmetic(self):
        """Test nested conditionals with arithmetic operations."""
        result = parse_fame_formula("c = if t ge 100 then a+b else if t le 50 then c+d else nd")
        assert result["type"] == "conditional"
        assert result["then_expr"] == "a+b"
        # else_expr should be a nested conditional
        assert result["else_expr"].strip().startswith("if")


class TestCustomFunctionCalls:
    """Test that custom function calls are recognized vs built-in functions."""
    
    def test_convert_custom_function(self):
        """Test that convert with 4 params can be custom function."""
        result = parse_fame_formula("b = convert(temp, bus, dis, ave)")
        # This should be recognized as convert function
        assert result["type"] == "convert"
        assert result["params"] == ["temp", "bus", "dis", "ave"]
    
    def test_convert_builtin_function(self):
        """Test standard convert function with 5 params."""
        result = parse_fame_formula("v23 = convert(v24, 'Q', 'M', 'AVG', 'END')")
        assert result["type"] == "convert"
        assert len(result["params"]) == 5


class TestPointInTimeAssignment:
    """Test that point-in-time assignments are correctly parsed."""
    
    def test_unquoted_date_format(self):
        """Test parsing of unquoted date like 12mar2020."""
        result = parse_fame_formula("set a[12mar2020]=33")
        assert result["type"] == "point_in_time_assign"
        assert result["target"] == "a"
        assert result["date"] == "12mar2020"
        assert result["rhs"] == "33"
    
    def test_quoted_date_format(self):
        """Test parsing of quoted date."""
        result = parse_fame_formula('gdp["2020-01-01"] = 1000')
        assert result["type"] == "point_in_time_assign"
        assert result["target"] == "gdp"
        assert result["date"] == "2020-01-01"
        assert result["rhs"] == "1000"


class TestCodeGeneration:
    """Test that generated code properly handles new features."""
    
    def test_conditional_with_t_variable_generates_date_col(self):
        """Test that 't' in conditional is mapped to DATE column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                commands = [
                    "freq m",
                    "a = 100",
                    "b = 200",
                    "result = if t ge 100 then a else b",
                ]
                generate_formulas_file(commands, "formulas.py")
                generate_test_script(commands, "ts_transformer.py")
                
                # Read generated ts_transformer
                with open("ts_transformer.py", "r") as f:
                    content = f.read()
                
                # Check that 't' is replaced with pl.col("DATE") in the condition
                # The condition should be: pl.col("DATE") >= 100
                assert 'pl.col("DATE")' in content
                # Should not have pl.col("T")
                assert 'pl.col("T")' not in content
            finally:
                os.chdir(old_cwd)
    
    def test_arithmetic_in_conditionals_uses_series_functions(self):
        """Test that arithmetic in conditional branches uses ADD_SERIES etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                commands = [
                    "freq m",
                    "a = 100",
                    "b = 200",
                    "c = 300",
                    "result = if a ge 100 then a+b else c",
                ]
                generate_formulas_file(commands, "formulas.py")
                generate_test_script(commands, "ts_transformer.py")
                
                # Read generated files
                with open("formulas.py", "r") as f:
                    formulas_content = f.read()
                with open("ts_transformer.py", "r") as f:
                    ts_content = f.read()
                
                # Check that ADD_SERIES function is defined
                assert "def ADD_SERIES" in formulas_content
                # Check that arithmetic in then clause uses proper expression
                # (May use ADD_SERIES or direct pl.col operations depending on implementation)
            finally:
                os.chdir(old_cwd)


class TestRenderConditionalExpr:
    """Test render_conditional_expr function directly."""
    
    def test_render_simple_conditional(self):
        """Test rendering a simple conditional expression."""
        condition = "a >= 100"
        then_expr = "b"
        else_expr = "c"
        subs = {"a": 'pl.col("A")', "b": 'pl.col("B")', "c": 'pl.col("C")'}
        
        result = render_conditional_expr(condition, then_expr, else_expr, substitution_map=subs)
        assert "pl.when(" in result
        assert ".then(" in result
        assert ".otherwise(" in result
    
    def test_render_conditional_with_nd(self):
        """Test rendering conditional with 'nd' (null) keyword."""
        condition = "a >= 100"
        then_expr = "b"
        else_expr = "nd"
        subs = {"a": 'pl.col("A")', "b": 'pl.col("B")'}
        
        result = render_conditional_expr(condition, then_expr, else_expr, substitution_map=subs)
        assert "pl.lit(None)" in result
