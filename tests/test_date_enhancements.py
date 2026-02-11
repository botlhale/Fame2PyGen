"""
Tests for enhanced date format support and conditional improvements.

This test module validates:
1. Extended date format parsing (annual, monthly, weekly)
2. Nested conditional (IF-ELSE IF) handling
3. Multiple conditions with AND/OR operators
4. Date-based conditionals in transformations
"""

import pytest
import polars as pl
from fame2pygen.formulas_generator import (
    convert_fame_date_to_iso,
    parse_fame_formula,
    render_conditional_expr,
    render_polars_expr,
)
from fame2pygen.fame2py_converter import generate_test_script
import tempfile


class TestEnhancedDateFormats:
    """Test enhanced date format parsing."""
    
    def test_iso_format(self):
        """Test ISO date format (YYYY-MM-DD)."""
        assert convert_fame_date_to_iso("2020-01-01") == "2020-01-01"
        assert convert_fame_date_to_iso("2021-12-31") == "2021-12-31"
    
    def test_quarterly_format(self):
        """Test quarterly format (YYYYQN)."""
        assert convert_fame_date_to_iso("2020Q1") == "2020-01-01"
        assert convert_fame_date_to_iso("2020Q2") == "2020-04-01"
        assert convert_fame_date_to_iso("2020Q3") == "2020-07-01"
        assert convert_fame_date_to_iso("2020Q4") == "2020-10-01"
    
    def test_fame_day_month_year_format(self):
        """Test FAME day-month-year format (DDmmmYYYY)."""
        assert convert_fame_date_to_iso("12jul1985") == "1985-07-12"
        assert convert_fame_date_to_iso("01Jan2020") == "2020-01-01"
        assert convert_fame_date_to_iso("31dec2021") == "2021-12-31"
    
    def test_annual_format(self):
        """Test annual format (YYYY) - returns first day of year."""
        assert convert_fame_date_to_iso("2020") == "2020-01-01"
        assert convert_fame_date_to_iso("2025") == "2025-01-01"
    
    def test_monthly_m_format(self):
        """Test monthly format with 'm' (YYYYmMM)."""
        assert convert_fame_date_to_iso("2020m01") == "2020-01-01"
        assert convert_fame_date_to_iso("2020m12") == "2020-12-01"
        assert convert_fame_date_to_iso("2021M06") == "2021-06-01"  # Case insensitive
    
    def test_monthly_name_format(self):
        """Test month name + year format (mmmYYYY)."""
        assert convert_fame_date_to_iso("jan2020") == "2020-01-01"
        assert convert_fame_date_to_iso("Dec2021") == "2021-12-01"
        assert convert_fame_date_to_iso("JUL2020") == "2020-07-01"
    
    def test_weekly_format(self):
        """Test weekly format (YYYY.WW) - approximate."""
        result = convert_fame_date_to_iso("2020.01")
        assert result.startswith("2020-01")  # Should be early January
        
        result = convert_fame_date_to_iso("2020.05")
        assert result.startswith("2020-01") or result.startswith("2020-02")  # Week 5
    
    def test_invalid_format_passthrough(self):
        """Test that invalid formats are returned as-is."""
        assert convert_fame_date_to_iso("invalid") == "invalid"
        assert convert_fame_date_to_iso("20x01") == "20x01"


class TestNestedConditionals:
    """Test nested conditional (IF-ELSE IF-ELSE) handling."""
    
    def test_parse_nested_if(self):
        """Test parsing of nested IF (ELSE IF) statement."""
        formula = "result = if t gt 10 then a else if t ge 5 then b else c"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert parsed["target"] == "result"
        assert parsed["condition"] == "t gt 10"
        assert parsed["then_expr"] == "a"
        assert "if t ge 5 then b else c" in parsed["else_expr"]
    
    def test_render_nested_if(self):
        """Test code generation for nested IF statement."""
        condition = "t gt 10"
        then_expr = "a"
        else_expr = "if t ge 5 then b else c"
        
        result = render_conditional_expr(condition, then_expr, else_expr)
        
        # Should have nested when/then/otherwise
        assert "pl.when" in result
        assert result.count(".when(") == 2  # Two when clauses
        assert result.count(".otherwise(") == 2  # Two otherwise clauses
    
    def test_triple_nested_if(self):
        """Test deeply nested IF statements."""
        formula = "result = if t gt 20 then a else if t gt 10 then b else if t gt 5 then c else d"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        # The else_expr should contain the rest of the nested conditions
        assert "if t gt 10" in parsed["else_expr"]


class TestMultipleConditions:
    """Test multiple conditions with AND/OR operators."""
    
    def test_and_condition(self):
        """Test AND condition in IF statement."""
        formula = "result = if (t gt 5 and t lt 10) then a else b"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert "and" in parsed["condition"]
    
    def test_and_condition_rendering(self):
        """Test that AND is converted to & in Polars."""
        condition = "(t gt 5 and t lt 10)"
        rendered = render_conditional_expr(condition, "a", "b")
        
        assert "&" in rendered
        assert "and" not in rendered.lower() or "when" in rendered.lower()
    
    def test_or_condition(self):
        """Test OR condition in IF statement."""
        formula = "result = if (t lt 5 or t gt 10) then a else b"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert "or" in parsed["condition"]
    
    def test_or_condition_rendering(self):
        """Test that OR is converted to | in Polars."""
        condition = "(t lt 5 or t gt 10)"
        rendered = render_conditional_expr(condition, "a", "b")
        
        assert "|" in rendered
        assert "or" not in rendered.lower() or "otherwise" in rendered.lower()
    
    def test_complex_condition(self):
        """Test complex condition with both AND and OR."""
        formula = "result = if (t gt 5 and t lt 10) or (t gt 20 and t lt 30) then a else b"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert "and" in parsed["condition"]
        assert "or" in parsed["condition"]


class TestDateBasedConditionals:
    """Test date-based conditionals in generated code."""
    
    def test_date_comparison_in_condition(self):
        """Test conditional with date comparison."""
        commands = [
            "freq m",
            "t = 1",
            "a = 10",
            "b = 20",
            "result = if t ge 100 then a else b"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        generate_test_script(commands, output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Should have date column comparison
        assert "pl.col(\"DATE\")" in content
        assert ">=" in content or "when" in content
    
    def test_nested_date_conditions(self):
        """Test nested conditionals with date comparisons."""
        commands = [
            "freq m",
            "t = 1",
            "a = 10",
            "b = 20",
            "c = 30",
            "result = if t gt 10 then a else if t ge 5 then b else c"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        generate_test_script(commands, output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Should have nested when clauses
        assert content.count("pl.when") >= 2
        assert "pl.col(\"DATE\")" in content


class TestConditionalWithNullValues:
    """Test conditionals with FAME null values (nd, na, nc)."""
    
    def test_nd_in_condition(self):
        """Test nd (null) in conditional else clause."""
        formula = "result = if t gt 100 then a else nd"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert parsed["else_expr"].lower() == "nd"
    
    def test_nd_rendering(self):
        """Test that nd is converted to pl.lit(None)."""
        rendered = render_conditional_expr("t gt 100", "a", "nd")
        
        assert "pl.lit(None)" in rendered
        assert "nd" not in rendered.lower() or "when" in rendered.lower()
    
    def test_na_in_condition(self):
        """Test na (not available) in conditional."""
        formula = "result = if t gt 100 then a else na"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert parsed["else_expr"].lower() == "na"
    
    def test_nc_in_condition(self):
        """Test nc (not computed) in conditional."""
        formula = "result = if t gt 100 then a else nc"
        parsed = parse_fame_formula(formula)
        
        assert parsed["type"] == "conditional"
        assert parsed["else_expr"].lower() == "nc"


class TestDateofWrapping:
    """Ensure DATEOF arguments are wrapped as columns regardless of casing."""

    def test_dateof_arguments_wrapped_in_pl_col(self):
        rendered = render_polars_expr('dateof(make(date(bus), "10aug2020"), *, contain, end)')
        assert "DATEOF_GENERIC(" in rendered
        assert rendered.count('pl.col(') >= 4  # all arguments wrapped
        assert 'pl.col("CONTAIN")' in rendered
        assert 'pl.col("END")' in rendered

    def test_dateof_case_insensitive_handling(self):
        rendered = render_polars_expr("DaTeOf(Bus)")
        assert rendered == 'DATEOF_GENERIC(pl.col("BUS"))'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
