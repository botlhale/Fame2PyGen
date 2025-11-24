"""
Tests for FAME LSUM function and nested IF improvements.

The LSUM function sums multiple arguments, treating null values as 0.
This is commonly used in FAME with conditional expressions that handle
missing values.

Example FAME formula:
AA = IF T GT DATEOF(A.FINALDATE,*,BEFORE,ENDING) THEN ND 
     ELSE LSUM((if exists(BBA) then (if BBA EQ NA then 0 ELSE IF BBA EQ NC THEN 0 
               ELSE IF BBA EQ ND THEN 0 ELSE BBA) else 0), ...)
"""
import pytest
import polars as pl
from fame2pygen.formulas_generator import (
    parse_fame_formula,
    generate_polars_functions,
    render_polars_expr,
    render_conditional_expr,
    _token_to_pl_expr,
    _split_lsum_args,
)


class TestLSUMParsing:
    """Test LSUM function parsing."""
    
    def test_simple_lsum(self):
        """Test parsing of simple LSUM expression."""
        result = parse_fame_formula("total = lsum(a, b, c)")
        assert result is not None
        assert result["type"] == "simple"
        assert result["target"] == "total"
        # lsum should not appear in refs as it's a function
        assert "lsum" not in [r.lower() for r in result["refs"]]
        assert "a" in [r.lower() for r in result["refs"]]
        assert "b" in [r.lower() for r in result["refs"]]
        assert "c" in [r.lower() for r in result["refs"]]
    
    def test_lsum_with_conditionals(self):
        """Test parsing LSUM with conditional arguments."""
        result = parse_fame_formula("total = lsum((if a gt 0 then a else 0), (if b gt 0 then b else 0))")
        assert result is not None
        assert result["type"] == "simple"
        # lsum should not appear in refs
        assert "lsum" not in [r.lower() for r in result["refs"]]
        # Variables should be in refs
        assert "a" in [r.lower() for r in result["refs"]]
        assert "b" in [r.lower() for r in result["refs"]]


class TestExistsParsing:
    """Test EXISTS function parsing."""
    
    def test_simple_exists(self):
        """Test parsing of simple exists expression."""
        result = parse_fame_formula("check = if exists(a) then a else 0")
        assert result is not None
        assert result["type"] == "conditional"
        # exists should not appear in refs as it's a function
        assert "exists" not in [r.lower() for r in result["refs"]]
        assert "a" in [r.lower() for r in result["refs"]]


class TestNANCNDHandling:
    """Test handling of FAME special values NA, NC, ND."""
    
    def test_token_na_handling(self):
        """Test that NA is converted to pl.lit(None)."""
        result = _token_to_pl_expr("NA")
        assert result == "pl.lit(None)"
        
    def test_token_nc_handling(self):
        """Test that NC is converted to pl.lit(None)."""
        result = _token_to_pl_expr("NC")
        assert result == "pl.lit(None)"
        
    def test_token_nd_handling(self):
        """Test that ND is converted to pl.lit(None)."""
        result = _token_to_pl_expr("ND")
        assert result == "pl.lit(None)"
        result_lower = _token_to_pl_expr("nd")
        assert result_lower == "pl.lit(None)"
    
    def test_parse_conditional_with_na(self):
        """Test that NA is not treated as a variable reference."""
        result = parse_fame_formula("check = if a eq na then 0 else a")
        assert result is not None
        assert result["type"] == "conditional"
        # NA should not be in refs
        assert "na" not in [r.lower() for r in result["refs"]]
        assert "a" in [r.lower() for r in result["refs"]]
    
    def test_parse_conditional_with_nc(self):
        """Test that NC is not treated as a variable reference."""
        result = parse_fame_formula("check = if a eq nc then 0 else a")
        assert result is not None
        assert result["type"] == "conditional"
        # NC should not be in refs
        assert "nc" not in [r.lower() for r in result["refs"]]


class TestSplitLsumArgs:
    """Test the _split_lsum_args helper function."""
    
    def test_simple_args(self):
        """Test splitting simple comma-separated arguments."""
        args = _split_lsum_args("a, b, c")
        assert len(args) == 3
        assert args[0] == "a"
        assert args[1] == "b"
        assert args[2] == "c"
    
    def test_nested_parens(self):
        """Test splitting with nested parentheses."""
        args = _split_lsum_args("(if a then b else c), (if d then e else f)")
        assert len(args) == 2
        assert "if a then b else c" in args[0]
        assert "if d then e else f" in args[1]
    
    def test_deeply_nested(self):
        """Test splitting with deeply nested parentheses."""
        args = _split_lsum_args("(if exists(a) then (if a eq na then 0 else a) else 0), b")
        assert len(args) == 2
        assert "exists(a)" in args[0]
        assert args[1].strip() == "b"


class TestLSUMFunctionGeneration:
    """Test LSUM function generation."""
    
    def test_lsum_function_generated(self):
        """Test that LSUM helper function is generated when lsum is used."""
        cmds = ["total = lsum(a, b, c)"]
        defs = generate_polars_functions(cmds)
        assert "LSUM" in defs
        assert "def LSUM" in defs["LSUM"]
        assert "fill_null(0)" in defs["LSUM"]
    
    def test_exists_function_generated(self):
        """Test that EXISTS helper function is generated when exists is used."""
        cmds = ["check = if exists(a) then a else 0"]
        defs = generate_polars_functions(cmds)
        assert "EXISTS" in defs
        assert "def EXISTS" in defs["EXISTS"]
        assert "is_not_null()" in defs["EXISTS"]


class TestComplexNestedIF:
    """Test complex nested IF expressions like the example in the issue."""
    
    def test_nested_if_with_special_values(self):
        """Test parsing deeply nested IF with NA, NC, ND checks."""
        formula = "(if BBA EQ NA then 0 ELSE IF BBA EQ NC THEN 0 ELSE IF BBA EQ ND THEN 0 ELSE BBA)"
        # This tests the tokenization and parsing
        result = parse_fame_formula(f"check = {formula}")
        assert result is not None
        # NA, NC, ND should not be in refs
        refs_lower = [r.lower() for r in result.get("refs", [])]
        assert "na" not in refs_lower
        assert "nc" not in refs_lower
        assert "nd" not in refs_lower
        # BBA should be in refs
        assert "bba" in refs_lower
    
    def test_exists_with_nested_if(self):
        """Test parsing exists with nested IF expressions."""
        formula = "(if exists(BBA) then (if BBA EQ NA then 0 ELSE BBA) else 0)"
        result = parse_fame_formula(f"check = {formula}")
        assert result is not None
        refs_lower = [r.lower() for r in result.get("refs", [])]
        # exists should not be in refs (it's a function)
        assert "exists" not in refs_lower
        assert "bba" in refs_lower


class TestCodeGenerationWithLSUM:
    """Test generated code with LSUM and EXISTS."""
    
    def test_lsum_in_transformer(self):
        """Test that LSUM generates correct transformer code."""
        from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
        import tempfile
        import os
        
        cmds = [
            "freq m",
            "total = lsum(a, b, c)"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            formulas_file = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            ts_file = f.name
        
        try:
            generate_formulas_file(cmds, formulas_file)
            generate_test_script(cmds, ts_file)
            
            with open(formulas_file, 'r') as f:
                formulas_content = f.read()
            with open(ts_file, 'r') as f:
                ts_content = f.read()
            
            # Check that LSUM function is defined
            assert "def LSUM" in formulas_content
            
            # Check that transformer uses LSUM
            assert "LSUM" in ts_content
            
            # Verify code compiles
            compile(formulas_content, formulas_file, 'exec')
            compile(ts_content, ts_file, 'exec')
        finally:
            if os.path.exists(formulas_file):
                os.unlink(formulas_file)
            if os.path.exists(ts_file):
                os.unlink(ts_file)
    
    def test_exists_in_conditional_transformer(self):
        """Test that exists() in conditional generates correct code."""
        from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
        import tempfile
        import os
        
        cmds = [
            "freq m",
            "check = if exists(a) then a else 0"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            formulas_file = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            ts_file = f.name
        
        try:
            generate_formulas_file(cmds, formulas_file)
            generate_test_script(cmds, ts_file)
            
            with open(formulas_file, 'r') as f:
                formulas_content = f.read()
            with open(ts_file, 'r') as f:
                ts_content = f.read()
            
            # Check that EXISTS function is defined in formulas
            assert "def EXISTS" in formulas_content
            
            # Check that transformer uses exists (imported from formulas as EXISTS)
            # The transformer generates lowercase 'exists' which is provided by formulas.py
            assert "exists" in ts_content.lower()
            
            # Verify code compiles
            compile(formulas_content, formulas_file, 'exec')
            compile(ts_content, ts_file, 'exec')
        finally:
            if os.path.exists(formulas_file):
                os.unlink(formulas_file)
            if os.path.exists(ts_file):
                os.unlink(ts_file)


class TestLSUMExecution:
    """Test LSUM function execution with actual data."""
    
    def test_lsum_execution_simple(self):
        """Test LSUM execution with simple values."""
        from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
        import tempfile
        import os
        import sys
        from datetime import date
        
        cmds = [
            "freq m",
            "a = 10",
            "b = 20",
            "c = 30",
            "total = lsum(a, b, c)"
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            formulas_file = os.path.join(tmpdir, "formulas.py")
            ts_file = os.path.join(tmpdir, "ts_transformer.py")
            
            generate_formulas_file(cmds, formulas_file)
            generate_test_script(cmds, ts_file)
            
            sys.path.insert(0, tmpdir)
            
            try:
                import importlib.util
                
                # First load formulas module to ensure LSUM is available
                formulas_spec = importlib.util.spec_from_file_location("formulas", formulas_file)
                formulas_module = importlib.util.module_from_spec(formulas_spec)
                sys.modules['formulas'] = formulas_module  # Register it so imports work
                formulas_spec.loader.exec_module(formulas_module)
                
                # Now load ts_transformer
                spec = importlib.util.spec_from_file_location("ts_transformer", ts_file)
                ts_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ts_module)
                
                # Create test DataFrame
                test_df = pl.DataFrame({
                    "DATE": [date(2020, 1, 1), date(2020, 2, 1)]
                })
                
                # Apply transformations
                result = ts_module.ts_transformer(test_df)
                
                # Verify results
                assert "TOTAL" in result.columns
                # LSUM(10, 20, 30) = 60
                assert all(result["TOTAL"] == 60)
            finally:
                sys.path.remove(tmpdir)
                # Clean up the formulas module from cache
                if 'formulas' in sys.modules:
                    del sys.modules['formulas']


class TestIssueExampleFormula:
    """Test the specific formula from the issue."""
    
    def test_issue_formula_parsing(self):
        """Test parsing the complex formula from the issue description."""
        # Simplified version of the issue formula
        formula = """AA = IF T GT 100 THEN ND ELSE LSUM((if exists(BBA) then (if BBA EQ NA then 0 ELSE IF BBA EQ NC THEN 0 ELSE IF BBA EQ ND THEN 0 ELSE BBA) else 0),(if exists(BBB) then (if BBB EQ NA then 0 ELSE BBB) else 0))"""
        
        result = parse_fame_formula(formula)
        assert result is not None
        assert result["type"] == "conditional"
        assert result["target"].lower() == "aa"
        
        # Check that the condition contains the comparison
        assert "gt" in result["condition"].lower() or ">" in result["condition"]
        
        # Check that the else_expr contains LSUM
        assert "lsum" in result["else_expr"].lower()
        
        # Check refs - should include variables but not function names or special values
        refs_lower = [r.lower() for r in result.get("refs", [])]
        assert "bba" in refs_lower or "bbb" in refs_lower
        # Special values and functions should not be in refs
        assert "na" not in refs_lower
        assert "nc" not in refs_lower
        assert "nd" not in refs_lower
        assert "lsum" not in refs_lower
        assert "exists" not in refs_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
