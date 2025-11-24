"""
Tests for NLRX, FIRSTVALUE, and LASTVALUE functions.
"""
import pytest
from fame2pygen.formulas_generator import parse_fame_formula, generate_polars_functions
from fame2pygen import generate_formulas_file, generate_test_script


class TestFirstValueParsing:
    """Test parsing of firstvalue() function."""
    
    def test_parse_firstvalue_simple(self):
        """Test parsing simple firstvalue assignment."""
        result = parse_fame_formula("start = firstvalue(a)")
        assert result is not None
        assert result["type"] == "firstvalue"
        assert result["target"] == "start"
        assert result["series"] == "a"
        assert "a" in result["refs"]
    
    def test_parse_firstvalue_with_complex_series(self):
        """Test parsing firstvalue with complex series name."""
        result = parse_fame_formula("mystart = firstvalue(series_123)")
        assert result is not None
        assert result["type"] == "firstvalue"
        assert result["target"] == "mystart"
        assert result["series"] == "series_123"


class TestLastValueParsing:
    """Test parsing of lastvalue() function."""
    
    def test_parse_lastvalue_simple(self):
        """Test parsing simple lastvalue assignment."""
        result = parse_fame_formula("end = lastvalue(a)")
        assert result is not None
        assert result["type"] == "lastvalue"
        assert result["target"] == "end"
        assert result["series"] == "a"
        assert "a" in result["refs"]
    
    def test_parse_lastvalue_with_complex_series(self):
        """Test parsing lastvalue with complex series name."""
        result = parse_fame_formula("myend = lastvalue(series_456)")
        assert result is not None
        assert result["type"] == "lastvalue"
        assert result["target"] == "myend"
        assert result["series"] == "series_456"


class TestNLRXParsing:
    """Test parsing of nlrx() function."""
    
    def test_parse_nlrx_basic(self):
        """Test parsing basic nlrx assignment."""
        result = parse_fame_formula("a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)")
        assert result is not None
        assert result["type"] == "nlrx"
        assert result["target"] == "a_t"
        assert len(result["params"]) == 8
        assert result["params"][0] == "lambda20"
        assert result["params"][1] == "a"
        assert result["params"][2] == "b1"
        assert result["params"][3] == "b2"
        assert result["params"][4] == "b3"
        assert result["params"][5] == "b4"
        assert result["params"][6] == "c"
        assert result["params"][7] == "d"
    
    def test_parse_nlrx_with_different_names(self):
        """Test parsing nlrx with different variable names."""
        result = parse_fame_formula("result = nlrx(20, y_var, w1_var, w2_var, w3_var, w4_var, gss_var, gpr_var)")
        assert result is not None
        assert result["type"] == "nlrx"
        assert result["target"] == "result"
        assert result["params"][0] == "20"
        assert result["params"][1] == "y_var"


class TestFunctionGeneration:
    """Test generation of helper functions."""
    
    def test_firstvalue_function_generation(self):
        """Test that FIRSTVALUE function is generated when needed."""
        commands = [
            "start = firstvalue(a)"
        ]
        functions = generate_polars_functions(commands)
        assert "FIRSTVALUE" in functions
        assert "def FIRSTVALUE" in functions["FIRSTVALUE"]
        assert "drop_nulls().first()" in functions["FIRSTVALUE"]
    
    def test_lastvalue_function_generation(self):
        """Test that LASTVALUE function is generated when needed."""
        commands = [
            "end = lastvalue(a)"
        ]
        functions = generate_polars_functions(commands)
        assert "LASTVALUE" in functions
        assert "def LASTVALUE" in functions["LASTVALUE"]
        assert "drop_nulls().last()" in functions["LASTVALUE"]
    
    def test_nlrx_function_generation(self):
        """Test that NLRX function is generated when needed."""
        commands = [
            "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)"
        ]
        functions = generate_polars_functions(commands)
        assert "NLRX" in functions
        assert "def NLRX" in functions["NLRX"]
        assert "polars_econ as ple" in functions["NLRX"]
        assert "ple.nlrx" in functions["NLRX"]
    
    def test_all_three_functions_generated(self):
        """Test that all three functions are generated when all are used."""
        commands = [
            "start = firstvalue(a)",
            "end = lastvalue(a)",
            "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)"
        ]
        functions = generate_polars_functions(commands)
        assert "FIRSTVALUE" in functions
        assert "LASTVALUE" in functions
        assert "NLRX" in functions


class TestCodeGeneration:
    """Test complete code generation including formulas.py and ts_transformer.py."""
    
    def test_nlrx_complete_example(self):
        """Test the complete example from the problem statement."""
        import tempfile
        import os
        
        commands = [
            "start = firstvalue(a)",
            "end = lastvalue(a)",
            "lambda20 = 20",
            "set <date start to end> b1 = 1",
            "set <date start-7 to end> b1 = 0",
            "set <date start to end> b2 = 0",
            "set <date start to end> b3 = 0",
            "set <date start to end> b4 = 0",
            "set <date start to end> c = 0",
            "set <date start to end> d = 0",
            "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d, begsa, endmona)"
        ]
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as formulas_file:
            formulas_path = formulas_file.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as transformer_file:
            transformer_path = transformer_file.name
        
        try:
            # Generate files
            generate_formulas_file(commands, formulas_path)
            generate_test_script(commands, transformer_path)
            
            # Read formulas file
            with open(formulas_path, 'r') as f:
                formulas_code = f.read()
            
            # Read transformer file
            with open(transformer_path, 'r') as f:
                transformer_code = f.read()
            
            # Check that all required functions are present
            assert "def FIRSTVALUE" in formulas_code
            assert "def LASTVALUE" in formulas_code
            assert "def NLRX" in formulas_code
            
            # Check imports
            assert "import polars as pl" in formulas_code
            
            # Check that transformer uses the functions correctly
            assert "FIRSTVALUE" in transformer_code
            assert "LASTVALUE" in transformer_code
            assert "NLRX" in transformer_code
            assert "from formulas import" in transformer_code
        finally:
            # Cleanup
            if os.path.exists(formulas_path):
                os.unlink(formulas_path)
            if os.path.exists(transformer_path):
                os.unlink(transformer_path)
    
    def test_firstvalue_in_transformer(self):
        """Test that firstvalue is properly used in transformer."""
        import tempfile
        import os
        
        commands = [
            "start = firstvalue(series_a)"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as transformer_file:
            transformer_path = transformer_file.name
        
        try:
            generate_test_script(commands, transformer_path)
            
            with open(transformer_path, 'r') as f:
                transformer_code = f.read()
            
            # Check that FIRSTVALUE is called
            assert "FIRSTVALUE" in transformer_code
            assert "SERIES_A" in transformer_code or "series_a" in transformer_code.upper()
        finally:
            if os.path.exists(transformer_path):
                os.unlink(transformer_path)
    
    def test_lastvalue_in_transformer(self):
        """Test that lastvalue is properly used in transformer."""
        import tempfile
        import os
        
        commands = [
            "end = lastvalue(series_b)"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as transformer_file:
            transformer_path = transformer_file.name
        
        try:
            generate_test_script(commands, transformer_path)
            
            with open(transformer_path, 'r') as f:
                transformer_code = f.read()
            
            # Check that LASTVALUE is called
            assert "LASTVALUE" in transformer_code
            assert "SERIES_B" in transformer_code or "series_b" in transformer_code.upper()
        finally:
            if os.path.exists(transformer_path):
                os.unlink(transformer_path)
    
    def test_nlrx_in_transformer(self):
        """Test that nlrx is properly used in transformer."""
        import tempfile
        import os
        
        commands = [
            "lambda20 = 20",
            "result = nlrx(lambda20, y, w1, w2, w3, w4, gss, gpr)"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as transformer_file:
            transformer_path = transformer_file.name
        
        try:
            generate_test_script(commands, transformer_path)
            
            with open(transformer_path, 'r') as f:
                transformer_code = f.read()
            
            # Check that NLRX is called
            assert "NLRX" in transformer_code
            # Check that pdf is reassigned (since NLRX returns a DataFrame)
            assert "pdf = NLRX" in transformer_code
        finally:
            if os.path.exists(transformer_path):
                os.unlink(transformer_path)


class TestIntegration:
    """Integration tests for NLRX with date filtering."""
    
    def test_nlrx_with_date_filters(self):
        """Test NLRX with date-filtered variable assignments."""
        import tempfile
        import os
        
        commands = [
            "freq m",
            "start = firstvalue(a)",
            "end = lastvalue(a)",
            "lambda20 = 20",
            "date start to end",
            "b1 = 1",
            "b2 = 0",
            "b3 = 0",
            "b4 = 0",
            "c = 0",
            "d = 0",
            "date *",
            "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as formulas_file:
            formulas_path = formulas_file.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as transformer_file:
            transformer_path = transformer_file.name
        
        try:
            generate_formulas_file(commands, formulas_path)
            generate_test_script(commands, transformer_path)
            
            with open(formulas_path, 'r') as f:
                formulas_code = f.read()
            with open(transformer_path, 'r') as f:
                transformer_code = f.read()
            
            # Check that all components are present
            assert "FIRSTVALUE" in formulas_code
            assert "LASTVALUE" in formulas_code
            assert "NLRX" in formulas_code
            assert "APPLY_DATE_FILTER" in formulas_code
            
            # Check transformer has proper flow
            assert "START" in transformer_code
            assert "END" in transformer_code
            assert "NLRX" in transformer_code
        finally:
            if os.path.exists(formulas_path):
                os.unlink(formulas_path)
            if os.path.exists(transformer_path):
                os.unlink(transformer_path)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nlrx_with_numeric_literal(self):
        """Test NLRX with numeric literal for lambda."""
        result = parse_fame_formula("a_t = nlrx(20, a, b1, b2, b3, b4, c, d)")
        assert result is not None
        assert result["type"] == "nlrx"
        assert result["params"][0] == "20"
    
    def test_nlrx_with_more_params(self):
        """Test NLRX with additional parameters (should still parse)."""
        result = parse_fame_formula("a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d, extra1, extra2)")
        assert result is not None
        assert result["type"] == "nlrx"
        assert len(result["params"]) == 10
    
    def test_firstvalue_case_insensitive(self):
        """Test that function names are case insensitive."""
        result1 = parse_fame_formula("start = firstvalue(a)")
        result2 = parse_fame_formula("start = FIRSTVALUE(a)")
        result3 = parse_fame_formula("start = FirstValue(a)")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["type"] == result2["type"] == result3["type"] == "firstvalue"
    
    def test_lastvalue_case_insensitive(self):
        """Test that function names are case insensitive."""
        result1 = parse_fame_formula("end = lastvalue(a)")
        result2 = parse_fame_formula("end = LASTVALUE(a)")
        result3 = parse_fame_formula("end = LastValue(a)")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["type"] == result2["type"] == result3["type"] == "lastvalue"
    
    def test_nlrx_case_insensitive(self):
        """Test that function names are case insensitive."""
        result1 = parse_fame_formula("a_t = nlrx(l, a, b1, b2, b3, b4, c, d)")
        result2 = parse_fame_formula("a_t = NLRX(l, a, b1, b2, b3, b4, c, d)")
        result3 = parse_fame_formula("a_t = NlRx(l, a, b1, b2, b3, b4, c, d)")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["type"] == result2["type"] == result3["type"] == "nlrx"
