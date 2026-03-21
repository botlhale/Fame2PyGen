"""
Master Test Suite for Fame2PyGen
================================
Consolidates all FAME-to-Python transpilation scenarios into a single,
comprehensive test module.  The suite exercises:

  - Parsing of every supported FAME construct
  - Polars expression rendering
  - Helper-function generation  (formulas.py)
  - Transformer code generation (ts_transformer.py)
  - End-to-end execution of generated code

Each section corresponds to a major FAME language feature and includes
both unit-level (parsing / rendering) and integration-level (code
generation + execution) tests.
"""

import os
import re
import sys
import tempfile
import textwrap

import polars as pl
import pytest

from fame2pygen.formulas_generator import (
    convert_fame_date_to_iso,
    extract_if_components,
    generate_polars_functions,
    normalize_formula_text,
    parse_fame_formula,
    render_conditional_expr,
    render_polars_expr,
    sanitize_func_name,
    token_to_pl_expr,
)
from fame2pygen.fame2py_converter import (
    generate_formulas_file,
    generate_test_script,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_and_read(cmds):
    """Generate both files for *cmds* and return their contents."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "formulas.py")
        tp = os.path.join(d, "ts_transformer.py")
        generate_formulas_file(cmds, fp)
        generate_test_script(cmds, tp)
        with open(fp) as fh:
            formulas = fh.read()
        with open(tp) as fh:
            transformer = fh.read()
    return formulas, transformer


def _load_and_run(cmds, df):
    """Generate code, load it, run the transformer, return result DataFrame."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "formulas.py")
        tp = os.path.join(d, "ts_transformer.py")
        generate_formulas_file(cmds, fp)
        generate_test_script(cmds, tp)

        import importlib.util
        # Load formulas first
        spec_f = importlib.util.spec_from_file_location("formulas", fp)
        mod_f = importlib.util.module_from_spec(spec_f)
        sys.modules["formulas"] = mod_f
        spec_f.loader.exec_module(mod_f)

        # Then load transformer
        spec_t = importlib.util.spec_from_file_location("ts_transformer", tp)
        mod_t = importlib.util.module_from_spec(spec_t)
        spec_t.loader.exec_module(mod_t)

        result = mod_t.ts_transformer(df)

        # Clean up sys.modules
        sys.modules.pop("formulas", None)
        sys.modules.pop("ts_transformer", None)

    return result


# ===================================================================
# 1. SIMPLE ASSIGNMENTS & ARITHMETIC
# ===================================================================

class TestSimpleAssignments:
    """Basic variable assignments and arithmetic operations."""

    def test_parse_simple_addition(self):
        r = parse_fame_formula("v1 = v2 + v3")
        assert r["type"] == "simple"
        assert r["target"] == "v1"
        assert "v2" in r["refs"]
        assert "v3" in r["refs"]

    def test_parse_simple_subtraction(self):
        r = parse_fame_formula("a = b - c")
        assert r["type"] == "simple"
        assert r["rhs"].strip() == "b - c"

    def test_parse_simple_multiplication(self):
        r = parse_fame_formula("out = x * y")
        assert r["type"] == "simple"

    def test_parse_simple_division(self):
        r = parse_fame_formula("ratio = a / b")
        assert r["type"] == "simple"

    def test_parse_mixed_arithmetic(self):
        r = parse_fame_formula("z = a + b * c - d")
        assert r["type"] == "simple"
        assert all(v in r["refs"] for v in ["a", "b", "c", "d"])

    def test_scalar_literal(self):
        r = parse_fame_formula("v1 = 150")
        assert r["type"] == "assign_series"
        assert r["rhs"] == "150"
        assert r["refs"] == []

    def test_scalar_literal_float(self):
        r = parse_fame_formula("rate = 3.14")
        assert r["type"] == "assign_series"
        assert r["rhs"] == "3.14"

    def test_set_keyword_stripped(self):
        r = parse_fame_formula("set x = y + z")
        assert r["type"] == "simple"
        assert r["target"] == "x"

    def test_dot_notation_preserved(self):
        r = parse_fame_formula("result.a = d.a + b.c")
        assert r is not None
        assert r["target"] is not None

    def test_scalar_code_generation(self):
        """Scalar assignment must wrap literal in pl.lit()."""
        _, ts = _generate_and_read(["freq m", "v1 = 150"])
        assert "pl.lit(150)" in ts
        # Must not produce bare 150.alias() which is a SyntaxError
        assert "150.alias" not in ts

    def test_arithmetic_code_generation(self):
        _, ts = _generate_and_read(["freq m", "v1 = v2 + v3"])
        assert "V1" in ts
        assert "V2" in ts
        assert "V3" in ts

    def test_scalar_execution(self):
        """Generated scalar assignment must execute without error."""
        cmds = ["freq m", "v1 = 42"]
        df = pl.DataFrame({"DATE": [None]})
        result = _load_and_run(cmds, df)
        assert "V1" in result.columns
        assert result["V1"][0] == 42

    def test_arithmetic_execution(self):
        cmds = ["freq m", "z = x + y"]
        df = pl.DataFrame({"DATE": [None], "X": [10], "Y": [20]})
        result = _load_and_run(cmds, df)
        assert "Z" in result.columns
        assert result["Z"][0] == 30


# ===================================================================
# 2. FREQUENCY COMMANDS
# ===================================================================

class TestFrequencyCommands:

    @pytest.mark.parametrize("cmd,expected", [
        ("freq m", "m"), ("freq q", "q"), ("freq a", "a"),
        ("freq w", "w"), ("freq d", "d"), ("freq b", "b"),
        ("freq bus", "bus"), ("freq M", "m"),
    ])
    def test_parse_freq(self, cmd, expected):
        r = parse_fame_formula(cmd)
        assert r["type"] == "freq"
        assert r["freq"] == expected


# ===================================================================
# 3. DATE COMMANDS & FILTERING
# ===================================================================

class TestDateCommands:

    def test_date_all(self):
        r = parse_fame_formula("date *")
        assert r["type"] == "date"
        assert r["filter"] is None

    def test_date_range_iso(self):
        r = parse_fame_formula("date 2020-01-01 to 2020-12-31")
        assert r["type"] == "date"
        assert r["filter"]["start"] == "2020-01-01"
        assert r["filter"]["end"] == "2020-12-31"

    def test_date_range_fame_format(self):
        r = parse_fame_formula("date 01jan2020 to 31dec2020")
        assert r["type"] == "date"
        assert r["filter"]["start"] == "01jan2020"

    def test_date_range_quarterly(self):
        r = parse_fame_formula("date 2020Q1 to 2020Q4")
        assert r["type"] == "date"

    def test_date_range_star_end(self):
        r = parse_fame_formula("date 01Jan2021 to *")
        assert r["type"] == "date"
        assert r["filter"]["end"] == "*"

    def test_date_filter_affects_code_generation(self):
        cmds = [
            "freq m",
            "date 2020-01-01 to 2020-12-31",
            "v1 = v2 + v3",
            "date *",
            "v4 = v5 + v6",
        ]
        _, ts = _generate_and_read(cmds)
        assert "# Date filter: 2020-01-01 to 2020-12-31" in ts
        assert "# Date filter: * (all dates)" in ts
        assert "APPLY_DATE_FILTER" in ts

    def test_multiple_date_ranges(self):
        cmds = [
            "freq m",
            "v_base = 100",
            "date 2020-01-01 to 2020-12-31",
            "v_2020 = v_base * 2",
            "date 2021-01-01 to 2021-12-31",
            "v_2021 = v_base * 3",
        ]
        _, ts = _generate_and_read(cmds)
        assert ts.count("APPLY_DATE_FILTER") >= 2


# ===================================================================
# 4. DATE FORMAT CONVERSION
# ===================================================================

class TestDateConversion:

    @pytest.mark.parametrize("fame_date,iso", [
        ("2020-01-01", "2020-01-01"),   # ISO pass-through
        ("2020Q1", "2020-01-01"),       # Quarterly
        ("2020Q3", "2020-07-01"),
        ("12jul1985", "1985-07-12"),    # FAME day-month-year
        ("01jan2020", "2020-01-01"),
        ("2020", "2020-01-01"),         # Annual
        ("2020m03", "2020-03-01"),      # Monthly numeric
        ("jan2020", "2020-01-01"),      # Month-name year
        ("dec2021", "2021-12-01"),
    ])
    def test_fame_to_iso(self, fame_date, iso):
        assert convert_fame_date_to_iso(fame_date) == iso


# ===================================================================
# 5. CONDITIONAL EXPRESSIONS
# ===================================================================

class TestConditionals:

    def test_parse_simple_if(self):
        r = parse_fame_formula("result = if a gt 100 then b else c")
        assert r["type"] == "conditional"
        assert r["target"] == "result"

    def test_comparison_operators_not_in_refs(self):
        r = parse_fame_formula("result = if a ge 100 then b else c")
        assert r["type"] == "conditional"
        assert "ge" not in r["refs"]

    def test_all_comparison_operators(self):
        for op in ["eq", "ne", "gt", "lt", "ge", "le"]:
            r = parse_fame_formula(f"r = if x {op} 10 then y else z")
            assert r["type"] == "conditional", f"Failed for operator {op}"
            assert op not in r["refs"], f"Operator {op} should not be in refs"

    def test_t_variable_not_in_refs(self):
        r = parse_fame_formula("result = if t ge 100 then a else b")
        assert r["type"] == "conditional"
        assert "t" not in r["refs"]

    def test_null_in_else_clause(self):
        r = parse_fame_formula("result = if a gt 0 then a else nd")
        assert r["type"] == "conditional"

    def test_nested_if(self):
        r = parse_fame_formula(
            "result = if a gt 100 then b else if a gt 50 then c else d"
        )
        assert r["type"] == "conditional"

    def test_and_or_operators(self):
        r = parse_fame_formula("r = if a gt 0 and b gt 0 then a else b")
        assert r["type"] == "conditional"

    def test_conditional_render(self):
        expr = render_conditional_expr("a >= 100", "b", "c")
        assert "pl.when(" in expr
        assert ".then(" in expr
        assert ".otherwise(" in expr

    def test_conditional_render_with_nd(self):
        expr = render_conditional_expr("a >= 100", "b", "nd")
        assert "pl.lit(None)" in expr

    def test_conditional_code_generation(self):
        cmds = ["freq m", "result = if a gt 100 then b else c"]
        _, ts = _generate_and_read(cmds)
        assert "pl.when(" in ts
        assert ".then(" in ts
        assert ".otherwise(" in ts

    def test_conditional_execution(self):
        cmds = [
            "freq m",
            "result = if x gt 5 then y else z",
        ]
        df = pl.DataFrame({
            "DATE": [None, None],
            "X": [10, 3],
            "Y": [100, 200],
            "Z": [0, 0],
        })
        result = _load_and_run(cmds, df)
        assert "RESULT" in result.columns
        vals = result["RESULT"].to_list()
        assert vals[0] == 100  # x=10 > 5, so y=100
        assert vals[1] == 0    # x=3 <= 5, so z=0


# ===================================================================
# 6. POINT-IN-TIME ASSIGNMENTS
# ===================================================================

class TestPointInTime:

    def test_parse_quoted_date(self):
        r = parse_fame_formula('gdp["2020-01-01"] = 1000')
        assert r["type"] == "point_in_time_assign"
        assert r["target"] == "gdp"
        assert r["date"] == "2020-01-01"

    def test_parse_unquoted_fame_date(self):
        r = parse_fame_formula("set a[12mar2020] = 33")
        assert r["type"] == "point_in_time_assign"
        assert r["date"] == "12mar2020"

    def test_parse_quarterly_date(self):
        r = parse_fame_formula("set cpi[2020Q1] = 105.5")
        assert r["type"] == "point_in_time_assign"
        assert r["date"] == "2020Q1"

    def test_date_indexed_refs_extract_base(self):
        """Point-in-time RHS refs should extract base variable name."""
        r = parse_fame_formula('gdp["2020-01-01"] = gdp["2019-12-31"] * 1.05')
        assert r["type"] == "point_in_time_assign"
        assert "gdp" in r["refs"]

    def test_code_generation(self):
        cmds = ["freq m", 'gdp["2020-01-01"] = 1000']
        _, ts = _generate_and_read(cmds)
        assert ".then(pl.lit(1000))" in ts
        assert '.alias("GDP")' in ts

    def test_unquoted_date_code_generation(self):
        cmds = ["freq m", "set a[12mar2020]=33"]
        _, ts = _generate_and_read(cmds)
        assert 'pl.lit("2020-03-12")' in ts
        assert ".then(pl.lit(33))" in ts

    def test_execution(self):
        from datetime import date as dt_date
        cmds = ["freq m", 'gdp["2020-01-01"] = 500']
        df = pl.DataFrame({
            "DATE": [dt_date(2020, 1, 1), dt_date(2020, 2, 1)],
        })
        result = _load_and_run(cmds, df)
        assert "GDP" in result.columns
        vals = result["GDP"].to_list()
        assert vals[0] == 500
        assert vals[1] is None


# ===================================================================
# 7. CONVERT FUNCTION
# ===================================================================

class TestConvert:

    def test_parse_convert(self):
        r = parse_fame_formula("set v23 = convert(v24, 'Q', 'M', 'AVG', 'END')")
        assert r["type"] == "convert"
        assert r["target"] == "v23"
        assert "convert_meta" in r

    def test_target_no_trailing_space(self):
        r = parse_fame_formula("set output = convert(input, Q, M)")
        assert r["type"] == "convert"
        assert r["target"] == "output"
        assert r["target"] == r["target"].strip()

    def test_convert_function_generated(self):
        cmds = ["freq m", "v1 = convert(v2, q, m, avg, end)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def CONVERT" in formulas


# ===================================================================
# 8. CHAIN / MCHAIN
# ===================================================================

class TestChain:

    def test_parse_chain(self):
        r = parse_fame_formula('set abcd = $chain("a - b - c - d", "2020")')
        assert r["type"] == "chain"
        assert r["target"] == "abcd"
        assert r["year"] == "2020"

    def test_target_no_trailing_space(self):
        r = parse_fame_formula('set result = $chain("x - y", "2021")')
        assert r["target"] == r["target"].strip()

    def test_chain_function_generated(self):
        cmds = ['set abc = $chain("a - b - c", "2020")']
        formulas, _ = _generate_and_read(cmds)
        assert "def CHAIN" in formulas

    def test_chain_code_generation(self):
        cmds = ['set abc = $chain("a - b", "2020")']
        _, ts = _generate_and_read(cmds)
        assert "CHAIN(" in ts


# ===================================================================
# 9. PCT FUNCTION
# ===================================================================

class TestPct:

    def test_parse_pct(self):
        r = parse_fame_formula("set v21 = pct(v22[t+1])")
        assert r["type"] == "simple"
        assert "pct" in r["rhs"].lower()

    def test_pct_function_generated(self):
        cmds = ["freq m", "v1 = pct(v2)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def PCT" in formulas


# ===================================================================
# 10. FISHVOL
# ===================================================================

class TestFishvol:

    def test_parse_fishvol(self):
        r = parse_fame_formula("fv = fishvol_rebase({a, b}, {c, d}, 2020)")
        assert r is not None


# ===================================================================
# 11. LSUM FUNCTION
# ===================================================================

class TestLsum:

    def test_parse_simple(self):
        r = parse_fame_formula("total = lsum(a, b, c)")
        assert r["type"] == "lsum"
        assert r["target"] == "total"
        assert "a" in [x.lower() for x in r["refs"]]

    def test_parse_with_conditionals(self):
        r = parse_fame_formula(
            "total = lsum((if a gt 0 then a else 0), (if b gt 0 then b else 0))"
        )
        assert r["type"] == "lsum"

    def test_function_generated(self):
        cmds = ["freq m", "total = lsum(a, b, c)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def LSUM" in formulas

    def test_code_generation(self):
        cmds = ["freq m", "total = lsum(a, b, c)"]
        _, ts = _generate_and_read(cmds)
        assert "lsum" in ts.lower() or "LSUM" in ts

    def test_execution(self):
        cmds = ["freq m", "total = lsum(a, b, c)"]
        df = pl.DataFrame({
            "DATE": [None], "A": [10], "B": [20], "C": [30],
        })
        result = _load_and_run(cmds, df)
        assert "TOTAL" in result.columns
        assert result["TOTAL"][0] == 60


# ===================================================================
# 12. EXISTS FUNCTION
# ===================================================================

class TestExists:

    def test_exists_function_generated(self):
        cmds = ["freq m", "check = if exists(a) then a else 0"]
        formulas, _ = _generate_and_read(cmds)
        assert "def EXISTS" in formulas

    def test_exists_rendered_as_is_not_null(self):
        expr = render_polars_expr("exists(a)")
        assert "is_not_null()" in expr


# ===================================================================
# 13. FIRSTVALUE / LASTVALUE
# ===================================================================

class TestFirstLastValue:

    def test_parse_firstvalue(self):
        r = parse_fame_formula("start = firstvalue(a)")
        assert r["type"] == "firstvalue"
        assert r["target"] == "start"
        assert r["series"] == "a"

    def test_parse_lastvalue(self):
        r = parse_fame_formula("end = lastvalue(b)")
        assert r["type"] == "lastvalue"
        assert r["target"] == "end"
        assert r["series"] == "b"

    def test_firstvalue_case_insensitive(self):
        r = parse_fame_formula("START = FIRSTVALUE(series_a)")
        assert r["type"] == "firstvalue"

    def test_lastvalue_case_insensitive(self):
        r = parse_fame_formula("END = LASTVALUE(series_b)")
        assert r["type"] == "lastvalue"

    def test_firstvalue_function_generated(self):
        cmds = ["start = firstvalue(a)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def FIRSTVALUE" in formulas
        assert "drop_nulls().first()" in formulas

    def test_lastvalue_function_generated(self):
        cmds = ["end = lastvalue(a)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def LASTVALUE" in formulas
        assert "drop_nulls().last()" in formulas

    def test_firstvalue_code_generation(self):
        cmds = ["start = firstvalue(series_a)"]
        _, ts = _generate_and_read(cmds)
        assert "FIRSTVALUE" in ts or ".first()" in ts

    def test_lastvalue_code_generation(self):
        cmds = ["end = lastvalue(series_b)"]
        _, ts = _generate_and_read(cmds)
        assert "LASTVALUE" in ts or ".last()" in ts


# ===================================================================
# 14. NLRX
# ===================================================================

class TestNlrx:

    def test_parse_nlrx(self):
        r = parse_fame_formula(
            "a_t = nlrx(lambda20, a, b1, b2, b3, b4, c, d)"
        )
        assert r["type"] == "nlrx"
        assert r["target"] == "a_t"
        assert len(r["params"]) == 8

    def test_nlrx_case_insensitive(self):
        r = parse_fame_formula(
            "A_T = NLRX(LAMBDA20, A, B1, B2, B3, B4, C, D)"
        )
        assert r["type"] == "nlrx"

    def test_nlrx_function_generated(self):
        cmds = ["r = nlrx(lam, a, b1, b2, b3, b4, c, d)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def NLRX" in formulas

    def test_nlrx_code_generation(self):
        cmds = ["r = nlrx(lam, a, b1, b2, b3, b4, c, d)"]
        _, ts = _generate_and_read(cmds)
        assert "NLRX(" in ts


# ===================================================================
# 15. SHIFT PCT
# ===================================================================

class TestShiftPct:

    def test_parse_shift_pct(self):
        r = parse_fame_formula(
            "v123s = v123s[t+1]/(1+(pct(v1014s[t+1])/100))"
        )
        assert r["type"] == "shift_pct"


# ===================================================================
# 16. LIST ALIASES
# ===================================================================

class TestListAlias:

    def test_parse_list_alias(self):
        r = parse_fame_formula("v27 = {a, b, c}")
        assert r["type"] == "list_alias"
        assert "a" in r["refs"]
        assert "b" in r["refs"]
        assert "c" in r["refs"]


# ===================================================================
# 17. SQRT FUNCTION
# ===================================================================

class TestSqrt:

    def test_sqrt_rendered(self):
        expr = render_polars_expr("sqrt(a)")
        assert ".sqrt()" in expr

    def test_sqrt_function_generated(self):
        cmds = ["freq m", "r = sqrt(a)"]
        formulas, _ = _generate_and_read(cmds)
        assert "def SQRT" in formulas


# ===================================================================
# 18. TOKEN CONVERSION
# ===================================================================

class TestTokenConversion:

    def test_nd_to_null(self):
        assert token_to_pl_expr("nd") == "pl.lit(None)"

    def test_na_to_null(self):
        assert token_to_pl_expr("na") == "pl.lit(None)"

    def test_nc_to_null(self):
        assert token_to_pl_expr("nc") == "pl.lit(None)"

    def test_t_to_date(self):
        assert token_to_pl_expr("T") == 'pl.col("DATE")'

    def test_numeric_literal(self):
        assert token_to_pl_expr("1000") == "pl.lit(1000)"

    def test_column_reference(self):
        result = token_to_pl_expr("abc")
        assert 'pl.col("ABC")' == result

    def test_time_indexed(self):
        result = token_to_pl_expr("v1[t+1]")
        assert "pl.col" in result
        assert ".shift(" in result


# ===================================================================
# 19. EXPRESSION RENDERING
# ===================================================================

class TestExpressionRendering:

    def test_simple_addition(self):
        expr = render_polars_expr("a + b")
        assert 'pl.col("A")' in expr
        assert 'pl.col("B")' in expr
        assert "+" in expr

    def test_comparison_operators(self):
        expr = render_polars_expr("a ge 100")
        assert ">=" in expr

    def test_or_operator(self):
        expr = render_polars_expr("a or b")
        assert "|" in expr

    def test_and_operator(self):
        expr = render_polars_expr("a and b")
        assert "&" in expr

    def test_if_expression(self):
        expr = render_polars_expr("if a gt 0 then b else c")
        assert "pl.when(" in expr

    def test_lsum_expression(self):
        expr = render_polars_expr("lsum(a, b, c)")
        assert "LSUM(" in expr or "lsum(" in expr


# ===================================================================
# 20. SANITIZE FUNCTION NAME
# ===================================================================

class TestSanitizeFuncName:

    def test_dots_preserved(self):
        # Dots are preserved by sanitize_func_name (replaced at upper level)
        assert sanitize_func_name("result.a") == "result.a"

    def test_dollar_sign(self):
        # Dollar sign is replaced with underscore
        assert sanitize_func_name("$abc") == "_abc"

    def test_basic_name(self):
        assert sanitize_func_name("v123") == "v123"


# ===================================================================
# 21. EXTRACT IF COMPONENTS
# ===================================================================

class TestExtractIfComponents:

    def test_simple_if(self):
        r = extract_if_components("if a > 0 then b else c")
        assert r is not None
        assert r["condition"].strip() == "a > 0"
        assert r["then_expr"].strip() == "b"
        assert r["else_expr"].strip() == "c"

    def test_nested_if(self):
        r = extract_if_components(
            "if a > 0 then b else if a > -1 then c else d"
        )
        assert r is not None
        assert "if" in r["else_expr"].lower()

    def test_no_if_returns_none(self):
        r = extract_if_components("a + b")
        assert r is None


# ===================================================================
# 22. COMPREHENSIVE END-TO-END
# ===================================================================

class TestEndToEnd:

    def test_multi_feature_pipeline(self):
        """Test a pipeline with multiple FAME features together."""
        cmds = [
            "freq m",
            "base = 100",
            "derived = base * 2",
            "cond_result = if derived gt 150 then derived else base",
        ]
        _, ts = _generate_and_read(cmds)
        # All variables should appear
        assert "BASE" in ts
        assert "DERIVED" in ts
        assert "COND_RESULT" in ts
        # Conditional pattern
        assert "pl.when(" in ts

    def test_multi_feature_execution(self):
        """Execute a generated pipeline with scalars and arithmetic."""
        cmds = [
            "freq m",
            "a = 10",
            "b = 20",
            "c = a + b",
        ]
        df = pl.DataFrame({"DATE": [None], "A": [10], "B": [20]})
        result = _load_and_run(cmds, df)
        assert "C" in result.columns

    def test_date_filter_with_scalar_execution(self):
        """Test date filter with scalar assignment executes."""
        from datetime import date as dt_date
        cmds = [
            "freq m",
            "date 2020-01-01 to 2020-12-31",
            "x = 42",
        ]
        df = pl.DataFrame({
            "DATE": [dt_date(2020, 6, 1), dt_date(2021, 6, 1)],
        })
        result = _load_and_run(cmds, df)
        assert "X" in result.columns

    def test_lsum_execution_with_nulls(self):
        """LSUM should treat nulls as 0."""
        cmds = ["freq m", "total = lsum(a, b)"]
        df = pl.DataFrame({
            "DATE": [None, None],
            "A": [10, None],
            "B": [None, 20],
        })
        result = _load_and_run(cmds, df)
        assert "TOTAL" in result.columns
        vals = result["TOTAL"].to_list()
        assert vals[0] == 10  # 10 + 0
        assert vals[1] == 20  # 0 + 20

    def test_conditional_with_nd_execution(self):
        """Conditional with nd (null) in else clause."""
        cmds = [
            "freq m",
            "result = if x gt 5 then x else nd",
        ]
        df = pl.DataFrame({
            "DATE": [None, None],
            "X": [10, 3],
        })
        result = _load_and_run(cmds, df)
        assert "RESULT" in result.columns
        vals = result["RESULT"].to_list()
        assert vals[0] == 10
        assert vals[1] is None

    def test_generated_code_compiles(self):
        """All generated code must compile without SyntaxError."""
        cmds = [
            "freq m",
            "v1 = 150",
            "v2 = 200",
            "v3 = v1 + v2",
            "result = if v3 gt 300 then v3 * 2 else v3",
        ]
        formulas, ts = _generate_and_read(cmds)
        # Both files must compile
        compile(formulas, "formulas.py", "exec")
        compile(ts, "ts_transformer.py", "exec")


# ===================================================================
# 23. CONVERT GROUPING
# ===================================================================

class TestConvertGrouping:

    def test_convert_with_frequency_metadata(self):
        r = parse_fame_formula("out = convert(inp, b, m, disc, end)")
        assert r["type"] == "convert"
        assert r["convert_meta"] is not None

    def test_polars_econ_import_generated(self):
        cmds = ["freq m", "v1 = convert(v2, q, m, avg, end)"]
        formulas, _ = _generate_and_read(cmds)
        assert "polars_econ" in formulas


# ===================================================================
# 24. DEPENDENCY ORDERING
# ===================================================================

class TestDependencyOrdering:

    def test_dependent_variables_ordered(self):
        """Variables should be computed in dependency order."""
        cmds = [
            "freq m",
            "c = a + b",
            "d = c * 2",
        ]
        _, ts = _generate_and_read(cmds)
        # C must be computed before D
        c_pos = ts.index('"C"')
        d_pos = ts.index('"D"')
        assert c_pos < d_pos


# ===================================================================
# 25. APPLY_DATE_FILTER
# ===================================================================

class TestApplyDateFilter:

    def test_function_always_generated(self):
        """APPLY_DATE_FILTER should always be available."""
        cmds = ["freq m", "v1 = 100"]
        formulas, _ = _generate_and_read(cmds)
        assert "def APPLY_DATE_FILTER" in formulas

    def test_preserve_existing_parameter(self):
        cmds = ["freq m", "v1 = 100"]
        formulas, _ = _generate_and_read(cmds)
        assert "preserve_existing" in formulas


# ===================================================================
# 26. NOT OPERATOR
# ===================================================================

class TestNotOperator:

    def test_not_rendered(self):
        expr = render_polars_expr("not a")
        assert "~" in expr


# ===================================================================
# 27. NORMALIZE FORMULA TEXT
# ===================================================================

class TestNormalizeFormula:

    def test_whitespace_collapsed(self):
        result = normalize_formula_text("  a   =   b  +  c  ")
        assert "  " not in result.strip()

    def test_preserve_content(self):
        result = normalize_formula_text("v1 = v2 + v3")
        assert "v1" in result
        assert "v2" in result
        assert "v3" in result


# ===================================================================
# 28. INLINE DATE RANGE
# ===================================================================

class TestInlineDateRange:

    def test_parse_inline_date(self):
        r = parse_fame_formula(
            "set <date 2020-01-01 to 2020-12-31> v1 = v2 + v3"
        )
        assert r is not None
        assert r["type"] == "simple"
        assert r["date_filter"]["start"] == "2020-01-01"
        assert r["date_filter"]["end"] == "2020-12-31"


# ===================================================================
# 29. MULTIPLE DATE-FILTERED ASSIGNMENTS
# ===================================================================

class TestMultipleDateFiltered:

    def test_different_date_ranges(self):
        cmds = [
            "freq bus",
            "date 01Feb2020 to 31Dec2020",
            "set a = 100",
            "date 01Jan2021 to *",
            "set a = 250",
        ]
        _, ts = _generate_and_read(cmds)
        # Both date ranges should appear
        assert "01Feb2020" in ts or "2020-02-01" in ts
        assert "01Jan2021" in ts or "2021-01-01" in ts


# ===================================================================
# 30. FORMULAS FILE STRUCTURE
# ===================================================================

class TestFormulasFileStructure:

    def test_imports_present(self):
        cmds = ["freq m", "v1 = 100"]
        formulas, _ = _generate_and_read(cmds)
        assert "import polars as pl" in formulas
        assert "from datetime import" in formulas

    def test_polars_econ_try_import(self):
        cmds = ["freq m", "v1 = convert(v2, q, m)"]
        formulas, _ = _generate_and_read(cmds)
        assert "import polars_econ" in formulas
        assert "except ImportError" in formulas


# ===================================================================
# 31. TRANSFORMER FILE STRUCTURE
# ===================================================================

class TestTransformerFileStructure:

    def test_function_signature(self):
        cmds = ["freq m", "v1 = 100"]
        _, ts = _generate_and_read(cmds)
        assert "def ts_transformer(pdf: pl.DataFrame)" in ts

    def test_returns_dataframe(self):
        cmds = ["freq m", "v1 = 100"]
        _, ts = _generate_and_read(cmds)
        assert "return pdf" in ts

    def test_imports_formulas(self):
        cmds = ["freq m", "v1 = 100"]
        _, ts = _generate_and_read(cmds)
        assert "from formulas import" in ts
