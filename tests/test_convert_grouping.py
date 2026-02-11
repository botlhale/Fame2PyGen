"""Tests for FAME convert frequency bridge grouping and code generation."""

import pytest
import tempfile
import os

from fame2pygen.formulas_generator import (
    parse_fame_formula,
    normalize_convert_frequency,
    normalize_convert_technique,
    normalize_convert_observed,
    parse_convert_args,
)
from fame2pygen.fame2py_converter import generate_test_script, generate_formulas_file


# ── Frequency normalization ──


class TestFrequencyNormalization:
    """Test FAME frequency alias resolution."""

    @pytest.mark.parametrize("alias,expected", [
        ("b", ("business", None)),
        ("bus", ("business", None)),
        ("business", ("business", None)),
        ("d", ("daily", None)),
        ("daily", ("daily", None)),
        ("w", ("weekly", None)),
        ("weekly", ("weekly", None)),
        ("m", ("monthly", None)),
        ("monthly", ("monthly", None)),
        ("mon", ("monthly", None)),
        ("q", ("quarterly", None)),
        ("quarterly", ("quarterly", None)),
        ("a", ("annual", None)),
        ("annual", ("annual", None)),
        ("y", ("annual", None)),
        ("yearly", ("annual", None)),
    ])
    def test_plain_frequency_aliases(self, alias, expected):
        assert normalize_convert_frequency(alias) == expected

    @pytest.mark.parametrize("alias,expected_day", [
        ("w(w)", "wednesday"),
        ("w(wed)", "wednesday"),
        ("w(f)", "friday"),
        ("w(fri)", "friday"),
        ("w(mon)", "monday"),
        ("w(m)", "monday"),
        ("weekly(wednesday)", "wednesday"),
        ("weekly(thu)", "thursday"),
        ("w(sun)", "sunday"),
    ])
    def test_weekly_with_day(self, alias, expected_day):
        canonical, day = normalize_convert_frequency(alias)
        assert canonical == "weekly"
        assert day == expected_day


class TestTechniqueNormalization:
    @pytest.mark.parametrize("alias,expected", [
        ("disc", "discrete"),
        ("discrete", "discrete"),
        ("linear", "linear"),
        ("lin", "linear"),
        ("cubic", "cubic"),
        ("constant", "constant"),
    ])
    def test_technique_aliases(self, alias, expected):
        assert normalize_convert_technique(alias) == expected


class TestObservedNormalization:
    @pytest.mark.parametrize("alias,expected", [
        ("ave", "average"),
        ("avg", "average"),
        ("average", "average"),
        ("sum", "sum"),
        ("first", "first"),
        ("last", "last"),
        ("hi", "high"),
        ("high", "high"),
        ("lo", "low"),
        ("end", "end"),
        ("begin", "beginning"),
    ])
    def test_observed_aliases(self, alias, expected):
        assert normalize_convert_observed(alias) == expected


# ── parse_convert_args ──


class TestParseConvertArgs:
    def test_business_daily_full(self):
        args = ["A", "b", "disc", "ave", "*", "off"]
        result = parse_convert_args(args)
        assert result["source_series"] == "A"
        assert result["target_freq_canonical"] == "business"
        assert result["suffix"] == "_BUSD"
        assert result["basis"] == "business"
        assert result["technique"] == "discrete"
        assert result["observed"] == "average"
        assert result["as_freq"] == "*"
        assert result["start_by"] is None  # 'off' means no start_by

    def test_weekly_wednesday(self):
        args = ["CC", "w(w)", "disc", "ave"]
        result = parse_convert_args(args)
        assert result["source_series"] == "CC"
        assert result["target_freq_canonical"] == "weekly"
        assert result["suffix"] == "_WK"
        assert result["technique"] == "discrete"
        assert result["observed"] == "average"
        assert result["start_by"] == "wednesday"

    def test_business_daily_long_names(self):
        args = ["DD", "b", "discrete", "average"]
        result = parse_convert_args(args)
        assert result["target_freq_canonical"] == "business"
        assert result["suffix"] == "_BUSD"
        assert result["technique"] == "discrete"
        assert result["observed"] == "average"

    def test_monthly_conversion(self):
        args = ["SERIES_A", "m", "linear", "end"]
        result = parse_convert_args(args)
        assert result["target_freq_canonical"] == "monthly"
        assert result["suffix"] == "_MON"
        assert result["to_freq"] == "1mo"

    def test_quarterly_conversion(self):
        args = ["GDP", "q", "disc", "ave"]
        result = parse_convert_args(args)
        assert result["target_freq_canonical"] == "quarterly"
        assert result["suffix"] == "_QTRLY"
        assert result["to_freq"] == "1q"

    def test_annual_conversion(self):
        args = ["CPI", "a", "disc", "ave"]
        result = parse_convert_args(args)
        assert result["target_freq_canonical"] == "annual"
        assert result["suffix"] == "_ANN"
        assert result["to_freq"] == "1y"

    def test_daily_conversion(self):
        args = ["PRICE", "d", "linear", "end"]
        result = parse_convert_args(args)
        assert result["target_freq_canonical"] == "daily"
        assert result["suffix"] == "_DD"
        assert result["to_freq"] == "1d"


# ── parse_fame_formula for convert ──


class TestConvertParsing:
    def test_convert_with_rich_metadata(self):
        result = parse_fame_formula("A = convert(A, b, disc, ave, *, off)")
        assert result["type"] == "convert"
        assert "convert_meta" in result
        meta = result["convert_meta"]
        assert meta["target_freq_canonical"] == "business"
        assert meta["technique"] == "discrete"
        assert meta["observed"] == "average"

    def test_convert_weekly_with_day(self):
        result = parse_fame_formula("CC = convert(CC, w(w), disc, ave)")
        assert result["type"] == "convert"
        meta = result["convert_meta"]
        assert meta["target_freq_canonical"] == "weekly"
        assert meta["start_by"] == "wednesday"

    def test_convert_weekly_friday(self):
        result = parse_fame_formula("X = convert(X, w(fri), disc, ave)")
        meta = result["convert_meta"]
        assert meta["target_freq_canonical"] == "weekly"
        assert meta["start_by"] == "friday"

    def test_convert_business_alias(self):
        result = parse_fame_formula("b = convert(temp, bus, dis, ave)")
        meta = result["convert_meta"]
        assert meta["target_freq_canonical"] == "business"

    def test_convert_weekly_long_form(self):
        result = parse_fame_formula("X = convert(X, WEEKLY(WED), disc, ave)")
        meta = result["convert_meta"]
        assert meta["target_freq_canonical"] == "weekly"
        assert meta["start_by"] == "wednesday"


# ── Code generation grouping ──


class TestConvertCodeGeneration:
    def _generate(self, cmds):
        """Helper: generate ts_transformer content from commands."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            ts_file = f.name
        try:
            generate_test_script(cmds, ts_file)
            with open(ts_file, "r") as f:
                return f.read()
        finally:
            os.unlink(ts_file)

    def test_grouping_business_daily_columns(self):
        """A and DD should be grouped together as business daily."""
        cmds = [
            "A = convert(A, b, disc, ave, *, off)",
            "DD = convert(DD, b, discrete, average)",
        ]
        content = self._generate(cmds)
        assert "business_cols = ['A', 'DD']" in content
        assert "_BUSD" in content
        assert "ple.convert" in content

    def test_grouping_weekly_separate(self):
        """CC should be in its own weekly group."""
        cmds = [
            "A = convert(A, b, disc, ave, *, off)",
            "CC = convert(CC, w(w), disc, ave)",
            "DD = convert(DD, b, discrete, average)",
        ]
        content = self._generate(cmds)
        # Business group
        assert "['A', 'DD']" in content
        # Weekly group
        assert "['CC']" in content
        assert "_WK" in content

    def test_business_day_filter_applied(self):
        """Business daily conversions should filter to business days."""
        cmds = ["X = convert(X, b, disc, ave)"]
        content = self._generate(cmds)
        assert ".filter(pl.col(\"DATE\").dt.is_business_day())" in content

    def test_weekly_no_business_day_filter(self):
        """Weekly conversions should NOT filter to business days."""
        cmds = ["X = convert(X, w, disc, ave)"]
        content = self._generate(cmds)
        assert "is_business_day" not in content

    def test_polars_econ_import(self):
        """polars_econ should be imported when converts are present."""
        cmds = ["X = convert(X, b, disc, ave)"]
        content = self._generate(cmds)
        assert "import polars_econ as ple" in content

    def test_no_polars_econ_without_convert(self):
        """polars_econ should NOT be imported when no converts."""
        cmds = ["X = Y + Z"]
        content = self._generate(cmds)
        assert "polars_econ" not in content

    def test_convert_ref_substitution(self):
        """Later formulas should reference suffixed column names."""
        cmds = [
            "A = convert(A, b, disc, ave)",
            "E = A + 100",
        ]
        content = self._generate(cmds)
        # E should reference A_BUSD not A
        assert 'pl.col("A_BUSD")' in content

    def test_convert_multiple_ref_substitution(self):
        """Multiple convert refs should all be substituted."""
        cmds = [
            "X = convert(X, b, disc, ave)",
            "Y = convert(Y, b, disc, ave)",
            "Z = X + Y",
        ]
        content = self._generate(cmds)
        assert 'pl.col("X_BUSD")' in content
        assert 'pl.col("Y_BUSD")' in content

    def test_convert_processed_before_levels(self):
        """Convert operations should appear before level computations."""
        cmds = [
            "A = convert(A, b, disc, ave)",
            "B = A + 1",
        ]
        content = self._generate(cmds)
        convert_pos = content.find("ple.convert")
        level_pos = content.find("Level")
        assert convert_pos < level_pos

    def test_join_and_rename_in_loop(self):
        """Generated code should have join and rename inside the loop."""
        cmds = ["X = convert(X, m, disc, ave)"]
        content = self._generate(cmds)
        assert ".rename({col_name:" in content
        assert 'pdf.join(temp_df, on="DATE", how="full")' in content
        assert '.drop("DATE_right")' in content

    def test_all_suffix_types(self):
        """Test all frequency suffixes are correctly applied."""
        cmds = [
            "A = convert(A, d, disc, ave)",
            "B = convert(B, b, disc, ave)",
            "C = convert(C, w, disc, ave)",
            "D = convert(D, m, disc, ave)",
            "E = convert(E, q, disc, ave)",
            "F = convert(F, a, disc, ave)",
        ]
        content = self._generate(cmds)
        assert "_DD" in content
        assert "_BUSD" in content
        assert "_WK" in content
        assert "_MON" in content
        assert "_QTRLY" in content
        assert "_ANN" in content

    def test_convert_start_by_kwarg(self):
        """Weekly with day spec should produce start_by kwarg."""
        cmds = ["X = convert(X, w(fri), disc, ave)"]
        content = self._generate(cmds)
        assert 'start_by="friday"' in content

    def test_convert_basis_kwarg_for_business(self):
        """Business daily should produce basis kwarg."""
        cmds = ["X = convert(X, b, disc, ave)"]
        content = self._generate(cmds)
        assert 'basis="business"' in content

    def test_drop_right_column(self):
        """Generated code should check for and drop *_right columns."""
        cmds = ["X = convert(X, b, disc, ave)"]
        content = self._generate(cmds)
        assert '_BUSD_right' in content
        assert 'pdf.drop(' in content
