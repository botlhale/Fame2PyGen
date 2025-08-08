import re
import write_formulagen as gen
import formulagen


def test_fishvol_partition_and_filtering():
    fame_script = """
freq m
a=v143*12
pa=v123*3
vol_idx=$fishvol_rebase({a},{pa},2020)
z = a + 1
z2 = vol_idx * 2
""".strip()

    parsed, _ = gen.preprocess_commands(fame_script)
    code = gen.generate_pipeline_code(parsed)

    # Main df should not be filtered by date
    assert "df.filter(pl.col('date')" not in "\n".join([l for l in code.splitlines() if l.startswith("df = df")])

    # There should be a fishvol sub-pipeline filtered to 2020-01-01
    assert "FISHVOL SUB-PIPELINE" in code
    assert "df_fv_vol_idx = df.filter(pl.col('date') >= pl.date(2020, 1, 1))" in code

    # Ensure that the fishvol closure receives with_columns on its own df_fv_...
    assert re.search(r"df_fv_vol_idx\s*=\s*df_fv_vol_idx\.with_columns", code)


def test_convert_partition_separate_df():
    fame_script = """
freq m
a=v143*12
q_a=convert(a,q,disc,ave)
""".strip()

    parsed, _ = gen.preprocess_commands(fame_script)
    code = gen.generate_pipeline_code(parsed)

    # Main df should exist; conversions should be in df_cv_q
    assert "# ---- CONVERT SUB-PIPELINE to 'q' ----" in code
    assert "df_cv_q = None" in code
    assert "group_by_dynamic('date', every='1q')" in code

    # Main df lines should not include conversion aggregations
    main_section = code.split("# ---- CONVERT SUB-PIPELINE")[0]
    assert "group_by_dynamic(" not in main_section


def test_formulas_generation_no_alias_and_names_match():
    fame_script = """
a=v143*12
b$=a*2
""".strip()

    parsed, _ = gen.preprocess_commands(fame_script)
    src = gen.generate_formulas_py_string(parsed)

    # Functions should be named A and B_
    assert re.search(r"def A\(\) -> pl\.Expr:", src)
    assert re.search(r"def B_\(\) -> pl\.Expr:", src)

    # No aliasing inside formula functions
    assert ".alias(" not in src

    # References should use original column tokens (e.g., 'a$' remains 'a$')
    # In this case, b$ depends on 'a' -> pl.col('a')
    assert "return (pl.col(\"v143\")*12)" in src or "return (pl.col('v143')*12)" in src
    assert "return (pl.col('a')*2)" in src or "return (pl.col(\"a\")*2)" in src