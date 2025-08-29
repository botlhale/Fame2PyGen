from fame2pygen.parser import parse_script
from fame2pygen.generators import build_generation_context, generate_formulas_module, generate_pipeline_script

def test_generation_roundtrip():
    script = [
        "freq m",
        "a$=v123*12",
        "a=v143*12",
        "aa=a$/a"
    ]
    parsed = parse_script(script)
    ctx = build_generation_context(parsed, script)
    formulas = generate_formulas_module(ctx)
    assert "def AA(" in formulas or "def AA()" in formulas
    pipeline = generate_pipeline_script(ctx, "cformulas")
    assert "final_result" in pipeline