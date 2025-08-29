from fame2pygen.parser import parse_script

def test_parse_basic():
    script = [
        "freq m",
        "a$=v123*12",
        "aa=a$/a",
        "abc=$mchain(\"a$ + a + bb\",\"2025\")"
    ]
    parsed = parse_script(script)
    assert any(p.type == "freq" for p in parsed)
    assert any(p.type == "simple" and p.target.lower() == "a$".lower() for p in parsed)
    assert any(p.type == "mchain" for p in parsed)