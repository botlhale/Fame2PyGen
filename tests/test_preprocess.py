from fame2pygen.preprocess import preprocess_lines

def test_loop_unroll_basic():
    lines = [
        "monthly_src = {g1, g2}",
        "quarterly_tgt = {q1, q2}",
        "loop for %i = 1 to length(monthly_src)",
        "    loop for %ms in { extract(monthly_src, %i) }",
        "        loop for %qs in { extract(quarterly_tgt, %i) }",
        "            set %qs = convert(%ms, q, ave, end)",
        "        end loop",
        "    end loop",
        "end loop",
    ]
    processed = preprocess_lines(lines)
    # Expect unrolled assignments (without 'set') and no loop markers
    assert any(p.startswith("q1 = convert(g1") for p in processed)
    assert any(p.startswith("q2 = convert(g2") for p in processed)
    assert not any("loop for" in p for p in processed)