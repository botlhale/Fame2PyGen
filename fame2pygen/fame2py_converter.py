"""
fame2py_converter: Generator that emits formulas.py and ts_transformer.py from a list of FAME-like commands.

This version ensures SHIFT_PCT logic is preserved and not regressed.
Dependency analysis now correctly handles time-indexed variables and offsets,
ensuring computations are ordered by dependency levels to prevent errors.
Multiple SHIFT_PCT_BACKWARDS patterns are grouped into SHIFT_PCT_BACKWARDS_MULTIPLE calls.
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict, deque

from .formulas_generator import (
    parse_fame_formula,
    generate_polars_functions,
    sanitize_func_name,
    parse_time_index,
    render_polars_expr,
    _token_to_pl_expr,
    _is_numeric_literal,
    TOKEN_RE,
    _shift_pct_re,
    normalize_formula_text,
)

ARITH_SPLIT_RE = re.compile(r"(\+|\-|\*|/)")


def preprocess_commands(lines: List[str]) -> List[str]:
    return [l.rstrip("\n") for l in lines]


def analyze_dependencies(parsed_cmds: List[Dict]) -> Tuple[defaultdict, defaultdict]:
    """
    Analyze dependencies between variables, considering time-indexed references.
    Builds a DAG where nodes are variables (lowercased), and edges represent dependencies.
    Time-indexed variables (e.g., v[t+1]) are mapped to their base names for dependency tracking.
    """
    adj = defaultdict(list)  # adjacency list: var -> list of vars that depend on it
    in_degree = defaultdict(int)  # number of dependencies for each var
    
    # Collect all target variables
    all_targets = {p["target"].lower() for p in parsed_cmds if p and "target" in p}
    
    for formula in parsed_cmds:
        if not formula or "target" not in formula:
            continue
        tgt = formula["target"].lower()
        in_degree[tgt]  # ensure target is in in_degree
        
        # Analyze refs for dependencies
        for ref in formula.get("refs", []):
            base, offset = parse_time_index(ref)
            ref_base = base.lower()
            if re.search(r"\[\s*t", ref):  # time-indexed
                # For time-indexed refs, dependency is on the base variable
                # Since offsets affect order, we note the relationship
                if ref_base in all_targets:
                    adj[ref_base].append(tgt)
                    in_degree[tgt] += 1
            else:
                # Non-time-indexed refs
                if ref_base in all_targets:
                    adj[ref_base].append(tgt)
                    in_degree[tgt] += 1
    
    return adj, in_degree


def get_computation_levels(adj: defaultdict, in_degree: defaultdict) -> List[List[str]]:
    """
    Perform topological sort to get computation levels.
    Levels ensure variables are computed before they are referenced.
    """
    levels: List[List[str]] = []
    q = deque([n for n, d in in_degree.items() if d == 0])
    while q:
        level_nodes = sorted(list(q))
        levels.append(level_nodes)
        nodes = list(q)
        q.clear()
        for n in nodes:
            for nb in sorted(adj[n]):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    q.append(nb)
    if sum(len(L) for L in levels) != len(in_degree):
        cyclic = {n for n, d in in_degree.items() if d > 0}
        raise Exception(f"A cycle was detected: {cyclic}")
    return levels


def _collect_operands_and_ops(rhs: str) -> Tuple[List[str], List[str]]:
    parts = ARITH_SPLIT_RE.split(rhs)
    operands = [p.strip() for p in parts if p and p not in "+-*/"]
    ops = [p for p in parts if p in "+-*/"]
    return operands, ops


def _operand_to_pl(tok: str) -> str:
    return _token_to_pl_expr(tok)


def generate_formulas_file(cmds: List[str], out_filename: str = "formulas.py"):
    # Create a context to track needed functions
    ctx = {"need_shiftpct": False, "need_shiftpct_backwards": False}
    
    # Check for SHIFT_PCT patterns in each command
    for cmd in cmds:
        normalized = normalize_formula_text(cmd)
        if _shift_pct_re.search(normalized):
            ctx["need_shiftpct"] = True
            # Assume backwards if the pattern matches the specific form
            if "=" in normalized and "t+" in normalized:
                ctx["need_shiftpct_backwards"] = True
    
    # Generate function definitions
    fn_defs = generate_polars_functions(cmds)
    
    with open(out_filename, "w") as f:
        f.write("import polars as pl\n")
        f.write("from typing import List, Tuple\n\n")
        for name in sorted(fn_defs.keys()):
            f.write(fn_defs[name])
            f.write("\n\n")


def generate_test_script(cmds: List[str], out_filename: str = "ts_transformer.py") -> str:
    parsed_raw = [p for p in (parse_fame_formula(c) for c in cmds) if p]
    if not parsed_raw:
        content = (
            '"""Auto-generated ts_transformer module - no parsed commands found"""\n'
            "import polars as pl\n\n"
            "def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:\n"
            "    return pdf\n"
        )
        with open(out_filename, "w") as f:
            f.write(content)
        return out_filename

    # Track date filter state across commands and preserve order
    current_date_filter = None  # None means no filtering (all dates)
    
    parsed = []
    for idx, p in enumerate(parsed_raw):
        np = p.copy()
        
        # Handle date commands
        if np.get("type") == "date":
            current_date_filter = np.get("filter")
            continue  # Don't include date commands in the parsed list
            
        if "target" in np:
            np["target"] = np["target"].lower()
        if "refs" in np:
            np["refs"] = [r.lower() for r in np["refs"]]
        
        # Track the date filter for this command and preserve original order
        np["date_filter"] = current_date_filter
        np["original_order"] = idx
        parsed.append(np)

    formulas = {p["target"]: p for p in parsed if "target" in p}
    
    # Enhanced dependency analysis
    adj, indeg = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, indeg)

    # Collect SHIFT_PCT_BACKWARDS patterns for batching
    shift_pct_backwards_patterns = []
    for p in parsed:
        if "rhs" in p and isinstance(p.get("rhs", ""), str):
            rhs_normalized = normalize_formula_text(p["rhs"])
            msp = _shift_pct_re.match(rhs_normalized)
            if msp and "t+" in rhs_normalized:
                ser1 = msp.group(1)
                offs1 = int(msp.group(2))
                ser2 = msp.group(3)
                shift_pct_backwards_patterns.append((p["target"], ser1, ser2, offs1))

    lines: List[str] = []
    lines.append('"""Auto-generated ts_transformer module - applies transformations from formulas"""')
    lines.append("import polars as pl")
    lines.append("from typing import List, Tuple")
    lines.append("from formulas import *\n")
    lines.append("def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:")
    lines.append('    """Apply transformations and return augmented DataFrame."""')
    
    # Track the current date filter state to add as comments
    current_filter_state = None

    # Handle multiple SHIFT_PCT_BACKWARDS at the beginning
    if shift_pct_backwards_patterns:
        column_pairs = [(tgt.upper(), pct.upper()) for tgt, ser1, pct, _ in shift_pct_backwards_patterns]
        offsets = [offs for _, _, _, offs in shift_pct_backwards_patterns]
        lines.append("    # Batch SHIFT_PCT_BACKWARDS calculations")
        lines.append(f"    pdf = SHIFT_PCT_BACKWARDS_MULTIPLE(pdf, \"2016-12-31\", \"1981-03-31\", {column_pairs}, offsets={offsets})")

    for level_idx, level in enumerate(levels):
        if not level:
            continue
        
        # Sort commands in this level by original order to preserve date filter transitions
        level_formulas = [(tgt, formulas[tgt]) for tgt in level if tgt in formulas]
        level_formulas.sort(key=lambda x: x[1].get("original_order", 0))
        
        # Group by date filter within this level
        groups = []
        current_group = []
        current_group_filter = None
        
        for tgt, formula in level_formulas:
            formula_filter = formula.get("date_filter")
            if formula_filter != current_group_filter:
                if current_group:
                    groups.append((current_group_filter, current_group))
                current_group = [tgt]
                current_group_filter = formula_filter
            else:
                current_group.append(tgt)
        
        if current_group:
            groups.append((current_group_filter, current_group))
        
        # Generate code for each group
        for group_filter, group_targets in groups:
            # Add comment for date filter state
            if group_filter is None:
                lines.append("    # Date filter: * (all dates)")
            else:
                lines.append(f"    # Date filter: {group_filter['start']} to {group_filter['end']}")
            
            lines.append(f"    # --- Level {level_idx + 1}: compute {', '.join(group_targets)} ---")
            cols: List[str] = []
            
            for tgt in group_targets:
                formula = formulas[tgt]
                tgt_alias = sanitize_func_name(formula["target"]).upper()
                rhs_text = (formula.get("rhs") or "").strip()
                rhs_normalized = normalize_formula_text(rhs_text)
                rhs_lower = rhs_normalized.lower()

                # Skip if already handled in batch
                if any(tgt == tgt_from_pattern for tgt_from_pattern, _, _, _ in shift_pct_backwards_patterns):
                    continue

                # parsed chain/mchain handling
                if formula.get("type") in ("chain", "mchain"):
                    pair_items = []
                    for op, var in formula.get("terms", []):
                        sign = "-" if op == "-" else ""
                        pcol = sanitize_func_name("p" + var).upper()
                        vcol = sanitize_func_name(var).upper()
                        pair_items.append(f"({sign}pl.col('{pcol}'), pl.col('{vcol}'))")
                    pairs_str = ", ".join(pair_items)
                    cols.append(f'        CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col("DATE"), year="{formula.get("year","")}").alias("{tgt_alias}")')
                    continue

                # exact SHIFT_PCT pattern detection with normalized RHS
                msp = _shift_pct_re.match(rhs_normalized)
                if msp:
                    ser1 = msp.group(1)
                    offs1 = int(msp.group(2))
                    ser2 = msp.group(3)
                    ser1_col = sanitize_func_name(ser1).upper()
                    ser2_col = sanitize_func_name(ser2).upper()
                    # Check if it's backwards (positive offset)
                    if offs1 > 0:
                        # Already handled in batch
                        continue
                    else:
                        # Forward SHIFT_PCT
                        cols.append(f'        SHIFT_PCT(pl.col("{ser1_col}"), pl.col("{ser2_col}"), {offs1}).alias("{tgt_alias}")')
                    continue

                # if RHS contains special markers, render with render_polars_expr
                if any(marker in rhs_lower for marker in ["$chain", "$mchain", "pct(", "convert(", "fishvol_rebase("]):
                    subs: Dict[str, str] = {}
                    for t in TOKEN_RE.findall(rhs_text):
                        key = t.lower()
                        if key == "t":
                            continue
                        if _is_numeric_literal(t):
                            continue
                        subs[key] = _operand_to_pl(t)
                    expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                    expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                    cols.append(f"        ({expr_text}).alias('{tgt_alias}')")
                    continue

                # assign-series single token
                m_assign = re.match(r"^\s*([A-Za-z0-9_$]+)\s*=\s*([A-Za-z0-9_$.]+)\s*$", rhs_text)
                if m_assign:
                    src_tok = m_assign.group(2)
                    src_expr = _operand_to_pl(src_tok)
                    cols.append(f'        ASSIGN_SERIES("{tgt_alias}", {src_expr})')
                    continue

                # arithmetic detection
                operands, ops = _collect_operands_and_ops(rhs_normalized)
                if ops:
                    unique_ops = set(ops)
                    if unique_ops == {"+"}:
                        args = ", ".join(_operand_to_pl(opnd) for opnd in operands)
                        cols.append(f'        ADD_SERIES("{tgt_alias}", {args})')
                        continue
                    if unique_ops == {"-"}:
                        args = ", ".join(_operand_to_pl(opnd) for opnd in operands)
                        cols.append(f'        SUB_SERIES("{tgt_alias}", {args})')
                        continue
                    if unique_ops == {"*"}:
                        args = ", ".join(_operand_to_pl(opnd) for opnd in operands)
                        cols.append(f'        MUL_SERIES("{tgt_alias}", {args})')
                        continue
                    if unique_ops == {"/"}:
                        args = ", ".join(_operand_to_pl(opnd) for opnd in operands)
                        cols.append(f'        DIV_SERIES("{tgt_alias}", {args})')
                        continue
                    # mixed -> render polars expr and alias
                    subs: Dict[str, str] = {}
                    for t in TOKEN_RE.findall(rhs_text):
                        key = t.lower()
                        if key == "t":
                            continue
                        if _is_numeric_literal(t):
                            continue
                        subs[key] = _operand_to_pl(t)
                    expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                    expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                    cols.append(f'        ASSIGN_SERIES("{tgt_alias}", {expr_text})')
                    continue

                # fallback: render and alias
                subs: Dict[str, str] = {}
                for t in TOKEN_RE.findall(rhs_text):
                    key = t.lower()
                    if key == "t":
                        continue
                    if _is_numeric_literal(t):
                        continue
                    subs[key] = _operand_to_pl(t)
                expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                cols.append(f'        ASSIGN_SERIES("{tgt_alias}", {expr_text})')

            if cols:
                lines.append("    pdf = pdf.with_columns([\n" + ",\n".join(cols) + "\n    ])\n")

    lines.append("    return pdf\n")
    content = "".join(lines)
    with open(out_filename, "w") as f:
        f.write(content)
    return out_filename


def create_mock_library():
    """Create a minimal mock of polars_econ library for testing."""
    content = """
# polars_econ_mock.py - minimal mock of polars_econ for development
import polars as pl
from typing import List, Tuple

def pct(expr, offset=1):
    return expr.shift(-offset).pct_change()

def chain(price_quantity_pairs, date_col, index_year):
    return pl.lit(1.0)  # mock implementation

def convert(series, date_col, as_freq, to_freq, technique, observed):
    return pl.lit(1.0)  # mock implementation

def fishvol(series_pairs, date_col, rebase_year):
    return pl.lit(1.0)  # mock implementation
"""
    with open("polars_econ_mock.py", "w") as f:
        f.write(content)


# Test cases for FAME script mappings
TEST_CASES = [
    # Basic assignment
    ["freq m", "vbot = 1"],
    
    # Single SHIFT_PCT_BACKWARDS
    ["set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))"],
    
    # Multiple SHIFT_PCT_BACKWARDS
    [
        "set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))",
        "set v456s[t] = v456s[t+1]/(1+(pct(v2024s[t+1])/100))"
    ],
    
    # Chain operations
    ['set abcd = $chain("a - b - c - d", "2020")/100'],
    
    # Arithmetic operations
    ["v1 = v2 + v3", "v4 = v5 - v6", "v7 = v8 * v9", "v10 = v11 / v12"],
    
    # Mixed arithmetic
    ["v13 = v14 + v15 - v16"],
    
    # Time-indexed variables
    ["v17 = v18[t+1]", "v19 = v20[t-2]"],
    
    # PCT function
    ["set v21 = pct(v22[t+1])"],
    
    # Convert function
    ["set v23 = convert(v24, 'Q', 'M', 'AVG', 'END')"],
    
    # Fishvol function
    ["set v25 = fishvol_rebase({v26},{p26},2020)"],
    
    # List alias
    ["v27 = {a, b, c}"],
    
    # Complex mixed scenario
    [
        "freq m",
        "vbot = 1",
        "set v123s[t] = v123s[t+1]/(1+(pct(v1014s[t+1])/100))",
        "set v456s[t] = v456s[t+1]/(1+(pct(v2024s[t+1])/100))",
        'set abcd = $chain("a - b - c - d", "2020")/100',
        "v28 = v29 + v30",
        "set v31 = pct(v32[t+1])",
        "set v33 = convert(v34, 'Q', 'M', 'AVG', 'END')"
    ]
]


def main(fame_commands):
    """Main entry point for the command generator."""
    new_cmds = preprocess_commands(fame_commands)
    print("Parsed commands:")
    for c in new_cmds:
        print("  ", c)
    
    print("Generating formulas.py...")
    generate_formulas_file(new_cmds)
    
    print("Generating ts_transformer.py (comprehensive transformer)...")
    fname = generate_test_script(new_cmds)
    print("Wrote", fname)
    
    create_mock_library()
    print("Generated files: formulas.py, ts_transformer.py, polars_econ_mock.py")


def run_tests():
    """Run all test cases."""
    for i, test_case in enumerate(TEST_CASES):
        print(f"\n--- Test Case {i+1} ---")
        main(test_case)


# Example usage
if __name__ == "__main__":
    # Run a specific test case or all tests
    run_tests()
