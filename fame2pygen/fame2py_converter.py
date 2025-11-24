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
    parse_date_index,
    render_polars_expr,
    render_conditional_expr,
    _token_to_pl_expr,
    _is_numeric_literal,
    _is_strict_number,
    TOKEN_RE,
    _shift_pct_re,
    normalize_formula_text,
    FUNCTION_NAMES,
    convert_fame_date_to_iso,
)

ARITH_SPLIT_RE = re.compile(r"(\+|\-|\*|/)")

# Additional function names used in conditional expressions for date functions
CONDITIONAL_FUNCTION_NAMES = FUNCTION_NAMES | {"dateof", "make", "date", "contain", "end"}


def preprocess_commands(lines: List[str]) -> List[str]:
    return [l.rstrip("\n") for l in lines]


def analyze_dependencies(parsed_cmds: List[Dict]) -> Tuple[defaultdict, defaultdict]:
    """
    Analyze dependencies between variables, considering time-indexed references.
    Builds a DAG where nodes are variables (lowercased), and edges represent dependencies.
    Time-indexed variables (e.g., v[t+1]) are mapped to their base names for dependency tracking.
    SHIFT_PCT patterns with self-references (e.g., v[t] = v[t+1]/...) are excluded from the
    dependency graph as they are handled specially.
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
        
        # Skip SHIFT_PCT patterns as they're handled specially (not part of regular dependency chain)
        if formula.get("type") == "shift_pct":
            continue
        
        # Analyze refs for dependencies
        for ref in formula.get("refs", []):
            base, offset = parse_time_index(ref)
            ref_base = base.lower()
            # Skip self-references for time-indexed variables
            if ref_base == tgt:
                continue
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

    # Build formulas dict - for point-in-time assignments and date-filtered assignments,
    # use target+identifier as key to avoid overwriting multiple assignments to the same variable
    formulas = {}
    for p in parsed:
        if "target" in p:
            if p.get("type") == "point_in_time_assign":
                # Use target+date as key for uniqueness
                key = f"{p['target']}@{p.get('date', '')}"
                formulas[key] = p
            elif p.get("date_filter") is not None:
                # Use target+date_filter as key for date-filtered assignments
                # to preserve multiple assignments to the same variable with different date ranges
                date_filter = p["date_filter"]
                date_key = f"{date_filter.get('start', '*')}_to_{date_filter.get('end', '*')}"
                key = f"{p['target']}@datefilter_{date_key}_{p.get('original_order', 0)}"
                formulas[key] = p
            else:
                formulas[p["target"]] = p
    
    # Enhanced dependency analysis
    adj, indeg = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, indeg)

    # Collect SHIFT_PCT_BACKWARDS patterns for batching
    shift_pct_backwards_patterns = []
    for p in parsed:
        # Check if it's a shift_pct type with positive offset (backwards pattern)
        if p.get("type") == "shift_pct":
            offset = p.get("offset", 0)
            if offset > 0:
                ser1 = p.get("ser1")
                ser2 = p.get("ser2")
                target = p.get("target")
                shift_pct_backwards_patterns.append((target, ser1, ser2, offset))

    lines: List[str] = []
    lines.append('"""Auto-generated ts_transformer module - applies transformations from formulas"""\n')
    lines.append("import polars as pl\n")
    lines.append("from typing import List, Tuple\n")
    lines.append("from datetime import date\n")
    lines.append("from formulas import *\n")
    lines.append("\n")
    lines.append("def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:\n")
    lines.append('    """Apply transformations and return augmented DataFrame."""\n')
    
    # Track the current date filter state to add as comments
    current_filter_state = None
    
    # Track columns that have been assigned to determine if we need to preserve existing values
    assigned_columns = set()

    # Collect point-in-time assignments grouped by target variable
    point_in_time_by_target = defaultdict(list)
    for p in parsed:
        if p.get("type") == "point_in_time_assign":
            target = p.get("target")
            point_in_time_by_target[target].append(p)

    # Handle multiple SHIFT_PCT_BACKWARDS at the beginning
    if shift_pct_backwards_patterns:
        column_pairs = [(tgt.upper(), pct.upper()) for tgt, ser1, pct, _ in shift_pct_backwards_patterns]
        offsets = [offs for _, _, _, offs in shift_pct_backwards_patterns]
        lines.append("    # Batch SHIFT_PCT_BACKWARDS calculations\n")
        lines.append(f"    pdf = SHIFT_PCT_BACKWARDS_MULTIPLE(pdf, \"2016-12-31\", \"1981-03-31\", {column_pairs}, offsets={offsets})\n")

    for level_idx, level in enumerate(levels):
        if not level:
            continue
        
        # Sort commands in this level by original order to preserve date filter transitions
        # For point-in-time assignments, find all formulas keys that start with target name
        level_formulas = []
        for tgt in level:
            # Check for exact match (regular assignment)
            if tgt in formulas:
                level_formulas.append((tgt, formulas[tgt]))
            # Check for point-in-time assignments (key format: target@date)
            for key, formula in formulas.items():
                if '@' in key and key.startswith(tgt + '@'):
                    level_formulas.append((key, formula))
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
                lines.append("    # Date filter: * (all dates)\n")
            else:
                lines.append(f"    # Date filter: {group_filter['start']} to {group_filter['end']}\n")
            
            lines.append(f"    # --- Level {level_idx + 1}: compute {', '.join(group_targets)} ---\n")
            cols: List[str] = []
            
            # Helper function to wrap expression with date filter if needed
            def wrap_with_date_filter(expr: str, target_col: str, preserve_existing: bool = False) -> str:
                """Wrap expression with APPLY_DATE_FILTER if group has date filter."""
                if group_filter is not None:
                    start = group_filter['start']
                    end = group_filter['end']
                    # Include preserve_existing parameter if True
                    if preserve_existing:
                        return f'APPLY_DATE_FILTER({expr}, "{target_col}", "{start}", "{end}", preserve_existing=True)'
                    else:
                        return f'APPLY_DATE_FILTER({expr}, "{target_col}", "{start}", "{end}")'
                return expr
            
            for tgt in group_targets:
                formula = formulas[tgt]
                tgt_alias = sanitize_func_name(formula["target"]).upper()
                
                # Check if this column has been assigned before to determine preserve_existing
                preserve_existing = tgt_alias in assigned_columns
                
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
                    expr = f'CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col("DATE"), year="{formula.get("year","")}")'
                    wrapped = wrap_with_date_filter(expr, tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue

                # exact SHIFT_PCT pattern detection
                if formula.get("type") == "shift_pct":
                    ser1 = formula.get("ser1")
                    ser2 = formula.get("ser2")
                    offs1 = formula.get("offset", 0)
                    ser1_col = sanitize_func_name(ser1).upper()
                    ser2_col = sanitize_func_name(ser2).upper()
                    # Check if it's backwards (positive offset)
                    if offs1 > 0:
                        # Already handled in batch
                        continue
                    else:
                        # Forward SHIFT_PCT
                        expr = f'SHIFT_PCT(pl.col("{ser1_col}"), pl.col("{ser2_col}"), {offs1})'
                        wrapped = wrap_with_date_filter(expr, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                        assigned_columns.add(tgt_alias)
                    continue

                # point-in-time assignment handling - skip here, will be handled after all levels
                if formula.get("type") == "point_in_time_assign":
                    continue

                # conditional expression handling
                if formula.get("type") == "conditional":
                    condition = formula.get("condition", "")
                    then_expr = formula.get("then_expr", "")
                    else_expr = formula.get("else_expr", "")
                    
                    # Build substitution map for all expressions
                    subs: Dict[str, str] = {}
                    # Collect all tokens from condition, then_expr, and else_expr
                    all_text = f"{condition} {then_expr} {else_expr}"
                    for t in TOKEN_RE.finditer(all_text):
                        tok = t.group(0)
                        key = tok.lower()
                        # In conditionals, 't' represents the current row's date (time variable in FAME)
                        # It should be mapped to pl.col("DATE"), not skipped or treated as regular column
                        if key == "t":
                            # Check if this is part of a time index like v[t+1]
                            # If so, skip it (it's handled in parse_time_index)
                            # Otherwise, map it to DATE column
                            if not re.search(r'\[\s*t\s*[+-]?\d*\s*\]', tok):
                                # Standalone t in conditional represents current date
                                subs[key] = 'pl.col("DATE")'
                            continue
                        if _is_numeric_literal(tok):
                            continue
                        if key in CONDITIONAL_FUNCTION_NAMES:
                            continue
                        if key in ('if', 'then', 'else', 'and', 'or', 'not', 'ge', 'gt', 'le', 'lt', 'eq', 'ne', 'nd'):
                            continue
                        subs[key] = _operand_to_pl(tok)
                    
                    # Render the conditional expression
                    expr_text = render_conditional_expr(condition, then_expr, else_expr, substitution_map=subs)
                    wrapped = wrap_with_date_filter(expr_text, tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue

                # convert function handling
                if formula.get("type") == "convert":
                    params = formula.get("params", [])
                    if len(params) >= 4:
                        # First param is the series - convert to uppercase column reference
                        series_col = sanitize_func_name(params[0]).upper()
                        # Build CONVERT call with parameters
                        # CONVERT expects: series (DataFrame subset), as_freq, to_freq, technique, observed
                        # The first param should be a DataFrame with the series column
                        params_str = ", ".join([f'"{p}"' if i > 0 else f'pdf.select(pl.col("{series_col}"))' for i, p in enumerate(params)])
                        expr_text = f'CONVERT({params_str})'
                        wrapped = wrap_with_date_filter(expr_text, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                        assigned_columns.add(tgt_alias)
                        continue
                
                # firstvalue function handling
                if formula.get("type") == "firstvalue":
                    series = formula.get("series", "")
                    series_col = sanitize_func_name(series).upper()
                    expr_text = f'FIRSTVALUE(pl.col("{series_col}"))'
                    # firstvalue returns a scalar, so we need to broadcast it
                    wrapped = wrap_with_date_filter(f'pl.lit({expr_text})', tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue
                
                # lastvalue function handling
                if formula.get("type") == "lastvalue":
                    series = formula.get("series", "")
                    series_col = sanitize_func_name(series).upper()
                    expr_text = f'LASTVALUE(pl.col("{series_col}"))'
                    # lastvalue returns a scalar, so we need to broadcast it
                    wrapped = wrap_with_date_filter(f'pl.lit({expr_text})', tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue
                
                # nlrx function handling
                if formula.get("type") == "nlrx":
                    params = formula.get("params", [])
                    if len(params) >= 8:
                        # Parameters: lambda, y, w1, w2, w3, w4, gss, gpr
                        # Convert parameter names to column expressions
                        lamb_expr = _operand_to_pl(params[0])
                        y_expr = _operand_to_pl(params[1])
                        w1_expr = _operand_to_pl(params[2])
                        w2_expr = _operand_to_pl(params[3])
                        w3_expr = _operand_to_pl(params[4])
                        w4_expr = _operand_to_pl(params[5])
                        gss_expr = _operand_to_pl(params[6])
                        gpr_expr = _operand_to_pl(params[7])
                        
                        # If there are pending columns to add, flush them first
                        if cols:
                            lines.append("    pdf = pdf.with_columns([\n" + ",\n".join(cols) + "\n    ])\n")
                            cols.clear()
                        
                        # NLRX returns a DataFrame, so we assign it back to pdf
                        # The result should be stored in a column with the target name
                        expr_text = f'NLRX(pdf, {lamb_expr}, {y_expr}, {w1_expr}, {w2_expr}, {w3_expr}, {w4_expr}, {gss_expr}, {gpr_expr})'
                        # Since NLRX returns a DataFrame, we need to handle it differently
                        # We'll emit a direct assignment that reassigns pdf
                        lines.append(f"    # NLRX function call for {tgt_alias}\n")
                        lines.append(f'    pdf = {expr_text}\n')
                        assigned_columns.add(tgt_alias)
                        continue

                # if RHS contains special markers, render with render_polars_expr
                if any(marker in rhs_lower for marker in ["$chain", "$mchain", "pct(", "convert(", "fishvol_rebase(", "sqrt("]):
                    subs: Dict[str, str] = {}
                    # Build substitution map, but skip function names
                    for t in TOKEN_RE.findall(rhs_text):
                        key = t.lower()
                        if key == "t":
                            continue
                        if _is_numeric_literal(t):
                            continue
                        if key in FUNCTION_NAMES:
                            continue  # Skip function names - they'll be handled by render_polars_expr
                        subs[key] = _operand_to_pl(t)
                    expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                    expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                    wrapped = wrap_with_date_filter(f"({expr_text})", tgt_alias, preserve_existing)
                    cols.append(f"        {wrapped}.alias('{tgt_alias}')")
                    assigned_columns.add(tgt_alias)
                    continue

                # assign-series single token (including time-indexed tokens like v2[t+1])
                # Check if RHS is a single token (possibly with time index)
                m_assign = re.match(r"^\s*([A-Za-z0-9_$.]+(?:\s*\[\s*t\s*[+-]?\d*\s*\])?)\s*$", rhs_text)
                if m_assign:
                    src_tok = m_assign.group(1)
                    src_expr = _operand_to_pl(src_tok)
                    # If it's a bare number, wrap it in pl.lit()
                    if _is_numeric_literal(src_tok):
                        src_expr = f"pl.lit({src_expr})"
                    wrapped = wrap_with_date_filter(src_expr, tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue

                # arithmetic detection
                operands, ops = _collect_operands_and_ops(rhs_normalized)
                if ops:
                    unique_ops = set(ops)
                    
                    # Helper to wrap numeric literals in pl.lit()
                    def wrap_operand(opnd: str) -> str:
                        expr = _operand_to_pl(opnd)
                        if _is_numeric_literal(opnd):
                            return f'pl.lit({expr})'
                        return expr
                    
                    if unique_ops == {"+"}:
                        args_list = [wrap_operand(opnd) for opnd in operands]
                        args_str = ', '.join(args_list)
                        base_expr = f'ADD_SERIES("{tgt_alias}", {args_str})'
                        wrapped = wrap_with_date_filter(base_expr, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}')
                        assigned_columns.add(tgt_alias)
                        continue
                    if unique_ops == {"-"}:
                        args_list = [wrap_operand(opnd) for opnd in operands]
                        args_str = ', '.join(args_list)
                        base_expr = f'SUB_SERIES("{tgt_alias}", {args_str})'
                        wrapped = wrap_with_date_filter(base_expr, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}')
                        assigned_columns.add(tgt_alias)
                        continue
                    if unique_ops == {"*"}:
                        args_list = [wrap_operand(opnd) for opnd in operands]
                        args_str = ', '.join(args_list)
                        base_expr = f'MUL_SERIES("{tgt_alias}", {args_str})'
                        wrapped = wrap_with_date_filter(base_expr, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}')
                        assigned_columns.add(tgt_alias)
                        continue
                    if unique_ops == {"/"}:
                        args_list = [wrap_operand(opnd) for opnd in operands]
                        args_str = ', '.join(args_list)
                        base_expr = f'DIV_SERIES("{tgt_alias}", {args_str})'
                        wrapped = wrap_with_date_filter(base_expr, tgt_alias, preserve_existing)
                        cols.append(f'        {wrapped}')
                        assigned_columns.add(tgt_alias)
                        continue
                    # mixed -> render polars expr and alias
                    subs: Dict[str, str] = {}
                    for t in TOKEN_RE.findall(rhs_text):
                        key = t.lower()
                        if key == "t":
                            continue
                        if _is_numeric_literal(t):
                            continue
                        if key in FUNCTION_NAMES:
                            continue
                        subs[key] = _operand_to_pl(t)
                    expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                    expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                    wrapped = wrap_with_date_filter(expr_text, tgt_alias, preserve_existing)
                    cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                    assigned_columns.add(tgt_alias)
                    continue

                # fallback: render and alias
                subs: Dict[str, str] = {}
                for t in TOKEN_RE.findall(rhs_text):
                    key = t.lower()
                    if key == "t":
                        continue
                    if _is_numeric_literal(t):
                        continue
                    if key in FUNCTION_NAMES:
                        continue
                    subs[key] = _operand_to_pl(t)
                expr_text = render_polars_expr(rhs_text, substitution_map=subs, memory=None, ctx=None)
                expr_text = re.sub(r"\)\s*/\s*100\b", ").truediv(100)", expr_text)
                wrapped = wrap_with_date_filter(expr_text, tgt_alias, preserve_existing)
                cols.append(f'        {wrapped}.alias("{tgt_alias}")')
                assigned_columns.add(tgt_alias)

            if cols:
                lines.append("    pdf = pdf.with_columns([\n" + ",\n".join(cols) + "\n    ])\n")

    # Process point-in-time assignments grouped by target variable
    # Generate chained when/then expressions for each target
    for target, assignments in sorted(point_in_time_by_target.items()):
        if not assignments:
            continue
        
        # Sort assignments by original order
        assignments.sort(key=lambda x: x.get("original_order", 0))
        
        tgt_alias = sanitize_func_name(target).upper()
        lines.append(f"    # Point-in-time assignments for {tgt_alias}\n")
        
        # Build chained when/then expression
        when_conditions = []
        for assignment in assignments:
            date_str = assignment.get("date", "")
            rhs = assignment.get("rhs", "").strip()
            
            # Convert FAME date to ISO format
            iso_date = convert_fame_date_to_iso(date_str)
            
            # Generate the value expression
            if _is_strict_number(rhs):
                value_expr = f"pl.lit({rhs})"
            else:
                # Build substitution map for RHS
                subs: Dict[str, str] = {}
                for t in TOKEN_RE.finditer(rhs):
                    tok = t.group(0)
                    key = tok.lower()
                    if key == "t":
                        continue
                    if _is_numeric_literal(tok):
                        continue
                    if key in FUNCTION_NAMES:
                        continue
                    subs[key] = _operand_to_pl(tok)
                value_expr = render_polars_expr(rhs, substitution_map=subs, memory=None, ctx=None)
            
            when_conditions.append((iso_date, value_expr))
        
        # Generate the chained when/then expression
        # Start with existing value if column exists, otherwise start fresh
        chain_parts = []
        for idx, (iso_date, value_expr) in enumerate(when_conditions):
            if idx == 0:
                chain_parts.append(f'        pl.when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                chain_parts.append(f'    .then({value_expr})')
            else:
                chain_parts.append(f'    .when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                chain_parts.append(f'    .then({value_expr})')
        
        # Add otherwise clause to preserve existing values if column already exists
        if tgt_alias in assigned_columns:
            chain_parts.append(f'    .otherwise(pl.col("{tgt_alias}"))')
        else:
            # If column doesn't exist yet, set to null for other dates
            chain_parts.append(f'    .otherwise(pl.lit(None))')
        
        chain_parts.append(f'    .alias("{tgt_alias}")')
        
        lines.append("    pdf = pdf.with_columns([\n")
        lines.append("\n".join(chain_parts) + "\n")
        lines.append("    ])\n")
        
        assigned_columns.add(tgt_alias)

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
