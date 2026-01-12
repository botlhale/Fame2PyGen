# Databricks notebook source
# MAGIC %run "./formgen"

import re
import sys
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

try:
    from . formulas_generator import (
        parse_fame_formula, generate_polars_functions, sanitize_func_name,
        render_polars_expr, render_conditional_expr, is_numeric_literal,
        is_strict_number, token_to_pl_expr, parse_time_index,
        convert_fame_date_to_iso, normalize_formula_text, split_args_balanced,
        split_local_db_name, LOCAL_DB_IGNORE
    )
except ImportError:
    # Fallback for direct execution
    from formulas_generator import (
        parse_fame_formula, generate_polars_functions, sanitize_func_name,
        render_polars_expr, render_conditional_expr, is_numeric_literal,
        is_strict_number, token_to_pl_expr, parse_time_index,
        convert_fame_date_to_iso, normalize_formula_text, split_args_balanced,
        split_local_db_name, LOCAL_DB_IGNORE
    )

# Regex for splitting arithmetic expressions
ARITH_SPLIT_RE = re.compile(r"(\+|\-|\*|/)")

# Sentinel to distinguish "no filter set" from "filter explicitly disabled with date *"
_NO_DATE_FILTER_SET = object()


def preprocess_commands(lines: List[str]) -> List[str]: 
    """Clean and preprocess command lines."""
    return [l.rstrip("\n") for l in lines]


def analyze_dependencies(parsed_cmds: List[Dict]) -> Tuple[defaultdict, defaultdict]:
    """Analyze dependencies between formulas for computation ordering."""
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_targets = {p["target"]. lower() for p in parsed_cmds if p and "target" in p}
    
    for formula in parsed_cmds: 
        if not formula or "target" not in formula:
            continue
        tgt = formula["target"]. lower()
        in_degree[tgt]  # ensure target is in in_degree
        
        # Skip SHIFT_PCT patterns as they're handled specially
        if formula.get("type") == "shift_pct":
            continue
        
        # Analyze refs for dependencies
        for ref in formula.get("refs", []):
            base, offset = parse_time_index(ref)
            ref_base = base. lower()
            if ref_base == tgt:
                continue
            if ref_base in all_targets:
                adj[ref_base].append(tgt)
                in_degree[tgt] += 1
    
    return adj, in_degree


def get_computation_levels(adj:  defaultdict, in_degree: defaultdict) -> List[List[str]]:
    """Topological sort to determine computation order levels."""
    levels = []
    # Make a copy of in_degree to avoid modifying the original
    in_deg = dict(in_degree)
    q = deque([n for n, d in in_deg.items() if d == 0])
    
    while q:
        level_nodes = sorted(list(q))
        levels.append(level_nodes)
        nodes = list(q)
        q.clear()
        for n in nodes: 
            for nb in sorted(adj[n]):
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    q.append(nb)
    
    # Handle any remaining nodes (cycles or disconnected)
    if sum(len(L) for L in levels) != len(in_degree):
        remaining = [n for n, d in in_deg.items() if d > 0]
        if remaining: 
            levels.append(remaining)
    
    return levels


def _collect_operands_and_ops(rhs:  str) -> Tuple[List[str], List[str]]:
    """Split an arithmetic expression into operands and operators."""
    parts = ARITH_SPLIT_RE. split(rhs)
    operands = [p.strip() for p in parts if p.strip() and p not in "+-*/"]
    ops = [p for p in parts if p in "+-*/"]
    return operands, ops


def _operand_to_pl_expr(tok: str) -> str:
    """Convert an operand token to a Polars expression string."""
    tok = tok.strip()
    if is_strict_number(tok):
        return f"pl.lit({tok})"
    return f'pl.col("{sanitize_func_name(tok).upper()}")'


def _get_series_function_for_ops(ops: List[str]) -> Optional[str]:
    """Determine which series function to use based on operators."""
    if not ops:
        return None
    if all(o == "+" for o in ops):
        return "ADD_SERIES"
    elif all(o == "-" for o in ops):
        return "SUB_SERIES"
    elif all(o == "*" for o in ops):
        return "MUL_SERIES"
    elif all(o == "/" for o in ops):
        return "DIV_SERIES"
    return None  # Mixed operators


def generate_formulas_file(cmds: List[str], out_filename: str = "formulas.py"):
    """Generate the formulas.py file with helper function definitions."""
    fn_defs = generate_polars_functions(cmds)
    
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("import polars as pl\n")
        f.write("from typing import List, Tuple, Dict, Callable, Optional\n")
        f.write("from datetime import date, datetime\n")
        f.write("import re\n\n")
        f.write("try:\n    import polars_econ as ple\nexcept ImportError:\n    ple = None\n\n")
        
        for name in sorted(fn_defs.keys()):
            f.write(fn_defs[name])
            f.write("\n\n")


def generate_test_script(cmds: List[str], out_filename: str = "ts_transformer.py") -> str:
    """Generate the ts_transformer.py file with the transformation pipeline."""
    
    # Parse all commands
    parsed_raw = [p for p in (parse_fame_formula(c) for c in cmds) if p]
    if not parsed_raw: 
        with open(out_filename, "w", encoding='utf-8') as f:
            f.write('"""No parsed commands"""\n')
        return out_filename

    # Track date filter state using sentinel for "never set"
    current_date_filter = _NO_DATE_FILTER_SET
    current_freq = None
    parsed = []

    def target_alias(name: str) -> str:
        db_prefix, series_name = split_local_db_name(name)
        col_name = f"{db_prefix}_{series_name}" if db_prefix else series_name
        return sanitize_func_name(col_name).upper()

    def series_alias_only(name: str) -> str:
        _, series_name = split_local_db_name(name)
        return sanitize_func_name(series_name).upper()

    local_db_series = defaultdict(set)
    
    for idx, p in enumerate(parsed_raw):
        np = p.copy()
        
        # Handle frequency commands
        if np. get("type") == "freq":
            current_freq = np. get("freq")
            continue
        
        # Handle date filter commands
        if np. get("type") == "date": 
            current_date_filter = np.get("filter")  # None for "date *", dict for range
            continue
        
        # Normalize target to lowercase
        if "target" in np: 
            np["target"] = np["target"].lower()
        
        # Track the date filter for this command
        if "date_filter" not in np:
            if current_date_filter is _NO_DATE_FILTER_SET: 
                np["date_filter"] = None
            else:
                np["date_filter"] = current_date_filter
        
        # Track frequency
        np["freq"] = current_freq
        np["original_order"] = idx

        if "target" in np:
            db_prefix, _ = split_local_db_name(np["target"])
            if db_prefix:
                local_db_series[db_prefix.upper()].add(series_alias_only(np["target"]))

        for ref in np.get("refs", []):
            db_prefix, _ = split_local_db_name(ref)
            if db_prefix:
                local_db_series[db_prefix.upper()].add(series_alias_only(ref))

        parsed.append(np)

    # Build formulas dictionary with unique keys
    formulas = {}
    for p in parsed:
        if "target" not in p:
            continue
            
        target = p["target"]
        
        # Handle point-in-time assignments (unique by target + date)
        if p.get("type") == "point_in_time_assign":
            key = f"{target}@pit_{p. get('date', '')}_{p.get('original_order', 0)}"
            formulas[key] = p
        # Handle date-filtered assignments
        elif p.get("date_filter") is not None:
            date_filter = p["date_filter"]
            date_key = f"{date_filter. get('start', '*')}_to_{date_filter.get('end', '*')}"
            key = f"{target}@datefilter_{date_key}_{p.get('original_order', 0)}"
            formulas[key] = p
        else:
            # Handle duplicate assignments by adding order suffix if needed
            if target in formulas: 
                key = f"{target}@_{p.get('original_order', 0)}"
                formulas[key] = p
            else:
                formulas[target] = p

    # Dependency Analysis
    adj, indeg = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, indeg)

    # Collect SHIFT_PCT_BACKWARDS patterns for batching
    shift_pct_backwards_patterns = []
    for p in parsed:
        if p.get("type") == "shift_pct": 
            offset = p.get("offset", 0)
            if offset > 0:  # Backwards pattern
                shift_pct_backwards_patterns.append((
                    p. get("target"),
                    p. get("ser1"),
                    p.get("ser2"),
                    p.get("offset")
                ))

    # Collect point-in-time assignments grouped by target
    point_in_time_by_target = defaultdict(list)
    for p in parsed:
        if p.get("type") == "point_in_time_assign": 
            target = p.get("target")
            point_in_time_by_target[target].append(p)

    # Start building output lines
    lines = []
    lines.append('"""Auto-generated ts_transformer module from FAME commands."""\n\n')
    lines.append("import polars as pl\n")
    lines.append("from datetime import date\n")
    lines.append("from formulas import *\n\n")
    lines.append("\n")
    lines.append("def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:\n")
    lines.append('    """Apply FAME-to-Polars transformations to the DataFrame."""\n')
    
    assigned_columns = set()
    processed_keys = set()
    lines.append("    local_databases = {}\n")

    # 1. Handle batch SHIFT_PCT_BACKWARDS first
    if shift_pct_backwards_patterns: 
        column_pairs = [(tgt.upper(), pct.upper()) for tgt, ser1, pct, _ in shift_pct_backwards_patterns]
        offsets = [offs for _, _, _, offs in shift_pct_backwards_patterns]
        lines.append("\n    # Batch SHIFT_PCT_BACKWARDS processing\n")
        lines.append(f'    pdf = SHIFT_PCT_BACKWARDS_MULTIPLE(pdf, "2016-12-31", "1981-03-31", {column_pairs}, offsets={offsets})\n')
        
        # Mark these as assigned
        for tgt, _, _, _ in shift_pct_backwards_patterns: 
            assigned_columns.add(tgt.upper())

    # 2. Process computation levels
    for level_idx, level in enumerate(levels):
        if not level:
            continue
        
        # Collect formulas for this level
        level_formulas = []
        for tgt_base_name in level:
            # Exact match
            if tgt_base_name in formulas and tgt_base_name not in processed_keys:
                level_formulas. append((tgt_base_name, formulas[tgt_base_name]))
                processed_keys.add(tgt_base_name)
            
            # Suffixed keys (point-in-time, date-filtered, duplicate assignments)
            for key, formula in formulas. items():
                if key in processed_keys: 
                    continue
                if key.startswith(tgt_base_name + '@'):
                    level_formulas.append((key, formula))
                    processed_keys.add(key)
        
        # Sort by original order to maintain command sequence
        level_formulas.sort(key=lambda x: x[1].get("original_order", 0))
        
        # Group formulas by date filter for efficient with_columns batching
        groups = []
        current_group = []
        current_group_filter = _NO_DATE_FILTER_SET
        
        for tgt_key, formula in level_formulas:
            formula_filter = formula.get("date_filter")
            formula_type = formula.get("type")
            
            # These types need isolated handling
            if formula_type in ("point_in_time_assign", "scalar", "nlrx"):
                if current_group: 
                    groups. append((current_group_filter, current_group))
                    current_group = []
                    current_group_filter = _NO_DATE_FILTER_SET
                groups.append((None, [tgt_key], formula_type))
                continue
            
            # Group by date filter
            if formula_filter != current_group_filter: 
                if current_group:
                    groups.append((current_group_filter, current_group))
                current_group = [tgt_key]
                current_group_filter = formula_filter
            else: 
                current_group. append(tgt_key)
        
        if current_group:
            groups. append((current_group_filter, current_group))

        # Generate code for each group
        for group_data in groups:
            if len(group_data) == 3:
                # Special type (scalar, nlrx, point_in_time)
                _, group_targets, special_type = group_data
                group_filter = None
            else:
                group_filter, group_targets = group_data
                special_type = None
            
            if not group_targets: 
                continue
            
            first_formula = formulas[group_targets[0]]
            ftype = first_formula. get("type") if special_type is None else special_type

            # --- HANDLE SCALAR ---
            if ftype == "scalar": 
                tgt_key = group_targets[0]
                formula = formulas[tgt_key]
                tgt_alias = target_alias(formula["target"])
                rhs = formula.get("rhs", "")
                
                lines.append(f"\n    # Scalar assignment: {tgt_alias}\n")
                
                rendered = render_polars_expr(rhs)
                if "__LOOKUP__" in rendered: 
                    m = re.match(r"__LOOKUP__: ([A-Za-z0-9_]+):([A-Za-z0-9_]+)", rendered)
                    if m:
                        ser, idx = m.groups()
                        lines.append(f'    {tgt_alias} = pdf.filter(pl.col("DATE") == {idx.upper()}).select(pl.col("{ser.upper()}")).item()\n')
                    else:
                        lines.append(f'    {tgt_alias} = {rendered}\n')
                elif any(agg in rendered for agg in [". mean()", ".last()", ".first()", ".sum()", ".min()", ".max()"]):
                    lines.append(f'    {tgt_alias} = pdf.select({rendered}).item()\n')
                else: 
                    lines.append(f'    {tgt_alias} = {rendered}\n')
                continue

            # --- HANDLE NLRX ---
            if ftype == "nlrx":
                tgt_key = group_targets[0]
                formula = formulas[tgt_key]
                tgt_alias = target_alias(formula["target"])
                args = formula.get("args", [])
                
                lines.append(f"\n    # NLRX computation: {tgt_alias}\n")
                
                if len(args) >= 8:
                    lamb_val = args[0]
                    cols = [sanitize_func_name(x).upper() for x in args[1:8]]
                    lines.append(f'    pdf = NLRX(pdf, {lamb_val}, y="{cols[0]}", w1="{cols[1]}", w2="{cols[2]}", w3="{cols[3]}", w4="{cols[4]}", gss="{cols[5]}", gpr="{cols[6]}")\n')
                    assigned_columns.add(tgt_alias)
                continue

            # --- HANDLE POINT IN TIME (skip here, handled later) ---
            if ftype == "point_in_time_assign":
                continue

            # --- HANDLE COLUMN OPERATIONS ---
            # Add comment for date filter if present
            if group_filter is not None and group_filter is not _NO_DATE_FILTER_SET:
                lines.append(f"\n    # Date filter:  {group_filter['start']} to {group_filter['end']}\n")
            elif group_filter is None and len(group_targets) > 0:
                lines.append(f"\n    # Date filter: * (all dates)\n")
            
            target_names = [target_alias(formulas[t]['target']) for t in group_targets]
            lines.append(f"    # Level {level_idx + 1}: compute {', '.join(target_names)}\n")
            
            cols_code = []
            
            for tgt_key in group_targets: 
                formula = formulas[tgt_key]
                tgt_alias = target_alias(formula["target"])
                preserve_existing = tgt_alias in assigned_columns
                
                # Check if this is a backwards shift_pct (already handled)
                if any(tgt_alias.lower() == t.lower() for t, _, _, _ in shift_pct_backwards_patterns):
                    continue

                # Helper to wrap expression with date filter
                def wrap_with_filter(expr_str:  str) -> str:
                    if group_filter is not None and group_filter is not _NO_DATE_FILTER_SET:
                        s, e = group_filter['start'], group_filter['end']
                        return f'APPLY_DATE_FILTER({expr_str}, "{tgt_alias}", "{s}", "{e}", preserve_existing={preserve_existing})'
                    return expr_str

                rhs_text = formula.get("rhs", "").strip()
                expr_code = None

                # --- Chain/MChain ---
                if formula.get("type") in ("chain", "mchain"):
                    pair_items = []
                    for op, var in formula.get("terms", []):
                        sign = "-" if op == "-" else ""
                        pcol = sanitize_func_name("p" + var).upper()
                        vcol = sanitize_func_name(var).upper()
                        if sign: 
                            pair_items.append(f"(-pl.col('{pcol}'), pl.col('{vcol}'))")
                        else:
                            pair_items. append(f"(pl.col('{pcol}'), pl.col('{vcol}'))")
                    pairs_str = ", ".join(pair_items)
                    raw_expr = f'CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col("DATE"), year="{formula. get("year", "")}")'
                    expr_code = wrap_with_filter(raw_expr)

                # --- Shift PCT (forward only, backwards handled above) ---
                elif formula.get("type") == "shift_pct":
                    s1 = sanitize_func_name(formula["ser1"]).upper()
                    s2 = sanitize_func_name(formula["ser2"]).upper()
                    off = formula["offset"]
                    if off <= 0:  # Forward shift
                        raw_expr = f'SHIFT_PCT(pl.col("{s1}"), pl.col("{s2}"), {off})'
                        expr_code = wrap_with_filter(raw_expr)

                # --- Convert ---
                elif formula. get("type") == "convert":
                    params = formula.get("params", [])
                    if params: 
                        scol = sanitize_func_name(params[0]).upper()
                        p_args = [f'"{p}"' for p in params[1:]]
                        raw_expr = f'CONVERT(pl.col("{scol}"), {", ".join(p_args)})'
                        expr_code = wrap_with_filter(raw_expr)

                # --- Conditional ---
                elif formula.get("type") == "conditional": 
                    raw_expr = render_conditional_expr(
                        formula["condition"],
                        formula["then_expr"],
                        formula["else_expr"]
                    )
                    expr_code = wrap_with_filter(raw_expr)

                # --- LSUM ---
                elif formula.get("type") == "lsum": 
                    raw_expr = render_polars_expr(rhs_text)
                    expr_code = wrap_with_filter(raw_expr)

                # --- Fishvol ---
                elif formula.get("type") == "fishvol":
                    pairs = formula.get("pairs", [])
                    year = formula.get("year", "")
                    pair_items = []
                    for vol, price in pairs:
                        vcol = sanitize_func_name(vol).upper()
                        pcol = sanitize_func_name(price).upper()
                        pair_items.append(f"(pl.col('{vcol}'), pl.col('{pcol}'))")
                    pairs_str = ", ". join(pair_items)
                    raw_expr = f'FISHVOL([{pairs_str}], pl.col("DATE"), {year})'
                    expr_code = wrap_with_filter(raw_expr)

                # --- Simple assignment or arithmetic ---
                elif formula.get("type") in ("simple", "assign_series"):
                    operands, ops = _collect_operands_and_ops(rhs_text)
                    
                    # Determine if we can use an optimized series function
                    series_func = _get_series_function_for_ops(ops)
                    
                    if series_func and not (group_filter is not None and group_filter is not _NO_DATE_FILTER_SET):
                        # Use optimized series function (no date filter)
                        args = [_operand_to_pl_expr(x) for x in operands]
                        expr_code = f'{series_func}("{tgt_alias}", {", ".join(args)})'
                    elif series_func and group_filter is not None:
                        # Has date filter - render normally and wrap
                        raw_expr = render_polars_expr(rhs_text)
                        expr_code = wrap_with_filter(raw_expr)
                    else:
                        # Mixed operators or no operators - use standard render
                        raw_expr = render_polars_expr(rhs_text)
                        expr_code = wrap_with_filter(raw_expr)

                # Finalize expression with alias if needed
                if expr_code:
                    # Series functions already include alias, others need it
                    needs_alias = not any(expr_code.startswith(fn) for fn in 
                                         ["ADD_SERIES", "SUB_SERIES", "MUL_SERIES", "DIV_SERIES"])
                    if needs_alias and not expr_code. endswith(f'. alias("{tgt_alias}")'):
                        expr_code = f'{expr_code}.alias("{tgt_alias}")'
                    
                    cols_code.append(expr_code)
                    assigned_columns.add(tgt_alias)

            # Write the with_columns block
            if cols_code: 
                if len(cols_code) == 1:
                    lines.append(f'    pdf = pdf.with_columns([{cols_code[0]}])\n')
                else: 
                    lines.append('    pdf = pdf.with_columns([\n')
                    for i, code in enumerate(cols_code):
                        comma = ',' if i < len(cols_code) - 1 else ''
                        lines.append(f'        {code}{comma}\n')
                    lines.append('    ])\n')

    # 3. Handle Point-in-Time Assignments (grouped by target)
    if point_in_time_by_target: 
        lines.append("\n    # Point-in-time assignments\n")
        
        for target, assignments in sorted(point_in_time_by_target.items()):
            if not assignments:
                continue
            
            # Sort by original order
            assignments.sort(key=lambda x: x.get("original_order", 0))
            tgt_alias = target_alias(target)
            
            lines.append(f"    # Point-in-time for {tgt_alias}\n")
            
            # Build chained when/then expression
            chain_parts = []
            for idx, assign in enumerate(assignments):
                date_str = assign.get("date", "")
                rhs = assign.get("rhs", "").strip()
                iso_date = convert_fame_date_to_iso(date_str)
                
                # Render the value expression
                if is_strict_number(rhs):
                    val_expr = f"pl.lit({rhs})"
                else:
                    val_expr = render_polars_expr(rhs)
                
                if idx == 0:
                    chain_parts.append(f'pl.when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                    chain_parts.append(f'. then({val_expr})')
                else:
                    chain_parts.append(f'. when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                    chain_parts.append(f'. then({val_expr})')
            
            # Add otherwise clause
            if tgt_alias in assigned_columns: 
                chain_parts.append(f'.otherwise(pl.col("{tgt_alias}"))')
            else:
                chain_parts.append('.otherwise(pl.lit(None))')
            
            chain_parts.append(f'. alias("{tgt_alias}")')
            
            # Write the expression
            expr_str = ''.join(chain_parts)
            lines.append(f'    pdf = pdf.with_columns([{expr_str}])\n')
            assigned_columns.add(tgt_alias)

    # 4. Build local database DataFrames (e.g., AA'ABC -> DataFrame AA with column ABC)
    if local_db_series:
        lines.append("\n    # Local databases extracted from pdf\n")
        for db_name, series_set in sorted(local_db_series.items()):
            db_var = sanitize_func_name(db_name).upper()
            lines.append(f"    {db_var}_cols = []\n")
            lines.append(f'    if "DATE" in pdf.columns:\n')
            lines.append(f"        {db_var}_cols.append(pl.col(\"DATE\"))\n")
            for series in sorted(series_set):
                col_name = f"{db_name.upper()}_{series}"
                lines.append(f'    if "{col_name}" in pdf.columns:\n')
                lines.append(f'        {db_var}_cols.append(pl.col("{col_name}").alias("{series}"))\n')
            lines.append(f"    if {db_var}_cols:\n")
            lines.append(f"        {db_var} = pdf.select({db_var}_cols)\n")
            lines.append(f'        local_databases[\"{db_name.upper()}\"] = {db_var}\n')

    # 5. Return the transformed DataFrame (with local databases attached)
    # Keep return type stable while exposing local databases for callers that need them.
    lines.append("    setattr(pdf, \"_local_databases\", local_databases)\n")
    lines.append("\n    return pdf\n")

    # Write the file
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("". join(lines))
    
    return out_filename


def main(fame_commands: List[str]):
    """Main entry point for processing FAME commands."""
    new_cmds = preprocess_commands(fame_commands)
    generate_formulas_file(new_cmds, out_filename="formulas.py")
    generate_test_script(new_cmds, out_filename="ts_transformer.py")
    print("Generated formulas.py and ts_transformer.py")


if __name__ == "__main__": 
    # Example usage with file input
    fame_scripts = ['test. inp']
    fame_commands = []
    
    for fs in fame_scripts: 
        if os.path.exists(f". /{fs}"):
            with open(f"./{fs}", "r") as file:
                fame_commands.extend(file.readlines())
        else:
            print(f"Warning: {fs} not found.")
    
    if fame_commands:
        main(fame_commands)
    else:
        # Demo with example commands
        demo_commands = [
            "freq m",
            "v_base = 100",
            "date 2020-01-01 to 2020-12-31",
            "v_2020 = v_base * 2",
            "date 2021-01-01 to 2021-12-31",
            "v_2021 = v_base * 3",
            "date *",
            "v_all = v_base + v_2020 + v_2021",
        ]
        print("Running demo with example commands...")
        main(demo_commands)
