# Databricks notebook source
# MAGIC %run "./formgen"

import re
import sys
import os
from typing import Dict, List, Tuple
from collections import defaultdict, deque

try:
    from .formulas_generator import *
except ImportError:
    pass

ARITH_SPLIT_RE = re.compile(r"(\+|\-|\*|/)")

def preprocess_commands(lines: List[str]) -> List[str]:
    return [l.rstrip("\n") for l in lines]

def analyze_dependencies(parsed_cmds: List[Dict]) -> Tuple[defaultdict, defaultdict]:
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_targets = {p["target"].lower() for p in parsed_cmds if p and "target" in p}
    
    for formula in parsed_cmds:
        if not formula or "target" not in formula:
            continue
        tgt = formula["target"].lower()
        in_degree[tgt]  # ensure target is in in_degree
        
        # Skip SHIFT_PCT patterns as they're handled specially
        if formula.get("type") == "shift_pct":
            continue
        
        # Analyze refs for dependencies
        for ref in formula.get("refs", []):
            base, offset = parse_time_index(ref)
            ref_base = base.lower()
            if ref_base == tgt:
                continue
            if ref_base in all_targets:
                adj[ref_base].append(tgt)
                in_degree[tgt] += 1
    
    return adj, in_degree

def get_computation_levels(adj: defaultdict, in_degree: defaultdict) -> List[List[str]]:
    levels = []
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
        remaining = [n for n, d in in_degree.items() if d > 0]
        if remaining: levels.append(remaining)
    return levels

def _collect_operands_and_ops(rhs: str) -> Tuple[List[str], List[str]]:
    parts = ARITH_SPLIT_RE.split(rhs)
    operands = [p.strip() for p in parts if p and p not in "+-*/"]
    ops = [p for p in parts if p in "+-*/"]
    return operands, ops

def _operand_to_pl(tok: str) -> str:
    return token_to_pl_expr(tok)

def generate_formulas_file(cmds: List[str], out_filename: str = "formulas.py"):
    # Create a context to track needed functions (repo logic)
    ctx = {"need_shiftpct": False, "need_shiftpct_backwards": False}
    # (Checking logic for shiftpct omitted for brevity, handled in generate_polars_functions)
    
    fn_defs = generate_polars_functions(cmds)
    
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("import polars as pl\n")
        f.write("from typing import List, Tuple, Dict, Callable, Optional\n")
        f.write("from datetime import date, datetime\n")
        f.write("import re\n\n")
        f.write("try:\n    import polars_econ as ple\nexcept ImportError:\n    pass\n\n")
        for name in sorted(fn_defs.keys()):
            f.write(fn_defs[name])
            f.write("\n\n")

def generate_test_script(cmds: List[str], out_filename: str = "ts_transformer.py") -> str:
    parsed_raw = [p for p in (parse_fame_formula(c) for c in cmds) if p]
    if not parsed_raw:
        with open(out_filename, "w", encoding='utf-8') as f:
            f.write('"""No parsed commands"""\n')
        return out_filename

    # Track date filter state
    current_date_filter = None
    parsed = []
    for idx, p in enumerate(parsed_raw):
        np = p.copy()
        if np.get("type") == "date":
            current_date_filter = np.get("filter")
            continue
        
        if "target" in np:
            np["target"] = np["target"].lower()
        
        # Track the date filter for this command
        if "date_filter" not in np:
            np["date_filter"] = current_date_filter
        
        np["original_order"] = idx
        parsed.append(np)

    # Build formulas dict
    formulas = {}
    for p in parsed:
        if "target" in p:
            # Handle point-in-time and date-filtered uniqueness (Repo Logic)
            if p.get("type") == "point_in_time_assign":
                key = f"{p['target']}@{p.get('date', '')}_{p.get('original_order', 0)}"
                formulas[key] = p
            elif p.get("date_filter") is not None:
                date_filter = p["date_filter"]
                date_key = f"{date_filter.get('start', '*')}_to_{date_filter.get('end', '*')}"
                key = f"{p['target']}@datefilter_{date_key}_{p.get('original_order', 0)}"
                formulas[key] = p
            else:
                # Handle duplicate assignments by adding order if needed
                if p["target"] in formulas:
                    key = f"{p['target']}@_{p.get('original_order', 0)}"
                    formulas[key] = p
                else:
                    formulas[p["target"]] = p

    # Dependency Analysis
    adj, indeg = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, indeg)

    # SHIFT_PCT_BACKWARDS batching
    shift_pct_backwards_patterns = []
    for p in parsed:
        if p.get("type") == "shift_pct":
            offset = p.get("offset", 0)
            if offset > 0:
                shift_pct_backwards_patterns.append((p.get("target"), p.get("ser1"), p.get("ser2"), p.get("offset")))

    # Collect point-in-time assignments (Repo Logic)
    point_in_time_by_target = defaultdict(list)
    for p in parsed:
        if p.get("type") == "point_in_time_assign":
            target = p.get("target")
            point_in_time_by_target[target].append(p)

    lines = []
    lines.append('"""Auto-generated ts_transformer module"""\n')
    lines.append("import polars as pl\n")
    lines.append("from datetime import date\n")
    lines.append("from formulas import *\n\n")
    lines.append("def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:\n")
    lines.append('    """Apply transformations."""\n')
    
    assigned_columns = set()
    processed_keys = set()

    # 1. Batch SHIFT_PCT_BACKWARDS
    if shift_pct_backwards_patterns:
        column_pairs = [(tgt.upper(), pct.upper()) for tgt, ser1, pct, _ in shift_pct_backwards_patterns]
        offsets = [offs for _, _, _, offs in shift_pct_backwards_patterns]
        lines.append("    # Batch SHIFT_PCT_BACKWARDS\n")
        lines.append(f"    pdf = SHIFT_PCT_BACKWARDS_MULTIPLE(pdf, \"2016-12-31\", \"1981-03-31\", {column_pairs}, offsets={offsets})\n")

    for level_idx, level in enumerate(levels):
        if not level: continue
        
        # Sort commands in level (Repo Logic)
        level_formulas = []
        for tgt_base_name in level:
            # Exact match
            if tgt_base_name in formulas and tgt_base_name not in processed_keys:
                level_formulas.append((tgt_base_name, formulas[tgt_base_name]))
                processed_keys.add(tgt_base_name)
            
            # Suffixed keys (point-in-time, multiple assignments)
            for key, formula in formulas.items():
                if key in processed_keys: continue
                # Match "target@..."
                if key.startswith(tgt_base_name + '@'):
                    level_formulas.append((key, formula))
                    processed_keys.add(key)
        
        level_formulas.sort(key=lambda x: x[1].get("original_order", 0))
        
        # Grouping Logic (Repo Logic - Critical for efficiency)
        groups = []
        current_group = []
        current_group_filter = None
        
        for tgt_key, formula in level_formulas:
            formula_filter = formula.get("date_filter")
            formula_type = formula.get("type")
            
            # SCALAR types and Point-in-Time break the group because they generate distinct non-column logic
            if formula_type in ("point_in_time_assign", "scalar", "nlrx"):
                if current_group:
                    groups.append((current_group_filter, current_group))
                    current_group = []
                groups.append((None, [tgt_key])) # Add as isolated group
            elif formula_filter != current_group_filter:
                if current_group:
                    groups.append((current_group_filter, current_group))
                current_group = [tgt_key]
                current_group_filter = formula_filter
            else:
                current_group.append(tgt_key)
        
        if current_group:
            groups.append((current_group_filter, current_group))
            
        # Generate code for groups
        for group_filter, group_targets in groups:
            if not group_targets: continue
            
            # Is this a special single-item group?
            first_formula = formulas[group_targets[0]]
            ftype = first_formula.get("type")
            
            # --- HANDLE SCALAR (New Feature) ---
            if ftype == "scalar":
                tgt_key = group_targets[0]
                formula = formulas[tgt_key]
                tgt_alias = sanitize_func_name(formula["target"]).upper() # Scalar variable name
                rhs = formula.get("rhs", "")
                
                # Check for dynamic lookup
                rendered = render_polars_expr(rhs)
                if "__LOOKUP__" in rendered:
                    m = re.match(r"__LOOKUP__:([A-Za-z0-9_]+):([A-Za-z0-9_]+)", rendered)
                    if m:
                        ser, idx = m.groups()
                        lines.append(f'    {tgt_alias} = pdf.filter(pl.col("DATE") == {idx.upper()}).select(pl.col("{ser.upper()}")).item()\n')
                    else:
                        lines.append(f'    {tgt_alias} = {rendered}\n')
                elif "mean()" in rendered or "last()" in rendered or "first()" in rendered:
                    lines.append(f'    {tgt_alias} = pdf.select({rendered}).item()\n')
                else:
                    lines.append(f'    {tgt_alias} = {rendered}\n')
                continue

            # --- HANDLE NLRX (New Feature) ---
            if ftype == "nlrx":
                tgt_key = group_targets[0]
                formula = formulas[tgt_key]
                tgt_alias = sanitize_func_name(formula["target"]).upper()
                args = formula.get("args", [])
                if len(args) >= 8:
                    lamb_val = args[0]
                    # Check if lambda is a known scalar variable or number
                    # We assume if it's a string not in quotes, it might be a variable from 'scalar' assignment
                    val_lambda = lamb_val 
                    # If it's a column, we need extraction, but usually lambda is scalar in NLRX context
                    if not is_numeric_literal(lamb_val) and not lamb_val.replace('.','',1).isdigit():
                         # It is a variable name. If it was defined as a scalar earlier (python var), use directly.
                         # If it is a column, we might need extraction. 
                         # For safety in this hybrid, assume it matches a python var if defined, or we extract.
                         pass 

                    cols = [sanitize_func_name(x).upper() for x in args[1:8]]
                    lines.append(f'    pdf = NLRX(pdf, {val_lambda}, y="{cols[0]}", w1="{cols[1]}", w2="{cols[2]}", w3="{cols[3]}", w4="{cols[4]}", gss="{cols[5]}", gpr="{cols[6]}")\n')
                    assigned_columns.add(tgt_alias)
                continue

            # --- HANDLE POINT IN TIME (Repo Logic) ---
            if ftype == "point_in_time_assign":
                # Skip here, handled at end of function or separately
                continue

            # --- HANDLE COLUMN OPERATIONS (Standard) ---
            # Add date filter comment
            if group_filter:
                lines.append(f"    # Date filter: {group_filter['start']} to {group_filter['end']}\n")
            
            lines.append(f"    # --- Level {level_idx + 1}: compute {', '.join([formulas[t]['target'] for t in group_targets])} ---\n")
            cols_code = []
            
            for tgt_key in group_targets:
                formula = formulas[tgt_key]
                tgt_alias = sanitize_func_name(formula["target"]).upper()
                preserve_existing = tgt_alias in assigned_columns
                
                # Helper to wrap
                def wrap(expr_str):
                    if group_filter:
                        s, e = group_filter['start'], group_filter['end']
                        return f'APPLY_DATE_FILTER({expr_str}, "{tgt_alias}", "{s}", "{e}", preserve_existing={preserve_existing})'
                    return expr_str

                if any(tgt_alias.lower() == t.lower() for t, _, _, _ in shift_pct_backwards_patterns):
                    continue

                rhs_text = formula.get("rhs", "").strip()
                expr_code = None

                if formula.get("type") in ("chain", "mchain"):
                    # ... (Chain logic from repo)
                    pair_items = []
                    for op, var in formula.get("terms", []):
                        sign = "-" if op == "-" else ""
                        pcol = sanitize_func_name("p" + var).upper()
                        vcol = sanitize_func_name(var).upper()
                        pair_items.append(f"({sign}pl.col('{pcol}'), pl.col('{vcol}'))")
                    pairs_str = ", ".join(pair_items)
                    raw_expr = f'CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col("DATE"), year="{formula.get("year","")}")'
                    expr_code = wrap(raw_expr)

                elif formula.get("type") == "shift_pct":
                    s1 = sanitize_func_name(formula["ser1"]).upper()
                    s2 = sanitize_func_name(formula["ser2"]).upper()
                    off = formula["offset"]
                    if off <= 0: # Forward only
                        raw_expr = f'SHIFT_PCT(pl.col("{s1}"), pl.col("{s2}"), {off})'
                        expr_code = wrap(raw_expr)

                elif formula.get("type") == "convert":
                    params = formula.get("params", [])
                    if params:
                        scol = sanitize_func_name(params[0]).upper()
                        p_args = [f'"{p}"' for p in params[1:]]
                        # First arg is series expression
                        raw_expr = f'CONVERT(pl.col("{scol}"), {", ".join(p_args)})'
                        expr_code = wrap(raw_expr)

                elif formula.get("type") == "conditional":
                    raw_expr = render_conditional_expr(formula["condition"], formula["then_expr"], formula["else_expr"])
                    expr_code = wrap(raw_expr)

                elif formula.get("type") == "simple":
                    # Arithmetic optimizations
                    ops = _collect_operands_and_ops(rhs_text)[1]
                    if ops and all(o == "+" for o in ops):
                        args = [f'pl.col("{sanitize_func_name(x).upper()}")' if not is_numeric_literal(x) else f'pl.lit({x})' for x in _collect_operands_and_ops(rhs_text)[0]]
                        raw_expr = f'ADD_SERIES("{tgt_alias}", {", ".join(args)})'
                        # ADD_SERIES handles alias internally, so we just wrap filter
                        if group_filter:
                             # This is tricky because ADD_SERIES returns an aliased expr. 
                             # We might need to wrap the *calls* inside ADD_SERIES or rely on standard render if filter is active.
                             # For simplicity, fallback to standard render if filtered to ensure APPLY_DATE_FILTER works on the result
                             raw_expr = render_polars_expr(rhs_text)
                             expr_code = wrap(raw_expr)
                        else:
                             expr_code = raw_expr # No alias needed here, it's inside
                    else:
                        raw_expr = render_polars_expr(rhs_text)
                        expr_code = wrap(raw_expr)

                # Finalize
                if expr_code:
                    if not expr_code.startswith(("ADD_SERIES", "SUB_SERIES", "MUL_SERIES", "DIV_SERIES")) and not expr_code.endswith(f'.alias("{tgt_alias}")'):
                         expr_code = f'{expr_code}.alias("{tgt_alias}")'
                    cols_code.append(expr_code)
                    assigned_columns.add(tgt_alias)

            if cols_code:
                lines.append(f'    pdf = pdf.with_columns([\n        ' + ',\n        '.join(cols_code) + '\n    ])\n')

    # 4. Handle Point-in-Time Assignments (Repo Logic - Chained)
    for target, assignments in sorted(point_in_time_by_target.items()):
        if not assignments: continue
        assignments.sort(key=lambda x: x.get("original_order", 0))
        tgt_alias = sanitize_func_name(target).upper()
        lines.append(f"    # Point-in-time assignments for {tgt_alias}\n")
        
        chain_parts = []
        for idx, assign in enumerate(assignments):
            date_str = assign.get("date", "")
            rhs = assign.get("rhs", "").strip()
            iso_date = convert_fame_date_to_iso(date_str)
            
            # Value expression
            if is_strict_number(rhs):
                val_expr = f"pl.lit({rhs})"
            else:
                val_expr = render_polars_expr(rhs) # Simple render for RHS
            
            if idx == 0:
                chain_parts.append(f'        pl.when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                chain_parts.append(f'        .then({val_expr})')
            else:
                chain_parts.append(f'        .when(pl.col("DATE") == pl.lit("{iso_date}").cast(pl.Date))')
                chain_parts.append(f'        .then({val_expr})')
        
        # Close chain
        if tgt_alias in assigned_columns:
            chain_parts.append(f'        .otherwise(pl.col("{tgt_alias}"))')
        else:
            chain_parts.append(f'        .otherwise(pl.lit(None))')
        chain_parts.append(f'        .alias("{tgt_alias}")')
        
        lines.append(f'    pdf = pdf.with_columns([\n{ "".join(chain_parts) }\n    ])\n')

    lines.append("\n    return pdf\n")
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("".join(lines))
    return out_filename

def main(fame_commands):
    new_cmds = preprocess_commands(fame_commands)
    generate_formulas_file(new_cmds, out_filename="formulas.py")
    generate_test_script(new_cmds, out_filename="ts_transformer.py")

if __name__ == "__main__":
    fame_scripts = ['CONQCFP_no_desc.inp']
    fame_commands = []
    for fs in fame_scripts:
        if os.path.exists(f"./{fs}"):
            with open(f"./{fs}", "r") as file:
                fame_commands.extend(file.readlines())
        else:
            print(f"Warning: {fs} not found.")
    if fame_commands:
        main(fame_commands)
    else:
        print("No commands loaded.")
