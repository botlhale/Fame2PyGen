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
        if not formula or "target" not in formula: continue
        tgt = formula["target"].lower()
        in_degree[tgt]
        if formula.get("type") == "shift_pct": continue
        for ref in formula.get("refs", []):
            base, offset = parse_time_index(ref)
            ref_base = base.lower()
            if ref_base == tgt: continue
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
                if in_degree[nb] == 0: q.append(nb)
    if sum(len(L) for L in levels) != len(in_degree):
        remaining = [n for n, d in in_degree.items() if d > 0]
        if remaining: levels.append(remaining)
    return levels

def _collect_operands_and_ops(rhs: str) -> Tuple[List[str], List[str]]:
    parts = ARITH_SPLIT_RE.split(rhs)
    operands = [p.strip() for p in parts if p and p not in "+-*/"]
    ops = [p for p in parts if p in "+-*/"]
    return operands, ops

def generate_formulas_file(cmds: List[str], out_filename: str = "formulas.py"):
    fn_defs = generate_polars_functions(cmds)
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("import polars as pl\nfrom typing import List, Tuple, Dict, Callable, Optional\n")
        f.write("from datetime import date, datetime\nimport re\n\n")
        f.write("try:\n    import polars_econ as ple\nexcept ImportError:\n    pass\n\n")
        for name in sorted(fn_defs.keys()):
            f.write(fn_defs[name])
            f.write("\n\n")

def generate_test_script(cmds: List[str], out_filename: str = "ts_transformer.py") -> str:
    parsed_raw = [p for p in (parse_fame_formula(c) for c in cmds) if p]
    current_date_filter = None
    parsed = []
    for idx, p in enumerate(parsed_raw):
        np = p.copy()
        if np.get("type") == "date":
            current_date_filter = np.get("filter")
            continue
        if "date_filter" not in np: np["date_filter"] = current_date_filter
        np["original_order"] = idx
        parsed.append(np)
    
    formulas = {}
    for p in parsed:
        if "target" in p:
            key = f"{p['target']}@{p.get('original_order')}"
            formulas[key] = p
            
    adj, indeg = analyze_dependencies(parsed)
    levels = get_computation_levels(adj, indeg)
    
    lines = []
    lines.append('"""Auto-generated ts_transformer module"""\nimport polars as pl\n')
    lines.append("from datetime import date\nfrom formulas import *\n\n")
    lines.append("def ts_transformer(pdf: pl.DataFrame) -> pl.DataFrame:\n")
    
    assigned_columns = set()
    processed_keys = set()
    
    for level in levels:
        if not level: continue
        level_formulas = []
        for tgt_base in level:
            for k, f in formulas.items():
                if f["target"].lower() == tgt_base and k not in processed_keys:
                    level_formulas.append((k, f))
                    processed_keys.add(k)
        level_formulas.sort(key=lambda x: x[1]["original_order"])
        
        for tgt_key, formula in level_formulas:
            tgt_alias = sanitize_func_name(formula["target"]).upper()
            ftype = formula.get("type")
            expr_code = None
            
            if ftype == "scalar":
                # Generate scalar assignment logic (aggregations vs lookups vs literals)
                rhs = formula.get("rhs", "")
                
                # Check for dynamic lookup: SERIES[SCALAR_DATE]
                # formgen parser returns "__LOOKUP__:SERIES:INDEX"
                rendered = render_polars_expr(rhs)
                if "mean()" in rendered or "last()" in rendered or "first()" in rendered:
                    # Aggregation
                    lines.append(f'    {tgt_alias} = pdf.select({rendered}).item()\n')
                elif "__LOOKUP__" in rendered:
                    # Extract parts from placeholder
                    m = re.match(r"__LOOKUP__:([A-Za-z0-9_]+):([A-Za-z0-9_]+)", rendered)
                    if m:
                        ser, idx = m.groups()
                        lines.append(f'    {tgt_alias} = pdf.filter(pl.col("DATE") == {idx.upper()}).select(pl.col("{ser.upper()}")).item()\n')
                    else:
                        lines.append(f'    {tgt_alias} = {rendered} # Complex lookup?\n')
                else:
                    # Literal or simple assignment
                    lines.append(f'    {tgt_alias} = {rendered}\n')
                continue

            if ftype == "nlrx":
                args = formula.get("args", [])
                if len(args) >= 8:
                    lamb_val = args[0]
                    is_num = lambda x: x.replace('.','',1).isdigit()
                    val_lambda = lamb_val if is_num(lamb_val) else f'pdf.select(pl.col("{sanitize_func_name(lamb_val).upper()}")).item(0)'
                    cols = [sanitize_func_name(x).upper() for x in args[1:8]]
                    lines.append(f'    pdf = NLRX(pdf, {val_lambda}, y="{cols[0]}", w1="{cols[1]}", w2="{cols[2]}", w3="{cols[3]}", w4="{cols[4]}", gss="{cols[5]}", gpr="{cols[6]}")\n')
                    assigned_columns.add(tgt_alias)
                continue
            
            if ftype == "conditional":
                expr_code = render_conditional_expr(formula["condition"], formula["then_expr"], formula["else_expr"])
            elif ftype == "simple":
                rhs = formula.get("rhs", "")
                ops = _collect_operands_and_ops(rhs)[1]
                if ops and all(o == "+" for o in ops):
                    args = [f'pl.col("{sanitize_func_name(x).upper()}")' if not _is_numeric_literal(x) else f'pl.lit({x})' for x in _collect_operands_and_ops(rhs)[0]]
                    expr_code = f'ADD_SERIES("{tgt_alias}", {", ".join(args)})'
                else:
                    expr_code = render_polars_expr(rhs)
            
            if expr_code:
                dfilter = formula.get("date_filter")
                preserve = tgt_alias in assigned_columns
                if dfilter:
                    expr_code = f'APPLY_DATE_FILTER({expr_code}, "{tgt_alias}", "{dfilter["start"]}", "{dfilter["end"]}", preserve_existing={preserve})'
                if not expr_code.startswith("ADD_SERIES"):
                    expr_code = f'{expr_code}.alias("{tgt_alias}")'
                lines.append(f'    pdf = pdf.with_columns([{expr_code}])\n')
                assigned_columns.add(tgt_alias)

    lines.append("\n    return pdf\n")
    with open(out_filename, "w", encoding='utf-8') as f: f.write("".join(lines))
    return out_filename

def main(fame_commands):
    new_cmds = preprocess_commands(fame_commands)
    generate_formulas_file(new_cmds, out_filename="formulas.py")
    generate_test_script(new_cmds, out_filename="ts_transformer.py")

if __name__ == "__main__":
    fame_scripts = ['fame1.inp']
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
