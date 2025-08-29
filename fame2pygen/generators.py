from typing import Dict, List, Tuple
import textwrap
from .model import ParsedCommand, GenerationContext
from .utils import sanitize_func_name

def fame_expr_to_polars(rhs_expr: str, memory: Dict[str, str]) -> str:
    import re
    expr = rhs_expr.strip()
    sorted_vars = sorted(memory.keys(), key=len, reverse=True)
    for v in sorted_vars:
        pattern = rf'(?<![a-zA-Z0-9_$]){re.escape(v)}(?![a-zA-Z0-9_$])'
        expr = re.sub(pattern, memory[v], expr)
    return expr

def build_generation_context(parsed_cmds: List[ParsedCommand], fame_commands: List[str]) -> GenerationContext:
    ctx = GenerationContext(fame_commands=fame_commands, parsed=parsed_cmds)
    for p in parsed_cmds:
        if p.type == "mchain": ctx.has_mchain = True
        if p.type == "convert": ctx.has_convert = True
        if p.type == "fishvol": ctx.has_fishvol = True
    return ctx

def generate_formulas_module(ctx: GenerationContext) -> str:
    # Collect calculable definitions
    definitions = {}
    memory = {}
    for cmd in ctx.parsed:
        if cmd.type in ("simple", "mchain", "convert", "fishvol"):
            t = cmd.target
            if not t: continue
            memory[t.lower()] = sanitize_func_name(t)
            definitions[t.lower()] = cmd

    fn_defs = []

    for target_lower, entry in definitions.items():
        if entry.type != "simple":
            continue
        sanitized_target = memory[target_lower]
        fn_name = sanitized_target.upper()

        doc = textwrap.dedent(f"""
        \"\"\"
        Computes values for FAME variable '{entry.target}'.
        Derived from FAME script line:
            {entry.target}={entry.original_rhs}
        \"\"\"
        """).strip()

        refs = sorted(set(r.lower() for r in entry.refs))
        arg_str = ", ".join([f"{sanitize_func_name(ref)}: pl.Expr" for ref in refs])
        signature = f"def {fn_name}({arg_str}) -> pl.Expr:"

        if not refs:
            # Attempt literal
            try:
                evaluated_rhs = eval(entry.rhs)  # noqa: S307 (controlled context)
                body_expr = f"pl.lit({evaluated_rhs})"
            except Exception:
                body_expr = f"pl.lit({entry.rhs})"
        else:
            substitution_map = {ref: sanitize_func_name(ref) for ref in refs}
            body_expr = fame_expr_to_polars(entry.rhs.lower(), substitution_map)

        body = f"""    res = (
        {body_expr}
    )
    return res.alias("{sanitized_target}")"""

        fn_defs.append(f"{signature}\n    {doc}\n{body}\n")

    if ctx.has_mchain:
        fn_defs.append(
            "def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))\n"
        )
    if ctx.has_convert:
        fn_defs.append(
            "def CONVERT(series: 'pl.DataFrame', as_freq: str, to_freq: str, technique: str, observed: str) -> pl.DataFrame:\n"
            "    import polars_econ as ple\n"
            "    return ple.convert(series, 'DATE', as_freq=as_freq, to_freq=to_freq, technique=technique, observed=observed)\n"
        )
    if ctx.has_fishvol:
        fn_defs.append(
            "def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    return ple.fishvol(series_pairs, date_col, rebase_year)\n"
        )

    header = "import polars as pl\nfrom typing import List, Tuple\n\n"
    return header + "\n".join(fn_defs)

def topological_levels(parsed: List[ParsedCommand]) -> List[List[str]]:
    from collections import defaultdict, deque
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    targets = {p.target.lower() for p in parsed if p.target}
    formulas_by_target = {p.target.lower(): p for p in parsed if p.target}

    for p in parsed:
        if not p.target:
            continue
        t = p.target.lower()
        in_degree[t]
        for r in p.refs:
            rl = r.lower()
            if rl in targets:
                adj[rl].append(t)
                in_degree[t] += 1

    q = deque([n for n, deg in in_degree.items() if deg == 0])
    levels = []
    visited = 0
    while q:
        level_nodes = sorted(list(q))
        q.clear()
        levels.append(level_nodes)
        for n in level_nodes:
            for nb in sorted(adj[n]):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    q.append(nb)
            visited += 1
    if visited != len(in_degree):
        raise RuntimeError("Cyclic dependency detected among formulas.")
    # Filter out nodes that are convert targets (handled later)
    return levels

def generate_pipeline_script(ctx: GenerationContext,
                             formulas_module_name: str,
                             output_columns_case_sensitive: bool = True) -> str:
    parsed = ctx.parsed
    formulas_by_target = {p.target.lower(): p for p in parsed if p.target}
    levels = topological_levels(parsed)
    convert_groups = {}
    convert_targets = set()
    current_as_freq = "1y"
    freq_map = {"m": "1mo", "q": "1q", "a": "1y"}
    tech_map = {"disc": "discrete", "lin": "linear", "const": "constant"}
    obs_map = {"ave": "average", "beg": "beginning", "end": "end", "sum": "sum"}

    for p in parsed:
        if p.type == "convert":
            to_freq_code = p.params[0]
            convert_groups.setdefault(to_freq_code, []).append(p)
            convert_targets.add(p.target.lower())

    original_case_map = {p.target.lower(): p.target for p in parsed if p.target}
    all_final_columns = [sanitize_func_name(p.target).upper() for p in parsed if p.target]

    lines = []
    lines.append('"""Auto-generated pipeline script."""')
    lines.append("import polars as pl")
    lines.append(f"from {formulas_module_name} import *")
    lines.append("")
    lines.append("# Example input DataFrame (replace with real data ingestion)")
    lines.append("pdf = pl.DataFrame({")
    lines.append('    "DATE": pl.date_range(pl.date(2018,1,1), pl.date(2023,12,31), "1mo", eager=True),')
    lines.append('    "A_IN": [i*1.1 for i in range(72)],')
    lines.append('    "B_IN": [i*1.2+5 for i in range(72)],')
    lines.append("})")
    lines.append("")
    lines.append("# Computation levels")
    for idx, level in enumerate(levels, start=1):
        level_non_convert = [t for t in level if t not in convert_targets]
        if not level_non_convert:
            continue
        lines.append(f"# --- Level {idx}: {', '.join(level_non_convert)} ---")
        cols_to_add = []
        for t in level_non_convert:
            formula = formulas_by_target[t]
            fn_name = sanitize_func_name(t).upper()
            if formula.type == "simple":
                deps = sorted(set(formula.refs))
                arg_exprs = [f'pl.col("{sanitize_func_name(r).upper()}")' for r in deps]
                cols_to_add.append(f"    {fn_name}({', '.join(arg_exprs)}).alias('{fn_name}')")
            elif formula.type == "mchain":
                pair_str = ", ".join([
                    f'({"-" if op=="-" else ""}pl.col(\'{sanitize_func_name("p"+var).upper()}\'), pl.col(\'{sanitize_func_name(var).upper()}\'))'
                    for op, var in formula.terms
                ])
                cols_to_add.append(
                    f"    CHAIN(price_quantity_pairs=[{pair_str}], date_col=pl.col('DATE'), year='{formula.year}').alias('{fn_name}')"
                )
            elif formula.type == "fishvol":
                pairs_str = ", ".join([
                    f"(pl.col('{sanitize_func_name(s1).upper()}'), pl.col('{sanitize_func_name(s2).upper()}'))"
                    for s1, s2 in formula.pairs
                ])
                base = f"FISHVOL(series_pairs=[{pairs_str}], date_col=pl.col('DATE'), rebase_year={formula.year})"
                if formula.trailing_op:
                    base = f"({base}{formula.trailing_op})"
                cols_to_add.append(f"    {base}.alias('{fn_name}')")
        if cols_to_add:
            lines.append("pdf = pdf.with_columns([")
            lines.extend(cols_to_add)
            lines.append("])")
            lines.append("")

    if convert_groups:
        lines.append("# --- Frequency Conversions ---")
        for freq_code, formulas_list in convert_groups.items():
            lines.append(f"# Conversions targeting {freq_code}")
            df_name = f"{freq_code}_convert_df"
            for i, formula in enumerate(formulas_list):
                target = formula.target
                source_series = formula.refs[0]
                to_freq, tech, obs = formula.params
                alias_target = sanitize_func_name(target).upper()
                alias_source = sanitize_func_name(source_series).upper()
                call_str = (
                    f"CONVERT(pdf.select(['DATE','{alias_source}']), "
                    f"as_freq='{current_as_freq}', "
                    f"to_freq='{freq_map.get(to_freq,to_freq)}', "
                    f"technique='{tech_map.get(tech,tech)}', "
                    f"observed='{obs_map.get(obs,obs)}')"
                )
                rename_str = f".rename({{'{alias_source}':'{alias_target}'}})"
                if i == 0:
                    lines.append(f"{df_name} = {call_str}{rename_str}")
                else:
                    lines.append(f"temp_df = {call_str}{rename_str}")
                    lines.append(f"{df_name} = {df_name}.join(temp_df, on='DATE', how='full').drop('DATE_right')")
            lines.append(f"pdf = pdf.join_asof({df_name}, on='DATE')")
            lines.append("")

    rename_map = {sanitize_func_name(k).upper(): v.upper() for k, v in original_case_map.items()}
    lines.append("# Final formatting")
    lines.append(f"final_result = (pdf.select(['DATE'] + {repr(all_final_columns)})"
                 f".rename({repr(rename_map)})"
                 ".melt(id_vars='DATE', variable_name='TIME_SERIES_NAME', value_name='VALUE')"
                 ".sort(['TIME_SERIES_NAME','DATE'])"
                 ".with_columns(SOURCE_SCRIPT_NAME=pl.lit('FAME_INPUT'))"
                 ")")
    lines.append("print(final_result)")
    return "\n".join(lines)