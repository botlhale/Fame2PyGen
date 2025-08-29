"""
┌───────────────────────────────────────────┐
│            Fame2PyGen                     │
│         FAME → Python                     │
│      Main Script Generator                │
└───────────────────────────────────────────┘
"""
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional

import polars as pl  # only used in generated code strings
import formulagen


def variable_to_function_name(var: str) -> str:
    """
    Convert a variable name to a valid Python function name for formulas.py.
    - 'a'   -> 'A'
    - 'a$'  -> 'A_'
    - 'pa$' -> 'PA_'
    """
    if var.endswith('$'):
        return var[:-1].upper() + '_'
    return var.upper()


def preprocess_commands(fame_script: str):
    """
    Expand alias definitions and simple 'loop ... end loop' constructs into flat commands,
    then parse them into structured commands via formulagen.
    """
    lines = fame_script.strip().split('\n')
    alias_dict: Dict[str, List[str]] = {}
    expanded_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'([a-zA-Z0-9_]+)\s*=\s*\{(.+)\}', line)
        if m:
            var = m.group(1)
            items = [item.strip() for item in m.group(2).split(',')]
            alias_dict[var] = items
            i += 1
            continue
        expanded_lines.append(line)
        i += 1

    def resolve_alias(alias: str) -> List[str]:
        if alias not in alias_dict:
            return [alias]
        result: List[str] = []
        for item in alias_dict[alias]:
            if item in alias_dict:
                result.extend(resolve_alias(item))
            else:
                result.append(item)
        return result

    for k in list(alias_dict.keys()):
        alias_dict[k] = resolve_alias(k)

    final_cmds: List[str] = []
    i = 0
    while i < len(expanded_lines):
        line = expanded_lines[i]
        loop_m = re.match(r'loop\s+([a-zA-Z0-9_]+)\s+as\s+([a-zA-Z0-9_]+):', line)
        if loop_m:
            alias = loop_m.group(1)
            varname = loop_m.group(2)
            block: List[str] = []
            i += 1
            while i < len(expanded_lines) and not expanded_lines[i].startswith('end loop'):
                block.append(expanded_lines[i])
                i += 1
            # Expand loop bodies with alias items
            for item in alias_dict.get(alias, []):
                for b in block:
                    final_cmds.append(b.replace(varname, item))
            i += 1
            continue
        final_cmds.append(line)
        i += 1

    parsed = [formulagen.parse_command(cmd) for cmd in final_cmds if formulagen.parse_command(cmd)]
    return parsed, alias_dict


def _collect_targets(parsed_commands: List[Dict]) -> Set[str]:
    return {c['target'] for c in parsed_commands if c and 'target' in c}


def _build_graph(parsed_commands: List[Dict]):
    targets = _collect_targets(parsed_commands)
    adj: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = defaultdict(int)
    for node in targets:
        indeg[node] = 0
    for c in parsed_commands:
        if not c or 'target' not in c:
            continue
        tgt = c['target']
        for ref in c.get('refs', []):
            if ref in targets:
                adj[ref].append(tgt)
                indeg[tgt] += 1
    return adj, indeg


def _topo_levels(adj: Dict[str, List[str]], indeg: Dict[str, int]) -> List[List[str]]:
    q = deque([n for n, d in indeg.items() if d == 0])
    levels: List[List[str]] = []
    visited = 0
    while q:
        level = sorted(list(q))
        levels.append(level)
        curr = list(q)
        q.clear()
        for n in curr:
            visited += 1
            for m in sorted(adj.get(n, [])):
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
    if visited != len(indeg):
        cycl = [n for n, d in indeg.items() if d > 0]
        raise ValueError(f"Cycle detected among nodes: {cycl}")
    return levels


def _closure_from_roots(adj: Dict[str, List[str]], roots: List[str]) -> Set[str]:
    seen: Set[str] = set()
    dq = deque(roots)
    while dq:
        n = dq.popleft()
        if n in seen:
            continue
        seen.add(n)
        for m in adj.get(n, []):
            dq.append(m)
    return seen


def partition_groups(parsed: List[Dict]):
    """
    Partition the dependency graph into:
      - main group: everything not dominated by fishvol or convert roots
      - per-FISHVOL groups: closure starting at each fishvol root
      - per-CONVERT groups: closure of each convert target, grouped by to_freq

    Returns:
      group_for_node: mapping target -> group label ('fishvol::<root>' or 'convert::<to_freq>')
      fishvol_roots: list of fishvol target names
      fishvol_closures: mapping fishvol root -> set of nodes in its closure
      convert_meta: mapping to_freq ('m','q','a') -> list of convert nodes (dicts)
      main_nodes: list of targets in main group
      adj: adjacency list of dependency graph
    """
    fishvol_roots = [c['target'] for c in parsed if c.get('type') == 'fishvol']
    convert_nodes = [c for c in parsed if c.get('type') == 'convert']

    convert_groups: Dict[str, List[str]] = defaultdict(list)
    convert_meta: Dict[str, List[Dict]] = defaultdict(list)
    for c in convert_nodes:
        to_freq = c['params'][0].lower()  # 'm','q','a'
        convert_groups[to_freq].append(c['target'])
        convert_meta[to_freq].append(c)

    adj, indeg = _build_graph(parsed)
    fishvol_closures: Dict[str, Set[str]] = {}
    for fr in fishvol_roots:
        fishvol_closures[fr] = _closure_from_roots(adj, [fr])

    group_for_node: Dict[str, str] = {}
    for fr, nodes in fishvol_closures.items():
        for n in nodes:
            group_for_node[n] = f'fishvol::{fr}'
    for to_freq, nodes in convert_groups.items():
        for n in _closure_from_roots(adj, nodes):
            group_for_node.setdefault(n, f'convert::{to_freq}')

    all_targets = _collect_targets(parsed)
    main_nodes = sorted([t for t in all_targets if t not in group_for_node])
    return group_for_node, fishvol_roots, fishvol_closures, convert_meta, main_nodes, adj


def _nodes_in_levels(levels: List[List[str]], filter_set: Set[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for lvl in levels:
        picked = [n for n in lvl if n in filter_set]
        if picked:
            out.append(picked)
    return out


def generate_formulas_py_string(parsed_cmds: List[Dict]) -> str:
    """
    Generate formulas.py source as a string.
    - Functions return unaliased pl.Expr.
    - Function names match sanitized target names (variable_to_function_name).
    - References inside expressions use the original target column names (e.g., 'a$', 'a').
    """
    header = [
        '"""',
        '┌───────────────────────────────────────────┐',
        '│            Fame2PyGen                     │',
        '│         FAME → Python                     │',
        '│     Auto-Generated Formulas              │',
        '└───────────────────────────────────────────┘',
        '',
        'This file was automatically generated by Fame2PyGen',
        'Contains individual formula functions for FAME script conversion',
        '"""',
        'import polars as pl',
        'from typing import List, Tuple',
        '',
    ]
    bodies: List[str] = []

    def expr_to_polars(expr: str, refs: List[str]) -> str:
        # Replace refs with pl.col("ref") using original ref token (case/punctuation preserved)
        expr_py = expr
        # Sort by length desc to avoid partial replacements
        for r in sorted(set(refs), key=len, reverse=True):
            expr_py = re.sub(rf'\\b{re.escape(r)}\\b', f'pl.col("{r}")', expr_py)
        return expr_py

    for cmd in parsed_cmds:
        if not cmd or 'target' not in cmd:
            continue
        target = cmd['target']
        fn_name = variable_to_function_name(target)
        if cmd['type'] == 'simple':
            expr_py = expr_to_polars(cmd['expr'], cmd.get('refs', []))
            body = [
                f"def {fn_name}() -> pl.Expr:",
                '    """',
                f"    Compute Polars expression for target '{target}'.",
                '    """',
                f"    return ({expr_py})",
                '',
            ]
            bodies.extend(body)
        elif cmd['type'] == 'mchain':
            # Build price-quantity tuples based on naming convention p<var> for prices (example heuristic)
            pairs = []
            for op, var in cmd.get('terms', []):
                # '+' or '-' op can be encoded by multiplying quantity by +1/-1
                sign = '-' if op == '-' else '+'
                q = f"(pl.col('{var}'))"
                p = f"(pl.col('p{var}'))"  # heuristic for price series name; adjust as needed
                if sign == '-':
                    q = f"(-1 * {q})"
                pairs.append(f"({p}, {q})")
            pairs_str = ", ".join(pairs) if pairs else ""
            body = [
                f"def {fn_name}() -> pl.Expr:",
                '    """',
                f"    CHAIN over multiple (price, quantity) pairs for '{target}'.",
                '    """',
                f"    return ple.chain([ {pairs_str} ], date_col=pl.col('date'))",
                '',
            ]
            bodies.extend(body)
        elif cmd['type'] == 'fishvol':
            # pairs based on refs order: (quantity, price)
            v, p = cmd['refs'][0], cmd['refs'][1]
            base_year = int(cmd['year'])
            trailing = cmd.get('trailing_op', '')
            inner = f"ple.fishvol(series_pairs=[(pl.col('{v}'), pl.col('{p}'))], date_col=pl.col('date'), rebase_year={base_year})"
            expr = f"({inner}{trailing})" if trailing else inner
            body = [
                f"def {fn_name}() -> pl.Expr:",
                '    """',
                f"    FISHVOL expression for '{target}'.",
                '    """',
                f"    return {expr}",
                '',
            ]
            bodies.extend(body)
        elif cmd['type'] == 'convert':
            src = cmd['refs'][0]
            to_freq, technique, observed = cmd['params']
            # This returns a placeholder; real conversion occurs in pipeline subframes
            body = [
                f"def {fn_name}() -> pl.Expr:",
                '    """',
                f"    Placeholder expression for convert target '{target}'.",
                '    """',
                f"    return pl.col('{src}')",
                '',
            ]
            bodies.extend(body)
        else:
            # Default: no-op expression referencing target if needed
            body = [
                f"def {fn_name}() -> pl.Expr:",
                '    """',
                f"    Default placeholder for '{target}'.",
                '    """',
                f"    return pl.col('{target}')",
                '',
            ]
            bodies.extend(body)

    return "\n".join(header + bodies)


def generate_pipeline_code(parsed_cmds: List[Dict]) -> str:
    """
    Generate the convpy4rmfame-like pipeline:
      - Main df computations (no filtering)
      - For each FISHVOL root, a df_fv_<root> filtered from Jan 1 of base year and its dependents
      - For each CONVERT target frequency, a df_cv_<freq> for conversions and their dependents
      - Melt each and union at the end
    """
    group_for_node, fishvol_roots, fishvol_closures, convert_meta, main_nodes, adj = partition_groups(parsed_cmds)

    adj_all, indeg_all = _build_graph(parsed_cmds)
    levels = _topo_levels(adj_all, indeg_all)

    def nodes_in_levels(filter_set: Set[str]) -> List[List[str]]:
        return _nodes_in_levels(levels, filter_set)

    code = []
    code.append('import polars as pl')
    code.append('from formulas import *')
    code.append('import ple')
    code.append('')
    code.append("# Base DataFrame (unfiltered)")
    code.append("df = pl.DataFrame({")
    code.append("    'date': pl.date_range(pl.date(2019, 1, 1), pl.date(2025, 1, 1), '1mo', eager=True),")
    code.append("    'v123': pl.Series('v123', range(1, 74)),")
    code.append("    'v143': pl.Series('v143', range(1, 74)),")
    code.append("    'prices': pl.Series('prices', range(1, 74)),")
    code.append("    'volumes': pl.Series('volumes', range(1, 74)),")
    code.append("})")
    code.append("")

    # MAIN PIPELINE
    if main_nodes:
        code.append("# ---- MAIN PIPELINE (no date filtering) ----")
        for idx, lvl in enumerate(nodes_in_levels(set(main_nodes))):
            code.append(f"# --- Level {idx+1}: {', '.join(lvl)}")
            cols = []
            for t in lvl:
                fn = variable_to_function_name(t)
                cols.append(f"{fn}().alias('{t}')")
            code.append("df = df.with_columns([")
            code.append("    " + ",\n    ".join(cols))
            code.append("])")
            code.append("")

    # FISHVOL PIPELINES
    for fr in fishvol_roots:
        closure = fishvol_closures[fr]
        fr_year = None
        for c in parsed_cmds:
            if c.get('type') == 'fishvol' and c.get('target') == fr:
                fr_year = int(c['year'])
                break
        year_guard = fr_year or 1900
        code.append(f"# ---- FISHVOL SUB-PIPELINE for root '{fr}' (filtered from {year_guard}-01-01) ----")
        code.append(f"df_fv_{fr} = df.filter(pl.col('date') >= pl.date({year_guard}, 1, 1))")
        for idx, lvl in enumerate(nodes_in_levels(set(closure))):
            code.append(f"# --- FV Level {idx+1}: {', '.join(lvl)}")
            cols = []
            for t in lvl:
                fn = variable_to_function_name(t)
                cols.append(f"{fn}().alias('{t}')")
            code.append(f"df_fv_{fr} = df_fv_{fr}.with_columns([")
            code.append("    " + ",\n    ".join(cols))
            code.append("])")
            code.append("")

    # CONVERT PIPELINES
    for to_freq, metas in convert_meta.items():
        code.append(f"# ---- CONVERT SUB-PIPELINE to '{to_freq}' ----")
        code.append(f"df_cv_{to_freq} = None")
        every = {'m': '1mo', 'q': '1q', 'a': '1y'}.get(to_freq, to_freq)
        for meta in metas:
            tgt = meta['target']
            src = meta['refs'][0]
            # Keep conversions separate; simple example using mean aggregation
            code.append(f"_cv_tmp = (df.select(['date', '{src}'])"
                        f".group_by_dynamic('date', every='{every}')"
                        f".agg(pl.col('{src}').mean().alias('{tgt}')))")
            code.append(f"df_cv_{to_freq} = _cv_tmp if df_cv_{to_freq} is None else df_cv_{to_freq}.join(_cv_tmp, on='date', how='full').select(pl.all().unique())")
        code.append("")

    # FINAL MELT + UNION
    main_cols = [t for t in main_nodes]
    code.append("# ---- FINAL MELT AND UNION ----")
    if main_cols:
        code.append(f"main_long = df.select(['date'{''.join([', ' + repr(c) for c in main_cols])}]).melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE')")
    else:
        code.append("main_long = pl.DataFrame({'date': [], 'TIME_SERIES_NAME': [], 'VALUE': []})")
    fv_names = []
    for fr in fishvol_roots:
        fv_names.append(f"fv_long_{fr}")
        code.append(f"fv_long_{fr} = df_fv_{fr}.select(pl.all().exclude(['v123','v143','prices','volumes']))"
                    f".melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE')")
    cv_names = []
    for to_freq in convert_meta.keys():
        cv_names.append(f"cv_long_{to_freq}")
        code.append(f"cv_long_{to_freq} = (df_cv_{to_freq} if df_cv_{to_freq} is not None else pl.DataFrame({{'date':[]}}))")
        code.append(f"cv_long_{to_freq} = cv_long_{to_freq}.melt(id_vars='date', variable_name='TIME_SERIES_NAME', value_name='VALUE') if df_cv_{to_freq} is not None else cv_long_{to_freq}")
    concat_sources = ['main_long'] + fv_names + cv_names
    code.append(f"final_long = pl.concat([{', '.join(concat_sources)}], how='vertical_relaxed')")
    code.append("print(final_long)")
    return '\n'.join(code)


def write_formulas_py(parsed_cmds: List[Dict], path: str = "formulas.py"):
    src = generate_formulas_py_string(parsed_cmds)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def write_pipeline_py(parsed_cmds: List[Dict], path: str = "convpy4rmfame.py"):
    src = generate_pipeline_code(parsed_cmds)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
