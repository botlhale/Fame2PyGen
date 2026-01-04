"""
formulas_generator: helpers to parse FAME-like formulas and render Polars expressions.

Responsibilities:
 - Robust parsing of nested IF statements and function calls.
 - Handling of FAME specific operators (EQ, NE, etc.) and keywords (ND, NA, T).
 - Support for complex functions like LSUM, NLRX, CHAIN, DATEOF.
 - Generation of helper wrapper definitions via generate_polars_functions.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# ---------- Constants ----------

FUNCTION_NAMES = {
    "pct", "convert", "fishvol_rebase", "chain", "mchain", "sqrt", 
    "nlrx", "lsum", "firstvalue", "lastvalue", "dateof", "exists",
    "make", "date", "diff", "ave"
}

LOGICAL_OPERATORS = {"or", "and", "not"}

COMPARISON_MAP = {
    ' eq ': ' == ', ' ne ': ' != ', ' gt ': ' > ', ' lt ': ' < ', ' ge ': ' >= ', ' le ': ' <= '
}

KEYWORDS_TO_SKIP = {
    "if", "then", "else", 
    "monthly", "quarterly", "annual", "weekly", "daily",
    "scalar"
}

FUNCTION_KEYWORDS = FUNCTION_NAMES | LOGICAL_OPERATORS | KEYWORDS_TO_SKIP

# ---------- Utilities ----------

def sanitize_func_name(name: Optional[str]) -> str:
    if name is None: return ""
    s = str(name)
    # Handle local DB syntax: gg'abc -> gg_abc
    s = s.replace("'", "_").replace("$", "_")
    # Preserve dots in column names
    s = re.sub(r"[^A-Za-z0-9_.]", "", s)
    return s.lower()

def normalize_formula_text(formula: Optional[str]) -> str:
    if formula is None: return ""
    s = str(formula)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    # Normalize time indices [T] -> [t]
    s = re.sub(r"\[\s*T\s*([+-]?\d*)\s*\]", r"[t\1]", s, flags=re.IGNORECASE)
    return s

def split_args_balanced(text: str) -> List[str]:
    args = []
    current_arg = []
    depth = 0
    quote_char = None
    
    for char in text:
        if quote_char:
            current_arg.append(char)
            if char == quote_char: quote_char = None
        else:
            if char in ('"', "'"):
                quote_char = char
                current_arg.append(char)
            elif char == '(': depth += 1; current_arg.append(char)
            elif char == ')': depth -= 1; current_arg.append(char)
            elif char == ',' and depth == 0:
                args.append("".join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
    if current_arg: args.append("".join(current_arg).strip())
    return [a for a in args if a]

def extract_if_components(text: str) -> Optional[Dict[str, str]]:
    if not re.match(r"^\s*if\b", text, re.IGNORECASE): return None
    tokens = re.split(r'(\s+|[()])', text)
    CONDITION, THEN_BLOCK, ELSE_BLOCK = 0, 1, 2
    state = CONDITION
    depth_if = 0
    depth_paren = 0
    parts = {CONDITION: [], THEN_BLOCK: [], ELSE_BLOCK: []}
    start_found = False
    
    for tok in tokens:
        clean_tok = tok.strip().lower()
        if not clean_tok: 
            if state in parts: parts[state].append(tok)
            continue
        if clean_tok == '(': depth_paren += 1; parts[state].append(tok); continue
        if clean_tok == ')': depth_paren -= 1; parts[state].append(tok); continue

        if not start_found:
            if clean_tok == 'if': start_found = True
            continue

        if depth_paren == 0:
            if clean_tok == 'if': depth_if += 1; parts[state].append(tok)
            elif clean_tok == 'then' and depth_if == 0 and state == CONDITION: state = THEN_BLOCK
            elif clean_tok == 'else' and depth_if == 0 and state == THEN_BLOCK: state = ELSE_BLOCK
            else: parts[state].append(tok)
        else: parts[state].append(tok)

    if state != ELSE_BLOCK: return None
    return {
        "condition": "".join(parts[CONDITION]).strip(),
        "then_expr": "".join(parts[THEN_BLOCK]).strip(),
        "else_expr": "".join(parts[ELSE_BLOCK]).strip()
    }

def parse_time_index(token: str) -> Tuple[str, int]:
    if not token: return "", 0
    t = token.strip()
    # Allow ' in names
    m = re.match(r"^\s*([A-Za-z0-9_$.']+)\s*(?:\[\s*t\s*([+-]?\d+)\s*\])?\s*$", t, re.IGNORECASE)
    if not m:
        m_dot = re.match(r"^\s*([A-Za-z0-9_$.']+)\s*$", t)
        if m_dot: return m_dot.group(1), 0
        return t, 0
    base = m.group(1)
    offs_raw = m.group(2)
    return base, (int(offs_raw) if offs_raw else 0)

def parse_date_index(token: str) -> Tuple[Optional[str], Optional[str]]:
    if not token: return None, None
    t = token.strip()
    m = re.match(r'^\s*([A-Za-z0-9_$.]+)\s*\[\s*["\']([^"\']+)["\']\s*\]\s*$', t)
    if m: return m.group(1), m.group(2)
    return None, None

def parse_dynamic_lookup(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Detects VAR[SCALAR_VAR] pattern."""
    if not token: return None, None
    m = re.match(r"^\s*([A-Za-z0-9_$.']+)\s*\[\s*([A-Za-z0-9_$.']+)\s*\]\s*$", token.strip())
    if m:
        base = m.group(1)
        idx = m.group(2)
        # Verify it's not a standard time index
        if idx.lower() == 't': return None, None 
        if re.match(r"t[+-]\d+", idx.lower()): return None, None
        return base, idx
    return None, None

TOKEN_RE = re.compile(r'[A-Za-z0-9_$.']+(?:\s*\[\s*(?:t\s*[+-]?\d+|["\'][^"\']+["\']|[A-Za-z0-9_$.']+\s*)\])?', re.IGNORECASE)

def _is_strict_number(tok: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", tok.strip()))

def _is_numeric_literal(tok: str) -> bool:
    if not _is_strict_number(tok): return False
    digits = tok.lstrip("+-").split(".", 1)[0]
    return len(digits) <= 3

def _token_to_pl_expr(tok: str) -> str:
    if tok.upper() in ('ND', 'NA', 'NC'): return "pl.lit(None)"
    # Map standalone T to DATE
    if tok.upper() == 'T': return 'pl.col("DATE")'
    if tok.startswith('"') or tok.startswith("'") and not re.match(r"[A-Za-z0-9]+'[A-Za-z0-9]+", tok): return tok
    
    # Check for Dynamic Lookup (Series[Scalar])
    base_dyn, idx_dyn = parse_dynamic_lookup(tok)
    if base_dyn:
        return f"__LOOKUP__:{base_dyn}:{idx_dyn}"

    base, offs = parse_time_index(tok)
    if base == "": return "pl.lit(None)"
    if _is_strict_number(base):
        if _is_numeric_literal(base): return base
    
    col = sanitize_func_name(base).upper()
    expr = f'pl.col("{col}")'
    if offs != 0: expr = f"{expr}.shift({-offs})"
    return expr

def _render_chain_calls(expr: str, date_col_name: str = "DATE") -> str:
    if not expr: return expr
    out_parts: List[str] = []
    idx = 0
    while True:
        m = re.search(r"\$(mchain|chain)\s*\(", expr[idx:], re.IGNORECASE)
        if not m: out_parts.append(expr[idx:]); break
        start = idx + m.start()
        out_parts.append(expr[idx:start])
        j = start + len(m.group(0))
        depth = 1
        while j < len(expr) and depth > 0:
            if expr[j] == "(": depth += 1
            elif expr[j] == ")": depth -= 1
            j += 1
        if depth != 0: out_parts.append(expr[start:j]); idx = j; continue
        inner = expr[start + len(m.group(0)) : j - 1].strip()
        parsed = re.match(r'^\s*"(.*?)"\s*,\s*"(.*?)"\s*$', inner, re.DOTALL)
        if not parsed: out_parts.append(expr[start:j]); idx = j; continue
        expr_str, year_raw = parsed.groups()
        year_m = re.search(r"(\d{4})", year_raw)
        year = year_m.group(1) if year_m else year_raw.strip()
        parts = re.split(r"([+-])", expr_str)
        parts = [p.strip() for p in parts if p.strip()]
        if parts and parts[0] not in ("+", "-"): parts.insert(0, "+")
        terms = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
        pair_items = []
        for op, var in terms:
            sign = "-" if op == "-" else ""
            pcol = sanitize_func_name("p" + var).upper()
            vcol = sanitize_func_name(var).upper()
            if sign: pair_items.append(f"(-pl.col('{pcol}'), pl.col('{vcol}'))")
            else: pair_items.append(f"(pl.col('{pcol}'), pl.col('{vcol}'))")
        pairs_str = ", ".join(pair_items)
        out_parts.append(f"CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col(\"{date_col_name}\"), year=\"{year}\")")
        idx = j
    return "".join(out_parts)

def _build_sub_map_and_placeholders(expr: str, substitution_map: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    placeholders: Dict[str, str] = {}
    parts: List[str] = []
    last = 0
    idx = 0
    for m in TOKEN_RE.finditer(expr):
        s, e = m.span()
        parts.append(expr[last:s])
        tok = m.group(0)
        key = tok.lower()
        
        if key in FUNCTION_KEYWORDS or tok == "__ND_PLACEHOLDER__":
            parts.append(tok); last = e; continue
        if _is_strict_number(tok) and _is_numeric_literal(tok):
            parts.append(tok); last = e; continue

        ph = f"__PH_{idx}__"
        if substitution_map and key in substitution_map:
            placeholders[ph] = substitution_map[key]
        else:
            placeholders[ph] = _token_to_pl_expr(tok)
        
        parts.append(ph)
        idx += 1
        last = e
    parts.append(expr[last:])
    return "".join(parts), placeholders

_shift_pct_re = re.compile(
    r"^\s*([A-Za-z0-9_$.]+)\s*\[\s*t\s*([+-]?\d+)\s*\]\s*/\s*\(\s*1\s*\+\s*\(\s*pct\s*\(\s*([A-Za-z0-9_$.]+)\s*\[\s*t\s*([+-]?\d+)\s*\]\s*\)\s*/\s*100\s*\)\s*\)\s*$", re.IGNORECASE)

def render_conditional_expr(condition: str, then_expr: str, else_expr: str, substitution_map: Optional[Dict[str, str]] = None) -> str:
    def clean_wrap(x):
        x = x.strip()
        while x.startswith('(') and x.endswith(')'):
            if re.match(r'^\(if\s+.+\s+then\s+.+\s+else\s+.+\)$', x, re.IGNORECASE) or "if " in x.lower():
                x = x[1:-1].strip()
            else: break
        return x

    condition = clean_wrap(condition)
    then_expr = clean_wrap(then_expr)
    else_expr = clean_wrap(else_expr)

    def replace_nd(t): return re.sub(r'\bnd\b', '__ND_PLACEHOLDER__', t, flags=re.IGNORECASE)
    def restore_nd(t): return t.replace('__ND_PLACEHOLDER__', 'pl.lit(None)')

    cond_expr = render_polars_expr(condition, substitution_map=substitution_map)
    
    def process_branch(branch):
        branch = replace_nd(branch)
        comps = extract_if_components(branch)
        if comps:
            res = render_conditional_expr(comps["condition"], comps["then_expr"], comps["else_expr"], substitution_map)
        else:
            res = render_polars_expr(branch, substitution_map=substitution_map)
        return restore_nd(res)

    then_polars = process_branch(then_expr)
    else_polars = process_branch(else_expr)
    
    return f"pl.when({cond_expr}).then({then_polars}).otherwise({else_polars})"

def render_polars_expr(rhs: str, substitution_map: Optional[Dict[str, str]] = None, memory: Optional[Dict[str, str]] = None, ctx: Optional[Dict[str, bool]] = None) -> str:
    if rhs is None: return ""
    sub_map = substitution_map if substitution_map is not None else (memory if memory is not None else {})
    expr = normalize_formula_text(rhs)
    
    # 0) Robust Nested IF check
    while expr.startswith('(') and expr.endswith(')') and "if" in expr.lower():
        depth = 0
        is_wrapped = True
        for i, c in enumerate(expr):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth == 0 and i < len(expr) - 1: is_wrapped = False; break
        if is_wrapped: expr = expr[1:-1].strip()
        else: break

    comps = extract_if_components(expr)
    if comps:
         return render_conditional_expr(comps["condition"], comps["then_expr"], comps["else_expr"], sub_map)

    # 1) Global Operator Cleaning
    expr_padded = f" {expr} "
    for fame_op, py_op in COMPARISON_MAP.items():
        expr_padded = re.sub(fame_op, py_op, expr_padded, flags=re.IGNORECASE)
    expr = expr_padded.strip()

    # 2) Inline Chains
    expr = _render_chain_calls(expr)

    # 3) Shift PCT
    msp = _shift_pct_re.match(expr)
    if msp:
        ser1, offs1, ser2, offs2 = msp.group(1), int(msp.group(2)), msp.group(3), int(msp.group(4))
        if offs1 == offs2:
            if ctx is not None: ctx["need_shiftpct"] = True
            return f"__SHIFT_PCT__:{ser1}:{ser2}:{offs1}"

    # 4) LSUM
    m_lsum = re.match(r"^\s*lsum\s*\((.+)\)\s*$", expr, re.IGNORECASE)
    if m_lsum:
        if ctx is not None: ctx["has_lsum"] = True
        inner_args = split_args_balanced(m_lsum.group(1))
        rendered_args = []
        for arg in inner_args:
            rendered_args.append(render_polars_expr(arg, sub_map, ctx=ctx))
        return f"LSUM([{', '.join(rendered_args)}])"

    # 5) PRE-TOKENIZATION Function Handling
    def process_generic_func(text, func_name, template_fn):
        out_parts = []
        i = 0
        pattern = re.compile(rf"\b{func_name}\s*\(", re.IGNORECASE)
        while True:
            m = pattern.search(text[i:])
            if not m: out_parts.append(text[i:]); break
            s_pos = i + m.start()
            out_parts.append(text[i:s_pos])
            j = s_pos + len(m.group(0))
            depth = 1
            inner_start = j
            while j < len(text) and depth > 0:
                if text[j] == "(": depth += 1
                elif text[j] == ")": depth -= 1
                j += 1
            if depth != 0: out_parts.append(text[s_pos:]); i = len(text); break
            inner = text[inner_start : j - 1].strip()
            out_parts.append(template_fn(inner))
            i = j
        return "".join(out_parts)

    def dateof_templ(inner):
        args = split_args_balanced(inner)
        if len(args) >= 3:
            suffix1 = re.sub(r'[^A-Z0-9]', '', args[-2].upper())
            suffix2 = re.sub(r'[^A-Z0-9]', '', args[-1].upper())
            if ctx is not None:
                if "dateof_variants" not in ctx: ctx["dateof_variants"] = set()
                ctx["dateof_variants"].add((suffix1, suffix2))
            return f"DATE_{suffix1}_{suffix2}"
        else:
            return f"DATEOF_GENERIC({render_polars_expr(inner, sub_map, memory, ctx)})"

    expr = process_generic_func(expr, "dateof", dateof_templ)
    
    def make_templ(inner):
        args = split_args_balanced(inner)
        if len(args) >= 2 and "date(" in args[0].lower() and (args[1].startswith('"') or args[1].startswith("'")):
            date_str = args[1].strip('"\'')
            safe_suffix = re.sub(r'[^A-Z0-9]', '_', date_str)
            if ctx is not None:
                if "make_variants" not in ctx: ctx["make_variants"] = set()
                ctx["make_variants"].add((safe_suffix, date_str))
            return f"DATE_{safe_suffix}"
        return f"MAKE({render_polars_expr(inner, sub_map, memory, ctx)})"

    expr = process_generic_func(expr, "make", make_templ)
    expr = process_generic_func(expr, "date", lambda x: f"DATE({x})") 

    # 6) Tokenization
    expr_with_ph, placeholders = _build_sub_map_and_placeholders(expr, substitution_map=sub_map)

    # 7) Process Standard Functions
    combined = expr_with_ph

    def pct_templ(inner):
        args = split_args_balanced(inner)
        if len(args) > 1: return f"PCT({args[0]}, offset={args[1]})"
        return f"PCT({inner})"
    
    combined = process_generic_func(combined, "pct", pct_templ)
    combined = process_generic_func(combined, "sqrt", lambda x: f"SQRT({x})")
    combined = process_generic_func(combined, "diff", lambda x: f"{x}.diff()")
    combined = process_generic_func(combined, "ave", lambda x: f"{x}.mean()")
    combined = process_generic_func(combined, "firstvalue", lambda x: f"{x}.first()")
    combined = process_generic_func(combined, "lastvalue", lambda x: f"{x}.last()")
    combined = process_generic_func(combined, "exists", lambda x: f"{x}.is_not_null()")

    def convert_templ(inner):
        args = [arg.strip() for arg in inner.split(",")]
        if not args: return "CONVERT()"
        series_expr = args[0]
        param_args = []
        for arg in args[1:]:
            if arg in placeholders:
                m_col = re.match(r'pl\.col\("([^"]+)"\)', placeholders[arg], re.IGNORECASE)
                if m_col: param_args.append(f'"{m_col.group(1).lower()}"')
                else: param_args.append(f'"{arg}"') 
            else: param_args.append(f'"{arg}"')
        return f"CONVERT({series_expr}, {', '.join(param_args)})"
    
    combined = process_generic_func(combined, "convert", convert_templ)

    # 8) Logical Operators
    combined = re.sub(r'\bor\b', '|', combined, flags=re.IGNORECASE)
    combined = re.sub(r'\band\b', '&', combined, flags=re.IGNORECASE)
    combined = re.sub(r'\bnot\b', '~', combined, flags=re.IGNORECASE)

    # 9) Final Placeholder Swap
    final = combined
    for ph, repl in placeholders.items():
        final = final.replace(ph, repl)
    return final

# ---------- Top Level Parsing ----------

def parse_fame_formula(line: str) -> Optional[Dict]:
    if line is None: return None
    s = normalize_formula_text(line)

    ch = _parse_chain_top_level(s)
    if ch: return ch

    if "nlrx(" in s.lower() and "=" in s:
        clean_s = s[4:].strip() if s.lower().startswith("set ") else s
        target, rhs = clean_s.split("=", 1)
        m_nlrx = re.match(r"nlrx\s*\((.+)\)", rhs.strip(), re.IGNORECASE)
        if m_nlrx:
            args = [a.strip() for a in split_args_balanced(m_nlrx.group(1))]
            return {"type": "nlrx", "target": target.strip(), "args": args, "refs": []}

    m_inline_date = re.match(r"^\s*set\s+<date\s+(.+?)\s+to\s+(.+?)>\s*([A-Za-z0-9_$.']+)\s*=\s*(.+)$", s, re.IGNORECASE)
    if m_inline_date:
        start, end, tgt, rhs = m_inline_date.groups()
        refs = [t for t in TOKEN_RE.findall(rhs) if t.lower() not in FUNCTION_KEYWORDS]
        return {"type": "simple", "target": tgt.strip(), "rhs": rhs.strip(), "refs": refs, "date_filter": {"start": start.strip(), "end": end.strip()}}

    m_list = re.match(r"^\s*([A-Za-z0-9_$.]+)\s*=\s*\{(.+)\}\s*$", s)
    if m_list:
        target, content = m_list.groups()
        items = [it.strip() for it in content.split(",")]
        return {"type": "list_alias", "target": target, "refs": items}

    # Scalar Assignment
    m_scalar = re.match(r"^\s*scalar\s+([A-Za-z0-9_$.']+)\s*=\s*(.+)$", s, re.IGNORECASE)
    if m_scalar:
        target, rhs = m_scalar.groups()
        refs = [t for t in TOKEN_RE.findall(rhs) if t.lower() not in FUNCTION_KEYWORDS and not _is_strict_number(t)]
        return {"type": "scalar", "target": target.strip(), "rhs": rhs.strip(), "refs": refs}

    m_freq = re.match(r"^\s*freq\s+([A-Za-z0-9]+)\s*$", s, re.IGNORECASE)
    if m_freq: return {"type": "freq", "freq": m_freq.group(1).lower()}

    m_date_all = re.match(r"^\s*date\s+\*\s*$", s, re.IGNORECASE)
    if m_date_all: return {"type": "date", "filter": None}
    
    m_date_range = re.match(r"^\s*date\s+(.+?)\s+to\s+(.+?)\s*$", s, re.IGNORECASE)
    if m_date_range: return {"type": "date", "filter": {"start": m_date_range.group(1).strip(), "end": m_date_range.group(2).strip()}}

    clean_s = s if not s.lower().startswith("set ") else s[4:].strip()
    m_date_assign = re.match(r'^\s*([A-Za-z0-9_$.]+)\s*\[\s*["\']([^"\']+)["\']\s*\]\s*=\s*(.+)\s*$', clean_s)
    if not m_date_assign:
         m_date_assign = re.match(r'^\s*([A-Za-z0-9_$.]+)\s*\[\s*(\d{1,2}[A-Za-z]{3}\d{4}|\d{4}Q[1-4]|\d{4}-\d{2}-\d{2})\s*\]\s*=\s*(.+)\s*$', clean_s, re.IGNORECASE)
    
    if m_date_assign:
        target, date_str, rhs = m_date_assign.groups()
        raw = TOKEN_RE.findall(rhs)
        refs = [t for t in raw if t.lower() != "t" and not _is_strict_number(t)]
        return {"type": "point_in_time_assign", "target": target, "date": date_str, "rhs": rhs.strip(), "refs": refs}

    m_convert = re.match(r"^\s*([A-Za-z0-9_$.]+)\s*=\s*convert\((.+)\)\s*$", clean_s, re.IGNORECASE)
    if m_convert:
        target, args_str = m_convert.groups()
        args = [a.strip().strip("'\"") for a in re.split(r""",\s*(?=(?:[^"']*["'][^"']*["'])*[^"']*$)""", args_str)]
        if len(args) >= 4:
            return {"type": "convert", "target": target, "refs": [args[0]], "params": args}

    m_fv = re.match(r"^\s*([A-Za-z0-9_$.]+)\s*=\s*\$?fishvol_rebase\((.+)\)(\s*[*/+-].*)?\s*$", clean_s, re.IGNORECASE)
    if m_fv:
        target, args_str, trailing = m_fv.groups()
        args = re.findall(r"\{([^}]+)\}|([^{},]+)", args_str)
        cleaned = [t[0] if t[0] else t[1].strip() for t in args]
        year = cleaned[-1]
        list_args = cleaned[:-1]
        if len(list_args) >= 2:
            vols = [v.strip() for v in list_args[0].split(",")]
            prices = [p.strip() for p in list_args[1].split(",")]
            pairs = list(zip(vols, prices))
            variable_refs = [s for s in vols + prices if not _is_strict_number(s)]
            return {"type": "fishvol", "target": target, "refs": variable_refs, "pairs": pairs, "year": year}

    if "=" in clean_s:
        lhs_temp, rhs_temp = clean_s.split("=", 1)
        lhs_temp = lhs_temp.strip(); rhs_temp = rhs_temp.strip()
        
        msp = _shift_pct_re.match(rhs_temp)
        if msp:
            ser1, offs1, ser2, offs2 = msp.group(1), int(msp.group(2)), msp.group(3), int(msp.group(4))
            if offs1 == offs2:
                target_match = re.match(r"^\s*([A-Za-z0-9_$.]+)", lhs_temp)
                target = target_match.group(1) if target_match else lhs_temp
                return {"type": "shift_pct", "target": target, "refs": [ser1, ser2], "ser1": ser1, "ser2": ser2, "offset": offs1}

        comps = extract_if_components(rhs_temp)
        if comps:
             return {"type": "conditional", "target": lhs_temp, "condition": comps["condition"], "then_expr": comps["then_expr"], "else_expr": comps["else_expr"], "refs": []}
        
        refs = [t for t in TOKEN_RE.findall(rhs_temp) if t.lower() not in FUNCTION_KEYWORDS and not _is_strict_number(t)]
        return {"type": "simple", "target": lhs_temp, "rhs": rhs_temp, "refs": refs}
    
    return None

def _parse_chain_top_level(line: str) -> Optional[Dict]:
    m = re.match(r'^\s*([A-Za-z0-9_$.]+)\s*=\s*\$mchain\s*\(\s*"(.*?)"\s*,\s*"\s*(\d{4})\s*"\s*\)\s*$', line, re.IGNORECASE)
    if m:
        target, inner, year = m.groups()
        return {"type": "mchain", "target": target, "terms": [], "year": year}
    return None

def generate_polars_functions(fame_cmds: List[str]) -> Dict[str, str]:
    ctx = defaultdict(bool)
    ctx["dateof_variants"] = set()
    ctx["make_variants"] = set()
    
    for cmd in fame_cmds:
        if not cmd: continue
        s = normalize_formula_text(str(cmd))
        if "=" in s:
            _, rhs = s.split("=", 1)
            render_polars_expr(rhs, ctx=ctx)
        
        if "set " in s.lower() and "<date" not in s.lower(): s = s[4:].strip()
        if re.search(r"\$(chain|mchain)\s*\(", s, re.IGNORECASE): ctx["has_mchain"] = True
        if re.search(r"\bconvert\s*\(", s, re.IGNORECASE): ctx["has_convert"] = True
        if re.search(r"\bfishvol_rebase\s*\(", s, re.IGNORECASE): ctx["has_fishvol"] = True
        if re.search(r"\bpct\s*\(", s, re.IGNORECASE): ctx["has_pct"] = True
        if re.search(r"\bsqrt\s*\(", s, re.IGNORECASE): ctx["has_sqrt"] = True
        if re.search(r"\bnlrx\s*\(", s, re.IGNORECASE): ctx["has_nlrx"] = True
        if re.search(r"\blsum\s*\(", s, re.IGNORECASE): ctx["has_lsum"] = True
        
        if re.match(r'^\s*[A-Za-z0-9_$.]+\s*\[\s*["\']', s) or re.match(r'^\s*[A-Za-z0-9_$.]+\s*\[\s*(\d{4}|12)', s, re.IGNORECASE):
            ctx["need_point_in_time_assign"] = True
            
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            if _shift_pct_re.match(rhs.strip()):
                ctx["need_shiftpct"] = True
                lhs_time = re.search(r"\[\s*t\s*([+-]?\d*)\s*\]", lhs)
                rhs_time = re.search(r"\[\s*t\s*\+\s*(\d+)\s*\]", rhs)
                if lhs_time and rhs_time:
                    l_off = int(lhs_time.group(1)) if lhs_time.group(1) and lhs_time.group(1) not in ('+','-') else 0
                    r_off = int(rhs_time.group(1))
                    if l_off == 0 and r_off > 0:
                        ctx["need_shiftpct_backwards"] = True
            
            if all(tok not in rhs.lower() for tok in ["pct(", "$chain", "convert", "fishvol"]):
                parts = re.split(r"([+\-*/])", rhs)
                ops = [p for p in parts if p in "+-*/"]
                if ops:
                    if all(op == "+" for op in ops): ctx["need_add_series"] = True
                    elif all(op == "-" for op in ops): ctx["need_sub_series"] = True
                    elif all(op == "*" for op in ops): ctx["need_mul_series"] = True
                    elif all(op == "/" for op in ops): ctx["need_div_series"] = True
                    elif len(set(ops)) > 1: ctx["need_arith"] = True

    defs = {}
    
    if ctx["has_nlrx"]:
        defs["NLRX"] = "def NLRX(df: pl.DataFrame, lamb: float, y: str, w1: str, w2: str, w3: str, w4: str, gss: str, gpr: str) -> pl.DataFrame:\n    import polars_econ as ple\n    return ple.nlrx(df, lamb, y=y, w1=w1, w2=w2, w3=w3, w4=w4, gss=gss, gpr=gpr)"
    if ctx["has_mchain"]:
        defs["CHAIN"] = "def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:\n    import polars_econ as ple\n    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))"
    if ctx["has_convert"]:
        defs["CONVERT"] = "def CONVERT(series: pl.Expr, *args) -> pl.Expr:\n    import polars_econ as ple\n    return ple.convert(series, *args)"
    if ctx["has_fishvol"]:
        defs["FISHVOL"] = "def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:\n    import polars_econ as ple\n    return ple.fishvol(series_pairs, date_col, rebase_year)"
    if ctx["has_pct"]:
        defs["PCT"] = "def PCT(expr: pl.Expr, offset: int = 1) -> pl.Expr:\n    import polars_econ as ple\n    return ple.pct(expr, offset=offset)"
    if ctx["has_sqrt"]:
        defs["SQRT"] = "def SQRT(expr: pl.Expr) -> pl.Expr:\n    return expr.sqrt()"
    
    if ctx["has_lsum"]:
        defs["LSUM"] = "def LSUM(args: List[pl.Expr]) -> pl.Expr:\n    return pl.sum_horizontal(args).fill_null(0)"
    
    for suffix1, suffix2 in ctx["dateof_variants"]:
        var_name = f"DATE_{suffix1}_{suffix2}"
        if suffix1 == "BEFORE" and suffix2 == "ENDING":
            defs[var_name] = f"{var_name} = pl.lit(date(9999, 12, 31))"
        elif suffix1 == "CONTAIN" and suffix2 == "END":
            defs[var_name] = f"{var_name} = pl.lit(date(9999, 12, 31))"
        else:
            defs[var_name] = f"{var_name} = pl.lit(None) # Undefined DATEOF"

    for safe_suffix, raw_date_str in ctx["make_variants"]:
        m = re.match(r"(\d{4})[:_]?(\d{1,2})", raw_date_str)
        if m:
            y, m_val = m.groups()
            defs[f"DATE_{safe_suffix}"] = f"DATE_{safe_suffix} = pl.lit(date({y}, {int(m_val)}, 1))"
        else:
            m_yr = re.match(r"(\d{4})", raw_date_str)
            if m_yr:
                defs[f"DATE_{safe_suffix}"] = f"DATE_{safe_suffix} = pl.lit(date({m_yr.group(1)}, 1, 1))"
            else:
                defs[f"DATE_{safe_suffix}"] = f"DATE_{safe_suffix} = pl.lit(None) # Unrecognized date fmt: {raw_date_str}"

    if ctx["need_shiftpct"]:
        defs["SHIFT_PCT"] = "def SHIFT_PCT(ser1: pl.Expr, ser2: pl.Expr, offset: int = 1) -> pl.Expr:\n    return (ser1.shift(-offset) / (1 + (PCT(ser2.shift(-offset)).truediv(100))))"
        
    if ctx["need_shiftpct_backwards"]:
        defs["SHIFT_PCT_BACKWARDS"] = '''
def SHIFT_PCT_BACKWARDS(
    df: pl.DataFrame, target_col: str, pct_col: str, start_date: str, end_date: str, offset: int = 1,
    initial_target: float = None, initial_pct: float = None
) -> pl.DataFrame:
    import polars as pl
    from datetime import datetime
    def PCT(expr: pl.Expr, offset: int = offset) -> pl.Expr:
        return ((expr.shift(offset) - expr) / expr) * 100
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if initial_target is None or initial_pct is None:
        next_period = df.filter(pl.col("DATE") > start).sort("DATE").limit(1)
        if not next_period.is_empty():
            if initial_target is None: initial_target = next_period.select(target_col).item()
            if initial_pct is None: initial_pct = next_period.select(pct_col).item()
    work_df = df.filter((pl.col("DATE") >= end) & (pl.col("DATE") <= start)).sort("DATE", descending=True)
    if work_df.is_empty(): return df
    pct_change_expr = pl.when(pl.col("DATE") == start).then(
        pl.when(pl.col(pct_col).is_not_null() & (pl.col(pct_col) != 0) & pl.lit(initial_pct).is_not_null()).then(
            ((pl.lit(initial_pct) - pl.col(pct_col)) / pl.col(pct_col)) * 100
        ).otherwise(None)
    ).otherwise(
        pl.when(pl.col(pct_col).is_not_null() & (pl.col(pct_col) != 0) & pl.col(pct_col).shift(1).is_not_null()).then(PCT(pl.col(pct_col))).otherwise(None)
    )
    denom_expr = pl.when(pct_change_expr.is_not_null()).then(1 + pct_change_expr / 100).otherwise(None)
    denom_cum_expr = denom_expr.cum_prod()
    target_new_expr = pl.when(denom_cum_expr.is_not_null()).then(pl.lit(initial_target) / denom_cum_expr).otherwise(None)
    processed_df = work_df.with_columns(target_new_expr.alias(f"{target_col}_new"))
    updated = processed_df.select(["DATE", pl.col(f"{target_col}_new").alias(target_col)])
    return df.join(updated, on="DATE", how="left", suffix="_updated").with_columns([
        pl.when((pl.col("DATE") >= end) & (pl.col("DATE") <= start)).then(pl.col(f"{target_col}_updated")).otherwise(pl.col(target_col)).alias(target_col)
    ]).drop(f"{target_col}_updated")

def SHIFT_PCT_BACKWARDS_MULTIPLE(df, start_date, end_date, column_pairs, offsets=None, initial_targets=None, initial_pcts=None):
    result = df.clone()
    if offsets is None: offsets = [1] * len(column_pairs)
    for (tgt, pct), off in zip(column_pairs, offsets):
        it = initial_targets.get(tgt) if initial_targets else None
        ip = initial_pcts.get(pct) if initial_pcts else None
        result = SHIFT_PCT_BACKWARDS(result, tgt, pct, start_date, end_date, off, it, ip)
    return result
'''

    if ctx["need_assign"] or ctx["need_arith"]:
        defs["ASSIGN_SERIES"] = "def ASSIGN_SERIES(out_ser: str, expr: pl.Expr) -> pl.Expr:\n    return expr.alias(out_ser)"
    if ctx["need_add_series"]:
        defs["ADD_SERIES"] = "def ADD_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n    res = series[0]\n    for s in series[1:]: res = res + s\n    return res.alias(out_ser)"
    if ctx["need_sub_series"]:
        defs["SUB_SERIES"] = "def SUB_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n    res = series[0]\n    for s in series[1:]: res = res - s\n    return res.alias(out_ser)"
    if ctx["need_mul_series"]:
        defs["MUL_SERIES"] = "def MUL_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n    res = series[0]\n    for s in series[1:]: res = res * s\n    return res.alias(out_ser)"
    if ctx["need_div_series"]:
        defs["DIV_SERIES"] = "def DIV_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n    res = series[0]\n    for s in series[1:]: res = res / s\n    return res.alias(out_ser)"
        
    defs["APPLY_DATE_FILTER"] = (
        "def APPLY_DATE_FILTER(expr: pl.Expr, col_name: str, start_date: str, end_date: str, date_col: str = 'DATE', preserve_existing: bool = False) -> pl.Expr:\n"
        "    import polars as pl\n"
        "    from datetime import datetime, date as dt_date\n"
        "    def get_date_expr(d_str):\n"
        "        if d_str == '*': return None\n"
        "        try:\n"
        "            dt = datetime.strptime(d_str, '%Y-%m-%d').date()\n"
        "            return pl.lit(dt)\n"
        "        except ValueError:\n"
        "            pass\n"
        "        try:\n"
        "             dt = datetime.strptime(d_str, '%d%b%Y').date()\n"
        "             return pl.lit(dt)\n"
        "        except ValueError:\n"
        "             pass\n"
        "        return pl.col(d_str.upper())\n"
        "    start_expr = get_date_expr(start_date)\n"
        "    end_expr = get_date_expr(end_date)\n"
        "    cond = pl.lit(True)\n"
        "    if start_expr is not None: cond = cond & (pl.col(date_col) >= start_expr)\n"
        "    if end_expr is not None: cond = cond & (pl.col(date_col) <= end_expr)\n"
        "    otherwise = pl.col(col_name) if preserve_existing else pl.lit(None)\n"
        "    return pl.when(cond).then(expr).otherwise(otherwise)"
    )
    
    return defs
