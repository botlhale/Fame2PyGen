"""
formulas_generator: helpers to parse FAME-like formulas and render Polars expressions.

Responsibilities:
 - Detect $chain / $mchain and render CHAIN(...) wrapper text.
 - Detect SHIFT_PCT pattern of the form:
      x[t+K] / (1 + (pct(y[t+K]) / 100))
   with arbitrary integer offsets (t+N or t-N). When found, set ctx["need_shiftpct"]=True
   and return a marker string "__SHIFT_PCT__:{x}:{y}:{K}" so the pipeline generator will emit
   SHIFT_PCT(pl.col(...), pl.col(...), K).alias("TARGET").
 - Provide render_polars_expr(rhs, substitution_map=..., ctx=...) that:
    * preserves already-generated pl.col(...) fragments using placeholders
    * converts pct(...) to PCT(...) wrapper calls (with offset parameter if provided)
    * does not re-tokenize generated pl.col fragments (prevents pl.col("pl.col(...)") bugs)
 - Generate helper wrapper definitions via generate_polars_functions(...) including:
    CHAIN, PCT, CONVERT, FISHVOL, SHIFT_PCT, SHIFT_PCT_BACKWARDS, ASSIGN_SERIES, ADD/SUB/MUL/DIV_SERIES as needed.
"""

import re
from typing import Dict, List, Tuple, Optional

# ---------- Constants ----------

# Recognized FAME function names that should not be tokenized as variables
FUNCTION_NAMES = {"pct", "convert", "fishvol_rebase", "chain", "mchain", "sqrt"}

# Logical operators that should be preserved during tokenization
LOGICAL_OPERATORS = {"or", "and"}

# All keywords that should not be treated as variable references
FUNCTION_KEYWORDS = FUNCTION_NAMES | LOGICAL_OPERATORS

# ---------- Utilities ----------

def sanitize_func_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    s = str(name)
    s = s.replace("$", "_")
    # Preserve dots in column names (Polars supports them)
    # Only remove other special characters
    s = re.sub(r"[^A-Za-z0-9_.]", "", s)
    return s.lower()


def normalize_formula_text(formula: Optional[str]) -> str:
    if formula is None:
        return ""
    s = str(formula)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    s = s.strip()
    if s.lower().startswith("set "):
        s = s[len("set "):].strip()
    # strip outer quotes if present
    while s and s[0] in "\"'":
        s = s[1:].lstrip()
    while s and s[-1] in "\"'":
        s = s[:-1].rstrip()
    s = s.rstrip(";").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_time_index(token: str) -> Tuple[str, int]:
    """
    Parse token forms:
      name
      name[t]
      name[t+1]
      name[t-2]
    Returns (base, offset) where offset is integer (0 if absent).
    """
    if not token:
        return "", 0
    t = token.strip()
    m = re.match(r"^\s*([A-Za-z0-9_$]+)\s*(?:\[\s*t\s*([+-]?\d+)\s*\])?\s*$", t, re.IGNORECASE)
    if not m:
        return t, 0
    base = m.group(1)
    offs_raw = m.group(2)
    offs = int(offs_raw) if offs_raw else 0
    return base, offs


def parse_date_index(token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse token forms with date indexing:
      name["2020-01-01"]
      name['2020Q1']
      name["2020-12-31"]
    Returns (base, date_str) where date_str is the literal date string or None if not date-indexed.
    """
    if not token:
        return None, None
    t = token.strip()
    # Match variable name followed by date string in brackets (with either single or double quotes)
    m = re.match(r'^\s*([A-Za-z0-9_$]+)\s*\[\s*["\']([^"\']+)["\']\s*\]\s*$', t)
    if m:
        base = m.group(1)
        date_str = m.group(2)
        return base, date_str
    return None, None


# ---------- Token regex and numeric heuristics ----------

# Match tokens including time-indexed (e.g., var[t+1]) and date-indexed (e.g., var["2020-01-01"])
TOKEN_RE = re.compile(r'[A-Za-z0-9_$.]+(?:\s*\[\s*(?:t\s*[+-]?\d+|["\'][^"\']+["\']\s*)\])?')

def _is_strict_number(tok: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", tok.strip()))

def _is_numeric_literal(tok: str) -> bool:
    """
    Heuristic to decide whether a numeric token is a literal number (e.g. 100, 234) or a series id.
    - Treat numeric tokens with <= 3 digits (before decimal) as numeric literal.
    - Treat numeric tokens with >= 4 digits as series identifier.
    """
    if not _is_strict_number(tok):
        return False
    digits = tok.lstrip("+-").split(".", 1)[0]
    return len(digits) <= 3


def _token_to_pl_expr(tok: str) -> str:
    """
    Convert a token (maybe time indexed) to pl expression text:
      - 'nd' (FAME null/missing) -> pl.lit(None)
      - numeric literal -> literal number (kept as-is)
      - series id or name -> pl.col("NAME").shift(-offset) (offset from [t+N] or [t-N])
    """
    # Handle 'nd' keyword
    if tok.lower() == 'nd':
        return "pl.lit(None)"
    
    base, offs = parse_time_index(tok)
    if base == "":
        return "pl.lit(None)"
    if _is_strict_number(base):
        if _is_numeric_literal(base):
            return base
        # else treat it as a series identifier (numeric series id)
    col = sanitize_func_name(base).upper()
    expr = f'pl.col("{col}")'
    if offs != 0:
        expr = f"{expr}.shift({-offs})"
    return expr


# ---------- Chain inline rendering helper ----------

def _render_chain_calls(expr: str, date_col_name: str = "DATE") -> str:
    """
    Replace occurrences of $chain(...) and $mchain(...) with CHAIN(...) call text using pl.col('...') for pairs.
    This happens before token placeholder substitution.
    """
    if not expr:
        return expr
    out_parts: List[str] = []
    idx = 0
    while True:
        m = re.search(r"\$(mchain|chain)\s*\(", expr[idx:], re.IGNORECASE)
        if not m:
            out_parts.append(expr[idx:])
            break
        start = idx + m.start()
        out_parts.append(expr[idx:start])
        j = start + len(m.group(0))
        depth = 1
        while j < len(expr) and depth > 0:
            if expr[j] == "(":
                depth += 1
            elif expr[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # unbalanced, keep as-is
            out_parts.append(expr[start:j])
            idx = j
            continue
        inner = expr[start + len(m.group(0)) : j - 1].strip()
        parsed = re.match(r'^\s*"(.*?)"\s*,\s*"(.*?)"\s*$', inner, re.DOTALL)
        if not parsed:
            out_parts.append(expr[start:j])
            idx = j
            continue
        expr_str, year_raw = parsed.groups()
        year_m = re.search(r"(\d{4})", year_raw)
        year = year_m.group(1) if year_m else year_raw.strip()
        parts = re.split(r"([+-])", expr_str)
        parts = [p.strip() for p in parts if p.strip()]
        if parts and parts[0] not in ("+", "-"):
            parts.insert(0, "+")
        terms = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
        pair_items = []
        for op, var in terms:
            sign = "-" if op == "-" else ""
            pcol = sanitize_func_name("p" + var).upper()
            vcol = sanitize_func_name(var).upper()
            if sign:
                pair_items.append(f"(-pl.col('{pcol}'), pl.col('{vcol}'))")
            else:
                pair_items.append(f"(pl.col('{pcol}'), pl.col('{vcol}'))")
        pairs_str = ", ".join(pair_items)
        chain_expr = f"CHAIN(price_quantity_pairs=[{pairs_str}], date_col=pl.col(\"{date_col_name}\"), year=\"{year}\")"
        out_parts.append(chain_expr)
        idx = j
    return "".join(out_parts)


# ---------- Placeholders builder (protect pl fragments) ----------

def _build_sub_map_and_placeholders(expr: str, substitution_map: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    """
    Replace tokens in expr by placeholders __PH_n__ and return (expr_with_placeholders, placeholders_map).
    substitution_map maps token.lower() -> replacement string to prefer (e.g. precomputed pl.col("X").shift(-1)).
    Numeric literals and bare 't' are left as-is.
    Function names (pct, convert, etc.) are left as-is to be processed later.
    """
    placeholders: Dict[str, str] = {}
    parts: List[str] = []
    last = 0
    idx = 0
    for m in TOKEN_RE.finditer(expr):
        s, e = m.span()
        parts.append(expr[last:s])
        tok = m.group(0)
        key = tok.lower()
        # keep bare 't' and small numeric literals
        if key == "t" or (_is_strict_number(tok) and _is_numeric_literal(tok)):
            parts.append(tok)
            last = e
            continue
        # keep function names as-is (they'll be processed by pct conversion logic)
        if key in FUNCTION_NAMES:
            parts.append(tok)
            last = e
            continue
        # keep logical operators as-is (they'll be processed later)
        if key in LOGICAL_OPERATORS:
            parts.append(tok)
            last = e
            continue
        # Skip __ND_PLACEHOLDER__ - it should remain as-is
        if tok == "__ND_PLACEHOLDER__":
            parts.append(tok)
            last = e
            continue
        # if substitution_map provides replacement, use placeholder for it
        if substitution_map and key in substitution_map:
            ph = f"__PH_{idx}__"
            placeholders[ph] = substitution_map[key]
            parts.append(ph)
            idx += 1
            last = e
            continue
        # default mapping
        ph = f"__PH_{idx}__"
        placeholders[ph] = _token_to_pl_expr(tok)
        parts.append(ph)
        idx += 1
        last = e
    parts.append(expr[last:])
    return "".join(parts), placeholders


# ---------- SHIFT_PCT detection regex ----------

_shift_pct_re = re.compile(
    r"^\s*([A-Za-z0-9_$]+)\s*\[\s*t\s*([+-]?\d+)\s*\]\s*/\s*\(\s*1\s*\+\s*\(\s*pct\s*\(\s*([A-Za-z0-9_$]+)\s*\[\s*t\s*([+-]?\d+)\s*\]\s*\)\s*/\s*100\s*\)\s*\)\s*$",
    re.IGNORECASE,
)


# ---------- Main renderer ----------

def render_conditional_expr(condition: str, then_expr: str, else_expr: str, substitution_map: Optional[Dict[str, str]] = None) -> str:
    """
    Render FAME conditional expression to Polars .when().then().otherwise() pattern.
    
    Handles:
    - Comparison operators: ge, gt, le, lt, eq, ne
    - 'nd' keyword mapping to pl.lit(None)
    - FAME functions: dateof, make, date, contain, end
    - Variable references with substitution
    """
    # Parse and render condition
    # FAME comparison operators: ge (>=), gt (>), le (<=), lt (<), eq (==), ne (!=)
    comparisons = {
        'ge': '>=',
        'gt': '>',
        'le': '<=',
        'lt': '<',
        'eq': '==',
        'ne': '!='
    }
    
    # Replace comparison operators
    cond_polars = condition
    for fame_op, py_op in comparisons.items():
        # Use word boundaries to avoid partial matches
        cond_polars = re.sub(r'\b' + fame_op + r'\b', py_op, cond_polars, flags=re.IGNORECASE)
    
    # Handle 'nd' keyword (null/missing in FAME) -> pl.lit(None)
    # Do this BEFORE tokenization to avoid splitting it up
    def replace_nd_keyword(expr: str) -> str:
        # Replace 'nd' keyword with a placeholder that won't be tokenized
        return re.sub(r'\bnd\b', '__ND_PLACEHOLDER__', expr, flags=re.IGNORECASE)
    
    def restore_nd_placeholder(expr: str) -> str:
        # Restore placeholder to pl.lit(None)
        return expr.replace('__ND_PLACEHOLDER__', 'pl.lit(None)')
    
    # Process condition - substitute variables
    cond_expr = cond_polars
    if substitution_map:
        # Build a replacement map with proper boundaries
        # Replace tokens in order from longest to shortest to avoid partial matches
        sorted_tokens = sorted(substitution_map.keys(), key=lambda x: len(x), reverse=True)
        for token_lower in sorted_tokens:
            # Use word boundaries to replace only complete tokens
            pattern = r'\b' + re.escape(token_lower) + r'\b'
            cond_expr = re.sub(pattern, substitution_map[token_lower], cond_expr, flags=re.IGNORECASE)
    
    # Process then_expr - replace nd first
    then_with_placeholder = replace_nd_keyword(then_expr)
    then_polars = render_polars_expr(then_with_placeholder, substitution_map=substitution_map)
    then_polars = restore_nd_placeholder(then_polars)
    
    # Process else_expr - check if it's a nested conditional first
    else_with_placeholder = replace_nd_keyword(else_expr)
    # Check if else_expr is a nested conditional (starts with "if")
    if re.match(r'^\s*if\s+.+\s+then\s+.+\s+else\s+.+\s*$', else_with_placeholder, re.IGNORECASE):
        # Recursively process nested conditional
        nested_result = parse_conditional_expr(else_with_placeholder)
        if nested_result:
            else_polars = render_conditional_expr(
                nested_result['condition'],
                nested_result['then_expr'],
                nested_result['else_expr'],
                substitution_map=substitution_map
            )
        else:
            # Fallback to regular expression rendering
            else_polars = render_polars_expr(else_with_placeholder, substitution_map=substitution_map)
            else_polars = restore_nd_placeholder(else_polars)
    else:
        else_polars = render_polars_expr(else_with_placeholder, substitution_map=substitution_map)
        else_polars = restore_nd_placeholder(else_polars)
    
    # Build Polars when/then/otherwise
    return f"pl.when({cond_expr}).then({then_polars}).otherwise({else_polars})"


def render_polars_expr(rhs: str, substitution_map: Optional[Dict[str, str]] = None, memory: Optional[Dict[str, str]] = None, ctx: Optional[Dict[str, bool]] = None) -> str:
    """
    Render RHS into a Polars expression string.

    Parameters:
      - rhs: original RHS text from FAME
      - substitution_map: mapping token.lower() -> replacement string, preferred
      - memory: alias for substitution_map for backward compatibility
      - ctx: optional dict that will be updated with flags (e.g. ctx['need_shiftpct']=True)

    Returns:
      - A string representing the Polars expression. If the exact SHIFT_PCT pattern is found,
        returns a marker string "__SHIFT_PCT__:{ser1}:{ser2}:{offset}" for the pipeline generator.
    """
    if rhs is None:
        return ""
    sub_map = substitution_map if substitution_map is not None else (memory if memory is not None else {})

    expr = normalize_formula_text(rhs)

    # 1) inline $chain/$mchain -> CHAIN(...)
    expr = _render_chain_calls(expr)

    # 2) SHIFT_PCT exact-match detection (both offsets must match)
    msp = _shift_pct_re.match(expr)
    if msp:
        ser1 = msp.group(1)
        offs1 = int(msp.group(2))
        ser2 = msp.group(3)
        offs2 = int(msp.group(4))
        if offs1 == offs2:
            if ctx is not None:
                ctx["need_shiftpct"] = True
            return f"__SHIFT_PCT__:{ser1}:{ser2}:{offs1}"

    # 3) Replace tokens with placeholders to prevent re-tokenization of generated fragments
    expr_with_ph, placeholders = _build_sub_map_and_placeholders(expr, substitution_map=sub_map)

    # 4) Process pct(...) constructs on the placeholder-containing string (balanced)
    out_parts: List[str] = []
    i = 0
    while True:
        m = re.search(r"\bpct\s*\(", expr_with_ph[i:], re.IGNORECASE)
        if not m:
            out_parts.append(expr_with_ph[i:])
            break
        s_pos = i + m.start()
        out_parts.append(expr_with_ph[i:s_pos])
        j = s_pos + len(m.group(0))
        depth = 1
        while j < len(expr_with_ph) and depth > 0:
            if expr_with_ph[j] == "(":
                depth += 1
            elif expr_with_ph[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            out_parts.append(expr_with_ph[s_pos:])
            i = len(expr_with_ph)
            break
        inner = expr_with_ph[s_pos + len(m.group(0)) : j - 1].strip()
        # split top-level comma if present
        depth2 = 0
        split_pos = None
        for idx_ch, ch in enumerate(inner):
            if ch == "(":
                depth2 += 1
            elif ch == ")":
                depth2 -= 1
            elif ch == "," and depth2 == 0:
                split_pos = idx_ch
                break
        if split_pos is not None:
            arg_expr = inner[:split_pos].strip()
            offset_expr = inner[split_pos + 1 :].strip()
        else:
            arg_expr = inner
            offset_expr = None
        if offset_expr:
            out_parts.append(f"PCT({arg_expr}, offset={offset_expr})")
        else:
            out_parts.append(f"PCT({arg_expr})")
        i = j

    combined = "".join(out_parts)

    # 4b) Process sqrt(...) constructs on the combined string (balanced)
    sqrt_parts: List[str] = []
    i = 0
    while True:
        m = re.search(r"\bsqrt\s*\(", combined[i:], re.IGNORECASE)
        if not m:
            sqrt_parts.append(combined[i:])
            break
        s_pos = i + m.start()
        sqrt_parts.append(combined[i:s_pos])
        j = s_pos + len(m.group(0))
        depth = 1
        while j < len(combined) and depth > 0:
            if combined[j] == "(":
                depth += 1
            elif combined[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            sqrt_parts.append(combined[s_pos:])
            i = len(combined)
            break
        inner = combined[s_pos + len(m.group(0)) : j - 1].strip()
        sqrt_parts.append(f"SQRT({inner})")
        i = j
    
    combined = "".join(sqrt_parts)

    # 4c) Handle logical operators: 'or' -> '|', 'and' -> '&'
    # Replace whole word 'or' and 'and' with proper Polars operators
    combined = re.sub(r'\bor\b', '|', combined, flags=re.IGNORECASE)
    combined = re.sub(r'\band\b', '&', combined, flags=re.IGNORECASE)

    # 5) Replace placeholders with actual replacement strings (no further tokenization)
    final = combined
    for ph, repl in placeholders.items():
        final = final.replace(ph, repl)

    return final


# ---------- Minimal parser used by pipeline generator ----------

def _parse_chain_top_level(line: str) -> Optional[Dict]:
    m = re.match(r'^\s*([A-Za-z0-9_$]+)\s*=\s*\$mchain\s*\(\s*"(.*?)"\s*,\s*"\s*(\d{4})\s*"\s*\)\s*$', line, re.IGNORECASE)
    if m:
        target, inner, year = m.groups()
        parts = re.split(r"([+-])", inner)
        parts = [p.strip() for p in parts if p.strip()]
        if parts and parts[0] not in ("+", "-"):
            parts.insert(0, "+")
        terms = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
        refs = [v for qv in [t[1] for t in terms] for v in (qv, "p" + qv)]
        return {"type": "mchain", "target": target, "refs": refs, "terms": terms, "year": year}
    m2 = re.match(r'^\s*([A-Za-z0-9_$]+)\s*=\s*\$chain\s*\(\s*"(.*?)"\s*,\s*"\s*(\d{4})\s*"\s*\)\s*$', line, re.IGNORECASE)
    if m2:
        target, inner, year = m2.groups()
        parts = re.split(r"([+-])", inner)
        parts = [p.strip() for p in parts if p.strip()]
        if parts and parts[0] not in ("+", "-"):
            parts.insert(0, "+")
        terms = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
        refs = [v for qv in [t[1] for t in terms] for v in (qv, "p" + qv)]
        return {"type": "chain", "target": target, "refs": refs, "terms": terms, "year": year}
    return None


def parse_conditional_expr(expr: str) -> Optional[Dict]:
    """
    Parse FAME-style conditional expressions:
      if <condition> then <then_expr> else <else_expr>
    
    Returns dict with:
      - condition: the condition expression
      - then_expr: expression for true case
      - else_expr: expression for false case
      - refs: list of variable references in all parts
    """
    # Match pattern: if ... then ... else ...
    # Need to handle nested parentheses in condition
    if_match = re.match(r'^\s*if\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+?)\s*$', expr, re.IGNORECASE)
    if not if_match:
        return None
    
    condition = if_match.group(1).strip()
    then_expr = if_match.group(2).strip()
    else_expr = if_match.group(3).strip()
    
    # Extract references from all parts
    # Exclude conditional keywords from refs
    conditional_keywords = {'t', 'if', 'then', 'else', 'and', 'or', 'not', 'ge', 'gt', 'le', 'lt', 'eq', 'ne', 'nd'}
    refs = []
    for part in [condition, then_expr, else_expr]:
        raw = TOKEN_RE.findall(part)
        for tkn in raw:
            if tkn.lower() not in conditional_keywords and not _is_strict_number(tkn):
                base, _ = parse_time_index(tkn)
                if base and not _is_strict_number(base):
                    refs.append(tkn)
    
    return {
        "type": "conditional",
        "condition": condition,
        "then_expr": then_expr,
        "else_expr": else_expr,
        "refs": refs
    }


def parse_fame_formula(line: str) -> Optional[Dict]:
    """
    Lightweight parser returning dicts with keys used by the pipeline generator.
    """
    if line is None:
        return None
    s_raw = line
    s = normalize_formula_text(line)

    # 1) top-level chain / mchain first (must not be parsed as arithmetic)
    ch = _parse_chain_top_level(s)
    if ch:
        return ch

    # 2) list alias
    m_list = re.match(r"^\s*([A-Za-z0-9_$]+)\s*=\s*\{(.+)\}\s*$", s)
    if m_list and not re.search(r"[+\-*/]", s.split("=", 1)[1].replace(f"{{{m_list.group(2)}}}", "")):
        target, content = m_list.groups()
        items = [it.strip() for it in content.split(",")]
        return {"type": "list_alias", "target": target, "refs": items, "original_rhs": f"{{{content}}}"}

    # 3) freq
    m_freq = re.match(r"^\s*freq\s+([A-Za-z0-9]+)\s*$", s, re.IGNORECASE)
    if m_freq:
        return {"type": "freq", "freq": m_freq.group(1).lower()}

    # 4) date command - handles both "date *" and "date <start> to <end>"
    m_date_all = re.match(r"^\s*date\s+\*\s*$", s, re.IGNORECASE)
    if m_date_all:
        return {"type": "date", "filter": None}  # None means no filtering (all dates)
    
    m_date_range = re.match(r"^\s*date\s+(.+?)\s+to\s+(.+?)\s*$", s, re.IGNORECASE)
    if m_date_range:
        start_date, end_date = m_date_range.groups()
        return {"type": "date", "filter": {"start": start_date.strip(), "end": end_date.strip()}}

    # 4b) point-in-time (date-indexed) assignment - e.g., variable["2020-01-01"] = value or variable[12mar2020] = value
    # Match patterns like: var["date"] = expr or var['date'] = expr or var[12mar2020] = expr
    # First try quoted dates
    m_date_assign = re.match(r'^\s*([A-Za-z0-9_$]+)\s*\[\s*["\']([^"\']+)["\']\s*\]\s*=\s*(.+)\s*$', s)
    if not m_date_assign:
        # Try unquoted date formats like: 12mar2020, 01Feb2020, 2020Q1, 2020-01-01
        m_date_assign = re.match(r'^\s*([A-Za-z0-9_$]+)\s*\[\s*(\d{1,2}[A-Za-z]{3}\d{4}|\d{4}Q[1-4]|\d{4}-\d{2}-\d{2})\s*\]\s*=\s*(.+)\s*$', s, re.IGNORECASE)
    
    if m_date_assign:
        target, date_str, rhs = m_date_assign.groups()
        # Parse RHS to extract references
        raw = TOKEN_RE.findall(rhs)
        refs = []
        for tkn in raw:
            if tkn.lower() != "t":
                # Check if token is date-indexed
                base, date_idx = parse_date_index(tkn)
                if base:
                    refs.append(base)
                else:
                    # Regular token - extract base if time-indexed
                    base_tok, _ = parse_time_index(tkn)
                    if base_tok and not _is_strict_number(base_tok):
                        refs.append(base_tok)
        return {
            "type": "point_in_time_assign",
            "target": target,
            "date": date_str,
            "rhs": rhs.strip(),
            "original_rhs": rhs.strip(),
            "refs": refs
        }

    # 5) convert
    m_convert = re.match(r"^\s*([A-Za-z0-9_$]+)\s*=\s*convert\((.+)\)\s*$", s, re.IGNORECASE)
    if m_convert:
        target, args_str = m_convert.groups()
        # Split args carefully, respecting quoted strings
        args = [a.strip().strip("'\"") for a in re.split(r""",\s*(?=(?:[^"']*["'][^"']*["'])*[^"']*$)""", args_str)]
        if len(args) >= 4:  # Accept 4 or more parameters
            return {"type": "convert", "target": target, "refs": [args[0]], "params": args}

    # 6) fishvol
    m_fv = re.match(r"^\s*([A-Za-z0-9_$]+)\s*=\s*\$?fishvol_rebase\((.+)\)(\s*[*/+-].*)?\s*$", s, re.IGNORECASE)
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
            return {"type": "fishvol", "target": target, "refs": variable_refs, "pairs": pairs, "year": year, "trailing_op": trailing.strip() if trailing else None}

    # 7) Check for SHIFT_PCT pattern explicitly - extract RHS first if assignment present
    if "=" in s:
        lhs_temp, rhs_temp = s.split("=", 1)
        lhs_temp = lhs_temp.strip()
        rhs_temp = rhs_temp.strip()
        msp = _shift_pct_re.match(rhs_temp)
        if msp:
            ser1 = msp.group(1)
            offs1 = int(msp.group(2))
            ser2 = msp.group(3)
            offs2 = int(msp.group(4))
            if offs1 == offs2:
                # Extract target from LHS - handle time-indexed target like v123s[t]
                target_match = re.match(r"^\s*([A-Za-z0-9_$]+)", lhs_temp)
                target = target_match.group(1) if target_match else lhs_temp
                return {"type": "shift_pct", "target": target, "refs": [ser1, ser2], "original_rhs": rhs_temp, 
                        "ser1": ser1, "ser2": ser2, "offset": offs1}

    # 8) conditional expression check (before generic assignment)
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Check for conditional expression in RHS
        if re.match(r'^\s*if\s+.+\s+then\s+.+\s+else\s+.+\s*$', rhs, re.IGNORECASE):
            cond_result = parse_conditional_expr(rhs)
            if cond_result:
                cond_result["target"] = lhs
                cond_result["original_rhs"] = rhs
                return cond_result
        
        # assign-series single token (e.g., v1234s = 1234) - RHS may be series id or numeric literal
        m_assign = re.match(r"^\s*([A-Za-z0-9_$]+)\s*=\s*([A-Za-z0-9_$.]+)\s*$", s)
        if m_assign:
            target = m_assign.group(1)
            rhs_tok = m_assign.group(2)
            return {"type": "assign_series", "target": target, "rhs": rhs_tok, "original_rhs": rhs}
        # fallback simple / arithmetic: collect refs tokens
        raw = TOKEN_RE.findall(rhs)
        # Filter out function names and logical operators from refs
        refs = [tkn for tkn in raw if tkn.lower() != "t" and tkn.lower() not in FUNCTION_KEYWORDS]
        return {"type": "simple", "target": lhs, "rhs": rhs, "original_rhs": rhs, "refs": refs}

    return None


# ---------- Helper generator ----------

def generate_polars_functions(fame_cmds: List[str]) -> Dict[str, str]:
    """
    Scan commands and emit only the helper wrappers that are needed:
      - CHAIN, PCT, CONVERT, FISHVOL
      - SHIFT_PCT, SHIFT_PCT_BACKWARDS when detected
      - ASSIGN_SERIES(out_ser, expr) used to alias result
      - ADD/SUB/MUL/DIV_SERIES for homogeneous arithmetic
    """
    ctx = {
        "has_mchain": False,
        "has_convert": False,
        "has_fishvol": False,
        "has_pct": False,
        "has_sqrt": False,
        "need_shiftpct": False,
        "need_shiftpct_backwards": False,
        "need_assign": False,
        "need_add_series": False,
        "need_sub_series": False,
        "need_mul_series": False,
        "need_div_series": False,
        "need_arith": False,
        "need_point_in_time_assign": False,
    }

    # scan commands properly
    for cmd in fame_cmds:
        if not cmd:
            continue
        s = str(cmd)
        if re.search(r"\$(chain|mchain)\s*\(", s, re.IGNORECASE):
            ctx["has_mchain"] = True
        if re.search(r"\bconvert\s*\(", s, re.IGNORECASE):
            ctx["has_convert"] = True
        if re.search(r"\bfishvol_rebase\s*\(", s, re.IGNORECASE):
            ctx["has_fishvol"] = True
        if re.search(r"\bpct\s*\(", s, re.IGNORECASE):
            ctx["has_pct"] = True
        if re.search(r"\bsqrt\s*\(", s, re.IGNORECASE):
            ctx["has_sqrt"] = True
        
        # SHIFT_PCT detection with enhanced pattern recognition - also detect in assignments
        normalized = normalize_formula_text(s)
        if "=" in normalized:
            lhs_check, rhs_check = normalized.split("=", 1)
            lhs_check = lhs_check.strip()
            rhs_check = rhs_check.strip()
            if _shift_pct_re.match(rhs_check):
                ctx["need_shiftpct"] = True
                # Check if this is a backwards pattern: LHS has [t] and RHS has [t+N] with N>0
                lhs_time_match = re.search(r"\[\s*t\s*([+-]?\d*)\s*\]", lhs_check)
                rhs_time_match = re.search(r"\[\s*t\s*\+\s*(\d+)\s*\]", rhs_check)
                if lhs_time_match and rhs_time_match:
                    lhs_offset = lhs_time_match.group(1)
                    lhs_offset = int(lhs_offset) if lhs_offset and lhs_offset not in ('+', '-') else 0
                    rhs_offset = int(rhs_time_match.group(1))
                    # Backwards if LHS is current time (t or t+0) and RHS references future time (t+N, N>0)
                    if lhs_offset == 0 and rhs_offset > 0:
                        ctx["need_shiftpct_backwards"] = True
        
        # point-in-time assignment pattern - match both quoted and unquoted dates
        # Use normalized version to remove "set " prefix
        normalized_for_point = normalize_formula_text(s)
        if re.match(r'^\s*[A-Za-z0-9_$]+\s*\[\s*["\'][^"\']+["\']\s*\]\s*=\s*.+\s*$', normalized_for_point):
            ctx["need_point_in_time_assign"] = True
        # Also check for unquoted date formats
        elif re.match(r'^\s*[A-Za-z0-9_$]+\s*\[\s*(\d{1,2}[A-Za-z]{3}\d{4}|\d{4}Q[1-4]|\d{4}-\d{2}-\d{2})\s*\]\s*=\s*.+\s*$', normalized_for_point, re.IGNORECASE):
            ctx["need_point_in_time_assign"] = True
        
        # assign-series pattern
        if re.match(r"^\s*[A-Za-z0-9_$]+\s*=\s*[A-Za-z0-9_$.]+\s*$", s):
            ctx["need_assign"] = True
        # arithmetic homogeneous detection but avoid special function RHS
        if "=" in s:
            rhs = s.split("=", 1)[1].strip()
            if all(tok not in rhs.lower() for tok in ["pct(", "$chain", "$mchain", "convert(", "fishvol_rebase("]):
                parts = re.split(r"([+\-*/])", rhs)
                ops = [p for p in parts if p in "+-*/"]
                if ops and all(op == "+" for op in ops):
                    ctx["need_add_series"] = True
                if ops and all(op == "-" for op in ops):
                    ctx["need_sub_series"] = True
                if ops and all(op == "*" for op in ops):
                    ctx["need_mul_series"] = True
                if ops and all(op == "/" for op in ops):
                    ctx["need_div_series"] = True
                if ops and len(set(ops)) > 1:
                    ctx["need_arith"] = True

    defs: Dict[str, str] = {}
    if ctx["has_mchain"]:
        defs["CHAIN"] = (
            "def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))"
        )
    if ctx["has_convert"]:
        defs["CONVERT"] = (
            "def CONVERT(series: pl.DataFrame, *args) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    # Handle both standard and custom convert signatures\n"
            "    if len(args) == 4:\n"
            "        # Standard: as_freq, to_freq, technique, observed\n"
            "        return ple.convert(series, 'DATE', as_freq=args[0], to_freq=args[1], technique=args[2], observed=args[3])\n"
            "    elif len(args) == 3:\n"
            "        # Custom 3-param variant\n"
            "        return ple.convert(series, 'DATE', *args)\n"
            "    else:\n"
            "        # Generic fallback - pass all args\n"
            "        return ple.convert(series, 'DATE', *args)"
        )
    if ctx["has_fishvol"]:
        defs["FISHVOL"] = (
            "def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    return ple.fishvol(series_pairs, date_col, rebase_year)"
        )
    if ctx["has_pct"]:
        defs["PCT"] = (
            "def PCT(expr: pl.Expr, offset: int = 1) -> pl.Expr:\n"
            "    import polars_econ as ple\n"
            "    return ple.pct(expr, offset=offset)"
        )
    if ctx["has_sqrt"]:
        defs["SQRT"] = (
            "def SQRT(expr: pl.Expr) -> pl.Expr:\n"
            "    \"\"\"Calculate the square root of an expression.\"\"\"\n"
            "    return expr.sqrt()"
        )
    if ctx["need_shiftpct"]:
        defs["SHIFT_PCT"] = (
            "def SHIFT_PCT(ser1: pl.Expr, ser2: pl.Expr, offset: int = 1) -> pl.Expr:\n"
            '    """Compute shift-and-pct-adjusted series for arbitrary offset:\n'
            "      ser1.shift(-offset) / (1 + (PCT(ser2.shift(-offset)).truediv(100)))\n"
            "    Arguments:\n"
            "      ser1: price or series to shift in numerator (pl.Expr)\n"
            "      ser2: series passed to PCT (pl.Expr)\n"
            "      offset: integer N where tokens are of the form [t+N] or [t-N].\n"
            "    Note: the function applies ser.shift(-offset) to map a reference like series[t+N]\n"
            "    into a current-period expression. Caller should alias the returned pl.Expr.\n"
            '    """\n'
            "    return (ser1.shift(-offset) / (1 + (PCT(ser2.shift(-offset)).truediv(100))))"
        )
    if ctx["need_shiftpct_backwards"]:
        defs["SHIFT_PCT_BACKWARDS"] = '''
def SHIFT_PCT_BACKWARDS(
    df: pl.DataFrame,
    target_col: str,
    pct_col: str,
    start_date: str,
    end_date: str,
    offset: int = 1,
    initial_target: float = None,
    initial_pct: float = None,
) -> pl.DataFrame:
    """
    Apply backwards time-series calculation matching FAME logic with robust None handling.
    
    FAME equivalent:
      loop for mydate = start_date to end_date step -1
        set target_col[t] = target_col[t+offset] / (1 + (pct(pct_col[t+offset]) / 100))
    
    Where pct(pct_col[t+offset]) = ((pct_col[t+offset] - pct_col[t]) / pct_col[t]) * 100
    
    Arguments:
      df: DataFrame with DATE column and time series
      target_col: Column to update (e.g., "V1234S")
      pct_col: Column for PCT calculation (e.g., "V1014S")
      start_date: Start date in "YYYY-MM-DD" format (most recent, e.g., "2016-12-31")
      end_date: End date in "YYYY-MM-DD" format (oldest, e.g., "1981-03-31")
      offset: Offset for shift (e.g., 1 for t+1)
      initial_target: Target value for the period after start_date (optional, inferred if None)
      initial_pct: PCT value for the period after start_date (optional, inferred if None)
    
    Returns:
      DataFrame with target_col updated within the date range
    """
    import polars as pl
    from datetime import datetime
    
    def PCT(expr: pl.Expr, offset: int = offset) -> pl.Expr:
        """
        Compute the percent change of a Polars expression.
        """
        return ((expr.shift(offset) - expr) / expr) * 100
    
    # Convert dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Infer initial values if not provided
    if initial_target is None or initial_pct is None:
        # Find the next period: smallest DATE > start_date
        next_period = df.filter(pl.col("DATE") > start).sort("DATE").limit(1)
        if next_period.is_empty():
            raise ValueError(f"No data found for period after start_date ({start_date}). Provide initial_target and initial_pct explicitly.")
        if initial_target is None:
            initial_target = next_period.select(target_col).item()
        if initial_pct is None:
            initial_pct = next_period.select(pct_col).item()
    
    # Filter and sort descending (newest first)
    work_df = (
        df.filter((pl.col("DATE") >= end) & (pl.col("DATE") <= start))
        .sort("DATE", descending=True)
    )
    
    if work_df.is_empty():
        return df
    
    # Compute pct_change column
    pct_change_expr = pl.when(pl.col("DATE") == start).then(
        pl.when(
            pl.col(pct_col).is_not_null() & (pl.col(pct_col) != 0) & pl.lit(initial_pct).is_not_null()
        ).then(
            ((pl.lit(initial_pct) - pl.col(pct_col)) / pl.col(pct_col)) * 100
        ).otherwise(None)
    ).otherwise(
        pl.when(
            pl.col(pct_col).is_not_null() & (pl.col(pct_col) != 0) & pl.col(pct_col).shift(1).is_not_null()
        ).then(
            PCT(pl.col(pct_col))
        ).otherwise(None)
    )
    
    # Compute denom = 1 + pct_change / 100, but only where pct_change not null
    denom_expr = pl.when(pct_change_expr.is_not_null()).then(1 + pct_change_expr / 100).otherwise(None)
    
    # Compute cumulative product of denom (this propagates None correctly)
    denom_cum_expr = denom_expr.cum_prod()
    
    # Compute the new target values
    target_new_expr = pl.when(denom_cum_expr.is_not_null()).then(
        pl.lit(initial_target) / denom_cum_expr
    ).otherwise(None)
    
    # Add the new target column
    processed_df = work_df.with_columns(target_new_expr.alias(f"{target_col}_new"))
    
    # Select only DATE and the new target for merging
    updated = processed_df.select(["DATE", pl.col(f"{target_col}_new").alias(target_col)])
    
    # Merge back into original DataFrame
    result = df.join(
        updated,
        on="DATE",
        how="left",
        suffix="_updated"
    ).with_columns([
        pl.when((pl.col("DATE") >= end) & (pl.col("DATE") <= start))
        .then(pl.col(f"{target_col}_updated"))
        .otherwise(pl.col(target_col))
        .alias(target_col)
    ]).drop(f"{target_col}_updated")
    
    return result

def SHIFT_PCT_BACKWARDS_MULTIPLE(
    df: pl.DataFrame,
    start_date: str,
    end_date: str,
    column_pairs: list[tuple[str, str]],
    offsets: list[int] = None,
    initial_targets: dict[str, float] = None,
    initial_pcts: dict[str, float] = None,
) -> pl.DataFrame:
    result = df.clone()
    if offsets is None:
        offsets = [1] * len(column_pairs)
    for (target_col, pct_col), offset in zip(column_pairs, offsets):
        init_target = initial_targets.get(target_col) if initial_targets else None
        init_pct = initial_pcts.get(pct_col) if initial_pcts else None
        
        result = SHIFT_PCT_BACKWARDS(
            df=result,
            target_col=target_col,
            pct_col=pct_col,
            start_date=start_date,
            end_date=end_date,
            offset=offset,
            initial_target=init_target,
            initial_pct=init_pct,
        )
    return result
'''
    # ASSIGN_SERIES as generic aliasing wrapper
    if ctx["need_assign"] or ctx["need_arith"]:
        defs["ASSIGN_SERIES"] = (
            "def ASSIGN_SERIES(out_ser: str, expr: pl.Expr) -> pl.Expr:\n"
            '    """Alias a pre-composed Polars expression to out_ser."""\n'
            "    return expr.alias(out_ser)"
        )
    if ctx["need_add_series"]:
        defs["ADD_SERIES"] = (
            "def ADD_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n"
            "    if not series:\n"
            "        return pl.lit(None).alias(out_ser)\n"
            "    res = series[0]\n"
            "    for s in series[1:]:\n"
            "        res = res + s\n"
            "    return res.alias(out_ser)"
        )
    if ctx["need_sub_series"]:
        defs["SUB_SERIES"] = (
            "def SUB_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n"
            "    if not series:\n"
            "        return pl.lit(None).alias(out_ser)\n"
            "    res = series[0]\n"
            "    for s in series[1:]:\n"
            "        res = res - s\n"
            "    return res.alias(out_ser)"
        )
    if ctx["need_mul_series"]:
        defs["MUL_SERIES"] = (
            "def MUL_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n"
            "    if not series:\n"
            "        return pl.lit(None).alias(out_ser)\n"
            "    res = series[0]\n"
            "    for s in series[1:]:\n"
            "        res = res * s\n"
            "    return res.alias(out_ser)"
        )
    if ctx["need_div_series"]:
        defs["DIV_SERIES"] = (
            "def DIV_SERIES(out_ser: str, *series: pl.Expr) -> pl.Expr:\n"
            "    if not series:\n"
            "        return pl.lit(None).alias(out_ser)\n"
            "    res = series[0]\n"
            "    for s in series[1:]:\n"
            "        res = res / s\n"
            "    return res.alias(out_ser)"
        )
    
    if ctx["need_point_in_time_assign"]:
        defs["POINT_IN_TIME_ASSIGN"] = (
            "def POINT_IN_TIME_ASSIGN(pdf: pl.DataFrame, target_col: str, date_str: str, value_expr, date_col: str = 'DATE') -> pl.DataFrame:\n"
            '    """Update a specific row in a time series based on a date string.\n'
            "    \n"
            "    Args:\n"
            "        pdf: Input DataFrame\n"
            "        target_col: Name of the column to update\n"
            "        date_str: Date string (e.g., '2020-01-01', '2020Q1')\n"
            "        value_expr: Expression (pl.Expr) or callable to compute the new value\n"
            "        date_col: Name of the date column (default 'DATE')\n"
            "    \n"
            "    Returns:\n"
            "        DataFrame with the updated column\n"
            '    """\n'
            "    import polars as pl\n"
            "    from datetime import datetime, date\n"
            "    import re\n"
            "    \n"
            "    # Parse date string - support multiple formats\n"
            "    try:\n"
            "        # Try YYYY-MM-DD format\n"
            "        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()\n"
            "    except ValueError:\n"
            "        try:\n"
            "            # Try YYYYQN format (e.g., 2020Q1)\n"
            "            m = re.match(r'^(\\d{4})Q([1-4])$', date_str, re.IGNORECASE)\n"
            "            if m:\n"
            "                year = int(m.group(1))\n"
            "                quarter = int(m.group(2))\n"
            "                month = (quarter - 1) * 3 + 1\n"
            "                target_date = date(year, month, 1)\n"
            "            else:\n"
            "                raise ValueError(f'Unsupported date format: {date_str}')\n"
            "        except:\n"
            "            raise ValueError(f'Cannot parse date: {date_str}')\n"
            "    \n"
            "    # If value_expr is callable, call it with the full dataframe\n"
            "    # For lambdas that reference specific dates, they need date objects\n"
            "    if callable(value_expr):\n"
            "        # Modify the lambda to use date objects instead of strings\n"
            "        result = value_expr(pdf)\n"
            "        # If the result is a scalar, wrap it in pl.lit()\n"
            "        if not isinstance(result, pl.Expr):\n"
            "            value_expr = pl.lit(result)\n"
            "        else:\n"
            "            value_expr = result\n"
            "    \n"
            "    # Create a temporary DataFrame with the computed value for the specific date\n"
            "    temp_df = pdf.filter(pl.col(date_col) == target_date).with_columns(\n"
            "        value_expr.alias(f'{target_col}_new')\n"
            "    ).select([date_col, f'{target_col}_new'])\n"
            "    \n"
            "    # Join back and update the column\n"
            "    result = pdf.join(temp_df, on=date_col, how='left').with_columns(\n"
            "        pl.when(pl.col(f'{target_col}_new').is_not_null())\n"
            "        .then(pl.col(f'{target_col}_new'))\n"
            "        .otherwise(pl.col(target_col))\n"
            "        .alias(target_col)\n"
            "    ).drop(f'{target_col}_new')\n"
            "    \n"
            "    return result"
        )
    
    # Add helper for date range filtering
    defs["APPLY_DATE_FILTER"] = (
        "def APPLY_DATE_FILTER(expr: pl.Expr, col_name: str, start_date: str, end_date: str, date_col: str = 'DATE', preserve_existing: bool = False) -> pl.Expr:\n"
        '    """Apply expression only to rows within date range, using null or existing values for other rows.\n'
        "    \n"
        "    Args:\n"
        "        expr: Polars expression to apply\n"
        "        col_name: Name of the column being updated\n"
        "        start_date: Start date string (e.g., '2020-01-01' or '01Feb2020')\n"
        "        end_date: End date string (e.g., '2020-12-31' or '*' for today)\n"
        "        date_col: Name of the date column (default 'DATE')\n"
        "        preserve_existing: If True, preserve existing column values outside date range (default False)\n"
        "    \n"
        "    Returns:\n"
        "        Expression that applies to filtered rows only\n"
        '    """\n'
        "    import polars as pl\n"
        "    from datetime import datetime, date as dt_date\n"
        "    \n"
        "    # Parse start date - support multiple formats\n"
        "    def parse_date(date_str):\n"
        "        # Try YYYY-MM-DD format\n"
        "        try:\n"
        "            return datetime.strptime(date_str, '%Y-%m-%d').date()\n"
        "        except ValueError:\n"
        "            pass\n"
        "        # Try ddMMMYYYY format (e.g., 01Feb2020)\n"
        "        try:\n"
        "            return datetime.strptime(date_str, '%d%b%Y').date()\n"
        "        except ValueError:\n"
        "            pass\n"
        "        # Try YYYYQN format (e.g., 2020Q1)\n"
        "        import re\n"
        "        m = re.match(r'^(\\d{4})Q([1-4])$', date_str, re.IGNORECASE)\n"
        "        if m:\n"
        "            year = int(m.group(1))\n"
        "            quarter = int(m.group(2))\n"
        "            month = (quarter - 1) * 3 + 1\n"
        "            return dt_date(year, month, 1)\n"
        "        raise ValueError(f'Cannot parse date: {date_str}')\n"
        "    \n"
        "    start = parse_date(start_date)\n"
        "    \n"
        "    # Handle special end date '*' (up to today)\n"
        "    if end_date == '*':\n"
        "        end = dt_date.today()\n"
        "    else:\n"
        "        end = parse_date(end_date)\n"
        "    \n"
        "    # Apply expression only within date range\n"
        "    # If preserve_existing is True, use existing column values outside the range\n"
        "    # Otherwise, use null for rows outside the range\n"
        "    otherwise_expr = pl.col(col_name) if preserve_existing else pl.lit(None)\n"
        "    return pl.when(\n"
        "        (pl.col(date_col) >= start) & (pl.col(date_col) <= end)\n"
        "    ).then(expr).otherwise(otherwise_expr)"
    )

    return defs
