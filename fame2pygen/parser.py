import re
from typing import List, Dict, Tuple, Optional
from .model import ParsedCommand
from .utils import sanitize_func_name

# --- Helper ---

def parse_mchain_expression(expr_str: str) -> List[Tuple[str, str]]:
    tokens = [t.strip() for t in re.split(r'([+-])', expr_str) if t and t.strip()]
    if tokens and tokens[0] not in ('+', '-'):
        tokens.insert(0, '+')
    return [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]

# --- Individual command parsers returning Optional[ParsedCommand] ---

def parse_series_declaration(line: str) -> Optional[ParsedCommand]:
    if match := re.match(r"^\s*series\s+(.+)$", line, re.IGNORECASE):
        vars_str = match.group(1)
        series_names = [v.strip() for v in vars_str.split(',') if v.strip()]
        if series_names:
            return ParsedCommand(type="declaration", raw=line, refs=series_names)
    return None

def parse_freq_command(line: str) -> Optional[ParsedCommand]:
    if freq_match := re.match(r"^\s*freq\s+([a-zA-Z0-9]+)\s*$", line, re.IGNORECASE):
        return ParsedCommand(type="freq", raw=line, freq=freq_match.group(1).lower())
    return None

def parse_list_assignment(line: str) -> Optional[ParsedCommand]:
    match = re.match(r"^\s*([a-zA-Z0-9_$]+)\s*=\s*\{(.+)\}\s*$", line, re.IGNORECASE)
    if match:
        target, content = match.groups()
        # Exclude single-item simple braces (handled by simple parser)
        rhs_after_equals = line.split('=', 1)[1]
        if ',' not in content or re.search(r"[\+\-\*\/]", rhs_after_equals.replace(f"{{{content}}}", "")):
            return None
        items = [item.strip() for item in content.split(',')]
        return ParsedCommand(type="list_alias", raw=line, target=target, refs=items, original_rhs=f"{{{content}}}")
    return None

def parse_convert_command(line: str) -> Optional[ParsedCommand]:
    var_pattern = r"[a-zA-Z0-9_$]+"
    if convert_match := re.match(fr"^\s*({var_pattern})\s*=\s*convert\((.+)\)\s*$", line, re.IGNORECASE):
        target, args_str = convert_match.groups()
        args = [arg.strip() for arg in re.split(r'\s*,\s*', args_str)]
        if len(args) == 4:
            source_series, to_freq, technique, observed = args
            return ParsedCommand(type="convert", raw=line, target=target, refs=[source_series],
                                 params=[to_freq, technique, observed])
    return None

def parse_fishvol_command(line: str) -> Optional[ParsedCommand]:
    var_pattern = r"[a-zA-Z0-9_$]+"
    fishvol_match = re.match(fr"^\s*({var_pattern})\s*=\s*\$?fishvol_rebase\((.+)\)(\s*[*/+\-].*)?$",
                             line, re.IGNORECASE)
    if fishvol_match:
        target, args_str, trailing_op = fishvol_match.groups()
        args = re.findall(r'\{([^}]+)\}|([^{},]+)', args_str)
        cleaned_args = [t[0] if t[0] else t[1].strip() for t in args]
        if len(cleaned_args) < 3:
            return None
        year = cleaned_args[-1]
        list_args = cleaned_args[:-1]
        if len(list_args) < 2:
            return None
        vols = [v.strip() for v in list_args[0].split(',')]
        prices = [p.strip() for p in list_args[1].split(',')]
        if len(vols) != len(prices):
            return None
        variable_refs = [s for s in vols + prices if not s.replace('.', '', 1).isdigit()]
        pairs = list(zip(vols, prices))
        return ParsedCommand(type="fishvol", raw=line, target=target, refs=variable_refs,
                             pairs=pairs, year=year,
                             trailing_op=trailing_op.strip() if trailing_op else None)
    return None

def parse_mchain_command(line: str) -> Optional[ParsedCommand]:
    var_pattern = r"[a-zA-Z0-9_$]+"
    mchain_pattern = fr"^\s*({var_pattern})\s*=\s*\$mchain\s*\(\s*\"(.*?)\",\s*\"(\d{{4}})\"\s*\)\s*$"
    if mchain_match := re.match(mchain_pattern, line, re.IGNORECASE):
        target, inner_expr, year = mchain_match.groups()
        terms = parse_mchain_expression(inner_expr)
        quantity_vars = [term[1] for term in terms]
        refs = [var for qv in quantity_vars for var in (qv, 'p' + qv)]
        return ParsedCommand(type="mchain", raw=line, target=target, refs=refs, terms=terms, year=year)
    return None

def parse_simple_command(line: str) -> Optional[ParsedCommand]:
    var_pattern = r"[a-zA-Z0-9_$.]+"
    if simple_match := re.match(fr"^\s*({var_pattern})\s*=\s*(.+)$", line, re.IGNORECASE):
        target, rhs_str = simple_match.groups()
        original_rhs = rhs_str.strip()
        rhs = re.sub(r'\{([^}]+)\}', r'\1', original_rhs)  # unwrap single-item braces
        all_potential_refs = re.findall(fr'({var_pattern})', rhs)
        refs = [r for r in all_potential_refs
                if not r.replace('.', '', 1).isdigit()
                and r.lower() != target.lower()]
        return ParsedCommand(type="simple", raw=line, target=target,
                             refs=list(set(refs)), rhs=rhs, original_rhs=original_rhs)
    return None

# Order of specialized parsers
PARSERS = [
    parse_series_declaration,
    parse_freq_command,
    parse_list_assignment,
    parse_fishvol_command,
    parse_convert_command,
    parse_mchain_command,
    parse_simple_command
]

def parse_fame_formula(line: str) -> Optional[ParsedCommand]:
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None
    for parser in PARSERS:
        result = parser(stripped)
        if result:
            return result
    # Could log a warning here
    return None

def parse_script(lines: List[str]) -> List[ParsedCommand]:
    results = []
    for line in lines:
        pc = parse_fame_formula(line)
        if pc:
            results.append(pc)
    return results