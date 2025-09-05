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

def is_timeseries_numeric_expression(rhs: str) -> bool:
    """
    Detect if an expression contains numeric patterns that should be treated as time series references.
    Examples:
    - "1234*12" -> True (number * factor)
    - "1233+2334+4827" -> True (multiple numbers being added)
    - "123.45" -> False (decimal number, likely literal)
    - "12" -> False (simple number, likely literal)
    """
    # Check for multiplication pattern: number*factor
    if re.match(r'^\s*\d{3,}\s*\*\s*\d+\s*$', rhs):
        return True
    
    # Check for addition pattern with multiple numbers: number+number+...
    if '+' in rhs and re.match(r'^\s*\d{3,}(\s*\+\s*\d{3,})+\s*$', rhs):
        return True
    
    return False

def extract_timeseries_numbers(rhs: str) -> List[str]:
    """Extract numeric time series references from an expression."""
    numbers = []
    
    # For multiplication: extract the first number
    mult_match = re.match(r'^\s*(\d{3,})\s*\*\s*\d+\s*$', rhs)
    if mult_match:
        numbers.append(mult_match.group(1))
    
    # For addition: extract all numbers
    elif '+' in rhs and re.match(r'^\s*\d{3,}(\s*\+\s*\d{3,})+\s*$', rhs):
        numbers.extend(re.findall(r'\d{3,}', rhs))
    
    return numbers

def contains_time_shift_references(rhs: str) -> bool:
    """Detect if expression contains time series references with offsets like var[t+1]."""
    return bool(re.search(r'\w+\[t[+-]\d+\]', rhs))

def contains_pct_function(rhs: str) -> bool:
    """Detect if expression contains pct() function calls."""
    return bool(re.search(r'pct\s*\(', rhs, re.IGNORECASE))

def extract_time_shift_references(rhs: str) -> List[str]:
    """Extract variable names that use time shift notation like var[t+1]."""
    # Find all patterns like variable[t+offset] or variable[t-offset]
    matches = re.findall(r'(\w+)\[t[+-]\d+\]', rhs)
    return list(set(matches))

def extract_pct_function_args(rhs: str) -> List[str]:
    """Extract variable references from pct() function calls."""
    # Find pct function calls and extract their arguments
    pct_matches = re.findall(r'pct\s*\(\s*([^)]+)\s*\)', rhs, re.IGNORECASE)
    refs = []
    for match in pct_matches:
        # Parse the argument, which could be a variable with time shift like v23s[t+1]
        arg = match.strip()
        # Extract the base variable name from patterns like var[t+1]
        var_match = re.match(r'(\w+)\[t[+-]\d+\]', arg)
        if var_match:
            refs.append(var_match.group(1))
        else:
            # Regular variable reference
            if re.match(r'^\w+$', arg):
                refs.append(arg)
    return refs

def parse_set_command(line: str) -> Optional[ParsedCommand]:
    """Parse set commands like 'set variable = expression'"""
    set_match = re.match(r"^\s*set\s+(.+)$", line, re.IGNORECASE)
    if set_match:
        # Extract the assignment part and parse it as a simple command
        assignment = set_match.group(1).strip()
        # Create a temporary line without 'set' and parse it
        temp_line = assignment
        result = parse_simple_command(temp_line)
        if result:
            # Update the raw line to include 'set'
            result.raw = line
            return result
    return None
    """Extract variable references from pct() function calls."""
    # Find pct function calls and extract their arguments
    pct_matches = re.findall(r'pct\s*\(\s*([^)]+)\s*\)', rhs, re.IGNORECASE)
    refs = []
    for match in pct_matches:
        # Parse the argument, which could be a variable with time shift like v23s[t+1]
        arg = match.strip()
        # Extract the base variable name from patterns like var[t+1]
        var_match = re.match(r'(\w+)\[t[+-]\d+\]', arg)
        if var_match:
            refs.append(var_match.group(1))
        else:
            # Regular variable reference
            if re.match(r'^\w+$', arg):
                refs.append(arg)
    return refs

def parse_simple_command(line: str) -> Optional[ParsedCommand]:
    # Updated pattern to include square brackets for time series indices
    var_pattern = r"[a-zA-Z0-9_$.]+(?:\[[^\]]+\])?"
    if simple_match := re.match(fr"^\s*({var_pattern})\s*=\s*(.+)$", line, re.IGNORECASE):
        target, rhs_str = simple_match.groups()
        original_rhs = rhs_str.strip()
        rhs = re.sub(r'\{([^}]+)\}', r'\1', original_rhs)  # unwrap single-item braces
        
        # Extract regular variable references (simpler pattern for finding vars in expression)
        basic_var_pattern = r"[a-zA-Z0-9_$.]+"
        all_potential_refs = re.findall(fr'({basic_var_pattern})', rhs)
        
        # Filter out numeric literals, but keep variable names
        refs = [r for r in all_potential_refs
                if not r.replace('.', '', 1).isdigit()
                and r.lower() != target.lower().split('[')[0]  # Compare base variable name
                and r.lower() != 't'  # Filter out 't' from time shift patterns
                and not re.match(r'^pct$', r, re.IGNORECASE)]  # Exclude 'pct' function name
        
        # Add time shift variable references
        time_shift_refs = extract_time_shift_references(rhs)
        refs.extend(time_shift_refs)
        
        # Add pct function argument references
        pct_refs = extract_pct_function_args(rhs)
        refs.extend(pct_refs)
        
        # Check if this expression contains time series numeric patterns
        if is_timeseries_numeric_expression(rhs):
            # Add numeric time series references
            timeseries_numbers = extract_timeseries_numbers(rhs)
            refs.extend(timeseries_numbers)
        
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
    parse_set_command,
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