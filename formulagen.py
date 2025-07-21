"""
┌─────────────────────────────────────┐
│            Fame2PyGen               │
│         FAME → Python               │
│        Formula Generator            │
└─────────────────────────────────────┘
"""
import re

def parse_series_declaration(line):
    m = re.match(r'series\s+([a-zA-Z0-9_$,\s]+)', line)
    if m:
        targets = [t.strip() for t in m.group(1).split(',')]
        return {'type': 'declaration', 'targets': targets}
    return None

def parse_freq_command(line):
    m = re.match(r'freq\s+([qa])', line)
    if m:
        return {'type': 'freq', 'freq': m.group(1)}
    return None

def parse_fishvol_list_command(line):
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*fishvol_rebase\(([a-zA-Z0-9_$]+),\s*([a-zA-Z0-9_$]+),\s*([0-9]+)\)', line)
    if m:
        return {'type': 'fishvol_list', 'target': m.group(1), 'refs': [m.group(2), m.group(3)], 'year': m.group(4)}
    return None

def parse_convert_command(line):
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*convert\(([a-zA-Z0-9_$]+),\s*([qa]),\s*([a-zA-Z]+),\s*([a-zA-Z]+)\)', line)
    if m:
        return {
            'type': 'convert',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3), m.group(4), m.group(5)]
        }
    return None

def parse_mchain_command(line):
    # Handle properly formatted mchain: $mchain("expr", "year")
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$mchain\("([^"]+)",\s*"([0-9]+)"\)', line)
    if m:
        expr = m.group(2)
        # Extract variable references (exclude pure numbers and operators)
        refs = []
        for match in re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr):
            refs.append(match)
        return {
            'type': 'mchain',
            'target': m.group(1),
            'expr': expr,
            'refs': refs,
            'base_year': m.group(3)
        }
    
    # Handle mchain without quotes around year: $mchain("expr",year)
    m3 = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$mchain\("([^"]+)",\s*([0-9]+)\)', line)
    if m3:
        expr = m3.group(2)
        # Extract variable references (exclude pure numbers and operators)
        refs = []
        for match in re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr):
            refs.append(match)
        return {
            'type': 'mchain',
            'target': m3.group(1),
            'expr': expr,
            'refs': refs,
            'base_year': m3.group(3)
        }
    
    # Handle malformed mchain: $mchain("expr"year") - missing comma
    m2 = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$mchain\("([^"]+)"([0-9]+)"\)', line)
    if m2:
        expr = m2.group(2)
        # Extract variable references (exclude pure numbers and operators)
        refs = []
        for match in re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr):
            refs.append(match)
        return {
            'type': 'mchain',
            'target': m2.group(1),
            'expr': expr,
            'refs': refs,
            'base_year': m2.group(3)
        }
    
    return None

def parse_chain_sum_command(line):
    """Parse chain sum commands for complex chaining operations with sums."""
    # Handle chain sum: $chainsum("expr", year, [list])
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$chainsum\("([^"]+)",\s*([0-9]+),\s*\[([^\]]+)\]\)', line)
    if m:
        expr = m.group(2)
        base_year = m.group(3)
        var_list = [v.strip().strip('"\'') for v in m.group(4).split(',')]
        
        # Extract all variable references from expression and list
        refs = set()
        for match in re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr):
            refs.add(match)
        refs.update(var_list)
        
        return {
            'type': 'chainsum',
            'target': m.group(1),
            'expr': expr,
            'refs': list(refs),
            'var_list': var_list,
            'base_year': base_year
        }
    
    return None

def parse_pct_command(line):
    """Parse percentage change commands like: set growth = pct(series, 4)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*pct\(([a-zA-Z0-9_$]+)(?:,\s*([0-9]+))?\)', line)
    if m:
        return {
            'type': 'pct',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3) or '1']  # default lag of 1
        }
    return None

def parse_interp_command(line):
    """Parse interpolation commands like: set filled = interp(series, linear)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*interp\(([a-zA-Z0-9_$]+)(?:,\s*([a-zA-Z]+))?\)', line)
    if m:
        return {
            'type': 'interp',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3) or 'linear']  # default method
        }
    return None

def parse_overlay_command(line):
    """Parse overlay commands like: set combined = overlay(series1, series2)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*overlay\(([a-zA-Z0-9_$]+),\s*([a-zA-Z0-9_$]+)\)', line)
    if m:
        return {
            'type': 'overlay',
            'target': m.group(1),
            'refs': [m.group(2), m.group(3)],
            'params': []
        }
    return None

def parse_mave_command(line):
    """Parse moving average commands like: set smooth = mave(series, 12)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*mave\(([a-zA-Z0-9_$]+),\s*([0-9]+)\)', line)
    if m:
        return {
            'type': 'mave',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3)]  # window size
        }
    return None

def parse_mavec_command(line):
    """Parse centered moving average commands like: set centered = mavec(series, 12)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*mavec\(([a-zA-Z0-9_$]+),\s*([0-9]+)\)', line)
    if m:
        return {
            'type': 'mavec',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3)]  # window size
        }
    return None

def parse_copy_command(line):
    """Parse copy commands like: set backup = copy(original)"""
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*copy\(([a-zA-Z0-9_$]+)\)', line)
    if m:
        return {
            'type': 'copy',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': []
        }
    return None

def parse_simple_command(line):
    # Skip alias definitions
    if re.match(r'[a-zA-Z0-9_$]+\s*=\s*\{.+\}', line):
        return None
    
    # Support variable names with $ and complex expressions with parentheses
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*(.+)', line)
    if m:
        target = m.group(1)
        rhs = m.group(2).strip()
        
        # Extract variable references (exclude pure numbers)
        refs = []
        for match in re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', rhs):
            refs.append(match)
        
        return {'type': 'simple', 'target': target, 'rhs': rhs, 'refs': refs}
    return None

def parse_fishvol_enhanced_command(line):
    """Parse enhanced fishvol commands with better dependency handling."""
    # Enhanced fishvol with explicit dependencies: fishvol_rebase(vols, prices, year, deps=[...])
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*fishvol_rebase\(([a-zA-Z0-9_$]+),\s*([a-zA-Z0-9_$]+),\s*([0-9]+)(?:,\s*deps=\[([^\]]*)\])?\)', line)
    if m:
        deps = []
        if m.group(5):
            deps = [d.strip().strip('"\'') for d in m.group(5).split(',')]
        return {
            'type': 'fishvol_enhanced', 
            'target': m.group(1), 
            'refs': [m.group(2), m.group(3)], 
            'year': m.group(4),
            'dependencies': deps
        }
    return None

def parse_convert_enhanced_command(line):
    """Parse enhanced convert commands with better dependency handling."""
    # Enhanced convert with dependencies: convert(series, freq, method, period, deps=[...])
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*convert\(([a-zA-Z0-9_$]+),\s*([qa]),\s*([a-zA-Z]+),\s*([a-zA-Z]+)(?:,\s*deps=\[([^\]]*)\])?\)', line)
    if m:
        deps = []
        if m.group(6):
            deps = [d.strip().strip('"\'') for d in m.group(6).split(',')]
        return {
            'type': 'convert_enhanced',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3), m.group(4), m.group(5)],
            'dependencies': deps
        }
    return None

def parse_command(line):
    for parser in [
        parse_series_declaration,
        parse_freq_command,
        parse_fishvol_enhanced_command,
        parse_fishvol_list_command,
        parse_convert_enhanced_command,
        parse_convert_command,
        parse_chain_sum_command,
        parse_mchain_command,
        parse_pct_command,
        parse_interp_command,
        parse_overlay_command,
        parse_mave_command,
        parse_mavec_command,
        parse_copy_command,
        parse_simple_command
    ]:
        result = parser(line)
        if result:
            return result
    return None