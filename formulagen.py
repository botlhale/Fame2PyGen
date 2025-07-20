import re

def parse_series_declaration(line):
    m = re.match(r'series\s+([a-zA-Z0-9_,\s]+)', line)
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
    m = re.match(r'([a-zA-Z0-9_]+)\s*=\s*fishvol_rebase\(([a-zA-Z0-9_]+),\s*([a-zA-Z0-9_]+),\s*([0-9]+)\)', line)
    if m:
        return {'type': 'fishvol_list', 'target': m.group(1), 'refs': [m.group(2), m.group(3)], 'year': m.group(4)}
    return None

def parse_convert_command(line):
    m = re.match(r'([a-zA-Z0-9_]+)\s*=\s*convert\(([a-zA-Z0-9_]+),\s*([qa]),\s*([a-zA-Z]+),\s*([a-zA-Z]+)\)', line)
    if m:
        return {
            'type': 'convert',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3), m.group(4), m.group(5)]
        }
    return None

def parse_mchain_command(line):
    m = re.match(r'([a-zA-Z0-9_]+)\s*=\s*\$mchain\("([^"]+)",\s*"([0-9]+)"\)', line)
    if m:
        expr = m.group(2)
        refs = re.findall(r'[a-zA-Z0-9_]+', expr)
        price_refs = ['p'+r if not r.startswith('p') else r for r in refs]
        all_refs = refs + price_refs
        return {
            'type': 'mchain',
            'target': m.group(1),
            'expr': expr,
            'refs': all_refs,
            'base_year': m.group(3)
        }
    return None

def parse_simple_command(line):
    if re.match(r'[a-zA-Z0-9_]+\s*=\s*\{.+\}', line):
        return None
    m = re.match(r'([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_+\-\*/\s]+)', line)
    if m:
        rhs = m.group(2)
        refs = re.findall(r'[a-zA-Z0-9_]+', rhs)
        return {'type': 'simple', 'target': m.group(1), 'rhs': rhs, 'refs': refs}
    return None

def parse_command(line):
    for parser in [
        parse_series_declaration,
        parse_freq_command,
        parse_fishvol_list_command,
        parse_convert_command,
        parse_mchain_command,
        parse_simple_command
    ]:
        result = parser(line)
        if result:
            return result
    return None