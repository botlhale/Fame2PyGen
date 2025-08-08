"""
┌───────────────────────────────────────────┐
│            Fame2PyGen                     │
│         FAME → Python                     │
│        Formula Generator                  │
└───────────────────────────────────────────┘
"""
import re
from typing import Dict, List, Optional

def parse_series_declaration(line: str):
    m = re.match(r'series\s+([a-zA-Z0-9_$,\s]+)', line)
    if m:
        targets = [t.strip() for t in m.group(1).split(',')]
        return {'type': 'declaration', 'targets': targets}
    return None

def parse_freq_command(line: str):
    m = re.match(r'freq\s+([mqa])', line, flags=re.IGNORECASE)
    if m:
        return {'type': 'freq', 'freq': m.group(1).lower()}
    return None

def parse_convert_command(line: str):
    # convert(x, q, disc, ave)
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*convert\(\s*([a-zA-Z0-9_$]+)\s*,\s*([mqa])\s*,\s*([a-zA-Z]+)\s*,\s*([a-zA-Z]+)\s*\)\s*$', line, re.IGNORECASE)
    if m:
        return {
            'type': 'convert',
            'target': m.group(1),
            'refs': [m.group(2)],
            'params': [m.group(3).lower(), m.group(4).lower(), m.group(5).lower()],  # to_freq, technique, observed
        }
    return None

def parse_fishvol_command(line: str):
    # xyz = $fishvol_rebase({a}, {pa}, 2017)  or fishvol_rebase(a, pa, 2017)
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$?fishvol_rebase\(\s*\{?([a-zA-Z0-9_$]+)\}?\s*,\s*\{?([a-zA-Z0-9_$]+)\}?\s*,\s*([0-9]{4})\s*\)\s*(\*[^#\n\r]+)?$', line, re.IGNORECASE)
    if m:
        trailing_op = (m.group(5) or '').strip()
        return {
            'type': 'fishvol',
            'target': m.group(1),
            'refs': [m.group(2), m.group(3)],
            'year': int(m.group(4)),
            'trailing_op': trailing_op,  # e.g., *12
        }
    return None

def parse_mchain_command(line: str):
    # abc = $mchain("a$ + a - bb", "2025")
    m = re.match(r'([a-zA-Z0-9_$]+)\s*=\s*\$?mchain\(\s*"(.*?)"\s*,\s*"?(.*?)"?\s*\)\s*$', line, re.IGNORECASE)
    if not m:
        return None
    target = m.group(1)
    terms_str = m.group(2)
    year = m.group(3)

    # tokenize +/- terms: e.g., "a$ + a - bb" -> [('+','a$'), ('+','a'), ('-','bb')]
    tokens = re.findall(r'([+-]?)\s*([a-zA-Z0-9_$]+)', terms_str)
    terms = []
    for sign, var in tokens:
        op = '-' if sign == '-' else '+'
        terms.append((op, var))
    return {'type': 'mchain', 'target': target, 'terms': terms, 'year': year}

def parse_simple_assignment(line: str):
    # set a = v143*12 or a = v143*12
    m = re.match(r'(?:set\s+)?([a-zA-Z0-9_$]+)\s*=\s*([^#\n\r]+)$', line.strip(), re.IGNORECASE)
    if not m:
        return None
    target = m.group(1)
    expr = m.group(2).strip()
    # Extract variable refs (conservative)
    refs = re.findall(r'[a-zA-Z_][a-zA-Z0-9_$]*', expr)
    # Filter out common function names/operators, numbers handled by not matching
    return {'type': 'simple', 'target': target, 'expr': expr, 'refs': list(set(refs))}

def parse_command(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    for parser in (parse_series_declaration, parse_freq_command, parse_fishvol_command, parse_convert_command, parse_mchain_command, parse_simple_assignment):
        res = parser(line)
        if res:
            return res
    return None
