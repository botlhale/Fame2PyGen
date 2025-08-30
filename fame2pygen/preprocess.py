"""
Preprocessing stage for Fame2PyGen:
 - Collect list definitions
 - Unroll supported FAME loop constructs into explicit 'set' lines
 - Return a flat list of commands to feed into the parser
"""

import re
from typing import List, Dict, Tuple

LIST_DEF_RE = re.compile(r"^\s*([a-zA-Z0-9_$]+)\s*=\s*\{([^}]*)\}\s*$")
TOP_LOOP_RE = re.compile(r"^\s*loop\s+for\s+%([a-zA-Z0-9_]+)\s*=\s*1\s*to\s*length\(\s*([a-zA-Z0-9_$]+)\s*\)\s*$", re.IGNORECASE)
EXTRACT_LOOP_RE = re.compile(r"^\s*loop\s+for\s+%([a-zA-Z0-9_]+)\s+in\s+\{\s*extract\(\s*([a-zA-Z0-9_$]+)\s*,\s*%([a-zA-Z0-9_]+)\s*\)\s*\}\s*$", re.IGNORECASE)
SET_RE = re.compile(r"^\s*set\s+(.+)$", re.IGNORECASE)
END_LOOP_RE = re.compile(r"^\s*end\s+loop\s*$", re.IGNORECASE)

class LoopUnrollError(Exception):
    pass

def _normalize_list_items(content: str) -> List[str]:
    return [x.strip() for x in content.split(',') if x.strip()]

def _collect_list_def(line: str, lists: Dict[str, List[str]]) -> bool:
    m = LIST_DEF_RE.match(line)
    if not m:
        return False
    name, content = m.groups()
    lists[name.lower()] = _normalize_list_items(content)
    return True

def preprocess_lines(lines: List[str]) -> List[str]:
    """
    Entry point:
      - Extract list definitions into dict
      - Unroll recognized loops
      - Preserve non-loop, non-definition lines
    """
    lists: Dict[str, List[str]] = {}
    output: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        # Capture list definitions
        if _collect_list_def(line, lists):
            # Keep definition line (parser may still use it if needed)
            output.append(line)
            i += 1
            continue
        # Detect start of a top-level loop
        m_top = TOP_LOOP_RE.match(line)
        if m_top:
            # Extract full loop block
            block_lines, advance = _capture_loop_block(lines, i)
            unrolled = _unroll_block(block_lines, lists)
            output.extend(unrolled)
            i += advance
            continue
        # Pass through everything else
        output.append(line)
        i += 1
    return output

def _capture_loop_block(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    """Capture nested loop block until matching end loop count returns to zero."""
    depth = 0
    collected: List[str] = []
    i = start_idx
    while i < len(lines):
        l = lines[i].rstrip()
        if re.match(r"^\s*loop\s+for", l, re.IGNORECASE):
            depth += 1
        if END_LOOP_RE.match(l):
            depth -= 1
        collected.append(l)
        i += 1
        if depth == 0:
            break
    if depth != 0:
        raise LoopUnrollError("Unterminated loop block starting at line index {}".format(start_idx))
    return collected, len(collected)

def _unroll_block(block_lines: List[str], lists: Dict[str, List[str]]) -> List[str]:
    """
    Currently supports a specific nested pattern:
      loop for %i = 1 to length(LIST_A)
        loop for %ms in { extract(MONTHLY_LIST, %i) }
          loop for %qs in { extract(QUARTERLY_LIST, %i) }
            set %qs = convert(%ms, q, ave, end)
          end loop
        end loop
      end loop

    General strategy:
      - Parse top loop gives index variable and reference list (only length used for iteration count)
      - Recursively process nested extract loops mapping placeholders -> concrete list elements
      - Expand set statements
    """
    # Parse top line
    top_match = TOP_LOOP_RE.match(block_lines[0])
    if not top_match:
        raise LoopUnrollError("Unsupported top loop syntax: {}".format(block_lines[0]))
    idx_var, length_list = top_match.groups()
    length_list_lower = length_list.lower()
    if length_list_lower not in lists:
        raise LoopUnrollError(f"List '{length_list}' not defined before loop.")
    total = len(lists[length_list_lower])

    # We'll recursively expand lines skipping first and final 'end loop'
    inner_lines = block_lines[1:-1]
    expanded: List[str] = []
    for idx in range(1, total + 1):
        expanded.extend(_expand_inner(inner_lines, idx_var, idx, lists))
    return expanded

def _expand_inner(lines: List[str], idx_var: str, idx: int, lists: Dict[str, List[str]],
                  context_map=None) -> List[str]:
    """
    Expand inner portion for a specific index 'idx' (1-based).
    context_map stores current placeholder -> concrete value assignments (e.g. %ms -> gdp_m).
    """
    if context_map is None:
        context_map = {}

    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Nested loop?
        if re.match(r"^\s*loop\s+for", line, re.IGNORECASE):
            block, advance = _capture_loop_block(lines, i)
            out.extend(_expand_nested_loop(block, idx_var, idx, lists, context_map))
            i += advance
            continue
        # 'set' line?
        m_set = SET_RE.match(line)
        if m_set:
            set_rhs = m_set.group(1)
            # Replace placeholders like %qs, %ms, and any %<idx_var> with current mappings
            concrete = _apply_context(set_rhs, context_map, idx_var, idx)
            out.append(concrete)
            i += 1
            continue
        # Other lines inside loops (comments or blank)
        i += 1
    return out

def _expand_nested_loop(block_lines: List[str], idx_var: str, idx: int,
                        lists: Dict[str, List[str]], context_map):
    """
    Expand inner 'loop for %alias in { extract(list_name, %idx_var) }'
    Provide a single mapping alias -> lists[list_name][idx-1].
    """
    head = block_lines[0]
    m = EXTRACT_LOOP_RE.match(head)
    if not m:
        raise LoopUnrollError(f"Unsupported nested loop syntax: {head}")
    alias, list_name, ref_idx_var = m.groups()
    if ref_idx_var != idx_var:
        raise LoopUnrollError(f"Index variable mismatch: expected %{idx_var}, found %{ref_idx_var}")
    list_name_lower = list_name.lower()
    if list_name_lower not in lists:
        raise LoopUnrollError(f"Referenced list '{list_name}' not defined.")
    items = lists[list_name_lower]
    if idx < 1 or idx > len(items):
        raise LoopUnrollError(f"Index {idx} out of bounds for list '{list_name}'")
    aliased_value = items[idx - 1]
    # New context layer
    new_context = dict(context_map)
    new_context[f"%{alias}"] = aliased_value
    # Recurse into its body
    body_lines = block_lines[1:-1]
    return _expand_inner(body_lines, idx_var, idx, lists, new_context)

def _apply_context(set_line: str, context_map: Dict[str, str], idx_var: str, idx: int) -> str:
    """Replace placeholders in a 'set' line."""
    result = set_line
    # Replace index placeholder if present (rare inside 'set', but safe)
    result = result.replace(f"%{idx_var}", str(idx))
    for placeholder, concrete in context_map.items():
        result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(placeholder)}(?![A-Za-z0-9_])", concrete, result)
    # Drop any leading spaces that came after substitution
    return result.strip()