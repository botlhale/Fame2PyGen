"""
┌─────────────────────────────────────┐
│            Fame2PyGen               │
│         FAME → Python               │
│      Main Script Generator          │
└─────────────────────────────────────┘
"""
import re
import polars as pl
import formulagen

def variable_to_function_name(var):
    """Convert a variable name to its corresponding function name."""
    if var.endswith('$'):
        # Convert a$ to A_, pa$ to PA_, etc.
        return var[:-1].upper() + '_'
    else:
        # Convert a to A, pa to PA, etc.
        return var.upper()

def should_use_function_call(target, expr):
    """Determine if we should use a function call for this expression.
    Only use function calls for simple expressions that reference base variables (v123, v143)."""
    # Check if the expression only contains v123, v143, numbers, and basic operators
    import re
    # Remove all v123, v143, numbers, operators and whitespace
    cleaned = re.sub(r'(v123|v143|\d+|[+\-*/().\s])', '', expr)
    # If nothing is left, this is a simple expression that can use function calls
    return len(cleaned) == 0

def preprocess_commands(fame_script):
    lines = fame_script.strip().split('\n')
    alias_dict = {}
    expanded_lines = []
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

    def resolve_alias(alias):
        if alias not in alias_dict:
            return [alias]
        result = []
        for item in alias_dict[alias]:
            if item in alias_dict:
                result.extend(resolve_alias(item))
            else:
                result.append(item)
        return result

    for k in alias_dict:
        alias_dict[k] = resolve_alias(k)

    final_cmds = []
    i = 0
    while i < len(expanded_lines):
        line = expanded_lines[i]
        loop_m = re.match(r'loop\s+([a-zA-Z0-9_]+)\s+as\s+([a-zA-Z0-9_]+):', line)
        if loop_m:
            alias = loop_m.group(1)
            varname = loop_m.group(2)
            block = []
            i += 1
            while not expanded_lines[i].startswith('end loop'):
                block.append(expanded_lines[i])
                i += 1
            if 'fishvol' in block[0]:
                for a in alias_dict[alias]:
                    new_block = [b.replace(varname, a) for b in block]
                    final_cmds.extend(new_block)
            else:
                for item in alias_dict[alias]:
                    for b in block:
                        final_cmds.append(b.replace(varname, item))
            i += 1
            continue
        final_cmds.append(line)
        i += 1

    parsed = [formulagen.parse_command(cmd) for cmd in final_cmds if formulagen.parse_command(cmd)]
    return parsed, alias_dict

def get_computation_levels(parsed_commands):
    all_targets = set()
    for c in parsed_commands:
        if c['type'] == 'declaration':
            all_targets.update(c['targets'])
        elif 'target' in c:
            all_targets.add(c['target'])
    all_refs = set()
    for c in parsed_commands:
        for ref in c.get('refs', []):
            all_refs.add(ref)
    decls = [t for c in parsed_commands if c['type'] == 'declaration' for t in c['targets']]
    input_vars = sorted(list(all_refs - all_targets))
    deps = {}
    for c in parsed_commands:
        if c['type'] == 'declaration':
            continue
        if 'target' in c:
            deps[c['target']] = set([r for r in c.get('refs', []) if r in all_targets])
    levels = []
    available = set(decls + input_vars)
    remaining = set(deps.keys())
    if decls:
        levels.append(decls)
    while remaining:
        level = []
        for t in list(remaining):
            needed = deps.get(t, set())
            if needed.issubset(available):
                level.append(t)
        if not level:
            missing = set()
            for t in remaining:
                needed = deps.get(t, set())
                missing.update(needed - available)
            raise Exception(f"Circular or missing dependencies! Missing: {missing}")
        levels.append(level)
        available.update(level)
        for t in level:
            remaining.remove(t)
    return levels

def convert_variable_name(name):
    """Convert FAME variable names to Python-compatible names.
    
    Args:
        name (str): FAME variable name (may contain $)
        
    Returns:
        str: Python-compatible variable name ($ replaced with _)
    """
    return name.replace('$', '_')

def generate_function_name(variable_name):
    """Generate a Python function name from a FAME variable name.
    
    Args:
        variable_name (str): FAME variable name
        
    Returns:
        str: Capitalized Python function name
    """
    converted = convert_variable_name(variable_name)
    return converted.upper()

def convert_expression_to_polars(expr, refs=None):
    """Convert a FAME expression to Polars expression syntax.
    
    Args:
        expr (str): FAME expression
        refs (list): List of variable references in the expression
        
    Returns:
        str: Polars expression string
    """
    import re
    
    if refs is None:
        refs = []
    
    # Find all variable names (including those with $)
    variables = re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr)
    polars_expr = expr
    
    # Sort variables by length (descending) to avoid partial replacements
    variables = sorted(set(variables), key=len, reverse=True)
    
    for var in variables:
        # Skip if it's a number
        if var.isdigit():
            continue
            
        # Escape the variable name for regex and use negative lookbehind/lookahead
        escaped_var = re.escape(var)
        pattern = r'(?<![a-zA-Z0-9_$])' + escaped_var + r'(?![a-zA-Z0-9_$])'
        converted_var = convert_variable_name(var)
        
        # If this variable is in refs (function parameters), use it directly
        # Otherwise, use pl.col()
        if var in refs:
            polars_expr = re.sub(pattern, converted_var, polars_expr)
        else:
            polars_expr = re.sub(pattern, f'pl.col("{converted_var}")', polars_expr)
    
    return polars_expr

def generate_formulas_py(parsed_commands=None, input_variables=None):
    """Generate formulas.py with individual function definitions for each FAME assignment.
    
    Args:
        parsed_commands (list): List of parsed FAME commands
        input_variables (list): List of input variable names (from external data)
        
    Returns:
        str: Generated Python code
    """
    if parsed_commands is None:
        parsed_commands = []
    if input_variables is None:
        input_variables = []
    
    script = []
    script.append('"""')
    script.append('┌─────────────────────────────────────┐')
    script.append('│            Fame2PyGen               │')
    script.append('│         FAME → Python               │')
    script.append('│     Auto-Generated Formulas        │')
    script.append('└─────────────────────────────────────┘')
    script.append('')
    script.append('This file was automatically generated by Fame2PyGen')
    script.append('Contains individual formula functions for FAME script conversion')
    script.append('"""')
    script.append('import polars as pl')
    script.append('import ple')
    script.append('from typing import List, Tuple')
    script.append('')
    
    # Generate individual function definitions for each command
    for cmd in parsed_commands:
        if cmd['type'] == 'declaration':
            # Skip declarations - they don't generate functions
            continue
        elif cmd['type'] == 'simple':
            # Generate function for simple mathematical expression
            target = cmd['target']
            expr = cmd['rhs']
            func_name = generate_function_name(target)
            converted_target = convert_variable_name(target)
            
            # Determine which references are computed variables vs input variables
            refs = cmd.get('refs', [])
            computed_refs = [ref for ref in refs if ref not in input_variables]
            input_refs = [ref for ref in refs if ref in input_variables]
            
            has_computed_refs = len(computed_refs) > 0
            
            if has_computed_refs:
                # Function with computed variable parameters - return pl.Series
                params = ', '.join([f'{convert_variable_name(ref)}: pl.Series' for ref in computed_refs])
                script.append(f'def {func_name}({params}) -> pl.Series:')
                script.append('    """')
                script.append(f'    Computes the values for the {converted_target} time series or variable using Polars expressions.')
                script.append('    Derived from FAME script(s):')
                script.append(f'        set {target} = {expr}')
                script.append('')
                script.append('    Returns:')
                script.append('        pl.Series: Polars Series to compute the time series or variable values.')
                script.append('    """')
                
                # Convert expression to Polars syntax with computed refs as parameters
                polars_expr = convert_expression_to_polars(expr, computed_refs)
                
                script.append('    res = (')
                script.append(f'        {polars_expr}')
                script.append('    )')
                script.append('    return res')
            else:
                # Function with only input variable references - return pl.Expr with alias
                script.append(f'def {func_name}() -> pl.Expr:')
                script.append('    """')
                script.append(f'    Computes the values for the {converted_target} time series or variable using Polars expressions.')
                script.append('    Derived from FAME script(s):')
                script.append(f'        set {target} = {expr}')
                script.append('')
                script.append('    Returns:')
                script.append('        pl.Expr: Polars expression to compute the time series or variable values.')
                script.append('    """')
                
                # Convert expression to Polars syntax using pl.col() for input variables
                polars_expr = convert_expression_to_polars(expr, [])
                
                script.append('    res = (')
                script.append(f'        {polars_expr}')
                script.append('    )')
                script.append(f'    return res.alias("{converted_target}")')
            
            script.append('')
            
        elif cmd['type'] == 'mchain':
            # Generate function for mchain expression
            target = cmd['target']
            expr = cmd['expr']
            base_year = cmd['base_year']
            func_name = generate_function_name(target)
            converted_target = convert_variable_name(target)
            
            refs = cmd.get('refs', [])
            computed_refs = [ref for ref in refs if ref not in input_variables]
            
            if computed_refs:
                params = ', '.join([f'{convert_variable_name(ref)}: pl.Series' for ref in computed_refs])
                script.append(f'def {func_name}({params}) -> pl.Series:')
                script.append('    """')
                script.append(f'    Computes the values for the {converted_target} time series or variable using Polars expressions.')
                script.append('    Derived from FAME script(s):')
                script.append(f'        set {target} = $mchain("{expr}",{base_year})')
                script.append('')
                script.append('    Returns:')
                script.append('        pl.Series: Polars Series to compute the time series or variable values.')
                script.append('    """')
                
                # For mchain, convert the expression and create a simple calculation
                polars_expr = convert_expression_to_polars(expr, computed_refs)
                
                script.append('    res = (')
                script.append(f'        # TODO: Fix this - placeholder for now')
                script.append(f'        {polars_expr}')
                script.append('    )')
                script.append('    return res')
                script.append('')
    
    # Add fallback generic functions for compatibility
    script.append('# Generic fallback functions for compatibility')
    script.append('def CONVERT(series: pl.DataFrame, as_freq: str, to_freq: str, technique: str, observed: str) -> pl.Expr:')
    script.append('    """Generic wrapper for convert using \'ple.convert\'.\"\"\"')
    script.append('    return ple.convert(series, "DATE", as_freq=as_freq, to_freq=to_freq, technique=technique, observed=observed)')
    script.append('')
    script.append('def FISHVOL(series_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, rebase_year: int) -> pl.Expr:')
    script.append('    """Generic wrapper for $fishvol_rebase using \'ple.fishvol\'.\"\"\"')
    script.append('    return ple.fishvol(series_pairs, date_col, rebase_year)')
    script.append('')
    script.append('def CHAIN(price_quantity_pairs: List[Tuple[pl.Expr, pl.Expr]], date_col: pl.Expr, year: str) -> pl.Expr:')
    script.append('    """Generic wrapper for $mchain using \'ple.chain\'.\"\"\"')
    script.append('    return ple.chain(price_quantity_pairs=price_quantity_pairs, date_col=date_col, index_year=int(year))')
    script.append('')
    script.append('def DECLARE_SERIES(df, name):')
    script.append('    return pl.lit(None, dtype=pl.Float64).alias(name)')
    script.append('')
    
    return '\n'.join(script)

def generate_convpy4rmfame_py(parsed_commands, alias_dict, levels):
    script = []
    script.append('"""')
    script.append('┌─────────────────────────────────────┐')
    script.append('│            Fame2PyGen               │')
    script.append('│         FAME → Python               │')
    script.append('│    Auto-Generated Pipeline         │')
    script.append('└─────────────────────────────────────┘')
    script.append('')
    script.append('This file was automatically generated by Fame2PyGen')
    script.append('Contains the conversion pipeline from FAME script to Python')
    script.append('"""')
    script.append("import polars as pl")
    script.append("from formulas import *")
    # Find all input/source columns
    all_targets = set()
    all_refs = set()
    for c in parsed_commands:
        if c['type'] == 'declaration':
            all_targets.update(c['targets'])
        elif 'target' in c:
            all_targets.add(c['target'])
        for ref in c.get('refs', []):
            all_refs.add(ref)
    input_vars = sorted(list(all_refs - all_targets))
    script.append("df = pl.DataFrame({")
    script.append("    'date': pl.date_range(pl.date(2019, 1, 1), pl.date(2025, 1, 1), '1mo', eager=True),")
    for col in input_vars:
        if col == 'date': continue
        script.append(f"    '{col}': pl.Series('{col}', range(1, 74)),")
    script.append("})")
    for alias, items in alias_dict.items():
        script.append(f"{alias} = {items}")
    script.append("# ---- DECLARE SERIES ----")
    for lvl, targets in enumerate(levels):
        cmds = [c for c in parsed_commands if (c.get('target') in targets) or (c['type'] == 'declaration' and set(c['targets']).intersection(targets))]
        for c in cmds:
            if c['type'] == 'declaration':
                for t in c['targets']:
                    script.append(f"# Declare series: {t}")
                    script.append(f"df = df.with_columns([DECLARE_SERIES(df, '{t}')])")
    script.append("# ---- COMPUTATIONS ----")
    for lvl, targets in enumerate(levels):
        cmds = [c for c in parsed_commands if (c.get('target') in targets)]
        for c in cmds:
            if c['type'] == 'fishvol_list':
                vols = alias_dict.get(c['refs'][0], [c['refs'][0]])
                prices = alias_dict.get(c['refs'][1], [c['refs'][1]])
                script.append(f"# fishvol function: {c['target']} = fishvol({vols}, {prices}, year={c['year']})")
                script.append(f"df = df.with_columns([FISHVOL(df, {vols}, {prices}, year={c['year']}).alias('{c['target']}')])")
            elif c['type'] == 'convert':
                freq, method, period = c['params']
                source = c['refs'][0]
                script.append(f"# convert function: {c['target']} = convert({source}, {freq}, {method}, {period})")
                script.append(f"df = df.with_columns([CONVERT(df, '{source}', '{freq}', '{method}', '{period}').alias('{c['target']}')])")
            elif c['type'] == 'mchain':
                refs = c['refs']
                script.append(f"# mchain function: {c['target']} = chain({refs}, base_year={c['base_year']})")
                # Convert refs list to pairs format for CHAIN function
                pair_exprs = []
                for i in range(0, len(refs), 2):
                    if i + 1 < len(refs):
                        pair_exprs.append(f"(pl.col('{refs[i]}'), pl.col('{refs[i+1]}'))")
                    else:
                        # If odd number, pair with itself or skip
                        pair_exprs.append(f"(pl.col('{refs[i]}'), pl.col('{refs[i]}'))")
                pairs_str = '[' + ', '.join(pair_exprs) + ']'
                script.append(f"df = df.with_columns([CHAIN({pairs_str}, pl.col(\"date\"), \"{c['base_year']}\").alias('{c['target']}')])")
            elif c['type'] == 'pct':
                source = c['refs'][0]
                lag = c['params'][0]
                script.append(f"# Using with_columns for pct function")
                script.append(f"df = df.with_columns([ple.pct(pl.col('{source}'), lag={lag}).alias('{c['target']}')])")
            elif c['type'] == 'interp':
                source = c['refs'][0]
                method = c['params'][0]
                script.append(f"# Using with_columns for interp function")
                script.append(f"df = df.with_columns([ple.interp(pl.col('{source}'), method='{method}').alias('{c['target']}')])")
            elif c['type'] == 'overlay':
                source1, source2 = c['refs']
                script.append(f"# Using with_columns for overlay function")
                script.append(f"df = df.with_columns([ple.overlay(pl.col('{source1}'), pl.col('{source2}')).alias('{c['target']}')])")
            elif c['type'] == 'mave':
                source = c['refs'][0]
                window = c['params'][0]
                script.append(f"# Using with_columns for mave function")
                script.append(f"df = df.with_columns([ple.mave(pl.col('{source}'), window={window}).alias('{c['target']}')])")
            elif c['type'] == 'mavec':
                source = c['refs'][0]
                window = c['params'][0]
                script.append(f"# Using with_columns for mavec function")
                script.append(f"df = df.with_columns([ple.mavec(pl.col('{source}'), window={window}).alias('{c['target']}')])")
            elif c['type'] == 'copy':
                source = c['refs'][0]
                script.append(f"# Using with_columns for copy function")
                script.append(f"df = df.with_columns([ple.copy(pl.col('{source}')).alias('{c['target']}')])")
            elif c['type'] == 'simple':
                # Generate proper Polars expression for mathematical operations
                expr = c['rhs']
                target = c['target']
                
                # Check if this is a simple expression that can use function calls
                if should_use_function_call(target, expr):
                    # Use function call approach
                    func_name = variable_to_function_name(target)
                    script.append(f"# Mathematical expression: {target} = {expr}")
                    script.append(f"df = df.with_columns([{func_name}().alias('{target}')])")
                else:
                    # Use traditional polars expression approach for complex expressions
                    import re
                    # Find all variable names, including those with $ (must start with letter)
                    variables = re.findall(r'[a-zA-Z][a-zA-Z0-9_$]*', expr)
                    polars_expr = expr
                    
                    # Sort variables by length (descending) to avoid partial replacements
                    variables = sorted(set(variables), key=len, reverse=True)
                    
                    for var in variables:
                        # Escape the variable name for regex and use negative lookbehind/lookahead
                        # to ensure we don't replace parts of other variable names
                        escaped_var = re.escape(var)
                        pattern = r'(?<![a-zA-Z0-9_$])' + escaped_var + r'(?![a-zA-Z0-9_$])'
                        polars_expr = re.sub(pattern, f'pl.col("{var}")', polars_expr)
                    
                    script.append(f"# Mathematical expression: {target} = {expr}")
                    script.append(f"df = df.with_columns([({polars_expr}).alias('{target}')])")
    script.append("print('Computation finished')")
    return '\n'.join(script)

if __name__ == '__main__':
    # Display branding banner
    print("┌─────────────────────────────────────┐")
    print("│            Fame2PyGen               │")
    print("│         FAME → Python               │")
    print("│      Code Generator v1.0            │")
    print("└─────────────────────────────────────┘")
    print("")
    
    fame_script = '''
a$=v123*12
a=v143*12
b=v143*2
b$=v123*6
c$=v123*5
d=v123*1
e=v123*2
f=v123*3
g=v123*4
h=v123*5
pa$=v123*3
pa=v143*4
pb=v143*1
pb$=v123*1
pc$=v123*2
pd=v123*3
pe=v123*4
pf=v123*5
pg=v123*1
ph=v123*2
aa=a$/a
bb=aa+a
paa=pa$/pa
pbb=pa+paa
hxz = (b*12)/a
abc$_d1=a$+b$+c$+a
c1 = $mchain("a + b + c$ + d + e + f + g + h",2017)
'''
    parsed, alias_dict = preprocess_commands(fame_script)
    levels = get_computation_levels(parsed)
    
    # Calculate input variables (those referenced but not computed)
    all_targets = set()
    all_refs = set()
    for c in parsed:
        if c['type'] == 'declaration':
            all_targets.update(c['targets'])
        elif 'target' in c:
            all_targets.add(c['target'])
        for ref in c.get('refs', []):
            all_refs.add(ref)
    input_vars = sorted(list(all_refs - all_targets))
    
    formulas_py = generate_formulas_py(parsed, input_vars)
    with open("formulas.py", "w", encoding="utf-8") as f:
        f.write(formulas_py)
    convpy4rmfame_py = generate_convpy4rmfame_py(parsed, alias_dict, levels)
    with open("convpy4rmfame.py", "w", encoding="utf-8") as f:
        f.write(convpy4rmfame_py)
    print("✓ formulas.py and convpy4rmfame.py have been generated.")
    print("✓ Fame2PyGen conversion completed successfully!")