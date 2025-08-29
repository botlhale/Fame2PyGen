import re

def sanitize_func_name(name: str) -> str:
    name = name.replace('$', '_')
    return re.sub(r'[^a-zA-Z0-9_]', '', name).lower()

def is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False