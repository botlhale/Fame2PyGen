from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class ParsedCommand:
    type: str
    raw: str
    target: Optional[str] = None
    refs: List[str] = field(default_factory=list)
    original_rhs: Optional[str] = None
    rhs: Optional[str] = None
    params: List[str] = field(default_factory=list)
    terms: List[Tuple[str, str]] = field(default_factory=list)
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    year: Optional[str] = None
    trailing_op: Optional[str] = None
    freq: Optional[str] = None

@dataclass
class GenerationContext:
    fame_commands: List[str]
    parsed: List[ParsedCommand]
    has_mchain: bool = False
    has_convert: bool = False
    has_fishvol: bool = False
    has_pct: bool = False