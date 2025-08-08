"""
┌───────────────────────────────────────────┐
│            Fame2PyGen                     │
│         FAME → Python                     │
│            cformgen (CLI)                 │
└───────────────────────────────────────────┘
"""
from typing import List
import argparse
import formulagen
import write_formulagen as gen


def generate_from_commands(commands: List[str], formulas_out: str = "cformulas.py", chain_out: str = "c_chain.py"):
    # Preprocess and parse
    parsed, _aliases = gen.preprocess_commands("\n".join(commands))
    # Generate formulas (no alias in functions)
    with open(formulas_out, "w", encoding="utf-8") as f:
        f.write(gen.generate_formulas_py_string(parsed))
    # Generate partitioned pipeline
    with open(chain_out, "w", encoding="utf-8") as f:
        f.write(gen.generate_pipeline_code(parsed))


def main():
    ap = argparse.ArgumentParser(description="Generate cformulas.py and c_chain.py from a FAME-like script.")
    ap.add_argument("-i", "--input", help="Path to FAME script file", required=True)
    ap.add_argument("--formulas", help="Output path for cformulas.py", default="cformulas.py")
    ap.add_argument("--chain", help="Output path for c_chain.py", default="c_chain.py")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        commands = [line.rstrip("\n") for line in f if line.strip()]

    generate_from_commands(commands, formulas_out=args.formulas, chain_out=args.chain)


if __name__ == "__main__":
    main()