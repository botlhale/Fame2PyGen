"""
┌───────────────────────────────────────────┐
│            Fame2PyGen                     │
│         FAME → Python                     │
│  chain_sum_write_formgen (orchestrator)   │
└───────────────────────────────────────────┘
"""
# This is a Python equivalent of the chain_sum_write_formgen.ipynb workflow.
# It invokes cformgen-like entry points and uses the partitioned sub-pipeline strategy.

from typing import List
import cformgen


def run_demo():
    fame_commands = """
freq m
a$=v123*12
a=v143*12
pa$=v123*3
pa=v143*4
aa=a$/a
paa=pa$/pa
bb=aa+a
pbb=paa+pa
vol_index=$fishvol_rebase({a},{pa},2020)
zz = vol_index * 2
q_a=convert(a,q,disc,ave)
""".strip().splitlines()

    # Generate cformulas.py and c_chain.py
    cformgen.generate_from_commands(fame_commands, formulas_out="cformulas.py", chain_out="c_chain.py")
    print("Generated cformulas.py and c_chain.py")


if __name__ == "__main__":
    run_demo()