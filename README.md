# Fame2PyGen

Convert FAME-style formula scripts (`.inp`) into executable, modular Python code using [Polars](https://pola.rs/) (and a companion economic toolkit library `polars_econ`).  
The tool parses FAME-like expressions and generates:

1. A formulas module (e.g. `cformulas.py`) containing one pure function per series (as **Polars expression builders**).
2. A pipeline script (e.g. `c_pipeline.py`) that composes those functions against an input DataFrame.

![Logo](logos/your_logo_primary.png)

---

## Key Features

- Reads FAME commands from `.inp` files (or stdin).
- Supports core constructs you've prototyped:
  - Simple assignments
  - `$mchain("...","YEAR")`
  - `convert(series,to_freq,technique,observed)`
  - `$fishvol_rebase({vols},{prices},YEAR)` (with optional trailing arithmetic)
  - `freq <m|q|a>` state changes
  - Braced single-item alias normalization (`x = {y}` -> `x = y`)
- Generates deterministic, importable formula functions returning `pl.Expr`.
- Optional generation of a mock `polars_econ.py` to dry-run without the real library.
- CLI customization for output filenames and module naming.
- Clean architecture for future "plugin" parsers.

---

## Quick Start

```bash
pip install --upgrade pip
pip install -e .
```

Generate code from an example:

```bash
fame2pygen examples/sample_basic.inp \
  --formulas-out cformulas.py \
  --pipeline-out c_chain.py \
  --write-mock-econ
```

Run the generated pipeline:

```bash
python c_chain.py
```

---

## CLI Usage

```
fame2pygen INPUT_INP \
  [--formulas-out FORMULAS_PY] \
  [--pipeline-out PIPELINE_PY] \
  [--formulas-module-name MODULE_IMPORT_NAME] \
  [--write-mock-econ] \
  [--version]
```

Examples:

```bash
# Basic
fame2pygen examples/sample_basic.inp

# Custom output names
fame2pygen examples/sample_basic.inp \
  --formulas-out my_formulas.py \
  --pipeline-out my_pipeline.py \
  --formulas-module-name my_formulas

# From stdin
cat examples/sample_basic.inp | fame2pygen - --pipeline-out alt_pipeline.py
```

---

## Generated Files (Overview)

### Formulas Module (e.g. `cformulas.py`)

- Imports Polars.
- One function per calculable series in UPPERCASE.
- Functions accept dependent series as `pl.Expr` arguments.
- Returns a `pl.Expr` with `.alias(...)` naming.

### Pipeline Script (e.g. `c_chain.py`)

- Imports the formulas module.
- Builds a sample `pdf` DataFrame (replace with real data ingestion).
- Computes series level-by-level (topological order).
- Executes special operations (`CHAIN`, `CONVERT`, `FISHVOL`) when referenced.
- Produces a normalized `final_result` in long format:
  ```
  DATE | TIME_SERIES_NAME | VALUE | SOURCE_SCRIPT_NAME
  ```

---

## Supported FAME Patterns (Current)

| Pattern | Example | Notes |
|---------|---------|-------|
| Frequency state | `freq m` | Influences `convert` interpretation |
| Simple assignment | `aa=a$/a` | Non-numeric refs become dependencies |
| Chain index | `abc=$mchain("a$ + a + bb","2025")` | Auto-expands price/quantity pairs: each quantity var requires a matching `p<var>` |
| Convert | `zed=convert(a$,q,disc,ave)` | Mapped to Polars dynamic group-by / custom econ lib |
| Fish volume | `xyz=$fishvol_rebase({a},{pa},2017)*12` | Supports trailing arithmetic |
| Single-item braces | `x={y}` | Normalized to `x=y` |
| Multi-item list alias | `d={aa,bb}` | Tracked (future expansion – no direct function yet) |

---

## Extending the Parser

Add a new specialized parser:

1. Create a function in `parser.py` returning `ParsedCommand`.
2. Insert it into `PARSERS` before `parse_simple_command`.
3. Update `generators.py` if code generation logic is needed.

For substantial categories, consider a plugin registry pattern later:

```python
PARSERS.insert(0, parse_new_thing)
```

---

## Roadmap Ideas

- Robust loop unrolling (ported from notebook prototype).
- Enhanced error reporting with line numbers.
- Configurable dependency graph visualization.
- Real `polars_econ` integration examples.
- Unit dimension / metadata propagation.
- Caching + incremental regeneration.
- Pluggable template system (Jinja2 already included; templates folder stub provided).

---

## Development

```bash
git clone https://github.com/botlhale/Fame2PyGen.git
cd Fame2PyGen
pip install -e ".[dev]"
pytest -q
```

(You can add a `[project.optional-dependencies]` section for `dev` extras if desired.)

---

## Tests

See `tests/` for parser & generation smoke tests. Expand with:
- Edge cases (numeric-only RHS)
- Chain with negative terms
- Mismatched fishvol pairs (expect graceful rejection)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome for:
- New FAME constructs
- Performance improvements
- Documentation enhancements

---

## CHANGELOG

See [CHANGELOG.md](CHANGELOG.md). Versioning follows Semantic Versioning.

---

## License

MIT (see `LICENSE`).

---

## Logos

Existing logos reused from the original repository in `logos/`.  
Example inline usage:

```markdown
![Alt Logo](logos/your_logo_symbol.png)
```

---

## FAQ

**Q:** Do I need real data first?  
**A:** No; start with the mock pipeline and then swap in real `pl.read_csv(...)` or other ingestion.

**Q:** What if a referenced price series (`p<var>`) is missing for `$mchain`?  
**A:** Currently you must ensure both exist; future versions may attempt auto-derivation or raise a clearer error.

**Q:** How do I add a custom output naming scheme?  
**A:** Use `--formulas-out` / `--pipeline-out` and optionally `--formulas-module-name`.

---

Happy transforming!