# Fame2PyGen

**Fame2PyGen** is a toolset for automatically converting FAME-style formula scripts into executable, modular Python code using [Polars](https://pola.rs/) for DataFrame operations. It supports generic data transformation functions, auto-generated pipelines, and mock FAME-style backend via the `ple.py` module.

## Project Structure

```
fame2pygen/
├── README.md
├── ple.py              # Mock backend for FAME-style calculations
├── formulagen.py       # Parsers and helpers for FAME script commands
├── write_formulagen.py # Main entry point: parses FAME scripts, generates formulas.py & convpy4rmfame.py
├── formulas.py         # Auto-generated: generic, reusable calculation functions
├── convpy4rmfame.py    # Auto-generated: pipeline that executes parsed FAME formulas using formulas.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fame2pygen.git
cd fame2pygen
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed.  
Install [Polars](https://pola.rs/) using pip:

```bash
pip install polars
```

### 3. Generate formulas and pipeline

Edit your FAME script inside `write_formulagen.py` (the `fame_script` variable).

Run:

```bash
python write_formulagen.py
```

This will generate:
- `formulas.py`: All generic calculation functions (CONVERT, FISHVOL, CHAIN, SUM_HORIZONTAL, etc)
- `convpy4rmfame.py`: The pipeline that executes your formulas using `formulas.py`

### 4. Run the generated pipeline

```bash
python convpy4rmfame.py
```

You should see output like:

```
Computation finished
```

## Customizing

- To use your own FAME script, edit the `fame_script` variable in `write_formulagen.py`.
- Regenerate the modules by rerunning `python write_formulagen.py`.
- You can further modify `formulas.py` and `convpy4rmfame.py` as needed for your project.

## How It Works

- **ple.py:** Mock backend for FAME-like calculations (can be replaced with real implementations).
- **formulagen.py:** Parses and interprets FAME scripts.
- **write_formulagen.py:** Generates generic `formulas.py` and a pipeline `convpy4rmfame.py` from your FAME script.
- **formulas.py (auto-generated):** Holds generic, reusable calculation functions.
- **convpy4rmfame.py (auto-generated):** Implements your formula pipeline using the generic functions.

## Example FAME Script

```
series gdp_q, cpi_q, vol_index_1
vols_g1 = {v_a, v_b}
prices_g1 = {p_a, p_b}
all_vols = {v_a, v_b}
list_of_vol_aliases = {vols_g1}
freq q
loop all_vols as VOL:
    gdp_q = convert(VOL, q, ave, end)
end loop
loop list_of_vol_aliases as ALIAS:
    gdp_real = fishvol_rebase(ALIAS, prices_g1, 2020)
end loop
vol_index_1 = v_a + v_b
gdp_chained = $mchain("gdp_q - cpi_q", "2022")
final_output = gdp_chained - vol_index_1
```

## License

MIT
Copyright (c) 13668754 Canada Inc
