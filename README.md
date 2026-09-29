# Crowd-to-Crash
**Endogenous liquidity and phase transitions in a heterogeneous mean-field market**
This repository develops a market model in which a representative trader reacts
to the distribution of a very large population of other traders. The long-run
research goal is to understand when individually sensible trading decisions can
produce crowding, vanishing liquidity, and abrupt sell-offs.
The project is deliberately built in layers:
1. a solvable linear-quadratic mean-field game;
2. a numerical Hamilton-Jacobi-Bellman/Fokker-Planck solver;
3. endogenous price impact and liquidity feedback;
4. heterogeneous fast and slow trader populations;
5. finite-agent validation and empirical calibration.
## What is implemented now
The first model describes traders by a one-dimensional inventory `q`. A trader
chooses a trading rate `u`, pays for trading quickly, dislikes holding inventory,
and may prefer to remain near the population's mean inventory. Small independent
inventory shocks spread the population distribution.
Because every cost is quadratic, the full mean-field PDE system reduces to four
ordinary differential equations. The dependency-free Python implementation in
`src/crowd_to_crash/baseline.py` solves their forward-backward fixed point.
The second implementation, `src/crowd_to_crash/grid_solver.py`, stores the full
value function and population density on an inventory-time grid. It solves the HJB
backward and the Fokker-Planck equation forward without assuming that either
function has a special shape. Read the typeset
[`output/pdf/02_full_grid_solver.pdf`](output/pdf/02_full_grid_solver.pdf) for the
derivation. The Markdown source is kept in `notes/02_math_grid_solver.md` for
version control.
The research extension in `src/crowd_to_crash/liquidity_model.py` adds aggregate
order flow, flow-dependent liquidity, and endogenous price drift. Its equilibrium
requires consistency in both the population mean and aggregate trading flow.
The derivation and first amplification experiment are documented in
`notes/03_endogenous_liquidity.md` and the typeset
[`output/pdf/03_endogenous_liquidity_feedback.pdf`](output/pdf/03_endogenous_liquidity_feedback.pdf).
The complete first research draft is maintained in `paper/draft.md`; the readable
PDF is [`output/pdf/crowd_to_crash_paper_draft.pdf`](output/pdf/crowd_to_crash_paper_draft.pdf).
The repository also contains a two-population solver, multi-start bifurcation
study, convergence study, and a calibration interface. See `CODE_OVERVIEW.md`
for the map.
For a simple but detailed walkthrough of the entire twelve-stage research plan,
read [`output/pdf/complete_research_math_handbook.pdf`](output/pdf/complete_research_math_handbook.pdf).
Its editable source is `handbook/research_math_handbook.md`.
Read the typeset [`output/pdf/01_mathematical_baseline.pdf`](output/pdf/01_mathematical_baseline.pdf)
before the code. It develops the mathematics from individual motion to the
population equation. The Markdown source in `notes/01_math_baseline.md` is kept
for version control, but some editors display its LaTeX notation as raw text.
## Run the baseline
The code requires Python 3.11 or newer. Install the project and NumPy with:
```bash
python3 -m pip install -e .
```
The analytical baseline itself has no third-party dependencies.
```bash
python3 -m crowd_to_crash.baseline --output results/baseline.csv
```
When running directly from a source checkout:
```bash
PYTHONPATH=src python3 -m crowd_to_crash.baseline --output results/baseline.csv
```
The grid solver requires NumPy:
```bash
PYTHONPATH=src python3 -m crowd_to_crash.grid_solver \
--output results/grid_baseline.csv
```
Run the tests with:
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
## Current research question
> How do interaction strength, trader heterogeneity, and liquidity-sensitive
> price impact change a market from smooth adjustment to synchronized trading?
The baseline is not yet a model of a crash. It is the transparent control case
needed to distinguish a genuine instability in later models from a numerical
artifact.
## Research standards
- Every model assumption must have an economic interpretation.
- Every numerical solver must be checked against a solvable special case.
- Probability mass, non-negativity, convergence, and grid sensitivity are tests,
not visual impressions.
- Empirical comparisons will target order flow, price impact, and liquidity
recovery—not unsupported claims of return prediction.
