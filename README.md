# Regulated PV Capacity for 3-Phase Distribution System via Robust-Scenario Program

Code accompanying the paper *"Regulated PV Capacity for 3-Phase Distribution System
via Robust-Scenario Program."* Given a radial 3-phase distribution feeder (via
OpenDSS), the code:

- builds a linearized 3-phase power-flow model (LinDistFlow, Section II.A),
- formulates PV hosting-capacity (PVHC) maximization as a robust-scenario LP
  (Section III, Proposition III.1),
- solves it with a scenario-count guarantee on the violation probability via
  the Campi-Garatti "wait-and-judge" approach (Algorithm 1, Theorem III.2),
  using a Clarkson-style iterative constraint-generation solver, and
- reproduces the paper's figures/tables: adoption studies, fairness ablation,
  battery/Volt-VAR/dual-sensitivity enhancements, and baseline comparisons.

This release replaces the real utility feeder and telemetry used in the paper
with a public, openly-licensed example feeder so the full pipeline can be run
end-to-end by anyone (see [Example data](#example-data) below). To protect our partner utility's data, 
no proprietary or customer data is included in this repository. Even though 
the example feeder is different, the code reproduces the paper's methodology and
  figure/table structure end-to-end.

## Setup

```bash
conda env create -f environment.yml
conda activate pvhc
```

All scripts below run entirely on open-source solvers (HiGHS for the plain LP
formulations, Clarabel for the log-fairness / exponential-cone objective,
both bundled with `cvxpy`) — no commercial solver license is required. If a
licensed Gurobi or MOSEK install is available, some scripts accept
`solver=cp.GUROBI` / `cp.MOSEK` for a speed-up (see comments in
`src/utils.py` and each script's `__main__`), but this is optional.

## Example data

The paper's case study uses a real utility's proprietary feeder model and
customer billing/telemetry data, which cannot be released. In its place,
`data/example_feeder/` ships:

- **Circuit model**: EPRI's "Ckt5" PV hosting-capacity test circuit, obtained
  unmodified from the official OpenDSS test-circuit distribution
  ([dss-extensions/electricdss-tst](https://github.com/dss-extensions/electricdss-tst),
  BSD-3-Clause — see `data/example_feeder/opendss/THIRD_PARTY_LICENSE.txt` for
  the full license and attribution). Ckt5 is a real-world-derived, unbalanced
  3-phase feeder purpose-built for PV hosting-capacity research, so it
  exercises the same model features as the paper's case study (radial 3-phase
  primary, single-phase laterals and service transformers, residential and
  commercial load classes). By default the code uses a ~1,200-bus subtree of
  Ckt5 (rooted at bus `63664`, see `src/constants.py`) rather than the full
  ~3,000-bus circuit, both to keep runtimes reasonable on a laptop and to
  roughly match the scale of the paper's own study.
- **Load timeseries**: Ckt5's own bundled hourly, annual (8,760-point)
  Residential / Commercial load shapes (`Loadshapes_ckt5.dss`), real data
  shipped with the EPRI circuit — not synthetic.
- **Solar timeseries**: `solar_capacity_factor.csv`, an hourly solar capacity
  factor series from NREL's public NSRDB for a public lat/lon grid cell (not
  tied to any specific customer or premise).
- **Feeder-head voltage**: held at a regulated 1.03 pu (a typical substation
  LTC setpoint; `v_root_setpoint_pu` in `src/constants.py`) with small seeded
  Gaussian noise kept within the regulator's +-1% band
  (`DataLoader.get_feeder_head_data`), since no historical feeder-head voltage
  telemetry is available for this substitute feeder — this is the one
  synthetic element in the example data and is documented as such in the code.
  At a 1.0 pu feeder head, the subtree's own load would take its lowest bus
  voltages below 0.95 pu.

Branch thermal ratings and voltage limits come directly from Ckt5's own
OpenDSS data (`Line.NormAmps` times bus base kV; a symmetric +-5% voltage
band) with no artificial safety margin applied. See
`PVHCProblemWithoutBattery._compute_network_limits()` in `src/analyzer.py` if
you want to adjust the adaptive safety-margin behavior
(`_relax_limits_adaptively()`, only engaged if the network is otherwise
infeasible even with zero PV installed -- a real, EPRI-documented condition on
a couple of branches of this subtree, not something PV additions cause).

Because this is a different feeder from the paper's case study, absolute
numbers (total hosting capacity, exact violation rates, etc.) will not match
the paper regardless -- the point of this release is to reproduce the
*methodology* and figure/table structure end-to-end.

### Per-customer capacity ceilings (zeta_max)

`zeta_max_u` / `zeta_max_c` are the per-site physical PV ceilings -- the box
constraints (12c)-(12d) that bound how much PV a site could host at all,
before the network is considered. The paper's case study sets them
**uniformly at 10 kW** for every site (Section IV.A); that is a deliberate
simplification, not part of the method, and the formulation admits any
per-site vector.

This release additionally ships a more realistic, site-varying way to set
them: `scripts/calculate_capacity_limit_and_store.py` sizes each customer's
ceiling from their own estimated annual consumption, scaled by
`capacity_multiplier` (default 3x). The base quotient is a *net-zero* sizing
-- the nameplate that would generate about as much energy as the site
consumes -- so `capacity_multiplier` models how far past net-metering parity
a program lets customers oversize (real NEM/interconnection rules vary). On
this feeder that yields ceilings of roughly 0.04-107 kW (mean ~31 kW) instead
of a flat 10 kW.

On this subtree the network binds under either ceiling (its upper voltage
limit). At a flat 10 kW, 124 of 553 customers receive the full ceiling and the
total, 4.10 MW, is 74% of the theoretical maximum; at 3x only 5 customers reach
their ceiling and the total, 4.24 MW, is 25% of it. (For reference, the
paper's feeder awards 43% of the theoretical maximum under its uniform 10 kW
ceiling, the bimodal distribution reported in Fig. 4.)

To reproduce the paper's exact ceiling convention instead, set
`USE_EXTRACTED_CAPACITY_LIMITS = False` in `scripts/run_pvhc_estimation.py`;
every script then falls back to the uniform
`DEFAULT_CANDIDATE_PV_CAPACITY_KW` (10 kW) from `src/constants.py`.

## Repository layout

```
src/                       Core library
  constants.py                Feeder registry (FEEDERS dict) and shared defaults
  data_loader.py               OpenDSS parsing (via altdss) + timeseries loading
  three_phase_lindistflow/    3-phase LinDistFlow model (Eq. 1-6, Appendix A), vendored (see below)
  simulator.py                Applies the LinDistFlow model to timeseries data
  analyzer.py                  RSP/LP formulation + Clarkson iterative solver
  utils.py                     Config dataclasses, result I/O, misc helpers
  visualization.py             Plotting primitives used by scripts/plot_figures.py

scripts/                   One script per experiment (see mapping table below)
tests/                     pytest suite (see Testing below)
data/example_feeder/       OpenDSS circuit + timeseries (see Example data above)
temp/                      All generated outputs (gitignored): caches, results, plots
```

`src/three_phase_lindistflow/` is an unmodified copy (plus its `LICENSE`) of the package in
[3-phase-lindistflow](https://github.com/caltech-netlab/3-phase-lindistflow) at commit `51232fb`.
To update it, replace the directory with the upstream `src/three_phase_lindistflow/`, then delete
`temp/cache/circuit_model/` and `temp/cache/limits/`: they hold the model matrices and the network
limits relaxed with them, which would otherwise keep coming from the previous version.

## Quickstart

Run from the repository root, with the `pvhc` environment active. Each step
writes to `temp/` and prints its own progress; nothing needs to be downloaded.

```bash
# 1. Set the per-site PV ceilings zeta_max (the Eq. (12c)-(12d) box
#    constraints) from each customer's estimated annual consumption, scaled by
#    capacity_multiplier (default 3x "net-zero"). This is an extension, not
#    the paper's own convention -- the paper sets zeta_max uniformly at 10 kW
#    for simplicity. Skipping this makes every downstream script fall back to
#    that flat 10 kW ceiling. See "Per-customer capacity ceilings" above.
python scripts/calculate_capacity_limit_and_store.py

# 2. PVHC estimation via the robust-scenario program (Algorithm 1, Table I).
#    Runs the epsilon/beta scenario-guarantee loop and stores results under
#    temp/optimization_results/ckt5/.
python scripts/run_pvhc_estimation.py

# 3. Time-series violation simulation using the optimized PV capacities
#    (Section IV.B, Figs. 2-3). Reads the results from step 2.
python scripts/run_pvhc_simulation.py

# 4. Generate plots. plot_figures.py is a function library (no CLI entrypoint);
#    call its functions directly for the figure(s) you want, e.g.:
python -c "
from scripts.plot_figures import plot_opt_results
plot_opt_results('ckt5', opt_subfolders=['iterative_eps0.01_beta0.05_utility_ratio_0pct'])
"
```

Every other script under `scripts/` (fairness sweep, method comparison,
battery/Volt-VAR/dual-sensitivity enhancements, unregulated adoption study) is
independently runnable — see the mapping table below for what each reproduces
and whether it depends on step 2's output.

## Scripts → paper mapping

| Script | Paper section / output | Depends on |
|---|---|---|
| `run_pvhc_estimation.py` | Section IV.C, Algorithm 1, Table I — RSP hosting-capacity estimation with epsilon/beta guarantee | — |
| `run_pvhc_simulation.py` | Section IV.B, Figs. 2-3 — timeseries violation simulation of the optimized solution | `run_pvhc_estimation.py` |
| `run_unregulated_adoption_study.py` | Section IV.B, Fig. 1 (adoption-rate ablation) and Fig. 2 (spatiotemporal violation heatmap) — *no* PVHC optimization, uncoordinated adoption only | — |
| `plot_figures.py` | Figs. 1-5 (function library; see its module docstring for the full mapping) | varies by function |
| `fairness_sweep.py` | Section IV.C.3, Eq. 16, Fig. 5 — gamma ablation of the fairness objective | — |
| `run_method_comparison.py` | Table IV — RSP vs. CVaR baseline vs. Robust (box) baseline | — (first run solves RSP fresh; set `SKIP_RSP=True` after that to reuse the saved result and iterate faster on the baselines alone) |
| `run_battery_enhancement.py` | Appendix C, Table II (battery rows) — open-loop utility battery storage extension | — |
| `print_summary_table.py` | Prints the Table II-style summary and a PV-vs-battery-capacity plot | `run_battery_enhancement.py` |
| `run_pvhc_voltvar_experiment.py` / `analyzer_voltvar.py` | Fixed-schedule Volt-VAR control extension on utility PV inverters | — |
| `dual_analysis.py` | Section IV.D.1, Appendix B, Fig. 6, Table V — dual-variable sensitivity ranking and upgrade re-optimization | `run_pvhc_estimation.py` |
| `calculate_capacity_limit_and_store.py` | *Extension, not in the paper* — sets the Eq. (12c)-(12d) per-site ceilings `zeta_max` from estimated annual consumption instead of the paper's uniform 10 kW (see [Per-customer capacity ceilings](#per-customer-capacity-ceilings-zeta_max)); run this first (see Quickstart) | — |

Each script's module docstring has more detail (equations, assumptions, scale
notes). Outputs land under `temp/{optimization_results,simulation_results,plots}/ckt5/...`;
every `optimization_results` and `simulation_results` folder gets its own
auto-generated `README.md` describing exactly what it contains.

## Testing

```bash
pytest tests/
```

On Windows you may see a `Windows fatal exception: code 0xe0465043` line during
test collection — this is a benign first-chance exception raised inside
`altdss` (the OpenDSS binding) during its own native DLL initialization, not a
real crash; the run completes normally afterward.

- `test_3pldf.py` — regression + smoke tests for the core LinDistFlow model
  (`src/three_phase_lindistflow/`) against fixed reference circuits.
- `test_data_loader.py` — end-to-end test of OpenDSS parsing, circuit-model
  construction, and simulation on the example feeder.
- `test_analyzer.py` — direct coverage of the RSP LP solver
  (`src/analyzer.py`), including the log-fairness (Clarabel) objective.

## Data formats

Use the loader functions in `src/utils.py` rather than reading the files
directly — the exact array layouts are documented in each function's
docstring (`store_optimization_results`/`load_optimization_results`,
`save_simulation_results`/`load_simulation_results`,
`save_capacity_limits`/`load_capacity_limits`), and each output folder also
gets its own auto-generated `README.md` (see `store_optimization_results`).

Phase connections for loads and PV sites are represented as `(bus, phases)`
dicts, where `phases` is a tuple of ints in `{0, 1, 2, 3}`:
- `(1, 0)` / `(2, 0)` / `(3, 0)` — single-phase, wye (phase-to-neutral) on A/B/C
- `(1, 2)`, `(2, 3)`, `(3, 1)` — phase-to-phase (delta) connections; order
  matters (`(1, 2)` and `(2, 1)` denote opposite current directions)

Bus counts: `N` denotes the number of non-slack (non-root) buses; `N+1`
includes the slack/root bus.

## Known limitations

- The log-fairness (Eq. 16) exponential-cone objective can be numerically hard
  for open-source conic solvers. `clarkson_solver` accepts a list of solvers
  (optionally with options) and tries them in order on each subproblem
  (`src/solver_utils.py`), falling back on any failure: an exception, a Rust
  panic inside Clarabel, or a status without a usable solution. Every failed
  attempt is printed, and if all fail the run stops with an error.
  `run_pvhc_estimation.py` and `fairness_sweep.py` use `FAIRNESS_SOLVERS`:
  Clarabel, then Clarabel with a shorter step and without equilibration, then
  ECOS, then MOSEK if it is installed. Clarabel is pinned to its QDLDL linear
  solver: from Clarabel 0.11 the default picks the faer solver for these dense
  subproblems, which can fail or panic. (GUROBI is not in the list: CVXPY's
  GUROBI interface does not take exponential cones.) `environment.yml` pins
  `clarabel<0.11`: on some of these subproblems Clarabel 0.11.1 stops with
  InsufficientProgress whatever its settings, where 0.10 solves them.
- A run that cannot be certified stops with an error rather than reporting a
  result: `run_iterative_pvhc_analysis` raises if the Clarkson solver does not
  converge (default `clarkson_max_iterations=30`), and `fairness_sweep.py` leaves
  such a gamma out. Support scenarios are counted from duals scaled by their
  constraint rows, relative to the largest one (`support_from_duals` in
  `src/analyzer.py`): interior-point solvers leave small non-zero duals on every
  inactive constraint, which an absolute threshold used to count at random.
- Appendix B's dual-sensitivity ranking in `dual_analysis.py` is derived for
  *voltage*-driven resistance sensitivity (dPVHC/d(resistance scale)), the
  mechanism that binds on this subtree. On a feeder where thermal/branch-flow
  binds instead, that quantity is identically zero everywhere, so
  `compute_thermal_rating_sensitivity` computes the analogous quantity for
  thermal rating (dPVHC/d(S_rating), a direct sum of each branch's own LP
  shadow prices -- no impedance-sensitivity matrices needed, since a branch's
  own rating is the direct RHS of its own flow constraints). The `__main__`
  block auto-detects which mechanism actually binds and upgrades branches
  accordingly (reconductoring for lower resistance, or a higher thermal
  rating).
