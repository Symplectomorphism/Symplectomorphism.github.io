---
title: "Development"
---

## Package layout

```text
code/src/wdm/
├── constants.py        physiological and unit constants, each with its source
├── config.py           subject.toml: Subject, Goal, Safety, Preferences
├── schema.py           the observations columns and their validation
├── io.py               observations.csv -> Episode arrays on a complete daily calendar
├── daily.py            file bookkeeping for wdm daily and wdm log; the plans log
├── models/
│   ├── types.py        State, Params, DailyInput (JAX pytrees); the timing convention
│   ├── energetics.py   resting rate, thermic effect of food, adaptive thermogenesis
│   ├── partition.py    Forbes' law: how a change of tissue splits into fat and lean
│   ├── glycogen.py     carbohydrate -> glycogen -> bound water; sodium water
│   ├── dynamics.py     the continuous-time model (Diffrax) and the discrete step map
│   └── observation.py  the scale and the waist tape
├── filters/
│   ├── common.py       shared result types, missing observations, the innovation gate
│   ├── linear.py       Kalman filter, RTS smoother, log-likelihood
│   ├── ukf.py          unscented filter and smoother, with an auxiliary nonlinear update
│   └── ekf.py          extended filter with autodiff Jacobians
├── estimation/
│   ├── noise_id.py     phase 1: maximum likelihood on a local linear trend
│   ├── reparam.py      constrained <-> unconstrained parameter coordinates
│   ├── joint.py        phase 2: the augmented-state filter over every episode
│   └── backtest.py     out-of-sample forecast scoring
├── control/
│   ├── objectives.py   cost terms for deadline and sustainable modes
│   └── mpc.py          the SLSQP receding-horizon solve, fallback, check, feasibility
├── routine.py          exercise energy from a routine CSV
├── exercise.py         exercise energy from a dated workout spreadsheet
├── tracker_write.py    appending a repeated session to that spreadsheet
├── synthetic.py        synthetic episodes from known parameters
├── viz.py              the figures
└── cli.py              the wdm command
```

Dependencies run one way, with no cycles: `constants` at the bottom; `models` on it; `filters` model-agnostic; `estimation` on both; `control` on `models`; `cli` on top. The exercise modules never load JAX. Double precision is enabled globally when `wdm` is imported.

The [API reference](reference/index.qmd) documents every module from its docstrings.

## Tests

```bash
cd weight-dynamics/code
uv run python -m pytest              # 194 tests, a few minutes on a CPU
uv run python -m pytest tests/test_daily.py -q
```

| File | Tests | Focus |
|---|---|---|
| `test_models.py` | 21 | Closed-form checks, energy conservation, integrator agreement, the waist prior |
| `test_filters.py` | 16 | Comparison with exact batch posteriors, sigma-point properties, the gate, Student-t |
| `test_estimation.py` | 21 | Parameter recovery and calibration, the tape, weight-only episodes, timing, the habitual reference, the backtest |
| `test_control.py` | 37 | Constraints, SLSQP without the fallback, the per-day rate limit, the protein check's basis, infeasibility, objective behavior |
| `test_io.py` | 29 | Schema validation, gaps, units, waist columns, habitual intake |
| `test_cli.py` | 6 | Staleness arithmetic, the plan's start, the default mode |
| `test_daily.py` | 10 | Recording readings, food, meals and steps; the plans log; prompting; `log` and its `eat` alias |
| `test_exercise.py` | 25 | MET arithmetic, tempo and hold parsing, classification |
| `test_tracker_write.py` | 17 | Spreadsheet mutation, merged ranges, duplicate guards |
| `test_routine.py` | 12 | The routine CSV, cross-checked against the set-level estimator |

Each test checks against something **independent**: a closed-form result, a separate integrator, a conservation law the code never uses, or a dense batch computation that shares no code with the recursion under test. Three of them each caught a real bug during development:

- `test_energy_balance_closes` rebuilds "intake minus expenditure equals the energy stored in tissue and glycogen" from the returned derivatives, catching sign errors and dropped terms in the glycogen coupling;
- `test_rk4_matches_adaptive_solver` guards the substep count, so a discretization error cannot pass for a physiological finding;
- `test_kalman_matches_batch_gaussian_solution` conditions the joint Gaussian over the whole sequence by dense linear algebra, which would catch a transposed gain or a mishandled missing observation.

## Figures

No dual axes: quantities on different scales get stacked panels, since a dual axis lets the author choose where the scales cross. Every series is labeled. Every state estimate carries its credible band, because a mean alone misstates how much is known, fat mass above all.

## The technical report

The full derivation lives in a LaTeX report in `weight-dynamics/report/`; these pages condense it. To rebuild it with fresh figures:

```bash
cd weight-dynamics/code
uv run wdm demo --days 120 --mode deadline --figures ../report/figures     # synthetic figures
uv run wdm filter --figures ../report/figures/real                         # real-data figures
uv run wdm plan --out /tmp/plan.csv --figures ../report/figures/real
uv run wdm backtest
cd ../report && make            # lualatex, biber, lualatex twice
```

## The presentation and these pages

The [deck](../weight-dynamics.html) and the [landing page](../index.qmd) never type a data-dependent number: `body-weight/assets/compute.py` runs the model on the current record and writes `body-weight/numbers.yml`, which the pages read with `{{{< meta bw.KEY >}}}`; `make_figures.py` then draws every figure from its JSON output. `body-weight/assets/README.md` has the commands.

The API reference is generated from the package's docstrings by quartodoc; regenerate it after the code changes:

```bash
body-weight/docs/build_reference.sh
```
