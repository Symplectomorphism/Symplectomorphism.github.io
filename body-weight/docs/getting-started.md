---
title: "Getting started"
---

## Install

The prerequisite is [`uv`](https://docs.astral.sh/uv/). The package pins Python 3.13, because JAX wheels lag new CPython releases; `uv` fetches 3.13 for this project only.

```bash
cd weight-dynamics/code
uv sync --extra dev            # one time: creates .venv with the pinned dependencies
uv run python -m pytest        # 203 tests, a few minutes
uv run wdm demo                # the whole pipeline on synthetic data: confirms the install
```

`uv run wdm ...` works from `weight-dynamics/code`. To type plain `wdm` from any directory, put the virtual environment's entry point on your `PATH`, for example:

```bash
ln -s "$PWD/.venv/bin/wdm" ~/.local/bin/wdm
```

The code is in a private repository. These pages document it in full, and the [API reference](reference/index.qmd) is generated from its docstrings.

## The repository

```text
weight-dynamics/
├── data/
│   ├── schema.md          every column of observations.csv, and the rules
│   ├── templates/         blank and filled-in examples of every input file
│   ├── raw/               your data: observations.csv, subject.toml, routine.csv
│   └── plans/             the plans log, written by wdm daily
├── code/                  the wdm package, its tests, pyproject.toml, plan.csv
└── report/                the technical report (LaTeX)
```

Every command reads `data/raw/observations.csv` and `data/raw/subject.toml` unless you give it other paths, so `wdm daily` and `wdm log` need no arguments. Body-weight and food logs are personal health data: keep `data/raw/` and `data/plans/` out of any public copy.

## Set up your files

1. **The subject file.** Copy `data/templates/subject_template.toml` to `data/raw/subject.toml`. Fill in sex, age, and height; review the safety limits; add a goal if you have a date in mind. Every field is documented in [The subject file](configuration.md).
2. **The observations file.** Copy `data/templates/observations_template.csv` to `data/raw/observations.csv` (`observations_example.csv` is a filled-in example). It holds one row per day; [The observations file](data.md) defines every column.
3. **History, if you have it.** Enter past diets one row per day, each with its own `episode` label (`past-01`, `past-02`, ...). Weight alone is enough: past weigh-ins without a food log still teach phase 1 how noisy your scale is.
4. **The first row of the current diet.** Add it by hand: the date, a new episode label such as `current`, and that morning's weight. From then on `wdm daily` and `wdm log` add rows to the latest episode for you.
5. **Check it.**

   ```bash
   wdm validate
   ```

   ```text
   150 rows | 3 episodes | 122 weigh-ins | 66 intake records
     no problems found
   ```

   Fix anything it reports. It exits with code 1 on errors.

## Run it

```bash
wdm daily            # each morning: records the weigh-in, plans, prints today's prescription
wdm log --kcal 520 --protein 38 --carb 41 --fat 22     # before each meal
```

That is the whole routine; [The daily routine](daily-routine.md) walks through it. Do not expect the numbers to mean much in the first week: the filter reports its prior back until the data have had time to speak ([what to expect, and when](reading-output.md#what-to-expect-and-when)).

## Try it without your own data

`wdm demo` generates a synthetic subject with known metabolic parameters (two weight-only past diets and a 120-day current one), runs both estimation phases and the controller on it, and reports how many of the true parameters fall inside their 90% intervals:

```bash
wdm demo --days 60 --figures /tmp/wdm-demo
```

```text
Phase 2 -- posterior parameter estimates

  parameter              median                90% interval  shift
  rmr_intercept          1352.7                [1079, 1694]   +3.8%
  ...
  ground-truth recovery: 9/9 parameters inside their 90% interval
  smoothed trend RMSE vs hidden truth: 211 g
  (single-weigh-in noise was 492 g)
```

It reads and writes nothing of yours. The [walkthrough](walkthrough/index.qmd) takes the same pipeline apart one step at a time.
