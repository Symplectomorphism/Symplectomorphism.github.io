---
title: "Commands"
---

```text
wdm COMMAND [options]

COMMAND: daily, log (eat), plan, validate, fit-noise, filter, backtest,
         session-kcal, exercise-kcal, demo
```

Every command that reads data takes the observations file as an optional first argument and the subject file as `--config`. Both default to the repository's own copies, `data/raw/observations.csv` and `data/raw/subject.toml`, so the commands work from any directory with no paths at all. Give paths to work on another file (a copy, a test set, someone else's data). `wdm COMMAND --help` prints the options below.

| Command | What it does |
|---|---|
| [`daily`](#wdm-daily) | The morning routine: records readings, then validates, plans, logs the plan |
| [`log`](#wdm-log) (alias `eat`) | Adds a meal, exercise, or steps to a day; shows what is left of today's plan |
| [`plan`](#wdm-plan) | Runs both estimation phases and the controller; prints the plan |
| [`validate`](#wdm-validate) | Checks the observations file |
| [`fit-noise`](#wdm-fit-noise) | Phase 1: the scale's noise |
| [`filter`](#wdm-filter) | Phase 2: metabolic parameters and the current state |
| [`backtest`](#wdm-backtest) | Scores out-of-sample forecasts of every weigh-in |
| [`session-kcal`](#wdm-session-kcal) | Exercise energy of a named session in a routine CSV |
| [`exercise-kcal`](#wdm-exercise-kcal) | Exercise energy from a dated workout spreadsheet |
| [`demo`](#wdm-demo) | The whole pipeline on synthetic data with known truth |

## The everyday commands

### `wdm daily`

```text
wdm daily [CSV] [--config TOML] [--weight LB] [--waist IN]
        [--date YYYY-MM-DD] [--mode {deadline,sustainable}] [--out FILE]
        [--log-dir DIR] [--no-input]
```

Records this morning's readings and any missing food from yesterday (asking in a terminal), validates the file, runs [`plan`](#wdm-plan) with `--log-dir`, and prints today's prescription. It reminds you when a waist measurement is due. Walkthrough: [The daily routine](daily-routine.md#morning-wdm-daily).

| Option | Meaning | Default |
|---|---|---|
| `CSV` | Observations file | `data/raw/observations.csv` |
| `--config TOML` | Subject file | `data/raw/subject.toml` |
| `--weight LB` | This morning's weigh-in, lb (not asked for if given) | asked |
| `--waist IN` | This morning's waist, inches | asked when 7 days have passed |
| `--date YYYY-MM-DD` | The morning to record | today |
| `--mode` | `deadline` or `sustainable` | `deadline` if the subject file has a `[goal]`, else `sustainable` |
| `--out FILE` | The plan as CSV, overwritten every run | `code/plan.csv` |
| `--log-dir DIR` | Plans archive and log | `data/plans` (a `plans` directory beside the CSV's directory) |
| `--no-input` | Never prompt; use only the flags; warn about missing food | off (prompts in a terminal) |

Exit code 1 if a value conflicts with one already in the file, if validation fails, or if planning fails.

### `wdm log`

```text
wdm log [CSV] [--kcal KCAL] [--protein G] [--carb G] [--fat G]
        [--fiber G] [--alcohol G] [--sodium MG] [--exercise KCAL]
        [--steps N] [--date YYYY-MM-DD] [--log-dir DIR]
```

Adds to a day's row, then prints the day's energy and macronutrient totals against today's prescription with what remains. With no value flags it only prints. Walkthrough: [During the day](daily-routine.md#during-the-day-wdm-log).

| Option | Column | Behavior |
|---|---|---|
| `--kcal KCAL` | `kcal` | adds |
| `--protein G` | `protein_g` | adds |
| `--carb G` | `carb_g` | adds (total carbohydrate, including fiber) |
| `--fat G` | `fat_g` | adds |
| `--fiber G` | `fiber_g` | adds |
| `--alcohol G` | `alcohol_g` | adds |
| `--sodium MG` | `sodium_mg` | adds |
| `--exercise KCAL` | `exercise_kcal` | adds |
| `--steps N` | `steps` | **replaces** (a step count is a running total) |
| `--date YYYY-MM-DD` | | the day to add to; default today |
| `--log-dir DIR` | | where to read today's plan from; default `data/plans` beside the data directory |
| `CSV` | | observations file; default `data/raw/observations.csv` |

The full command, every option at once:

```bash
wdm log --kcal 650 --protein 45 --carb 60 --fat 22 \
        --fiber 9 --alcohol 0 --sodium 900 \
        --exercise 310 --steps 8400 --date 2026-10-28
```

A day without a row gets one in the latest episode, as long as it comes after the file's last date; add earlier rows by hand. `wdm eat` is an alias.

### `wdm plan`

```text
wdm plan [CSV] [--config TOML] [--mode {deadline,sustainable}]
        [--horizon DAYS] [--block DAYS] [--target-weight-lb LB]
        [--target-date YYYY-MM-DD] [--max-fat-fraction FRAC]
        [--carb-aversion W] [--min-carb-g G] [--target-carb-fraction FRAC]
        [--target-fat-fraction FRAC] [--out FILE] [--log-dir DIR]
        [--figures DIR] [--full]
```

Runs phase 1, phase 2, and the controller, then checks the plan against every safety constraint independently of the optimizer. Prints the first 14 days, writes the whole plan to `--out`, and prints today's prescription. Theory: [Control](theory/control.md); output: [Reading the output](reading-output.md#the-plan).

| Option | Meaning | Default |
|---|---|---|
| `CSV`, `--config TOML` | Observations and subject files | `data/raw/observations.csv`, `data/raw/subject.toml` |
| `--mode` | `deadline`: reach `[goal]` by its date. `sustainable`: lose at the safe rate, no date | `deadline` if there is a `[goal]`, else `sustainable` |
| `--horizon DAYS` | Planning window | deadline: the days to the goal date, at most 365 (at least one block); sustainable: 28 |
| `--block DAYS` | Days per decision block (intake is constant within a block) | 7 |
| `--target-weight-lb LB` | Override `[goal] target_weight_lb` for this run | the subject file |
| `--target-date YYYY-MM-DD` | Override `[goal] target_date` for this run | the subject file |
| `--max-fat-fraction FRAC` | Ceiling on fat's share of energy | 0.75 |
| `--carb-aversion W` | Weight on a preference for less carbohydrate; 3 to 10 is a real low-carb push | 0 (off) |
| `--min-carb-g G` | Hard floor on carbohydrate, g/day | 20 |
| `--target-carb-fraction FRAC` | Carbohydrate's share of energy to hold (0.05 to 0.10 is keto-style); wins over the fat target | `[preferences]`, else unset |
| `--target-fat-fraction FRAC` | Fat's share of energy when nothing else sets the split | `[preferences]`, else the midpoint of 0.15 and `--max-fat-fraction` |
| `--out FILE` | The plan as CSV, overwritten every run | `code/plan.csv` |
| `--log-dir DIR` | Also archive the plan as `DIR/<start>.csv` and append it to `DIR/log.csv` | off |
| `--figures DIR` | Write `plan.pdf` (the first 28 days) | off |
| `--full` | Print every day of the horizon, not 14 | off |

Both overrides must be given together if the subject file has no `[goal]` to fall back on. Planning is refused when the latest estimate is more than three days old (the current episode has no recent weigh-in), and when the goal date has passed.

The plan CSV has one row per day: `date, kcal, protein_g, carb_g, fat_g, protein_pct, carb_pct, fat_pct, predicted_weight_lb, lb_per_week, tissue_loss_pct_per_week`.

## Diagnostics

`plan` and `daily` run both estimation phases themselves. These commands show the intermediate results; none of them writes to your data.

### `wdm validate`

```text
wdm validate [CSV]
```

Checks the file against the [schema](data.md#validation): required columns, parseable dates, episode labels, duplicate days, calendar gaps within an episode, numeric values in plausible ranges, both units of one quantity in a row, and whether the macronutrients account for the energy total. Prints a one-line summary and any problems. Exit code 1 on errors; warnings alone exit 0.

### `wdm fit-noise`

```text
wdm fit-noise [CSV]
```

Phase 1 on weight alone: lists each episode's coverage, then the scale's observation noise, the trend's innovation noise, the filter's steady-state gain, and its effective window. Refuses when there are too few weigh-ins to identify three noise scales. Output: [Phase 1](reading-output.md#phase-1-the-scales-noise).

### `wdm filter`

```text
wdm filter [CSV] [--config TOML] [--figures DIR]
```

Phase 1, then phase 2: the posterior of every parameter (median, 90% interval, shift from the prior), then the current state (trend weight, fat and lean mass, adaptive thermogenesis, maintenance, weight-stable intake). Output: [Phase 2](reading-output.md#phase-2-parameters-and-state).

| Option | Meaning | Default |
|---|---|---|
| `--figures DIR` | Write `trajectory.pdf`, `composition.pdf`, `innovations.pdf`, `parameters.pdf`, `intake_log.pdf` for the latest episode | off |

### `wdm backtest`

```text
wdm backtest [CSV] [--config TOML] [--horizons DAYS [DAYS ...]]
        [--student-t DOF] [--out FILE]
```

Replays every episode day by day and scores the filter's forecasts of each weigh-in it had not yet seen, against persistence ("tomorrow equals the last reading"). Output: [The backtest](reading-output.md#the-backtest); method: [Estimation](theory/estimation.md).

| Option | Meaning | Default |
|---|---|---|
| `--horizons DAYS ...` | Forecast horizons in days | `1 7` |
| `--student-t DOF` | Use Student-t observation noise with DOF degrees of freedom in both phases, to compare with the Gaussian default by log score | off (Gaussian with a 4-sigma gate) |
| `--out FILE` | Write every scored forecast: `episode, origin, horizon, predicted_lb, sd_lb, observed_lb, persistence_lb, z` | off |

## Exercise energy

Both commands print an estimate; neither writes to `observations.csv`. Add the number with `wdm log --exercise KCAL`. Details: [Exercise energy](exercise.md).

### `wdm session-kcal`

```text
wdm session-kcal [ROUTINE] [--session NAME] [--weight-lb LB]
        [--list]
```

| Option | Meaning | Default |
|---|---|---|
| `ROUTINE` | Routine CSV, one row per exercise | `data/raw/routine.csv` |
| `--session NAME` | The session to estimate, such as `"Lower A"` | required unless `--list` |
| `--weight-lb LB` | Body weight for the MET formula | the last weigh-in in `data/raw/observations.csv` |
| `--list` | List the sessions in the routine | off |

### `wdm exercise-kcal`

```text
wdm exercise-kcal [XLSX] [--weight-lb LB]
        [--repeat SESSION --date YYYY-MM-DD [--append]]
```

| Option | Meaning | Default |
|---|---|---|
| `XLSX` | Workout tracker with `Upper`, `Lower`, and `Muscle Guide` sheets | `data/raw/workout_tracker.xlsx` |
| `--weight-lb LB` | Body weight for the MET formula | the last weigh-in in `data/raw/observations.csv` |
| `--repeat SESSION` | Estimate SESSION as if repeated on `--date`, reusing its latest logged instance | off: estimate every logged session |
| `--date YYYY-MM-DD` | The date for `--repeat` and `--append` | |
| `--append` | With `--repeat` and `--date`, also write the copied block into the spreadsheet as a dated entry (backs the file up as `.bak` first; refuses a duplicate) | off |

## `wdm demo`

```text
wdm demo [--days N] [--mode {deadline,sustainable}] [--figures DIR]
        [--max-fat-fraction FRAC] [--carb-aversion W] [--min-carb-g G]
        [--target-fat-fraction FRAC] [--target-carb-fraction FRAC]
```

Generates a synthetic subject with known parameters (two weight-only past episodes and a current one), runs phase 1, phase 2, and the controller, and reports parameter recovery against the truth. Its subject and limits are built in; it reads and writes nothing of yours.

| Option | Meaning | Default |
|---|---|---|
| `--days N` | Length of the synthetic current episode | 120 |
| `--mode` | Planning mode | `deadline` |
| `--figures DIR` | Write the diagnostic and plan figures | off |
| `--max-fat-fraction`, `--carb-aversion`, `--min-carb-g`, `--target-fat-fraction`, `--target-carb-fraction` | As for [`plan`](#wdm-plan) | as for `plan` |

## Exit codes

All commands exit 0 on success and 1 on a refusal or error (validation errors, a conflicting value, a stale estimate, a passed goal date, a missing required option). Messages go to standard error.
