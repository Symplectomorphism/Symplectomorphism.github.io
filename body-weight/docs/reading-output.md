---
title: "Reading the output"
---

The examples on this page come from a synthetic subject (70 days of a current diet plus two weight-only past diets), so the numbers are illustrative, not anyone's.

## The plan

```text
Plan (deadline mode, 155 day horizon)

  date           kcal   pro  carb   fat  pro%  carb%  fat%   pred lb   lb/wk loss%/wk
  2026-10-28     2164   149    64   146    27     12    61     204.5    2.02     0.63
  2026-10-29     2164   149    64   146    27     12    61     204.2    1.89     0.64
  ...
  ... 141 more days

  plan written to plan.csv

  all safety constraints satisfied

  Today: 2164 kcal | 149 g protein (27%) | 64 g carb (12%) | 146 g fat (61%)
  Only today's line is a prescription. The rest is what the model currently
  expects, and it will change as more weigh-ins arrive.
```

| Column | Meaning |
|---|---|
| `kcal`, `pro`, `carb`, `fat` | Prescribed energy and grams of protein, carbohydrate, fat for that day. Constant within each decision block (7 days by default). |
| `pro%`, `carb%`, `fat%` | Each macronutrient's share of energy |
| `pred lb` | Predicted morning scale weight, water included |
| `lb/wk` | Predicted rate of **scale-weight** loss, water included |
| `loss%/wk` | Predicted rate of **tissue** loss, as a percentage per week: the quantity to compare with `max_loss_frac_per_week` |

**The `Today:` line is the prescription.** Every other row is what the model currently expects, and tomorrow's run replaces it. That is the receding-horizon idea: plan weeks ahead, eat one day, re-estimate, replan.

The plan starts this morning while today's row is dated today, however much food you have already logged, so a rerun at any time of day gives the same prescription for the same data. If the last row is a finished day with its food logged, the plan starts the next morning.

`lb/wk` and `loss%/wk` disagree, especially early: scale weight includes water that moves fast, while the rate limit governs tissue. In the first days of a diet the scale can drop several pounds a week with tissue loss well inside the limit.

**What can appear below the plan:**

| Message | Meaning |
|---|---|
| `all safety constraints satisfied` | An independent check of the finished plan (not the optimizer's own word) found every limit in `[safety]` met. |
| `CONSTRAINT VIOLATIONS (do not follow this plan)` | That check failed; the list says which limit on which day. Do not eat this plan. Loosen nothing in a hurry: rerun, and if it persists, report it. |
| `NOTE: target of 170 lb by 2026-12-01 is not reachable within your 1%/week limit; the earliest achievable date at that maximum rate is 2027-03-07` | The deadline is too aggressive for your rate limit. The plan still respects the limit (it goes as fast as allowed); move the date in `subject.toml`, or use sustainable mode. |
| `Refusing to plan: the most recent state estimate is from ... days ago` | The current episode's last weigh-in is more than three days old. Weigh in and log it. |
| `goal date ... has already passed` | Set a new `target_date`. |

In **deadline mode** the controller aims at the goal by its date, without exceeding the limits. In **sustainable mode** there is no date: it loses at a safe steady rate over a 28-day window and reports the timeline that results. Without a real date in mind, sustainable mode is the more honest choice. In **maintain mode** it holds the weight in `[maintain]`: expect intake near the weight-stable level, a flat `pred lb` column, and no `NOTE`.

`wdm plots` writes all of this as figures, along with the history of the parameter estimates and of the maintenance estimate ([Commands](commands.md#wdm-plots)).

## Phase 1: the scale's noise

`wdm fit-noise` (and the top of `wdm filter`):

```text
Phase 1 -- noise identification
  episodes used         3
  weigh-ins used        114
  log-likelihood        -106.32

  observation sd        0.492 kg (1.09 lb)   x/1.09
  level innovation sd   0.09789 kg/day   x/2.03
  slope innovation sd   0.00182 kg/day^2 unidentified

  steady-state gain     0.195
  effective window      5.1 days
```

- **Observation sd** is how far a single morning reading scatters around your true weight: mostly fluid, not the instrument. `x/1.09` is the multiplicative uncertainty (a factor of 1.09 either way). Well above 0.6 kg usually means inconsistent weighing; fix that before anything else.
- **Level and slope innovation sd** are how much the underlying trend and its rate wander from day to day. `unidentified` means the data cannot pin that scale down; it is reported, not hidden, and does no harm.
- **Steady-state gain** is how much of each new reading's surprise moves the trend. **Effective window** is its reciprocal: roughly how many days of readings the trend averages over. Near 10 days, one odd weigh-in moves the trend by a tenth of its error; a short window means the trend follows readings closely. A very long window means the trend looks nearly straight over the record, so readings are averaged heavily.

Phase 1 uses weight alone, from every episode, food log or not. Theory: [Estimation](theory/estimation.md).

## Phase 2: parameters and state

`wdm filter`:

```text
Phase 2 -- posterior parameter estimates

  parameter                    median                90% interval  shift
  rmr_intercept                1338.6                [1071, 1671]   +2.7%
  rmr_slope                     15.33              [12.13, 19.35]   +5.7%
  at_gain                     0.13141           [0.04568, 0.3245]  -12.4%
  at_tau                       12.927                [5.66, 29.6]   -7.7%
  gly_gain                  0.0027997        [0.001351, 0.005868]   -6.7%
  gly_tau                      1.5735             [0.8391, 2.952]   +4.9%
  na_gain                  1.1672e-05      [3.629e-06, 3.727e-05]   -2.7%
  na_tau                      0.99015             [0.5074, 1.917]   -1.0%
  partition_kappa            0.047795          [-0.03371, 0.1321]   -4.4%
  waist_offset_cm              68.425              [59.21, 77.56]   +1.8%
  waist_slope_cm_per_kg        1.2473             [0.9729, 1.604]   +4.3%

Estimate as of 2026-10-28 (episode 'current')
  trend weight          204.9 lb
  fat mass              28.2 kg
  lean mass             65.1 kg
  adaptive thermogenesis -118 kcal/day
  maintenance           2765 kcal/day
  weight-stable intake  3072 kcal/day
```

**Read the `shift` column first.** It is how far each parameter's median has moved from its prior. A shift near zero means the data have not spoken about that parameter yet, and its interval is the prior's. The parameters are defined in [the model](theory/model.md#parameters); the two `waist_` rows appear only when the file has waist readings.

**Then read the state.** *Maintenance* is the energy you would spend today at your trend weight, and *weight-stable intake* the logged intake that would hold your weight steady; these combinations are much better determined than the coefficients behind them. *Fat mass* is weakly identified from a scale: read its trend, not its level, and expect its interval to span several kilograms. *Adaptive thermogenesis* is the expenditure your body has shed in response to the deficit so far.

`--figures DIR` writes the trajectory (readings, smoothed trend, 95% band), body composition (fat and lean with bands), the innovations, the parameter posteriors, and the intake log. In the **innovations**, the standardized surprises of each reading, look for a standard deviation near 1 and no structure; an sd far from 1, or a weekly pattern, means something the model does not capture (see [Troubleshooting](#troubleshooting)).

## The backtest

`wdm backtest` replays each episode day by day and scores the forecasts of weigh-ins the filter had not yet seen:

```text
Backtest -- forecasts of the scale reading (Gaussian noise)

  horizon     n   bias    RMSE  persist   skill   z sd   50% cov   90% cov   log score
    1 d      58  -0.19    1.10     1.47    +25%   0.85      43%       98%      -0.747
    7 d      52  -0.40    1.15     2.12    +46%   0.74      54%       96%      -0.832
```

- **`skill`** first: the RMSE improvement over persistence ("the next reading equals the last one"). A model that cannot beat persistence at a week is not earning its complexity.
- **Calibration** next: `z sd` near 1, and coverage near 50% and 90%, mean the stated intervals are honest. Below 1 and above 90% means the intervals are too wide (cautious); the reverse means overconfident.
- **`log score`** is the mean log predictive density per reading; higher is better. It is the fair way to compare noise models (`--student-t 4` against the Gaussian default), because it rewards honest intervals as well as accurate means.

Two caveats: the phase-1 noise scales are fitted on the whole record, so coverage is slightly optimistic; and neighboring 7-day forecasts overlap, so the effective sample is several times smaller than `n`.

## The plans log

`wdm daily` (and `wdm plan --log-dir DIR`) archive every plan as `data/plans/<start>.csv`, in the same format as `plan.csv`, and append one row to `data/plans/log.csv`:

```text
run_at,start,mode,target_weight_lb,target_date,horizon_days,kcal,protein_g,carb_g,fat_g,trend_lb,maintenance_kcal,constraints_ok,wdm_commit,command
2026-10-28T18:04:25,2026-10-28,deadline,185,2027-04-01,155,2164,149,64,146,204.9,2765,yes,bb15534,wdm daily
```

| Field | Meaning |
|---|---|
| `run_at` | When the plan was made |
| `start` | The plan's first day; the prescription below is for this day |
| `mode`, `target_weight_lb`, `target_date`, `horizon_days` | What it aimed at (the target is blank in sustainable mode) |
| `kcal`, `protein_g`, `carb_g`, `fat_g` | The day's prescription |
| `trend_lb`, `maintenance_kcal` | The estimate it was based on |
| `constraints_ok` | Whether the independent check passed |
| `wdm_commit` | The package's git commit, with `+dirty` if it had uncommitted changes |
| `command` | The exact command line |

A replan on the same day overwrites that day's archived plan but adds a log row, so every prescription ever issued is kept. `plan.csv` holds only the latest; the log can reproduce a past decision. `wdm log` reads today's prescription from it.

## What to expect, and when

| Elapsed | What becomes available |
|---|---|
| Week 1 | Nothing reliable. The trend exists but is dominated by its prior. |
| Weeks 2 to 3 | A usable trend and rate of loss. This alone is worth the trouble. |
| Weeks 4 to 6 | Maintenance separates from its prior. The prescriptions become worth acting on rather than merely looking at. |
| Months 2 to 3 | The glycogen parameters are well identified; the water/tissue split becomes informative about individual days. |
| Month 4 and on | The adaptation parameters begin to move, **if** intake varied enough to produce a transient. If not, they stay at their priors. |

::: {.callout-tip}
## The most common mistake is looking too early
In week 1 the filter reports its prior back, slightly perturbed, in output that looks as authoritative as it ever will. Shifts near zero mean the data have not spoken yet.
:::

After four months of consistent data, expect a trend much better than the raw scale, a maintenance figure good to perhaps 5 to 8% with an honest interval, a usable tissue/water split, and prescriptions grounded in your own data. Do not expect a reliable body-composition measurement, an adaptation estimate unless you varied intake, or any insight into whether you will follow the plan, which is, empirically, what decides the outcome.

## Improving your estimates

Roughly in order of value per unit of effort:

1. **Weigh under identical conditions**, every morning. The noise is fluid variation, not instrument error, so this habit is worth more than a better scale or any refinement of the model.
2. **Weigh daily.** Gaps cost precision.
3. **Log macronutrients, not just calories.** This unlocks the glycogen compartment and the thermic effect of food.
4. **Log with constant rigor**, not increasing rigor: a drifting log bias looks like metabolic adaptation. Note any change of method in `notes`.
5. **Track sodium** if your tracker reports it. Before reading anything into an overnight gain, think about the previous evening.
6. **Estimate what you cannot measure.** At a restaurant or a friend's, log a best guess at calories, all three macronutrients and sodium; to the model a blank means a specific assumption, usually a worse one than your guess ([When you don't know exactly](data.md#estimating)).
7. **Set your habitual intake** in `subject.toml` if you know it.
8. **Measure your waist weekly.** Expect a fat-mass interval about a tenth narrower.
9. **Log `exercise_kcal`** on training days ([Exercise energy](exercise.md)).
10. **Vary intake deliberately.** A diet break or refeed week creates the transient that makes adaptation identifiable: for once the experimentally optimal action is the pleasant one.

**Outliers.** Leave a mistyped weight in the file for the outlier gate to downweight; deleting readings by hand corrupts the noise estimate. A real excursion (illness, a long flight, a salty meal) is downweighted too. If one dominates a week, note it and read that period cautiously, or start a new episode.

## Troubleshooting

| Symptom | Likely cause and response |
|---|---|
| Parameters barely move off their priors | Not enough data yet, or intake too constant to be informative. Expected before weeks 4 to 6. |
| Innovation sd well above 1 | Noise understated. Check for inconsistent weighing or a stretch of erratic logging. |
| Weekly pattern in the innovations | Weekend eating the log misses. Log weekends more carefully rather than changing the model. |
| Maintenance implausibly low | A drifting log bias, or real adaptation. The data cannot tell these apart; treat the estimate cautiously. |
| The plan sits on the rate limit every day | The deadline is aggressive for your limits. Look for the `NOTE`, or use sustainable mode. |
| Fat-mass interval enormous | Expected. Read the trend, not the level. |
| `wdm fit-noise` refuses: too few weigh-ins | Correct. Three noise scales cannot be identified from a handful of points. |
| Warning or refusal: the state is days old | The latest episode's last weigh-in is more than three days old. Weigh in (`wdm daily`). |
| `wdm daily` refuses to write a value | The cell already holds a different value. If the old one is wrong, `wdm log --date ... --set` replaces it. |
| `wdm log` says no plan is logged for today | Run `wdm daily` first. |
| `wdm daily` asks for yesterday's food when you did eat | Yesterday's row has neither `kcal` nor a full macro split. Answer, or fill it in by hand. A day of zeros is read as a fast, not as unknown. |
