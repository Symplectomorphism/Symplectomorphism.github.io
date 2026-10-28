---
title: "Exercise energy"
---

`exercise_kcal` is optional. Left blank, training is absorbed into the estimated maintenance energy, which is harmless if your training is steady. If you lift and want expenditure to follow the sessions that happened, `wdm` estimates a session's energy in one of two ways, for two ways of keeping track of a program:

| You keep | Use | Reads |
|---|---|---|
| A routine held roughly constant, referred to by name ("did Lower A today") | `wdm session-kcal` (**recommended**) | `data/raw/routine.csv`, one row per exercise, no dates |
| A full dated log, set by set, with the weights of each day | `wdm exercise-kcal` | `data/raw/workout_tracker.xlsx` |

Neither command writes to `observations.csv`. Add the number to the day with `wdm log`:

```bash
wdm session-kcal --session "Lower A"       # prints the estimate
wdm log --exercise 194                     # adds it to today's exercise_kcal
```

## The method

Both use the MET (metabolic equivalent of task) method,

$$
\text{kcal} = \text{MET} \times W_{\text{kg}} \times \text{hours},
$$

with values from the Compendium of Physical Activities (Ainsworth et al.):

| Category | MET | Rest between sets |
|---|---|---|
| `compound` | 6.0 (vigorous resistance training) | 90 s |
| `isolation` | 3.5 (light to moderate effort) | 60 s |
| `core` (isometric holds, core-only work) | 3.0 (light calisthenics) | 45 s |

The tracker records sets and reps, not a stopwatch, so the **duration is rebuilt** set by set: reps times the seconds per rep, plus the category's rest. Seconds per rep come from the tempo: a dash-separated tempo such as `4-2-2-0` (eccentric, pause, concentric, pause) sums to 8 s; `Normal`, blank, or unparseable means 3 s. An isometric set (no reps) takes its hold time from an `M:SS` duration in the notes, such as `1:00 held`, and 45 s otherwise. `W_kg` is your body weight, by default your last weigh-in.

It is a rule of thumb, not a measurement: it answers "roughly how much," not "exactly how much." Treat it with the suspicion you would give a wearable's calorie count. If you rest notably longer or shorter than the table, the estimate errs in the direction you would expect.

## A routine: `wdm session-kcal`

Copy `data/templates/routine_template.csv` (or the filled-in `routine_example.csv`) to `data/raw/routine.csv`. One row **per exercise**, not per set:

```text
session,exercise,sets,reps,weight_lb,tempo,category,notes
Push A,Flat Bench Press,4,8,135,Normal,compound,
Push A,Overhead Press,3,10,65,Normal,compound,
Push A,Lateral Raise,3,15,20,Normal,isolation,
Push A,Triceps Pushdown,3,12,40,Normal,isolation,
Core,Plank,3,,,—,core,1:00 held
```

| Column | Meaning |
|---|---|
| `session` | The session's name, such as `Lower A`; what `--session` looks up. **Required.** |
| `exercise` | The exercise. **Required.** |
| `sets` | Working sets. **Required.** |
| `reps` | Reps per set; blank for holds |
| `weight_lb` | Load; blank for body weight |
| `tempo` | `4-2-2-0`-style seconds, or `Normal` |
| `category` | `compound`, `isolation`, or `core`. **Required**, stated directly |
| `notes` | Free text; an `M:SS` here is a hold's duration |

Update a row only when the routine itself changes (a new weight, a swapped exercise), never per session.

```bash
wdm session-kcal --list                         # the sessions in your routine
wdm session-kcal --session "Push A"             # at your last weigh-in
wdm session-kcal --session "Push A" --weight-lb 220
```

```text
Push A: 194 kcal   13 sets, ~24 min
```

Stating `category` in the file avoids a real failure of the spreadsheet path: there, an exercise logged as "Weighted Leg Raise" but listed in the Muscle Guide as "Weighted Leg Raise / Russian Twist" silently missed the join, defaulted to compound instead of core, and overstated that session by about 13%.

## A dated log: `wdm exercise-kcal`

For per-session history (different weights on different days, recorded rather than assumed), keep a workout spreadsheet with three sheets:

- **`Upper`** and **`Lower`**: the log, one block of dated rows per session, with columns `Date, Session, Exercise, Set #, Weight (lb/hand), Reps, Tempo, Equipment Cap?, Notes`. Title, instruction, and trailing note rows are ignored (a row needs a parseable date).
- **`Muscle Guide`**: keyed by `(Session, Exercise)`, with a `Movement Pattern` column that sets the category: a pattern containing "isometric", or exactly "core", is `core`; one containing "isolation" is `isolation`; anything else is `compound`.

```bash
wdm exercise-kcal                                   # every logged day's estimate
wdm exercise-kcal --repeat "Lower A" --date 2026-10-28            # Lower A again today
wdm exercise-kcal --repeat "Lower A" --date 2026-10-28 --append   # ... and write it into the sheet
```

`--repeat` reuses the latest logged instance of a session for a new date, which is right only if the routine has not changed since. `--append` also copies that block into the spreadsheet as a real dated entry: it backs the file up as `.bak` first, refuses a duplicate (session, date), and works around an openpyxl 3.1.5 defect that moves cell values but not merged ranges.
