---
title: "The daily routine"
---

Two commands: `wdm daily` once in the morning, and `wdm log` whenever you eat, train, or want to know what is left of today's plan. You never have to open `observations.csv`, though you can edit it by hand at any time and both commands will read what you wrote.

Row *k* of the file holds the weigh-in on the morning of day *k* and everything eaten during day *k*. The routine follows that order: the weight goes in first thing, and the food fills in as the day goes.

## Morning: `wdm daily`

Weigh in under identical conditions every day: after the bathroom, before food or drink, same clothing (or none), same scale on the same spot. Then run

```bash
wdm daily
```

In a terminal it asks, in this order, only for what is missing:

```text
Yesterday (2026-10-27) has no food logged. Enter your tracker's totals; blank skips.
  energy, kcal: 1709
  protein, g: 180
  carbohydrate, g: 102
  fat, g: 65
  fiber, g:
  alcohol, g:
  sodium, mg: 2872
  exercise, kcal (wdm session-kcal on training days, 0 if none):
  steps:
  updated 2026-10-27: kcal 1709, protein_g 180, carb_g 102, fat_g 65, sodium_mg 2872
This morning's weight (2026-10-28), lb: 203.6
Waist, in (due; last 2026-10-21); blank skips: 38.2
  updated 2026-10-28: weight_lb 203.6, waist_in 38.2
```

- **Yesterday's food** is asked for only if yesterday's row has no food at all (no `kcal` and no full macro split), and then only for the fields still empty. If you log meals with `wdm log` as you go, you will never see this question.
- **This morning's weight** is asked for unless today's row already has one.
- **The waist** is asked for when seven days have passed since the last reading (see [the waist tape](#the-waist-tape)).

A blank answer skips the item. A value that conflicts with one already in the file is refused, not overwritten, because a silent overwrite is how a typo becomes data; fix the file by hand if the old value is the wrong one. A day's row that does not exist yet is created in the latest episode.

Then it validates the file, runs both estimation phases and the controller, archives the plan in the [plans log](reading-output.md#the-plans-log), and prints the plan and today's prescription:

```text
150 rows | 3 episodes | 122 weigh-ins | 66 intake records
  no problems found

Plan (deadline mode, 155 day horizon)

  date           kcal   pro  carb   fat  pro%  carb%  fat%   pred lb   lb/wk loss%/wk
  2026-10-28     2164   149    64   146    27     12    61     204.5    2.02     0.63
  2026-10-29     2164   149    64   146    27     12    61     204.2    1.89     0.64
  ...
  plan written to /home/you/weight-dynamics/code/plan.csv

  all safety constraints satisfied

  plan logged to /home/you/weight-dynamics/data/plans/log.csv

  Today: 2164 kcal | 149 g protein (27%) | 64 g carb (12%) | 146 g fat (61%)
  Only today's line is a prescription. The rest is what the model currently
  expects, and it will change as more weigh-ins arrive.
```

The run takes a minute or two (the filter passes over your whole history). [Reading the output](reading-output.md#the-plan) explains each column. The mode is deadline when `subject.toml` has a `[goal]` and sustainable otherwise; `--mode` overrides it.

**Flags.** `--weight LB` and `--waist IN` supply the readings without prompting; `--date YYYY-MM-DD` records a different morning; `--no-input` never prompts (for scripts and cron jobs; it then only warns about missing food). The [command reference](commands.md#wdm-daily) lists them all.

## During the day: `wdm log`

Before each meal, add it to today's row and see what is left:

```bash
wdm log --kcal 520 --protein 38 --carb 41 --fat 22
```

```text
  updated 2026-10-28: kcal 520, protein_g 38, carb_g 41, fat_g 22
  2026-10-28 so far, against today's plan:
    kcal        520 / 2164 kcal   1644 kcal left
    protein      38 / 149 g   111 g left
    carb         41 / 64 g   23 g left
    fat          22 / 146 g   124 g left
```

Every food flag **adds** to what the day already holds, so three meals logged separately sum to the day's totals. The same goes for `--fiber`, `--alcohol`, `--sodium`, and `--exercise`. `--steps` is the exception: a step counter reports a running total, so `--steps` **replaces** the day's value and the latest total wins.

```bash
wdm log --exercise 310               # after a training session (see Exercise energy)
wdm log --steps 8400                 # at night, the counter's total for the day
wdm log                              # no flags: just show today's totals against the plan
wdm log --date 2026-10-27 --kcal 180 # a snack you forgot yesterday
```

Eating out, or at a friend's, without exact numbers? Estimate all of it (calories, the three macros, sodium) rather than leaving cells blank; [When you don't know exactly](data.md#estimating) has rules of thumb and explains why a blank is worse than a rough guess.

Logging never shifts the plan. Today's prescription is fixed by the latest plan that starts today (the last row of `data/plans/log.csv` with today's start date), and `wdm log` compares against it. If you have not run `wdm daily` yet today it shows the totals alone and says so.

`wdm eat` is an older name for the same command and still works.

### Food and exercise are separate columns

`--kcal` adds to `kcal` (energy eaten); `--exercise` adds to `exercise_kcal` (energy spent training). Neither is subtracted from the other: the model does the energy balance itself. Leave `exercise_kcal` blank on rest days; blank counts as zero. If you run the same `--exercise` command twice, the session is counted twice; fix the cell by hand.

## The waist tape

A weekly waist measurement is the only reading that responds to fat mass directly, and it narrows the fat-mass interval by roughly a tenth. Measure at the navel, standing, tape level and snug but not compressing, at the end of a relaxed exhale, before eating. The same time and place each time matter more than the exact protocol.

`wdm daily` asks for it every seven days; you can also pass `--waist IN`, or type the value into the `waist_in` column of any row by hand. Measuring more often does no harm. The filter learns your personal offset and slope (centimeters of waist per kilogram of fat), so the tape needs no calibration. See [the model](theory/model.md) for the observation equation.

## Running the plan yourself

`wdm daily` is `wdm plan` plus the bookkeeping. To replan at any time without recording anything, or to try a different target:

```bash
wdm plan                           # daily's plan, not logged
wdm plan --log-dir DIR             # ... logged in DIR (daily: data/plans)
wdm plan --mode sustainable        # ignore the goal; lose at the safe rate
wdm plan --target-date 2027-06-01  # a later deadline, this run only
```

A replan at any time of day gives today's prescription for the same data: the plan starts this morning while today's row is today, however much food it already holds. Change the deadline itself in `subject.toml` (`[goal] target_date`).

## Summary

| When | What | Command |
|---|---|---|
| Every morning | Weigh in; get today's prescription | `wdm daily` |
| Weekly (it asks) | Waist at the navel | answer `wdm daily`'s prompt, or `--waist IN` |
| Before each meal | Add the meal; see what is left | `wdm log --kcal ... --protein ... --carb ... --fat ...` |
| After training | Add the session's energy | `wdm session-kcal --session "Lower A"`, then `wdm log --exercise KCAL` |
| At night (optional) | The day's step count | `wdm log --steps N` |
| Any time | Replan, or try a different target | `wdm plan ...` |
