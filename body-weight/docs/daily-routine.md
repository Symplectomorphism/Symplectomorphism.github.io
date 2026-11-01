---
title: "The daily routine"
---

Two commands: `wdm daily` once in the morning, and `wdm log` whenever you eat, train, or want to know what is left of today's plan. You never have to open `observations.csv`, though you can edit it by hand at any time and both commands will read what you wrote.

Row *k* of the file holds the weigh-in on the morning of day *k* and everything eaten during day *k*. The routine follows that order: the weight goes in first thing, and the food fills in as the day goes.

Do not expect the numbers to mean much in the first week: the filter reports its prior back until the data have had time to speak ([what to expect, and when](reading-output.md#what-to-expect-and-when)).

## Morning: `wdm daily`

Weigh in under the [same conditions every day](data.md#weighing), then run

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

Then it validates the file, runs both estimation phases and the controller in the [mode the subject file selects](configuration.md#mode), archives the plan in the [plans log](reading-output.md#the-plans-log), and prints the plan ([how to read it](reading-output.md#the-plan)), ending with today's prescription:

```text
  Today: 2164 kcal | 149 g protein (27%) | 64 g carb (12%) | 146 g fat (61%)
```

That line is the only prescription. The run takes a minute or two. `--weight LB` and `--waist IN` supply the readings without prompting, and `--no-input` never prompts (for scripts); [the command reference](commands.md#wdm-daily) lists every flag.

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

Food flags **add** to the day, so meals logged separately sum to the day's totals. Training goes in its own column (`wdm log --exercise KCAL`; the model does the energy balance, so neither column is subtracted from the other), and the step counter's total with `wdm log --steps N`. Eating out, estimate everything and add `--estimate` ([When you don't know exactly](data.md#estimating)). [Commands](commands.md#wdm-log) covers every flag, including corrections with `--set` and `--clear`.

Logging never shifts the plan: `wdm log` compares against the latest plan logged for today, and before `wdm daily` has run it shows the totals alone.

## The waist tape

A weekly waist measurement is the only reading that responds to fat mass directly, and it narrows the fat-mass interval by roughly a tenth: not more, because how many centimeters of waist a kilogram of fat is worth differs from person to person and has to be learned. Measure at the navel, standing, tape level and snug but not compressing, at the end of a relaxed exhale, before eating. The same time and place each time matter more than the exact protocol.

`wdm daily` asks for it every seven days; you can also pass `--waist IN`, or type the value into the `waist_in` column of any row. Measuring more often does no harm. The filter learns your personal offset and slope, so the tape needs no calibration ([the model](theory/model.md#waist)).

## Beyond the routine

- **Replan or try another target** without recording anything: [`wdm plan`](commands.md#wdm-plan).
- **At the goal**, replace `[goal]` with [`[maintain]`](configuration.md#maintain) and carry on with the same two commands.
- **Every figure**: [`wdm plots`](commands.md#wdm-plots); the [Latest figures](../figures.qmd) page shows how to read them.

## Summary

| When | What | Command |
|---|---|---|
| Every morning | Weigh in; get today's prescription | `wdm daily` |
| Weekly (it asks) | Waist at the navel | answer `wdm daily`'s prompt, or `--waist IN` |
| Before each meal | Add the meal; see what is left | `wdm log --kcal ... --protein ... --carb ... --fat ...` |
| After training | Add the session's energy | `wdm session-kcal --session "Lower A"`, then `wdm log --exercise KCAL` |
| At night (optional) | The day's step count | `wdm log --steps N` |
| Eating out | Log a best guess, marked as one | `wdm log ... --estimate` |
| Any time | Replan, or try a different target | `wdm plan ...` |
| At the goal | Hold the weight | `[maintain] weight_lb = ...` in `subject.toml` |
