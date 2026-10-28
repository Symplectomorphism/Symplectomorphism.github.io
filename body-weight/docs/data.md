---
title: "The observations file"
---

`data/raw/observations.csv` holds one row per **calendar day**, per **episode**. Every column except `date` and `episode` may be left blank: the filter is built for missing data, which is the reason for a state-space model rather than curve fitting. `wdm daily` and `wdm log` write to it for you; you can also edit it by hand at any time.

Templates: `data/templates/observations_template.csv` (blank) and `observations_example.csv` (filled in).

```text
date,episode,weight_lb,kcal,protein_g,carb_g,fat_g,fiber_g,alcohol_g,sodium_mg,exercise_kcal,steps,waist_in,notes
2026-10-27,current,207.9,2456,180,242,85,,,2781,310,8400,,
2026-10-28,current,203.6,520,38,41,22,,,,,,38.2,salty dinner the night before
```

Row *k* is day *k*: the weigh-in on its morning and everything eaten during it. Today's row starts with the weight and fills in as you eat.

## Columns

| Column | Units | Range accepted | Meaning |
|---|---|---|---|
| `date` | `YYYY-MM-DD` | | Calendar date. **Required.** Unique within an episode. |
| `episode` | text | | One continuous diet attempt. **Required.** |
| `weight_lb` | lb | 50 to 800 | Morning scale weight. Blank on days you did not weigh in. (`weight_kg`, 25 to 360, also accepted.) |
| `kcal` | kcal/day | 0 to 12,000 | Total energy eaten. |
| `protein_g` | g/day | 0 to 700 | Protein. |
| `carb_g` | g/day | 0 to 2,000 | Carbohydrate, **total** (including fiber), not net. |
| `fat_g` | g/day | 0 to 700 | Fat. |
| `fiber_g` | g/day | 0 to 300 | Dietary fiber; slightly reduces metabolizable energy. |
| `alcohol_g` | g/day | 0 to 500 | Alcohol, 7 kcal/g; often left out of logs. |
| `sodium_mg` | mg/day | 0 to 30,000 | Sodium; a major driver of day-to-day water weight. |
| `exercise_kcal` | kcal/day | 0 to 8,000 | Deliberate exercise energy above baseline. Blank counts as 0. |
| `steps` | count | 0 to 150,000 | Step count. Kept as a diagnostic, not used in the energy balance. |
| `waist_in` | in | 15 to 90 | Waist at the navel. (`waist_cm`, 38 to 230, also accepted.) |
| `notes` | text | | Free text, never parsed. Useful when reviewing outliers. |

The ranges catch unit errors and typos (a weight in kg typed into the lb column, a slipped decimal point), not unusual days. A row may give a quantity in one unit only: `weight_lb` or `weight_kg`, `waist_in` or `waist_cm`. Columns not listed here are ignored with a warning.

## Rows and episodes

**One row per day, no skipped days.** On a day you missed, keep the row and leave its values blank. Explicit gaps let the filter propagate uncertainty through the missing days; an absent row would be treated as if no time had passed. `wdm validate` warns about calendar gaps within an episode.

**An episode is one continuous attempt.** Use any short label: `past-01`, `past-02`, `current`. Episodes matter because:

- each gets its **own initial condition**: you started each attempt at a different weight, glycogen level, and adaptation state;
- **parameters are pooled across episodes**: your maintenance energy and metabolic response are properties of you, so every episode contributes to estimating them, and weight-only history still teaches phase 1 how noisy your scale is;
- a long stretch with no records between two diets needs no blank rows: start a new episode instead.

Within an episode, dates should be contiguous. Between episodes, gaps are fine. `wdm daily` and `wdm log` add new days to the most recent episode, so the first row of a new attempt is entered by hand.

## What each column buys

### `weight_lb`

**Consistency matters far more than precision.** Weigh under the same conditions every time: first thing in the morning, after the bathroom, before eating or drinking, ideally unclothed. A scale that reads 1.5 lb heavy is harmless: the bias is absorbed into the initial condition. A scale used at random times of day injects noise that no filter can remove. If you weighed twice, keep the morning fasted value.

### The food columns

`kcal` and the macronutrients are the model's **input**. Their quality decides what it can learn:

| You log | You get |
|---|---|
| nothing | trend estimation only; no maintenance estimate |
| `kcal` only | maintenance and adaptation estimates, but no glycogen-water correction |
| `kcal` and macronutrients | the full model: carbohydrate drives the glycogen-water compartment, which explains much of the day-to-day scale noise, and the protein/carb/fat split changes the thermic effect of food |

A day counts as logged when it has `kcal` or a full protein/carb/fat split. A day with neither is treated as intake unknown, not as a fast; a real fast is `kcal` = 0.

**On accuracy.** Self-reported intake under-reports true intake by roughly 20% on average in doubly-labeled-water studies. A **constant** proportional bias is mostly harmless here: it is absorbed into the estimated maintenance energy and cancels out of the prescriptions. What hurts is a bias that **shrinks over time** as your logging improves, because "apparently eating more while losing the same" is exactly the signature of metabolic adaptation, and the filter cannot tell the two apart from weight data alone. Log with the same rigor on day 90 as on day 1, and note any change of method in `notes`.

### `sodium_mg`

Optional but valuable if your tracker reports it. Sodium shifts extracellular water over one to two days and is often a larger source of scale noise than carbohydrate. Supplied, it enters the sodium-water compartment as a second driver; left blank, its effect is absorbed into the observation noise, which makes your data noisier but not wrong. A restaurant meal can carry 3 g of extra sodium and hold a kilogram of water for two days.

### `exercise_kcal` and `steps`

Both optional. Left blank, steady activity is absorbed into the estimated maintenance, which is often the more honest choice if your activity is stable. If you lift, [Exercise energy](exercise.md) estimates `exercise_kcal` per session. Treat wearable calorie estimates with suspicion; they are usually noisier than food logs. If you supply both columns, only `exercise_kcal` enters the energy balance.

### `waist_in`

Optional, and the only reading that responds to fat mass directly. Weekly is plenty and daily does no harm; protocol in [The daily routine](daily-routine.md#the-waist-tape). The filter learns your personal offset and slope, so the tape needs no calibration, but expect a modest effect: it narrows the fat-mass estimate by about a tenth, because how many centimeters of waist a kilogram of fat is worth differs from person to person.

## Missing data and outliers

| Situation | What to do |
|---|---|
| Missed a weigh-in | Keep the row; leave `weight_lb` blank |
| Missed logging food | Keep the row; leave the food columns blank |
| Weighed twice in a day | Keep the morning fasted value only |
| Long gap between diets | Start a new episode |
| A reading looks wrong | Keep it and say why in `notes`; the filter's outlier gate downweights it |

Do not delete inconvenient rows or interpolate missing values by hand: both corrupt the noise estimates that everything downstream depends on. Reporting a gap honestly is strictly better than filling it in.

## Validation

`wdm validate` (and `wdm daily`, before it plans) checks the file. **Errors** mean it cannot be interpreted and block everything: a missing `date` or `episode` column, an unparseable date, a blank episode label, a duplicate (episode, date), a non-numeric value, a value outside its range, or both units of one quantity in a row. **Warnings** mean it can be read but deserves a look: unrecognized columns, no weight column, calendar gaps in an episode, and days whose macronutrients do not account for the energy total:

$$
\text{kcal} \approx 4\,\text{protein\_g} + 4\,\text{carb\_g} + 9\,\text{fat\_g} + 7\,\text{alcohol\_g},
$$

within 12% or 60 kcal, whichever is larger. A persistent gap usually means net rather than total carbohydrate, unlogged alcohol, or an under-recorded macronutrient, all of which bias the maintenance estimate.
