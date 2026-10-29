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
| `estimated` | 1 or blank | 0 to 1 | 1 when the day's food is an estimate; the filter then treats its energy as uncertain by about a quarter. `wdm log --estimate` sets it (and adds the column if the file lacks it). |
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

Optional but valuable if your tracker reports it. Sodium shifts extracellular water over one to two days and is often a larger source of scale noise than carbohydrate. Supplied, it drives the sodium-water compartment: roughly 0.2 kg of water per 1,000 mg/day of sustained change, and about half a kilogram the morning after a salty restaurant meal, gone in two or three days. Left blank, the day is taken at your habitual sodium (4,000 mg by default), so estimate rather than leave it blank on days you eat out ([When you don't know exactly](#estimating)).

### `exercise_kcal` and `steps`

Both optional. Left blank, steady activity is absorbed into the estimated maintenance, which is often the more honest choice if your activity is stable. If you lift, [Exercise energy](exercise.md) estimates `exercise_kcal` per session. Treat wearable calorie estimates with suspicion; they are usually noisier than food logs. If you supply both columns, only `exercise_kcal` enters the energy balance.

### `waist_in`

Optional, and the only reading that responds to fat mass directly. Weekly is plenty and daily does no harm; protocol in [The daily routine](daily-routine.md#the-waist-tape). The filter learns your personal offset and slope, so the tape needs no calibration, but expect a modest effect: it narrows the fat-mass estimate by about a tenth, because how many centimeters of waist a kilogram of fat is worth differs from person to person.

## When you don't know exactly {#estimating}

**Estimate.** A rough figure on the right day beats a blank, because a blank is not "unknown" to the model in the way you might expect:

| Left blank | What the model assumes that day |
|---|---|
| `kcal`, with no macros either | You ate exactly what you spent, give or take 600 kcal: no information about the day's balance |
| The macros, with `kcal` given | Carbohydrate at your habitual level (260 g by default), no protein effect on the fat/lean split, a flat thermic effect of food |
| One or two macros | The same as all three: a partial split (protein only, say) is ignored, because the model needs all three |
| `sodium_mg` | Your habitual sodium (4,000 mg by default): about a restaurant day, which is wrong for a quiet day at home |

So at a restaurant or a friend's, log your best guess at all of it, and mark the day as an estimate:

```bash
wdm log --kcal 1100 --protein 45 --carb 90 --fat 62 --sodium 2200 --estimate
```

`--estimate` sets the day's `estimated` column to 1. The filter then takes the day's energy as the logged value give or take a quarter of it, instead of exactly, so a guessed day teaches it less about your metabolism than a weighed one. A day without a food log is still worth more as a rough estimate than as a blank: a blank says "no information", an estimate says "about this much".

### Calories

Use the posted number when there is one (US chains with 20 or more locations must post calories; many publish full nutrition online). Posted values are close on average, but individual dishes can be off by 100 kcal or more. Otherwise take the closest entry in your food tracker, and count the extras: bread, drinks, dressing, dessert, the cooking oil.

### Macronutrients

When the restaurant publishes them, use them. Otherwise estimate in order of what matters to the model:

1. **Carbohydrate first.** It drives the glycogen water, the largest day-to-day swing on the scale.
2. **Protein second.** It shifts the fat/lean split and costs the most to digest.
3. **Fat by difference**, so the three add up to the calories: fat ≈ (kcal − 4 × protein − 4 × carb) / 9. This also keeps `wdm validate` from flagging the day.

| Portion | Roughly |
|---|---|
| Palm-sized piece of cooked meat or fish (about 4 oz) | 25 to 35 g protein |
| Two eggs | 12 g protein |
| A cup of cooked rice or pasta | 45 g carbohydrate |
| A slice of bread or a small roll | 15 g carbohydrate |
| A large flour tortilla | 35 g carbohydrate |
| A medium potato, or a medium order of fries | 35 to 45 g carbohydrate |
| A can of regular soda | 40 g carbohydrate |
| Restaurant dishes generally | 40 to 50% of calories from fat (butter, oil, sauces) |

### Sodium

| Meal | Sodium, roughly |
|---|---|
| Sit-down restaurant entrée | 1,500 to 2,500 mg, more with bread, soup or appetizers |
| Fast-food meal | 1,000 to 2,000 mg |
| Soy-sauce dishes, ramen, pho, curries | 3,000 mg or more |
| Pizza | 600 to 700 mg per slice |
| Home cooking at a friend's | 800 to 1,200 mg per plate |

### What an error costs

- **Calories.** A random error averages out over the days around it; a consistent under-estimate (everyone's) is absorbed into the estimated maintenance and cancels out of the prescriptions. What hurts is a habit that drifts ([On accuracy](#the-food-columns)).
- **Carbohydrate.** An error of 100 g shifts the predicted glycogen water by up to about 0.3 kg for a day or two: less than one weigh-in's own noise. Guessing 80 g when it was 130 g is fine; leaving it blank on a 60 g day (and so assuming 260 g) is not.
- **Sodium.** It moves only water, and only for a day or two; its error never reaches the tissue estimate.

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
