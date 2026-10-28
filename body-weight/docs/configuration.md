---
title: "The subject file"
---

`data/raw/subject.toml` describes you, your goal, the limits the controller may never cross, and your taste in macronutrients. Copy `data/templates/subject_template.toml`, which documents every field in comments. Only `[subject]` is required.

```toml
[subject]
sex = "male"
age_years = 41
height_cm = 175.0
# body_fat_frac = 0.28
# habitual_carb_g = 260
# habitual_sodium_mg = 4000

[goal]
target_weight_lb = 170.0
target_date = "2027-04-01"

[safety]
min_kcal = 1500.0
max_loss_frac_per_week = 0.01
min_protein_g_per_kg = 1.6
# max_kcal = 6000.0

[preferences]
# target_carb_fraction = 0.08
# target_fat_fraction = 0.45
```

A misspelled key in `[subject]`, `[safety]`, or `[preferences]` is an error, not a silent default, and every value is checked when the file is read.

## `[subject]`

These build the **priors**, the starting beliefs the filter updates from your data. They do not fix any estimate.

| Key | Meaning | Accepted | Default |
|---|---|---|---|
| `sex` | `"male"` or `"female"`, for the Mifflin-St Jeor resting-metabolic-rate prior and the habitual-intake defaults | | required |
| `age_years` | Age at the start of the current episode | 0 to 130 | required |
| `height_cm` | Height in centimeters (5 ft 10 in = 177.8 cm) | 50 to 260 | required |
| `body_fat_frac` | Measured or estimated body-fat fraction at the start of the current episode; initializes the fat/lean split. The filter tolerates a rough guess. | 0.02 to 0.75 | the Deurenberg estimate from BMI, age, and sex |
| `habitual_carb_g` | Carbohydrate you habitually ate before a diet, g/day | 0 to 1,000 | 260 (male), 200 (female) |
| `habitual_sodium_mg` | Sodium you habitually ate before a diet, mg/day | 0 to 15,000 | 4,000 (male), 3,000 (female) |

**Why the habitual intake matters.** The water compartments measure deviations from what you ate before the diet, so a diet starting from ordinary eating starts with both at zero, and the drop in glycogen and sodium water over the first days is predicted instead of being mistaken for fat loss. The defaults are rough US adult averages. If a few ordinary days in a food tracker tell you your usual intake, put it here. See [the model](theory/model.md).

## `[goal]`

| Key | Meaning |
|---|---|
| `target_weight_lb` | Goal weight, lb |
| `target_date` | The date to reach it, `"YYYY-MM-DD"` |

When this section is present, `wdm plan` and `wdm daily` use **deadline mode** by default; without it they use **sustainable mode**. `--mode` overrides either, and in sustainable mode the goal is ignored. `wdm plan --target-weight-lb` and `--target-date` override it for one run. To change your deadline for good, edit `target_date` here.

If the goal cannot be reached within your rate limit, the plan still respects the limit, and a `NOTE` gives the earliest date that can be met ([reading the plan](reading-output.md#the-plan)).

## `[safety]`

Hard limits. The optimizer treats them as constraints, and an independent check of the finished plan reports any violation as `CONSTRAINT VIOLATIONS (do not follow this plan)`.

| Key | Meaning | Default |
|---|---|---|
| `min_kcal` | Floor on prescribed intake, kcal/day | 1,500 |
| `max_kcal` | Ceiling on prescribed intake, kcal/day | 6,000 |
| `max_loss_frac_per_week` | Ceiling on the rate of **tissue** loss, as a fraction per week (0.01 = 1%/week); applied per day as a fraction 1/7 of this of the day's tissue | 0.01 |
| `min_protein_g_per_kg` | Floor on protein, g per kg of body weight at the plan's start | 1.6 |

`min_kcal` must be positive and below `max_kcal`, and `max_loss_frac_per_week` must lie in (0, 0.10). The defaults are deliberately conservative. Do not lower the intake floor or raise the rate limit without talking to a doctor or dietitian.

The rate limit applies to tissue, not scale weight: a fast water drop in the first days of a diet is neither progress nor danger, and the controller is not fooled by it. [Control](theory/control.md) has the formulation.

## `[preferences]`

Diet style, with no principled default but yours. Nothing here is a guardrail.

| Key | Meaning | Default |
|---|---|---|
| `target_carb_fraction` | Carbohydrate's share of energy to hold: 0.05 to 0.10 for keto-style, 0.20 to 0.30 for low-carb. A strongly weighted preference, not just a floor. | unset |
| `target_fat_fraction` | Fat's share of energy to settle on when nothing else sets the split. Used only if `target_carb_fraction` is unset. | unset: the midpoint of 0.15 and the maximum fat fraction (0.75) |

State one of the two. Pulling on either share moves the other, and carbohydrate is the more direct way to name most diet styles. Both have per-run overrides on `wdm plan` (`--target-carb-fraction`, `--target-fat-fraction`), along with `--carb-aversion`, `--min-carb-g` (default 20 g), and `--max-fat-fraction` (default 0.75); see [Commands](commands.md#wdm-plan).
