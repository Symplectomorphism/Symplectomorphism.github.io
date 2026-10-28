---
title: "2. Synthetic data"
---

A record with a known answer: 120 days of morning weigh-ins, food logs and weekly waist readings, generated from a metabolism that is deliberately not the prior. Steps 3 to 7 all use this episode, rebuilt from the same seed, and score themselves against its hidden truth.

```bash
uv run python walkthrough/02_synthetic.py
```

<details><summary>The script</summary>

```python
"""Step 2: synthetic data with a known truth.

Generate 120 days of the record the subject would keep: a morning weigh-in on
most days, a food log on most days, and a tape-measure waist reading once a
week. The generator runs the same model as step 1, with parameters deliberately
away from the prior, and keeps the hidden states so later steps can be scored.
"""

import jax.numpy as jnp
import numpy as np

from wdm.config import Subject
from wdm.models import Params
from wdm.synthetic import SyntheticSpec, make_episode

subj = Subject(sex="male", age_years=40, height_cm=175.0, body_fat_frac=0.30)
c0, c1 = subj.expenditure_prior(100.0)
prior = Params.prior(c0, c1)
# The "true" metabolism: offset from the prior, so recovering it is a real test.
true = Params(
    rmr_intercept=jnp.asarray(c0 * 1.18),
    rmr_slope=jnp.asarray(c1 * 0.90),
    at_gain=jnp.asarray(0.22),
    at_tau=jnp.asarray(11.0),
    gly_gain=jnp.asarray(0.0042),
    gly_tau=jnp.asarray(2.0),
    na_gain=jnp.asarray(1.5e-5),
    na_tau=jnp.asarray(1.0),
    partition_kappa=jnp.asarray(0.06),
)
carb_ref, sodium_ref = subj.habitual_intake()
spec = SyntheticSpec(
    n_days=120,
    start_weight_kg=100.0,
    body_fat_frac=0.30,
    intake_fraction=0.80,  # eat 80 % of true initial maintenance, +-250 kcal/day
    obs_sd_kg=0.45,
    weigh_in_rate=0.85,
    logging_rate=0.90,
    waist_every_days=7,  # waist = 75 cm + 1.2 cm/kg x fat, read to +-1.5 cm
    habitual_carb_g=carb_ref,
    habitual_sodium_mg=sodium_ref,
)
ep, truth = make_episode(spec, true, name="current", start_date="2026-06-01", seed=7)

print(ep.summary())
print(f"reference intake: {ep.carb_ref_g:.0f} g carb, {ep.sodium_ref_mg:.0f} mg sodium (habitual)")
print()
print("true parameters vs the prior center")
for name in ("rmr_intercept", "rmr_slope", "at_gain", "at_tau", "gly_gain", "partition_kappa"):
    t, p = float(getattr(true, name)), float(getattr(prior, name))
    print(f"  {name:16s} {t:10.4g}  prior {p:10.4g}  ({t / p - 1:+.0%})")
print()

print("first ten days ('-' = not recorded)")
print("  date        reading   truth   kcal   pro  carb  fat  sodium  waist")
for k in range(10):
    w = f"{ep.weight_kg[k]:7.2f}" if ep.weight_observed[k] else "      -"
    food = (
        (
            f"{ep.energy_kcal[k]:5.0f} {ep.protein_g[k]:5.0f} {ep.carb_g[k]:5.0f}"
            f" {ep.fat_g[k]:4.0f} {ep.sodium_mg[k]:7.0f}"
        )
        if ep.intake_known[k]
        else "    -     -     -    -       -"
    )
    waist = f"{ep.waist_cm[k]:6.1f}" if np.isfinite(ep.waist_cm[k]) else "     -"
    print(f"  {ep.dates[k]}  {w}  {truth['scale_kg'][k]:6.2f}  {food}  {waist}")
print()

resid = ep.weight_kg[ep.weight_observed] - truth["scale_kg"][ep.weight_observed]
logged = ep.intake_known
print(
    f"weigh-ins {ep.n_observations}/{ep.n_days}, food logs {logged.sum()}/{ep.n_days},"
    f" waist readings {ep.n_waist}"
)
print(
    f"reading minus truth: mean {resid.mean():+.3f} kg, sd {resid.std(ddof=1):.3f} kg"
    f" (set to {spec.obs_sd_kg})"
)
print(
    f"logged intake: mean {np.nanmean(ep.energy_kcal):.0f} kcal/day,"
    f" carb {ep.carb_g[logged].mean():.0f} g/day"
)
print()
print("hidden truth, day 0 -> day 119")
for key in ("scale_kg", "tissue_kg", "fat_kg", "gly_water_kg", "na_water_kg", "adapt_kcal"):
    a, b = truth[key][0], truth[key][-1]
    print(f"  {key:13s} {a:8.2f} -> {b:8.2f}   ({b - a:+.2f})")
```

</details>

```text
current      2026-06-01 to 2026-09-28   120 d  weigh-ins  107 ( 89.2%)  intake  106 ( 88.3%)  macros yes
reference intake: 260 g carb, 4000 mg sodium (habitual)

true parameters vs the prior center
  rmr_intercept          1538  prior       1303  (+18%)
  rmr_slope             13.05  prior       14.5  (-10%)
  at_gain                0.22  prior       0.15  (+47%)
  at_tau                   11  prior         14  (-21%)
  gly_gain             0.0042  prior      0.003  (+40%)
  partition_kappa        0.06  prior       0.05  (+20%)

first ten days ('-' = not recorded)
  date        reading   truth   kcal   pro  carb  fat  sodium  waist
  2026-06-01        -  100.00      -     -     -    -       -   111.0
  2026-06-02    99.88   99.90   2537   180   278   78    2527       -
  2026-06-03    99.90   99.83   2440   180   264   74    2905       -
  2026-06-04    99.90   99.88   2156   180   205   68    1402       -
  2026-06-05   100.12   99.26   2056   180   172   72    2095       -
  2026-06-06    98.29   98.76   1792   180   116   68    3290       -
  2026-06-07    97.58   98.28   2102   180   174   76    1297       -
  2026-06-08    97.92   98.37   2609   180   277   87    3677   108.8
  2026-06-09    97.71   98.31   2339   180   243   72    1603       -
  2026-06-10    98.79   98.46      -     -     -    -       -       -

weigh-ins 107/120, food logs 106/120, waist readings 18
reading minus truth: mean -0.091 kg, sd 0.410 kg (set to 0.45)
logged intake: mean 2233 kcal/day, carb 211 g/day

hidden truth, day 0 -> day 119
  scale_kg        100.00 ->    88.41   (-11.59)
  tissue_kg       100.00 ->    89.16   (-10.84)
  fat_kg           30.00 ->    22.40   (-7.60)
  gly_water_kg      0.00 ->    -0.73   (-0.73)
  na_water_kg       0.00 ->    -0.01   (-0.01)
  adapt_kcal        0.00 ->  -162.04   (-162.04)
```

## What to notice

- **Missing days.** 107 of 120 mornings have a weigh-in (89.2 %) and 106 days have a food log (88.3 %), against set rates of 85 % and 90 %. Each is drawn independently every day, so 2026-06-10 has a weigh-in but no food log, and 2026-06-01 has neither (only the tape). The episode keeps every calendar day; a missing value is a missing observation, not a missing row.
- **The truth is off the prior.** The maintenance intercept is 18 % above the prior and the mass slope 10 % below; adaptation is 47 % stronger and 21 % faster; glycogen holds 40 % more water per gram. These are the same values `wdm demo` and the test suite use. If the truth sat at the prior, an estimator that ignored the data would look perfect.
- **Readings scatter around the truth.** On 2026-06-05 the scale says 100.12 kg while the truth is 99.26 kg, a +0.86 kg reading (1.9 standard deviations at 0.45). Across all 107 readings the error has mean −0.091 kg and standard deviation 0.410 kg, a sample from the 0.45 kg that was set.
- **The food log.** Protein is 180 g every day (1.8 g/kg of the starting 100 kg), fat is 30 % of energy, and carbohydrate takes the rest plus a weekly ±60 g cycle: 278 g on 2026-06-02, 116 g on 2026-06-06, back up to 277 g on 2026-06-08. The cycle exercises the glycogen compartment. Logged energy averages 2233 kcal/day; the target was 80 % of the true day-0 maintenance, 0.80 × (1538 + 13.05 × 100) = 2274 kcal/day, with ±250 kcal of day-to-day scatter.
- **The habitual reference.** Carbohydrate averages 211 g/day against the habitual 260 g, and sodium is drawn around the generator's default 3000 mg against a habitual 4000 mg, so both water compartments sit below zero on average. On day 119 glycogen water is −0.73 kg: that is one day's value, which includes the generator's 0.15 kg/day of random glycogen jitter (step 3 measures its effect).
- **The tape.** 18 readings, on days 0, 7, ..., 119. The day-0 reading of 111.0 cm matches the waist model, 75 + 1.2 × 30.00 = 111.0 cm, plus a 1.5 cm reading error.
- **The hidden truth.** Over 120 days the scale falls 11.59 kg, tissue 10.84 kg and fat 7.60 kg (70 % of the tissue). Adaptation ends at −162 kcal/day. These are the numbers steps 4 and 5 try to recover.

## The code

| What | Where |
|---|---|
| The episode's settings | `synthetic.py:27` (`SyntheticSpec`) |
| The generator | `synthetic.py:69` (`make_episode`) |
| Intake target, protein, the carbohydrate cycle | `synthetic.py:117`, `synthetic.py:120`, `synthetic.py:123` |
| Morning state recorded before the day's food | `synthetic.py:144` |
| Process noise on tissue and glycogen water | `synthetic.py:160–162` (scales from `NoiseModel.default`, `models/dynamics.py:75`) |
| Readings, missing days, waist | `synthetic.py:167–170`, `synthetic.py:176` |
| The episode container and its summary line | `io.py:40` (`Episode`), `io.py:137` (`summary`) |
| Past episodes with weigh-ins only (step 3) | `synthetic.py:219` (`make_weight_only_episode`) |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| The data come from the estimator's own model | `synthetic.py:69` | This tests the estimator, not the model. Whether the model describes a person is the backtest's question (step 6) and the real record's |
| The food log is exact when present | `synthetic.py:119–124` | No logging bias. A real log's bias lands in `rmr_intercept` ([Limitations](../theory/limitations.md#drifting-bias)) |
| Reading noise is Gaussian, 0.45 kg, with no gross errors | `synthetic.py:168` | The filter's 4-sigma gate is not exercised here |
| Process noise only on tissue (0.08 kg/day) and glycogen water (0.15 kg/day) | `synthetic.py:160–162` | Keeps the data from being cleaner than the filter assumes |
| Weigh-ins and logs are missing at random, independently | `synthetic.py:170` | Real missing days are not random: unlogged days tend to be high-intake days |
| Waist is linear in fat mass, 75 cm + 1.2 cm/kg, read to ±1.5 cm | `synthetic.py:176`, defaults in `synthetic.py:27` | The same form the filter assumes ([The waist tape](../theory/model.md#waist)) |

## Try this

- Lower `logging_rate` to 0.5 and rerun steps 2 and 4. On unlogged days the filter assumes intake equals expenditure and widens its uncertainty ([Unlogged days](../theory/model.md#timescales)); watch the maintenance interval widen.
- Under-log by 10 %: in step 4, fit `dataclasses.replace(ep, energy_kcal=0.9 * ep.energy_kcal)` and see how far `rmr_intercept` and the maintenance estimate move. The fitted maintenance describes the log, not the body.
- Change `seed=7` to another value. Every later step rebuilds this episode, so all their numbers change; the conclusions should not.
