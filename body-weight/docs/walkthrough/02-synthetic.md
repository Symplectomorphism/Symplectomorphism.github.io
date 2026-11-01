---
title: "2. Synthetic data"
---

A record with a known answer: 120 days of morning weigh-ins, food logs and weekly waist readings, generated from a metabolism that is deliberately not the prior. Steps 3 to 8 all use this episode, rebuilt from the same seed, and score themselves against its hidden truth.

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
    na_gain=jnp.asarray(2.5e-4),
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
  2026-06-02    99.75   99.76   2537   180   278   78    2527       -
  2026-06-03    99.64   99.56   2440   180   264   74    2905       -
  2026-06-04    99.64   99.62   2156   180   205   68    1402       -
  2026-06-05    99.64   98.78   2056   180   172   72    2095       -
  2026-06-06    97.83   98.30   1792   180   116   68    3290       -
  2026-06-07    97.31   98.01   2102   180   174   76    1297       -
  2026-06-08    97.42   97.87   2609   180   277   87    3677   108.8
  2026-06-09    97.49   98.09   2339   180   243   72    1603       -
  2026-06-10    98.36   98.02      -     -     -    -       -       -

weigh-ins 107/120, food logs 106/120, waist readings 18
reading minus truth: mean -0.091 kg, sd 0.410 kg (set to 0.45)
logged intake: mean 2233 kcal/day, carb 211 g/day

hidden truth, day 0 -> day 119
  scale_kg        100.00 ->    88.22   (-11.78)
  tissue_kg       100.00 ->    89.20   (-10.80)
  fat_kg           30.00 ->    22.43   (-7.57)
  gly_water_kg      0.00 ->    -0.73   (-0.73)
  na_water_kg       0.00 ->    -0.24   (-0.24)
  adapt_kcal        0.00 ->  -161.41   (-161.41)
```

## What to notice

- **Missing days.** 107 of 120 mornings have a weigh-in and 106 days a food log, against set rates of 85 % and 90 %, drawn independently: 2026-06-10 has a weigh-in but no food log, 2026-06-01 neither. A missing value is a missing observation, not a missing row.
- **The truth is off the prior.** The maintenance intercept is 18 % above the prior and the mass slope 10 % below; adaptation is 47 % stronger and 21 % faster; glycogen holds 40 % more water per gram; the sodium gain (not printed; step 4 scores it) is 25 % above its prior. If the truth sat at the prior, an estimator that ignored the data would look perfect.
- **Readings scatter around the truth.** On 2026-06-05 the scale reads 99.64 kg against a truth of 98.78 kg, +0.86 kg (1.9 standard deviations). Over all 107 readings the error has mean −0.091 kg and standard deviation 0.410 kg, a sample from the 0.45 kg set.
- **The food log.** Protein is 180 g every day, fat 30 % of energy, and carbohydrate the rest plus a weekly ±60 g cycle (278 g on 06-02, 116 g on 06-06) that exercises the glycogen compartment. Logged energy averages 2233 kcal/day, near the target of 80 % of true day-0 maintenance, 0.80 × (1538 + 13.05 × 100) = 2274.
- **Sodium moves the truth from day to day.** The 1402 mg on 2026-06-04 pulls sodium water toward 2.5·10⁻⁴ × (1402 − 4000) = −0.65 kg, 63 % of which lands by the next morning: the truth on 06-05 is 0.84 kg below the day before. Step 3 shows what this does to the scale's apparent noise.
- **Both water compartments sit below zero**, because carbohydrate averages 211 g against a habitual 260 g and sodium is drawn around 3000 mg against 4000 mg. On day 119 sodium water is −0.24 kg, close to 2.5·10⁻⁴ × (3000 − 4000).
- **The tape.** 18 weekly readings. Day 0's 111.0 cm matches 75 + 1.2 × 30.00 = 111.0 cm plus a 1.5 cm reading error.
- **The hidden truth.** Over 120 days the scale falls 11.78 kg, tissue 10.80 kg and fat 7.57 kg (70 % of tissue); adaptation ends at −161 kcal/day. Steps 4 and 5 try to recover these.

## The code

| What | Where (under `wdm`) |
|---|---|
| The episode's settings and the generator | `synthetic`: `SyntheticSpec`, `make_episode` |
| Process noise scales | `models.dynamics.NoiseModel` |
| The episode container and its summary line | `io.Episode` |
| Past episodes with weigh-ins only (step 3) | `synthetic.make_weight_only_episode` |

## How the data are generated

- The data come from the estimator's own model, so this tests the estimator, not the model; whether the model describes a person is the backtest's question (step 6).
- The food log is exact when present. A real log's bias lands in `rmr_intercept` ([Limitations](../theory/limitations.md#drifting-bias)).
- Reading noise is Gaussian, 0.45 kg, with no gross errors, so the filter's outlier gate is not exercised.
- Process noise only on tissue (0.08 kg/day) and glycogen water (0.15 kg/day).
- Sodium is drawn independently each day, 3000 ± 800 mg, and logged exactly; this day-to-day variation is what lets phase 2 see the sodium gain.
- Weigh-ins and logs are missing at random. Real missing days are not: unlogged days tend to be high-intake days.
- Waist is linear in fat mass, 75 cm + 1.2 cm/kg, read to ±1.5 cm: the form the filter assumes ([The waist](../theory/model.md#waist)).

## Try this

- Lower `logging_rate` to 0.5 and rerun steps 2 and 4. On unlogged days the filter assumes intake equals expenditure and widens its uncertainty ([Unlogged days](../theory/model.md#timescales)); watch the maintenance interval widen.
- Under-log by 10 %: in step 4, fit `dataclasses.replace(ep, energy_kcal=0.9 * ep.energy_kcal)` and see how far `rmr_intercept` and the maintenance estimate move. The fitted maintenance describes the log, not the body.
- Change `seed=7` to another value. Every later step rebuilds this episode, so all their numbers change; the conclusions should not.
