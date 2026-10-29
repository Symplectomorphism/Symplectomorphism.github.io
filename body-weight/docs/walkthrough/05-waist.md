---
title: "5. The waist tape"
---

The scale sees fat mass only through the energy density of what is lost, which is weak evidence, so step 4's fat estimate ended almost as uncertain as it began, and centered on whatever body fat the config said. A tape measure responds to fat directly. This step fits the same episode without and with its 18 weekly waist readings, once with the configured body fat right (30 %) and once 5 points low (25 %). The waist model is in [The waist tape](../theory/model.md#waist) and its update in [The waist update](../theory/estimation.md#waist-update).

```bash
uv run python walkthrough/05_waist.py
```

<details><summary>The script</summary>

```python
"""Step 5: the tape measure as a second sensor.

The scale sees fat mass only through the energy density of what is lost, so
step 4's fat estimate kept most of its starting uncertainty, and its starting
point was the body-fat fraction in the config. Fit the same episode without
and with its weekly waist readings, once with the configured body fat right
(30 %) and once with it 5 points low, and compare fat mass against the truth.
"""

import dataclasses

import jax.numpy as jnp

from wdm.config import Config, Preferences, Safety, Subject
from wdm.estimation import fit_joint, identify_noise
from wdm.models import NoiseModel, Params
from wdm.models.observation import waist_prior
from wdm.synthetic import SyntheticSpec, make_episode

subj = Subject(sex="male", age_years=40, height_cm=175.0, body_fat_frac=0.30)
c0, c1 = subj.expenditure_prior(100.0)
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
    n_days=120, waist_every_days=7, habitual_carb_g=carb_ref, habitual_sodium_mg=sodium_ref
)
ep, truth = make_episode(spec, true, name="current", start_date="2026-06-01", seed=7)

ni = identify_noise([ep])  # phase 1 uses weigh-ins only, so it is the same either way
noise = dataclasses.replace(NoiseModel.default(), obs_sd=jnp.asarray(ni.obs_sd))

w0 = ep.first_weight_kg()
a0, b0, sd_a = waist_prior(subj.sex, subj.age_years, 0.30 * w0, w0)
print(f"waist prior at 30 % fat: offset {a0:.1f} +- {sd_a:.1f} cm, slope {b0:.3f} cm/kg (+-20 %)")
print(
    f"waist truth: offset {spec.waist_offset_cm:.1f} cm, slope {spec.waist_slope_cm_per_kg:.3f}"
    f" cm/kg; {ep.n_waist} readings, sd {spec.waist_sd_cm} cm"
)
print(f"true fat mass: day 0 {truth['fat_kg'][0]:.2f} kg, day 119 {truth['fat_kg'][-1]:.2f} kg")
print()

print("configured   data          fat day 0         fat day 119      error day 119")
fits = {}
for bf in (0.30, 0.25):
    cfg = Config(
        subject=dataclasses.replace(subj, body_fat_frac=bf),
        goal=None,
        safety=Safety(),
        preferences=Preferences(),
    )
    for label, data in (("scale only", dataclasses.replace(ep, waist_cm=None)), ("with waist", ep)):
        fit = fit_joint([data], cfg, noise)
        fits[bf, label] = fit
        e = fit.episodes[0]
        f, s = e.fat_kg(), e.state_sd()[:, 1]
        err = f[-1] - truth["fat_kg"][-1]
        print(
            f"  {bf:.0%} fat    {label}   {f[0]:5.2f} +- {s[0]:4.2f}     "
            f"{f[-1]:5.2f} +- {s[-1]:4.2f}     {err:+5.2f} kg ({err / s[-1]:+.2f} sd)"
        )
print()

print("waist calibration with 30 % configured: posterior median and 90% interval")
for s in fits[0.30, "with waist"].waist_summaries():
    print(f"  {s.name:22s} {s.median:7.3f}  [{s.q05:.3f}, {s.q95:.3f}]  shift {s.shrinkage:+.1%}")
```

</details>

```text
waist prior at 30 % fat: offset 64.8 +- 7.2 cm, slope 1.238 cm/kg (+-20 %)
waist truth: offset 75.0 cm, slope 1.200 cm/kg; 18 readings, sd 1.5 cm
true fat mass: day 0 30.00 kg, day 119 22.43 kg

configured   data          fat day 0         fat day 119      error day 119
  30% fat    scale only   30.05 +- 3.99     22.53 +- 3.79     +0.11 kg (+0.03 sd)
  30% fat    with waist   31.36 +- 3.57     23.90 +- 3.39     +1.47 kg (+0.44 sd)
  25% fat    scale only   25.15 +- 3.98     18.07 +- 3.68     -4.35 kg (-1.18 sd)
  25% fat    with waist   27.97 +- 3.47     20.87 +- 3.25     -1.56 kg (-0.48 sd)

waist calibration with 30 % configured: posterior median and 90% interval
  waist_offset_cm         69.799  [61.396, 78.126]  shift +7.7%
  waist_slope_cm_per_kg    1.303  [1.020, 1.672]  shift +5.3%
```

## What to notice

- **The tape narrows the fat interval by about a tenth.** At 30 % configured, the day-119 standard deviation falls from 3.79 to 3.39 kg (−11 %); at 25 %, from 3.68 to 3.25 kg (−12 %). The tape sees the product of a personal slope, known only to ±20 %, and the fat change, so it cannot pin fat mass down; what it adds is the population waist-to-fat relation as a second, independent anchor.
- **A wrong starting guess is where the tape earns its keep.** With 25 % configured, the scale-only fit starts at 25.15 kg and stays wrong: −4.35 kg on day 119, −1.18 standard deviations. The scale has no way to correct the starting fat level. With the tape, day 0 moves to 27.97 kg and the day-119 error shrinks to −1.56 kg (−0.48 standard deviations).
- **With a right guess, the tape adds a little bias.** At 30 % configured, the scale-only fit is off by +0.11 kg, but only because it was told the truth. With the tape it is off by +1.47 kg (+0.44 standard deviations). The prior line predicts 64.8 + 1.238 × 30 = 101.9 cm on day 0, while the subject's true line gives 75 + 1.2 × 30 = 111 cm (step 2 read 111.0). The filter splits that 9 cm excess between a larger offset (posterior 69.8 cm, shift +7.7 %) and more fat (31.36 kg on day 0 instead of 30.05). Offset and fat level trade off; only the prior on the offset, ±7.2 cm, decides the split.
- **The true subject is unusual for the prior.** The true offset, 75.0 cm, is (75.0 − 64.8) / 7.2 = 1.4 prior standard deviations above the population line; the true slope, 1.200 cm/kg, is 3 % below the prior's 1.238. Both true values sit inside the posterior intervals: offset [61.4, 78.1] cm, slope [1.020, 1.672] cm/kg.
- **Every error here is inside 1.645 standard deviations.** The intervals are honest in all four fits; the tape makes them narrower without making them dishonest.

## The code

| What | Where |
|---|---|
| The waist prior from the Lean, Han and Deurenberg (1996) equation | `models/observation.py:129` (`waist_prior`), coefficients at 123 |
| Predicted waist; reading noise 1.5 cm | `models/observation.py:174` (`predict_waist`), `models/observation.py:118` |
| The waist block, added only when readings exist | `estimation/joint.py:433` (`use_waist`); transforms at 351; slope log sd at 86 |
| The waist as a function of the augmented state | `estimation/joint.py:358` (`_waist_observation`) |
| The scalar unscented update for the tape | `filters/ukf.py:189` (`aux_update`), gate at 199 |
| The posterior of offset and slope | `estimation/joint.py:180` (`waist_summaries`) |
| The configured body fat as the fat prior | `estimation/joint.py:281` (`initial_body_fat_fraction`) |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Waist is linear in fat mass with a personal offset and slope | `models/observation.py:174` | Simple, and the slope is estimated rather than fixed |
| The prior line comes from a cross-sectional equation, at the configured body fat | `models/observation.py:129`, `estimation/joint.py:351` | No better population anchor is available; its 4.1-point standard error becomes ±7.2 cm on the offset |
| The slope is known to ±20 % (log scale 0.2) | `estimation/joint.py:86` | The equation's implied slope itself changes as fat is lost |
| Offset and slope are personal constants with the same slow random walk as the metabolic parameters | `estimation/joint.py:330` | They are carried and relaxed the same way, but never enter the dynamics |
| One self-measured reading has a 1.5 cm standard deviation | `models/observation.py:118` | Tape placement, posture and breath |
| The waist block is left out entirely when no reading exists | `estimation/joint.py:433` | The unscented spread depends on the state dimension, so idle entries would perturb every other estimate |

## Try this

- Make the subject typical: set `waist_offset_cm=70.0` in the spec. With 30 % configured, the with-waist fit's day-119 fat mass then comes out at 23.15 kg (+0.73 kg), about half the bias above.
- Read the tape daily (`waist_every_days=1`) and see how far the fat interval narrows when the slope uncertainty, not the reading noise, is the limit.
- Configure 35 % body fat (5 points high). The tape cannot help here: this subject's high true offset makes the readings look like extra fat, which is the direction of the error already made. The day-119 fat estimate barely moves, 27.14 kg from the scale alone and 27.16 kg with the tape, against a truth of 22.43; only the interval narrows.
