---
title: "6. The backtest"
---

Synthetic recovery (steps 4 and 5) tests the inference on data the model generated. The backtest asks the question that also applies to a real record: replaying the episode day by day, with the filter knowing only what had happened by then, how close were its forecasts of the reading 1 and 7 days ahead, and were its intervals honest? The forecasts are driven by the food that was logged, so they test the model, not adherence. The theory is in [Backtest](../theory/estimation.md#backtest); the command-line equivalent is [`wdm backtest`](../commands.md#wdm-backtest).

```bash
uv run python walkthrough/06_backtest.py
```

<details><summary>The script</summary>

```python
"""Step 6: the backtest, out-of-sample forecasts of the scale.

Replay the episode day by day. From each morning's filtered belief, push the
state forward 1 and 7 days on the food that was logged, with no weigh-ins, and
compare the predicted reading with the one the scale gave. The yardstick is
persistence: "the next reading equals the last one".
"""

import dataclasses

import jax.numpy as jnp
import numpy as np

from wdm.config import Config, Preferences, Safety, Subject
from wdm.constants import LB_PER_KG
from wdm.estimation import backtest, identify_noise
from wdm.models import NoiseModel, Params
from wdm.synthetic import SyntheticSpec, make_episode

subj = Subject(sex="male", age_years=40, height_cm=175.0, body_fat_frac=0.30)
cfg = Config(subject=subj, goal=None, safety=Safety(), preferences=Preferences())
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

ni = identify_noise([ep])
noise = dataclasses.replace(NoiseModel.default(), obs_sd=jnp.asarray(ni.obs_sd))
bt = backtest([ep], cfg, noise, horizons=(1, 7))
print(bt.render())
print()

# The forecast interval widens with the horizon; the reading noise sets its floor.
for h in (1, 7):
    fs = [f for f in bt.forecasts if f.horizon == h]
    sd = np.array([f.sd_kg for f in fs]) * LB_PER_KG
    print(
        f"h = {h}: forecast sd {sd.min():.2f} to {sd.max():.2f} lb, median {np.median(sd):.2f};"
        f" reading noise alone {ni.obs_sd * LB_PER_KG:.2f} lb"
    )

# Skill by stretch of the episode: early on, the parameters are still near the prior.
print()
print("7-day forecasts by stretch     n    RMSE  persist   90% cov")
day = {d: k for k, d in enumerate(ep.dates)}
week = [f for f in bt.forecasts if f.horizon == 7]
for lo, hi in ((0, 30), (30, 60), (60, 120)):
    fs = [f for f in week if lo <= day[f.origin] < hi]
    err = np.array([f.observed_kg - f.predicted_kg for f in fs]) * LB_PER_KG
    per = np.array([f.observed_kg - f.persistence_kg for f in fs]) * LB_PER_KG
    cov = np.mean([abs(f.z) < 1.645 for f in fs])
    print(
        f"  origin day {lo:3d}-{hi - 1:3d}       {len(fs):3d}   {np.sqrt(np.mean(err**2)):5.2f}"
        f"    {np.sqrt(np.mean(per**2)):5.2f}     {cov:5.0%}"
    )

print()
print("five 7-day forecasts, lb      predicted     sd   observed   last reading")
for f in week[::20]:
    print(
        f"  from {f.origin}        {f.predicted_kg * LB_PER_KG:7.1f}  {f.sd_kg * LB_PER_KG:5.2f}"
        f"    {f.observed_kg * LB_PER_KG:7.1f}        {f.persistence_kg * LB_PER_KG:7.1f}"
    )
```

</details>

```text
Backtest -- forecasts of the scale reading (Gaussian noise)

  horizon     n   bias    RMSE  persist   skill   z sd   50% cov   90% cov   log score
    1 d     106  -0.14    1.18     1.57    +25%   0.75      61%       96%      -0.850
    7 d     100  -0.23    1.22     2.06    +41%   0.68      65%       98%      -0.916

  bias, RMSE and persistence RMSE in lb. 'persist' forecasts the last reading;
  'skill' is the RMSE improvement over it. A calibrated model has z sd near 1
  and coverage near 50% and 90%. Log score is the mean log predictive density
  per reading (kg units); higher is better, and it is the fair way to compare
  noise models, because it rewards honest intervals as well as accurate means.

h = 1: forecast sd 1.48 to 2.02 lb, median 1.52; reading noise alone 1.28 lb
h = 7: forecast sd 1.60 to 2.56 lb, median 1.65; reading noise alone 1.28 lb

7-day forecasts by stretch     n    RMSE  persist   90% cov
  origin day   0- 29        27    1.55     2.63      100%
  origin day  30- 59        25    1.07     1.67      100%
  origin day  60-119        48    1.09     1.87       96%

five 7-day forecasts, lb      predicted     sd   observed   last reading
  from 2026-06-02          215.7   2.56      213.6          219.4
  from 2026-06-24          210.0   1.81      210.9          212.5
  from 2026-07-16          208.3   1.66      208.2          209.4
  from 2026-08-09          202.6   1.64      203.4          202.2
  from 2026-08-31          198.4   1.62      199.1          200.5
```

## What to notice

- **The model beats persistence, more so at a week.** "The scale reads the same as last time" has an RMSE of 1.57 lb at 1 day and 2.06 lb at 7, because the weight is falling. The model's RMSE barely grows, 1.18 to 1.22 lb, because it knows what was eaten: skill +25 % and +41 %. A model that cannot beat persistence at a week is not earning its complexity.
- **No bias**: −0.14 and −0.23 lb.
- **The intervals are a little too wide.** Standardized errors have standard deviation 0.75 and 0.68 instead of 1, and the 90 % intervals cover 96 % and 98 %. Erring wide is the safer direction; the likely cause is the fast water counted twice (step 3).
- **The floor on any forecast is the reading noise**, 1.28 lb; the median forecast sd is 1.52 lb at 1 day and 1.65 at 7. The widest are the earliest, before the parameters have contracted.
- **Early forecasts are the worst.** For origins in days 0 to 29 the 7-day RMSE is 1.55 lb, then 1.07 and 1.09. Persistence is also worst early (2.63 lb), when the water drop is fastest.
- **Single forecasts are noisy.** The five above miss the reading by 0.1 lb (from 2026-07-16) to 2.1 lb (from 2026-06-02, the widest interval), which is why the table scores a hundred of them.
- **The caveats** (phase 1 saw the whole record; overlapping 7-day forecasts are not independent) are in [Backtest](../theory/estimation.md#backtest).

## The code

| What | Where (under `wdm`) |
|---|---|
| Refit, then forecast from every morning's filtered belief; persistence; scoring | `estimation.backtest`: `backtest`, `HorizonScore` |
| h unscented time updates, no measurement updates, on the food actually logged | `filters.ukf.unscented_predict` |

## Try this

- Score longer horizons with `horizons=(1, 7, 14, 28)` and watch persistence fall further behind.
- Compare noise models fairly by log score: fit both phases with `dof=4.0` (`identify_noise([ep], dof=4.0)`, then `backtest(..., dof=4.0)`). Here z sd moves to 1.07 and 0.90, closer to 1, and the log scores rise from −0.850 and −0.916 to −0.825 and −0.897: on this record the Student-t model forecasts slightly better.
