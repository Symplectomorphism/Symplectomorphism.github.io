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
    na_gain=jnp.asarray(1.5e-5),
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
    1 d     106  -0.09    1.11     1.41    +21%   0.85      57%       92%      -0.751
    7 d     100  -0.11    1.19     1.97    +40%   0.78      61%       97%      -0.819

  bias, RMSE and persistence RMSE in lb. 'persist' forecasts the last reading;
  'skill' is the RMSE improvement over it. A calibrated model has z sd near 1
  and coverage near 50% and 90%. Log score is the mean log predictive density
  per reading (kg units); higher is better, and it is the fair way to compare
  noise models, because it rewards honest intervals as well as accurate means.

h = 1: forecast sd 1.26 to 1.48 lb, median 1.28; reading noise alone 1.08 lb
h = 7: forecast sd 1.38 to 1.79 lb, median 1.42; reading noise alone 1.08 lb

7-day forecasts by stretch     n    RMSE  persist   90% cov
  origin day   0- 29        27    1.67     2.37       96%
  origin day  30- 59        25    0.91     1.78      100%
  origin day  60-119        48    0.97     1.82       96%

five 7-day forecasts, lb      predicted     sd   observed   last reading
  from 2026-06-02          218.3   1.79      215.4          220.2
  from 2026-06-24          211.8   1.53      212.7          215.0
  from 2026-07-16          209.5   1.43      209.6          210.5
  from 2026-08-09          204.3   1.41      204.9          204.4
  from 2026-08-31          200.4   1.39      201.2          202.6
```

## What to notice

- **The model beats persistence, more so at a week.** "Tomorrow reads the same as the last reading" has an RMSE of 1.41 lb at 1 day and 1.97 lb at 7 days, because the weight is falling. The model's RMSE barely grows, 1.11 to 1.19 lb, because it knows what was eaten. Skill is 1 − 1.11 / 1.41 = +21 % at 1 day and 1 − 1.19 / 1.97 = +40 % at 7.
- **No bias.** The mean error is −0.09 lb at 1 day and −0.11 lb at 7.
- **The intervals are a little too wide.** A calibrated forecast has standardized errors with standard deviation 1; these have 0.85 and 0.78. The 90 % intervals cover 92 % and 97 % of the readings. Erring wide is the safer direction, and a likely cause is step 3's: the glycogen jitter is counted once in phase 1's reading noise and again as phase 2 process noise.
- **The floor on any forecast is the reading noise.** Phase 1's 0.492 kg is 1.08 lb, and no forecast sd goes below it: the 1-day median is 1.28 lb, the 7-day median 1.42 lb. The widest are the earliest, before the parameters have contracted (1.79 lb for the forecast from 2026-06-02).
- **Early forecasts are the worst.** For origins in days 0 to 29 the 7-day RMSE is 1.67 lb; after that it is 0.91 and 0.97 lb. Persistence is also worst early (2.37 lb), when the water drop and the initial loss are fastest.
- **One forecast in detail.** From 2026-06-02 the model predicted 218.3 ± 1.79 lb for a week later; the scale said 215.4 (1.6 standard deviations below), and persistence said 220.2. From 2026-08-09 persistence (204.4) happened to be closer to the reading (204.9) than the model (204.3): single forecasts are noisy, which is why the table scores hundreds.
- **Two caveats** are part of the result. Phase 1 saw the whole record, so the intervals have seen a little of the future. And neighboring 7-day forecasts overlap, so the 100 forecasts at 7 days carry fewer than 100 independent errors.

## The code

| What | Where |
|---|---|
| The backtest: refit, then forecast from every morning | `estimation/backtest.py:123` (`backtest`) |
| Start each forecast from the filtered (not smoothed) belief | `estimation/backtest.py:155–156` |
| Persistence: the last reading at or before each day | `estimation/backtest.py:159–165` |
| h unscented time updates, no measurement updates | `estimation/backtest.py:181–188` (`unscented_predict` at 184), `filters/ukf.py:104` |
| Scoring: bias, RMSE, skill, z, coverage, log score | `estimation/backtest.py:207` (`_score`); Gaussian quantiles at 211; skill at 89 |
| The printed table | `estimation/backtest.py:100` (`render`) |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Forecasts use the food actually logged on the days ahead | `estimation/backtest.py:154` | Separates model error from adherence |
| Unlogged days ahead: intake equals expenditure, with a wider variance | `models/energetics.py:95`, `models/dynamics.py:262–263` | The neutral assumption absent a log |
| Each forecast carries the parameter uncertainty of its origin | `estimation/backtest.py:155–156` | The sigma points span parameters as well as states |
| Phase-1 noise is fitted on the whole record | `estimation/backtest.py:21–23` (module note) | Three scalars; the optimism is small |
| Coverage is read against Gaussian quantiles (or Student-t with `dof`) | `estimation/backtest.py:210–211` | The same noise model the filter used |

## Try this

- Score longer horizons with `horizons=(1, 7, 14, 28)` and watch persistence fall further behind.
- Compare noise models fairly by log score: fit both phases with `dof=4.0` (`identify_noise([ep], dof=4.0)`, then `backtest(..., dof=4.0)`). Here z sd moves to 1.09 and 0.95, closer to 1, but the log scores fall from −0.751 and −0.819 to −0.766 and −0.845: on Gaussian data the Gaussian model still forecasts better.
- Run [`wdm backtest`](../commands.md#wdm-backtest) on your own record. A model that cannot beat persistence at a week is not earning its complexity.
