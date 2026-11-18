---
title: "3. Phase 1: the scale's noise"
---

Phase 1 fits a local linear trend (a level, a slope, and white reading noise) to the weigh-ins alone, by maximum likelihood, and ignores the food log. Its output is the reading noise phase 2 will assume, and how hard any trend filter should smooth. The theory is in [Estimation: phase 1](../theory/estimation.md#phase-1); the command-line equivalent is [`wdm fit-noise`](../commands.md#wdm-fit-noise).

```bash
uv run python walkthrough/03_phase1.py
```

<details><summary>The script</summary>

```python
"""Step 3: phase 1, noise identification.

Fit the local linear trend model (level + slope + reading noise) to the weigh-ins
alone, by maximum likelihood. Food logs are ignored. The output is the reading
noise that phase 2 will assume, and how hard a trend filter should smooth.
Then add two older weight-only episodes, as a subject with a past would.
"""

import jax.numpy as jnp
import numpy as np

from wdm.config import Subject
from wdm.estimation import identify_noise
from wdm.filters import trend_filter
from wdm.models import NoiseModel, Params
from wdm.synthetic import SyntheticSpec, make_episode, make_weight_only_episode

# Same subject, truth and episode as step 2.
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

ni = identify_noise([ep])
print(ni.render())
print()

# What "reading noise" means to phase 1: everything faster than a smooth trend.
# The truth has three such sources: the scale's own 0.45 kg; glycogen water,
# which the generator jitters by gly_sd per day and which relaxes with gly_tau;
# and sodium water, which follows each day's random sodium through na_tau.
a = np.exp(-1.0 / float(true.gly_tau))
gly_sd = float(NoiseModel.default().gly_sd) / np.sqrt(1.0 - a**2)
b = np.exp(-1.0 / float(true.na_tau))
na_sd = float(true.na_gain) * spec.sodium_sd_mg * np.sqrt((1.0 - b) / (1.0 + b))
fast = np.sqrt(spec.obs_sd_kg**2 + gly_sd**2 + na_sd**2)
print(
    f"true reading noise {spec.obs_sd_kg:.3f} kg; water scatter: glycogen {gly_sd:.3f} kg,"
    f" sodium {na_sd:.3f} kg; combined {fast:.3f} kg"
)

# Run the trend filter at the fitted scales and score it against the hidden truth.
y = jnp.asarray(np.nan_to_num(ep.weight_kg))
o = jnp.asarray(ep.weight_observed.astype(float))
res = trend_filter(y, o, ni.level_sd, ni.slope_sd, ni.obs_sd)
level = np.asarray(res.filtered_mean)[:, 0]
obs = ep.weight_observed
rmse_read = np.sqrt(np.mean((ep.weight_kg[obs] - truth["scale_kg"][obs]) ** 2))
rmse_trend = np.sqrt(np.mean((level[30:] - truth["scale_kg"][30:]) ** 2))
print(
    f"error vs the true scale weight: one reading {rmse_read:.3f} kg,"
    f" filtered trend (days 30-119) {rmse_trend:.3f} kg"
)
print()

# Two older episodes with weigh-ins but no food log: more data for phase 1 only.
past = [
    make_weight_only_episode(
        SyntheticSpec(n_days=40, start_weight_kg=105.0),
        true,
        name=f"past-0{i}",
        start_date=d,
        seed=10 + i,
    )
    for i, d in enumerate(["2023-03-01", "2024-01-15"], start=1)
]
ni_all = identify_noise([*past, ep])
print("with two past weight-only episodes added")
print(
    f"  weigh-ins {ni_all.n_observations}, observation sd {ni_all.obs_sd:.3f} kg"
    f" (x/{ni_all.obs_sd_factor:.2f}), gain {ni_all.steady_state_gain:.3f},"
    f" window {ni_all.effective_window_days:.1f} days"
)
```

</details>

```text
Phase 1 -- noise identification
  episodes used         1
  weigh-ins used        107
  log-likelihood        -110.72

  observation sd        0.582 kg (1.28 lb)   x/1.10
  level innovation sd   0.12737 kg/day   x/1.84
  slope innovation sd   0.00000 kg/day^2 unidentified

  steady-state gain     0.196
  effective window      5.1 days

true reading noise 0.450 kg; water scatter: glycogen 0.189 kg, sodium 0.136 kg; combined 0.507 kg
error vs the true scale weight: one reading 0.418 kg, filtered trend (days 30-119) 0.385 kg

with two past weight-only episodes added
  weigh-ins 170, observation sd 0.581 kg (x/1.13), gain 0.267, window 3.7 days
```

## What to notice

- **The reading noise is pinned down.** 107 weigh-ins give 0.582 kg (1.28 lb), known to within a factor of 1.10 either way.
- **It is larger than the 0.45 kg set, and the second block says part of why.** Phase 1 has no water or gut compartments, so anything faster than a smooth trend counts as reading noise. Glycogen jitter of 0.15 kg/day with a 2-day time constant settles to a day-to-day spread of 0.15 / √(1 − e^(−1)) = 0.189 kg. Sodium drawn with an 800 mg spread, at the true gain and a 1-day time constant, adds 2.5·10⁻⁴ × 800 × √((1 − e^(−1)) / (1 + e^(−1))) = 0.136 kg. Together, √(0.45² + 0.189² + 0.136²) = 0.507 kg. The fitted 0.582 is above that by a factor of 1.15, more than its own factor of 1.10, so these three sources do not account for all of it. The printed rule leaves out two fast sources the generator now has: carbohydrate moves sodium water too (each gram below the habitual 260 g acts like 4000 / 260 = 15.4 mg less sodium, and carbohydrate swings by tens of grams from day to day and ±60 g over the week), and gut content follows each day's intake with a 3-day lag.
- **That water is counted twice.** Phase 2 models glycogen jitter as process noise and sodium water and gut content as states driven by the log, yet takes phase 1's figure as its reading noise. This is a likely reason step 6 finds the forecast intervals a little wide.
- **The slope innovation is "unidentified".** It prints as 0.00000 kg/day² (4·10⁻⁶), next to the search's lower bound of 10⁻⁶: the trend moves only through its level, as a random walk, with no wander in its slope. The data cannot pin such a value down, so the report says so instead of printing a standard error. The level innovation, 0.127 kg/day, is known only to a factor of 1.84.
- **Gain and window.** A steady-state gain of 0.196 moves the trend 19.6 % of the way toward each weigh-in; the effective window is its reciprocal, 5.1 days.
- **The trend beats a single reading, narrowly.** Against the true scale weight, one reading is off by 0.418 kg (root mean square), the causal trend by 0.385 kg (days 30 to 119). The trend cannot follow the water wiggle.
- **More history, the same reading noise.** Two older weight-only episodes raise the count to 170 weigh-ins and leave the reading noise at 0.581 kg, but the gain rises to 0.267 and the window shortens to 3.7 days. On a real record, old episodes are how a short current one gets a usable noise estimate; phase 2 ignores them ([Which episodes are filtered](../theory/estimation.md#which-episodes)).
- **Standard errors come from the Hessian at the optimum**, so they are optimistic: the trend model has no water and no food.

## The code

| What | Where (under `wdm`) |
|---|---|
| The fit (L-BFGS-B within search bounds), gain, window, standard errors, report | `estimation.noise_id`: `identify_noise`, `NoiseIdentification` |
| Likelihood: the Kalman filter's prediction-error decomposition | `estimation.noise_id.total_loglik`, `filters.linear`: `trend_filter`, `kalman_filter` |
| The local linear trend matrices | `filters.linear.local_linear_trend_matrices` |

## Try this

- Set `obs_sd_kg=0.25` in the spec. The fit gives 0.186 kg, below both the 0.25 kg set and the 0.341 kg the rule above predicts (√(0.25² + 0.189² + 0.136²)), because the level innovation grows to 0.437 kg/day and follows most of the water wiggle as level moves. The gain rises to 0.865 and the window shrinks to 1.2 days: a quieter scale is trusted more.
- Fit Student-t noise with `identify_noise([ep], dof=4.0)`. The log-likelihood falls from −110.72 to −116.24 and the reading noise drops to 0.367 kg, because heavy tails let the variance shrink. On Gaussian data the Gaussian fits better, as it should.
