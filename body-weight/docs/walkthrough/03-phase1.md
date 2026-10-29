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
  log-likelihood        -86.53

  observation sd        0.488 kg (1.08 lb)   x/1.07
  level innovation sd   0.00010 kg/day   unidentified
  slope innovation sd   0.00367 kg/day^2 x/1.88

  steady-state gain     0.115
  effective window      8.7 days

true reading noise 0.450 kg; water scatter: glycogen 0.189 kg, sodium 0.136 kg; combined 0.507 kg
error vs the true scale weight: one reading 0.418 kg, filtered trend (days 30-119) 0.290 kg

with two past weight-only episodes added
  weigh-ins 170, observation sd 0.491 kg (x/1.06), gain 0.111, window 9.0 days
```

## What to notice

- **The reading noise is pinned down.** 107 weigh-ins give an observation standard deviation of 0.488 kg (1.08 lb), known to within a factor of 1.07 either way.
- **It is larger than the 0.45 kg that was set, and the second block says why.** Phase 1 has no water compartments, so anything faster than a smooth trend counts as reading noise. The generator jitters glycogen water by 0.15 kg/day, and with its 2-day time constant that settles to a day-to-day spread of 0.15 / √(1 − e^(−1)) = 0.189 kg.
- **Sodium water now adds its own share.** Each day's sodium is drawn with an 800 mg spread, and at the true gain of 2.5·10⁻⁴ kg per mg/day with a 1-day time constant that gives 2.5·10⁻⁴ × 800 × √((1 − e^(−1)) / (1 + e^(−1))) = 0.136 kg of day-to-day scatter. All three together give √(0.45² + 0.189² + 0.136²) = 0.507 kg; the fitted 0.488 is within its factor of 1.07 of that. Phase 2 models sodium water from the sodium log, so this share is counted twice once phase 2 takes phase 1's figure (step 6 sees the result).
- **The level innovation is "unidentified".** It sits at 0.00010 kg/day, the lower bound of the search. The maximum-likelihood trend moves only through its slope, which is a legitimate fit, but an estimate on a bound has no two-sided standard error, so the report says so instead of printing one. The slope innovation, 0.00367 kg/day², is known only to a factor of 1.88.
- **Gain and window.** A steady-state gain of 0.115 means each weigh-in moves the trend 11.5 % of the way toward itself; the effective window is its reciprocal, 8.7 days. A single surprising morning moves the estimate by about a ninth of the surprise.
- **The trend beats a single reading.** Against the true scale weight, one reading is off by 0.418 kg (root mean square), the causal trend by 0.290 kg (days 30 to 119, after start-up). The trend cannot do better than the water wiggle it is not built to follow.
- **More history, same answer.** Two older 40-day weight-only episodes raise the count to 170 weigh-ins and barely move anything: 0.491 kg, a factor of 1.06, gain 0.111, window 9.0 days. They come from the same generator, so this is expected; on a real record, old episodes are how a short current one gets a usable noise estimate. Phase 2 ignores weight-only episodes ([Which episodes are filtered](../theory/estimation.md#which-episodes)).

## The code

| What | Where |
|---|---|
| Fit, bounds, L-BFGS-B | `estimation/noise_id.py:153` (`identify_noise`), bounds at 200, `minimize` at 202 |
| Likelihood: the Kalman filter's prediction-error decomposition | `estimation/noise_id.py:139` (`total_loglik`), `filters/linear.py:168` (`trend_filter`), `filters/linear.py:50` (`kalman_filter`) |
| The local linear trend matrices | `filters/linear.py:43` |
| Steady-state gain and window | `estimation/noise_id.py:80`, `:99`; Riccati iteration at 228 |
| Standard-error factors, "unidentified" | `estimation/noise_id.py:248` |
| The printed report | `estimation/noise_id.py:110` |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Weight is a level plus a slope, each a random walk, plus white reading noise | `filters/linear.py:43`, `:168` | Needs no food log, so any weigh-in history can be used |
| Every episode shares the three noise scales and has its own diffuse start (first reading, 4 kg² variance) | `estimation/noise_id.py:139`, `filters/linear.py:193–197` | The scale and the subject's habits are the same across episodes; the starting weight is not |
| Gaussian readings by default | `estimation/noise_id.py:153` (`dof=None`) | Phase 1 and phase 2 must make the same choice |
| Search bounds: reading noise 0.03 to 5 kg | `estimation/noise_id.py:200` | Keeps the optimizer in physically sane territory |
| Standard errors from the Hessian at the optimum | `estimation/noise_id.py:248` | Optimistic: the trend model is misspecified (no water, no food) |
| The fast water that phase 1 calls reading noise is also in phase 2: glycogen jitter as process noise, sodium water as a state driven by the log | `models/dynamics.py:75` (`gly_sd`), `models/glycogen.py:52` (`sodium_rate`) | Counted twice, it is a likely reason step 6 finds the forecast intervals a little wide |

## Try this

- Set `obs_sd_kg=0.25` in the spec. The fit gives 0.273 kg, below the 0.341 kg the rule above predicts (√(0.25² + 0.189² + 0.136²)), because the level innovation is no longer pinned at its bound: at 0.175 kg/day it now follows part of the water wiggle as level moves. The gain rises to 0.468 and the window shrinks to 2.1 days: a quieter scale is trusted more.
- Fit Student-t noise with `identify_noise([ep], dof=4.0)`. The log-likelihood falls from −86.53 to −93.96 and the reading noise drops to 0.359 kg, because heavy tails let the variance shrink. On Gaussian data the Gaussian fits better, as it should.
- Run [`wdm fit-noise`](../commands.md#wdm-fit-noise) on your own file and compare your window with 8.7 days.
