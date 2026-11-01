---
title: "4. Phase 2: state and metabolism"
---

Phase 2 appends the nine metabolic parameters to the five physiological states and runs an unscented Kalman filter and smoother over the scale and the food log. Each weigh-in updates both where the subject is and what kind of metabolism produced the path so far. This step leaves the tape out (step 5 adds it) and scores the result against the truth from step 2. The theory is in [Estimation: phase 2](../theory/estimation.md#phase-2); the command-line equivalent is [`wdm filter`](../commands.md#wdm-filter).

```bash
uv run python walkthrough/04_phase2.py
```

<details><summary>The script</summary>

```python
"""Step 4: phase 2, joint state and parameter estimation.

Run the augmented unscented Kalman filter and smoother on the scale and the food
log (no tape yet: step 5 adds it). Then score the result against the truth the
generator kept: is each true parameter inside its 90 % interval, how far did
each median move from the prior, and how close are the hidden states?
"""

import dataclasses

import jax.numpy as jnp
import numpy as np

from wdm.config import Config, Preferences, Safety, Subject
from wdm.constants import LB_PER_KG
from wdm.estimation import fit_joint, identify_noise
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
ep = dataclasses.replace(ep, waist_cm=None)  # scale and food log only

# Phase 1's reading noise replaces the default; the process noise stays default.
ni = identify_noise([ep])
noise = dataclasses.replace(NoiseModel.default(), obs_sd=jnp.asarray(ni.obs_sd))
fit = fit_joint([ep], cfg, noise)
print(fit.render_parameters())
print()

print("  parameter            truth   in 90%?   truth vs prior")
for s in fit.summaries:
    t = float(getattr(true, s.name))
    inside = "yes" if s.q05 <= t <= s.q95 else "NO"
    print(f"  {s.name:16s} {t:10.4g}   {inside:>5s}     {t / s.prior_center - 1:+6.1%}")
print()

# The intercept and slope trade off; their sum at the current weight is what the
# data pin down. The truth's value is computed the same way.
last = fit.episodes[-1]
w_lb = float(last.scale_trend_kg()[-1] * LB_PER_KG)
lo, med, hi = fit.maintenance_interval_kcal(w_lb)
w_kg = w_lb / LB_PER_KG
true_maint = float(true.rmr_intercept + true.rmr_slope * w_kg)
prior_maint = c0 + c1 * w_kg
print(
    f"maintenance at {w_lb:.1f} lb: {med:.0f} kcal/day, 90% [{lo:.0f}, {hi:.0f}];"
    f" truth {true_maint:.0f}, prior {prior_maint:.0f}"
)
print()

# Hidden states on the last day: smoothed mean +- sd against the truth.
sd = last.state_sd()[-1]
for name, j, est in [
    ("tissue_kg", 0, last.tissue_kg()),
    ("fat_kg", 1, last.fat_kg()),
    ("adapt_kcal", 4, last.adaptation_kcal()),
]:
    print(f"  {name:10s} day 119: {est[-1]:8.2f} +- {sd[j]:6.2f}   truth {truth[name][-1]:8.2f}")
err = last.scale_trend_kg() - truth["scale_kg"]
print(f"smoothed scale weight vs truth, all 120 days: RMSE {np.sqrt(np.mean(err**2)):.3f} kg")
```

</details>

```text
Phase 2 -- posterior parameter estimates

  parameter              median                90% interval  shift
  rmr_intercept          1353.1                [1081, 1691]   +3.8%
  rmr_slope              15.405              [12.22, 19.41]   +6.2%
  at_gain                0.1532           [0.05403, 0.3637]   +2.1%
  at_tau                 12.631              [5.513, 28.91]   -9.8%
  gly_gain            0.0035331        [0.001965, 0.006445]  +17.8%
  gly_tau                1.2272             [0.6757, 2.236]  -18.2%
  na_gain            7.2163e-05      [2.125e-05, 0.0002437]  -63.9%
  na_tau                 1.3121             [0.6846, 2.506]  +31.2%
  partition_kappa      0.049742          [-0.03239, 0.1347]   -0.5%

  'shift' is movement of the posterior median away from the prior center. Values near zero mean the data was uninformative about that parameter.

  parameter            truth   in 90%?   truth vs prior
  rmr_intercept          1538     yes     +18.0%
  rmr_slope             13.05     yes     -10.0%
  at_gain                0.22     yes     +46.7%
  at_tau                   11     yes     -21.4%
  gly_gain             0.0042     yes     +40.0%
  gly_tau                   2     yes     +33.3%
  na_gain             0.00025      NO     +25.0%
  na_tau                    1     yes      +0.0%
  partition_kappa        0.06     yes     +20.0%

maintenance at 195.3 lb: 2739 kcal/day, 90% [2547, 2953]; truth 2694, prior 2588

  tissue_kg  day 119:    88.91 +-   0.24   truth    89.20
  fat_kg     day 119:    22.53 +-   3.79   truth    22.43
  adapt_kcal day 119:  -136.08 +-  79.15   truth  -161.41
smoothed scale weight vs truth, all 120 days: RMSE 0.208 kg
```

## What to notice

- **Eight of the nine true values are inside their 90 % intervals.** That is the minimum an honest estimator must do on data from its own model, and not the same as learning all nine; one miss in nine at 90 % is within chance.
- **The sodium gain is the miss.** The median, 7.2·10⁻⁵ kg per mg/day, moved −63.9 % from the prior center of 2.0·10⁻⁴; the truth, 2.5·10⁻⁴, sits just above the interval. This is weak information, not bias: ±800 mg of random sodium moves the scale by about 0.1 kg, so the likelihood is nearly flat in the gain and one seed's data can favor a low value. With two or three times the sodium variation the estimates converge on the truth ([Identifiability](../theory/estimation.md#identifiability)). It matters because a 1000 mg change is worth 0.25 kg at the true gain, so an estimate 3.5 times low leaves much of the sodium swing for the filter to explain some other way.
- **What "shift" means.** (Posterior median − prior center) / |prior center|. Near zero, the line is reading back the prior: `partition_kappa` shifts −0.5 %, since 120 days at one protein level cannot reveal a protein effect. Its interval is even slightly wider than the prior's, because the parameter random walk (0.02 prior standard deviations a day, so the filter keeps listening) adds variance the data never removed.
- **The two expenditure coefficients are not learned separately.** The intercept moved +3.8 % toward a truth at +18 %; the slope moved +6.2 % against a truth at −10 %. Over the 11.8 kg this episode spans, raising one and lowering the other leaves the trajectory almost unchanged.
- **Their combination is learned.** Maintenance at 195.3 lb is 2739 kcal/day, 90 % [2547, 2953], against a truth of 2694 and a prior of 2588. This number, not either coefficient, is what the controller relies on.
- **The glycogen gain is partly learned**, +17.8 % toward a truth at +40 %, helped by the weekly carbohydrate cycle.
- **States.** Tissue on day 119 is 88.91 ± 0.24 kg against a true 89.20. Fat is 22.53 ± 3.79 kg against 22.43, but the 3.79 kg is nearly the 4 kg prior it started with: the scale taught the filter little about fat, and the estimate is right only because the configured body fat matches the generator's. Step 5 shows what happens when it does not.
- **The smoothed scale weight** is within 0.208 kg of the truth over all 120 days, against step 3's 0.290 kg causal trend: the smoother has the food log and both directions in time.

## The code

| What | Where (under `wdm`) |
|---|---|
| The fit, episode by episode; summaries, shift, maintenance | `estimation.joint`: `fit_joint`, `ParameterSummary.shrinkage`, `JointFit.maintenance_interval_kcal` |
| Unconstrained coordinates and prior scales | `estimation.reparam`: `Transform`, `ParamTransform` |
| Extra process noise on unlogged and estimated days | `models.dynamics.process_noise`, `constants.ESTIMATED_INTAKE_FRAC_SD` |
| Unscented filter and RTS smoother | `filters.ukf`: `unscented_filter`, `unscented_rts_smoother` |
| The gate on a surprising reading | `models.observation.innovation_gate` |

## Try this

- Widen the prior with `fit_joint([ep], cfg, noise, prior_scale=2.0)`. Informed parameters move further; uninformed ones only widen. Does the sodium gain's interval reach the truth?
- Under-log by 10 % (`dataclasses.replace(ep, energy_kcal=0.9 * ep.energy_kcal)`) and compare the maintenance line. The fit explains a short log by a lower maintenance: it describes the log, not the body.
- Mark days as estimated, as `wdm log --estimate` does for a restaurant meal: `dataclasses.replace(ep, intake_sd_kcal=0.25 * np.nan_to_num(ep.energy_kcal))` treats every logged day as known only to ±25 %. Each day then teaches the filter less about the metabolism, and the maintenance interval widens, from [2547, 2953] to [2522, 2973].
- Print `fit.episodes[0].adaptation_kcal()` every 10 days next to `truth["adapt_kcal"]`, and see how long the filter takes to believe in adaptation.
