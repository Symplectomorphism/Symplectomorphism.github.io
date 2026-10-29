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

- **Eight of the nine true values are inside their 90 % intervals.** That is the minimum an honest estimator must do on data from its own model, and it is not the same as learning all nine. At 90 %, one miss in nine is within what chance allows; the next bullet is about the miss.
- **The sodium gain is the miss, and it now matters.** The median, 7.2·10⁻⁵ kg per mg/day, moved −63.9 % from the prior center of 2.0·10⁻⁴, while the truth, 2.5·10⁻⁴, sits 25 % above it; the interval [2.1·10⁻⁵, 2.4·10⁻⁴] stops just short of the truth. The direction is the warning: the data pulled the gain away from the truth, not toward it. With the earlier prior (1.2·10⁻⁵) the sodium gain could not move the scale enough to matter; at the true value a 1000 mg change in daily sodium is worth 0.25 kg at equilibrium (step 2), so an estimate a factor of 3.5 low leaves much of the sodium swing for the filter to explain some other way. Its time constant moved +31.2 % (truth +0 %) and still covers the true 1 day: [0.68, 2.51].
- **What "shift" means.** Shift is (posterior median − prior center) / |prior center|. Near zero, the data said nothing and the line is reading back the prior. `partition_kappa` shifts −0.5 %: 120 days at one protein level cannot reveal a protein effect. Its interval, [−0.032, 0.135], is even slightly wider than the prior's 0.05 ± 1.645 × 0.05 = [−0.032, 0.132], because the parameter random walk (0.02 prior standard deviations a day) adds variance that the data never removed.
- **The two expenditure coefficients are not learned separately.** The intercept moved +3.8 % toward a truth at +18 %; the slope moved +6.2 % while the truth is −10 %, the wrong direction. Over the 11.8 kg this episode spans (step 2), raising one and lowering the other leaves the trajectory almost unchanged, so the data constrain only their combination.
- **Their combination is learned.** Maintenance at the current 195.3 lb is 2739 kcal/day, 90 % interval [2547, 2953]. The truth is 2694 and the prior 2588: the estimate moved 151 kcal/day from the prior and ended 45 kcal/day past the truth, well inside its interval. This number, not either coefficient, is what the controller relies on ([Identifiability](../theory/estimation.md#identifiability)).
- **The glycogen gain is partly learned.** It moved +17.8 % toward a truth at +40 %, helped by the weekly carbohydrate cycle. The glycogen time constant moved −18.2 % while the truth is +33.3 %; it stays inside its interval, [0.68, 2.24] days, which contains the true 2.
- **States.** Tissue on day 119 is 88.91 ± 0.24 kg against a true 89.20 (1.2 standard deviations off). Fat is 22.53 ± 3.79 kg against 22.43: close, but the 3.79 kg is nearly the 4 kg it started with, so the scale taught the filter little about fat. The estimate is right because the configured body fat (30 %) is the generator's; step 5 shows what happens when it is not. Adaptation is −136 ± 79 kcal/day against −161.
- **The smoothed scale weight** is within 0.208 kg of the truth over all 120 days (root mean square). Step 3's causal trend, scored on days 30 to 119 only, was 0.290 kg; the two are not the same measure, but the smoother has the food log and both directions in time.

## The code

| What | Where |
|---|---|
| The fit, episode by episode | `estimation/joint.py:383` (`fit_joint`) |
| The augmented transition and process noise | `estimation/joint.py:318`, `estimation/joint.py:330` |
| Parameter random walk, 0.02 per day | `estimation/joint.py:91` |
| Initial state belief: fat sd 4 kg, water at zero | `estimation/joint.py:299` (covariance at 314) |
| Unconstrained coordinates and prior scales | `estimation/reparam.py:30` (`Transform`), `estimation/reparam.py:75` (`ParamTransform`; scales at 88–100, the sodium gain at 96 from `constants.py:215` and `:231`) |
| Extra process noise on unlogged days and on days marked estimated | `models/dynamics.py:231` (`process_noise`), the intake variance at 264; `io.py:248–250`, `constants.py:96` (`ESTIMATED_INTAKE_FRAC_SD`) |
| Unscented filter and RTS smoother | `filters/ukf.py:123`, `filters/ukf.py:265`; sigma-point spread `alpha = 0.3` at 60 |
| The 4-sigma gate on a reading | `filters/ukf.py:221`, `models/observation.py:59` |
| Monte Carlo summaries and shift | `estimation/joint.py:537` (`_summarise`), `estimation/joint.py:147` (`shrinkage`), `estimation/joint.py:263` (`render_parameters`) |
| Maintenance and its interval | `estimation/joint.py:217` (`maintenance_interval_kcal`) |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Parameters drift as a slow random walk, 0.02 prior standard deviations a day | `estimation/joint.py:91` | Without it the parameter covariance only shrinks and the filter stops listening |
| Gaussian belief in unconstrained coordinates; log, logit or affine maps to natural units | `estimation/reparam.py:75` | A sigma point can never be a negative time constant |
| Fat mass starts at the configured body fat with a 4 kg standard deviation | `estimation/joint.py:281`, `estimation/joint.py:314` | About the standard error of a population body-fat formula |
| Water deviations start at zero | `estimation/joint.py:306` | The episode starts from habitual eating |
| Reading noise from phase 1; process noise at the package defaults | `models/dynamics.py:75` | Only the reading noise has data behind it |
| Gaussian readings with a 4-sigma gate | `filters/ukf.py:135` (`gate_sigma`), `filters/ukf.py:221` | Student-t fit worse on the real record; the gate handles a mistyped weight |
| An estimated day is known to ±25 % of its logged energy, as process noise along the tissue/fat direction | `constants.py:96`, `models/dynamics.py:264` | Restaurant and guessed meals miss by 100 kcal or more; a quarter of the day is the honest order of magnitude |
| Thermic-effect fractions are not estimated | `models/types.py:109` | Confounded with maintenance |

## Try this

- Widen the prior with `fit_joint([ep], cfg, noise, prior_scale=2.0)`. Parameters the data inform should move further; uninformed ones only widen. Watch whether the sodium gain's interval reaches the truth.
- Under-log by 10 % (`dataclasses.replace(ep, energy_kcal=0.9 * ep.energy_kcal)`) and compare the maintenance line. The fit explains a short log by a lower maintenance: it describes the log, not the body.
- Mark days as estimated, as `wdm log --estimate` does for a restaurant meal: `dataclasses.replace(ep, intake_sd_kcal=0.25 * np.nan_to_num(ep.energy_kcal))` treats every logged day as known only to ±25 %. Each day then teaches the filter less about the metabolism, and the maintenance interval widens, from [2547, 2953] to [2522, 2973].
- Print `fit.episodes[0].adaptation_kcal()` every 10 days next to `truth["adapt_kcal"]`, and see how long the filter takes to believe in adaptation.
