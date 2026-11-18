---
title: "7. Planning"
---

The controller takes the filter's last state and the posterior-mean parameters, and chooses weekly blocks of energy and fat share by SLSQP, with protein held at 1.6 g/kg by default, subject to the safety limits. This step asks it three questions: a reachable deadline, sustainable mode with no target, and a deadline that the 1 %-per-week limit cannot reach. Then it checks how the first plan would turn out for the true subject. [Step 8](08-maintain.md) asks the same controller to hold a weight instead. The theory is in [Control](../theory/control.md); the command-line equivalent is [`wdm plan`](../commands.md#wdm-plan).

```bash
uv run python walkthrough/07_plan.py
```

<details><summary>The script</summary>

```python
"""Step 7: planning with the model-predictive controller.

Fit the episode (scale, food log and tape), take the last filtered state and the
posterior-mean parameters, and ask the SLSQP controller for weekly intake blocks:
once toward a reachable deadline, once in sustainable mode (no target), and once
toward a deadline that the 1 %-per-week tissue-loss limit cannot reach.
"""

import dataclasses
import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np

from wdm.config import Config, Goal, Preferences, Safety, Subject
from wdm.constants import LB_PER_KG
from wdm.control import PlanSpec, check_constraints, feasibility_report, solve
from wdm.estimation import fit_joint, identify_noise
from wdm.models import (
    N_STATES,
    STATE_NAMES,
    NoiseModel,
    Params,
    State,
    inputs_from_episode,
    simulate,
    step,
)
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
fit = fit_joint([ep], cfg, dataclasses.replace(NoiseModel.default(), obs_sd=jnp.asarray(ni.obs_sd)))

# As `wdm plan` does: the last morning's filtered state, advanced through that
# day's logged food, so the plan starts the next morning.
u_last = jax.tree_util.tree_map(lambda a: a[-1], inputs_from_episode(ep))
x0 = step(
    State.from_vector(fit.episodes[-1].filtered.filtered_mean[-1][:N_STATES]),
    u_last,
    fit.params,
    ep.carb_ref_g,
)
start = ep.dates[-1] + np.timedelta64(1, "D")
prev_kcal = float(ep.energy_kcal[np.flatnonzero(ep.intake_known)[-1]])
print(
    f"plan starts {start} at {float(x0.scale_kg) * LB_PER_KG:.1f} lb; last logged intake"
    f" {prev_kcal:.0f} kcal"
)


def run(mode, goal=None):
    c = dataclasses.replace(cfg, goal=goal)
    left = int((np.datetime64(goal.target_date) - start) / np.timedelta64(1, "D")) if goal else None
    # As `wdm plan`: a deadline plan runs to the goal date, a sustainable one 28 days.
    ps = PlanSpec(mode=mode, horizon_days=max(7, min(left, 365)) if goal else 28, block_days=7)
    plan = solve(
        x0,
        fit.params,
        c,
        ps,
        carb_ref_g=ep.carb_ref_g,
        start_date=start,
        previous_kcal=prev_kcal,
        days_remaining=left,
    )
    print(
        f"\n== {mode}" + (f": {goal.target_weight_lb:.0f} lb by {goal.target_date}" if goal else "")
    )
    print("  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk")
    tis = plan.tissue_loss_frac_per_week()
    for b in range(0, len(plan.kcal), 7):
        e = min(b + 6, len(plan.kcal) - 1)
        lbs = plan.predicted_weight_lb
        rate = (lbs[b - 1] if b else float(x0.scale_kg) * LB_PER_KG) - lbs[e]
        print(
            f"  {plan.dates[b]}  {plan.kcal[b]:5.0f} {plan.protein_g[b]:4.0f} {plan.carb_g[b]:5.0f}"
            f" {plan.fat_g[b]:4.0f}  {lbs[e]:7.1f}  {rate * 7 / (e - b + 1):5.2f}"
            f"     {plan.predicted_tissue_kg[e]:6.2f}  {tis[b : e + 1].max():6.2f}"
        )
    print(
        f"  converged {plan.converged}; constraints: {check_constraints(plan, c, ps) or 'all met'}"
    )
    note = feasibility_report(plan, c, params=fit.params, carb_ref_g=carb_ref)
    print(f"  feasibility: {note or 'on track'}")
    return plan


plan = run("deadline", Goal(185.0, dt.date(2026, 12, 31)))
run("sustainable")
run("deadline", Goal(175.0, dt.date(2026, 11, 15)))

# Certainty equivalence: the plan is optimal for the posterior mean. The same
# intake, eaten by the true subject from the true state, ends somewhere else.
x_true_last = State.from_vector(jnp.asarray([truth[k][-1] for k in STATE_NAMES]))
x_true = step(x_true_last, u_last, true, carb_ref)
u_plan = jax.tree_util.tree_map(lambda a: jnp.full(len(plan.kcal), a), u_last)
u_plan = dataclasses.replace(
    u_plan,
    energy_kcal=jnp.asarray(plan.kcal),
    protein_g=jnp.asarray(plan.protein_g),
    fat_g=jnp.asarray(plan.fat_g),
    carb_dev_g=jnp.asarray(plan.carb_g - carb_ref),
    sodium_dev_mg=jnp.zeros(len(plan.kcal)),
)
end_true = float(simulate(x_true, u_plan, true, carb_ref).scale_kg[-1]) * LB_PER_KG
print(
    f"\nfirst plan's last day: predicted {plan.predicted_weight_lb[-1]:.1f} lb;"
    f" the true subject eating it {end_true:.1f} lb"
)
```

</details>

```text
plan starts 2026-09-29 at 194.1 lb; last logged intake 2322 kcal

== deadline: 185 lb by 2026-12-31
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   2404  141   174  127    193.3   0.73      86.79    0.44
  2026-10-06   2435  141   188  124    192.8   0.57      86.42    0.42
  2026-10-13   2445  141   194  123    192.1   0.70      86.06    0.42
  2026-10-20   2446  141   196  122    191.3   0.76      85.70    0.42
  2026-10-27   2444  141   196  122    190.5   0.79      85.34    0.42
  2026-11-03   2440  141   196  121    189.7   0.81      84.98    0.43
  2026-11-10   2435  141   195  121    188.9   0.82      84.62    0.43
  2026-11-17   2429  141   195  121    188.1   0.82      84.25    0.44
  2026-11-24   2421  141   193  121    187.2   0.84      83.89    0.44
  2026-12-01   2413  141   192  120    186.4   0.85      83.51    0.45
  2026-12-08   2401  141   190  120    185.5   0.87      83.14    0.46
  2026-12-15   2387  141   189  119    184.6   0.89      82.75    0.47
  2026-12-22   2372  141   190  117    183.8   0.88      82.35    0.48
  2026-12-29   2365  141   192  115    183.5   0.82      82.24    0.49
  converged True; constraints: all met
  feasibility: on track

== sustainable
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   1887  141   213   52    192.1   1.95      86.27    1.00
  2026-10-06   1856  141   126   88    189.1   3.01      85.48    1.00
  2026-10-13   1849  141   119   90    187.1   1.97      84.63    1.00
  2026-10-20   1844  141   113   92    185.2   1.93      83.80    1.00
  converged True; constraints: all met
  feasibility: on track

== deadline: 175 lb by 2026-11-15
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   1883  141   197   59    191.9   2.15      86.28    1.00
  2026-10-06   1858  141   133   85    189.2   2.72      85.47    1.00
  2026-10-13   1840  141    81  106    186.7   2.52      84.66    1.00
  2026-10-20   1823  141    20  131    184.1   2.61      83.86    1.00
  2026-10-27   1825  141    29  127    182.3   1.73      83.02    1.00
  2026-11-03   1844  141   114   92    181.6   0.77      82.14    1.00
  2026-11-10   1863  141   192   59    181.2   0.48      81.51    1.00
  converged True; constraints: all met
  feasibility: target of 175 lb by 2026-11-15 is not reachable within your 1%/week limit; the earliest achievable date at that maximum rate is 2026-12-18

first plan's last day: predicted 183.5 lb; the true subject eating it 184.3 lb
```

## What to notice

- **Where the plan starts.** The last row (2026-09-28) has its food logged, so the plan starts the next morning, 2026-09-29, at 194.1 lb ([Which day the plan starts](../theory/control.md#plan-start)).
- **A reachable deadline asks for little.** 185 lb by 2026-12-31 is 0.69 lb a week. The plan eats about 2365 to 2446 kcal and uses under half of the rate limit (0.42 to 0.49 % of tissue a week against 1 %). This script calls `solve` without the recent exercise, sodium and steps that `wdm plan` passes, so its rollout assumes no exercise and habitual sodium from day 1.
- **The target is tissue, not the scale reading.** The plan aims at the settled tissue that reads 185 lb (83.91 kg) on the scale once the subject eats at maintenance: 83.91 − 0.0006 × (1344 + 15.34 × 83.91) / 0.90 = 82.16 kg, the goal less the gut content of weight-stable eating ([The tissue target](../theory/control.md#target)). It ends at 82.19 kg of settled tissue. The predicted scale then reads 183.5 lb, 1.5 lb below the goal, because mid-diet the gut holds the content of about 2365 kcal rather than 2923 and the glycogen and sodium water are still drained; both refill when intake returns to maintenance.
- **Protein is held, not chosen.** 141 g in every block is 1.6 g/kg × 88.0 kg (194.1 lb), `PlanSpec`'s default `protein_g_per_kg`. An objective that scores tissue mass would push protein to its floor, since more protein keeps lean tissue and so slows the loss of mass ([Why protein is held fixed](../theory/control.md#protein)); state another level with `PlanSpec(..., protein_g_per_kg=2.0)`, or 0 to let the optimizer choose.
- **Sustainable mode rides the rate limit.** With no target, a linear reward on lost tissue against a quadratic cost on the deficit lands on the constraint: 1.00 % of tissue a week in every block, at 1844 to 1887 kcal. The second block's 3.01 lb a week, against about 1.95 around it, comes with carbohydrate falling from 213 to 126 g: water, since tissue loss stays at the limit.
- **An impossible deadline is reported, not attempted.** 175 lb by 2026-11-15 needs about 2.8 lb a week. The plan runs at the limit, reaches 181.2 lb, and meets every constraint; the feasibility report says separately that the target is out of reach and that the earliest date at the maximum rate is 2026-12-18, measured from today's settled tissue (87.05 kg) to the tissue that reads 175 lb at maintenance (77.67 kg).
- **Certainty equivalence has a cost.** Eaten by the true subject, the first plan ends at 184.3 lb, not the predicted 183.5. Most of that is the maintenance error from step 4: 22 kcal/day × 93 days ≈ 2050 kcal, about 0.3 kg or 0.6 lb at roughly 7000 kcal per kg of tissue. Replanning every morning with a fresh weigh-in corrects this in use ([Certainty equivalence](../theory/control.md#certainty-equivalence)).

## The code

| What | Where (under `wdm`) |
|---|---|
| The solver: 7-day blocks, bounds, SLSQP, trajectory constraints | `control.mpc`: `solve`, `PlanSpec`, `Plan` |
| Deadline and sustainable objectives; the reference path; default weights | `control.objectives`: `deadline_cost`, `sustainable_cost`, `reference_trajectory`, `CostWeights` |
| The tissue target a goal weight sets | `control.mpc`: `tissue_at_scale_weight`, `Plan.target_tissue_kg` |
| The independent constraint check and the earliest reachable date, at the rate limit and at the calorie floor | `control.mpc`: `check_constraints`, `feasibility_report`, `earliest_at_floor` |
| Safety limits | `config.Safety` |
| Plan start and horizon, as `wdm plan` does them | `cli` |

## Try this

- Plan with the truth: replace `fit.params` with `true` in `solve` and compare the deadline plan's energy with the one above. The difference is what a perfect metabolism estimate would change.
- Lower the rate limit with `Safety(max_loss_frac_per_week=0.0075)` and rerun sustainable mode: it should again sit on the (new) limit.
- State a diet style with `PlanSpec(..., target_carb_fraction=0.10)` and see where the energy goes; then add `weights=CostWeights(carb_aversion=1.0)` instead and compare ([Macro preferences](../theory/control.md#macros)).
- Close the loop by hand: each simulated morning, solve, feed day 0 to the true subject with `step`, add 0.45 kg of noise to a new reading, refit, and repeat. That is what `wdm daily` does with a real scale.
