---
title: "7. Planning"
---

The controller takes the filter's last state and the posterior-mean parameters, and chooses weekly blocks of energy, protein and fat share by SLSQP, subject to the safety limits. This step asks it three questions: a reachable deadline, sustainable mode with no target, and a deadline that the 1 %-per-week limit cannot reach. Then it checks how the first plan would turn out for the true subject. [Step 8](08-maintain.md) asks the same controller to hold a weight instead. The theory is in [Control](../theory/control.md); the command-line equivalent is [`wdm plan`](../commands.md#wdm-plan).

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
from wdm.models import STATE_NAMES, NoiseModel, Params, State, inputs_from_episode, simulate, step
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
    State.from_vector(fit.episodes[-1].filtered.filtered_mean[-1][:5]),
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
    print(f"  feasibility: {feasibility_report(plan, c) or 'on track'}")
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
plan starts 2026-09-29 at 195.2 lb; last logged intake 2322 kcal

== deadline: 185 lb by 2026-12-31
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   2417  142   174  128    194.3   0.83      88.46    0.45
  2026-10-06   2458  142   192  125    193.7   0.69      88.08    0.44
  2026-10-13   2475  142   200  123    192.9   0.74      87.72    0.42
  2026-10-20   2482  142   203  122    192.1   0.77      87.35    0.41
  2026-10-27   2484  142   204  122    191.4   0.78      87.00    0.41
  2026-11-03   2484  142   205  122    190.6   0.79      86.64    0.41
  2026-11-10   2482  142   204  122    189.8   0.79      86.28    0.41
  2026-11-17   2479  142   204  122    189.0   0.79      85.93    0.41
  2026-11-24   2476  142   203  122    188.2   0.79      85.57    0.42
  2026-12-01   2471  142   202  122    187.4   0.79      85.21    0.42
  2026-12-08   2466  142   203  121    186.6   0.78      84.86    0.43
  2026-12-15   2464  142   207  119    185.9   0.76      84.50    0.52
  2026-12-22   2474  142   288   84    185.6   0.26      84.09    0.57
  2026-12-29   2501  142   390   42    185.9  -1.12      83.95    0.61
  converged True; constraints: all met
  feasibility: on track

== sustainable
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   1897  142   210   54    193.5   1.67      87.95    1.00
  2026-10-06   1884  142   209   53    191.6   1.93      87.08    1.00
  2026-10-13   1878  142   208   53    189.7   1.91      86.21    1.00
  2026-10-20   1876  142   208   53    187.8   1.89      85.35    1.00
  converged True; constraints: all met
  feasibility: on track

== deadline: 175 lb by 2026-11-15
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  max tissue %/wk
  2026-09-29   1891  142   207   55    193.5   1.71      87.95    1.00
  2026-10-06   1881  142   204   55    191.5   1.94      87.07    1.00
  2026-10-13   1876  142   203   55    189.6   1.92      86.21    1.00
  2026-10-20   1874  142   203   55    187.7   1.89      85.35    1.00
  2026-10-27   1874  142   203   55    185.9   1.87      84.50    1.00
  2026-11-03   1875  142   204   55    184.0   1.85      83.66    1.00
  2026-11-10   1876  142   204   55    182.7   1.83      83.07    1.00
  converged True; constraints: all met
  feasibility: target of 175 lb by 2026-11-15 is not reachable within your 1%/week limit; the earliest achievable date at that maximum rate is 2026-12-14

first plan's last day: predicted 185.9 lb; the true subject eating it 187.4 lb
```

## What to notice

- **Where the plan starts.** The last row (2026-09-28) has its food logged, so, as `wdm plan` does, the filtered morning state is advanced through that day's 2322 kcal and the plan starts the next morning, 2026-09-29, at 195.2 lb ([Which day the plan starts](../theory/control.md#plan-start)).
- **A reachable deadline asks for little.** 185 lb by 2026-12-31 is 10.2 lb in 93 days, 0.77 lb a week. The plan eats about 2417 to 2484 kcal and loses 0.69 to 0.79 lb a week through the middle blocks, using under half the rate limit (0.41 to 0.45 % of tissue a week against 1 %).
- **The first block's water nets out.** Its 174 g of carbohydrate, the plan's lowest, drains glycogen water, but the rollout assumes habitual sodium (4000 mg) from day 1, so the sodium water drained by the last days' below-habitual sodium comes back at the same time (step 8 prints both at the start: −0.18 kg glycogen, −0.12 kg sodium). The first block loses 0.83 lb on the scale, and its tissue alone falls from 88.83 kg (step 8) to 88.46, 0.82 lb. The same refill slows the first block of the other two plans (1.67 and 1.71 lb against about 1.9 after).
- **Protein sits on its floor in every block of every plan.** 142 g is 1.6 g/kg × 88.5 kg (195.2 lb), the minimum in `Safety`. The objective scores tissue mass and intake, not body composition, so nothing in it rewards protein's leaner loss (`partition_kappa`). A preference for more protein has to be stated as a bound, `Safety(min_protein_g_per_kg=...)`.
- **The tail of the deadline plan is an end effect, not advice.** In the last two blocks carbohydrate jumps to 288 and 390 g and fat drops to 84 and 42 g at nearly the same energy. Tissue keeps falling, to 83.95 kg, which is the target: 185 lb is 83.91 kg. But the scale rises from 185.6 to 185.9 lb, because carbohydrate above the habitual 260 g refills glycogen water (185.9 lb is 84.32 kg, so 0.37 kg above tissue). The cost tracks tissue, so water on the scale at the horizon's end is free, and energy parked in glycogen is energy that tissue did not get, which shows as faster tissue loss in those blocks (0.57 and 0.61 %/wk, against about 0.42 before). Only today's line is ever prescribed, and the horizon moves every day, but `plan.csv` will show this tail.
- **Sustainable mode rides the rate limit.** With no target, the default weights trade a linear reward on lost tissue against a quadratic cost on the deficit, and the optimum lands on the constraint: 1.00 % of tissue a week in every block, about 1.9 lb a week after the first, at 1876 to 1897 kcal.
- **An impossible deadline is reported, not attempted.** 175 lb by 2026-11-15 is 20.2 lb in 47 days, about 3.0 lb a week. The plan runs at the limit (1.00 %/wk in every block) and reaches only 182.7 lb by the end of its horizon. It satisfies every constraint; the feasibility report says separately that the target is out of reach and that the earliest date at the maximum rate is 2026-12-14.
- **Certainty equivalence has a cost.** The plan is optimal for the posterior-mean parameters. Eaten by the true subject from the true state, the first plan ends at 187.4 lb, not the predicted 185.9. Most of that 1.5 lb is the size of the maintenance error from step 4 (that fit left out the tape; this one uses it): the estimate was 45 kcal/day above the truth, and 45 × 93 days ≈ 4200 kcal is roughly 0.55 kg, 1.2 lb. Replanning every morning with a fresh weigh-in is what corrects this in use ([Certainty equivalence](../theory/control.md#certainty-equivalence)).

## The code

| What | Where |
|---|---|
| The solver | `control/mpc.py:303` (`solve`) |
| Decisions per block: energy, protein per kg of starting weight, fat share | `control/mpc.py:285` (`_decode`), protein at 297 |
| Bounds and the SLSQP call | `control/mpc.py:442` (bounds), `control/mpc.py:460–476` (`method="SLSQP"` at 464) |
| Trajectory constraints: daily tissue-loss rate, carbohydrate floor | `control/mpc.py:418` |
| Penalty fallback, used only if SLSQP fails | `control/mpc.py:483` |
| Deadline and sustainable objectives; the reference path | `control/objectives.py:236`, `control/objectives.py:295`, `control/objectives.py:124` |
| Terminal and tracking terms on tissue | `control/objectives.py:276–277` |
| Default cost weights | `control/objectives.py:59` (`CostWeights`) |
| The independent constraint check, 5 % tolerance on the rate | `control/mpc.py:625` (`check_constraints`; day 1 measured from the starting tissue at 656), `control/mpc.py:525` |
| The earliest reachable date | `control/mpc.py:578` (`feasibility_report`), formula at 613–614 |
| Plan start and horizon sizing, as the command does them | `cli.py:304` (`_plan_start`), `cli.py:328` (`_solve_plan`; horizon at 408–415), `cli.py:442` (`cmd_plan`) |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Certainty equivalence: plan with the posterior-mean parameters and state | `control/mpc.py:303` | Cheap; replanning daily corrects the error it causes |
| Decisions are held for 7-day blocks | `control/mpc.py:68`, `control/mpc.py:285` | Nobody eats to a different target every day |
| Objective and rate limit are on tissue, not scale weight | `control/objectives.py:153`, `control/objectives.py:236` | A scale-weight objective could be met by cutting carbohydrate alone |
| Safety: at least 1500 kcal and 1.6 g/kg protein; at most 1 % of tissue a week | `constants.py:281`, `constants.py:291`, `constants.py:288` | Conservative defaults; the subject file can change them |
| Carbohydrate at least 20 g; fat share 15 % to 75 % | `control/mpc.py:87–93` | A glycogen-exploit guard and a nutritional floor |
| The rollout assumes habitual sodium, no logged exercise, every day logged | `control/mpc.py:264–276` | The plan prescribes food only |
| The feasibility check assumes the water compartments end where they start | `control/mpc.py:587–588` | The target is stated in scale weight |

## Try this

- Plan with the truth: replace `fit.params` with `true` in `solve` and compare the deadline plan's energy with the one above. The difference is what a perfect metabolism estimate would change.
- Lower the rate limit with `Safety(max_loss_frac_per_week=0.0075)` and rerun sustainable mode: it should again sit on the (new) limit.
- State a diet style with `PlanSpec(..., target_carb_fraction=0.10)` and see where the energy goes; then add `weights=CostWeights(carb_aversion=1.0)` instead and compare ([Macro preferences](../theory/control.md#macros)).
- Close the loop by hand: each simulated morning, solve, feed day 0 to the true subject with `step`, add 0.45 kg of noise to a new reading, refit, and repeat. That is what `wdm daily` does with a real scale.
