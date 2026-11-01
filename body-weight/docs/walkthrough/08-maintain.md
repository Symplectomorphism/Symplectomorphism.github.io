---
title: "8. Maintain"
---

Once a goal is reached, the question changes from "how fast" to "what intake holds this weight". Maintain mode answers it with the deadline objective and no deadline: the reference eases tissue from where it is today to the held weight over the 28-day horizon. This step asks the controller to hold today's tissue (the default, with no held weight set) and then a weight 3 lb below it, from the same fitted state as step 7, and compares its intake with the fitted maintenance. The theory is in [Maintain mode](../theory/control.md#maintain); the command-line equivalent is [`wdm plan --mode maintain`](../commands.md#wdm-plan), or a [`[maintain]` section](../configuration.md#maintain) in the subject file.

```bash
uv run python walkthrough/08_maintain.py
```

<details><summary>The script</summary>

```python
"""Step 8: maintain mode.

Once a goal is reached, the question changes from "how fast" to "how much keeps
me here". Maintain mode is the deadline objective with no deadline: its
reference eases tissue from where it is today to the held weight over the 28-day
horizon. Ask it, from the same fitted state as step 7, to hold today's tissue
(the default, with no held weight set) and then a weight 3 lb below it, and
compare its intake with the fitted maintenance.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from wdm.config import Config, Maintain, Preferences, Safety, Subject
from wdm.constants import LB_PER_KG
from wdm.control import PlanSpec, check_constraints, feasibility_report, solve
from wdm.estimation import fit_joint, identify_noise
from wdm.models import NoiseModel, Params, State, inputs_from_episode, step
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

# The plan starts where step 7's does: the last filtered state, advanced a day.
u_last = jax.tree_util.tree_map(lambda a: a[-1], inputs_from_episode(ep))
x0 = step(
    State.from_vector(fit.episodes[-1].filtered.filtered_mean[-1][:5]),
    u_last,
    fit.params,
    ep.carb_ref_g,
)
start = ep.dates[-1] + np.timedelta64(1, "D")
prev_kcal = float(ep.energy_kcal[np.flatnonzero(ep.intake_known)[-1]])
w_lb = float(x0.scale_kg) * LB_PER_KG
lo, med, hi = fit.maintenance_interval_kcal(w_lb)
print(f"plan starts {start} at {w_lb:.1f} lb; last logged intake {prev_kcal:.0f} kcal")
print(f"fitted maintenance at {w_lb:.1f} lb: {med:.0f} kcal/day, 90% [{lo:.0f}, {hi:.0f}]")
print(
    f"start: tissue {float(x0.tissue_kg):.2f} kg; water: glycogen {float(x0.gly_water_kg):+.2f} kg,"
    f" sodium {float(x0.na_water_kg):+.2f} kg; adaptation {float(x0.adapt_kcal):.0f} kcal/day"
)


def hold(weight_lb=None):
    held = Maintain(weight_lb=weight_lb) if weight_lb is not None else None
    c = dataclasses.replace(cfg, maintain=held)
    ps = PlanSpec(mode="maintain", horizon_days=28, block_days=7)  # as `wdm plan`
    plan = solve(
        x0, fit.params, c, ps, carb_ref_g=ep.carb_ref_g, start_date=start, previous_kcal=prev_kcal
    )
    print(f"\n== maintain {f'{weight_lb:.1f} lb' if held else 'today (no held weight)'}")
    print("  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  adapt")
    lbs = plan.predicted_weight_lb
    for b in range(0, len(plan.kcal), 7):
        e = b + 6
        rate = (lbs[b - 1] if b else w_lb) - lbs[e]
        print(
            f"  {plan.dates[b]}  {plan.kcal[b]:5.0f} {plan.protein_g[b]:4.0f} {plan.carb_g[b]:5.0f}"
            f" {plan.fat_g[b]:4.0f}  {lbs[e]:7.1f}  {rate:5.2f}"
            f"     {plan.predicted_tissue_kg[e]:6.2f}  {plan.predicted_adapt_kcal[e]:5.0f}"
        )
    print(
        f"  converged {plan.converged}; constraints: {check_constraints(plan, c, ps) or 'all met'};"
        f" feasibility report: {feasibility_report(plan, c) or 'none'}"
    )
    return plan


hold()  # no [maintain] weight: hold today's tissue
hold(round(float(x0.tissue_kg) * LB_PER_KG - 3.0, 1))
```

</details>

```text
plan starts 2026-09-29 at 195.2 lb; last logged intake 2322 kcal
fitted maintenance at 195.2 lb: 2740 kcal/day, 90% [2549, 2952]
start: tissue 88.83 kg; water: glycogen -0.18 kg, sodium -0.12 kg; adaptation -135 kcal/day

== maintain today (no held weight)
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  adapt
  2026-09-29   2717  142   108  191    194.5   0.63      88.78    -92
  2026-10-06   2885  142   188  174    195.1  -0.55      88.75    -57
  2026-10-13   2954  142   228  164    195.4  -0.35      88.77    -33
  2026-10-20   2979  162   180  179    195.2   0.25      88.83    -19
  converged True; constraints: all met; feasibility report: none

== maintain 192.8 lb
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  adapt
  2026-09-29   2469  142   163  139    194.4   0.80      88.51   -109
  2026-10-06   2532  142   196  131    193.9   0.45      88.19    -90
  2026-10-13   2564  142   220  124    193.5   0.48      87.89    -77
  2026-10-20   2598  142   411   43    194.0  -0.59      87.48    -70
  converged True; constraints: all met; feasibility report: none
```

## What to notice

- **The starting point is step 7's**: 195.2 lb on the scale, tissue 88.83 kg, both water compartments below habitual, adaptation −135 kcal/day after four months of deficit. The fitted maintenance line, 2740 kcal/day, is c0 + c1 W alone, without adaptation or the thermic effect of food.
- **The held weight is tissue, not today's scale reading**: 88.83 kg is 195.8 lb, 0.30 kg above the scale, the drained water that refills on ordinary eating. Why the plan holds tissue is in [Maintain mode](../theory/control.md#maintain).
- **Holding today's tissue means eating near maintenance.** The plan eats 2717 to 2979 kcal and tissue stays within 0.08 kg of where it started. Intake rises because adaptation fades once the deficit ends (−92 to −19 kcal/day). By the last block expenditure is roughly 2740 − 19 plus about 248 kcal/day of thermic effect, about 2970, against 2979 eaten.
- **The scale ends below the tissue it holds**, at 195.2 lb rather than 195.8, because carbohydrate stays below the habitual 260 g and glycogen water stays partly drained.
- **A drift is corrected gently.** Holding 192.8 lb, 3 lb below today's tissue, the reference covers the gap in 28 days: tissue falls about 0.7 lb a week at 2469 to 2564 kcal and ends at 87.48 kg against the 87.45 held, nowhere near the calorie floor or the rate limit.
- **The horizon's end.** Both plans change their mix in the last block (the second jumps to 411 g of carbohydrate), the same end effect as step 7: the cost tracks tissue, so the mix in the last days is nearly free.
- **No feasibility report.** Maintain mode has no date, so the check that flagged step 7's impossible deadline does not run. Replanning each morning keeps easing toward the held weight, and the 1 %-per-week limit still caps any correction.

## The code

| What | Where (under `wdm`) |
|---|---|
| The maintain reference (today's tissue to the held weight over the horizon) and the shared deadline objective | `control.mpc.solve`, `control.objectives`: `reference_trajectory`, `deadline_cost` |
| The planning mode and the held weight | `control.mpc.PlanSpec.mode`, `config.Maintain` |
| Fitted maintenance | `estimation.joint.JointFit.maintenance_interval_kcal` |
| Adaptation relaxing as the gap closes | `models.energetics.adaptation_rate` |

## Try this

- Hold today's scale reading explicitly with `hold(round(w_lb, 1))` (195.2 lb): the plan then has about 0.3 kg of tissue to lose, the drained water's worth.
- Hold 3 lb above today's tissue instead. The plan eats above maintenance, and the rate limit, which only caps loss, plays no part.
- Give the 3-lb plan twice the horizon, `PlanSpec(mode="maintain", horizon_days=56, block_days=7)`: the same correction spread over eight weeks, at about half the weekly pace.
- Feed the first plan to the true subject, as step 7 does with `simulate`, and see how far the posterior-mean metabolism lets the true weight drift in four weeks. Replanning every morning is what keeps that drift small in use.
