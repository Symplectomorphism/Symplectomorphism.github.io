---
title: "8. Maintain"
---

Once a goal is reached, the question changes from "how fast" to "what intake holds this weight". Maintain mode answers it with the deadline objective and no deadline: the reference eases settled tissue from where it is today to the tissue that reads the held weight on the scale at maintenance, over the 28-day horizon. This step asks the controller to hold today's tissue (the default, with no held weight set) and then a held weight 3 lb below today's tissue in pounds, from the same fitted state as step 7, and compares its intake with the fitted maintenance. The theory is in [Maintain mode](../theory/control.md#maintain); the command-line equivalent is [`wdm plan --mode maintain`](../commands.md#wdm-plan), or a [`[maintain]` section](../configuration.md#maintain) in the subject file.

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
from wdm.models import N_STATES, NoiseModel, Params, State, inputs_from_episode, step
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
    State.from_vector(fit.episodes[-1].filtered.filtered_mean[-1][:N_STATES]),
    u_last,
    fit.params,
    ep.carb_ref_g,
)
start = ep.dates[-1] + np.timedelta64(1, "D")
prev_kcal = float(ep.energy_kcal[np.flatnonzero(ep.intake_known)[-1]])
w_lb = float(x0.scale_kg) * LB_PER_KG
tissue_lb = float(x0.tissue_kg) * LB_PER_KG
lo, med, hi = fit.maintenance_interval_kcal(tissue_lb)  # expenditure scales with tissue
print(f"plan starts {start} at {w_lb:.1f} lb; last logged intake {prev_kcal:.0f} kcal")
print(
    f"fitted maintenance at {tissue_lb:.1f} lb of tissue: {med:.0f} kcal/day,"
    f" 90% [{lo:.0f}, {hi:.0f}]"
)
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
plan starts 2026-09-29 at 194.1 lb; last logged intake 2322 kcal
fitted maintenance at 192.1 lb of tissue: 2699 kcal/day, 90% [2513, 2908]
start: tissue 87.13 kg; water: glycogen -0.20 kg, sodium -0.24 kg; adaptation -125 kcal/day

== maintain today (no held weight)
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  adapt
  2026-09-29   2698  141   136  177    193.8   0.22      87.09    -85
  2026-10-06   2854  141   207  163    195.0  -1.12      87.06    -54
  2026-10-13   2917  141   240  155    195.5  -0.58      87.07    -32
  2026-10-20   2940  141   252  152    195.8  -0.27      87.11    -17
  converged True; constraints: all met; feasibility report: none

== maintain 189.1 lb
  block from    kcal  pro  carb  fat   end lb  lb/wk  tissue kg  adapt
  2026-09-29   2094  141   183   89    192.4   1.63      86.49   -123
  2026-10-06   1984  141   148   92    190.2   2.22      85.76   -128
  2026-10-13   1921  141   133   92    188.2   2.02      84.98   -133
  2026-10-20   1883  141   130   89    186.3   1.90      84.16   -138
  converged True; constraints: all met; feasibility report: none
```

## What to notice

- **The starting point is step 7's**: 194.1 lb on the scale, tissue 87.13 kg, both water compartments below habitual, adaptation −125 kcal/day after four months of deficit. The fitted maintenance line, 2699 kcal/day, is c0 + c1 T at today's tissue alone (192.1 lb), without adaptation or the thermic effect of food, as `wdm filter` reports it.
- **With no held weight, the plan holds today's settled tissue, not today's scale reading**: 87.13 kg is 192.1 lb, 2.0 lb below the scale. The scale also weighs gut content, about 1.35 kg here (88.04 kg on the scale, less 87.13 of tissue, plus the 0.44 kg of drained water), and that outweighs the drained water. Why the plan holds tissue is in [Maintain mode](../theory/control.md#maintain).
- **Holding today's tissue means eating near maintenance.** The plan eats 2698 to 2940 kcal and tissue stays within 0.07 kg of where it started. Intake rises because adaptation fades once the deficit ends (−85 to −17 kcal/day). By the last block expenditure is roughly 2699 (maintenance at 87.11 kg of tissue) − 17 plus about 244 kcal/day of thermic effect, about 2926, against 2940 eaten.
- **The scale ends above the tissue it holds**, at 195.8 lb against 192.0, because it also weighs gut content: about 0.0006 × 2940 ≈ 1.8 kg (3.9 lb) at this intake. With carbohydrate back near the habitual 260 g, little water is left drained, so 195.8 lb is roughly what holding today's tissue reads on the scale at maintenance.
- **A held weight is a scale reading.** 189.1 lb is 3 lb below today's tissue in pounds, but read on the scale at maintenance it means 84.00 kg of tissue: 189.1 lb (85.77 kg) less about 1.77 kg of gut content. That is 3.1 kg (6.8 lb) below today's settled tissue (87.10 kg), so this is not a gentle 3 lb correction: the reference covers the gap in 28 days, the plan eats 1883 to 2094 kcal, the scale falls about 2 lb a week, adaptation deepens to −138 kcal/day, and tissue ends at 84.16 kg (84.07 settled) against the 84.00 held. By the last block tissue falls about 0.9 % a week, close to the 1 % limit.
- **Protein is held at 141 g** in every block of both plans, 1.6 g/kg of the starting scale weight, as in step 7.
- **No feasibility report.** Maintain mode has no date, so the check that flagged step 7's impossible deadline does not run. Replanning each morning keeps easing toward the held weight, and the 1 %-per-week limit still caps any correction.

## The code

| What | Where (under `wdm`) |
|---|---|
| The maintain reference (today's tissue to the held weight over the horizon) and the shared deadline objective | `control.mpc.solve`, `control.objectives`: `reference_trajectory`, `deadline_cost` |
| The planning mode and the held weight | `control.mpc.PlanSpec.mode`, `config.Maintain` |
| The tissue a held weight sets | `control.mpc`: `tissue_at_scale_weight`, `Plan.target_tissue_kg` |
| Fitted maintenance | `estimation.joint.JointFit.maintenance_interval_kcal` |
| Adaptation relaxing as the gap closes | `models.energetics.adaptation_rate` |

## Try this

- Hold today's scale reading explicitly with `hold(round(w_lb, 1))` (194.1 lb). At maintenance that reading means 86.25 kg of tissue, 88.04 kg less about 1.8 kg of gut content, so the plan has about 0.85 kg of tissue to lose from today's 87.10 kg settled: it eats 2531 to 2647 kcal and ends at 86.29 kg. A weight 3 lb below today's scale reading, `hold(round(w_lb - 3.0, 1))` (191.1 lb), is a 4.8 lb correction in tissue, about 1.3 lb a week at 2188 to 2269 kcal.
- Hold 3 lb above today's tissue instead. The plan eats above maintenance, and the rate limit, which only caps loss, plays no part.
- Give the 189.1-lb plan twice the horizon, `PlanSpec(mode="maintain", horizon_days=56, block_days=7)`: the same correction spread over eight weeks, at about half the weekly pace.
- Feed the first plan to the true subject, as step 7 does with `simulate`, and see how far the posterior-mean metabolism lets the true weight drift in four weeks. Replanning every morning is what keeps that drift small in use.
