---
title: "1. The model"
---

The six-state model on its own: one constant low-carbohydrate diet, simulated for 16 weeks at the prior parameters, with no noise, no data and no filter. The equations are in [The model](../theory/model.md).

```bash
uv run python walkthrough/01_model.py
```

<details><summary>The script</summary>

```python
"""Step 1: the model on its own.

Simulate 16 weeks of a constant low-carbohydrate diet at the prior parameters:
no noise, no data, no filter. The point is to see the six states move: the
scale drops fast in the first days (glycogen and sodium water), then settles
into slower tissue loss, which itself slows as adaptation builds.
"""

import jax
import jax.numpy as jnp
import numpy as np

from wdm.config import Subject
from wdm.constants import GUT_KG_PER_KCAL
from wdm.models import DailyInput, Params, State, simulate, simulate_diffrax
from wdm.models.energetics import TEF_MIXED_DIET_FRACTION, baseline_expenditure
from wdm.models.partition import fat_fraction

# A 40-year-old man, 175 cm, 100 kg at 30 % body fat.
subj = Subject(sex="male", age_years=40, height_cm=175.0, body_fat_frac=0.30)
w0 = 100.0
c0, c1 = subj.expenditure_prior(w0)  # Mifflin-St Jeor x activity, split c0 + c1 W
params = Params.prior(c0, c1)  # every other parameter at its literature prior
carb_ref, sodium_ref = subj.habitual_intake()  # habitual 260 g carb, 4000 mg sodium

# The diet, held constant every day: 180 g protein, 100 g carb, 98 g fat, 2500 mg sodium.
n = 112
protein, carb, fat, sodium = 180.0, 100.0, 98.0, 2500.0
kcal = 4 * protein + 4 * carb + 9 * fat


def const(v):
    return jnp.full(n, v)


u = DailyInput(
    energy_kcal=const(kcal),
    protein_g=const(protein),
    carb_dev_g=const(carb - carb_ref),
    fat_g=const(fat),
    alcohol_g=const(0.0),
    sodium_dev_mg=const(sodium - sodium_ref),
    exercise_kcal=const(0.0),
    intake_known=const(1.0),
    macro_known=const(1.0),
)
# Gut content starts where weight-stable pre-diet eating held it, part of the
# starting weight: tissue is the rest.
habitual_kcal = (c0 + c1 * w0) / (1.0 - TEF_MIXED_DIET_FRACTION)
gut0 = GUT_KG_PER_KCAL * habitual_kcal
x0 = State(
    tissue_kg=jnp.asarray(w0 - gut0),
    fat_kg=jnp.asarray(0.30 * (w0 - gut0)),
    gly_water_kg=jnp.asarray(0.0),
    na_water_kg=jnp.asarray(0.0),
    adapt_kcal=jnp.asarray(0.0),
    gut_kg=jnp.asarray(gut0),
)
u0 = jax.tree_util.tree_map(lambda a: a[0], u)  # day 0's input, for the printout

e0 = float(baseline_expenditure(x0, u0, params, carb_ref))
print(
    f"prior: c0 = {c0:.0f} kcal/day, c1 = {c1:.2f} kcal/day/kg,"
    f" at_gain = {float(params.at_gain):.2f}"
)
print(
    f"day-0 expenditure (no adaptation) {e0:.0f} kcal/day, intake {kcal:.0f}: gap {e0 - kcal:.0f}"
)
print(f"fat share of tissue lost on day 0: {float(fat_fraction(x0, u0, params)):.3f}")
print()

traj = simulate(x0, u, params, carb_ref)  # state after each day, length n
s = np.vstack([np.asarray(x0.to_vector()), np.asarray(traj.to_vector()).T])  # (n + 1, 6)
scale = s[:, 0] + s[:, 2] + s[:, 3] + s[:, 5]

print(" day   scale  tissue     fat  gly H2O  Na H2O     gut   adapt   lost   of which water+gut")
for d in (0, 1, 2, 3, 7, 14, 28, 56, 84, 112):
    lost = scale[0] - scale[d]
    transient = (s[0, 2] + s[0, 3] + s[0, 5]) - (s[d, 2] + s[d, 3] + s[d, 5])
    print(
        f"{d:4d}  {scale[d]:6.2f}  {s[d, 0]:6.2f}  {s[d, 1]:6.2f}  {s[d, 2]:+7.3f}  {s[d, 3]:+6.3f}"
        f"  {s[d, 5]:6.3f}  {s[d, 4]:+6.0f}  {lost:5.2f}   {transient:5.2f}"
    )
print()

print(
    "scale loss by week (kg): "
    + "  ".join(f"w{w}:{scale[7 * (w - 1)] - scale[7 * w]:.2f}" for w in (1, 2, 4, 8, 12, 16))
)
fat_lost, tissue_lost = s[0, 1] - s[-1, 1], s[0, 0] - s[-1, 0]
print(
    f"after {n} days: tissue -{tissue_lost:.2f} kg, of which fat {fat_lost:.2f} kg"
    f" ({fat_lost / tissue_lost:.0%})"
)

# The fast RK4 map against the adaptive Diffrax reference.
ref = simulate_diffrax(x0, u, params, carb_ref)
err = np.max(np.abs(np.asarray(ref.scale_kg) - scale[1:]))
print(f"RK4 (4 substeps/day) vs adaptive Tsit5: max scale difference {err * 1e6:.2f} mg")
```

</details>

```text
prior: c0 = 1303 kcal/day, c1 = 14.50 kcal/day/kg, at_gain = 0.15
day-0 expenditure (no adaptation) 2954 kcal/day, intake 2002: gap 952
fat share of tissue lost on day 0: 0.749

 day   scale  tissue     fat  gly H2O  Na H2O     gut   adapt   lost   of which water+gut
   0  100.00   98.16   29.45   +0.000  +0.000   1.835      +0   0.00    0.00
   1   98.99   98.07   29.38   -0.234  -0.501   1.656     -10   1.01    0.91
   2   98.45   97.96   29.30   -0.353  -0.685   1.527     -19   1.55    1.35
   3   98.11   97.85   29.21   -0.415  -0.753   1.435     -27   1.89    1.57
   7   97.37   97.38   28.86   -0.475  -0.792   1.263     -56   2.63    1.84
  14   96.51   96.57   28.26   -0.480  -0.792   1.207     -89   3.49    1.90
  28   94.98   95.05   27.13   -0.480  -0.792   1.201    -120   5.02    1.91
  56   92.11   92.18   25.03   -0.480  -0.792   1.201    -130   7.89    1.91
  84   89.35   89.43   23.06   -0.480  -0.792   1.201    -126  10.65    1.91
 112   86.69   86.76   21.19   -0.480  -0.792   1.201    -121  13.31    1.91

scale loss by week (kg): w1:2.63  w2:0.86  w4:0.75  w8:0.70  w12:0.68  w16:0.66
after 112 days: tissue -11.40 kg, of which fat 8.26 kg (72%)
RK4 (4 substeps/day) vs adaptive Tsit5: max scale difference 12.81 mg
```

## What to notice

- **The start.** The scale reads 100 kg, but 1.835 kg of that is gut content: the subject was eating the weight-stable 3059 kcal/day of the prior, (1303 + 14.50 × 100) / 0.90, and the gut holds 0.0006 kg per kcal/day eaten. Tissue starts at the remaining 98.16 kg, fat at 30 % of it, 29.45 kg.
- **The prior.** Mifflin-St Jeor scaled by an activity level of 1.45 gives c0 = 1303 kcal/day and c1 = 14.50 kcal/day per kg of tissue. Day-0 expenditure is 2954 kcal/day, so 2954 − 1303 − 14.50 × 98.16 = 228 kcal/day is the thermic effect of this day's food. Against 2002 kcal eaten, the gap is 952 kcal/day.
- **The early drop.** Day 1 loses 1.01 kg, 0.91 kg of it water and gut content. Eating 100 g of carbohydrate against a habitual 260 g drains glycogen water toward gly_gain × (100 − 260) = 0.0030 × (−160) = −0.480 kg; with a 1.5-day time constant it is 86 % of the way there by day 3 (1 − e^(−2) = 0.865).
- **Sodium water is the largest share of it.** Water held per milligram of sodium is fixed by the extracellular concentration, so the prior gain is large: 2.0·10⁻⁴ kg per mg/day. Eating 2500 mg against a habitual 4000 mg is −1500 mg/day, and cutting carbohydrate by 160 of 260 g makes the kidneys shed another 4000 × 160/260 = 2462 mg/day; together they settle at 2.0·10⁻⁴ × (−3962) = −0.792 kg. With a 1-day time constant 63 % of that (−0.501 kg) is gone by the first morning: more than half of day 1's drop. The same gain makes one salty day visible: 3000 mg above habitual adds about 0.38 kg the next morning.
- **Gut content follows intake.** At 2002 kcal/day it settles at 0.0006 × 2002 = 1.201 kg, 0.634 kg below where it started, with a 3-day time constant: 63 % of the way by day 3 (1.435 kg).
- **Then slower loss.** Week 1 loses 2.63 kg, 1.84 kg of it water and gut content; week 2 loses 0.86 kg, only 0.06 kg of it water and gut. The weekly loss keeps shrinking, to 0.66 kg in week 16, because expenditure falls with tissue (c1 × 11.40 kg ≈ 165 kcal/day by day 112) and adaptation builds.
- **Adaptation is a lagged state.** A reaches −56 kcal/day after a week (time constant 14 days), peaks at −130 around day 56, then shrinks as the gap closes. It relaxes toward −at_gain × gap: at day 56 the gap is 1303 + 14.50 × 92.18 + 228 − 2002 ≈ 866 kcal/day, and 0.15 × 866 = 130.
- **What is lost.** Forbes' law at 29.45 kg of fat gives 0.739, and 1.8 g/kg of protein (0.2 above the 1.6 reference) adds 0.010, so the day-0 fat share is 0.749. It falls as fat falls: 8.26 kg of fat in 11.40 kg of tissue over 112 days, 72 %.
- **The integrator.** The filter's fixed-step map (RK4, four substeps a day) agrees with an adaptive Tsit5 solve to 12.81 mg over 112 days.

## The code

All under `wdm` (see the [API reference](../reference/index.qmd)).

| What | Where |
|---|---|
| State, parameters, daily inputs | `models.types`: `State`, `Params` (`Params.prior`), `DailyInput` |
| The vector field and the one-day RK4 step | `models.dynamics`: `rate`, `step` |
| Rollouts | `models.dynamics`: `simulate`, `simulate_diffrax` |
| Expenditure, thermic effect, adaptation | `models.energetics`: `baseline_expenditure`, `thermic_effect`, `adaptation_rate` |
| Water compartments and gut content | `models.glycogen`: `glycogen_rate`, `sodium_rate`, `effective_sodium_dev_mg`, `gut_rate`; priors and fixed constants in `constants` |
| Fat share and energy density | `models.partition`: `forbes_fat_fraction`, `fat_fraction`, `effective_energy_density` |
| Expenditure prior; habitual intake | `config.Subject`: `expenditure_prior`, `habitual_intake` |

## Try this

- Keep the energy and eat the habitual 260 g of carbohydrate instead (carb 260, fat 27, 2003 kcal): the glycogen column stays at zero, sodium water stops at the −0.300 kg of the sodium cut alone, and week 1 drops 1.76 kg instead of 2.63. Tissue falls further, 11.90 kg against 11.40, because carbohydrate's thermic effect is higher than fat's (the gap is 986 kcal/day instead of 952), yet the scale ends higher on day 112, 87.17 kg against 86.69: the 0.97 kg more glycogen and sodium water the low-carbohydrate diet drained outweighs the extra tissue.
- Switch adaptation off with `params = dataclasses.replace(Params.prior(c0, c1), at_gain=jnp.asarray(0.0))`. Week 16 then loses 0.76 kg instead of 0.66, and tissue falls 13.05 kg instead of 11.40 over the 112 days.
- Move 70 g of carbohydrate to protein (protein 250, carb 30, fat 98, the same 2002 kcal). Two effects pull against each other: the fat share rises to 0.784, so each kilogram lost carries more energy, but protein's thermic effect widens the gap to 1001 kcal/day. Here the second wins slightly: tissue falls 11.57 kg instead of 11.40, and fat 8.81 kg instead of 8.26 (76 % of the loss). The deeper carbohydrate cut also drains more glycogen water (−0.690 kg) and more sodium water (−1.008 kg).
- Set `sodium = 4000.0`, the habitual level. The sodium column no longer reaches zero: carbohydrate's effect alone holds it at 2.0·10⁻⁴ × (−2462) = −0.492 kg, and day 1's drop shrinks from 1.01 kg to 0.82.
