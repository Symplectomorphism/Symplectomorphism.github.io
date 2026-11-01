---
title: "1. The model"
---

The five-state model on its own: one constant low-carbohydrate diet, simulated for 16 weeks at the prior parameters, with no noise, no data and no filter. The equations are in [The model](../theory/model.md).

```bash
uv run python walkthrough/01_model.py
```

<details><summary>The script</summary>

```python
"""Step 1: the model on its own.

Simulate 16 weeks of a constant low-carbohydrate diet at the prior parameters:
no noise, no data, no filter. The point is to see the five states move: the
scale drops fast in the first days (glycogen and sodium water), then settles
into slower tissue loss, which itself slows as adaptation builds.
"""

import jax
import jax.numpy as jnp
import numpy as np

from wdm.config import Subject
from wdm.models import DailyInput, Params, State, simulate, simulate_diffrax
from wdm.models.energetics import baseline_expenditure
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
x0 = State(
    tissue_kg=jnp.asarray(w0),
    fat_kg=jnp.asarray(0.30 * w0),
    gly_water_kg=jnp.asarray(0.0),
    na_water_kg=jnp.asarray(0.0),
    adapt_kcal=jnp.asarray(0.0),
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
s = np.vstack([np.asarray(x0.to_vector()), np.asarray(traj.to_vector()).T])  # (n + 1, 5)
scale = s[:, 0] + s[:, 2] + s[:, 3]

print(" day   scale  tissue     fat  gly H2O  Na H2O   adapt   lost   of which water")
for d in (0, 1, 2, 3, 7, 14, 28, 56, 84, 112):
    lost = scale[0] - scale[d]
    water = 0.0 - (s[d, 2] + s[d, 3])
    print(
        f"{d:4d}  {scale[d]:6.2f}  {s[d, 0]:6.2f}  {s[d, 1]:6.2f}  {s[d, 2]:+7.3f}  {s[d, 3]:+6.3f}"
        f"  {s[d, 4]:+6.0f}  {lost:5.2f}   {water:5.2f}"
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
day-0 expenditure (no adaptation) 2981 kcal/day, intake 2002: gap 979
fat share of tissue lost on day 0: 0.753

 day   scale  tissue     fat  gly H2O  Na H2O   adapt   lost   of which water
   0  100.00  100.00   30.00   +0.000  +0.000      +0   0.00    0.00
   1   99.48   99.90   29.93   -0.234  -0.190     -10   0.52    0.42
   2   99.18   99.79   29.84   -0.353  -0.259     -19   0.82    0.61
   3   98.98   99.68   29.76   -0.415  -0.285     -28   1.02    0.70
   7   98.42   99.20   29.40   -0.475  -0.300     -57   1.58    0.78
  14   97.61   98.39   28.79   -0.480  -0.300     -91   2.39    0.78
  28   96.06   96.84   27.64   -0.480  -0.300    -122   3.94    0.78
  56   93.15   93.93   25.51   -0.480  -0.300    -132   6.85    0.78
  84   90.36   91.14   23.50   -0.480  -0.300    -129   9.64    0.78
 112   87.66   88.44   21.59   -0.480  -0.300    -123  12.34    0.78

scale loss by week (kg): w1:1.58  w2:0.82  w4:0.76  w8:0.71  w12:0.69  w16:0.67
after 112 days: tissue -11.56 kg, of which fat 8.41 kg (73%)
RK4 (4 substeps/day) vs adaptive Tsit5: max scale difference 5.49 mg
```

## What to notice

- **The prior.** Mifflin-St Jeor scaled by an activity level of 1.45 gives c0 = 1303 kcal/day and c1 = 14.50 kcal/day per kg. Day-0 expenditure is 2981 kcal/day, so 2981 − 1303 − 14.50 × 100 = 228 kcal/day is the thermic effect of this day's food. Against 2002 kcal eaten, the gap is 979 kcal/day.
- **The early water drop.** Day 1 loses 0.52 kg, 0.42 kg of it water. Eating 100 g of carbohydrate against a habitual 260 g drains glycogen water toward gly_gain × (100 − 260) = 0.0030 × (−160) = −0.480 kg; with a 1.5-day time constant it is 86 % of the way there by day 3 (1 − e^(−2) = 0.865).
- **Sodium water is a large share of it.** Water held per milligram of sodium is fixed by the extracellular concentration, so the prior gain is large: 2.0·10⁻⁴ kg per mg/day. Eating 2500 mg against a habitual 4000 mg settles at −0.300 kg, and with a 1-day time constant 63 % of that (−0.190 kg) is gone by the first morning: almost half of day 1's water. The same gain makes one salty day visible: 3000 mg above habitual adds about 0.38 kg the next morning.
- **Then slower loss.** Week 1 loses 1.58 kg, 0.78 kg of it water; week 2 loses 0.82 kg and almost no water. The weekly loss keeps shrinking, to 0.67 kg in week 16, because expenditure falls with mass (c1 × 12.34 kg ≈ 179 kcal/day by day 112) and adaptation builds.
- **Adaptation is a lagged state.** A reaches −57 kcal/day after a week (time constant 14 days), peaks at −132 around day 56, then shrinks as the gap closes. It relaxes toward −at_gain × gap: at day 56 the gap is about 880 kcal/day, and 0.15 × 880 = 132.
- **What is lost.** Forbes' law at 30 kg of fat gives 0.743, and 1.8 g/kg of protein (0.2 above the 1.6 reference) adds 0.010, so the day-0 fat share is 0.753. It falls as fat falls: 8.41 kg of fat in 11.56 kg of tissue over 112 days, 73 %.
- **The integrator.** The filter's fixed-step map (RK4, four substeps a day) agrees with an adaptive Tsit5 solve to 5.49 mg over 112 days.

## The code

All under `wdm` (see the [API reference](../reference/index.qmd)).

| What | Where |
|---|---|
| State, parameters, daily inputs | `models.types`: `State`, `Params` (`Params.prior`), `DailyInput` |
| The vector field and the one-day RK4 step | `models.dynamics`: `rate`, `step` |
| Rollouts | `models.dynamics`: `simulate`, `simulate_diffrax` |
| Expenditure, thermic effect, adaptation | `models.energetics`: `baseline_expenditure`, `thermic_effect`, `adaptation_rate` |
| Water compartments | `models.glycogen`: `glycogen_rate`, `sodium_rate`; priors in `constants` |
| Fat share and energy density | `models.partition`: `forbes_fat_fraction`, `fat_fraction`, `effective_energy_density` |
| Expenditure prior; habitual intake | `config.Subject`: `expenditure_prior`, `habitual_intake` |

## Try this

- Keep the energy and eat the habitual 260 g of carbohydrate instead (carb 260, fat 27): the glycogen column stays at zero, and week 1 drops 1.20 kg instead of 1.58 (the 0.30 kg of sodium water still goes). By day 112 the scale ends lower, 87.57 kg against 87.66, because carbohydrate's thermic effect is higher than fat's (the gap is 1013 kcal/day instead of 979).
- Switch adaptation off with `params = dataclasses.replace(Params.prior(c0, c1), at_gain=jnp.asarray(0.0))`. Week 16 then loses 0.77 kg instead of 0.67, and tissue falls 13.22 kg instead of 11.56 over the 112 days.
- Move 70 g of carbohydrate to protein (protein 250, carb 30, fat 98, the same 2002 kcal). Two effects pull against each other: the fat share rises to 0.788, so each kilogram lost carries more energy, but protein's thermic effect widens the gap to 1028 kcal/day. Here the second wins slightly: tissue falls 11.70 kg instead of 11.56, and fat 8.92 kg instead of 8.41 (76 % of the loss). The deeper carbohydrate cut also drains more glycogen water (−0.690 kg).
- Set `sodium = 4000.0`, the habitual level: the sodium column stays at zero and day 1's drop shrinks to the glycogen water and tissue alone.
