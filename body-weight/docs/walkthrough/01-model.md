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

- **The prior.** For a 40-year-old, 175 cm, 100 kg man, Mifflin-St Jeor scaled by an activity level of 1.45 gives c0 = 1303 kcal/day and c1 = 14.50 kcal/day per kg. Day-0 expenditure is 2981 kcal/day, so 2981 − 1303 − 14.50 × 100 = 228 kcal/day is the thermic effect of this day's food (25 % of the protein energy, 7.5 % of the carbohydrate, 2 % of the fat). Against 2002 kcal eaten, the gap is 979 kcal/day.
- **The early water drop.** On day 1 the scale falls 0.52 kg, and 0.42 kg of it is water: 81 %. Eating 100 g of carbohydrate against a habitual 260 g drains glycogen-bound water toward its steady state, gly_gain × (100 − 260) = 0.0030 × (−160) = −0.480 kg, which is the value the column settles at. With a 1.5-day time constant it is 86 % of the way there by day 3 (−0.415 of −0.480, and 1 − e^(−2) = 0.865).
- **Sodium water is now a real share of it.** The prior sodium gain is 2.0·10⁻⁴ kg per mg/day, so eating 2500 mg against a habitual 4000 mg settles at 2.0·10⁻⁴ × (2500 − 4000) = −0.300 kg, the value the column reaches by day 7. Its time constant is 1 day, so 63 % of that (−0.190 kg, 1 − e^(−1) = 0.632) is gone by the first morning: almost half of day 1's water. The same gain makes a single salty day visible on the scale: 3000 mg above habitual for one day adds about 2.0·10⁻⁴ × 3000 × 0.632 = 0.38 kg the next morning.
- **Then slower loss.** Week 1 loses 1.58 kg, of which 0.78 kg is water; week 2 loses 0.82 kg and almost no water (the water column stays at 0.78). After that the weekly loss keeps shrinking, to 0.67 kg in week 16, for two reasons: expenditure falls with body mass (c1 × 12.34 kg ≈ 179 kcal/day less by day 112), and adaptation builds.
- **Adaptation is a lagged state.** A reaches −57 kcal/day after a week and −91 after two (time constant 14 days), peaks at −132 around day 56, then shrinks to −123 by day 112. It relaxes toward −at_gain × gap, and the gap itself closes as mass falls. At day 56 the gap is 1303 + 14.50 × 93.15 + 228 − 2002 ≈ 880 kcal/day, and 0.15 × 880 = 132, the printed value.
- **What is lost.** Forbes' law at 30 kg of fat gives 30 / (30 + 10.4) = 0.743, and 1.8 g/kg of protein (0.2 above the 1.6 reference) adds 0.05 × 0.2 = 0.010, so the day-0 fat share is 0.753. Over 112 days the share falls as fat falls, and the cumulative split is 8.41 kg of fat in 11.56 kg of tissue, 73 %.
- **The integrator.** The filter's fixed-step map (RK4, four substeps a day) agrees with an adaptive Tsit5 solve to 5.49 mg of scale weight over 112 days.

## The code

| What | Where |
|---|---|
| State, parameters, daily inputs | `models/types.py:54` (`State`), `models/types.py:109` (`Params`; `prior` at 158), `models/types.py:181` (`DailyInput`) |
| The vector field; energy into glycogen withheld from tissue | `models/dynamics.py:88` (`rate`), the energy line at 103 |
| One-day RK4 step | `models/dynamics.py:116` (`step`); `DEFAULT_SUBSTEPS = 4` at 43 |
| Rollouts | `models/dynamics.py:161` (`simulate`), `models/dynamics.py:184` (`simulate_diffrax`) |
| Expenditure, thermic effect, adaptation | `models/energetics.py:78` (`baseline_expenditure`), `:40` (`thermic_effect`), `:112` (`adaptation_rate`) |
| Water compartments | `models/glycogen.py:36` (`glycogen_rate`), `models/glycogen.py:52` (`sodium_rate`); priors `constants.py:202` (glycogen gain), `constants.py:215` (sodium gain), `constants.py:234` (sodium time constant) |
| Fat share and energy density | `models/partition.py:52` (`forbes_fat_fraction`), `:58` (`fat_fraction`), `:72` (`effective_energy_density`) |
| Expenditure prior; habitual intake | `config.py:93` (`expenditure_prior`), `config.py:62` (`habitual_intake`), defaults at `config.py:31` |

## Assumptions made here

| Assumption | Where | Why |
|---|---|---|
| Each day's intake is held constant over the day | `models/dynamics.py:116` | A food log reports daily totals; this makes the step an exact zero-order hold |
| Water compartments are measured from habitual intake: 260 g carbohydrate and 4000 mg sodium for a man | `config.py:31` | A diet that starts from ordinary eating starts both at zero, so the first-week drain is predicted rather than credited to tissue |
| Expenditure prior: Mifflin-St Jeor × 1.45 | `config.py:93`, `constants.py:167` | Only a prior center; phase 2 re-estimates both coefficients (step 4) |
| Thermic-effect fractions are fixed, not estimated | `constants.py:115–129` | Confounded with maintenance on scale data |
| Fat share from Forbes' law plus a linear protein shift, clipped to [0.50, 0.98] | `models/partition.py:58`, `constants.py:259–260` | Keeps the energy density physical whatever the parameters do |
| Sodium water: 2.0·10⁻⁴ kg per mg/day at equilibrium, 1-day time constant | `constants.py:215`, `constants.py:234` | Extracellular fluid holds sodium at a fixed concentration; excretion is the lag, so the gain is the equilibrium value |
| Glycogen complex stores 1025 kcal/kg | `constants.py:58` | Energy stored as glycogen is not available to tissue, so a refeed is not credited as fat gain |
| Adaptation is a first-order lag toward −at_gain × gap | `models/energetics.py:112` | The lag is what separates it, partly, from an error in c0 |

## Try this

- Keep the energy and eat the habitual 260 g of carbohydrate instead (carb 260, fat 27): the glycogen column stays at zero, and week 1 drops 1.20 kg instead of 1.58 (the 0.30 kg of sodium water still goes). By day 112 the scale ends lower, 87.57 kg against 87.66, because carbohydrate's thermic effect is higher than fat's (the gap is 1013 kcal/day instead of 979).
- Switch adaptation off with `params = dataclasses.replace(Params.prior(c0, c1), at_gain=jnp.asarray(0.0))`. Week 16 then loses 0.77 kg instead of 0.67, and tissue falls 13.22 kg instead of 11.56 over the 112 days.
- Move 70 g of carbohydrate to protein (protein 250, carb 30, fat 98, the same 2002 kcal). Two effects pull against each other: the fat share rises to 0.788, so each kilogram lost carries more energy, but protein's thermic effect widens the gap to 1028 kcal/day. Here the second wins slightly: tissue falls 11.70 kg instead of 11.56, and fat 8.92 kg instead of 8.41 (76 % of the loss). The deeper carbohydrate cut also drains more glycogen water (−0.690 kg).
- Set `sodium = 4000.0`, the habitual level: the sodium column stays at zero and day 1's drop shrinks to the glycogen water and tissue alone.
