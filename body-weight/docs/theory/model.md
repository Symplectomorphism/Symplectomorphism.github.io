---
title: "The model"
---

`wdm` describes one person's body weight as a five-state, continuous-time
dynamical system driven by a daily food log. This page states each mechanism
with its equation and source, then the observation model that connects the
hidden state to a bathroom scale and an optional tape measure.
[Estimation](estimation.md) covers discretization and inference,
[Control](control.md) the diet planner, and [Notation](notation.md) every
symbol. The [slide deck](../../weight-dynamics.html) is a visual tour of the
same material. Units: kilograms, kilocalories, days.

## The five states {#states}

| Symbol | Code name | Units | Meaning |
|:--|:--|:--|:----------|
| $T$ | `tissue_kg` | kg | Tissue mass: fat plus fat-free solids and their structural water. What a diet changes, and not what the scale reads. |
| $F$ | `fat_kg` | kg | Fat mass, a component of $T$. |
| $G_c$ | `gly_water_kg` | kg | Glycogen and its bound water, as a deviation from its level at habitual intake. Fast, carbohydrate-driven. |
| $G_n$ | `na_water_kg` | kg | Sodium-associated extracellular water, as a deviation from its habitual level. Zero unless sodium is logged. |
| $A$ | `adapt_kcal` | kcal/day | Adaptive thermogenesis. Negative during a deficit. |

Lean mass $M = T - F$ is not a separate state, so the fat/lean split is
consistent by construction. The scale weighs total mass $W = T + G_c + G_n$.

## Energy balance {#energy-balance}

Stored energy $S$ changes by intake minus expenditure,
$\mathrm{d}S/\mathrm{d}t = I - E$, with $I$ the metabolizable intake and $E$ the
expenditure (kcal/day). Nobody disputes this; the hard parts are what form the
stored energy takes, how $E$ depends on the body, and how well either is
measured. Expenditure is not measured outside a metabolic ward, and
self-reported intake is biased, often badly (Dhurandhar et al., 2015), so logged
intake is treated as a noisy input. A constant bias is absorbed into an
estimated intercept; a drifting one is the framework's most serious weakness
([Limitations](limitations.md#drifting-bias)). Before use, logged energy is
rebuilt from Atwater factors (4, 4, 9, 7 kcal/g) when only macronutrients are
given, and 2 kcal per gram of fiber is subtracted, since labels count fiber at
the full carbohydrate factor.

## Fat and lean: where the mass comes from {#partition}

Adipose tissue stores $\rho_F = 9441$ kcal/kg (39.5 MJ/kg) and fat-free tissue
$\rho_L = 1816$ kcal/kg (7.6 MJ/kg) (Hall, 2008); fat-free tissue is about 70%
water.

::: {.callout-note title="The most important fact in the model"}
The same deficit moves the scale five times as far if the mass comes off as lean
tissue rather than fat. A rule such as 3500 kcal per pound assumes this away,
along with falling expenditure and adaptation, and so predicts unbounded linear
loss where real trajectories flatten (Hall et al., 2011).
:::

**Forbes' law.** Forbes (1987) found lean mass approximately logarithmic in fat
mass across a wide range of adiposity, $M = 10.4 \ln F + \text{const}$ (kg).
Since $\mathrm{d}M/\mathrm{d}F = 10.4/F$ and $\mathrm{d}W = \mathrm{d}F +
\mathrm{d}M$, the fat share of any mass change is

$$
p_{\mathrm{Forbes}} = \frac{\mathrm{d}F}{\mathrm{d}W} = \frac{F}{F + 10.4},
$$

with $F$ floored at 2 kg in code, where the relation is singular. The energy
cost of a kilogram of tissue is then

$$
\rho_{\mathrm{eff}} = p\,\rho_F + (1 - p)\,\rho_L .
$$

| $F$ (kg) | 50 | 30 | 20 | 15 |
|:--|--:|--:|--:|--:|
| $p_{\mathrm{Forbes}}$ | 0.83 | 0.74 | 0.66 | 0.59 |
| $\rho_{\mathrm{eff}}$ (kcal/kg) | 8128 | 7478 | 6832 | 6319 |

The fatter you are, the larger the fat share of each kilogram lost, so the last
stretch of a diet is harder than the first. Forbes' law is a population
regression with substantial individual scatter, used as a structural prior that
the scale barely constrains.

**Protein.** Higher protein intake during a deficit spares lean mass (Helms et
al., 2014). With $P_g$ the protein eaten (g/day),

$$
p = \operatorname{clip}\!\left( \frac{F}{F + 10.4}
  + \kappa_P \left( \frac{P_g}{W} - 1.6 \right),\; 0.50,\; 0.98 \right),
$$

applied only on days with a logged macronutrient split. The effect is real but
its size is uncertain, and $\kappa_P$ is essentially not recoverable from scale
data ([identifiability](estimation.md#identifiability)). It stays because
omitting it would assert that protein has no effect.

## Expenditure {#expenditure}

Total expenditure is roughly 60 to 70% resting metabolism, 20 to 30% activity
and 10% thermic effect of food. The model writes

$$
E = \underbrace{c_0 + c_1 W}_{\text{resting + baseline activity}}
  + \mathrm{TEF} + X + A ,
$$

with $X$ the logged exercise (kcal/day).

**Resting expenditure.** The Mifflin-St Jeor equation (Mifflin et al., 1990),
$\mathrm{RMR} = 10\,W + 6.25 \times \text{height} - 5 \times \text{age} + s$
(kg, cm, years; $s = +5$ for men, $-161$ for women), only centers the priors.
The code scales it by an activity level of 1.45: $c_1$ is centered at 14.5
kcal/day per kg and $c_0$ at $1.45\,(6.25 \times \text{height} - 5 \times
\text{age} + s)$. Population equations miss an individual by several hundred
kilocalories a day, so both are estimated. Never quote $c_0$ as physiology: it
absorbs mass-independent expenditure, baseline activity and any constant bias in
the food log (a subject who under-reports by 20% has that 20% absorbed into
$c_0$), which is what makes a constant bias harmless.

**Thermic effect of food.** With carbohydrate $C$, fat $F_g$ and alcohol $A_g$
in g/day,

$$
\mathrm{TEF} = \alpha_P \cdot 4 P_g + \alpha_C \cdot 4 C
  + \alpha_F \cdot 9 F_g + \alpha_A \cdot 7 A_g ,
$$

with $\alpha_P = 0.25$, $\alpha_C = 0.075$, $\alpha_F = 0.02$ and
$\alpha_A = 0.15$ (reported ranges 20 to 30%, 5 to 10%, 0 to 3% and 10 to 20%;
Westerterp, 2004), or 10% of intake when no split is logged. Moving 100 g of
carbohydrate to protein at constant energy raises expenditure by about
$(0.25 - 0.075) \times 400 \approx 70$ kcal/day. The coefficients are held fixed
because their effect is confounded with maintenance energy.

**Adaptive thermogenesis.** During restriction, expenditure falls by more than
mass loss explains (Leibel et al., 1995; Rosenbaum and Leibel, 2010), and the
effect can persist for years (Fothergill et al., 2016). Adaptation responds to
the gap against expenditure without adaptation (using total expenditure would
make the equation refer to itself):

$$
E_{\mathrm{base}} = c_0 + c_1 W + \mathrm{TEF} + X, \qquad
D = E_{\mathrm{base}} - I, \qquad
\dot A = \frac{-\beta_A D - A}{\tau_A},
$$

with $\beta_A \approx 0.15$ (reported 0.10 to 0.25; Hall, 2010) and
$\tau_A \approx 14$ days, an order-of-magnitude figure.

::: {.callout-tip title="The plateau"}
Under a sustained gap $D$, adaptation settles at $-\beta_A D$. A 500 kcal/day
deficit at $\beta_A = 0.15$ eventually lowers expenditure by 75 kcal/day beyond
what mass loss explains, shrinking the effective deficit to 425 kcal/day; loss slows
toward a new equilibrium at constant intake. The plateau is a property of the
system, not a failure of the person in it.
:::

The intake that holds weight steady is not $c_0 + c_1 W$ but
$(c_0 + c_1 W)/(1 - \phi)$ for a thermic fraction $\phi$, about 11% higher at
$\phi = 0.10$; mid-deficit it overestimates, since it ignores adaptation.

## Water compartments {#water}

Two first-order lags move the scale within days with no change in tissue.

**Glycogen water.** Glycogen is stored with roughly 3 g of water per gram
(Olsson and Saltin, 1970), and swinging between carbohydrate restriction and
repletion moves about 200 g of glycogen plus 600 g of water:

$$
\dot G_c = \frac{k_G\,\Delta C - G_c}{\tau_G}, \qquad \Delta C = C - \bar C .
$$

A sustained +250 g/day swing gives about 0.8 kg on the scale, so
$k_G \approx 0.8/250 = 0.0032$, rounded to a prior center of 0.0030 kg per g/day.
This compartment explains the dramatic first week of a low-carbohydrate diet,
the "relapse" after one large meal, and much of the daily scale noise; modeling
it keeps that water from masquerading as fat.

**Sodium water.** Extracellular fluid tracks sodium load over one to two days,
often a larger and almost never logged source of scale noise:

$$
\dot G_n = \frac{k_{Na}\,\Delta\mathrm{Na} - G_n}{\tau_{Na}}, \qquad
\Delta\mathrm{Na} = \mathrm{Na} - \overline{\mathrm{Na}} ,
$$

with $\mathrm{Na}$ in mg/day. Days without a sodium entry have
$\Delta\mathrm{Na} = 0$.

The gain is the equilibrium water per unit of sustained sodium change.
Extracellular fluid holds sodium at about 140 mmol/L (3.2 g per liter), and its
volume moves until excretion matches intake: raising intake by about 4,400 mg/day
adds on the order of a kilogram over several days, about $2.3 \times 10^{-4}$ kg
per mg/day. The prior is centered at $2 \times 10^{-4}$ with a log-scale spread of
1.0 (a factor of about 2.7 either way), since some studies find sodium stored
without water. Excretion is the lag's job: a salty restaurant meal, a one-day
pulse of 3,000 mg, raises the next morning's weight by roughly 0.4 to 0.7 kg,
gone within two or three days.

### The habitual-intake reference {#habitual}

$\bar C$ and $\overline{\mathrm{Na}}$ are the subject's habitual, pre-diet
intakes: by default 260 g and 4000 mg a day for men, 200 g and 3000 mg for women
(roughly US adult averages), overridden by `habitual_carb_g` and
`habitual_sodium_mg` in `subject.toml` ([Configuration](../configuration.md)).
A diet that starts from ordinary eating starts with both compartments at their
habitual fill, so $G_c(0) = G_n(0) = 0$, and the drop to diet intake drains them
over the first days: the model predicts that drain instead of crediting it to
tissue. For example, 80 g/day of carbohydrate against a habitual 260 g sends
$G_c$ toward $k_G \Delta C \approx -0.54$ kg with a 1.5-day time constant. The
earlier reference, each episode's mean intake, hid that drop and changed as days
were added; on the report's logged record the habitual reference fits better.
Called without a habitual level, the library still falls back to episode means.

## The assembled model {#assembled}

Energy not spent goes into tissue, except energy diverted into glycogen, which
is unavailable to fat. With state $x = (T, F, G_c, G_n, A)$ and daily input
$u = (I, P_g, \Delta C, F_g, A_g, \Delta\mathrm{Na}, X)$:

$$
\begin{aligned}
\rho_{\mathrm{eff}}\,\dot T &= I - E - \rho_G\,\dot G_c, &
\dot F &= p\,\dot T, \\
\dot G_c &= (k_G\,\Delta C - G_c)/\tau_G, &
\dot G_n &= (k_{Na}\,\Delta\mathrm{Na} - G_n)/\tau_{Na}, \\
\dot A &= \bigl(-\beta_A (E_{\mathrm{base}} - I) - A\bigr)/\tau_A .
\end{aligned}
$$

$\rho_G \approx 1025$ kcal/kg is the energy density of the glycogen-water
complex: glycogen carries about 4.1 kcal/g but is stored with 3 g of water, so
$4100/4 \approx 1025$. Using 4100 would overstate the cost of refeeding
fourfold. Sodium water carries no energy term.

## Parameters and priors {#parameters}

Nine parameters are estimated. Each prior is a transform of a standard-normal
coordinate $\eta$ ([reparameterization](estimation.md#reparameterization)):
"log, $s$" means $\theta = c\,e^{s\eta}$ ($s$ is the sd of $\ln\theta$), "logit,
$s$" means $\theta = \sigma(\operatorname{logit} c + s\eta)$ with $\sigma$ the
logistic function, and "normal, $s$" means $\theta = c + s\eta$.

| Code name | Symbol | Meaning | Prior center $c$ | Spread | Units | Source |
|:--|:--|:------|:----|:--|:--|:----|
| `rmr_intercept` | $c_0$ | Mass-independent expenditure, plus constant log bias | $1.45\,(6.25 \times \text{height} - 5 \times \text{age} + s)$ | log, 0.15 | kcal/day | Mifflin et al. (1990) |
| `rmr_slope` | $c_1$ | Mass dependence of expenditure | 14.5 | log, 0.25 | kcal/day per kg | Mifflin et al. (1990) |
| `at_gain` | $\beta_A$ | Fraction of a sustained deficit offset by adaptation | 0.15 | logit, 0.70 | dimensionless | Hall (2010) |
| `at_tau` | $\tau_A$ | Adaptation time constant | 14 | log, 0.50 | days | Order of magnitude |
| `gly_gain` | $k_G$ | Glycogen-water gain | 0.0030 | log, 0.50 | kg per g/day | Olsson and Saltin (1970) |
| `gly_tau` | $\tau_G$ | Glycogen-water time constant | 1.5 | log, 0.40 | days | Assumed fast lag |
| `na_gain` | $k_{Na}$ | Sodium-water gain | $2 \times 10^{-4}$ | log, 1.0 | kg per mg/day | Extracellular fluid at 140 mmol/L |
| `na_tau` | $\tau_{Na}$ | Sodium-water time constant | 1.0 | log, 0.40 | days | Standard physiology |
| `partition_kappa` | $\kappa_P$ | Fat-share shift per g/kg/day of protein above 1.6 | 0.05 | normal, 0.05 | per g/kg/day | Helms et al. (2014) |

<!-- Source note: the report gives prior centers but not the prior spreads, the activity-level factor 1.45 or the k_Na prior center; those come from the code (estimation/reparam.py, config.py, constants.py). -->

The thermic-effect coefficients, tissue energy densities and Forbes coefficient
are fixed; [identifiability](estimation.md#identifiability) explains why. The
tape adds two estimated constants, $a_w$ and $b_w$ ([below](#waist)).

## Unlogged days and timescales {#timescales}

On a day with no food log, $I$ is set to current total expenditure (no net
imbalance), carbohydrate and sodium to habitual levels, and the process noise is
inflated ([process noise](estimation.md#process-noise)). People forget to log a
party more often than a salad; the inflation widens the uncertainty but does not
correct that bias.

| Component | Timescale | Consequence |
|:--|:--|:----|
| Water compartments | 1 to 2 days | Fast scale swings with no tissue change |
| Adaptation | about 14 days | The plateau; separable from mass effects only because it lags |
| Tissue | months | The actual objective |

If adaptation were instantaneous it would be perfectly confounded with an error
in $c_0$. Because it lags, a change in intake produces a brief transient in which
the two differ, and that transient is the whole basis for estimating $\beta_A$.

## The observation model {#observation}

**The scale** reads total mass plus noise on the morning of day $k$:

$$
y_k = H x_k + v_k, \qquad
H = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \end{pmatrix}, \qquad
v_k \sim \mathcal{N}(0, R).
$$

The map is linear, so the scale update is exact. $R$ is not instrument error
(tens of grams for a decent scale) but fluid and gut-content variation the water
compartments miss; for consistent morning fasted weights its standard deviation
is near 0.4 to 0.6 kg (prior 0.45 kg), and [phase 1](estimation.md#phase-1)
estimates it from your history. Consistent conditions therefore matter more
than accuracy: a constant scale offset is absorbed into the initial condition,
while weighing at random times adds variance no filter can remove. The noise is
Gaussian on evidence, with a gate against typing errors
([noise model](estimation.md#noise-model)).

### The waist tape {#waist}

The scale sees fat only through $\rho_{\mathrm{eff}}$, which is weak evidence; a
tape measure responds to fat directly. The optional `waist_in` or `waist_cm`
column ([Data](../data.md)) is modeled, in centimeters, as

$$
y^w_k = a_w + b_w\,F_k + v^w_k, \qquad
v^w_k \sim \mathcal{N}(0, \sigma_w^2), \qquad \sigma_w = 1.5\ \text{cm},
$$

the agreement of repeated self-measurements. The offset $a_w$ and slope $b_w$
are personal constants, estimated like the metabolic parameters and added to the
filter only when some episode has a waist reading.

The priors come from Lean, Han and Deurenberg (1996), who predict
$\%\mathrm{BF} = 0.567\,y^w + 0.101\,\text{age} - 31.8$ for men and
$0.439\,y^w + 0.221\,\text{age} - 9.4$ for women, with standard errors of 4.1
and 4.9 percentage points. With $k_w$ the waist coefficient and
$\%\mathrm{BF} = 100F/W$, differentiating at fixed lean mass gives the prior
slope

$$
b_{w,0} = \frac{100\,M}{k_w\,W^2},
$$

at the starting composition of the latest episode, with a log-scale prior sd of
0.2 (about $\pm 20\%$). A man of 100 kg with 30 kg of fat gets about 1.2 cm per
kilogram of fat, consistent with the common observation of about a centimeter
per kilogram lost. The prior offset puts the line through the Lean prediction at
the starting fat mass, with sd equal to the standard error in centimeters,
$4.1/0.567 \approx 7.2$ cm for men ($4.9/0.439 \approx 11.2$ cm for women). The
Lean equation's implied slope steepens as fat falls (about 10% per 5 kg lost),
so the prior slope is a starting value that the estimated slope must track. The
tape constrains only the product of $b_w$ and the fat share, so it anchors the
fat level without pinning down the share
([what the tape identifies](estimation.md#waist-update)).

## How firm each piece is {#firmness}

| Relationship | Source | Firmness |
|:--|:--|:--|
| Energy balance | First law | Certain |
| Tissue energy densities | Hall (2008) | Strong |
| Forbes partitioning | Forbes (1987) | Moderate |
| Protein lean-sparing | Helms et al. (2014) | Weak |
| RMR from body measurements | Mifflin et al. (1990) | Prior only |
| Thermic-effect ordering; magnitudes | Westerterp (2004) | Strong; moderate, held fixed |
| Adaptive thermogenesis exists | Leibel et al. (1995) | Strong |
| Adaptation parameters | Hall (2010) | Weak |
| Glycogen-water coupling | Olsson and Saltin (1970) | Mechanism certain, ratio moderate |
| Sodium-water coupling | Standard physiology | Moderate |
| Habitual intake levels | US adult averages | Default only; set your own |
| Waist against fat mass | Lean et al. (1996) | Prior only; slope estimated |

A wide posterior on a weakly founded parameter is the estimator correctly
reporting that neither the literature nor your data has much to say.

## References {#references}

- Dhurandhar, N. V., Schoeller, D., Brown, A. W., Heymsfield, S. B., Thomas, D., Sørensen, T. I. A., Speakman, J. R., Jeansonne, M., and Allison, D. B. (2015). Energy balance measurement: when something is not better than nothing. *International Journal of Obesity*.
- Forbes, G. B. (1987). Lean body mass-body fat interrelationships in humans. *Nutrition Reviews*.
- Fothergill, E., Guo, J., Howard, L., Kerns, J. C., Knuth, N. D., Brychta, R., Chen, K. Y., Skarulis, M. C., Walter, M., Walter, P. J., and Hall, K. D. (2016). Persistent metabolic adaptation 6 years after "The Biggest Loser" competition. *Obesity*.
- Hall, K. D. (2008). What is the required energy deficit per unit weight loss? *International Journal of Obesity*.
- Hall, K. D. (2010). Predicting metabolic adaptation, body weight change, and energy intake in humans. *American Journal of Physiology: Endocrinology and Metabolism*.
- Hall, K. D., Sacks, G., Chandramohan, D., Chow, C. C., Wang, Y. C., Gortmaker, S. L., and Swinburn, B. A. (2011). Quantification of the effect of energy imbalance on bodyweight. *The Lancet*.
- Helms, E. R., Zinn, C., Rowlands, D. S., and Brown, S. R. (2014). A systematic review of dietary protein during caloric restriction in resistance trained lean athletes: a case for higher intakes. *International Journal of Sport Nutrition and Exercise Metabolism*.
- Lean, M. E. J., Han, T. S., and Deurenberg, P. (1996). Predicting body composition by densitometry from simple anthropometric measurements. *The American Journal of Clinical Nutrition*, 63(1), 4-14.
- Leibel, R. L., Rosenbaum, M., and Hirsch, J. (1995). Changes in energy expenditure resulting from altered body weight. *New England Journal of Medicine*, 332, 621-628.
- Mifflin, M. D., St Jeor, S. T., Hill, L. A., Scott, B. J., Daugherty, S. A., and Koh, Y. O. (1990). A new predictive equation for resting energy expenditure in healthy individuals. *American Journal of Clinical Nutrition*, 51, 241-247.
- Olsson, K.-E., and Saltin, B. (1970). Variation in total body water with muscle glycogen changes in man. *Acta Physiologica Scandinavica*.
- Rosenbaum, M., and Leibel, R. L. (2010). Adaptive thermogenesis in humans. *International Journal of Obesity*.
- Westerterp, K. R. (2004). Diet induced thermogenesis. *Nutrition & Metabolism*.
