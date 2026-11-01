---
title: "The model"
---

`wdm` models one person's body weight as a five-state, continuous-time system
driven by a daily food log, observed through a scale and an optional tape
measure. Units: kilograms, kilocalories, days; symbols are in
[Notation](notation.md).

## The five states {#states}

| Symbol | Code name | Units | Meaning |
|:--|:--|:--|:----------|
| $T$ | `tissue_kg` | kg | Tissue: fat plus fat-free solids and their structural water. What a diet changes. |
| $F$ | `fat_kg` | kg | Fat mass, part of $T$; lean mass is $M = T - F$. |
| $G_c$ | `gly_water_kg` | kg | Glycogen and its bound water, relative to habitual intake. |
| $G_n$ | `na_water_kg` | kg | Sodium-associated water, relative to habitual intake. |
| $A$ | `adapt_kcal` | kcal/day | Adaptive thermogenesis, negative in a deficit. |

The scale weighs $W = T + G_c + G_n$.

## Energy balance {#energy-balance}

Stored energy changes by metabolizable intake $I$ minus expenditure $E$.
Self-reported intake is biased (Dhurandhar et al., 2015), so it enters as a
noisy input ([log bias](limitations.md#drifting-bias)). Energy is rebuilt from
Atwater factors (4, 4, 9, 7 kcal/g) when only macronutrients are logged, and
2 kcal per gram of fiber is subtracted, since labels count fiber as
carbohydrate.

## Fat and lean: where the mass comes from {#partition}

Adipose tissue stores $\rho_F = 9441$ kcal/kg and fat-free tissue
$\rho_L = 1816$ kcal/kg (Hall, 2008).

::: {.callout-note title="The most important fact in the model"}
The same deficit moves the scale five times as far if the mass comes off as lean
tissue rather than fat. A fixed 3500 kcal per pound ignores this and falling
expenditure, and so predicts linear loss where real trajectories flatten (Hall
et al., 2011).
:::

**Forbes' law.** Lean mass is approximately logarithmic in fat mass,
$M = 10.4 \ln F + \text{const}$ (Forbes, 1987), so the fat share of a mass
change and the energy cost of a kilogram of tissue are

$$
p_{\mathrm{Forbes}} = \frac{F}{F + 10.4}, \qquad
\rho_{\mathrm{eff}} = p\,\rho_F + (1 - p)\,\rho_L .
$$

| $F$ (kg) | 50 | 30 | 20 | 15 |
|:--|--:|--:|--:|--:|
| $p_{\mathrm{Forbes}}$ | 0.83 | 0.74 | 0.66 | 0.59 |
| $\rho_{\mathrm{eff}}$ (kcal/kg) | 8128 | 7478 | 6832 | 6319 |

The leaner the body, the more lean tissue each kilogram lost contains, so the
last stretch of a diet is harder than the first.

**Protein.** Higher protein intake during a deficit spares lean mass (Helms et
al., 2014). With $P_g$ the protein eaten (g/day),

$$
p = \operatorname{clip}\!\left( \frac{F}{F + 10.4}
  + \kappa_P \left( \frac{P_g}{W} - 1.6 \right),\; 0.50,\; 0.98 \right),
$$

on days with a logged macronutrient split. $\kappa_P$ is not recoverable from
scale data ([identifiability](estimation.md#identifiability)); it stays because
omitting it would assert that protein has no effect.

## Expenditure {#expenditure}

$$
E = \underbrace{c_0 + c_1 W}_{\text{resting + baseline activity}}
  + \mathrm{TEF} + X + A ,
$$

with $X$ the logged exercise (kcal/day).

**Resting expenditure.** The Mifflin-St Jeor equation (Mifflin et al., 1990),
times an activity level of 1.45, centers the priors on $c_0$ and $c_1$; both are
estimated, since population equations miss an individual by hundreds of
kilocalories. $c_0$ also absorbs baseline activity and any constant log bias.

**Thermic effect of food**, with macronutrients in g/day:

$$
\mathrm{TEF} = \alpha_P \cdot 4 P_g + \alpha_C \cdot 4 C
  + \alpha_F \cdot 9 F_g + \alpha_A \cdot 7 A_g ,
$$

with $\alpha_P = 0.25$, $\alpha_C = 0.075$, $\alpha_F = 0.02$, $\alpha_A = 0.15$
(Westerterp, 2004), or 10% of intake when no split is logged.

**Adaptive thermogenesis.** In a deficit, expenditure falls by more than mass
loss explains (Leibel et al., 1995), sometimes for years (Fothergill et al.,
2016). Adaptation responds to the gap against expenditure without it:

$$
E_{\mathrm{base}} = c_0 + c_1 W + \mathrm{TEF} + X, \qquad
D = E_{\mathrm{base}} - I, \qquad
\dot A = \frac{-\beta_A D - A}{\tau_A},
$$

with $\beta_A \approx 0.15$ (Hall, 2010) and $\tau_A \approx 14$ days.

::: {.callout-tip title="The plateau"}
Under a sustained gap $D$, adaptation settles at $-\beta_A D$: a 500 kcal/day
deficit eventually loses 75 kcal/day to it, and loss slows at constant intake.
The plateau is a property of the system, not a failure of the person in it.
:::

The intake that holds weight steady is $(c_0 + c_1 W)/(1 - \phi)$ for a thermic
fraction $\phi$, about 11% above $c_0 + c_1 W$ at $\phi = 0.10$.

## Water compartments {#water}

Two first-order lags move the scale within days with no change in tissue.

**Glycogen water** (about 3 g of water per gram of glycogen; Olsson and Saltin,
1970):

$$
\dot G_c = \frac{k_G\,\Delta C - G_c}{\tau_G}, \qquad \Delta C = C - \bar C .
$$

A sustained +250 g/day swing puts about 0.8 kg on the scale. This compartment
explains the dramatic first week of a low-carbohydrate diet and the "relapse"
after one large meal, and keeps that water from masquerading as fat.

**Sodium water** tracks sodium load:

$$
\dot G_n = \frac{k_{Na}\,\Delta\mathrm{Na} - G_n}{\tau_{Na}}, \qquad
\Delta\mathrm{Na} = \mathrm{Na} - \overline{\mathrm{Na}} ,
$$

with $\mathrm{Na}$ in mg/day and $\Delta\mathrm{Na} = 0$ when sodium is not
logged. A salty restaurant meal raises the next morning's weight by roughly 0.4
to 0.7 kg, gone within two or three days.

### The habitual-intake reference {#habitual}

$\bar C$ and $\overline{\mathrm{Na}}$ are the subject's habitual, pre-diet
intakes: by default 260 g and 4000 mg a day for men, 200 g and 3000 mg for women
([The subject file](../configuration.md)). A diet starts with both compartments
at their habitual fill, $G_c(0) = G_n(0) = 0$, so the drop to diet intake drains
them in the first days: 80 g/day of carbohydrate against a habitual 260 g sends
$G_c$ toward $-0.54$ kg. The reference is fixed rather than each episode's mean
intake, because a mean hides that drop and moves as days are added.

## The assembled model {#assembled}

With state $x = (T, F, G_c, G_n, A)$ and daily input
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
complex (4.1 kcal/g of glycogen, diluted by 3 g of water).

## Parameters and priors {#parameters}

Nine parameters are estimated, each through a transform of a standard-normal
coordinate $\eta$ ([reparameterization](estimation.md#reparameterization)):
"log, $s$" means $\theta = c\,e^{s\eta}$, "logit, $s$" means
$\theta = \sigma(\operatorname{logit} c + s\eta)$, and "normal, $s$" means
$\theta = c + s\eta$.

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


## Unlogged days and timescales {#timescales}

An unlogged day assumes intake equal to expenditure, habitual carbohydrate and
sodium, and inflated process noise ([process noise](estimation.md#process-noise)).

The water compartments move in 1 to 2 days, adaptation in about 14, tissue over
months. Instantaneous adaptation would be confounded with an error in $c_0$; the
transient after a change in intake is the whole basis for estimating $\beta_A$.

## The observation model {#observation}

**The scale** reads total mass plus noise on the morning of day $k$:

$$
y_k = H x_k + v_k, \qquad
H = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \end{pmatrix}, \qquad
v_k \sim \mathcal{N}(0, R).
$$

$R$ is mostly fluid and gut-content variation, not instrument error: an sd near
0.4 to 0.6 kg for consistent morning weights ([how to weigh](../data.md#weighing)),
estimated in [phase 1](estimation.md#phase-1) and gated against typing errors
([noise model](estimation.md#noise-model)).

### The waist tape {#waist}

The scale sees fat only through $\rho_{\mathrm{eff}}$; a tape measure responds to
fat directly. The optional waist reading (cm) is modeled as

$$
y^w_k = a_w + b_w\,F_k + v^w_k, \qquad
v^w_k \sim \mathcal{N}(0, \sigma_w^2), \qquad \sigma_w = 1.5\ \text{cm}.
$$

The offset $a_w$ and slope $b_w$ are personal constants, estimated with the
metabolic parameters. Their priors come from the Lean, Han and Deurenberg (1996)
waist equation for body fat at the starting composition: about 1.2 cm per
kilogram of fat for a 100 kg man with 30 kg of fat, offset sd 7.2 cm (11.2 cm for
women). The tape anchors the fat level but not the fat share
([what the tape identifies](estimation.md#waist-update)).

## How firm each piece is {#firmness}

Energy balance is certain and the tissue energy densities and the existence of
adaptation are strong. Forbes partitioning, the water couplings and the
thermic-effect magnitudes are moderate; protein sparing and the adaptation
parameters are weak; Mifflin-St Jeor, Lean and the habitual intakes only center
priors. A wide posterior on a weakly founded parameter is the estimator correctly
reporting that neither the literature nor the data has much to say.

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
- Westerterp, K. R. (2004). Diet induced thermogenesis. *Nutrition & Metabolism*.
