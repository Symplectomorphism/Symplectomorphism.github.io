---
title: "Notation"
---

Every symbol used on more than one of the theory pages is defined here, with the
name it carries in the `wdm` code where there is one. A few letters have two
jobs; the [last section](#reused) lists them.

## Conventions {#conventions}

Mass in kilograms, energy in kilocalories, time in days; pounds and inches
appear only at the input and output boundary. Subscript $k$ indexes days, and
day $k$'s state is its morning state; subscript 0 marks a starting value
($T_0$, $W_0$). A hat denotes an estimate, a dot a time derivative, a star a
target ($\phi^\star$, $W^\star$), a superscript $(i)$ a sigma point, a
superscript $+$ a value just after a measurement update, and $^{\top}$ a
transpose. A $\Delta$ before a symbol is a change or deviation ($\Delta C$,
$\Delta I$); standing alone it is the distance to the target.

## States {#states}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $T$ | Tissue mass: fat plus fat-free solids and structural water | kg | `tissue_kg` |
| $F$ | Fat mass, a component of $T$ | kg | `fat_kg` |
| $M$ | Lean mass, $T - F$ | kg | `lean_kg` |
| $G_c$ | Glycogen-water compartment, deviation from its level at habitual intake | kg | `gly_water_kg` |
| $G_n$ | Sodium-water compartment, deviation from its level at habitual intake | kg | `na_water_kg` |
| $A$ | Adaptive thermogenesis ($\le 0$ in a deficit) | kcal/day | `adapt_kcal` |
| $W$ | Total mass, $T + G_c + G_n$: what the scale weighs | kg | `scale_kg` |
| $S$ | Stored energy | kcal | |
| $t$, $s$ | Continuous time; time within one day in the step map | days | |

## Parameters {#parameters}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $c_0$ | Mass-independent expenditure (absorbs constant log bias) | kcal/day | `rmr_intercept` |
| $c_1$ | Mass dependence of expenditure | kcal/day per kg | `rmr_slope` |
| $\beta_A$ | Fraction of a sustained deficit offset by adaptation | dimensionless | `at_gain` |
| $\tau_A$ | Adaptation time constant | days | `at_tau` |
| $k_G$ | Glycogen-water gain | kg per g/day | `gly_gain` |
| $\tau_G$ | Glycogen-water time constant | days | `gly_tau` |
| $k_{Na}$ | Sodium-water gain | kg per mg/day | `na_gain` |
| $\tau_{Na}$ | Sodium-water time constant | days | `na_tau` |
| $\kappa_P$ | Shift in fat share per g/kg/day of protein above 1.6 | per g/kg/day | `partition_kappa` |
| $\theta$ | The nine parameters above, in natural units | | `Params` |
| $a_w$ | Waist offset; estimated only when waist readings exist | cm | `waist_offset_cm` |
| $b_w$ | Waist slope: centimeters of waist per kilogram of fat | cm per kg | `waist_slope_cm_per_kg` |

Priors for all of them are in the [parameter table](model.md#parameters).

## Derived model quantities {#derived}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $E$ | Total expenditure, $c_0 + c_1 W + \mathrm{TEF} + X + A$ | kcal/day | `total_expenditure` |
| $E_{\mathrm{base}}$ | Expenditure excluding adaptation | kcal/day | `baseline_expenditure` |
| $\mathrm{TEF}$ | Thermic effect of food | kcal/day | `thermic_effect` |
| $D$ | Energy gap $E_{\mathrm{base}} - I$ driving adaptation; the same quantity is the controller's deficit | kcal/day | |
| $p$ | Fat share of a tissue change, Forbes plus protein shift, clipped to $[0.50, 0.98]$ | dimensionless | `fat_fraction` |
| $p_{\mathrm{Forbes}}$ | Forbes fat share, $F/(F + 10.4)$ | dimensionless | `forbes_fat_fraction` |
| $\rho_{\mathrm{eff}}$ | Energy cost of a kilogram of tissue, $p\rho_F + (1 - p)\rho_L$ | kcal/kg | `effective_energy_density` |
| $c_0 + c_1 W$ | Maintenance energy at weight $W$ (no TEF, no adaptation) | kcal/day | `maintenance_kcal`, `maintenance_interval_kcal` |
| $I^\star$ | Weight-stable intake, $(c_0 + c_1 W)/(1 - \phi)$ at thermic fraction $\phi = 0.10$ | kcal/day | `weight_stable_intake_kcal` |

## Fixed constants {#constants}

| Symbol | Meaning | Value | Code name |
|:--|:------|:--|:--|
| $\rho_F$ | Adipose energy density | 9441 kcal/kg | `RHO_FAT` |
| $\rho_L$ | Fat-free energy density | 1816 kcal/kg | `RHO_LEAN` |
| $\rho_G$ | Glycogen-complex energy density | 1025 kcal/kg | `RHO_GLYCOGEN_COMPLEX` |
| $\alpha_P$ | Thermic effect, protein | 0.25 | `TEF_PROTEIN` |
| $\alpha_C$ | Thermic effect, carbohydrate | 0.075 | `TEF_CARB` |
| $\alpha_F$ | Thermic effect, fat | 0.02 | `TEF_FAT` |
| $\alpha_A$ | Thermic effect, alcohol | 0.15 | `TEF_ALCOHOL` |
| | Thermic effect when only total energy is logged | 10% of intake | `TEF_MIXED_DIET_FRACTION` |
| | Forbes coefficient | 10.4 | `FORBES_COEFFICIENT` |
| | Reference protein intake for the partition shift | 1.6 g/kg/day | `PROTEIN_REF_G_PER_KG` |
| | Floor on $F$ inside Forbes' law | 2 kg | `MIN_FAT_KG` |
| | Physical-activity level scaling the Mifflin-St Jeor prior | 1.45 | `DEFAULT_PAL` |
| | Fiber correction subtracted from logged energy | 2 kcal/g | `KCAL_PER_G_FIBER_OVERCOUNT` |
| $\sigma_w$ | Waist-reading noise standard deviation | 1.5 cm | `WAIST_OBS_SD_CM` |
| $\sigma_I$ | Spread of intake on an unlogged day | 600 kcal/day (default) | `unlogged_intake_sd` |
| $s_k$ | Uncertainty of an estimated day's logged energy | a quarter of it | `intake_sd_kcal` |

## Inputs {#inputs}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $I$ | Metabolizable energy intake | kcal/day | `energy_kcal` |
| $P_g$ | Protein | g/day | `protein_g` |
| $C$ | Carbohydrate (in the controller, the grams prescribed) | g/day | `carb_g` |
| $\bar C$ | Habitual (pre-diet) carbohydrate intake | g/day | `habitual_carb_g`, `carb_ref_g` |
| $\Delta C$ | $C - \bar C$ | g/day | `carb_dev_g` |
| $F_g$ | Fat | g/day | `fat_g` |
| $A_g$ | Alcohol | g/day | `alcohol_g` |
| $\mathrm{Na}$ | Sodium | mg/day | `sodium_mg` |
| $\overline{\mathrm{Na}}$ | Habitual (pre-diet) sodium intake | mg/day | `habitual_sodium_mg`, `sodium_ref_mg` |
| $\Delta\mathrm{Na}$ | $\mathrm{Na} - \overline{\mathrm{Na}}$ | mg/day | `sodium_dev_mg` |
| $X$ | Logged exercise | kcal/day | `exercise_kcal` |
| $q$ | Constant under-reporting fraction of logged intake | dimensionless | |
| $u$, $u_k$ | Daily input vector; the food of day $k$, eaten after the morning-$k$ weigh-in | | `DailyInput` |
| $m_k$ | 1 if a food log exists for day $k$ | | `intake_known` |
| $o_k$ | 1 if a weigh-in exists for day $k$ | | `weight_observed` |

The CSV columns behind these are described in [Data](../data.md).

## Observations and noise {#observations}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $y_k$ | Scale reading on the morning of day $k$ | kg | `weight_kg` (CSV `weight_lb`) |
| $y^w_k$ | Waist reading on day $k$ | cm | `waist_cm` (CSV `waist_in` or `waist_cm`) |
| $H$ | Observation row $(1, 0, 1, 1, 0)$ | | `H` |
| $v_k$, $R$ | Scale noise and its variance, $R = \sigma_v^2$ | kg, kg² | `obs_sd` |
| $v^w_k$ | Waist-reading noise, standard deviation $\sigma_w$ | cm | |
| $w_k$, $Q_k$ | Process noise and its covariance | | `process_noise` |
| $\sigma_T, \sigma_F, \sigma_G, \sigma_N, \sigma_A$ | Per-day process-noise standard deviations of the five states (defaults 0.08, 0.04, 0.15, 0.10 kg and 3.0 kcal/day) | state units | `tissue_sd`, `fat_sd`, `gly_sd`, `na_sd`, `adapt_sd` |
| $b$ | Direction of the unlogged-intake inflation, $\rho_{\mathrm{eff}}^{-1}(1, p, 0, 0, 0)^{\top}$ | | |
| $\nu$ | Student-$t$ degrees of freedom, when that noise model is selected; the pipeline is Gaussian | | `dof` (`--student-t`) |

## Estimation {#estimation}

| Symbol | Meaning | Dimension | Code name |
|:--|:------|:--|:--|
| $x$, $x_k$ | Physiological state $(T, F, G_c, G_n, A)$ on the morning of day $k$ | 5 | `State` |
| $\eta$ | Unconstrained (standardized) metabolic parameters | 9 | `eta_mean` |
| $\eta_w$ | Unconstrained waist calibration ($a_w$, $b_w$) | 2 | `waist_eta_mean` |
| $z$, $z_k$ | Augmented state $(x_k, \eta)$, or $(x_k, \eta, \eta_w)$ with waist readings | 14 or 16 | |
| $\sigma_\eta$ | Per-day random-walk scale on $\eta$, in prior-sigma units (0.02) | | `PARAM_DRIFT_PER_DAY` |
| $\hat x_{k \mid j}$, $P_{k \mid j}$ | Mean and covariance of $x_k$ given $y_{1:j}$ | | |
| $P_\eta$ | Covariance of the parameter block, carried between episodes | $9 \times 9$ | `eta_cov` |
| $e_k$, $S_k$ | Innovation and its predicted variance | | `innovation`, `innovation_var` |
| $z_k$ | Standardized innovation, $e_k/\sqrt{S_k}$ | | |
| $K_k$ | Kalman gain | | |
| $F$, $F_k$ | Transition matrix of a linear model; its Jacobian in the extended filter | $5 \times 5$ | |
| $J_k$, $C_k$ | Smoother gain; sigma-point cross-covariance used in its place | | `cross_cov` |
| $\ell$, $b$ | Local linear trend model: weight level, daily slope | | |
| $\sigma_\ell, \sigma_b, \sigma_v$ | Local-linear-trend noise standard deviations: level, slope, observation | | `level_sd`, `slope_sd`, `obs_sd` |
| $n$ | Dimension of the belief being sigma-pointed | | |
| $L$ | Cholesky factor of the belief covariance | $n \times n$ | |
| $\alpha$, $\beta$, $\kappa$ | Unscented-transform tuning: spread, prior-knowledge correction, secondary spread (0.3, 2, 0) | | `SigmaPointSpec` |
| $\lambda$ | Derived scaling, $\alpha^2(n + \kappa) - n$ | | |
| $\mathcal{X}^{(i)}$, $\mathcal{Y}^{(i)}$ | Sigma points; their images under the step map | | |
| $W_m^{(i)}$, $W_c^{(i)}$ | Sigma-point weights for the mean and covariance | | |
| $\mathcal{Z}^{(i)}$ | Sigma points mapped through the waist observation | | |
| $f$, $g$ | One-day step map; the continuous-time vector field it integrates | | `step`, `rate` |
| $c$, $s$ | Center and scale of a parameter's reparameterization map | | `Transform.center`, `Transform.scale` |
| $w$, $\Delta t$, $\tau_{\mathrm{forget}}$ | Prior weight when relaxing across a gap; the gap in days; the relaxation time constant (730 days) | | `FORGET_TAU_DAYS` |
| $n_\sigma$ | Gross-error gate threshold, in predicted standard deviations (4) | | `gate_sigma` |
| $R_k^{\mathrm{eff}}$ | Observation variance after gating or Student-$t$ reweighting | kg² | |
| $\omega$ | Latent precision multiplier of the Student-$t$ scale mixture | | |
| $\varepsilon$ | Intake error on an unlogged day | kcal/day | |
| $h$ | Forecast horizon in the backtest (default 1 and 7 days) | days | `horizons` |
| $j$ | Origin day in the backtest; conditioning index in $\hat x_{k \mid j}$ | | |

## Control {#control}

| Symbol | Meaning | Units | Code name |
|:--|:------|:--|:--|
| $W_k$, $W_H$ | Predicted scale weight on plan day $k$; at the horizon's end | kg | `predicted_weight_kg` |
| $T_k$, $T_H$ | Predicted tissue mass on plan day $k$; at the horizon's end (what the rate limit and the objective score) | kg | `predicted_tissue_kg` |
| $W_0$, $T_0$ | Scale weight and tissue at the start of the plan | kg | `start_weight_kg` |
| $W^\star$ | Target scale weight | kg | `target_weight_lb` (in lb) |
| $r_k$, $r_H$ | Reference path toward the goal; its value at the horizon's end | kg | `reference_trajectory` |
| $J$ | Controller objective | | |
| $\Delta I_k$ | Day-to-day change in prescribed intake | kcal/day | |
| $D_k$ | Deficit, expenditure without adaptation minus prescribed intake | kcal/day | |
| $\mathbf{d}$ | Decision vector: energy, protein per kg and fat share for each of the $B$ blocks | $3B$ | |
| $\tilde d$; $d_{\min}$, $d_{\max}$ | A decision scaled to $[0, 1]$ across its bounds; the bounds | | |
| $B$, $b$ | Number of decision blocks; block index | | `n_blocks` |
| $H$ | Horizon length; as a subscript, its last day | days | `horizon_days` |
| $I_{\min}$, $I_{\max}$ | Bounds on prescribed energy | kcal/day | `min_kcal`, `max_kcal` |
| $P_{\min}$ | Protein floor, per kg of the starting weight | g/kg/day | `min_protein_g_per_kg` |
| $C_b$, $C_{\min}$ | Carbohydrate in block $b$; the carbohydrate floor | g/day | `min_carb_g` |
| $\phi_F$ | Fat share of energy, a decision | dimensionless | `min_fat_fraction`, `max_fat_fraction` |
| $\phi$, $\phi^\star$ | Energy share of the targeted macronutrient; its target | dimensionless | `target_carb_fraction`, `target_fat_fraction` |
| $\bar\phi$ | Midpoint of the fat-share bounds, the default target | dimensionless | |
| $\gamma$ | Permitted weekly loss as a fraction of tissue, applied daily as $T_{k-1} - T_k \le \gamma T_k/7$ | per week | `max_loss_frac_per_week` |
| $\mathbf{g}$ | Inequality constraints $\mathbf{g} \ge 0$: one rate row per day, one carbohydrate row per block | | |
| $\Phi$ | Weighted total constraint violation, used by the fallback | | |
| $\mu$ | Escalating penalty multiplier on $\Phi$; zero for SLSQP | | `penalty_scale` |
| $w_T, w_t, w_p, w_s, w_e, w_c$ | Cost weights: terminal, tracking, progress, smoothness, effort, carbohydrate aversion | | `CostWeights` fields |
| $w_\phi$ | Macro-share weight: 0.1 as a tie-breaker, 100 for a stated target | | `NEUTRAL_FAT_FRAC_WEIGHT`, `FAT_TARGET_WEIGHT`, `CARB_TARGET_WEIGHT` |
| $\Delta$, $n_{\min}$ | Distance from the current scale weight to the target; fewest days to cover it at the maximum rate | kg, days | |

## Symbols with two jobs {#reused}

Within the section that defines it, every symbol above has one meaning. These
letters also have a second job; a subscript, a typeface or the context
separates them.

| Symbol | Uses |
|:--|:------|
| $\alpha$, $\beta$, $\kappa$ | Bare: unscented-transform tuning. Subscripted ($\alpha_P$, $\beta_A$, $\kappa_P$, ...): thermic-effect and model coefficients. |
| $b$ | Local-linear-trend slope; unlogged-intake direction; block index in the controller. $b_w$: waist slope. |
| $s$ | Time within a day; the sex constant in Mifflin-St Jeor; a reparameterization scale. |
| $z$ | $z_k$ as a vector: augmented state. $z_k$ as a scalar: standardized innovation. |
| $g$ | Italic: the continuous-time vector field. Bold $\mathbf{g}$: the controller's constraints. |
| $F$ | Fat mass; transition matrix $F$, $F_k$ in the generic filter equations; $F_g$: fat grams eaten. |
| $H$ | Observation row in the filter; horizon length in the controller. |
| $C$ | Carbohydrate grams; $C_k$: smoother cross-covariance. |
| $W$ | Total mass; $W_m^{(i)}$, $W_c^{(i)}$: sigma-point weights. |
| $P$ | Covariance; $P_g$: protein grams. |
| $w$ | $w_k$: process noise; bare $w$: prior weight across a gap; $w_T, \dots, w_\phi$: cost weights. |
| $\sigma$ | Subscripted: a standard deviation. As a function: the logistic sigmoid in the logit map. |
| $c$, $J$ | $c_0$, $c_1$: expenditure coefficients; bare $c$: a reparameterization center. $J$: controller objective; $J_k$: smoother gain. |
| $\phi$ | The thermic fraction in the weight-stable intake formula; the controller's macro energy share. |
| $j$ | Backtest origin day; conditioning index in $\hat x_{k \mid j}$. |

$D$, the energy gap that drives adaptation, and the controller's deficit are the
same physical quantity, not a reuse.
