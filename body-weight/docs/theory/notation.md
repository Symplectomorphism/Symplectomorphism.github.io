---
title: "Notation"
---

The symbols used across the theory pages, the slides and the walkthrough. Code
names for the states and parameters are in the [model](model.md#states) tables.
A few letters have two jobs; the [last section](#reused) lists them.

## Conventions {#conventions}

Mass in kilograms, energy in kilocalories, time in days; pounds and inches
appear only at the input and output boundary. Subscript $k$ indexes days, and
day $k$'s state is its morning state; subscript 0 marks a starting value. A hat
denotes an estimate, a dot a time derivative, a star a target, and a $\Delta$
before a symbol a change or deviation; standing alone, $\Delta$ is the distance
to the target.

## Model {#model}

| Symbol | Meaning | Units |
|:--|:------|:--|
| $T$, $F$, $M$ | Tissue, fat and lean mass ($M = T - F$) | kg |
| $G_c$, $G_n$ | Glycogen water and sodium water, deviations from habitual levels | kg |
| $A$ | Adaptive thermogenesis ($\le 0$ in a deficit) | kcal/day |
| $W$ | Scale weight, $T + G_c + G_n$ | kg |
| $x$, $u$ | State $(T, F, G_c, G_n, A)$; the day's inputs | |
| $c_0$, $c_1$ | Expenditure intercept (absorbs constant log bias) and mass slope | kcal/day; per kg |
| $\beta_A$, $\tau_A$ | Adaptation gain and time constant | ; days |
| $k_G$, $\tau_G$ | Glycogen-water gain and time constant | kg per g/day; days |
| $k_{Na}$, $\tau_{Na}$ | Sodium-water gain and time constant | kg per mg/day; days |
| $\kappa_P$ | Fat-share shift per g/kg/day of protein above 1.6 | |
| $a_w$, $b_w$ | Waist offset and slope (cm of waist per kg of fat) | cm; cm/kg |
| $E$, $E_{\mathrm{base}}$ | Total expenditure; expenditure without adaptation | kcal/day |
| $\mathrm{TEF}$, $\alpha_P, \alpha_C, \alpha_F, \alpha_A$ | Thermic effect of food and its per-macronutrient coefficients | |
| $D$ | Energy gap $E_{\mathrm{base}} - I$: drives adaptation, and is the controller's deficit | kcal/day |
| $p$, $p_{\mathrm{Forbes}}$ | Fat share of a tissue change; its Forbes baseline $F/(F + 10.4)$ | |
| $\rho_F$, $\rho_L$, $\rho_G$ | Energy density of fat, fat-free tissue and the glycogen-water complex | kcal/kg |
| $\rho_{\mathrm{eff}}$ | Energy cost of a kilogram of tissue, $p\rho_F + (1 - p)\rho_L$ | kcal/kg |
| $T^{\mathrm{s}}$ | Settled tissue, $T + (\rho_G/\rho_{\mathrm{eff}})\,G_c$ ([control](control.md#settled)) | kg |

## Inputs {#inputs}

| Symbol | Meaning | Units |
|:--|:------|:--|
| $I$ | Metabolizable energy intake | kcal/day |
| $P_g$, $C$, $F_g$, $A_g$ | Protein, carbohydrate, fat and alcohol eaten | g/day |
| $\bar C$, $\Delta C$ | Habitual carbohydrate intake; $C - \bar C$ | g/day |
| $\mathrm{Na}$, $\overline{\mathrm{Na}}$, $\Delta\mathrm{Na}$ | Sodium; its habitual level; the deviation | mg/day |
| $X$ | Logged exercise | kcal/day |
| $q$ | Under-reporting fraction of logged intake | |
| $m_k$, $o_k$ | 1 if day $k$ has a food log; 1 if it has a weigh-in | |

## Estimation {#estimation}

| Symbol | Meaning |
|:--|:------|
| $y_k$, $y^w_k$ | Scale reading (kg) and waist reading (cm) on morning $k$ |
| $H$ | Scale observation row $(1, 0, 1, 1, 0)$ |
| $R = \sigma_v^2$, $\sigma_w$ | Scale noise variance; waist noise sd (1.5 cm) |
| $Q_k$, $\sigma_I$, $s_k$ | Process-noise covariance; intake spread on an unlogged day (600 kcal); an estimated day's uncertainty |
| $f$, $g$ | One-day step map; the vector field it integrates |
| $\eta$, $\eta_w$ | Standardized metabolic parameters (9) and waist calibration (2) |
| $\theta$; $c$, $s$ | Parameters in natural units; the center and scale of their prior map |
| $z_k$ | Augmented state $(x_k, \eta, \eta_w)$, 14 or 16 entries |
| $\sigma_\eta$ | Daily random-walk scale on $\eta$ (0.02 prior sigmas) |
| $\hat x_{k \mid j}$, $P$ | Estimate of $x_k$ given readings through day $j$; a covariance |
| $e_k$, $S_k$ | Innovation and its predicted variance |
| $\alpha$, $\beta$, $\kappa$, $\lambda$ | Unscented-transform tuning (0.3, 2, 0) and its derived scaling |
| $W_m$, $W_c$ | Sigma-point weights for the mean and covariance |
| $\ell$, $b$ | Phase-1 trend: weight level and daily slope |
| $\sigma_\ell, \sigma_b, \sigma_v$ | Phase-1 noise sds: level, slope, observation |
| $n_\sigma$, $R_k^{\mathrm{eff}}$ | Gate threshold (4 sds); the gated observation variance |
| $\nu$ | Student-$t$ degrees of freedom (optional noise model) |
| $\tau_{\mathrm{forget}}$ | Prior-relaxation time between episodes (730 days) |
| $h$, $j$ | Backtest horizon and origin day |

## Control {#control}

| Symbol | Meaning |
|:--|:------|
| $H$, $B$ | Horizon length (days); number of week-long decision blocks |
| $\mathbf{d}$ | Decisions: energy, protein per kg and fat share for each block |
| $W_0$, $T_0$ | Scale weight and tissue at the start of the plan |
| $W^\star$, $W_h$ | Target weight (deadline mode); held weight (maintain mode) |
| $T_k$, $T_H$ | Predicted settled tissue on plan day $k$; at the horizon's end |
| $r_k$, $r_H$ | Reference path; its value at the horizon's end |
| $J$ | Objective |
| $w_T, w_t, w_p, w_s, w_e, w_c, w_\phi$ | Cost weights: terminal, tracking, progress, smoothness, effort, carbohydrate aversion, macro share |
| $\Delta I_k$, $D_k$ | Day-to-day intake change; deficit |
| $\phi_F$; $\phi$, $\phi^\star$, $\bar\phi$ | Fat share of energy; the targeted share, its target, the default midpoint |
| $I_{\min}$, $I_{\max}$, $P_{\min}$, $C_{\min}$ | Calorie bounds; protein floor (g/kg); carbohydrate floor (g/day) |
| $\gamma$ | Permitted weekly tissue loss as a fraction of tissue |
| $\mathbf{g}$ | Trajectory constraints, $\mathbf{g} \ge 0$ |
| $n_{\min}$ | Fewest days to the target at the maximum rate |

## Symbols with two jobs {#reused}

| Symbol | Uses |
|:--|:------|
| $\alpha$, $\beta$, $\kappa$ | Bare: unscented-transform tuning. Subscripted ($\alpha_P$, $\beta_A$, $\kappa_P$): model coefficients. |
| $b$ | Phase-1 slope; the unlogged-intake direction in $Q_k$. $b_w$: waist slope. |
| $s$ | Time within a day; the sex constant in Mifflin-St Jeor; a prior scale. |
| $F$ | Fat mass; $F_g$: fat grams eaten. |
| $H$ | Observation row in the filter; horizon length in the controller. |
| $W$ | Scale weight; $W_m$, $W_c$: sigma-point weights. |
| $P$ | Covariance; $P_g$: protein grams. |
| $\phi$ | Thermic fraction in the weight-stable intake; the controller's macro share. |
