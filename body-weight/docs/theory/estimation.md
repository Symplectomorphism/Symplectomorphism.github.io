---
title: "Estimation"
metadata-files: [../numbers_synthetic.yml]
---

The [model](model.md) is continuous and deterministic; the data are daily, noisy
and full of gaps. Estimation runs in two phases: phase 1 identifies the noise
from weight history alone, and phase 2 runs a joint unscented Kalman filter over
the state and nine parameters (Särkkä, 2013, covers the theory).

## Discretization {#discretization}

Each day's inputs are held constant over the day, which makes the one-day step
an exact zero-order-hold discretization:

$$
x_{k+1} = f(x_k, u_k) = x_k + \int_0^1 g\bigl(x(s), u_k\bigr)\,\mathrm{d}s ,
$$

with $g$ the vector field of the [assembled model](model.md#assembled),
integrated by RK4 with four substeps per day and tested against an adaptive
solver (Tsitouras, 2011; Kidger, 2021).

## Timing: the morning state and the day's food {#timing}

A row of the [data file](../data.md) holds the weigh-in on morning $k$ and the
food eaten after it, so $x_k$ is the morning state and day $k$'s food moves it to
morning $k+1$: Monday's dinner first shows on Tuesday's scale. Where a plan then
starts is a calendar question for the [controller](control.md#plan-start).

## Process noise {#process-noise}

Process noise stands for unlogged snacks, log error and activity changes:

$$
Q_k = \operatorname{diag}(\sigma_T^2, \sigma_F^2, \sigma_G^2, \sigma_N^2, \sigma_A^2)
    + \bigl[(1 - m_k)\,\sigma_I^2 + m_k\, s_k^2\bigr]\, b b^{\top},
\qquad
b = \frac{1}{\rho_{\mathrm{eff}}}\,(1, p, 0, 0, 0)^{\top},
$$

where $m_k$ marks a food log, $\sigma_I = 600$ kcal/day is the intake spread on
an unlogged day, and $s_k$ is a quarter of the energy of a day logged as an
[estimate](../data.md#estimating) (zero otherwise). The inflation is rank one
because an unknown intake moves tissue and fat together.

## Missing data and episodes {#missing-data}

A missing weigh-in skips the update, which for data missing at random costs
precision but adds no bias (Durbin and Koopman, 2012). Separate diet attempts
(episodes) share the metabolism but not the state ([pooling](#pooling)).

## The filter {#filter}

In the Kalman filter (Kalman, 1960), the innovation
$e_k = y_k - H\hat x_{k \mid k-1}$ is the surprise in today's reading and $S_k$
its predicted variance; together they give the log-likelihood that all
parameter estimation maximizes:

$$
\log p(y_{1:n} \mid \theta) = -\frac{1}{2}\sum_{k:\,o_k = 1}
  \left[ \log 2\pi + \log S_k + \frac{e_k^2}{S_k} \right],
$$

with $o_k = 1$ on days with a weigh-in.

The dynamics are nonlinear, so the time update is unscented (Julier and
Uhlmann, 2004; Wan and van der Merwe, 2000): $2n + 1$ sigma points are pushed
through the exact step map, with $\alpha = 0.3$, $\beta = 2$, $\kappa = 0$ (van
der Merwe, 2004). The scale reading is linear in the state, so its update is
exact. Figures show the Rauch-Tung-Striebel smoothed history (Rauch et al.,
1965); the [controller](control.md) starts from the filtered state.

## Phase 1: noise from the weight record {#phase-1}

Weight alone cannot identify metabolism, but it can say how noisy a reading is
and how fast the trend wanders, and their ratio sets how hard to smooth. Phase 1
fits the local linear trend model
(Harvey, 1989),

$$
\begin{pmatrix} \ell_{k+1} \\ b_{k+1} \end{pmatrix}
= \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
  \begin{pmatrix} \ell_k \\ b_k \end{pmatrix} + w_k,
\qquad y_k = \ell_k + v_k,
$$

by maximum likelihood over $(\sigma_\ell, \sigma_b, \sigma_v)$, on every
episode, logged or not. Phase 2 takes $\sigma_v$ from it. The fit also reports
the steady-state gain, the fraction of each surprise the filter acts on, whose
reciprocal is the effective averaging window in days. On the synthetic subject
it finds $\sigma_v = {{< meta sy.ObsSd >}}$ kg against a true 0.45
([walkthrough step 3](../walkthrough/03-phase1.md)).

## Phase 2: joint state and parameter estimation {#phase-2}

Phase 2 appends the parameters to the state and filters everything together:

$$
z_k = \bigl(\underbrace{T, F, G_c, G_n, A}_{x_k},\;
            \underbrace{\eta}_{\in \mathbb{R}^9},\;
            \underbrace{\eta_w}_{\in \mathbb{R}^2}\bigr) \in \mathbb{R}^{16}.
$$

The waist block exists only when some episode has a waist reading.

### Reparameterization {#reparameterization}

A sigma point can land anywhere, and a negative time constant makes the
dynamics explode, so the filter works in coordinates $\eta$ with prior
$\mathcal{N}(0, \mathbb{I})$, mapped to natural units by the log, logit and
affine maps of the [parameter table](model.md#parameters).

$\eta$ follows a slow random walk ($\sigma_\eta = 0.02$ prior sigmas per day),
so the filter never goes deaf to evidence that its early estimate, from the
least reliable logs, was wrong.

### Which episodes are filtered {#which-episodes}

Phase 2 filters episodes with a food log, plus the latest episode, from which
planning starts. Past weight-only diets feed phase 1 only: with intake assumed
equal to expenditure, a past diet would look like a run of same-signed
surprises, explicable only by bending parameters.

### Pooling across episodes {#pooling}

Each episode starts from its first weigh-in, with fat from the configured
body-fat fraction or the Deurenberg et al. (1991) prediction. The parameter
posterior carries forward as the next episode's prior, its covariance relaxed
toward the prior across the gap $\Delta t$ (days):

$$
P_\eta \leftarrow (1 - w) P_\eta + w P_\eta^{\mathrm{prior}},
\qquad w = 1 - e^{-\Delta t / \tau_{\mathrm{forget}}},
\qquad \tau_{\mathrm{forget}} = 730\ \text{days}.
$$

Elapsed time makes the estimate less certain, not wrong. The reported "shift"
is the relative move of each posterior median from its prior center; near zero
means the data said nothing.

## Noise model: Gaussian with a 4-sigma gate {#noise-model}

So that a mistyped weight does not count as a real gain, the gate inflates the
observation variance beyond $n_\sigma = 4$ predicted standard deviations:

$$
R_k^{\mathrm{eff}} = R \cdot
  \max\!\left(1, \frac{\lvert e_k \rvert}{n_\sigma \sqrt{S_k}}\right)^{2}.
$$

A sustained shift still moves the estimate. A Student-$t$ alternative (Lange et
al., 1989) is in `wdm backtest --student-t 4`; on the real record it fits worse.

## The waist update {#waist-update}

The tape prediction $a_w + b_w F$ multiplies two uncertain entries, so its
update is unscented, after the scale update. Between two readings,
$\Delta y^w \approx b_w\, p\, \Delta W$, so the data constrain only the product
$b_w p$: a 20% error in the fat share can hide behind a 20% error in the slope.
What the tape adds is an independent anchor for the fat level,
$F = (y^w - a_w)/b_w$; on the synthetic subject it narrows the fat-mass interval
by {{< meta sy.TapeNarrowPct >}}% ([walkthrough step 5](../walkthrough/05-waist.md)).

## Identifiability {#identifiability}

Read this before quoting any parameter.

| Status | Quantity | Why |
|:--|:--|:------|
| Well identified | Maintenance $c_0 + c_1 W$ | Each coefficient is wide over one dataset's narrow mass range, but their combination is well determined. |
| Well identified | $k_G$, $\tau_G$ | Carbohydrate varies daily and the water swings are large next to the noise. |
| Weak | $\beta_A$, $\tau_A$ | Separable from an error in $c_0$ only through the lag; a diet break or refeed week helps. |
| Weak | $k_{Na}$ | Everyday sodium swings move the scale less than one weigh-in's noise; logging sodium helps. |
| Weak | Fat mass $F$ | Seen only through $\rho_{\mathrm{eff}}$, which a 15 kg loss changes by under 10%. |
| Not identified | $\kappa_P$ | A second-order effect through $\rho_{\mathrm{eff}}$. |
| Not identified | Thermic-effect coefficients | Held fixed: their effect looks like a logging bias, which $c_0$ absorbs. |

::: {.callout-note title="Quote maintenance, not its coefficients"}
Quoting $c_0$ or $c_1$ alone overstates the uncertainty in what the subject can
eat; `maintenance_interval_kcal` carries their full covariance.
:::

## Validation on synthetic data {#synthetic}

`wdm demo` simulates a subject with true parameters deliberately offset from the
priors (if the truth sat at the prior center, an estimator that ignored the data
would pass). {{< meta sy.HitsWord >}} true values lie inside their 90% posterior intervals,
and maintenance at 100 kg is estimated at {{< meta sy.MaintMed >}} kcal/day against a true
{{< meta sy.TrueMaint >}}. This validates the inference, not the physiology; the
[walkthrough](../walkthrough/index.qmd) takes the same run apart step by step.

## Backtest: forecasting readings not yet seen {#backtest}

`wdm backtest` replays each phase-2 episode day by day. For every origin day $j$
and horizon $h$ (default 1 and 7 days) whose target morning has a weigh-in, it
forecasts that reading from the filtered belief on morning $j$ and the food
logged since, which separates model error from adherence. The scores are bias,
RMSE, skill against persistence (tomorrow equals the last reading), which a
model must beat at a week to earn its complexity, calibration (standardized-error
sd and 50/90% coverage) and log predictive density. On the synthetic subject the
skill is {{< meta sy.BtOneSkill >}}% at a day and {{< meta sy.BtSevenSkill >}}% at a week
([walkthrough step 6](../walkthrough/06-backtest.md)).

On real data the first mornings of a diet drop more than the water compartments
explain, most likely gut contents, which no state represents.

**Caveats.** Phase-1 noise is fitted on the whole record, so coverage is slightly
optimistic. Forecasts from neighboring origins overlap (at $h = 7$, consecutive
forecasts share six days of inputs), so the effective sample is several times
smaller than $n$.

## References {#references}

- Deurenberg, P., Weststrate, J. A., and Seidell, J. C. (1991). Body mass index as a measure of body fatness: age- and sex-specific prediction formulas. *British Journal of Nutrition*.
- Durbin, J., and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Julier, S. J., and Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. *Proceedings of the IEEE*, 92, 401-422.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82, 35-45.
- Kidger, P. (2021). *On Neural Differential Equations*. PhD thesis, University of Oxford.
- Lange, K. L., Little, R. J. A., and Taylor, J. M. G. (1989). Robust statistical modeling using the *t* distribution. *Journal of the American Statistical Association*, 84(408), 881-896.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3, 1445-1450.
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.
- Tsitouras, C. (2011). Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption. *Computers & Mathematics with Applications*.
- van der Merwe, R. (2004). *Sigma-point Kalman filters for probabilistic inference in dynamic state-space models*. PhD thesis, Oregon Health & Science University.
- Wan, E. A., and van der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation. In *Proceedings of the IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium*.
