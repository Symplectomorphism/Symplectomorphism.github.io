---
title: "Estimation"
metadata-files: [../numbers_synthetic.yml]
---

The [model](model.md) is continuous and deterministic; the data are daily, noisy
and full of gaps. This page makes the model a discrete stochastic state-space
system, then estimates it in two phases: phase 1 identifies the noise from
weight history alone, and phase 2 runs a joint unscented Kalman filter over the
state and nine parameters. It ends with what the data can identify, checks on
synthetic data, and a forecast backtest. Särkkä (2013) covers the filtering
theory in depth.

## Discretization {#discretization}

Food logs report daily totals, so each day's inputs are held constant over the
day, which makes the one-day step an exact zero-order-hold discretization:

$$
x_{k+1} = f(x_k, u_k) = x_k + \int_0^1 g\bigl(x(s), u_k\bigr)\,\mathrm{d}s ,
$$

with $g$ the vector field of the [assembled model](model.md#assembled). The
integral uses fixed-step RK4 with four substeps per day; the fastest mode, the
1.5-day glycogen lag, is six substeps long. A test requires the RK4 map (`step`)
to agree with an adaptive Tsit5 reference (`simulate_diffrax`; Tsitouras, 2011;
Kidger, 2021) to $10^{-5}$ (measured: $2.5 \times 10^{-7}$), so that a
discretization error cannot pass for a physiological finding.

## Timing: the morning state and the day's food {#timing}

Each row of the [data file](../data.md) holds the weigh-in on the morning of day
$k$ and the food eaten during day $k$, after it. The state $x_k$ is the morning
state, so day $k$'s food moves morning $k$ to morning $k+1$:

$$
x_{k+1} = f(x_k, u_k), \qquad y_k = H x_k + v_k .
$$

Monday's reading reflects food through Sunday; Monday's dinner first shows on
Tuesday's scale. The filter shifts the inputs by a day (`previous_day_inputs`)
so that step $k$ propagates with day $k-1$'s food and then applies morning $k$'s
reading, and step 0 is an update only (`predict_first=False`). Where a plan
starts when today's food is partly logged is a calendar question for the
[controller](control.md#plan-start).

## Process noise {#process-noise}

Process noise stands for unlogged snacks, hydration swings, log error and
changes in spontaneous activity. Standard deviations are per day, so a week
without weigh-ins widens the band by $\sqrt{7}$, not 7:

$$
Q_k = \operatorname{diag}(\sigma_T^2, \sigma_F^2, \sigma_G^2, \sigma_N^2, \sigma_A^2)
    + \bigl[(1 - m_k)\,\sigma_I^2 + m_k\, s_k^2\bigr]\, b b^{\top},
\qquad
b = \frac{1}{\rho_{\mathrm{eff}}}\,(1, p, 0, 0, 0)^{\top},
$$

where $m_k \in \{0, 1\}$ indicates a food log, $\sigma_I = 600$ kcal/day is
the spread of plausible intake on an unlogged day, and $s_k$ is a logged day's own
uncertainty: zero for a day logged as eaten, a quarter of its energy for a day
marked as an estimate (`wdm log --estimate`), which then informs the energy
balance less than a weighed day. The code's diagonal defaults
are $\sigma_T = 0.08$, $\sigma_F = 0.04$, $\sigma_G = 0.15$, $\sigma_N = 0.10$ kg
and $\sigma_A = 3.0$ kcal/day.

<!-- Source note: the report does not state the diagonal process-noise defaults; they are NoiseModel.default() in models/dynamics.py. -->

The inflation is rank one because an unknown intake moves tissue and fat
together, in the ratio $p$ and with the physical sensitivity
$1/\rho_{\mathrm{eff}}$; independent noise on $T$ and $F$ would let the filter
learn about body composition from a day it knows nothing about.

## Missing data and episodes {#missing-data}

The loader reindexes each episode onto a complete daily calendar, so a missing
day is a missing observation: with $o_k \in \{0, 1\}$ indicating a weigh-in, the
update is skipped when $o_k = 0$ and the predicted belief carries forward. For
data missing at random this is exact (Durbin and Koopman, 2012): gaps cost
precision but add no bias, so never fill in a missed weigh-in (an invented
observation makes the filter overconfident) and never delete an inconvenient one
(it corrupts the noise estimate); mistyped values are left to the
[gate](#noise-model). Separate diet attempts (episodes), possibly years apart, share the subject's
metabolism but not their state, so parameters are shared and states
episode-specific ([pooling](#pooling)).

## The filter {#filter}

For a linear model $x_{k+1} = F x_k + w_k$, $y_k = H x_k + v_k$ with
$w \sim \mathcal{N}(0, Q)$, $v \sim \mathcal{N}(0, R)$, the Kalman filter
(Kalman, 1960) predicts and updates in closed form:

$$
\begin{aligned}
\hat x_{k \mid k-1} &= F \hat x_{k-1 \mid k-1},
& P_{k \mid k-1} &= F P_{k-1 \mid k-1} F^{\top} + Q, \\
e_k &= y_k - H \hat x_{k \mid k-1},
& S_k &= H P_{k \mid k-1} H^{\top} + R, \\
K_k &= P_{k \mid k-1} H^{\top} S_k^{-1},
& \hat x_{k \mid k} &= \hat x_{k \mid k-1} + K_k e_k .
\end{aligned}
$$

The innovation $e_k$ is the surprise in today's reading and $S_k$ its predicted
variance. The covariance update uses the Joseph form,
$P_{k \mid k} = (\mathbb{I} - K_k H) P_{k \mid k-1} (\mathbb{I} - K_k H)^{\top} +
K_k R K_k^{\top}$, a sum of positive semi-definite terms that round-off cannot
drive negative. The same quantities give the log-likelihood that all parameter
estimation maximizes:

$$
\log p(y_{1:n} \mid \theta) = -\frac{1}{2}\sum_{k:\,o_k = 1}
  \left[ \log 2\pi + \log S_k + \frac{e_k^2}{S_k} \right].
$$

**Unscented time update.** The dynamics are nonlinear (partitioning saturates,
and parameters in the state make them bilinear), and linearization (the extended
filter, kept only as a test reference) is first-order accurate. The unscented
transform (Julier and Uhlmann, 2004; Wan and van der Merwe, 2000) instead pushes
$2n + 1$ deterministic points through the exact step map. For a belief
$(\hat x, P)$ of dimension $n$ with Cholesky factor $L$ and
$\lambda = \alpha^2(n + \kappa) - n$,

$$
\mathcal{X}^{(0)} = \hat x, \quad
\mathcal{X}^{(\pm i)} = \hat x \pm \sqrt{n + \lambda}\,L_{:,i}, \qquad
W_m^{(0)} = \frac{\lambda}{n + \lambda}, \quad
W_c^{(0)} = W_m^{(0)} + 1 - \alpha^2 + \beta, \quad
W_m^{(\pm i)} = W_c^{(\pm i)} = \frac{1}{2(n + \lambda)} .
$$

The points reproduce the mean and covariance exactly, and the predicted belief
is the weighted mean and covariance of their images
$\mathcal{Y}^{(i)} = f(\mathcal{X}^{(i)}, u)$, plus $Q_k$. The package uses
$\alpha = 0.3$, $\beta = 2$, $\kappa = 0$; the classic $\alpha = 10^{-3}$ makes
$W_c^{(0)}$ large and negative in 14 or 16 dimensions and can break positive
definiteness (van der Merwe, 2004).

::: {.callout-tip title="The scale update is exact"}
The scale reading is linear in the state and does not involve the parameters, so
the linear update above applies exactly. The filter's only approximations are
the time update and, on days with a tape reading, the waist update.
:::

**Smoothing and numerics.** A Rauch-Tung-Striebel pass (Rauch et al., 1965)
runs backwards to give $p(x_k \mid y_{1:n})$, with the sigma-point
cross-covariance the time update already stores in place of $P_{k \mid k}
F^{\top}$. Figures show the smoothed history; the [controller](control.md) uses
the filtered state at the last step, the only estimate that exists when deciding
what to eat. The code runs in double precision, symmetrizes every covariance
update and adds $10^{-9}$ to the diagonal before each Cholesky factorization;
tests check the linear filter and smoother against a dense batch posterior to
$10^{-8}$ and $10^{-7}$.

## Phase 1: noise from the weight record {#phase-1}

Weight without a food log cannot identify metabolism, but it can say how noisy a
reading is and how fast the trend wanders. Their ratio sets how hard to smooth:
wrong one way and the estimate chases every salty dinner, wrong the other and it
takes weeks to notice a real change. Phase 1 fits the local linear trend model
(Harvey, 1989),

$$
\begin{pmatrix} \ell_{k+1} \\ b_{k+1} \end{pmatrix}
= \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
  \begin{pmatrix} \ell_k \\ b_k \end{pmatrix} + w_k,
\qquad y_k = \ell_k + v_k,
$$

with level $\ell$, daily slope $b$, $Q = \operatorname{diag}(\sigma_\ell^2,
\sigma_b^2)$ and $R = \sigma_v^2$, by maximum likelihood over
$(\sigma_\ell, \sigma_b, \sigma_v)$. Every episode with at least two weigh-ins
contributes, logged or not; episodes share the noise scales, each with its own
diffuse start. The search is L-BFGS-B (Byrd et al., 1995) in log coordinates with
JAX gradients, and it refuses to run on fewer than 10 weigh-ins. Phase 2 takes
$\sigma_v$ from this fit.

Standard errors come from the observed information, reported as multiplicative
factors, except that the code reports "unidentified" when the optimum sits on a
bound, the Hessian's condition number exceeds $10^{10}$, or a variance comes out
non-positive. The first case is common: maximum likelihood often drives
$\sigma_\ell \to 0$, a fit in which the trend moves through changes of slope.
For the same reason the fit reports the steady-state gain on the level (the
fraction of each surprise the filter acts on, from iterating the covariance
recursion to a fixed point) rather than $\sigma_\ell/\sigma_v$. Its reciprocal
is the effective averaging window in days. The trend $\ell$ with its slope $b$
is, on its own, the package's most useful output.

| Synthetic: {{< meta sy.NWeighIns >}} weigh-ins, {{< meta sy.NEpisodes >}} episodes, true $\sigma_v = 0.45$ kg | Estimate | Uncertainty |
|:--|--:|:--|
| Observation sd | {{< meta sy.ObsSd >}} kg | {{< meta sy.ObsSdFactor >}} |
| Level innovation sd | {{< meta sy.LevelSd >}} kg/day | {{< meta sy.LevelSdFactor >}} |
| Slope innovation sd | {{< meta sy.SlopeSd >}} kg/day² | {{< meta sy.SlopeSdFactor >}} |
| Steady-state gain; effective window | {{< meta sy.Gain >}}; {{< meta sy.Window >}} days | |

## Phase 2: joint state and parameter estimation {#phase-2}

Phase 2 appends the parameters to the state and filters everything together:

$$
z_k = \bigl(\underbrace{T, F, G_c, G_n, A}_{x_k},\;
            \underbrace{\eta}_{\in \mathbb{R}^9},\;
            \underbrace{\eta_w}_{\in \mathbb{R}^2}\bigr) \in \mathbb{R}^{16}.
$$

The waist block $\eta_w$ exists only when some episode has a waist reading
($\mathbb{R}^{14}$ otherwise), because the sigma-point spread depends on the
dimension and idle entries would perturb every other estimate. $\eta$ and
$\eta_w$ follow a slow random walk, $\eta_w$ never enters the dynamics, and each
weigh-in updates all of $z_k$ through the cross-covariances the time update
builds.

### Reparameterization {#reparameterization}

Time constants and gains are positive and $\beta_A \in [0, 1)$, but a Gaussian
sigma point can land anywhere, and a negative time constant makes the dynamics
explode. The filter therefore works in unconstrained coordinates and maps to
natural units only inside the dynamics:

| Kind | Map | Used for |
|:--|:--|:----|
| log | $\theta = c\,e^{s\eta}$ | $c_0$, $c_1$, the time constants and gains; the waist slope $b_w$ ($s = 0.2$) |
| logit | $\theta = \sigma(\operatorname{logit} c + s\eta)$ | $\beta_A$ |
| affine | $\theta = c + s\eta$ | $\kappa_P$; the waist offset $a_w$ ($s = 7.2$ cm for men) |

Centers and scales put the prior at $\eta \sim \mathcal{N}(0, \mathbb{I})$ exactly
([parameter table](model.md#parameters), [waist priors](model.md#waist)).

**Slow drift.** A random walk of $\sigma_\eta = 0.02$ per day, in prior-sigma
units, keeps the filter from locking in early. Without it the covariance only
shrinks and the filter goes deaf to later evidence, including evidence that its
early estimate (from the least reliable food logs) was wrong. The fiction is
partly true, since maintenance does drift with season, activity and age. By
design, each posterior then settles where information gain and drift balance; on
synthetic data the 90% width of $c_0$ plateaus at about 0.9 times the prior width
from day 30 to day 180.

### Which episodes are filtered {#which-episodes}

Phase 2 filters episodes with a food log, plus the latest episode, logged or not,
because planning starts from its state. Past weight-only diets feed phase 1
only. On an unlogged day, intake is modeled as unknown around expenditure, so a
past diet that was in fact losing weight looks like a long run of same-signed
surprises, which the filter can explain only by bending parameters; on the
report's record it pulled the glycogen time constant far from its prior. A test
checks that adding a weight-only past episode leaves the metabolic posterior
unchanged.

### Pooling across episodes {#pooling}

Each episode starts from its first weigh-in: $T_0$ is that reading, $F_0$ the
configured body-fat fraction times it or else the Deurenberg et al. (1991)
prediction from BMI, age and sex (standard error about 4 percentage points),
and $G_c = G_n = A = 0$, with initial standard deviations of 4 kg on fat and
5 kcal/day on adaptation.

<!-- Source note: the report says "fat mass from config or Deurenberg"; the formula (1.20 BMI + 0.23 age - 10.8 sex - 5.4, clipped to 5-60 %) and the initial covariance diag(sigma_v^2 + 0.25, 16, 0.25, 0.10, 25) are in estimation/joint.py. -->

The parameter posterior carries forward as the next episode's prior (sequential
Bayes), with its covariance relaxed toward the prior across the gap $\Delta t$
(days):

$$
P_\eta \leftarrow (1 - w) P_\eta + w P_\eta^{\mathrm{prior}},
\qquad w = 1 - e^{-\Delta t / \tau_{\mathrm{forget}}},
\qquad \tau_{\mathrm{forget}} = 730\ \text{days}.
$$

The mean is left alone: elapsed time makes you less certain about your
metabolism, not wrong about it. The waist block is carried and relaxed the same
way. Each filtered episode is then smoothed, and the final parameter summaries
come from 20,000 Monte Carlo draws of the unconstrained posterior mapped to
natural units, since the exponential and logistic maps make the posteriors
skewed. The reported "shift" is the relative move of each posterior median from
its prior center; near zero means the data said nothing
([Reading the output](../reading-output.md)).

## Noise model: Gaussian with a 4-sigma gate {#noise-model}

A weight mistyped as 231 lb instead of 213 lb would otherwise count as a real
18 lb gain, and the filter would revise your metabolism to match. The gate
inflates the observation variance by the square of any exceedance beyond
$n_\sigma = 4$ predicted standard deviations:

$$
R_k^{\mathrm{eff}} = R \cdot
  \max\!\left(1, \frac{\lvert e_k \rvert}{n_\sigma \sqrt{S_k}}\right)^{2}.
$$

It downweights rather than discards, so a sustained shift still moves the
estimate once its first readings have pulled the prediction along, and an honest
Gaussian reading is almost never touched (probability about $6 \times 10^{-5}$).
It guards the waist update too.

**Student-$t$, as an option.** Treating the noise as a Gaussian with a random
precision $\omega \sim \mathrm{Gamma}(\nu/2, \text{rate } \nu/2)$ gives, for a
standardized innovation $z_k = e_k/\sqrt{S_k}$, the one-step approximation

$$
R_k^{\mathrm{eff}} = R\,\frac{\nu + z_k^2}{\nu + 1},
$$

followed by an ordinary update, with a Student-$t$ likelihood term. At
$\nu = 4$ (Lange et al., 1989) a ten-sigma reading keeps about a twentieth of its
weight ($5/104$). It replaces the gate and is available through
`wdm backtest --student-t 4` ([Commands](../commands.md)).

**Why the Gaussian was kept.** Refitting phase 1 on the real record with
heavier tails ($\nu = \infty, 30, 10, 6, 4, 3$), the report finds:

- the likelihood falls monotonically as the tails get heavier;
- the Gaussian's residuals look Gaussian, with kurtosis near 3 and no
  standardized residual beyond 3;
- heavier tails move variance from the readings into the trend and raise the
  steady-state gain, so the filter chases readings, the opposite of smoothing.

The backtest log score agrees, on too few forecasts to decide anything, and the
comparison should be rerun as the record grows. Under Student-$t$, $\sigma_v$
is a scale (the sd is $\sqrt{2}\,\sigma_v$ at $\nu = 4$), so the same $\nu$ must
be used in both phases.

## The waist update {#waist-update}

On a day with a tape reading, the exact scale update comes first and the tape
second. The tape prediction $a_w + b_w F$ multiplies two uncertain entries, so
its update is unscented: fresh sigma points from the post-scale belief
$(\hat z^{+}, P^{+})$ give predicted readings $\mathcal{Z}^{(i)}$, and

$$
\begin{aligned}
\hat y^w_k &= \textstyle\sum_i W_m^{(i)} \mathcal{Z}^{(i)}, &
S^w_k &= \max\Bigl(0, \textstyle\sum_i W_c^{(i)} (\mathcal{Z}^{(i)} - \hat y^w_k)^2\Bigr) + \sigma_w^2, \\
K^w_k &= \textstyle\sum_i W_c^{(i)} (\mathcal{X}^{(i)} - \hat z^{+})(\mathcal{Z}^{(i)} - \hat y^w_k) / S^w_k, &
\hat z_{k \mid k} &= \hat z^{+} + K^w_k (y^w_k - \hat y^w_k), \\
P_{k \mid k} &= P^{+} - S^w_k K^w_k K^{w\top}_k .
\end{aligned}
$$

The floor at zero guards against a negative central weight. The innovation adds
a likelihood term under the same gate. Without tape readings the filter is
unchanged.

**What the tape identifies.** Between two readings,
$\Delta y^w \approx b_w\, p\, \Delta W$, so the data constrain only the product
$b_w p$: a 20% error in the fat share can hide behind a 20% error in the slope.
What the tape adds is a second, independent anchor for the fat level,
$F = (y^w - a_w)/b_w$ with $a_w$ known to 7.2 cm, alongside the Deurenberg
starting value.

| Synthetic, 120 days, waist weekly | With tape | Without |
|:--|--:|--:|
| Final fat mass, posterior sd | {{< meta sy.TapeFatSdWith >}} kg | {{< meta sy.TapeFatSdWithout >}} kg |
| Final fat mass, estimate (truth {{< meta sy.TapeFatTruth >}} kg) | {{< meta sy.TapeFatWith >}} kg | {{< meta sy.TapeFatWithout >}} kg |
| Waist slope, 90% interval (truth 1.2; prior [{{< meta sy.TapePriorLo >}}, {{< meta sy.TapePriorHi >}}]) | [{{< meta sy.TapeSlopeLo >}}, {{< meta sy.TapeSlopeHi >}}] | n/a |

The fat-mass interval narrows by {{< meta sy.TapeNarrowPct >}}%. A
test asserts this much (at least 5% narrower, truth inside) and no more.

## Identifiability {#identifiability}

Read this before quoting any parameter. Synthetic figures are from the run
[below](#synthetic).

| Status | Quantity | Why |
|:--|:--|:------|
| Well identified | Maintenance $c_0 + c_1 W$ at a given weight | Over the narrow mass range of one dataset, raising $c_0$ and lowering $c_1$ barely changes the fit, so each coefficient is wide while their combination is well determined. |
| Well identified | $k_G$, $\tau_G$ | Carbohydrate varies daily and the water swings are large next to the noise. On synthetic data $k_G$ moved {{< meta sy.ShiftKG >}}% off its prior. |
| Weak | $\beta_A$, $\tau_A$ | Separable from a mis-estimated $c_0$ only through the lag; constant intake gives almost no information. A diet break or refeed week helps. |
| Weak | Sodium gain $k_{Na}$ | Learned only from how much sodium varies. Everyday variation moves sodium water by about 0.1 kg, under one weigh-in's noise, and each swing fades within a day; on the demo subject the likelihood moves about one nat across a fivefold range of the gain, and estimates scatter low. A profile of the likelihood peaks where the posterior sits, and with two to three times the variation the estimates converge on the truth: weak information, not a biased estimator. Logging sodium, and estimating it when eating out, makes it learnable. |
| Weak | Fat mass $F$ | The scale sees it only through $\rho_{\mathrm{eff}}$, which a 15 kg loss changes by under 10%. Always shown with a credible band. |
| Not identified | $\kappa_P$ | Second-order effect through $\rho_{\mathrm{eff}}$. On synthetic data it moved less than 1%, with a 90% interval spanning zero. |
| Not identified | Thermic-effect coefficients | Held fixed: their effect looks like a proportional logging bias, which $c_0$ already absorbs. |

::: {.callout-note title="Quote maintenance, not its coefficients"}
Quoting $c_0$ or $c_1$ alone overstates the uncertainty in what you can eat;
treating them as independent understates it. `maintenance_interval_kcal`
carries the full covariance through the combination, and a test asserts it is
more than 20% narrower than the naive independent interval.
:::

## Validation on synthetic data {#synthetic}

`wdm demo` generates two weight-only past episodes of 40 days and a current
episode of {{< meta sy.Days >}} days with macronutrients ({{< meta sy.WeighInPct >}}% weigh-in and {{< meta sy.LoggedPct >}}% logging coverage),
from true parameters deliberately offset from the priors (if the truth sat at
the prior center, an estimator that ignored the data would pass). {{< meta sy.HitsWord >}} true
values lie inside their 90% posterior intervals; for example, $k_G$ (truth
{{< meta sy.TruthKG >}}) has median {{< meta sy.MedKG >}}, and $\kappa_P$ (truth {{< meta sy.TruthKappaP >}}) has median {{< meta sy.MedKappaP >}} with an
interval from {{< meta sy.LoKappaP >}} to {{< meta sy.HiKappaP >}}. Maintenance at 100 kg is
estimated at {{< meta sy.MaintMed >}} kcal/day [{{< meta sy.MaintLo >}}, {{< meta sy.MaintHi >}}] against a true {{< meta sy.TrueMaint >}} ({{< meta sy.MaintErrPct >}}% error), and the
smoothed trend's error against the hidden truth is {{< meta sy.TrendRmseG >}} g RMS, against {{< meta sy.ObsNoiseG >}} g of
single-weigh-in noise. This validates the inference machinery and says nothing
about whether the model resembles a person: the data come from the dynamics the
estimator assumes.

## Backtest: forecasting readings not yet seen {#backtest}

`wdm backtest` replays each phase-2 episode day by day. For every origin day $j$
and horizon $h$ (default 1 and 7 days) whose target morning has a weigh-in, it
takes the filtered belief on morning $j$, applies $h$ unscented time updates
driven by the food logged on days $j$ to $j + h - 1$, and scores the predictive
distribution $\mathcal{N}(H\hat z, H P H^{\top} + R)$ against the reading. Using
the logged food separates model error from adherence.

Scores: bias and RMSE (lb); skill, the fractional RMSE improvement over
persistence (tomorrow equals the last reading), which a model must beat at a
week to earn its complexity; the standardized-error sd and 50% and 90% coverage,
which a calibrated forecast puts near 1, 50% and 90%; and the mean log
predictive density (higher is better), the fair score for comparing noise
models.

| Synthetic | $h$ | $n$ | Bias | RMSE | Persist. | Skill | $z$ sd | Cov. 50/90 | Log score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Current episode | 1 | {{< meta sy.BtOneN >}} | {{< meta sy.BtOneBias >}} | {{< meta sy.BtOneRmse >}} | {{< meta sy.BtOnePersist >}} | +{{< meta sy.BtOneSkill >}}% | {{< meta sy.BtOneZsd >}} | {{< meta sy.BtOneCovFifty >}}/{{< meta sy.BtOneCovNinety >}} | {{< meta sy.BtOneLogScore >}} |
| Current episode | 7 | {{< meta sy.BtSevenN >}} | {{< meta sy.BtSevenBias >}} | {{< meta sy.BtSevenRmse >}} | {{< meta sy.BtSevenPersist >}} | +{{< meta sy.BtSevenSkill >}}% | {{< meta sy.BtSevenZsd >}} | {{< meta sy.BtSevenCovFifty >}}/{{< meta sy.BtSevenCovNinety >}} | {{< meta sy.BtSevenLogScore >}} |

On its own synthetic data the model's intervals are honest to slightly wide, and
it beats persistence by {{< meta sy.BtOneSkill >}}% at a day and {{< meta sy.BtSevenSkill >}}% at a week. A test asserts the
one-day calibration, the one-week skill, and that the one-day forecast equals the
filter's own prediction. On real data the first mornings of a diet drop more
than the water compartments explain, most likely gut contents, which no state
represents.

**Caveats.** Phase-1 noise is fitted on the whole record, so coverage is slightly
optimistic. Forecasts from neighboring origins overlap (at $h = 7$, consecutive
forecasts share six days of inputs), so the effective sample is several times
smaller than $n$.

## References {#references}

- Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16, 1190-1208.
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
