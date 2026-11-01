---
title: "Control"
metadata-files: [../numbers_synthetic.yml]
---

Given today's belief about the subject's state and metabolism, what should the
subject eat today? `wdm` answers with a model-predictive controller (Mayne et
al., 2000; Rawlings et al., 2017) that optimizes a calorie and macronutrient
plan against the [model](model.md) from the filtered state of
[phase 2](estimation.md#phase-2), under safety limits set in
[The subject file](../configuration.md).

## The receding-horizon problem {#receding-horizon}

Each run optimizes a plan over $H$ days and prescribes only the first, so model
errors are corrected daily instead of accumulating. The horizon
is 28 days in sustainable and maintain mode; in deadline mode it runs to the goal
date, capped at 365 days. Which mode runs by default is set by the subject file
([mode](../configuration.md#mode)).

### Which day the plan starts {#plan-start}

The filtered state is the morning of the last row's date
([timing](estimation.md#timing)), and whether that day is over is a calendar
question:

- **Last row dated today:** the plan starts this morning, however much of
  today's food is already logged, so logging a meal never shifts the plan.
- **Last row dated earlier, food logged:** that day is finished; the state is
  advanced one step through it and the plan starts the next morning.
- **Last row dated earlier, no food:** the plan starts on that date, and
  `wdm plan` refuses to plan if the estimate is more than three days old.

## Decision variables {#decisions}

Decisions are held constant over week-long blocks (`--block`). Per block they
are energy $I$, protein per kilogram of the plan's starting weight, and fat's
share of energy $\phi_F$, so $F_g = \phi_F I/9$ and carbohydrate is the
residual,

$$
C = \frac{I - 4 P_g - 9 F_g}{4},
$$

which keeps the three choices independent. Planned days have no alcohol,
sodium deviation or exercise.

## The objective {#objective}

**Deadline mode.**

$$
J = w_T \bigl(T_H - r_H\bigr)^2
  + \frac{w_t}{H} \sum_{k=1}^{H} (T_k - r_k)^2
  + w_s \sum_{k} \Delta I_k^2
  + w_e \sum_{k} D_k^2
  + w_c \sum_{k} C_k^2
  + w_\phi \sum_{k} \bigl(\phi_k - \phi^\star\bigr)^2 .
$$

$T_k$ is predicted settled tissue at the end of plan day $k$
([below](#settled)). The reference
$r_k = W_0 + \min(k/N, 1)\,(W^\star - W_0)$ runs linearly from the current scale
weight $W_0$ to the target $W^\star$ over the $N$ days remaining.
$\Delta I_k$ is the day-to-day change in intake ($I_0$ is the last finished
day's logged intake), $D_k = E_{\mathrm{base}} - I_k$ the deficit
([model](model.md#expenditure)), $C_k$ the carbohydrate prescribed, and
$\phi_k$, $\phi^\star$ a macronutrient's energy share and its target
([macros](#macros)).

**Sustainable mode.**

$$
J = w_p T_H + w_s \sum_k \Delta I_k^2 + w_e \sum_k D_k^2 + w_c \sum_k C_k^2
  + w_\phi \sum_k \bigl(\phi_k - \phi^\star\bigr)^2 .
$$

The timeline becomes an output. Progress is linear in final tissue, so that
against quadratic effort it has an interior optimum.

| Weight | Code name | Default | Role |
|:--|:--|--:|:------|
| $w_T$ | `CostWeights.terminal` | 40 | Terminal miss against $r_H$ |
| $w_t$ | `CostWeights.tracking` | 1 | Deviation from the path |
| $w_p$ | `CostWeights.progress` | 12 | Final tissue (sustainable) |
| $w_s$ | `CostWeights.smoothness` | 3 | Day-to-day intake changes |
| $w_e$ | `CostWeights.effort` | 0.4 | Squared deficit |
| $w_c$ | `CostWeights.carb_aversion` | 0 | Carbohydrate preference |
| $w_\phi$ | module constants | 0.1 or 100 | Macro share ([macros](#macros)) |

Two choices close loopholes. **The terminal term tracks $r_H$, not the goal**:
over a horizon shorter than the time remaining, a penalty on the gap to the goal
would pin intake to the calorie floor. **Both objectives score tissue, not scale
weight**: a cost on $W = T + G_c + G_n$ can be lowered by cutting carbohydrate
alone, draining $G_c$. (The reference is still built from scale weights, so it
differs from tissue by the starting water offset.)

### Settled tissue {#settled}

Energy stored as glycogen is withheld from tissue (the $\rho_G \dot G_c$ term of
the [assembled model](model.md#assembled)) and handed back when glycogen drains.
Scored on bare tissue, a plan could load carbohydrate in its last weeks and let
the energy return to tissue after the horizon closes, and a rate limit could be
loosened by draining glycogen. The controller therefore scores **settled
tissue**, the tissue left once glycogen returns to its habitual level:

$$
T^{\mathrm{s}}_k = T_k + \frac{\rho_G}{\rho_{\mathrm{eff}}}\,G_{c,k}.
$$

Every $T$ in the objectives and the rate limit on this page means
$T^{\mathrm{s}}$, and the weekly rate `wdm plan` reports (`loss%/wk`) is the
loss of settled tissue as a percentage of it.

### Maintain mode {#maintain}

Once the goal is reached the question becomes what intake holds the subject
there. Maintain mode uses the deadline objective with a different reference: from
today's tissue $T_0$ to the held weight $W_h$ (`[maintain] weight_lb`, or $T_0$
itself when none is set) over the horizon,

$$
r_k = T_0 + \frac{k}{H}\,(W_h - T_0), \qquad k = 1, \dots, H,
$$

with no goal date. Tissue is the right basis: at maintenance the water
compartments refill and the scale settles at tissue, so holding "today's weight"
mid-diet, with the water drained, would cut tissue to stay there. At the held
weight the reference is flat and the effort term pulls intake to the
weight-stable level; a 3 lb drift is corrected over the 28-day horizon, under
1 lb a week. Every constraint still applies. Usage is in
[The subject file](../configuration.md#maintain).

## Constraints {#constraints}

| Constraint | Form | Default | Enforced as |
|:--|:--|:--|:--|
| Calorie floor and ceiling | $I_{\min} \le I_b \le I_{\max}$ | 1500 and 6000 kcal/day (`min_kcal`, `max_kcal`) | Box bound |
| Protein | $P_{\min} \le$ g/kg of the plan's starting weight $\le 3.0$ | $P_{\min} = 1.6$ (`min_protein_g_per_kg`) | Box bound |
| Fat share | $0.15 \le \phi_{F,b} \le 0.75$ | ceiling via `--max-fat-fraction` | Box bound |
| Tissue-loss rate | $T_{k-1} - T_k \le \gamma\,T_k/7$, $k = 1, \dots, H$ | $\gamma = 1\%$/week (`max_loss_frac_per_week`) | One inequality per day |
| Carbohydrate floor | $C_b \ge C_{\min}$ | 20 g/day (`min_carb_g`) | One inequality per block |

$T_0$ is the settled tissue at the start of the plan.

::: {.callout-tip title="The rate limit applies to tissue"}
Cutting carbohydrate sharply sheds a kilogram of glycogen water within days,
which reads as 4 lb/week on the scale with no tissue lost; a limit on scale
weight would refuse a reasonable plan or raise calories in response to water.
The limit is therefore a fraction $\gamma$ per week of each day's settled tissue
([above](#settled)), and it tightens as tissue falls.
:::

The carbohydrate floor is a safety limit, distinct from the carbohydrate
preference, and no value of $w_c$ crosses it.

## Solving {#slsqp}

The rate limit and carbohydrate floor act on the predicted trajectory, as
inequalities $\mathbf{g}(\mathbf{d}) \ge 0$. The solver is SLSQP (Kraft, 1988;
Nocedal and Wright, 2006), which holds an active constraint rather than trading
it against the objective, with exact JAX derivatives through the rollout. If it
fails, or its plan fails the check below, a penalty-method fallback takes over.

## The independent check {#check}

Before anything is displayed, the plan is checked against every floor and the
rate limit on every day: a constraint never verified is not a constraint. The
rate check allows a 5% relative tolerance.

## Infeasible goals and the earliest date {#feasibility}

Asked for the unachievable, an optimizer returns something that looks like a
plan rather than a refusal. So deadline mode also computes the earliest date at
the maximum rate, where tissue falls geometrically:

$$
T_n = T_0 \left(1 + \frac{\gamma}{7}\right)^{-n},
\qquad
n_{\min} = \frac{-\ln\bigl(1 - \Delta/T_0\bigr)}{\ln(1 + \gamma/7)} ,
$$

with $\Delta = W_0 - W^\star$ the distance to the target on the scale, assuming
the water compartments end where they are now. The earliest date is
$\lceil n_{\min} \rceil$ days after the plan starts (20 kg from 100 kg at 1% per
week takes 157 days, not the fixed-rate 140). The message reads

> `target of <W*> lb by <date> is not reachable within your 1%/week limit; the earliest achievable date at that maximum rate is <date>`

## Macro preferences {#macros}

**Carbohydrate aversion.** $w_c \sum_k C_k^2$ (`--carb-aversion W`, default 0)
prefers less carbohydrate at every level. Since carbohydrate is the residual it
works by pushing protein and fat up (`--max-fat-fraction` gives the freed energy
somewhere to go), so total energy can rise: on synthetic data, raising it from 0
to 3 moved a day from 1903 kcal at 40% carbohydrate to 2250 kcal at 4%.

**Tie-breaker.** Tissue dynamics barely distinguish one fat/carbohydrate split
from another, and a flat direction makes the solver unreliable, so a weak term
($w_\phi = 0.1$) pulls fat's share toward the midpoint $\bar\phi$ of its bounds.

**Target fractions.** `target_carb_fraction` or `target_fat_fraction`, on the
command line or under `[preferences]`, replaces $\bar\phi$ with $w_\phi = 100$.
Only one term is used: the carbohydrate target if set, then the fat target, then
$\bar\phi$. Shares are of energy ($9F_g/I$, $4C/I$).

## Certainty equivalence {#certainty-equivalence}

The optimization uses posterior mean parameters and ignores their uncertainty:
standard, cheap and optimistic. A maintenance estimate 200 kcal too high
under-delivers until the weigh-ins disagree, and daily replanning then corrects
it. The controller also assumes its plan is followed exactly
([Limitations](limitations.md#statistical)).

## Behavior on synthetic data {#behavior}

On the synthetic subject ({{< meta sy.BehWeightLb >}} lb, maintenance near {{< meta sy.BehMaint >}} kcal/day), a
reachable deadline loses {{< meta sy.BehDeadRate >}} lb a week, peaking at {{< meta sy.BehDeadPct >}}% of tissue; the sustainable plan
sits on the 1% limit, since with the default weights the optimum of progress
against effort lies at or beyond it; and an impossible deadline is flagged and
pushed to whichever limit binds first, here {{< meta sy.BehImpBinds >}}. [Walkthrough step
7](../walkthrough/07-plan.md) runs all three.

## References {#references}

- Kraft, D. (1988). *A software package for sequential quadratic programming*. Forschungsbericht DFVLR-FB 88-28, DFVLR, Oberpfaffenhofen.
- Mayne, D. Q., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M. (2000). Constrained model predictive control: stability and optimality. *Automatica*, 36, 789-814.
- Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Rawlings, J. B., Mayne, D. Q., and Diehl, M. M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
