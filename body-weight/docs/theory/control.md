---
title: "Control"
metadata-files: [../numbers_synthetic.yml]
---

Given today's belief about your state and metabolism, what should you eat today?
`wdm` answers with a receding-horizon model-predictive controller (Mayne et al.,
2000; Rawlings et al., 2017) that optimizes a calorie and macronutrient plan
against the [model](model.md), starting from the filtered state of
[phase 2](estimation.md#phase-2), under safety limits it verifies before showing
anything. Usage is in [Commands](../commands.md); the limits are set in
[Configuration](../configuration.md).

## The receding-horizon problem {#receding-horizon}

Each run optimizes a plan over $H$ days, prescribes only the first day, and
discards the rest; the next run repeats this with one more weigh-in. Because the
plan is never trusted beyond a day, model and parameter errors are corrected
continuously instead of accumulating. The horizon is 28 days in sustainable
mode; in deadline mode it runs to the goal date, capped at 365 days and never
shorter than one block. `wdm plan` uses deadline mode when `subject.toml` has a
`[goal]` section and sustainable mode otherwise.

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

Decisions are held constant over blocks of a week (`--block`, default 7), since
nobody eats to a different target every day and 28 daily targets would be
numerically awkward. With $B$ blocks the decision vector $\mathbf{d}$ has $3B$
entries: per block, energy $I$, protein per kilogram, and fat's share of energy
$\phi_F$. Fat grams are $F_g = \phi_F I/9$, protein grams are the chosen g/kg
times the scale weight at the start of the plan, and carbohydrate is the
residual,

$$
C = \frac{I - 4 P_g - 9 F_g}{4},
$$

which keeps the three choices independent without a sum-to-one constraint. The
rollout treats every planned day as fully logged, with no alcohol, sodium
deviation or exercise.

## The objective {#objective}

**Deadline mode.**

$$
J = w_T \bigl(T_H - r_H\bigr)^2
  + \frac{w_t}{H} \sum_{k=1}^{H} (T_k - r_k)^2
  + w_s \sum_{k} \Delta I_k^2
  + w_e \sum_{k} D_k^2
  + w_c \sum_{k} C_k^2
  + w_\phi \sum_{k} \bigl(\phi_k - \phi^\star\bigr)^2
  + \mu \Phi .
$$

$T_k$ is predicted settled tissue at the end of plan day $k$
([below](#settled)). The reference
$r_k = W_0 + \min(k/N, 1)\,(W^\star - W_0)$ runs linearly from the current scale
weight $W_0$ to the target $W^\star$ over the $N$ days remaining.
$\Delta I_k = I_k - I_{k-1}$ is the day-to-day change in intake ($I_0$ is the last
logged intake, or $I_{\min} + 600$). $D_k = E_{\mathrm{base}} - I_k$ is the
deficit against expenditure without adaptation ([model](model.md#expenditure)).
$C_k$ is the carbohydrate prescribed, $\phi_k$ and $\phi^\star$ are the energy
share of the targeted macronutrient and its target ([macros](#macros)), and
$\mu\Phi$ is a violation penalty, zero for the main solver
([solving](#slsqp)). In code $\Delta I$ and $D$ are in thousands of kcal and $C$
in hundreds of grams, so each term is of order one.

**Sustainable mode.**

$$
J = w_p T_H + w_s \sum_k \Delta I_k^2 + w_e \sum_k D_k^2 + w_c \sum_k C_k^2
  + w_\phi \sum_k \bigl(\phi_k - \phi^\star\bigr)^2 + \mu \Phi .
$$

The timeline becomes an output, which usually makes this the more honest mode.
Progress is linear in final tissue: a quadratic reward would grow without bound
and drive intake to the floor, while linear progress against quadratic effort has
an interior optimum, unless the rate limit binds first.

| Weight | Code name | Default | Role |
|:--|:--|--:|:------|
| $w_T$ | `CostWeights.terminal` | 40 | Terminal miss against $r_H$ |
| $w_t$ | `CostWeights.tracking` | 1 | Deviation from the path; stops a loaf-then-crash plan |
| $w_p$ | `CostWeights.progress` | 12 | Final tissue (sustainable) |
| $w_s$ | `CostWeights.smoothness` | 3 | Day-to-day intake changes |
| $w_e$ | `CostWeights.effort` | 0.4 | Squared deficit: large deficits cost adherence, training and social life |
| $w_c$ | `CostWeights.carb_aversion` | 0 | Carbohydrate preference (`--carb-aversion`) |
| $w_\phi$ | module constants | 0.1 or 100 | Macro share: tie-breaker or stated target |


::: {.callout-note title="Two bugs that shaped the objective"}
**The terminal term tracks $r_H$, not the goal.** Penalizing the gap to the final
target over a horizon shorter than the time remaining asks the controller to
arrive by the horizon's end, and it pinned intake to the calorie floor for a
month in a plausible-looking plan. Tracking the reference makes the horizon a
window onto a longer plan.

**Both objectives score tissue $T$, not scale weight $W$.** Since
$W = T + G_c + G_n$ and carbohydrate moves $G_c$ directly, a cost on $W$ can be
lowered by cutting carbohydrate alone; with no carbohydrate preference,
carbohydrate landed on its floor at every floor setting tried. The reference is
still built from the goal's scale weight, so it differs from $T$ by the water
offset present at the start.
:::

### Settled tissue {#settled}

Bare tissue leaves a smaller loophole of the same kind. Energy stored as glycogen
is withheld from tissue (the $\rho_G \dot G_c$ term of the
[energy balance](model.md)) and handed back when glycogen drains. Scored on bare
tissue, deadline plans loaded carbohydrate in their last weeks: in an 84-day
synthetic plan it rose from 115 g to 254 g a day while fat fell to its floor. The
refill lowers $T_H$, and the energy returns to tissue after the horizon closes,
where the objective cannot see it. The controller therefore scores **settled
tissue**, the tissue left once glycogen returns to its habitual level,

$$
T^{\mathrm{s}}_k = T_k + \frac{\rho_G}{\rho_{\mathrm{eff}}}\,G_{c,k},
$$

and every $T$ in the objectives and the rate limit on this page means
$T^{\mathrm{s}}$. A carbohydrate-for-fat swap then moves the score only through
its real effect, carbohydrate's higher thermic cost. The rate limit needs the same
treatment: on bare tissue it can be loosened by draining glycogen, whose energy
props tissue up for a few days, and plans crashed carbohydrate to its floor in the
final week instead.

### Maintain mode {#maintain}

Once the goal is reached the question becomes "how much keeps me here". Maintain
mode uses the deadline objective with a different reference: from today's tissue
$T_0$ to the held weight $W_h$ (`[maintain] weight_lb` in the subject file, or
$T_0$ itself when none is set) over the horizon,

$$
r_k = T_0 + \frac{k}{H}\,(W_h - T_0), \qquad k = 1, \dots, H,
$$

with no goal date. Tissue is the right basis: at maintenance, eating as usual,
the water compartments return to their habitual level and the scale settles at
tissue. Mid-diet the scale sits below tissue by the drained water, so holding
"today's weight" on the scale would cut tissue to stay there; holding today's
tissue does not. At the held weight the reference is flat and the effort term
pulls intake to the weight-stable level; after a drift of 3 lb the correction is
spread over the 28-day horizon, under 1 lb a week, instead of a crash. Every
constraint still applies, and there is no feasibility report, since there is no
deadline to miss. A `[maintain]` section makes this mode the default for
`wdm plan` and `wdm daily` ([The subject file](../configuration.md#maintain)).

## Constraints {#constraints}

| Constraint | Form | Default | Enforced as |
|:--|:--|:--|:--|
| Calorie floor and ceiling | $I_{\min} \le I_b \le I_{\max}$ | 1500 and 6000 kcal/day (`min_kcal`, `max_kcal`) | Box bound |
| Protein | $P_{\min} \le$ g/kg of the plan's starting weight $\le 3.0$ | $P_{\min} = 1.6$ (`min_protein_g_per_kg`) | Box bound |
| Fat share | $0.15 \le \phi_{F,b} \le 0.75$ | ceiling via `--max-fat-fraction` | Box bound |
| Tissue-loss rate | $T_{k-1} - T_k \le \gamma\,T_k/7$, $k = 1, \dots, H$ | $\gamma = 1\%$/week (`max_loss_frac_per_week`) | One inequality per day |
| Carbohydrate floor | $C_b \ge C_{\min}$ | 20 g/day (`min_carb_g`) | One inequality per block |

$T_0$ is the tissue at the start of the plan. Protein is bounded per kilogram of
the starting weight, not each day's predicted weight.

::: {.callout-tip title="The rate limit applies to tissue"}
Cutting carbohydrate sharply sheds a kilogram of glycogen water within days. On
the scale that reads as 4 lb/week and would trip any limit on scale weight, yet
no tissue was lost; a controller limited on scale weight would refuse a
reasonable plan or raise calories in response to water loss. The limit is
therefore a fraction $\gamma$ per week of each day's settled tissue
([above](#settled)), which is the energy deficit. It tightens as
tissue falls, which is what "a fraction of body mass per week" means; a fixed cap
of $\gamma T_0/7$ kg a day would let a long plan exceed $\gamma$ in its later
weeks.
:::

The carbohydrate floor is a safety limit, distinct from the carbohydrate
preference: it stops the optimizer crashing carbohydrate for a fast water drop,
and no value of $w_c$ crosses it.

## Solving: SLSQP, with a penalty fallback {#slsqp}

The box bounds act on the decisions; the rate limit and carbohydrate floor act on
the predicted trajectory, so they become inequalities $\mathbf{g}(\mathbf{d})
\ge 0$:

$$
g_k = \frac{\gamma}{7} T_k - (T_{k-1} - T_k), \quad k = 1, \dots, H;
\qquad
g_{H+b} = \frac{C_b - C_{\min}}{100\ \text{g}}, \quad b = 1, \dots, B,
$$

the 100 g scaling making the carbohydrate rows comparable to the rate rows. The
solver is SLSQP (Kraft, 1988; Nocedal and Wright, 2006), which solves a quadratic
model of $J$ subject to the linearized constraints at each iterate, so an active
constraint is held rather than traded against the objective. JAX differentiates
exactly through the model rollout in forward mode, which suits the shape: $3B =
12$ inputs and $H + B = 32$ outputs for a 28-day plan. Each decision is scaled to
$[0, 1]$ across its bounds, $\tilde d = (d - d_{\min})/(d_{\max} - d_{\min})$
(energy and fat share differ by four orders of magnitude), and $J$ is divided by
its starting value. The start puts energy mid-way in
$[I_{\min}, \min(I_{\max}, 3000)]$, protein at $P_{\min} + 0.2$ and the fat share
mid-range; SLSQP runs for at most 500 iterations.

If SLSQP fails, or its plan fails the check below, the older scheme takes over
from SLSQP's point: the trajectory limits become hinge penalties,

$$
\Phi = 4000 \sum_k \max\!\Bigl(0, (T_{k-1} - T_k) - \tfrac{\gamma}{7} T_k\Bigr)^2
     + 50 \sum_k \max\!\bigl(0, (C_{\min} - C_k)/100\bigr)^2 ,
$$

minimized with L-BFGS-B (Byrd et al., 1995) from $\mu = 1$, multiplying $\mu$ by
12 after each round that still violates a limit, for up to nine rounds. SLSQP
plans carry `penalty_scale = 0`; a positive value marks a fallback plan.

## The independent check {#check}

Before anything is displayed, the plan is checked against the calorie floor, the
protein floor (on the starting weight), the carbohydrate floor (1 g of slack) and
the rate limit on every day, the first measured from the starting tissue: a constraint never verified is not a constraint. The rate check
allows a 5% relative tolerance, since rejecting 1.027%/week against a 1%/week
limit would make a distinction with no physiological content.


## Infeasible goals and the earliest date {#feasibility}

Asked for the unachievable, an optimizer returns the closest thing it can manage,
which looks like a plan rather than a refusal. So in deadline mode the package
computes the earliest date the target can be reached at the maximum rate. With
the rate limit at equality, $T_k = T_{k-1}/(1 + \gamma/7)$, tissue falls
geometrically:

$$
T_n = T_0 \left(1 + \frac{\gamma}{7}\right)^{-n},
\qquad
n_{\min} = \frac{-\ln\bigl(1 - \Delta/T_0\bigr)}{\ln(1 + \gamma/7)} ,
$$

with $\Delta = W_0 - W^\star$ the distance to the target on the scale, assuming
the water compartments end where they are now. The earliest date is
$\lceil n_{\min} \rceil$ days after the plan starts, and if $\Delta \ge T_0$ the
message asks you to check the goal. The fixed-rate shortcut
$\Delta/(\gamma T_0/7)$ is always too early, since $-\ln(1 - x) > x$ and
$\ln(1 + y) < y$: for example, 20 kg from $T_0 = 100$ kg at 1% per week takes
$\lceil 156.3 \rceil = 157$ days, not 140. In code $W_0$ and $T_0$ are the plan's
first predicted day, so the date can differ from a hand calculation by a day, and
the check is skipped within 0.1 kg of the target. The message reads

> `target of <W*> lb by <date> is not reachable within your 1%/week limit; the earliest achievable date at that maximum rate is <date>`

with the limit printed unrounded so it matches the config file. An impossible
deadline pushes intake to whichever limit binds first, the rate limit or the
calorie floor ([Reading the output](../reading-output.md)).

## Macro preferences {#macros}

**Carbohydrate aversion.** $w_c \sum_k C_k^2$ (`--carb-aversion W`, default 0)
is a preference for less carbohydrate at every level. Being squared, it trades
off smoothly instead of driving to the floor, and since carbohydrate is the
residual it works by pushing protein and fat toward their ceilings
(`--max-fat-fraction` gives the freed energy somewhere to go). In sustainable
mode on synthetic data, raising it from 0 to 3 moved a day from 1903 kcal at 40%
carbohydrate, 30% protein, 30% fat to 2250 kcal at 4%, 47%, 49%, with
carbohydrate on its 20 g floor: total energy rose, the opposite of "aversion just
cuts calories".

**Tie-breaker.** With no preference and carbohydrate above its floor, nothing
distinguishes one fat/carbohydrate split from another, since tissue dynamics do
not depend on it. On that flat direction L-BFGS-B, the solver used before SLSQP,
failed outright, so a term pulls fat's share toward the midpoint $\bar\phi$ of
its bounds with $w_\phi = 0.1$, the smallest of $10^{-3}, 10^{-2}, 10^{-1}, 1$
that stopped the failure. It is kept for both solvers.

**Target fractions.** `target_carb_fraction` or `target_fat_fraction`, on the
command line or under `[preferences]` in `subject.toml`, replaces $\bar\phi$ with
$w_\phi = 100$, since at the tie-breaking weight a stated target did not move a
plan against a moderate deadline. Carbohydrate and fat share the residual, so
only one term is used: the carbohydrate target if set, then the fat target, then
$\bar\phi$. Carbohydrate wins because it states most diet styles directly (0.05
to 0.10 for keto, say). Shares are of energy: $\phi = 9F_g/I$ for fat and
$4C/I$ for carbohydrate.

## Certainty equivalence {#certainty-equivalence}

The optimization uses posterior mean parameters and ignores their uncertainty:
standard, cheap and optimistic. A plan built on a maintenance estimate 200 kcal
too high under-delivers until the weigh-ins disagree, and daily replanning then
corrects it within days. For the first fortnight treat the numbers as a starting
point. The controller also assumes its plan is followed exactly; it has no model
of adherence ([Limitations](limitations.md#statistical)).

## Behavior on synthetic data {#behavior}

On the `wdm demo` subject ({{< meta sy.BehWeightLb >}} lb after {{< meta sy.Days >}} simulated days, estimated
maintenance near {{< meta sy.BehMaint >}} kcal/day) with the default limits and a 28-day horizon:

| Scenario | kcal by week | lb/wk | Peak %/wk | Outcome |
|:--|:--|--:|--:|:--|
| Reachable deadline, 175 lb by 2027-03-01 | {{< meta sy.BehDeadWeeks >}} | {{< meta sy.BehDeadRate >}} | {{< meta sy.BehDeadPct >}} | {{< meta sy.BehDeadOutcome >}} |
| Sustainable | {{< meta sy.BehSustWeeks >}} | {{< meta sy.BehSustRate >}} | {{< meta sy.BehSustPct >}} | {{< meta sy.BehSustOutcome >}} |
| Impossible deadline, 150 lb by 2027-01-31 | {{< meta sy.BehImpWeeks >}} | {{< meta sy.BehImpRate >}} | {{< meta sy.BehImpPct >}} | {{< meta sy.BehImpOutcome >}} |

An impossible deadline pushes intake to whichever limit binds first, here {{< meta sy.BehImpBinds >}}. With the default weights the sustainable plan's optimum of progress against effort sits at or beyond the 1% limit. {{< meta sy.BehSolver >}}

## References {#references}

- Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16, 1190-1208.
- Kraft, D. (1988). *A software package for sequential quadratic programming*. Forschungsbericht DFVLR-FB 88-28, DFVLR, Oberpfaffenhofen.
- Mayne, D. Q., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M. (2000). Constrained model predictive control: stability and optimality. *Automatica*, 36, 789-814.
- Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Rawlings, J. B., Mayne, D. Q., and Diehl, M. M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
