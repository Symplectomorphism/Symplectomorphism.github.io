---
title: "Limitations"
---

Every model here is wrong in specific, knowable ways. This page lists them,
roughly by how much they should worry you, then what the model has no opinion
about and the extensions most worth doing. Details are on the
[model](model.md), [estimation](estimation.md) and [control](control.md) pages.

## Drifting log bias looks like adaptation {#drifting-bias}

This is the most serious threat to the whole framework.

**Constant bias is harmless.** Self-reported intake under-reports true intake by
roughly 20% on average in doubly-labeled-water studies (Lichtman et al., 1992;
Schoeller, 1990). If you log $I_{\mathrm{log}} = (1 - q)\,I_{\mathrm{true}}$
with $q$ constant, the model fits
$I_{\mathrm{log}} - (c_0 + c_1 W + \dots) \approx \rho_{\mathrm{eff}}\,\dot T$
and $c_0$ absorbs the shortfall. The controller prescribes in the same biased
units it learned in, so the prescription is correct in your logging units even
though intake and maintenance are both underestimates.

::: {.callout-note}
The model does not need your food log to be accurate. It needs it to be
consistently inaccurate.
:::

**Drifting bias is not.** If $q$ shrinks as logging improves, as it usually
does, the model sees apparently eating more while losing the same amount:
exactly the signature of metabolic adaptation. The two are observationally
equivalent from weight and logged intake, so the filter reads better logging as
a slowing metabolism, inflates $\beta_A$, deflates maintenance, and prescribes a
larger deficit than intended. There is no algorithmic fix within this data;
breaking the tie needs an independent measure of expenditure or a period of
unusually careful logging to calibrate against. The defense is behavioral: log
with the same rigor on day 90 as on day 1, and record any change of method (a
new app, a food scale) in the `notes` column of the [data file](../data.md).

Early logs are usually the worst, and all four reasons push the same way:
portion estimates improve with practice; database errors and forgotten oil,
condiments and drinks are mostly omissions, so they bias low; early days are
more often reconstructed from memory; and week 1 brings unfamiliar foods, while
by week 6 most people eat easily logged repeats. These change the bias more than
the variance, which is why the problem is drift rather than noise.

## Other structural limits {#structural}

- **Self-reported intake is the input.** Dhurandhar et al. (2015) argue that
  free-living energy-balance measurements are often poor enough to mislead. The
  cancellation of a constant bias defends against nothing else.
- **Expenditure is never observed**, only its consequences. A different model
  structure would give a different maintenance figure with equally narrow
  intervals.
- **One subject, no replication.** Synthetic validation tests the estimator, not
  the physiology; the backtest tests forecasts, but on one person.
- **Unlogged days are treated as neutral**, although people forget to log a
  party more often than a salad. The process-noise inflation widens the
  uncertainty without correcting the bias.

## Modeling simplifications {#simplifications}

| Simplification | Consequence |
|:--|:------|
| Expenditure linear in total mass | Lean and fat tissue burn at different rates; a composition-resolved form adds two correlated, weakly identified coefficients. |
| First-order adaptation, one time constant | Real adaptation has several mechanisms on different timescales. |
| Fixed thermic-effect coefficients | They vary between people and meals; fixed because they are confounded with logging bias. |
| Forbes' law as the partitioning baseline | A population regression with wide individual variation, applied to one person. |
| No gut-contents state | Eating much less leaves less food in transit, so the first days of a diet drop more than predicted. |
| No explicit hydration state | Absorbed into observation noise, inflating $R$ and costing responsiveness. |
| No menstrual-cycle term | A large, systematic water-weight signal for subjects who cycle; it would appear as serial correlation in the innovations, with no way to attribute it. |
| No exercise-composition model | Resistance training changes the fat/lean split in ways the protein term only crudely proxies. |
| Population priors | Mifflin-St Jeor, Deurenberg and Lean are cross-sectional regressions that each miss an individual widely; they only center priors. |
| Waist linear in fat mass | The Lean equation's implied slope steepens about 10% per 5 kg of fat lost; the estimated slope must track it. |

## Statistical limitations {#statistical}

- **Gaussian noise.** For the scale it is a tested choice
  ([noise model](estimation.md#noise-model)), but on a small record; the process
  noise is Gaussian by assumption, and more data could favor heavier tails.
- **Weak identification.** Adaptation needs a change in intake to show itself,
  fat mass stays wide, the tape cannot pin the fat share (it sees only the
  product of its slope and the share), and $\kappa_P$ is not recoverable from
  scale data ([identifiability](estimation.md#identifiability)). A shift near
  zero is the prior read back.
- **The parameter random walk is a fiction.** Its rate is chosen, not estimated,
  and it sets the steady-state width of every posterior.
- **Certainty-equivalent control.** The controller ignores parameter
  uncertainty, so it is optimistic, above all in the first fortnight. Daily
  replanning limits the damage without removing it.
- **Reported intervals are optimistic.** Every interval is conditional on the
  model being right, and structural uncertainty is represented nowhere. Phase-1
  standard errors also assume the trend model is right, which it is not; the
  backtest's noise is fitted on the whole record, and overlapping forecasts
  shrink its effective sample ([backtest](estimation.md#backtest)).
- **Early output is the prior.** In the first week the filter reports its prior
  back in output that looks as authoritative as it ever will. Maintenance
  separates from its prior after roughly four to six weeks; adaptation moves
  only if intake varied.

The controller's own approximations are stated where they arise: its reference
is a scale weight compared against tissue
([control](control.md#objective)), the earliest-date formula assumes the water
compartments end where they are ([feasibility](control.md#feasibility)), and the
safety limits are general-guidance defaults, not anything specific to you.

## Things the model has no opinion about {#no-opinion}

- **Adherence**, the largest single determinant of whether a diet works. The
  controller optimizes what would happen if its plan were followed exactly.
- **Health**: blood pressure, lipids, sleep, mood, bone density, or anything else
  that matters beyond mass and composition.
- **Nutritional adequacy.** The macronutrient bounds are crude guard rails;
  nothing checks micronutrients.
- **Eating disorders.** Turning eating into a daily optimization problem with a
  numerical target is not appropriate for everyone, and this documentation
  cannot judge whether it is appropriate for you.
- **Whether the goal is a good one.** It takes the target as given.

::: {.callout-note title="Not medical advice"}
This is a modeling exercise built on population-level physiology and fitted to
self-reported data. Anyone acting on its output should talk to a doctor or a
registered dietitian first.
:::

## Extensions worth doing {#extensions}

Roughly by value:

1. **Stochastic or minimax MPC**, propagating parameter uncertainty into the plan.
   The clearest methodological improvement available.
2. **An adherence model** relating what is prescribed to what is eaten; the plans
   log already records the prescriptions. The just-in-time adaptive intervention
   literature (Nahum-Shani et al., 2018) treats this as the central object.
3. **Composition-resolved expenditure**, $c_M M + c_F F$ instead of $c_1 W$, if
   enough data accumulate to identify it.
4. **Full Bayesian inference** by MCMC or sequential Monte Carlo: more honest
   uncertainty at much more compute.
5. **Body-composition measurements.** Even occasional DEXA scans would make the
   fat-mass estimate well constrained, which the tape cannot do.

## How to read the output {#interpretation}

The output is a set of conditional statements: given this model, your logged data
and these priors, here is what follows. The intervals quantify uncertainty within
that frame and say nothing about the frame itself. The most useful output is not
any parameter estimate but the trend line with its credible band, which needs
almost no modeling assumptions and answers the question people actually have
when they step off the scale ([Reading the output](../reading-output.md)).

## References {#references}

- Dhurandhar, N. V., Schoeller, D., Brown, A. W., Heymsfield, S. B., Thomas, D., Sørensen, T. I. A., Speakman, J. R., Jeansonne, M., and Allison, D. B. (2015). Energy balance measurement: when something is not better than nothing. *International Journal of Obesity*.
- Lichtman, S. W., Pisarska, K., Berman, E. R., Pestone, M., Dowling, H., Offenbacher, E., Weisel, H., Heshka, S., Matthews, D. E., and Heymsfield, S. B. (1992). Discrepancy between self-reported and actual caloric intake and exercise in obese subjects. *New England Journal of Medicine*, 327, 1893-1898.
- Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., and Murphy, S. A. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health: key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*.
- Schoeller, D. A. (1990). How accurate is self-reported dietary energy intake? *Nutrition Reviews*.
