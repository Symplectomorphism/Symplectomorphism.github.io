---
title: "Limitations"
---

Every model here is wrong in specific, knowable ways, listed roughly by how much
they matter.

## Drifting log bias looks like adaptation {#drifting-bias}

This is the most serious threat to the whole framework.

**Constant bias is harmless.** Self-reported intake under-reports true intake by
roughly 20% on average in doubly-labeled-water studies (Lichtman et al., 1992;
Schoeller, 1990). If the subject logs $I_{\mathrm{log}} = (1 - q)\,I_{\mathrm{true}}$
with $q$ constant, the estimated intercept $c_0$ absorbs the shortfall, and the
controller prescribes in the same biased units it learned in, so the
prescription is correct in the subject's logging units even though intake and
maintenance are both underestimates.

::: {.callout-note}
The model does not need the food log to be accurate. It needs it to be
consistently inaccurate.
:::

**Drifting bias is not.** If $q$ shrinks as logging improves, as it usually
does, the model sees the subject apparently eating more while losing the same
amount: exactly the signature of metabolic adaptation. The two are
observationally equivalent from weight and logged intake, so the filter reads
better logging as a slowing metabolism, inflates $\beta_A$, deflates
maintenance, and prescribes a larger deficit than intended. No algorithm can
break the tie with this data; that needs an independent measure of expenditure.
The defense is behavioral: log with the same rigor on day 90 as on day 1, and
record any change of method (a new app, a food scale) in the `notes` column of
the [data file](../data.md).

Early logs are usually the worst (portions misjudged, oil and drinks forgotten,
days reconstructed from memory), which is why the problem is drift rather than
noise.

## Other structural limits {#structural}

- **Self-reported intake is the input**, often poor enough to mislead
  (Dhurandhar et al., 2015).
- **Expenditure is never observed**; a different model structure would give a
  different maintenance figure with equally narrow intervals.
- **One subject, no replication.** Synthetic validation tests the estimator, not
  the physiology.
- **Unlogged days are treated as neutral**, although people forget to log a
  party more often than a salad.

## Modeling simplifications {#simplifications}

| Simplification | Consequence |
|:--|:------|
| Expenditure linear in total mass | Lean and fat tissue burn at different rates. |
| First-order adaptation, one time constant | Real adaptation has several mechanisms on different timescales. |
| Fixed thermic-effect coefficients | They vary between people and meals. |
| Forbes' law as the partitioning baseline | A population regression applied to one person. |
| No gut-contents state | The first days of a diet drop more than predicted. |
| No menstrual-cycle term | A large, systematic water signal for subjects who cycle, left in the noise. |
| No exercise-composition model | Resistance training changes the fat/lean split in ways the protein term only crudely proxies. |
| Waist linear in fat mass | The true slope steepens as fat falls; the estimated slope must track it. |

## Statistical limitations {#statistical}

- **Gaussian noise**, tested for the scale on a small record
  ([noise model](estimation.md#noise-model)), assumed for the process noise.
- **Weak identification**: a shift near zero is the prior read back
  ([identifiability](estimation.md#identifiability)).
- **The parameter random walk is chosen, not estimated**, and it sets the
  steady-state width of every posterior.
- **Certainty-equivalent control** is optimistic, above all in the first
  fortnight ([control](control.md#certainty-equivalence)).
- **Every interval is conditional on the model being right**; structural
  uncertainty is represented nowhere ([backtest caveats](estimation.md#backtest)).
- **Early output is the prior**, looking as authoritative as it ever will
  ([what to expect, and when](../reading-output.md#what-to-expect-and-when)).

## Things the model has no opinion about {#no-opinion}

- **Adherence**, the largest single determinant of whether a diet works.
- **Health**: blood pressure, lipids, sleep, mood, bone density, or anything else
  beyond mass and composition.
- **Nutritional adequacy.** The macronutrient bounds are crude guard rails;
  nothing checks micronutrients.
- **Eating disorders.** A daily numerical eating target is not appropriate for
  everyone.
- **Whether the goal is a good one.** It takes the target as given.

::: {.callout-note title="Not medical advice"}
This is a modeling exercise built on population-level physiology and fitted to
self-reported data. Anyone acting on its output should talk to a doctor or a
registered dietitian first.
:::

## Extensions worth doing {#extensions}

1. **Stochastic or minimax MPC**, propagating parameter uncertainty into the plan.
2. **An adherence model** relating what is prescribed to what is eaten, using the
   plans log (Nahum-Shani et al., 2018).
3. **Composition-resolved expenditure**, $c_M M + c_F F$ instead of $c_1 W$, if
   enough data accumulate to identify it.
4. **Full Bayesian inference** by MCMC or sequential Monte Carlo.
5. **Body-composition measurements.** Occasional DEXA scans would constrain fat
   mass, which the tape cannot do.

The output is a set of conditional statements: given this model, the logged data
and these priors, here is what follows. The most useful output is not any
parameter estimate but the trend line with its credible band, which needs almost
no modeling assumptions.

## References {#references}

- Dhurandhar, N. V., Schoeller, D., Brown, A. W., Heymsfield, S. B., Thomas, D., Sørensen, T. I. A., Speakman, J. R., Jeansonne, M., and Allison, D. B. (2015). Energy balance measurement: when something is not better than nothing. *International Journal of Obesity*.
- Lichtman, S. W., Pisarska, K., Berman, E. R., Pestone, M., Dowling, H., Offenbacher, E., Weisel, H., Heshka, S., Matthews, D. E., and Heymsfield, S. B. (1992). Discrepancy between self-reported and actual caloric intake and exercise in obese subjects. *New England Journal of Medicine*, 327, 1893-1898.
- Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., and Murphy, S. A. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health: key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*.
- Schoeller, D. A. (1990). How accurate is self-reported dietary energy intake? *Nutrition Reviews*.
