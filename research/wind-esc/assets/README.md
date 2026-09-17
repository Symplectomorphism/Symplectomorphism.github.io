# Figures for the wind-turbine gradient-estimation deck

All 15 figures are generated from the NREL 5-MW reference turbine's constants
and from numbers quoted in the source papers:

    ../../../.venv/bin/python make_figures.py

Output is SVG with `svg.fonttype: none`, so the slide's own typeface is used and
the palette matches `deck.scss` exactly.

| file | module | what it shows |
|------|--------|---------------|
| `windkit.py` | (shared) | palette, turbine constants, `C_P(λ)`, plot styling, block-diagram primitives |
| `fig_setup.py` | `regions`, `cp-curve`, `blind`, `ray` | the control problem and why it is blind |
| `fig_esc.py` | `esc-loop`, `logfix`, `evidence`, `piesc` | extremum seeking and the published results |
| `fig_open.py` | `estimand`, `channels`, `biasvar`, `stability`, `identify`, `hessian`, `bound` | the five open questions |

## The `C_P` curve

`windkit.cp` is Heier's analytic surface, affinely remapped so the peak sits at
`(λ, C_P) = (7.5, 0.49)`, the pair the 2019 LES results are quoted against.
(The 2022 *Energies* paper uses `7.55` and `0.48`.)

Heier is used rather than a parabola for one reason: the real curve is steep on
the stall side and gentle above the peak, so `J'''` at the optimum is non-zero.
That non-zero third derivative is exactly what displaces a finite-difference
estimator's fixed point from the true optimum, which is the subject of the `Q1`
slides. A symmetric stand-in would quietly erase the effect.

## Numbers and where they come from

| quantity | value | source |
|---|---|---|
| `R`, `I`, `N` | 63 m, 35.44e6 kg·m², 97 | Jonkman et al. 2009; Kumar & Rotea 2022 Table 1 |
| settling times under shear + 10% TI | 31.0 / 8.0 / unstable vs 8.1 / 8.1 / 8.0 min | Ciri et al. 2019, Table 4 |
| converged `λ̄` under shear + TI | 7.85 against a design 7.5 | Ciri et al. 2019, Table 4 |
| energy vs. an oracle baseline | LP-ESC −14.2%, LP-PIESC −0.3% | Kumar & Rotea 2022, Table 4 |
| dither amplitudes | 33% of `u_opt`; LP-PIESC 25% of that | Kumar & Rotea 2022, Tables 2 and 3 |
| farm power gain, 12-turbine tunnel | +8.9% | Rotea et al. 2024, §3.2.1 |

`bound.svg` is explicitly schematic and labelled as such on the figure: the
performance bound it draws does not exist yet, which is the point of that slide.
