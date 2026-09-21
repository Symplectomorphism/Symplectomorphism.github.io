# Chapters 2–4 and 12–13: coverage and validation

## Source baseline and authorization

Primary source: Satici, *Kinematics and Machine Dynamics*, May 25, 2020 PDF, source repository commit `232d73ae379cbecf66664a9f060e62e0f2160687`. The instructor authorized completion, merging to master, and publication without a separate review on September 21, 2026.

Chapters 2–4 were also checked against `Lectures/Lecture01/slides/{mobility,classification,quick_return}.tex`. Chapters 12–13 follow the compiled textbook; its numerous missing-figure placeholders are replaced with original vector diagrams and plots.

| Chapter | Source coverage | Added lecture support |
|---|---|---|
| 2 | pp. 5–6: joint freedoms, planar and spatial mobility, common pairs | Four-bar, slider-crank, cam and serial-chain examples; constraint rank, coaxial hinges, singularity qualification |
| 3 | pp. 7–9: closure, Grashof, inversions, change points, §3.1 transmission angle | Worked inversion and variable-ground classification; law-of-cosines calculation and output torque |
| 4 | pp. 11–13: §4.1 in-line/offset limits, stroke and timing; §4.2 quick return | Branch-aware position derivation, endpoint acceleration, numerical offset example, slotted-lever tangent construction and drag-link discussion |
| 12 | pp. 89–101: §§12.1–12.8, including involute, pressure angle, changing center distance, backlash, nomenclature, interference, gear families, belts/chains, and all train types | Involute equations, contact ratio, force resolution, efficiency assumptions, reverted geometry, full planetary relative-speed derivation and examples |
| 13 | pp. 103–107: §§13.1–13.4, including terminology, S–V–A–J, polynomials, single/double dwells and sizing | Chain-rule derivations, 3–4–5 and 4–5–6–7 solutions, cycloidal comparison, single-dwell polynomial, full-cycle plots, pressure-angle bound, curvature and profile-offset construction |

## Corrections and qualifications

- The Lecture01 planar one-DoF-joint count omits a factor of two; use `3(n_L-1)-2J_1-J_2`. Equality of counts and mobility requires independent constraints at a regular configuration. Jacobian nullity at a singular posture does not necessarily equal finite mobility.
- Distinguish four-bar closure equality (a degenerate quadrilateral) from Grashof equality (a change-point linkage). The variable-ground example with other lengths 100, 200, 300 mm has Grashof range 200–400 mm; endpoints are change points.
- Offset slider formulas assume `L>R+|E|`, a specified assembly branch, and constant speed for angle-based time ratios. The source's acceleration statement is qualified; zero velocity does not imply zero acceleration.
- The source/lecture quick-return stroke statements mix topology and link numbering and disagree (`2l_3` versus `l_3`). No universal stroke formula is asserted for an unspecified drag-link/slotted arrangement. The tangent construction explicitly states its pivot separation and crank radius.
- Gear torque ratios use magnitudes and a stated power-flow efficiency. The text's signed ratio typography and claims about absence of forces are not carried over. Teeth and bearings do transmit forces.
- Worms can have multiple starts and are not universally self-locking. Backdrivability depends on lead angle and friction. Catalog efficiency ranges are replaced with explicitly assumed example efficiencies.
- A root below the base circle does not alone establish interference. Contact path, cutter geometry, profile shift, and mating tooth counts matter.
- Polynomial coefficients use `c_m`, correcting the repeated `c_0` in source Eq. 13.1. Continuity through the second derivative is described as continuous S, V, A rather than ambiguous “third-order continuity.”
- For cams, distinguish the largest centered circle contained inside the cam from the minimum radius of the roller-center pitch curve. Normal offset, not radial subtraction, generates the general roller cam surface.
- Pressure-angle and curvature equations are scoped to an in-line translating radial roller follower; they are not applied to offset or flat-faced followers. Sampled curvature is a preliminary numerical check, not manufacturing certification.

## Figure provenance

All `assets/final/*.svg` files are original mathematical schematics/plots: four-bar, slider-crank, tangent quick-return geometry, pitch circles and line of action, involute, compound train topology, planetary pitch circles, radial cam offset, and full-cycle S–V–A–J. Gear circles depict pitch geometry, not manufactured tooth profiles. The compound drawing is a shaft/mesh schematic, not a scale assembly.

The existing Chapter 1 crank-rocker PNG is reused with its original compiled-textbook Fig. 3.1 attribution. No textbook PDF or new third-party scan is published.

Supplemental dimension reference: KHK, *Calculation of Gear Dimensions*, linked in Chapter 12. Main conceptual coverage remains the user's compiled textbook.

## Verification

Independent polynomial coefficient/endpoint and derivative-extremum checks; dense numerical slider-position extrema versus analytic dead centers; piecewise Grashof classification against sorted-length inequalities; transmission-angle/torque arithmetic; compound and planetary mesh ratios; full-cycle cam angle and curvature sampling. Browser checks and final rendering results are recorded after inspection.

Final validation: 114 new slides (17, 17, 18, 32, 30 by chapter) rendered in the default HTML/Reveal.js formats. All slides visually inspected; corrected the S–V–A–J plot height and gear-diagram labels. Browser checks report no overflow, broken images, KaTeX errors, JavaScript errors, or failed resources after corrections. All 25 local landing-page links return HTTP 200; all five inherited HTML variants redirect to their Reveal.js decks. `git diff --check` passes.
