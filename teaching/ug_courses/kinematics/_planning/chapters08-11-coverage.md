# Chapters 8–11 coverage and validation

## Scope and source baseline

Source repository: `Symplectomorphism/kinematics`, tree `232d73ae379cbecf66664a9f060e62e0f2160687`. Primary authority is `Textbook/kinematics_and_dynamics.pdf`, May 25, 2020 / Fall 2020 version. The matching Chapter08–11 TeX sources and Lecture05/06 slides were compared. Lecture07 and Lecture08 are gears and cams, so their numbers are not used as chapter numbers.

Review branch: `codex/kinematics-chapters08-11`. The user requested all four chapters, including Chapter 11's advanced material. A dedicated warning slide precedes the advanced sequence. Speaker notes contain sources, assumptions, and editorial qualifications, not discussion questions or suggested answers.

## Coverage

| Source | Deck coverage |
|---|---|
| 8.1 Mass center | Particle formula, zero first mass moment, original lecture cube example with completed optimization |
| 8.2 Curves, surfaces, solids | Density units, continuous formula, nonuniform rod, composite body/cutout |
| 8.3 Inertia vector and scalars | Vector operation, symmetry, moment, product, radius of gyration |
| 8.4 Orthogonal unit vectors | Linearity, three basis directions, bilinear scalar form |
| 8.5 Inertia matrix | Cartesian entries/sign convention, properties, rod integration, basis transformation |
| 8.6 Parallel axes | Matrix and scalar forms, original lecture cylindrical pendulum, numerical shift |
| 8.7 Principal moments | Eigenproblem, symmetry, repeated eigenvalues, planar formula, worked eigenbasis |
| 9.1 Moments, bound vectors, resultants | Lines of action, transfer theorem, offset-force and beam examples |
| 9.2 Couples and torque | Couple diagram, reference-point independence, numerical example |
| 9.3 Equivalence and replacement | Force–couple reduction, single-force condition, wrench decomposition, distributed load |
| Chapter 9 additions | Power identity, generalized forces through virtual work, revolute-joint example |
| 10.1 Static analysis | Assumptions, free-body diagrams, 3D/planar balance, supported beam |
| 10.2 Dynamic analysis | Newton–Euler derivation, basis conventions, Euler components, gyroscopic torque, COM block form, shifted-point derivation and block form, planar restriction |
| Chapter 10 additions | Eccentricity scaling, falling rod with angular acceleration and pin reactions, driven pendulum, energy check, per-link inverse-force workflow, bridge to generalized coordinates |
| 11.1 Rigid bodies and friction | Gap, nonadhesive contact, impulse idealization |
| 11.1.1 Coulomb friction | Sliding/sticking, ramp and stopping example, graph completion, regularization, cone, maximum dissipation |
| 11.1.2 Impact models | Momentum jump, Newton restitution, normal-impact example, Poisson model, limitations of a constant restitution parameter |
| Advanced transition | Explicit warning that the remaining material goes beyond standard undergraduate friction/impact |
| 11.1.3 Painleve problem | Original figure, consistent rod equations, contact acceleration, numerical inconsistent sliding mode, impulses |
| 11.1.4 Complementarity | Componentwise definition, scalar solvable/unsolvable LCPs, KKT conditions, solver qualifications |
| 11.1.5 Measure differential inclusions | BV velocity, atomic derivative, measure momentum, gap/measure complementarity, friction variation measure, recession cone |
| 11.2 Formulation/simulation | Four numerical approaches and their modeling tradeoffs |
| 11.2.1 Continuous problem | Generalized Jacobians, Lagrange equations, smooth contact equations, measure formulation, initial conditions and separate impact law |
| 11.2.2 Numerical methods | Frozen-geometry impulse step, normal complementarity, numerical frictional impact, polyhedral cone, friction KKT, contact-space matrix, full LCP, copositivity, exact-cone formulation, drift and validation |

The advanced material presents formulations and worked examples. It does not reproduce the source's functional-analysis existence and convergence proofs or its historical claims about unresolved research as present-day claims.

## Editorial corrections and notation

- **Chapter 8 Eq. (8.6):** remove the extraneous vector `p` preceding the scalar integrand. The correct integrand is `(p × na) · (p × nb) dm`.
- **Chapter 8 p. 57:** equal diagonal entries `Iaa = Ibb` imply arbitrary in-plane principal directions only when `Iab = 0`. Use `atan2(2B, A-D)` and the symmetric 2×2 eigenvalues to handle equal diagonals with nonzero product.
- Use `G` for the mass center and column coordinate vectors consistently.
- Recompute the cylindrical-pendulum values before rounding. The support sketch explains why `d = 0.25 m` is larger than `h/2 = 0.215 m`. The source lecture's separate -75-degree matrix contains `0.3117` in its last entry; that erroneous value is not carried into this deck.
- **Chapter 9:** distinguish sliding a force along its line of action from preserving internal stresses. The virtual-work slides are additions, since the compiled chapter focuses on force systems despite its title.
- **Chapter 10:** `F` is the external resultant, not necessarily a physical force applied at the reference point. Inertial acceleration expressed in body coordinates differs from the ordinary derivative of velocity coordinates. Observation frame and expression basis are distinct.
- **Chapter 10 planar restriction:** only the normal component of the gyroscopic term automatically vanishes. The full vector can require out-of-plane reaction moments.
- **Chapter 11 Painleve geometry:** Fig. 11.2 measures theta from horizontal. Consequently `yc = y - (l/2) sin(theta)`, and the centrifugal gap-acceleration term is `(l/2) sin(theta) theta_dot^2`. The PDF uses inconsistent cosines. The normal-force coefficient is re-derived from the same geometry.
- **Chapter 11 restitution:** use positive Poisson impulse magnitudes to avoid a sign ambiguity. Apply impact laws to closing/candidate contacts, not all separated or merely touching configurations. Separation is permitted with zero impulse.
- **Chapter 11 generalized coordinates:** use normal and tangent Jacobians explicitly. Their transposes map physical reactions to generalized forces. `v = q_dot` is a local-coordinate assumption; an arbitrary orientation-speed parametrization can need a separate kinematic map.
- **Chapter 11 p. 81:** use `f = Qext - C(q,v)v - grad V` instead of copying the source expression for `k`, which omits velocity factors.
- **Chapter 11 friction stationarity:** use the unscaled KKT conditions consistent with the stated dissipation objective and constraint. The extra `mu` in Eqs. (11.17)/(11.21) is absent in Eqs. (11.27)/(11.28) and is not carried over.
- **Chapter 11 recession cone:** replace the malformed limit in Eq. (11.14) by the closed-convex-set definition.
- Distinguish normal force `N` from impulse `Pn`, restitution `e` from the vector of ones, and gap `g` from gravity `g0` where both appear together.
- Use a frozen-mass-matrix variant of the source time step and state this assumption. Specify the common free velocity in the LCP right-hand side. Copositivity alone is not presented as an unconditional existence theorem.

## Figures and documents

- `assets/chapter08/cube.png`: unchanged `Lectures/Lecture05/figures/cube.png`, also used in the original mass-center lecture example.
- `assets/chapter11/brick_on_ramp.png`, `painleve_prob.png`, `regularized_Coulomb.png`, `polyhedral_fc.png`: unchanged `Textbook/gfx/` images, corresponding to Figs. 11.1–11.4. Chapter 11 follows David E. Stewart, SIAM Review 42(1), 3–39 (2000), DOI 10.1137/S0036144599360110. No claim of original artwork is made for these figures.
- Nine new editable SVG schematics show the cylindrical pendulum, offset force, beam loads, force couple, revolute-joint loading, supported beam, body-point shift, falling-rod free body, and normal impact. They match the deck-specific geometry and are labeled as original additions in the notes.
- `resources/syllabus-fall2021.pdf`: byte-for-byte copy of `Syllabus/Fall_21/main.pdf`, blob `b357ba72cb9a103a796abebe203ccf8fde6cd988`, 42,953 bytes. The current GitHub file was checked against this blob. The document retains its original Fall 2021 date and institutional details.
- Landing-page links use the instructor-specified Cornell Dynamics PDF and Caltech MLS second-edition information page. The latter is labeled as an information page, not a completed downloadable second edition.

## Validation

Numeric checks independently verify the cube projection and optimum, rod integration, inertia shift/rotation/eigenvalues, force/moment transfer, rigid-body power, shifted Newton–Euler blocks, falling-rod reactions, gyroscopic torque, ramp stopping, restitution impulse/energy, and a complete frictional-impact LCP solution. Browser checks cover every slide, local math dependencies, missing images, overflow, and the syllabus/course links. All four decks are rendered in Reveal.js and the default multi-format publication path.

Final slide counts (including titles): Chapter 8: 27; Chapter 9: 22; Chapter 10: 28; Chapter 11: 48. Total: 125. All slides were visually reviewed. The final browser pass reported zero overflowing elements, broken images, KaTeX errors, JavaScript errors, or failed local resources. The landing-page local links returned HTTP 200; the syllabus returned a PDF signature and matched the source file hash. Source attribution remains in hidden speaker notes, and inherited HTML variants are noindex redirects. `git diff --check` passed.
