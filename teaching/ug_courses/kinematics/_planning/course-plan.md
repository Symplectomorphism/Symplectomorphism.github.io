# Kinematics and Machine Dynamics: editorial plan

## Source baseline

- Source repository: Symplectomorphism/kinematics, commit 232d73ae379cbecf66664a9f060e62e0f2160687.
- Primary authority: Textbook/kinematics_and_dynamics.pdf, dated May 25, 2020 (Fall 2020).
- Supporting sources: Textbook/Chapters/Chapter01.tex, Textbook/Bibliography.bib, Lectures/Lecture01/slides/terminology.tex.
- Preserve the PDF's chapter numbering and four main parts. Existing LectureNN and WeekNN directories are supporting material, not the new chapter numbering.
- First delivery: landing page and complete Chapter 1. User requested lecture content, without discussion questions and suggested-answer notes.
- Speaker notes contain source attribution and necessary technical qualifications only.
- Chapter 1 was reviewed and published. Preview Chapters 5–7 together on codex/kinematics-chapters05-07 before merging or publishing.

## Chapter 1 coverage

Printed pages 3–4 (PDF pages 19–20), section 1.1:

| Definitions | Deck coverage |
|---|---|
| 1.1–1.2: machine, mechanism | Machines and mechanisms; four-bar example |
| 1.3: kinematics | Kinematics |
| 1.4–1.5: dynamics, statics | Dynamics and statics |
| 1.6–1.7: synthesis, analysis | Analysis and synthesis |
| 1.8: link | Links and the fixed frame |
| 1.9: joint | Joints |
| 1.10–1.11: closed/open kinematic chains | Kinematic chains; arm and four-bar examples |
| 1.12: planar linkage | Planar motion |
| 1.13: spatial linkage | Spatial motion |

The topology wording clarifies mixed open/closed mechanisms instead of treating the textbook's simplified criterion as a universal graph classification. Wiper and spatial-arm examples are instructional additions. Mobility formulas and position-analysis derivations belong to later chapters.

## Figure provenance

- assets/chapter01/crank-rocker.png: unchanged Textbook/gfx/crank_rocker.png, PDF Fig. 3.1, printed p. 7. Chapter 3 follows Wilson and Sadler; the compiled notes do not give a separate figure credit.
- assets/chapter01/two-link-arm.png: unchanged Textbook/gfx/two_link_manipulator.png, PDF Fig. 6.1, printed p. 36. Retain the compiled-note attribution; no claim of newly created artwork.
- The compiled PDF is a local research source and is not copied into the public website.

## Subsequent chapters

Use one deck per textbook chapter initially. Split a long chapter into explicitly labeled parts only when reviewing that chapter, retaining its textbook chapter number.

Build order requested by the instructor on 2026-09-20 (textbook numbering stays unchanged):

1. Kinematics, Chapters 5–7: rigid motion, position analysis, velocity and acceleration. Build and review these three together.
2. Dynamics, Chapters 8–11: mass distribution, generalized forces, equations of motion, friction and impact.
3. Components, Chapters 12–13: gears and cams.
4. Return to Chapters 2–4: mobility, four-bar classification, slider-crank/quick-return mechanisms.

Chapter 11 includes complementarity problems, measure differential inclusions, and numerical methods. Decide with the instructor whether to cover the full treatment or separate foundational undergraduate material from advanced extensions before building that deck. Appendices A–C are potential support notes, not currently promised slide decks.

For each chapter: inventory every PDF section, compare the matching lecture sources, keep notation consistent, add worked examples where the source supports them, cite figures, render and inspect every slide, then review before proceeding.
