# Kinematics block: coverage and editorial decisions

Primary source: Textbook/kinematics_and_dynamics.pdf, Fall 2020, source repository Symplectomorphism/kinematics at 232d73ae379cbecf66664a9f060e62e0f2160687. The matching Chapter05–07.tex files were used for equation transcription. Supporting lecture files: Lecture02/slides/rotational.tex, Lecture03/slides/forward.tex, and Lecture04/slides/vel_acc_pts.tex.

## Coverage

| Source section | Slide treatment |
|---|---|
| 5.1 Rigid Body Transformations | Rigidity, points/vectors, handedness, distance and cross-product preservation, polarization identity |
| 5.2.1 Properties of rotation matrices | Frame columns, SO(3)/SO(n), group properties, coordinate conversion, composition, hat map, cross-product identities |
| 5.2.2 Exponential representation | Axis motion, matrix exponential, skew powers, Rodrigues formula, validity, axis recovery, zero/pi cases, nonuniqueness |
| 5.2.3 Euler representation | Elementary rotations, intrinsic ZYZ, expanded matrix, angle recovery, singularities, explicit ZYX convention |
| 5.3 Rigid Motion | Position and orientation, point versus vector action, SE(3) configuration set |
| 5.3.1 Homogeneous representation | Point/vector fourth coordinate, matrix action, composition, inverse and group properties |
| 5.3.2 Exponential coordinates and twists | Revolute/prismatic examples, linear-first wedge/vee maps, se(3), finite displacement, initial-pose multiplication, surjectivity; closed-form exponential added |
| 6.1 Introduction | Forward/inverse tasks, frames, open and closed chains |
| 6.2 Forward Kinematics | Full adjacent-transform derivation for elbow arm; four-bar closure; slider-crank closure; assembly feasibility; Newton solve and singularity checks |
| 6.3 Inverse Kinematics | Both elbow branches, shoulder-angle derivation, reachability, position versus pose, four-bar and slider-crank solutions |
| 7.1 Angular Velocity | Formal basis definition, cross-product differentiation theorem and derivation; matrix connection added |
| 7.2 Simple Angular Velocity | Definition, signed rate, basis derivation |
| 7.3 Auxiliary Reference Frames | Addition theorem, derivation, yaw/pitch example |
| 7.4 Angular Acceleration | Reference-frame derivatives, changing-axis term, nonadditivity and derived two-frame formula |
| 7.5 Velocity and Acceleration | Reference-frame point definitions |
| 7.6 Two Points Fixed On a Rigid Body | Velocity and acceleration theorems, derivations, normal/tangential interpretation, numerical example |
| 7.7 One Point Moving On a Rigid Body | Expanded transport formulas, Coriolis derivation, original coincident-point formulation, rotating bead and moving-line examples |

Chapter 5 has 36 slides; Chapter 6 has 29; Chapter 7 has 34, each including a title and references/transition slide. All are lecture content; no discussion questions or suggested-answer blocks were added.

## Notation and corrections

- R_ab maps B components to A components; p_ab is expressed in A. All three decks retain this convention.
- Chapter 5 uses u for the unit rotation axis, reserving omega for angular velocity in Chapter 7. Twist coordinates remain linear-first (v,u), as in the source.
- Non-unit Rodrigues evaluation uses the vector norm and a zero-angle limit.
- Axis recovery from (R-R^T)/(2 sin theta) requires 0 < theta < pi. Added the omitted half-turn case and corrected r31's missing subscript.
- Euler rotation order is explicit: intrinsic ZYZ; yaw-pitch-roll uses intrinsic ZYX, equivalently extrinsic XYZ applied roll then pitch then yaw. Corrected the contradictory source description of body-fixed XYZ with the ZYX label.
- SE(3) is identified with R^3 x SO(3) as a configuration set; its group law is the coupled homogeneous-matrix product.
- Offset revolute generator uses v=-u cross q for a fixed point q on the axis. The source matrix incorrectly substitutes the moving point p in that column.
- Four-bar figure 6.2 shows theta2 from the horizontal, while the text uses a relative coupler angle. The deck explicitly defines absolute alpha, beta, gamma; beta=theta1+theta2 in the textbook equation convention and gamma=theta3+pi for the figure's reversed output axis.
- Slider-crank position B=(q3,-a0) requires -q3 in the horizontal closure. Corrected Eq. (6.9) and standardized a1,a2,q1,q2,q3.
- Full-pose IK is distinguished from position-only IK. Added unreachable, branch-merging, and equal-length folded-arm cases.
- Simple angular-velocity derivation has db3/dt=0; the source repeats b1 in its last derivative.
- Explicit angular-acceleration addition and Coriolis derivations clarify the source's statements without changing their meaning.

## Figures and additions

- Chapter 5 uses source Textbook/gfx/rotation_about_point.png, simple_rot.png, general_rigid_motion.png, general_revolute.png, and general_prismatic.png (Figs. 5.1–5.4). Credits identify the compiled notes and Murray/Li/Sastry chapter source.
- Chapter 6 reuses the course's original two_link_manipulator.png (Fig. 6.1), renders Textbook/gfx/fourbar.eps to PNG (Fig. 6.2), and renders slider_crank.pdf to PNG (Fig. 6.3).
- Chapter 7 retains the original Lecture04 figures point_on_door.svg and bead_on_wire.svg. PNG exports use Inkscape's drawing bounds and white background; diagram contents are unchanged. Editable SVG originals are retained.
- All numerical values are instructional additions and are identified as such in notes. The Chapter 6 four-bar example continues into Chapter 7's velocity and acceleration solves.
- Transport theorem support comes from Appendix B; Newton support comes from Appendix C. No separate appendix decks are promised yet.
- Book authorship checked against https://www.cds.caltech.edu/~murray/books/MLS/; the compiled bibliography lists only Murray under a 2017 edition.

## Rendering and review

KaTeX 0.16.22 is vendored with license and WOFF2 fonts under assets/katex. It is scoped to these new decks; Chapter 1 and other courses retain their renderers. The project resources declaration includes the font files needed at runtime. Chapter HTML links target the Reveal output directly; alternate inherited HTML outputs follow the existing no-index redirect workflow.

Validated by rendering all three decks and both landing pages, inspecting every slide, browser checks with external HTTPS requests blocked, and numerical comparisons against SciPy matrix exponentials, forward-kinematics residuals, and finite differences. Review remains local until the instructor approves merging/publishing.
