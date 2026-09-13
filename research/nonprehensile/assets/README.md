# Figures for the nonprehensile-manipulation deck

All 18 figures are generated, dependency-free, from the running example's
geometry:

    python make_figures.py

Output is self-contained SVG. Animation is SMIL, so it plays inside an
`<img>` tag with no JavaScript and no library.

| file | module |
|------|--------|
| `svgkit.py` | canvas, world→pixel transform, scene primitives, entity fixups |
| `fig_task.py` | `task`, `strip`, `prior-work` |
| `fig_state.py` | `features`, `template`, `catalog`, `state`, `actions` |
| `fig_solver.py` | `solver`, `gates`, `support`, `normals` |
| `fig_learn.py` | `graph`, `live`, `ranker`, `greedy`, `pipeline`, `budget` |

## Conventions

Identical to the report: `+x_W` points **left** on the page, `+z_W` up, `+y_W`
out of the page, so `z_W x x_W = y_W`. `θ` is the rotation about `+y_W` and
positive `θ` is the counterclockwise topple. The box therefore travels
right-to-left across the page.

    R(θ)(ξx, ξz) = (ξx cos θ + ξz sin θ,  −ξx sin θ + ξz cos θ)
    R(π/2)(ξx, ξz) = (ξz, −ξx)          # the report's Lemma "Box faces at M3"

Scene geometry, world origin at the floor–riser corner: floor `x ≤ 0, z ≤ 0`;
riser `x ≥ 0, 0 ≤ z ≤ h`; platform `0 ≤ x ≤ 2a, z = h`; box of side `a` with
`h = 3a/8`; manipulator radius `r = a/8`.

All world-space geometry is written once and is independent of the drawing
direction. Only three things carry the sign, and they are centralized in
`svgkit.py`:

| | |
|---|---|
| `Svg.P(x, z)` | world → pixel |
| `Svg.dx(Δx)` | horizontal pixel delta for SMIL `translate` |
| `Svg.rot_deg(θ)` | SVG angle for a world rotation, for SMIL `rotate` |

Set `X_RIGHT = True` in `svgkit.py` to mirror the page; nothing else changes.

## Figure titles

Figures omit their own title and subtitle by default, because the slide heading
already carries them; the band they would occupy is cropped from the `viewBox`.
Set `SHOW_TITLES = True` in `svgkit.py` and regenerate to get standalone
figures (for the report, or for sharing a single image).
