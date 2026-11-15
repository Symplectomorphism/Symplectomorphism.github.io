"""Step 8b: what each search round tried, rejected and accepted.

Runs the same two batches as 08_search.py, but records every solver call the
search makes, rejected ones included (the search itself keeps only accepted
results). For each attempt it recomputes

  * the search's verdict: every requested bit in band at the final step, and
  * the solver's own verdict (MPPISolver's success checks): no penetration
    beyond epsilon_pen at any step, requested bits within epsilon_on from the
    deadline on, dropped bits at least R away, and dwell drift within tolerance,

and draws one panel per attempt in the x-z plane: start pose (dashed), final
pose (solid), the disk's path, and both verdicts. Writes
walkthrough/08b_search_rounds.png (and a zoomable .svg) and prints the same
information.
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works over SSH)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from cmg.config.search_config import SearchConfig
from cmg.config.solver_config import SolverConfig
from cmg.physics.world import State
from cmg.search import Search

from _common import CMG, template_names

base = SolverConfig.from_yaml(str(CMG / "assets/configs/solver/step-base.yaml"))
solver_config = SolverConfig(
    solver_type=base.solver_type,
    solver_args=[],
    solver_overrides={**base.solver_overrides, "n_samples": 64, "maxiters": 2},
    scene_spec=base.scene_spec,
)
search = Search.from_specs(SearchConfig(batch_size=4), solver_config, base.scene_spec)
catalog = search.candidate_formations.catalog
world = search.candidate_formations.exec_world
solver = search.local_solver
names = [n.replace(":boundary", "") for n in template_names(base.scene_spec, catalog)]
EPS_PEN, EPS_ON = float(catalog.epsilon_pen), float(catalog.epsilon_on)
gaps_along = jax.jit(jax.vmap(lambda q: catalog.gap(pose=world.pose(State(q, jnp.zeros_like(q))))))

# Record every batch the search sends to the solver, accepted or not.
calls = []
solve_batch = search._solve_batch


def recording(states, old, new, op_types):
    out = solve_batch(states, old, new, op_types)
    calls.append((np.asarray(old), np.asarray(new), out))
    return out


search._solve_batch = recording


def verdicts(q, old, new):
    """Both verdicts for one attempt, from its qpos trajectory (T, 5)."""
    g = np.asarray(gaps_along(jnp.asarray(q)))  # (T, 24)
    final = g[-1]
    in_band = (final >= -EPS_PEN) & (final < EPS_ON)
    missing = [(names[i], final[i]) for i in np.flatnonzero(new & ~in_band)]
    reasons = []
    t, c = np.unravel_index(np.argmin(g), g.shape)
    if g[t, c] < -EPS_PEN:
        reasons.append(f"penetration {-g[t, c] * 1e3:.1f} mm on {names[c]}")
    deadline = g[solver.horizon - 1:]
    late = [names[i] for i in np.flatnonzero(new & np.any(deadline >= EPS_ON, axis=0))]
    if late:
        reasons.append("out of band after deadline: " + ", ".join(late))
    leaving = old & ~new
    if np.any(leaving & np.any(deadline < solver.R, axis=0)):
        reasons.append("dropped contact not cleared")
    drift = np.abs(q[solver.horizon:] - q[solver.horizon - 1]).max()
    if drift > solver.dwell_tolerance:
        reasons.append(f"dwell drift {drift * 1e3:.1f} mm")
    return missing, reasons, int(in_band.sum())


def edit_label(old, new):
    made = [names[i] for i in np.flatnonzero(new & ~old)]
    broken = [names[i] for i in np.flatnonzero(old & ~new)]
    return " ".join([f"make {m}" for m in made] + [f"break {b}" for b in broken]) or "transport"


def draw_scene(ax, q_start, q):
    h, a, r = 0.03125, 0.125, 0.015625
    ax.fill([-0.3, 0, 0, 0.12, 0.12, -0.3], [0, 0, h, h, -0.02, -0.02], color="0.85", lw=0)
    ax.plot([-0.3, 0, 0, 0.12], [0, 0, h, h], color="0.3", lw=1)

    def box(qq):
        c, s = np.cos(qq[2]), np.sin(qq[2])
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * a / 2
        return np.c_[qq[0] + local[:, 0] * c + local[:, 1] * s,
                     qq[1] - local[:, 0] * s + local[:, 1] * c]

    ax.add_patch(Polygon(box(q_start), fill=False, ls="--", ec="0.5", lw=0.8))
    ax.add_patch(Circle(q_start[3:5], r, fill=False, ls="--", ec="0.5", lw=0.8))
    ax.add_patch(Polygon(box(q[-1]), fc="#e8a33d", ec="#7a4f0e", alpha=0.85, lw=1))
    ax.add_patch(Circle(q[-1, 3:5], r, fc="#3d6fe8", ec="#14307a", lw=1))
    ax.plot(q[:, 3], q[:, 4], color="#3d6fe8", lw=0.8)
    ax.plot(q[:, 0], q[:, 1], color="#7a4f0e", lw=0.8)
    ax.set_xlim(0.08, -0.25)  # +x points left, as in the paper's figures
    ax.set_ylim(-0.015, 0.165)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


rounds = []
for batch in (1, 2):
    first = len(calls)
    search.run_single()
    attempts = []
    for old, new, (trajectories, _, accepted) in calls[first:]:
        q = np.asarray(trajectories.configuration)
        for i in range(len(new)):
            missing, reasons, nband = verdicts(q[i], old[i], new[i])
            attempts.append(dict(label=edit_label(old[i], new[i]), q=q[i], accepted=bool(accepted[i]),
                                 missing=missing, reasons=reasons, stored=int(new[i].sum()), nband=nband))
    rounds.append(attempts)

fig, axes = plt.subplots(len(rounds), 4, figsize=(16, 4.4 * len(rounds)), squeeze=False)
for r, attempts in enumerate(rounds):
    print(f"\nRound {r + 1}: {len(attempts)} tried, {sum(a['accepted'] for a in attempts)} accepted")
    for k, (ax, att) in enumerate(zip(axes[r], attempts)):
        draw_scene(ax, att["q"][0], att["q"])
        search_line = ("search: ACCEPT" if att["accepted"] else
                       "search: REJECT, not in band at the end: "
                       + ", ".join(f"{n} ({g * 1e3:+.1f} mm)" for n, g in att["missing"]))
        solver_line = "solver: success" if not att["reasons"] else "solver: fail, " + "; ".join(att["reasons"])
        node_line = (f"node stores {att['stored']} bits; {att['nband']} in band at its pose"
                     if att["accepted"] else "")
        print(f"  {att['label']}\n    {search_line}\n    {solver_line}" + (f"\n    {node_line}" if node_line else ""))
        ax.set_title(f"Round {r + 1}: {att['label']}", color="#1a7f37" if att["accepted"] else "#b42318",
                     fontsize=10, fontweight="bold")
        text = "\n".join(textwrap.fill(t, 62) for t in (search_line, solver_line, node_line) if t)
        ax.text(0.0, -0.04, text, transform=ax.transAxes, va="top", fontsize=8)
        for side in ax.spines.values():
            side.set_edgecolor("#1a7f37" if att["accepted"] else "#b42318")
            side.set_linewidth(2)
fig.suptitle("Search rounds (+x to the left, as in the paper): dashed = start pose, solid = final pose, lines = paths of disk (blue) "
             "and box center (brown); green = accepted, red = rejected", fontsize=11)
fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12, wspace=0.08, hspace=0.75)
out = Path(__file__).with_name("08b_search_rounds.png")
fig.savefig(out, dpi=200)
fig.savefig(out.with_suffix(".svg"))  # vector copy: zoom without blur
print(f"\nWrote {out} and {out.with_suffix('.svg').name}")
