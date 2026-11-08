"""Planar build, step 3: figures from the real code and the experiment's JSON.

Run from anywhere (Python 3.14 env of c-gnn/cmg):

    JAX_PLATFORMS=cpu ~/repos/research/c-gnn/cmg/.venv314/bin/python 03_figures.py <media dir>

1. step3_sweep.svg: break(g_v) with the disk low on -x_B. The line is the
   at-pose gate's margin of {m_x, s_v} along the tilt (the step-2 figure's
   column at -40 mm); the dots are the transition sweep's grid from 30 deg
   (recognized poses of {m_x, g_v, s_v}: tilt and the disk along -x_B),
   filled where {m_x, s_v} holds. The ring marks chi_t (the nearest pose
   with margin >= 0.02), the triangle the largest-margin pose.
2. step3_success.svg: successes per cost variant and edit (GPU run).
Prints the numbers the page quotes.
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "media"
CMG = os.path.expanduser("~/repos/research/c-gnn/cmg")
sys.path.insert(0, os.path.join(CMG, "tests"))
os.chdir(CMG)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.physics.planar_hold import PlanarHold
from cmg.physics.transition import TransitionSweep
from cmg.physics.world import ExecutionWorld, State
from cmg.scene.spec import SceneSpec
from cmg.storage import resolve_path
from test_planar_step2 import disk_on, mask, pivot

INK, MUTED, GRID = "#1f2328", "#656d76", "#d0d7de"
BLUE, RED = "#1c5cab", "#c0392b"

spec = SceneSpec.from_yaml(resolve_path("src/cmg/scene/families/step2d.yaml"))
cat = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
hold = PlanarHold(spec, cat, world)
sweep = TransitionSweep(hold, q0=spec.start_qpos)
old, new = mask(cat, "m_x", "g_v", "s_v"), mask(cat, "m_x", "s_v")

# --- 1. the sweep -------------------------------------------------------------
tilts = np.linspace(20, 70, 101)
at = jax.jit(lambda q: hold.edit_report(State(configuration=q, velocity=jnp.zeros(5)), old, new)["margin"])
curve = []
for d in tilts:
    p = pivot(math.radians(d))
    curve.append(float(at(jnp.asarray(p + disk_on(p, "xn", -0.04), jnp.float32))))
curve = np.array(curve)
first = tilts[np.argmax(curve >= 0)]
p30 = pivot(math.radians(30))
q30 = jnp.asarray(p30 + disk_on(p30, "xn", -0.04), jnp.float32)
r = jax.jit(lambda q: sweep(q, old, new, details=True))(q30)
gq = np.asarray(r["grid_q"]); rec = np.asarray(r["grid_recognized"]); sup = np.asarray(r["grid_supported"])
gm = np.asarray(r["grid_margin"])
keep = rec
seen = set()
uniq = []
for i in np.flatnonzero(keep):
    key = tuple(np.round(gq[i], 5))
    if key not in seen:
        seen.add(key); uniq.append(i)
uniq = np.array(uniq)
chi, best = np.asarray(r["chi_t"]), np.asarray(r["chi_best"])
print(f"at-pose margin at 30 deg: {curve[np.argmin(abs(tilts - 30))]:.3f}; first tilt held (disk -40 mm): {first:.1f} deg")
print(f"sweep: free directions {int(r['n_free'])}, distinct recognized poses {len(uniq)}, of which held "
      f"{int(sup[uniq].sum())}; chi_t tilt {math.degrees(chi[2]):.1f} deg margin {float(r['margin']):.3f}; "
      f"max-margin tilt {math.degrees(best[2]):.1f} deg margin {float(r['margin_best']):.3f}")
for pts, rng_ in ((9, 0.05), (9, 0.08)):
    fine = TransitionSweep(hold, q0=spec.start_qpos, points=pts, range_=rng_)
    rf = jax.jit(lambda q: fine(q, old, new, details=True))(q30)
    qf = np.asarray(rf["grid_q"]); recf = np.asarray(rf["grid_recognized"]); supf = np.asarray(rf["grid_supported"])
    _, first_idx = np.unique(np.round(qf, 5), axis=0, return_index=True)
    u = np.zeros(len(qf), bool); u[first_idx] = True
    print(f"sweep with {pts} points, range {rng_}: recognized {int((recf & u).sum())}, held {int((supf & u).sum())}; "
          f"chi_t tilt {math.degrees(float(rf['chi_t'][2])):.1f} deg margin {float(rf['margin']):.3f}; "
          f"max-margin tilt {math.degrees(float(rf['chi_best'][2])):.1f} deg margin {float(rf['margin_best']):.3f}")


fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.axhline(0.0, color=GRID, lw=1.0, zorder=0)
ax.plot(tilts, curve, color=INK, lw=2.0, label="gate at the pose (disk 40 mm below center)")
deg = np.degrees(gq[uniq, 2])
ax.scatter(deg[~sup[uniq]], np.clip(gm[uniq][~sup[uniq]], -1, 1), s=22, facecolors="white", edgecolors=MUTED,
           linewidths=1.0, label="sweep pose, not held", zorder=3)
ax.scatter(deg[sup[uniq]], np.clip(gm[uniq][sup[uniq]], -1, 1), s=22, color=BLUE, label="sweep pose, held", zorder=6)
ax.scatter([30], [curve[np.argmin(abs(tilts - 30))]], s=60, color=RED, zorder=4)
ax.annotate("χ₀ = 30°: refused at the pose", (30, curve[np.argmin(abs(tilts - 30))]), xytext=(21, 0.55),
            fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.scatter([math.degrees(chi[2])], [float(r["margin"])], s=140, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
ax.annotate(f"χ_t (nearest held): {math.degrees(chi[2]):.0f}°", (math.degrees(chi[2]), float(r["margin"])),
            xytext=(47, -0.55), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.scatter([math.degrees(best[2])], [min(float(r["margin_best"]), 1.0)], marker="^", s=70, color=INK, zorder=5)
ax.annotate("largest margin", (math.degrees(best[2]), min(float(r["margin_best"]), 1.0)), xytext=(57, 0.8),
            fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("box tilt θ (deg)", fontsize=9, color=INK)
ax.set_ylabel("margin of {m_x, s_v} (m g, clipped to ±1)", fontsize=9, color=INK)
ax.set_ylim(-1.05, 1.05)
ax.tick_params(labelsize=8, colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "step3_sweep.svg", bbox_inches="tight")

# --- 2. the experiment --------------------------------------------------------
path = OUT / "step3_experiment_gpu.json"
if path.exists():
    data = json.load(open(path))
    variants = data["variants"]
    edits = list(dict.fromkeys(r["edit"] for r in data["records"]))
    grid = np.zeros((len(variants), len(edits)))
    text = [["" for _ in edits] for _ in variants]
    for i, v in enumerate(variants):
        for j, e in enumerate(edits):
            rs = [r for r in data["records"] if r["variant"] == v and r["edit"] == e]
            k = sum(r["success"] for r in rs)
            grid[i, j] = k / max(1, len(rs))
            text[i][j] = f"{k}/{len(rs)}"
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f8fa", "#9ec5f0", "#3987e5", "#1c5cab"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(variants)):
        for j in range(len(edits)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > 0.6 else INK)
    names = {"A": "A current", "B": "B paper", "C": "C +margin, dwell", "D": "D +subgoal", "E": "E D, mm-scaled"}
    ax.set_yticks(range(len(variants)), [names.get(v, v) for v in variants], fontsize=8, color=INK)
    short = [e.replace("compound(s -> s_v, g -> g_v)", "pivot").replace("transfer(s_v -> t)", "s_v→t (land)")
             .replace(" from 30 deg", "\n@30°") for e in edits]
    ax.set_xticks(range(len(edits)), short, fontsize=7.5, color=INK, rotation=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "step3_success.svg", bbox_inches="tight")
    for v in variants:
        print(v, json.dumps({k: val for k, val in data["summary"][v].items() if k != "per_edit"}))
