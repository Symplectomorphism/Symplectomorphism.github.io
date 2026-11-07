"""Planar build, step 2: where the disk must push for the box to hold on the step corner.

Run from c-gnn/cmg on branch phase1/planar (Python 3.14 env):

    JAX_PLATFORMS=cpu .venv314/bin/python <this file> <website media dir>

For the box tipped by theta onto the step corner (bottom corner on the floor,
+x_B on v_S) and the disk on -x_B at local height `mid`, computes H for
break(g_v) at {m_x, g_v, s_v}, i.e. whether {m_x, s_v} alone holds the box,
with its support margin (PlanarHold, the default gate: LP with the disk's
actuator limits, l1 margin). Writes step2_pivot_margin.svg
and prints the numbers the page quotes.
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from cmg.catalog.templates import Catalog
from cmg.physics.planar_hold import PlanarHold
from cmg.physics.world import ExecutionWorld
from cmg.scene.spec import SceneSpec

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
spec = SceneSpec.from_yaml("src/cmg/scene/families/step2d.yaml")
cat = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
hold = PlanarHold(spec, cat, world)
A, R, H = 0.125, 0.015625, 0.125 / 4
INK, MUTED = "#1f2328", "#656d76"


def pivot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [-H * math.tan(theta) - A / 2 * (c - s), A / 2 * (s + c), theta]


def disk_on_xn(box, mid):
    x, z, theta = box
    c, s = math.cos(theta), math.sin(theta)
    u, w = -A / 2 - R, mid
    return [x + u * c + w * s, z - u * s + w * c]


def mask(*names):
    f = np.zeros(len(cat), dtype=bool)
    for n in names:
        f[cat.index(n)] = True
    return jnp.asarray(f)


old, new = mask("m_x", "g_v", "s_v"), mask("m_x", "s_v")
degs = np.linspace(4, 80, 39)
mids = np.linspace(-A / 2 + 0.002, A / 2 - 0.002, 41)
Q = jnp.asarray([pivot(math.radians(d)) + disk_on_xn(pivot(math.radians(d)), m) for d in degs for m in mids],
                jnp.float32)
band = jax.jit(jax.vmap(lambda q: cat.band(config=q)))(Q)
on_ok = np.asarray(jnp.all(~old | band, axis=1)).reshape(len(degs), len(mids))
rep = jax.jit(jax.vmap(lambda q: hold.edit_report(q, old, new)))(Q)
margin = np.asarray(rep["margin"]).reshape(len(degs), len(mids))
held = np.asarray(rep["supported"]).reshape(len(degs), len(mids))
print(f"pose grid {len(degs)} x {len(mids)}; formation {{m_x, g_v, s_v}} on at {on_ok.mean():.0%} of poses")
capped = np.asarray(rep["capped"]).reshape(len(degs), len(mids))
print(f"break(g_v) admitted at {held[on_ok].mean():.0%} of them; capped margin at {capped[on_ok].mean():.0%}")
print(f"margin range where held: {margin[on_ok & held].min():.3f} .. {margin[on_ok & held].max():.3f}")
for d in (10, 30, 45, 60):
    i = int(np.argmin(abs(degs - d)))
    ok = mids[held[i] & on_ok[i]]
    print(f"  {d:2d} deg: holds for disk height {1e3 * ok.min():+.0f} .. {1e3 * ok.max():+.0f} mm" if ok.size else f"  {d} deg: never")

shown = np.where(on_ok, np.clip(margin, -1.0, 1.0), np.nan)
cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
fig, ax = plt.subplots(figsize=(7.6, 4.2))
im = ax.pcolormesh(1e3 * mids, degs, shown, cmap=cmap, norm=TwoSlopeNorm(0.0, -1.0, 1.0), shading="nearest")
unbounded = on_ok & capped
yy, xx = np.nonzero(unbounded)
ax.scatter(1e3 * mids[xx], degs[yy], s=5, marker=".", color="white", linewidths=0)
ax.contour(1e3 * mids, degs, np.where(on_ok, held.astype(float), np.nan), levels=[0.5], colors=INK, linewidths=1.2)
ax.plot([0], [30], "o", mfc="none", mec=INK, ms=9, mew=1.4)
ax.annotate("disk at mid-face, 30°: refused", (0, 30), xytext=(6, 77.5), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.plot([-40], [30], "D", mfc="none", mec=INK, ms=7, mew=1.4)
ax.annotate("disk low, 30°: refused\n(motors too weak)", (-40, 30), xytext=(-25, 14), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.plot([-40], [45], "s", mfc="none", mec=INK, ms=8, mew=1.4)
ax.annotate("chain pose (disk low, 45°): admitted", (-40, 45), xytext=(-60, 77.5), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(degs[0] - 1, 82)
ax.set_xlabel("disk height on −x_B from the face center (mm; negative = toward the floor)", fontsize=9, color=INK)
ax.set_ylabel("box tilt θ (deg)", fontsize=9, color=INK)
ax.tick_params(labelsize=8, colors=MUTED)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("support margin of {m_x, s_v} (units of m g; clipped to ±1)", fontsize=8, color=INK)
cb.ax.tick_params(labelsize=7, colors=MUTED)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "step2_pivot_margin.svg", bbox_inches="tight")
print("wrote step2_pivot_margin.svg")
