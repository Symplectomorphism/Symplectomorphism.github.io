"""Planar build, node identity (Decisions 17 and 18): which pairs of states the search treats as one node.

Run from c-gnn/cmg on branch phase1/planar (Python 3.14 env):

    JAX_PLATFORMS=cpu PYTHONPATH=src:tests .venv314/bin/python <this file> <website media dir>

Uses the real code: the node key ``NodeKey`` (cmg/search/contact_state.py) with the
merge rule of ``SearchState`` (cmg/search/__init__.py), the earlier grid key
``Discretizer`` (1 mm / 0.01 rad cells from step2d.yaml), and the recognizer
``Catalog.band``. For each example pair it prints and saves (node_identity.json):
the two formations, the grid keys, the full Xiao-Ji contact state (formation +
connected component; computed here as "same formation, both poses in the one
component of the box's floor region", which holds for these flat poses), and the
node key. Writes node_identity.svg (the scene with the example poses).
"""
import json
import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.physics.world import State
from cmg.scene.spec import SceneSpec
from cmg.search import Discretizer, RealizedPath, SearchState
from cmg.search.contact_state import NodeKey
from cmg.storage import resolve_path
from test_planar_step2 import A, FLAT, H, R

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
spec = SceneSpec.from_yaml(resolve_path("src/cmg/scene/families/step2d.yaml"))
cat = Catalog.from_spec(spec)
grid = Discretizer.from_spec(spec)
nk = NodeKey.from_spec(spec, cat)


def box_on_floor(x):
    return [x, A / 2, 0.0]


def disk_free_left(box):          # disk to the box's -x side, clear of everything
    return [box[0] - A / 2 - R - 0.03, 0.06]


def disk_free_above(box):         # disk above the box, clear of it
    return [box[0] + 0.02, A + R + 0.04]


def state(q):
    return State(jnp.asarray(q, jnp.float32), jnp.zeros(5, jnp.float32))


def formation(q):
    return np.asarray(cat.band(config=jnp.asarray(q, jnp.float32)))


def names(f):
    return "{" + ", ".join(n for n, on in zip(cat.names, f) if on) + "}"


def node_key_same(q1, q2):
    """Real merge rule: make a node at q1 (as a search would), then key q2 against it."""
    f1, f2 = formation(q1), formation(q2)
    st = SearchState(nk, "page")
    st.add(state(q1), tuple(f1))
    k1 = st.key(state(q1), tuple(f1))
    k2 = st.key(state(q2), tuple(f2))
    return k1 == k2, k1, k2


examples = []
x0 = FLAT[0]
cases = [
    ("disk moved, box fixed",
     "The disk hovers in two different places; the box rests at the same pose.",
     box_on_floor(x0) + disk_free_left(box_on_floor(x0)),
     box_on_floor(x0) + disk_free_above(box_on_floor(x0))),
    ("box 0.2 mm apart, across a cell boundary",
     "Physically the same state; the 1 mm grid puts the two poses in different cells.",
     box_on_floor(-0.0999) + disk_free_left(box_on_floor(-0.0999)),
     box_on_floor(-0.1001) + disk_free_left(box_on_floor(-0.1001))),
    ("box 5 mm apart",
     "Within the merge radius (9 mm placeholder until the per-key gate).",
     box_on_floor(x0) + disk_free_left(box_on_floor(x0)),
     box_on_floor(x0 - 0.005) + disk_free_left(box_on_floor(x0 - 0.005))),
    ("box 30 cm apart",
     "Near the step vs far from it: one Xiao-Ji contact state, but only the disk can move "
     "the box, so the futures differ (the pivot needs the box at the step).",
     box_on_floor(x0) + disk_free_left(box_on_floor(x0)),
     box_on_floor(x0 - 0.30) + disk_free_left(box_on_floor(x0 - 0.30))),
]
for title, why, q1, q2 in cases:
    f1, f2 = formation(q1), formation(q2)
    g1, g2 = grid(state(q1)), grid(state(q2))
    same_form = bool(np.array_equal(f1, f2))
    same_nk, k1, k2 = node_key_same(q1, q2)
    row = dict(case=title, why=why, q1=[round(v, 4) for v in q1], q2=[round(v, 4) for v in q2],
               formation=names(f1), same_formation=same_form,
               grid_same=bool(same_form and g1 == g2),
               xiao_ji_same=same_form,     # flat on the floor: one connected region per formation
               node_key_same=bool(same_nk),
               box_distance_mm=round(1000 * nk.box_distance(nk.box_pose(state(q1)), nk.box_pose(state(q2))), 2),
               merge_radius_mm=round(1000 * nk.R_ID_PLACEHOLDER, 1))
    examples.append(row)
    print(f"{title}: formation {row['formation']} | grid same {row['grid_same']} | "
          f"Xiao-Ji same {row['xiao_ji_same']} | node key same {row['node_key_same']} "
          f"(box {row['box_distance_mm']} mm)")

(out / "node_identity.json").write_text(json.dumps(dict(
    examples=examples, eta_mm=1000 * spec.execution.eta, eta_theta=spec.execution.eta_theta,
    key_kind=nk.kind, merge_radius_placeholder_mm=1000 * nk.R_ID_PLACEHOLDER), indent=1))

# Figure: the scene (+x to the left, as in the paper) with the example poses.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.fill([-0.55, 0, 0, 0.12, 0.12, -0.55], [0, 0, H, H, -0.03, -0.03], color="#cfcfcf", lw=0)
ax.plot([-0.55, 0, 0, 0.12], [0, 0, H, H], color="#555", lw=1.2)


def box(x, **kw):
    ax.add_patch(Rectangle((x - A / 2, 0), A, A, fill=False, lw=1.6, **kw))


box(x0, ec="#1f6fb4")
box(x0 - 0.30, ec="#c0392b", ls="--")
for p, c in ((disk_free_left(box_on_floor(x0)), "#1f6fb4"), (disk_free_above(box_on_floor(x0)), "#1f6fb4")):
    ax.add_patch(Circle(p, R, fill=False, lw=1.4, ec=c))
lbl = (x0 - 0.10, 0.215)
for p in (disk_free_left(box_on_floor(x0)), disk_free_above(box_on_floor(x0))):
    ax.annotate("", xy=p, xytext=lbl, arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=6, shrinkB=11))
ax.text(*lbl, "disk here or here: same node", fontsize=8, ha="center", va="bottom")
ax.text(x0 - 0.30, A + 0.035, "box 30 cm farther from the step:\nsame Xiao–Ji contact state, different node",
        fontsize=8, color="#c0392b", ha="center", va="bottom")
ax.text(0.06, H + 0.012, "platform", fontsize=7, ha="center")
ax.text(-0.52, 0.008, "floor", fontsize=7)
ax.set_xlim(0.12, -0.55)            # +x to the left
ax.set_ylim(-0.03, 0.26)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig(out / "node_identity.svg")
print("wrote", out / "node_identity.json", out / "node_identity.svg")
