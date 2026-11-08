"""Planar build, Decision 9 (Xiao and Ji's construction): the counts and the landing figure.

Run from c-gnn/cmg on branch phase1/planar (Python 3.14 env):

    JAX_PLATFORMS=cpu .venv314/bin/python <this file> <website media dir>

For the counts before Decision 9, run the same file against the code of commit
84827c0 (the last one before it), extracted with
    git archive 84827c0 cmg | tar -x -C <old dir>
and pass --before; then only the counts are printed:

    cd <old dir>/cmg && JAX_PLATFORMS=cpu PYTHONPATH=src <c-gnn>/cmg/.venv314/bin/python <this file> --before

Prints: the transfer relation (pairs, classified as Xiao-Ji containment, round a
corner, or both features changing), the valid markings, the compound edges, and
the ties on the demo and on 4,000 jittered demo poses (the stress test's sample).
Writes xj_landing.svg (current code only).
"""
import itertools
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.scene.spec import SceneSpec

BEFORE = "--before" in sys.argv
OUT = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "."))

spec = SceneSpec.from_yaml("src/cmg/scene/families/step2d.yaml")
cat = Catalog.from_spec(spec)
A, R, H = 0.125, 0.015625, 0.125 / 4
inv = {i: n for n, i in cat.named.items()}
label = [inv.get(i, cat.names[i]) for i in range(len(cat))]
VERTEX = {"xpzp", "xpzn", "xnzp", "xnzn", "end", "S/P"}
print("code:", "before Decision 9 (84827c0)" if BEFORE else "current (Decision 9)")
print(f"templates: {len(cat)}")


# ------------------------------------------------------------ the transfer relation

def kind(f):
    return "vertex" if f.feature_type in VERTEX else "edge"


def classify(t1, t2):
    diff = [(a, b) for a, b in ((t1.feature1, t2.feature1), (t1.feature2, t2.feature2))
            if a.feature_type != b.feature_type]
    if len(diff) == 2:
        return "both features change"
    a, b = diff[0]
    if {kind(a), kind(b)} == {"vertex", "edge"}:
        return "containment (Xiao-Ji cases 2-3)"
    return "edge to edge round a corner"


T = list(cat)
pairs = [(a, b) for a, b in itertools.combinations(T, 2) if cat.transfer_adjacent(a, b)]
classes = {}
for a, b in pairs:
    classes.setdefault(classify(a, b), []).append((a, b))
print(f"transfer relation: {len(pairs)} unordered pairs")
for k, v in sorted(classes.items()):
    print(f"  {len(v):3d}  {k}")
named_set = set(cat.named.values())
step = sorted(f"{label[a.template_id]} <-> {label[b.template_id]}" for a, b in pairs
              if a.template_id in named_set and b.template_id in named_set)
print("  between named contacts:", ", ".join(step))
mx, mz = cat.named["m_x"], cat.named["m_z"]
print("  m_x <-> m_z a transfer:", any({a.template_id, b.template_id} == {mx, mz} for a, b in pairs))


# ------------------------------------------------------------ markings and compound edges

from cmg.marking.edits import EditSet  # noqa: E402

edits = EditSet(cat, spec)
group = np.asarray(cat.slot if hasattr(cat, "slot") else cat._pair)
G = int(group.max()) + 1
groups = [[None] + list(np.flatnonzero(group == p)) for p in range(G)]
marks = []
for choice in itertools.product(*groups):
    f = np.zeros(len(cat), dtype=bool)
    f[[c for c in choice if c is not None]] = True
    marks.append(f)
marks = np.array(marks)
valid = np.asarray(jax.jit(jax.vmap(edits.valid))(jnp.asarray(marks)))
V = marks[valid]
edges = np.asarray(jax.jit(jax.vmap(edits.edges))(jnp.asarray(V)))
types = np.asarray(edits.types)
compound_type = int(types.max())            # the compound / pass-through kind is the last one
multi = [i for i in range(len(edits)) if types[i] == compound_type and edits.labels[i].startswith(("pass", "compound"))]
unit = "convex element pair" if hasattr(cat, "slot") else "body pair"
print(f"markings with at most one contact per {unit}: {len(marks)}, valid: {len(V)}")
print(f"compound edges: {sorted(edits.labels[i] for i in multi)}; "
      f"instances over the valid markings: {int(edges[:, multi].sum())}")


# ------------------------------------------------------------ ties

from cmg.storage import resolve_path  # noqa: E402

demo = np.load(resolve_path("src/cmg/scene/families/step_example_trajectory.npz"))["qpos"].astype(np.float32)
ties = jax.jit(jax.vmap(lambda c: cat.tie_breaks(config=c)))
print(f"ties on the demo: {int((np.asarray(ties(jnp.asarray(demo))) > 0).sum())} of {len(demo)} frames")
rng = np.random.default_rng(0)                     # the stress test's sample (test_tie_stress)
q = demo[rng.integers(0, len(demo), 4000)] + rng.normal(
    0, [0.003, 0.003, 0.03, 0.003, 0.003], (4000, 5)).astype(np.float32)
tq = np.asarray(ties(jnp.asarray(q)))
print(f"ties on 4,000 jittered demo poses: {int((tq > 0).sum())}")
if not BEFORE:
    band = jax.jit(jax.vmap(lambda c: cat.band(config=c)))
    inband = jax.jit(jax.vmap(lambda c: cat.in_band(config=c)))
    gaps = jax.jit(jax.vmap(lambda c: cat.gap(config=c)))
    tied = q[tq > 0]
    on, ib, ph = (np.asarray(f(jnp.asarray(tied))) for f in (band, inband, gaps))
    sv, t = cat.named["s_v"], cat.named["t"]
    lost = ib[:, sv] & ~on[:, sv] & on[:, t]
    deep = (np.where(ib, ph, np.inf).min(1) < -float(cat.epsilon_on))
    print(f"  tied poses with a contact deeper than eps_on: {int(deep.sum())} of {len(tied)}")
    print(f"  tied poses where s_v is in band but loses to t: {int(lost.sum())}, "
          f"median deepest penetration there {1e3 * float(np.median(-np.where(ib, ph, np.inf).min(1)[lost])):.1f} mm")
if BEFORE:
    sys.exit()


# ------------------------------------------------------------ the landing figure

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

INK, MUTED, BOXFILL = "#1f2328", "#656d76", "#f3e2c4"
SLOT_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
slot = np.asarray(cat.slot)
kset = jax.jit(lambda c: cat.contact_set(config=c)[0])
on1 = jax.jit(lambda c: cat.band(config=c))
FAR = [-0.2, 0.0625]


def X(x):
    return -x


def corners(x, z, th):
    c, s = math.cos(th), math.sin(th)
    return [(x + u * c + w * s, z - u * s + w * c) for u, w in
            [(A / 2, A / 2), (A / 2, -A / 2), (-A / 2, -A / 2), (-A / 2, A / 2)]]


def lifted(theta, along):
    c, s = math.cos(theta), math.sin(theta)
    vx, vz = -along * s, H - along * c
    return [vx - A / 2 * (c - s), vz + A / 2 * (s + c), theta]


PANELS = [
    ("resting on the corner (60°)", lifted(math.radians(60), 0.03) + FAR),
    ("landed, overhanging the corner", [0.02, H + A / 2, math.pi / 2] + FAR),
    ("landed, clear of the corner", [A / 2 + 0.01, H + A / 2, math.pi / 2] + FAR),
]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
for ax, (title, q) in zip(axes, PANELS):
    ax.plot([X(-0.25), X(0)], [0, 0], color=INK, lw=1.6)
    ax.plot([X(0), X(0)], [0, H], color=INK, lw=1.6)
    ax.plot([X(0), X(0.2)], [H, H], color=INK, lw=1.6)
    ax.fill([X(-0.25), X(0), X(0), X(0.2), X(0.2), X(-0.25)], [0, 0, H, H, -0.02, -0.02],
            color="#d9dde1", lw=0, zorder=0)
    ax.plot([X(0)], [H], "o", ms=5, color=INK, zorder=6)
    ax.add_patch(Polygon([(X(px), pz) for px, pz in corners(*q[:3])], closed=True,
                         fc=BOXFILL, ec="#7a4f0e", lw=1.2, zorder=2))
    qj = jnp.asarray(q, dtype=jnp.float32)
    on = np.asarray(on1(qj))
    ks = np.asarray(kset(qj))
    names = []
    for i in np.flatnonzero(on):
        (x0, z0), (x1, z1) = ks[i]
        col = SLOT_COLOR[slot[i]]
        if math.hypot(x1 - x0, z1 - z0) < 1e-4:
            ax.plot([X(x0)], [z0], "o", ms=7, color=col, mec="white", mew=1.0, zorder=7)
        else:
            ax.plot([X(x0), X(x1)], [z0, z1], color=col, lw=4.5, solid_capstyle="round", zorder=5)
        ax.annotate(label[i], (X((x0 + x1) / 2), (z0 + z1) / 2), xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.8, alpha=0.9))
        names.append(label[i])
    ax.annotate("v_S", (X(0), H), xytext=(6, -12), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_title(f"{title}\n{{{', '.join(sorted(names))}}}", fontsize=9, color=INK)
    print(f"landing figure: {title}: {{{', '.join(sorted(names))}}}")
    ax.set_xlim(X(0.16), X(-0.12))
    ax.set_ylim(-0.012, 0.19)
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "xj_landing.svg", bbox_inches="tight")
print("wrote xj_landing.svg")
