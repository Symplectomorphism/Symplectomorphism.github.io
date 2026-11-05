"""Planar build, step 1: export the contact layer's results for the website.

Run from c-gnn/cmg on branch phase1/planar (Python 3.14 env):

    JAX_PLATFORMS=cpu .venv314/bin/python <this file> <website media dir>

Writes into the media directory
  * step1_scene.svg   the planar step scene and its features,
  * step1_chain.svg   the paper's chain sigma_0 ... sigma_8 at constructed poses,
                      each with its principal-contact on-set and contact sets,
  * step1_tilt.svg    the box's tilt through the pivot of the demo, with the
                      edge-edge tilt threshold theta_e,
  * step1_demo.json   per-frame data for the interactive demo viewer,
and prints the numbers the page quotes.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from cmg.catalog.templates import Catalog
from cmg.scene.spec import SceneSpec

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

spec = SceneSpec.from_yaml("src/cmg/scene/families/step2d.yaml")
cat = Catalog.from_spec(spec)
A, R, H = 0.125, 0.015625, 0.125 / 4
EPS_ON = float(cat.epsilon_on)
L_MIN = float(cat.min_edge_length)
THETA_E = math.degrees(cat.theta_e)

inv = {i: n for n, i in cat.named.items()}
label = [inv.get(i, cat.names[i]) for i in range(len(cat))]
pair = np.asarray(cat._pair)
PAIRS = ["M-B", "B-L", "B-S", "B-P"]          # pair ids 0..3 in catalog order
# Categorical slots 1, 2, 3, 7 of the reference palette; every contact is also labeled.
PAIR_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, MUTED, BOXFILL = "#1f2328", "#656d76", "#f3e2c4"

band = jax.jit(jax.vmap(lambda q: cat.band(config=q)))
inband = jax.jit(jax.vmap(lambda q: cat.in_band(config=q)))
ties = jax.jit(jax.vmap(lambda q: cat.tie_breaks(config=q)))
kset = jax.jit(jax.vmap(lambda q: cat.contact_set(config=q)[0]))


# ------------------------------------------------------------ poses (as in the tests)

def pivot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [-H * math.tan(theta) - A / 2 * (c - s), A / 2 * (s + c), theta]


def disk_on_xn(box, mid=0.0):
    x, z, theta = box
    c, s = math.cos(theta), math.sin(theta)
    u, w = -A / 2 - R, mid
    return [x + u * c + w * s, z - u * s + w * c]


def lifted(theta, along):
    c, s = math.cos(theta), math.sin(theta)
    vx, vz = -along * s, H - along * c
    return [vx - A / 2 * (c - s), vz + A / 2 * (s + c), theta]


FAR = [-0.2, 0.0625]
FLAT = [-0.1, A / 2, 0.0]
FLUSH = [-A / 2, A / 2, 0.0]
LANDED = [A / 4, H + A / 2, math.pi / 2]
SLID = [A / 2 + 0.01, H + A / 2, math.pi / 2]
CHAIN = [
    ("σ₀  rest", FLAT + FAR),
    ("σ₁  disk on −x_B", FLAT + disk_on_xn(FLAT)),
    ("σ₂  pushed flush", FLUSH + disk_on_xn(FLUSH)),
    ("σ₃  pivot (30°)", pivot(math.radians(30)) + disk_on_xn(pivot(math.radians(30)))),
    ("σ₄  lifted (60°)", lifted(math.radians(60), 0.03) + disk_on_xn(lifted(math.radians(60), 0.03))),
    ("σ₅  landed, disk on top", LANDED + [A / 4, H + A + R]),
    ("σ₆  disk away", LANDED + FAR),
    ("σ₇  disk behind", LANDED + [-A / 4 - R, H + A / 2]),
    ("σ₈  slid clear", SLID + [SLID[0] - A / 2 - R, H + A / 2]),
]


# ------------------------------------------------------------ drawing (+x to the left)

def box_corners(x, z, th):
    c, s = math.cos(th), math.sin(th)
    loc = [(A / 2, A / 2), (A / 2, -A / 2), (-A / 2, -A / 2), (-A / 2, A / 2)]
    return [(x + u * c + w * s, z - u * s + w * c) for u, w in loc]


def X(x):
    return -x


def draw_env(ax, xmin=-0.32, xmax=0.2):
    ax.plot([X(xmin), X(0)], [0, 0], color=INK, lw=1.6, solid_capstyle="butt")
    ax.plot([X(0), X(0)], [0, H], color=INK, lw=1.6)
    ax.plot([X(0), X(xmax)], [H, H], color=INK, lw=1.6, solid_capstyle="butt")
    ax.fill([X(xmin), X(0), X(0), X(xmax), X(xmax), X(xmin)], [0, 0, H, H, -0.02, -0.02],
            color="#d9dde1", lw=0, zorder=0)


def draw_bodies(ax, q):
    ax.add_patch(Polygon([(X(px), pz) for px, pz in box_corners(*q[:3])], closed=True,
                         fc=BOXFILL, ec="#7a4f0e", lw=1.2, zorder=2))
    ax.add_patch(Circle((X(q[3]), q[4]), R, fc="#c9d8fb", ec="#14307a", lw=1.2, zorder=3))


def draw_contacts(ax, q, fs=7.5):
    qj = jnp.asarray([q], dtype=jnp.float32)
    on = np.asarray(band(qj))[0]
    ks = np.asarray(kset(qj))[0]
    for i in np.flatnonzero(on):
        (x0, z0), (x1, z1) = ks[i]
        col = PAIR_COLOR[pair[i]]
        if math.hypot(x1 - x0, z1 - z0) < 1e-4:
            ax.plot([X(x0)], [z0], "o", ms=6, color=col, mec="white", mew=1.0, zorder=5)
        else:
            ax.plot([X(x0), X(x1)], [z0, z1], color=col, lw=4.5, solid_capstyle="round", zorder=4)
        mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
        ax.annotate(label[i], (X(mx), mz), xytext=(4, 4), textcoords="offset points",
                    fontsize=fs, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.8, alpha=0.9))
    return [label[i] for i in np.flatnonzero(on)]


def frame(ax, xlim=(-0.24, 0.14), zlim=(-0.012, 0.2)):
    ax.set_xlim(*sorted((X(xlim[0]), X(xlim[1]))))
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------ figure 1: the scene

fig, ax = plt.subplots(figsize=(8.2, 3.6))
draw_env(ax, -0.3, 0.2)
q = FLAT + [-0.2, 0.0625]
draw_bodies(ax, q)
x, z = q[0], q[1]
for name, (u, w) in {"xp": (A / 2, 0), "xn": (-A / 2, 0), "zp": (0, A / 2), "zn": (0, -A / 2)}.items():
    ax.annotate(f"B:{name}", (X(x + u), z + w), xytext=(X(x + 0.62 * u), z + 0.62 * w),
                fontsize=8.5, ha="center", va="center", color=INK)
for name, (u, w) in {"xpzp": (1, 1), "xpzn": (1, -1), "xnzp": (-1, 1), "xnzn": (-1, -1)}.items():
    px, pz = x + u * A / 2, z + w * A / 2
    ax.plot([X(px)], [pz], "o", ms=4, color="#7a4f0e", zorder=4)
    ax.annotate(f"B:{name}", (X(px), pz), xytext=(-u * 6, w * 7), textcoords="offset points",
                fontsize=7.5, ha="center", va="center", color=MUTED)
ax.annotate("M (disk, r = a/8)", (X(-0.2), 0.0625 + R), xytext=(0, 10), textcoords="offset points",
            fontsize=8.5, ha="center", color=INK)
ax.annotate("L:edge (floor)", (X(-0.27), 0), xytext=(0, -12), textcoords="offset points",
            fontsize=8.5, ha="center", color=INK)
ax.annotate("S:edge (riser)", (X(0), H / 2), xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, ha="left", va="center", color=INK)
ax.plot([X(0)], [H], "o", ms=5.5, color=INK, zorder=4)
ax.annotate("S:end = v_S (owned by the riser)", (X(0), H), xytext=(-10, 30), textcoords="offset points",
            fontsize=8.5, ha="right", color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.plot([X(0)], [H], "o", ms=10, mfc="none", mec=MUTED, lw=0.8, zorder=3)
ax.annotate("P:edge (platform, open at its start)", (X(0.12), H), xytext=(0, 8), textcoords="offset points",
            fontsize=8.5, ha="center", color=INK)
ax.annotate("(0, 0): concave junction, not a feature", (X(0), 0), xytext=(-6, -14),
            textcoords="offset points", fontsize=7.5, color=MUTED, ha="right")
ax.annotate("", (X(-0.29), 0.17), xytext=(X(-0.24), 0.17),
            arrowprops=dict(arrowstyle="<-", color=MUTED, lw=1))
ax.text(X(-0.265), 0.176, "+x", fontsize=8, color=MUTED, ha="center")
frame(ax, (-0.3, 0.2), (-0.03, 0.19))
fig.tight_layout()
fig.savefig(OUT / "step1_scene.svg", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------ figure 2: the chain

fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.6))
chain_out = []
for ax, (title, q) in zip(axes.flat, CHAIN):
    draw_env(ax)
    draw_bodies(ax, q)
    on = draw_contacts(ax, q)
    order = {"m_x": 0, "m_z": 1, "g": 2, "g_v": 3, "s": 4, "s_v": 5, "t": 6}
    on = sorted(on, key=lambda n: order.get(n, 9))
    chain_out.append((title, on))
    ax.set_title(f"{title}\n{{{', '.join(on)}}}", fontsize=9, color=INK)
    frame(ax)
fig.tight_layout()
fig.savefig(OUT / "step1_chain.svg", bbox_inches="tight")
plt.close(fig)
print("chain:")
for t, on in chain_out:
    print(f"  {t:28s} {{{', '.join(on)}}}")


# ------------------------------------------------------------ the demo

qpos = np.load("src/cmg/scene/families/step_example_trajectory.npz")["qpos"].astype(np.float32)
Q = jnp.asarray(qpos)
on = np.asarray(band(Q))
ib = np.asarray(inband(Q))
tb = np.asarray(ties(Q))
ks = np.asarray(kset(Q))
th = np.degrees(qpos[:, 2])
tilt = np.abs(((th + 45) % 90) - 45)

per_pair_on = np.stack([on[:, pair == p].sum(1) for p in range(4)], 1)
per_pair_ib = np.stack([ib[:, pair == p].sum(1) for p in range(4)], 1)
print(f"demo: {len(qpos)} frames; max on-bits per pair {per_pair_on.max()}; "
      f"max in-band bits per pair {per_pair_ib.max()}; tie frames {int((tb > 0).sum())}")

seq = []
for k in range(len(on)):
    s = tuple(label[i] for i in np.flatnonzero(on[k]))
    if not seq or seq[-1]["on"] != list(s):
        seq.append({"start": k, "n": 1, "on": list(s)})
    else:
        seq[-1]["n"] += 1
for s in seq:
    print(f"  {s['start']:5d} +{s['n']:4d}  {{{', '.join(s['on'])}}}")

rest = tilt[:600][on[:600, cat.named["g"]]]
print(f"resting tilt (frames 0-599, g on): max {rest.max():.2f} deg")
sw = np.flatnonzero(on[:-1, cat.named["g"]] & on[1:, cat.named["g_v"]])
print(f"g -> g_v at frame {sw.tolist()}, tilt {[round(float(tilt[i]), 2) for i in sw]} -> "
      f"{[round(float(tilt[i + 1]), 2) for i in sw]}; theta_e = {THETA_E:.2f} deg")
first_2deg = int(np.flatnonzero((np.arange(len(tilt)) > 600) & (tilt > 2.0))[0])
print(f"first frame after 600 with tilt > 2 deg: {first_2deg} ({tilt[first_2deg]:.2f} deg)")

frames = []
for k in range(len(qpos)):
    idx = np.flatnonzero(on[k])
    frames.append({
        "q": [round(float(v), 5) for v in qpos[k]],
        "on": idx.tolist(),
        "k": [[round(float(v), 5) for v in ks[k, i].ravel()] for i in idx],
        "ib": per_pair_ib[k].tolist(),
        "tie": int(tb[k]),
        "tilt": round(float(tilt[k]), 2),
    })
data = {
    "a": A, "r": R, "h": H, "eps_on": EPS_ON, "l_min": L_MIN, "theta_e": round(THETA_E, 2),
    "theta_bound": round(math.degrees(math.asin(EPS_ON / L_MIN)), 2),
    "dt_note": "frame index of step_example_trajectory.npz",
    "pairs": PAIRS, "pair_color": PAIR_COLOR,
    "templates": [{"name": label[i], "pair": int(pair[i])} for i in range(len(cat))],
    "segments": seq,
    "frames": frames,
}
(OUT / "step1_demo.json").write_text(json.dumps(data, separators=(",", ":")))
print(f"wrote step1_demo.json ({(OUT / 'step1_demo.json').stat().st_size / 1e3:.0f} kB)")


# ------------------------------------------------------------ figure 3: tilt through the pivot

k0, k1 = 604, 720
ks_ = np.arange(k0, k1)
fig, ax = plt.subplots(figsize=(8.2, 3.3))
for j, s in enumerate(seq):
    a, b = s["start"], s["start"] + s["n"]
    if b <= k0 or a >= k1:
        continue
    lo, hi = max(a, k0) - 0.5, min(b, k1) - 0.5
    ax.axvspan(lo, hi, color="#e6eaee" if j % 2 else "#f6f7f8", lw=0, zorder=0)
    text = "{" + ", ".join(s["on"]) + "}"
    if hi - lo >= 14:
        ax.text((lo + hi) / 2, 11.4, text, fontsize=7.5, color=MUTED, ha="center", va="top")
    elif hi - lo >= 2:
        ax.text((lo + hi) / 2, 11.4, text, fontsize=6.8, color=MUTED, ha="center", va="top", rotation=90)
ax.plot(ks_, tilt[k0:k1], color="#2a78d6", lw=2, marker="o", ms=3.5, zorder=3)
BOUND = math.degrees(math.asin(EPS_ON / L_MIN))
ax.axhspan(1.5, 2.5, color=MUTED, alpha=0.10, lw=0, zorder=1)
ax.axhline(THETA_E, color=INK, lw=1, ls="--", zorder=2)
ax.text(k0 + 20, THETA_E + 0.65, f"θ_e = {THETA_E:.0f}° (adopted); grey band 1.5–2.5° = proposed hysteresis",
        fontsize=7.5, color=INK)
ax.axhline(BOUND, color=MUTED, lw=1, ls=":", zorder=2)
ax.text(k0 + 0.3, BOUND + 0.8, f"{BOUND:.2f}° (old default, the bound)", fontsize=8, color=MUTED)
ax.axhspan(0, rest.max(), color="#2a78d6", alpha=0.12, lw=0, zorder=1)
ax.text(k0 + 0.3, -1.25, f"resting jitter, frames 0–599: ≤ {rest.max():.2f}°", fontsize=7.5,
        color=MUTED)
ax.set_xlim(k0 - 0.5, k1 - 0.5)
ax.set_ylim(-1.6, 11.6)
peak = int(np.argmax(tilt[k0:k1])) + k0
ax.annotate(f"pivot: peaks at {tilt[peak]:.0f}° (frame {peak})", (peak, 11.6), xytext=(peak - 20, 5.5), ha="right",
            fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax.set_xlabel("demo frame", fontsize=9, color=INK)
ax.set_ylabel("box tilt (deg)", fontsize=9, color=INK)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8, colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "step1_tilt.svg", bbox_inches="tight")
plt.close(fig)
print("wrote step1_scene.svg, step1_chain.svg, step1_tilt.svg")
