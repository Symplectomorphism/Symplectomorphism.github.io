"""Background tutorial: deciding support exactly (the planar hold test).

Run from c-gnn/cmg on branch phase1/planar (Python 3.14 env):

    JAX_PLATFORMS=cpu .venv314/bin/python <this file> <website media dir>

Builds the worked example of the tutorial: a square box on the floor with
mu = 0.2, held by both bottom corners (supported) or by one corner (not).
Runs cmg's cone_test on it, prints every number the page quotes, and writes
bg_cone_slice.svg: the physical picture (top) and the cone seen through the
slice f_z = 1 (bottom).
"""
import itertools
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from cmg.physics.planar_hold import cone_test

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MU = 0.2
NEED = np.array([0.0, 1.0, 0.0])                     # -(gravity wrench) / m g
INK, MUTED, FAINT = "#1f2328", "#656d76", "#d0d7de"
CA, CB = "#2a78d6", "#eb6834"                         # corner A, corner B


def generators(mu, corners):
    """Ray wrenches (f_x, f_z, tau / l) of a square box's bottom corners.

    Arms from the center of mass in units of l = a/2: A = (-1, -1), B = (+1, -1).
    Rays n +- mu t with n = (0, 1); tau = r_z f_x - r_x f_z (moment about +y).
    """
    arms = {"A": (-1.0, -1.0), "B": (1.0, -1.0)}
    out, names = [], []
    for c in corners:
        rx, rz = arms[c]
        for s, tag in ((1, "+"), (-1, "-")):
            fx, fz = s * mu, 1.0
            out.append((fx, fz, rz * fx - rx * fz))
            names.append(f"{c}{tag}")
    return np.array(out), names


def report(g):
    r = cone_test(jnp.asarray(g, jnp.float32), jnp.ones(len(g), dtype=bool), jnp.asarray(NEED, jnp.float32))
    return {k: float(v) for k, v in r.items()}


both, names2 = generators(MU, "AB")
one, names1 = generators(MU, "A")
r2, r1 = report(both), report(one)
print(f"mu = {MU}")
for n, g in zip(names2, both):
    print(f"  generator {n}: ({g[0]:+.1f}, {g[1]:.1f}, {g[2]:+.1f})")
print("two corners:", r2)
print("one corner :", r1)

# Witness: the smallest subset that reproduces NEED with nonnegative weights.
for k in (1, 2, 3):
    found = None
    for idx in itertools.combinations(range(4), k):
        lam, *_ = np.linalg.lstsq(both[list(idx)].T, NEED, rcond=None)
        if np.all(lam >= -1e-9) and np.linalg.norm(both[list(idx)].T @ lam - NEED) < 1e-9:
            found = idx, lam
            break
    if found:
        print("witness:", [names2[i] for i in found[0]], "weights", np.round(found[1], 3))
        break

# Facets of the two-corner cone and their values y . need.
print("facets (unit normal y, y . need):")
for i, j in itertools.combinations(range(4), 2):
    y = np.cross(both[i], both[j])
    y /= np.linalg.norm(y)
    for sgn in (1, -1):
        if np.all(sgn * both @ y >= -1e-12):
            print(f"  {names2[i]},{names2[j]}: y = {np.round(sgn * y, 3)}, value {sgn * y @ NEED:.4f}")

# Certificate for one corner: the plane through its two rays.
y = np.cross(one[0], one[1])
y /= np.linalg.norm(y)
y = y if np.all(one @ y >= -1e-12) and y @ NEED < 0 else -y
print(f"one-corner certificate y = {np.round(y, 3)}: y.g = {np.round(one @ y, 6)}, y.need = {y @ NEED:.4f}")
for mu in (0.2, 0.5, 1 / math.sqrt(2), 1.0):
    print(f"  flat-box margin at mu = {mu:.3f}: {report(generators(mu, 'AB')[0])['margin']:.4f}"
          f"  (mu/sqrt(1+mu^2) = {mu / math.hypot(1, mu):.4f}, 1/sqrt(3) = {1 / math.sqrt(3):.4f})")
for d in (0.19, 0.21):
    pushed = cone_test(jnp.asarray(both, jnp.float32), jnp.ones(4, dtype=bool), jnp.asarray([d, 1.0, 0.0], jnp.float32))
    print(f"  sideways push {d} m g: supported = {bool(pushed['feasible'])}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4), gridspec_kw={"height_ratios": [1, 1.15]})


def box_panel(ax, corners, title):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-1.6, 1.6], [-1, -1], color=INK, lw=1.2)
    for x in np.linspace(-1.55, 1.5, 22):
        ax.plot([x, x + 0.08], [-1, -1.08], color=MUTED, lw=0.6)
    ax.add_patch(Rectangle((-1, -1), 2, 2, fc="#f6f8fa", ec=INK, lw=1.3))
    ax.plot([0], [0], "o", color=INK, ms=4)
    ax.annotate("", xy=(0, -0.62), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.08, -0.38, "mg", fontsize=9, color=INK, style="italic")
    for c, x0, col in (("A", -1.0, CA), ("B", 1.0, CB)):
        on = c in corners
        colr = col if on else FAINT
        tips = [(x0 + s * 0.62 * MU, -1 + 0.62) for s in (1, -1)]
        cone = Polygon([(x0, -1), *tips], closed=True, fc=colr, ec="none", alpha=0.18 if on else 0.35)
        ax.add_patch(cone)
        for s in (1, -1):
            ax.annotate("", xy=(x0 + s * 0.62 * MU, -1 + 0.62), xytext=(x0, -1),
                        arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.3))
        ax.text(x0 - 0.13 if c == "A" else x0 + 0.05, -1.32, c if on else f"{c} (gone)", fontsize=9,
                color=colr if on else MUTED, ha="center" if on else "left")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.45, 1.6)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


box_panel(axes[0, 0], "AB", "(a) Flat box on two corners, μ = 0.2")
# The witness: half the weight on A+ and half on B-, both leaning inward.
for x0, s, col in ((-1.0, 1, CA), (1.0, -1, CB)):
    axes[0, 0].add_patch(FancyArrowPatch((x0, -1), (x0 + s * 0.5 * MU * 2.2, -1 + 0.5 * 2.2), arrowstyle="-|>",
                                         mutation_scale=12, color=col, lw=2.6))
axes[0, 0].text(0, 1.22, "witness: ½ mg along A₊ and ½ mg along B₋", fontsize=8, color=INK, ha="center")
box_panel(axes[0, 1], "A", "(b) The same box on corner A alone")
axes[0, 1].text(0, 1.15, "A's force always has a moment about the\ncenter of mass; gravity has none to cancel it",
                fontsize=8, color=INK, ha="center")


def slice_panel(ax, g, names, title):
    u, v = g[:, 0] / g[:, 1], g[:, 2] / g[:, 1]
    ax.axhline(0, color=FAINT, lw=0.8, zorder=0)
    ax.axvline(0, color=FAINT, lw=0.8, zorder=0)
    if len(g) == 4:
        order = [0, 1, 3, 2]                        # A+, A-, B-, B+ around the hull
        ax.add_patch(Polygon(np.c_[u[order], v[order]], closed=True, fc="#e8f0fb", ec=INK, lw=1.0, zorder=1))
        ax.plot([u[0], u[3]], [v[0], v[3]], color=INK, lw=2.2, zorder=2)
        ax.annotate("witness: need is the\nmidpoint of A₊ and B₋", (-0.1, -0.4), xytext=(-0.69, 0.25), fontsize=8,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.annotate("sliding facet\n(margin 0.196)", (0.2, -0.2), xytext=(0.3, 0.45), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.annotate("tipping facet\n(about B)", (0.0, -1.0), xytext=(-0.62, -1.38), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    else:
        ax.plot(u, v, color=INK, lw=1.0, zorder=1)
        xs = np.linspace(-0.7, 0.7, 2)
        ax.fill_between(xs, -1.6, 1 - xs, color="#f3f4f6", zorder=0)
        ax.plot(xs, 1 - xs, color=MUTED, lw=1.2, ls="--", zorder=1)
        ax.text(-0.66, -1.3, "y · x < 0 here: no mix of A's rays\nreaches this side", fontsize=8, color=MUTED)
        ax.text(-0.66, 0.35, "separator  y = (1, −1, 1)/√3:\n y·g = 0 on both rays,\n y·need = −0.577 < 0",
                fontsize=8, color=INK)
    for k, n in enumerate(names):
        col = CA if n[0] == "A" else CB
        ax.plot(u[k], v[k], "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
        sub = "₊" if n[1] == "+" else "₋"
        ax.text(u[k] + (0.05 if u[k] > 0 else -0.05), v[k], n[0] + sub, fontsize=9, color=INK,
                ha="left" if u[k] > 0 else "right", va="center")
    ax.plot(0, 0, "s", color=INK, ms=7, zorder=4)
    ax.text(-0.04, -0.12, "need", fontsize=8.5, color=INK, ha="right")
    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("f_x / f_z", fontsize=9, color=INK)
    ax.set_ylabel("τ / (f_z ℓ)", fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


slice_panel(axes[1, 0], both, names2, "(c) Slice f_z = 1: need inside → supported")
slice_panel(axes[1, 1], one, names1, "(d) Slice f_z = 1: need outside → not supported")
fig.tight_layout(h_pad=1.5)
fig.savefig(OUT / "bg_cone_slice.svg", bbox_inches="tight")
print("wrote bg_cone_slice.svg")
