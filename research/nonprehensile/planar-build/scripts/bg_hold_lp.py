"""Background tutorial: the hold test as a linear program.

Run with any Python that has numpy, scipy and matplotlib (the c-gnn env works):

    python <this file> <website media dir>

Solves the tutorial's LPs with SciPy's HiGHS (dual simplex, float64) on the
running example of bg-exact-hold-test (square box, mu = 0.2, wrenches in
units of m g and m g l) and prints every number the page quotes. Writes
bg_lp_exits.svg: the exit distance along each of the six axis directions.
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MU, D = 0.2, 3
NEED = np.array([0.0, 1.0, 0.0])
CAP = 1e3                                             # t <= CAP keeps every LP bounded
INK, MUTED, FAINT = "#1f2328", "#656d76", "#d0d7de"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def point_rays(point, normal, mu):
    """Planar friction cone at `point` (arm from the center of mass, units of l):
    two ray wrenches (f_x, f_z, tau / l), tau = r_z f_x - r_x f_z."""
    n = np.asarray(normal, float)
    t = np.array([-n[1], n[0]])
    out = []
    for s in (1, -1):
        f = n + s * mu * t
        out.append((f[0], f[1], point[1] * f[0] - point[0] * f[1]))
    return out


A = point_rays((-1, -1), (0, 1), MU)                  # corner A (left)
B = point_rays((1, -1), (0, 1), MU)                   # corner B (right)
DISK = point_rays((-1, 0), (1, 0), MU)                # disk on the left face, at center height


def lp1(W, need=NEED, bounds_rows=None, f_max=None):
    """Elastic feasibility LP: min 1's+ + 1's-  s.t.  W lam + s+ - s- = need, all >= 0."""
    k = W.shape[1]
    c = np.r_[np.zeros(k), np.ones(2 * D)]
    A_eq = np.c_[W, np.eye(D), -np.eye(D)]
    A_ub = None if bounds_rows is None else np.c_[bounds_rows, np.zeros((len(bounds_rows), 2 * D))]
    r = linprog(c, A_ub=A_ub, b_ub=f_max, A_eq=A_eq, b_eq=need, bounds=[(0, None)] * (k + 2 * D),
                method="highs-ds")
    return r.fun, r.x[:k], r.eqlin.marginals


def lp2(W, u, need=NEED, bounds_rows=None, f_max=None):
    """Exit distance along u: max t  s.t.  W lam = need + t u, lam >= 0, 0 <= t <= CAP."""
    k = W.shape[1]
    c = np.r_[np.zeros(k), -1.0]
    A_eq = np.c_[W, -u]
    A_ub = None if bounds_rows is None else np.c_[bounds_rows, np.zeros(len(bounds_rows))]
    r = linprog(c, A_ub=A_ub, b_ub=f_max, A_eq=A_eq, b_eq=need, bounds=[(0, None)] * k + [(0, CAP)],
                method="highs-ds")
    return -r.fun if r.status == 0 else -np.inf


DIRS = [("+f_x", np.array([1.0, 0, 0])), ("−f_x", np.array([-1.0, 0, 0])),
        ("+f_z", np.array([0, 1.0, 0])), ("−f_z", np.array([0, -1.0, 0])),
        ("+τ", np.array([0, 0, 1.0])), ("−τ", np.array([0, 0, -1.0]))]

# 1. Flat box on both corners.
W = np.array(A + B).T
delta, lam, y = lp1(W)
print(f"[1] flat box: LP-1 optimum {delta:.3g}; witness weights {np.round(lam, 4)} "
      f"(nonzero: {np.count_nonzero(lam > 1e-9)})")

# 2. Corner A alone: certificate from the dual.
WA = np.array(A).T
delta, lam, y = lp1(WA)
z = -y
print(f"[2] corner A alone: LP-1 optimum (l1 distance) {delta:.4f}; dual y = {np.round(y, 4)}")
print(f"    certificate z = -y: z.w = {np.round(WA.T @ z, 4)} (>= 0), z.need = {z @ NEED:.4f} (< 0)")

# 3. Six exit distances for the flat box: the l1 margin.
exits = {name: lp2(W, u) for name, u in DIRS}
print("[3] flat box exit distances:", {k: round(v, 4) for k, v in exits.items()},
      " l1 margin =", round(min(exits.values()), 4))
print(f"    compare the Euclidean margin mu/sqrt(1+mu^2) = {MU / math.hypot(1, MU):.4f}")

# 4. Flat box plus the disk at center height, without and with a force limit.
WD = np.array(A + B + DISK).T
exits_free = {name: lp2(WD, u) for name, u in DIRS}
rows = np.zeros((1, WD.shape[1]))
rows[0, 4:6] = 1.0                                    # normal force of the disk = sum of its two ray weights
F_MAX = 0.5
exits_lim = {name: lp2(WD, u, bounds_rows=rows, f_max=[F_MAX]) for name, u in DIRS}
print("[4] with the disk, unlimited:", {k: (round(v, 4) if v < CAP else "cap") for k, v in exits_free.items()})
print(f"    with the disk, normal force <= {F_MAX} m g:", {k: round(v, 4) for k, v in exits_lim.items()})

# 5. Consistency: LP-1 distance equals the l1 distance by brute force on a grid
#    (for the corner-A cone: min over lam >= 0 of |W lam - need|_1).
g = np.linspace(0, 1.5, 601)
best = min(np.abs(WA @ np.array([a, b]) - NEED).sum() for a in g for b in g)
print(f"[5] corner A: grid search l1 distance {best:.4f} (LP-1 gave {lp1(WA)[0]:.4f})")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
names = [n for n, _ in DIRS]
SHOW = 2.0
for ax, data, title in ((axes[0], exits, "(a) Flat box on two corners"),
                        (axes[1], exits_lim, f"(b) Plus the disk, normal force ≤ {F_MAX} mg")):
    vals = np.array([data[n] for n in names])
    m = vals.min()
    for i, (n, v) in enumerate(zip(names, vals)):
        shown = min(v, SHOW)
        col = BLUE if math.isclose(v, m, rel_tol=1e-6) else FAINT
        ax.barh(i, shown, height=0.62, color=col)
        label = "∞ (pushing down only loads the corners)" if v >= CAP else f"{v:.3g}"
        ax.text(shown + 0.04 if v < SHOW else SHOW - 0.04, i, label, va="center", fontsize=8,
                color=INK, ha="left" if v < SHOW else "right")
    ax.axvline(m, color=INK, lw=1.0, ls="--")
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlim(0, SHOW)
    ax.set_xlabel("exit distance t (units of mg and mg·ℓ)", fontsize=8.5, color=INK)
    ax.set_title(f"{title}: margin {m:.3g}", fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("extra wrench the contacts must supply", fontsize=8.5, color=INK)
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "bg_lp_exits.svg", bbox_inches="tight")
print("wrote bg_lp_exits.svg")
