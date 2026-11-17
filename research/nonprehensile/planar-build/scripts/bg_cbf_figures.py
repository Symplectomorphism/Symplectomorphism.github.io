#!/usr/bin/env python3
"""Figure for the Background page on control barrier functions.

A small, checkable example, not a record. No arrows are drawn. Run:

    python bg_cbf_figures.py OUTDIR

Writes into OUTDIR:
  bg_cbf_wall.svg   a block pushed toward a wall (double integrator, x'' = u): the phase plane with the wall, the
                    exponential-CBF set and the set from which bounded braking can still stop, and three runs
                    (unfiltered; filtered; filtered with |u| <= a_max from too fast a start), plus the input over time
  bg_cbf_proof_p1.svg ... bg_cbf_proof_p5.svg
                    one sketch of the margin h(x(t)) against t per proposition of the proofs section, the zero line
                    marked: P1 never decreases; P2 decreases toward zero but stays above the comparison solution;
                    P3 steps down at least geometrically; P4 the counterexample h = -x^2 leaving at once; P5 dips below
                    zero but never below -c w/gamma
  bg_cbf_lyap_vs_barrier.svg
                    left: nested Lyapunov level sets, the flow inward on every one; right: one barrier set, the flow
                    free inside, tangent or inward on the boundary, inward outside (horizontal/vertical arrows only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#1a4f9c"
RED = "#c62828"
GREEN = "#2e7d32"
GREY = "#666666"
ORANGE = "#b26a00"

X_MAX = 0.10  # wall position (m)
LAM = 5.0  # exponential-CBF gain, both poles at -LAM (1/s)
A_MAX = 1.0  # braking limit (m/s^2)
U_NOM = 0.5  # nominal push toward the wall (m/s^2)


def run(v0: float, filt: bool, sat: bool, T: float = 1.2, dt: float = 1e-4):
    """Forward-Euler run of x'' = u from x = 0. With filt, u = min(u_nom, lam^2 h - 2 lam v) (the CBF-QP's closed
    form for this one constraint); with sat, u is clipped to -A_MAX when the QP has no admissible solution."""
    n = int(T / dt)
    t = np.arange(n) * dt
    x = np.empty(n)
    v = np.empty(n)
    u = np.empty(n)
    xi, vi, infeasible = 0.0, v0, False
    for k in range(n):
        h = X_MAX - xi
        uk = U_NOM
        if filt:
            uk = min(U_NOM, LAM * LAM * h - 2 * LAM * vi)
            if sat and uk < -A_MAX:
                infeasible = True
                uk = -A_MAX
        x[k], v[k], u[k] = xi, vi, uk
        xi += vi * dt
        vi += uk * dt
    return t, x, v, u, infeasible


def wall(out: Path) -> None:
    runs = {
        "A": run(0.10, filt=False, sat=False),
        "B": run(0.10, filt=True, sat=True),
        "C": run(0.48, filt=True, sat=True),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})

    # phase plane: position (mm) against speed toward the wall (mm/s)
    xs = np.linspace(0, X_MAX, 200)
    a1.fill_between(1e3 * xs, 0, 1e3 * LAM * (X_MAX - xs), color=BLUE, alpha=0.08, lw=0)
    a1.plot(1e3 * xs, 1e3 * LAM * (X_MAX - xs), color=BLUE, ls="--", lw=1,
            label=r"$h_1=0$: $v=\lambda(x_{\max}-x)$")
    a1.plot(1e3 * xs, 1e3 * np.sqrt(2 * A_MAX * (X_MAX - xs)), color=GREEN, ls=":", lw=1.4,
            label=r"$v=\sqrt{2a_{\max}(x_{\max}-x)}$")
    a1.axvline(1e3 * X_MAX, color="black", lw=2)
    a1.text(1e3 * X_MAX + 2, 545, "wall", fontsize=7, ha="left", va="top")
    styles = {"A": (GREY, "A: no filter"), "B": (BLUE, "B: CBF filter"), "C": (RED, r"C: filter, $|u|\leq a_{\max}$")}
    for key, (t, x, v, u, _) in runs.items():
        col, lab = styles[key]
        hit = int(np.argmax(x > X_MAX)) if np.any(x > X_MAX) else len(x)
        a1.plot(1e3 * x[:hit], 1e3 * v[:hit], color=col, lw=1.6, label=lab)
        a1.plot(0, 1e3 * v[0], "o", color=col, ms=3.5)
        if hit < len(x):  # the block reaches the wall: mark the impact
            a1.plot(1e3 * X_MAX, 1e3 * v[hit], "x", color=col, ms=6, mew=1.8)
    a1.set_xlim(0, 125)
    a1.set_ylim(0, 560)
    a1.set_xlabel("position $x$ (mm)", fontsize=8)
    a1.set_ylabel("speed toward the wall $v$ (mm/s)", fontsize=8)

    # the input over time
    for key in ("B", "C"):
        t, x, v, u, _ = runs[key]
        col, lab = styles[key]
        stop = np.argmax(x > X_MAX) if np.any(x > X_MAX) else len(t)
        a2.plot(1e3 * t[:stop], u[:stop], color=col, lw=2.2 if key == "C" else 1.6, zorder=3)
    a2.axhline(U_NOM, color=GREY, lw=1, ls="--")
    a2.text(1200, U_NOM + 0.06, r"nominal $u_{\rm nom}$", fontsize=7, color=GREY, ha="right")
    a2.axhline(-A_MAX, color=ORANGE, lw=1, ls=":", zorder=1)
    a2.text(1200, -A_MAX + 0.06, r"limit $-a_{\max}$", fontsize=7, color=ORANGE, ha="right")
    a2.set_xlabel("time (ms)", fontsize=8)
    a2.set_ylabel(r"input $u$ (m/s$^2$)", fontsize=8)
    a2.set_ylim(-1.25, 0.8)
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(rf"Toy: $\ddot x=u$, wall at $x_{{\max}}={1e3 * X_MAX:.0f}$ mm, $u_{{\rm nom}}={U_NOM}$ m/s$^2$, "
                 rf"$\lambda={LAM:.0f}$ 1/s, $a_{{\max}}={A_MAX:.0f}$ m/s$^2$", fontsize=8)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "bg_cbf_wall.svg")
    plt.close(fig)

    for key, (t, x, v, u, infeasible) in runs.items():
        hit = np.argmax(x > X_MAX) if np.any(x > X_MAX) else None
        msg = f"hits wall at t={1e3 * t[hit]:.0f} ms, v={1e3 * v[hit]:.0f} mm/s" if hit is not None \
            else f"stays: max x={1e3 * x.max():.2f} mm"
        print(f"{key}: v0={1e3 * v[0]:.0f} mm/s, min u={u.min():.3f}, QP infeasible somewhere: {infeasible}; {msg}")


def _sketch(out: Path, name: str, draw, ylim, xlabel="time $t$") -> None:
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    ax.axhline(0, color="black", lw=1.0)
    draw(ax)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel(r"margin $h(x(t))$", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def proofs(out: Path) -> None:
    t = np.linspace(0, 3, 400)

    def zero_label(ax, x=3.0, below=True):
        ax.text(x, -0.012 if below else 0.012, "zero line: the boundary of $C$", fontsize=6.5, ha="right",
                va="top" if below else "bottom")

    def p1(ax):  # x' = -x, h = 1 - x^2, from x0 = 0.9: h = 1 - 0.81 e^{-2t}
        ax.plot(t, 1 - 0.81 * np.exp(-2 * t), color=BLUE, lw=1.6, label=r"$h=1-x^2$, $\dot x=-x$")
        ax.axhline(0.19, color=GREY, lw=0.9, ls="--")
        ax.text(3.0, 0.21, r"start $h(x_0)=0.19$", fontsize=6.5, color=GREY, ha="right", va="bottom")
        zero_label(ax)
        ax.legend(fontsize=6.5, frameon=False, loc="center right")

    def integrate(rate, h0):  # forward Euler on the fine grid t
        h = np.empty_like(t)
        h[0] = h0
        for i in range(len(t) - 1):
            h[i + 1] = h[i] + (t[i + 1] - t[i]) * rate(t[i], h[i])
        return h

    def p2(ax):  # a margin obeying h' = -2h + p(t) with p >= 0, from h0 = 0.5; the comparison y = 0.5 e^{-2t}
        h = integrate(lambda s, x: -2 * x + 0.12 * (1 + np.sin(4 * s)) * np.exp(-s), 0.5)
        y = 0.5 * np.exp(-2 * t)
        ax.plot(t, h, color=BLUE, lw=1.6, label=r"a margin with $\dot h\geq-2h$")
        ax.plot(t, y, color=RED, lw=1.2, ls="--", label=r"comparison $y=0.5e^{-2t}$")
        zero_label(ax)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    def p3(ax):  # h_{k+1} >= (1 - gamma) h_k with gamma = 0.5, h0 = 1: the worst case 1, 0.5, 0.25, ...
        k = np.arange(8)
        lo = 0.5 ** k
        ax.step(k, lo, where="post", color=RED, lw=1.2, ls="--", label=r"worst case $(1-\gamma)^k h_0$")
        ax.plot(k, lo, "o", color=BLUE, ms=3.5, label=r"$\gamma=0.5$, $h_0=1$: $1, 0.5, 0.25,\dots$")
        ax.set_xlim(-0.2, 7.2)
        ax.text(7.2, -0.03, "zero line: the boundary of $C$", fontsize=6.5, ha="right", va="top")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    def p4(ax):  # h = -x^2, x' = 1 from x0 = 0: h(x(t)) = -t^2
        tt = np.linspace(0, 1, 200)
        ax.plot(tt, -tt ** 2, color=RED, lw=1.6, label=r"$h=-x^2$, $\dot x=1$: $h(x(t))=-t^2$")
        ax.plot([0, 0.35], [0, 0], color=BLUE, lw=2.4, label=r"slope at $t=0$: $\dot h=0$")
        ax.set_xlim(-0.03, 1.0)
        ax.text(1.0, 0.03, "zero line: $C=\\{0\\}$", fontsize=6.5, ha="right", va="bottom")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")

    def p5(ax):  # h' = -2h - 0.2 + p(t) with p >= 0, so h' >= -2h - 0.2: the floor is -0.1
        h = integrate(lambda s, x: -2 * x - 0.2 + 0.08 * (1 + np.sin(3 * s)) * np.exp(-0.8 * s), 0.3)
        ax.plot(t, h, color=BLUE, lw=1.6, label=r"a margin with $\dot h\geq-2h-0.2$")
        ax.plot(t, -0.1 + 0.4 * np.exp(-2 * t), color=RED, lw=1.2, ls="--", label=r"comparison $-0.1+0.4e^{-2t}$")
        ax.axhline(-0.1, color=ORANGE, lw=1.1, ls=":")
        ax.text(0.05, -0.11, r"floor $-c\bar w/\gamma=-0.1$", fontsize=6.5, color=ORANGE, ha="left", va="top")
        zero_label(ax, below=False)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    _sketch(out, "bg_cbf_proof_p1", p1, (-0.16, 1.1))
    _sketch(out, "bg_cbf_proof_p2", p2, (-0.1, 0.6))
    _sketch(out, "bg_cbf_proof_p3", p3, (-0.15, 1.15), xlabel="step $k$")
    _sketch(out, "bg_cbf_proof_p4", p4, (-1.05, 0.25))
    _sketch(out, "bg_cbf_proof_p5", p5, (-0.2, 0.42))
    print("p1: h(1) =", round(1 - 0.81 * np.exp(-2), 4), "; p2: bound at t=1, 3:", round(0.5 * np.exp(-2), 4),
          round(0.5 * np.exp(-6), 5), "; p3:", [0.5 ** k for k in range(4)])


def _arrow(ax, x0, y0, x1, y1, color, lw=1.4):
    assert x0 == x1 or y0 == y1, "arrows are horizontal or vertical only"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def lyap_vs_barrier(out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.3))
    th = np.linspace(0, 2 * np.pi, 300)
    # left: V = x^2 + y^2, nested sublevel sets, inward everywhere
    for r in (0.45, 0.75, 1.05):
        a1.plot(r * np.cos(th), r * np.sin(th), color=GREY, lw=1.0)
        L = 0.17
        _arrow(a1, r, 0, r - L, 0, BLUE)
        _arrow(a1, -r, 0, -r + L, 0, BLUE)
        _arrow(a1, 0, r, 0, r - L, BLUE)
        _arrow(a1, 0, -r, 0, -r + L, BLUE)
    a1.plot(0, 0, "o", color="black", ms=4)
    a1.text(0.0, -0.08, "equilibrium", fontsize=6.5, ha="center", va="top")
    a1.text(0.78, 0.86, r"$\{V\leq c\}$ for every $c$", fontsize=7, color=GREY, ha="left")
    a1.set_title(r"Lyapunov: $\dot V\leq0$ on every level set", fontsize=8)
    # right: one barrier set C = {h >= 0}, h = 1 - x^2 - y^2
    a2.fill(np.cos(th), np.sin(th), color=BLUE, alpha=0.07, lw=0)
    a2.plot(np.cos(th), np.sin(th), color="black", lw=1.4)
    a2.text(0.62, 0.86, r"$\partial C:\ h=0$", fontsize=7, ha="left")
    a2.text(-0.05, 0.42, r"$C=\{h\geq0\}$", fontsize=7, ha="center", color=BLUE)
    # inside: any direction, including toward the boundary (margin spent, h decreasing)
    for x0, y0, x1, y1 in [(0.15, 0.1, 0.5, 0.1), (-0.15, -0.05, -0.5, -0.05), (0.15, -0.2, 0.15, -0.55),
                           (-0.35, 0.15, -0.35, 0.45)]:
        _arrow(a2, x0, y0, x1, y1, BLUE)
    # boundary: tangent or inward only
    _arrow(a2, 1.0, -0.2, 1.0, 0.2, GREEN, lw=1.8)  # tangent
    _arrow(a2, 0.2, 1.0, -0.2, 1.0, GREEN, lw=1.8)  # tangent
    _arrow(a2, -1.0, 0.0, -0.75, 0.0, GREEN, lw=1.8)  # inward
    _arrow(a2, 0.0, -1.0, 0.0, -0.75, GREEN, lw=1.8)  # inward
    # outside: pushed back
    _arrow(a2, 1.42, -0.45, 1.17, -0.45, RED)
    a2.text(1.45, -0.55, "outside:\npushed back", fontsize=6.5, color=RED, ha="right", va="top")
    a2.text(0.0, -1.2, "no equilibrium needed", fontsize=6.5, ha="center", va="top")
    a2.set_title(r"Barrier: $\dot h\geq-\alpha(h)$, one set", fontsize=8)
    for ax in (a1, a2):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.45, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, lw=1.4), plt.Line2D([], [], color=GREEN, lw=1.8),
               plt.Line2D([], [], color=RED, lw=1.4)]
    fig.legend(handles, ["flow inside (any direction on the right)", "flow on the boundary: tangent or inward",
                         "flow outside $C$"], loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "bg_cbf_lyap_vs_barrier.svg")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    wall(out)
    proofs(out)
    lyap_vs_barrier(out)
