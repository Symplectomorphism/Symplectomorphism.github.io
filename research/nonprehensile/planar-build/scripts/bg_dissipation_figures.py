#!/usr/bin/env python3
"""Figures for the Background page on dissipation (passivity) certificates.

Every figure is a small, checkable example; none plots a record. Arrows are horizontal or vertical only. Run:

    python bg_dissipation_figures.py OUTDIR

Writes into OUTDIR:
  bg_dissipation_forces.svg   the block sliding on the platform with the disk on its face: the velocities, the
                              friction forces that dissipate and the hold force that supplies
  bg_dissipation_toy.svg      the toy: a block pushed by F against friction; its energy and position over time with
                              the bounds the dissipation inequality gives (they are exact here)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

BLUE = "#1a4f9c"
ORANGE = "#b26a00"
GREEN = "#2e7d32"
RED = "#c62828"
GREY = "#666666"
FILL = "#f3e2c4"
EDGE = "#7a4f0e"


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def arrow(ax, x0, y0, x1, y1, color, label=None, lx=0.0, ly=0.0, ha="center"):
    assert x0 == x1 or y0 == y1, "arrows are horizontal or vertical only"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, color=color, fontsize=7.5, ha=ha, va="center")


def forces(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot([-1.6, 1.8], [0, 0], color="black", lw=1.2)
    ax.add_patch(Rectangle((-1.6, -0.25), 3.4, 0.25, fc="#d9dde1", ec="none"))
    ax.text(1.75, -0.13, "platform", fontsize=7, color=GREY, ha="right", va="center")
    ax.add_patch(Rectangle((-0.5, 0), 1.0, 1.0, fc=FILL, ec=EDGE, lw=1.2))
    ax.text(0, 0.5, "block", fontsize=8, ha="center", va="center", color=EDGE)
    ax.add_patch(Circle((-0.62, 0.55), 0.12, fc="#c9d8fb", ec="#14307a", lw=1.0))
    ax.text(-0.62, 0.32, "disk", fontsize=7, ha="center", va="center", color="#14307a")
    # the slide and the slip
    arrow(ax, 0.6, 0.85, 1.15, 0.85, BLUE, r"$v_B$ (slide)", ly=0.09)
    arrow(ax, -0.62, 0.70, -0.62, 1.05, BLUE, r"$v_s$ (slip up the face)", lx=-0.06, ha="right")
    # dissipation: friction opposes each slip
    arrow(ax, 0.2, -0.06, -0.35, -0.06, RED, r"$\mu_t N_t$", ly=-0.12)
    arrow(ax, -0.5, 0.75, -0.5, 0.35, RED, r"$\mu_d N_m$", lx=0.07, ha="left")
    # supply: the hold force pushes the block forward
    arrow(ax, -1.45, 0.55, -0.76, 0.55, GREEN, r"hold $c_n$", ly=-0.1)
    # weight and support
    arrow(ax, 0.25, 0.5, 0.25, 0.15, GREY, r"$m_B g$", lx=0.06, ha="left")
    ax.text(0.0, -0.45, r"dissipation (red) $\geq$ supply (green): $\dot E \leq -c_B v_B - c_s v_s$",
            fontsize=7.5, ha="center", color="black")
    ax.set_xlim(-1.65, 1.85)
    ax.set_ylim(-0.6, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out, "bg_dissipation_forces")


def toy(out: Path) -> None:
    m, mu, g, F, v0 = 0.1, 0.2, 9.81, 0.1, 0.05
    c = mu * m * g - F
    a = c / m
    T = v0 / a
    t = np.linspace(0, 1.25 * T, 400)
    v = np.clip(v0 - a * t, 0, None)
    x = np.where(t < T, v0 * t - 0.5 * a * t ** 2, v0 * T - 0.5 * a * T ** 2)
    E = 0.5 * m * v ** 2
    E0 = 0.5 * m * v0 ** 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a1.plot(1e3 * t, 1e6 * E, color=BLUE, lw=1.6, label=r"$E(t)=\frac{1}{2} m v^2$")
    a1.axvline(1e3 * np.sqrt(2 * m * E0) / c, color=RED, ls="--", lw=1, label=r"bound $T\leq\sqrt{2mE_0}/c$")
    a1.set_xlabel("time (ms)", fontsize=8)
    a1.set_ylabel(r"kinetic energy ($\mu$J)", fontsize=8)
    a1.legend(fontsize=7, frameon=False)
    a2.plot(1e3 * t, 1e3 * x, color=BLUE, lw=1.6, label="position")
    a2.axhline(1e3 * E0 / c, color=RED, ls="--", lw=1, label=r"bound $\Delta x\leq E_0/c$")
    a2.set_xlabel("time (ms)", fontsize=8)
    a2.set_ylabel("distance slid (mm)", fontsize=8)
    a2.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(rf"Toy: $m={m}$ kg, $\mu={mu}$, push $F={F}$ N, $v_0={1e3 * v0:.0f}$ mm/s; $c=\mu mg-F={c:.4f}$ N",
                 fontsize=8)
    save(fig, out, "bg_dissipation_toy")
    print(f"toy: c={c:.5f} N, E0={E0:.3e} J, dx bound={1e3 * E0 / c:.3f} mm, exact={1e3 * v0 ** 2 / (2 * a):.3f} mm, "
          f"T bound={1e3 * np.sqrt(2 * m * E0) / c:.2f} ms, exact={1e3 * T:.2f} ms")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    forces(out)
    toy(out)
