#!/usr/bin/env python3
"""Figures for the Background pages on certification (interval arithmetic, zonotopes, funnels, techniques).

Every figure is a small, checkable example; none plots a record. Run:

    python bg_certification_figures.py OUTDIR

Writes into OUTDIR:
  bg_dependency.svg      x - y over the box [1,2]^2 versus over the diagonal x = y (the dependency loss)
  bg_bisection.svg       certifying x*y >= 0.3 on [0.5,1]^2: whole box, 2x2, 4x4 pieces
  bg_same_hull.svg       two sets with the same interval hull, then rotated by -45 degrees
  bg_band_vs_time.svg    time-indexed versus progress-indexed (band) tubes on a diverging tipping motion
  bg_switch_band.svg     a support switch inside one formation: two force models and the band's min over both
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

BLUE = "#1a4f9c"
ORANGE = "#b26a00"
GREEN = "#2e7d32"
RED = "#c62828"
GREY = "#666666"


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def dependency(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.add_patch(Rectangle((1, 1), 1, 1, fc="#dbe6f5", ec=BLUE, lw=1.2))
    ax.plot([1, 2], [1, 2], color=ORANGE, lw=3, solid_capstyle="round")
    for v in (-1, -0.5, 0, 0.5, 1):
        xs = np.linspace(0.8, 2.2, 2)
        ax.plot(xs, xs - v, ":", color=GREY, lw=0.8)
    ax.annotate("the box: x and y independent\nx − y ranges over [−1, 1]", xy=(1.85, 1.15), xytext=(0.9, 0.55),
                fontsize=7, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax.annotate("the true set: y = x\nx − x = 0", xy=(1.5, 1.5), xytext=(0.55, 2.2), fontsize=7, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax.set_xlim(0.5, 2.6)
    ax.set_ylim(0.4, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Dependency: [x] − [x] treats the two x's as independent\n(dotted: lines of constant x − y)", fontsize=8)
    save(fig, out, "bg_dependency")


def bisection(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    lo, hi, need = 0.5, 1.0, 0.3
    xs = np.linspace(lo, hi, 200)
    for ax, n in zip(axes, (1, 2, 4)):
        X, Y = np.meshgrid(xs, xs)
        ax.contourf(X, Y, X * Y, levels=[0, need, 2], colors=["#f6d5d5", "#e8f3e8"], alpha=0.8)
        ax.contour(X, Y, X * Y, levels=[need], colors=[RED], linewidths=1.0)
        h = (hi - lo) / n
        ok = 0
        for i in range(n):
            for j in range(n):
                x0, y0 = lo + i * h, lo + j * h
                lower = x0 * y0  # interval product of [x0, x0+h] * [y0, y0+h], both positive
                good = lower >= need
                ok += good
                ax.add_patch(Rectangle((x0, y0), h, h, fill=False, ec=GREEN if good else RED, lw=1.4))
                ax.text(x0 + h / 2, y0 + h / 2, f"{lower:.3f}", fontsize=6 if n == 4 else 7, ha="center",
                        va="center", color=GREEN if good else RED)
        ax.set_title(f"{n * n} piece{'s' if n > 1 else ''}: {ok} certified", fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("y", fontsize=8)
    for ax in axes:
        ax.set_xlabel("x", fontsize=8)
    fig.suptitle("Is x·y ≥ 0.3 on [0.5, 1]²?  Number in each piece: the interval lower bound of x·y "
                 "(red curve: x·y = 0.3)", fontsize=7.5)
    save(fig, out, "bg_bisection")


def same_hull(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    R = np.array([[np.cos(-np.pi / 4), -np.sin(-np.pi / 4)], [np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
    box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    G = np.array([[1.0, 0.1], [1.0, -0.1]])  # a thin diagonal zonotope with hull [-1.1, 1.1]^2
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    zon = signs @ G.T
    boxh = 1.1 * box
    for ax, (A, title) in zip(axes, ((np.eye(2), "same interval hull [−1.1, 1.1]²"),
                                      (R, "after a rotation by −45°"))):
        b = boxh @ A.T
        z = zon @ A.T
        ax.add_patch(Polygon(b, closed=True, fc="#dbe6f5", ec=BLUE, lw=1.2, label="the box"))
        ax.add_patch(Polygon(z, closed=True, fc="#f4c27a", ec=ORANGE, lw=1.2, label="the thin zonotope"))
        for P, col in ((b, BLUE), (z, ORANGE)):
            lo, hi = P.min(axis=0), P.max(axis=0)
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=col, lw=0.8, ls="--"))
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.suptitle("Dashed: each set's interval hull. Before: identical. After, in half-widths: the zonotope's hull "
                 "1.41 × 0.14, the box's 1.56 × 1.56", fontsize=7)
    save(fig, out, "bg_same_hull")


def band_vs_time(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a = 1.2
    t = np.linspace(0, 2.0, 400)
    starts = np.linspace(0.08, 0.12, 9)
    for th0 in starts:
        for ax in axes:
            ax.plot(t, th0 * np.exp(a * t), color=GREY, lw=0.7)
    t_star = 1.6
    th_t = starts * np.exp(a * t_star)
    axes[0].plot([t_star, t_star], [th_t.min(), th_t.max()], color=RED, lw=3)
    axes[0].set_title(f"time-indexed: at t = {t_star} the set spans\n{th_t.min():.2f}–{th_t.max():.2f} in tilt "
                      "(and keeps widening)", fontsize=7.5)
    th_star = 0.5
    t_hit = np.log(th_star / starts) / a
    axes[1].plot([t_hit.min(), t_hit.max()], [th_star, th_star], color=GREEN, lw=3)
    axes[1].set_title(f"progress-indexed (band): at tilt {th_star} the spread\nis only in arrival time "
                      f"({t_hit.min():.2f}–{t_hit.max():.2f})", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("time", fontsize=8)
        ax.set_ylabel("tilt (progress)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1.2)
    save(fig, out, "bg_band_vs_time")


def switch_band(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.linspace(0, 1, 400)
    xs = 0.55
    mA = 0.30 - 0.25 * x  # margin under force model A (support at the step corner)
    mB = (0.30 - 0.25 * xs) + 0.30 * (x - xs)  # margin under model B (support at the block corner)
    true = np.where(x < xs, mA, mB)
    ax.plot(x, mA, color=BLUE, lw=1, ls="--", label="model A (left support at the step corner)")
    ax.plot(x, mB, color=ORANGE, lw=1, ls="--", label="model B (left support at the block corner)")
    ax.plot(x, true, color="k", lw=2, label="true margin: A before the switch, B after (a kink)")
    w = 0.12
    band = (x > xs - w) & (x < xs + w)
    ax.fill_between(x[band], 0, np.minimum(mA, mB)[band], color=GREEN, alpha=0.25,
                    label="band across the switch: its bound is min(A, B)")
    ax.axvline(xs, color=GREY, lw=0.8, ls=":")
    ax.text(xs + 0.01, 0.02, "switch", fontsize=7, color=GREY)
    ax.set_xlabel("progress along the transport (schematic)", fontsize=8)
    ax.set_ylabel("certified margin", fontsize=8)
    ax.set_ylim(0, 0.32)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_title("One formation, two force models: a band that straddles the switch takes the min over both",
                 fontsize=7.5)
    save(fig, out, "bg_switch_band")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    dependency(out)
    bisection(out)
    same_hull(out)
    band_vs_time(out)
    switch_band(out)


if __name__ == "__main__":
    main()
