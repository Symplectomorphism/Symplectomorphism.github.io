"""Figures drawn from the simulation results in the `rotea` repo's `src/`.

Each one reads a .npz written by an experiment script; nothing here recomputes
physics, so the slides and the simulation cannot drift apart.
"""

from __future__ import annotations

import os

import numpy as np

from windkit import (
    BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, SOFT, VIOLET, figure, save,
    style,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")


def _load(name):
    return np.load(os.path.join(SIM, name))


def sim_gate(path):
    """Curvature bias against how far the second harmonic sits into the rotor."""
    d = _load("exp1_gate.npz")
    x, he, ge = d["w2tau"], d["h_err"], d["g_err"]
    fig, ax = figure(11.6, 3.9)

    ax.axhline(0, color=FAINT, linewidth=1.2)
    ax.axhspan(-5, 5, color=SOFT, zorder=0)
    ax.text(0.048, 3.2, "within 5%", fontsize=10, color=PRIMARY)

    ax.plot(x, he, "o-", color=VIOLET, linewidth=2.6, markersize=8,
            label=r"curvature  $\hat H$,  read at $2\omega$")
    ax.plot(x, ge, "s-", color=HOT, linewidth=2.2, markersize=7,
            label=r"gradient  $\hat g$,  read at $\omega$")

    i = int(np.argmin(np.abs(d["period"] - 150.0)))
    ax.plot([x[i]], [he[i]], "o", color=GOLD, markersize=15, zorder=6,
            markerfacecolor="none", markeredgewidth=2.5)
    ax.annotate(f"published $T = 150$ s\nreads {abs(he[i]):.0f}% low",
                xy=(x[i], he[i]), xytext=(0.30, -31),
                fontsize=11, color=GOLD, linespacing=1.4, ha="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.5))

    ax.annotate(f"floor at ${ge[-1]:.1f}\\%$: the finite-amplitude bias,\nnot a dynamic one",
                xy=(x[-1], ge[-1]), xytext=(0.068, -16.5), fontsize=10,
                color=HOT, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=HOT, linewidth=1.1))

    ax.set_xscale("log")
    ax.set_xlim(0.040, 0.95)
    ax.set_ylim(-46, 10)
    style(ax, r"$2\omega\tau_{\rm rotor}$   (second harmonic, in units of the rotor pole)",
          "error in the estimate  [%]")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=11,
                    bbox_to_anchor=(0.0, 0.02))
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def sim_curvature(path):
    """Settling with and without the curvature in the loop gain."""
    import matplotlib.pyplot as plt
    d = _load("exp2_curvature.npz")
    sharp = [0.6, 1.0, 1.6]
    cols = {0.6: BLUE, 1.0: PRIMARY, 1.6: HOT}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)

    for ax, mode, title in ((a1, "gradient", "gradient ascent:  rate $=\\kappa|H|$"),
                            (a2, "newton", "Newton step:  rate $=\\kappa$")):
        for s in sharp:
            lam = d[f"{mode}_s{s}"]
            n = np.arange(1, len(lam) + 1)
            ax.plot(n, lam, color=cols[s], linewidth=2.4,
                    label=f"$|H|\\times{s:.1f}$")
        ax.axhline(float(d["lamstar_s1.0"]), color=FAINT, linewidth=1.2,
                   linestyle=(0, (4, 3)))
        ax.set_xlim(1, 22)
        style(ax, "dither period", "tip-speed ratio $\\lambda$" if mode == "gradient" else None,
              title)

    a1.set_ylim(7.02, 7.56)
    leg = a1.legend(loc="lower right", frameon=False, fontsize=10.5,
                    title="plant curvature", ncols=3, columnspacing=1.0)
    leg.get_title().set_fontsize(10)
    for t in leg.get_texts():
        t.set_color(INK)

    def span(v):
        return f"{v.min()} periods" if v.min() == v.max() else f"{v.min()}-{v.max()} periods"

    g, nw = d["settle_gradient"], d["settle_newton"]
    a1.text(21.4, 7.185, f"settles in {span(g)}\n"
            f"{g.max()/max(g.min(),1):.0f}$\\times$ spread across the three plants",
            fontsize=10.5, color=HOT, linespacing=1.45, ha="right")
    a2.text(21.4, 7.185, f"settles in {span(nw)}\n"
            "the same, whatever the curvature",
            fontsize=10.5, color=PRIMARY, linespacing=1.45, ha="right")
    fig.tight_layout()
    save(fig, path)


def sim_rolling(path):
    """How much averaging the curvature estimate needs."""
    d = _load("exp3_noise.npz")
    ht = abs(float(d["h_true"]))
    N = 2.0 / d["gammas"] - 1.0
    rel = d["rolling_std"] / ht
    fig, ax = figure(11.6, 3.9)

    white = rel[0] * np.sqrt(N[0] / N)
    ax.loglog(N, 100 * white, color=FAINT, linewidth=2.2, linestyle=(0, (5, 3)),
              label=r"white-noise prediction, $\propto N_H^{-1/2}$")
    ax.loglog(N, 100 * rel, "o-", color=VIOLET, linewidth=2.8, markersize=9,
              label=r"measured, $\propto N_H^{-0.72}$")

    ax.axhline(20, color=GOLD, linewidth=2.0, linestyle=(0, (6, 3)))
    ax.text(1.25, 23, "20% of $|H|$: the accuracy a usable gain needs",
            fontsize=10.5, color=GOLD)

    p = np.polyfit(np.log(N[1:]), np.log(rel[1:]), 1)
    need = np.exp((np.log(0.20) - p[1]) / p[0])
    Nx = np.logspace(np.log10(N[-1]), np.log10(need * 1.5), 50)
    ax.loglog(Nx, 100 * np.exp(p[1]) * Nx ** p[0], color=VIOLET, linewidth=1.5,
              linestyle=(0, (2, 3)))
    ax.plot([need], [20], "*", color=GOLD, markersize=22, zorder=7)
    ax.annotate(f"$N_H \\approx {need:.0f}$ periods\n= {need*600/86400:.0f} days at $T=600$ s",
                xy=(need, 20), xytext=(need * 0.30, 3.4), fontsize=11,
                color=GOLD, linespacing=1.4, ha="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.5))

    ax.set_xlim(0.9, need * 2.2)
    ax.set_ylim(2.5, 6e3)
    style(ax, r"$N_H = 2/\gamma - 1$   (periods the rolling estimate averages)",
          r"scatter of $\hat A$   [% of $|H|$]")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)
