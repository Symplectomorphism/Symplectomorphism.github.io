"""Figures for the stochastic-wind deck: the spectrum, and what the
demodulator does to it."""

from __future__ import annotations

import numpy as np

from windkit import (
    BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, SOFT, VIOLET, figure, save,
    style,
)

TI, L, U = 0.10, 340.0, 8.0


def S_eps(f):
    """One-sided PSD of the relative wind fluctuation, IEC 61400-1 Kaimal."""
    return TI**2 * (4 * L / U) / (1 + 6 * np.abs(f) * L / U) ** (5 / 3)


def kaimal(path):
    """The Kaimal density, with the dither frequencies compared in the Design section."""
    f = np.logspace(-4.2, -0.9, 800)
    fig, ax = figure(11.6, 3.9)
    ax.loglog(f, S_eps(f), color=PRIMARY, linewidth=2.8)
    ax.axvline(U / (6 * L), color=FAINT, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(U / (6 * L) * 1.12, 2.4e-2, "corner\n$f = \\bar V/6L$", fontsize=10.5,
            color=MUTED, linespacing=1.3)
    ax.text(1.2e-4, 0.42, "flat:  $S_\\varepsilon \\to \\mathrm{TI}^2\\,4L/\\bar V$", fontsize=11,
            color=PRIMARY)
    ax.text(6e-2, 5e-2, "$\\propto f^{-5/3}$", fontsize=12, color=PRIMARY, ha="right")
    for T, dy in ((600.0, 6), (150.0, 6), (60.0, 6)):
        ax.plot([1 / T], [S_eps(1 / T)], "o", color=GOLD, markersize=10, zorder=6)
        ax.annotate(f"$T={T:.0f}$ s", xy=(1 / T, S_eps(1 / T)), xytext=(1 / T * 1.6, S_eps(1 / T) * dy),
                    fontsize=10.5, color=GOLD, ha="left",
                    arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.1))
    ax.set_ylim(4e-3, 12)
    style(ax, "frequency $f$  [Hz]", "$S_\\varepsilon(f)$   [1/Hz]")
    save(fig, path)


def _w2(f, T, N=1):
    """|w_hat_N(f)|^2 for sin(2 pi t / T) on [0, N T]."""
    a, b = 2 * np.pi / T, 2 * np.pi * np.asarray(f, dtype=float)
    near = np.abs(np.abs(b) - a) < 1e-9
    z = np.abs(a * (1 - np.exp(-1j * b * N * T)) / (a**2 - np.where(near, 1, b**2))) ** 2
    return np.where(near, (N * T / 2) ** 2, z)


def window(path):
    """One period's window spans the whole lobe; N periods' window narrows onto f_1."""
    T = 150.0
    f = np.linspace(1e-6, 3.2 / T, 6000)
    fig, ax = figure(11.6, 3.9)
    ax.plot(f * T, S_eps(f) / S_eps(1 / T), color=PRIMARY, linewidth=2.6,
            label=r"wind density $S_\varepsilon(f)/S_\varepsilon(f_1)$")
    for N, col, lw in ((1, VIOLET, 2.8), (4, HOT, 2.2)):
        w = _w2(f, T, N) / (N * T / 4) / T      # unit area over f/f_1 > 0
        lab = ("one period, $N = 1$" if N == 1 else f"$N = {N}$ periods") + r": window weight $|\hat w_N|^2$, unit area"
        ax.plot(f * T, w, color=col, linewidth=lw, label=lab)
    ax.axvline(1.0, color=GOLD, linewidth=1.6, linestyle=(0, (6, 3)))
    ax.text(1.05, 5.1, "$f_1 = 1/T$", fontsize=10.5, color=GOLD)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 5.6)
    style(ax, "frequency, in units of the dither frequency $f/f_1$", "normalized")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def roadmap(path):
    """From the wind's density to the power lost: the chain of the deck."""
    from windkit import blank, box
    fig, ax = figure(11.6, 3.3)
    blank(ax)
    ax.set_xlim(-0.05, 11.75)
    ax.set_ylim(0.0, 3.2)
    steps = [
        ("Link A", r"$y - \mathbb{E}y = 3\varepsilon$", "one random\nprocess", PRIMARY),
        ("Link B", r"$\hat g_{\rm noise} = \frac{6}{aT}Z$", "a linear\nfunctional", VIOLET),
        ("Link C", r"$\mathrm{Var}\,Z = \int S^{2s}|\hat w|^2 df$", "its variance", BLUE),
        ("Link D", r"Kaimal $S_\varepsilon$", "a number,\nchecked", HOT),
        ("Loop, Design", r"loss$(T, a, \tau_c)$", "the parameters", GOLD),
    ]
    W, GAP = 2.0, 0.36
    for i, (tag, out, what, col) in enumerate(steps):
        x = i * (W + GAP)
        box(ax, x, 0.35, W, 2.5, f"{tag}\n\n{out}\n\n{what}", ec=col, fs=10.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + W + GAP - 0.04, 1.6), xytext=(x + W + 0.04, 1.6),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6))
    save(fig, path, pad=0.05)


def design(path):
    """Least power lost at a one-day tracking time, against the dither period."""
    TAU, TAUC = 6.36, 86400.0
    T = np.geomspace(15, 900, 400)
    R = 9 * 0.5 * S_eps(1 / T)
    G1 = 1 / (1 + (2 * np.pi * TAU / T) ** 2)
    fig, ax = figure(11.6, 3.9)
    ax.axvspan(15, 45, color=SOFT, zorder=0)
    ax.text(16.5, 0.505, "period nearing the rotor's\n6.4 s time constant", fontsize=10,
            color=MUTED, linespacing=1.3, va="top")
    ax.semilogx(T, 100 * np.sqrt(R / (2 * TAUC * G1)), color=PRIMARY, linewidth=2.8,
                label=r"in-phase demodulation: $\sqrt{R/(2\tau_c G_1)}$")
    ax.semilogx(T, 100 * np.sqrt(R / (2 * TAUC)), color=BLUE, linewidth=2.0,
                linestyle=(0, (6, 3)), label=r"phase-compensated: $\sqrt{R/(2\tau_c)}$")
    Rp = 9 * 0.5 * S_eps(1 / 150.0); Gp = 1 / (1 + (2 * np.pi * TAU / 150.0) ** 2)
    a = 0.136 * 1.0; J2 = 0.7022
    su2 = Rp / (a**2 * Gp**2 * TAUC * J2**2)
    pub = 100 * (0.5 * J2 * su2 + 0.25 * J2 * Gp * a**2)
    ax.plot([150], [pub], "s", color=HOT, markersize=10, zorder=6)
    ax.annotate(f"published: $T = 150$ s, $a = 13.6\\%$\n{pub:.2f}%", xy=(150, pub), xytext=(62, 0.44),
                fontsize=10.5, color=HOT, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=HOT, linewidth=1.1))
    for Tm in (60.0, 150.0):
        Rm = 9 * 0.5 * S_eps(1 / Tm); Gm = 1 / (1 + (2 * np.pi * TAU / Tm) ** 2)
        v = 100 * np.sqrt(Rm / (2 * TAUC * Gm))
        ax.plot([Tm], [v], "o", color=PRIMARY, markersize=9, zorder=6)
        ax.annotate(f"{v:.2f}%", xy=(Tm, v), xytext=(Tm * 1.06, v - 0.045), fontsize=10.5,
                    color=PRIMARY, ha="left")
    ax.set_xlim(15, 900)
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks([20, 30, 60, 150, 300, 600], ["20", "30", "60", "150", "300", "600"])
    ax.minorticks_off()
    style(ax, "dither period $T$  [s]", "power lost  [%]\n(at the best amplitude)")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def validate(path):
    """Each link of the chain, checked against the full nonlinear simulation."""
    import matplotlib.pyplot as plt
    import os
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sim", "exp5_chain.npz"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.0),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})

    e, y = d["eps_sample"], d["y_sample"]
    a1.plot(3 * e, y, ".", color=PRIMARY, markersize=2.0, alpha=0.30)
    lim = 1.05 * np.abs(3 * e).max()
    a1.plot([-lim, lim], [-lim, lim], color=INK, linewidth=2.0,
            linestyle=(0, (5, 3)), label="$y-\\bar y = 3\\varepsilon$, predicted")
    a1.plot([-lim, lim], [-lim * d["slope"] / 3, lim * d["slope"] / 3],
            color=HOT, linewidth=2.2,
            label=f"fitted slope {float(d['slope']):.2f}")
    a1.set_xlim(-lim, lim)
    a1.set_ylim(-lim, lim)
    style(a1, r"$3\varepsilon(t)$", r"$y(t)-\bar y$",
          f"Link A:  correlation {float(d['corr']):.4f}")
    leg = a1.legend(loc="upper left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)

    names = ["A\n$y-\\bar y$ vs $3\\varepsilon$", "C\n$\\sigma_Z^2$",
             "D\n$\\sigma(\\hat g)$"]
    ratio = [float(d["sd_ratio"]), float(d["var_meas"] / d["var_pred"]),
             float(d["sg_meas"] / d["sg_pred"])]
    cols = [PRIMARY, BLUE, HOT]
    a2.axhspan(0.95, 1.05, color=SOFT, zorder=0)
    a2.axhline(1.0, color=INK, linewidth=1.6)
    a2.bar(range(3), ratio, 0.5, color="#ffffff", edgecolor=cols, linewidth=2.4)
    for i, r in enumerate(ratio):
        a2.text(i, r + 0.012, f"{r:.3f}", ha="center", fontsize=12, color=cols[i])
    a2.text(2.42, 1.052, "within $5\\%$", fontsize=10, color=PRIMARY, ha="right")
    a2.set_xticks(range(3))
    a2.set_xticklabels(names, fontsize=10.5)
    a2.set_ylim(0.80, 1.14)
    style(a2, None, "measured / predicted", "every link, independently")
    fig.tight_layout()
    save(fig, path)
