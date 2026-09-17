"""Figures for the three turbine-level open questions.

Q1 numbers are Tables 3 and 4 of Ciri, Leonardi & Rotea (Wind Energy 2019),
against the design values k_opt = 2.2 and lambda_opt = 7.5 quoted in the same
paper. Q2 calibrates the loop gain from that paper's own settling time.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from windkit import (
    BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, RULE, SOFT, VIOLET,
    arrow, blank, box, figure, save, style,
)

K_OPT, LAM_OPT = 2.2, 7.5


def anomaly(path):
    """Every reported torque gain sits below the design value."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    labels = ["4", "8", "12", "4", "8", "12"]
    esc = [2.19, 2.06, np.nan, 2.13, 2.03, np.nan]
    lp = [2.18, 2.07, 2.10, 2.10, 2.02, 2.03]
    x = np.arange(6)
    w = 0.36

    a1.axhline(0, color=INK, linewidth=1.4)
    a1.bar(x - w/2, [100*(v/K_OPT - 1) for v in esc], w, color="#ffffff",
           edgecolor=HOT, linewidth=1.8, hatch="///", label="ESC")
    a1.bar(x + w/2, [100*(v/K_OPT - 1) for v in lp], w, color=PRIMARY,
           edgecolor=PRIMARY, linewidth=1.4, label="LP-ESC")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=11)
    a1.text(1.0, -10.3, "uniform inflow", ha="center", fontsize=10, color=MUTED)
    a1.text(4.0, -10.3, "shear + 10% TI", ha="center", fontsize=10, color=MUTED)
    a1.set_ylim(-11.4, 2.0)
    style(a1, "wind speed [m/s] and inflow", "converged $\\bar k$ vs design  [%]",
          "the torque gain lands low, every time")
    leg = a1.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)

    lam_esc = [7.60, 7.75, np.nan, 7.72, 7.85, np.nan]
    lam_lp = [7.60, 7.73, 7.70, 7.75, 7.85, 7.85]
    a2.axhline(LAM_OPT, color=INK, linewidth=1.6)
    a2.text(5.42, LAM_OPT + 0.020, "design $\\lambda^\\star = 7.5$", fontsize=10.5,
            color=INK, ha="right", va="bottom")
    a2.plot(x, lam_esc, "o", color=HOT, markersize=10, label="ESC")
    a2.plot(x, lam_lp, "s", color=PRIMARY, markersize=9, label="LP-ESC")
    a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=11)
    a2.text(1.0, 7.425, "uniform inflow", ha="center", fontsize=10, color=MUTED)
    a2.text(4.0, 7.425, "shear + 10% TI", ha="center", fontsize=10, color=MUTED)
    a2.set_ylim(7.39, 8.04)
    style(a2, "wind speed [m/s] and inflow", "converged $\\bar\\lambda$",
          "and the tip-speed ratio lands high")
    leg = a2.legend(loc="upper left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    save(fig, path)


def budget(path):
    """The two obvious explanations do not come close."""
    fig, ax = figure(9.4, 4.9)

    items = [
        ("finite-difference bias\nof the central difference", 0.012, GOLD),
        ("turbulent optimum\n(Jensen, $\\sigma_\\lambda = 0.5$)", 0.015, VIOLET),
        ("averaging artifact\nmean of $\\lambda$ vs $\\lambda$ of the mean", 0.015, BLUE),
        ("the three together", 0.042, MUTED),
        ("observed gap", 0.350, HOT),
    ]
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, col) in zip(y, items):
        ax.barh(yi, v, 0.62, color=col if col != HOT else "#ffffff",
                edgecolor=col, linewidth=2.0,
                hatch="///" if col == HOT else None)
        ax.text(v + 0.008, yi, f"  {v:.3f}", va="center", fontsize=11.5,
                color=col, fontweight="bold" if col == HOT else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=10.5)
    ax.set_xlim(0, 0.44)
    style(ax, "shift in $\\lambda$", None)
    ax.axvline(0.2, color=FAINT, linewidth=1.4, linestyle=(0, (5, 3)))
    ax.text(0.206, 4.35, "what erosion moves $\\lambda^\\star$ by", fontsize=10.5,
            color=MUTED)
    ax.text(0.0, -1.62, "Computed on a fitted surface: explaining the gap by "
            "finite-difference bias alone would need a third derivative about "
            "28 times larger.", fontsize=10, color=FAINT)
    save(fig, path)


def loop(path):
    """What is actually around the loop, including the update delay."""
    fig, ax = figure(9.8, 3.9)
    blank(ax)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.0)

    yf, h = 2.55, 0.68
    box(ax, 0.20, yf, 1.55, h, "rotor\n$\\frac{1}{1+\\tau s}$", ec=BLUE, fs=11)
    box(ax, 2.25, yf, 1.70, h, "$\\ln P$", ec=PRIMARY, fs=12)
    box(ax, 4.45, yf, 2.35, h, "average over $T$", ec=PRIMARY, fs=11)
    box(ax, 7.30, yf, 3.05, h, "hold until period ends", ec=HOT, fs=11)

    yr = 1.05
    box(ax, 6.05, yr, 2.35, h, "$\\kappa$ per update", ec=PRIMARY, fs=11)
    box(ax, 2.05, yr, 2.60, h, "$+\\,a\\,g(t)$", ec=PRIMARY, fs=11)

    ym = yf + h/2
    for p, q in ((1.75, 2.25), (3.95, 4.45), (6.80, 7.30)):
        arrow(ax, (p, ym), (q, ym))
    ax.plot([10.35, 10.55], [ym, ym], color=INK, linewidth=1.5, zorder=2)
    ax.plot([10.55, 10.55], [ym, yr + h/2], color=INK, linewidth=1.5, zorder=2)
    arrow(ax, (10.55, yr + h/2), (8.40, yr + h/2))
    arrow(ax, (6.05, yr + h/2), (4.65, yr + h/2))
    ax.plot([2.05, 1.00], [yr + h/2, yr + h/2], color=INK, linewidth=1.5, zorder=2)
    ax.plot([1.00, 1.00], [yr + h/2, yf], color=INK, linewidth=1.5, zorder=2)
    arrow(ax, (1.00, yf - 0.02), (1.00, yf))

    ax.annotate("", xy=(8.82, yf), xytext=(8.82, 1.00),
                arrowprops=dict(arrowstyle="->", color=HOT, linewidth=1.6))
    ax.text(8.82, 0.78, "this is the second pole", ha="center", fontsize=11,
            color=HOT)
    ax.text(0.20, 0.14, "$\\tau \\approx 8$ s,  $T = 150$ s, so the rotor is fully "
            "settled between updates. The delay is not.",
            fontsize=10.5, color=MUTED)
    save(fig, path, pad=0.05)


def boundary(path):
    """Calibrate K from the published settling time, then extrapolate."""
    fig, ax = figure(9.4, 4.7)
    V = np.linspace(3.5, 13, 400)
    K8 = 1 - 0.01 ** (1 / (11 * 60 / 150))

    ax.fill_between(V, 1.0, 3.3, color="#fbf0ee", zorder=0)
    ax.plot(V, K8 * (V / 8) ** 3, color=HOT, linewidth=2.8)
    ax.plot(V, np.full_like(V, K8), color=PRIMARY, linewidth=2.8)
    ax.axhline(1.0, color=INK, linewidth=2.0, linestyle=(0, (6, 3)))

    ax.text(3.75, 3.00, "$K > 1$:  unstable", fontsize=12.5, color=HOT)
    ax.text(10.15, K8 * (10.15 / 8) ** 3 - 0.30, "ESC,  $K \\propto V^3$",
            fontsize=11.5, color=HOT, rotation=30, ha="center")
    ax.text(12.85, K8 + 0.10, "LP-ESC,  $K$ constant", fontsize=11.5,
            color=PRIMARY, ha="right")

    for v, ok, dy in ((4, True, 0.20), (8, True, -0.26), (12, False, 0.22)):
        k = K8 * (v / 8) ** 3
        ax.plot([v], [k], "o", color=HOT, markersize=11, zorder=6)
        ax.text(v, k + dy, "LES: stable" if ok else "LES: unstable",
                fontsize=10.5, color=HOT, ha="center",
                va="top" if dy < 0 else "bottom")

    vb = 8 * (1 / K8) ** (1 / 3)
    ax.plot([vb], [1.0], "*", color=GOLD, markersize=22, zorder=7)
    ax.annotate(f"predicted boundary\n{vb:.1f} m/s\n(untested)", xy=(vb, 1.0),
                xytext=(vb - 0.55, 1.95), fontsize=11.5, color=GOLD,
                linespacing=1.4, ha="right",
                arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.5))

    ax.set_xlim(3.5, 13)
    ax.set_ylim(-0.42, 3.3)
    style(ax, "wind speed  [m/s]", "loop gain per update  $K$")
    ax.set_yticks([0, 1, 2, 3])
    ax.text(3.6, -0.36, "$K$ fixed by the paper's own 11-minute settling time at "
            "8 m/s. Nothing else is fitted.", fontsize=10.5, color=MUTED)
    save(fig, path)


def excitation(path):
    """Turbulence already sweeps lambda about as far as the probe does."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                 gridspec_kw={"width_ratios": [1.5, 1.0]})
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1200, 2400)

    w = rng.standard_normal(t.size)
    kern = np.exp(-np.linspace(0, 5, 300)); kern /= kern.sum()
    turb = np.convolve(w, kern, mode="same")
    turb = 0.245 * turb / turb.std()
    dither = 0.338 * np.sign(np.sin(2 * np.pi * t / 150.0))

    a1.plot(t, 7.5 + turb, color=HOT, linewidth=1.3,
            label="turbulence alone")
    a1.plot(t, 7.5 + turb + dither, color=PRIMARY, linewidth=1.0, alpha=0.75,
            label="turbulence + probe")
    a1.axhline(7.5, color=FAINT, linewidth=1.2)
    a1.set_xlim(0, 1200)
    a1.set_ylim(6.5, 8.5)
    style(a1, "time  [s]", "tip-speed ratio  $\\lambda$",
          "the wind is already sweeping the curve")
    leg = a1.legend(loc="upper right", frameon=False, fontsize=10)
    for x in leg.get_texts():
        x.set_color(INK)

    a2.bar([0], [0.245], 0.52, color="#ffffff", edgecolor=HOT, linewidth=2.2,
           hatch="///")
    a2.bar([1], [0.338], 0.52, color=PRIMARY, edgecolor=PRIMARY, linewidth=1.5)
    for xi, v, c in ((0, 0.245, HOT), (1, 0.338, PRIMARY)):
        a2.text(xi, v + 0.018, f"{v:.2f}", ha="center", fontsize=12, color=c)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["turbulence", "probe"], fontsize=11)
    a2.set_ylim(0, 0.44)
    style(a2, None, "$\\sigma_\\lambda$ contributed", "same order")

    fig.tight_layout()
    save(fig, path)


def zplane(path):
    """Root locus of z^2 - z + K = 0: the pair leaves the circle at 60 degrees."""
    fig, ax = figure(7.0, 5.6)
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK, linewidth=2.0)
    ax.fill(np.cos(th), np.sin(th), color=SOFT, zorder=0)

    Kc = np.linspace(0.25, 2.6, 300)
    ax.plot(np.full_like(Kc, 0.5), np.sqrt(4*Kc-1)/2, color=HOT, linewidth=2.2)
    ax.plot(np.full_like(Kc, 0.5), -np.sqrt(4*Kc-1)/2, color=HOT, linewidth=2.2)
    Kr = np.linspace(0.0, 0.25, 120)
    for sgn in (1, -1):
        ax.plot((1 + sgn*np.sqrt(1-4*Kr))/2, np.zeros_like(Kr), color=HOT,
                linewidth=2.2)

    for K, lab, col, dx in ((0.649, "$K=0.65$,  8 m/s", PRIMARY, -0.14),
                            (1.0, "$K=1$,  boundary", GOLD, 0.14),
                            (2.19, "$K=2.19$,  12 m/s", HOT, 0.14)):
        z = 0.5 + 1j*np.sqrt(4*K-1)/2
        ax.plot([z.real, z.real], [z.imag, -z.imag], "o", color=col,
                markersize=11, zorder=6)
        ax.text(z.real + dx, z.imag, lab, fontsize=11, color=col,
                va="center", ha="left" if dx > 0 else "right")

    ax.plot([0, 0.5], [0, np.sqrt(3)/2], color=GOLD, linewidth=1.4,
            linestyle=(0, (4, 3)))
    ax.text(0.10, 0.30, "$60^\\circ$", fontsize=12, color=GOLD)

    ax.set_xlim(-1.25, 1.55)
    ax.set_ylim(-2.05, 1.7)
    ax.set_aspect("equal")
    style(ax, "Re $z$", "Im $z$")
    ax.axhline(0, color=RULE, linewidth=1.0)
    ax.axvline(0, color=RULE, linewidth=1.0)
    ax.text(-1.22, -2.00, "leaving at $60^\\circ$ means the unstable mode rings "
            "at $6T$,\nwhich is 15 minutes at the published dither period.",
            fontsize=10.5, color=MUTED, linespacing=1.4, va="bottom")
    save(fig, path)


def probe_cost(path):
    """What the machine pays for being probed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.3))

    a1.bar([0, 1], [14.2, 0.3], 0.5,
           color=["#ffffff", PRIMARY], edgecolor=[HOT, PRIMARY],
           linewidth=[2.2, 1.5], hatch=["///", None])
    for x, v, c in ((0, 14.2, HOT), (1, 0.3, PRIMARY)):
        a1.text(x, v + 0.5, f"{v}%", ha="center", fontsize=13, color=c)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["LP-ESC", "LP-PIESC"], fontsize=11)
    a1.set_ylim(0, 17)
    style(a1, None, "energy lost vs. an oracle  [%]", "the cost of searching")

    labels = ["thrust DEL\nESC vs LP-ESC\n4 m/s", "drivetrain DEL\none step change\n4 m/s"]
    a2.bar([0, 1], [65, 70], 0.5, color="#ffffff", edgecolor=HOT, linewidth=2.2,
           hatch="///")
    for x, v in ((0, 65), (1, 70)):
        a2.text(x, v + 2, f"+{v}%", ha="center", fontsize=13, color=HOT)
    a2.set_xticks([0, 1]); a2.set_xticklabels(labels, fontsize=9.5)
    a2.set_ylim(0, 88)
    style(a2, None, "increase in damage-equivalent load  [%]",
          "the cost in fatigue")

    fig.tight_layout()
    save(fig, path)
