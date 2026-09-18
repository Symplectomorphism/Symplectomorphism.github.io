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


