"""Act 1 figures: extremum seeking, the log trick, and the published evidence."""

from __future__ import annotations

import numpy as np

from windkit import (
    BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, RULE, SOFT, VIOLET,
    arrow, blank, box, figure, save, style,
)


def esc_loop(path):
    """The classical dither-demodulate-integrate loop, with log feedback."""
    fig, ax = figure(9.6, 4.2)
    blank(ax)
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 4.4)

    y_fwd, y_ret = 2.9, 0.75
    h = 0.62

    box(ax, 0.30, y_fwd, 1.50, h, "turbine\n$u \\mapsto P$", ec=BLUE, fs=11)
    box(ax, 2.30, y_fwd, 1.15, h, "$\\ln(\\cdot)$", ec=PRIMARY, fs=13)
    box(ax, 4.70, y_fwd, 1.35, h, "high-pass", ec=PRIMARY, fs=10.5)
    box(ax, 6.55, y_fwd, 1.05, h, "$\\times$", ec=GOLD, fs=15)
    box(ax, 8.10, y_fwd, 1.35, h, "low-pass", ec=PRIMARY, fs=10.5)

    box(ax, 8.30, y_ret, 1.15, h, "$k/s$", ec=PRIMARY, fs=13)
    box(ax, 4.55, y_ret, 1.55, h, "$\\hat u$  estimate", ec=PRIMARY, fs=10.5)
    box(ax, 1.90, y_ret, 1.45, h, "$+\\,a\\sin\\omega t$", ec=HOT, fs=11)

    ym = y_fwd + h / 2
    for a, b in ((1.80, 2.30), (3.45, 4.70), (6.05, 6.55),
                 (7.60, 8.10)):
        arrow(ax, (a, ym), (b, ym))

    # demodulation input
    arrow(ax, (7.075, 4.12), (7.075, y_fwd + h), color=GOLD)
    ax.text(7.17, 4.18, "$\\sin(\\omega t + \\theta)$", fontsize=11, color=GOLD,
            va="bottom")

    # return path
    yr = y_ret + h / 2
    arrow(ax, (9.45, ym), (10.20, ym))
    ax.plot([10.20, 10.20], [ym, yr], color=INK, linewidth=1.5, zorder=2)
    arrow(ax, (10.20, yr), (9.45, yr))
    arrow(ax, (8.30, yr), (6.10, yr))
    arrow(ax, (4.55, yr), (3.35, yr))
    ax.plot([1.90, 1.05], [yr, yr], color=INK, linewidth=1.5, zorder=2)
    ax.plot([1.05, 1.05], [yr, y_fwd - 0.02], color=INK, linewidth=1.5, zorder=2)
    arrow(ax, (1.05, y_fwd - 0.02), (1.05, y_fwd))

    ax.text(2.05, ym + 0.20, "$P$", fontsize=11.5, color=MUTED)
    ax.text(4.075, ym + 0.20, "$y = \\ln P$", fontsize=11.5, color=PRIMARY,
            ha="center")

    ax.text(10.27, yr + 0.62, "gradient\nestimate", fontsize=10.5, color=GOLD,
            ha="left", va="bottom", linespacing=1.3)
    ax.text(0.55, y_fwd - 0.33, "$u$", fontsize=12, color=HOT)

    ax.text(0.30, 0.10,
            "The gradient is never measured. It is inferred from how much of the injected "
            "tone comes back out.",
            fontsize=11, color=MUTED)
    save(fig, path, pad=0.05)


def logfix(path):
    """What taking the logarithm actually fixes: the 27:1 rate spread."""
    fig, (a1, a2) = __import__("matplotlib.pyplot", fromlist=["x"]).subplots(
        1, 2, figsize=(9.6, 4.0))
    v = np.linspace(4, 12, 400)

    a1.plot(v, (v / 8.0) ** 3, color=HOT, linewidth=2.8)
    a1.axhline(1.0, color=FAINT, linewidth=1.1, linestyle=(0, (4, 3)))
    a1.plot([8], [1], "o", color=HOT, markersize=8, zorder=5)
    a1.annotate("tuned here", xy=(8, 1), xytext=(6.1, 2.1), fontsize=10.5,
                color=MUTED, arrowprops=dict(arrowstyle="->", color=FAINT, lw=1.2))
    a1.text(4.3, 3.05, "$\\omega_c \\propto \\rho V^3$", fontsize=14, color=HOT)
    a1.text(4.3, 2.55, "27 : 1 across region 2", fontsize=11, color=MUTED)
    a1.set_ylim(0, 3.6)
    style(a1, "wind speed  [m/s]", "loop bandwidth  (8 m/s = 1)",
          "power feedback,  $J = P$")

    a2.plot(v, np.ones_like(v), color=PRIMARY, linewidth=2.8)
    a2.axhline(1.0, color=FAINT, linewidth=1.1, linestyle=(0, (4, 3)))
    a2.plot([8], [1], "o", color=PRIMARY, markersize=8, zorder=5)
    a2.text(4.3, 3.05, "$\\nu_c = \\frac{\\kappa}{C_P^{\\max}}"
                       "\\left|\\frac{\\partial^2 C_P}{\\partial u^2}\\right|$",
            fontsize=13, color=PRIMARY)
    a2.text(4.3, 2.35, "no $V$, no $\\rho$: tune once", fontsize=11, color=MUTED)
    a2.set_ylim(0, 3.6)
    style(a2, "wind speed  [m/s]", None, "log-power feedback,  $J = \\ln P$")

    fig.tight_layout()
    save(fig, path)


def evidence(path):
    """Ciri/Leonardi/Rotea 2019, Table 4: settling time under shear + 10% TI."""
    fig, ax = figure(9.2, 4.3)
    speeds = ["4 m/s", "8 m/s", "12 m/s"]
    esc = [31.0, 8.0, np.nan]
    lp = [8.1, 8.1, 8.0]
    x = np.arange(3)
    w = 0.34

    ax.bar(x - w / 2, esc, w, color="#ffffff", edgecolor=HOT, linewidth=2.0,
           hatch="///", label="ESC  (power feedback)")
    ax.bar(x + w / 2, lp, w, color=PRIMARY, edgecolor=PRIMARY, linewidth=1.5,
           label="LP-ESC  (log-power)")

    for xi, val in zip(x - w / 2, esc):
        if np.isnan(val):
            ax.text(xi, 1.2, "unstable", ha="center", fontsize=12, color=HOT,
                    rotation=90, fontweight="bold", va="bottom")
        else:
            ax.text(xi, val + 0.7, f"{val:.1f}", ha="center", fontsize=11, color=HOT)
    for xi, val in zip(x + w / 2, lp):
        ax.text(xi, val + 0.7, f"{val:.1f}", ha="center", fontsize=11, color=PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(speeds)
    ax.set_ylim(0, 36)
    style(ax, None, "99% settling time  [min]")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(0.0, 34.2, "Both tuned once, at 8 m/s, then left alone.",
            fontsize=11, color=MUTED, ha="center")
    save(fig, path)


def piesc(path):
    """LP-PIESC replaces demodulation with recursive least squares."""
    fig, ax = figure(9.6, 4.3)
    blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)

    box(ax, 0.15, 2.45, 4.55, 1.85,
        "LP-ESC\n\n"
        "inject $a\\sin\\omega t$, multiply the\n"
        "response by $\\sin(\\omega t+\\theta)$, low-pass\n"
        "$\\Rightarrow$  one gradient per dither period",
        ec=MUTED, fs=11)

    box(ax, 5.30, 2.45, 4.55, 1.85,
        "LP-PIESC\n\n"
        "fit  $\\dot y = \\theta_0 + \\theta_1(u-\\hat u)$  by\n"
        "recursive least squares; $\\theta_1$ is the\n"
        "gradient. Add a proportional term.",
        ec=PRIMARY, fs=11)

    arrow(ax, (4.70, 3.35), (5.30, 3.35), color=PRIMARY, lw=2.0)

    box(ax, 0.15, 0.20, 3.05, 1.85,
        "Dither amplitude\n\n"
        "$a$ cut to 25%\nof the LP-ESC value",
        fc=SOFT, ec=PRIMARY, fs=11)
    box(ax, 3.48, 0.20, 3.05, 1.85,
        "Energy vs. a controller\nthat already knows $k_{opt}$\n\n"
        "LP-ESC  $-14.2\\%$\nLP-PIESC  $-0.3\\%$",
        fc=SOFT, ec=PRIMARY, fs=11)
    box(ax, 6.80, 0.20, 3.05, 1.85,
        "Wake steering, wind tunnel\n12 turbines, yaw\n\n"
        "$+8.9\\%$ farm power,\nconverged in seconds",
        fc=SOFT, ec=PRIMARY, fs=11)
    save(fig, path, pad=0.05)

def piesc_loop(path):
    """LP-PIESC on the same skeleton as esc_loop, with the estimator in full."""
    fig, ax = figure(10.0, 6.9)
    blank(ax)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 6.85)

    y_f, h = 5.30, 0.62
    ymf = y_f + h / 2

    box(ax, 0.30, y_f, 1.50, h, "turbine\n$u \\mapsto P$", ec=BLUE, fs=11)
    box(ax, 2.30, y_f, 1.15, h, "$\\ln(\\cdot)$", ec=PRIMARY, fs=13)
    arrow(ax, (1.80, ymf), (2.30, ymf))
    arrow(ax, (3.45, ymf), (4.55, ymf))
    ax.text(4.00, ymf + 0.20, "$y = \\ln P$", fontsize=11.5, color=PRIMARY,
            ha="center")

    # ---- the estimator, in full -------------------------------------------
    box(ax, 4.55, 3.10, 5.75, 3.00, "", ec=VIOLET)
    ax.text(7.425, 5.98, "recursive least squares on\n"
            "$\\dot y = \\theta_0 + \\theta_1(u-\\hat u) = \\phi^{T}\\theta$",
            ha="center", va="top", fontsize=11, color=VIOLET,
            fontweight="bold", linespacing=1.5)
    ax.text(7.425, 5.25,
            "$e = y - \\hat y$\n"
            "$\\dot{\\hat y} = \\phi^{T}\\hat\\theta + Ke + c^{T}\\dot{\\hat\\theta}$\n"
            "$\\dot c^{T} = -Kc^{T} + \\phi^{T}$\n"
            "$\\dot\\Sigma^{-1} = -\\Sigma^{-1}cc^{T}\\Sigma^{-1} + k_T\\Sigma^{-1}"
            " - \\sigma\\Sigma^{-2}$\n"
            "$\\dot{\\hat\\theta} = \\mathrm{Proj}(\\Sigma^{-1}(ce - "
            "\\sigma\\hat\\theta),\\; \\hat\\theta)$",
            ha="center", va="top", fontsize=10, color=INK, linespacing=1.5)

    # ---- theta-1-hat out, down to the two PI branches ---------------------
    box(ax, 8.20, 2.05, 1.40, h, "$-k_p$", ec=GOLD, fs=13)
    box(ax, 8.20, 0.95, 1.40, h, "$-1/\\tau_I s$", ec=GOLD, fs=12)

    ax.plot([10.30, 10.95], [4.60, 4.60], color=VIOLET, linewidth=1.6, zorder=2)
    ax.plot([10.95, 10.95], [4.60, 1.26], color=VIOLET, linewidth=1.6, zorder=2)
    arrow(ax, (10.95, 2.36), (9.60, 2.36), color=VIOLET)
    arrow(ax, (10.95, 1.26), (9.60, 1.26), color=VIOLET)
    ax.text(11.08, 3.45, "$\\hat\\theta_1$", fontsize=13, color=VIOLET)

    # ---- back to the summing junction -------------------------------------
    box(ax, 2.80, 2.00, 0.60, 0.60, "$+$", ec=HOT, fs=15)
    arrow(ax, (8.20, 2.36), (3.40, 2.36), color=GOLD)
    ax.text(4.40, 2.50, "proportional", fontsize=10, color=GOLD)
    ax.plot([8.20, 3.10], [1.26, 1.26], color=GOLD, linewidth=1.6, zorder=2)
    arrow(ax, (3.10, 1.26), (3.10, 2.00), color=GOLD)
    ax.text(4.40, 1.40, "integral,  $\\hat u$", fontsize=10, color=GOLD)

    arrow(ax, (3.10, 3.30), (3.10, 2.60), color=HOT)
    ax.text(3.25, 3.06, "$a\\sin\\omega t$", fontsize=11, color=HOT, ha="left")

    ax.plot([2.80, 1.05], [2.30, 2.30], color=INK, linewidth=1.6, zorder=2)
    ax.plot([1.05, 1.05], [2.30, y_f - 0.02], color=INK, linewidth=1.6, zorder=2)
    arrow(ax, (1.05, y_f - 0.02), (1.05, y_f))
    ax.text(0.55, y_f - 0.33, "$u$", fontsize=12, color=HOT)

    arrow(ax, (6.60, 2.92), (6.60, 3.10), color=VIOLET, ls=(0, (3, 2)))
    ax.text(6.60, 2.86, "$u,\\ \\hat u$", fontsize=10.5, color=VIOLET,
            ha="center", va="top")

    ax.text(0.30, 0.28,
            "Compare slide 12. The high-pass, the multiplier, the low-pass and "
            "the $\\sin(\\omega t+\\theta)$ reference are gone,\nreplaced by the one "
            "block above. The $k/s$ integrator has become a PI pair. Everything "
            "else is unchanged.",
            fontsize=11, color=MUTED, linespacing=1.45)
    save(fig, path, pad=0.05)
