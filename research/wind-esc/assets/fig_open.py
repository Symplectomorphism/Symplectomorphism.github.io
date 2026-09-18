"""Act 2 figures: the five open questions."""

from __future__ import annotations

import numpy as np

from windkit import (
    BLUE, CP_MAX, FAINT, GOLD, HOT, INK, LAMBDA_OPT, MUTED, PRIMARY, RULE,
    SOFT, VIOLET, arrow, blank, box, cp, figure, save, style,
)


def estimand(path):
    """Q1. Three different 'optima', and the gap between them beats the signal."""
    fig, ax = figure(9.2, 4.6)
    lam = np.linspace(5.4, 10.4, 800)
    ax.plot(lam, cp(lam), color=PRIMARY, linewidth=2.8, zorder=4)

    marks = [
        (7.50, PRIMARY, "(a)  $\\arg\\max C_P(\\lambda)$\nsteady, from a BEM curve\n"
                        "what every paper scores against"),
        (7.85, HOT, "(c)  where the 2019 LES actually\nconverged, both algorithms,\n"
                    "every wind speed"),
    ]
    for lm, col, lab in marks:
        ax.plot([lm], [cp(lm)], "o", color=col, markersize=9, zorder=6)
        ax.plot([lm, lm], [0.30, cp(lm)], color=col, linewidth=1.2,
                linestyle=(0, (3, 3)), zorder=2)

    ax.text(6.55, 0.335, marks[0][2], fontsize=10.5, color=PRIMARY,
            ha="center", va="top", linespacing=1.35)
    ax.text(9.35, 0.335, marks[1][2], fontsize=10.5, color=HOT,
            ha="center", va="top", linespacing=1.35)

    # both lengths on a common left edge, so the comparison is visual
    y = 0.534
    ax.annotate("", xy=(7.50, y), xytext=(7.85, y),
                arrowprops=dict(arrowstyle="<|-|>", color=HOT, linewidth=2.0))
    ax.text(7.93, y, "the gap:  $0.35$", ha="left", va="center", fontsize=11.5,
            color=HOT, fontweight="bold")

    y2 = 0.508
    ax.annotate("", xy=(7.50, y2), xytext=(7.70, y2),
                arrowprops=dict(arrowstyle="<|-|>", color=GOLD, linewidth=2.0))
    ax.text(7.93, y2, "what erosion moves $\\lambda^\\star$ by:  $0.2$",
            ha="left", va="center", fontsize=11.5, color=GOLD, fontweight="bold")
    ax.plot([7.50, 7.50], [0.500, 0.542], color=FAINT, linewidth=1.0, zorder=1)

    ax.set_xlim(5.4, 10.4)
    ax.set_ylim(0.20, 0.558)
    style(ax, "tip-speed ratio  $\\lambda$", "power coefficient  $C_P$")
    save(fig, path)


def channels(path):
    """Q1/Q3. What fraction of the log-power signal is actually the turbine."""
    fig, ax = figure(9.2, 4.4)
    rng = np.random.default_rng(4)
    t = np.linspace(0, 600, 3000)

    # turbulence: a low-pass filtered noise with ~30 s correlation time
    w = rng.standard_normal(t.size)
    kern = np.exp(-np.linspace(0, 5, 260))
    kern /= kern.sum()
    turb = np.convolve(w, kern, mode="same")
    turb = 0.10 * turb / turb.std()          # 10% TI  ->  sigma(ln V) ~ 0.10
    wind = 3.0 * turb                         # 3 ln V

    dither = 0.02 * np.sin(2 * np.pi * t / 150.0)   # what the turbine contributes

    ax.plot(t, wind, color=HOT, linewidth=1.5, zorder=3,
            label="$3\\ln V(t)$,  the wind,  $\\sigma \\approx 0.30$")
    ax.plot(t, dither, color=PRIMARY, linewidth=2.2, zorder=4,
            label="$\\ln C_P$ response to the probe,  $\\sigma \\approx 0.02$")

    ax.set_xlim(0, 600)
    ax.set_ylim(-1.55, 1.05)
    style(ax, "time  [s]", "contribution to  $\\ln P$  [nepers]")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11)
    for tt in leg.get_texts():
        tt.set_color(INK)
    ax.text(12, -1.44,
            "$\\ln P = \\ln(\\frac{1}{2}\\rho A) + 3\\ln V + \\ln C_P(\\lambda)$ "
            "The middle term is $u$-independent, so it vanishes from the\n"
            "gradient exactly. It does not vanish from any finite-time estimate of it.",
            fontsize=11, color=MUTED, linespacing=1.4)
    save(fig, path)


def biasvar(path):
    """Dither amplitude: bias does not average down, scatter does."""
    fig, ax = figure(9.4, 4.7)
    x = np.linspace(0.04, 1.0, 400)          # dither, as a fraction of k_opt
    N = 3                                     # N_eff = 2/g - 1 from their own
                                              # constant-gain tuning, not a run length
    # constants computed from the C_P surface and the 2019 dither settings:
    #   |bias| = 0.682 x^2   from the C_P curve
    #   scatter = 0.033 / (x sqrt(N))   an UPPER BOUND calibrated from the
    #   run-to-run spread in Ciri et al. 2019 Table 4; predicting it from the
    #   turbulence overshoots by 24-40x and is the open problem
    bias = 0.6817 * x ** 2
    scat = 0.0330 / (x * np.sqrt(N))
    rms = np.sqrt(bias ** 2 + scat ** 2)

    ax.loglog(x, bias, color=GOLD, linewidth=2.2, linestyle=(0, (5, 3)),
              label="bias  $\\propto a^2$,  does not average down")
    ax.loglog(x, scat, color=HOT, linewidth=2.2, linestyle=(0, (5, 3)),
              label="scatter  $\\propto 1/(a\\sqrt{N})$,  does")
    ax.loglog(x, rms, color=PRIMARY, linewidth=3.0, label="total error in $\\lambda$")

    i = int(np.argmin(rms))
    ax.plot([x[i]], [rms[i]], "o", color=PRIMARY, markersize=11, zorder=6)
    ax.annotate(f"best at {100*x[i]:.0f}% of $k_{{opt}}$", xy=(x[i], rms[i]),
                xytext=(x[i] * 2.05, rms[i] * 2.6), fontsize=11, color=PRIMARY,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=PRIMARY, linewidth=1.3))

    ax.axvline(0.136, color=BLUE, linewidth=1.6, linestyle=(0, (2, 3)), zorder=2)
    ax.annotate("published:  $a=0.3$,  $13.6\\%$\nscatter beats bias $11\\times$",
                xy=(0.136, 0.16), xytext=(0.044, 1.15), fontsize=10.5, color=BLUE,
                linespacing=1.4, ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.2))

    ax.set_xlim(0.04, 1.0)
    ax.set_ylim(2e-2, 3.0)
    style(ax, "dither amplitude, as a fraction of $k_{opt}$",
          "error in $\\lambda$, at the tuned $N_{eff} = 3$")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def stability(path):
    """Q2. A first-order ODE cannot go unstable. The LES did."""
    fig, (a1, a2) = __import__("matplotlib.pyplot", fromlist=["x"]).subplots(
        1, 2, figsize=(9.6, 4.1))

    # left: the 2017 model - a pole on the negative real axis, always
    a1.axhline(0, color=RULE, linewidth=1.2)
    a1.axvline(0, color=RULE, linewidth=1.2)
    for g, col, ms in ((0.35, FAINT, 8), (1.0, PRIMARY, 10), (3.4, PRIMARY, 12)):
        a1.plot([-g], [0], "x", color=col, markersize=ms, markeredgewidth=3)
    a1.annotate("", xy=(-3.55, 0.42), xytext=(-0.35, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, linewidth=1.6))
    a1.text(-1.95, 0.52, "faster as $V$ rises", ha="center", fontsize=11,
            color=PRIMARY)
    a1.axvspan(0, 1.2, color="#fbf0ee", zorder=0)
    a1.text(0.6, -0.72, "unstable\nhalf-plane", ha="center", fontsize=10.5,
            color=HOT, linespacing=1.3)
    a1.set_xlim(-4.2, 1.2)
    a1.set_ylim(-1.0, 1.0)
    a1.set_yticks([])
    style(a1, "Re $s$", None, "2017 model:  $\\dot{\\tilde u} = -\\omega_c\\tilde u$")
    a1.text(-4.05, 0.80, "one pole, always in the left half-plane\nfor any $\\kappa>0$",
            fontsize=10.5, color=MUTED, linespacing=1.35)

    # right: what a sampled loop actually does
    v = np.linspace(4, 13, 300)
    a2.plot(v, (v / 8.0) ** 3, color=HOT, linewidth=2.8, label="ESC  (gain $\\propto V^3$)")
    a2.plot(v, np.ones_like(v), color=PRIMARY, linewidth=2.8, label="LP-ESC")
    a2.axhline(2.6, color=HOT, linewidth=1.8, linestyle=(0, (5, 3)))
    a2.text(4.2, 2.75, "sampled-data stability boundary", fontsize=10.5, color=HOT)
    a2.fill_between(v, 2.6, 5.0, color="#fbf0ee", zorder=0)
    a2.plot([12], [(12 / 8) ** 3], "o", color=HOT, markersize=10, zorder=6)
    a2.annotate("LES: unstable at 12 m/s", xy=(12, (12 / 8) ** 3),
                xytext=(7.4, 4.35), fontsize=11, color=HOT,
                arrowprops=dict(arrowstyle="->", color=HOT, lw=1.3))
    a2.set_ylim(0, 5.0)
    style(a2, "wind speed  [m/s]", "loop gain  (8 m/s = 1)",
          "one update per 150 s vs. a 6 s rotor")
    leg = a2.legend(loc="lower right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    save(fig, path)


def identify(path):
    """Q3. Joint (V, C_Q) estimation has exactly one blind direction."""
    fig, (a1, a2) = __import__("matplotlib.pyplot", fromlist=["x"]).subplots(
        1, 2, figsize=(9.6, 4.2))
    lam = np.linspace(3, 12, 500)
    c = 1.12

    a1.plot(lam, cp(lam), color=PRIMARY, linewidth=2.6, label="true  $C_P(\\lambda)$")
    a1.plot(lam, cp(c * lam) / c ** 3, color=VIOLET, linewidth=2.2,
            linestyle=(0, (6, 3)),
            label="$\\tilde C_P(x) = C_P(cx)/c^3$,  $c=1.12$")
    for f, col in ((1.0, PRIMARY), (1 / c, VIOLET)):
        lm = LAMBDA_OPT * f
        y = cp(lm) if f == 1.0 else cp(c * lm) / c ** 3
        a1.plot([lm], [y], "o", color=col, markersize=8, zorder=6)
    a1.set_xlim(3, 12)
    a1.set_ylim(0, 0.62)
    style(a1, "tip-speed ratio", "$C_P$", "Two different curves...")
    leg = a1.legend(loc="upper right", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)

    om = np.linspace(0.4, 1.6, 300)
    taero = om ** 2
    a2.plot(om, taero, color=PRIMARY, linewidth=3.0, label="from $(V, C_Q)$")
    a2.plot(om, taero, color=VIOLET, linewidth=1.6, linestyle=(0, (4, 4)),
            label="from $(cV, \\tilde C_Q)$")
    a2.set_xlim(0.4, 1.6)
    a2.set_ylim(0, 2.8)
    style(a2, "rotor speed  $\\Omega$", "$\\tau_{aero} = I\\dot\\Omega + \\tau_g$",
          "...produce identical data")
    leg = a2.legend(loc="upper left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)
    a2.text(0.47, 0.22,
            "$\\lambda^\\star$ is identifiable only up to the\n"
            "wind-speed scale $c$. Nothing else\nabout the curve is ambiguous.",
            fontsize=10.5, color=MUTED, linespacing=1.4)

    fig.tight_layout()
    save(fig, path)


def hessian(path):
    """Q4. Two routes to the optimal asymptotic covariance."""
    fig, ax = figure(9.6, 5.3)
    blank(ax)
    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(0, 5.45)

    box(ax, 0.0, 4.25, 10.0, 1.05,
        "In stochastic approximation the asymptotic covariance is "
        "$\\Sigma^{PR} = A^{-1}\\Sigma_W A^{-T}$,\n"
        "and for a gradient method $A = \\nabla^2\\Gamma(\\theta^\\star)$.\n"
        "The Hessian is in the answer whether or not you estimate it.",
        fc=SOFT, ec=PRIMARY, fs=10.5, lw=1.4)

    box(ax, 0.0, 1.30, 4.80, 2.70, "", ec=VIOLET)
    ax.text(2.40, 3.76, "Zap SA  (Meyn)", ha="center", va="top", fontsize=11.5,
            color=VIOLET, fontweight="bold")
    ax.text(2.40, 3.18,
            "$\\hat A_{n+1} = \\hat A_n + \\varepsilon_{n+1}[A_{n+1}-\\hat A_n]$\n"
            "$\\theta_{n+1} = \\theta_n + \\alpha_{n+1}\\,G_{n+1}f_{n+1}$\n"
            "$G_{n+1} = -[\\varepsilon I + \\hat A^T \\hat A]^{-1}\\hat A$",
            ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.55)
    ax.text(2.40, 1.92, "Estimate the Jacobian on a fast\ntimescale, invert it, "
            "use it as a gain.", ha="center", va="top", fontsize=10,
            color=MUTED, linespacing=1.4)

    box(ax, 5.20, 1.30, 4.80, 2.70, "", ec=PRIMARY)
    ax.text(7.60, 3.76, "Polyak-Ruppert averaging", ha="center", va="top",
            fontsize=11.5, color=PRIMARY, fontweight="bold")
    ax.text(7.60, 3.00,
            "$\\theta^{PR}_N \\;=\\; \\frac{1}{N-N_0}\\;"
            "(\\theta_{N_0} + \\cdots + \\theta_N)$",
            ha="center", va="top", fontsize=11.5, color=INK)
    ax.text(7.60, 2.10, "Average the iterates.\nNo curvature estimate anywhere.",
            ha="center", va="top", fontsize=10, color=MUTED, linespacing=1.4)

    arrow(ax, (2.40, 1.30), (2.40, 1.02), color=VIOLET, lw=2.0)
    arrow(ax, (7.60, 1.30), (7.60, 1.02), color=PRIMARY, lw=2.0)

    box(ax, 0.0, 0.10, 10.0, 0.90,
        "Both attain the same optimal covariance. Meyn: “Given the simplicity of "
        "the recursions\ndefining $\\theta^{PJR}$, this is probably the first "
        "algorithm to try in most applications.”",
        fc="#fbf6e8", ec=GOLD, fs=10.5, lw=1.4)
    save(fig, path, pad=0.05)


def bound(path):
    """Q5. There is no benchmark, so nobody knows if this is a research problem."""
    fig, ax = figure(9.2, 4.5)
    t = np.logspace(1.7, 3.4, 300)

    ax.loglog(t, 3.2 / np.sqrt(t), color=HOT, linewidth=2.4)
    ax.loglog(t, 1.35 / np.sqrt(t), color=BLUE, linewidth=2.4)
    ax.loglog(t, 0.34 / np.sqrt(t), color=GOLD, linewidth=2.6,
              linestyle=(0, (6, 4)))
    for c, col, lab in ((3.2, HOT, "LP-ESC"), (1.35, BLUE, "LP-PIESC"),
                        (0.34, GOLD, "best possible?")):
        ax.text(t[-1] * 1.04, c / np.sqrt(t[-1]), lab, color=col, fontsize=11,
                va="center", ha="left")

    ax.fill_between(t, 0.34 / np.sqrt(t), 1.35 / np.sqrt(t), color="#fbf6e8",
                    zorder=0)
    ax.annotate("", xy=(430, 0.34 / np.sqrt(430)), xytext=(430, 1.35 / np.sqrt(430)),
                arrowprops=dict(arrowstyle="<|-|>", color=GOLD, linewidth=1.8))
    ax.text(470, 0.036, "?", fontsize=22, color=GOLD, fontweight="bold",
            va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.004, 2.6)
    style(ax, "observation time  $T$  [s]",
          "scatter in $\\hat\\lambda^\\star$   $\\sigma_\\lambda$")

    ax.text(t[0] * 1.15, 2.15,
            "Is this gap a factor of 1.2, or a factor of 20?\n"
            "Nobody has written the bound down, so there is no way to tell\n"
            "whether this is a research programme or an engineering one.",
            fontsize=11.5, color=GOLD, linespacing=1.45, va="top", ha="left")

    ax.text(t[0] * 1.15, 0.0050, "Schematic: the solid curves have the shape of "
            "the published claims, not measured values.",
            fontsize=10, color=FAINT, ha="left")
    save(fig, path)
