"""Figures for the Zap-SA-for-wind-turbines deck.

Numbers are taken from the same sources as the first deck: the plant time
constants and dither settings are Kumar & Rotea (Energies 2022, Tables 2-3) and
Rotea et al., J. Phys. Conf. Ser. 2767 (2024) 032043, Table A1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from windkit import (
    BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, RULE, SOFT, VIOLET,
    arrow, blank, box, figure, save, style,
)


def dictionary(path):
    """The four Zap objects, and what each one is for a turbine."""
    fig, ax = figure(9.8, 5.0)
    blank(ax)
    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(0, 5.2)

    rows = [
        ("$\\theta_n$", "the parameter",
         "torque gain $u = k/I$, or $(u,\\beta)$,\nor a vector of yaw angles $\\gamma$"),
        ("$f_{n+1}(\\theta_n)$", "the noisy update direction",
         "one demodulated gradient estimate\nof $\\ln P$, per dither period"),
        ("$A_{n+1}$", "its Jacobian in $\\theta$",
         "the curvature of $\\ln C_P$, read off the\n"
         "SECOND harmonic of the same record"),
        ("$\\hat A_n \\to A(\\theta)$", "the averaged Jacobian",
         "the Hessian at the operating point,\nwhich erosion moves over months"),
    ]
    y = 4.20
    for sym, role, meaning in rows:
        ax.text(0.10, y + 0.32, sym, fontsize=15, color=PRIMARY, va="center")
        ax.text(1.55, y + 0.32, role, fontsize=11, color=MUTED, va="center")
        box(ax, 4.35, y, 5.65, 0.66, meaning, ec=RULE, fs=10.5, lw=1.2)
        y -= 0.86

    box(ax, 0.0, 0.10, 10.0, 0.72,
        "Everything Zap needs is a function of $\\ln P(t)$ and the probe. No new "
        "sensor, no extra excitation.",
        fc=SOFT, ec=PRIMARY, fs=11, lw=1.4)
    save(fig, path, pad=0.05)


def harmonics(path):
    """Gradient lives at omega, curvature at 2 omega, in the same record."""
    fig, ax = figure(9.4, 4.6)
    f = np.logspace(-3.1, -0.4, 700)
    wd = 0.02 / (2 * np.pi)            # 0.02 rad/s in Hz

    turb = 2.5e-4 * (f / 1e-3) ** (-5.0 / 3.0)
    ax.loglog(f, turb, color=HOT, linewidth=2.2,
              label="turbulence in $\\ln P$  ($\\propto f^{-5/3}$)")

    for mult, col, lab, amp in ((1, PRIMARY, "$\\omega$:  gradient", 3.0e-2),
                                (2, VIOLET, "$2\\omega$:  curvature", 6.0e-3)):
        ax.plot([mult * wd, mult * wd], [1e-8, amp], color=col, linewidth=3.4,
                solid_capstyle="butt", zorder=5)
        ax.plot([mult * wd], [amp], "o", color=col, markersize=8, zorder=6)
        ax.text(mult * wd * (0.80 if mult == 1 else 1.20), amp * 1.9, lab,
                fontsize=11.5, color=col,
                ha="right" if mult == 1 else "left")

    ax.axvline(0.125 / (2 * np.pi), color=FAINT, linewidth=1.4,
               linestyle=(0, (5, 3)), zorder=2)
    ax.text(0.125 / (2 * np.pi) * 1.12, 1.5e-6, "plant bandwidth",
            fontsize=10.5, color=MUTED, rotation=90, va="bottom")

    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(1e-8, 2.0)
    style(ax, "frequency  [Hz]", "power spectral density of  $\\ln P$")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(f[0] * 1.2, 1.6e-8,
            "The probe already writes the curvature into the record. Reading it "
            "costs one more correlation,\nnot one more experiment.",
            fontsize=10.5, color=MUTED, linespacing=1.45)
    save(fig, path)


def demod(path):
    """The dual demodulation and the Zap matrix gain, as a loop."""
    fig, ax = figure(9.8, 4.9)
    blank(ax)
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 5.25)

    box(ax, 0.25, 3.50, 1.60, 0.70, "turbine\n$\\theta \\mapsto P$",
        ec=BLUE, fs=10.5)
    box(ax, 2.25, 3.50, 1.10, 0.70, "$\\ln(\\cdot)$", ec=PRIMARY, fs=13)

    box(ax, 4.10, 4.05, 2.75, 0.70, "$\\times\\, N(t)$,  average",
        ec=VIOLET, fs=10.5)
    box(ax, 4.10, 2.95, 2.75, 0.70,
        "$\\times\\, \\frac{2}{a}\\sin\\omega t$,  average",
        ec=PRIMARY, fs=10.5)

    box(ax, 7.60, 4.05, 2.90, 0.70,
        "$\\hat A_n + \\varepsilon_n[\\hat H_n - \\hat A_n]$", ec=VIOLET, fs=10)
    box(ax, 7.60, 2.95, 2.90, 0.70,
        "$G_n = -[\\varepsilon I + \\hat A^T\\hat A]^{-1}\\hat A$",
        ec=GOLD, fs=9.5)

    box(ax, 3.60, 1.70, 3.25, 0.70,
        "$\\theta_{n+1} = \\theta_n + \\alpha_n G_n \\hat g_n$",
        ec=PRIMARY, fs=11)
    box(ax, 0.25, 1.70, 2.35, 0.70, "$+\\; a\\sin\\omega t$", ec=HOT, fs=11)

    arrow(ax, (1.85, 3.85), (2.25, 3.85))
    arrow(ax, (3.35, 3.85), (4.10, 4.40), rad=-0.18)
    arrow(ax, (3.35, 3.85), (4.10, 3.30), rad=0.18)

    arrow(ax, (6.85, 4.40), (7.60, 4.40), color=VIOLET)
    ax.text(7.22, 4.52, "$\\hat H_n$", fontsize=12, color=VIOLET, ha="center")

    arrow(ax, (9.05, 4.05), (9.05, 3.65), color=VIOLET)

    ax.plot([9.05, 9.05], [2.95, 1.15], color=GOLD, linewidth=1.5, zorder=2)
    ax.plot([9.05, 5.20], [1.15, 1.15], color=GOLD, linewidth=1.5, zorder=2)
    arrow(ax, (5.20, 1.15), (5.20, 1.70), color=GOLD)
    ax.text(9.20, 2.05, "$G_n$", fontsize=12, color=GOLD)

    ax.plot([6.85, 7.15], [3.30, 3.30], color=PRIMARY, linewidth=1.5, zorder=2)
    ax.plot([7.15, 7.15], [3.30, 2.05], color=PRIMARY, linewidth=1.5, zorder=2)
    arrow(ax, (7.15, 2.05), (6.85, 2.05), color=PRIMARY)
    ax.text(7.28, 2.62, "$\\hat g_n$", fontsize=12, color=PRIMARY)

    arrow(ax, (3.60, 2.05), (2.60, 2.05))
    ax.plot([1.05, 1.05], [2.40, 3.42], color=INK, linewidth=1.5, zorder=2)
    arrow(ax, (1.05, 3.42), (1.05, 3.50))

    ax.text(0.25, 0.72, "$N(t) = \\frac{16}{a^2}\\left(\\sin^2\\omega t - "
            "\\frac{1}{2}\\right)$", fontsize=13, color=VIOLET)
    ax.text(0.25, 0.14, "The lower path is the existing algorithm. Everything in "
            "violet and gold is added.", fontsize=10.5, color=MUTED)
    save(fig, path, pad=0.05)


def newton_rate(path):
    """Gradient ascent inherits the plant's curvature; a Newton step does not."""
    fig, ax = figure(9.4, 4.5)
    h = np.linspace(0.25, 3.0, 300)

    ax.plot(h, h, color=HOT, linewidth=2.8,
            label="gradient ascent:  rate $= \\kappa\\,|H|$")
    ax.plot(h, np.ones_like(h), color=PRIMARY, linewidth=2.8,
            label="Newton / Zap step:  rate $= \\kappa$")
    ax.axhline(1.0, color=FAINT, linewidth=1.0, linestyle=(0, (3, 4)), zorder=1)
    ax.plot([1.0], [1.0], "o", color=INK, markersize=9, zorder=6)
    ax.annotate("tuned here", xy=(1.0, 1.0), xytext=(1.30, 1.48),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))

    # the two ways the plant drifts away from the value it was tuned at
    ax.axvspan(0.25, 0.70, color=SOFT, zorder=0)
    ax.axvspan(1.60, 3.00, color="#faf6ec", zorder=0)
    ax.text(0.475, 3.02, "flatter peak\n$|H|$ smaller", ha="center", va="top",
            fontsize=10.5, color=PRIMARY, linespacing=1.4)
    ax.text(2.30, 3.02, "sharper peak\n$|H|$ larger", ha="center", va="top",
            fontsize=10.5, color=GOLD, linespacing=1.4)

    for x, lab, col in ((0.40, "$2.5\\times$ too slow", HOT),
                        (2.50, "$2.5\\times$ too fast", HOT)):
        ax.annotate("", xy=(x, x), xytext=(x, 1.0),
                    arrowprops=dict(arrowstyle="<|-|>", color=HOT, linewidth=1.6))
        ax.text(x + 0.10, (x + 1.0) / 2, lab, fontsize=10.5, color=HOT,
                va="center", ha="left")

    ax.set_xlim(0.25, 3.0)
    ax.set_ylim(0.0, 3.35)
    style(ax, "curvature at the peak  $|H|$   (value it was tuned at = 1)",
          "closed-loop rate   (design value = 1)")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)

def conditioning(path):
    """Where a matrix gain actually pays: a long ridge in two yaw angles."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    g1 = np.linspace(-40, 10, 260)
    g2 = np.linspace(-40, 10, 260)
    G1, G2 = np.meshgrid(g1, g2)

    # a long, tilted ridge, in the spirit of the measured farm power map
    c, s = np.cos(np.radians(35.0)), np.sin(np.radians(35.0))
    X = (G1 + 29) * c + (G2 + 19) * s
    Y = -(G1 + 29) * s + (G2 + 19) * c
    Z = 1.09 - (X / 26.0) ** 2 - (Y / 5.2) ** 2

    def path_of(newton, n=40):
        p = np.array([-6.0, -38.0])
        out = [p.copy()]
        M = np.array([[c, s], [-s, c]])
        H = M.T @ np.diag([2 / 26.0 ** 2, 2 / 5.2 ** 2]) @ M
        Hi = np.linalg.inv(H)
        for _ in range(n):
            x = M @ (p + np.array([29.0, 19.0]))
            grad = -M.T @ np.array([2 * x[0] / 26.0 ** 2, 2 * x[1] / 5.2 ** 2])
            step = (Hi @ grad) * 0.60 if newton else grad * 13.0
            p = p + step
            out.append(p.copy())
        return np.array(out)

    for ax, newton, title, col in ((a1, False, "gradient ascent", HOT),
                                   (a2, True, "matrix gain", PRIMARY)):
        ax.contour(G1, G2, Z, levels=np.linspace(0.2, 1.06, 11),
                   colors=[RULE], linewidths=0.9)
        ax.contour(G1, G2, Z, levels=[1.03], colors=[FAINT], linewidths=1.4)
        w = path_of(newton)
        ax.plot(w[:, 0], w[:, 1], "-o", color=col, markersize=3.2, linewidth=1.6)
        ax.plot([-29], [-19], "*", color=GOLD, markersize=17, zorder=6)
        ax.set_xlim(-40, 10)
        ax.set_ylim(-40, 10)
        style(ax, "$\\gamma_{row\\,1}$  [deg]",
              "$\\gamma_{row\\,2}$  [deg]" if not newton else None, title)

    fig.tight_layout()
    save(fig, path)


def relerr(path):
    """Near the peak the curvature is the better conditioned of the two."""
    fig, ax = figure(9.4, 4.6)
    d = np.logspace(-2.4, -0.1, 400)      # distance from the peak, in u_opt
    # a must be the dither as a FRACTION of u_opt for d to be in the same units:
    # Ciri et al. use a = 0.3 N.m.rpm^-2 against u_opt = 2.2, i.e. 13.6%.
    a = 0.136
    ratio = 8 / a * 2 ** (-5 / 6)         # sigma(Hhat)/sigma(ghat), per u_opt

    rel_g = 1.0 / d
    rel_h = np.full_like(d, ratio)

    ax.loglog(d, rel_g, color=HOT, linewidth=2.8,
              label="gradient  $\\sigma(\\hat g)/|g|$")
    ax.loglog(d, rel_h, color=VIOLET, linewidth=2.8,
              label="curvature  $\\sigma(\\hat H)/|H|$")

    xc = 1 / ratio
    ax.plot([xc], [ratio], "o", color=GOLD, markersize=11, zorder=6)
    ax.annotate(f"they cross at {100*xc:.1f}% from the peak",
                xy=(xc, ratio), xytext=(xc * 1.7, ratio * 3.2),
                fontsize=11, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.4))
    ax.axvspan(d[0], xc, color="#f4eefa", zorder=0)
    ax.text(d[0] * 1.3, 3.6,
            "where the controller\nactually lives",
            fontsize=10.5, color=VIOLET, linespacing=1.4)

    ax.set_xlim(d[0], d[-1])
    ax.set_ylim(0.62, 1.2e3)
    style(ax, "distance from the peak   (fraction of $u_{opt}$)",
          "relative error of the estimate")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(d[0] * 1.3, 0.70,
            "The gradient signal vanishes at the optimum. The curvature signal "
            "does not.", fontsize=10.5, color=MUTED)
    save(fig, path)


def bandwidth(path):
    """The constraint that decides feasibility: 2w must stay in the passband."""
    fig, ax = figure(9.4, 4.6)
    r = np.logspace(-1.3, 1.0, 400)
    ax.semilogx(r, 1 / np.sqrt(1 + r ** 2), color=INK, linewidth=2.4)
    ax.axhline(1 / np.sqrt(2), color=FAINT, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(9.4, 0.735, "$-3$ dB", fontsize=10.5, color=MUTED, ha="right")

    kr = 2 * 0.02 / 0.125
    ax.plot([kr], [1 / np.sqrt(1 + kr ** 2)], "o", color=PRIMARY, markersize=10,
            zorder=6)
    ax.annotate("Kumar & Rotea 2022\n$2\\omega$ at 0.04 rad/s,  gain 0.95",
                xy=(kr, 1 / np.sqrt(1 + kr ** 2)), xytext=(0.062, 0.38),
                fontsize=10.5, color=PRIMARY, linespacing=1.4, ha="left",
                va="top",
                arrowprops=dict(arrowstyle="->", color=PRIMARY, linewidth=1.2))

    for x in (1.6, 2.0):
        y = 1 / np.sqrt(1 + x ** 2)
        ax.plot([x], [y], "o", color=HOT, markersize=10, zorder=6)
        ax.annotate("", xy=(x, y), xytext=(2.6, 0.20),
                    arrowprops=dict(arrowstyle="->", color=HOT, linewidth=1.2))
    ax.text(2.75, 0.17, "wind tunnel, yaw\n$2\\omega$ at 0.8 and 1.0 rad/s,\n"
            "gain 0.53 and 0.45", fontsize=10.5, color=HOT, linespacing=1.4,
            ha="left", va="center")

    ax.axvspan(1.0, r[-1], color="#fbf0ee", zorder=0)
    ax.text(4.2, 0.92, "second harmonic\noutside the passband", ha="center",
            fontsize=10.5, color=HOT, linespacing=1.4)

    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(0, 1.08)
    style(ax, "$2\\omega \\,/\\, \\omega_{plant}$", "plant gain at $2\\omega$")
    ax.text(0.055, 0.035, "The torque-gain problem is already in range. The yaw "
            "problem needs its dither slowed by about $3\\times$.",
            fontsize=10.5, color=MUTED)
    save(fig, path)


def timescales(path):
    """Keep Meyn's recursion; run the gain as the slow variable, not the fast one."""
    fig, ax = figure(9.8, 4.4)
    blank(ax)
    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(0, 4.6)

    W, GAP = 4.75, 0.50
    x2 = W + GAP
    box(ax, 0.0, 2.55, W, 1.80,
        "What Zap assumes\n\n"
        "$\\varepsilon_n/\\alpha_n \\to \\infty$\n\n"
        "Jacobian estimate fast,\nparameter slow.",
        ec=VIOLET, fs=11)

    box(ax, x2, 2.55, W, 1.80,
        "What this plant does\n\n"
        "$\\theta^\\star$ drifts over minutes.\n"
        "$H$ drifts over months, with erosion.\n\n"
        "The Jacobian is the SLOW variable.",
        ec=PRIMARY, fs=11)

    ax.annotate("", xy=(x2 - 0.06, 3.45), xytext=(W + 0.06, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6))

    box(ax, 0.0, 0.20, 2 * W + GAP, 2.05,
        "Keep Meyn's recursion; feed it the demodulated $\\hat H_n$ as innovation:\n"
        r"$\hat A_{n+1} = \hat A_n + \gamma\,(\hat H_n - \hat A_n)$,"
        "   so   " r"$N_H = 2/\gamma - 1$" "\n"
        "Small $\\gamma$ makes $\\hat A$ slow and quiet. Erosion at months against\n"
        "an estimator at a day is still a two-timescale separation,\n"
        "only the other way up.",
        fc="#fbf6e8", ec=GOLD, fs=10.5, lw=1.4)
    save(fig, path, pad=0.05)

def probes(path):
    """Bernoulli probing carries the gradient but not the curvature."""
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 4.9), sharex=True)
    rng = np.random.default_rng(3)
    n = 220
    t = np.arange(n)

    bern = rng.choice([-1.0, 1.0], n)
    gauss = rng.standard_normal(n)

    for col, (xi, name, col_c) in enumerate(
            ((bern, "Bernoulli $\\pm 1$", HOT), (gauss, "Gaussian", PRIMARY))):
        ax = axes[0][col]
        ax.step(t, xi, color=col_c, linewidth=1.4, where="mid")
        ax.set_ylim(-3.6, 3.6)
        style(ax, None, "$\\xi_n$" if col == 0 else None, name)

        ax = axes[1][col]
        ax.step(t, xi ** 2 - 1.0, color=col_c, linewidth=1.4, where="mid")
        ax.axhline(0, color=FAINT, linewidth=1.0)
        ax.set_ylim(-2.2, 9.0)
        style(ax, "dither period $n$",
              "$\\xi_n^2 - 1$" if col == 0 else None)
        if col == 0:
            ax.text(8, 5.4, "identically zero:\nno curvature information",
                    fontsize=11, color=HOT, linespacing=1.4)
        else:
            ax.text(8, 6.4, "$\\mathrm{Var}(\\xi^2) = 2$", fontsize=11.5,
                    color=PRIMARY)

    fig.tight_layout()
    save(fig, path)


def spread(path):
    """Both schemes difference two windows. They weight the spectrum differently."""
    fig, ax = figure(9.4, 4.7)
    f = np.logspace(-3.6, -1.6, 700)
    fp = 0.88e-3                       # spectral peak, Ciri et al. 2019
    fN = 1.0 / (2 * 150.0)             # sign-sequence Nyquist, 3.3e-3 Hz

    turb = (f / fp) ** (-5.0 / 3.0)
    ax.loglog(f, turb, color=HOT, linewidth=2.4)
    ax.text(f[-1] * 0.9, turb[-1] * 2.0, "turbulence in $\\ln P$",
            fontsize=11, color=HOT, ha="right")

    band = f <= fN
    ax.fill_between(f[band], 1e-2, turb[band], color="#f2ecf8", zorder=0)
    ax.annotate("", xy=(f[0] * 1.02, 900), xytext=(fN, 900),
                arrowprops=dict(arrowstyle="<|-|>", color=VIOLET, linewidth=2.0))
    ax.text(np.sqrt(f[0] * fN), 1500, "random signs weight all of this, evenly",
            fontsize=11.5, color=VIOLET, ha="center")

    sd = (fN / fp) ** (-5.0 / 3.0)
    ax.plot([fN, fN], [1e-2, 420], color=PRIMARY, linewidth=3.4,
            solid_capstyle="butt", zorder=5)
    ax.plot([fN], [sd], "o", color=PRIMARY, markersize=10, zorder=6)
    ax.text(fN * 1.15, 420, "alternating signs\nweight only this bin",
            fontsize=11.5, color=PRIMARY, va="center", linespacing=1.4)

    ax.plot([fp], [1.0], "o", color=HOT, markersize=8, zorder=6)
    ax.text(fp * 0.88, 1.5, "spectral peak", fontsize=10.5, color=HOT, ha="right")

    ax.annotate("", xy=(fN * 0.80, sd), xytext=(fN * 0.80, 16 * sd),
                arrowprops=dict(arrowstyle="<|-|>", color=GOLD, linewidth=1.8))
    ax.text(fN * 0.72, 4 * sd, "about $16\\times$\nin noise power,\n"
            "$4\\times$ in scatter", fontsize=11, color=GOLD, ha="center",
            va="bottom", linespacing=1.45)

    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(1e-2, 1e4)
    style(ax, "frequency  [Hz]", "PSD, normalized to the peak")
    save(fig, path)


def multiplex(path):
    """Three ways to separate p parameters in one scalar measurement."""
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.0), sharex=True)
    t = np.linspace(0, 1, 1200)
    rng = np.random.default_rng(11)
    cols = [PRIMARY, VIOLET, GOLD]
    names = ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"]

    ax = axes[0]
    for k, (c, nm) in enumerate(zip(cols, names)):
        ax.plot(t, 0.8*np.sin(2*np.pi*(4 + 5*k)*t) + 2.4*(2 - k),
                color=c, linewidth=1.5)
    style(ax, None, None, "Frequency division: what ESC does")
    ax.text(1.01, 2.4, "all active, one tone each\nexactly orthogonal,\nneeds spectral room",
            transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED,
            va="center", linespacing=1.4)

    ax = axes[1]
    for k, c in enumerate(cols):
        seg = (t >= k/3) & (t < (k+1)/3)
        y = np.zeros_like(t) + 2.4*(2 - k)
        y[seg] += 0.8*np.sin(2*np.pi*14*t[seg])
        ax.plot(t, y, color=c, linewidth=1.5)
    style(ax, None, None, "Time division: one parameter at a time")
    ax.text(1.01, 2.4, "exactly orthogonal,\nno spectral room needed,\ncosts $p\\times$ the clock",
            transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED,
            va="center", linespacing=1.4)

    ax = axes[2]
    n = 40
    for k, c in enumerate(cols):
        s = rng.choice([-1.0, 1.0], n)
        ax.step(np.linspace(0, 1, n), 0.8*s + 2.4*(2 - k), color=c,
                linewidth=1.4, where="mid")
    style(ax, "time", None, "Code division: what SPSA does")
    ax.text(1.01, 2.4, "all active, random signs\northogonal only on average,\n"
            "leakage shows up as variance", transform=ax.get_yaxis_transform(),
            fontsize=10, color=MUTED, va="center", linespacing=1.4)

    for ax in axes:
        ax.set_yticks([0, 2.4, 4.8])
        ax.set_yticklabels(["$\\theta_3$", "$\\theta_2$", "$\\theta_1$"])
        ax.set_ylim(-1.2, 6.0)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    fig.tight_layout()
    save(fig, path, pad=0.3)


def sidon(path):
    """The frequency plan for a Hessian, and how fast it outgrows the band."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})

    tones = [4, 5, 7]
    sums = sorted({a + b for i, a in enumerate(tones) for b in tones[i:]})
    diffs = sorted({a - b for a in tones for b in tones if a > b})

    for xs, col, y, lab in ((diffs, VIOLET, 0.55, "$\\omega_i-\\omega_j$"),
                            (tones, PRIMARY, 1.55, "$\\omega_i$  (gradient)"),
                            (sums, VIOLET, 1.05, "$\\omega_i+\\omega_j$,  $2\\omega_i$")):
        a1.vlines(xs, y - 0.30, y + 0.30, color=col, linewidth=3.0)
        a1.text(15.4, y, "  " + lab, fontsize=10.5, color=col, va="center")

    a1.axvspan(0, 3.6, color="#fbf0ee", zorder=0)
    a1.text(1.8, 0.10, "turbulence\nlives here", ha="center", fontsize=10,
            color=HOT, linespacing=1.35)
    a1.axvline(14, color=FAINT, linewidth=1.6, linestyle=(0, (5, 3)))
    a1.text(13.6, 2.05, "$\\omega_{plant}$", fontsize=11, color=MUTED, ha="right")

    a1.set_xlim(0, 15.2)
    a1.set_ylim(0, 2.25)
    a1.set_yticks([])
    style(a1, "frequency, units of $2\\pi/T$", None,
          "$p=3$:  twelve channels, all distinct")

    p = np.arange(1, 7)
    tmin = np.array([101, 302, 704, 1206, 1910, 2915])
    a2.plot(p, tmin / 60.0, "-o", color=PRIMARY, linewidth=2.4, markersize=8)
    a2.axhline(314 / 60.0, color=HOT, linewidth=1.8, linestyle=(0, (5, 3)))
    a2.text(6.2, 314/60.0 + 1.8, "published dither period,  314 s",
            fontsize=10.5, color=HOT, ha="right")
    a2.set_xlim(0.7, 6.3)
    a2.set_ylim(0, 54)
    style(a2, "parameters $p$", "minimum averaging window  [min]",
          "what the frequency plan costs")
    for xi, yi in zip(p, tmin / 60.0):
        if xi in (3, 6):
            a2.annotate(f"{yi:.0f} min", xy=(xi, yi), xytext=(xi - 0.12, yi + 3.0),
                        fontsize=10.5, color=PRIMARY, ha="right")

    fig.tight_layout()
    save(fig, path)
