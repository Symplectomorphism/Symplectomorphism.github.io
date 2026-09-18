"""Act 0 figures: what the machine is doing and why the problem is hard."""

from __future__ import annotations

import numpy as np

from windkit import (
    BLUE, CP_MAX, FAINT, GOLD, HOT, INK, LAMBDA_OPT, MUTED, PRIMARY, R, RULE,
    SOFT, VIOLET, arrow, blank, box, cp, figure, k_from_lambda, save, style,
)


def regions(path):
    """Power curve: the only region with an extremum to seek is region 2."""
    fig, ax = figure(9.2, 4.3)
    v = np.linspace(0, 25, 600)
    cut_in, rated, cut_out = 3.0, 11.4, 25.0
    p = np.where(v < cut_in, 0.0,
                 np.where(v < rated, 5.0 * (v / rated) ** 3, 5.0))
    p = np.where(v >= cut_out, 0.0, p)

    ax.axvspan(cut_in, rated, color=SOFT, zorder=0)
    ax.axvspan(rated, cut_out, color="#faf6ec", zorder=0)
    ax.plot(v, p, color=PRIMARY, linewidth=2.8, zorder=3)

    ax.text((cut_in + rated) / 2, 4.55, "Region 2", ha="center", fontsize=13,
            color=PRIMARY, fontweight="bold")
    ax.text((cut_in + rated) / 2, 4.28, "maximize $C_P$\nactuator: generator torque",
            ha="center", va="top", fontsize=10.5, color=MUTED, linespacing=1.35)
    ax.text((rated + cut_out) / 2, 4.55, "Region 3", ha="center", fontsize=13,
            color=GOLD, fontweight="bold")
    ax.text((rated + cut_out) / 2, 4.28, "hold rated power\nactuator: blade pitch",
            ha="center", va="top", fontsize=10.5, color=MUTED, linespacing=1.35)

    for x, lab, ha, dx in ((cut_in, "cut-in\n3 m/s", "right", -0.25),
                           (rated, "rated\n11.4 m/s", "left", 0.25)):
        ax.axvline(x, color=FAINT, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(x + dx, 0.55, lab, ha=ha, va="top", fontsize=9.5, color=MUTED,
                linespacing=1.3)

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 5.6)
    style(ax, "wind speed $V$  [m/s]", "electrical power  [MW]")
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    save(fig, path, pad=0.12)


def cp_curve(path):
    """The hill: flat at the top, and it moves."""
    fig, ax = figure(9.2, 4.6)
    lam = np.linspace(2, 13, 800)

    ax.plot(lam, cp(lam), color=PRIMARY, linewidth=2.8, zorder=4,
            label="design surface")
    ax.plot(lam, cp(lam, lam_opt=LAMBDA_OPT + 0.2, cp_max=CP_MAX * 0.955),
            color=HOT, linewidth=2.2, linestyle=(0, (6, 3)), zorder=3,
            label="after blade erosion / soiling")

    ax.plot([LAMBDA_OPT], [CP_MAX], "o", color=PRIMARY, markersize=8, zorder=6)
    ax.plot([LAMBDA_OPT + 0.2], [CP_MAX * 0.955], "o", color=HOT, markersize=7,
            zorder=6)

    # the flat-peak band: how little Cp changes over a wide lambda window
    band = cp(lam) >= 0.99 * CP_MAX
    lo, hi = lam[band][0], lam[band][-1]
    ax.axvspan(lo, hi, color=SOFT, zorder=0)
    mid = (lo + hi) / 2
    ax.annotate("", xy=(lo, 0.455), xytext=(hi, 0.455),
                arrowprops=dict(arrowstyle="<|-|>", color=PRIMARY, linewidth=1.6))
    ax.annotate("", xy=(mid, 0.435), xytext=(mid, 0.315),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, linewidth=1.2))
    ax.text(mid, 0.295,
            f"the peak is flat:\n$\\Delta\\lambda = {hi-lo:.1f}$ costs only 1% of $C_P$",
            ha="center", va="top", fontsize=11, color=PRIMARY, linespacing=1.4)

    ax.annotate(f"$\\lambda^\\star$ moves ~0.2",
                xy=(LAMBDA_OPT + 0.2, CP_MAX * 0.955), xytext=(10.4, 0.40),
                fontsize=11, color=HOT,
                arrowprops=dict(arrowstyle="->", color=HOT, linewidth=1.3))

    ax.set_xlim(2, 13)
    ax.set_ylim(0, 0.56)
    style(ax, "tip-speed ratio  $\\lambda = R\\Omega/V$", "power coefficient  $C_P$")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def blind(path):
    """What you can measure, and what the objective is written in."""
    fig, ax = figure(9.2, 5.1)
    blank(ax)
    ax.set_xlim(-0.05, 10.1)
    ax.set_ylim(0, 5.45)

    box(ax, 0.02, 3.05, 4.62, 2.30,
        "You measure\n\n"
        r"$\Omega$  rotor speed:  encoder, exact" "\n"
        r"$\tau_g$  generator torque:  you command it" "\n"
        r"$P_g$  generator power:  a wattmeter",
        fc="#ffffff", ec=PRIMARY, fs=10.5)

    box(ax, 5.42, 3.05, 4.62, 2.30,
        "The curve is drawn in\n\n"
        r"$\lambda = R\Omega/V$      horizontal axis" "\n"
        r"$C_P = P/(\frac{1}{2}\rho A V^3)$   vertical axis" "\n"
        r"$\partial C_P/\partial\lambda$        the slope, from both" "\n\n"
        r"all three need $V$",
        fc="#fbf0ee", ec=HOT, fs=10.5)

    ax.text(5.0, 4.20, "$\\neq$", ha="center", va="center", fontsize=30,
            color=HOT)

    box(ax, 0.02, 0.28, 10.02, 2.35,
        "So you could try to build the curve directly, from $\\hat V$. Both axes carry the same\n"
        "unknown, so an error there does not scatter the points: it rescales the curve and\n"
        "carries the peak along, "
        r"$\hat\lambda^\star = \lambda^\star/c$." "  "
        "Resolving the $0.2$ that erosion produces\n"
        "would need $\\hat V$ good to $3\\%$. Extremum seeking does not go this way at all.",
        fc=SOFT, ec=PRIMARY, fs=10.5, lw=1.3)
    save(fig, path, pad=0.05)


def ray(path):
    """The kOmega^2 law has no speed setpoint: its equilibrium is a ray."""
    fig, ax = figure(9.2, 4.6)
    v = np.linspace(3, 12, 200)

    for lam, col, lw, lab in (
        (8.8, FAINT, 1.7, "$k$ too small $\\rightarrow$ $\\lambda_{eq}=8.8$"),
        (LAMBDA_OPT, PRIMARY, 3.0, "$k=k_{opt}$ $\\rightarrow$ $\\lambda_{eq}=7.5$"),
        (6.2, FAINT, 1.7, "$k$ too large $\\rightarrow$ $\\lambda_{eq}=6.2$"),
    ):
        omega = lam * v / R * 60 / (2 * np.pi)   # rotor rpm
        ax.plot(v, omega, color=col, linewidth=lw, zorder=3)
        ax.text(12.5, omega[-1], "  " + lab, fontsize=10.5, color=col,
                va="center", ha="left")

    for vv in (4.0, 8.0, 12.0):
        om = LAMBDA_OPT * vv / R * 60 / (2 * np.pi)
        ax.plot([vv], [om], "o", color=PRIMARY, markersize=8, zorder=5)
        ax.annotate(f"{om:.1f} rpm", xy=(vv, om), xytext=(vv - 0.15, om + 1.15),
                    fontsize=10, color=PRIMARY, ha="center")

    ax.set_xlim(3, 12.5)
    ax.set_ylim(0, 15)
    style(ax, "wind speed $V$  [m/s]", "rotor speed $\\Omega$  [rpm]")
    ax.text(12.2, 1.0,
            "the equilibrium is a ray,\nnot a point: there is no\nrotor speed to regulate to",
            ha="right", va="bottom", fontsize=10.5, color=MUTED, linespacing=1.35)
    save(fig, path)


def rest(path):
    """Where the rotor rests: a fixed curve meeting a line you choose."""
    from scipy.optimize import brentq
    fig, ax = figure(9.2, 4.8)
    lam = np.linspace(3.0, 12.5, 800)
    phi = lambda l: cp(l) / l ** 3
    p0 = phi(LAMBDA_OPT)

    ax.plot(lam, phi(lam) / p0, color=PRIMARY, linewidth=3.0, zorder=4)
    ax.text(11.9, phi(11.9) / p0 + 0.06, "$C_P(\\lambda)\\,/\\,\\lambda^3$",
            fontsize=13, color=PRIMARY, ha="right")

    for f, col, lab in ((0.75, FAINT, "$k = 0.75\\,k_{opt}$"),
                        (1.00, HOT, "$k = k_{opt}$"),
                        (1.35, FAINT, "$k = 1.35\\,k_{opt}$")):
        r = brentq(lambda l: phi(l) / p0 - f, 3.0, 12.5)
        ax.axhline(f, color=col, linewidth=1.8,
                   linestyle="-" if f == 1.0 else (0, (5, 3)), zorder=2)
        ax.plot([r], [f], "o", color=col, markersize=11, zorder=6)
        ax.plot([r, r], [0, f], color=col, linewidth=1.0,
                linestyle=(0, (2, 3)), zorder=1)
        ax.text(3.15, f + 0.045, lab, fontsize=11, color=col)
        ax.text(r + 0.20, f + 0.055, f"$\\lambda = {r:.2f}$", fontsize=11.5,
                color=col, ha="left", va="bottom")

    ax.set_xlim(3.0, 12.5)
    ax.set_ylim(0, 2.35)
    style(ax, "tip-speed ratio  $\\lambda$",
          "$C_P/\\lambda^3$,  in units of its value at $\\lambda^\\star$")
    ax.text(6.6, 2.05,
            "the curve is fixed by the blades;\nthe line is the number you choose",
            fontsize=11.5, color=MUTED, linespacing=1.4)
    ax.text(3.15, 0.10, "the curve falls here, so every crossing is attracting",
            fontsize=10.5, color=PRIMARY)
    save(fig, path, pad=0.16)
