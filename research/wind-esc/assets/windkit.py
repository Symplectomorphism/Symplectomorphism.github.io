"""Shared style and turbine model for the wind-ESC deck figures.

The palette matches `deck.scss` exactly, so a figure dropped on a slide reads
as part of the same document rather than as an imported plot.

Turbine constants are the NREL 5-MW reference machine as used throughout the
UTD extremum-seeking line (Jonkman et al. 2009):

    R = 63 m (radius; note Ciri et al. 2019 Table 1 labels 126 m as "R")
    I = 35.44e6 kg m^2,  N = 97,  rated 5 MW,  cut-in 3, rated 11.4 m/s

lambda_opt and Cp_max differ slightly between papers (7.5/0.49 in the 2019
LES, 7.55/0.48 in the 2022 Energies paper); we use the 2019 pair, which is
what the LES results in this deck are quoted against.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ palette

INK = "#1c2b26"
MUTED = "#6b7a74"
FAINT = "#97a49e"
RULE = "#dce3dd"
PRIMARY = "#175b48"
SOFT = "#eef3f0"
HOT = "#c0392b"
GOLD = "#b8860b"
BLUE = "#2f6690"
VIOLET = "#6b4c9a"

# ------------------------------------------------------------------ turbine

R = 63.0
INERTIA = 35.444067e6
RHO = 1.225
LAMBDA_OPT = 7.5
CP_MAX = 0.49


def cp(lam, beta=0.0, lam_opt=LAMBDA_OPT, cp_max=CP_MAX):
    """Heier's analytic Cp surface, affinely remapped onto (lam_opt, cp_max).

    The point of using Heier rather than a parabola is the asymmetry: the
    curve is steep on the low-lambda (stall) side and gentle on the high side,
    which is what makes the third derivative at the peak non-zero.  That
    non-zero J''' is exactly what puts a finite-difference estimator's fixed
    point off the true optimum, so a symmetric stand-in would quietly erase
    the effect this deck is about.
    """
    lam = np.asarray(lam, dtype=float)
    # Heier peaks near lambda = 8.12 with Cp = 0.4805; rescale onto ours.
    scale = 8.1226 / lam_opt
    l = np.clip(lam * scale, 1e-6, None)
    li = 1.0 / (1.0 / (l + 0.08 * beta) - 0.035 / (beta**3 + 1.0))
    out = 0.5176 * (116.0 / li - 0.4 * beta - 5.0) * np.exp(-21.0 / li) + 0.0068 * l
    return np.maximum(out, 0.0) * (cp_max / 0.48048)


def cq(lam, **kw):
    """Torque coefficient, C_Q = C_P / lambda."""
    lam = np.asarray(lam, dtype=float)
    return cp(lam, **kw) / np.clip(lam, 1e-9, None)


def k_from_lambda(lam):
    """Torque gain whose kOmega^2 equilibrium sits at `lam`.

    Equilibrium of I Omega-dot = T_aero - k Omega^2 is C_P(l)/l^3 = 2k/(rho pi R^5).
    """
    return 0.5 * RHO * np.pi * R**5 * cp(lam) / np.asarray(lam, dtype=float) ** 3


# ------------------------------------------------------------------ styling

def figure(w=9.2, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def style(ax, xlabel=None, ylabel=None, title=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=11, length=4, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=12.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=12.5)
    if title:
        ax.set_title(title, color=PRIMARY, fontsize=13.5, fontweight="bold", loc="left")
    ax.grid(True, color=RULE, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def blank(ax):
    """Turn an axes into a bare drawing canvas for block diagrams."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=PRIMARY, tc=INK, fs=11, lw=1.6, r=0.02):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.35)


def arrow(ax, p, q, color=INK, lw=1.5, ls="-", rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
            color=color, linestyle=ls, zorder=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def save(fig, path, pad=0.12):
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=pad,
                transparent=True)
    plt.close(fig)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans Pro", "DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})
