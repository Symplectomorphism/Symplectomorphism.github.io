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
    """What the Kaimal spectrum is, and where the dither sits in it."""
    import matplotlib.pyplot as plt
    f = np.logspace(-4.2, -0.7, 800)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.0))

    a1.loglog(f, S_eps(f), color=PRIMARY, linewidth=2.8)
    a1.axvline(U / (6 * L), color=FAINT, linewidth=1.4, linestyle=(0, (4, 3)))
    a1.text(U / (6 * L) * 1.15, 3e-3, f"corner\n$U/6L$", fontsize=10,
            color=MUTED, linespacing=1.3)
    a1.text(2.0e-4, 2.0, "flat:  $S \\to \\mathrm{TI}^2\\,4L/U$", fontsize=10.5,
            color=PRIMARY)
    a1.text(2.2e-2, 2.0e-2, "$\\propto f^{-5/3}$", fontsize=11.5, color=PRIMARY,
            ha="right")
    for T, lab in ((150.0, "$T=150$ s"), (600.0, "$T=600$ s")):
        a1.plot([1 / T], [S_eps(1 / T)], "o", color=GOLD, markersize=10, zorder=6)
        a1.annotate(lab, xy=(1 / T, S_eps(1 / T)), xytext=(1 / T, S_eps(1 / T) * 7),
                    fontsize=10, color=GOLD, ha="center",
                    arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=1.1))
    a1.set_ylim(2e-3, 30)
    style(a1, "frequency $f$  [Hz]", "$S_\\varepsilon(f)$   [1/Hz]",
          "the density: how much variance per Hz")

    a2.semilogx(f, f * S_eps(f), color=HOT, linewidth=2.8)
    pk = f[int(np.argmax(f * S_eps(f)))]
    a2.axvline(pk, color=HOT, linewidth=1.2, linestyle=(0, (3, 3)))
    a2.text(pk * 1.2, 0.85 * np.max(f * S_eps(f)),
            f"most energy sits here,\n$f = U/4L = {U/(4*L):.4f}$ Hz",
            fontsize=10, color=HOT, linespacing=1.35)
    for T in (150.0, 600.0):
        a2.plot([1 / T], [(1 / T) * S_eps(1 / T)], "o", color=GOLD, markersize=10,
                zorder=6)
    style(a2, "frequency $f$  [Hz]", "$f\\,S_\\varepsilon(f)$",
          "energy per decade: area under this is variance")
    fig.tight_layout()
    save(fig, path)


def window(path):
    """The demodulation window is not narrow, so S cannot be pulled out."""
    T = 150.0
    f = np.linspace(1e-6, 8 / T, 4000)
    a = 2 * np.pi / T
    b = 2 * np.pi * f
    z = (a - a * np.exp(-1j * b * T) * np.cos(a * T)
         - 1j * b * np.exp(-1j * b * T) * np.sin(a * T)) / (a**2 - b**2)
    w2 = np.abs(z) ** 2
    w2 /= w2.max()

    fig, ax = figure(11.6, 3.9)
    ax.fill_between(f * T, 0, w2, color=SOFT, zorder=0)
    ax.plot(f * T, w2, color=VIOLET, linewidth=2.8,
            label=r"$|\hat w(f)|^2$, the demodulation window, normalized")
    ax.plot(f * T, S_eps(f) / S_eps(1 / T), color=PRIMARY, linewidth=2.6,
            label=r"$S_\varepsilon(f)/S_\varepsilon(1/T)$, the spectrum it samples")
    ax.axvline(1.0, color=GOLD, linewidth=2.0, linestyle=(0, (6, 3)))
    ax.text(1.08, 1.02, "dither\n$f_1 = 1/T$", fontsize=10.5, color=GOLD,
            linespacing=1.35)
    ax.annotate("", xy=(0.02, 0.14), xytext=(2.0, 0.14),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, linewidth=1.5))
    ax.text(1.62, 0.20, "main lobe spans $0$ to $2f_1$", fontsize=10.5,
            color=MUTED, ha="center")
    ax.text(3.4, 1.75, "Across that lobe $S_\\varepsilon$ falls $12\\times$,\n"
            "so it is not flat and cannot be\npulled out of the integral.",
            fontsize=10.5, color=PRIMARY, linespacing=1.45)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.1)
    style(ax, "frequency, in units of the dither frequency $f/f_1$",
          "normalized")
    leg = ax.legend(loc="upper center", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def roadmap(path):
    """The four links, what each needs, and what each produces."""
    from windkit import blank, box
    fig, ax = figure(11.6, 4.3)
    blank(ax)
    ax.set_xlim(-0.05, 11.7)
    ax.set_ylim(0, 4.6)

    steps = [
        ("A", "reduce the\nrandomness", r"$y-\bar y = 3\varepsilon$",
         r"needs $P,\ C_P,\ \lambda$", PRIMARY),
        ("B", "make it a\nlinear functional", r"$\hat g_{\rm noise}=\frac{6}{aT}Z$",
         "needs the estimator", VIOLET),
        ("C", "variance of a\nlinear functional", r"$\sigma^2_Z$ from $S$ and $\hat w$",
         "needs Wiener-Khinchin", BLUE),
        ("D", "put in the real\nspectrum", "a number",
         r"needs Kaimal and $L$", HOT),
    ]
    W, GAP = 2.62, 0.38
    for i, (tag, title, out, needs, col) in enumerate(steps):
        x = i * (W + GAP)
        box(ax, x, 1.45, W, 2.45,
            f"{tag}.  {title}\n\n{out}\n\n{needs}", ec=col, fs=10.5)
        if i < 3:
            ax.annotate("", xy=(x + W + GAP - 0.04, 2.67),
                        xytext=(x + W + 0.04, 2.67),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6))

    box(ax, 0.0, 0.18, 4 * W + 3 * GAP, 1.05,
        r"Goal: $\sigma(\hat g)$, the scatter of one period's gradient estimate."
        "\nIt fixes the dither amplitude, the averaging time, and the terminal error.",
        fc=SOFT, ec=PRIMARY, fs=11, lw=1.3)
    save(fig, path, pad=0.05)


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
