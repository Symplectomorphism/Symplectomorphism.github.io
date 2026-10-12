"""Figures for the consolidated deck "Gradient Noise in Extremum Seeking for Wind Turbines".

Every figure is drawn from results copied from the `rotea` repo's `src/results/` into
`sim/`, or from the closed-form formulas of the technical report; nothing here is
fitted. Numbers quoted on the figures are the ones the report states and the rotea claim
checker verifies.

    python fig_report.py            # writes rpt-*.svg next to this file
"""

from __future__ import annotations

import os

import numpy as np

from windkit import (BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, RULE, SOFT, VIOLET,
                     arrow, blank, box, figure, plt, save, style)
from fig_wind import S_eps, corner

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
TAU_R = 6.8776          # rotor time constant at 8 m/s (I = 4.38e7 kg m^2)
J2 = 0.7022             # |J''(u*)| u*^2
TAUC = 86400.0          # tracking time, one day
NU = 0.061              # normalized optimal gain
MODELS = (("kaimal", "Kaimal", PRIMARY, "-"), ("vonkarman", "von Kármán", VIOLET, "--"))
# Draw text as outlines: hats and tildes over symbols do not survive svg.fonttype "none".
plt.rcParams["svg.fonttype"] = "path"


def _load(name):
    return np.load(os.path.join(SIM, name))


def _legend(ax, **kw):
    kw.setdefault("fontsize", 11.5)
    leg = ax.legend(frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def gains(T):
    wt = 2 * np.pi * TAU_R / np.asarray(T, dtype=float)
    g1 = 1 / (1 + wt**2)
    return wt, g1, wt * g1


def psi(T, model):
    return 9 * 0.5 * S_eps(1 / np.asarray(T, dtype=float), model)


def lmin(T, model, quadrature=False):
    wt, g1, gq = gains(T)
    g = gq if quadrature else g1
    return np.sqrt(psi(T, model) * g1 / (2 * TAUC * g**2))


# ---------------------------------------------------------------- energy results

CTRL = (("PUB_LPESC", "LP-ESC\nKumar & Rotea 2022", HOT),
        ("PUB_PIESC", "LP-PIESC\nKumar & Rotea 2022", GOLD),
        ("LIT_DEMOD", "LP-ESC\nCiri, Leonardi & Rotea 2019", BLUE),
        ("REC_PIESC", "LP-PIESC\nour design", VIOLET),
        ("REC_DEMOD", "LP-ESC\nour design", PRIMARY))


def _energy(model):
    suf = "" if model == "kaimal" else "_vonkarman"
    d = dict(_load(f"exp13_energy{suf}.npz"))
    d.update({k: v for k, v in _load(f"exp13_energy{suf}_lit_demod.npz").items() if k.startswith("LIT")})
    return d


def _loss(d, kind, scen, window):
    vals = []
    for s in range(int(d["seeds"])):
        p, pr = d[f"{kind}_{scen}_{s}_P"], d[f"ref_{scen}_{s}"]
        sl = slice(0, 24) if window == "day1" else slice(24, None)
        vals.append(100 * (1 - p[sl].sum() / pr[sl].sum()))
    return np.mean(vals), np.min(vals), np.max(vals)


def energy_bars(path):
    """Headline: energy lost by the five controllers, both spectra, log scale."""
    fig, ax = figure(11.4, 5.2)
    E = {m: _energy(m) for m, *_ in MODELS}
    y = np.arange(len(CTRL))
    for j, (m, lab, col, _) in enumerate(MODELS):
        for i, (kind, name, c) in enumerate(CTRL):
            mean, lo, hi = _loss(E[m], kind, "steady", "rest")
            yy = y[i] + (0.19 if j == 0 else -0.19)
            ax.barh(yy, mean, height=0.36, color=c, alpha=1.0 if j == 0 else 0.45,
                    edgecolor=c, linewidth=1.2)
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.0, alpha=0.55)
            ax.text(hi * 1.08, yy, f"{mean:.2f}%" if mean < 1 else f"{mean:.1f}%",
                    va="center", fontsize=12.5 if j == 0 else 11.5, color=c,
                    alpha=1.0 if j == 0 else 0.7, fontweight="bold" if j == 0 else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 12)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n, _ in CTRL], fontsize=12)
    style(ax, xlabel="energy lost against a turbine held at the optimum, days 2–6 (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    from matplotlib.patches import Patch
    _legend(ax, handles=[Patch(color=MUTED, alpha=1.0, label="Kaimal (upper bar)"),
                         Patch(color=MUTED, alpha=0.45, label="von Kármán (lower bar)")],
            loc="upper right")
    save(fig, path)


def wander(path):
    """The gain over six days: Kumar & Rotea 2022 LP-PIESC against our LP-ESC design."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    d = _energy("kaimal")
    t = np.arange(len(d["PUB_PIESC_steady_0_uh"])) / 24
    for ax, kind, title, col in ((axs[0], "PUB_PIESC", "LP-PIESC, Kumar & Rotea 2022", GOLD),
                                 (axs[1], "REC_DEMOD", "LP-ESC, our design", PRIMARY)):
        for s in range(int(d["seeds"])):
            ax.plot(t, d[f"{kind}_steady_{s}_uh"] / NU, color=col, lw=1.6, alpha=0.9 - 0.15 * s)
        ax.axhline(1, color=INK, lw=1.0, ls=":")
        style(ax, xlabel="time (days)", title=title)
        ax.set_ylim(0.5, 1.75)
    axs[0].set_ylabel("gain / optimal gain", color=INK, fontsize=12.5)
    axs[1].text(5.9, 0.56, "four wind records each", ha="right", fontsize=11, color=MUTED)
    fig.tight_layout()
    save(fig, path)


def tracking(path):
    """After the optimum moves: the gain from 0.8 u*, hourly means, one wind record."""
    fig, ax = figure(11.0, 4.6)
    d = _energy("kaimal")
    s = 1
    t = np.arange(len(d[f"REC_DEMOD_tracking_{s}_uh"])) / 24
    for kind, name, col in CTRL:
        ax.plot(t, d[f"{kind}_tracking_{s}_uh"] / NU, color=col,
                lw=1.1 if kind == "PUB_LPESC" else 2.2, alpha=0.55 if kind == "PUB_LPESC" else 1.0,
                label=name.replace("\n", ", "))
    ax.axhline(1, color=INK, lw=1.0, ls=":")
    ax.axhline(0.8, color=FAINT, lw=1.0, ls="--")
    ax.text(0.05, 0.78, "start: 0.8 × optimum", ha="left", fontsize=11, color=MUTED, va="top")
    style(ax, xlabel="time (days)", ylabel="gain / optimal gain")
    ax.set_ylim(0.4, 1.6)
    _legend(ax, loc="upper left", ncol=3, fontsize=10)
    save(fig, path)


# ---------------------------------------------------------------- the chain

def chain(path):
    """The argument at a glance."""
    fig, ax = figure(14.0, 2.6)
    blank(ax)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0.6, 2.4)
    steps = [("wind\nspectrum", BLUE),
             ("log-power\n$\\approx 3\\varepsilon$", PRIMARY),
             ("one estimate:\nspectrum\n× window", PRIMARY),
             ("slow loop:\ndensity at $f_1$", PRIMARY),
             ("gain\nscatters", GOLD),
             ("loss =\nscatter\n+ dither", HOT),
             ("design\n$a$, $T$, step", PRIMARY)]
    w, gap = 1.72, 0.3
    for i, (txt, col) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.75, w, 1.5, txt, ec=col, fs=12, lw=2.0)
        if i:
            arrow(ax, (x - gap + 0.02, 1.5), (x - 0.02, 1.5), color=MUTED, lw=1.6)
    save(fig, path)


def logpower(path):
    """Link 1: at the optimum, log-power fluctuates as 3 epsilon."""
    fig, ax = figure(7.6, 5.2)
    for m, lab, col, ls in MODELS:
        d = _load("exp5_chain.npz" if m == "kaimal" else "exp5_chain_vonkarman.npz")
        e, y = d["eps_sample"], d["y_sample"]
        y = y - y.mean()
        ax.scatter(3 * e[::6], y[::6], s=5, color=col, alpha=0.35, linewidths=0)
        ax.plot([], [], color=col, lw=0, marker="o", label=f"{lab}: slope {float(d['slope']):.3f}, corr. {float(d['corr']):.3f}")
    x = np.linspace(-0.8, 0.8, 2)
    ax.plot(x, x, color=INK, lw=1.4, ls="--", label="prediction: slope 3")
    style(ax, xlabel="$3\\varepsilon$  (three times the relative wind fluctuation)",
          ylabel="log-power fluctuation $y - \\mathrm{E}[y]$")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    _legend(ax, loc="upper left", fontsize=11)
    save(fig, path)


def window(path):
    """Link 2: one estimate sees the spectrum through the window."""
    fig, ax = figure(11.0, 4.6)
    T = 150.0
    f1 = 1 / T
    x = np.linspace(0.02, 4, 900)
    f = x * f1
    om, b = 2 * np.pi / T, 2 * np.pi * f
    w2 = (2 * om * np.sin(b * T / 2)) ** 2 / (om**2 - b**2) ** 2
    w2c = (f / f1) ** 2 * w2
    for m, lab, col, ls in MODELS:
        ax.plot(x, S_eps(f, m) / S_eps(f1, m), color=col, lw=2.2, ls=ls, label=f"wind density, {lab} (÷ its value at $f_1$)")
    ax.fill_between(x, 0, w2 / (T**2 / 4), color=PRIMARY, alpha=0.12, label="sine window (in-phase channel)")
    ax.plot(x, w2c / (T**2 / 4), color=GOLD, lw=1.8, label="cosine window (quadrature channel)")
    ax.axvline(1, color=INK, lw=1.0, ls=":")
    ax.text(1.03, 2.6, "dither\nfrequency $f_1$", fontsize=11, color=MUTED, va="top")
    style(ax, xlabel="frequency / dither frequency  ($T = 150$ s)", ylabel="weight (normalized at $f_1$)")
    ax.set_ylim(0, 2.8)
    ax.set_xlim(0, 4)
    _legend(ax, loc="upper right", fontsize=10.5)
    save(fig, path)


def noise_period(path):
    """Link 2 checked: noise of one estimate, predicted and simulated, over a tenfold range of periods."""
    fig, ax = figure(9.6, 4.6)
    Ts = [60, 150, 314, 600]
    for j, (m, lab, col, ls) in enumerate(MODELS):
        d = _load("exp6_period_sweep.npz" if m == "kaimal" else "exp6_period_sweep_vonkarman.npz")
        pred = np.array([float(d[f"sg_pred_{T}"]) for T in Ts]) * 1e7
        sim = np.array([float(d[f"sg_meas_{T}"]) for T in Ts]) * 1e7
        ax.plot(Ts, pred, color=col, lw=2.2, ls=ls, label=f"predicted, {lab}")
        ax.plot(Ts, sim, color=col, lw=0, marker="o", ms=9, mfc="white", mew=2.2, label=f"simulated, {lab}")
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    style(ax, xlabel="dither period $T$ (s)", ylabel="std. of one estimate ($10^{-7}$ (N m s$^2$)$^{-1}$)")
    _legend(ax, loc="lower left", fontsize=10.5, ncol=2)
    ax.text(0.98, 0.95, "simulated / predicted: 1.005 to 1.022\n240 periods each, no fitted parameter",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, color=MUTED)
    save(fig, path)


def spectra(path):
    """The two IEC spectra, with the three dither frequencies."""
    fig, ax = figure(10.0, 4.6)
    f = np.logspace(-4, 0, 600)
    for m, lab, col, ls in MODELS:
        ax.loglog(f, S_eps(f, m), color=col, lw=2.4, ls=ls, label=f"{lab}, $L$ = {340 if m == 'kaimal' else 147} m")
    for T, name in ((314.16, "314 s"), (150, "150 s"), (60, "60 s")):
        ax.axvline(1 / T, color=FAINT, lw=1.0, ls=":")
        ax.text(1 / T * 1.05, 1.2e-3, name, fontsize=11, color=MUTED, rotation=90, va="bottom")
    style(ax, xlabel="frequency $f$ (Hz)", ylabel="one-sided density $S(f)$ (s)")
    _legend(ax, loc="lower left")
    save(fig, path)


def channels(path):
    """Link 3: what each channel carries, against the period."""
    fig, ax = figure(10.0, 4.6)
    T = np.logspace(np.log10(15), np.log10(700), 400)
    wt, g1, gq = gains(T)
    ax.semilogx(T, g1, color=PRIMARY, lw=2.6, label="in-phase gain $G_1$ (demodulation, LP-ESC)")
    ax.semilogx(T, gq, color=GOLD, lw=2.6, label="quadrature gain $G_q = \\omega\\tau_r G_1$ (LP-PIESC, published)")
    for Tm in (60, 314.16):
        _, a, b = gains(Tm)
        ax.plot([Tm, Tm], [b, a], color=HOT, lw=1.6)
        ax.text(Tm * 1.06, (a + b) / 2, f"×{a / b:.1f}", color=HOT, fontsize=13, fontweight="bold", va="center")
    ax.axvline(2 * np.pi * TAU_R, color=FAINT, ls=":", lw=1.0)
    ax.text(2 * np.pi * TAU_R * 1.04, 0.05, "$2\\pi\\tau_r$ = 43 s", fontsize=11, color=MUTED)
    style(ax, xlabel="dither period $T$ (s)", ylabel="share of the gradient signal")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([20, 43, 60, 150, 314, 600])
    ax.set_xticklabels(["20", "43", "60", "150", "314", "600"])
    _legend(ax, loc="upper left", fontsize=11)
    save(fig, path)


def loop(path):
    """Link 4 checked: the scatter of a slow loop sees the long-run variance."""
    fig, ax = figure(10.0, 4.6)
    Ts = [60, 150, 314]
    x = np.arange(len(Ts))
    for j, (m, lab, col, ls) in enumerate(MODELS):
        d = _load("exp9_long_run.npz" if m == "kaimal" else "exp9_long_run_vonkarman.npz")
        meas = [float(d[f"u_sd_meas_{T}"]) for T in Ts]
        lr = [float(d[f"u_sd_pred_lr_{T}"]) for T in Ts]
        pp = [float(d[f"u_sd_pred_pp_{T}"]) for T in Ts]
        xx = x + (-0.2 if j == 0 else 0.2)
        ax.bar(xx, meas, width=0.36, color=col, alpha=0.85, label=f"measured, {lab}")
        ax.scatter(xx, lr, marker="_", s=900, color=INK, linewidths=2.6, zorder=4,
                   label="long-run prediction" if j == 0 else None)
        ax.scatter(xx, pp, marker="_", s=900, color=HOT, linewidths=2.0, zorder=4,
                   label="one-period variance (overstates)" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T = {T} s" for T in Ts], fontsize=12)
    style(ax, ylabel="scatter of the gain / optimal gain")
    ax.grid(axis="x", visible=False)
    _legend(ax, loc="upper left", fontsize=10.5)
    save(fig, path)


def loss_amplitude(path):
    """Link 5: the loss is wander plus dither; the best amplitude balances them."""
    fig, ax = figure(10.0, 4.8)
    T = 60.0
    wt, g1, gq = gains(T)
    a = np.linspace(0.02, 0.3, 400)
    P = psi(T, "kaimal")
    A = P / (2 * g1**2 * TAUC * J2)
    B = 0.25 * J2 * g1
    ax.plot(a, 100 * A / a**2, color=GOLD, lw=2.2, label="cost of the scatter (falls as $1/a^2$)")
    ax.plot(a, 100 * B * a**2, color=BLUE, lw=2.2, label="cost of the dither (rises as $a^2$)")
    ax.plot(a, 100 * (A / a**2 + B * a**2), color=PRIMARY, lw=3.0, label="total loss")
    aopt = (A / B) ** 0.25
    lm = 100 * 2 * np.sqrt(A * B)
    ax.plot([aopt], [lm], "o", color=HOT, ms=10, zorder=5)
    ax.annotate(f"$a_{{opt}}$ = {aopt:.3f} $u^\\star$\nloss {lm:.2f}%", (aopt, lm), (aopt + 0.04, lm + 0.25),
                fontsize=12.5, color=HOT, arrowprops=dict(arrowstyle="-", color=HOT, lw=1.0))
    style(ax, xlabel="dither amplitude $a / u^\\star$", ylabel="power lost (%)")
    ax.set_ylim(0, 1.2)
    ax.set_title("LP-ESC, $T$ = 60 s, tracking time one day, Kaimal", loc="left", fontsize=11.5, color=MUTED)
    _legend(ax, loc="upper right", fontsize=11)
    save(fig, path)


def design(path):
    """Least loss against the period: both routes, both spectra, with the published designs."""
    fig, ax = figure(10.6, 4.9)
    T = np.logspace(np.log10(20), np.log10(700), 400)
    for m, lab, col, ls in MODELS:
        ax.loglog(T, 100 * lmin(T, m), color=PRIMARY, lw=2.6, ls=ls, label=f"LP-ESC (in-phase), {lab}")
        ax.loglog(T, 100 * lmin(T, m, True), color=GOLD, lw=2.6, ls=ls, label=f"LP-PIESC (quadrature), {lab}")
    ax.axvspan(20, 45, color=RULE, alpha=0.5)
    ax.text(21, 3.2, "rotor model\nleast reliable", fontsize=10.5, color=MUTED)
    for Tm, txt in ((60, "60 s:\nrecommended"), (314.16, "314 s:\npublished")):
        ax.axvline(Tm, color=INK, lw=1.0, ls=":")
        ax.text(Tm * 1.05, 4.6, txt, fontsize=11, color=INK, va="top")
    style(ax, xlabel="dither period $T$ (s)", ylabel="least loss $\\ell_{min}$ (%)")
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_ylim(0.1, 5)
    ax.set_xticks([20, 45, 60, 150, 314, 600])
    ax.set_xticklabels(["20", "45", "60", "150", "314", "600"])
    _legend(ax, loc="lower right", fontsize=10, ncol=2)
    save(fig, path)


def reproduce(path):
    """The published runs repeated: LP-ESC converges; LP-PIESC holds wherever its integrator starts."""
    d = _load("exp11_lppiesc.npz")
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    cols = {4: BLUE, 8: HOT, 12: INK}
    titles = {"lpesc": "LP-ESC, published tuning", "from_0.2": "LP-PIESC, published, from 0.2",
              "from_opt": "LP-PIESC, published, from the optimum"}
    for ax, case in zip(axs, ("lpesc", "from_0.2", "from_opt")):
        for v in (4, 8, 12):
            ax.plot(d[f"{case}_{v}__t"], d[f"{case}_{v}__u"], color=cols[v], lw=1.0, label=f"{v} m/s")
        ax.axhline(0.061, color=PRIMARY, lw=1.6, ls="-.")
        ax.axvline(150, color=FAINT, lw=1.0, ls=":")
        style(ax, xlabel="time (s)", title=titles[case])
        ax.set_ylim(0, 0.23)
    axs[0].set_ylabel("gain $\\nu$ (optimum 0.061)", color=INK, fontsize=12.5)
    _legend(axs[0], loc="upper right", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)


def piesc_loop(path):
    """LP-PIESC's structure: a least-squares fit, then proportional and integral paths."""
    from matplotlib.patches import Circle
    fig, ax = figure(12.0, 4.6)
    blank(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    box(ax, 0.2, 3.1, 1.6, 1.0, "turbine\n$u \\mapsto P$", ec=BLUE, fs=12.5)
    box(ax, 2.3, 3.1, 1.2, 1.0, "$\\ln(\\cdot)$", fs=13)
    box(ax, 4.1, 2.8, 4.0, 1.6,
        "least-squares fit, memory 0.05 s\n$\\dot y \\approx \\theta_0 + \\theta_1(\\nu - \\hat\\nu)$\n"
        "gradient estimate $\\hat\\theta_1$", ec=VIOLET, fs=12.5)
    box(ax, 8.8, 3.2, 2.0, 0.8, "proportional\n$k_p \\hat\\theta_1$", ec=GOLD, fs=11.5)
    box(ax, 8.8, 1.9, 2.0, 0.8, "integral\n$\\hat\\nu = \\int \\hat\\theta_1 / \\tau_I$", ec=GOLD, fs=11.5)
    ax.add_patch(Circle((11.4, 1.0), 0.28, fc="white", ec=INK, lw=1.6, zorder=3))
    ax.text(11.4, 1.0, "+", ha="center", va="center", fontsize=16, zorder=4)
    box(ax, 8.9, 0.15, 1.8, 0.7, "dither $a\\sin\\omega t$", ec=HOT, fs=11.5)
    arrow(ax, (1.8, 3.6), (2.3, 3.6))
    arrow(ax, (3.5, 3.6), (4.1, 3.6))
    arrow(ax, (8.1, 3.6), (8.8, 3.6))
    arrow(ax, (8.1, 3.0), (8.8, 2.3))
    arrow(ax, (10.8, 3.6), (11.4, 1.28))
    arrow(ax, (10.8, 2.3), (11.2, 1.2))
    arrow(ax, (10.7, 0.5), (11.15, 0.85))
    arrow(ax, (11.12, 1.0), (1.0, 1.0))
    arrow(ax, (1.0, 1.0), (1.0, 3.1))
    ax.text(5.5, 1.15, "gain $\\nu = k_p\\hat\\theta_1 + \\hat\\nu + a\\sin\\omega t$", fontsize=12.5, color=INK)
    save(fig, path)


def optimum(path):
    """Backup: where the optimum sits in turbulence (fixed gains, no dither)."""
    fig, ax = figure(10.0, 4.6)
    for m, lab, col, ls in MODELS:
        d = _load("exp14_turbulent_optimum.npz" if m == "kaimal" else "exp14_turbulent_optimum_vonkarman.npz")
        g = d["gains"]
        ax.plot(g, 100 * d["energy_rel"], color=col, lw=2.4, ls=ls, marker="o", ms=5,
                label=f"energy, {lab} (peak {float(d['opt_energy_rel']):.3f} $u^\\star$)")
    g = np.linspace(0.9, 1.15, 200)
    ax.plot(g, -100 * J2 / 2 * (g - 1) ** 2, color=FAINT, lw=1.6, ls=":", label="calm-wind curvature")
    ax.axvline(1, color=INK, lw=1.0, ls=":")
    style(ax, xlabel="fixed gain / $u^\\star$", ylabel="energy against $u^\\star$ (%)")
    _legend(ax, loc="lower left", fontsize=11)
    save(fig, path)


JOBS = [(energy_bars, "rpt-energy.svg"), (wander, "rpt-wander.svg"), (tracking, "rpt-tracking.svg"),
        (chain, "rpt-chain.svg"), (logpower, "rpt-logpower.svg"), (window, "rpt-window.svg"),
        (noise_period, "rpt-noise-period.svg"), (spectra, "rpt-spectra.svg"),
        (channels, "rpt-channels.svg"), (loop, "rpt-loop.svg"), (loss_amplitude, "rpt-loss-amplitude.svg"),
        (design, "rpt-design.svg"), (reproduce, "rpt-reproduce.svg"), (piesc_loop, "rpt-piesc-loop.svg"),
        (optimum, "rpt-optimum.svg")]

if __name__ == "__main__":
    for fn, name in JOBS:
        fn(os.path.join(HERE, name))
        print("wrote", name)
