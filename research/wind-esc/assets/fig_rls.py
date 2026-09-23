"""Figures for the least-squares deck, drawn from exp7 and exp8 results.

Each reads a .npz copied from the `rotea` repo's `src/results/`; nothing here
recomputes the simulation.
"""

from __future__ import annotations

import os

import numpy as np

from windkit import BLUE, FAINT, GOLD, HOT, INK, MUTED, PRIMARY, SOFT, VIOLET, figure, save, style

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
WT = 0.02 * 6.357          # omega * tau_rotor at 8 m/s


def _load(name):
    return np.load(os.path.join(SIM, name))


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=11, **kw)
    for t in leg.get_texts():
        t.set_color(INK)


def phase(path):
    """The power's response to a slow dither: large in phase, small in quadrature."""
    re_g, im_g = 1 / (1 + WT**2), -WT / (1 + WT**2)
    t = np.linspace(0, 1, 600)
    s, c = np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)
    fig, ax = figure(11.6, 3.7)
    ax.axhline(0, color=FAINT, linewidth=1.0)
    ax.plot(t, s, color=GOLD, linewidth=2.0, linestyle=(0, (6, 3)),
            label=r"dither $d(t)/a = \sin\omega t$")
    ax.plot(t, re_g * s + im_g * c, color=INK, linewidth=2.8,
            label="power response $y_{\\rm det}(t)$ / (J\u2032a)")
    ax.plot(t, re_g * s, color=PRIMARY, linewidth=2.2,
            label=rf"in-phase part, amplitude $\mathrm{{Re}}\,G = {re_g:.3f}$")
    ax.plot(t, im_g * c, color=HOT, linewidth=2.2,
            label=rf"quadrature part, amplitude $|\mathrm{{Im}}\,G| = {abs(im_g):.3f}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.15)
    style(ax, r"time, in dither periods  $t/T$", "normalized")
    _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, path)


def identity(path):
    """The published estimate, period by period, against (u - u_hat) * y_dot."""
    d = _load("exp7_piesc_estimator.npz")
    x, y = 1e6 * d["dydot__turb_1.0"], 1e6 * d["th1_PUB__turb_1.0"]
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = figure(7.6, 4.4)
    lim = 1.05 * max(np.abs(x).max(), np.abs(y).max())
    ax.plot([-lim, lim], [-lim, lim], color=FAINT, linewidth=1.4, linestyle=(0, (5, 3)),
            label="identity")
    ax.plot(x, y, "o", color=PRIMARY, markersize=4.5, alpha=0.75,
            label=f"400 dither periods, correlation {r:.3f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    style(ax, r"period mean of $(u-\hat u)\,\dot y$   [$10^{-6}$ s$^{-1}$]",
          r"period mean of $\hat\theta_1$   [$10^{-6}$]")
    _legend(ax, loc="upper left")
    save(fig, path)


def channels(path):
    """Signal and noise of the in-phase and quadrature channels.

    Signal: both measured in calm wind; the quadrature bar is predicted from the
    in-phase one times omega*tau.  Noise: one period (exp7, the turbine) and the
    long run a slow integrator sees (exp9, batch means of synthesized wind),
    each against the variance rule.
    """
    import matplotlib.pyplot as plt
    d = _load("exp7_piesc_estimator.npz")
    e9 = _load("exp9_long_run.npz")
    sig = {k: abs(d[f"{k}__calm_0.9"].mean() - d[f"{k}__calm_1.1"].mean()) / 2
           for k in ("demod_sin", "demod_cos")}
    one = {k: d[f"{k}__turb_1.0"].std() for k in ("demod_sin", "demod_cos")}
    one_pred = {"demod_sin": 28.90, "demod_cos": 25.17}
    g = 6 / (0.005 * 314.0)
    long = {"demod_sin": g * np.sqrt(float(e9["lrv_sin_meas_314"])),
            "demod_cos": g * np.sqrt(float(e9["lrv_cos_meas_314"]))}
    long_pred = g * np.sqrt(float(e9["lrv_pred_314"]))
    pred_sig_cos = sig["demod_sin"] * WT
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.9), width_ratios=[1, 1.55])
    keys = ["demod_sin", "demod_cos"]
    cols = [PRIMARY, HOT]

    xs = np.arange(2)
    a1.bar(xs, [sig[k] for k in keys], width=0.55, color=cols, alpha=0.85, label="measured")
    a1.plot([1], [pred_sig_cos], "D", color=INK, markersize=9, markerfacecolor="white",
            markeredgewidth=2, label=r"predicted: in phase $\times\,\omega\tau$")
    a1.text(0.32, sig["demod_sin"] * 1.02, f"{sig['demod_sin']:.3g}", fontsize=10, color=INK)
    a1.text(1.32, sig["demod_cos"] + 0.05, f"{sig['demod_cos']:.3g}\npred. {pred_sig_cos:.3g}",
            fontsize=10, color=INK, linespacing=1.3)
    a1.set_ylim(0, 1.5 * sig["demod_sin"])
    a1.set_xticks(xs, ["in phase\n(sine)", "quadrature\n(cosine)"])
    a1.set_xlim(-0.5, 2.0)
    style(a1, None, "per unit gain", "signal: 10% off $u^\\star$")

    xn = np.array([0, 1, 2.7, 3.7])
    meas = [one["demod_sin"], one["demod_cos"], long["demod_sin"], long["demod_cos"]]
    pred = [one_pred["demod_sin"], one_pred["demod_cos"], long_pred, long_pred]
    a2.bar(xn, meas, width=0.55, color=cols * 2, alpha=0.85, label="measured")
    a2.plot(xn, pred, "D", color=INK, markersize=9, markerfacecolor="white",
            markeredgewidth=2, label="predicted: variance rule")
    for x_, m, p_ in zip(xn, meas, pred):
        a2.text(x_ + 0.3, max(m, p_) + 0.8, f"{m:.3g}\npred. {p_:.3g}", fontsize=10,
                color=INK, linespacing=1.3)
    a2.set_ylim(0, 1.5 * max(meas))
    a2.set_xticks(xn, ["sine", "cosine", "sine", "cosine"])
    a2.set_xlim(-0.5, 4.6)
    a2.text(0.5, -0.30 * max(meas), "one period", ha="center", fontsize=11, color=MUTED)
    a2.text(3.2, -0.30 * max(meas), "long run, per period", ha="center", fontsize=11, color=MUTED)
    style(a2, None, None, "noise per period at $u^\\star$")
    for ax in (a1, a2):
        leg = ax.legend(frameon=False, fontsize=10, loc="upper left", ncols=1 if ax is a1 else 2)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    save(fig, path)


def calibration(path):
    """The static-map RLS at u*, with the error band computed from R and its own Sigma."""
    import matplotlib.pyplot as plt
    d = _load("exp7_series_ustar.npz")
    t, th = d["t"] / (2 * np.pi / 0.02), d["th1_STATIC"]
    band = np.sqrt(float(d["R"]) / 2 * d["sinv22_STATIC"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.9), width_ratios=[2.3, 1])
    a1.fill_between(t, -2 * band, 2 * band, color=SOFT, label=r"$\pm2$ reported standard deviations")
    a1.fill_between(t, -band, band, color="#d6e4dc", label=r"$\pm1$")
    a1.plot(t, th, color=PRIMARY, linewidth=1.3, label=r"$\hat\theta_1(t)$")
    a1.axhline(0, color=INK, linewidth=1.0)
    a1.set_xlim(0, t[-1])
    style(a1, "time, in dither periods", r"$\hat\theta_1$  (true value $\approx 0$)",
          "estimate and its reported error band")
    _legend(a1, loc="upper right", ncols=3)
    a1.set_ylim(-75, 95)
    counts, edges = d["hist_counts"], d["hist_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    dens = counts / counts.sum() / (edges[1] - edges[0])
    s_rep = float(d["sd_reported"])
    a2.bar(centers, dens, width=edges[1] - edges[0], color=PRIMARY, alpha=0.35,
           label=f"400 periods, sd {float(d['sd_actual']):.1f}")
    xg = np.linspace(-80, 80, 400)
    a2.plot(xg, np.exp(-xg**2 / (2 * s_rep**2)) / np.sqrt(2 * np.pi) / s_rep, color=HOT,
            linewidth=2.2, label=f"normal, reported sd {s_rep:.1f}")
    a2.set_ylim(0, 1.45 * dens.max())
    style(a2, r"$\hat\theta_1$", "density", "distribution")
    leg = a2.legend(frameon=False, fontsize=9.5, loc="upper center", ncols=1)
    for tx in leg.get_texts():
        tx.set_color(INK)
    save(fig, path)


def schedule(path):
    """Fixed gains against gains scaled by the calibrated signal-to-noise ratio."""
    d = _load("exp8_snr_gain.npz")
    rows, kind = d["rows"], d["kind"]
    fig, ax = figure(11.6, 4.0)
    fx = rows[kind == "fixed"]
    ax.plot(fx[:, 4], fx[:, 3], "o-", color=PRIMARY, linewidth=2.4, markersize=8,
            label=r"fixed gain $K = T/\tau_c$")
    for r in fx:
        ax.annotate(f"$\\tau_c$ = {int(r[0])} h", (r[4], r[3]), textcoords="offset points",
                    xytext=(7, 6), fontsize=10.5, color=PRIMARY)
    sc = rows[kind == "sched"]
    ax.plot(sc[:, 4], sc[:, 3], "s", color=VIOLET, markersize=8,
            label="gain scaled by signal-to-noise (8 variants)")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    style(ax, "steady-state power loss  [%]   (days 5 to 15)",
          "transient power loss  [%]\n(days 0 to 2, from $0.6u^\\star$)")
    _legend(ax, loc="upper left")
    save(fig, path)
