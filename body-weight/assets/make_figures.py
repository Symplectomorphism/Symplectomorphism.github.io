#!/usr/bin/env python3
"""Draw every figure in the body-weight dynamics deck from `data/*.json`.

    ../../.venv/bin/python make_figures.py

Needs only numpy and matplotlib. The JSON comes from `compute.py`, which is the
one step that needs the model and the observation log.

Output is SVG with `svg.fonttype: none`, so the slide's own typeface is used.
Series colors are the deck's four categorical slots, checked as a set for
color-vision-deficiency separation on every pair, not just adjacent ones. A
figure never uses more than four, and identity is always carried by a direct
label or a legend as well as by color.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ------------------------------------------------------------------ palette

INK = "#1c2b26"
MUTED = "#6b7a74"
FAINT = "#97a49e"
RULE = "#dce3dd"
SOFT = "#eef3f0"
PRIMARY = "#175b48"

GREEN = "#0f8a6c"   # scale weight, the trend: the thing the scale reads
GOLD = "#b07d06"    # tissue, carbohydrate
BLUE = "#3a6fb0"    # water, protein, the causal filter
RED = "#c0392b"     # adaptation, fat, the naive rule

LB_PER_KG = 2.2046226

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 11,
    }
)


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def dates(x) -> np.ndarray:
    return np.array([np.datetime64(d) for d in x])


def style(ax, xlabel=None, ylabel=None, title=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=11, length=4, width=1.0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=12.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=12.5)
    if title:
        ax.set_title(title, color=PRIMARY, fontsize=13.5, fontweight="bold", loc="left")
    ax.grid(True, color=RULE, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def label(ax, x, y, text, color=INK, dx=6, dy=0, ha="left", va="center", fs=11.5, weight="bold"):
    """A direct label, offset in points from a data point.

    Text is always ink (or muted ink for secondary notes), never the series
    color: the mark beside it carries identity, and a gold or green label at
    slide size reads poorly.  `color` is accepted so call sites can say which
    series a label belongs to; only MUTED changes the rendering.
    """
    ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points",
        color=MUTED if color == MUTED else INK, fontsize=fs, ha=ha, va=va, fontweight=weight,
    )


def weighins(ax, x, y, **kw):
    """Raw scale readings: hollow-free gray dots with a surface ring."""
    m = np.isfinite(y)
    ax.plot(
        x[m], y[m], "o", markersize=kw.pop("ms", 6.5), color=kw.pop("color", MUTED),
        markeredgecolor="white", markeredgewidth=1.2, zorder=kw.pop("zorder", 4), **kw,
    )


def save(fig, name: str):
    fig.savefig(HERE / name, format="svg", bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ the record

EPISODE_TITLES = {
    "past-01": "Jan 2025",
    "past-02": "Aug 2025",
    "past-03": "Mar 2026",
    "past-04": "Aug 2026",
    "current": "Oct 2026: now",
}


def timeline():
    """Every weigh-in on record, one panel per attempt, shared weight axis.

    The calendar gaps between attempts are months long and carry no data, so a
    single calendar axis would squeeze each attempt into a sliver. Panels are
    sized by duration (with a floor, so two-day attempts stay legible) and the
    gap is written between them instead of drawn.
    """
    r = load("real")
    eps = r["trends"]
    n = np.array([len(e["dates"]) for e in eps], dtype=float)
    widths = np.maximum(n, 9)
    fig, axes = plt.subplots(
        1, len(eps), figsize=(12.6, 4.4), sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.12},
    )
    for i, (ax, e) in enumerate(zip(axes, eps, strict=True)):
        t = np.arange(len(e["dates"]))
        y, s, sd = arr(e["obs_lb"]), arr(e["smoothed_lb"]), arr(e["smoothed_sd_lb"])
        ax.fill_between(t, s - 1.96 * sd, s + 1.96 * sd, color=GREEN, alpha=0.14, linewidth=0)
        ax.plot(t, s, color=GREEN, zorder=3)
        weighins(ax, t, y)
        style(ax)
        ax.set_xlim(-0.8, max(len(t) - 1, 8) + 0.8)
        span = len(t)
        ax.set_title(
            f"{EPISODE_TITLES.get(e['name'], e['name'])}\n{span} day{'s' if span > 1 else ''}",
            color=PRIMARY if e["name"] == "current" else INK, fontsize=12, loc="left",
            fontweight="bold",
        )
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(7 if span > 10 else 4))
        if i:
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("weight (lb)", color=INK, fontsize=12.5)
    fig.supxlabel("day of attempt", color=INK, fontsize=12.5, y=-0.02)
    axes[0].set_ylim(203, 241)
    axes[2].plot([], [], "o", color=MUTED, markersize=6.5, label="weigh-in")
    axes[2].plot([], [], color=GREEN, label="smoothed trend, 95% band")
    axes[2].legend(loc="upper right")
    save(fig, "timeline.svg")


def smoother():
    """One attempt in detail: the raw readings, the filter, and the smoother."""
    r = load("real")
    e = [x for x in r["trends"] if x["name"] == "past-03"][0]
    t = dates(e["dates"])
    y, f = arr(e["obs_lb"]), arr(e["filtered_lb"])
    s, sd = arr(e["smoothed_lb"]), arr(e["smoothed_sd_lb"])

    fig, ax = plt.subplots(figsize=(11.6, 4.5))
    ax.fill_between(t, s - 1.96 * sd, s + 1.96 * sd, color=GREEN, alpha=0.14, linewidth=0)
    ax.plot(t, f, color=BLUE, linewidth=1.6, zorder=3, label="filtered (uses only the past)")
    ax.plot(t, s, color=GREEN, zorder=3, label="smoothed (uses the whole record), 95% band")
    weighins(ax, t, y, label="weigh-in")
    style(ax, ylabel="weight (lb)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.TH))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(loc="upper right", ncol=1)
    label(ax, t[-1], s[-1], "smoothed", GREEN, dx=8)
    label(ax, t[-1], f[-1], "filtered", BLUE, dx=8, dy=-14)
    ax.set_xlim(t[0] - np.timedelta64(1, "D"), t[-1] + np.timedelta64(4, "D"))
    save(fig, "smoother.svg")


def first_days():
    """Every attempt, aligned on its first weigh-in: the opening cliff."""
    r = load("real")
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    colors = {"past-01": BLUE, "past-03": GOLD, "past-04": RED, "current": GREEN}
    for e in r["trends"]:
        if e["name"] not in colors:
            continue
        y = arr(e["obs_lb"])[:15]
        t = np.arange(len(y))
        m = np.isfinite(y)
        d = y - y[m][0]
        c = colors[e["name"]]
        lw = 2.6 if e["name"] == "current" else 1.6
        ax.plot(t[m], d[m], "-o", color=c, linewidth=lw, markersize=6.5,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4,
                label=EPISODE_TITLES[e["name"]])
        last = np.flatnonzero(m)[-1]
        label(ax, t[last], d[last], EPISODE_TITLES[e["name"]].replace(": now", ""), c, dx=8)
    ax.axhline(0, color=FAINT, linewidth=1.0, zorder=1)
    style(ax, xlabel="day of attempt", ylabel="change since first weigh-in (lb)")
    ax.set_xlim(-0.4, 15.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.legend(loc="lower left", ncol=4)
    save(fig, "first-days.svg")


# ------------------------------------------------------------------ the model

def structure():
    """The loop: person, scale, filter, controller."""
    fig, ax = plt.subplots(figsize=(12.4, 3.9))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    def box(x, y, w, h, head, body, ec=PRIMARY, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h - 0.2, head, ha="center", va="top", fontsize=13.5,
                fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=11, color=INK,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1, text=None, tx=None, ty=None, ha="center", conn="arc3"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=1.6, color=MUTED, connectionstyle=conn))
        if text:
            ax.text(tx if tx is not None else (x0 + x1) / 2, ty if ty is not None else y0 + 0.14,
                    text, ha=ha, va="bottom", fontsize=10, color=MUTED, style="italic")

    box(0.05, 1.75, 2.85, 1.65, "Person",
        "5 hidden states: tissue, fat,\nglycogen water, sodium\nwater, adaptation")
    box(3.8, 1.75, 1.9, 1.65, "Scale", "$y = T + G + N + v$\none number a day")
    box(6.6, 1.75, 2.45, 1.65, "Filter", "unscented Kalman,\n14 states: 5 physical\n+ 9 parameters")
    box(9.95, 1.75, 2.4, 1.65, "Controller", "28-day horizon in\nweekly blocks;\nonly day 1 is eaten")
    arrow(2.9, 2.6, 3.8, 2.6)
    arrow(5.7, 2.6, 6.6, 2.6, "weigh-in", ty=2.68)
    arrow(9.05, 2.6, 9.95, 2.6, "belief", ty=2.68)
    # the return path: tomorrow's food, controller back round to the person
    ax.plot([11.1, 11.1, 1.55, 1.55], [1.75, 0.75, 0.75, 1.2], color=GREEN, lw=1.8)
    ax.add_patch(FancyArrowPatch((1.55, 1.1), (1.55, 1.75), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=GREEN))
    ax.text(6.2, 0.82, "tomorrow's calories and macros (kcal, protein, carbohydrate, fat)",
            ha="center", va="bottom", fontsize=11.5, color=GREEN, fontweight="bold")
    ax.text(6.2, 0.62, "what was actually eaten, logged exercise and sodium also feed the filter",
            ha="center", va="top", fontsize=10.5, color=MUTED, style="italic")
    save(fig, "structure.svg")


def forbes():
    f = load("forbes")
    fat, p = arr(f["fat_kg"]), arr(f["p"])
    r = load("real")
    fnow = r["current"]["fat_kg"][-1]
    sd = r["current"]["fat_sd_kg"][-1]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axvspan(fnow - 1.645 * sd, fnow + 1.645 * sd, color=GREEN, alpha=0.10, linewidth=0)
    ax.plot(fat, p, color=GREEN)
    pn = fnow / (fnow + 10.4)
    ax.plot([fnow], [pn], "o", color=GREEN, markersize=9, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)
    label(ax, fnow, pn, f"today's estimate: {fnow:.0f} kg fat\n{pn:.0%} of a lost pound is fat",
          INK, dx=12, dy=-26, va="top", weight="normal")
    for fk in (15, 25):
        pk = fk / (fk + 10.4)
        ax.plot([fk], [pk], "o", color=MUTED, markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        label(ax, fk, pk, f"{fk} kg: {pk:.0%}", MUTED, dx=-8, dy=10, ha="right", weight="normal")
    style(ax, xlabel="fat mass F (kg)", ylabel="fat share of mass lost, p")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.3, 0.9)
    ax.set_xlim(5, 60)
    ax.text(fnow, 0.315, "90% band", ha="center", va="bottom", fontsize=10, color=MUTED)
    save(fig, "forbes.svg")


def timescales():
    """One diet simulated at prior parameters: three clocks, three panels."""
    s = load("timescales")
    t = np.array(s["day"])
    scale, tissue = arr(s["scale_lb"]), arr(s["tissue_lb"])
    gly, adapt = arr(s["gly_lb"]), arr(s["adapt_kcal"])
    naive = np.where(t <= 0, s["start_lb"], s["start_lb"] - 1000.0 * np.maximum(t, 0) / 3500.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), gridspec_kw={"width_ratios": [1.75, 1, 1],
                                                                       "wspace": 0.32})
    ax = axes[0]
    ax.plot(t, naive, color=RED, linewidth=1.6, linestyle=(0, (5, 3)))
    ax.plot(t, tissue, color=GOLD, linestyle=(0, (2, 1.5)))
    ax.plot(t, scale, color=GREEN)
    label(ax, t[-1], naive[-1], f"3500 kcal/lb rule: \u2212{naive[0] - naive[-1]:.0f} lb", RED)
    label(ax, t[-1], scale[-1], f"model: \u2212{scale[0] - scale[-1]:.0f} lb", GREEN)
    ax.set_xlim(-20, 330)
    ax.set_xticks([0, 60, 120, 180])
    style(ax, xlabel="day", ylabel="weight (lb)", title="Tissue: months")
    ax.legend(["3500 kcal/lb rule", "tissue T", "scale weight W"], loc="lower left", fontsize=10.5)

    ax = axes[1]
    ax.plot(t, gly, color=BLUE)
    style(ax, xlabel="day", ylabel="glycogen water (lb)", title="Water: days")
    ax.set_xlim(-7, 21)
    i = np.searchsorted(t, 2)
    label(ax, t[i], gly[i], "carbs 250 → 50 g", BLUE, dx=10, dy=6, va="bottom", weight="normal")

    ax = axes[2]
    ax.plot(t, adapt, color=RED)
    style(ax, xlabel="day", ylabel="adaptation A (kcal/day)", title="Adaptation: weeks")
    i = int(np.nanargmin(adapt))
    label(ax, t[i], adapt[i], f"\u2212{-adapt[i]:.0f} kcal/day", RED, dx=0, dy=-16, ha="center",
          weight="normal")
    ax.set_ylim(adapt[i] * 1.22, 12)
    for a in axes:
        a.axvline(0, color=FAINT, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    save(fig, "timescales.svg")


# ------------------------------------------------------------------ estimation

# Prior scale of each parameter's unconstrained coordinate, from
# wdm.estimation.reparam.ParamTransform. Centers come from the JSON.
TRANSFORMS = {
    "rmr_intercept": ("log", 0.15), "rmr_slope": ("log", 0.25), "at_gain": ("logit", 0.70),
    "at_tau": ("log", 0.50), "gly_gain": ("log", 0.50), "gly_tau": ("log", 0.40),
    "na_gain": ("log", 0.70), "na_tau": ("log", 0.40), "partition_kappa": ("affine", 0.05),
}
SYMBOLS = {
    "rmr_intercept": r"$c_0$  intercept", "rmr_slope": r"$c_1$  per-kg slope",
    "at_gain": r"$\beta_A$  adaptation gain", "at_tau": r"$\tau_A$  adaptation lag",
    "gly_gain": r"$k_G$  glycogen gain", "gly_tau": r"$\tau_G$  glycogen lag",
    "na_gain": r"$k_{Na}$  sodium gain", "na_tau": r"$\tau_{Na}$  sodium lag",
    "partition_kappa": r"$\kappa_P$  protein sparing",
}


def to_eta(name, v, center):
    kind, s = TRANSFORMS[name]
    if kind == "log":
        return np.log(v / center) / s
    if kind == "logit":
        lg = lambda q: np.log(q / (1 - q))  # noqa: E731
        return (lg(v) - lg(center)) / s
    return (v - center) / s


def parameters():
    """Posterior 90% intervals in prior-sigma units: synthetic beside real."""
    syn, real = load("synthetic"), load("real")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True, gridspec_kw={"wspace": 0.08})
    names = list(TRANSFORMS)
    ypos = np.arange(len(names))[::-1]
    # Same x-range on both panels, wide enough for the farthest interval end.
    reach = max(to_eta(p["name"], p["q95"], p["prior"]) for p in syn["params"] + real["params"])
    xmax = max(6.4, np.ceil(reach) + 0.6)
    for ax, src, title in (
        (axes[0], syn, "Synthetic subject, 120 logged days"),
        (axes[1], real, "Real subject so far: 2 logged days"),
    ):
        ax.axvspan(-1.645, 1.645, color=SOFT, zorder=0)
        ax.axvline(0, color=FAINT, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        rows = {p["name"]: p for p in src["params"]}
        for y, n in zip(ypos, names, strict=True):
            p = rows[n]
            lo, med, hi = (to_eta(n, p[k], p["prior"]) for k in ("q05", "median", "q95"))
            ax.plot([lo, hi], [y, y], color=GREEN, linewidth=3.0, solid_capstyle="round", zorder=3)
            ax.plot([med], [y], "o", color=GREEN, markersize=8, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=4)
            if "truth" in p:
                tr = to_eta(n, p["truth"], p["prior"])
                ax.plot([tr], [y], "D", color=RED, markersize=7.5, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=5)
        style(ax, xlabel="shift from prior center (prior standard deviations)", title=title)
        ax.set_xlim(-3.2, xmax)
        ax.grid(axis="y", visible=False)
        ax.text(0, len(names) - 0.35, "prior 90%", ha="center", va="bottom", fontsize=10,
                color=MUTED)
    axes[0].set_yticks(ypos, [SYMBOLS[n] for n in names], fontsize=11.5, color=INK)
    axes[0].set_ylim(-0.7, len(names) + 0.2)
    axes[0].plot([], [], "-o", color=GREEN, linewidth=3, markersize=8, label="median, 90% interval")
    axes[0].plot([], [], "D", color=RED, markersize=7.5, label="truth")
    axes[0].legend(loc="lower right", fontsize=10.5)
    save(fig, "parameters.svg")


def ridge():
    """c0 against c1: individually loose, maintenance tight."""
    s = load("synthetic")
    m = s["maintenance"]
    c0, c1 = arr(m["c0"]), arr(m["c1"])
    w = m["weight_kg"]
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.plot(c0, c1, "o", color=GREEN, alpha=0.16, markersize=3.2, markeredgewidth=0, zorder=2)
    lo, med, hi = m["interval"]
    xs = np.linspace(np.nanpercentile(c0, 0.2), np.nanpercentile(c0, 99.8), 50)
    for v in (lo, hi):
        ax.plot(xs, (v - xs) / w, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.plot(xs, (s["true_maintenance"] - xs) / w, color=RED, linewidth=1.8, zorder=3)
    ax.plot([s["true_c0"]], [s["true_c1"]], "D", color=RED, markersize=9, markeredgecolor="white",
            markeredgewidth=1.4, zorder=5)
    r = np.corrcoef(c0, c1)[0, 1]
    handles = [
        ax.plot([], [], "o", color=GREEN, alpha=0.5, markersize=5)[0],
        ax.plot([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 3)))[0],
        ax.plot([], [], color=RED, linewidth=1.8, marker="D", markersize=7)[0],
    ]
    ax.legend(
        handles,
        [f"posterior draws (correlation {r:+.2f})",
         f"maintenance 90% interval: {lo:.0f} to {hi:.0f} kcal/day",
         f"truth: $c_0 + c_1W$ = {s['true_maintenance']:.0f} kcal/day"],
        loc="upper right", fontsize=10.5,
    )
    style(ax, xlabel=r"intercept $c_0$ (kcal/day)", ylabel=r"slope $c_1$ (kcal/day per kg)")
    ax.set_ylim(np.nanpercentile(c1, 0.5), np.nanpercentile(c1, 99.5) * 1.12)
    ax.set_xlim(xs[0], xs[-1])
    save(fig, "ridge.svg")


def synthetic_fit():
    """120 synthetic days: scale trend against truth, fat mass against truth."""
    s = load("synthetic")
    t = np.arange(s["days"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), gridspec_kw={"wspace": 0.24})
    ax = axes[0]
    y, sc, sd = arr(s["obs_lb"]), arr(s["scale_lb"]), arr(s["scale_sd_lb"])
    weighins(ax, t, y, ms=4.5, label="weigh-in")
    ax.fill_between(t, sc - 1.96 * sd, sc + 1.96 * sd, color=GREEN, alpha=0.14, linewidth=0)
    ax.plot(t, sc, color=GREEN, zorder=4, label="estimate, 95% band")
    ax.plot(t, arr(s["true_scale_lb"]), color=RED, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5,
            label="hidden truth")
    ax.legend(loc="upper right")
    style(ax, xlabel="day", ylabel="scale weight (lb)",
          title=f"Trend error {s['trend_rmse_g']:.0f} g vs. {s['obs_sd_kg'] * 1000:.0f} g per reading")

    ax = axes[1]
    f, fsd = arr(s["fat_kg"]), arr(s["fat_sd_kg"])
    ax.fill_between(t, f - 1.645 * fsd, f + 1.645 * fsd, color=GOLD, alpha=0.16, linewidth=0)
    ax.plot(t, f, color=GOLD, zorder=4, label="estimate, 90% band")
    ax.plot(t, arr(s["true_fat_kg"]), color=RED, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5,
            label="hidden truth")
    ax.legend(loc="upper right")
    style(ax, xlabel="day", ylabel="fat mass (kg)", title="Fat mass: weakly observed")
    save(fig, "synthetic-fit.svg")


# ------------------------------------------------------------------ control

def plan():
    """The plan being followed, from the last `wdm plan` run, to its end date.

    Intake as stacked macronutrient energy (the controller decides in weekly
    blocks, so the bands are steps), and the predicted scale weight it implies,
    continued from the current attempt's weigh-ins and filtered trend.
    """
    r = load("real")
    p, cur = r["followed_plan"], r["current"]
    d = dates(p["date"])
    P = np.array(p["protein_g"]) * 4
    C = np.array(p["carb_g"]) * 4
    F = np.array(p["fat_g"]) * 9
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.45], "hspace": 0.14})
    ax = axes[0]
    kw = dict(step="post", linewidth=0, edgecolor="white")
    ax.fill_between(d, 0, P, color=BLUE, label="protein", **kw)
    ax.fill_between(d, P, P + C, color=GOLD, label="carbohydrate", **kw)
    ax.fill_between(d, P + C, P + C + F, color=RED, label="fat", **kw)
    for edge in (P, P + C):  # a surface gap between stacked bands
        ax.step(d, edge, where="post", color="white", linewidth=1.5)
    style(ax, ylabel="kcal/day")
    ax.set_ylim(0, 1650)
    ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.2))
    label(ax, d[0], P[0] + C[0] + F[0], f"{p['kcal'][0]:.0f} kcal", dx=0, dy=8, ha="left",
          va="bottom")
    label(ax, d[-1], p["kcal"][-1], f"{p['kcal'][-1]:.0f} kcal: the floor", dx=0, dy=8,
          ha="right", va="bottom")

    ax = axes[1]
    hd = dates(cur["dates"])
    weighins(ax, hd, arr(cur["obs_lb"]), label="weigh-in")
    ax.plot(hd, arr(cur["scale_lb"]), color=GREEN, alpha=0.45)
    w = np.array(p["predicted_weight_lb"])
    ax.plot(d, w, color=GREEN, label="predicted scale weight")
    label(ax, d[-1], w[-1], f"{w[-1]:.0f} lb\n{d[-1].astype(dt.date):%b %-d, %Y}", dx=8)
    label(ax, hd[0], cur["obs_lb"][0], f"{cur['obs_lb'][0]:.1f} lb", MUTED, dx=8, dy=4,
          weight="normal")
    ax.legend(loc="upper right")
    style(ax, ylabel="weight (lb)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(hd[0] - np.timedelta64(2, "D"), d[-1] + np.timedelta64(18, "D"))
    save(fig, "plan.svg")


JOBS = [timeline, smoother, first_days, structure, forbes, timescales, parameters, ridge,
        synthetic_fit, plan]

if __name__ == "__main__":
    for job in JOBS:
        job()
