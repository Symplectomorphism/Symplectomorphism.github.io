#!/usr/bin/env python3
"""Run the weight-dynamics package and save everything the deck plots.

    uv run --project ~/repos/glorious-beard/weight-dynamics/code python compute.py

This is the only step that needs the `wdm` package (JAX, Diffrax) and the
private observation log. It writes small JSON files to `data/`, and
`make_figures.py` draws the slides' figures from those with nothing but numpy
and matplotlib, so restyling a figure never needs the model or the data.

Four computations:

    real.json       the observation log: phase-1 noise fit, a local-linear-trend
                    smoother per episode, the joint filter, and the plan being
                    followed (the last `wdm plan` output, code/plan.csv)
    synthetic.json  the package's own end-to-end demo (known ground truth)
    timescales.json one simulated diet from the model at prior parameters
    forbes.json     the partition curve, nothing fitted
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from wdm.cli import _load_noise
from wdm.config import Config, Goal, Preferences, Safety, Subject
from wdm.constants import LB_PER_KG
from wdm.estimation import fit_joint, identify_noise
from wdm.filters.linear import local_linear_trend_matrices, rts_smoother, trend_filter
from wdm.io import load_episodes
from wdm.models.dynamics import simulate
from wdm.models.energetics import total_expenditure
from wdm.models.partition import forbes_fat_fraction
from wdm.models.types import DailyInput, Params, State
from wdm.synthetic import SyntheticSpec, make_episode, make_weight_only_episode

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
DEFAULT_DATA = Path.home() / "repos/glorious-beard/weight-dynamics/data/raw"


def _dates(d) -> list[str]:
    return [str(x)[:10] for x in np.asarray(d)]


def _list(a, nd=4) -> list:
    """Floats rounded for the JSON, NaN as null."""
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def _summaries(fit, truth: Params | None = None) -> list[dict]:
    out = []
    for s in fit.summaries:
        row = {
            "name": s.name,
            "median": s.median,
            "q05": s.q05,
            "q95": s.q95,
            "prior": s.prior_center,
        }
        if truth is not None:
            row["truth"] = float(getattr(truth, s.name))
        out.append(row)
    return out


def _maintenance_draws(fit, weight_lb: float, n: int = 4000) -> dict:
    """Posterior draws of (c0, c1), for the identifiability slide."""
    cov = 0.5 * (fit.eta_cov + fit.eta_cov.T) + 1e-10 * np.eye(fit.eta_cov.shape[0])
    chol = jnp.asarray(np.linalg.cholesky(cov))
    z = jax.random.normal(jax.random.key(1), (n, cov.shape[0]))
    eta = jnp.asarray(fit.eta_mean) + z @ chol.T
    p = jax.vmap(fit.transform.constrain)(eta)
    return {
        "c0": _list(p.rmr_intercept, 1),
        "c1": _list(p.rmr_slope, 3),
        "weight_kg": weight_lb / LB_PER_KG,
        "interval": fit.maintenance_interval_kcal(weight_lb),
    }


# --------------------------------------------------------------------- real data

def real(data_dir: Path) -> dict:
    episodes = load_episodes(data_dir / "observations.csv")
    cfg = Config.from_toml(data_dir / "subject.toml")

    ni = identify_noise(episodes)
    noise = {
        "obs_sd_kg": ni.obs_sd,
        "level_sd_kg": ni.level_sd,
        "slope_sd_kg": ni.slope_sd,
        "gain": ni.steady_state_gain,
        "window_days": ni.effective_window_days,
        "n_obs": ni.n_observations,
        "n_episodes": ni.n_episodes,
    }

    # Phase 1's own model, smoothed per episode: the trend a weight-only
    # history supports, with no food log involved.
    f, _ = local_linear_trend_matrices()
    trends = []
    for ep in episodes:
        y = jnp.asarray(np.nan_to_num(ep.weight_kg))
        o = jnp.asarray(ep.weight_observed, dtype=float)
        res = trend_filter(y, o, ni.level_sd, ni.slope_sd, ni.obs_sd)
        sm = rts_smoother(res, f)
        obs = np.where(ep.weight_observed, ep.weight_kg * LB_PER_KG, np.nan)
        trends.append(
            {
                "name": ep.name,
                "dates": _dates(ep.dates),
                "obs_lb": _list(obs, 2),
                "filtered_lb": _list(np.asarray(res.filtered_mean)[:, 0] * LB_PER_KG, 3),
                "smoothed_lb": _list(np.asarray(sm.mean)[:, 0] * LB_PER_KG, 3),
                "smoothed_sd_lb": _list(np.asarray(sm.sd)[:, 0] * LB_PER_KG, 3),
                "slope_lb_per_wk": _list(np.asarray(sm.mean)[:, 1] * LB_PER_KG * 7, 3),
            }
        )

    fit = fit_joint(episodes, cfg, _load_noise(ni.obs_sd))
    last = fit.episodes[-1]
    last_ep = [e for e in episodes if e.name == last.name][0]
    sd = last.state_sd()
    trend_lb = float(last.scale_trend_kg()[-1] * LB_PER_KG)
    current = {
        "name": last.name,
        "dates": _dates(last.dates),
        "obs_lb": _list(np.where(last_ep.weight_observed, last_ep.weight_kg * LB_PER_KG, np.nan), 2),
        "scale_lb": _list(last.scale_trend_kg() * LB_PER_KG, 3),
        "tissue_lb": _list(last.tissue_kg() * LB_PER_KG, 3),
        "tissue_sd_lb": _list(sd[:, 0] * LB_PER_KG, 3),
        "fat_kg": _list(last.fat_kg(), 3),
        "fat_sd_kg": _list(sd[:, 1], 3),
        "gly_lb": _list(np.asarray(last.smoothed.mean)[:, 2] * LB_PER_KG, 3),
        "adapt_kcal": _list(last.adaptation_kcal(), 1),
        "kcal": _list(np.where(last_ep.intake_known, last_ep.energy_kcal, np.nan), 0),
        "trend_lb": trend_lb,
        "maintenance_kcal": fit.maintenance_kcal(trend_lb),
        "weight_stable_kcal": fit.weight_stable_intake_kcal(trend_lb),
    }

    # The plan actually being followed, as last written by `wdm plan --out`.
    followed = None
    plan_csv = data_dir.parent.parent / "code" / "plan.csv"
    if plan_csv.exists():
        import csv

        with plan_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        followed = {k: [r[k] if k == "date" else float(r[k]) for r in rows] for k in rows[0]}

    return {
        "generated": dt.date.today().isoformat(),
        "followed_plan": followed,
        "noise": noise,
        "trends": trends,
        "params": _summaries(fit),
        "current": current,
        "maintenance": _maintenance_draws(fit, trend_lb),
    }


# ----------------------------------------------------------------- synthetic demo

def synthetic(days: int = 120) -> dict:
    """The `wdm demo` subject, verbatim, so the deck matches the package."""
    subj = Subject(sex="male", age_years=40, height_cm=175.0, body_fat_frac=0.30)
    cfg = Config(
        subject=subj,
        goal=Goal(target_weight_lb=170.0, target_date=dt.date(2026, 12, 31)),
        safety=Safety(),
        preferences=Preferences(),
    )
    c0, c1 = subj.expenditure_prior(100.0)
    true = Params(
        rmr_intercept=jnp.asarray(c0 * 1.18),
        rmr_slope=jnp.asarray(c1 * 0.90),
        at_gain=jnp.asarray(0.22),
        at_tau=jnp.asarray(11.0),
        gly_gain=jnp.asarray(0.0042),
        gly_tau=jnp.asarray(2.0),
        na_gain=jnp.asarray(1.5e-5),
        na_tau=jnp.asarray(1.0),
        partition_kappa=jnp.asarray(0.06),
    )
    past = [
        make_weight_only_episode(
            SyntheticSpec(n_days=40, start_weight_kg=105.0),
            true, name=f"past-0{i}", start_date=d, seed=10 + i,
        )
        for i, d in enumerate(["2023-03-01", "2024-01-15"], start=1)
    ]
    current, truth = make_episode(
        SyntheticSpec(n_days=days, start_weight_kg=100.0),
        true, name="current", start_date="2026-08-03", seed=7,
    )
    episodes = [*past, current]
    ni = identify_noise(episodes)
    fit = fit_joint(episodes, cfg, _load_noise(ni.obs_sd))
    last = fit.episodes[-1]
    sd = last.state_sd()
    w_end_lb = float(last.scale_trend_kg()[-1] * LB_PER_KG)

    true_maint = float(true.rmr_intercept + true.rmr_slope * w_end_lb / LB_PER_KG)
    return {
        "days": days,
        "obs_sd_kg": ni.obs_sd,
        "obs_lb": _list(np.where(current.weight_observed, current.weight_kg * LB_PER_KG, np.nan), 2),
        "kcal": _list(np.where(current.intake_known, current.energy_kcal, np.nan), 0),
        "scale_lb": _list(last.scale_trend_kg() * LB_PER_KG, 3),
        "scale_sd_lb": _list(np.sqrt(sd[:, 0] ** 2 + sd[:, 2] ** 2) * LB_PER_KG, 3),
        "true_scale_lb": _list(truth["scale_kg"] * LB_PER_KG, 3),
        "tissue_lb": _list(last.tissue_kg() * LB_PER_KG, 3),
        "true_tissue_lb": _list(np.asarray(truth["tissue_kg"]) * LB_PER_KG, 3),
        "fat_kg": _list(last.fat_kg(), 3),
        "fat_sd_kg": _list(sd[:, 1], 3),
        "true_fat_kg": _list(np.asarray(truth["fat_kg"]), 3),
        "adapt_kcal": _list(last.adaptation_kcal(), 1),
        "true_adapt_kcal": _list(np.asarray(truth["adapt_kcal"]), 1),
        "trend_rmse_g": float(np.sqrt(np.mean((last.scale_trend_kg() - truth["scale_kg"]) ** 2)) * 1000),
        "params": _summaries(fit, true),
        "maintenance": _maintenance_draws(fit, w_end_lb),
        "true_maintenance": true_maint,
        "true_c0": float(true.rmr_intercept),
        "true_c1": float(true.rmr_slope),
    }


# ----------------------------------------------------------- timescales and Forbes

def timescales(weight_lb: float = 235.0, fat_kg: float = 38.0) -> dict:
    """One diet, simulated at prior parameters.

    Two weeks at maintenance on 250 g/day of carbohydrate, then from day 0 a
    1000 kcal/day cut to the same total that also drops carbohydrate to 50 g.
    Carbohydrate is measured from the pre-diet level, which is what the water
    compartment relaxes away from.
    """
    w = weight_lb / LB_PER_KG
    subj = Subject(sex="male", age_years=40, height_cm=175.0)
    c0, c1 = subj.expenditure_prior(w)
    p = Params.prior(c0, c1)
    carb_ref = 250.0
    x0 = State(
        tissue_kg=jnp.asarray(w), fat_kg=jnp.asarray(fat_kg), gly_water_kg=jnp.asarray(0.0),
        na_water_kg=jnp.asarray(0.0), adapt_kcal=jnp.asarray(0.0),
    )

    def day(kcal, protein, carb, fat):
        return DailyInput(
            energy_kcal=jnp.asarray(kcal), protein_g=jnp.asarray(protein),
            carb_dev_g=jnp.asarray(carb - carb_ref), fat_g=jnp.asarray(fat),
            alcohol_g=jnp.asarray(0.0), sodium_dev_mg=jnp.asarray(0.0),
            exercise_kcal=jnp.asarray(0.0), intake_known=jnp.asarray(1.0),
            macro_known=jnp.asarray(1.0),
        )

    # maintenance at x0 for a 25/45/30 protein/carb/fat split, by fixed point
    kcal = 3000.0
    for _ in range(30):
        u = day(kcal, 0.25 * kcal / 4, carb_ref, 0.30 * kcal / 9)
        kcal = float(total_expenditure(x0, u, p, carb_ref))
    protein = 0.25 * kcal / 4
    cut = kcal - 1000.0
    cut_fat = (cut - 4 * protein - 4 * 50.0) / 9

    pre, post = 14, 180
    rows = [day(kcal, protein, carb_ref, (kcal - 4 * protein - 4 * carb_ref) / 9)] * pre
    rows += [day(cut, protein, 50.0, cut_fat)] * post
    u = jax.tree.map(lambda *xs: jnp.stack(xs), *rows)
    traj = simulate(x0, u, p, carb_ref)

    t = np.arange(-pre + 1, post + 1)
    return {
        "day": t.tolist(),
        "maintenance_kcal": kcal,
        "cut_kcal": cut,
        "start_lb": weight_lb,
        "scale_lb": _list(np.asarray(traj.scale_kg) * LB_PER_KG, 3),
        "tissue_lb": _list(np.asarray(traj.tissue_kg) * LB_PER_KG, 3),
        "fat_kg": _list(np.asarray(traj.fat_kg), 3),
        "gly_lb": _list(np.asarray(traj.gly_water_kg) * LB_PER_KG, 3),
        "adapt_kcal": _list(np.asarray(traj.adapt_kcal), 1),
    }


def forbes() -> dict:
    f = np.linspace(5.0, 60.0, 221)
    return {"fat_kg": _list(f, 2), "p": _list(forbes_fat_fraction(jnp.asarray(f)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="directory with observations.csv")
    ap.add_argument("--only", nargs="*", choices=["real", "synthetic", "timescales", "forbes"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = {
        "real": lambda: real(args.data),
        "synthetic": synthetic,
        "timescales": timescales,
        "forbes": forbes,
    }
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        print(f"computing {name} ...", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(job(), indent=1, default=float))
    os.utime(OUT)


if __name__ == "__main__":
    main()
