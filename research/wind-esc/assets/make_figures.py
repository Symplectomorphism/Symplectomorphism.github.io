#!/usr/bin/env python3
"""Regenerate every figure used by the wind-turbine gradient-estimation deck.

    python make_figures.py

Needs numpy and matplotlib (the site venv has both). Output is SVG with
`svg.fonttype: none`, so the slide's own font is used and the files stay small.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fig_esc      # noqa: E402
import fig_three    # noqa: E402
import fig_zap      # noqa: E402
import fig_open     # noqa: E402
import fig_setup    # noqa: E402
import fig_sim      # noqa: E402
import fig_wind     # noqa: E402
import fig_rls      # noqa: E402

JOBS = [
    (fig_setup.regions, "regions.svg"),
    (fig_setup.cp_curve, "cp-curve.svg"),
    (fig_setup.blind, "blind.svg"),
    (fig_setup.ray, "ray.svg"),
    (fig_setup.rest, "rest.svg"),
    (fig_esc.esc_loop, "esc-loop.svg"),
    (fig_esc.logfix, "logfix.svg"),
    (fig_esc.evidence, "evidence.svg"),
    (fig_esc.piesc, "piesc.svg"),
    (fig_esc.piesc_loop, "piesc-loop.svg"),
    (fig_open.estimand, "estimand.svg"),
    (fig_open.channels, "channels.svg"),
    (fig_open.biasvar, "biasvar.svg"),
    (fig_open.stability, "stability.svg"),
    (fig_open.identify, "identify.svg"),
    (fig_open.hessian, "hessian.svg"),
    (fig_open.bound, "bound.svg"),
    (fig_zap.dictionary, "zap-dictionary.svg"),
    (fig_zap.harmonics, "zap-harmonics.svg"),
    (fig_zap.demod, "zap-demod.svg"),
    (fig_zap.newton_rate, "zap-rate.svg"),
    (fig_zap.conditioning, "zap-conditioning.svg"),
    (fig_zap.relerr, "zap-relerr.svg"),
    (fig_zap.bandwidth, "zap-bandwidth.svg"),
    (fig_zap.timescales, "zap-timescales.svg"),
    (fig_zap.probes, "zap-probes.svg"),
    (fig_zap.spread, "zap-spread.svg"),
    (fig_zap.multiplex, "zap-multiplex.svg"),
    (fig_zap.sidon, "zap-sidon.svg"),
    (fig_wind.kaimal, "wind-kaimal.svg"),
    (fig_wind.window, "wind-window.svg"),
    (fig_wind.roadmap, "wind-roadmap.svg"),
    (fig_wind.validate, "wind-validate.svg"),
    (fig_wind.design, "wind-design.svg"),
    (fig_sim.sim_gate, "sim-gate.svg"),
    (fig_sim.sim_curvature, "sim-curvature.svg"),
    (fig_sim.sim_rolling, "sim-rolling.svg"),
    (fig_sim.sim_averaging, "sim-averaging.svg"),
    (fig_sim.sim_period, "sim-period.svg"),
    (fig_rls.phase, "rls-phase.svg"),
    (fig_rls.identity, "rls-identity.svg"),
    (fig_rls.channels, "rls-channels.svg"),
    (fig_rls.calibration, "rls-calibration.svg"),
    (fig_rls.schedule, "rls-schedule.svg"),
    (fig_three.anomaly, "q1-anomaly.svg"),
    (fig_three.budget, "q1-budget.svg"),
]


def main():
    for fn, name in JOBS:
        out = os.path.join(HERE, name)
        fn(out)
        print(f"  {name:<16} {os.path.getsize(out)/1024:6.1f} kB")
    print(f"\n{len(JOBS)} figures written to {HERE}")


if __name__ == "__main__":
    main()
