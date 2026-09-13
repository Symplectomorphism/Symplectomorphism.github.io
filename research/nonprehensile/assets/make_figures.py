#!/usr/bin/env python3
"""Regenerate every figure used by the nonprehensile-manipulation deck.

    python make_figures.py

Writes self-contained SVG (animation is SMIL, so nothing but a browser is
needed) next to this script.  No third-party dependencies.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fig_learn      # noqa: E402
import fig_solver     # noqa: E402
import fig_state      # noqa: E402
import fig_task       # noqa: E402

JOBS = [
    (fig_task.task_animation, "task.svg"),
    (fig_task.panel_strip, "strip.svg"),
    (fig_task.prior_work, "prior-work.svg"),
    (fig_state.overview, "overview.svg"),
    (fig_state.features, "features.svg"),
    (fig_state.template_gap, "template.svg"),
    (fig_state.catalog, "catalog.svg"),
    (fig_state.state, "state.svg"),
    (fig_state.actions, "actions.svg"),
    (fig_state.markgraph, "markgraph.svg"),
    (fig_state.walk, "walk.svg"),
    (fig_solver.solver, "solver.svg"),
    (fig_solver.gates, "gates.svg"),
    (fig_solver.support_theorem, "support.svg"),
    (fig_solver.normals_m3, "normals.svg"),
    (fig_learn.search_cost, "search.svg"),
    (fig_learn.contact_graph, "graph.svg"),
    (fig_learn.live_subgraph, "live.svg"),
    (fig_learn.ranker, "ranker.svg"),
    (fig_learn.greedy_misses, "greedy.svg"),
    (fig_learn.pipeline, "pipeline.svg"),
    (fig_learn.budget, "budget.svg"),
]


def main():
    for fn, name in JOBS:
        out = os.path.join(HERE, name)
        fn(out)
        size = os.path.getsize(out)
        print(f"  {name:<18} {size/1024:6.1f} kB")
    print(f"\n{len(JOBS)} figures written to {HERE}")


if __name__ == "__main__":
    main()
