"""Ties under candidate values of epsilon_pen (step 3). The jittered-demo stress
test of step 2 (4,000 poses, the same jitter and seed as test_tie_stress) with
the scene's epsilon_pen replaced. Writes JSON.

    python 03_ties_eps_pen.py --out media/step3_ties.json --values 2,3,4,5,10 [--cpu]
"""
import argparse
import copy
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--values", default="2,3,4,5,10", help="mm")
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
args.out = os.path.abspath(args.out)
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"
CMG = os.path.expanduser("~/repos/research/c-gnn/cmg")
os.chdir(CMG)

import jax
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.scene.spec import SceneSpec
from cmg.storage import resolve_path

base = SceneSpec.from_yaml(resolve_path("src/cmg/scene/families/step2d.yaml"))
rng = np.random.default_rng(0)
demo = np.load(resolve_path("src/cmg/scene/families/step_example_trajectory.npz"))["qpos"].astype(np.float32)
q = demo[rng.integers(0, len(demo), 4000)] + rng.normal(0, [0.003, 0.003, 0.03, 0.003, 0.003], (4000, 5)).astype(np.float32)
out = {"poses": len(q), "eps_pen_mm": {}}
for mm in [float(v) for v in args.values.split(",")]:
    spec = copy.deepcopy(base)
    spec.execution.epsilon_pen = mm / 1000.0
    cat = Catalog.from_spec(spec)
    ties = np.asarray(jax.jit(jax.vmap(lambda c: cat.tie_breaks(config=c)))(jnp.asarray(q)))
    phi = np.asarray(jax.jit(jax.vmap(lambda c: cat.gap(config=c)))(jnp.asarray(q)))
    deep = (phi.min(axis=1) < -cat.epsilon_on)
    out["eps_pen_mm"][str(mm)] = {"ties": int((ties > 0).sum()), "ties_with_contact_deeper_than_eps_on":
                                  int(((ties > 0) & deep).sum())}
    print(mm, out["eps_pen_mm"][str(mm)], flush=True)
with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
