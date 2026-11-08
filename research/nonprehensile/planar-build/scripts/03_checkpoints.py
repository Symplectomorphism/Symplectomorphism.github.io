"""Checkpoint 1 (step 3): how much does H prune? Runs Cand (c-gnn phase1/planar)
on the chain's starts and on nodes from the hand-moved demo trajectory
(every 25th frame, plus jittered copies projected back onto their formation),
and counts, among candidate edits that are edges of the marking graph and pass
Near, how many H rejects at chi_0 (step 2) and with the transition sweep
(Decision 10). Writes JSON.

    python 03_checkpoints.py --out media/step3_checkpoints.json [--cpu]
"""
import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
args.out = os.path.abspath(args.out)
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"
CMG = os.path.expanduser("~/repos/research/c-gnn/cmg")
sys.path.insert(0, os.path.join(CMG, "tests"))
os.chdir(CMG)

import jax
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.marking.candidate import Cand
from cmg.marking.edits import TYPE_NAMES
from cmg.physics.world import ExecutionWorld
from cmg.scene.spec import SceneSpec
from cmg.storage import resolve_path
from test_planar_step2 import A, FLAT, FLUSH, H, LANDED, R, disk_on, lifted, pivot

spec = SceneSpec.from_yaml(resolve_path("src/cmg/scene/families/step2d.yaml"))
cat = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
cand = Cand.from_spec(spec, cat, world)
gates = jax.jit(lambda q: cand.gate_report(config=q))
band = jax.jit(lambda q: cat.band(config=q))
ties = jax.jit(lambda q: cat.tie_breaks(config=q))
project = jax.jit(lambda q, target: cand.sweep.project(q, q, target, jnp.ones(q.shape[0], bool)))
types = np.asarray(cand.edits.types)

p30, p45, l60 = pivot(math.radians(30)), pivot(math.radians(45)), lifted(math.radians(60), 0.03)
chain = [FLAT + [FLAT[0] - A / 2 - R - 0.01, A / 2], FLAT + disk_on(FLAT, "xn"), FLUSH + disk_on(FLUSH, "xn"),
         p30 + disk_on(p30, "xn", -0.04), p45 + disk_on(p45, "xn", -0.04), l60 + disk_on(l60, "xn", -0.04),
         LANDED + [A / 4, H + A + R], LANDED + [-A / 4 - R - 0.01, H + A / 2], LANDED + [-A / 4 - R, H + A / 2]]
demo = np.load(resolve_path("src/cmg/scene/families/step_example_trajectory.npz"))["qpos"].astype(np.float32)
rng = np.random.default_rng(1)
nodes = []
for q in demo[::25]:
    f = band(jnp.asarray(q))
    if int(ties(jnp.asarray(q))) == 0 and int(np.sum(np.asarray(f))) > 0:
        nodes.append(q)
    for _ in range(2):
        qq = q + rng.normal(0, [0.004, 0.004, 0.03, 0.004, 0.004]).astype(np.float32)
        qq, err = project(jnp.asarray(qq), f)
        if float(err) < 1e-4 and np.array_equal(np.asarray(band(qq)), np.asarray(f)) and int(ties(qq)) == 0:
            nodes.append(np.asarray(qq))


def tally(qs):
    out = {"nodes": 0, "candidates": 0, "rejected_at_pose": 0, "rejected_sweep": 0, "overflow": 0,
           "by_type": {}}
    t = time.perf_counter()
    for q in qs:
        r = gates(jnp.asarray(q, jnp.float32))
        cheap = np.asarray(r["edge"] & r["near"])
        at, sw = np.asarray(r["hold_at_pose"]), np.asarray(r["hold"])
        out["nodes"] += 1
        out["candidates"] += int(cheap.sum())
        out["rejected_at_pose"] += int((cheap & ~at).sum())
        out["rejected_sweep"] += int((cheap & ~sw).sum())
        out["overflow"] += int(r["overflow"])
        for k in np.unique(types[cheap]):
            d = out["by_type"].setdefault(TYPE_NAMES[int(k)], [0, 0, 0])
            sel = cheap & (types == k)
            d[0] += int(sel.sum()); d[1] += int((sel & ~at).sum()); d[2] += int((sel & ~sw).sum())
    out["seconds_per_node"] = round((time.perf_counter() - t) / max(1, len(qs)), 3)
    return out


gates(jnp.asarray(chain[0], jnp.float32))      # compile
result = {"backend": jax.default_backend(), "chain": tally(chain), "demo_nodes": tally(nodes)}
for part in ("chain", "demo_nodes"):
    r = result[part]
    r["fraction_rejected_at_pose"] = round(r["rejected_at_pose"] / max(1, r["candidates"]), 3)
    r["fraction_rejected_sweep"] = round(r["rejected_sweep"] / max(1, r["candidates"]), 3)
with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "by_type"})
                  for k, v in result.items()}, indent=1))
