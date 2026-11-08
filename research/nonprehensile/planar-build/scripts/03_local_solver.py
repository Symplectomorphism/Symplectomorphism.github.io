"""Step 3 experiment: Lambda's cost on the chain's edits (c-gnn phase1/planar).

Runs the real code: MPPI with the current cost (A), the paper's compiled cost
(B), B plus the terminal margin and dwell terms (C), and C plus the
transition subgoal of Decision 10 (D), on every edit of the Decision-9
chain from its canonical start and from perturbed starts (projected back
onto the formation, recognized, and passing Cand), plus break(g_v) from
30 deg, where the gate needs the transition sweep. Success is the shared
predicate (solver/success.py). Writes JSON (records and summary) and a
progress log (``done i of N``) for the ETA script.

    python 03_local_solver.py --out media/step3_experiment_gpu.json --log LOG [--cpu] [--starts 3]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--starts", type=int, default=3, help="perturbed starts per edit (besides the canonical one)")
ap.add_argument("--variants", default="ABCDE")
ap.add_argument("--next", type=int, default=1, help="run the next chain edit from successful ends")
args = ap.parse_args()
args.out = os.path.abspath(args.out)
args.log = os.path.abspath(args.log)
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

CMG = os.path.expanduser("~/repos/research/c-gnn/cmg")
sys.path.insert(0, os.path.join(CMG, "tests"))
os.chdir(CMG)

import jax
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.cost.compiler import CompiledCost, CostOptions
from cmg.marking.candidate import Cand
from cmg.physics.planar_hold import PlanarHold
from cmg.physics.world import ExecutionWorld, State
from cmg.scene.spec import SceneSpec
from cmg.solver.mppi import MPPISolver
from cmg.storage import resolve_path
from test_planar_step2 import A, FLAT, FLUSH, H, LANDED, R, disk_on, lifted, mask, pivot

LOG = open(args.log, "a")
T0 = time.time()


def log(msg):
    print(f"{time.time() - T0:8.1f}s {msg}", file=LOG, flush=True)


spec = SceneSpec.from_yaml(resolve_path("src/cmg/scene/families/step2d.yaml"))
cat = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
hold = PlanarHold(spec, cat, world)
cand = Cand.from_spec(spec, cat, world)
GOAL = (0.25, 0.09375, math.pi / 2)            # step.xml's target body (x, z, theta)
IDLE = jnp.asarray([0.0, 0.02 * 9.804])        # the disk's weight on its z motor
# assets/configs/solver/step2d-base.yaml, with the executed record lengthened from
# 20 + 10 to 60 + 20 steps (0.8 s) and the 30-step lookahead kept: at 0.3 s
# every variant failed the chain (dry run).
BASE = dict(horizon=60, dwell_steps=20, planning_horizon=30, n_samples=256, maxiters=5, noise_scale=0.1,
            band_alpha=0.1, band_weight=1000.0, control_hold=3)
COMMON = dict(hold=hold, goal=GOAL, goal_progress=A / 8)


def solver(variant):
    if variant == "A":
        return MPPISolver(cat, world, temperature=1e-3, **BASE, **COMMON)
    opts = {"B": CostOptions(), "C": CostOptions(margin=True, dwell=True),
            "D": CostOptions(margin=True, dwell=True, subgoal=True),
            # E: D with contact terms over eps_on (not l), a smooth terminal and a smaller barrier
            "E": CostOptions(margin=True, dwell=True, subgoal=True, scale="tolerance",
                             smooth_terminal=True, barrier=100.0)}[variant]
    cost = CompiledCost(cat, world, hold, options=opts, u_idle=IDLE, q0=spec.start_qpos, goal=GOAL)
    return MPPISolver(cat, world, temperature=0.1, temperature_mode="normalized", u_idle=IDLE,
                      cost=cost, **BASE, **COMMON)


p30, p45, l60 = pivot(math.radians(30)), pivot(math.radians(45)), lifted(math.radians(60), 0.03)
CHAIN = [  # (start, sigma, edit label, sigma')
    (FLAT + [FLAT[0] - A / 2 - R - 0.01, A / 2], ("g",), "make(m_x)", ("m_x", "g")),
    (FLAT + disk_on(FLAT, "xn"), ("m_x", "g"), "make(s)", ("m_x", "g", "s")),
    (FLUSH + disk_on(FLUSH, "xn"), ("m_x", "g", "s"), "compound(s -> s_v, g -> g_v)", ("m_x", "g_v", "s_v")),
    (p45 + disk_on(p45, "xn", -0.04), ("m_x", "g_v", "s_v"), "break(g_v)", ("m_x", "s_v")),
    (l60 + disk_on(l60, "xn", -0.04), ("m_x", "s_v"), "transfer(s_v -> t)", ("m_x", "t")),
    (LANDED + [A / 4, H + A + R], ("m_x", "t"), "break(m_x)", ("t",)),
    (LANDED + [-A / 4 - R - 0.01, H + A / 2], ("t",), "make(m_z)", ("m_z", "t")),
    (LANDED + [-A / 4 - R, H + A / 2], ("m_z", "t"), "transport", ("m_z", "t")),
]
EXTRA = [(p30 + disk_on(p30, "xn", -0.04), ("m_x", "g_v", "s_v"), "break(g_v)", ("m_x", "s_v"), "break(g_v) from 30 deg")]
labels = cand.edits.labels
edit_index = {lab: i for i, lab in enumerate(labels)}
legal = jax.jit(lambda q: cand.gate_report(config=q))
band = jax.jit(lambda q: cat.band(config=q))
ties = jax.jit(lambda q: cat.tie_breaks(config=q))
sweep = jax.jit(cand.sweep.__call__)
project = jax.jit(lambda q, target: cand.sweep.project(q, q, target, jnp.ones(q.shape[0], bool)))


def names(f):
    inv = {i: n for n, i in cat.named.items()}
    return tuple(sorted(inv.get(i, cat.names[i]) for i in np.flatnonzero(np.asarray(f))))


def starts_for(q0, sigma, label, rng, k):
    """Canonical start plus k perturbed ones, projected onto sigma, recognized, legal."""
    out = [np.asarray(q0, np.float32)]
    target = mask(cat, *sigma)
    tries = 0
    while len(out) < k + 1 and tries < 40 * (k + 1):
        tries += 1
        q = np.asarray(q0, np.float32) + rng.normal(0, [0.004, 0.0, 0.03, 0.004, 0.004]).astype(np.float32)
        q, err = project(jnp.asarray(q), target)
        if float(err) > 1e-4 or names(band(q)) != tuple(sorted(sigma)) or int(ties(q)) != 0:
            continue
        r = legal(q)
        i = edit_index[label]
        if bool(r["legal"][i]):
            out.append(np.asarray(q))
    return out


def run(variant, fn, q, sigma, label, new):
    st = State(configuration=jnp.asarray(q), velocity=jnp.zeros(len(q)))
    old_m, new_m = mask(cat, *sigma), mask(cat, *new)
    chi_t = None
    if variant in "DE":
        rep = sweep(st, old_m, new_m)
        chi_t = rep["chi_t"]
    t = time.perf_counter()
    res = fn(st, old_m, new_m, jnp.asarray(0), chi_t=chi_t) if chi_t is not None else fn(st, old_m, new_m, jnp.asarray(0))
    jax.block_until_ready(res)
    dt = time.perf_counter() - t
    poses = jax.vmap(world.pose)(res.traj)
    phis = jax.vmap(lambda p: cat.gap(pose=p))(poses)
    final = jax.tree.map(lambda x: x[-1], res.traj)
    margin = float(hold.supported_report(final, new_m)["margin"])
    cfg = res.traj.configuration
    window = cfg[-BASE["dwell_steps"]:]
    drift_box = float(jnp.max(jnp.abs(window[:, :3] - window[0, :3]) * jnp.asarray([1.0, 1.0, hold.length])))
    on_seq = jax.vmap(lambda p: cat._on_set(cat.query(pose=p))[0])(poses)
    reached = bool(jnp.any(jnp.all(on_seq == new_m[None], axis=1)))
    return {"success": bool(res.success), "min_phi": float(jnp.min(phis)), "margin": margin,
            "reached": reached, "drift_box_m": drift_box,
            "final_on": names(band(final.configuration)), "time": dt,
            "final": np.asarray(final.configuration).tolist()}, final


jobs = []
rng = np.random.default_rng(0)
cases = []
for i, (q0, sigma, label, new) in enumerate(CHAIN):
    for j, q in enumerate(starts_for(q0, sigma, label, rng, args.starts)):
        cases.append((i, f"{label}", "canonical" if j == 0 else f"perturbed {j}", q, sigma, label, new))
for q0, sigma, label, new, name in EXTRA:
    cases.append((3, name, "canonical", np.asarray(q0, np.float32), sigma, label, new))
total = len(cases) * len(args.variants)
log(f"backend {jax.default_backend()}: {len(cases)} cases x {len(args.variants)} variants = {total} calls (+ next edits)")
records = []
done = 0
for variant in args.variants:
    fn = jax.jit(solver(variant))
    for (ci, name, kind, q, sigma, label, new) in cases:
        rec, final = run(variant, fn, q, sigma, label, new)
        rec.update(variant=variant, edit=name, start=kind)
        # The next chain edit from the end state (same variant).
        rec["next"] = None
        if args.next and rec["success"] and ci + 1 < len(CHAIN):
            _, nsig, nlabel, nnew = CHAIN[ci + 1]
            qe = np.asarray(final.configuration)
            if names(band(qe)) == tuple(sorted(nsig)) and bool(legal(jnp.asarray(qe))["legal"][edit_index[nlabel]]):
                nrec, _ = run(variant, fn, qe, nsig, nlabel, nnew)
                rec["next"] = nrec["success"]
            else:
                rec["next"] = False          # not even offered from the end state
        records.append(rec)
        done += 1
        log(f"done {done} of {total}: {variant} {name} {kind} success={rec['success']} "
            f"min_phi={rec['min_phi']*1e3:.1f}mm t={rec['time']:.1f}s next={rec['next']}")

summary = {}
for variant in args.variants:
    rs = [r for r in records if r["variant"] == variant]
    succ = [r for r in rs if r["success"]]
    nxt = [r for r in succ if r["next"] is not None]
    summary[variant] = {
        "calls": len(rs), "success": len(succ),
        "success_canonical": sum(r["success"] for r in rs if r["start"] == "canonical"),
        "reached_sigma_prime": sum(r["reached"] for r in rs),
        "median_box_drift_mm": round(1e3 * float(np.median([r["drift_box_m"] for r in rs])), 2),
        "canonical": sum(r["start"] == "canonical" for r in rs),
        "max_penetration_mm": round(-1e3 * min(r["min_phi"] for r in rs), 2),
        "p95_penetration_mm": round(float(np.percentile([-1e3 * r["min_phi"] for r in rs], 95)), 2),
        "median_final_margin_success": (round(float(np.median([r["margin"] for r in succ])), 3) if succ else None),
        "next_success": f"{sum(bool(r['next']) for r in nxt)}/{len(nxt)}",
        "median_time_s": round(float(np.median([r["time"] for r in rs])), 2),
        "per_edit": {e: f"{sum(r['success'] for r in rs if r['edit'] == e)}/{sum(r['edit'] == e for r in rs)}"
                     for e in dict.fromkeys(r["edit"] for r in rs)},
    }
out = {"backend": jax.default_backend(), "variants": args.variants, "starts": args.starts,
       "summary": summary, "records": records}
with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
log("finished")
