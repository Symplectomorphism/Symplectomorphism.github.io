"""Background tutorial (hold test as an LP): linrax's feasibility LP vs the exact test.

Same cases and float64 HiGHS reference as the step-2 hold benchmark (1,000
random, 416 within 2 mm of disk height of the support boundary). Needs
linrax importable (pip install linrax, or its source on PYTHONPATH). Run from
c-gnn/cmg on branch phase1/planar:

    JAX_PLATFORMS=cpu .venv314/bin/python <this file>          # float32, CPU
    python <this file>                                         # float32, GPU
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv314/bin/python <this file>   # float64
"""
import json, math, os, sys, time, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "tests")
import linrax
from cmg.physics.world import State
from test_planar_step2 import _reference, pivot, lifted, disk_on, A, H, FLAT, FAR
from cmg.catalog.templates import Catalog
from cmg.scene.spec import SceneSpec
from cmg.physics.world import ExecutionWorld
from cmg.physics.planar_hold import PlanarHold
from cmg.physics.hold import create_predicates
spec = SceneSpec.from_yaml('src/cmg/scene/families/step2d.yaml'); cat = Catalog.from_spec(spec); world = ExecutionWorld(spec.execution)
pair = np.asarray(cat._pair); rng = np.random.default_rng(10)
demo = np.load('src/cmg/scene/families/step_example_trajectory.npz')['qpos'].astype(np.float32)
band = jax.jit(lambda q: cat.band(config=q)); nonempty = jax.jit(lambda q: cat.query(config=q).nonempty)
# random cases
cases = []
while len(cases) < 1000:
    q = demo[rng.integers(len(demo))].copy()
    if rng.random() < 0.7: q[3:5] = q[:2] + rng.normal(0, 0.06, 2)
    q += rng.normal(0, [0.002, 0.002, 0.02, 0, 0]).astype(np.float32)
    f = np.asarray(band(jnp.asarray(q))) & (rng.random(len(cat)) < 0.8); ne = np.asarray(nonempty(jnp.asarray(q)))
    for p in range(4):
        if rng.random() < 0.25 and ne[pair == p].any(): f[pair == p] = False; f[rng.choice(np.flatnonzero((pair == p) & ne))] = True
    cases.append(("random", q, f))
# near-boundary cases: bisect the disk height at which {m_x, s_v} stops holding, then sample within +-2 mm
keep = np.zeros(len(cat), bool); keep[[cat.index("m_x"), cat.index("s_v")]] = True
for box_of in (pivot, lambda th: lifted(th, 0.03)):
    for deg in np.linspace(8, 70, 16):
        box = box_of(math.radians(deg)); qf = lambda mid: np.asarray(box + disk_on(box, "xn", mid), np.float32)
        lo, hi = -A/2 + 0.002, A/2 - 0.002
        if _reference(cat, world, qf(lo), keep) == _reference(cat, world, qf(hi), keep): continue
        a, b = lo, hi; va = _reference(cat, world, qf(a), keep)
        for _ in range(30):
            m = (a + b) / 2
            if _reference(cat, world, qf(m), keep) == va: a = m
            else: b = m
        for d in np.linspace(-0.002, 0.002, 13): cases.append(("boundary", qf((a + b) / 2 + d), keep))
print("cases:", {k: sum(c[0] == k for c in cases) for k in ("random", "boundary")}, "backend:", jax.default_backend())
truth = np.array([_reference(cat, world, q, f) for _, q, f in cases])
Q = jnp.asarray(np.stack([c[1] for c in cases])); F = jnp.asarray(np.stack([c[2] for c in cases]))

hold = PlanarHold(spec, cat, world, method="exact")
def problem(q, f):
    state = State(configuration=q, velocity=jnp.zeros_like(q))
    pose = world.pose(state); query = cat.query(pose=pose)
    points, normals, mu, sign, manip, on, count_ok = hold._slots(f, f, query)
    com3 = world.center_of_mass(state)[hold.object]; com = jnp.stack((com3[0], com3[2]))
    gens = hold._generators(points, normals, mu, sign, com).reshape(-1, 3)
    active = jnp.repeat(on, 4)
    need = hold._needs(state)[0]
    return jnp.where(active[None, :], gens.T, 0.0), need, jnp.all(count_ok)
def lx_feas(q, f):
    A, need, ok = problem(q, f)
    sol, st = linrax.linprog(jnp.zeros(A.shape[1], A.dtype), A_eq=A, b_eq=need)
    return ok & st.feasible[0]
def lx_elastic(q, f):
    A, need, ok = problem(q, f)
    k = A.shape[1]
    Ae = jnp.hstack((A, jnp.eye(3, dtype=A.dtype), -jnp.eye(3, dtype=A.dtype)))
    c = jnp.concatenate((jnp.zeros(k, A.dtype), jnp.ones(6, A.dtype)))
    sol, st = linrax.linprog(c, A_eq=Ae, b_eq=need)
    return ok & st.success[0] & (sol.fun <= 1e-5 * jnp.maximum(1.0, jnp.linalg.norm(need)))
methods = {"exact (certified)": hold.supported, "linrax feasibility": lx_feas, "linrax elastic": lx_elastic}
tag = jax.default_backend() + ("-f64" if jax.config.jax_enable_x64 else "-f32")
out = {}
for name, fn in methods.items():
    run = jax.jit(jax.vmap(fn))
    t = time.perf_counter(); r = np.asarray(run(Q[:256], F[:256])); compile_s = time.perf_counter() - t
    ts = []
    for _ in range(3):
        t = time.perf_counter(); r = np.asarray(run(Q[:256], F[:256])); ts.append(time.perf_counter() - t)
    n = len(Q) // 256 * 256
    res = np.concatenate([np.asarray(run(Q[i:i+256], F[i:i+256])) for i in range(0, n, 256)] + [np.asarray(jax.jit(jax.vmap(fn))(Q[n:], F[n:]))])
    kinds = np.array([c[0] for c in cases])
    row = {"compile_s": round(compile_s, 1), "ms_per_256": round(1e3 * min(ts), 1)}
    for k in ("random", "boundary"):
        m = kinds == k
        row[k] = {"agree": f"{int((res[m] == truth[m]).sum())}/{int(m.sum())}",
                  "false_reject": int((~res[m] & truth[m]).sum()), "false_accept": int((res[m] & ~truth[m]).sum())}
    out[name] = row; print(tag, name, json.dumps(row), flush=True)
