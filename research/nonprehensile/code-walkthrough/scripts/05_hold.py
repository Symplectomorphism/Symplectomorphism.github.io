"""Step 5: the static-hold LP (Supported and EnvStable).

create_predicates returns two jitted functions of (state, formation):
Supported admits every asserted contact, EnvStable drops the manipulator's.
Each solves one friction-cone LP for the box's wrench balance at the
asserted templates' gap midpoints and normals.
"""
import time

import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.physics.hold import create_predicates
from cmg.physics.world import ExecutionWorld, State

from _common import fixtures, load_spec, on, template_names

spec = load_spec()
catalog = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
names = template_names(spec, catalog)
# The same tolerance that marking/candidate.py uses inside the search.
supported, env_stable = create_predicates(spec, catalog, world, tolerance=1e-3)

print("Friction per body pair (MuJoCo's priority rule):")
for a, b in ((0, 1), (1, 2), (1, 3), (1, 4)):
    print(f"  {spec.bodies[a].name}-{spec.bodies[b].name}: mu = {world.pair_friction(a, b)}")
print("Box mass:", float(world.model.body_mass[1]), "kg")


def state_at(qpos):
    qpos = jnp.asarray(qpos)
    return State(qpos, jnp.zeros_like(qpos))


def mask(*wanted):
    return jnp.array([n in wanted for n in names])


start = time.perf_counter()
supported(state_at(fixtures()["flat"]), mask())
print(f"\nFirst call (compiles the LP): {time.perf_counter() - start:.1f} s")

print("\nBand formation at each fixture:")
print(f"  {'fixture':16s} Supported EnvStable")
for name, qpos in fixtures().items():
    state = state_at(qpos)
    band = catalog.band(world.pose(state))
    print(f"  {name:16s} {bool(supported(state, band))!s:9s} {bool(env_stable(state, band))}")

flat = state_at(fixtures()["flat"])
print("\nHand-picked formations at 'flat':")
for formation in (
    (),
    ("B:zn-L:boundary",),
    ("B:xn-L:boundary",),
    ("B:xp-L:boundary", "B:xn-L:boundary"),
):
    f = mask(*formation)
    print(f"  {on(f, names):40s} Supported={bool(supported(flat, f))}")

start = time.perf_counter()
for _ in range(20):
    supported(flat, mask("B:zn-L:boundary")).block_until_ready()
print(f"\nCompiled call: {(time.perf_counter() - start) / 20 * 1e3:.1f} ms each")
