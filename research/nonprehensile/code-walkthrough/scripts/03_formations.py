"""Step 3: formations (the band test) at named poses and along a trajectory.

The formation the search starts from is Catalog.band(pose): every template
whose gap lies in [-epsilon_pen, epsilon_on). Print it at each fixture in
tests/configs/step.txt, then count, along the hand-moved demo trajectory,
how many bits each body pair has on at once.
"""
import jax
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.physics.world import ExecutionWorld, State

from _common import TRAJECTORY, fixtures, load_spec, on, template_names

spec = load_spec()
catalog = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
names = template_names(spec, catalog)


def band_of(qpos):
    qpos = jnp.asarray(qpos)
    return catalog.band(world.pose(State(qpos, jnp.zeros_like(qpos))))


for name, qpos in fixtures().items():
    print(f"{name:16s} {on(band_of(qpos), names)}")

# Which body pair does each template belong to?
pair_of = [tuple(sorted((t.feature1.body(), t.feature2.body()))) for t in catalog]
pairs = sorted(set(pair_of), key=pair_of.index)
member = np.array([[p == q for q in pair_of] for p in pairs])  # (pairs, templates)

qpos = jnp.asarray(np.load(TRAJECTORY)["qpos"])
bands = np.asarray(jax.jit(jax.vmap(band_of))(qpos))  # (frames, templates)
per_pair = bands.astype(int) @ member.T.astype(int)   # (frames, pairs)

print(f"\nDemo trajectory: {len(qpos)} frames")
for k, (a, b) in enumerate(pairs):
    label = f"{spec.bodies[a].name}-{spec.bodies[b].name}"
    counts = np.bincount(per_pair[:, k], minlength=6)
    print(f"  {label}: frames with 0..5 bits on = {counts[:6].tolist()}")
print("Distinct formations visited:", len({tuple(r) for r in bands}))
