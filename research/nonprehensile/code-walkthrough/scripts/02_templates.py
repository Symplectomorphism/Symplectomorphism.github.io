"""Step 2: templates and their gaps.

Build the catalog, list its templates, and evaluate every template's signed
gap at the 'flat' fixture: box resting on the floor 3.75 cm short of the
riser, disk 4.375 cm behind the box.
"""
import jax.numpy as jnp
import numpy as np

from cmg.catalog.templates import Catalog
from cmg.physics.world import ExecutionWorld, State

from _common import fixtures, load_spec, template_names

spec = load_spec()
catalog = Catalog.from_spec(spec)
world = ExecutionWorld(spec.execution)
names = template_names(spec, catalog)

qpos = jnp.array(fixtures()["flat"])
state = State(qpos, jnp.zeros_like(qpos))
pose = world.pose(state)  # (5 bodies, 7) = x, y, z, qw, qx, qy, qz

np.set_printoptions(precision=4, suppress=True)
print("qpos [box x, box z, box angle, disk x, disk z] =", np.asarray(qpos))
print("Body poses in scene-ID order (x, y, z, qw, qx, qy, qz):")
for body, row in zip(spec.bodies, np.asarray(pose)):
    print(f"  {body.name}: {row}")

gaps = np.asarray(catalog.gap(pose))
band = np.asarray(catalog.band(pose))
print(f"\n{'id':>2}  {'template':22s} {'class':22s} {'gap':>9}  band")
for i, (name, t, g, b) in enumerate(zip(names, catalog, gaps, band)):
    print(f"{i:>2}  {name:22s} {type(t).__name__:22s} {g:9.5f}  {'*' if b else ''}")

print(f"\nband = gap >= -{catalog.epsilon_pen} and gap < {catalog.epsilon_on}")
print("template_id stored on each template:", {t.template_id for t in catalog})
