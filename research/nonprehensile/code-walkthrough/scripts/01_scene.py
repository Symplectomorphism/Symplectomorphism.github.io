"""Step 1: the scene as data.

Load step.yaml into a SceneSpec, print what the loader derived from it, and
compare it with the MuJoCo model that the execution world will simulate.
"""
import mujoco

from _common import load_spec

spec = load_spec()

print("Bodies (scene ID, name, body set, geometry):")
for body in spec.bodies:
    kind, params = spec.local_geometry[body.id]
    print(f"  {body.id}  {body.name}  {body.body_set.value:5s}  {kind.value:6s} {params}")

print("\nFeature inventories:")
for body_id, features in spec.feature_inventories.items():
    print(f"  {spec.bodies[body_id].name}: {[f.value for _, f in features]}")

print("\nRelative pairs (derived: every pair that contains an object body):")
for pair in spec.relative_pairs:
    print("  ", sorted(spec.bodies[i].name for i in pair))

count = sum(
    len(spec.feature_inventories[a]) * len(spec.feature_inventories[b])
    for a, b in map(sorted, spec.relative_pairs)
)
print(f"\nTemplates = sum over pairs of |F(A)| x |F(A')| = {count}")
print("Exclusion table:", spec.exclusion_table)

e = spec.execution
print("\nExecution spec:")
print(f"  epsilon_pen={e.epsilon_pen}  epsilon_on={e.epsilon_on}  phi_gate={e.phi_gate}")
print(f"  eta={e.eta}  eta_theta={e.eta_theta}  timestep={e.timestep}")
print(f"  mjx_body_ids={e.mjx_body_ids}")

model = mujoco.MjModel.from_xml_path(e.reference)
print(f"\nMuJoCo model: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
print("  joints:", [model.joint(i).name for i in range(model.njnt)])
print("  bodies:", [model.body(i).name for i in range(model.nbody)])
print("  scene body -> MuJoCo body:",
      {b.name: model.body(i).name for b, i in zip(spec.bodies, e.mjx_body_ids)})

a = 2 * spec.local_geometry[1][1][0]
print(f"\nBox side a = {a}, disk radius r = {spec.local_geometry[0][1][0]} = a/8,"
      f" step height h = {model.body('P').pos[2]} = a/4")
print("start_qpos:", spec.start_qpos)
