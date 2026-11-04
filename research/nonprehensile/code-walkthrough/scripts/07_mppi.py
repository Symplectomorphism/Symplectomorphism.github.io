"""Step 7: one local-solver call (Lambda) with MPPI.

Realize a single make from the start pose: bring the disk into contact
with the box's back face (M:boundary-B:xn, the paper's m_x). The solver is
built from assets/configs/solver/step-base.yaml, exactly as the search
builds it.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cmg.config.solver_config import SolverConfig
from cmg.physics.world import State
from cmg.solver import create_local_solver

from _common import CMG, template_names

config = SolverConfig.from_yaml(str(CMG / "assets/configs/solver/step-base.yaml"))
solver = create_local_solver(config)
catalog, world = solver.catalog, solver.exec_world
names = template_names(config.scene_spec, catalog)
print("Solver:", type(solver).__name__, config.solver_overrides)

qpos = jnp.asarray(config.scene_spec.start_qpos)
state = State(qpos, jnp.zeros_like(qpos), world._state_data(State(qpos, jnp.zeros_like(qpos))))
old = catalog.band(world.pose(state))
m_x = names.index("M:boundary-B:xn")
new = old.at[m_x].set(True)

solve = jax.jit(solver)
for attempt in ("first call (compiles)", "second call (cached)"):
    start = time.perf_counter()
    result = solve(state, old, new, jnp.asarray(0))
    jax.block_until_ready(result.success)
    print(f"{attempt}: {time.perf_counter() - start:.1f} s")

np.set_printoptions(precision=4, suppress=True)
traj = result.traj.configuration
gaps = jax.vmap(lambda c: catalog.gap(pose=world.pose(State(c, jnp.zeros_like(c)))))(traj)
print(f"\nTrajectory: {traj.shape[0]} physics steps of {world.model.opt.timestep} s")
print("Start qpos:   ", np.asarray(qpos))
print("Terminal qpos:", np.asarray(traj[-1]))
print("Gap of m_x every 5 steps:", np.asarray(gaps[::5, m_x]))
print("Controls (first 9 steps):\n", np.asarray(result.ctrl[:9]))
print("\nsolver success flag:", bool(result.success))
worst = np.unravel_index(int(jnp.argmin(gaps)), gaps.shape)
print(f"  deepest gap {float(gaps[worst]):.4f} (limit -{catalog.epsilon_pen})"
      f" on {names[worst[1]]} at step {worst[0]}")
drift = jnp.abs(traj[solver.horizon:] - traj[solver.horizon - 1]).max()
print(f"  largest drift during dwell {float(drift):.4f} (limit {solver.dwell_tolerance})")
terminal_band = catalog.band(world.pose(State(traj[-1], jnp.zeros_like(traj[-1]))))
print("search would accept (destination bits in band at the end):",
      bool(jnp.all(~new | terminal_band)))
