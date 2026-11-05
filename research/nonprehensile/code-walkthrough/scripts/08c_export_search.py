"""Step 8c: export a multi-round search run for the interactive viewer.

Runs the walkthrough's breadth-first search (the lightened solver of
08_search.py, batches of 4) for ROUNDS rounds and writes everything the viewer
needs to walkthrough/08c_search_run.json:

  * every search node: depth, qpos, the formation it stores, the bits actually
    in band at its pose, and every legal edit at that node (Cand.legal_ops);
  * every attempt, round by round: source node, edit, the search's verdict
    (every requested bit in band at the final step), the solver's own success
    checks recomputed from the trajectory, the 30-step qpos trajectory, and the
    node it created if accepted;
  * the size of the frontier (legal edits not yet tried) after each round.

    python walkthrough/08c_export_search.py          # 12 rounds, about 7 min
    python walkthrough/08c_export_search.py 6        # fewer rounds
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cmg.config.search_config import SearchConfig
from cmg.config.solver_config import SolverConfig
from cmg.physics.world import State
from cmg.search import Search

from _common import CMG, template_names

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

base = SolverConfig.from_yaml(str(CMG / "assets/configs/solver/step-base.yaml"))
solver_config = SolverConfig(
    solver_type=base.solver_type,
    solver_args=[],
    solver_overrides={**base.solver_overrides, "n_samples": 64, "maxiters": 2},
    scene_spec=base.scene_spec,
)
search = Search.from_specs(SearchConfig(batch_size=4), solver_config, base.scene_spec)
cand = search.candidate_formations
catalog, world, solver = cand.catalog, cand.exec_world, search.local_solver
names = [n.replace(":boundary", "") for n in template_names(base.scene_spec, catalog)]
N = len(names)
EPS_PEN, EPS_ON = float(catalog.epsilon_pen), float(catalog.epsilon_on)
gaps_along = jax.jit(jax.vmap(lambda q: catalog.gap(pose=world.pose(State(q, jnp.zeros_like(q))))))
band_at = jax.jit(lambda q: catalog.band(pose=world.pose(State(q, jnp.zeros_like(q)))))

# Record every batch the search sends to the solver, accepted or not.
calls = []
solve_batch = search._solve_batch


def recording(states, old, new, op_types):
    out = solve_batch(states, old, new, op_types)
    calls.append((np.asarray(states.configuration), np.asarray(old), np.asarray(new),
                  np.asarray(op_types), out))
    return out


search._solve_batch = recording

node_ids = {}   # id(SearchNode) -> int
nodes_out = []  # exported node records


def bits(mask):
    return [int(i) for i in np.flatnonzero(np.asarray(mask))]


def edit_label(old, new, op_type):
    made = [names[i] for i in np.flatnonzero(new & ~old)]
    broken = [names[i] for i in np.flatnonzero(old & ~new)]
    if op_type == 2 or not (made or broken):
        return "transport"
    return " ".join([f"make {m}" for m in made] + [f"break {b}" for b in broken])


def register_nodes():
    """Give new search nodes an id and export them with their legal edits."""
    for node in search.search_state.nodes.values():
        if id(node) in node_ids:
            continue
        node_ids[id(node)] = len(nodes_out)
        q = np.asarray(node.scene_state.configuration)
        formation = jnp.asarray(node.formation, dtype=jnp.bool_)
        valid, op_types = search._legal_ops(node.scene_state, formation=formation)
        legal = []
        for index in np.flatnonzero(np.asarray(valid)):
            destination = np.asarray(cand.apply_op(formation, int(index)))
            old = np.asarray(node.formation)
            legal.append(dict(op=int(index), label=edit_label(old, destination, int(op_types[index])),
                              dest=bits(destination)))
        nodes_out.append(dict(id=len(nodes_out), depth=int(node.stats.get("depth", 0)),
                              qpos=[round(float(x), 5) for x in q], stored=bits(node.formation),
                              actual=bits(band_at(jnp.asarray(q))), legal=legal, born=None))


def verdicts(q, old, new):
    g = np.asarray(gaps_along(jnp.asarray(q)))
    final = g[-1]
    in_band = (final >= -EPS_PEN) & (final < EPS_ON)
    missing = [dict(name=names[i], gap_mm=round(float(final[i]) * 1e3, 1))
               for i in np.flatnonzero(new & ~in_band)]
    reasons = []
    t, c = np.unravel_index(np.argmin(g), g.shape)
    if g[t, c] < -EPS_PEN:
        reasons.append(f"penetration {-g[t, c] * 1e3:.1f} mm on {names[c]}")
    deadline = g[solver.horizon - 1:]
    late = [names[i] for i in np.flatnonzero(new & np.any(deadline >= EPS_ON, axis=0))]
    if late:
        reasons.append("out of band after deadline: " + ", ".join(late))
    if np.any((old & ~new) & np.any(deadline < solver.R, axis=0)):
        reasons.append("dropped contact not cleared")
    drift = float(np.abs(q[solver.horizon:] - q[solver.horizon - 1]).max())
    if drift > solver.dwell_tolerance:
        reasons.append(f"dwell drift {drift * 1e3:.1f} mm")
    return missing, reasons


register_nodes()
rounds = []
for r in range(1, ROUNDS + 1):
    start = time.perf_counter()
    before_calls = len(calls)
    before_attempted = set(search.search_state.attempted)
    paths = search.run_single()
    new_pairs = [p for p in search.search_state.attempted if p not in before_attempted]
    register_nodes()
    attempts = []
    for q0, old, new, op_types, (traj, _, accepted) in calls[before_calls:]:
        q = np.asarray(traj.configuration)
        for i in range(len(new)):
            # Match the recorded row to the (node, formation) pair the search tried.
            source = next(node for node, f in new_pairs
                          if tuple(bool(x) for x in new[i]) == f
                          and tuple(node.formation) == tuple(bool(x) for x in old[i])
                          and np.allclose(np.asarray(node.scene_state.configuration), q0[i]))
            child = None
            if accepted[i]:
                path = next(p for p in paths if p.prev_node is source
                            and tuple(p.cur_node.formation) == tuple(bool(x) for x in new[i]))
                child = node_ids[id(path.cur_node)]
                if nodes_out[child]["born"] is None:
                    nodes_out[child]["born"] = r
            missing, reasons = verdicts(q[i], old[i], new[i])
            attempts.append(dict(
                node=node_ids[id(source)], label=edit_label(old[i], new[i], int(op_types[i])),
                dest=bits(new[i]), accepted=bool(accepted[i]), child=child,
                missing=missing, solver=reasons,
                traj=[[round(float(x), 4) for x in row] for row in q[i]],
            ))
    frontier = 0
    for node in search.search_state.nodes.values():
        record = nodes_out[node_ids[id(node)]]
        frontier += sum((node, tuple(bool(i in leg["dest"]) for i in range(N)))
                        not in search.search_state.attempted for leg in record["legal"])
    rounds.append(dict(round=r, attempts=attempts, frontier=frontier,
                       seconds=round(time.perf_counter() - start, 1)))
    print(f"round {r:2d}: {sum(a['accepted'] for a in attempts)}/{len(attempts)} accepted, "
          f"{len(nodes_out)} nodes, frontier {frontier}, {rounds[-1]['seconds']:.0f} s", flush=True)

for record in nodes_out:
    if record["born"] is None:
        record["born"] = 0
run = dict(
    names=names,
    scene=dict(a=0.125, h=0.03125, r=0.015625, eps_on=EPS_ON, eps_pen=EPS_PEN,
               horizon=int(solver.horizon), dwell=int(solver.dwell_steps), dt=0.01),
    solver=dict(n_samples=64, maxiters=2, batch_size=4),
    nodes=nodes_out, rounds=rounds,
)
out = Path(__file__).with_name("08c_search_run.json")
out.write_text(json.dumps(run, separators=(",", ":")))
print(f"Wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
