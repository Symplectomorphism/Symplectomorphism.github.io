"""Step 8: the search loop, two batches.

Search.from_specs builds Cand, the solver and the discretizer, and seeds
the graph with one node at start_qpos. run_single() ranks every untried
(node, legal edit) pair by BFS depth, sends the best batch_size of them to
the solver in one vmapped call, and adds a node for each accepted result.
The solver is lightened (64 samples, 2 iterations, batch of 4) so the run
fits in a few minutes on a CPU.
"""
import tempfile
import time

import numpy as np

from cmg.config.search_config import SearchConfig
from cmg.config.solver_config import SolverConfig
from cmg.search import Search

from _common import CMG, on, template_names

base = SolverConfig.from_yaml(str(CMG / "assets/configs/solver/step-base.yaml"))
solver_config = SolverConfig(
    solver_type=base.solver_type,
    solver_args=[],
    solver_overrides={**base.solver_overrides, "n_samples": 64, "maxiters": 2},
    scene_spec=base.scene_spec,
)
search = Search.from_specs(SearchConfig(batch_size=4), solver_config, base.scene_spec)
names = template_names(base.scene_spec, search.candidate_formations.catalog)


def show_nodes():
    for key, node in search.search_state.nodes.items():
        depth = node.stats.get("depth", 0)
        print(f"  depth {depth}  qpos {np.round(np.asarray(node.scene_state.configuration), 4)}")
        print(f"           {on(np.array(node.formation), names)}")


print("Root:")
show_nodes()
root = next(iter(search.search_state.nodes.values()))
tried = []
for batch in (1, 2):
    start = time.perf_counter()
    paths = search.run_single()
    print(f"\nBatch {batch}: {len(paths)} accepted in {time.perf_counter() - start:.0f} s;"
          f" {len(search.search_state.attempted)} (node, formation) pairs tried so far")
    new = [f for node, f in search.search_state.attempted if node is root and f not in tried]
    tried.extend(new)
    added = [np.array(f) & ~np.array(root.formation) for f in new]
    for bits in sorted(added, key=lambda bits: int(np.argmax(bits))):
        print(f"  tried from root: make {on(bits, names)}")
    for path in paths:
        added = np.array(path.cur_node.formation) & ~np.array(path.prev_node.formation)
        dropped = np.array(path.prev_node.formation) & ~np.array(path.cur_node.formation)
        print(f"  + {on(added, names)}  - {on(dropped, names)}  ({len(path.path)} steps)")

print(f"\nGraph after two batches: {len(search.search_state.nodes)} nodes")
show_nodes()

with tempfile.TemporaryDirectory() as directory:
    search.save(directory)
    print("\nSaved:", sorted(p.name for p in __import__("pathlib").Path(directory).iterdir()))
    search.load(directory)
    print("Reloaded nodes:", len(search.search_state.nodes))
