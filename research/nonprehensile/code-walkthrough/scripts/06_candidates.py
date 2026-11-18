"""Step 6: candidate edits (Cand.legal_ops).

From a node's formation, every single-bit make and break is a candidate,
plus one transport (no change). A make or break is legal when it changes
the formation, the persisting contacts pass Supported (Hold), and every
added template is within phi_gate (Near). Print the legal edits at three
fixtures.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cmg.marking.candidate import Cand

from _common import fixtures, load_spec, on, template_names

spec = load_spec()
cand = Cand.from_spec(spec)
names = template_names(spec, cand.catalog)
n = len(names)
legal_ops = jax.jit(lambda q: cand.legal_ops(config=q))

for fixture in ("flat", "flat_stable_close", "toppled30_close"):
    qpos = jnp.array(fixtures()[fixture])
    start = time.perf_counter()
    ops, types = legal_ops(qpos)
    ops = np.asarray(ops)
    elapsed = time.perf_counter() - start
    formation = cand.catalog.band(config=qpos)
    print(f"== {fixture}  ({elapsed:.1f} s)")
    print("  formation:", on(formation, names))
    print("  make: ", on(ops[:n], names))
    print("  break:", on(ops[n:2 * n], names))
    print("  transport:", bool(ops[2 * n]))
