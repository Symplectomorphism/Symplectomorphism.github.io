"""Step 4: the contact graph (catalog-level edges).

ContactGraph builds two edge types over templates: transfer-adjacent (same
body pair, features adjacent on both bodies) and shared-feature (the two
templates share a feature). Count them, and look at what the edge list
actually stores.
"""
from collections import Counter

from cmg.catalog.graph import ContactGraph, EdgeTypes

from _common import STEP_YAML, load_spec, template_names

graph = ContactGraph.from_yaml(str(STEP_YAML))
catalog = graph.catalog
names = template_names(load_spec(), catalog)

print("Edges by type:", dict(Counter(EdgeTypes(t).name for _, _, t in graph.edges)))
print("First three stored edges:", graph.edges[:3])
print("Has an excluded-edge list:", hasattr(graph, "edges_excluded"))

# Recompute with catalog positions so the edges can be read.
index = {id(t): i for i, t in enumerate(catalog)}
first = names.index("B:zn-L:boundary")
print(f"\nNeighbours of {names[first]} (template {first}):")
for t1, t2 in ((a, b) for a in catalog for b in catalog if a is not b):
    if index[id(t1)] != first:
        continue
    kinds = []
    if catalog.transfer_adjacent(t1, t2):
        kinds.append("transfer-adjacent")
    if set(t1.key()) & set(t2.key()):
        kinds.append("shared-feature")
    if kinds:
        print(f"  {names[index[id(t2)]]:18s} {', '.join(kinds)}")
