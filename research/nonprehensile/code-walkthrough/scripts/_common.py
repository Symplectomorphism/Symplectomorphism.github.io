"""Shared helpers for the walkthrough scripts (not part of cmg).

Run every script from the cmg/ directory, for example
    python walkthrough/01_scene.py
"""
from pathlib import Path

import jax.numpy as jnp

from cmg.scene.spec import SceneSpec

CMG = Path(__file__).resolve().parents[1]
STEP_YAML = CMG / "src/cmg/scene/families/step.yaml"
TRAJECTORY = CMG / "src/cmg/scene/families/step_example_trajectory.npz"
FIXTURES = CMG / "tests/configs/step.txt"


def load_spec(path=STEP_YAML) -> SceneSpec:
    return SceneSpec.from_yaml(str(path))


def fixtures() -> dict[str, list[float]]:
    """Named qpos vectors from tests/configs/step.txt, in file order.

    qpos = [box x, box z, box angle about y, disk x, disk z].
    """
    named = {}
    for line in FIXTURES.read_text().splitlines():
        line = line.split("#", 1)[0]
        if ":" not in line:
            continue
        name, values = line.split(":", 1)
        named[name.strip()] = [float(v) for v in values.split(",")]
    return named


def template_names(spec: SceneSpec, catalog) -> list[str]:
    """Readable template names such as 'M:boundary-B:xn'."""
    body = {b.id: b.name for b in spec.bodies}
    return [
        f"{body[t.feature1.body()]}:{t.feature1.feature_type.value}"
        f"-{body[t.feature2.body()]}:{t.feature2.feature_type.value}"
        for t in catalog
    ]


def on(mask, names) -> str:
    """The names of the True entries of a Boolean mask, or '{}'."""
    picked = [n for n, bit in zip(names, list(jnp.asarray(mask))) if bit]
    return "{" + ", ".join(picked) + "}"
