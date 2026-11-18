"""Optional: loop a qpos trajectory in MuJoCo's interactive viewer.

    python walkthrough/replay.py                 # the hand-moved demo
    python walkthrough/replay.py my_path.npz     # any .npz with a 'qpos' array

Needs a display. Close the window to stop.
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from _common import CMG, TRAJECTORY

model = mujoco.MjModel.from_xml_path(str(CMG / "src/cmg/scene/families/step.xml"))
data = mujoco.MjData(model)
qpos = np.load(sys.argv[1] if len(sys.argv) > 1 else TRAJECTORY)["qpos"]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for frame in qpos:
            if not viewer.is_running():
                break
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
        time.sleep(1.0)
