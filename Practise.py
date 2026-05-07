import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.3 0.5 1 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()