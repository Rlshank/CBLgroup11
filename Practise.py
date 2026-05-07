import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 1">
      <joint name="box_x" type="slide" axis="1 0 0"/>
      <joint name="box_y" type="slide" axis="0 1 0"/>
      <joint name="box_z" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.3 0.5 1 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

jx = model.joint("box_x").qposadr[0]
jy = model.joint("box_y").qposadr[0]
jz = model.joint("box_z").qposadr[0]

def fake_sensor():
    t = time.time()
    x = np.sin(t * 0.5) * 1
    y = np.cos(t * 0.3) * 1
    z = np.sin(t * 0.3) * 1
    print(f"x={x:.2f}  y={y:.2f}  z={z:.2f}")
    return x, y, z

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        x, y, z = fake_sensor()
        data.qpos[jx] = x
        data.qpos[jy] = y
        data.qpos[jz] = z
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)