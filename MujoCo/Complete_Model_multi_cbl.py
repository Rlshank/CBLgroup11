# camera_support_sim.py with inverse kinematics
import time
import threading
import tkinter as tk
import numpy as np
import mujoco
import mujoco.viewer


xml = """
<mujoco model="camera_cable_system">

    <compiler angle="degree"/>
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <default>
        <geom friction="0.8 0.1 0.1"
              density="500"/>
        <site size="0.025"/>
    </default>

    <worldbody>

        <!-- Floor -->
        <geom name="floor"
              type="plane"
              size="4 4 0.1"
              rgba="0.9 0.9 0.9 1"/>

        <!-- LEFT TRIANGULAR SUPPORT -->

        <geom name="left_support_left"
              type="capsule"
              fromto="-1.15 0 0.05   -1.0 0 1.25"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <geom name="left_support_right"
              type="capsule"
              fromto="-0.85 0 0.05   -1.0 0 1.25"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <geom name="left_support_base"
              type="capsule"
              fromto="-1.15 0 0.05   -0.85 0 0.05"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <!-- RIGHT TRIANGULAR SUPPORT -->

        <geom name="right_support_left"
              type="capsule"
              fromto="0.85 0 0.05   1.0 0 1.25"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <geom name="right_support_right"
              type="capsule"
              fromto="1.15 0 0.05   1.0 0 1.25"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <geom name="right_support_base"
              type="capsule"
              fromto="0.85 0 0.05   1.15 0 0.05"
              size="0.05"
              rgba="0 0.6 1 1"/>

        <!-- TOP PULLEYS -->

        <geom name="left_top_pulley"
              type="cylinder"
              pos="-1.0 0 1.25"
              euler="90 0 0"
              size="0.11 0.04"
              rgba="1 0 0 1"/>

        <geom name="right_top_pulley"
              type="cylinder"
              pos="1.0 0 1.25"
              euler="90 0 0"
              size="0.11 0.04"
              rgba="1 0 0 1"/>

        <!-- LOWER PULLEYS -->

        <geom name="left_lower_pulley"
              type="cylinder"
              pos="-1.0 0 0.28"
              euler="90 0 0"
              size="0.09 0.03"
              rgba="1 0 0 1"/>

        <geom name="right_lower_pulley"
              type="cylinder"
              pos="1.0 0 0.28"
              euler="90 0 0"
              size="0.09 0.03"
              rgba="1 0 0 1"/>

        <!-- MOTORS -->

        <geom name="left_motor"
              type="box"
              pos="-1.55 0 0.12"
              size="0.12 0.08 0.08"
              rgba="0.6 0.3 0.7 1"/>

        <geom name="right_motor"
              type="box"
              pos="1.55 0 0.12"
              size="0.12 0.08 0.08"
              rgba="0.6 0.3 0.7 1"/>

        <!-- CABLE ROUTING SITES -->

        <site name="left_lower_site"
              pos="-1.0 0 0.28"
              rgba="1 0 0 1"/>

        <site name="right_lower_site"
              pos="1.0 0 0.28"
              rgba="0 0 1 1"/>

        <site name="left_top_site"
              pos="-1.0 0 1.25"
              rgba="1 0 0 1"/>

        <site name="right_top_site"
              pos="1.0 0 1.25"
              rgba="0 0 1 1"/>
        
        <site name="left_motor_site"
                pos="-1.55 0 0.12"
                rgba="0.6 0.3 0.7 1"/>
        
        <site name="right_motor_site"
                pos="1.55 0 0.12"
                rgba="0.6 0.3 0.7 1"/>

        <!-- CAMERA BODY -->

        <body name="camera_body"
              mocap="true"
              pos="0 0 0.75">

            <geom name="camera_box"
                  type="box"
                  size="0.10 0.06 0.07"
                  rgba="0.05 0.05 0.05 1"
                  contype="0"
                  conaffinity="0"/>

            <geom name="camera_lens"
                  type="cylinder"
                  pos="0 -0.075 0"
                  euler="90 0 0"
                  size="0.035 0.025"
                  rgba="0.01 0.01 0.01 1"
                  contype="0"
                  conaffinity="0"/>

            <site name="camera_site"
                  pos="0 0 0.06"
                  rgba="0 1 0 1"/>

        </body>

    </worldbody>

    <!-- VISUAL CABLES -->

    <tendon>

        <spatial name="left_cable"
                 width="0.008"
                 rgba="0 0 0 1">

            <site site="left_motor_site"/>     
            <site site="left_lower_site"/>
            <site site="left_top_site"/>
            <site site="camera_site"/>

        </spatial>

        <spatial name="right_cable"
                 width="0.008"
                 rgba="0 0 0 1">
            <site site="right_motor_site"/>
            <site site="right_lower_site"/>
            <site site="right_top_site"/>
            <site site="camera_site"/>

        </spatial>

    </tendon>

</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

camera_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "camera_body"
)

camera_mocap_id = model.body_mocapid[camera_body_id]


# =========================================================
# INVERSE KINEMATICS PARAMETERS
# =========================================================

spool_r = 0.10

anchor_1 = np.array([-1.0, 1.25])
anchor_2 = np.array([1.0, 1.25])

prev_pos = np.array([0.0, 0.75])

left_motor_angle = 0.0
right_motor_angle = 0.0


# =========================================================
# INVERSE KINEMATICS FUNCTION
# =========================================================

def inverse_kinematics(des_x, des_z):
    global prev_pos

    des_pos = np.array([des_x, des_z])

    # Current cable lengths
    L1 = np.linalg.norm(anchor_1 - des_pos)
    L2 = np.linalg.norm(anchor_2 - des_pos)

    # Previous cable lengths
    L1_prev = np.linalg.norm(anchor_1 - prev_pos)
    L2_prev = np.linalg.norm(anchor_2 - prev_pos)

    # Cable length changes
    dL1 = L1 - L1_prev
    dL2 = L2 - L2_prev

    # Convert cable length to spool rotation
    dtheta1 = np.degrees(dL1 / spool_r)
    dtheta2 = np.degrees(dL2 / spool_r)

    prev_pos = des_pos.copy()

    return dtheta1, dtheta2


# =========================================================
# CAMERA POSITION VARIABLES
# =========================================================

camera_position = np.array([0.0, 0.0, 0.75], dtype=float)
target_position = np.array([0.0, 0.0, 0.75], dtype=float)


# =========================================================
# MOVEMENT LIMITS
# =========================================================

MIN_X = -0.70
MAX_X = 0.70

MIN_Z = 0.50
MAX_Z = 1.05

MOVEMENT_SPEED = 2.0


target_lock = threading.Lock()
running = True


# =========================================================
# MUJOCO SIMULATION THREAD
# =========================================================

def run_mujoco():
    global running
    global camera_position
    global left_motor_angle
    global right_motor_angle

    data.mocap_pos[camera_mocap_id, :] = camera_position
    data.mocap_quat[camera_mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            dt = model.opt.timestep

            with target_lock:
                target_copy = target_position.copy()

            # Smooth movement toward slider target
            camera_position += (
                target_copy - camera_position
            ) * MOVEMENT_SPEED * dt

            # =================================================
            # INVERSE KINEMATICS
            # =================================================

            dtheta1, dtheta2 = inverse_kinematics(
                camera_position[0],
                camera_position[2]
            )

            left_motor_angle += dtheta1
            right_motor_angle += dtheta2

            # Print motor angles
            print(
                f"Left motor: {left_motor_angle:.2f} deg | "
                f"Right motor: {right_motor_angle:.2f} deg"
            )

            # Move camera body
            data.mocap_pos[camera_mocap_id, :] = camera_position
            data.mocap_quat[camera_mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0])

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(dt)

    running = False


# =========================================================
# TKINTER CONTROL WINDOW
# =========================================================

def run_slider_window():
    global running

    root = tk.Tk()
    root.title("Camera Slider Control")

    root.geometry("420x420")

    title = tk.Label(
        root,
        text="Move the camera using the sliders",
        font=("Arial", 14, "bold")
    )
    title.pack(pady=10)

    # =====================================================
    # X SLIDER
    # =====================================================

    x_label = tk.Label(root, text="Horizontal position X")
    x_label.pack()

    x_slider = tk.Scale(
        root,
        from_=MIN_X,
        to=MAX_X,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        length=350
    )

    x_slider.set(0.0)
    x_slider.pack(pady=10)

    # =====================================================
    # Z SLIDER
    # =====================================================

    z_label = tk.Label(root, text="Camera height Z")
    z_label.pack()

    z_slider = tk.Scale(
        root,
        from_=MAX_Z,
        to=MIN_Z,
        resolution=0.01,
        orient=tk.VERTICAL,
        length=220
    )

    z_slider.set(0.75)
    z_slider.pack(pady=10)

    # =====================================================
    # POSITION LABEL
    # =====================================================

    position_label = tk.Label(
        root,
        text="x = 0.00, z = 0.75"
    )

    position_label.pack(pady=5)

    # =====================================================
    # UPDATE FUNCTION
    # =====================================================

    def update_target_from_sliders():

        with target_lock:
            target_position[0] = float(x_slider.get())
            target_position[1] = 0.0
            target_position[2] = float(z_slider.get())

        position_label.config(
            text=f"x = {target_position[0]:.2f}, z = {target_position[2]:.2f}"
        )

        if running:
            root.after(20, update_target_from_sliders)

    # =====================================================
    # RESET BUTTON
    # =====================================================

    def reset_camera():
        x_slider.set(0.0)
        z_slider.set(0.75)

    reset_button = tk.Button(
        root,
        text="Reset camera to centre",
        command=reset_camera
    )

    reset_button.pack(pady=10)

    # =====================================================
    # CLOSE PROGRAM
    # =====================================================

    def close_program():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_program)

    update_target_from_sliders()
    root.mainloop()


# =========================================================
# START PROGRAM
# =========================================================

mujoco_thread = threading.Thread(
    target=run_mujoco,
    daemon=True
)

mujoco_thread.start()

run_slider_window()



