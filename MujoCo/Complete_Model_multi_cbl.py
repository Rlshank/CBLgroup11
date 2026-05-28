# =========================================================
# camera_support_sim.py
# Cable camera simulation using MuJoCo tendons
# =========================================================

import time
import threading
import tkinter as tk
import numpy as np
import mujoco
import mujoco.viewer


# =========================================================
# MUJOCO XML
# =========================================================

xml = r"""
<mujoco model="camera_cable_system">

    <compiler angle="degree" coordinate="local"/>

    <option gravity="0 0 -9.81"
            timestep="0.002"/>

    <visual>
        <headlight ambient="0.4 0.4 0.4"/>
    </visual>

    <asset>
        <texture type="skybox"
                 builtin="gradient"
                 rgb1="0.7 0.8 0.9"
                 rgb2="0.1 0.1 0.1"
                 width="512"
                 height="512"/>
    </asset>

    <worldbody>

        <!-- ================================================= -->
        <!-- GROUND -->
        <!-- ================================================= -->

        <geom type="plane"
              size="3 3 0.1"
              rgba="0.85 0.85 0.85 1"/>

        <!-- ================================================= -->
        <!-- base -->
        <!-- ================================================= -->

        <!-- Base -->
            <geom type="box"
                  pos="0 0 0.01"
                  size="0.5 0.10 0.01"
                  rgba="0.55 0.35 0.2 1"/>


        <!-- ================================================= -->
        <!-- LEFT SUPPORT -->
        <!-- ================================================= -->

        <body name="left_support"
              pos="-0.485 0 0">

            <!-- Vertical pillar -->
            <geom type="box"
                  pos="0 0 0.15"
                  size="0.016 0.02 0.15"
                  rgba="0.52 0.40 0.25 1"/>

            <!-- vertical sheet l1 -->
            <geom type="box"
                  pos="-0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>      

            <!-- vertical sheet l2 -->
            <geom type="box"
                  pos="-0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>         
                  
            <!-- horizontal sheet l1 -->
            <geom type="box"
                  pos="-0.065 0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>
                  
            <!-- horizontal sheet l2 -->
            <geom type="box"
                  pos="-0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- tilted diagonal brace sheet l1 -->
            <geom type="box"
                  pos="-0.050 0.015 0.38"
                  euler="0 52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- tilted diagonal brace sheet l2 -->
            <geom type="box"
                  pos="-0.050 -0.015 0.38"
                  euler="0 52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- wood support plate -->
            <geom type="box"
                  pos="-0.043 0 0.30"
                  size="0.07 0.10 0.01"
                  rgba="0.9 0.8 0.9 1"/>

            <geom type="capsule"
                  fromto="0 0 0.15   -0.09 0 0.29"
                  size="0.015"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- Pulley -->
            <geom type="cylinder"
                  size="0.01 0.01"
                  rgba="0.5 0.5 0.5 1"/>

            <site name="left_pulley"
                  pos="0 0 0.50"
                  size="0.01"
                  rgba="0 0 1 1"/>

            <!-- Motor -->
            <geom type="box"
                  pos="-0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02"
                  rgba="0.1 0.1 0.1 1"/>

            <site name="left_motor_site"
                  pos="-0.0655 0.0 0.33"
                  size="0.008"
                  rgba="1 0 0 1"/>

            <!-- Motor Winch Left -->
            <geom type="cylinder"
                  pos="-0.0655 0.0 0.33"
                  euler="90 0 0"
                  size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>

            <!-- Top Winch Left -->
            <geom type="cylinder"
                  pos="0 0.0 0.485"
                  euler="90 0 0"
                  size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>      
        </body>
        
        
        <!-- ================================================= -->
        <!-- RIGHT SUPPORT -->
        <!-- ================================================= -->

        <body name="right_support"
              pos="0.485 0 0">

            <!-- Vertical pillar -->
            <geom type="box"
                  pos="0 0 0.15"
                  size="0.016 0.02 0.15"
                  rgba="0.52 0.40 0.25 1"/>

            <!-- vertical sheet r1 -->
            <geom type="box"
                  pos="0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>      

            <!-- vertical sheet r2 -->
            <geom type="box"
                  pos="0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>         
                  
            <!-- horizontal sheet r1 -->
            <geom type="box"
                  pos="0.065 0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>
                  
            <!-- horizontal sheet r2 -->
            <geom type="box"
                  pos="0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- horizontal sheet r2 -->
            <geom type="box"
                  pos="0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>
                        
            <!-- tilted diagonal brace sheet r1 -->
            <geom type="box"
                  pos="0.050 0.015 0.38"
                  euler="0 -52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- tilted diagonal brace sheet r2 -->
            <geom type="box"
                  pos="0.050 -0.015 0.38"
                  euler="0 -52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- wood support plate -->
            <geom type="box"
                  pos="0.043 0 0.30"
                  size="0.07 0.10 0.01"
                  rgba="0.9 0.8 0.9 1"/>

            <geom type="capsule"
                  fromto="0 0 0.15   0.09 0 0.29"
                  size="0.015"
                  rgba="0.72 0.5 0.35 1"/>

            <!-- Pulley -->
                <geom type="cylinder"
                  size="0.01 0.01"
                  rgba="0.5 0.5 0.5 1"/>

            <site name="right_pulley"
                  pos="0 0 0.50"
                  size="0.01"
                  rgba="0 0 1 1"/>

            <!-- Motor -->
            <geom type="box"
                  pos="0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02"
                  rgba="0.1 0.1 0.1 1"/>

            <site name="motor_site"
                  pos="0.0655 0.0 0.33"
                  size="0.008"
                  rgba="1 0 0 1"/>

            <!-- Motor Wench Right -->
            <geom type="cylinder"
                  pos="0.0655 0.0 0.33"
                  euler="90 0 0"
                  size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>

            <!-- Top Wench Right -->
            <geom type="cylinder"
                  pos="0 0.0 0.485"
                  euler="90 0 0"
                  size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>      
        </body>

        <!-- ================================================= -->
        <!-- CAMERA -->
        <!-- ================================================= -->

        <body name="camera"
              mocap="true"
              pos="0 0 0.20">

            <geom type="box"
                  size="0.0125 0.0125 0.0125"
                  mass="0.2"
                  rgba="0 0 0 1"/>

            <site name="camera_site"
                  pos="0 0 0"
                  size="0.008"
                  rgba="1 0 0 1"/>
        </body>

    </worldbody>

    <!-- ================================================= -->
    <!-- CABLE TENDONS -->
    <!-- ================================================= -->

    <tendon>

        <!-- Left cable -->
        <spatial name="left_cable"
                 width="0.004"
                 rgba="0 0.7 1 1">

            <site site="left_motor_site"/>
            <site site="left_pulley"/>
            <site site="camera_site"/>

        </spatial>

        <!-- Right cable -->
        <spatial name="right_cable"
                 width="0.004"
                 rgba="0 0.7 1 1">

            <site site="motor_site"/>
            <site site="right_pulley"/>
            <site site="camera_site"/>

        </spatial>

    </tendon>

</mujoco>
"""


# =========================================================
# CREATE MODEL
# =========================================================

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


# =========================================================
# CAMERA IDS
# =========================================================

camera_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "camera"
)

camera_mocap_id = model.body_mocapid[camera_body_id]


# =========================================================
# REAL SYSTEM DIMENSIONS
# =========================================================

PULLEY_HEIGHT = 0.50
HALF_SPAN = 0.465

anchor_1 = np.array([-HALF_SPAN, PULLEY_HEIGHT])
anchor_2 = np.array([ HALF_SPAN, PULLEY_HEIGHT])


# =========================================================
# WINCH PARAMETERS
# =========================================================

# 1 cm spool radius
spool_r = 0.01

left_motor_angle = 0.0
right_motor_angle = 0.0


# =========================================================
# INITIAL POSITION
# =========================================================

prev_pos = np.array([0.0, 0.20])


# =========================================================
# CAMERA POSITION
# =========================================================

camera_position = np.array(
    [0.0, 0.0, 0.20],
    dtype=float
)

target_position = np.array(
    [0.0, 0.0, 0.20],
    dtype=float
)


# =========================================================
# MOVEMENT LIMITS
# =========================================================

MIN_X = -0.40
MAX_X =  0.40

MIN_Z = 0.10
MAX_Z = 0.45


# =========================================================
# MOVEMENT SPEED
# =========================================================

MOVEMENT_SPEED = 2.0


target_lock = threading.Lock()
running = True


# =========================================================
# INVERSE KINEMATICS
# =========================================================

def inverse_kinematics(des_x, des_z):

    global prev_pos

    des_pos = np.array([des_x, des_z])

    # Current cable lengths
    L1 = np.linalg.norm(des_pos - anchor_1)
    L2 = np.linalg.norm(des_pos - anchor_2)

    # Previous cable lengths
    L1_prev = np.linalg.norm(prev_pos - anchor_1)
    L2_prev = np.linalg.norm(prev_pos - anchor_2)

    # Cable length change
    dL1 = L1 - L1_prev
    dL2 = L2 - L2_prev

    # Convert to spool rotation
    dtheta1 = np.degrees(dL1 / spool_r)
    dtheta2 = np.degrees(dL2 / spool_r)

    prev_pos = des_pos.copy()

    return (
        dtheta1,
        dtheta2,
        L1,
        L2
    )


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():

    global running
    global camera_position
    global left_motor_angle
    global right_motor_angle

    data.mocap_pos[camera_mocap_id] = camera_position

    data.mocap_quat[camera_mocap_id] = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            dt = model.opt.timestep

            # =============================================
            # TARGET POSITION
            # =============================================

            with target_lock:
                target_copy = target_position.copy()

            # =============================================
            # SMOOTH MOVEMENT
            # =============================================

            camera_position += (
                target_copy - camera_position
            ) * MOVEMENT_SPEED * dt

            # =============================================
            # IK
            # =============================================

            dtheta1, dtheta2, L1, L2 = inverse_kinematics(
                camera_position[0],
                camera_position[2]
            )

            left_motor_angle += dtheta1
            right_motor_angle += dtheta2

            # =============================================
            # PRINT INFO
            # =============================================

            print(
                f"L cable: {L1:.3f} m | "
                f"R cable: {L2:.3f} m | "
                f"L motor: {left_motor_angle:.1f} deg | "
                f"R motor: {right_motor_angle:.1f} deg"
            )

            # =============================================
            # MOVE CAMERA
            # =============================================

            data.mocap_pos[camera_mocap_id] = camera_position

            data.mocap_quat[camera_mocap_id] = np.array(
                [1.0, 0.0, 0.0, 0.0]
            )

            mujoco.mj_forward(model, data)

            viewer.sync()

            time.sleep(dt)

    running = False


# =========================================================
# TKINTER GUI
# =========================================================

def run_slider_window():

    global running

    root = tk.Tk()

    root.title("Cable Camera Control")

    root.geometry("420x420")

    title = tk.Label(
        root,
        text="Cable Camera Controller",
        font=("Arial", 14, "bold")
    )

    title.pack(pady=10)

    # =====================================================
    # X SLIDER
    # =====================================================

    x_label = tk.Label(
        root,
        text="Horizontal Position X"
    )

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

    z_label = tk.Label(
        root,
        text="Camera Height Z"
    )

    z_label.pack()

    z_slider = tk.Scale(
        root,
        from_=MAX_Z,
        to=MIN_Z,
        resolution=0.01,
        orient=tk.VERTICAL,
        length=220
    )

    z_slider.set(0.20)

    z_slider.pack(pady=10)

    # =====================================================
    # POSITION LABEL
    # =====================================================

    position_label = tk.Label(
        root,
        text="x = 0.00 , z = 0.20"
    )

    position_label.pack(pady=5)

    # =====================================================
    # UPDATE TARGET
    # =====================================================

    def update_target():

        with target_lock:

            target_position[0] = float(x_slider.get())
            target_position[1] = 0.0
            target_position[2] = float(z_slider.get())

        position_label.config(
            text=(
                f"x = {target_position[0]:.2f} , "
                f"z = {target_position[2]:.2f}"
            )
        )

        if running:
            root.after(20, update_target)

    # =====================================================
    # RESET BUTTON
    # =====================================================

    def reset_camera():

        x_slider.set(0.0)
        z_slider.set(0.20)

    reset_button = tk.Button(
        root,
        text="Reset Camera",
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

    root.protocol(
        "WM_DELETE_WINDOW",
        close_program
    )

    update_target()

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