"""
camera_support_sim_v2.py — gravity + cable stress simulation

Key changes from v1:
  - camera_body is now a real dynamic body (free joint) with mass + inertia
  - Two spatial tendons drive the camera via position actuators
  - Inverse kinematics computes target cable lengths → actuator ref values
  - Cable stress (N) is read from data.actuator_force each step
  - Gravity acts normally; cables must support the camera weight
"""

import time
import threading
import tkinter as tk
import numpy as np
import mujoco
import mujoco.viewer


# =========================================================
# XML MODEL
# =========================================================

xml = """
<mujoco model="camera_cable_system">

    <compiler angle="degree"/>
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <default>
        <geom friction="0.8 0.1 0.1" density="500"/>
        <site size="0.025"/>
        <!-- Cable actuators: position-controlled, high gain -->
        <position kp="800" kv="40"/>
    </default>

    <worldbody>

        <!-- Floor -->
        <geom name="floor"
              type="plane"
              size="4 4 0.1"
              rgba="0.9 0.9 0.9 1"/>

        <!-- LEFT TRIANGULAR SUPPORT (static) -->
        <geom name="left_support_left"   type="capsule" fromto="-1.15 0 0.05  -1.0 0 1.25" size="0.05" rgba="0 0.6 1 1"/>
        <geom name="left_support_right"  type="capsule" fromto="-0.85 0 0.05  -1.0 0 1.25" size="0.05" rgba="0 0.6 1 1"/>
        <geom name="left_support_base"   type="capsule" fromto="-1.15 0 0.05  -0.85 0 0.05" size="0.05" rgba="0 0.6 1 1"/>

        <!-- RIGHT TRIANGULAR SUPPORT (static) -->
        <geom name="right_support_left"  type="capsule" fromto="0.85 0 0.05   1.0 0 1.25" size="0.05" rgba="0 0.6 1 1"/>
        <geom name="right_support_right" type="capsule" fromto="1.15 0 0.05   1.0 0 1.25" size="0.05" rgba="0 0.6 1 1"/>
        <geom name="right_support_base"  type="capsule" fromto="0.85 0 0.05   1.15 0 0.05" size="0.05" rgba="0 0.6 1 1"/>

        <!-- TOP PULLEYS -->
        <geom name="left_top_pulley"  type="cylinder" pos="-1.0 0 1.25" euler="90 0 0" size="0.11 0.04" rgba="1 0 0 1"/>
        <geom name="right_top_pulley" type="cylinder" pos="1.0 0 1.25"  euler="90 0 0" size="0.11 0.04" rgba="1 0 0 1"/>

        <!-- LOWER PULLEYS -->
        <geom name="left_lower_pulley"  type="cylinder" pos="-1.0 0 0.28" euler="90 0 0" size="0.09 0.03" rgba="1 0 0 1"/>
        <geom name="right_lower_pulley" type="cylinder" pos="1.0 0 0.28"  euler="90 0 0" size="0.09 0.03" rgba="1 0 0 1"/>

        <!-- MOTORS -->
        <geom name="left_motor"  type="box" pos="-1.55 0 0.12" size="0.12 0.08 0.08" rgba="0.6 0.3 0.7 1"/>
        <geom name="right_motor" type="box" pos="1.55 0 0.12"  size="0.12 0.08 0.08" rgba="0.6 0.3 0.7 1"/>

        <!-- CABLE ROUTING SITES (static world) -->
        <site name="left_motor_site"  pos="-1.55 0 0.12" rgba="0.6 0.3 0.7 1"/>
        <site name="right_motor_site" pos="1.55 0 0.12"  rgba="0.6 0.3 0.7 1"/>
        <site name="left_lower_site"  pos="-1.0 0 0.28"  rgba="1 0 0 1"/>
        <site name="right_lower_site" pos="1.0 0 0.28"   rgba="0 0 1 1"/>
        <site name="left_top_site"    pos="-1.0 0 1.25"  rgba="1 0 0 1"/>
        <site name="right_top_site"   pos="1.0 0 1.25"   rgba="0 0 1 1"/>

        <!--
            CAMERA BODY — now a real dynamic body.
            Mass = 1.5 kg (realistic camera + rig weight).
            Collision disabled so it doesn't snag the support frames.
            A free joint lets gravity act on it naturally.
        -->
        <body name="camera_body" pos="0 0 0.75">

            <freejoint name="camera_free"/>

            <inertial pos="0 0 0"
                      mass="1.5"
                      diaginertia="0.005 0.005 0.003"/>

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

            <!-- Attachment point for both cables -->
            <site name="camera_site" pos="0 0 0.06" rgba="0 1 0 1"/>

        </body>

    </worldbody>

    <!--
        TENDONS — spatial cables routed through pulleys.
        These are the structural elements that carry load.
        MuJoCo computes their length (data.ten_length) each step.
    -->
    <tendon>

        <spatial name="left_cable" width="0.008" rgba="0.2 0.2 0.2 1"
                 limited="true" range="0 4.0"
                 stiffness="0" damping="0">
            <site site="left_motor_site"/>
            <site site="left_lower_site"/>
            <site site="left_top_site"/>
            <site site="camera_site"/>
        </spatial>

        <spatial name="right_cable" width="0.008" rgba="0.2 0.2 0.2 1"
                 limited="true" range="0 4.0"
                 stiffness="0" damping="0">
            <site site="right_motor_site"/>
            <site site="right_lower_site"/>
            <site site="right_top_site"/>
            <site site="camera_site"/>
        </spatial>

    </tendon>

    <!--
        ACTUATORS — position-controlled tendon actuators.
        Setting ctrl[i] = target cable length drives the motor.
        data.actuator_force[i] gives the tension in Newtons.

        gear="1"  → ctrl units = metres of cable length
        kp/kv     → stiffness / damping of the virtual spring
    -->
    <actuator>

        <position name="left_winch"
                  tendon="left_cable"
                  gear="1"
                  kp="800"
                  kv="40"/>

        <position name="right_winch"
                  tendon="right_cable"
                  gear="1"
                  kp="800"
                  kv="40"/>

    </actuator>

</mujoco>
"""


# =========================================================
# BUILD MODEL
# =========================================================

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

# Joint/body IDs
camera_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_body")
camera_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "camera_free")

# Tendon IDs (for length + stress readout)
left_tendon_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "left_cable")
right_tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "right_cable")

# Actuator IDs
left_act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_winch")
right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_winch")


# =========================================================
# INVERSE KINEMATICS
#
# Given desired (x, z) of the camera attachment site,
# compute the straight-line cable lengths from each top pulley.
#
# Routing: motor → lower pulley → top pulley → camera
# The motor→lower and lower→top segments are fixed lengths.
# Only the top-pulley→camera segment changes with camera pos.
# =========================================================

# Pulley world positions (x, z)
ANCHOR_LEFT  = np.array([-1.0, 1.25])
ANCHOR_RIGHT = np.array([ 1.0, 1.25])

# Fixed segment lengths (motor→lower pulley + lower→top pulley)
# Measured from XML positions
_left_motor  = np.array([-1.55, 0.12])
_left_lower  = np.array([-1.0,  0.28])
_left_top    = np.array([-1.0,  1.25])
FIXED_LEFT   = (np.linalg.norm(_left_lower  - _left_motor) +
                np.linalg.norm(_left_top    - _left_lower))

_right_motor = np.array([ 1.55, 0.12])
_right_lower = np.array([ 1.0,  0.28])
_right_top   = np.array([ 1.0,  1.25])
FIXED_RIGHT  = (np.linalg.norm(_right_lower - _right_motor) +
                np.linalg.norm(_right_top   - _right_lower))

# Camera site is 0.06 m above body centre
SITE_OFFSET_Z = 0.06


def inverse_kinematics(des_x: float, des_z: float):
    """
    Return target total cable lengths (in metres) for each winch
    so that the camera attachment site sits at (des_x, des_z).
    """
    site_pos = np.array([des_x, des_z + SITE_OFFSET_Z])

    L1 = FIXED_LEFT  + np.linalg.norm(ANCHOR_LEFT  - site_pos)
    L2 = FIXED_RIGHT + np.linalg.norm(ANCHOR_RIGHT - site_pos)

    return L1, L2


# =========================================================
# SHARED STATE
# =========================================================

target_position = np.array([0.0, 0.0, 0.75], dtype=float)
stress_values   = {"left": 0.0, "right": 0.0, "total": 0.0}   # Newtons
target_lock     = threading.Lock()
stress_lock     = threading.Lock()
running         = True

# Movement limits
MIN_X, MAX_X = -0.70,  0.70
MIN_Z, MAX_Z =  0.45,  1.05


# =========================================================
# MUJOCO SIMULATION THREAD
# =========================================================

def run_mujoco():
    global running

    # Initialise cable lengths for start position
    L1, L2 = inverse_kinematics(0.0, 0.75)
    data.ctrl[left_act_id]  = L1
    data.ctrl[right_act_id] = L2

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            with target_lock:
                tx, tz = target_position[0], target_position[2]

            # Compute IK → set actuator target lengths
            L1, L2 = inverse_kinematics(tx, tz)
            data.ctrl[left_act_id]  = L1
            data.ctrl[right_act_id] = L2

            # Step physics (gravity + actuator forces + dynamics)
            mujoco.mj_step(model, data)

            # ── Read cable stress ──────────────────────────────
            # actuator_force is the generalised force; for a
            # tendon actuator with gear=1 this equals cable
            # tension in Newtons (positive = pulling).
            left_tension  = float(data.actuator_force[left_act_id])
            right_tension = float(data.actuator_force[right_act_id])

            # Also available: data.ten_length for actual cable length
            left_len  = float(data.ten_length[left_tendon_id])
            right_len = float(data.ten_length[right_tendon_id])

            with stress_lock:
                stress_values["left"]  = left_tension
                stress_values["right"] = right_tension
                stress_values["total"] = left_tension + right_tension
                stress_values["left_len"]  = left_len
                stress_values["right_len"] = right_len

            print(
                f"T_left={left_tension:+7.2f} N  "
                f"T_right={right_tension:+7.2f} N  "
                f"L_left={left_len:.4f} m  "
                f"L_right={right_len:.4f} m"
            )

            viewer.sync()

    running = False


# =========================================================
# TKINTER CONTROL + STRESS DISPLAY
# =========================================================

def run_slider_window():
    global running

    root = tk.Tk()
    root.title("Camera Slider Control — Cable Stress")
    root.geometry("480x560")
    root.configure(bg="#1a1a2e")

    # ── Title ──────────────────────────────────────────────
    tk.Label(
        root, text="CABLE SUPPORT SIMULATOR",
        font=("Courier", 13, "bold"),
        fg="#00d4ff", bg="#1a1a2e"
    ).pack(pady=(14, 2))

    tk.Label(
        root, text="gravity + cable stress active",
        font=("Courier", 9), fg="#888", bg="#1a1a2e"
    ).pack(pady=(0, 10))

    # ── X Slider ───────────────────────────────────────────
    tk.Label(root, text="Horizontal position X",
             font=("Courier", 10), fg="#ccc", bg="#1a1a2e").pack()

    x_slider = tk.Scale(
        root, from_=MIN_X, to=MAX_X, resolution=0.01,
        orient=tk.HORIZONTAL, length=370,
        bg="#16213e", fg="#00d4ff", troughcolor="#0f3460",
        highlightthickness=0, activebackground="#00d4ff"
    )
    x_slider.set(0.0)
    x_slider.pack(pady=6)

    # ── Z Slider ───────────────────────────────────────────
    tk.Label(root, text="Camera height Z",
             font=("Courier", 10), fg="#ccc", bg="#1a1a2e").pack()

    z_slider = tk.Scale(
        root, from_=MAX_Z, to=MIN_Z, resolution=0.01,
        orient=tk.VERTICAL, length=180,
        bg="#16213e", fg="#00d4ff", troughcolor="#0f3460",
        highlightthickness=0, activebackground="#00d4ff"
    )
    z_slider.set(0.75)
    z_slider.pack(pady=6)

    # ── Position label ─────────────────────────────────────
    pos_label = tk.Label(
        root, text="x = 0.00 m  |  z = 0.75 m",
        font=("Courier", 10), fg="#aaa", bg="#1a1a2e"
    )
    pos_label.pack(pady=4)

    # ── Stress display ─────────────────────────────────────
    stress_frame = tk.Frame(root, bg="#0f3460", bd=1, relief="solid")
    stress_frame.pack(fill="x", padx=20, pady=8)

    tk.Label(
        stress_frame, text="  CABLE TENSIONS",
        font=("Courier", 10, "bold"), fg="#ff6b6b", bg="#0f3460"
    ).pack(anchor="w", pady=(6, 2))

    left_stress_lbl = tk.Label(
        stress_frame, text="  Left  cable :    0.00 N",
        font=("Courier", 10), fg="#00d4ff", bg="#0f3460"
    )
    left_stress_lbl.pack(anchor="w")

    right_stress_lbl = tk.Label(
        stress_frame, text="  Right cable :    0.00 N",
        font=("Courier", 10), fg="#00d4ff", bg="#0f3460"
    )
    right_stress_lbl.pack(anchor="w")

    total_stress_lbl = tk.Label(
        stress_frame, text="  Total load  :    0.00 N  (gravity: 14.72 N)",
        font=("Courier", 10), fg="#ffd700", bg="#0f3460"
    )
    total_stress_lbl.pack(anchor="w", pady=(2, 8))

    # ── Warning label ──────────────────────────────────────
    warn_label = tk.Label(
        root, text="", font=("Courier", 10, "bold"),
        fg="#ff4444", bg="#1a1a2e"
    )
    warn_label.pack(pady=2)

    # ── Reset button ───────────────────────────────────────
    def reset_camera():
        x_slider.set(0.0)
        z_slider.set(0.75)

    tk.Button(
        root, text="Reset to centre",
        command=reset_camera,
        font=("Courier", 10), fg="#1a1a2e", bg="#00d4ff",
        activebackground="#009bb5", relief="flat", padx=10, pady=4
    ).pack(pady=10)

    # ── Update loop ────────────────────────────────────────
    GRAVITY_FORCE = 1.5 * 9.81   # weight of camera in N
    CABLE_LIMIT   = 80.0          # warn above this tension

    def update():
        if not running:
            return

        # Push target to sim thread
        with target_lock:
            target_position[0] = float(x_slider.get())
            target_position[1] = 0.0
            target_position[2] = float(z_slider.get())

        pos_label.config(
            text=f"x = {target_position[0]:+.2f} m  |  z = {target_position[2]:.2f} m"
        )

        # Pull stress from sim thread
        with stress_lock:
            lt = stress_values.get("left",  0.0)
            rt = stress_values.get("right", 0.0)
            tt = stress_values.get("total", 0.0)

        left_stress_lbl.config( text=f"  Left  cable : {lt:+8.2f} N")
        right_stress_lbl.config(text=f"  Right cable : {rt:+8.2f} N")
        total_stress_lbl.config(
            text=f"  Total load  : {tt:+8.2f} N  (gravity: {GRAVITY_FORCE:.2f} N)"
        )

        # Colour-code by load level
        if max(abs(lt), abs(rt)) > CABLE_LIMIT:
            warn_label.config(text="⚠  CABLE OVERLOAD")
        elif max(abs(lt), abs(rt)) > CABLE_LIMIT * 0.7:
            warn_label.config(text="△  High tension")
        else:
            warn_label.config(text="✓  Tension nominal")

        root.after(33, update)   # ~30 Hz UI refresh

    # ── Close ──────────────────────────────────────────────
    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    update()
    root.mainloop()


# =========================================================
# START
# =========================================================

mujoco_thread = threading.Thread(target=run_mujoco, daemon=True)
mujoco_thread.start()

run_slider_window()