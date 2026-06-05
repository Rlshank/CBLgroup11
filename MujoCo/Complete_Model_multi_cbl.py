# =========================================================
# camera_support_sim.py
# Digital twin – stepper steps + current drives MuJoCo sim
# Torque calculated from cable geometry and motor current
# =========================================================

import time
import threading
import tkinter as tk
import numpy as np
import mujoco
import mujoco.viewer
import serial
import serial.tools.list_ports


# =========================================================
# SERIAL CONFIG
# =========================================================

IMU_BAUD    = 115200
IMU_PORT    = None       # None = auto-detect, or e.g. "COM3"
IMU_TIMEOUT = 0.01


# =========================================================
# SYSTEM PARAMETERS
# =========================================================

PULLEY_HEIGHT = 0.50     # m  – height of top pulleys
HALF_SPAN     = 0.485    # m  – half distance between poles

# Stepper motor
STEPS_PER_REV = 200      # steps per full revolution
SPOOL_RADIUS  = 0.01     # m  – cable spool radius (1 cm)

# Motor torque constant  τ = Kt * I
# For a typical NEMA17: Kt ≈ 0.1–0.2 N·m/A  – adjust to your motor
KT = 0.1                 # N·m / A

# Anchor positions in world XZ plane (x, z)
anchor_L = np.array([-HALF_SPAN, PULLEY_HEIGHT])
anchor_R = np.array([ HALF_SPAN, PULLEY_HEIGHT])


# =========================================================
# MUJOCO XML
# =========================================================

xml = r"""
<mujoco model="camera_cable_system">

    <compiler angle="degree" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <visual>
        <headlight ambient="0.4 0.4 0.4"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient"
                 rgb1="0.7 0.8 0.9" rgb2="0.1 0.1 0.1"
                 width="512" height="512"/>
    </asset>

    <worldbody>

        <geom type="plane" size="3 3 0.1" rgba="0.85 0.85 0.85 1"/>

        <geom type="box" pos="0 0 0.01" size="0.5 0.10 0.01"
              rgba="0.55 0.35 0.2 1"/>

        <!-- ================================================= -->
        <!-- LEFT SUPPORT -->
        <!-- ================================================= -->

        <body name="left_support" pos="-0.485 0 0">
            <geom type="box" pos="0 0 0.15"
                  size="0.016 0.02 0.15" rgba="0.52 0.40 0.25 1"/>
            <geom type="box" pos="-0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.065 0.015 0.33"
                  size="0.050 0.0015 0.02" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.050 0.015 0.38"
                  euler="0 52 0" size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.050 -0.015 0.38"
                  euler="0 52 0" size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.043 0 0.30"
                  size="0.07 0.10 0.01" rgba="0.9 0.8 0.9 1"/>
            <geom type="capsule" fromto="0 0 0.15  -0.09 0 0.29"
                  size="0.015" rgba="0.72 0.5 0.35 1"/>
            <geom type="cylinder" size="0.01 0.01" rgba="0.5 0.5 0.5 1"/>

            <site name="left_pulley" pos="0 0 0.50"
                  size="0.01" rgba="0 0 1 1"/>

            <geom type="box" pos="-0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02" rgba="0.1 0.1 0.1 1"/>
            <site name="left_motor_site" pos="-0.0655 0.0 0.33"
                  size="0.008" rgba="1 0 0 1"/>
            <geom type="cylinder" pos="-0.0655 0.0 0.33"
                  euler="90 0 0" size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>
            <geom type="cylinder" pos="0 0.0 0.485"
                  euler="90 0 0" size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>
        </body>

        <!-- ================================================= -->
        <!-- RIGHT SUPPORT -->
        <!-- ================================================= -->

        <body name="right_support" pos="0.485 0 0">
            <geom type="box" pos="0 0 0.15"
                  size="0.016 0.02 0.15" rgba="0.52 0.40 0.25 1"/>
            <geom type="box" pos="0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.065 0.015 0.33"
                  size="0.050 0.0015 0.02" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02" rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.050 0.015 0.38"
                  euler="0 -52 0" size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.050 -0.015 0.38"
                  euler="0 -52 0" size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="0.043 0 0.30"
                  size="0.07 0.10 0.01" rgba="0.9 0.8 0.9 1"/>
            <geom type="capsule" fromto="0 0 0.15   0.09 0 0.29"
                  size="0.015" rgba="0.72 0.5 0.35 1"/>
            <geom type="cylinder" size="0.01 0.01" rgba="0.5 0.5 0.5 1"/>

            <site name="right_pulley" pos="0 0 0.50"
                  size="0.01" rgba="0 0 1 1"/>

            <geom type="box" pos="0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02" rgba="0.1 0.1 0.1 1"/>
            <site name="motor_site" pos="0.0655 0.0 0.33"
                  size="0.008" rgba="1 0 0 1"/>
            <geom type="cylinder" pos="0.0655 0.0 0.33"
                  euler="90 0 0" size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>
            <geom type="cylinder" pos="0 0.0 0.485"
                  euler="90 0 0" size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>
        </body>

        <!-- ================================================= -->
        <!-- CAMERA  (mocap = driven by step counts)           -->
        <!-- ================================================= -->

        <body name="camera" mocap="true" pos="0 0 0.20">
            <geom type="box" size="0.0125 0.0125 0.0125"
                  mass="0.2" rgba="0 0 0 1"/>
            <site name="camera_site" pos="0 0 0"
                  size="0.008" rgba="1 0 0 1"/>
        </body>

    </worldbody>

    <tendon>
        <spatial name="left_cable" width="0.004" rgba="0 0.7 1 1">
            <site site="left_motor_site"/>
            <site site="left_pulley"/>
            <site site="camera_site"/>
        </spatial>
        <spatial name="right_cable" width="0.004" rgba="0 0.7 1 1">
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
data  = mujoco.MjData(model)

camera_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera")
camera_mocap_id = model.body_mocapid[camera_body_id]


# =========================================================
# SHARED STATE
# =========================================================

state_lock = threading.Lock()

# Camera position estimated from step counts
cam_position = np.array([0.0, 0.0, 0.20], dtype=float)

# Latest torque values
torque_L = 0.0
torque_R = 0.0

# Cable tension estimated from torque
tension_L = 0.0
tension_R = 0.0

# Raw serial values
raw_steps_L = 0
raw_steps_R = 0
raw_current = 0.0

running = True

MIN_X = -0.40
MAX_X =  0.40
MIN_Z =  0.10
MAX_Z =  0.45


# =========================================================
# FORWARD KINEMATICS
# Steps → cable lengths → camera XZ position
# =========================================================

def steps_to_cable_length(steps, initial_length):
    """
    Convert cumulative step count to absolute cable length.
    Positive steps = motor winds cable in = cable shortens.
    """
    delta = (steps / STEPS_PER_REV) * (2 * np.pi * SPOOL_RADIUS)
    return initial_length - delta


def forward_kinematics(length_L, length_R):
    """
    Given two cable lengths from the fixed anchors, find
    the camera position (x, z) by trilateration.

    anchor_L = (-HALF_SPAN, PULLEY_HEIGHT)
    anchor_R = ( HALF_SPAN, PULLEY_HEIGHT)
    """
    xL, zL = anchor_L
    xR, zR = anchor_R

    # Distance between anchors
    d = xR - xL   # = 2 * HALF_SPAN

    # Trilateration in XZ plane
    # x measured from left anchor
    a = (length_L**2 - length_R**2 + d**2) / (2 * d)
    h_sq = length_L**2 - a**2

    if h_sq < 0:
        h_sq = 0.0   # clamp numerical error

    h = np.sqrt(h_sq)

    # Camera is BELOW the anchors so z = zL - h
    cam_x = xL + a
    cam_z = zL - h

    return cam_x, cam_z


# =========================================================
# CABLE ANGLE CALCULATION
# Returns unit vector along each cable from pulley to camera
# =========================================================

def cable_angles(cam_x, cam_z):
    """
    Returns the angle each cable makes with the vertical (radians),
    and the unit direction vectors from pulley → camera.

    angle_L : angle of left  cable from vertical
    angle_R : angle of right cable from vertical
    vec_L   : unit vector left  pulley → camera  (x, z)
    vec_R   : unit vector right pulley → camera  (x, z)
    """
    cam = np.array([cam_x, cam_z])

    vec_L = cam - anchor_L
    vec_R = cam - anchor_R

    len_L = np.linalg.norm(vec_L)
    len_R = np.linalg.norm(vec_R)

    if len_L < 1e-6:
        len_L = 1e-6
    if len_R < 1e-6:
        len_R = 1e-6

    unit_L = vec_L / len_L
    unit_R = vec_R / len_R

    # Angle from vertical downward axis (0, -1)
    vertical = np.array([0.0, -1.0])
    angle_L  = np.arccos(np.clip(np.dot(unit_L, vertical), -1, 1))
    angle_R  = np.arccos(np.clip(np.dot(unit_R, vertical), -1, 1))

    return angle_L, angle_R, unit_L, unit_R

# =========================================================
# TENDON COLOUR IDS
# =========================================================

left_tendon_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "left_cable")
right_tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "right_cable")

# =========================================================
# TORQUE AND TENSION CALCULATION
#
# Total motor torque from current:
#   tau_total = Kt * I_total
#
# The total torque is split between the two motors
# proportionally to their cable tension contribution.
#
# Cable tension from torque:
#   T = tau / spool_radius
#
# Force balance on camera (static):
#   T_L * sin(angle_L) + T_R * sin(angle_R) = 0   (x)
#   T_L * cos(angle_L) + T_R * cos(angle_R) = m*g (z)
#
# This gives us the expected tension ratio, which we use
# to split the total torque between left and right motors.
# =========================================================

def calculate_torques(total_current, cam_x, cam_z):
    """
    Given total motor current and camera position, compute:
      - total torque
      - individual left/right torques (split by geometry)
      - cable tensions
      - cable angles
    """
    angle_L, angle_R, unit_L, unit_R = cable_angles(cam_x, cam_z)

    # Total torque produced by both motors combined
    tau_total = KT * total_current

    # ── Static force balance to find tension ratio ────────
    # Vertical equilibrium:  T_L * cos(a_L) + T_R * cos(a_R) = m*g
    # Horizontal equilibrium: -T_L * sin(a_L) + T_R * sin(a_R) = 0
    #   → T_L / T_R = sin(a_R) / sin(a_L)
    #
    # Define ratio r = T_L / T_R
    sin_L = np.abs(np.sin(angle_L))
    sin_R = np.abs(np.sin(angle_R))
    cos_L = np.abs(np.cos(angle_L))
    cos_R = np.abs(np.cos(angle_R))

    # Avoid divide-by-zero when camera is directly below a pulley
    if sin_L < 1e-6:
        sin_L = 1e-6
    if sin_R < 1e-6:
        sin_R = 1e-6

    ratio = sin_R / sin_L          # T_L = ratio * T_R

    # Total torque = tau_L + tau_R = (T_L + T_R) * r_spool
    #              = T_R * (ratio + 1) * r_spool
    # → T_R = tau_total / ((ratio + 1) * r_spool)
    T_R = tau_total / ((ratio + 1) * SPOOL_RADIUS)
    T_L = ratio * T_R

    tau_L = T_L * SPOOL_RADIUS
    tau_R = T_R * SPOOL_RADIUS

    return tau_L, tau_R, T_L, T_R, np.degrees(angle_L), np.degrees(angle_R)


# =========================================================
# PARSE SERIAL LINE
# =========================================================

def parse_serial_line(line: str):
    """
    Expected CSV format from Arduino:
        steps_left, steps_right, current
    Example:
        1200,-340,1.45
    steps_left / steps_right : cumulative signed step counts
    current                  : total motor current in Amps
    """
    try:
        parts = [v.strip() for v in line.strip().split(",")]
        if len(parts) >= 3:
            sl = int(parts[0])
            sr = int(parts[1])
            It = float(parts[2])
            return sl, sr, It
    except (ValueError, IndexError):
        pass
    return None, None, None

# =========================================================
# TENSION TO COLOUR
# =========================================================

def tension_to_color(T):

    T_MAX = 3.0
    ratio = np.clip(T / T_MAX, 0.0, 1.0)

    if ratio < 0.25:
        t     = ratio / 0.25
        red   = 0.0
        green = t
        blue  = 1.0 - t

    elif ratio < 0.50:
        t     = (ratio - 0.25) / 0.25
        red   = t
        green = 1.0
        blue  = 0.0

    elif ratio < 0.75:
        t     = (ratio - 0.50) / 0.25
        red   = 1.0
        green = 1.0 - 0.35 * t
        blue  = 0.0

    else:
        t     = (ratio - 0.75) / 0.25
        red   = 1.0
        green = 0.65 * (1.0 - t)
        blue  = 0.0

    return np.array([red, green, blue, 1.0])


# =========================================================
# AUTO-DETECT SERIAL PORT
# =========================================================

def find_serial_port():
    if IMU_PORT is not None:
        return IMU_PORT

    candidates = serial.tools.list_ports.comports()
    keywords   = ["cp210", "ch340", "ftdi", "arduino", "usb serial", "uart"]

    for port in candidates:
        desc = (port.description + port.hwid).lower()
        if any(k in desc for k in keywords):
            print(f"[SERIAL] Auto-detected: {port.device} ({port.description})")
            return port.device

    if candidates:
        print(f"[SERIAL] Trying first port: {candidates[0].device}")
        return candidates[0].device

    return None


# =========================================================
# INITIAL CABLE LENGTHS
# (camera starts at centre, z = 0.20)
# =========================================================

init_cam = np.array([0.0, 0.20])
INITIAL_LENGTH_L = np.linalg.norm(init_cam - anchor_L)
INITIAL_LENGTH_R = np.linalg.norm(init_cam - anchor_R)


# =========================================================
# SERIAL READER THREAD
# =========================================================

def serial_reader_thread():

    global cam_position, torque_L, torque_R
    global tension_L, tension_R
    global raw_steps_L, raw_steps_R, raw_current
    global running

    port = find_serial_port()

    if port is None:
        print("[SERIAL] No port found. Camera held at start position.")
        return

    try:
        ser = serial.Serial(port, IMU_BAUD, timeout=IMU_TIMEOUT)
        print(f"[SERIAL] Connected on {port} at {IMU_BAUD} baud.")
    except serial.SerialException as e:
        print(f"[SERIAL] Could not open {port}: {e}")
        return

    while running:

        try:
            raw = ser.readline().decode("utf-8", errors="ignore")
        except serial.SerialException:
            print("[SERIAL] Read error, retrying…")
            time.sleep(0.5)
            continue

        if not raw.strip():
            continue

        sl, sr, current = parse_serial_line(raw)

        if sl is None:
            continue

        # ── Cable lengths from step counts ────────────────
        len_L = steps_to_cable_length(sl, INITIAL_LENGTH_L)
        len_R = steps_to_cable_length(sr, INITIAL_LENGTH_R)

        # ── Camera position from cable lengths ────────────
        cam_x, cam_z = forward_kinematics(len_L, len_R)
        cam_x = np.clip(cam_x, MIN_X, MAX_X)
        cam_z = np.clip(cam_z, MIN_Z, MAX_Z)

        # ── Torque and tension from current + geometry ────
        tL, tR, tensL, tensR, aL, aR = calculate_torques(
            current, cam_x, cam_z
        )

        # ── Write shared state ────────────────────────────
        with state_lock:
            cam_position[0] = cam_x
            cam_position[2] = cam_z
            torque_L        = tL
            torque_R        = tR
            tension_L       = tensL
            tension_R       = tensR
            raw_steps_L     = sl
            raw_steps_R     = sr
            raw_current     = current

        print(
            f"[DATA] steps=({sl:+6d},{sr:+6d}) | "
            f"I={current:.2f} A | "
            f"pos=({cam_x:+.3f}, {cam_z:+.3f}) m | "
            f"τ_L={tL:.4f} N·m  τ_R={tR:.4f} N·m | "
            f"T_L={tensL:.3f} N  T_R={tensR:.3f} N | "
            f"θ_L={aL:.1f}°  θ_R={aR:.1f}°"
        )

    ser.close()
    print("[SERIAL] Port closed.")


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():

    global running

    data.mocap_pos[camera_mocap_id]  = [0.0, 0.0, 0.20]
    data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            with state_lock:
                pos  = cam_position.copy()
                tenL = tension_L
                tenR = tension_R

            # ── Update camera position ────────────────────
            data.mocap_pos[camera_mocap_id]  = pos
            data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]

            # ── Update cable colours from tension ─────────
            model.tendon_rgba[left_tendon_id]  = tension_to_color(tenL)
            model.tendon_rgba[right_tendon_id] = tension_to_color(tenR)

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)

# =========================================================
# TKINTER STATUS WINDOW
# =========================================================

def run_status_window():

    global running

    root = tk.Tk()
    root.title("Digital Twin – Cable Camera")
    root.geometry("480x420")

    tk.Label(root, text="Cable Camera Digital Twin",
             font=("Arial", 14, "bold")).pack(pady=8)

    # ── Live readouts ─────────────────────────────────────
    pos_label     = tk.Label(root, text="Position : x=---  z=---",
                             font=("Courier", 11))
    pos_label.pack(pady=3)

    steps_label   = tk.Label(root, text="Steps    : L=---  R=---",
                             font=("Courier", 10))
    steps_label.pack(pady=3)

    current_label = tk.Label(root, text="Current  : ---  A",
                             font=("Courier", 10))
    current_label.pack(pady=3)

    torque_label  = tk.Label(root, text="Torque   : L=---  R=---  N·m",
                             font=("Courier", 10))
    torque_label.pack(pady=3)

    tension_label = tk.Label(root, text="Tension  : L=---  R=---  N",
                             font=("Courier", 10))
    tension_label.pack(pady=3)

    angle_label   = tk.Label(root, text="Angle    : L=---  R=---  deg",
                             font=("Courier", 10))
    angle_label.pack(pady=3)

    cable_label   = tk.Label(root, text="Cables   : L=---  R=---  m",
                             font=("Courier", 10))
    cable_label.pack(pady=3)

    # ── Reset button ──────────────────────────────────────
    def reset_position():
        with state_lock:
            cam_position[:] = [0.0, 0.0, 0.20]
        print("[GUI] Position reset.")

    tk.Button(root, text="Reset Position",
              command=reset_position).pack(pady=10)

    # ── Polling update ────────────────────────────────────
    def update_labels():
        with state_lock:
            pos  = cam_position.copy()
            tL   = torque_L
            tR   = torque_R
            tenL = tension_L
            tenR = tension_R
            sL   = raw_steps_L
            sR   = raw_steps_R
            I    = raw_current

        # Recalculate cable lengths for display
        len_L = steps_to_cable_length(sL, INITIAL_LENGTH_L)
        len_R = steps_to_cable_length(sR, INITIAL_LENGTH_R)

        # Cable angles for display
        _, _, aL_deg, aR_deg = (0, 0, 0, 0)
        try:
            aL, aR, _, _ = cable_angles(pos[0], pos[2])
            aL_deg = np.degrees(aL)
            aR_deg = np.degrees(aR)
        except Exception:
            pass

        pos_label.config(
            text=f"Position : x={pos[0]:+.3f} m   z={pos[2]:+.3f} m")
        steps_label.config(
            text=f"Steps    : L={sL:+7d}   R={sR:+7d}")
        current_label.config(
            text=f"Current  : {I:.3f} A")
        torque_label.config(
            text=f"Torque   : L={tL:.4f}  R={tR:.4f}  N·m")
        tension_label.config(
            text=f"Tension  : L={tenL:.3f}  R={tenR:.3f}  N")
        angle_label.config(
            text=f"Angle    : L={aL_deg:.1f}°   R={aR_deg:.1f}°")
        cable_label.config(
            text=f"Cables   : L={len_L:.3f}  R={len_R:.3f}  m")

        if running:
            root.after(50, update_labels)

    def close_program():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_program)
    update_labels()
    root.mainloop()


# =========================================================
# START
# =========================================================

threading.Thread(target=serial_reader_thread, daemon=True).start()
threading.Thread(target=run_mujoco,           daemon=True).start()

run_status_window()