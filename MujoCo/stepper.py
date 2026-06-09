# =========================================================
# camera_support_sim.py
# Digital twin – stepper motor encoders drive position,
# current sensors provide cable tension feedback
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

SERIAL_BAUD    = 115200
SERIAL_PORT    = "COM7"     # None = auto-detect, or set e.g. "COM3" / "/dev/ttyUSB0"
SERIAL_TIMEOUT = 0.01       # seconds


# =========================================================
# STEPPER CONFIG
# =========================================================

STEPS_PER_REV  = 200        # full steps per revolution (adjust for your motor)
MICROSTEP      = 1        # microstepping factor (e.g. 1, 2, 4, 8, 16)
SPOOL_RADIUS   = 0.01275    # metres – radius of cable spool


# =========================================================
# CURRENT SENSOR CONFIG
# =========================================================

# Current (A) → Tension (N) conversion factor
# Calibrate this for your motor + drivetrain
CURRENT_TO_TENSION = 0.0376    # N per Amp

# Threshold below which tension is considered slack
TENSION_MIN = 0.0
TENSION_MAX = 50.0          # N – safety limit for display


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

        <geom type="plane"
              size="3 3 0.1"
              rgba="0.85 0.85 0.85 1"/>

        <geom type="box"
              pos="0 0 0.01"
              size="0.5 0.10 0.01"
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
                  euler="0 -52 0" size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>
            <geom type="box" pos="-0.043 0 0.30"
                  size="0.07 0.10 0.01" rgba="0.9 0.8 0.9 1"/>
            <geom type="capsule" fromto="0 0 0.15   -0.09 0 0.29"
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
        <!-- CAMERA (mocap = driven by stepper position)       -->
        <!-- ================================================= -->

        <body name="camera" mocap="true" pos="0 0 0.05">

            <geom type="box"
                  size="0.0125 0.0125 0.0125"
                  mass="0.2"
                  rgba="0 0 0 1"/>

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
# MODEL
# =========================================================

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

camera_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera")
camera_mocap_id = model.body_mocapid[camera_body_id]

# ── Tendon IDs for colour updates ─────────────────────────
left_tendon_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "left_cable")
right_tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "right_cable")
# =========================================================
# SYSTEM DIMENSIONS
# =========================================================

PULLEY_HEIGHT = 0.50
HALF_SPAN     = 0.465

anchor_L = np.array([-HALF_SPAN, PULLEY_HEIGHT])   # left pulley (x, z)
anchor_R = np.array([ HALF_SPAN, PULLEY_HEIGHT])   # right pulley (x, z)

# Steps per metre of cable = (steps_per_rev * microstep) / (2π * spool_radius)
STEPS_PER_METRE = (STEPS_PER_REV * MICROSTEP) / (2.0 * np.pi * SPOOL_RADIUS)


# =========================================================
# FORWARD KINEMATICS
# Compute (x, z) of camera from two cable lengths.
# Uses trilateration in the x-z plane.
# =========================================================

def forward_kinematics(L_left: float, L_right: float):
    """
    Given left and right cable lengths, return camera (x, z) position.
    Anchors are at anchor_L and anchor_R in the x-z plane.
    """
    x1, z1 = anchor_L
    x2, z2 = anchor_R
    d      = np.linalg.norm(anchor_R - anchor_L)  # span between pulleys

    # Clamp cable lengths to physically reachable range
    L_left  = np.clip(L_left,  abs(x1 - x2) * 0.01, d + 0.5)
    L_right = np.clip(L_right, abs(x1 - x2) * 0.01, d + 0.5)

    # Cosine rule to find angle at left anchor
    cos_a = (L_left**2 + d**2 - L_right**2) / (2.0 * L_left * d)
    cos_a = np.clip(cos_a, -1.0, 1.0)

    # Camera x relative to left anchor
    dx = L_left * cos_a
    dz = -np.sqrt(max(L_left**2 - dx**2, 0.0))   # camera hangs below pulleys

    x = x1 + dx
    z = z1 + dz

    return float(x), float(z)


# =========================================================
# MOVEMENT LIMITS
# =========================================================

MIN_X = -0.40
MAX_X =  0.40
MIN_Z =  0.05
MAX_Z =  0.45


# =========================================================
# SHARED STATE  (updated by serial reader thread)
# =========================================================

state_lock = threading.Lock()

# Cable lengths derived from stepper counts (metres)
cable_length_left  = np.linalg.norm(np.array([0.0, 0.05]) - anchor_L)
cable_length_right = np.linalg.norm(np.array([0.0, 0.05]) - anchor_R)

# Camera position derived via forward kinematics
camera_position = np.array([0.0, 0.0, 0.05], dtype=float)

# Current sensor readings (Amps) and derived tensions (N)
current_left   = 0.0
current_right  = 0.0
tension_left   = 0.0
tension_right  = 0.0

# Cumulative stepper counts
steps_left  = 0
steps_right = 0

running = True

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
    if SERIAL_PORT is not None:
        return SERIAL_PORT

    candidates = serial.tools.list_ports.comports()
    keywords   = ["cp210", "ch340", "ftdi", "arduino", "usb serial", "uart"]

    for port in candidates:
        desc = (port.description + port.hwid).lower()
        if any(k in desc for k in keywords):
            print(f"[SERIAL] Auto-detected: {port.device}  ({port.description})")
            return port.device

    if candidates:
        print(f"[SERIAL] Trying first port: {candidates[0].device}")
        return candidates[0].device

    return None


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
    current_left_A / current_right_A : motor current in Amps
    """
    try:
        parts = [v.strip() for v in line.strip().split(",")]
        if len(parts) >= 3:
            sl  = int(parts[0])
            sr  = int(parts[1])
            It  = float(parts[2])
            return sl, sr, It
    except (ValueError, IndexError):
        pass
    return None, None, None, None

def split_tension_by_angle(total_current: float, cam_x: float, cam_z: float):
    """
    Splits the total motor current into individual left/right
    cable tensions using the cable geometry.

    Force balance on camera:
        Horizontal:  T_L * sin(a_L) = T_R * sin(a_R)
        Vertical:    T_L * cos(a_L) + T_R * cos(a_R) = m * g

    From horizontal equilibrium:
        T_L / T_R = sin(a_R) / sin(a_L)

    Total force from current:
        F_total = Kt * I_total / spool_radius
    """

    # ── Cable geometry ────────────────────────────────────
    cam = np.array([cam_x, cam_z])

    vec_L = anchor_L - cam
    vec_R = anchor_R - cam

    len_L = max(np.linalg.norm(vec_L), 1e-6)
    len_R = max(np.linalg.norm(vec_R), 1e-6)

    unit_L = vec_L / len_L
    unit_R = vec_R / len_R

    # Angle from vertical (z axis)
    sin_L = max(abs(unit_L[0]), 1e-6)   # horizontal component
    sin_R = max(abs(unit_R[0]), 1e-6)
    cos_L = abs(unit_L[1])              # vertical component
    cos_R = abs(unit_R[1])

    # ── Tension ratio from horizontal equilibrium ─────────
    # T_L * sin_L = T_R * sin_R  →  T_L = T_R * (sin_R / sin_L)
    ratio = sin_R / sin_L              # T_L = ratio * T_R

    # ── Total tension from current ────────────────────────
    # Both motors share the total current
    # Total vertical force = m * g (static equilibrium)
    # T_total = T_L + T_R = T_R * (ratio + 1)
    # Use current as a scale factor on top of the geometric split
    F_total = abs(total_current) * CURRENT_TO_TENSION

    T_R = F_total / (ratio + 1)
    T_L = ratio * T_R

    T_L = np.clip(T_L, TENSION_MIN, TENSION_MAX)
    T_R = np.clip(T_R, TENSION_MIN, TENSION_MAX)

    return float(T_L), float(T_R)


# =========================================================
# SERIAL READER THREAD
# Reads stepper counts + current from Arduino and updates
# camera position via forward kinematics.
# =========================================================

def serial_reader_thread():

    global cable_length_left, cable_length_right
    global camera_position
    global current_left, current_right
    global tension_left, tension_right
    global steps_left, steps_right
    global running

    port = find_serial_port()

    if port is None:
        print("[SERIAL] ERROR: No serial port found.")
        return

    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        print(f"[SERIAL] Connected on {port}")
    except serial.SerialException as e:
        print(f"[SERIAL] Could not open port: {e}")
        return

    while running:

        try:
            raw = ser.readline().decode("utf-8", errors="ignore")
        except serial.SerialException:
            print("[SERIAL] Read error")
            continue

        if not raw.strip():
            continue

        sl, sr, It = parse_serial_line(raw)
        
        sl, sr, It = parse_serial_line(raw)

        if sl is None:
            continue

        # ── Cable length from stepper counts ──────────────
        L_left  = max(sl / STEPS_PER_METRE, 0.05)
        L_right = max(sr / STEPS_PER_METRE, 0.05)

        # ── Forward kinematics → camera position ──────────
        cam_x, cam_z = forward_kinematics(L_left, L_right)
        cam_x = np.clip(cam_x, MIN_X, MAX_X)
        cam_z = np.clip(cam_z, MIN_Z, MAX_Z)

        # ── Split current into left/right tension by angle ─
        t_left, t_right = split_tension_by_angle(It, cam_x, cam_z)

        # ── Write shared state ────────────────────────────
        with state_lock:
            steps_left         = sl
            steps_right        = sr
            cable_length_left  = L_left
            cable_length_right = L_right
            camera_position[:] = [cam_x, 0.0, cam_z]
            current_left       = It * (t_left  / max(t_left + t_right, 1e-6))
            current_right      = It * (t_right / max(t_left + t_right, 1e-6))
            tension_left       = t_left
            tension_right      = t_right

    ser.close()


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():

    global running

    data.mocap_pos[camera_mocap_id]  = [0.0, 0.0, 0.05]
    data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            with state_lock:
                pos = camera_position.copy()
                tL  = tension_left
                tR  = tension_right

            # ── Drive mocap camera to computed position ────
            data.mocap_pos[camera_mocap_id]  = pos
            data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]

            # ── Update cable colours from tension ─────────
            model.tendon_rgba[left_tendon_id]  = tension_to_color(tL)
            model.tendon_rgba[right_tendon_id] = tension_to_color(tR)

            mujoco.mj_forward(model, data)

            print(
                f"[SIM] x={pos[0]:+.3f}  z={pos[2]:+.3f} | "
                f"T_L={tL:.2f} N  T_R={tR:.2f} N"
            )

            viewer.sync()
            time.sleep(model.opt.timestep)
    running = False


# =========================================================
# TKINTER STATUS WINDOW
# =========================================================

def run_status_window():

    global running

    root = tk.Tk()
    root.title("Digital Twin – Motor & Tension Status")
    root.geometry("460x380")

    tk.Label(root, text="Cable Camera Digital Twin",
             font=("Arial", 14, "bold")).pack(pady=10)

    # ── Camera position ───────────────────────────────────
    pos_label = tk.Label(root, text="Position:  x = ---   z = ---",
                         font=("Courier", 11))
    pos_label.pack(pady=4)

    # ── Cable lengths ─────────────────────────────────────
    cable_label = tk.Label(root, text="Cables:  L = ---   R = ---",
                           font=("Courier", 10))
    cable_label.pack(pady=4)

    # ── Stepper counts ────────────────────────────────────
    steps_label = tk.Label(root, text="Steps:   L = ---   R = ---",
                           font=("Courier", 10))
    steps_label.pack(pady=4)

    # ── Current readings ──────────────────────────────────
    current_label = tk.Label(root, text="Current:  L = ---  R = ---  A",
                             font=("Courier", 10))
    current_label.pack(pady=4)

    # ── Tension readings ──────────────────────────────────
    tension_label = tk.Label(root, text="Tension:  L = ---  R = ---  N",
                             font=("Courier", 10))
    tension_label.pack(pady=4)

    # ── Reset / zero position button ──────────────────────
    def reset_position():
        """
        Reset the camera to the home position.
        NOTE: This resets the *display* only — the physical
        steppers must also be homed for the counts to match.
        """
        global camera_position
        with state_lock:
            camera_position[:] = [0.0, 0.0, 0.05]
        print("[SIM] Position reset to home.")

    tk.Button(root, text="Reset / Zero Position",
              command=reset_position).pack(pady=10)

    # ── Polling update ────────────────────────────────────
    def update_labels():
        with state_lock:
            pos  = camera_position.copy()
            LL   = cable_length_left
            LR   = cable_length_right
            sL   = steps_left
            sR   = steps_right
            iL   = current_left
            iR   = current_right
            tL   = tension_left
            tR   = tension_right

        pos_label.config(
            text=f"Position:  x = {pos[0]:+.3f} m   z = {pos[2]:+.3f} m"
        )
        cable_label.config(
            text=f"Cables:    L = {LL:.4f} m    R = {LR:.4f} m"
        )
        steps_label.config(
            text=f"Steps:     L = {sL:+d}    R = {sR:+d}"
        )
        current_label.config(
            text=f"Current:   L = {iL:.3f} A    R = {iR:.3f} A"
        )
        tension_label.config(
            text=f"Tension:   L = {tL:.2f} N    R = {tR:.2f} N"
        )

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
# FAKE STEPPER/CURRENT THREAD
# Simulates motor step counts + current for testing
# without physical hardware.
# =========================================================

def fake_motor_thread():

    global cable_length_left, cable_length_right
    global camera_position
    global current_left, current_right
    global tension_left, tension_right
    global steps_left, steps_right
    global running

    t0 = time.perf_counter()

    while running:

        t = time.perf_counter() - t0

        # ── Target trajectory (used only to compute step counts) ──
        x_target = 0.35 * np.sin(1.2 * t)
        z_target = 0.25 + 0.08 * np.sin(2.8 * t)

        # ── True cable lengths at target position ─────────
        LL = np.linalg.norm(np.array([x_target, z_target]) - anchor_L)
        LR = np.linalg.norm(np.array([x_target, z_target]) - anchor_R)

        # ── Convert to step counts (only sensor output) ───
        sL = int(LL * STEPS_PER_METRE)
        sR = int(LR * STEPS_PER_METRE)

        # ── From here identical to real hardware ──────────
        L_left  = max(sL / STEPS_PER_METRE, 0.05)
        L_right = max(sR / STEPS_PER_METRE, 0.05)

        cam_x, cam_z = forward_kinematics(L_left, L_right)
        cam_x = np.clip(cam_x, MIN_X, MAX_X)
        cam_z = np.clip(cam_z, MIN_Z, MAX_Z)

        # ── Simulate total current with noise ─────────────
        I_total  = 1.2 + 0.3 * np.sin(0.3 * t)
        I_total += 0.05 * np.random.randn()
        I_total  = max(I_total, 0.0)

        t_left, t_right = split_tension_by_angle(I_total, cam_x, cam_z)

        with state_lock:
            steps_left         = sL
            steps_right        = sR
            cable_length_left  = L_left
            cable_length_right = L_right
            camera_position[:] = [cam_x, 0.0, cam_z]
            current_left       = I_total * (t_left  / max(t_left + t_right, 1e-6))
            current_right      = I_total * (t_right / max(t_left + t_right, 1e-6))
            tension_left       = t_left
            tension_right      = t_right

        time.sleep(0.02)


# =========================================================
# START ALL THREADS
# =========================================================

# Switch between real hardware and fake simulation:
#   real hardware  → serial_reader_thread
#   testing        → fake_motor_thread

# threading.Thread(target=serial_reader_thread, daemon=True).start()
threading.Thread(target=fake_motor_thread, daemon=True).start()

threading.Thread(target=run_mujoco, daemon=True).start()

run_status_window()