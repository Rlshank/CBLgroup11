# =========================================================
# camera_support_sim.py
# Digital twin – joystick INV_KIN drives camera position
# Tension is estimated from camera position change + mass
# =========================================================

import time
import threading
import tkinter as tk
import math
import numpy as np
import mujoco
import mujoco.viewer
import serial
import serial.tools.list_ports


# =========================================================
# SERIAL CONFIG
# =========================================================

SERIAL_BAUD    = 115200
SERIAL_PORT    = "COM9"      # Use your Arduino port, e.g. "COM4". Set to None for auto-detect.
SERIAL_TIMEOUT = 0.01


# =========================================================
# INV_KIN / STEPPER CONFIG
# =========================================================

SPOOL_RADIUS   = 0.0125      # metres
STEPS_PER_REV  = 200
MAX_SPEED      = 0.25        # m/s joystick max camera speed
DEADZONE       = 50
CONTROL_DT     = 0.05        # seconds, same as INV_KIN DT

MIN_STEP_DELAY_US = 500
MAX_STEP_DELAY_US = 8000


# =========================================================
# PHYSICAL SYSTEM CONFIG
# =========================================================

CAMERA_MASS = 0.20           # kg
GRAVITY     = 9.81           # m/s^2

TENSION_MIN = 0.0
TENSION_MAX = 50.0

# Optional damping term to make the tension estimate less jumpy
DAMPING = 0.05


# =========================================================
# GEOMETRY
# INV_KIN coordinates:
#   left anchor  = (0.00, 0.50)
#   right anchor = (0.93, 0.50)
#   camera x from 0.00 to 0.93
#   camera y = vertical height
#
# MuJoCo coordinates:
#   sim_x = cam_x - 0.465
#   sim_y = cam_y
# =========================================================

SPAN = 0.93
HALF_SPAN = SPAN / 2.0
PULLEY_HEIGHT = 0.50

ax1, ay1 = 0.0,  PULLEY_HEIGHT
ax2, ay2 = SPAN, PULLEY_HEIGHT

anchor_L_sim = np.array([-HALF_SPAN, PULLEY_HEIGHT], dtype=float)
anchor_R_sim = np.array([ HALF_SPAN, PULLEY_HEIGHT], dtype=float)

# Home position in INV_KIN coordinates
cam_x = HALF_SPAN
cam_y = 0.05

# Movement limits in INV_KIN coordinates
MIN_X = 0.0
MAX_X = SPAN
MIN_Y = 0
MAX_Y = 0.5


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

        <!-- LEFT SUPPORT -->
        <body name="left_support" pos="-0.485 0 0">

            <geom type="box" pos="0 0 0.15"
                  size="0.016 0.02 0.15"
                  rgba="0.52 0.40 0.25 1"/>

            <geom type="box" pos="-0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.065 0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.050 0.015 0.38"
                  euler="0 52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.050 -0.015 0.38"
                  euler="0 -52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="-0.043 0 0.30"
                  size="0.07 0.10 0.01"
                  rgba="0.9 0.8 0.9 1"/>

            <geom type="capsule"
                  fromto="0 0 0.15 -0.09 0 0.29"
                  size="0.015"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="cylinder"
                  size="0.01 0.01"
                  rgba="0.5 0.5 0.5 1"/>

            <site name="left_pulley"
                  pos="0 0 0.50"
                  size="0.01"
                  rgba="0 0 1 1"/>

            <geom type="box"
                  pos="-0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02"
                  rgba="0.1 0.1 0.1 1"/>

            <site name="left_motor_site"
                  pos="-0.0655 0.0 0.33"
                  size="0.008"
                  rgba="1 0 0 1"/>

            <geom type="cylinder"
                  pos="-0.0655 0.0 0.33"
                  euler="90 0 0"
                  size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>

            <geom type="cylinder"
                  pos="0 0.0 0.485"
                  euler="90 0 0"
                  size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>
        </body>

        <!-- RIGHT SUPPORT -->
        <body name="right_support" pos="0.485 0 0">

            <geom type="box" pos="0 0 0.15"
                  size="0.016 0.02 0.15"
                  rgba="0.52 0.40 0.25 1"/>

            <geom type="box" pos="0.0015 0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.0015 -0.015 0.40"
                  size="0.02 0.0015 0.10"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.065 0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.065 -0.015 0.33"
                  size="0.050 0.0015 0.02"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.050 0.015 0.38"
                  euler="0 -52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.050 -0.015 0.38"
                  euler="0 -52 0"
                  size="0.010 0.00075 0.0725"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="box" pos="0.043 0 0.30"
                  size="0.07 0.10 0.01"
                  rgba="0.9 0.8 0.9 1"/>

            <geom type="capsule"
                  fromto="0 0 0.15 0.09 0 0.29"
                  size="0.015"
                  rgba="0.72 0.5 0.35 1"/>

            <geom type="cylinder"
                  size="0.01 0.01"
                  rgba="0.5 0.5 0.5 1"/>

            <site name="right_pulley"
                  pos="0 0 0.50"
                  size="0.01"
                  rgba="0 0 1 1"/>

            <geom type="box"
                  pos="0.0655 0.0465 0.33"
                  size="0.02 0.03 0.02"
                  rgba="0.1 0.1 0.1 1"/>

            <site name="motor_site"
                  pos="0.0655 0.0 0.33"
                  size="0.008"
                  rgba="1 0 0 1"/>

            <geom type="cylinder"
                  pos="0.0655 0.0 0.33"
                  euler="90 0 0"
                  size="0.03 0.01 0.03"
                  rgba="0.15 0.15 0.15 1"/>

            <geom type="cylinder"
                  pos="0 0.0 0.485"
                  euler="90 0 0"
                  size="0.0175 0.01 0.0175"
                  rgba="0.15 0.15 0.15 1"/>
        </body>

        <!-- CAMERA -->
        <body name="camera" mocap="true" pos="0 0 0.05">

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
# INV_KIN FUNCTIONS
# =========================================================

def rope_lengths(x, y):
    l1 = math.sqrt((x - ax1) ** 2 + (y - ay1) ** 2)
    l2 = math.sqrt((x - ax2) ** 2 + (y - ay2) ** 2)
    return l1, l2


def speed_to_delay(cable_speed):
    if abs(cable_speed) < 0.001:
        return 0

    steps_per_sec = abs(cable_speed) / (2 * math.pi * SPOOL_RADIUS) * STEPS_PER_REV

    if steps_per_sec <= 0:
        return 0

    # Arduino toggles HIGH/LOW, so one full step happens every two toggles
    delay_us = int(1e6 / (2 * steps_per_sec))

    return max(MIN_STEP_DELAY_US, min(delay_us, MAX_STEP_DELAY_US))


def joystick_to_velocity(rx, ry):
    if abs(rx - 512) > DEADZONE:
        vx = (rx - 512) / 512.0
    else:
        vx = 0.0

    if abs(ry - 512) > DEADZONE:
        vy = -(ry - 512) / 512.0
    else:
        vy = 0.0

    vx *= MAX_SPEED
    vy *= MAX_SPEED

    return vx, vy


def send_stop(ser):
    ser.write(b"0,0,0,0\n")


def send_command(ser, d1, sp1, d2, sp2):
    cmd = f"{d1},{sp1},{d2},{sp2}\n"
    ser.write(cmd.encode())


def inv_to_sim_position(x_inv, y_inv):
    """
    INV_KIN:
        x = horizontal (left/right)
        y = vertical (up/down)

    MuJoCo:
        x = horizontal (left/right)
        z = vertical (up/down)
    """
    sim_x = x_inv - HALF_SPAN
    sim_y = y_inv

    return np.array([sim_x, 0.0, sim_y], dtype=float)


# =========================================================
# TENSION CALCULATION
# =========================================================

def calculate_dynamic_tension(pos_now, pos_prev, vel_prev, dt):
    """
    Estimate left/right cable tensions using camera acceleration and mass.

    Force balance:
        T_L*u_L + T_R*u_R + [0, -m*g] = m*a

    Rearranged:
        T_L*u_L + T_R*u_R = [m*a_x, m*(a_z + g)]

    u_L and u_R are unit vectors from camera to each pulley.
    """
    if dt <= 1e-6:
        return (
            0.0,
            0.0,
            np.array([0.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float)
        )

    p_now = np.array([pos_now[0], pos_now[2]], dtype=float)
    p_prev = np.array([pos_prev[0], pos_prev[2]], dtype=float)

    vel_now = (p_now - p_prev) / dt
    acc = (vel_now - vel_prev) / dt

    vec_L = anchor_L_sim - p_now
    vec_R = anchor_R_sim - p_now

    norm_L = np.linalg.norm(vec_L)
    norm_R = np.linalg.norm(vec_R)

    if norm_L < 1e-6 or norm_R < 1e-6:
        return 0.0, 0.0, vel_now, acc

    uL = vec_L / norm_L
    uR = vec_R / norm_R

    A = np.array([
        [uL[0], uR[0]],
        [uL[1], uR[1]]
    ], dtype=float)

    b = np.array([
        CAMERA_MASS * acc[0],
        CAMERA_MASS * (acc[1] + GRAVITY)
    ], dtype=float)

    try:
        tensions = np.linalg.solve(A, b)
        tL = float(tensions[0])
        tR = float(tensions[1])
    except np.linalg.LinAlgError:
        tL = 0.0
        tR = 0.0

    # Small extra damping tension during movement
    tL += DAMPING * abs(np.dot(vel_now, uL))
    tR += DAMPING * abs(np.dot(vel_now, uR))

    # Cables cannot push, so negative tension means slack
    tL = float(np.clip(max(tL, 0.0), TENSION_MIN, TENSION_MAX))
    tR = float(np.clip(max(tR, 0.0), TENSION_MIN, TENSION_MAX))

    return tL, tR, vel_now, acc

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
# SHARED STATE
# =========================================================

state_lock = threading.Lock()

camera_position = inv_to_sim_position(cam_x, cam_y)

velocity_xz = np.array([0.0, 0.0], dtype=float)
acceleration_xz = np.array([0.0, 0.0], dtype=float)

cable_length_left, cable_length_right = rope_lengths(cam_x, cam_y)

current_motor_2_mA = 0.0

tension_left = 0.0
tension_right = 0.0

step_delay_left = 0
step_delay_right = 0

direction_left = 0
direction_right = 0

joystick_rx = 512
joystick_ry = 512

running = True


# =========================================================
# SERIAL HELPERS
# =========================================================

def find_serial_port():
    if SERIAL_PORT is not None:
        return SERIAL_PORT

    candidates = serial.tools.list_ports.comports()

    keywords = [
        "cp210",
        "ch340",
        "ftdi",
        "arduino",
        "usb serial",
        "uart"
    ]

    for port in candidates:
        desc = (port.description + port.hwid).lower()

        if any(k in desc for k in keywords):
            print(f"[SERIAL] Auto-detected: {port.device} ({port.description})")
            return port.device

    if candidates:
        print(f"[SERIAL] Trying first port: {candidates[0].device}")
        return candidates[0].device

    return None


def parse_arduino_line(line):
    """
    Expected Arduino line format:
        POS,x,y,joy_x,joy_y,current_mA

    Example:
        POS,0.1200,0.2500,420,0,0.0
    """
    line = line.strip()

    if not line.startswith("POS,"):
        return None

    try:
        parts = line.split(",")

        if len(parts) < 6:
            return None

        x = float(parts[1])
        y = float(parts[2])
        joy_x = int(parts[3])
        joy_y = int(parts[4])
        current_mA = float(parts[5])

        return x, y, joy_x, joy_y, current_mA

    except ValueError:
        return None


# =========================================================
# SERIAL + INV_KIN THREAD
# =========================================================

def arduino_position_thread():
    global cam_x, cam_y
    global camera_position, velocity_xz, acceleration_xz
    global cable_length_left, cable_length_right
    global current_motor_2_mA
    global tension_left, tension_right
    global joystick_rx, joystick_ry
    global running

    port = find_serial_port()

    if port is None:
        print("[SERIAL] ERROR: No serial port found.")
        return

    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        print(f"[SERIAL] Connected on {port}")

    except serial.SerialException as e:
        print(f"[SERIAL] Could not connect to Arduino on {port}: {e}")
        return

    prev_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    prev_vel = np.array([0.0, 0.0], dtype=float)
    last_time = time.perf_counter()

    while running:
        try:
            raw = ser.readline().decode(errors="ignore").strip()
        except serial.SerialException:
            print("[SERIAL] Read error")
            continue

        parsed = parse_arduino_line(raw)

        if parsed is None:
            continue

        arduino_x, arduino_y, joy_x, joy_y, current_mA = parsed

        now = time.perf_counter()
        dt = max(now - last_time, 0.001)
        last_time = now

        # Arduino x is already centered: -0.465 to +0.465
        sim_pos = np.array([arduino_x, 0.0, arduino_y], dtype=float)

        # Convert to old INV_KIN coordinates only for cable length calculation
        cam_x = arduino_x + HALF_SPAN
        cam_y = arduino_y

        l1, l2 = rope_lengths(cam_x, cam_y)

        tL, tR, vel_now, acc_now = calculate_dynamic_tension(
            sim_pos,
            prev_pos,
            prev_vel,
            dt
        )

        with state_lock:
            camera_position[:] = sim_pos
            velocity_xz[:] = vel_now
            acceleration_xz[:] = acc_now

            cable_length_left = l1
            cable_length_right = l2

            current_motor_2_mA = current_mA

            tension_left = tL
            tension_right = tR

            joystick_rx = joy_x
            joystick_ry = joy_y

        prev_pos = sim_pos.copy()
        prev_vel = vel_now.copy()

        print(
            f"[ARDUINO] x={arduino_x:+.3f} y={arduino_y:+.3f} | "
            f"L1={l1:.3f} L2={l2:.3f} | "
            f"T_L={tL:.2f}N T_R={tR:.2f}N | "
            f"I={current_mA:.2f}mA"
        )

    try:
        ser.close()
    except Exception:
        pass

# =========================================================
# FAKE TEST THREAD WITHOUT ARDUINO
# =========================================================

def fake_inv_kin_thread():
    global cam_x, cam_y
    global camera_position, velocity_xz, acceleration_xz
    global cable_length_left, cable_length_right
    global current_motor_2_mA
    global tension_left, tension_right
    global step_delay_left, step_delay_right
    global direction_left, direction_right
    global joystick_rx, joystick_ry
    global running

    prev_pos = inv_to_sim_position(cam_x, cam_y)
    prev_vel = np.array([0.0, 0.0], dtype=float)

    t0 = time.perf_counter()

    while running:
        t = time.perf_counter() - t0

        cam_x = HALF_SPAN + 0.25 * math.sin(0.5 * t)
        cam_y = 0.22 + 0.08 * math.sin(0.9 * t)

        cam_x = max(MIN_X, min(MAX_X, cam_x))
        cam_y = max(MIN_Y, min(MAX_Y, cam_y))

        l1, l2 = rope_lengths(cam_x, cam_y)

        pos_now = inv_to_sim_position(cam_x, cam_y)

        tL, tR, vel_now, acc_now = calculate_dynamic_tension(
            pos_now,
            prev_pos,
            prev_vel,
            CONTROL_DT
        )

        with state_lock:
            camera_position[:] = pos_now
            velocity_xz[:] = vel_now
            acceleration_xz[:] = acc_now

            cable_length_left = l1
            cable_length_right = l2

            current_motor_2_mA = 0.0

            tension_left = tL
            tension_right = tR

            step_delay_left = 0
            step_delay_right = 0

            direction_left = 0
            direction_right = 0

            joystick_rx = 512
            joystick_ry = 512

        prev_pos = pos_now.copy()
        prev_vel = vel_now.copy()

        time.sleep(CONTROL_DT)


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():
    global running

    data.mocap_pos[camera_mocap_id] = camera_position.copy()
    data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        
        while viewer.is_running() and running:

            with state_lock:
                pos = camera_position.copy()
                LL  = cable_length_left
                LR  = cable_length_right
                tL  = tension_left
                tR  = tension_right

            data.mocap_pos[camera_mocap_id]  = pos
            data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]

            # ── Update cable colours from tension ─────────
            model.tendon_rgba[left_tendon_id]  = tension_to_color(tL)
            model.tendon_rgba[right_tendon_id] = tension_to_color(tR)

            mujoco.mj_forward(model, data)

            print(
                f"[SIM] x={pos[0]:+.3f} y={pos[2]:+.3f} | "
                f"L1={LL:.3f}m L2={LR:.3f}m | "
                f"T_L={tL:.2f}N T_R={tR:.2f}N"
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
    root.title("Digital Twin – INV_KIN & Tension")
    root.geometry("520x460")

    tk.Label(
        root,
        text="Cable Camera Digital Twin",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    tk.Label(
        root,
        text="Position from INV_KIN output + dynamic tension estimate",
        font=("Arial", 10)
    ).pack(pady=2)

    pos_label = tk.Label(root, text="Position: ---", font=("Courier", 10))
    pos_label.pack(pady=5)

    inv_label = tk.Label(root, text="INV_KIN position: ---", font=("Courier", 10))
    inv_label.pack(pady=5)

    cable_label = tk.Label(root, text="Cables: ---", font=("Courier", 10))
    cable_label.pack(pady=5)

    speed_label = tk.Label(root, text="Velocity / Acceleration: ---", font=("Courier", 10))
    speed_label.pack(pady=5)

    tension_label = tk.Label(root, text="Tension: ---", font=("Courier", 10))
    tension_label.pack(pady=5)

    motor_label = tk.Label(root, text="Motor command: ---", font=("Courier", 10))
    motor_label.pack(pady=5)

    current_label = tk.Label(root, text="Current sensor: ---", font=("Courier", 10))
    current_label.pack(pady=5)

    joystick_label = tk.Label(root, text="Joystick: ---", font=("Courier", 10))
    joystick_label.pack(pady=5)

    def reset_position():
        global cam_x, cam_y, camera_position, velocity_xz, acceleration_xz
        global cable_length_left, cable_length_right

        with state_lock:
            cam_x = HALF_SPAN
            cam_y = 0

            camera_position[:] = inv_to_sim_position(cam_x, cam_y)
            velocity_xz[:] = [0.0, 0.0]
            acceleration_xz[:] = [0.0, 0.0]

            cable_length_left, cable_length_right = rope_lengths(cam_x, cam_y)

        print("[SIM] Reset to home position.")

    tk.Button(
        root,
        text="Reset / Zero Position",
        command=reset_position
    ).pack(pady=12)

    def update_labels():
        with state_lock:
            pos = camera_position.copy()
            vel = velocity_xz.copy()
            acc = acceleration_xz.copy()

            LL = cable_length_left
            LR = cable_length_right

            tL = tension_left
            tR = tension_right

            sp1 = step_delay_left
            sp2 = step_delay_right

            d1 = direction_left
            d2 = direction_right

            i2 = current_motor_2_mA

            rx = joystick_rx
            ry = joystick_ry

            inv_x = cam_x
            inv_y = cam_y

        pos_label.config(
            text=f"MuJoCo pos:   x={pos[0]:+.3f} m   y={pos[2]:+.3f} m"
        )

        inv_label.config(
            text=f"INV_KIN pos:  x={inv_x:+.3f} m   y={inv_y:+.3f} m"
        )

        cable_label.config(
            text=f"Cables:       L1={LL:.4f} m   L2={LR:.4f} m"
        )

        speed_label.config(
            text=f"v=({vel[0]:+.2f},{vel[1]:+.2f}) m/s   "
                 f"a=({acc[0]:+.2f},{acc[1]:+.2f}) m/s²"
        )

        tension_label.config(
            text=f"Tension:      T_L={tL:.2f} N   T_R={tR:.2f} N"
        )

        motor_label.config(
            text=f"Motor cmd:    d1={d1} sp1={sp1} us   d2={d2} sp2={sp2} us"
        )

        current_label.config(
            text=f"Current:      I2={i2:.2f} mA"
        )

        joystick_label.config(
            text=f"Joystick:     rx={rx}   ry={ry}"
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
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    # Use real Arduino / joystick:
    threading.Thread(target=arduino_position_thread, daemon=True).start()

    # For testing without Arduino, comment the line above and uncomment this:
    # threading.Thread(target=fake_inv_kin_thread, daemon=True).start()

    threading.Thread(target=run_mujoco, daemon=True).start()

    run_status_window()