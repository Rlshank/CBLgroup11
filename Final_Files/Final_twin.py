# =========================================================
# camera_support_sim.py
# Digital twin – Arduino sends position + tension directly
# Rope lengths calculated from position geometry
# Tension warning label + joystick scaling feedback
# Boundary warning label when camera near limits
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
SERIAL_PORT    = "COM4"
SERIAL_TIMEOUT = 0.01


# =========================================================
# PHYSICAL SYSTEM CONFIG
# =========================================================

CAMERA_MASS = 0.20
GRAVITY     = 9.81

TENSION_MIN = 0.0
TENSION_MAX = 50.0

# Tension thresholds
TENSION_WARN_N  = 3.5  # N – start reducing joystick speed
TENSION_RED_N   = 4.5  # N – cable turns red, show popup


# =========================================================
# BOUNDARY WARNING
# =========================================================

BOUNDARY_WARN_MARGIN = 0.05   # metres – warn when within 5 cm of limit


# =========================================================
# GEOMETRY
# =========================================================

SPAN          = 0.93
HALF_SPAN     = SPAN / 2.0
PULLEY_HEIGHT = 0.505

ax1, ay1 = 0.0,  PULLEY_HEIGHT
ax2, ay2 = SPAN, PULLEY_HEIGHT

anchor_L_sim = np.array([-HALF_SPAN, PULLEY_HEIGHT], dtype=float)
anchor_R_sim = np.array([ HALF_SPAN, PULLEY_HEIGHT], dtype=float)

# Movement limits in MuJoCo coordinates
MIN_X_SIM = -HALF_SPAN
MAX_X_SIM =  HALF_SPAN
MIN_Z_SIM =  0.0
MAX_Z_SIM =  PULLEY_HEIGHT

# Home position in INV_KIN coordinates
cam_x = HALF_SPAN
cam_y = 0.05


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

            <geom type="capsule" fromto="0 0 0.15 -0.09 0 0.29"
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

        <!-- RIGHT SUPPORT -->
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

            <geom type="capsule" fromto="0 0 0.15 0.09 0 0.29"
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

        <!-- CAMERA -->
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

left_tendon_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "left_cable")
right_tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "right_cable")


# =========================================================
# GEOMETRY HELPERS
# =========================================================

def rope_lengths(x, y):
    l1 = math.sqrt((x - ax1)**2 + (y - ay1)**2)
    l2 = math.sqrt((x - ax2)**2 + (y - ay2)**2)
    return l1, l2


def inv_to_sim_position(x_inv, y_inv):
    sim_x = x_inv - HALF_SPAN
    sim_z = y_inv
    return np.array([sim_x, 0.0, sim_z], dtype=float)


# =========================================================
# TENSION TO COLOUR
# =========================================================

def tension_to_color(T):

    T_MAX = TENSION_RED_N
    ratio = np.clip(T / T_MAX, 0.0, 1.0)

    if ratio < 0.25:
        t     = ratio / 0.25
        red, green, blue = 0.0, t, 1.0 - t

    elif ratio < 0.50:
        t     = (ratio - 0.25) / 0.25
        red, green, blue = t, 1.0, 0.0

    elif ratio < 0.75:
        t     = (ratio - 0.50) / 0.25
        red, green, blue = 1.0, 1.0 - 0.35 * t, 0.0

    else:
        t     = (ratio - 0.75) / 0.25
        red, green, blue = 1.0, 0.65 * (1.0 - t), 0.0

    return np.array([red, green, blue, 1.0])


# =========================================================
# JOYSTICK SCALE FROM TENSION
# Returns 0.0 – 1.0 scale factor sent back to Arduino
# =========================================================

def tension_speed_scale(t_left, t_right):
    """
    Linearly reduce joystick speed between TENSION_WARN_N
    and TENSION_RED_N. At or above TENSION_RED_N → scale = 0.
    """
    T_max = max(t_left, t_right)

    if T_max >= TENSION_RED_N:
        return 0.0

    if T_max <= TENSION_WARN_N:
        return 1.0

    return 1.0 - (T_max - TENSION_WARN_N) / (TENSION_RED_N - TENSION_WARN_N)


# =========================================================
# BOUNDARY CHECK
# Returns list of warning strings (empty = all clear)
# =========================================================

def boundary_warnings(pos):
    """
    pos: MuJoCo position [x, y, z]
    Returns a list of warning strings if camera is within
    BOUNDARY_WARN_MARGIN of any limit.
    """
    warnings = []
    m = BOUNDARY_WARN_MARGIN

    if pos[0] < MIN_X_SIM + m:
        warnings.append("⚠ Near LEFT boundary")
    if pos[0] > MAX_X_SIM - m:
        warnings.append("⚠ Near RIGHT boundary")
    if pos[2] < MIN_Z_SIM + m:
        warnings.append("⚠ Near BOTTOM boundary")
    if pos[2] > MAX_Z_SIM - m:
        warnings.append("⚠ Near TOP boundary")

    return warnings


# =========================================================
# SHARED STATE
# =========================================================

state_lock = threading.Lock()

camera_position    = inv_to_sim_position(cam_x, cam_y)
cable_length_left  = 0.0
cable_length_right = 0.0
tension_left       = 0.0
tension_right      = 0.0
joystick_rx        = 512
joystick_ry        = 512
speed_scale        = 1.0    # sent back to Arduino

running            = True

# Warning messages are now shown as labels in the Tkinter window.
# No pop-up/messagebox warnings are used.


# =========================================================
# SERIAL HELPERS
# =========================================================

def find_serial_port():
    if SERIAL_PORT is not None:
        return SERIAL_PORT

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


def parse_arduino_line(line):
    """
    Expected format:
        POS,x,y,joystick_x,joystick_y,loadcell1,loadcell2

    x, y          : camera position in INV_KIN coordinates (metres)
    joystick_x/y  : raw joystick ADC values (0–1023)
    loadcell1/2   : cable tensions in Newtons
    """
    line = line.strip()

    if not line.startswith("POS,"):
        return None

    try:
        parts = line.split(",")

        if len(parts) < 7:
            return None

        x         = float(parts[1])
        y         = float(parts[2])
        joy_x     = float(parts[3])
        joy_y     = float(parts[4])
        tension_l = float(parts[5])
        tension_r = float(parts[6])

        return -x, y, joy_x, joy_y, tension_r, tension_l

    except ValueError:
        return None


# =========================================================
# ARDUINO SERIAL THREAD
# =========================================================

def arduino_position_thread():

    global cam_x, cam_y
    global camera_position
    global cable_length_left, cable_length_right
    global tension_left, tension_right
    global joystick_rx, joystick_ry
    global speed_scale
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
        print(f"[SERIAL] Could not connect: {e}")
        return

    last_scale_sent = 1.0

    while running:

        try:
            raw = ser.readline().decode(errors="ignore").strip()
        except serial.SerialException:
            print("[SERIAL] Read error")
            continue

        parsed = parse_arduino_line(raw)

        if parsed is None:
            print(f"[PARSE FAIL] raw={raw!r}")
            continue

        arduino_x, arduino_y, joy_x, joy_y, t_left, t_right = parsed

        # ── Rope lengths from geometry ────────────────────
        inv_x = arduino_x + HALF_SPAN
        inv_y = arduino_y
        l1, l2 = rope_lengths(inv_x, inv_y)

        sim_pos = inv_to_sim_position(inv_x, inv_y)

        # ── Clamp tensions ────────────────────────────────
        t_left  = np.clip(abs(t_left),  TENSION_MIN, TENSION_MAX)
        t_right = np.clip(abs(t_right), TENSION_MIN, TENSION_MAX)

        # ── Compute joystick scale from tension ───────────
        scale = tension_speed_scale(t_left, t_right)

        # Send scale back to Arduino only when it changes
        # Arduino multiplies joystick input by this value
        # Format: "S0.75\n"
        if abs(scale - last_scale_sent) > 0.01:
            try:
                ser.write(f"S{scale:.2f}\n".encode())
                last_scale_sent = scale
                print(f"[SERIAL] Sent scale: {scale:.2f}")
            except serial.SerialException:
                pass

        # ── Write shared state ────────────────────────────
        with state_lock:
            cam_x              = inv_x
            cam_y              = inv_y
            camera_position[:] = sim_pos
            cable_length_left  = l1
            cable_length_right = l2
            tension_left       = t_left
            tension_right      = t_right
            joystick_rx        = joy_x
            joystick_ry        = joy_y
            speed_scale        = scale

        print(
            f"[ARDUINO] x={arduino_x:+.3f} y={arduino_y:+.3f} | "
            f"L1={l1:.3f}m L2={l2:.3f}m | "
            f"T_L={t_left:.2f}N T_R={t_right:.2f}N | "
            f"scale={scale:.2f}"
        )

    try:
        ser.close()
    except Exception:
        pass


# =========================================================
# FAKE TEST THREAD
# =========================================================

def fake_inv_kin_thread():

    global cam_x, cam_y
    global camera_position
    global cable_length_left, cable_length_right
    global tension_left, tension_right
    global joystick_rx, joystick_ry
    global speed_scale
    global running

    t0 = time.perf_counter()

    while running:

        t = time.perf_counter() - t0

        inv_x = HALF_SPAN + 0.25 * math.sin(0.5 * t)
        inv_y = 0.22 + 0.08 * math.sin(0.9 * t)

        inv_x = max(0.0,  min(SPAN, inv_x))
        inv_y = max(0.05, min(0.5,  inv_y))

        l1, l2 = rope_lengths(inv_x, inv_y)

        # Static tension from geometry
        cam    = np.array([inv_x - HALF_SPAN, inv_y], dtype=float)
        vec_L  = anchor_L_sim - cam
        vec_R  = anchor_R_sim - cam
        unit_L = vec_L / max(np.linalg.norm(vec_L), 1e-6)
        unit_R = vec_R / max(np.linalg.norm(vec_R), 1e-6)

        A = np.array([[unit_L[0], unit_R[0]],
                      [unit_L[1], unit_R[1]]], dtype=float)
        b = np.array([0.0, CAMERA_MASS * GRAVITY], dtype=float)

        try:
            tensions = np.linalg.solve(A, b)
            t_left  = float(np.clip(tensions[0], TENSION_MIN, TENSION_MAX))
            t_right = float(np.clip(tensions[1], TENSION_MIN, TENSION_MAX))
        except np.linalg.LinAlgError:
            t_left  = 0.0
            t_right = 0.0

        t_left  += 0.02 * np.random.randn()
        t_right += 0.02 * np.random.randn()
        t_left  = float(np.clip(t_left,  TENSION_MIN, TENSION_MAX))
        t_right = float(np.clip(t_right, TENSION_MIN, TENSION_MAX))

        scale   = tension_speed_scale(t_left, t_right)
        sim_pos = inv_to_sim_position(inv_x, inv_y)

        with state_lock:
            cam_x              = inv_x
            cam_y              = inv_y
            camera_position[:] = sim_pos
            cable_length_left  = l1
            cable_length_right = l2
            tension_left       = t_left
            tension_right      = t_right
            joystick_rx        = 512
            joystick_ry        = 512
            speed_scale        = scale

        time.sleep(0.05)


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():

    global running

    data.mocap_pos[camera_mocap_id]  = camera_position.copy()
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

            model.tendon_rgba[left_tendon_id]  = tension_to_color(tL)
            model.tendon_rgba[right_tendon_id] = tension_to_color(tR)

            mujoco.mj_forward(model, data)

            print(
                f"[SIM] x={pos[0]:+.3f} z={pos[2]:+.3f} | "
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
    root.title("Digital Twin – Cable Camera")
    root.geometry("560x500")

    tk.Label(root, text="Cable Camera Digital Twin",
             font=("Arial", 14, "bold")).pack(pady=8)

    tk.Label(root, text="Position from Arduino  |  Tension from load cells",
             font=("Arial", 10)).pack(pady=2)

    pos_label     = tk.Label(root, text="Position:  ---",
                             font=("Courier", 10))
    pos_label.pack(pady=3)

    cable_label   = tk.Label(root, text="Cables:    ---",
                             font=("Courier", 10))
    cable_label.pack(pady=3)

    tension_label = tk.Label(root, text="Tension:   ---",
                             font=("Courier", 10))
    tension_label.pack(pady=3)

    scale_label   = tk.Label(root, text="Joy scale: ---",
                             font=("Courier", 10))
    scale_label.pack(pady=3)

    joy_label     = tk.Label(root, text="Joystick:  ---",
                             font=("Courier", 10))
    joy_label.pack(pady=3)

    # ── Warning labels (no pop-ups) ──────────────────────
    tension_warn_label = tk.Label(
        root,
        text="",
        font=("Courier", 10, "bold"),
        fg="red",
        wraplength=520,
        justify="center"
    )
    tension_warn_label.pack(pady=4)

    boundary_warn_label = tk.Label(
        root,
        text="",
        font=("Courier", 10, "bold"),
        fg="red",
        wraplength=520,
        justify="center"
    )
    boundary_warn_label.pack(pady=4)

    def reset_position():
        global cable_length_left, cable_length_right, tension_left, tension_right, speed_scale

        with state_lock:
            camera_position[:] = inv_to_sim_position(HALF_SPAN, 0.05)
            cable_length_left, cable_length_right = rope_lengths(HALF_SPAN, 0.05)
            tension_left = 0.0
            tension_right = 0.0
            speed_scale = 1.0

        print("[SIM] Reset to home.")

    tk.Button(root, text="Reset / Zero Position",
              command=reset_position).pack(pady=8)

    # ── Polling update ────────────────────────────────────
    def update_labels():

        with state_lock:
            pos   = camera_position.copy()
            LL    = cable_length_left
            LR    = cable_length_right
            tL    = tension_left
            tR    = tension_right
            scale = speed_scale
            rx    = joystick_rx
            ry    = joystick_ry

        # ── Update labels ─────────────────────────────────
        pos_label.config(
            text=f"Position:  x={pos[0]:+.3f} m   z={pos[2]:+.3f} m"
        )
        cable_label.config(
            text=f"Cables:    L1={LL:.4f} m   L2={LR:.4f} m"
        )
        tension_label.config(
            text=f"Tension:   T_R={tL:.2f} N   T_L={tR:.2f} N"
        )

        # Colour the scale label by severity
        pct        = int(scale * 100)
        scale_color = (
            "red"        if scale == 0.0  else
            "orange"     if scale < 0.75  else
            "dark green"
        )
        scale_label.config(
            text=f"Joy scale: {pct}%",
            fg=scale_color
        )

        joy_label.config(
            text=f"Joystick:  rx={rx}   ry={ry}"
        )

        # ── Tension warning label ─────────────────────────
        # This replaces the old pop-up warning.
        if max(tL, tR) >= TENSION_RED_N:
            tension_warn_label.config(
                text=(
                    "⚠ Cable tension is too high! "
                    "Joystick input has been reduced. "
                    "Move the camera back towards the centre."
                )
            )
        elif max(tL, tR) >= TENSION_WARN_N:
            tension_warn_label.config(
                text=(
                    "⚠ Cable tension is getting high. "
                    "Joystick speed is being reduced."
                )
            )
        else:
            tension_warn_label.config(text="")

        # ── Boundary warning label ────────────────────────
        # This replaces the old boundary pop-up.
        bwarns = boundary_warnings(pos)

        if bwarns:
            boundary_warn_label.config(
                text=(
                    "  ".join(bwarns)
                    + "  |  Camera is within 5 cm of the movement limit."
                )
            )
        else:
            boundary_warn_label.config(text="")

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

    # Real Arduino:
    threading.Thread(target=arduino_position_thread, daemon=True).start()

    # Testing without Arduino — comment above and uncomment below:
    #threading.Thread(target=fake_inv_kin_thread, daemon=True).start()

    threading.Thread(target=run_mujoco, daemon=True).start()

    run_status_window()