# =========================================================
# camera_support_sim.py
# Digital twin – physical IMU drives the MuJoCo simulation
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
# IMU SERIAL CONFIG
# =========================================================

IMU_BAUD     = 115200
IMU_PORT = "COM7"        # None = auto-detect, or set e.g. "COM3" / "/dev/ttyUSB0"
IMU_TIMEOUT  = 0.01        # seconds


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
                  euler="0 52 0" size="0.010 0.00075 0.0725"
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
        <!-- CAMERA (mocap = driven by IMU data)               -->
        <!-- ================================================= -->

        <body name="camera" mocap="true" pos="0 0 0.20">

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


# =========================================================
# SYSTEM DIMENSIONS
# =========================================================

PULLEY_HEIGHT = 0.50
HALF_SPAN     = 0.465

anchor_1 = np.array([-HALF_SPAN, PULLEY_HEIGHT])
anchor_2 = np.array([ HALF_SPAN, PULLEY_HEIGHT])

spool_r           = 0.01
left_motor_angle  = 0.0
right_motor_angle = 0.0
prev_pos          = np.array([0.0, 0.20])


# =========================================================
# SHARED IMU STATE
# =========================================================

imu_lock = threading.Lock()

# Position estimated from physical IMU (metres)
imu_position = np.array([0.0, 0.0, 0.05], dtype=float)

# Raw readings (for GUI display)
imu_accel_raw = np.zeros(3)
imu_gyro_raw  = np.zeros(3)

# Integration state
imu_velocity  = np.zeros(3)

# Gravity in world frame
GRAVITY = np.array([0.0, 0.0, 9.81])

# Complementary filter coefficient (0 = trust accel only, 1 = trust integration only)
ALPHA = 0.98

running = True


# =========================================================
# MOVEMENT LIMITS
# =========================================================

MIN_X = -0.40
MAX_X =  0.40
MIN_Z =  0.10
MAX_Z =  0.45


# =========================================================
# AUTO-DETECT IMU PORT
# =========================================================

def find_imu_port():
    """
    Try to auto-detect the IMU serial port.
    Looks for common USB-serial chips (CP210x, CH340, FTDI).
    Override IMU_PORT at the top of the file if auto-detect fails.
    """
    if IMU_PORT is not None:
        return IMU_PORT

    candidates = serial.tools.list_ports.comports()
    keywords   = ["cp210", "ch340", "ftdi", "arduino", "usb serial", "uart"]

    for port in candidates:
        desc = (port.description + port.hwid).lower()
        if any(k in desc for k in keywords):
            print(f"[IMU] Auto-detected port: {port.device}  ({port.description})")
            return port.device

    # Fall back to first available port
    if candidates:
        print(f"[IMU] No match found, trying first port: {candidates[0].device}")
        return candidates[0].device

    return None


# =========================================================
# PARSE IMU LINE
# =========================================================

def parse_imu_line(line: str):

    """
    Expected format:

    ax,ay,az,gx,gy,gz,qw,qx,qy,qz

    Example:
    0.01,0.02,9.80,0.001,0.002,0.000,1,0,0,0
    """
    try:
        parts = [float(v) for v in line.strip().split(",")]
        if len(parts) >= 7:
            accel = np.array(parts[0:3], dtype=float)
            gyro = np.array(parts[3:6], dtype=float)
            return accel, gyro
    except ValueError:
        pass
    return None, None, None

# =========================================================
# IMU READER THREAD
# Reads serial data from the physical IMU and integrates
# acceleration to estimate the camera position.
# =========================================================

def imu_reader_thread():

    global imu_position
    global imu_velocity
    global imu_accel_raw
    global imu_gyro_raw
    global imu_quat
    global running

    port = find_imu_port()
    
    if port is None:
        print("[IMU] ERROR: No serial port found.")
        return

    try:

        ser = serial.Serial(
            port,
            IMU_BAUD,
            timeout=IMU_TIMEOUT
        )

        print(f"[IMU] Connected on {port}")

    except serial.SerialException as e:

        print(f"[IMU] Could not open port: {e}")
        return

    prev_time = time.perf_counter()

    while running:

        try:

            raw = ser.readline().decode(
                "utf-8",
                errors="ignore"
            )

        except serial.SerialException:

            print("[IMU] Serial read error")
            continue

        if not raw.strip():
            continue

        accel, gyro, quat = parse_imu_line(raw)

        if accel is None:
            continue

        # ============================================
        # TIMING
        # ============================================

        now = time.perf_counter()

        dt = now - prev_time

        dt = np.clip(dt, 1e-4, 0.05)

        prev_time = now

        # ============================================
        # REMOVE GRAVITY
        # ============================================

        accel_linear = accel - np.array([0, 0, 9.81])

        # deadband filter

        accel_linear[np.abs(accel_linear) < 0.08] = 0.0

        # ============================================
        # INTEGRATE
        # ============================================

        imu_velocity += accel_linear * dt

        # velocity damping

        imu_velocity *= 0.98

        imu_position += imu_velocity * dt

        # ============================================
        # LIMITS
        # ============================================

        imu_position[0] = np.clip(
            imu_position[0],
            MIN_X,
            MAX_X
        )

        imu_position[2] = np.clip(
            imu_position[2],
            MIN_Z,
            MAX_Z
        )

        # ============================================
        # SAVE STATE
        # ============================================

        with imu_lock:

            imu_accel_raw[:] = accel

            imu_gyro_raw[:] = gyro

            imu_quat[:] = quat

    ser.close()

# =========================================================
# INVERSE KINEMATICS
# =========================================================

def inverse_kinematics(des_x, des_z):

    global prev_pos

    des_pos = np.array([des_x, des_z])
    L1      = np.linalg.norm(des_pos - anchor_1)
    L2      = np.linalg.norm(des_pos - anchor_2)
    L1_prev = np.linalg.norm(prev_pos - anchor_1)
    L2_prev = np.linalg.norm(prev_pos - anchor_2)

    dtheta1 = np.degrees((L1 - L1_prev) / spool_r)
    dtheta2 = np.degrees((L2 - L2_prev) / spool_r)

    prev_pos = des_pos.copy()

    return dtheta1, dtheta2, L1, L2


# =========================================================
# MUJOCO THREAD
# =========================================================

def run_mujoco():

    global running, left_motor_angle, right_motor_angle

    data.mocap_pos[camera_mocap_id]  = [0.0, 0.0, 0.20]
    data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and running:

            dt = model.opt.timestep

            # ── Read latest IMU position ──────────────────
            with imu_lock:
                pos   = imu_position.copy()
                accel = imu_accel_raw.copy()

            # ── IK ────────────────────────────────────────
            dtheta1, dtheta2, L1, L2 = inverse_kinematics(
                pos[0], pos[2]
            )

            left_motor_angle  += dtheta1
            right_motor_angle += dtheta2

            # ── Drive mocap camera to IMU position ────────
            data.mocap_pos[camera_mocap_id]  = pos
            data.mocap_quat[camera_mocap_id] = [1.0, 0.0, 0.0, 0.0]

            mujoco.mj_forward(model, data)

            print(
                f"[SIM] x={pos[0]:+.3f}  z={pos[2]:+.3f} | "
                f"accel={accel} | "
                f"L={L1:.3f} m  R={L2:.3f} m | "
                f"L motor={left_motor_angle:.1f}°  "
                f"R motor={right_motor_angle:.1f}°"
            )

            viewer.sync()
            time.sleep(dt)

    running = False


# =========================================================
# TKINTER STATUS WINDOW
# (sliders replaced by live readouts from the IMU)
# =========================================================

def run_status_window():

    global running

    root = tk.Tk()
    root.title("Digital Twin – IMU Status")
    root.geometry("420x320")

    tk.Label(root, text="Cable Camera Digital Twin",
             font=("Arial", 14, "bold")).pack(pady=10)

    # ── Live position display ─────────────────────────────
    pos_label = tk.Label(root, text="Position:  x = ---   z = ---",
                         font=("Courier", 11))
    pos_label.pack(pady=4)

    # ── Raw accelerometer display ─────────────────────────
    accel_label = tk.Label(root, text="Accel (m/s²):  ax=---  ay=---  az=---",
                           font=("Courier", 10))
    accel_label.pack(pady=4)

    # ── Raw gyro display ──────────────────────────────────
    gyro_label = tk.Label(root, text="Gyro  (rad/s):  gx=---  gy=---  gz=---",
                          font=("Courier", 10))
    gyro_label.pack(pady=4)

    # ── Cable lengths ─────────────────────────────────────
    cable_label = tk.Label(root, text="Cables:  L = ---   R = ---",
                           font=("Courier", 10))
    cable_label.pack(pady=4)

    # ── Reset drift button ────────────────────────────────
    def reset_drift():
        """Zero the velocity integrator to remove accumulated drift."""
        global imu_velocity, imu_position
        with imu_lock:
            imu_velocity[:] = 0.0
            imu_position[:]  = [0.0, 0.0, 0.20]
        print("[IMU] Drift reset.")

    tk.Button(root, text="Reset / Zero Position",
              command=reset_drift).pack(pady=10)

    # ── Polling update ────────────────────────────────────
    def update_labels():
        with imu_lock:
            pos   = imu_position.copy()
            accel = imu_accel_raw.copy()
            gyro  = imu_gyro_raw.copy()

        L1 = np.linalg.norm(np.array([pos[0], pos[2]]) - anchor_1)
        L2 = np.linalg.norm(np.array([pos[0], pos[2]]) - anchor_2)

        pos_label.config(
            text=f"Position:  x = {pos[0]:+.3f} m   z = {pos[2]:+.3f} m"
        )
        accel_label.config(
            text=f"Accel:  {accel[0]:+.2f}  {accel[1]:+.2f}  {accel[2]:+.2f}  m/s²"
        )
        gyro_label.config(
            text=f"Gyro:   {gyro[0]:+.4f}  {gyro[1]:+.4f}  {gyro[2]:+.4f}  rad/s"
        )
        cable_label.config(
            text=f"Cables:  L = {L1:.3f} m    R = {L2:.3f} m"
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
# FAKE IMU THREAD
# Generates simulated IMU motion for testing
# =========================================================

def fake_imu_thread():

    global imu_position
    global imu_quat
    global imu_accel_raw
    global imu_gyro_raw
    global running

    t0 = time.perf_counter()

    while running:

        t = time.perf_counter() - t0

        # ============================================
        # SIMULATED POSITION
        # ============================================

        # horizontal sine motion
        x = 0.20 * np.sin(0.5 * t)

        # vertical sine motion
        z = 0.25 + 0.08 * np.sin(1.2 * t)

        # ============================================
        # SIMULATED VELOCITY
        # ============================================

        vx = 0.20 * 0.5 * np.cos(0.5 * t)

        vz = 0.08 * 1.2 * np.cos(1.2 * t)

        # ============================================
        # SIMULATED ACCELERATION
        # ============================================

        ax = -0.20 * (0.5**2) * np.sin(0.5 * t)

        az = -0.08 * (1.2**2) * np.sin(1.2 * t)

        # include gravity
        accel = np.array([
            ax,
            0.0,
            9.81 + az
        ])

        # ============================================
        # SIMULATED ORIENTATION
        # ============================================

        # small oscillating roll angle

        roll = np.radians(
            10 * np.sin(0.8 * t)
        )

        # quaternion from roll

        qw = np.cos(roll / 2)

        qx = np.sin(roll / 2)

        quat = np.array([
            qw,
            qx,
            0.0,
            0.0
        ])

        # ============================================
        # WRITE SHARED STATE
        # ============================================

        with imu_lock:

            imu_position[:] = [x, 0.0, z]

            imu_quat[:] = quat

            imu_accel_raw[:] = accel

            imu_gyro_raw[:] = [0.1, 0.5, 0.1]

        time.sleep(0.01)


# =========================================================
# START ALL THREADS
# =========================================================

threading.Thread(target=imu_reader_thread, daemon=True).start()
threading.Thread(target=run_mujoco,        daemon=True).start()

run_status_window()


