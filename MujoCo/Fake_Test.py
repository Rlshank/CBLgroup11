# =========================================================
# fake_arduino.py
# Simulates the Arduino serial output for testing
# camera_support_sim.py without physical hardware.
#
# Creates a virtual serial port pair and streams fake
# stepper counts + current sensor data.
#
# USAGE:
#   1. Install dependencies:
#        pip install pyserial
#        # Linux/Mac also needs:
#        pip install ptyprocess
#
#   2. Run this script first:
#        python fake_arduino.py
#
#   3. It will print the virtual port name, e.g.:
#        [FAKE] Virtual port: /dev/pts/3   (Linux/Mac)
#        [FAKE] Using loopback port: COM7  (Windows)
#
#   4. Set that port in camera_support_sim.py:
#        SERIAL_PORT = "/dev/pts/3"   (Linux/Mac)
#        SERIAL_PORT = "COM7"         (Windows)
#
#   5. Run camera_support_sim.py normally.
#
# MOTION PROFILES (set PROFILE below):
#   "sine"     – smooth horizontal + vertical sine sweep
#   "zigzag"   – sharp left/right traversal
#   "circle"   – circular arc motion
#   "hover"    – gentle hover with small disturbances
#   "manual"   – control position with keyboard (WASD)
# =========================================================

import sys
import time
import math
import threading
import numpy as np

# =========================================================
# CONFIG
# =========================================================

PROFILE        = "sine"     # sine | zigzag | circle | hover | manual
BAUD           = 115200
UPDATE_RATE_HZ = 50         # how many packets per second to send

# Must match camera_support_sim.py exactly
STEPS_PER_REV  = 200
MICROSTEP      = 16
SPOOL_RADIUS   = 0.01275    # metres
PULLEY_HEIGHT  = 0.50
HALF_SPAN      = 0.465

STEPS_PER_METRE = (STEPS_PER_REV * MICROSTEP) / (2.0 * math.pi * SPOOL_RADIUS)

anchor_L = np.array([-HALF_SPAN, PULLEY_HEIGHT])
anchor_R = np.array([ HALF_SPAN, PULLEY_HEIGHT])

# Current noise amplitude (Amps)
CURRENT_NOISE = 0.04

# Base current proportional to tension (rough model)
BASE_CURRENT  = 1.2


# =========================================================
# GEOMETRY HELPERS
# =========================================================

def camera_to_steps(x: float, z: float):
    """Convert camera (x, z) to left/right stepper counts."""
    pos  = np.array([x, z])
    LL   = float(np.linalg.norm(pos - anchor_L))
    LR   = float(np.linalg.norm(pos - anchor_R))
    sL   = int(LL * STEPS_PER_METRE)
    sR   = int(LR * STEPS_PER_METRE)
    return sL, sR, LL, LR


def tension_to_current(tension_N: float) -> float:
    """Rough inverse of CURRENT_TO_TENSION=5.0 in sim."""
    return tension_N / 5.0 + CURRENT_NOISE * np.random.randn()


def gravity_tension(x: float, z: float, mass_kg=0.2) -> tuple:
    """
    Approximate static cable tensions from geometry + gravity.
    Resolves weight (mass × g) into left and right cable components.
    """
    g   = 9.81
    W   = mass_kg * g

    pos = np.array([x, z])
    dL  = anchor_L - pos
    dR  = anchor_R - pos

    # Unit vectors along each cable
    uL  = dL / np.linalg.norm(dL)
    uR  = dR / np.linalg.norm(dR)

    # Solve: TL*uL + TR*uR = [0, W] (2D equilibrium)
    A   = np.array([[uL[0], uR[0]],
                    [uL[1], uR[1]]])
    try:
        T   = np.linalg.solve(A, np.array([0.0, W]))
        TL  = max(T[0], 0.0)
        TR  = max(T[1], 0.0)
    except np.linalg.LinAlgError:
        TL = TR = W / 2.0

    return TL, TR


# =========================================================
# MOTION PROFILES
# Each returns (x, z) at time t
# =========================================================

def profile_sine(t):
    x = 0.30 * math.sin(0.4 * t)
    z = 0.25 + 0.08 * math.sin(0.9 * t)
    return x, z

def profile_zigzag(t):
    period = 4.0
    phase  = (t % period) / period       # 0 → 1
    x      = 0.35 * (2 * abs(2 * phase - 1) - 1)   # triangle wave
    z      = 0.22 + 0.06 * math.sin(0.5 * t)
    return x, z

def profile_circle(t):
    r  = 0.20
    cx, cz = 0.0, 0.28
    x  = cx + r * math.cos(0.5 * t)
    z  = cz + r * 0.4 * math.sin(0.5 * t)   # flattened ellipse
    return x, z

def profile_hover(t):
    x = 0.02 * math.sin(1.1 * t) + 0.01 * math.sin(3.3 * t)
    z = 0.25 + 0.01 * math.cos(0.7 * t)
    return x, z

PROFILES = {
    "sine"   : profile_sine,
    "zigzag" : profile_zigzag,
    "circle" : profile_circle,
    "hover"  : profile_hover,
}


# =========================================================
# MANUAL KEYBOARD CONTROL
# =========================================================

manual_pos  = [0.0, 0.25]
manual_lock = threading.Lock()
STEP_SIZE   = 0.01   # metres per keypress

def keyboard_listener():
    """Non-blocking WASD input. Runs in its own thread."""
    import sys, tty, termios

    fd   = sys.stdin.fileno()
    old  = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            with manual_lock:
                x, z = manual_pos
                if ch in ('a', 'A'):   x -= STEP_SIZE
                elif ch in ('d', 'D'): x += STEP_SIZE
                elif ch in ('w', 'W'): z += STEP_SIZE
                elif ch in ('s', 'S'): z -= STEP_SIZE
                elif ch == 'q':
                    break
                x = np.clip(x, -0.40, 0.40)
                z = np.clip(z, 0.10,  0.45)
                manual_pos[0], manual_pos[1] = x, z
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# =========================================================
# VIRTUAL SERIAL PORT  (platform-specific)
# =========================================================

def open_virtual_port():
    """
    Returns a writable file-like object and prints the port
    name to set in camera_support_sim.py.

    Linux/Mac : uses a PTY (pseudo-terminal) pair.
    Windows   : uses a real COM port via loopback cable,
                or a virtual COM port tool like com0com.
    """
    if sys.platform.startswith("win"):
        # On Windows you need a virtual COM port pair tool
        # (e.g. com0com) and then set both ends manually.
        import serial
        port_name = "COM8"   # ← one end of the virtual pair
        ser = serial.Serial(port_name, BAUD)
        print(f"[FAKE] Windows virtual port: {port_name}")
        print(f"[FAKE] Set SERIAL_PORT = 'COM7' in camera_support_sim.py")
        print(f"       (the other end of your virtual COM pair)")
        return ser, port_name

    else:
        # Linux / Mac: create a PTY pair
        import pty, os, tty
        master_fd, slave_fd = pty.openpty()
        slave_name = os.ttyname(slave_fd)
        # Set raw mode so bytes pass through cleanly
        tty.setraw(master_fd)
        master = os.fdopen(master_fd, "wb", buffering=0)
        print(f"[FAKE] Virtual serial port created.")
        print(f"[FAKE] Set SERIAL_PORT = '{slave_name}' in camera_support_sim.py")
        return master, slave_name


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 55)
    print("  Fake Arduino – Camera Support Sim Test")
    print(f"  Profile  : {PROFILE}")
    print(f"  Rate     : {UPDATE_RATE_HZ} Hz")
    print("=" * 55)

    port_obj, port_name = open_virtual_port()

    # Start keyboard thread for manual mode
    if PROFILE == "manual":
        if sys.platform.startswith("win"):
            print("[FAKE] Keyboard control not supported on Windows in this script.")
            print("[FAKE] Falling back to sine profile.")
            profile_fn = profile_sine
        else:
            print("[FAKE] Keyboard control: W=up  S=down  A=left  D=right  Q=quit")
            kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
            kb_thread.start()
            profile_fn = None   # handled separately below
    else:
        profile_fn = PROFILES.get(PROFILE, profile_sine)

    t0       = time.perf_counter()
    interval = 1.0 / UPDATE_RATE_HZ

    print(f"\n[FAKE] Streaming to {port_name} at {BAUD} baud ...\n")

    try:
        while True:
            t = time.perf_counter() - t0

            # ── Get position ──────────────────────────────
            if PROFILE == "manual" and profile_fn is None:
                with manual_lock:
                    x, z = manual_pos[0], manual_pos[1]
            else:
                x, z = profile_fn(t)

            # Clamp to valid workspace
            x = np.clip(x, -0.40, 0.40)
            z = np.clip(z,  0.10,  0.45)

            # ── Step counts ───────────────────────────────
            sL, sR, LL, LR = camera_to_steps(x, z)

            # ── Tension → current ─────────────────────────
            TL, TR = gravity_tension(x, z)
            iL     = tension_to_current(TL)
            iR     = tension_to_current(TR)

            # ── Build CSV packet ──────────────────────────
            line = f"{sL},{sR},{iL:.4f},{iR:.4f}\n"

            # ── Write to virtual port ─────────────────────
            try:
                if hasattr(port_obj, "write"):
                    port_obj.write(line.encode("utf-8"))
                else:
                    port_obj.write(line.encode("utf-8"))
            except (OSError, BrokenPipeError):
                print("[FAKE] Port closed — exiting.")
                break

            # ── Console log ───────────────────────────────
            print(
                f"t={t:6.2f}s | "
                f"x={x:+.3f}  z={z:.3f} | "
                f"sL={sL:6d}  sR={sR:6d} | "
                f"LL={LL:.3f}m  LR={LR:.3f}m | "
                f"iL={iL:.3f}A  iR={iR:.3f}A | "
                f"TL={TL:.2f}N  TR={TR:.2f}N"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[FAKE] Stopped by user.")

    finally:
        try:
            port_obj.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()