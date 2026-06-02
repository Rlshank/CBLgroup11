import serial
import time
import math

# =========================
# Settings
# =========================
PORT = "COM3"
BAUD = 115200

SPOOL_RADIUS = 0.02
STEPS_PER_REV = 200
MAX_SPEED = 0.3          # I lowered this for smoother movement
DEADZONE = 70
DT = 0.05

MOTOR1_DIR_INVERT = 1
MOTOR2_DIR_INVERT = 1

# Joystick center values
CENTER_X = 512
CENTER_Y = 512

# Anchors: 1 meter apart, 50 cm high
ax1, ay1 = 0.0, 0.5
ax2, ay2 = 1.0, 0.5

# Start position
cam_x = 0.5
cam_y = 0.0

# Movement limits
MIN_X = 0.05
MAX_X = 0.95
MIN_Y = -0.45
MAX_Y = 0.45


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

    delay_us = int(1_000_000 / steps_per_sec)

    # Safety limits
    delay_us = max(500, min(delay_us, 8000))

    return delay_us


def joystick_to_velocity(rx, ry):
    if abs(rx - CENTER_X) > DEADZONE:
        vx = (rx - CENTER_X) / 512.0
    else:
        vx = 0.0

    if abs(ry - CENTER_Y) > DEADZONE:
        vy = -(ry - CENTER_Y) / 512.0
    else:
        vy = 0.0

    vx *= MAX_SPEED
    vy *= MAX_SPEED

    return vx, vy


def send_motor_command(ser, dir1, sp1, dir2, sp2):
    cmd = f"{dir1},{sp1},{dir2},{sp2}\n"
    ser.write(cmd.encode())


try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)

    print("Connected to Arduino")
    print(f"Start position: x={cam_x:.3f}, y={cam_y:.3f}")

    while True:
        t0 = time.time()

        line = ser.readline().decode(errors="ignore").strip()

        if "," not in line:
            continue

        try:
            rx, ry = map(int, line.split(","))
        except ValueError:
            continue

        vx, vy = joystick_to_velocity(rx, ry)

        # Stop motors when joystick is centered
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            send_motor_command(ser, 0, 0, 0, 0)
            continue

        # Current rope lengths
        old_l1, old_l2 = rope_lengths(cam_x, cam_y)

        # =========================
        # Desired next position
        # =========================

        new_x = cam_x + vx * DT
        new_y = cam_y + vy * DT

        # Limit movement area
        new_x = max(MIN_X, min(MAX_X, new_x))
        new_y = max(MIN_Y, min(MAX_Y, new_y))

        # Calculate required rope lengths for the new position
        new_l1, new_l2 = rope_lengths(new_x, new_y)

        # Cable speed needed to reach that new position
        l1_dot = (new_l1 - old_l1) / DT
        l2_dot = (new_l2 - old_l2) / DT

        sp1 = speed_to_delay(l1_dot)
        sp2 = speed_to_delay(l2_dot)

        dir1 = 1 if l1_dot * MOTOR1_DIR_INVERT > 0 else 0
        dir2 = 1 if l2_dot * MOTOR2_DIR_INVERT > 0 else 0

        send_motor_command(ser, dir1, sp1, dir2, sp2)

        # Update simulated position
        cam_x = new_x
        cam_y = new_y

        print(
            f"Joystick: ({rx}, {ry}) | "
            f"pos: ({cam_x:.2f}, {cam_y:.2f}) | "
            f"L1: {new_l1:.3f}, L2: {new_l2:.3f} | "
            f"sp1: {sp1} us, sp2: {sp2} us | "
            f"dir1: {dir1}, dir2: {dir2}"
        )

        elapsed = time.time() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

except serial.SerialException:
    print(f"Could not connect to Arduino on {PORT}.")
    print("Check if the Arduino IDE Serial Monitor is closed and if COM3 is correct.")

except KeyboardInterrupt:
    print("\nStopping motors...")
    try:
        send_motor_command(ser, 0, 0, 0, 0)
        ser.close()
    except:
        pass
    print("Program stopped.")