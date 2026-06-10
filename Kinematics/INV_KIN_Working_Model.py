import serial
import time
import math

PORT = 'COM4'

BAUD = 115200
SPOOL_RADIUS = 0.0125
STEPS_PER_REV = 200
MAX_SPEED = 0.25
DEADZONE = 50
DT = 0.05


# Anchors 50 cm high, 93 cm apart
ax1, ay1 = 0.0, 0.5
ax2, ay2 = 0.93,0.5


# Home position: bottom center
cam_x, cam_y = 0.5, 0.0

# Movement limits
MIN_X = 0.
MAX_X = 0.93
MIN_Y = -0.5
MAX_Y = 0.5


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

    # Because Arduino toggles HIGH/LOW, one full step happens every two toggles
    delay_us = int(1e6 / (2 * steps_per_sec))
    return max(500, min(delay_us, 8000))

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
    ser.write(b'0,0,0,0\n')


def send_command(ser, d1, sp1, d2, sp2):
    cmd = f'{d1},{sp1},{d2},{sp2}\n'
    ser.write(cmd.encode())

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    l1, l2 = rope_lengths(cam_x, cam_y)
    print("Connected")
    print(f"Initial rope lengths — L1: {l1:.3f} m  L2: {l2:.3f} m")

    while True:
        t0 = time.time()
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("DATA,"):
            continue
        try:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            rx = int(parts[1])
            ry = int(parts[2])
            current2_mA = float(parts[3])

        except ValueError:
            continue
        vx, vy = joystick_to_velocity(rx, ry)
        # Stop motors when joystick is centered
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            send_stop(ser)
            print(
                f"Joystick: ({rx}, {ry}) | STOP | "
                f"I2: {current2_mA:.2f} mA"
            )
            continue

        # Current rope lengths
        old_l1, old_l2 = rope_lengths(cam_x, cam_y)
        # Desired next position
        new_x = cam_x + vx * DT
        new_y = cam_y + vy * DT
        # Limit movement area
        new_x = max(MIN_X, min(MAX_X, new_x))
        new_y = max(MIN_Y, min(MAX_Y, new_y))


        # Required rope lengths at new position
        new_l1, new_l2 = rope_lengths(new_x, new_y)

        # Required rope speed
        l1_dot = (new_l1 - old_l1) / DT
        l2_dot = (new_l2 - old_l2) / DT

        sp1 = speed_to_delay(l1_dot)
        sp2 = speed_to_delay(l2_dot)

        d1 = 1 if l1_dot > 0 else 0
        d2 = 1 if l2_dot > 0 else 0

        send_command(ser, d1, sp1, d2, sp2)

        # Update estimated position
        cam_x = new_x
        cam_y = new_y

        # Final rope lengths after position update
        l1, l2 = rope_lengths(cam_x, cam_y)
        
        print(

            f"Joystick: ({rx}, {ry}) | "

            f"pos: ({cam_x:.2f}, {cam_y:.2f}) | "

            f"L1: {l1:.3f} m | L2: {l2:.3f} m | "

            f"sp1: {sp1} us | sp2: {sp2} us | "

            f"I2: {current2_mA:.2f} mA"

        )
        elapsed = time.time() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)


except serial.SerialException:
    print(f"Could not connect to Arduino on {PORT}.")
    print("Check the COM port and close the Arduino Serial Monitor.")

except KeyboardInterrupt:
    print("\nStopping motors...")
    try:
        send_stop(ser)
        ser.close()
    except:
        pass
    print("Program stopped.")