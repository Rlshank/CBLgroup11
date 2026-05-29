import serial
import time
import math

PORT = 'COM4'
BAUD = 115200
SPOOL_RADIUS = 0.02
STEPS_PER_REV = 200
MAX_SPEED = 0.3
DEADZONE = 50
DT = 0.05

# Anchors 50cm high, 1m apart
ax1, ay1 = 0.0, 0.5
ax2, ay2 = 1.0, 0.5

# Home position: bottom center
cam_x, cam_y = 0.5, 0.0

l1 = math.sqrt((cam_x - ax1)**2 + (cam_y - ay1)**2)
l2 = math.sqrt((cam_x - ax2)**2 + (cam_y - ay2)**2)

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)
print("Connected")
print(f"Initial rope lengths — L1: {l1:.3f}m  L2: {l2:.3f}m")

def speed_to_delay(cable_dot):
    if abs(cable_dot) < 0.001:
        return 0
    steps_per_sec = abs(cable_dot) / (2 * math.pi * SPOOL_RADIUS) * STEPS_PER_REV
    delay_us = int(1e6 / (2 * steps_per_sec))
    return max(300, min(delay_us, 5000))

while True:
    t0 = time.time()
    line = ser.readline().decode().strip()
    if ',' not in line:
        continue
    try:
        rx, ry = map(int, line.split(','))
    except:
        continue

    # rx (A0) = side-to-side → moves gondola left/right (vx)
    # ry (A1) = forward/back → moves gondola up/down (vy)
    vx =  (rx - 512) / 512.0 if abs(rx - 512) > DEADZONE else 0.0
    vy = -(ry - 512) / 512.0 if abs(ry - 512) > DEADZONE else 0.0

    vx *= MAX_SPEED
    vy *= MAX_SPEED

    if abs(vx) < 0.01 and abs(vy) < 0.01:
        ser.write(b'0,0,0,0\n')
        continue

    # Inverse kinematics
    d1x, d1y = cam_x - ax1, cam_y - ay1
    d2x, d2y = cam_x - ax2, cam_y - ay2
    l1_dot = (d1x * vx + d1y * vy) / l1
    l2_dot = (d2x * vx + d2y * vy) / l2

    sp1 = speed_to_delay(l1_dot)
    sp2 = speed_to_delay(l2_dot)
    d1 = 1 if l1_dot > 0 else 0
    d2 = 1 if l2_dot > 0 else 0

    cmd = f'{d1},{sp1},{d2},{sp2}\n'
    ser.write(cmd.encode())

    # Update position
    l1 += l1_dot * DT
    l2 += l2_dot * DT
    cam_x += vx * DT
    cam_y += vy * DT

    cam_x = max(0.05, min(0.95, cam_x))
    cam_y = max(-0.45, min(0.45, cam_y))

    print(f"pos: ({cam_x:.2f}, {cam_y:.2f})  L1: {l1:.3f}m  L2: {l2:.3f}m  sp1: {sp1}us  sp2: {sp2}us")

    elapsed = time.time() - t0
    if elapsed < DT:
        time.sleep(DT - elapsed)