import numpy as np

ANCHOR_LEFT  = np.array([0.0, 1.0])
ANCHOR_RIGHT = np.array([1.0, 1.0])

SPOOL_RADIUS = 0.1  # metres

HOME = np.array([0.5, 1.0])
prev_pos = HOME.copy()

def inverse_kinematics(x, y):
    global prev_pos

    camera = np.array([x, y])

    L_left_new   = np.linalg.norm(ANCHOR_LEFT  - camera)
    L_right_new  = np.linalg.norm(ANCHOR_RIGHT - camera)

    L_left_prev  = np.linalg.norm(ANCHOR_LEFT  - prev_pos)
    L_right_prev = np.linalg.norm(ANCHOR_RIGHT - prev_pos)

    dL_left  = L_left_new  - L_left_prev
    dL_right = L_right_new - L_right_prev

    dAngle_left  = np.degrees(dL_left  / SPOOL_RADIUS)
    dAngle_right = np.degrees(dL_right / SPOOL_RADIUS)

    prev_pos = camera.copy()

    return dAngle_left, dAngle_right

def is_reachable(x, y):
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

def go_home():
    return inverse_kinematics(0.5, 1)
# --- TEST loop ---
if __name__ == "__main__":
    print("SpiderCam IK controller")
    print("Frame: 1x1m | Home: (0.5, 1.0)")
    print("Type 'home' to return home, 'quit' to exit")
    print(f"Starting at home position: {HOME}")
    print()

    while True:
        user_input = input("Enter target (x y): ").strip().lower()

        if user_input == "quit":
            print("Shutting down.")
            break

        elif user_input == "home":
            dL, dR = inverse_kinematics(HOME[0], HOME[1])
            print(f"  Going home → (0.5, 1.0)")
            print(f"  Left:  {dL:+.2f}° ({'reel in' if dL < 0 else 'reel out'})")
            print(f"  Right: {dR:+.2f}° ({'reel in' if dR < 0 else 'reel out'})")
            print()

        else:
            try:
                x, y = map(float, user_input.split())

                if not is_reachable(x, y):
                    print(f"  Out of bounds — x and y must be between 0 and 1")
                    print()
                    continue

                dL, dR = inverse_kinematics(x, y)
                print(f"  Moving to ({x}, {y})")
                print(f"  Left:  {dL:+.2f}° ({'reel in' if dL < 0 else 'reel out'})")
                print(f"  Right: {dR:+.2f}° ({'reel in' if dR < 0 else 'reel out'})")
                print(f"  Current position: {prev_pos}")
                print()

            except ValueError:
                print("  Invalid input — type two numbers like: 0.5 0.3")
                print()