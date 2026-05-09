import numpy as np
spool_r = 0.1
anchor_1 = np.array([0.0, 1.0])
anchor_2 = np.array([1.0, 1.0])
prev_pos = np.array([0.5, 1.0])


def inverse_kinematics(des_x, des_y):
    global prev_pos
    des_pos = np.array([des_x, des_y])

    L1 = np.linalg.norm(anchor_1 - des_pos)
    L2 = np.linalg.norm(anchor_2 - des_pos)

    L1_prev = np.linalg.norm(anchor_1 - prev_pos)
    L2_prev = np.linalg.norm(anchor_2 - prev_pos)

    des_L = np.array([L1, L2])
    prev_L = np.array([L1_prev, L2_prev])
    dL = des_L - prev_L
    dL1 = dL[0]
    dL2 = dL[1]

    dtheta1 = float(np.degrees(dL1 / spool_r))
    dtheta2 = float(np.degrees(dL2 / spool_r))

    prev_pos = des_pos.copy()
    return dtheta1, dtheta2

