import numpy as np
spool_r = 0.1
anchor_1 = np.array([0.0, 1.0])
anchor_2 = np.array([1.0, 1.0])
prev_L = np.array([0.5, 0.5])

def inverse_kinematics(des_x, des_y):
    global prev_L
    des_pos = np.array([des_x, des_y])
    
    L1 = np.linalg.norm(anchor_1 - des_pos)
    L2 = np.linalg.norm(anchor_2 - des_pos)

    des_L = np.array([L1, L2])

    dL = des_L - prev_L
    dL1 = dL[0]
    dL2 = dL[1]
    
    dtheta1 = float(np.degrees(dL1 / spool_r))
    dtheta2 = float(np.degrees(dL2 / spool_r))
    
    prev_L = des_L.copy()
    return dtheta1 , dtheta2

print(inverse_kinematics(1.0, 0.0))