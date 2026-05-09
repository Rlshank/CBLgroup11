from Inverse_Kinematics import inverse_kinematics as IK
from Inverse_Kinematics import go_home, is_reachable

x_des = 0.5
y_des = 0.5

dtheta_L , dtheta_R = IK(x_des, y_des)
print(dtheta_L, dtheta_R)