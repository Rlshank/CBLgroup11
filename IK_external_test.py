from Inv_Kin import inverse_kinematics as IK

x_des = 0.5
y_des = 0.5

dtheta_L, dtheta_R = IK(x_des, y_des)
print(dtheta_L, dtheta_R)
dtheta_L, dtheta_R = IK(x_des, y_des)
print(dtheta_L, dtheta_R)
dtheta_L, dtheta_R = IK(x_des, y_des)
print(dtheta_L, dtheta_R)
