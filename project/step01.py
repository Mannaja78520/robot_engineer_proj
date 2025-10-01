#luaExec wrapper='pythonWrapper' -- using the old wrapper for backw. compat.
# To switch to the new wrapper, simply remove above line, and add sim=require('sim')
# as the first instruction in sysCall_init() or sysCall_thread()
import time
import numpy as np
import math

pi = np.pi
d2r = pi/180
r2d = 1/d2r

def dh_transform(alpha, a, d, theta):
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    ct = math.cos(theta)
    st = math.sin(theta)
    transform = np.array([[   ct,   -st,   0,     a],
                          [st*ca, ct*ca, -sa, -d*sa],
                          [st*sa, ct*sa,  ca,  d*ca],
                          [    0,     0,   0,     1]])
    return transform

# --- helper ???? ?: ???? R ???? Euler X-Y-Z ??????????? getObjectOrientation ---
def rot_to_eulXYZ(R):
    # match CoppeliaSim's XYZ with a gimbal-lock branch
    sy = -R[2,0]                        # = sin(beta)
    sy = 1.0 if sy > 1.0 else (-1.0 if sy < -1.0 else sy)
    beta = math.asin(sy)
    cb = math.cos(beta)

    if abs(cb) < 1e-8:
        # --- gimbal lock branch (beta ≈ ±90°) ---
        # choose the same branch SIM uses:
        alpha = 0.0
        # gamma from the in-plane terms
        gamma = -math.atan2(R[0,1], R[1,1])
    else:
        # regular branch
        alpha = math.atan2(R[2,1], R[2,2])
        gamma = math.atan2(R[1,0], R[0,0])

    return np.array([alpha, beta, gamma], dtype=float)

def calculate_forward_kinematics(theta : dict):
    theta1, theta2, theta3, theta4, theta5, theta6 = theta[0]*d2r, theta[1]*d2r, theta[2]*d2r, theta[3]*d2r, theta[4]*d2r, theta[5]*d2r
    
    # Calculate transformation matrices (???????)
    T_01 = dh_transform(0.0      , 0.0      ,   0.0892 ,   theta1 - np.pi/2)
    T_12 = dh_transform(np.pi/2  , 0.0      ,   0.0    ,   theta2 + np.pi/2)
    T_23 = dh_transform(0.0      , 0.4251   ,   0.0    ,   theta3          )
    T_34 = dh_transform(0.0      , 0.39215  ,   0.11   ,   theta4 + np.pi/2)
    T_45 = dh_transform(np.pi/2  , 0.0      ,   0.09475,   theta5          )
    T_56 = dh_transform(-np.pi/2 , 0.0      ,   0.26658,   theta6          )
    
    # End effector transformation ? ???????????:
    # ???????? Rz(+90). ????????? EndPoint ????? ??? Rz(-90) ??????? Rx(180)
    T_6E = np.array([
                        [ 0.0,  0.0, -1.0, 0.0],
                        [-1.0,  0.0,  0.0, 0.0],
                        [ 0.0,  1.0,  0.0, 0.0],
                        [ 0.0,  0.0,  0.0, 1.0],
                    ])

    return T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_6E

def sysCall_init():
    sim=require("sim")
def sysCall_thread():
    # define handles for axis (???????)
    hdl_j={}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    
    hdl_end = sim.getObject("/UR5/EndPoint")
    
    t = 0
    t1 = time.time()
    th = {}
    
    while t<10:
        p = 45*pi/180*np.sin(0.2*pi*t)
        for i in range(0,6):       
            sim.setJointTargetPosition(hdl_j[i], p)
         
        for i in range(0,6):
            th[i] = round(sim.getJointPosition(hdl_j[i])*r2d, 2)
            
        end_pos = sim.getObjectPosition(hdl_end,-1)
        end_ori = sim.getObjectOrientation(hdl_end,-1)  # Euler XYZ (rad)

        # FK
        T0E = calculate_forward_kinematics(th)
        fk_pos = T0E[0:3,3]
        fk_eul = rot_to_eulXYZ(T0E[0:3,0:3])

        print("-----------------------")
        print("Joint Position (deg): {}".format(th))
        print("FK  T0E:\n{}".format(np.array(T0E).round(5)))
        print("FK  pos (m): {}".format(np.array(fk_pos).round(4)))
        print("FK  eulXYZ (deg): {}".format((fk_eul*r2d).round(2)))
        print("SIM pos (m): {}".format(np.array(end_pos).round(4)))
        print("SIM eulXYZ (deg): {}".format((np.array(end_ori)*r2d).round(2)))

        # time
        t = time.time()-t1
        sim.switchThread() # resume in next simulation step
    pass
