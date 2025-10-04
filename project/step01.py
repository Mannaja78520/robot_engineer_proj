import time, sys, numpy as np, math

pi = np.pi
d2r = pi/180
r2d = 1/d2r

def dh_transform_matrix(alpha : float, a : float, d : float, theta : float) -> np.ndarray:
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    A = np.array([[   ct,   -st,    0,     a],
                  [st*ca, ct*ca, -sa , -d*sa],
                  [st*sa, ct*sa,  ca ,  d*ca],
                  [    0,     0,   0 ,     1]])
    return A

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])
        gamma = math.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])

    return np.array([alpha, beta, gamma])

def ur5_forward_kinematrix(theta : list) -> tuple:
    theta1, theta2, theta3, theta4, theta5, theta6 = theta
    T_01 = dh_transform_matrix(0.0      , 0.0      , 0.0892 , theta1 - np.pi/2)
    T_12 = dh_transform_matrix(np.pi/2  , 0.0      , 0.0    , theta2 + np.pi/2)
    T_23 = dh_transform_matrix(0.0      , 0.4251   , 0.0    , theta3          )
    T_34 = dh_transform_matrix(0.0      , 0.39215  , 0.11   , theta4 + np.pi/2)
    T_56 = dh_transform_matrix(-np.pi/2 , 0.0      , 0.26658, theta6          )
    T_45 = dh_transform_matrix(np.pi/2  , 0.0      , 0.09475, theta5          )
    T_6E = np.eye(4)
    T_0E = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_6E
    position = T_0E[:3, 3]
    R = T_0E[:3, :3]
    euler_angles = rotation_matrix_to_euler(R)
    return position, euler_angles, T_0E

def ur5_geometric_jacobian(theta: list) -> np.ndarray:
    
    # Geometric Jacobian J (6x6) in base frame:
    #   [ v ] = J(q) * qdot
    #   [ ω ]
    # สำหรับข้อแบบหมุนทั้งหมด:
    #   Jv_i = z_{i-1} × (p_e - p_{i-1})
    #   Jw_i = z_{i-1}
    
    # สร้าง A_i ตาม DH ให้ตรงกับ FK เดิม
    A1 = dh_transform_matrix(0.0      , 0.0     , 0.0892 , theta[0] - np.pi/2)
    A2 = dh_transform_matrix(np.pi/2  , 0.0     , 0.0    , theta[1] + np.pi/2)
    A3 = dh_transform_matrix(0.0      , 0.4251  , 0.0    , theta[2])
    A4 = dh_transform_matrix(0.0      , 0.39215 , 0.11   , theta[3] + np.pi/2)
    A5 = dh_transform_matrix(np.pi/2  , 0.0     , 0.09475, theta[4])
    A6 = dh_transform_matrix(-np.pi/2 , 0.0     , 0.26658, theta[5])

    As = [A1, A2, A3, A4, A5, A6]

    # คูณสะสมให้ได้ T_0i ทุกข้อ
    T = np.eye(4, dtype=float)
    T_list = [np.eye(4, dtype=float)]  # T_00 = I
    for Ai in As:
        T = T @ Ai
        T_list.append(T)               # T_01 ... T_06

    p_e = T_list[6][:3, 3]
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        T_im1 = T_list[i]              # T_0(i-1)
        z = T_im1[:3, 2]               # z_{i-1} in base
        p = T_im1[:3, 3]               # p_{i-1} in base
        J[:3, i] = np.cross(z, (p_e - p))  # linear part
        J[3:, i] = z                      # angular part
    return J

def sysCall_init():
    sim=require("sim")
    
def sysCall_thread():
    sim = require("sim")

    # ---------- Handles ----------
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    # ---------- Motion params ----------
    A = 90.0 * d2r
    period = 30.0
    cycles = 6
    tol = 0.5 * d2r
    omega_max = 8.0 * d2r
    scales = np.array([1.0, 0.4, 0.6, 0.7, 0.2, 0.8], dtype=float)
    USE_TARGET = True

    # ---------- Logging (ลื่น + บรรทัดเยอะ) ----------
    LOG_PERIOD = 0.5
    USE_AUX_CONSOLE = True
    COMPUTE_FK_IN_LOG = True
    COMPUTE_JAC_IN_LOG = True

    # เพิ่มบรรทัดสูงสุด + ขยายหน้าต่าง
    AUX_MAX_LINES = 1000          # เดิม 12 -> 1000 บรรทัด
    AUX_SIZE = [820, 560]         # กว้าง x สูง (px)
    AUX_POS  = [10, 10]

    # (ทางเลือก) เก็บลงไฟล์ไม่ให้หาย
    LOG_TO_FILE = True
    LOG_FILE_PATH = 'ur5_log.txt'
    _f = None

    q0 = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], dtype=float)
    theta_cmd_prev = q0.copy()

    aux = None
    if USE_AUX_CONSOLE:
        aux = sim.auxiliaryConsoleOpen('UR5 log', AUX_MAX_LINES, 0, AUX_POS, AUX_SIZE, [1,1,1], [0,0,0])
        try:
            # เผื่อบางเวอร์ชันรองรับ show/hide
            sim.auxiliaryConsoleShow(aux, True)
        except Exception:
            pass

    if LOG_TO_FILE:
        try:
            _f = open(LOG_FILE_PATH, 'w', encoding='utf-8')
        except Exception:
            _f = None  # ถ้าเปิดไม่ได้ ก็ข้ามไฟล์ไป

    def log_once(s: str):
        if aux is not None:
            sim.auxiliaryConsolePrint(aux, s + '\n')
        else:
            sys.stdout.write(s + '\n')
        if _f is not None:
            _f.write(s + '\n'); _f.flush()   # เขียนเก็บทุกครั้ง (กันหาย)

    log_once("UR5 Forward Kinematics and Orientation Display\n" + "="*60)

    # ---------- Time loop ----------
    t0 = sim.getSimulationTime()
    T_total = (cycles * period) if cycles > 0 else 1e12
    last_log = -1.0

    while (sim.getSimulationTime() - t0) < T_total:
        t = sim.getSimulationTime() - t0
        dt = sim.getSimulationTimeStep()
        s = math.sin(2 * pi * (t / period))

        joint_angles = q0 + (A * s) * scales

        err = joint_angles - theta_cmd_prev
        step = np.clip(err, -omega_max * dt, omega_max * dt)
        theta_cmd = theta_cmd_prev + step
        theta_cmd_prev = theta_cmd

        for i in range(6):
            if USE_TARGET:
                sim.setJointTargetPosition(hdl_j[i], float(theta_cmd[i]))
            else:
                sim.setJointPosition(hdl_j[i], float(theta_cmd[i]))

        if (last_log < 0) or ((t - last_log) >= LOG_PERIOD):
            th_now = [sim.getJointPosition(hdl_j[i]) for i in range(6)]
            end_pos = sim.getObjectPosition(hdl_end, -1)
            end_eul = sim.getObjectOrientation(hdl_end, -1)

            lines = []
            lines.append(f"t={t:5.2f}s")
            lines.append("q(deg): " + ", ".join(f"{ang*r2d:6.2f}" for ang in th_now))
            lines.append("sim_pos:    {:.4f}, {:.4f}, {:.4f}".format(*end_pos))
            lines.append("sim_eul:    {:.2f}, {:.2f}, {:.2f}".format(*(np.array(end_eul) * r2d)))

            if COMPUTE_FK_IN_LOG:
                calc_pos, calc_eul, _ = ur5_forward_kinematrix(th_now)
                lines.append("FK_pos:     {:.4f}, {:.4f}, {:.4f}".format(*calc_pos))
                lines.append("FK_eul:     {:.2f}, {:.2f}, {:.2f}".format(*(calc_eul * r2d)))
            if COMPUTE_JAC_IN_LOG:
                J = ur5_geometric_jacobian(th_now)
                lines.append("Jacobian [v;omega] (6x6, base frame):")
                for r in range(6):
                    lines.append("  " + " ".join(f"{J[r,c]: 7.3f}" for c in range(6)))

            log_once("\n".join(lines))
            last_log = t

        sim.switchThread()

    log_once("\nReturning to start pose...")
    while True:
        dt = sim.getSimulationTimeStep()
        for i in range(6):
            if USE_TARGET:
                sim.setJointTargetPosition(hdl_j[i], float(q0[i]))
            else:
                sim.setJointPosition(hdl_j[i], float(q0[i]))
        errs = [abs(sim.getJointPosition(hdl_j[i]) - q0[i]) for i in range(6)]
        if max(errs) <= tol:
            break
        sim.switchThread()

    log_once("Done. Settled at start pose.")

    # ปิดไฟล์เมื่อจบ
    if _f is not None:
        try: _f.close()
        except Exception: pass


    log_once("Done. Settled at start pose.")

def sysCall_actuation():
    pass


def sysCall_sensing():
    pass


def sysCall_cleanup():
    pass