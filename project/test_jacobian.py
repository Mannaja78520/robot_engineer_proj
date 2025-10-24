import time, sys, numpy as np, math

pi = np.pi
d2r = pi/180
r2d = 1/d2r

# ====== DH tables (as provided) ======
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]])  # [alpha, a, d, theta]
DH = np.array([
    [ 0.0   , 0.0    , 0.0   , 0.0      ],
    [ pi/2  , 0.0    , 0.0   , 0.0      ],
    [ 0.0   , 0.4251 , 0.0   , 0.0      ],
    [ 0.0   , 0.39215, 0.11  , 0.0      ],
    [ -pi/2 , 0.0    , 0.09475,0.0      ],
    [  pi/2 , 0.0    , 0.0   , 0.0      ],
], dtype=float)
DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

th_offset = np.array([0.0, np.pi/2, 0.0, -np.pi/2, 0.0, 0.0], dtype=float)

def _chain_from_DH(theta_list: list) -> tuple[list, np.ndarray]:
    
    # สร้างลิสต์ทรานสฟอร์มสะสม [T_00, T_01, ..., T_06, T_0E] ตาม DH_Base, DH (+th_offset), DH_EE
    # คืนค่า (T_list, T_0E)
    
    # มุมร่วม offset
    th = np.asarray(theta_list, dtype=float) + th_offset

    # ตัวคูณเริ่มจาก I
    T_list = [np.eye(4, dtype=float)]
    T = np.eye(4, dtype=float)

    # Base transform
    aB, AB = DH_Base[0], DH_Base  # just to name
    A_base = dh_transform_matrix(DH_Base[0,0], DH_Base[0,1], DH_Base[0,2], DH_Base[0,3])
    T = T @ A_base
    T_list.append(T)  # T_0(base)

    # 6 joints
    for i in range(6):
        alpha, a, d, _ = DH[i]
        A_i = dh_transform_matrix(alpha, a, d, th[i])
        T = T @ A_i
        T_list.append(T)  # T_0i

    # EE transform
    A_ee = dh_transform_matrix(DH_EE[0,0], DH_EE[0,1], DH_EE[0,2], DH_EE[0,3])
    T = T @ A_ee
    T_list.append(T)  # T_0E

    return T_list, T

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
    
    # ใช้ DH_Base, DH (+th_offset), และ DH_EE
    # คืน (position(3,), eulerXYZ(3,), T_0E(4x4))
    
    T_list, T_0E = _chain_from_DH(theta)
    position = T_0E[:3, 3]
    R = T_0E[:3, :3]
    euler_angles = rotation_matrix_to_euler(R)  # ตามฟังก์ชันเดิมของคุณ (XYZ)
    return position, euler_angles, T_0E

def ur5_geometric_jacobian(theta: list) -> np.ndarray:
    
    # Jacobian แบบ geometric ในกรอบฐาน (base frame)
    # ใช้ p_e ที่รวม EE แล้ว และแกน z_{i-1} จากทรานสฟอร์มสะสม
    # รูป: [v; ω] = J * qdot, joints ทั้งหมดเป็น revolute
    
    T_list, T_0E = _chain_from_DH(theta)
    # ดัชนีใน T_list:
    #   0: T_00 = I
    #   1: หลัง Base (T_0B)
    #   2..7: T_01..T_06
    #   8: T_0E
    p_e = T_0E[:3, 3].copy()

    J = np.zeros((6, 6), dtype=float)
    # ข้อที่ i ใช้ z_{i-1} และ p_{i-1}:
    # i=1 ใช้ T_list[1] (หลัง Base) เป็น "ข้อ 0" ก่อนเข้า joint1? ระวังอินเด็กซ์:
    # - joint1 ใช้ T_list[1] เป็นกรอบก่อนข้อ 1 (คือหลัง base)
    # - joint2 ใช้ T_list[2]
    # ...
    # - joint6 ใช้ T_list[6]
    for i in range(6):
        T_im1 = T_list[i+1]         # ก่อนข้อ i+1 (เพราะ +1 คือผ่าน Base หรือข้อก่อนหน้าแล้ว)
        z = T_im1[:3, 2]            # z_{i-1} ในกรอบฐาน
        p = T_im1[:3, 3]            # p_{i-1}
        J[:3, i] = np.cross(z, (p_e - p))  # linear
        J[3:, i] = z                          # angular
    return J

# --- 1) ตัวช่วยเลือกคอนเวนชันของ Jacobian: ใช้ z_i & p_i (แบบอ้างอิงของคุณ) ---
def ur5_geometric_jacobian_ref(theta: list) -> np.ndarray:
    T_list, T_0E = _chain_from_DH(theta)
    p_e = T_0E[:3, 3].copy()

    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        # pre/post ตามอ้างอิง: คอลัมน์ 1 ใช้ pre (หลัง Base), ที่เหลือใช้ post
        T_use = T_list[1] if i == 0 else T_list[i+2]
        z = T_use[:3, 2]
        p = T_use[:3, 3]

        if i == 0:
            # ข้อ 1: z × (p_e - p)
            J[:3, i] = np.cross(z, (p_e - p))
        else:
            # ข้อ 2–6: (p_e - p) × z
            J[:3, i] = np.cross((p_e - p), z)
        J[3:, i] = z
    return J

# --- 2) ตัวเลือก "จัดกรอบฐานให้ตรงกับ reference" โดยกลับสัญญาณแกน X ---
def align_with_reference_base(T_0E: np.ndarray, J: np.ndarray):
    # สะท้อนแกน X ของโลก: x' = -x (ตำแหน่ง), และแถวเชิงเส้นของ J (row 0) ก็กลับสัญญาณด้วย
    T_fix = np.diag([-1, 1, 1, 1])   # เฉพาะตำแหน่ง
    T_0E_fixed = T_0E.copy()
    T_0E_fixed[:3, 3] = (T_fix @ np.r_[T_0E[:3, 3], 1.0])[:3]

    J_fixed = J.copy()
    J_fixed[0, :] *= -1.0            # flip เฉพาะแถว vx
    # ส่วน ω ไม่ต้องแตะ เพราะกรอบหมุนเดียวกัน (คุณเช็คแล้วว่า Euler ตรง)
    return T_0E_fixed, J_fixed

def match_reference_table(J: np.ndarray) -> np.ndarray:
    J2 = J.copy()
    # พลิกเฉพาะส่วนเชิงเส้น vy(แถว 1) และ vz(แถว 2) ในคอลัมน์ 2..5
    J2[0:3, 1:5] *= -1.0
    return J2

# ========== SE(3) / SO(3) Utils ==========
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)

def log_SO3(R):
    tr = np.clip((np.trace(R)-1.0)*0.5, -1.0, 1.0)
    theta = math.acos(tr)
    if abs(theta) < 1e-9:
        return np.zeros(3)
    w = (1.0/(2*math.sin(theta))) * np.array([R[2,1]-R[1,2],
                                              R[0,2]-R[2,0],
                                              R[1,0]-R[0,1]])
    return w*theta

# (ใช้ FK ของคุณอยู่แล้ว) -> ur5_forward_kinematrix

# ---------- Spatial Jacobian (มาตรฐานสำหรับ IK) ----------
# ใช้ pre-joint: z_{i-1}, p_{i-1}
def ur5_jacobian_spatial(theta: list) -> np.ndarray:
    T_list, T_0E = _chain_from_DH(theta)
    p_e = T_0E[:3, 3].copy()
    J = np.zeros((6,6), dtype=float)
    for i in range(6):
        T_im1 = T_list[i+1]   # 1..6 = pre-joint i
        z = T_im1[:3, 2]
        p = T_im1[:3, 3]
        J[:3, i] = np.cross(z, (p_e - p))
        J[3:, i] = z
    return J

# ---------- IK (Damped Least Squares) ----------
def ik_solve_dls(theta0, p_des, R_des, max_iter=200, tol_pos=1e-4, tol_rot=1e-3,
                 lam=1e-3, step_max=4.0*np.pi/180, verbose=False, log_fn=None):
    """
    theta0: เริ่มต้น (rad)
    p_des, R_des: เป้าหมายในกรอบฐาน (m, rotation 3x3)
    lam: damping
    step_max: จำกัด |dq_i| ต่อสเต็ป (rad)
    """
    th = np.asarray(theta0, dtype=float).copy()
    for k in range(max_iter):
        # FK ปัจจุบัน
        p_cur, _, T_0E = ur5_forward_kinematrix(th)
        R_cur = T_0E[:3,:3]
        # ตำแหน่ง/มุมต่าง
        e_p = p_des - p_cur
        # นำ error หมุนจาก body มายัง spatial (base): e_w = R_cur * Log(R_cur^T R_des)
        Re = R_des @ R_cur.T
        e_w_body = log_SO3(Re)
        e_w = R_cur @ e_w_body
        # รวมเวกเตอร์เออเรอร์
        e_twist = np.hstack([e_p, e_w])

        if verbose and log_fn is not None:
            log_fn(f"[IK] iter {k:03d} |ep|={np.linalg.norm(e_p):.6f}, |ew|={np.linalg.norm(e_w):.6f}")

        # ตรวจจบ
        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_w) < tol_rot:
            return th, True, k

        # Jacobian (spatial)
        J = ur5_jacobian_spatial(th)
        # DLS: dq = J^T (J J^T + lam^2 I)^-1 e
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), e_twist)

        # จำกัดก้าว
        dq = np.clip(dq, -step_max, step_max)
        th = th + dq

    return th, False, max_iter

# ---------- Helpers ----------
def eulXYZ_to_R(eul_xyz):
    ex, ey, ez = eul_xyz
    cx, sx = math.cos(ex), math.sin(ex)
    cy, sy = math.cos(ey), math.sin(ey)
    cz, sz = math.cos(ez), math.sin(ez)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rx @ Ry @ Rz  # XYZ

def get_pose_of(sim, h, rel=-1):
    p = np.array(sim.getObjectPosition(h, rel), dtype=float)
    e = np.array(sim.getObjectOrientation(h, rel), dtype=float)  # Euler XYZ (rad)
    R = eulXYZ_to_R(e)
    return p, R, e

def set_joints(sim, hdl_j, th):
    for i in range(6):
        sim.setJointTargetPosition(hdl_j[i], float(th[i]))
        
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

    th = np.array([10, -10, 20, -20, 30, -30]) * np.pi/180

   # FK (ใช้ DH ที่แก้แล้ว)
    pos, eul, T = ur5_forward_kinematrix(th)

    # Jacobian แบบอ้างอิงของคุณ (hybrid + cross ตามที่แก้แล้ว)
    J_ref = ur5_geometric_jacobian_ref(th)

    # 🔧 อย่าเรียก align_with_reference_base อีกต่อไป
    # T_fix, J_fix = align_with_reference_base(T, J_ref)

    # ปรับ vy,vz เฉพาะคอลัมน์ 2..5 (ถ้าตารางอ้างอิงคุณต้องการ)
    J_fix = match_reference_table(J_ref)

    print("Position (m):", np.round(T[:3, 3], 5))
    print("Euler XYZ (deg):", np.round(eul * r2d, 2))
    print("Jacobian (6x6):\n", np.round(J_fix, 5))

    # # ---------- Motion params ----------
    # A = 90.0 * d2r
    # period = 30.0
    # cycles = 6
    # tol = 0.5 * d2r
    # omega_max = 8.0 * d2r
    # scales = np.array([1.0, 0.4, 0.6, 0.7, 0.2, 0.8], dtype=float)
    # USE_TARGET = True

    # # ---------- Logging (ลื่น + บรรทัดเยอะ) ----------
    # LOG_PERIOD = 0.5
    # USE_AUX_CONSOLE = True
    # COMPUTE_FK_IN_LOG = True
    # COMPUTE_JAC_IN_LOG = True

    # # เพิ่มบรรทัดสูงสุด + ขยายหน้าต่าง
    # AUX_MAX_LINES = 1000          # เดิม 12 -> 1000 บรรทัด
    # AUX_SIZE = [820, 560]         # กว้าง x สูง (px)
    # AUX_POS  = [10, 10]

    # # (ทางเลือก) เก็บลงไฟล์ไม่ให้หาย
    # LOG_TO_FILE = True
    # LOG_FILE_PATH = 'ur5_log.txt'
    # _f = None

    # q0 = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], dtype=float)
    # theta_cmd_prev = q0.copy()

    # aux = None
    # if USE_AUX_CONSOLE:
    #     aux = sim.auxiliaryConsoleOpen('UR5 log', AUX_MAX_LINES, 0, AUX_POS, AUX_SIZE, [1,1,1], [0,0,0])
    #     try:
    #         # เผื่อบางเวอร์ชันรองรับ show/hide
    #         sim.auxiliaryConsoleShow(aux, True)
    #     except Exception:
    #         pass

    # if LOG_TO_FILE:
    #     try:
    #         _f = open(LOG_FILE_PATH, 'w', encoding='utf-8')
    #     except Exception:
    #         _f = None  # ถ้าเปิดไม่ได้ ก็ข้ามไฟล์ไป

    # def log_once(s: str):
    #     if aux is not None:
    #         sim.auxiliaryConsolePrint(aux, s + '\n')
    #     else:
    #         sys.stdout.write(s + '\n')
    #     if _f is not None:
    #         _f.write(s + '\n'); _f.flush()   # เขียนเก็บทุกครั้ง (กันหาย)

    # log_once("UR5 Forward Kinematics and Orientation Display\n" + "="*60)

    # # ---------- Time loop ----------
    # t0 = sim.getSimulationTime()
    # T_total = (cycles * period) if cycles > 0 else 1e12
    # last_log = -1.0

    # while (sim.getSimulationTime() - t0) < T_total:
    #     t = sim.getSimulationTime() - t0
    #     dt = sim.getSimulationTimeStep()
    #     s = math.sin(2 * pi * (t / period))

    #     joint_angles = q0 + (A * s) * scales

    #     err = joint_angles - theta_cmd_prev
    #     step = np.clip(err, -omega_max * dt, omega_max * dt)
    #     theta_cmd = theta_cmd_prev + step
    #     theta_cmd_prev = theta_cmd

    #     for i in range(6):
    #         if USE_TARGET:
    #             sim.setJointTargetPosition(hdl_j[i], float(theta_cmd[i]))
    #         else:
    #             sim.setJointPosition(hdl_j[i], float(theta_cmd[i]))

    #     if (last_log < 0) or ((t - last_log) >= LOG_PERIOD):
    #         th_now = [sim.getJointPosition(hdl_j[i]) for i in range(6)]
    #         end_pos = sim.getObjectPosition(hdl_end, -1)
    #         end_eul = sim.getObjectOrientation(hdl_end, -1)

    #         lines = []
    #         lines.append(f"t={t:5.2f}s")
    #         lines.append("q(deg): " + ", ".join(f"{ang*r2d:6.2f}" for ang in th_now))
    #         lines.append("sim_pos:    {:.4f}, {:.4f}, {:.4f}".format(*end_pos))
    #         lines.append("sim_eul:    {:.2f}, {:.2f}, {:.2f}".format(*(np.array(end_eul) * r2d)))

    #         if COMPUTE_FK_IN_LOG:
    #             calc_pos, calc_eul, _ = ur5_forward_kinematrix(th_now)
    #             lines.append("FK_pos:     {:.4f}, {:.4f}, {:.4f}".format(*calc_pos))
    #             lines.append("FK_eul:     {:.2f}, {:.2f}, {:.2f}".format(*(calc_eul * r2d)))
    #         if COMPUTE_JAC_IN_LOG:
    #             J = ur5_geometric_jacobian(th_now)
    #             lines.append("Jacobian [v;omega] (6x6, base frame):")
    #             for r in range(6):
    #                 lines.append("  " + " ".join(f"{J[r,c]: 7.3f}" for c in range(6)))

    #         log_once("\n".join(lines))
    #         last_log = t

    #     sim.switchThread()

    # log_once("\nReturning to start pose...")
    # while True:
    #     dt = sim.getSimulationTimeStep()
    #     for i in range(6):
    #         if USE_TARGET:
    #             sim.setJointTargetPosition(hdl_j[i], float(q0[i]))
    #         else:
    #             sim.setJointPosition(hdl_j[i], float(q0[i]))
    #     errs = [abs(sim.getJointPosition(hdl_j[i]) - q0[i]) for i in range(6)]
    #     if max(errs) <= tol:
    #         break
    #     sim.switchThread()

    # log_once("Done. Settled at start pose.")

    # # ปิดไฟล์เมื่อจบ
    # if _f is not None:
    #     try: _f.close()
    #     except Exception: pass


    # log_once("Done. Settled at start pose.")

def sysCall_actuation():
    pass


def sysCall_sensing():
    pass


def sysCall_cleanup():
    pass