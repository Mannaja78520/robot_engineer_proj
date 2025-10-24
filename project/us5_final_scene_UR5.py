import time, sys, numpy as np, math

# ===================== ค่าพื้นฐาน =====================
pi = np.pi
d2r = pi/180     # องศา -> เรเดียน
r2d = 1/d2r      # เรเดียน -> องศา

# ===================== DH (ยืนยันแล้ว) =====================
# เปลี่ยนค่าในตาราง DH = เปลี่ยนคิเนเมติกส์ทั้งหมดของแขน
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float)  # [alpha, a, d, theta]
DH = np.array([
    [ 0.0   , 0.0    , 0.0    , 0.0],
    [ pi/2  , 0.0    , 0.0    , 0.0],
    [ 0.0   , 0.4251 , 0.0    , 0.0],
    [ 0.0   , 0.39215, 0.11   , 0.0],
    [ -pi/2 , 0.0    , 0.09475, 0.0],
    [  pi/2 , 0.0    , 0.0    , 0.0],
], dtype=float)
DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

# ออฟเซ็ตมุมศูนย์ของข้อต่อ (เปลี่ยน = เลื่อนตำแหน่งศูนย์แต่ละแกน)
th_offset = np.array([0.0, np.pi/2, 0.0, -np.pi/2, 0.0, 0.0], dtype=float)
# ===== User-tunable gripper yaw =====
USER_YAW_DEG = 0.0   # ใส่เองได้เลย (+ หมุนตามเข็ม ถ้ามองจากปลายหัวลงมา)

# === Add these near your other constants ===
UR5_BASE_WORLD = [0.00, 0.00, 0.0] # ตำแหน่งฐาน UR5 (world)
BOX_START_WORLD = [0.275, -0.5, 0.125] # จุดเริ่มของกล่อง (world)
PLACE_IN_CONV = [0.32459, 0.51, 0.25] # จุดหย่อนกล่องบนสายพาน (world)


GRASP_CLEAR_Z = 0.15 # ระยะเผื่อยก/ลงจากด้านบน
PLACE_CLEAR_Z = 0.12
DESCENT_Z = 0.12 # ระยะลงไปคีบ/หย่อน

# ===================== Joint kinematics limits (velocity/accel) =====================
# ตั้งค่าตัวอย่าง (ปรับตามบอทจริงของคุณ)
QDOT_MAX = np.array([2.5, 2.0, 2.5, 3.5, 3.5, 3.5], float)     # rad/s   ต่อข้อ
QDDOT_MAX = np.array([8.0, 7.0, 8.0, 10.0, 10.0, 10.0], float)  # rad/s^2 ต่อข้อ

# ===================== ขีดจำกัดข้อต่อ =====================
Q_MIN = np.array([-2*np.pi]*6, float)  # แคบลง = ปลอดภัยขึ้นแต่ IK หาได้น้อยลง
Q_MAX = np.array([ 2*np.pi]*6, float)
Q_NOM = np.zeros(6, float)             # จุดอ้างอิงสำหรับ nullspace

# ===================== คอนเวนชันทิศเข้าใกล้ของ tool =====================
# +1 = ให้ +Z ของ tool ชี้เข้าหาวัตถุ, -1 = ให้ -Z ชี้เข้า (สลับถ้าทิศ top-down กลับหัว)
APPROACH_TOOL_Z = +1

# ===================== Helpers =====================
def R_from_top_mode(mode, yaw):
    # เลือกเมทริกซ์หมุนตามโหมด TOP-DOWN/TOP-UP แล้วบิดด้วย yaw (รอบ Z_tool)
    if mode == 'TOP-DOWN':
        return make_R_topdown(yaw=yaw)
    elif mode == 'TOP-UP':
        return make_R_topup(yaw=yaw)
    else:
        raise ValueError("R_from_top_mode: mode ต้องเป็น 'TOP-DOWN' หรือ 'TOP-UP'")

def solve_with_yaw_sweep(q0, p_target, base_yaw, mode, solver_fn,
                         yaw_deg_list=(-180, -135, -90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 135)):
    # สแกน yaw หลายค่าเพื่อหลบลิมิต/ซิงกูลาร์และลดคอสต์ ep+0.35*eo
    best = None
    for yd in yaw_deg_list:
        yaw = base_yaw + yd*d2r
        R_try = R_from_top_mode(mode, yaw)
        q, it, ep, eo, ok = solver_fn(q0, p_target, R_try)
        cost = ep + 0.35*eo
        cand = dict(q=q, it=it, ep=ep, eo=eo, ok=ok, yaw=yaw, R=R_try, cost=cost)
        if best is None or (ok and not best['ok']) or (ok == best['ok'] and cost < best['cost']):
            best = cand
        if ok and cost < 2e-3:
            break
    return best

def clamp_q(q):
    # จำกัดมุมให้อยู่ในลิมิตข้อ
    return np.minimum(np.maximum(q, Q_MIN), Q_MAX)

def ang_wrap(x):
    # wrap มุมให้อยู่ในช่วง (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

def ang_wrap_vec(q):
    # wrap ทุกองค์ประกอบของเวกเตอร์มุม
    q2 = np.array(q, float)
    for i in range(q2.shape[0]):
        q2[i] = ang_wrap(q2[i])
    return q2

def dh_transform_matrix(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    # เมทริกซ์ทรานส์ฟอร์มตาม DH
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    A = np.array([[   ct,   -st,    0,     a],
                  [st*ca, ct*ca, -sa , -d*sa],
                  [st*sa, ct*sa,  ca ,  d*ca],
                  [    0,     0,   0 ,     1]], dtype=float)
    return A

def _chain_from_DH(theta_list: list) -> tuple[list, np.ndarray]:
    # ต่อเชนทรานส์ฟอร์มตั้งแต่ฐานถึงปลายมือ และเก็บลิสต์ไว้ใช้คำนวณ J
    th = np.asarray(theta_list, dtype=float) + th_offset
    T_list = [np.eye(4, dtype=float)]
    T = np.eye(4, dtype=float)
    A_base = dh_transform_matrix(DH_Base[0,0], DH_Base[0,1], DH_Base[0,2], DH_Base[0,3])
    T = T @ A_base
    T_list.append(T)
    for i in range(6):
        alpha, a, d, _ = DH[i]
        A_i = dh_transform_matrix(alpha, a, d, th[i])
        T = T @ A_i
        T_list.append(T)
    A_ee = dh_transform_matrix(DH_EE[0,0], DH_EE[0,1], DH_EE[0,2], DH_EE[0,3])
    T = T @ A_ee
    T_list.append(T)
    return T_list, T

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    # แปลง R -> Euler XYZ (ระวังลำดับแกน)
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])
        gamma = math.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma], dtype=float)

def ur5_forward_kinematrix(theta: list) -> tuple:
    # FK คืน (ตำแหน่ง, ยูเลอร์, ทรานส์ฟอร์มเต็ม)
    T_list, T_0E = _chain_from_DH(theta)
    position = T_0E[:3, 3]
    R = T_0E[:3, :3]
    euler_angles = rotation_matrix_to_euler(R)
    return position, euler_angles, T_0E

# ===================== Jacobian =====================
def ur5_geometric_jacobian_ref(theta: list) -> np.ndarray:
    # J เรขาคณิตอ้างอิงจากเฟรมแต่ละข้อ (ใช้เปรียบเทียบ/debug)
    T_list, T_0E = _chain_from_DH(theta)
    p_e = T_0E[:3, 3].copy()
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        T_use = T_list[1] if i == 0 else T_list[i+2]
        z = T_use[:3, 2]
        p = T_use[:3, 3]
        if i == 0:
            J[:3, i] = np.cross(z, (p_e - p))
        else:
            J[:3, i] = np.cross((p_e - p), z)
        J[3:, i] = z
    return J

def match_reference_table(J: np.ndarray) -> np.ndarray:
    # ปรับสัญญาณบางแถวให้ตรงกับตารางอ้างอิงภายนอก
    J2 = J.copy()
    J2[1:3, 1:5] *= -1.0
    return J2

# ===================== SO(3) =====================
def log_SO3(R):
    # แปลง R -> เวกเตอร์แกน-มุม (log map)
    tr = np.clip((np.trace(R)-1.0)*0.5, -1.0, 1.0)
    theta = math.acos(tr)
    if abs(theta) < 1e-9:
        return np.zeros(3, float)
    w = (1.0/(2*math.sin(theta))) * np.array([R[2,1]-R[1,2],
                                              R[0,2]-R[2,0],
                                              R[1,0]-R[0,1]])
    return w*theta

# ===================== Euler / Rotation helpers =====================
def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0,0],[0,ca,-sa,0],[0,sa,ca,0],[0,0,0,1]], float)
def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa,0],[0,1,0,0],[-sa,0,ca,0],[0,0,0,1]], float)
def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0,0],[sa,ca,0,0],[0,0,1,0],[0,0,0,1]], float)
def eulXYZ_to_R(eul_xyz):
    # Euler XYZ -> R (อย่าเปลี่ยนลำดับ)
    rx, ry, rz = eul_xyz
    return (Rx(rx) @ Ry(ry) @ Rz(rz))[:3,:3]

# ===================== Auto-approach =====================
Z_TOPDOWN = 0.60         # z ต่ำกว่า = โน้มไป TOP-DOWN
ELEV_DEG  = 35.0         # elevation แบนกว่า = TOP-DOWN
YAW_ALIGN_WITH_HEADING = True  # True = base_yaw = heading(x,y)

def rot_from_approach(approach_world, yaw=0.0):
    # ให้ Z_tool ชี้ตาม approach_world แล้วบิดรอบ Z_tool ด้วย yaw
    a = np.array(approach_world, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    z_tool = APPROACH_TOOL_Z * a
    up_hint = np.array([0,0,1.0], float)
    if abs(np.dot(z_tool, up_hint)) > 0.99:
        up_hint = np.array([1,0,0], float)
    x_tool = np.cross(up_hint, z_tool); x_tool /= (np.linalg.norm(x_tool)+1e-12)
    y_tool = np.cross(z_tool, x_tool)
    R = np.column_stack([x_tool, y_tool, z_tool])
    c, s = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    return R @ R_yaw

def make_R_topdown(yaw=0.0):
    # approach = -Z_world (จากด้านบน)
    return rot_from_approach([0,0,-1], yaw=yaw)

def make_R_topup(yaw=0.0):
    # approach = +Z_world (จากด้านล่าง)
    return rot_from_approach([0,0, 1], yaw=yaw)

def make_R_side_toward_xy(p_target, yaw=0.0):
    # approach ระนาบ XY ชี้จากฐานไปเป้า
    x, y = float(p_target[0]), float(p_target[1])
    ax = np.array([x, y, 0.0], float)
    n = np.linalg.norm(ax)
    if n < 1e-6: ax = np.array([1.0,0.0,0.0], float)
    else: ax /= n
    return rot_from_approach(ax, yaw=yaw)

# ย้ำค่า (ไม่มีผลเพิ่ม)
YAW_ALIGN_WITH_HEADING = True
ELEV_DEG  = 35.0
Z_TOPDOWN = 0.60

def pick_auto_approach(p_target, force_mode=None):
    # เลือกโหมด TOP-DOWN/SIDE อัตโนมัติจาก z และ elevation (TOP-UP ต้องบังคับ)
    x, y, z = float(p_target[0]), float(p_target[1]), float(p_target[2])
    rxy = math.hypot(x, y)
    elev = math.degrees(math.atan2(z, max(rxy,1e-12)))
    heading = math.atan2(y, x)
    yaw = heading if YAW_ALIGN_WITH_HEADING else 0.0

    mode = force_mode
    if mode is None:
        if (z < Z_TOPDOWN) or (abs(elev) < ELEV_DEG):
            mode = 'TOP-DOWN'
        else:
            mode = 'SIDE'

    if mode == 'TOP-DOWN':
        R = make_R_topdown(yaw=yaw)
    elif mode == 'TOP-UP':
        R = make_R_topup(yaw=yaw)
    elif mode == 'SIDE':
        R = make_R_side_toward_xy(p_target, yaw=0.0)
    else:
        R = make_R_side_toward_xy(p_target, yaw=0.0); mode = 'SIDE'
    return R, mode

def try_ik_with_fallback(q0, p_target, eul_deg_if_needed, solver_fn, force_mode=None):
    # ลองโหมดอัตโนมัติ; ถ้าเป็น TOP-* สแกน yaw; ถ้าไม่ได้ค่อย fallback ใช้ Euler ที่ป้อนมา
    eul_xyz = np.array(eul_deg_if_needed, float)*d2r
    R_euler = eulXYZ_to_R(eul_xyz)

    R_auto, mode_auto = pick_auto_approach(p_target, force_mode=force_mode)
    x, y = float(p_target[0]), float(p_target[1])
    heading = math.atan2(y, x)
    base_yaw = heading if YAW_ALIGN_WITH_HEADING else 0.0

    if mode_auto in ('TOP-DOWN', 'TOP-UP'):
        best = solve_with_yaw_sweep(q0, p_target, base_yaw, mode_auto, solver_fn)
        if best and (best['ok'] or best['cost'] < 1e-2):
            chosen = {"mode": mode_auto, "R": best['R'], "eul": rotation_matrix_to_euler(best['R']), "yaw": best['yaw']}
            return best['q'], best['it'], best['ep'], best['eo'], best['ok'], chosen
        q2, it2, ep2, eo2, ok2 = solver_fn(q0, p_target, R_euler)
        chosen = {"mode": "EULER-FALLBACK", "R": R_euler, "eul": rotation_matrix_to_euler(R_euler)}
        return q2, it2, ep2, eo2, ok2, chosen

    q1, it1, ep1, eo1, ok1 = solver_fn(q0, p_target, R_auto)
    if ok1:
        chosen = {"mode":mode_auto, "R":R_auto, "eul":rotation_matrix_to_euler(R_auto)}
        return q1, it1, ep1, eo1, ok1, chosen

    q2, it2, ep2, eo2, ok2 = solver_fn(q0, p_target, R_euler)
    chosen = {"mode":"EULER-FALLBACK", "R":R_euler, "eul":rotation_matrix_to_euler(R_euler)}
    return q2, it2, ep2, eo2, ok2, chosen

# ระยะก่อนเข้าหาเป้าตามแกน Z_tool (เพิ่ม = ถอยก่อนมากขึ้น)
PREAPPROACH_DIST = 0.10

def plan_pre_and_target(q_start, p_target, R_target, solver_fn):
    # คำนวณจุด PRE และ TARGET พร้อม q ที่ถึงจุดเหล่านั้น
    z_tool = R_target[:,2]
    p_pre  = p_target - PREAPPROACH_DIST * (APPROACH_TOOL_Z * z_tool)
    q_pre, *_ = solver_fn(q_start, p_pre, R_target)
    q_fin, it, ep, eo, ok = solver_fn(q_pre, p_target, R_target)
    return (p_pre, q_pre), (p_target, q_fin), ok

# ===================== จาคอบเบียนเชิงตัวเลข =====================
def jacobian_numeric(theta, h=1e-5):
    # central difference; h เล็กไปอาจมี noise, ใหญ่ไปจะหยาบ
    theta = np.array(theta, float)
    p0, _, T0 = ur5_forward_kinematrix(theta)
    R0 = T0[:3, :3]
    J = np.zeros((6, 6), float)
    for j in range(6):
        qp = theta.copy(); qp[j] += h
        qm = theta.copy(); qm[j] -= h
        pp, _, Tp = ur5_forward_kinematrix(qp)
        pm, _, Tm = ur5_forward_kinematrix(qm)
        Rp = Tp[:3,:3]; Rm = Tm[:3,:3]
        dp = (pp - pm) / (2*h)
        wp = log_SO3(Rp @ R0.T)
        wm = log_SO3(Rm @ R0.T)
        dw_body = (wp - wm) / (2*h)
        dw = R0 @ dw_body
        J[0:3, j] = dp
        J[3:6, j] = dw
    return J

# ===================== ตัวช่วย SVD / Nullspace =====================
def svd_damped_pinv(J, lam):
    # pseudoinverse แบบ DLS (lam สูงขึ้น = เสถียรขึ้นแต่ตอบช้าลง)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_d = s / (s*s + lam*lam)
    return (Vt.T * s_d) @ U.T

def limit_avoidance_grad(q):
    # ดึงมุมเข้าใกล้ Q_NOM ใน nullspace
    w = 0.1
    return -w*(q - Q_NOM)

def clamp_step(dq, dq_max=10.0*d2r):
    # จำกัดก้าวต่อรอบ
    return np.clip(dq, -dq_max, dq_max)

# ===================== IK: Position-only + 6D DLS =====================
def ik_pos_only(theta0, p_target, max_iters=300, tol=2e-4, lam0=0.03, dq_max=10.0*d2r):
    # เริ่มต้นด้วยตำแหน่งล้วน (ช่วยทำให้เข้าใกล้ก่อน)
    th = ang_wrap_vec(theta0)
    for _ in range(max_iters):
        p_cur, _, _ = ur5_forward_kinematrix(th)
        ep = p_target - p_cur
        if np.linalg.norm(ep) < tol:
            break
        J = jacobian_numeric(th)
        Jp = J[:3,:]
        try:
            cond_p = np.linalg.cond(Jp)
        except:
            cond_p = 1e6
        lam = float(np.clip(lam0*(cond_p/50.0), 1e-3, 2e-1))
        dq = svd_damped_pinv(Jp, lam) @ ep
        dq += 0.03*limit_avoidance_grad(th)
        dq = np.clip(dq, -dq_max, dq_max)
        th = ang_wrap_vec(clamp_q(th + dq))
    return th

def R_with_tool_yaw(R_base, yaw):
    # """หมุน R_base เพิ่มรอบ Z_tool ด้วยมุม yaw (rad)"""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz_tool = np.array([[c,-s,0],
                        [s, c,0],
                        [0, 0,1]], float)
    return R_base @ Rz_tool

def dls_6d_weighted(J, e6, lam, w_pos=1.0, w_ori=1.0):
    # DLS 6D ถ่วงน้ำหนักตำแหน่ง/ทิศทาง
    W = np.diag([w_pos, w_pos, w_pos, w_ori, w_ori, w_ori])
    JW = W @ J
    eW = W @ e6
    JJt = JW @ JW.T
    dq = JW.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), eW)
    return dq

def ik_priority_dls(theta0, p_target, R_target,
                    max_iters=1000, tol_pos=1e-4, tol_ori=2e-3,
                    lam_pos0=0.02, lam_ori0=0.02, dq_max=12.0*d2r,
                    ko_base=0.06, w_null=0.08, warmup=True, do_polish=True):
    # เฟสตำแหน่ง -> โอเรียนเตชัน (nullspace) -> 6D รวม -> polish
    th = ang_wrap_vec(np.array(theta0, float))
    if warmup:
        th = ik_pos_only(th, p_target, max_iters=300, tol=2e-4, lam0=lam_pos0, dq_max=dq_max)
    alphas = [1.0, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04]
    pos_phase_done = False
    for k in range(1, max_iters+1):
        p_cur, _, Tcur = ur5_forward_kinematrix(th)
        R_cur = Tcur[:3,:3]
        ep = p_target - p_cur
        ew_body = log_SO3(R_target @ R_cur.T)
        ew = R_cur @ ew_body
        epos = np.linalg.norm(ep)
        eori = np.linalg.norm(ew)
        if epos < tol_pos and eori < tol_ori:
            break
        J = jacobian_numeric(th)
        Jp = J[:3,:]
        Jo = J[3:,:]
        if (not pos_phase_done) or (epos > 3e-4):
            try: cond_p = np.linalg.cond(Jp)
            except: cond_p = 1e6
            lam_p = float(np.clip(lam_pos0 * (cond_p/50.0), 1e-3, 2e-1))
            Jp_pinv = svd_damped_pinv(Jp, lam_p)
            dq_pos = Jp_pinv @ ep
            Np = np.eye(6) - Jp_pinv @ Jp
            ko_now = ko_base + 0.6*(1.0 - math.exp(-epos/0.04))
            Jo_bar = Jo @ Np
            try: cond_o = np.linalg.cond(Jo_bar)
            except: cond_o = 1e6
            lam_o = float(np.clip(lam_ori0 * (cond_o/50.0), 1e-3, 2e-1))
            Jo_bar_pinv = svd_damped_pinv(Jo_bar, lam_o)
            dq_ori = Jo_bar_pinv @ (ko_now * ew)
            dq_ns = w_null * (Q_NOM - th)
            dq = dq_pos + Np @ dq_ori + Np @ dq_ns
            if epos < 3e-4:
                pos_phase_done = True
        else:
            try: cond6 = np.linalg.cond(J)
            except: cond6 = 1e6
            lam6 = float(np.clip(lam_ori0 * (cond6/50.0), 1e-3, 1.5e-1))
            w_pos = 8.0
            w_ori = 2.0 + 6.0*(1.0 - math.exp(-eori/0.08))
            e6 = np.hstack([ep, ew])
            dq = dls_6d_weighted(J, e6, lam6, w_pos=w_pos, w_ori=w_ori)
        dq = np.clip(dq, -dq_max, dq_max)
        p0 = p_cur; e0w = ew
        c0 = np.linalg.norm(p_target - p0) + 0.25*np.linalg.norm(e0w)
        improved = False
        th_best = th; c_best = c0
        for a in alphas:
            th_try = ang_wrap_vec(clamp_q(th + a*dq))
            p_t, _, T_t = ur5_forward_kinematrix(th_try)
            R_t = T_t[:3,:3]
            ep_t = p_target - p_t
            ew_t = R_t @ log_SO3(R_target @ R_t.T)
            c_t = np.linalg.norm(ep_t) + 0.25*np.linalg.norm(ew_t)
            if c_t < c_best:
                th_best = th_try; c_best = c_t; improved = True
                break
        th = th_best if improved else ang_wrap_vec(clamp_q(th + 0.05*dq))
    if do_polish:
        th = ik_pos_only(th, p_target, max_iters=120, tol=1.0e-4, lam0=0.02, dq_max=dq_max)
        for _ in range(60):
            p_cur, _, Tcur = ur5_forward_kinematrix(th)
            R_cur = Tcur[:3,:3]
            ep = p_target - p_cur
            if np.linalg.norm(ep) < 1.2e-4:
                break
            J = jacobian_numeric(th)
            Jp = J[:3,:]; Jo = J[3:,:]
            Jp_pinv = svd_damped_pinv(Jp, 0.02)
            Np = np.eye(6) - Jp_pinv @ Jp
            ew = R_cur @ log_SO3(R_target @ R_cur.T)
            dq = Jp_pinv @ ep + Np @ (0.015 * Jo.T @ ew)
            th = ang_wrap_vec(clamp_q(th + np.clip(dq, -8.0*d2r, 8.0*d2r)))
    p_cur, _, Tcur = ur5_forward_kinematrix(th)
    R_cur = Tcur[:3,:3]
    ep = p_target - p_cur
    ew = R_cur @ log_SO3(R_target @ R_cur.T)
    ok = (np.linalg.norm(ep) < tol_pos) and (np.linalg.norm(ew) < tol_ori)
    return th, k, float(np.linalg.norm(ep)), float(np.linalg.norm(ew)), ok

# ===================== ซีน/ข้อต่อ =====================
def get_or_create_target_cube(sim):
    # สร้างลูกบาศก์เป้า หากยังไม่มีในซีน
    try:
        return sim.getObject("/TargetCube")
    except:
        pass
    size = [0.05, 0.05, 0.05]
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 12)
    sim.setObjectAlias(h, "TargetCube", 0)
    sim.setObjectPosition(h, -1, [0.45, 0.0, 0.65])
    sim.setObjectOrientation(h, -1, [0.0, 0.0, 0.0])
    return h

def get_q(sim, hdl_j):
    # อ่านมุมข้อต่อทั้ง 6
    return np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)

def set_q(sim, hdl_j, q, use_target=True):
    # ส่งคำสั่งมุมไปยังคอนโทรลเลอร์หรือเซ็ตตรง
    qn = ang_wrap_vec(q)
    for i in range(6):
        if use_target:
            sim.setJointTargetPosition(hdl_j[i], float(qn[i]))
        else:
            sim.setJointPosition(hdl_j[i], float(qn[i]))

def seed_face_target_xy(p_t, q_now, shift_pi=False):
    # ตั้งมุมฐานหันเข้าหาเป้าในระนาบ XY (shift 180° ได้)
    q = q_now.copy()
    base = math.atan2(p_t[1], p_t[0])
    if shift_pi:
        base = ang_wrap(base + math.pi)
    q[0] = ang_wrap(base)
    return q

def print_log(sim, s):
    # พิมพ์ข้อความไปคอนโซลในซีน/STDOUT
    try: sim.auxiliaryConsolePrint(aux, s+"\n")
    except: sys.stdout.write(s+"\n")

# เวลาการเคลื่อนและเวลาค้างที่เป้า
MOVE_TIME = 5.0
DWELL_TIME = 2.0

def log_once(sim, aux, s):
    # log บรรทัดเดียวสั้น ๆ
    try:
        sim.auxiliaryConsolePrint(aux, s+"\n")
    except:
        sys.stdout.write(s+"\n")

def run_leg(sim, aux, hdl_j, hdl_end, q_from, q_to, target_pos, target_eul_rad, label):
    # เคลื่อนแขนจาก q_from -> q_to ภายใน MOVE_TIME พร้อม log ระหว่างทาง (ย้ายออกจาก sysCall_thread)
    t_start = sim.getSimulationTime()
    t_end   = t_start + MOVE_TIME
    dq_full = ang_wrap_vec(q_to - q_from)
    last_log = -1.0
    while True:
        t_now = sim.getSimulationTime()
        s = 0.0 if t_end == t_start else (t_now - t_start) / (t_end - t_start)
        if s < 0.0: s = 0.0
        if s > 1.0: s = 1.0
        q_cmd = ang_wrap_vec(q_from + s * dq_full)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q_cmd[i]))
        if (last_log < 0) or (t_now - last_log > 0.2):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            msg = []
            msg.append(f"{label} s={s:5.2f} t={t_now:6.2f}s")
            msg.append("  target_pos:      " + ", ".join(f"{x:7.4f}" for x in target_pos))
            msg.append("  target_eul(deg): " + ", ".join(f"{x*r2d:7.2f}" for x in target_eul_rad))
            msg.append("  sim_pos:         " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul(deg):    " + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now
        if s >= 1.0:
            break
        sim.switchThread()

# ===================== Gripper =====================
def get_gripper_handle(sim):
    return sim.getObject("/UR5/RG2/openCloseJoint")

# grip_action: 1 = ปิด, 2 = เปิด
def gripper_action(sim, hdl_gripper, grip_action):
    if grip_action == 1:
        sim.setJointTargetForce(hdl_gripper, -20)
        sim.setJointTargetVelocity(hdl_gripper, -0.2)
    elif grip_action == 2:
        sim.setJointTargetForce(hdl_gripper, 3)
        sim.setJointTargetVelocity(hdl_gripper, 0.2)
        
# ===================== Trajectory helpers =====================
# LSPB scalar profile (0->1)
def lspb_s(t, T, tb=None):
    T = float(T)
    t = max(0.0, min(float(t), T))
    if tb is None:
        tb = min(0.25*T, 0.4) # สัดส่วนเร่ง/หน่วง
    if t < tb:
        return 0.5*(t/tb)**2
    elif t <= T - tb:
        return (t - tb/2.0) / (T - tb)
    else:
        td = T - t
        return 1.0 - 0.5*(td/tb)**2

def sample_times(T, dt):
    n = max(2, int(T/dt)+1)
    return [i*float(T)/(n-1) for i in range(n)]


# ============ Joint-space trajectory ============

def plan_joint_traj(q_from, q_to, T=3.0, dt=0.02):
    q_from = np.array(q_from, float); q_to = np.array(q_to, float)
    times = sample_times(T, dt)
    dq = ang_wrap_vec(q_to - q_from)
    return [ang_wrap_vec(q_from + lspb_s(t, T)*dq) for t in times]

# ============ Task-space straight line (with IK) ============

def plan_task_line(q_seed, p_start, p_end, R_target, solver_fn, T=3.0, dt=0.04):
    p_start = np.array(p_start, float); p_end = np.array(p_end, float)
    times = sample_times(T, dt)
    traj_q = []
    q = np.array(q_seed, float)
    for t in times:
        s = lspb_s(t, T)
        p = (1-s)*p_start + s*p_end
        q, it, ep, eo, ok = solver_fn(q, p, R_target)
        traj_q.append(q)
    return traj_q

def plan_yaw_in_place(q_seed, p_fixed, R_start, yaw_target_rad, solver_fn,
                      steps=20, per_wp_max_iters=90):
    # """ไล่หมุนหัวจาก R_start  to  R_with_tool_yaw(R_start, yaw_target_rad) ที่ตำแหน่งเดิม"""
    traj = []
    q = np.array(q_seed, float)
    for i in range(steps+1):
        s = i/steps
        R_t = R_with_tool_yaw(R_start, s*yaw_target_rad)
        q, it, ep, eo, ok = solver_fn(q, p_fixed, R_t)
        # กันค้าง: ผ่อน iters นิดถ้าไม่คอนเวิร์จ แต่เก็บ q ล่าสุดไว้ให้ต่อเนื่อง
        if (not ok) and (per_wp_max_iters > 50):
            q, it, ep, eo, ok = ik_priority_dls(
                q, p_fixed, R_t,
                max_iters=50, tol_pos=1e-4, tol_ori=3e-3,
                lam_pos0=0.03, lam_ori0=0.03, dq_max=12.0*d2r,
                ko_base=0.05, w_null=0.08, warmup=False, do_polish=False
            )
        traj.append(q.copy())
    return traj

# ============ Execute a list of joint waypoints ============
def _seg_time_from_limits(dq, v_max=QDOT_MAX, a_max=QDDOT_MAX):
    dq = np.abs(np.array(dq, float))
    # เวลาตามข้อที่ช้าที่สุด (per-joint LSPB)
    T_v = dq / np.maximum(1e-9, v_max)
    # ใช้กฎสามเหลี่ยม/สี่เหลี่ยมอย่างหยาบสำหรับเวลาเร่งหน่วง
    T_a = 2.0*np.sqrt(dq/np.maximum(1e-9, a_max))
    return float(np.max(np.maximum(T_v, T_a)))


def execute_traj(sim, hdl_j, q_list, label="[TRAJ]", v_scale=0.9, a_scale=0.9):
    # """
    # เล่น trajectory แบบ smooth: interpolate ระหว่าง q[k] -> q[k+1]
    # ด้วย LSPB ต่อเนื่อง (เหมือน run_leg) แทนการ set เป็นขั้น ๆ
    # """
    if not q_list: return
    q_list = [ang_wrap_vec(q) for q in q_list]

    for k in range(len(q_list)-1):
        q0 = q_list[k].copy()
        q1 = q_list[k+1].copy()
        dq = ang_wrap_vec(q1 - q0)

        # เวลาเซกเมนต์จากลิมิตข้อต่อ
        Tseg = _seg_time_from_limits(dq, v_max=v_scale*QDOT_MAX, a_max=a_scale*QDDOT_MAX)
        if Tseg < 1e-3: Tseg = 1e-3

        # พารามิเตอร์ LSPB ต่อเนื่อง
        tb, T, is_tri = lspb_time_params_scalar(np.max(np.abs(dq)), v_scale*np.max(QDOT_MAX), a_scale*np.max(QDDOT_MAX))
        # ใช้เวลาจริงของเซกเมนต์แทน
        T = max(T, Tseg)

        t0 = sim.getSimulationTime()
        while True:
            t = sim.getSimulationTime() - t0
            s = lspb_s_time_hold(t, tb, T, is_tri)   # 0..1 ต่อเนื่อง
            q_cmd = ang_wrap_vec(q0 + s*dq)
            for i in range(6):
                sim.setJointTargetPosition(hdl_j[i], float(q_cmd[i]))
            if t >= T: break
            sim.switchThread()

        # small settle
        for _ in range(2):
            for i in range(6):
                sim.setJointTargetPosition(hdl_j[i], float(q1[i]))
            sim.switchThread()

# ===================== Pick & Place sequence =====================

def do_pick_and_place(sim, hdl_j, hdl_end,
                      box_base,               # พิกัดที่จะคีบ (BASE) [x,y,z] จุดกึ่งกลางฝาบนกล่อง
                      place_base,             # พิกัดจุดวาง (BASE) [x,y,z] จุดกึ่งกลางตำแหน่งวาง
                      yaw_deg=0.0,
                      pre_back=0.14,
                      descend=0.08,
                      t_hint=15.0):

    GRASP_CLEAR_Z = 0.12
    PLACE_CLEAR_Z = 0.12
    DESCENT_Z     = float(descend)

    # --- จุดใน BASE สำหรับหยิบ (ABOVE/GRASP) ---
    p_box_top_b   = np.array(box_base,   float) + np.array([0,0, GRASP_CLEAR_Z])
    p_box_grasp_b = np.array(box_base,   float) - np.array([0,0, DESCENT_Z])

    # --- จุดใน BASE สำหรับวาง (ABOVE/DROP) ---
    p_place_top_b  = np.array(place_base, float) + np.array([0,0, PLACE_CLEAR_Z])
    p_place_drop_b = np.array(place_base, float) - np.array([0,0, DESCENT_Z])  # ถ้าแหย่ลงลึกไป ให้ลด DESCENT_Z

    # --- TOP-DOWN + PRE ตามแกน Z_tool (สำหรับหยิบ) ---
    heading_pick = math.atan2(p_box_top_b[1], p_box_top_b[0])
    R_top_pick   = make_R_topdown(yaw = heading_pick + yaw_deg*d2r)
    z_tool_pick  = R_top_pick[:,2]
    p_pre_b      = p_box_top_b - pre_back * (APPROACH_TOOL_Z * z_tool_pick)

    # --- TOP-DOWN (สำหรับวาง): จะหันหัวตาม heading ไปยังจุดวาง ---
    heading_place = math.atan2(p_place_top_b[1], p_place_top_b[0])
    R_top_place   = make_R_topdown(yaw = heading_place + yaw_deg*d2r)

    # --- q เริ่ม ---
    q_now = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)

    # 1) PRE  to  ABOVE (หยิบ)
    T_pre   = choose_T_task_with_blend(p_pre_b, p_box_top_b, ALIN_MAX, t_hint)
    q_above = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_now, p_start=p_pre_b, p_final=p_box_top_b, R_hold=R_top_pick,
        T_move=T_pre, label="[PRE to ABOVE]")

    # 2) ABOVE  to  GRASP (หยิบ)
    T_down = choose_T_task_with_blend(p_box_top_b, p_box_grasp_b, ALIN_MAX, t_hint)
    q_at_grasp = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_above, p_start=p_box_top_b, p_final=p_box_grasp_b, R_hold=R_top_pick,
        T_move=T_down, label="[ABOVE to GRASP]")

    # 3) GRIP (ปิดกริปเปอร์)
    hdl_grip = get_gripper_handle(sim)
    gripper_action(sim, hdl_grip, 1)
    t0 = sim.getSimulationTime()
    while sim.getSimulationTime() - t0 < 0.25:
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q_at_grasp[i]))
        sim.switchThread()

    # 4) LIFT (กลับขึ้น ABOVE หยิบ)
    T_up = choose_T_task_with_blend(p_box_grasp_b, p_box_top_b, ALIN_MAX, t_hint)
    q_lift = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_at_grasp, p_start=p_box_grasp_b, p_final=p_box_top_b,
        R_hold=R_top_pick, T_move=T_up, label="[LIFT]")

    # 5) TRANSIT (ถือของไปเหนือจุดวาง)
    T_go_place = choose_T_task_with_blend(p_box_top_b, p_place_top_b, ALIN_MAX, t_hint)
    q_over_place = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_lift, p_start=p_box_top_b, p_final=p_place_top_b,
        R_hold=R_top_place, T_move=T_go_place, label="[TRANSIT to PLACE_ABOVE]")

    # 6) DROP (ลงไปวาง)
    T_down_place = choose_T_task_with_blend(p_place_top_b, p_place_drop_b, ALIN_MAX, t_hint)
    q_at_drop = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_over_place, p_start=p_place_top_b, p_final=p_place_drop_b,
        R_hold=R_top_place, T_move=T_down_place, label="[ABOVE to DROP]")

    # 7) RELEASE (เปิดกริปเปอร์)
    gripper_action(sim, hdl_grip, 2)
    t1 = sim.getSimulationTime()
    while sim.getSimulationTime() - t1 < 0.20:
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q_at_drop[i]))
        sim.switchThread()

    # 8) RETRACT (ยกขึ้นจากจุดวางกลับ ABOVE วาง)
    T_up_place = choose_T_task_with_blend(p_place_drop_b, p_place_top_b, ALIN_MAX, t_hint)
    q_final = run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
        q_start=q_at_drop, p_start=p_place_drop_b, p_final=p_place_top_b,
        R_hold=R_top_place, T_move=T_up_place, label="[RETRACT]")

    return q_final

def lspb_time_params(L, v_max, a_max):
    # return (tb, T, is_triangular)
    # ถ้า v_max^2 / a_max >= L  => โปรไฟล์สามเหลี่ยม (ไม่มีช่วงคงความเร็ว)
    if v_max*v_max / a_max >= L:
        tb = math.sqrt(L / a_max)
        T  = 2.0 * tb
        return tb, T, True
    else:
        tb = v_max / a_max
        T  = tb + (L - v_max*tb) / v_max + tb  # = L/v_max + tb
        return tb, T, False

def lspb_s_time(t, tb, T, is_tri):
    t = max(0.0, min(t, T))
    if is_tri:
        # triangular: accel tb, decel tb
        if t <= tb:
            return 0.5*(t/tb)**2
        elif t <= (T - tb):
            # สำหรับสามเหลี่ยม T-2*tb == 0 ดังนั้นสัดส่วนตรงกลางแทบไม่มี
            # map linear ระหว่าง s(tb)=0.5 กับ s(T-tb)=0.5? แทบไม่ใช้
            return 0.5 + 0.5*((t - tb)/(T - 2*tb + 1e-12))
        else:
            td = T - t
            return 1.0 - 0.5*(td/tb)**2
    else:
        # trapezoid: accel tb, cruise, decel tb
        if t <= tb:
            return 0.5*(t/tb)**2
        elif t <= (T - tb):
            return (t - tb/2.0) / (T - tb)
        else:
            td = T - t
            return 1.0 - 0.5*(td/tb)**2
        
def plan_task_line_lspb(q_seed, p_start, p_end, R_target, solver_fn,
                        v_max=0.30, a_max=0.60, dt=0.02,
                        max_backtrack=6, sim=None, yield_every=6,
                        per_wp_max_iters=90, ori_relax_deg=6.0,
                        consec_fail_skip=4,      # fail ติดกันเท่าไรจึงกระโดดข้าม
                        max_skip_per_hit=5,      # เวลาข้าม ข้ามได้มากสุดกี่ waypoint
                        posonly_fallback=True):  # ยอม pos-only ชั่วคราวเพื่อให้เคลื่อนต่อ
    # """
    # เดินเส้นตรง p_start -> p_end ด้วย LSPB (จำกัด v_max, a_max) และ orientation คงเดิม (R_target)
    # คืนรายชื่อ q waypoints ที่แซมเปิลตาม dt
    # ถ้า IK จุดใดไม่คอนเวิร์จ จะ backtrack ลดระยะก้าว (halve) ได้สูงสุด max_backtrack ครั้ง
    # """
    p_start = np.array(p_start, float); p_end = np.array(p_end, float)
    L = float(np.linalg.norm(p_end - p_start))
    if L < 1e-9:
        return [np.array(q_seed, float)]

    tb, T, is_tri = lspb_time_params(L, v_max, a_max)
    n = max(2, int(T/dt)+1)
    times = [i*float(T)/(n-1) for i in range(n)]

    q_list = []
    q = np.array(q_seed, float)

    def solver_wp(qs, pt, Rt, w_ori_scale=1.0, iters=per_wp_max_iters):
        return ik_priority_dls(
            qs, pt, Rt,
            max_iters=int(iters), tol_pos=1e-4, tol_ori=2e-3,
            lam_pos0=0.03, lam_ori0=0.03, dq_max=12.0*d2r,
            ko_base=0.06*w_ori_scale, w_null=0.08, warmup=False, do_polish=False
        )

    fail_streak = 0
    i = 0
    while i < n:
        if sim and (i % yield_every) == 0:
            sim.switchThread()

        s = lspb_s_time(times[i], tb, T, is_tri)
        p = (1.0 - s)*p_start + s*p_end

        ok_step = False
        p_hi = p.copy()
        w_scale = 1.0

        # งบต่อ waypoint น้อยลง: backtrack นิดหน่อย + ลด iters เมื่อวนหลายรอบ
        for b in range(min(max_backtrack, 4)+1):
            iters = max(50, per_wp_max_iters - 10*b)  # ยิ่ง backtrack ยิ่งลดงบ
            q_try, it, ep, eo, ok = solver_wp(q, p_hi, R_target, w_ori_scale=w_scale, iters=iters)
            if ok:
                q = q_try; ok_step = True; break

            # ลอง pos-only fallback (ช่วยขยับเข้าใกล้ไว ๆ)
            if posonly_fallback and b >= 1:
                q_pos = ik_pos_only(q, p_hi, max_iters=60, tol=6e-4, lam0=0.03, dq_max=14.0*d2r)
                if np.all(np.isfinite(q_pos)):
                    q = q_pos; ok_step = True; break

            # ลดสเต็ปและผ่อน orientation เร็วขึ้น
            s_prev = lspb_s_time(times[max(i-1,0)], tb, T, is_tri)
            s_half = 0.5*(s + s_prev)
            p_hi = (1.0 - s_half)*p_start + s_half*p_end
            w_scale = max(0.30, 0.6*w_scale)

        if not ok_step:
            # ยังไม่ได้: ไม่รอ—ข้าม waypoint ทีละหลายขั้นเลย
            fail_streak += 1
            if sim:
                sim.auxiliaryConsolePrint(aux, f"[WARN] IK failed at wp {i+1}/{n}, skip.\n")
            skip = max(1, min(max_skip_per_hit, 1 + fail_streak // consec_fail_skip))
            i += skip
            q_list.append(q.copy())  # เก็บค่าเดิมไว้ให้ทางเดินต่อเนื่อง
            continue

        # ถ้าทำได้ รีเซ็ต fail และไปต่อปกติ
        fail_streak = 0
        q_list.append(q.copy())
        i += 1

    return q_list

def plan_task_line_hybrid(q_seed, p_start, p_end, R_target, solver_fn,
                          keep_ori_after_s=0.75,      # ล็อก orientation หลังสัดส่วนเส้นทางนี้
                          v_max=0.30, a_max=0.60, dt=0.02,
                          per_wp_max_iters=80, ori_relax_deg=8.0,
                          posonly_iters=80, max_backtrack=3, sim=None, yield_every=6):
    p_start = np.array(p_start, float); p_end = np.array(p_end, float)
    L = float(np.linalg.norm(p_end - p_start))
    if L < 1e-9: return [np.array(q_seed, float)]

    tb, T, is_tri = lspb_time_params(L, v_max, a_max)
    n = max(2, int(T/dt)+1)
    times = [i*float(T)/(n-1) for i in range(n)]

    q_list = []
    q = np.array(q_seed, float)

    # ตัวช่วย IK (ล็อกพารามิเตอร์ให้เบา)
    def solve_ori(qs, pt, Rt, wscale=1.0, iters=per_wp_max_iters):
        return ik_priority_dls(qs, pt, Rt,
            max_iters=int(iters), tol_pos=1e-4, tol_ori=2e-3,
            lam_pos0=0.03, lam_ori0=0.03, dq_max=12.0*d2r,
            ko_base=0.06*wscale, w_null=0.08, warmup=False, do_polish=False)

    for i, t in enumerate(times, 1):
        if sim and (i % yield_every) == 0: sim.switchThread()
        s = lspb_s_time(t, tb, T, is_tri)
        p = (1.0 - s)*p_start + s*p_end

        if s < keep_ori_after_s:
            # ช่วงแรก: pos-only ให้ลื่น ๆ
            q = ik_pos_only(q, p, max_iters=posonly_iters, tol=4e-4, lam0=0.03, dq_max=14.0*d2r)
        else:
            # ช่วงท้าย: ล็อก TOP-DOWN แต่ผ่อนหน่อย
            ok_step = False
            p_try = p.copy(); wscale = 1.0
            for b in range(max_backtrack+1):
                q_try, it, ep, eo, ok = solve_ori(q, p_try, R_target, wscale, per_wp_max_iters-10*b)
                if ok:
                    q = q_try; ok_step = True; break
                # ผ่อนความเข้มงวด & ลดสเต็ป
                wscale = max(0.35, 0.7*wscale)
                s_prev = lspb_s_time(times[max(i-1,0)], tb, T, is_tri)
                s_half = 0.5*(s + s_prev)
                p_try = (1.0 - s_half)*p_start + s_half*p_end
            if not ok_step:
                # ถ้าไม่ไหวจริง ๆ ใช้ pos-only ต่อ เพื่อไม่หยุด
                q = ik_pos_only(q, p, max_iters=posonly_iters//2, tol=6e-4, lam0=0.03, dq_max=14.0*d2r)

        q_list.append(q.copy())
    return q_list

def plan_task_line_lspb_oriented(
    q_seed, p_start, p_end, R_target, solver_fn,
    v_max=0.28, a_max=0.65, dt=0.02,
    per_wp_max_iters=80, max_backtrack=4,
    max_ori_err_deg=8.0,       # "leash" ให้หัวเอียงได้ไม่เกินกี่องศาระหว่างทาง
    consec_fail_skip=3,        # fail ติดกันกี่ครั้งถึงจะกระโดดข้าม
    max_skip_per_hit=4,        # ข้ามทีละมากสุดกี่ waypoint
    sim=None
):
    p_start = np.array(p_start, float); p_end = np.array(p_end, float)

    # เวลาตามความยาวเส้นทาง
    L = float(np.linalg.norm(p_end - p_start))
    if L < 1e-9: return [np.array(q_seed, float)]
    tb, T, is_tri = lspb_time_params(L, v_max, a_max)
    n = max(2, int(T/dt)+1)
    times = [i*float(T)/(n-1) for i in range(n)]

    q_list = []
    q = np.array(q_seed, float)
    leash = float(max_ori_err_deg)*d2r

    def _solve(qs, pt, Rt, w_ori_scale=1.0, iters=per_wp_max_iters):
        return ik_priority_dls(
            qs, pt, Rt,
            max_iters=int(iters), tol_pos=1e-4, tol_ori=2e-3,
            lam_pos0=0.03, lam_ori0=0.03, dq_max=12.0*d2r,
            ko_base=0.06*w_ori_scale, w_null=0.08,
            warmup=False, do_polish=False
        )

    fail_streak = 0
    i = 0
    while i < n:
        if sim and (i % 6) == 0: sim.switchThread()

        s = lspb_s_time(times[i], tb, T, is_tri)
        p = (1.0 - s)*p_start + s*p_end

        # ลองแก้ IK แบบคุมทิศ (หัวตรง) แต่ไม่ยอมวนยาว
        ok_step = False
        p_try = p.copy()
        w_scale = 1.0

        for b in range(max_backtrack+1):
            iters = max(50, per_wp_max_iters - 10*b)
            q_try, it, ep, eo, ok = _solve(q, p_try, R_target, w_ori_scale=w_scale, iters=iters)
            if ok:
                # เช็คว่า orientation ไม่หลุดเกิน leash
                _, _, Tcur = ur5_forward_kinematrix(q_try)
                R_cur = Tcur[:3,:3]
                eo_vec = R_cur @ log_SO3(R_target @ R_cur.T)
                if np.linalg.norm(eo_vec) <= leash:
                    q = q_try; ok_step = True
                    break
            # backtrack + ผ่อนน้ำหนักทิศลง (ลด w  to  ง่ายขึ้น)
            s_prev = lspb_s_time(times[max(i-1,0)], tb, T, is_tri)
            s_half = 0.5*(s + s_prev)
            p_try = (1.0 - s_half)*p_start + s_half*p_end
            w_scale = max(0.35, 0.7*w_scale)

        if ok_step:
            q_list.append(q.copy())
            i += 1
            continue
        
        if not ok_step:
            # --- NO-SKIP POLICY ---
            # 1) ลอง pos-only เพื่อ "ลงจุด" waypoint นี้ให้ได้ก่อน
            q_pos = ik_pos_only(q, p, max_iters=80, tol=4e-4, lam0=0.03, dq_max=14.0*d2r)
            if np.all(np.isfinite(q_pos)):
                q = q_pos
                q_list.append(q.copy())
                i += 1
                continue

            # 2) ยังไม่ได้: ลดสเต็ปเฉพาะบริเวณ (refine waypoint แทนที่จะข้าม)
            if i+1 < n:
                # แทรกเวลาตรงกลางเพื่อให้มี waypoint ระหว่างทางเพิ่ม
                t_mid = 0.5*(times[i] + times[i+1])
                times.insert(i+1, t_mid)
                n += 1
                # ไม่เพิ่ม i เพื่อให้ลอง waypoint เดิมใหม่ (ละเอียดขึ้น)
                if sim: sim.auxiliaryConsolePrint(aux, f"[REFINE] inserted mid waypoint at i={i+1}\n")
                continue

            # 3) ปลายสุดแล้วยังไม่ได้: เก็บ q เดิมไว้เพื่อความต่อเนื่อง แล้วไปต่อ (ไม่ข้าม)
            q_list.append(q.copy())
            i += 1
            continue

    # snap orientation เบา ๆ ที่ปลาย (pos แน่น, ori ผ่อนนิดหน่อย)
    if len(q_list) > 0:
        q_last = q_list[-1]
        q_snap, _, _, _, _ = ik_priority_dls(
            q_last, p_end, R_target,
            max_iters=80, tol_pos=8e-5, tol_ori=min(0.12, leash),  # ~7°
            lam_pos0=0.03, lam_ori0=0.03, dq_max=10.0*d2r,
            ko_base=0.05, w_null=0.06, warmup=False, do_polish=False
        )
        q_list[-1] = q_snap

    return q_list

def execute_traj(sim, hdl_j, q_list, label="[TRAJ]", settle_last=6):
    # """
    # ส่งชุด q_list เข้า controller ทีละสเต็ป พร้อมสลับเธรดทุกเฟรม
    # เพิ่ม settle frame ท้ายๆ อีกนิด เพื่อให้ปลายแขนทันเกาะค่าปลาย ไม่ดูค้างที่ step N-1/N
    # """
    for k, q in enumerate(q_list, 1):
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        sim.switchThread()
        # log เบาๆ เพื่อลด overhead
        if (k % 25) == 0 or k == len(q_list):
            try:
                sim.auxiliaryConsolePrint(aux, f"{label} {k}/{len(q_list)}\n")
            except:
                pass
    # ถือปลายทางอีกหลายเฟรม (กันค้าง 150/151)
    if len(q_list) > 0:
        q_last = q_list[-1]
        for _ in range(settle_last):
            for i in range(6):
                sim.setJointTargetPosition(hdl_j[i], float(q_last[i]))
            sim.switchThread()
            
def lspb_time_params_scalar(L, v_max, a_max):
    # L = |delta angle| (rad)
    if v_max*v_max / a_max >= L:
        tb = math.sqrt(L / a_max)  # triangular
        T  = 2.0 * tb
        return tb, T, True
    else:
        tb = v_max / a_max
        T  = L / v_max + tb  # trapezoid min time
        return tb, T, False

def lspb_s_time_hold(t, tb, T, is_tri):
    # s(t) แบบเดิม แต่ถ้า t>=T ให้คืน 1.0 (hold ปลาย)
    if t <= 0.0: return 0.0
    if t >= T:   return 1.0
    if is_tri:
        if t <= tb:
            return 0.5*(t/tb)**2
        elif t <= (T - tb):
            # ช่วงยอด (สำหรับสามเหลี่ยมจริง ๆ จะสั้นมาก)
            return 0.5 + 0.5*((t - tb)/(T - 2*tb + 1e-12))
        else:
            td = T - t
            return 1.0 - 0.5*(td/tb)**2
    else:
        if t <= tb:
            return 0.5*(t/tb)**2
        elif t <= (T - tb):
            return (t - tb/2.0) / (T - tb)
        else:
            td = T - t
            return 1.0 - 0.5*(td/tb)**2
        
def plan_joint_traj_perjoint(q_from, q_to, v_max_vec=None, a_max_vec=None, dt=0.02):
    q_from = np.array(q_from, float)
    q_to   = np.array(q_to,   float)
    dq     = ang_wrap_vec(q_to - q_from)
    if v_max_vec is None: v_max_vec = QDOT_MAX
    if a_max_vec is None: a_max_vec = QDDOT_MAX
    v_max_vec = np.array(v_max_vec, float)
    a_max_vec = np.array(a_max_vec, float)

    # เวลาอย่างน้อยของแต่ละข้อ
    tb_list, T_list, tri_list = [], [], []
    for i in range(6):
        L = abs(float(dq[i]))
        if L < 1e-12:
            tb_i, T_i, tri_i = 0.0, 0.0, True
        else:
            tb_i, T_i, tri_i = lspb_time_params_scalar(L, v_max_vec[i], a_max_vec[i])
        tb_list.append(tb_i); T_list.append(T_i); tri_list.append(tri_i)

    T_all = max(T_list) if len(T_list) else 0.0
    if T_all < 1e-6:
        return [ang_wrap_vec(q_from.copy())]  # ไม่มีการเคลื่อน

    n = max(2, int(T_all/dt)+1)
    times = [i*float(T_all)/(n-1) for i in range(n)]

    traj = []
    for t in times:
        q = np.zeros(6, float)
        for i in range(6):
            if T_list[i] <= 0.0:
                s_i = 1.0
            else:
                s_i = lspb_s_time_hold(t, tb_list[i], T_list[i], tri_list[i])
            q[i] = ang_wrap(q_from[i] + s_i * dq[i])
        traj.append(q)
    return traj
            
def ik_pos_only_fast(theta0, p_target, max_iters=120, tol=4e-4, lam0=0.03, dq_max=14.0*d2r):
    # ใช้ ik_pos_only เดิม แต่ตั้งงบน้อยเพื่อความไว
    return ik_pos_only(theta0, p_target, max_iters=max_iters, tol=tol, lam0=lam0, dq_max=dq_max)

def plan_fast_to_pre(sim, q_now, p_box, solver_fn, dt=0.02):
    # เลือก approach เฉียง (SIDE) สำหรับจุดพรี
    R_side = make_R_side_toward_xy(p_box, yaw=0.0)
    # จุด PRE: ถอยจากกล่องตามแกน Z_tool ของ SIDE เล็กน้อย
    z_tool = R_side[:,2]
    p_pre  = np.array(p_box, float) - 0.12 * z_tool   # ถอย ~12 ซม. (ปรับได้)

    # 1) ใช้ IK ตำแหน่งอย่างเดียวไปหา PRE แบบเร็ว ๆ
    q_pre_seed = ik_pos_only_fast(q_now, p_pre, max_iters=150, tol=5e-4)

    # 2) วิ่ง joint-space ด้วย per-joint limits ให้คมและไว
    q_traj = plan_joint_traj_perjoint(q_now, q_pre_seed, v_max_vec=QDOT_MAX, a_max_vec=QDDOT_MAX, dt=dt)
    return q_traj, p_pre, R_side            

def auto_calibrate_DH_EE_Z(sim, hdl_j, tcp_handle):
    # อ่าน q ปัจจุบัน
    q_now = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    # FK จากโมเดล (ก่อนแก้)
    p_fk, _, T_fk = ur5_forward_kinematrix(q_now)
    # ตำแหน่ง TCP จริงจากซีน
    p_tcp = np.array(sim.getObjectPosition(tcp_handle, -1), float)
    # ส่วนต่างตามแกน Z_world ในท่านิ่ง (ถือว่าเฟรมปลายมือขนานกัน)
    dz = float(p_tcp[2] - p_fk[2])
    # ปรับความยาว along Z_tool (DH_EE[0,2]) ด้วย dz
    DH_EE[0,2] += dz
    # (ออปชัน) พิมพ์เช็ค
    sim.auxiliaryConsolePrint(aux, f"[CAL] Adjust DH_EE[0,2] by dz={dz:+.4f} -> {DH_EE[0,2]:.5f}\n")


def solver_posonly_fast(qs, pt):
    # Jacobian + DLS เหมือน ik_pos_only เดิม แต่ลดงบให้ไว
    return ik_pos_only(qs, pt, max_iters=120, tol=3e-4, lam0=0.03, dq_max=14.0*d2r)


def plan_task_line_lspb_posonly(q_seed, p_start, p_end,
                                v_max=0.35, a_max=0.9, dt=0.02,
                                max_backtrack=6, sim=None, yield_every=6,
                                per_wp_max_iters=90):
    p_start = np.array(p_start, float); p_end = np.array(p_end, float)
    L = float(np.linalg.norm(p_end - p_start))
    if L < 1e-9: return [np.array(q_seed, float)]

    tb, T, is_tri = lspb_time_params(L, v_max, a_max)
    n = max(2, int(T/dt)+1)
    times = [i*float(T)/(n-1) for i in range(n)]

    # ห่อ solver ให้จำกัดงบ/เร็ว
    def _solve(qs, pt):
        return ik_pos_only(qs, pt, max_iters=per_wp_max_iters, tol=3e-4, lam0=0.03, dq_max=14.0*d2r)

    q_list = []
    q = np.array(q_seed, float)
    for i, t in enumerate(times, 1):
        if sim and (i % yield_every) == 0: sim.switchThread()
        s = lspb_s_time(t, tb, T, is_tri)
        p = (1.0 - s)*p_start + s*p_end

        step_ok = False; p_hi = p.copy()
        for b in range(max_backtrack+1):
            q_try = _solve(q, p_hi)
            # _solve ของ ik_pos_only คืนค่า th อย่างเดียว
            if q_try is not None and np.all(np.isfinite(q_try)):
                q = q_try; step_ok = True; break
            s_prev = lspb_s_time(times[i-2], tb, T, is_tri) if i > 1 else 0.0
            s_half = 0.5*(s + s_prev)
            p_hi = (1.0 - s_half)*p_start + s_half*p_end

        if not step_ok and sim:
            sim.auxiliaryConsolePrint(aux, f"[WARN] pos-only IK fail wp {i}/{n}, keep last q\n")
        q_list.append(q.copy())
    return q_list

# ========== Jacobian resolved-rate 1 step ==========
def jacobian_resolved_rate_step(q_cur, p_des, R_des,
                                dt,                      # <<< เพิ่ม dt
                                lam=0.05,                # damping มากขึ้น (เดิม 0.02)
                                w_pos=6.0, w_ori=1.5,    # gain เบาลง
                                dq_clip_deg=8.0,         # cap ต่อรอบ
                                alpha=0.5):              # low-pass step
    # error
    p_now, _, Tcur = ur5_forward_kinematrix(q_cur)
    R_cur = Tcur[:3,:3]
    ep = np.asarray(p_des, float) - p_now
    ew_body = log_SO3(R_des @ R_cur.T)
    ew = R_cur @ ew_body

    # ผ่อนน้ำหนักทิศทางระหว่างยังไกลจุด (กัน “บิดหัว” แรง)
    if np.linalg.norm(ep) > 0.06:   # > 6 ซม. ให้เน้นตำแหน่งก่อน
        w_ori = 0.6 * w_ori

    # Jacobian step (DLS 6D)
    J = jacobian_numeric(q_cur)
    dq = dls_6d_weighted(J, np.hstack([ep, ew]), lam, w_pos=w_pos, w_ori=w_ori)

    # จำกัดก้าว: 1) ตามความเร็วข้อจริง 2) ตามมุมสูงสุดต่อรอบ 3) ทำ low-pass
    dq_clip = np.deg2rad(dq_clip_deg)
    dq = np.clip(dq, -dq_clip, dq_clip)                 # กัน spike เชิงตัวเลข
    dq = np.clip(dq, -QDOT_MAX*dt, QDOT_MAX*dt)         # <<< คุมความเร็วข้อรายสเต็ป
    q_next = ang_wrap_vec(clamp_q(q_cur + alpha*dq))    # <<< เอาแค่สัดส่วนของก้าว (นุ่ม)

    return q_next, ep, ew

# ========== Task LIPB utilities ==========
ALIN_MAX = np.array([1.0, 1.0, 1.0], float)  # m/s^2 ต่อแกน XYZ (ปรับได้)

def _amax_s_from_disp(d, amax_lin, eps=1e-9):
    d = np.asarray(d, float); amax_lin = np.asarray(amax_lin, float)
    mask = np.abs(d) > eps
    return float(np.min(amax_lin[mask] / np.abs(d[mask]))) if np.any(mask) else 1.0

def choose_T_task_with_blend(p0, pf, amax_lin, T_hint):
    amax_s = _amax_s_from_disp(np.asarray(pf)-np.asarray(p0), amax_lin)
    T_req = 2.0 / math.sqrt(max(amax_s, 1e-12))  # มี tb จริง
    return max(float(T_hint), float(T_req))

def lipb_scalar_params(p0, pf, amax_lin, T):
    d = np.asarray(pf, float) - np.asarray(p0, float)
    amax_s = _amax_s_from_disp(d, amax_lin)
    disc = max(0.0, T*T - 4.0/max(amax_s,1e-12))
    tb = 0.5 * (T - math.sqrt(disc))
    tb = float(np.clip(tb, 0.0, T/2.0))
    v_s = amax_s * tb
    return amax_s, tb, v_s, d

def lipb_scalar_eval(t, T, amax_s, tb):
    if t < tb:
        s=0.5*amax_s*t*t; sd=amax_s*t; sdd=amax_s
    elif t <= (T - tb):
        s=0.5*amax_s*tb*tb + (amax_s*tb)*(t - tb); sd=amax_s*tb; sdd=0.0
    else:
        td=T-t; s=1.0-0.5*amax_s*td*td; sd=amax_s*td; sdd=-amax_s
    return s, sd, sdd

# ========== ตัววิ่งเส้นตรงแบบ LIPB ด้วย resolved-rate ==========
def run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
                      q_start, p_start, p_final, R_hold,
                      T_move, label="[TASK-LIPB]"):
    amax_s, tb, v_s, d = lipb_scalar_params(p_start, p_final, ALIN_MAX, T_move)
    q_cmd = ang_wrap_vec(q_start.copy())
    t0 = sim.getSimulationTime()
    last_log = -1.0
    log_once(sim, aux, f"{label} start (T={T_move:.2f}s, tb={tb:.3f}s)")
    while True:
        t_now = sim.getSimulationTime()
        t = min(max(0.0, t_now - t0), T_move)
        s, sd, sdd = lipb_scalar_eval(t, T_move, amax_s, tb)
        p_des = p_start + s * d
        dt_sim = sim.getSimulationTimeStep()
        q_cmd, ep, ew = jacobian_resolved_rate_step(
            q_cmd, p_des, R_hold,
            dt=dt_sim,                                  # <<< ส่ง dt เข้าไป
            lam=0.05, w_pos=6.0, w_ori=1.5,
            dq_clip_deg=8.0, alpha=0.5
        )
        # ส่งคำสั่ง
        for i in range(6): sim.setJointTargetPosition(hdl_j[i], float(q_cmd[i]))
        # log เบา ๆ
        if (last_log < 0) or (t_now - last_log > 0.30):
            log_once(sim, aux, f"{label} s={t/T_move:5.2f} |ep|={np.linalg.norm(ep):.3e} |ew|={np.linalg.norm(ew):.3e}")
            last_log = t_now
        if t >= T_move: break
        sim.switchThread()
    return q_cmd

# ===================== Hooks ของ CoppeliaSim =====================
def sysCall_init():
    sim = require("sim")

def sysCall_thread():
    sim = require("sim")

    # ---------- ดึง handle ข้อต่อและปลายมือ ----------
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")
    hdl_grip = get_gripper_handle(sim)
    gripper_action(sim, hdl_grip, 2)   # เริ่มต้นด้วยการ "ปล่อย" กริปเปอร์

    # ---------- เปิดคอนโซล ----------
    global aux
    AUX_MAX_LINES = 3000
    AUX_SIZE = [850, 820]
    AUX_POS  = [10, 10]
    aux = sim.auxiliaryConsoleOpen(
        'UR5 IK via DLS | pick-place mission (loop)', 
        AUX_MAX_LINES, 0, AUX_POS, AUX_SIZE, [1,1,1], [0,0,0]
    )
    sim.auxiliaryConsolePrint(aux, "=== UR5 pick & place (looping) ===\n")

    # ---------- พารามิเตอร์ IK ----------
    MAX_ITERS = 1000
    TOL_POS   = 1e-4
    TOL_ORI   = 2e-3
    DQ_MAX    = 14.0*d2r

    # ---------- สร้าง solver ----------
    _solver = lambda qs, pt, Rt: ik_priority_dls(
        qs, pt, Rt,
        max_iters=MAX_ITERS, tol_pos=TOL_POS, tol_ori=TOL_ORI,
        lam_pos0=0.02, lam_ori0=0.02, dq_max=DQ_MAX,
        ko_base=0.06, w_null=0.08, warmup=True, do_polish=True
    )

    # ---------- ตัวเลือกลูป ----------
    NUM_CYCLES = 0       # 0 = วนตลอด, >0 = จำนวนรอบ
    CYCLE_DWELL = 0.6    # เวลาพักคั่นรอบ (วินาที)

    # ---------- พิกัด pick & place (ใน BASE) ----------
    box_base_fixed   = BOX_START_WORLD  # จุดหยิบ
    place_base_fixed = PLACE_IN_CONV  # จุดวาง

    # ---------- ลูปภารกิจ ----------
    cycle = 0
    while (NUM_CYCLES == 0) or (cycle < NUM_CYCLES):
        cycle += 1
        sim.auxiliaryConsolePrint(aux, f"\n--- Cycle {cycle} ---\n")
        try:
            q_end = do_pick_and_place(
                sim, hdl_j, hdl_end,
                box_base=box_base_fixed,
                place_base=place_base_fixed,
                yaw_deg=0.0,
                pre_back=0.14,
                descend=0.08,
                t_hint=15.0
            )
            sim.auxiliaryConsolePrint(aux, "Cycle complete.\n")
        except Exception as e:
            sim.auxiliaryConsolePrint(aux, f"[ERROR @ cycle {cycle}] {e}\n")
            break

        # ค้างท่าปลายทางสั้น ๆ เพื่อความนิ่ง
        t_hold = sim.getSimulationTime()
        while sim.getSimulationTime() - t_hold < CYCLE_DWELL:
            for i in range(6):
                sim.setJointTargetPosition(hdl_j[i], float(q_end[i]))
            sim.switchThread()

    sim.auxiliaryConsolePrint(aux, "Loop finished.\n")

def sysCall_actuation():
    pass

def sysCall_sensing():
    pass

def sysCall_cleanup():
    pass
