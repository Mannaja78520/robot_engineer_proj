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
            dq = Jp_pinv @ ep + Np @ (0.02 * Jo.T @ ew)
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

# ===================== Hooks ของ CoppeliaSim =====================
def sysCall_init():
    sim = require("sim")

def sysCall_thread():
    sim = require("sim")
    
    # ดึง handle ข้อต่อและปลายมือ
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    # เปิดคอนโซลแสดงผล
    global aux
    AUX_MAX_LINES = 3000
    AUX_SIZE = [1120, 820]
    AUX_POS  = [10, 10]
    aux = sim.auxiliaryConsoleOpen(
        'UR5 IK via DLS (priority+polish, numeric J) | sim_pos/eul | J_used',
        AUX_MAX_LINES, 0, AUX_POS, AUX_SIZE, [1,1,1], [0,0,0]
    )

    # เตรียมวัตถุเป้าในซีน
    h_cube = get_or_create_target_cube(sim)

    # ค่าพารามิเตอร์ของตัวแก้ IK (ปรับได้)
    MAX_ITERS = 1000
    TOL_POS   = 1e-4
    TOL_ORI   = 2e-3
    DQ_MAX    = 14.0*d2r

    # เซ็ตชุดทดสอบเป้าหมาย
    tests = [
        ([0.33,  0.25, 0.10], [  0.0,   0.0,   0.0]),
        ([0.45,  0.10, 0.65], [  0.0,  20.0,  30.0]),
        ([0.35, -0.25, 0.70], [ 30.0, -25.0,  60.0]),
        ([0.55,  0.20, 0.55], [-20.0,  15.0, -40.0]),
        ([0.40,  0.00, 0.80], [ 90.0,   0.0,   0.0]),
        ([0.50,  0.15, 0.60], [ 10.0, -30.0,  45.0]),
        ([0.20, -0.10, 0.75], [-90.0,   0.0, 180.0]),
    ]

    log_once(sim, aux, "UR5 IK via DLS (priority + nullspace + final polish, numeric J)")
    log_once(sim, aux, "="*100)

    # สร้าง solver ที่ล็อกพารามิเตอร์ไว้ (lambda) — ไม่ต้องประกาศฟังก์ชันย่อยในเธรดแล้ว
    _solver = lambda qs, pt, Rt: ik_priority_dls(
        qs, pt, Rt,
        max_iters=MAX_ITERS, tol_pos=TOL_POS, tol_ori=TOL_ORI,
        lam_pos0=0.02, lam_ori0=0.02, dq_max=DQ_MAX,
        ko_base=0.06, w_null=0.08, warmup=True, do_polish=True
    )

    # วิ่งทุกเคสทดสอบ
    for ti, (p_t, eul_deg) in enumerate(tests, 1):
        # ตั้งค่าการมองเห็นในซีน
        eul_rad = (np.array(eul_deg)*d2r).tolist()
        sim.setObjectPosition(h_cube, -1, p_t)
        sim.setObjectOrientation(h_cube, -1, eul_rad)

        # อ่านเป้าที่ใช้งานจริงจากซีน
        p_target = np.array(sim.getObjectPosition(h_cube, -1), float)
        eul_xyz  = np.array(sim.getObjectOrientation(h_cube, -1), float)
        R_target = eulXYZ_to_R(eul_xyz)

        # สถานะปัจจุบันของข้อต่อ
        q_now = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)

        log_once(sim, aux, f"[Test {ti}] TARGET pos=({p_target[0]:.4f}, {p_target[1]:.4f}, {p_target[2]:.4f}) "
                 f"eulXYZ(deg)=({eul_xyz[0]*r2d: .2f}, {eul_xyz[1]*r2d: .2f}, {eul_xyz[2]*r2d: .2f})")

        # เลือก approach และแก้ IK ครั้งแรก (ใช้รายงาน/สำรอง)
        q_sol0, iters0, perr0, oerr0, ok0, chosen = try_ik_with_fallback(q_now, p_target, eul_deg, _solver, force_mode=None)

        # รายงานโหมดและ Euler ที่เลือก
        log_once(sim, aux, f"[Test {ti}] TO pos=({p_target[0]:.4f},{p_target[1]:.4f},{p_target[2]:.4f}) "
                 f"MODE={chosen['mode']} target_eulXYZ(deg)=("
                 f"{chosen['eul'][0]*r2d: .2f},{chosen['eul'][1]*r2d: .2f},{chosen['eul'][2]*r2d: .2f})")

        # วางแผน pre-approach จากท่าปัจจุบัน
        R_cmd = chosen['R']
        q_start = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
        (pre_P, pre_Q), (fin_P, fin_Q), ok_path = plan_pre_and_target(q_start, p_target, R_cmd, _solver)

        # เดินงาน: direct หรือสองช่วง (PRE -> TARGET)
        if not ok_path:
            run_leg(sim, aux, hdl_j, hdl_end, q_start, q_sol0, p_target,
                    rotation_matrix_to_euler(R_cmd), f"[Test {ti}][LEG: direct]")
            q_sol = q_sol0
            iters, perr, oerr, ok = iters0, perr0, oerr0, ok0
        else:
            run_leg(sim, aux, hdl_j, hdl_end, q_start, pre_Q, pre_P,
                    rotation_matrix_to_euler(R_cmd), f"[Test {ti}][LEG 1: to PRE]")

            # ค้างก่อนเข้าเป้าเล็กน้อย
            t_pre_hold = sim.getSimulationTime()
            while sim.getSimulationTime() - t_pre_hold < 0.4:
                for i in range(6):
                    sim.setJointTargetPosition(hdl_j[i], float(pre_Q[i]))
                sim.switchThread()

            run_leg(sim, aux, hdl_j, hdl_end, pre_Q, fin_Q, p_target,
                    rotation_matrix_to_euler(R_cmd), f"[Test {ti}][LEG 2: to TARGET]")

            q_sol = fin_Q
            # คำนวณ error จริงเทียบกับ R_cmd
            p_cur, _, Tcur = ur5_forward_kinematrix(q_sol)
            R_cur = Tcur[:3,:3]
            perr = float(np.linalg.norm(p_target - p_cur))
            ew   = R_cur @ log_SO3(R_cmd @ R_cur.T)
            oerr = float(np.linalg.norm(ew))
            ok   = (perr < TOL_POS) and (oerr < TOL_ORI)
            iters = iters0

        # ค้างที่จุดเป้า
        t_hold_start = sim.getSimulationTime()
        while sim.getSimulationTime() - t_hold_start < DWELL_TIME:
            for i in range(6):
                sim.setJointTargetPosition(hdl_j[i], float(q_sol[i]))
            sim.switchThread()

        # สรุปผล/แสดง J ที่ใช้
        sim_pos  = sim.getObjectPosition(hdl_end, -1)
        sim_eul  = sim.getObjectOrientation(hdl_end, -1)
        fk_pos, fk_eul, _ = ur5_forward_kinematrix(q_sol)

        msg = []
        msg.append("  target_pos:      " + ", ".join(f"{x:7.4f}" for x in p_target))
        msg.append("  target_eul(deg): " + ", ".join(f"{x:7.2f}" for x in (eul_xyz*r2d)))
        msg.append(f"[Test {ti}] Reached target and dwelled {DWELL_TIME:.2f}s")
        msg.append(f"  -> IK {'OK' if ok else 'NOT CONVERGED'} in {iters} iters")
        msg.append(f"     |ep|={perr:.6f} m, |eo|={oerr:.6f} rad")
        msg.append("  Solution q(deg): " + ", ".join(f"{x*r2d:7.2f}" for x in q_sol))
        msg.append("  sim_pos:         " + ", ".join(f"{x:7.4f}" for x in sim_pos))
        msg.append("  sim_eul(deg):    " + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
        msg.append("  FK pos:          " + ", ".join(f"{x:7.4f}" for x in fk_pos))
        msg.append("  FK eulXYZ(deg):  " + ", ".join(f"{x:7.2f}" for x in (fk_eul*r2d)))
        Jnum = jacobian_numeric(q_sol)
        msg.append("  Jacobian USED (numeric):")
        for r in range(6):
            msg.append("    " + " ".join(f"{Jnum[r,c]: 8.5f}" for c in range(6)))
        log_once(sim, aux, "\n".join(msg))
        log_once(sim, aux, "-"*100)

    log_once(sim, aux, "All tests finished.")

def sysCall_actuation():
    pass

def sysCall_sensing():
    pass

def sysCall_cleanup():
    pass
